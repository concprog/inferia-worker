package cloudenv

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
)

func TestDetect_EnvOverrideSetsKind(t *testing.T) {
	t.Setenv("INFERIA_RUNTIME_ENV", "aws-ec2")
	t.Setenv("INFERIA_INSTANCE_ID", "i-test-1234")
	t.Setenv("INFERIA_REGION", "us-east-1")
	t.Setenv("INFERIA_AZ", "us-east-1a")
	// Force IMDS path off so we don't depend on real network.
	t.Setenv("INFERIA_CLOUDENV_IMDS_URL", "http://127.0.0.1:1") // unreachable

	got := detectFresh() // bypasses cache for tests
	if got.Kind != KindAWSEC2 {
		t.Fatalf("Kind = %q, want %q", got.Kind, KindAWSEC2)
	}
	if got.InstanceID != "i-test-1234" {
		t.Errorf("InstanceID = %q", got.InstanceID)
	}
	if got.Region != "us-east-1" {
		t.Errorf("Region = %q", got.Region)
	}
	if got.AvailabilityZone != "us-east-1a" {
		t.Errorf("AZ = %q", got.AvailabilityZone)
	}
}

func TestDetect_NoEnvNoIMDSReturnsLocal(t *testing.T) {
	t.Setenv("INFERIA_RUNTIME_ENV", "")
	t.Setenv("INFERIA_CLOUDENV_IMDS_URL", "http://127.0.0.1:1") // unreachable
	got := detectFresh()
	if got.Kind != KindLocal {
		t.Fatalf("Kind = %q, want %q", got.Kind, KindLocal)
	}
}

// TestDetect_CacheReturnsSameValue verifies that Detect() caches the result and
// returns identical values on repeated calls.
// NOTE: cacheOnce and cached are process-global (sync.Once). Any test that calls
// Detect() MUST reset them first:
//
//	cacheOnce = sync.Once{}
//	cached = RuntimeInfo{}
func TestDetect_CacheReturnsSameValue(t *testing.T) {
	// Reset the process-global cache so this test owns the first call.
	cacheOnce = sync.Once{}
	cached = RuntimeInfo{}
	t.Cleanup(func() {
		cacheOnce = sync.Once{}
		cached = RuntimeInfo{}
	})

	t.Setenv("INFERIA_RUNTIME_ENV", "aws-ec2")
	t.Setenv("INFERIA_INSTANCE_ID", "i-cache-test")
	t.Setenv("INFERIA_REGION", "us-west-2")
	t.Setenv("INFERIA_AZ", "us-west-2b")
	// Prevent any real IMDS call.
	t.Setenv("INFERIA_CLOUDENV_IMDS_URL", "http://127.0.0.1:1")

	first := Detect()
	if first.Kind != KindAWSEC2 {
		t.Fatalf("first call: Kind = %q, want %q", first.Kind, KindAWSEC2)
	}
	if first.InstanceID != "i-cache-test" {
		t.Fatalf("first call: InstanceID = %q, want %q", first.InstanceID, "i-cache-test")
	}

	second := Detect()
	third := Detect()

	if second != first {
		t.Errorf("second call differs: got %+v, want %+v", second, first)
	}
	if third != first {
		t.Errorf("third call differs: got %+v, want %+v", third, first)
	}
}

func TestDetect_IMDSv2Success(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPut && r.URL.Path == "/latest/api/token":
			if r.Header.Get("X-aws-ec2-metadata-token-ttl-seconds") == "" {
				t.Errorf("missing TTL header")
			}
			w.Write([]byte("imds-token-abc"))
		case r.Method == http.MethodGet && r.URL.Path == "/latest/dynamic/instance-identity/document":
			if r.Header.Get("X-aws-ec2-metadata-token") != "imds-token-abc" {
				t.Errorf("missing/wrong token")
			}
			w.Write([]byte(`{"instanceId":"i-real","region":"eu-west-2","availabilityZone":"eu-west-2b"}`))
		default:
			http.Error(w, "unexpected", 400)
		}
	}))
	defer ts.Close()
	t.Setenv("INFERIA_CLOUDENV_IMDS_URL", ts.URL)
	t.Setenv("INFERIA_RUNTIME_ENV", "")

	got := detectFresh()
	if got.Kind != KindAWSEC2 || got.InstanceID != "i-real" || got.Region != "eu-west-2" || got.AvailabilityZone != "eu-west-2b" {
		t.Fatalf("got %+v", got)
	}
}

func TestDetect_IMDSv1Disabled(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Reject any GET without a token (IMDSv1 disabled), accept PUT.
		if r.Method == http.MethodPut {
			w.Write([]byte("tok"))
			return
		}
		if r.Header.Get("X-aws-ec2-metadata-token") == "tok" {
			w.Write([]byte(`{"instanceId":"i-v2","region":"us-west-2","availabilityZone":"us-west-2c"}`))
			return
		}
		http.Error(w, "v1 disabled", 401)
	}))
	defer ts.Close()
	t.Setenv("INFERIA_CLOUDENV_IMDS_URL", ts.URL)
	t.Setenv("INFERIA_RUNTIME_ENV", "")

	got := detectFresh()
	if got.Kind != KindAWSEC2 || got.InstanceID != "i-v2" {
		t.Fatalf("got %+v", got)
	}
}

func TestDetect_IMDSPayloadOversizeIsBounded(t *testing.T) {
	// Server returns a 1 MB JSON document. We should not OOM; we should fail
	// to parse and fall back to KindLocal.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut {
			w.Write([]byte("tok"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		padding := make([]byte, 1024*1024)
		for i := range padding {
			padding[i] = 'x'
		}
		w.Write([]byte(`{"instanceId":"i","region":"r","availabilityZone":"a","pad":"`))
		w.Write(padding)
		w.Write([]byte(`"}`))
	}))
	defer ts.Close()
	t.Setenv("INFERIA_CLOUDENV_IMDS_URL", ts.URL)
	t.Setenv("INFERIA_RUNTIME_ENV", "")

	got := detectFresh()
	if got.Kind != KindLocal {
		t.Fatalf("oversize payload should fall back to local, got %+v", got)
	}
}

func TestDetect_Cached(t *testing.T) {
	var hits atomic.Int64
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		if r.Method == http.MethodPut {
			w.Write([]byte("tok"))
			return
		}
		w.Write([]byte(`{"instanceId":"i","region":"r","availabilityZone":"a"}`))
	}))
	defer ts.Close()
	t.Setenv("INFERIA_CLOUDENV_IMDS_URL", ts.URL)
	t.Setenv("INFERIA_RUNTIME_ENV", "")

	// Reset the package-level cache for a clean test.
	cacheOnce = sync.Once{}
	cached = RuntimeInfo{}
	t.Cleanup(func() {
		cacheOnce = sync.Once{}
		cached = RuntimeInfo{}
	})

	_ = Detect()
	_ = Detect()
	_ = Detect()
	if h := hits.Load(); h < 1 || h > 2 {
		t.Fatalf("expected 1-2 IMDS hits (PUT + GET), got %d", h)
	}
}
