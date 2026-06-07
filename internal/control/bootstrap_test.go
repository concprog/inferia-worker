package control

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/inferia/inferia-worker/internal/cloudenv"
)

func TestBootstrap_HappyPath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/workers/register" {
			http.Error(w, "wrong path", 404)
			return
		}
		if r.Method != "POST" {
			http.Error(w, "wrong method", 405)
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer bootstrap-x" {
			http.Error(w, "bad auth: "+got, 401)
			return
		}
		var req RegisterRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if req.NodeName == "" || req.PoolID == "" {
			http.Error(w, "missing required fields", 400)
			return
		}
		_ = json.NewEncoder(w).Encode(RegisterResponse{
			NodeID:    "node-uuid",
			WorkerJWT: "jwt-token",
		})
	}))
	defer server.Close()

	b := &Bootstrapper{
		ControlPlaneURL: server.URL,
		BootstrapToken:  "bootstrap-x",
		HTTP:            server.Client(),
	}
	resp, err := b.Register(context.Background(), RegisterRequest{
		NodeName: "n", PoolID: "p", AdvertiseURL: "https://w", Allocatable: map[string]string{"gpu": "1"},
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	if resp.NodeID != "node-uuid" || resp.WorkerJWT != "jwt-token" {
		t.Errorf("got %+v", resp)
	}
}

func TestBootstrap_401(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "no", 401)
	}))
	defer server.Close()
	b := &Bootstrapper{ControlPlaneURL: server.URL, BootstrapToken: "x", HTTP: server.Client()}
	_, err := b.Register(context.Background(), RegisterRequest{NodeName: "n", PoolID: "p"})
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Errorf("expected 401, got %v", err)
	}
}

func TestBootstrap_409Conflict(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "node name taken", 409)
	}))
	defer server.Close()
	b := &Bootstrapper{ControlPlaneURL: server.URL, BootstrapToken: "x", HTTP: server.Client()}
	_, err := b.Register(context.Background(), RegisterRequest{NodeName: "n", PoolID: "p"})
	if err == nil || !strings.Contains(err.Error(), "409") {
		t.Errorf("expected 409, got %v", err)
	}
}

func TestBootstrap_MalformedJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("not json"))
	}))
	defer server.Close()
	b := &Bootstrapper{ControlPlaneURL: server.URL, BootstrapToken: "x", HTTP: server.Client()}
	_, err := b.Register(context.Background(), RegisterRequest{NodeName: "n", PoolID: "p"})
	if err == nil {
		t.Errorf("expected json decode error")
	}
}

func TestBootstrap_NetworkError(t *testing.T) {
	b := &Bootstrapper{ControlPlaneURL: "http://127.0.0.1:1", BootstrapToken: "x", HTTP: http.DefaultClient}
	_, err := b.Register(context.Background(), RegisterRequest{NodeName: "n", PoolID: "p"})
	if err == nil {
		t.Errorf("expected dial error")
	}
}

func TestBootstrap_DefaultHTTPClient(t *testing.T) {
	b := &Bootstrapper{ControlPlaneURL: "http://127.0.0.1:1", BootstrapToken: "x"}
	// We just want to exercise the lazy default — won't connect.
	_, _ = b.Register(context.Background(), RegisterRequest{NodeName: "n", PoolID: "p"})
}

func TestRegisterRequest_IncludesCloudEnv(t *testing.T) {
	info := cloudenv.RuntimeInfo{
		Kind:             cloudenv.KindAWSEC2,
		InstanceID:       "i-abc",
		Region:           "us-east-1",
		AvailabilityZone: "us-east-1a",
	}
	req := BuildRegisterRequest(BuildRegisterInput{
		NodeName:    "node-1",
		PoolID:      "pool-x",
		Allocatable: map[string]string{"cpu": "8", "gpu": "1"},
		Runtime:     info,
	})
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	s := string(data)
	for _, want := range []string{
		`"runtime_env":"aws-ec2"`,
		`"instance_id":"i-abc"`,
		`"region":"us-east-1"`,
		`"availability_zone":"us-east-1a"`,
	} {
		if !strings.Contains(s, want) {
			t.Errorf("missing %q in %s", want, s)
		}
	}
}

func TestRegisterRequest_OmitsCloudEnvWhenLocal(t *testing.T) {
	info := cloudenv.RuntimeInfo{Kind: cloudenv.KindLocal}
	req := BuildRegisterRequest(BuildRegisterInput{
		NodeName:    "node-1",
		PoolID:      "pool-x",
		Allocatable: map[string]string{"cpu": "8"},
		Runtime:     info,
	})
	data, _ := json.Marshal(req)
	s := string(data)
	if strings.Contains(s, "instance_id") || strings.Contains(s, "region") || strings.Contains(s, "availability_zone") {
		t.Errorf("local runtime should omit cloud-env fields: %s", s)
	}
	// runtime_env is always serialized because Kind=KindLocal is the non-empty
	// string "local" (omitempty drops empty strings, not specific values).
	if !strings.Contains(s, `"runtime_env":"local"`) {
		t.Errorf("runtime_env=local should be present: %s", s)
	}
}

func TestRegisterRequest_IncludesBootstrapToken(t *testing.T) {
	req := BuildRegisterRequest(BuildRegisterInput{
		NodeName:       "node-1",
		PoolID:         "pool-x",
		Allocatable:    map[string]string{"cpu": "1"},
		BootstrapToken: "tok-xyz",
	})
	data, _ := json.Marshal(req)
	if !strings.Contains(string(data), `"bootstrap_token":"tok-xyz"`) {
		t.Errorf("missing bootstrap_token in %s", data)
	}
}

func TestRegisterRequest_OmitsBootstrapTokenWhenEmpty(t *testing.T) {
	req := BuildRegisterRequest(BuildRegisterInput{
		NodeName:    "node-1",
		PoolID:      "pool-x",
		Allocatable: map[string]string{"cpu": "1"},
	})
	data, _ := json.Marshal(req)
	if strings.Contains(string(data), "bootstrap_token") {
		t.Errorf("bootstrap_token should be omitted when empty: %s", data)
	}
}
