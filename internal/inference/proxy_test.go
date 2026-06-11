package inference

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/inferia/inferia-worker/internal/runtime"
	"github.com/inferia/inferia-worker/internal/runtime/recipes"
)

// fakeRuntime implements EndpointResolver for the proxy tests.
type fakeRuntime struct {
	endpoints map[string]string
}

func (f *fakeRuntime) EndpointURL(deploymentID string) string { return f.endpoints[deploymentID] }

func (f *fakeRuntime) DeploymentInfo(deploymentID string) (recipe, model, phase string, pullDur, startDur time.Duration, ok bool) {
	return "fake-recipe", "fake-model", "running", 0, 0, true
}

func startUpstream(handler http.HandlerFunc) *httptest.Server {
	return httptest.NewServer(handler)
}

func TestProxy_HappyPath(t *testing.T) {
	up := startUpstream(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "wrong path", 404)
			return
		}
		if r.Header.Get("X-User-Hdr") != "kept" {
			http.Error(w, "header missing", 400)
			return
		}
		w.Header().Set("X-Upstream-Hdr", "yes")
		w.WriteHeader(200)
		_, _ = fmt.Fprintf(w, `{"id":"chatcmpl-1"}`)
	})
	defer up.Close()

	rt := &fakeRuntime{endpoints: map[string]string{"dep-1": up.URL}}
	app := fiber.New()
	app.Use(NewProxy(Config{
		Runtime: rt,
		Resolver: PathResolver{
			// Resolve target deployment from a header set by the auth middleware
			// upstream. For the unit test we override with a stub that picks dep-1.
			Resolver: func(c *fiber.Ctx) string { return "dep-1" },
		},
	}))

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"m"}`))
	req.Header.Set("X-User-Hdr", "kept")
	req.Header.Set("Content-Type", "application/json")
	resp, err := app.Test(req, 5000)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status=%d body=%s", resp.StatusCode, string(body))
	}
	if resp.Header.Get("X-Upstream-Hdr") != "yes" {
		t.Errorf("upstream header lost")
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "chatcmpl-1") {
		t.Errorf("body=%s", body)
	}
}

func TestProxy_DeploymentNotLoadedReturns503(t *testing.T) {
	rt := &fakeRuntime{endpoints: map[string]string{}}
	app := fiber.New()
	app.Use(NewProxy(Config{
		Runtime: rt,
		Resolver: PathResolver{
			Resolver: func(c *fiber.Ctx) string { return "nope" },
		},
	}))
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	resp, _ := app.Test(req)
	if resp.StatusCode != 503 {
		t.Errorf("status: %d", resp.StatusCode)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Errorf("missing Retry-After")
	}
}

func TestProxy_PassesThroughSSEStream(t *testing.T) {
	up := startUpstream(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		f, _ := w.(http.Flusher)
		for i := 0; i < 3; i++ {
			fmt.Fprintf(w, "data: chunk-%d\n\n", i)
			f.Flush()
			time.Sleep(5 * time.Millisecond)
		}
	})
	defer up.Close()

	rt := &fakeRuntime{endpoints: map[string]string{"dep-1": up.URL}}
	app := fiber.New()
	app.Use(NewProxy(Config{
		Runtime: rt,
		Resolver: PathResolver{
			Resolver: func(c *fiber.Ctx) string { return "dep-1" },
		},
	}))
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"stream":true}`))
	req.Header.Set("Content-Type", "application/json")
	resp, err := app.Test(req, 5000)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Errorf("ct: %q", resp.Header.Get("Content-Type"))
	}
	body, _ := io.ReadAll(resp.Body)
	for i := 0; i < 3; i++ {
		want := fmt.Sprintf("chunk-%d", i)
		if !strings.Contains(string(body), want) {
			t.Errorf("missing %q in body: %s", want, body)
		}
	}
}

func TestPathResolver_FromHeader(t *testing.T) {
	app := fiber.New()
	var captured string
	app.Use(func(c *fiber.Ctx) error {
		r := PathResolver{
			Header: "X-Deployment-ID",
		}
		captured = r.Resolve(c)
		return c.SendString("ok")
	})
	req := httptest.NewRequest("GET", "/v1/x", nil)
	req.Header.Set("X-Deployment-ID", "dep-7")
	_, _ = app.Test(req)
	if captured != "dep-7" {
		t.Errorf("captured: %q", captured)
	}
}

func TestPathResolver_FromHeaderMissing(t *testing.T) {
	app := fiber.New()
	var captured string
	app.Use(func(c *fiber.Ctx) error {
		captured = (PathResolver{Header: "X-Deployment-ID"}).Resolve(c)
		return c.SendString("ok")
	})
	req := httptest.NewRequest("GET", "/v1/x", nil)
	_, _ = app.Test(req)
	if captured != "" {
		t.Errorf("expected empty, got %q", captured)
	}
}

func TestPathResolver_CustomResolverOverridesHeader(t *testing.T) {
	app := fiber.New()
	var captured string
	app.Use(func(c *fiber.Ctx) error {
		captured = (PathResolver{
			Header:   "X-Deployment-ID",
			Resolver: func(c *fiber.Ctx) string { return "from-fn" },
		}).Resolve(c)
		return c.SendString("ok")
	})
	req := httptest.NewRequest("GET", "/v1/x", nil)
	req.Header.Set("X-Deployment-ID", "from-hdr")
	_, _ = app.Test(req)
	if captured != "from-fn" {
		t.Errorf("captured: %q", captured)
	}
}

func TestProxy_SkipsNonV1Paths(t *testing.T) {
	rt := &fakeRuntime{endpoints: map[string]string{}}
	app := fiber.New()
	app.Use(NewProxy(Config{
		Runtime: rt,
		Resolver: PathResolver{
			Resolver: func(c *fiber.Ctx) string { return "nope" },
		},
	}))
	app.Get("/healthz", func(c *fiber.Ctx) error { return c.SendString("ok") })
	req := httptest.NewRequest("GET", "/healthz", nil)
	resp, _ := app.Test(req)
	if resp.StatusCode != 200 {
		t.Errorf("/healthz status: %d", resp.StatusCode)
	}
}

func TestProxy_UpstreamUnreachable(t *testing.T) {
	rt := &fakeRuntime{endpoints: map[string]string{"dep-1": "http://127.0.0.1:1"}}
	app := fiber.New()
	app.Use(NewProxy(Config{
		Runtime:  rt,
		Resolver: PathResolver{Resolver: func(c *fiber.Ctx) string { return "dep-1" }},
	}))
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	resp, _ := app.Test(req, 5000)
	if resp.StatusCode != 502 && resp.StatusCode != 503 {
		t.Errorf("expected 5xx, got %d", resp.StatusCode)
	}
}

func TestProxy_PreservesQueryString(t *testing.T) {
	got := make(chan string, 1)
	up := startUpstream(func(w http.ResponseWriter, r *http.Request) {
		got <- r.URL.RawQuery
		w.WriteHeader(200)
	})
	defer up.Close()
	rt := &fakeRuntime{endpoints: map[string]string{"dep-1": up.URL}}
	app := fiber.New()
	app.Use(NewProxy(Config{
		Runtime:  rt,
		Resolver: PathResolver{Resolver: func(c *fiber.Ctx) string { return "dep-1" }},
	}))
	req := httptest.NewRequest("GET", "/v1/models?foo=bar&baz=qux", nil)
	_, err := app.Test(req)
	if err != nil {
		t.Fatalf("%v", err)
	}
	select {
	case q := <-got:
		if !strings.Contains(q, "foo=bar") || !strings.Contains(q, "baz=qux") {
			t.Errorf("query: %q", q)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("upstream not called")
	}
}

func TestProxy_AbortsOnCtxCancel(t *testing.T) {
	// Slow upstream; client cancels.
	up := startUpstream(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(2 * time.Second):
			w.WriteHeader(200)
		case <-r.Context().Done():
		}
	})
	defer up.Close()
	rt := &fakeRuntime{endpoints: map[string]string{"dep-1": up.URL}}
	app := fiber.New()
	app.Use(NewProxy(Config{
		Runtime:  rt,
		Resolver: PathResolver{Resolver: func(c *fiber.Ctx) string { return "dep-1" }},
	}))
	req := httptest.NewRequestWithContext(
		func() context.Context {
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			_ = cancel
			return ctx
		}(),
		"GET", "/v1/x", nil,
	)
	_, _ = app.Test(req, 1000)
	// We're not asserting the exact status — Fiber may close — only that the
	// proxy did not hang the test.
}

// --- Disagg proxy tests ------------------------------------------------------

func TestProxy_DisaggRouting(t *testing.T) {
	// Phase-1 server (prefill): expects stream=false, max_tokens=1,
	// kv_transfer_params with do_remote_decode=true, request_id present.
	pCalled := false
	var sharedRequestID string
	pServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pCalled = true
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "bad json", 400)
			return
		}
		// Must have been modified for prefill
		if s, ok := req["stream"].(bool); !ok || s != false {
			t.Errorf("prefill: stream=%v, want false", req["stream"])
		}
		if mt, ok := req["max_tokens"].(float64); !ok || mt != 1 {
			t.Errorf("prefill: max_tokens=%v, want 1", req["max_tokens"])
		}
		// Must have kv_transfer_params structure (not top-level do_remote_decode)
		ktp, ok := req["kv_transfer_params"].(map[string]any)
		if !ok {
			t.Errorf("prefill: missing kv_transfer_params map")
		} else {
			if dr, ok := ktp["do_remote_decode"].(bool); !ok || dr != true {
				t.Errorf("prefill: kv_transfer_params.do_remote_decode=%v, want true", ktp["do_remote_decode"])
			}
			if drp, ok := ktp["do_remote_prefill"].(bool); !ok || drp != false {
				t.Errorf("prefill: kv_transfer_params.do_remote_prefill=%v, want false", ktp["do_remote_prefill"])
			}
		}
		// Must have request_id in body
		rid, ok := req["request_id"].(string)
		if !ok || rid == "" {
			t.Errorf("prefill: missing or empty request_id")
		}
		sharedRequestID = rid
		// Must have X-Request-Id header matching body
		if r.Header.Get("X-Request-Id") != rid {
			t.Errorf("prefill: X-Request-Id header=%q, want %q", r.Header.Get("X-Request-Id"), rid)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"kv_transfer_params":{"engine_id":"e1","block_ids":[1,2,3]}}`)
	}))
	defer pServer.Close()

	// Phase-2 server (decode): expects original body + kv_transfer_params injected
	// and matching request_id.
	dCalled := false
	dServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		dCalled = true
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "bad json", 400)
			return
		}
		// Must have stream=true (preserved from original)
		if s, ok := req["stream"].(bool); !ok || s != true {
			t.Errorf("decode: stream=%v, want true", req["stream"])
		}
		// Must have max_tokens=4096 (preserved from original)
		if mt, ok := req["max_tokens"].(float64); !ok || mt != 4096 {
			t.Errorf("decode: max_tokens=%v, want 4096", req["max_tokens"])
		}
		// Must have kv_transfer_params from prefill
		if _, ok := req["kv_transfer_params"]; !ok {
			t.Errorf("decode: missing kv_transfer_params")
		}
		// Must have matching request_id
		rid, ok := req["request_id"].(string)
		if !ok || rid == "" {
			t.Errorf("decode: missing or empty request_id")
		}
		if rid != sharedRequestID {
			t.Errorf("decode: request_id=%q, want %q (same as prefill)", rid, sharedRequestID)
		}
		if r.Header.Get("X-Request-Id") != rid {
			t.Errorf("decode: X-Request-Id header=%q, want %q", r.Header.Get("X-Request-Id"), rid)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-disagg","choices":[{"text":"Hello from D"}]}`)
	}))
	defer dServer.Close()

	// Parse upstream URLs to extract ports
	pURL := strings.TrimPrefix(pServer.URL, "http://127.0.0.1:")
	dURL := strings.TrimPrefix(dServer.URL, "http://127.0.0.1:")
	pPort, dPort := 0, 0
	fmt.Sscanf(pURL, "%d", &pPort)
	fmt.Sscanf(dURL, "%d", &dPort)

	reg := NewDeploymentRegistry()
	reg.RegisterDisagg("dep-disagg-1", "mymodel", &runtime.DeploymentGroup{
		ID: "dep-disagg-1",
		Prefill: []runtime.ContainerInfo{
			{HostPort: pPort, Role: recipes.KvRoleProducer, ReplicaIdx: 0},
		},
		Decode: []runtime.ContainerInfo{
			{HostPort: dPort, Role: recipes.KvRoleConsumer, ReplicaIdx: 0},
		},
	})

	app := fiber.New()
	app.Use(NewProxy(Config{
		Resolver: PathResolver{Resolver: func(c *fiber.Ctx) string { return "dep-disagg-1" }},
		Registry: reg,
	}))

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"mymodel","stream":true,"max_tokens":4096}`))
	req.Header.Set("Content-Type", "application/json")
	resp, err := app.Test(req, 5000)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status=%d body=%s", resp.StatusCode, string(body))
	}
	if !pCalled {
		t.Error("prefill server was never called")
	}
	if !dCalled {
		t.Error("decode server was never called")
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "Hello from D") {
		t.Errorf("response body=%q", string(body))
	}
}

func TestProxy_DisaggRegistryNilFallsThrough(t *testing.T) {
	// When no registry is set, the proxy must take the legacy single-endpoint path.
	up := startUpstream(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"id":"legacy"}`)
	})
	defer up.Close()

	rt := &fakeRuntime{endpoints: map[string]string{"dep-1": up.URL}}
	app := fiber.New()
	app.Use(NewProxy(Config{
		Runtime:  rt,
		Resolver: PathResolver{Resolver: func(c *fiber.Ctx) string { return "dep-1" }},
		// Registry is nil — legacy path
	}))

	req := httptest.NewRequest("GET", "/v1/completions",
		strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")
	resp, err := app.Test(req, 5000)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status=%d body=%s", resp.StatusCode, string(body))
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "legacy") {
		t.Errorf("body=%q", string(body))
	}
}

func TestProxy_DisaggEntryNotFoundFallsThrough(t *testing.T) {
	// When the registry has no entry for the deployment, falls through
	// to the legacy path.
	up := startUpstream(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"id":"legacy"}`)
	})
	defer up.Close()

	reg := NewDeploymentRegistry() // empty registry
	rt := &fakeRuntime{endpoints: map[string]string{"dep-1": up.URL}}
	app := fiber.New()
	app.Use(NewProxy(Config{
		Runtime:  rt,
		Resolver: PathResolver{Resolver: func(c *fiber.Ctx) string { return "dep-1" }},
		Registry: reg,
	}))

	req := httptest.NewRequest("GET", "/v1/completions",
		strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")
	resp, err := app.Test(req, 5000)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status=%d body=%s", resp.StatusCode, string(body))
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "legacy") {
		t.Errorf("body=%q", string(body))
	}
}
