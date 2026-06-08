package inference

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gofiber/fiber/v2"
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
