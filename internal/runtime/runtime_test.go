package runtime

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/inferia/inferia-worker/internal/runtime/dockerclient"
	"github.com/inferia/inferia-worker/internal/runtime/dockerclient/fake"
	"github.com/inferia/inferia-worker/internal/runtime/recipes"
)

func newRT(t *testing.T, fc *fake.Client, probe func(url string) bool) *Runtime {
	t.Helper()
	return New(Config{
		Docker:                fc,
		Network:               "inferia-models",
		PullTimeout:           5 * time.Second,
		ReadinessTimeout:      2 * time.Second,
		ReadinessPollInterval: 10 * time.Millisecond,
		ReadinessProbe:        probe,
		HostPortAllocator:     fixedPort(19000),
	})
}

func fixedPort(p int) func() (int, error) { return func() (int, error) { return p, nil } }

func okProbe(url string) bool { return true }

func samplePlan(deploymentID string) recipes.Plan {
	r, _ := recipes.Get("vllm")
	plan, _ := r.BuildPlan(recipes.BuildInput{
		DeploymentID: deploymentID,
		ArtifactURI:  "hf://meta/m",
		GPUIndices:   []int{0},
		HostPort:     19000,
	})
	return plan
}

func TestLoadModel_HappyPath(t *testing.T) {
	fc := fake.New()
	rt := newRT(t, fc, okProbe)

	res, err := rt.LoadModel(context.Background(), "dep-1", samplePlan("dep-1"))
	if err != nil {
		t.Fatalf("%v", err)
	}
	if res.EndpointURL == "" || !strings.Contains(res.EndpointURL, "19000") {
		t.Errorf("endpoint: %q", res.EndpointURL)
	}
	if len(fc.Pulled) != 1 {
		t.Errorf("expected 1 pull, got %d", len(fc.Pulled))
	}
	if len(fc.Created) != 1 || len(fc.Started) != 1 {
		t.Errorf("expected 1 create + 1 start")
	}
	if !contains(rt.LoadedDeployments(), "dep-1") {
		t.Errorf("LoadedDeployments missing dep-1: %v", rt.LoadedDeployments())
	}
}

func TestLoadModel_Idempotent(t *testing.T) {
	fc := fake.New()
	rt := newRT(t, fc, okProbe)
	ctx := context.Background()

	res1, err := rt.LoadModel(ctx, "dep-1", samplePlan("dep-1"))
	if err != nil {
		t.Fatalf("%v", err)
	}
	res2, err := rt.LoadModel(ctx, "dep-1", samplePlan("dep-1"))
	if err != nil {
		t.Fatalf("%v", err)
	}
	if res1.EndpointURL != res2.EndpointURL {
		t.Errorf("endpoint changed: %q vs %q", res1.EndpointURL, res2.EndpointURL)
	}
	if len(fc.Pulled) != 1 {
		t.Errorf("expected pull called once across two loads, got %d", len(fc.Pulled))
	}
}

func TestLoadModel_PullFailureNoLeak(t *testing.T) {
	fc := fake.New()
	fc.PullErr = errors.New("registry boom")
	rt := newRT(t, fc, okProbe)

	_, err := rt.LoadModel(context.Background(), "dep-1", samplePlan("dep-1"))
	if err == nil {
		t.Fatalf("expected error")
	}
	if len(fc.Created) != 0 {
		t.Errorf("no container should be created on pull failure")
	}
	if contains(rt.LoadedDeployments(), "dep-1") {
		t.Errorf("failed deployment should not be in LoadedDeployments")
	}
}

func TestLoadModel_ReadinessTimeoutCleansUp(t *testing.T) {
	fc := fake.New()
	rt := newRT(t, fc, func(url string) bool { return false }) // never ready

	_, err := rt.LoadModel(context.Background(), "dep-1", samplePlan("dep-1"))
	if err == nil {
		t.Fatalf("expected timeout error")
	}
	// Container should have been removed.
	if len(fc.Removed) != 1 {
		t.Errorf("expected 1 remove on readiness fail, got %d", len(fc.Removed))
	}
	if contains(rt.LoadedDeployments(), "dep-1") {
		t.Errorf("failed deployment should not be in LoadedDeployments")
	}
}

func TestUnloadModel_StopsAndRemoves(t *testing.T) {
	fc := fake.New()
	rt := newRT(t, fc, okProbe)
	ctx := context.Background()

	if _, err := rt.LoadModel(ctx, "dep-1", samplePlan("dep-1")); err != nil {
		t.Fatalf("%v", err)
	}
	if err := rt.UnloadModel(ctx, "dep-1"); err != nil {
		t.Fatalf("%v", err)
	}
	if len(fc.Stopped) != 1 || len(fc.Removed) != 1 {
		t.Errorf("expected stop+remove")
	}
	if contains(rt.LoadedDeployments(), "dep-1") {
		t.Errorf("dep-1 still in LoadedDeployments after unload")
	}
}

func TestUnloadModel_AbsentOK(t *testing.T) {
	fc := fake.New()
	rt := newRT(t, fc, okProbe)
	if err := rt.UnloadModel(context.Background(), "nope"); err != nil {
		t.Errorf("Unload of absent should be ok, got %v", err)
	}
}

func TestStatusOf(t *testing.T) {
	fc := fake.New()
	rt := newRT(t, fc, okProbe)
	if rt.StatusOf("dep-1") != StateAbsent {
		t.Errorf("expected absent")
	}
	_, _ = rt.LoadModel(context.Background(), "dep-1", samplePlan("dep-1"))
	if rt.StatusOf("dep-1") != StateRunning {
		t.Errorf("expected running, got %v", rt.StatusOf("dep-1"))
	}
}

func TestEndpointURL(t *testing.T) {
	fc := fake.New()
	rt := newRT(t, fc, okProbe)
	if rt.EndpointURL("nope") != "" {
		t.Errorf("expected empty")
	}
	_, _ = rt.LoadModel(context.Background(), "dep-1", samplePlan("dep-1"))
	if !strings.Contains(rt.EndpointURL("dep-1"), "19000") {
		t.Errorf("expected port 19000, got %q", rt.EndpointURL("dep-1"))
	}
}

func TestLoadModel_RealProbeAgainstHTTPServer(t *testing.T) {
	// Verify the default HTTP probe path with a real httptest server.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(200)
			return
		}
		w.WriteHeader(404)
	}))
	defer ts.Close()
	if !httpProbe(ts.URL + "/health") {
		t.Errorf("expected probe to pass")
	}
	if httpProbe(ts.URL + "/not-there") {
		t.Errorf("expected probe to fail on 404")
	}
	if httpProbe("http://127.0.0.1:1") {
		t.Errorf("expected probe to fail on dial error")
	}
}

func TestConcurrentLoad_Serialised(t *testing.T) {
	fc := fake.New()
	rt := newRT(t, fc, okProbe)
	ctx := context.Background()

	// 50 concurrent LoadModel calls for the same deployment id should result
	// in exactly one underlying Create.
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = rt.LoadModel(ctx, "dep-x", samplePlan("dep-x"))
		}()
	}
	wg.Wait()
	if got := len(fc.Created); got != 1 {
		t.Errorf("expected 1 Create call, got %d", got)
	}
}

func TestPortAllocator_AutoAssigns(t *testing.T) {
	a := NewPortAllocator(20000, 20100)
	seen := map[int]bool{}
	for i := 0; i < 5; i++ {
		p, err := a.Next()
		if err != nil {
			t.Fatalf("%v", err)
		}
		if seen[p] {
			t.Errorf("duplicate port %d", p)
		}
		seen[p] = true
	}
}

func TestPortAllocator_Exhaustion(t *testing.T) {
	a := NewPortAllocator(20000, 20001)
	if _, err := a.Next(); err != nil {
		t.Fatalf("%v", err)
	}
	if _, err := a.Next(); err != nil {
		t.Fatalf("%v", err)
	}
	if _, err := a.Next(); err == nil {
		t.Errorf("expected exhaustion error")
	}
}

func TestPortAllocator_Release(t *testing.T) {
	a := NewPortAllocator(20000, 20001)
	p1, _ := a.Next()
	p2, _ := a.Next()
	if _, err := a.Next(); err == nil {
		t.Fatalf("expected exhaustion")
	}
	a.Release(p1)
	p3, err := a.Next()
	if err != nil {
		t.Fatalf("%v", err)
	}
	if p3 != p1 {
		t.Errorf("expected released %d back, got %d", p1, p3)
	}
	_ = p2
}

func TestEnsureNetwork_OnNew(t *testing.T) {
	fc := fake.New()
	rt := newRT(t, fc, okProbe)
	if err := rt.EnsureNetwork(context.Background()); err != nil {
		t.Fatalf("%v", err)
	}
	if len(fc.NetworksCreated) != 1 || fc.NetworksCreated[0] != "inferia-models" {
		t.Errorf("nets created: %v", fc.NetworksCreated)
	}
}

func TestPing(t *testing.T) {
	fc := fake.New()
	rt := newRT(t, fc, okProbe)
	if err := rt.Ping(context.Background()); err != nil {
		t.Fatalf("%v", err)
	}
	if fc.Pinged != 1 {
		t.Errorf("expected 1 ping, got %d", fc.Pinged)
	}
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

// silence unused warning when only some tests reference these types
var _ dockerclient.Client = (*fake.Client)(nil)

// ollamaSamplePlan builds a minimal ollama Plan for integration tests.
// deploymentID is omitted — it is passed separately to LoadModel.
func ollamaSamplePlan(containerName, model string, port int) recipes.Plan {
	return recipes.Plan{
		Image:         "ollama/ollama:latest",
		ContainerName: containerName,
		Cmd:           []string{"serve"},
		Env:           map[string]string{"INFERIA_OLLAMA_MODEL": model},
		ContainerPort: port,
		HostPort:      port,
		ReadyPath:     "/",
	}
}

func TestLoadModel_OllamaPullsAfterReady(t *testing.T) {
	// Stand up a fake ollama server that records the pull.
	var pulled int
	pullSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/pull" {
			pulled++
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"status":"success"}`))
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer pullSrv.Close()

	u, err := url.Parse(pullSrv.URL)
	if err != nil {
		t.Fatalf("parse pull url: %v", err)
	}
	_, portStr, _ := strings.Cut(u.Host, ":")
	var port int
	_, _ = fmt.Sscanf(portStr, "%d", &port)

	fc := fake.New()
	rt := newRT(t, fc, okProbe)

	// Regression guard (server misbehaving / NXDOMAIN under --network host):
	// the pull MUST reach the host-bound port (AdvertiseHost:HostPort), NOT the
	// bridge ContainerName. ContainerName is deliberately a bogus, unresolvable
	// name here; HostPort points at the fake ollama server (on 127.0.0.1, the
	// default AdvertiseHost). If the pull regressed to dialing ContainerName,
	// this test would fail to reach the server and `pulled` would stay 0.
	plan := ollamaSamplePlan("inferia-ollama-bridge-unresolvable", "qwen3:0.6b", port)

	_, err = rt.LoadModel(context.Background(), "dep-1", plan)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	if pulled != 1 {
		t.Errorf("ollama /api/pull called %d times, want 1", pulled)
	}
	if !contains(rt.LoadedDeployments(), "dep-1") {
		t.Errorf("deployment not loaded after successful pull")
	}
}

func TestLoadModel_OllamaPullFailure_CleansUp(t *testing.T) {
	// Server returns 4xx for /api/pull → no retry, immediate failure.
	pullSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/pull" {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":"manifest not found"}`))
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer pullSrv.Close()

	u, _ := url.Parse(pullSrv.URL)
	_, portStr, _ := strings.Cut(u.Host, ":")
	var port int
	_, _ = fmt.Sscanf(portStr, "%d", &port)

	fc := fake.New()
	rt := newRT(t, fc, okProbe)

	plan := ollamaSamplePlan("inferia-ollama-dep-1", "qwen3:0.6b", port)
	plan.ContainerName = u.Hostname()

	_, err := rt.LoadModel(context.Background(), "dep-1", plan)
	if err == nil {
		t.Fatalf("expected pull failure, got nil")
	}
	if !strings.Contains(err.Error(), "ollama pull-after-ready") {
		t.Errorf("error %q does not mention pull-after-ready", err)
	}
	if len(fc.Removed) != 1 {
		t.Errorf("expected 1 container remove on pull failure, got %d", len(fc.Removed))
	}
	if contains(rt.LoadedDeployments(), "dep-1") {
		t.Errorf("failed deployment should not be loaded")
	}
}

func TestLoadModel_NonOllamaRecipe_DoesNotPull(t *testing.T) {
	// Track pulls; vllm should never hit /api/pull.
	pullSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/pull" {
			t.Errorf("vllm recipe must not call /api/pull")
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer pullSrv.Close()

	fc := fake.New()
	rt := newRT(t, fc, okProbe)
	_, err := rt.LoadModel(context.Background(), "dep-1", samplePlan("dep-1")) // vllm
	if err != nil {
		t.Fatalf("%v", err)
	}
}
