package dispatcher

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/inferia/inferia-worker/internal/control"
	"github.com/inferia/inferia-worker/internal/metrics"
	"github.com/inferia/inferia-worker/internal/runtime"
	"github.com/inferia/inferia-worker/internal/runtime/recipes"
)

type fakeRT struct {
	loaded      map[string]string
	loadCalls   []recipes.Plan
	loadErr     error
	unloadErr   error
	unloadedIDs []string
}

func newFakeRT() *fakeRT { return &fakeRT{loaded: map[string]string{}} }

func (f *fakeRT) LoadModel(ctx context.Context, id string, plan recipes.Plan) (*runtime.LoadResult, error) {
	f.loadCalls = append(f.loadCalls, plan)
	if f.loadErr != nil {
		return nil, f.loadErr
	}
	f.loaded[id] = "http://endpoint/" + id
	return &runtime.LoadResult{EndpointURL: f.loaded[id]}, nil
}

func (f *fakeRT) UnloadModel(ctx context.Context, id string) error {
	f.unloadedIDs = append(f.unloadedIDs, id)
	if f.unloadErr != nil {
		return f.unloadErr
	}
	delete(f.loaded, id)
	return nil
}

func (f *fakeRT) LoadedDeployments() []string {
	out := make([]string, 0, len(f.loaded))
	for k := range f.loaded {
		out = append(out, k)
	}
	return out
}

func (f *fakeRT) DeploymentInfo(deploymentID string) (recipe, model, phase string, pullDur, startDur time.Duration, ok bool) {
	return "vllm", "llama3.1", "running", 0, 0, true
}

func (f *fakeRT) EndpointURL(deploymentID string) string { return f.loaded[deploymentID] }

type fakeTelemetry struct{ data map[string]string }

func (f *fakeTelemetry) Read() map[string]string { return f.data }

func TestLoadModel_HappyPath(t *testing.T) {
	rt := newFakeRT()
	d := &Dispatcher{Rt: rt}
	endpoint, err := d.LoadModel(context.Background(), control.LoadModelBody{
		DeploymentID: "dep-1",
		Recipe:       "vllm",
		Model:        control.ModelRef{ArtifactURI: "hf://o/m"},
		GPUIndices:   []int{0},
		Port:         1234,
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	if endpoint == "" {
		t.Errorf("empty endpoint")
	}
	if len(rt.loadCalls) != 1 {
		t.Errorf("expected 1 load call, got %d", len(rt.loadCalls))
	}
	if rt.loadCalls[0].HostPort != 1234 {
		t.Errorf("plan HostPort: %d", rt.loadCalls[0].HostPort)
	}
}

func TestLoadModel_PortZeroSignalsAllocator(t *testing.T) {
	rt := newFakeRT()
	d := &Dispatcher{Rt: rt}
	_, err := d.LoadModel(context.Background(), control.LoadModelBody{
		DeploymentID: "dep-2",
		Recipe:       "vllm",
		Model:        control.ModelRef{ArtifactURI: "hf://o/m"},
		GPUIndices:   []int{0},
		Port:         0,
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	if got := rt.loadCalls[0].HostPort; got != 0 {
		t.Errorf("plan HostPort: %d (expected 0 to signal allocator)", got)
	}
}

func TestLoadModel_UnknownRecipe(t *testing.T) {
	d := &Dispatcher{Rt: newFakeRT()}
	_, err := d.LoadModel(context.Background(), control.LoadModelBody{
		DeploymentID: "x", Recipe: "nope", Model: control.ModelRef{ArtifactURI: "hf://o/m"},
		GPUIndices: []int{0}, Port: 1,
	})
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestLoadModel_BadPlanRejected(t *testing.T) {
	d := &Dispatcher{Rt: newFakeRT()}
	_, err := d.LoadModel(context.Background(), control.LoadModelBody{
		DeploymentID: "x", Recipe: "vllm",
		Model:      control.ModelRef{ArtifactURI: "javascript:bad"},
		GPUIndices: []int{0}, Port: 1,
	})
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestLoadModel_RuntimeError(t *testing.T) {
	rt := newFakeRT()
	rt.loadErr = errors.New("docker boom")
	d := &Dispatcher{Rt: rt}
	_, err := d.LoadModel(context.Background(), control.LoadModelBody{
		DeploymentID: "x", Recipe: "vllm",
		Model: control.ModelRef{ArtifactURI: "hf://o/m"}, GPUIndices: []int{0}, Port: 1,
	})
	if err == nil {
		t.Errorf("expected runtime error")
	}
}

func TestUnloadModel_HappyPath(t *testing.T) {
	rt := newFakeRT()
	d := &Dispatcher{Rt: rt}
	if err := d.UnloadModel(context.Background(), control.UnloadModelBody{DeploymentID: "x"}); err != nil {
		t.Errorf("%v", err)
	}
	if len(rt.unloadedIDs) != 1 || rt.unloadedIDs[0] != "x" {
		t.Errorf("unloaded: %v", rt.unloadedIDs)
	}
}

func TestUnloadModel_RuntimeError(t *testing.T) {
	rt := newFakeRT()
	rt.unloadErr = errors.New("stop boom")
	d := &Dispatcher{Rt: rt}
	if err := d.UnloadModel(context.Background(), control.UnloadModelBody{DeploymentID: "x"}); err == nil {
		t.Errorf("expected error")
	}
}

func TestHeartbeatSnapshot_WithTelemetry(t *testing.T) {
	rt := newFakeRT()
	_, _ = rt.LoadModel(context.Background(), "dep-1", recipes.Plan{})
	d := &Dispatcher{
		Rt:        rt,
		Telemetry: &fakeTelemetry{data: map[string]string{"cpu_pct": "42.5"}},
	}
	hb := d.HeartbeatSnapshot()
	if hb.Used["cpu_pct"] != "42.5" {
		t.Errorf("used: %v", hb.Used)
	}
	if len(hb.LoadedModels) != 1 || hb.LoadedModels[0] != "dep-1" {
		t.Errorf("loaded: %v", hb.LoadedModels)
	}
}

func TestHeartbeatSnapshot_NilTelemetryIsSafe(t *testing.T) {
	d := &Dispatcher{Rt: newFakeRT()}
	hb := d.HeartbeatSnapshot()
	if hb.Used == nil || len(hb.Used) != 0 {
		t.Errorf("expected empty map, got %v", hb.Used)
	}
}

func TestSafeFmt(t *testing.T) {
	if SafeFmt("%d", 7) != "7" {
		t.Errorf("SafeFmt")
	}
}

func TestMetricsPipeline_VerboseEndToEnd(t *testing.T) {
	// ================================================================
	// 1. SETUP — create the full pipeline: Collector → Dispatcher
	// ================================================================
	t.Log("=== SETUP: Creating Collector, fakeRuntime, Dispatcher ===")

	mc := metrics.NewCollector()
	rt := newFakeRT()
	rt.loaded["dep-vllm"] = "http://placeholder:9999"
	rt.loaded["dep-ollama"] = "http://placeholder:9998"

	tel := &fakeTelemetry{data: map[string]string{"cpu_pct": "23.5", "mem_pct": "45.2"}}

	d := &Dispatcher{
		Rt:        rt,
		Metrics:   mc,
		Telemetry: tel,
	}

	// ================================================================
	// 2. SIMULATE PROXY MIDDLEWARE — IncActive / RecordRequest / DecActive
	// ================================================================
	t.Log("=== PROXY: Simulating 15 vLLM requests with varying latency ===")
	for i := 0; i < 15; i++ {
		mc.IncActive("dep-vllm")
		latency := int64(50 + i*15)
		mc.RecordRequest("dep-vllm", "vllm", "llama3.1", latency)
		mc.DecActive("dep-vllm")
		t.Logf("  req %2d: latency=%dms  → bucket recipe=vllm model=llama3.1", i+1, latency)
	}
	t.Log("")

	t.Log("=== PROXY: Simulating 5 Ollama requests with higher latency ===")
	for i := 0; i < 5; i++ {
		mc.IncActive("dep-ollama")
		latency := int64(200 + i*50)
		mc.RecordRequest("dep-ollama", "ollama", "gemma3:4b", latency)
		mc.DecActive("dep-ollama")
		t.Logf("  req %2d: latency=%dms  → bucket recipe=ollama model=gemma3:4b", i+1, latency)
	}
	t.Log("")

	// ================================================================
	// 3. VLLM SCRAPE — fake /metrics endpoint
	// ================================================================
	t.Log("=== VLLM SCRAPE: Starting fake /metrics endpoint ===")

	vllmMetrics := `vllm:num_requests_running 2.0
vllm:request_success_total 42.0
vllm:request_failed_total 1.0
vllm:avg_generation_throughput_toks_per_sec 58.3`

	vllmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Logf("  ← vLLM scraper hit %s (200)", r.URL.Path)
		w.WriteHeader(200)
		_, _ = w.Write([]byte(vllmMetrics))
	}))
	defer vllmSrv.Close()

	rt.loaded["dep-vllm"] = vllmSrv.URL

	if err := mc.ScrapeVLLM("dep-vllm", vllmSrv.URL); err != nil {
		t.Fatalf("ScrapeVLLM(dep-vllm) error: %v", err)
	}
	t.Log("  ✓ Engine metrics after scrape:")
	for k, v := range mc.GetVLLMMetrics("dep-vllm") {
		t.Logf("    %s = %v", k, v)
	}

	// Non-vLLM scrape should be a no-op (recipe check in ScrapeVLLM)
	if err := mc.ScrapeVLLM("dep-ollama", "http://irrelevant"); err != nil {
		t.Fatalf("ScrapeVLLM(dep-ollama) error: %v", err)
	}
	t.Log("  ✓ ScrapeVLLM(dep-ollama) correctly returned nil (recipe != vllm)")
	t.Log("")

	// ================================================================
	// 4. HEARTBEAT — generate HeartbeatSnapshot
	// ================================================================
	t.Log("=== HEARTBEAT: Calling HeartbeatSnapshot() ===")

	hb := d.HeartbeatSnapshot()

	t.Logf("  Host telemetry:  cpu_pct=%s  mem_pct=%s",
		hb.Used["cpu_pct"], hb.Used["mem_pct"])
	t.Logf("  Loaded models:   %v", hb.LoadedModels)
	t.Logf("  Deploy metrics:  %d deployments", len(hb.DeployMetrics))
	t.Log("")

	// ================================================================
	// 5. INSPECT every metric field
	// ================================================================
	for _, dm := range hb.DeployMetrics {
		t.Logf("  Deployment: %s", dm.DeploymentID)
		t.Logf("    recipe=%-6s model=%-10s phase=%-8s",
			dm.Recipe, dm.Model, dm.Phase)
		t.Logf("    requests_total=%-3d  active_requests=%d",
			dm.RequestsTotal, dm.ActiveRequests)
		t.Logf("    latencies: p50=%-4dms  p95=%dms",
			dm.RequestLatencyP50Ms, dm.RequestLatencyP95Ms)
		t.Logf("    durations: pull=%-4dms  start=%dms",
			dm.PullDurationMs, dm.StartDurationMs)
		if len(dm.EngineMetrics) > 0 {
			t.Logf("    engine_metrics:")
			for k, v := range dm.EngineMetrics {
				t.Logf("      %s = %v", k, v)
			}
		}
	}
	t.Log("")

	// ================================================================
	// 6. ASSERT — verify every value
	// ================================================================
	t.Log("=== ASSERT: Verifying metric values ===")

	var vllmDM, ollamaDM *control.DeploymentMetric
	for i := range hb.DeployMetrics {
		switch hb.DeployMetrics[i].DeploymentID {
		case "dep-vllm":
			vllmDM = &hb.DeployMetrics[i]
		case "dep-ollama":
			ollamaDM = &hb.DeployMetrics[i]
		}
	}
	if vllmDM == nil {
		t.Fatal("dep-vllm missing from DeployMetrics")
	}
	if ollamaDM == nil {
		t.Fatal("dep-ollama missing from DeployMetrics")
	}

	// --- vLLM checks ---
	if vllmDM.RequestsTotal != 15 {
		t.Errorf("dep-vllm requests_total: want 15, got %d", vllmDM.RequestsTotal)
	} else {
		t.Log("  ✓ dep-vllm requests_total = 15")
	}
	if vllmDM.ActiveRequests != 0 {
		t.Errorf("dep-vllm active_requests: want 0, got %d", vllmDM.ActiveRequests)
	} else {
		t.Log("  ✓ dep-vllm active_requests = 0")
	}
	if vllmDM.RequestLatencyP50Ms == 0 {
		t.Errorf("dep-vllm p50: expected >0")
	} else {
		t.Logf("  ✓ dep-vllm request_latency_p50_ms = %d", vllmDM.RequestLatencyP50Ms)
	}
	if vllmDM.RequestLatencyP95Ms == 0 {
		t.Errorf("dep-vllm p95: expected >0")
	} else {
		t.Logf("  ✓ dep-vllm request_latency_p95_ms = %d", vllmDM.RequestLatencyP95Ms)
	}
	if vllmDM.Recipe != "vllm" {
		t.Errorf("dep-vllm recipe: want vllm, got %s", vllmDM.Recipe)
	} else {
		t.Log("  ✓ dep-vllm recipe = vllm")
	}
	if vllmDM.Phase != "running" {
		t.Errorf("dep-vllm phase: want running, got %s", vllmDM.Phase)
	} else {
		t.Log("  ✓ dep-vllm phase = running")
	}
	if vllmDM.EngineMetrics == nil || len(vllmDM.EngineMetrics) == 0 {
		t.Errorf("dep-vllm engine_metrics: expected non-empty")
	} else {
		if v, ok := vllmDM.EngineMetrics["vllm:num_requests_running"]; !ok || v != 2.0 {
			t.Errorf("vllm:num_requests_running: want 2.0, got %v", v)
		} else {
			t.Logf("  ✓ dep-vllm vllm:num_requests_running = 2.0")
		}
		if v, ok := vllmDM.EngineMetrics["vllm:request_success_total"]; !ok || v != 42.0 {
			t.Errorf("vllm:request_success_total: want 42.0, got %v", v)
		} else {
			t.Logf("  ✓ dep-vllm vllm:request_success_total = 42.0")
		}
	}

	// --- Ollama checks ---
	if ollamaDM.RequestsTotal != 5 {
		t.Errorf("dep-ollama requests_total: want 5, got %d", ollamaDM.RequestsTotal)
	} else {
		t.Log("  ✓ dep-ollama requests_total = 5")
	}
	if len(ollamaDM.EngineMetrics) != 0 {
		t.Errorf("dep-ollama engine_metrics: expected empty, got %v", ollamaDM.EngineMetrics)
	} else {
		t.Log("  ✓ dep-ollama engine_metrics = empty (not vLLM)")
	}

	// --- Host telemetry ---
	if hb.Used["cpu_pct"] != "23.5" {
		t.Errorf("cpu_pct: want 23.5, got %s", hb.Used["cpu_pct"])
	} else {
		t.Log("  ✓ host cpu_pct = 23.5")
	}
	if hb.Used["mem_pct"] != "45.2" {
		t.Errorf("mem_pct: want 45.2, got %s", hb.Used["mem_pct"])
	} else {
		t.Log("  ✓ host mem_pct = 45.2")
	}
	if len(hb.LoadedModels) != 2 {
		t.Errorf("loaded_models: want 2, got %d", len(hb.LoadedModels))
	} else {
		t.Log("  ✓ loaded_models = [dep-vllm dep-ollama]")
	}
	t.Log("")

	// ================================================================
	// 7. UNLOAD — verify metric cleanup
	// ================================================================
	t.Log("=== UNLOAD: UnloadModel → RemoveDeployment → Snapshot ===")

	if err := d.UnloadModel(context.Background(), control.UnloadModelBody{DeploymentID: "dep-vllm"}); err != nil {
		t.Fatalf("UnloadModel(dep-vllm) error: %v", err)
	}
	t.Log("  ✓ UnloadModel(dep-vllm) succeeded")

	hb2 := d.HeartbeatSnapshot()
	for _, dm := range hb2.DeployMetrics {
		if dm.DeploymentID == "dep-vllm" {
			t.Errorf("dep-vllm should have been removed from metrics after UnloadModel")
		}
	}
	t.Log("  ✓ dep-vllm correctly absent from subsequent HeartbeatSnapshot")

	// Verify dep-ollama still present (not unloaded)
	foundOllama := false
	for _, dm := range hb2.DeployMetrics {
		if dm.DeploymentID == "dep-ollama" {
			foundOllama = true
			break
		}
	}
	if !foundOllama {
		t.Errorf("dep-ollama should still be present")
	} else {
		t.Log("  ✓ dep-ollama still present (not unloaded)")
	}

	// ================================================================
	// 8. SECOND-HEARTBEAT — verify cumulative counter + sliding histogram
	// ================================================================
	t.Log("")
	t.Log("=== SECOND HEARTBEAT: RequestsTotal is cumulative, histogram resets ===")

	// RequestsTotal is now Load() (cumulative), so it stays at 5.
	// The histogram is Reset() after each snapshot, so P50/P95 become 0.
	for _, dm := range hb2.DeployMetrics {
		if dm.DeploymentID == "dep-ollama" {
			if dm.RequestsTotal != 5 {
				t.Errorf("dep-ollama requests_total (cumulative): want 5, got %d", dm.RequestsTotal)
			} else {
				t.Log("  ✓ dep-ollama requests_total stays 5 (cumulative, not windowed)")
			}
			if dm.RequestLatencyP50Ms != 0 || dm.RequestLatencyP95Ms != 0 {
				t.Errorf("dep-ollama latency should be 0 after histogram reset, got p50=%d p95=%d",
					dm.RequestLatencyP50Ms, dm.RequestLatencyP95Ms)
			} else {
				t.Log("  ✓ dep-ollama latencies reset to 0 (sliding histogram)")
			}
		}
	}

	t.Log("")
	t.Log("=== ALL VERIFIED: metrics pipeline end-to-end ==")
}
