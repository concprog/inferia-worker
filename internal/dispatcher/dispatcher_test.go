package dispatcher

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/inferia/inferia-worker/internal/control"
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
