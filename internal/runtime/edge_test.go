package runtime

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/inferia/inferia-worker/internal/runtime/dockerclient/fake"
	"github.com/inferia/inferia-worker/internal/runtime/recipes"
)

func TestNew_DefaultsApplied(t *testing.T) {
	rt := New(Config{Docker: fake.New(), Network: "n"})
	if rt.cfg.ReadinessPollInterval == 0 {
		t.Errorf("ReadinessPollInterval default missing")
	}
	if rt.cfg.ReadinessTimeout == 0 {
		t.Errorf("ReadinessTimeout default missing")
	}
	if rt.cfg.PullTimeout == 0 {
		t.Errorf("PullTimeout default missing")
	}
	if rt.cfg.HostPortAllocator == nil {
		t.Errorf("HostPortAllocator default missing")
	}
	if rt.cfg.AdvertiseHost == "" {
		t.Errorf("AdvertiseHost default missing")
	}
	if rt.cfg.ReadinessProbe == nil {
		t.Errorf("ReadinessProbe default missing")
	}
}

func TestLoadModel_PortAllocatorUsedWhenPlanHostPortZero(t *testing.T) {
	fc := fake.New()
	rt := New(Config{
		Docker: fc, Network: "n",
		ReadinessProbe:        okProbe,
		ReadinessPollInterval: 1 * time.Millisecond,
		ReadinessTimeout:      100 * time.Millisecond,
		HostPortAllocator:     fixedPort(21000),
		AdvertiseHost:         "127.0.0.1",
	})
	r, _ := recipes.Get("vllm")
	plan, _ := r.BuildPlan(recipes.BuildInput{
		DeploymentID: "d", ArtifactURI: "hf://o/m", GPUIndices: []int{0}, HostPort: 1, // any non-zero
	})
	plan.HostPort = 0 // force allocator
	res, err := rt.LoadModel(context.Background(), "d", plan)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if res.EndpointURL == "" {
		t.Errorf("empty endpoint")
	}
}

func TestLoadModel_PortAllocatorFailure(t *testing.T) {
	fc := fake.New()
	rt := New(Config{
		Docker: fc, Network: "n",
		ReadinessProbe:        okProbe,
		ReadinessPollInterval: 1 * time.Millisecond,
		ReadinessTimeout:      100 * time.Millisecond,
		HostPortAllocator:     func() (int, error) { return 0, errors.New("exhausted") },
		AdvertiseHost:         "127.0.0.1",
	})
	r, _ := recipes.Get("vllm")
	plan, _ := r.BuildPlan(recipes.BuildInput{
		DeploymentID: "d", ArtifactURI: "hf://o/m", GPUIndices: []int{0}, HostPort: 1,
	})
	plan.HostPort = 0
	if _, err := rt.LoadModel(context.Background(), "d", plan); err == nil {
		t.Errorf("expected allocator error")
	}
}

func TestLoadModel_CreateFailure(t *testing.T) {
	fc := fake.New()
	fc.CreateErr = errors.New("create boom")
	rt := newRT(t, fc, okProbe)
	if _, err := rt.LoadModel(context.Background(), "d", samplePlan("d")); err == nil {
		t.Errorf("expected error")
	}
}

func TestLoadModel_StartFailure(t *testing.T) {
	fc := fake.New()
	fc.StartErr = errors.New("start boom")
	rt := newRT(t, fc, okProbe)
	if _, err := rt.LoadModel(context.Background(), "d", samplePlan("d")); err == nil {
		t.Errorf("expected error")
	}
	if len(fc.Removed) != 1 {
		t.Errorf("expected cleanup remove, got %d", len(fc.Removed))
	}
}

func TestUnloadModel_StopFailure(t *testing.T) {
	fc := fake.New()
	rt := newRT(t, fc, okProbe)
	_, _ = rt.LoadModel(context.Background(), "d", samplePlan("d"))
	fc.StopErr = errors.New("stop boom")
	if err := rt.UnloadModel(context.Background(), "d"); err == nil {
		t.Errorf("expected error")
	}
}

func TestUnloadModel_RemoveFailure(t *testing.T) {
	fc := fake.New()
	rt := newRT(t, fc, okProbe)
	_, _ = rt.LoadModel(context.Background(), "d", samplePlan("d"))
	fc.RemoveErr = errors.New("remove boom")
	if err := rt.UnloadModel(context.Background(), "d"); err == nil {
		t.Errorf("expected error")
	}
}

func TestWaitReady_ContextCancelled(t *testing.T) {
	rt := New(Config{
		Docker:                fake.New(),
		Network:               "n",
		ReadinessProbe:        func(string) bool { return false },
		ReadinessPollInterval: 1 * time.Second,
		ReadinessTimeout:      5 * time.Second,
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if rt.waitReady(ctx, "http://x", 5*time.Second) {
		t.Errorf("expected false after ctx cancel")
	}
}

func TestState_String(t *testing.T) {
	cases := map[State]string{
		StateAbsent: "absent", StatePulling: "pulling", StateStarting: "starting",
		StateRunning: "running", StateStopping: "stopping", StateFailed: "failed",
		State(999): "unknown",
	}
	for s, want := range cases {
		if got := s.String(); got != want {
			t.Errorf("State(%d).String() = %q, want %q", s, got, want)
		}
	}
}

func TestLoadModel_BadSpecRejected(t *testing.T) {
	fc := fake.New()
	rt := newRT(t, fc, okProbe)
	bad := samplePlan("d")
	bad.Image = "" // invalid plan
	if _, err := rt.LoadModel(context.Background(), "d", bad); err == nil {
		t.Errorf("expected error for bad spec")
	}
}

func TestReadinessTimeout_PerPlanOverrideElseGlobal(t *testing.T) {
	rt := New(Config{
		Docker:           fake.New(),
		Network:          "n",
		ReadinessTimeout: 180 * time.Second,
	})
	// Plan sets its own (e.g. diffusion) → use it.
	if got := rt.readinessTimeout(recipes.Plan{ReadinessTimeout: 1800 * time.Second}); got != 1800*time.Second {
		t.Errorf("per-plan timeout = %v, want 1800s", got)
	}
	// Plan leaves it 0 → fall back to the worker global default.
	if got := rt.readinessTimeout(recipes.Plan{}); got != 180*time.Second {
		t.Errorf("global fallback = %v, want 180s", got)
	}
}
