package inference

import (
	"testing"

	"github.com/inferia/inferia-worker/internal/runtime"
	"github.com/inferia/inferia-worker/internal/runtime/recipes"
)

func TestRegistry_RegisterAndLookup(t *testing.T) {
	reg := NewDeploymentRegistry()

	group := &runtime.DeploymentGroup{
		ID:    "dep-disagg-1",
		Model: "meta-llama/Llama-3.1-8B",
		Prefill: []runtime.ContainerInfo{
			{HostPort: 19001, Role: recipes.KvRoleProducer},
		},
		Decode: []runtime.ContainerInfo{
			{HostPort: 19501, Role: recipes.KvRoleConsumer},
		},
	}
	reg.RegisterDisagg("dep-disagg-1", "meta-llama/Llama-3.1-8B", group)

	entry, ok := reg.LookupByID("dep-disagg-1")
	if !ok {
		t.Fatal("expected to find entry")
	}
	if entry.Mode != ModeDisagg {
		t.Errorf("Mode=%d, want ModeDisagg", entry.Mode)
	}
	if entry.Model != "meta-llama/Llama-3.1-8B" {
		t.Errorf("Model=%q", entry.Model)
	}
	if entry.Group == nil {
		t.Fatal("Group is nil")
	}
	if len(entry.Group.Prefill) != 1 || len(entry.Group.Decode) != 1 {
		t.Errorf("unexpected replica counts: P=%d D=%d", len(entry.Group.Prefill), len(entry.Group.Decode))
	}
}

func TestRegistry_DeregisterRemovesEntry(t *testing.T) {
	reg := NewDeploymentRegistry()
	reg.RegisterDisagg("dep-1", "model", &runtime.DeploymentGroup{ID: "dep-1"})

	if _, ok := reg.LookupByID("dep-1"); !ok {
		t.Fatal("expected entry before deregister")
	}

	reg.Deregister("dep-1")
	if _, ok := reg.LookupByID("dep-1"); ok {
		t.Error("entry still found after deregister")
	}
}

func TestRegistry_LookupUnknownReturnsFalse(t *testing.T) {
	reg := NewDeploymentRegistry()
	if _, ok := reg.LookupByID("nonexistent"); ok {
		t.Error("expected false for unknown ID")
	}
}

func TestRegistry_DeregisterUnknownNoPanic(t *testing.T) {
	reg := NewDeploymentRegistry()
	reg.Deregister("ghost-id") // must not panic
}
