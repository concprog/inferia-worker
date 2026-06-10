package runtime

import (
	"testing"

	"github.com/inferia/inferia-worker/internal/runtime/recipes"
)

func TestDeploymentGroup_RoundRobin(t *testing.T) {
	g := &DeploymentGroup{
		Prefill: []ContainerInfo{
			{HostPort: 19001, Role: recipes.KvRoleProducer, ReplicaIdx: 0},
			{HostPort: 19002, Role: recipes.KvRoleProducer, ReplicaIdx: 1},
		},
		Decode: []ContainerInfo{
			{HostPort: 19501, Role: recipes.KvRoleConsumer, ReplicaIdx: 0},
			{HostPort: 19502, Role: recipes.KvRoleConsumer, ReplicaIdx: 1},
		},
	}

	wantPrefills := []string{
		"http://127.0.0.1:19001",
		"http://127.0.0.1:19002",
		"http://127.0.0.1:19001",
		"http://127.0.0.1:19002",
	}
	for i, want := range wantPrefills {
		if got := g.NextPrefill(); got != want {
			t.Errorf("prefill iter %d: want %q, got %q", i, want, got)
		}
	}

	wantDecodes := []string{
		"http://127.0.0.1:19501",
		"http://127.0.0.1:19502",
		"http://127.0.0.1:19501",
	}
	for i, want := range wantDecodes {
		if got := g.NextDecode(); got != want {
			t.Errorf("decode iter %d: want %q, got %q", i, want, got)
		}
	}
}

func TestDeploymentGroup_EmptyReturnsEmptyString(t *testing.T) {
	g := &DeploymentGroup{}
	if got := g.NextPrefill(); got != "" {
		t.Errorf("expected empty string, got %q", got)
	}
	if got := g.NextDecode(); got != "" {
		t.Errorf("expected empty string, got %q", got)
	}
}
