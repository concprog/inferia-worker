package shellbridge

import (
	"strings"
	"testing"
)

// fakeRuntime stubs shellbridge.Runtime so we can drive ResolveContainer
// without a real runtime.Runtime in scope.
type fakeRuntime struct {
	loaded    []string
	container map[string]string
}

func (f fakeRuntime) ContainerForDeployment(id string) string { return f.container[id] }
func (f fakeRuntime) LoadedDeployments() []string             { return f.loaded }

func TestResolveContainer_ExplicitContainerWins(t *testing.T) {
	rt := fakeRuntime{loaded: []string{"d-1"}, container: map[string]string{"d-1": "rt-cid"}}
	cid, err := ResolveContainer(nil, rt, "explicit-cid", "d-1")
	if err != nil {
		t.Fatalf("ResolveContainer: %v", err)
	}
	if cid != "explicit-cid" {
		t.Errorf("expected explicit-cid, got %q", cid)
	}
}

func TestResolveContainer_DeploymentResolvedViaRuntime(t *testing.T) {
	rt := fakeRuntime{container: map[string]string{"d-42": "rt-cid-42"}}
	cid, err := ResolveContainer(nil, rt, "", "d-42")
	if err != nil {
		t.Fatalf("ResolveContainer: %v", err)
	}
	if cid != "rt-cid-42" {
		t.Errorf("expected rt-cid-42, got %q", cid)
	}
}

func TestResolveContainer_FirstLoadedWhenNoArgs(t *testing.T) {
	rt := fakeRuntime{
		loaded:    []string{"d-1", "d-2"},
		container: map[string]string{"d-1": "cid-1", "d-2": "cid-2"},
	}
	cid, err := ResolveContainer(nil, rt, "", "")
	if err != nil {
		t.Fatalf("ResolveContainer: %v", err)
	}
	if cid != "cid-1" {
		t.Errorf("expected cid-1 (first loaded), got %q", cid)
	}
}

func TestResolveContainer_NoTargetIsError(t *testing.T) {
	rt := fakeRuntime{}
	cid, err := ResolveContainer(nil, rt, "", "")
	if err == nil {
		t.Fatalf("expected error, got cid=%q", cid)
	}
	if !strings.Contains(err.Error(), "no active deployment") {
		t.Errorf("expected 'no active deployment' error, got %v", err)
	}
}

func TestResolveContainer_DeploymentMissingFromBothSources(t *testing.T) {
	rt := fakeRuntime{}
	_, err := ResolveContainer(nil, rt, "", "ghost-deploy")
	if err == nil {
		t.Fatalf("expected error for missing deployment")
	}
	if !strings.Contains(err.Error(), "ghost-deploy") {
		t.Errorf("expected error to mention deployment id, got %v", err)
	}
}

func TestResolveContainer_NilRuntimeWithNoContainerErrors(t *testing.T) {
	// Defence-in-depth: ResolveContainer must not panic on a nil
	// Runtime — passing nil from a misconfigured caller should produce
	// a clean error.
	_, err := ResolveContainer(nil, nil, "", "")
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
}

func TestLookupHelpers_NilClientReturnsEmpty(t *testing.T) {
	if cid := lookupByDeploymentID(nil, "dep"); cid != "" {
		t.Errorf("expected empty cid, got %q", cid)
	}
	if cid := lookupMostRecent(nil); cid != "" {
		t.Errorf("expected empty cid, got %q", cid)
	}
	if cid := lookupByDeploymentID(nil, ""); cid != "" {
		t.Errorf("expected empty cid for empty dep id, got %q", cid)
	}
}
