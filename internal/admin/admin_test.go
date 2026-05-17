// Tests for resolveContainerCore — the resolution precedence is the
// load-bearing part of the admin package; the WS / docker plumbing
// around it is exercised by the integration tests we ran during
// development. These keep the precedence chain locked down so
// regressions surface in CI rather than in a live shell session.
package admin

import (
	"testing"
)

// fakeRuntime stubs the admin.Runtime interface with explicit maps.
type fakeRuntime struct {
	loaded    []string
	container map[string]string // deploymentID → containerID
}

func (f fakeRuntime) ContainerForDeployment(id string) string { return f.container[id] }
func (f fakeRuntime) LoadedDeployments() []string             { return f.loaded }

// fakeLookup stubs containerLookup with explicit return values so we can
// assert which branch the resolution took without spinning a docker
// daemon.
type fakeLookup struct {
	byDep      map[string]string // depID → containerID
	mostRecent string
	// Counters so a test can verify the lookup was (or was not) consulted.
	calledByDep      int
	calledMostRecent int
}

func (f *fakeLookup) ByDeploymentID(depID string) string {
	f.calledByDep++
	return f.byDep[depID]
}
func (f *fakeLookup) MostRecent() string {
	f.calledMostRecent++
	return f.mostRecent
}

func queryFn(q map[string]string) func(string) string {
	return func(k string) string { return q[k] }
}

func TestResolveContainer_PrefersExplicitContainerQuery(t *testing.T) {
	rt := fakeRuntime{loaded: []string{"d-1"}, container: map[string]string{"d-1": "rt-container"}}
	look := &fakeLookup{}
	got, errMsg := resolveContainerCore(
		queryFn(map[string]string{"container": "explicit-cid", "deployment": "d-1"}),
		rt, look,
	)
	if got != "explicit-cid" {
		t.Fatalf("expected explicit-cid (the ?container= override), got %q", got)
	}
	if errMsg != "" {
		t.Fatalf("expected no error, got %q", errMsg)
	}
	if look.calledByDep != 0 || look.calledMostRecent != 0 {
		t.Fatalf("docker lookup must not be consulted when ?container= is set; got byDep=%d mostRecent=%d",
			look.calledByDep, look.calledMostRecent)
	}
}

func TestResolveContainer_RuntimeMappingWins(t *testing.T) {
	rt := fakeRuntime{container: map[string]string{"d-42": "rt-container-42"}}
	look := &fakeLookup{byDep: map[string]string{"d-42": "docker-fallback"}}
	got, errMsg := resolveContainerCore(
		queryFn(map[string]string{"deployment": "d-42"}),
		rt, look,
	)
	if got != "rt-container-42" {
		t.Fatalf("expected runtime mapping to win, got %q", got)
	}
	if errMsg != "" {
		t.Fatalf("expected no error, got %q", errMsg)
	}
	if look.calledByDep != 0 {
		t.Fatalf("docker lookup must not be consulted when runtime knows the deployment")
	}
}

func TestResolveContainer_DockerFallbackAfterRuntimeMiss(t *testing.T) {
	// This is the post-restart case: runtime registry is empty, model
	// container is still up. Docker filter should be consulted and the
	// container surfaced.
	rt := fakeRuntime{} // empty
	look := &fakeLookup{byDep: map[string]string{"d-99": "found-via-docker"}}
	got, errMsg := resolveContainerCore(
		queryFn(map[string]string{"deployment": "d-99"}),
		rt, look,
	)
	if got != "found-via-docker" {
		t.Fatalf("expected docker fallback to surface the container, got %q (err=%q)", got, errMsg)
	}
	if look.calledByDep != 1 {
		t.Fatalf("expected one ByDeploymentID call, got %d", look.calledByDep)
	}
}

func TestResolveContainer_FirstLoadedDeploymentWhenNoQuery(t *testing.T) {
	rt := fakeRuntime{
		loaded:    []string{"d-first", "d-second"},
		container: map[string]string{"d-first": "cid-first", "d-second": "cid-second"},
	}
	look := &fakeLookup{}
	got, errMsg := resolveContainerCore(queryFn(nil), rt, look)
	if got != "cid-first" {
		t.Fatalf("expected first loaded deployment to be chosen, got %q (err=%q)", got, errMsg)
	}
	if look.calledMostRecent != 0 {
		t.Fatalf("docker MostRecent must not be consulted when runtime has loaded deployments")
	}
}

func TestResolveContainer_MostRecentFallbackOnEmptyRuntime(t *testing.T) {
	// No query params, no runtime state → last-ditch docker most-recent
	// fallback so a freshly-restarted worker still serves Logs/Shell.
	rt := fakeRuntime{}
	look := &fakeLookup{mostRecent: "newest-inferia-cid"}
	got, errMsg := resolveContainerCore(queryFn(nil), rt, look)
	if got != "newest-inferia-cid" {
		t.Fatalf("expected most-recent fallback, got %q (err=%q)", got, errMsg)
	}
	if look.calledMostRecent != 1 {
		t.Fatalf("expected one MostRecent call, got %d", look.calledMostRecent)
	}
}

func TestResolveContainer_NoTargetReturnsError(t *testing.T) {
	rt := fakeRuntime{}
	look := &fakeLookup{} // empty
	got, errMsg := resolveContainerCore(queryFn(nil), rt, look)
	if got != "" {
		t.Fatalf("expected empty container id, got %q", got)
	}
	if errMsg != "no active deployment on this worker" {
		t.Fatalf("expected the 'no active deployment' error, got %q", errMsg)
	}
}

func TestResolveContainer_DeploymentMissingFromBothSourcesReportsTheDeployment(t *testing.T) {
	rt := fakeRuntime{}
	look := &fakeLookup{} // empty
	_, errMsg := resolveContainerCore(
		queryFn(map[string]string{"deployment": "ghost-deploy"}),
		rt, look,
	)
	if errMsg == "" {
		t.Fatalf("expected an error explaining the deployment is missing")
	}
	// The exact phrasing carries the deployment id back to the operator so
	// they can tell whether they asked for the wrong id vs. the worker is
	// off-state.
	if want := `"ghost-deploy"`; !contains(errMsg, want) {
		t.Fatalf("expected error to mention the deployment id (%q), got %q", want, errMsg)
	}
}

// contains avoids strings.Contains in the test file's import set since
// every other test uses no stdlib at all — kept the file dependency-free.
func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
