// Package runtime owns the lifecycle of model containers on a worker host.
// It accepts LoadModel / UnloadModel calls from the control channel, talks to
// the Docker daemon via internal/runtime/dockerclient, and exposes the current
// inference endpoint for the inference proxy.
package runtime

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/inferia/inferia-worker/internal/runtime/dockerclient"
	"github.com/inferia/inferia-worker/internal/runtime/recipes"
)

// LoadResult is returned by LoadModel.
type LoadResult struct {
	EndpointURL string // base URL the inference proxy should route to
}

// Config wires up the Runtime's dependencies.
type Config struct {
	Docker                dockerclient.Client
	Network               string
	PullTimeout           time.Duration
	ReadinessTimeout      time.Duration
	ReadinessPollInterval time.Duration
	ReadinessProbe        func(url string) bool // injected for tests
	HostPortAllocator     func() (int, error)   // returns next host port; injectable for tests
	AdvertiseHost         string                // e.g. "127.0.0.1" — host portion of EndpointURL
}

// Runtime is the per-worker model lifecycle owner.
type Runtime struct {
	cfg Config

	mu          sync.Mutex
	deployments map[string]*deployment // by deploymentID
}

type deployment struct {
	containerID string
	hostPort    int
	endpointURL string
	state       State
	plan        recipes.Plan

	// init guards a single concurrent LoadModel for the same id.
	init sync.Mutex
}

// New constructs a Runtime. Sensible defaults applied for missing optional fields.
func New(cfg Config) *Runtime {
	if cfg.ReadinessProbe == nil {
		cfg.ReadinessProbe = httpProbe
	}
	if cfg.ReadinessPollInterval == 0 {
		cfg.ReadinessPollInterval = 500 * time.Millisecond
	}
	if cfg.ReadinessTimeout == 0 {
		cfg.ReadinessTimeout = 180 * time.Second
	}
	if cfg.PullTimeout == 0 {
		cfg.PullTimeout = 600 * time.Second
	}
	if cfg.HostPortAllocator == nil {
		alloc := NewPortAllocator(19000, 19999)
		cfg.HostPortAllocator = alloc.Next
	}
	if cfg.AdvertiseHost == "" {
		cfg.AdvertiseHost = "127.0.0.1"
	}
	return &Runtime{
		cfg:         cfg,
		deployments: map[string]*deployment{},
	}
}

// Ping checks docker reachability.
func (r *Runtime) Ping(ctx context.Context) error { return r.cfg.Docker.Ping(ctx) }

// EnsureNetwork creates the inferia-models bridge network if missing.
func (r *Runtime) EnsureNetwork(ctx context.Context) error {
	return r.cfg.Docker.EnsureNetwork(ctx, r.cfg.Network)
}

// LoadedDeployments returns the ids currently in the running state.
func (r *Runtime) LoadedDeployments() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, 0, len(r.deployments))
	for id, d := range r.deployments {
		if d.state == StateRunning {
			out = append(out, id)
		}
	}
	return out
}

// StatusOf returns the current state of a deployment, or StateAbsent.
func (r *Runtime) StatusOf(deploymentID string) State {
	r.mu.Lock()
	defer r.mu.Unlock()
	d, ok := r.deployments[deploymentID]
	if !ok {
		return StateAbsent
	}
	return d.state
}

// EndpointURL returns the local inference URL for a deployment, or "" if absent.
func (r *Runtime) EndpointURL(deploymentID string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	d, ok := r.deployments[deploymentID]
	if !ok {
		return ""
	}
	return d.endpointURL
}

// LoadModel pulls the image, runs the container, and waits for readiness.
// Idempotent: a duplicate call for the same deploymentID either waits for the
// first to finish and returns its result, or — if already running — returns
// immediately.
func (r *Runtime) LoadModel(ctx context.Context, deploymentID string, plan recipes.Plan) (*LoadResult, error) {
	d := r.getOrCreate(deploymentID)
	d.init.Lock()
	defer d.init.Unlock()

	// Re-check state under the per-deployment lock.
	if r.StatusOf(deploymentID) == StateRunning && r.EndpointURL(deploymentID) != "" {
		return &LoadResult{EndpointURL: r.EndpointURL(deploymentID)}, nil
	}

	// Allocate host port. Use the recipe's HostPort if provided; otherwise the allocator.
	hostPort := plan.HostPort
	if hostPort == 0 {
		p, err := r.cfg.HostPortAllocator()
		if err != nil {
			return nil, fmt.Errorf("port allocation: %w", err)
		}
		hostPort = p
		plan.HostPort = hostPort
	}

	// Pull.
	r.setState(deploymentID, StatePulling)
	pullCtx, cancel := context.WithTimeout(ctx, r.cfg.PullTimeout)
	if err := r.cfg.Docker.Pull(pullCtx, plan.Image); err != nil {
		cancel()
		r.setState(deploymentID, StateFailed)
		r.drop(deploymentID)
		return nil, fmt.Errorf("pull: %w", err)
	}
	cancel()

	// Create + start.
	spec, err := dockerclient.BuildContainerSpec(plan, r.cfg.Network)
	if err != nil {
		r.setState(deploymentID, StateFailed)
		r.drop(deploymentID)
		return nil, fmt.Errorf("spec: %w", err)
	}
	cid, err := r.cfg.Docker.Create(ctx, spec)
	if err != nil {
		r.setState(deploymentID, StateFailed)
		r.drop(deploymentID)
		return nil, fmt.Errorf("create: %w", err)
	}
	r.setContainerID(deploymentID, cid)

	r.setState(deploymentID, StateStarting)
	if err := r.cfg.Docker.Start(ctx, cid); err != nil {
		_ = r.cfg.Docker.Remove(ctx, cid)
		r.setState(deploymentID, StateFailed)
		r.drop(deploymentID)
		return nil, fmt.Errorf("start: %w", err)
	}

	// Wait for readiness. Probe the container via its name + container-port
	// on the shared inferia-models network — the worker and the model both
	// attach to that network, so DNS resolution + intra-bridge routing
	// succeed without relying on the host's loopback (which would point at
	// the worker container itself, not the model container).
	probeURL := fmt.Sprintf("http://%s:%d%s", plan.ContainerName, plan.ContainerPort, plan.ReadyPath)
	endpoint := fmt.Sprintf("http://%s:%d", r.cfg.AdvertiseHost, hostPort)
	if !r.waitReady(ctx, probeURL) {
		_ = r.cfg.Docker.Stop(ctx, cid, 5)
		_ = r.cfg.Docker.Remove(ctx, cid)
		r.setState(deploymentID, StateFailed)
		r.drop(deploymentID)
		return nil, errors.New("readiness probe timed out")
	}

	r.mu.Lock()
	d.containerID = cid
	d.hostPort = hostPort
	d.endpointURL = endpoint
	d.plan = plan
	d.state = StateRunning
	r.mu.Unlock()

	return &LoadResult{EndpointURL: endpoint}, nil
}

// UnloadModel stops and removes the container for deploymentID. Absent is OK.
func (r *Runtime) UnloadModel(ctx context.Context, deploymentID string) error {
	r.mu.Lock()
	d, ok := r.deployments[deploymentID]
	r.mu.Unlock()
	if !ok {
		return nil
	}
	d.init.Lock()
	defer d.init.Unlock()

	r.setState(deploymentID, StateStopping)
	if d.containerID != "" {
		if err := r.cfg.Docker.Stop(ctx, d.containerID, 10); err != nil {
			return fmt.Errorf("stop: %w", err)
		}
		if err := r.cfg.Docker.Remove(ctx, d.containerID); err != nil {
			return fmt.Errorf("remove: %w", err)
		}
	}
	r.drop(deploymentID)
	return nil
}

// internal mutators -----------------------------------------------------------

func (r *Runtime) getOrCreate(deploymentID string) *deployment {
	r.mu.Lock()
	defer r.mu.Unlock()
	d, ok := r.deployments[deploymentID]
	if !ok {
		d = &deployment{state: StateAbsent}
		r.deployments[deploymentID] = d
	}
	return d
}

func (r *Runtime) setState(id string, s State) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if d, ok := r.deployments[id]; ok {
		d.state = s
	}
}

func (r *Runtime) setContainerID(id, cid string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if d, ok := r.deployments[id]; ok {
		d.containerID = cid
	}
}

func (r *Runtime) drop(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.deployments, id)
}

func (r *Runtime) waitReady(ctx context.Context, url string) bool {
	deadline := time.Now().Add(r.cfg.ReadinessTimeout)
	for time.Now().Before(deadline) {
		if r.cfg.ReadinessProbe(url) {
			return true
		}
		select {
		case <-ctx.Done():
			return false
		case <-time.After(r.cfg.ReadinessPollInterval):
		}
	}
	return false
}

// httpProbe is the default readiness probe: 200 = ready.
func httpProbe(url string) bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}
