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

	// prog buffers lifecycle/pull-progress lines surfaced as logs before the
	// container exists. Always non-nil for a registered deployment.
	prog *progressLog

	// init guards a single concurrent LoadModel for the same id.
	init sync.Mutex

	enteredAt map[State]time.Time
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
		// Matches config.defaultPullTimeout: large vLLM images (~35 GB
		// uncompressed) need well over 10 min to pull+extract on gp3.
		cfg.PullTimeout = 1800 * time.Second
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

// DeploymentInfo returns a summary of a deployment's metadata and timing.
func (r *Runtime) DeploymentInfo(deploymentID string) (recipe, model, phase string, pullDur, startDur time.Duration, ok bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	d, ok := r.deployments[deploymentID]
	if !ok {
		return "", "", "", 0, 0, false
	}

	recipe = d.plan.Image // Simplification: Use image as recipe name or extend recipes.Plan
	model = ""
	if d.plan.Env != nil {
		model = d.plan.Env["INFERIA_OLLAMA_MODEL"]
	}

	phase = fmt.Sprintf("%v", d.state)

	if t1, ok1 := d.enteredAt[StatePulling]; ok1 {
		if t2, ok2 := d.enteredAt[StateStarting]; ok2 {
			pullDur = t2.Sub(t1)
		}
	}
	if t2, ok2 := d.enteredAt[StateStarting]; ok2 {
		if t3, ok3 := d.enteredAt[StateRunning]; ok3 {
			startDur = t3.Sub(t2)
		}
	}

	return recipe, model, phase, pullDur, startDur, true
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

// ContainerForDeployment returns the docker container ID currently running
// the given deployment, or "" if no such container exists. Used by the
// admin logs / shell endpoints to target the right container.
func (r *Runtime) ContainerForDeployment(deploymentID string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	d, ok := r.deployments[deploymentID]
	if !ok {
		return ""
	}
	return d.containerID
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

	// Pull. Surface progress as deployment logs (visible in the dashboard's
	// terminal while the container does not yet exist).
	r.setState(deploymentID, StatePulling)
	d.prog.append("Pulling image " + plan.Image + " …")
	pullCtx, cancel := context.WithTimeout(ctx, r.cfg.PullTimeout)
	if err := r.cfg.Docker.Pull(pullCtx, plan.Image, func(line string) {
		d.prog.append(line)
	}); err != nil {
		cancel()
		d.prog.append("Failed: pull: " + err.Error())
		d.prog.markFailed()
		r.setState(deploymentID, StateFailed)
		r.drop(deploymentID)
		return nil, fmt.Errorf("pull: %w", err)
	}
	cancel()
	d.prog.append("Image ready. Creating container…")

	// Create + start.
	spec, err := dockerclient.BuildContainerSpec(plan, r.cfg.Network)
	if err != nil {
		d.prog.append("Failed: spec: " + err.Error())
		d.prog.markFailed()
		r.setState(deploymentID, StateFailed)
		r.drop(deploymentID)
		return nil, fmt.Errorf("spec: %w", err)
	}
	// Idempotency: a prior load attempt may have left a container occupying this
	// deployment's FIXED name (inferia-<recipe>-<deployment_id>). That happens
	// when the control plane restarts and cancels the request ctx mid-load — the
	// failure-path cleanup below then no-ops on the dead ctx — or when the worker
	// process itself restarts before cleanup. Docker rejects the next create with
	// "container name already in use" and the deploy is stuck DEPLOYING forever
	// with no running container. Force-remove any stale same-name container first
	// so a re-drive (or retry) always recreates cleanly. Missing container is a
	// no-op; a real removal error is surfaced as a log but not fatal — the create
	// below reports the actual conflict if the stale container truly survived.
	if err := r.cfg.Docker.RemoveByName(ctx, spec.Name); err != nil {
		d.prog.append("Note: could not clear stale container " + spec.Name + ": " + err.Error())
	}

	cid, err := r.cfg.Docker.Create(ctx, spec)
	if err != nil {
		d.prog.append("Failed: create: " + err.Error())
		d.prog.markFailed()
		r.setState(deploymentID, StateFailed)
		r.drop(deploymentID)
		return nil, fmt.Errorf("create: %w", err)
	}
	r.setContainerID(deploymentID, cid)

	r.setState(deploymentID, StateStarting)
	d.prog.append("Starting container…")
	if err := r.cfg.Docker.Start(ctx, cid); err != nil {
		cctx, ccancel := cleanupCtx()
		_ = r.cfg.Docker.Remove(cctx, cid)
		ccancel()
		d.prog.append("Failed: start: " + err.Error())
		d.prog.markFailed()
		r.setState(deploymentID, StateFailed)
		r.drop(deploymentID)
		return nil, fmt.Errorf("start: %w", err)
	}
	// Container is up — hand off log following to real container logs.
	d.prog.append("Container started. Streaming container logs…")
	d.prog.markContainerStarted(cid)

	// Wait for readiness. Probe the model container at the SAME address the
	// inference proxy will route to: AdvertiseHost:hostPort (the host-bound
	// port). The worker runs with `--network host` (see the AWS bootstrap),
	// so it shares the host network namespace — the model container's
	// 127.0.0.1:<hostPort> host binding is reachable, but a bridge container
	// NAME is NOT resolvable from host-network. Probing by container name
	// (the previous behaviour) therefore always timed out on the AWS path →
	// the worker killed the freshly-loaded container after ReadinessTimeout.
	// Probing the host-bound endpoint keeps probe + proxy consistent.
	endpoint := fmt.Sprintf("http://%s:%d", r.cfg.AdvertiseHost, hostPort)
	probeURL := endpoint + plan.ReadyPath
	if !r.waitReady(ctx, probeURL) {
		cctx, ccancel := cleanupCtx()
		_ = r.cfg.Docker.Stop(cctx, cid, 5)
		_ = r.cfg.Docker.Remove(cctx, cid)
		ccancel()
		r.setState(deploymentID, StateFailed)
		r.drop(deploymentID)
		return nil, errors.New("readiness probe timed out")
	}

	if err := ollamaPullIfNeeded(ctx, plan, endpoint, r.cfg.PullTimeout); err != nil {
		cctx, ccancel := cleanupCtx()
		_ = r.cfg.Docker.Stop(cctx, cid, 5)
		_ = r.cfg.Docker.Remove(cctx, cid)
		ccancel()
		r.setState(deploymentID, StateFailed)
		r.drop(deploymentID)
		return nil, fmt.Errorf("ollama pull-after-ready: %w", err)
	}

	r.mu.Lock()
	d.containerID = cid
	d.hostPort = hostPort
	d.endpointURL = endpoint
	d.plan = plan
	r.mu.Unlock()

	r.setState(deploymentID, StateRunning)

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

// cleanupCtx returns a short-lived context for best-effort teardown (Stop /
// Remove) that MUST run even when the caller's ctx was cancelled — e.g. a
// control-plane restart broke the control channel and cancelled the load
// request mid-flight. Using the (possibly cancelled) request ctx for cleanup
// would no-op the Stop/Remove and leak the container, which is the root cause
// of the "container name already in use" stuck-DEPLOYING failure.
func cleanupCtx() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 30*time.Second)
}

// internal mutators -----------------------------------------------------------

func (r *Runtime) getOrCreate(deploymentID string) *deployment {
	r.mu.Lock()
	defer r.mu.Unlock()
	d, ok := r.deployments[deploymentID]
	if !ok {
		d = &deployment{
			state:     StateAbsent,
			prog:      newProgressLog(),
			enteredAt: make(map[State]time.Time),
		}
		r.deployments[deploymentID] = d
	} else if d.prog == nil {
		d.prog = newProgressLog()
	}
	return d
}

func (r *Runtime) setState(id string, s State) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if d, ok := r.deployments[id]; ok {
		d.state = s
		d.enteredAt[s] = time.Now()
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
