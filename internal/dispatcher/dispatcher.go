// Package dispatcher adapts the runtime to the control.Dispatcher interface.
// Split into its own package so cmd/worker stays a thin wiring layer and the
// adapter logic gets unit coverage.
package dispatcher

import (
	"context"
	"fmt"
	"time"

	"github.com/inferia/inferia-worker/internal/control"
	"github.com/inferia/inferia-worker/internal/metrics"
	"github.com/inferia/inferia-worker/internal/runtime"
	"github.com/inferia/inferia-worker/internal/runtime/recipes"
)

// DeploymentRegistrar is the narrow interface the dispatcher uses to
// register/deregister disagg deployments with the proxy registry.
// Implemented by *inference.DeploymentRegistry.
type DeploymentRegistrar interface {
	RegisterDisagg(id, model string, group *runtime.DeploymentGroup)
	Deregister(id string)
}

// Runtime defines the subset of *runtime.Runtime operations needed by the
// dispatcher. Avoids packages importing each other recursively.
type Runtime interface {
	LoadModel(ctx context.Context, id string, plan recipes.Plan) (*runtime.LoadResult, error)
	UnloadModel(ctx context.Context, id string) error
	LoadedDeployments() []string
	DeploymentInfo(deploymentID string) (recipe, model, phase string, pullDur, startDur time.Duration, ok bool)
	EndpointURL(deploymentID string) string

	// Disagg (multi-container) lifecycle.
	LoadDeploymentGroup(ctx context.Context, plan recipes.DeploymentPlan) (*runtime.DeploymentGroup, error)
	UnloadDeploymentGroup(ctx context.Context, id string) error
}

// Dispatcher implements control.Dispatcher. It is the primary adapter between
// the control plane (WS channel) and the worker's local runtime.
type Dispatcher struct {
	Rt        Runtime
	Telemetry TelemetryReader
	Metrics   *metrics.Collector
	GPUName   string
	GPUMemMiB uint64
	Registry  DeploymentRegistrar // optional; nil = no disagg support
}

func NewDispatcher(rt Runtime, tel TelemetryReader, mc *metrics.Collector, gpuName string, gpuMem uint64) *Dispatcher {
	return &Dispatcher{
		Rt:        rt,
		Telemetry: tel,
		Metrics:   mc,
		GPUName:   gpuName,
		GPUMemMiB: gpuMem,
	}
}


// LoadResult mirrors runtime.LoadResult so this package doesn't import runtime
// (avoiding cycles when runtime imports dispatcher in some future refactor).
type LoadResult struct {
	EndpointURL string
}

// TelemetryReader returns one snapshot of host CPU/memory/GPU usage.
type TelemetryReader interface {
	Read() (used map[string]string)
}

// LoadModel converts the WS body into a recipes.Plan and asks the runtime to
// load it. Returning a non-nil error becomes CommandResult{status:"failed"}
// in the control package.
//
// When body.Recipe names a MultiContainerBuilder and body.PrefillReplicas > 0,
// it routes to the disagg (multi-container) path instead.
func (d *Dispatcher) LoadModel(ctx context.Context, body control.LoadModelBody) (string, error) {
	// Disagg path: multi-container prefill/decode split.
	if body.PrefillReplicas > 0 {
		mc, err := recipes.MultiGet(body.Recipe)
		if err != nil {
			return "", err
		}
		return d.loadDeploymentGroup(ctx, mc, body)
	}

	// Single-container path — unchanged.
	r, err := recipes.Get(body.Recipe)
	if err != nil {
		return "", err
	}
	port := body.Port
	if port == 0 {
		// recipes.BuildPlan rejects HostPort=0, so use a placeholder and let
		// the runtime allocator override it.
		port = 1
	}
	plan, err := r.BuildPlan(recipes.BuildInput{
		DeploymentID: body.DeploymentID,
		ArtifactURI:  body.Model.ArtifactURI,
		Config:       body.Config,
		GPUIndices:   body.GPUIndices,
		HostPort:     port,
		Env:          body.Env,
		GPUName:      d.GPUName,
		GPUMemoryMiB: d.GPUMemMiB,
	})
	if err != nil {
		return "", err
	}
	if body.Port == 0 {
		plan.HostPort = 0 // signal runtime to allocate
	}
	res, err := d.Rt.LoadModel(ctx, body.DeploymentID, plan)
	if err != nil {
		return "", err
	}
	return res.EndpointURL, nil
}

// loadDeploymentGroup builds and launches a disagg deployment group, then
// registers it with the proxy registry.
func (d *Dispatcher) loadDeploymentGroup(ctx context.Context, mc recipes.MultiContainerBuilder, body control.LoadModelBody) (string, error) {
	plan, err := mc.BuildDeploymentPlan(recipes.BuildInput{
		DeploymentID:     body.DeploymentID,
		ArtifactURI:      body.Model.ArtifactURI,
		Config:           body.Config,
		GPUIndices:       body.GPUIndices,
		HostPort:         1, // placeholder — runtime allocates per-container
		Env:              body.Env,
		GPUName:          d.GPUName,
		GPUMemoryMiB:     d.GPUMemMiB,
		PrefillReplicas:  body.PrefillReplicas,
		DecodeReplicas:   body.DecodeReplicas,
	})
	if err != nil {
		return "", err
	}
	group, err := d.Rt.LoadDeploymentGroup(ctx, plan)
	if err != nil {
		return "", err
	}
	if d.Registry != nil {
		d.Registry.RegisterDisagg(body.DeploymentID, plan.Model, group)
	}
	if len(group.Prefill) > 0 {
		return fmt.Sprintf("http://127.0.0.1:%d", group.Prefill[0].HostPort), nil
	}
	return "", nil
}

// UnloadModel stops and removes a deployment. It tries both the disagg group
// path and the single-container path — both are idempotent for absent IDs.
func (d *Dispatcher) UnloadModel(ctx context.Context, body control.UnloadModelBody) error {
	if d.Registry != nil {
		d.Registry.Deregister(body.DeploymentID)
	}
	// Try both; at most one will actually have containers.
	_ = d.Rt.UnloadDeploymentGroup(ctx, body.DeploymentID)
	err := d.Rt.UnloadModel(ctx, body.DeploymentID)
	if err == nil && d.Metrics != nil {
		d.Metrics.RemoveDeployment(body.DeploymentID)
	}
	return err
}

func (d *Dispatcher) HeartbeatSnapshot() control.HeartbeatBody {
	var used map[string]string
	if d.Telemetry != nil {
		used = d.Telemetry.Read()
	} else {
		used = map[string]string{}
	}
	models := d.Rt.LoadedDeployments()

	body := control.HeartbeatBody{
		Used:         used,
		LoadedModels: models,
	}

	if d.Metrics != nil {
		// Gather runtime info for all loaded deployments to enrich metrics
		infoMap := make(map[string]metrics.RuntimeInfo)
		for _, id := range models {
			r, m, p, pd, sd, ok := d.Rt.DeploymentInfo(id)
			if ok {
				infoMap[id] = metrics.RuntimeInfo{
					Recipe:   r,
					Model:    m,
					Phase:    p,
					PullDur:  pd,
					StartDur: sd,
				}
			}
		}
		body.DeployMetrics = d.Metrics.Snapshot(infoMap)
	}

	return body
}

// StartScraper runs a background loop that periodically scrapes vLLM metrics.
func (d *Dispatcher) StartScraper(ctx context.Context, interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				models := d.Rt.LoadedDeployments()
				for _, id := range models {
					recipe, _, _, _, _, ok := d.Rt.DeploymentInfo(id)
					if ok && recipe == "vllm" {
						endpoint := d.Rt.EndpointURL(id)
						if endpoint != "" && d.Metrics != nil {
							_ = d.Metrics.ScrapeVLLM(id, endpoint)
						}
					}
				}
			}
		}
	}()
}

// SafeFmt is a tiny convenience wrapper exposed so cmd/worker can build its

// own TelemetryReader without re-importing fmt. Kept package-private otherwise.
func SafeFmt(format string, args ...any) string { return fmt.Sprintf(format, args...) }
