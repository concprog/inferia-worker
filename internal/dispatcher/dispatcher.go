// Package dispatcher adapts the runtime to the control.Dispatcher interface.
// Split into its own package so cmd/worker stays a thin wiring layer and the
// adapter logic gets unit coverage.
package dispatcher

import (
	"context"
	"fmt"
	"strings"
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
	TotalGPUs int                // total physical GPUs on the host
	Registry  DeploymentRegistrar // optional; nil = no disagg support
	Allocator *GPUAllocator       // optional; nil = no GPU tracking
}

func NewDispatcher(rt Runtime, tel TelemetryReader, mc *metrics.Collector, gpuName string, gpuMem uint64, totalGPUs int) *Dispatcher {
	return &Dispatcher{
		Rt:        rt,
		Telemetry: tel,
		Metrics:   mc,
		GPUName:   gpuName,
		GPUMemMiB: gpuMem,
		TotalGPUs: totalGPUs,
		Allocator: NewGPUAllocator(totalGPUs),
	}
}


// LoadResult mirrors runtime.LoadResult so this package doesn't import runtime
// (avoiding cycles when runtime imports dispatcher in some future refactor).
type LoadResult struct {
	EndpointURL string
}

// TelemetryReader returns one snapshot of host usage: the legacy opaque
// `used` map (consumed by inventory) plus an optional structured metrics
// sample for the dashboard Metrics tab.
type TelemetryReader interface {
	Read() (used map[string]string, metrics *control.MetricsSample)
}

// validateGPUIndices rejects any index outside [0, totalGPUs).  Returns nil
// when totalGPUs is 0 (no GPU info available — skip validation).
func validateGPUIndices(indices []int, totalGPUs int) error {
	if totalGPUs <= 0 {
		return nil
	}
	for _, idx := range indices {
		if idx < 0 || idx >= totalGPUs {
			return fmt.Errorf("GPU index %d out of range [0, %d)", idx, totalGPUs)
		}
	}
	return nil
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

	gpuIndices := body.GPUIndices
	if err := validateGPUIndices(gpuIndices, d.TotalGPUs); err != nil {
		return "", err
	}
	if d.Allocator != nil {
		gpuIndices = d.Allocator.Allocate(body.DeploymentID, body.GPUIndices)
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
		GPUIndices:   gpuIndices,
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
	prefillDesired := body.PrefillGPUIndices
	if len(prefillDesired) == 0 {
		prefillDesired = body.GPUIndices
	}
	decodeDesired := body.DecodeGPUIndices
	if len(decodeDesired) == 0 {
		decodeDesired = body.GPUIndices
	}

	prefillGPUs := prefillDesired
	decodeGPUs := decodeDesired
	if d.Allocator != nil {
		if err := validateGPUIndices(prefillGPUs, d.TotalGPUs); err != nil {
			return "", fmt.Errorf("prefill: %w", err)
		}
		if err := validateGPUIndices(decodeGPUs, d.TotalGPUs); err != nil {
			return "", fmt.Errorf("decode: %w", err)
		}
		prefillGPUs = d.Allocator.Allocate(body.DeploymentID+"/prefill", prefillDesired)
		decodeGPUs = d.Allocator.Allocate(body.DeploymentID+"/decode", decodeDesired)
	}

	plan, err := mc.BuildDeploymentPlan(recipes.BuildInput{
		DeploymentID:       body.DeploymentID,
		ArtifactURI:        body.Model.ArtifactURI,
		Config:             body.Config,
		GPUIndices:         body.GPUIndices,
		PrefillGPUIndices:  prefillGPUs,
		DecodeGPUIndices:   decodeGPUs,
		HostPort:           1, // placeholder — runtime allocates per-container
		Env:                body.Env,
		GPUName:            d.GPUName,
		GPUMemoryMiB:       d.GPUMemMiB,
		PrefillReplicas:    body.PrefillReplicas,
		DecodeReplicas:     body.DecodeReplicas,
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
	if len(group.Prefill) == 0 {
		return "", fmt.Errorf("deployment %s: no prefill containers started", plan.DeploymentID)
	}
	return fmt.Sprintf("http://127.0.0.1:%d", group.Prefill[0].HostPort), nil
}

// UnloadModel stops and removes a deployment. It tries both the disagg group
// path and the single-container path — if either succeeded the deployment is
// gone and metrics are cleaned up. Only returns an error when both paths
// report "not found", indicating the ID was never loaded.
func (d *Dispatcher) UnloadModel(ctx context.Context, body control.UnloadModelBody) error {
	if d.Registry != nil {
		d.Registry.Deregister(body.DeploymentID)
	}
	if d.Allocator != nil {
		d.Allocator.Release(body.DeploymentID)
		d.Allocator.Release(body.DeploymentID + "/prefill")
		d.Allocator.Release(body.DeploymentID + "/decode")
	}
	disaggErr := d.Rt.UnloadDeploymentGroup(ctx, body.DeploymentID)
	singleErr := d.Rt.UnloadModel(ctx, body.DeploymentID)
	if disaggErr != nil && singleErr != nil {
		return singleErr
	}
	if d.Metrics != nil {
		d.Metrics.RemoveDeployment(body.DeploymentID)
	}
	return nil
}

func (d *Dispatcher) HeartbeatSnapshot() control.HeartbeatBody {
	var used map[string]string
	var sample *control.MetricsSample
	if d.Telemetry != nil {
		used, sample = d.Telemetry.Read()
	} else {
		used = map[string]string{}
	}
	models := d.Rt.LoadedDeployments()

	body := control.HeartbeatBody{
		Used:         used,
		LoadedModels: models,
		Metrics:      sample,
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
					if ok && (recipe == "vllm" || strings.Contains(recipe, "vllm-openai") || strings.Contains(recipe, "vllm-omni")) {
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
