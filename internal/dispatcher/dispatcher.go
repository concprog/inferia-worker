// Package dispatcher adapts the runtime to the control.Dispatcher interface.
// Split into its own package so cmd/worker stays a thin wiring layer and the
// adapter logic gets unit coverage.
package dispatcher

import (
	"context"
	"fmt"

	"github.com/inferia/inferia-worker/internal/control"
	"github.com/inferia/inferia-worker/internal/runtime/recipes"
)

// Runtime is the narrow view of runtime.Runtime that Dispatcher needs.
// Defined as an interface so tests can supply a fake.
type Runtime interface {
	LoadModel(ctx context.Context, deploymentID string, plan recipes.Plan) (*LoadResult, error)
	UnloadModel(ctx context.Context, deploymentID string) error
	LoadedDeployments() []string
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

// Dispatcher implements control.Dispatcher.
type Dispatcher struct {
	Rt        Runtime
	Telemetry TelemetryReader
	GPUName   string // populated by main.go from telemetry.ReadGPU()
	GPUMemMiB uint64 // populated by main.go from telemetry.ReadGPU()
}

// LoadModel converts the WS body into a recipes.Plan and asks the runtime to
// load it. Returning a non-nil error becomes CommandResult{status:"failed"}
// in the control package.
func (d *Dispatcher) LoadModel(ctx context.Context, body control.LoadModelBody) (string, error) {
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

// UnloadModel is a direct passthrough.
func (d *Dispatcher) UnloadModel(ctx context.Context, body control.UnloadModelBody) error {
	return d.Rt.UnloadModel(ctx, body.DeploymentID)
}

// HeartbeatSnapshot composes the periodic heartbeat body.
func (d *Dispatcher) HeartbeatSnapshot() control.HeartbeatBody {
	var used map[string]string
	if d.Telemetry != nil {
		used = d.Telemetry.Read()
	} else {
		used = map[string]string{}
	}
	return control.HeartbeatBody{
		Used:         used,
		LoadedModels: d.Rt.LoadedDeployments(),
	}
}

// SafeFmt is a tiny convenience wrapper exposed so cmd/worker can build its
// own TelemetryReader without re-importing fmt. Kept package-private otherwise.
func SafeFmt(format string, args ...any) string { return fmt.Sprintf(format, args...) }
