// Package dockerclient wraps the Docker SDK with the narrow surface the worker
// runtime needs. It also exposes a pure BuildContainerSpec function that
// translates a recipes.Plan into a host-agnostic ContainerSpec — the SDK calls
// then turn that into the corresponding container.Config / HostConfig.
package dockerclient

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/inferia/inferia-worker/internal/runtime/recipes"
)

// CDINvidiaVendor is the CDI vendor prefix for NVIDIA GPUs.
const CDINvidiaVendor = "nvidia.com/gpu"

// ContainerSpec is a wire-format-agnostic description of a container we'd
// like Docker to run. Keeps the dockerclient testable without importing the
// SDK types into tests.
type ContainerSpec struct {
	Name            string
	Image           string
	Cmd             []string
	Entrypoint      []string // overrides image entrypoint; nil → use image default
	Env             map[string]string
	Mounts          []Mount
	PortBinding     PortBinding
	NetworkName     string
	RestartPolicy   string
	Labels          map[string]string
	GPUDeviceIDs    []string
	GPUCapabilities [][]string // capability groups; each group is ANDed, groups are ORed
	ShmSize         int64       // shared memory in bytes; 0 = daemon default
}

// Mount mirrors recipes.Mount in dockerclient types to avoid importing
// the recipes package into client tests.
type Mount struct {
	Type     string
	Source   string
	Target   string
	ReadOnly bool
}

// PortBinding describes one host:container port mapping. We always bind to
// 127.0.0.1 so the model server is reachable only by the worker proxy.
type PortBinding struct {
	HostIP        string
	HostPort      string
	ContainerPort string
}

// BuildContainerSpec validates the plan and returns the spec.
func BuildContainerSpec(p recipes.Plan, networkName string) (*ContainerSpec, error) {
	if p.Image == "" {
		return nil, errors.New("plan.Image is required")
	}
	if p.ContainerName == "" {
		return nil, errors.New("plan.ContainerName is required")
	}
	if p.ContainerPort <= 0 {
		return nil, errors.New("plan.ContainerPort must be > 0")
	}
	if p.HostPort <= 0 {
		return nil, errors.New("plan.HostPort must be > 0")
	}

	deviceIDs := make([]string, 0, len(p.GPUIndices))
	for _, i := range p.GPUIndices {
		deviceIDs = append(deviceIDs, CDINvidiaVendor+"="+strconv.Itoa(i))
	}

	labels := map[string]string{
		"inferia.managed_by":    "inferia-worker",
		"inferia.deployment_id": labelDeploymentID(p.ContainerName),
	}

	mounts := make([]Mount, len(p.Mounts))
	for i, m := range p.Mounts {
		mounts[i] = Mount{
			Type:     m.Type,
			Source:   m.Source,
			Target:   m.Target,
			ReadOnly: m.ReadOnly,
		}
	}

	return &ContainerSpec{
		Name:        p.ContainerName,
		Image:       p.Image,
		Cmd:         p.Cmd,
		Entrypoint:  p.Entrypoint,
		Env:         p.Env,
		Mounts:      mounts,
		PortBinding: PortBinding{
			HostIP:        "127.0.0.1",
			HostPort:      strconv.Itoa(p.HostPort),
			ContainerPort: fmt.Sprintf("%d/tcp", p.ContainerPort),
		},
		NetworkName:     networkName,
		RestartPolicy:   "no",
		Labels:          labels,
		GPUDeviceIDs:    deviceIDs,
		GPUCapabilities: [][]string{{"gpu"}},
		ShmSize:         p.ShmSize,
	}, nil
}

// labelDeploymentID extracts a stable deployment id from the container name
// (which the recipe builds as <prefix>-<deploymentID>). If the prefix scheme
// changes, we fall back to the full name.
func labelDeploymentID(containerName string) string {
	// Container names look like inferia-vllm-<dep-id>; we strip the prefix
	// "inferia-<recipe>-".
	prefixes := []string{
		"inferia-vllm-", "inferia-sglang-", "inferia-ollama-", "inferia-infinity-",
		"inferia-triton-", "inferia-diff-",
	}
	for _, p := range prefixes {
		if len(containerName) > len(p) && containerName[:len(p)] == p {
			return containerName[len(p):]
		}
	}
	return containerName
}
