package recipes

import "fmt"

// KvRole is the mooncake KV cache role assigned to a container in a
// disagg deployment group.
type KvRole string

const (
	KvRoleProducer KvRole = "kv_producer"
	KvRoleConsumer KvRole = "kv_consumer"
)

// ContainerPlan describes one container within a multi-container deployment.
// Fields mirror Plan but omit ContainerName and HostPort — the runtime
// generates those dynamically.
type ContainerPlan struct {
	Image         string
	Cmd           []string
	Entrypoint    []string
	Env           map[string]string
	Mounts        []Mount
	ContainerPort int
	GPUIndices    []int
	ReadyPath     string
	Role          KvRole
	ReplicaIdx    int
}

// ToPlan converts the ContainerPlan into a runnable Plan by filling in the
// runtime-assigned container name and host port.
func (cp ContainerPlan) ToPlan(containerName string, hostPort int) Plan {
	return Plan{
		Image:         cp.Image,
		ContainerName: containerName,
		Cmd:           cp.Cmd,
		Entrypoint:    cp.Entrypoint,
		Env:           cp.Env,
		Mounts:        cp.Mounts,
		ContainerPort: cp.ContainerPort,
		HostPort:      hostPort,
		GPUIndices:    cp.GPUIndices,
		ReadyPath:     cp.ReadyPath,
	}
}

// DeploymentPlan is the blueprint for a full disagg deployment group.
// The runtime uses it to launch N prefill + M decode containers.
type DeploymentPlan struct {
	DeploymentID string
	Model        string
	Prefill      []ContainerPlan
	Decode       []ContainerPlan
}

// MultiContainerBuilder is the interface a recipe satisfies when it
// produces multiple container plans (e.g. vllm-prefill-decode).
// The dispatcher type-asserts Recipe to this interface to detect
// disagg deployments.
type MultiContainerBuilder interface {
	BuildDeploymentPlan(in BuildInput) (DeploymentPlan, error)
}

// MultiGet returns the MultiContainerBuilder registered under name.
func MultiGet(name string) (MultiContainerBuilder, error) {
	r, ok := multiRegistry[name]
	if !ok {
		return nil, fmt.Errorf("unknown multi-container recipe: %q", name)
	}
	return r, nil
}

// multiRegistry holds recipes that produce multiple container plans.
var multiRegistry = map[string]MultiContainerBuilder{}
