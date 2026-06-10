package runtime

import (
	"fmt"
	"sync/atomic"

	"github.com/inferia/inferia-worker/internal/runtime/recipes"
)

// ContainerInfo describes one container within a deployment group.
type ContainerInfo struct {
	ContainerID string
	HostPort    int
	Role        recipes.KvRole
	ReplicaIdx  int
}

// DeploymentGroup groups together the P and D containers for a single
// disagg deployment. Used for lifecycle management and round-robin
// endpoint selection by the proxy.
type DeploymentGroup struct {
	ID          string
	Model       string
	Prefill     []ContainerInfo
	Decode      []ContainerInfo
	prefillNext atomic.Uint32
	decodeNext  atomic.Uint32
}

// NextPrefill returns the next prefill endpoint in round-robin order.
func (g *DeploymentGroup) NextPrefill() string {
	if len(g.Prefill) == 0 {
		return ""
	}
	i := g.prefillNext.Add(1) - 1
	ci := g.Prefill[int(i)%len(g.Prefill)]
	return fmt.Sprintf("http://127.0.0.1:%d", ci.HostPort)
}

// NextDecode returns the next decode endpoint in round-robin order.
func (g *DeploymentGroup) NextDecode() string {
	if len(g.Decode) == 0 {
		return ""
	}
	i := g.decodeNext.Add(1) - 1
	ci := g.Decode[int(i)%len(g.Decode)]
	return fmt.Sprintf("http://127.0.0.1:%d", ci.HostPort)
}
