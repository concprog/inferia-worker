package inference

import (
	"sync"

	"github.com/inferia/inferia-worker/internal/runtime"
)

// DeploymentMode distinguishes single-container from disagg deployments.
type DeploymentMode int

const (
	ModeSingle DeploymentMode = iota
	ModeDisagg
)

// DeploymentEntry is what the registry stores for a single deployment ID.
type DeploymentEntry struct {
	Mode         DeploymentMode
	DeploymentID string
	Model        string
	Group        *runtime.DeploymentGroup // non-nil only for ModeDisagg
}

// DeploymentRegistry maps deployment IDs to their P/D endpoints.
// The dispatcher writes (Register/Deregister); the proxy reads (Lookup).
type DeploymentRegistry struct {
	mu   sync.RWMutex
	byID map[string]*DeploymentEntry
}

// NewDeploymentRegistry creates an empty registry.
func NewDeploymentRegistry() *DeploymentRegistry {
	return &DeploymentRegistry{byID: map[string]*DeploymentEntry{}}
}

// RegisterDisagg stores a disagg deployment group under id.
func (r *DeploymentRegistry) RegisterDisagg(id, model string, group *runtime.DeploymentGroup) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.byID[id] = &DeploymentEntry{
		Mode:         ModeDisagg,
		DeploymentID: id,
		Model:        model,
		Group:        group,
	}
}

// Deregister removes id from the registry. Safe to call for unknown ids.
func (r *DeploymentRegistry) Deregister(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.byID, id)
}

// LookupByID returns the entry for id, or false if not registered.
func (r *DeploymentRegistry) LookupByID(id string) (*DeploymentEntry, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.byID[id]
	return e, ok
}
