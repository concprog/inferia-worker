package metrics

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/inferia/inferia-worker/internal/control"
)

type deploymentBucket struct {
	requestsTotal    atomic.Int64
	activeRequests   atomic.Int64
	latencyHistogram *SlidingHistogram

	recipe string
	model  string
}

type Collector struct {
	mu          sync.RWMutex
	deployments map[string]*deploymentBucket
}

func NewCollector() *Collector {
	return &Collector{
		deployments: make(map[string]*deploymentBucket),
	}
}

func (c *Collector) getBucket(id string, recipe, model string) *deploymentBucket {
	c.mu.RLock()
	b, ok := c.deployments[id]
	c.mu.RUnlock()

	if ok {
		return b
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	// Double check
	if b, ok := c.deployments[id]; ok {
		return b
	}

	b = &deploymentBucket{
		latencyHistogram: NewSlidingHistogram([]int64{
			10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000,
		}),
		recipe: recipe,
		model:  model,
	}
	c.deployments[id] = b
	return b
}

func (c *Collector) RecordRequest(id, recipe, model string, latencyMs int64) {
	b := c.getBucket(id, recipe, model)
	b.requestsTotal.Add(1)
	b.latencyHistogram.Observe(latencyMs)
}

func (c *Collector) IncActive(id string) {
	b := c.getBucket(id, "", "")
	b.activeRequests.Add(1)
}

func (c *Collector) DecActive(id string) {
	b := c.getBucket(id, "", "")
	b.activeRequests.Add(-1)
}

func (c *Collector) RemoveDeployment(id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.deployments, id)
}

// RuntimeInfo is a map payload helper to pass states from runtime to the metrics collector.
type RuntimeInfo struct {
	Recipe, Model, Phase string
	PullDur, StartDur    time.Duration
}

func (c *Collector) Snapshot(runtimeInfo map[string]RuntimeInfo) []control.DeploymentMetric {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var results []control.DeploymentMetric
	for id, b := range c.deployments {
		p50, p95 := b.latencyHistogram.Snapshot()
		
		info, ok := runtimeInfo[id]
		recipe, model, phase := b.recipe, b.model, "unknown"
		var pullDur, startDur int64

		if ok {
			recipe = info.Recipe
			model = info.Model
			phase = info.Phase
			pullDur = info.PullDur.Milliseconds()
			startDur = info.StartDur.Milliseconds()
		}

		results = append(results, control.DeploymentMetric{
			DeploymentID:        id,
			Recipe:              recipe,
			Model:               model,
			RequestsTotal:       b.requestsTotal.Swap(0),
			ActiveRequests:      b.activeRequests.Load(),
			RequestLatencyP50Ms: p50,
			RequestLatencyP95Ms: p95,
			PullDurationMs:      pullDur,
			StartDurationMs:     startDur,
			Phase:               phase,
		})
		b.latencyHistogram.Reset()
	}

	return results
}
