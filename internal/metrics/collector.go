package metrics

import (
	"bufio"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/inferia/inferia-worker/internal/control"
)

type deploymentBucket struct {
	requestsTotal    atomic.Int64
	activeRequests   atomic.Int64
	latencyHistogram *PeakHistogram

	recipe string
	model  string

	// vLLM specific scrape data
	vllmMetrics map[string]float64
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
		if recipe != "" {
			b.recipe = recipe
		}
		if model != "" {
			b.model = model
		}
		return b
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if b, ok := c.deployments[id]; ok {
		if recipe != "" {
			b.recipe = recipe
		}
		if model != "" {
			b.model = model
		}
		return b
	}

	b = &deploymentBucket{
		latencyHistogram: NewPeakHistogram(1000, []int64{
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

func (c *Collector) ScrapeVLLM(id string, url string) error {
	b := c.getBucket(id, "", "")
	if b.recipe != "vllm" && !strings.Contains(b.recipe, "vllm-openai") && !strings.Contains(b.recipe, "vllm-omni") {
		return nil
	}

	resp, err := http.Get(url + "/metrics")
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	c.mu.Lock()
	defer c.mu.Unlock()
	if b.vllmMetrics == nil {
		b.vllmMetrics = make(map[string]float64)
	}

	re := regexp.MustCompile(`^(vllm:\w+)\s+([\d.]+)$`)
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		matches := re.FindStringSubmatch(line)
		if len(matches) == 3 {
			val, _ := strconv.ParseFloat(matches[2], 64)
			b.vllmMetrics[matches[1]] = val
		}
	}
	return nil
}

func (c *Collector) GetVLLMMetrics(id string) map[string]float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if b, ok := c.deployments[id]; ok {
		return b.vllmMetrics
	}
	return nil
}

// RuntimeInfo is a map payload helper to pass states from runtime to the metrics collector.
type RuntimeInfo struct {
	Recipe, Model, Phase string
	PullDur, StartDur    time.Duration
}

func (c *Collector) Snapshot(runtimeInfo map[string]RuntimeInfo) []control.DeploymentMetric {
	c.mu.Lock()
	defer c.mu.Unlock()

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
			RequestsTotal:       b.requestsTotal.Load(),
			ActiveRequests:      b.activeRequests.Load(),
			RequestLatencyP50Ms: p50,
			RequestLatencyP95Ms: p95,
			PullDurationMs:      pullDur,
			StartDurationMs:     startDur,
			Phase:               phase,
			EngineMetrics:       b.vllmMetrics,
		})
	}

	return results
}
