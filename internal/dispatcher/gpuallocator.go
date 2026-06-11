package dispatcher

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

func makeRange(start, end int) []int {
	n := end - start + 1
	r := make([]int, n)
	for i := range r {
		r[i] = start + i
	}
	return r
}

type Agent struct {
	Name        string
	MinResource float64 // Hard static floor (e.g., VRAM size required)
}

// StaticDeploymentPlan holds the immutable deployment configuration
type StaticDeploymentPlan struct {
	Allocations map[string]float64
	TotalUsed   float64
}

type GPUAllocatorInterface interface {
	AllocateStatic(agents []Agent, poolCapacity float64) (*StaticDeploymentPlan, error)
}

type StaticEqualAllocator struct{}

func (a *StaticEqualAllocator) AllocateStatic(agents []Agent, poolCapacity float64) (*StaticDeploymentPlan, error) {
	allocations := make(map[string]float64)
	numAgents := len(agents)

	if numAgents == 0 {
		return &StaticDeploymentPlan{Allocations: allocations, TotalUsed: 0}, nil
	}

	// 1. Hard Admission Check (OOM Prevention)
	var absoluteMinSum float64 = 0.0
	for _, agent := range agents {
		absoluteMinSum += agent.MinResource
	}

	if absoluteMinSum > poolCapacity {
		return nil, fmt.Errorf("Admission failed: absolute minimum VRAM required (%.2f) exceeds pool capacity (%.2f)", absoluteMinSum, poolCapacity)
	}

	// 2. Calculate Base Equal Share
	equalShare := poolCapacity / float64(numAgents)
	var gAllocated float64 = 0.0
	tempAllocations := make([]float64, numAgents)

	// 3. Attempt Equal Distribution with Minimum Floors
	for i, agent := range agents {
		tempAllocations[i] = math.Max(equalShare, agent.MinResource)
		gAllocated += tempAllocations[i]
	}

	// 4. Static Constraint Resolution
	if gAllocated > poolCapacity {
		// We have enough for the absolute minimums, but not enough for the calculated equal shares.
		// Since deployments are static, we cannot use proportional scaling (which might drop an agent below MinResource).
		// Instead, we lock in the minimums and divide the remaining slack equally.
		remainingCapacity := poolCapacity - absoluteMinSum

		// Find how many agents were relying on the equal share rather than their minimum
		var slackReceivers int
		for _, agent := range agents {
			if equalShare > agent.MinResource {
				slackReceivers++
			}
		}

		slackShare := 0.0
		if slackReceivers > 0 {
			slackShare = remainingCapacity / float64(slackReceivers)
		}

		gAllocated = 0.0
		for _, agent := range agents {
			if equalShare > agent.MinResource {
				allocations[agent.Name] = agent.MinResource + slackShare
			} else {
				allocations[agent.Name] = agent.MinResource // Locked to hard minimum
			}
			gAllocated += allocations[agent.Name]
		}
	} else {
		// Pool is large enough to handle the pure equal split
		for i, agent := range agents {
			allocations[agent.Name] = tempAllocations[i]
		}
	}

	return &StaticDeploymentPlan{
		Allocations: allocations,
		TotalUsed:   gAllocated,
	}, nil
}

// GPUAllocator tracks GPU device assignment across deployments.
// It uses a static equal allocator logic internally to distribute shares
// and assigns discrete physical GPUs based on lease loads.
type GPUAllocator struct {
	mu             sync.Mutex
	allocated      map[string][]int // deploymentID -> GPU indices currently held
	equalAllocator GPUAllocatorInterface
	totalGPUs      int // total physical GPUs on host; 0 = no GPUs
}

// NewGPUAllocator creates an allocator that knows about totalGPUs on the host.
// When totalGPUs > 0 and a call to Allocate receives an empty desirable set, the
// allocator auto-generates [0..totalGPUs-1] so the deployment always gets GPUs.
func NewGPUAllocator(totalGPUs int) *GPUAllocator {
	return &GPUAllocator{
		allocated:      make(map[string][]int),
		equalAllocator: &StaticEqualAllocator{},
		totalGPUs:      totalGPUs,
	}
}

// Allocate assigns physical GPUs to a deployment. When desirable is empty but
// the host has GPUs (totalGPUs > 0), it auto-fills desirable to all available
// GPUs so the deployment never silently falls back to CPU-only.
func (a *GPUAllocator) Allocate(id string, desirable []int) []int {
	a.mu.Lock()
	defer a.mu.Unlock()

	if len(desirable) == 0 {
		if a.totalGPUs > 0 {
			desirable = makeRange(0, a.totalGPUs-1)
		} else {
			return desirable
		}
	}

	// 1. Build Agent list representing active deployments + the new one
	agents := []Agent{}
	for depID, gpus := range a.allocated {
		// Avoid duplicate registration if client retries
		if depID == id {
			continue
		}
		agents = append(agents, Agent{Name: depID, MinResource: float64(len(gpus))})
	}
	agents = append(agents, Agent{Name: id, MinResource: float64(len(desirable))})

	// Capacity can be dynamically inferred. To be extremely robust across different
	// platforms and environments, we calculate poolCapacity.
	// We can set poolCapacity as the max index seen + 1, or a default capacity.
	// Let's find the max physical GPU index requested/used to determine total pool size.
	maxIdx := 0
	for _, g := range desirable {
		if g > maxIdx {
			maxIdx = g
		}
	}
	for _, gpus := range a.allocated {
		for _, g := range gpus {
			if g > maxIdx {
				maxIdx = g
			}
		}
	}
	poolCapacity := float64(maxIdx + 1)
	if poolCapacity < 1 {
		poolCapacity = 1
	}

	// 2. Evaluate equal allocation plan
	plan, err := a.equalAllocator.AllocateStatic(agents, poolCapacity)
	if err != nil {
		// Admission failed (OOM/Capacity prevention), but we still allow sharing as fallback.
		// When sharing under pressure, we just share the desirable set directly.
		a.allocated[id] = append([]int{}, desirable...)
		return desirable
	}

	// 3. Map fractional shares to physical GPU indices by picking the least loaded desirable GPUs
	loads := make(map[int]float64)
	for depID, gpus := range a.allocated {
		if depID == id {
			continue
		}
		share := plan.Allocations[depID]
		gpuShare := share / float64(len(gpus))
		for _, g := range gpus {
			loads[g] += gpuShare
		}
	}

	// Sort desirable indices by current load (ascending)
	sortedDesirable := append([]int{}, desirable...)
	sort.Slice(sortedDesirable, func(i, j int) bool {
		return loads[sortedDesirable[i]] < loads[sortedDesirable[j]]
	})

	// Select the top len(desirable) least-loaded physical GPUs
	selected := sortedDesirable[:len(desirable)]
	a.allocated[id] = append([]int{}, selected...)

	return selected
}

// Release drops any allocation held by id.
func (a *GPUAllocator) Release(id string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.allocated, id)
}

// Used returns the set of GPUs held by any deployment.
func (a *GPUAllocator) Used() []int {
	a.mu.Lock()
	defer a.mu.Unlock()
	seen := map[int]struct{}{}
	for _, gpus := range a.allocated {
		for _, g := range gpus {
			seen[g] = struct{}{}
		}
	}
	out := make([]int, 0, len(seen))
	for g := range seen {
		out = append(out, g)
	}
	sort.Ints(out)
	return out
}
