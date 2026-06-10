package metrics

import (
	"sync"
)

// PeakHistogram tracks latency samples in fixed buckets, retaining only the
// last maxSamples observations in FIFO order. Unlike SlidingHistogram it is
// never time-reset, so percentiles are always based on recent requests and
// never return 0 for an idle window.
type PeakHistogram struct {
	mu         sync.Mutex
	buckets    []int64
	limits     []int64
	buffer     []int64 // ring buffer of last maxSamples values
	head       int     // next write position
	cur        int     // current number of samples in buffer
}

func NewPeakHistogram(maxSamples int, limits []int64) *PeakHistogram {
	return &PeakHistogram{
		buckets: make([]int64, len(limits)),
		limits:  limits,
		buffer:  make([]int64, maxSamples),
	}
}

func (h *PeakHistogram) bucketIndex(val int64) int {
	for i, limit := range h.limits {
		if val <= limit {
			return i
		}
	}
	return len(h.limits) - 1
}

func (h *PeakHistogram) Observe(val int64) {
	h.mu.Lock()
	defer h.mu.Unlock()

	idx := h.bucketIndex(val)
	if h.cur < len(h.buffer) {
		h.buffer[h.head] = val
		h.buckets[idx]++
		h.head++
		h.cur++
	} else {
		oldVal := h.buffer[h.head]
		h.buckets[h.bucketIndex(oldVal)]--
		h.buffer[h.head] = val
		h.buckets[idx]++
		h.head = (h.head + 1) % len(h.buffer)
	}
}

func (h *PeakHistogram) Snapshot() (p50, p95 int64) {
	h.mu.Lock()
	defer h.mu.Unlock()

	total := int64(h.cur)
	if total == 0 {
		return 0, 0
	}

	p95Target := (total*95 + 99) / 100
	p98Target := (total*98 + 99) / 100

	var sum int64
	p95Idx, p98Idx := -1, -1
	for i, count := range h.buckets {
		sum += count
		if p95Idx == -1 && sum >= p95Target {
			p95Idx = i
		}
		if p98Idx == -1 && sum >= p98Target {
			p98Idx = i
		}
	}

	if p95Idx != -1 {
		p50 = h.limits[p95Idx]
	}
	if p98Idx != -1 {
		p95 = h.limits[p98Idx]
	}
	return
}

// Reset is retained for API compatibility but should NOT be called for
// peak-based latency tracking — the whole point is to never reset.
func (h *PeakHistogram) Reset() {
	h.mu.Lock()
	defer h.mu.Unlock()
	for i := range h.buckets {
		h.buckets[i] = 0
	}
	h.head = 0
	h.cur = 0
}
