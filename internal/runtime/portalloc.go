package runtime

import (
	"errors"
	"sync"
)

// PortAllocator hands out host ports from [lo, hi]. Released ports are recycled.
type PortAllocator struct {
	lo, hi int
	mu     sync.Mutex
	used   map[int]bool
}

// NewPortAllocator constructs an allocator over [lo, hi].
func NewPortAllocator(lo, hi int) *PortAllocator {
	return &PortAllocator{lo: lo, hi: hi, used: map[int]bool{}}
}

// Next returns the smallest unused port in range, or an error if exhausted.
func (a *PortAllocator) Next() (int, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for p := a.lo; p <= a.hi; p++ {
		if !a.used[p] {
			a.used[p] = true
			return p, nil
		}
	}
	return 0, errors.New("port range exhausted")
}

// Release marks a port available again. Releasing a port that was never
// allocated is a no-op.
func (a *PortAllocator) Release(p int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.used, p)
}
