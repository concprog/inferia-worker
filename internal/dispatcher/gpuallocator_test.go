package dispatcher

import (
	"sort"
	"testing"
)

func TestAllocator_AllFree_GivesDesirable(t *testing.T) {
	a := NewGPUAllocator(8)
	got := a.Allocate("dep-a", []int{0, 1, 2})
	if len(got) != 3 || got[0] != 0 || got[1] != 1 || got[2] != 2 {
		t.Errorf("Allocate({0,1,2}) = %v, want [0 1 2]", got)
	}
	used := a.Used()
	sort.Ints(used)
	if len(used) != 3 || used[0] != 0 || used[1] != 1 || used[2] != 2 {
		t.Errorf("Used() = %v, want [0 1 2]", used)
	}
}

func TestAllocator_MultipleDeploymentsShareGPUs(t *testing.T) {
	a := NewGPUAllocator(8)
	_ = a.Allocate("dep-a", []int{0, 1})
	_ = a.Allocate("dep-b", []int{2, 3})
	got := a.Allocate("dep-c", []int{0, 1, 2, 3, 4})
	// All deployments get their full desirable set under the static-equal model.
	if len(got) != 5 {
		t.Errorf("Allocate({0..4}) = %v, want 5 entries", got)
	}
	used := a.Used()
	sort.Ints(used)
	// All three deployments are tracked.
	if len(used) != 5 {
		t.Errorf("Used() = %v, want 5 entries (0-4)", used)
	}
}

func TestAllocator_AllBusy_ReturnsAllDesirable(t *testing.T) {
	a := NewGPUAllocator(8)
	_ = a.Allocate("dep-a", []int{0, 1})
	_ = a.Allocate("dep-b", []int{2, 3})
	got := a.Allocate("dep-c", []int{0, 1, 2, 3})
	if len(got) != 4 {
		t.Errorf("Allocate({0..3}) all busy = %v, want [0 1 2 3]", got)
	}
	used := a.Used()
	sort.Ints(used)
	if len(used) != 4 {
		t.Errorf("Used() = %v, want 4 entries (0-3)", used)
	}
}

func TestAllocator_Release_FreesGpus(t *testing.T) {
	a := NewGPUAllocator(8)
	_ = a.Allocate("dep-a", []int{0, 1})
	_ = a.Allocate("dep-b", []int{2, 3})
	a.Release("dep-a")
	used := a.Used()
	sort.Ints(used)
	if len(used) != 2 || used[0] != 2 || used[1] != 3 {
		t.Errorf("Used() after release dep-a = %v, want [2 3]", used)
	}
}

func TestAllocator_ReleaseIdempotent(t *testing.T) {
	a := NewGPUAllocator(8)
	_ = a.Allocate("dep-a", []int{0})
	a.Release("dep-a")
	a.Release("dep-a")
	a.Release("nonexistent")
}

func TestAllocator_AllocateIdempotentKey(t *testing.T) {
	a := NewGPUAllocator(8)
	_ = a.Allocate("dep-a", []int{0, 1})
	got := a.Allocate("dep-a", []int{2, 3})
	if len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Errorf("re-Allocate same id = %v, want [2 3]", got)
	}
	used := a.Used()
	sort.Ints(used)
	if len(used) != 2 || used[0] != 2 || used[1] != 3 {
		t.Errorf("Used() after re-Allocate = %v, want [2 3]", used)
	}
}

func TestAllocator_EmptyDesirable_NoGPUs(t *testing.T) {
	a := NewGPUAllocator(0)
	got := a.Allocate("dep-a", nil)
	if got != nil {
		t.Errorf("Allocate(nil) on no-GPU host = %v, want nil", got)
	}
	got = a.Allocate("dep-a", []int{})
	if len(got) != 0 {
		t.Errorf("Allocate(empty) on no-GPU host = %v, want empty", got)
	}
}

func TestAllocator_AutoFillDesirable(t *testing.T) {
	a := NewGPUAllocator(4)
	got := a.Allocate("dep-a", nil)
	if len(got) != 4 {
		t.Errorf("Allocate(nil) on 4-GPU host = %v, want [0 1 2 3]", got)
	}
	sort.Ints(got)
	for i, g := range got {
		if g != i {
			t.Errorf("Allocate(nil)[%d] = %d, want %d", i, g, i)
		}
	}
	got = a.Allocate("dep-b", []int{})
	if len(got) != 4 {
		t.Errorf("Allocate(empty) on 4-GPU host = %v, want [0 1 2 3]", got)
	}
}

func TestAllocator_UsedEmptyWhenNoAllocs(t *testing.T) {
	a := NewGPUAllocator(8)
	used := a.Used()
	if len(used) != 0 {
		t.Errorf("Used() on fresh allocator = %v, want empty", used)
	}
}

func TestAllocator_LeastLoadedGPUsPreferred(t *testing.T) {
	a := NewGPUAllocator(8)
	// dep-a gets 2 GPUs; dep-b gets 2 GPUs; both share [0-3].
	_ = a.Allocate("dep-a", []int{0, 1, 2, 3})
	_ = a.Allocate("dep-b", []int{0, 1, 2, 3})
	// dep-c requests [0,2,4]; allocator picks least-loaded among the three.
	got := a.Allocate("dep-c", []int{0, 2, 4})
	if len(got) != 3 {
		t.Errorf("Allocate({0,2,4}) = %v, want 3 entries", got)
	}
	// 4 was never used, so it must be included.
	has4 := false
	for _, g := range got {
		if g == 4 {
			has4 = true
			break
		}
	}
	if !has4 {
		t.Errorf("Allocate({0,2,4}) = %v, want GPU 4 included (least loaded)", got)
	}
}

func TestAllocator_SubsequentAllocSeesReleasedGpus(t *testing.T) {
	a := NewGPUAllocator(8)
	_ = a.Allocate("dep-a", []int{0, 1})
	_ = a.Allocate("dep-b", []int{0, 1})
	a.Release("dep-a")
	got := a.Allocate("dep-d", []int{0, 1})
	if len(got) != 2 {
		t.Errorf("Allocate({0,1}) after dep-a released = %v, want [0 1]", got)
	}
	a.Release("dep-d")
	got = a.Allocate("dep-e", []int{0, 1})
	if len(got) != 2 {
		t.Errorf("Allocate({0,1}) after all released = %v, want [0 1]", got)
	}
}

func TestStaticEqualAllocator_ExactFit(t *testing.T) {
	alloc := &StaticEqualAllocator{}
	agents := []Agent{
		{Name: "a", MinResource: 2},
		{Name: "b", MinResource: 2},
	}
	plan, err := alloc.AllocateStatic(agents, 4)
	if err != nil {
		t.Fatalf("AllocateStatic(2+2, 4) = %v", err)
	}
	if plan.TotalUsed != 4 {
		t.Errorf("TotalUsed = %v, want 4", plan.TotalUsed)
	}
	if plan.Allocations["a"] != 2 || plan.Allocations["b"] != 2 {
		t.Errorf("Allocations = %v, want {a:2, b:2}", plan.Allocations)
	}
}

func TestStaticEqualAllocator_OvercapacityFails(t *testing.T) {
	alloc := &StaticEqualAllocator{}
	agents := []Agent{
		{Name: "a", MinResource: 3},
		{Name: "b", MinResource: 3},
	}
	_, err := alloc.AllocateStatic(agents, 4)
	if err == nil {
		t.Fatal("AllocateStatic(3+3, 4) should fail (6 > 4)")
	}
}

func TestStaticEqualAllocator_SlackRedistribution(t *testing.T) {
	alloc := &StaticEqualAllocator{}
	agents := []Agent{
		{Name: "big", MinResource: 3},
		{Name: "small1", MinResource: 0.5},
		{Name: "small2", MinResource: 0.5},
	}
	plan, err := alloc.AllocateStatic(agents, 5)
	if err != nil {
		t.Fatalf("AllocateStatic(3+0.5+0.5, 5) = %v", err)
	}
	// Equal share = 5/3 ≈ 1.67
	// big: max(1.67, 3) = 3, small1: max(1.67, 0.5) = 1.67, small2: max(1.67, 0.5) = 1.67
	// Total = 3 + 1.67 + 1.67 = 6.34 > 5
	// absoluteMinSum = 3 + 0.5 + 0.5 = 4
	// remaining = 5 - 4 = 1
	// slackReceivers = 2 (small1, small2)
	// slackShare = 1 / 2 = 0.5
	// big: 3, small1: 0.5 + 0.5 = 1.0, small2: 0.5 + 0.5 = 1.0
	// Total = 3 + 1 + 1 = 5  ✓
	if plan.TotalUsed != 5 {
		t.Errorf("TotalUsed = %v, want 5", plan.TotalUsed)
	}
	if plan.Allocations["big"] != 3 {
		t.Errorf("big = %v, want 3", plan.Allocations["big"])
	}
	if plan.Allocations["small1"] != 1.0 || plan.Allocations["small2"] != 1.0 {
		t.Errorf("small allocations = %v, want {small1:1, small2:1}", plan.Allocations)
	}
}

func TestStaticEqualAllocator_EmptyAgents(t *testing.T) {
	alloc := &StaticEqualAllocator{}
	plan, err := alloc.AllocateStatic(nil, 10)
	if err != nil {
		t.Fatalf("AllocateStatic(nil, 10) = %v", err)
	}
	if plan.TotalUsed != 0 {
		t.Errorf("TotalUsed = %v, want 0", plan.TotalUsed)
	}
}