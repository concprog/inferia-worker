package telemetry

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// CPUInfo summarises CPU usage across all cores.
type CPUInfo struct {
	Cores       int
	UsedPercent float64 // 0..100
}

type reader func() (string, error)

func readProcStat() (string, error) {
	b, err := os.ReadFile("/proc/stat")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// ReadCPU samples /proc/stat twice (sleep ~250ms in between) to compute a busy
// percentage averaged over the interval.
func ReadCPU() (CPUInfo, error) {
	first := readProcStat
	second := func() (string, error) {
		time.Sleep(250 * time.Millisecond)
		return readProcStat()
	}
	return readCPUFrom(first, second)
}

func readCPUFrom(first, second reader) (CPUInfo, error) {
	a, err := first()
	if err != nil {
		return CPUInfo{}, err
	}
	t1, i1, err := parseProcStat(a)
	if err != nil {
		return CPUInfo{}, err
	}
	b, err := second()
	if err != nil {
		return CPUInfo{}, err
	}
	t2, i2, err := parseProcStat(b)
	if err != nil {
		return CPUInfo{}, err
	}
	cores := countCoreLines(a)
	return CPUInfo{
		Cores:       cores,
		UsedPercent: cpuDelta(t1, i1, t2, i2),
	}, nil
}

// parseProcStat extracts the aggregate cpu line from /proc/stat and returns
// (total_jiffies, idle_jiffies). Fields: user nice system idle iowait irq softirq steal guest guest_nice
// (idle is field index 3).
func parseProcStat(s string) (uint64, uint64, error) {
	for _, line := range strings.Split(s, "\n") {
		if !strings.HasPrefix(line, "cpu ") && line != "cpu" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 5 {
			return 0, 0, fmt.Errorf("malformed cpu line: %q", line)
		}
		var total uint64
		for i := 1; i < len(fields); i++ {
			n, err := strconv.ParseUint(fields[i], 10, 64)
			if err != nil {
				return 0, 0, fmt.Errorf("malformed cpu line value %q: %v", fields[i], err)
			}
			total += n
		}
		idle, err := strconv.ParseUint(fields[4], 10, 64)
		if err != nil {
			return 0, 0, fmt.Errorf("malformed idle: %v", err)
		}
		return total, idle, nil
	}
	return 0, 0, errors.New("no aggregate 'cpu ' line found in /proc/stat")
}

func countCoreLines(s string) int {
	n := 0
	for _, line := range strings.Split(s, "\n") {
		if len(line) < 4 {
			continue
		}
		if !strings.HasPrefix(line, "cpu") {
			continue
		}
		// Per-core lines look like "cpu0 ", "cpu1 ", ... The aggregate "cpu " is excluded.
		rest := line[3:]
		if rest == "" || rest[0] == ' ' {
			continue
		}
		if len(rest) > 0 && rest[0] >= '0' && rest[0] <= '9' {
			n++
		}
	}
	return n
}

// cpuDelta returns busy-percent in 0..100. Treats invalid samples (no time
// elapsed, monotonic violations) as 0%.
func cpuDelta(t1, i1, t2, i2 uint64) float64 {
	if t2 <= t1 {
		return 0
	}
	dTotal := t2 - t1
	if i2 < i1 {
		i2 = i1
	}
	dIdle := i2 - i1
	busy := float64(dTotal-dIdle) / float64(dTotal)
	if busy < 0 {
		busy = 0
	}
	if busy > 1 {
		busy = 1
	}
	return busy * 100
}
