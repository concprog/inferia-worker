package telemetry

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// MemInfo reports total and used host memory in bytes.
type MemInfo struct {
	Total uint64
	Used  uint64
}

func readProcMemInfo() (string, error) {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// ReadMemory returns total/used host memory from /proc/meminfo.
func ReadMemory() (MemInfo, error) {
	return readMemoryFrom(readProcMemInfo)
}

func readMemoryFrom(r reader) (MemInfo, error) {
	s, err := r()
	if err != nil {
		return MemInfo{}, err
	}
	total, avail, err := parseMemInfo(s)
	if err != nil {
		return MemInfo{}, err
	}
	if avail > total {
		avail = total
	}
	return MemInfo{Total: total, Used: total - avail}, nil
}

// parseMemInfo returns (total_bytes, available_bytes). If MemAvailable is
// absent (older kernels), falls back to MemFree.
func parseMemInfo(s string) (uint64, uint64, error) {
	var total, avail, free uint64
	var haveTotal, haveAvail, haveFree bool

	for _, line := range strings.Split(s, "\n") {
		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		switch strings.TrimSpace(key) {
		case "MemTotal":
			n, err := parseKB(val)
			if err != nil {
				return 0, 0, fmt.Errorf("MemTotal: %v", err)
			}
			total, haveTotal = n, true
		case "MemAvailable":
			n, err := parseKB(val)
			if err != nil {
				return 0, 0, fmt.Errorf("MemAvailable: %v", err)
			}
			avail, haveAvail = n, true
		case "MemFree":
			n, err := parseKB(val)
			if err != nil {
				return 0, 0, fmt.Errorf("MemFree: %v", err)
			}
			free, haveFree = n, true
		}
	}

	if !haveTotal {
		return 0, 0, errors.New("MemTotal missing in /proc/meminfo")
	}
	if haveAvail {
		return total, avail, nil
	}
	if haveFree {
		return total, free, nil
	}
	return total, 0, nil
}

func parseKB(s string) (uint64, error) {
	f := strings.Fields(s)
	if len(f) < 1 {
		return 0, fmt.Errorf("missing value: %q", s)
	}
	n, err := strconv.ParseUint(f[0], 10, 64)
	if err != nil {
		return 0, err
	}
	// /proc/meminfo values are in kB. Convert to bytes; saturate on overflow
	// rather than wrap.
	const kb = uint64(1024)
	const maxKB = ^uint64(0) / kb
	if n > maxKB {
		return ^uint64(0), nil
	}
	return n * kb, nil
}
