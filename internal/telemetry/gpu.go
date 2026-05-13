package telemetry

import (
	"os/exec"
	"strconv"
	"strings"
)

// GPUInfo describes one GPU's current memory state.
type GPUInfo struct {
	Name           string
	MemoryTotalMiB uint64
	MemoryUsedMiB  uint64
}

// commandRunner returns combined stdout of nvidia-smi.
type commandRunner func() (string, error)

func defaultNvidiaSMI() (string, error) {
	out, err := exec.Command(
		"nvidia-smi",
		"--query-gpu=name,memory.total,memory.used",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

// ReadGPU shells out to nvidia-smi. Returns an empty slice (no error) when
// nvidia-smi is missing or fails — hosts without a GPU stay usable for
// CPU-only deployments.
func ReadGPU() ([]GPUInfo, error) {
	return readGPUFrom(defaultNvidiaSMI)
}

func readGPUFrom(run commandRunner) ([]GPUInfo, error) {
	out, err := run()
	if err != nil {
		// Intentional: treat any execution error as "no GPUs visible". Caller
		// gets an empty slice. Scheduler then never places GPU models here.
		return nil, nil
	}
	return parseNvidiaSMI(out)
}

// parseNvidiaSMI parses CSV output. Some driver versions emit names with
// embedded commas, so we always interpret the last two CSV fields as
// memory.total and memory.used, and rejoin everything else as the name.
func parseNvidiaSMI(s string) ([]GPUInfo, error) {
	var gpus []GPUInfo
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 3 {
			continue
		}
		// Take the last two fields as memory values; everything before is the name.
		usedRaw := strings.TrimSpace(parts[len(parts)-1])
		totalRaw := strings.TrimSpace(parts[len(parts)-2])
		nameParts := parts[:len(parts)-2]
		for i, p := range nameParts {
			nameParts[i] = strings.TrimSpace(p)
		}
		name := strings.Join(nameParts, ", ")

		total, err := strconv.ParseUint(totalRaw, 10, 64)
		if err != nil {
			continue
		}
		used, err := strconv.ParseUint(usedRaw, 10, 64)
		if err != nil {
			continue
		}
		gpus = append(gpus, GPUInfo{
			Name:           name,
			MemoryTotalMiB: total,
			MemoryUsedMiB:  used,
		})
	}
	return gpus, nil
}
