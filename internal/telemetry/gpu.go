package telemetry

import (
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

// GPUInfo describes one GPU's current memory state.
type GPUInfo struct {
	Name           string
	MemoryTotalMiB uint64
	MemoryUsedMiB  uint64
	UtilPct        float64 // 0..100; 0 when nvidia-smi unavailable or unparseable
}

// commandRunner returns combined stdout of nvidia-smi.
type commandRunner func() (string, error)

// devLister returns the names of files in /dev whose path matches the
// nvidia[0-9]+ pattern. Pulled out as an indirection so unit tests can
// substitute a fake listing without writing to the host /dev.
type devLister func() ([]string, error)

func defaultNvidiaSMI() (string, error) {
	out, err := exec.Command(
		"nvidia-smi",
		"--query-gpu=name,memory.total,memory.used,utilization.gpu",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func defaultListDev() ([]string, error) {
	entries, err := os.ReadDir("/dev")
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		out = append(out, e.Name())
	}
	return out, nil
}

// ReadGPU reports the GPUs visible to this process. It first tries
// nvidia-smi (which yields real model names + memory totals); if the
// binary is missing or fails — common in distroless / minimal images —
// it falls back to counting /dev/nvidia[0-9]+ device files exposed by
// the NVIDIA container runtime. The fallback can't recover the model
// name (no driver to query), so it reports "NVIDIA" and zero memory.
// Callers that care about real memory totals (e.g. scheduler-driven
// VRAM checks) must use a worker image that ships nvidia-smi; callers
// that only care about GPU count (the placement filter only multiplies
// `replicas * gpu_per_replica`) get accurate counts either way.
func ReadGPU() ([]GPUInfo, error) {
	return readGPUFrom(defaultNvidiaSMI, defaultListDev)
}

func readGPUFrom(run commandRunner, listDev devLister) ([]GPUInfo, error) {
	out, err := run()
	if err == nil {
		gpus, perr := parseNvidiaSMI(out)
		// nvidia-smi succeeded and returned at least one GPU — trust it.
		if perr == nil && len(gpus) > 0 {
			return gpus, nil
		}
	}
	// nvidia-smi missing, errored, or yielded no GPUs. Try /dev fallback —
	// in distroless containers with `--gpus all` the runtime mounts
	// /dev/nvidia0, /dev/nvidia1, ... even though no nvidia binaries
	// were copied into the image.
	gpus := readGPUFromDevices(listDev)
	if len(gpus) > 0 {
		return gpus, nil
	}
	// No GPUs visible by either path. Return empty (no error) so the
	// host stays usable for CPU-only deployments — placement just
	// skips it for GPU workloads.
	return nil, nil
}

var nvidiaDevRe = regexp.MustCompile(`^nvidia[0-9]+$`)

// readGPUFromDevices counts /dev/nvidia[0-9]+ files. /dev/nvidiactl,
// /dev/nvidia-uvm, /dev/nvidia-modeset, etc. are NOT counted — those
// exist once per host regardless of GPU count and would inflate the
// total. Returns empty slice (never nil-with-error) for any error so
// the fallback is non-fatal.
func readGPUFromDevices(listDev devLister) []GPUInfo {
	entries, err := listDev()
	if err != nil {
		return nil
	}
	var gpus []GPUInfo
	for _, name := range entries {
		if nvidiaDevRe.MatchString(name) {
			gpus = append(gpus, GPUInfo{Name: "NVIDIA"})
		}
	}
	return gpus
}

// parseNvidiaSMI parses CSV output from nvidia-smi queried with
// --query-gpu=name,memory.total,memory.used,utilization.gpu.
// Some driver versions emit names with embedded commas, so we always
// interpret the last THREE CSV fields as memory.total, memory.used, and
// utilization.gpu, and rejoin everything before as the name.
// If utilization.gpu is unparseable (e.g. "[N/A]"), UtilPct defaults to 0
// and the GPU is still included.
func parseNvidiaSMI(s string) ([]GPUInfo, error) {
	var gpus []GPUInfo
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 4 {
			continue
		}
		utilRaw := strings.TrimSpace(parts[len(parts)-1])
		usedRaw := strings.TrimSpace(parts[len(parts)-2])
		totalRaw := strings.TrimSpace(parts[len(parts)-3])
		nameParts := parts[:len(parts)-3]
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
		util, uerr := strconv.ParseFloat(utilRaw, 64)
		if uerr != nil {
			util = 0
		}
		gpus = append(gpus, GPUInfo{
			Name:           name,
			MemoryTotalMiB: total,
			MemoryUsedMiB:  used,
			UtilPct:        util,
		})
	}
	return gpus, nil
}
