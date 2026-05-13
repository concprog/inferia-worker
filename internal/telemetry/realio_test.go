package telemetry

import (
	"runtime"
	"testing"
)

// These tests exercise the real-I/O wrappers. They are best-effort on Linux
// and skipped on platforms without /proc.

func TestReadCPU_RealProcStat(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("requires /proc/stat")
	}
	info, err := ReadCPU()
	if err != nil {
		t.Fatalf("ReadCPU: %v", err)
	}
	if info.Cores <= 0 {
		t.Errorf("expected ≥1 core, got %d", info.Cores)
	}
	if info.UsedPercent < 0 || info.UsedPercent > 100 {
		t.Errorf("UsedPercent out of range: %.2f", info.UsedPercent)
	}
}

func TestReadMemory_RealProcMemInfo(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("requires /proc/meminfo")
	}
	info, err := ReadMemory()
	if err != nil {
		t.Fatalf("ReadMemory: %v", err)
	}
	if info.Total == 0 {
		t.Errorf("expected non-zero total")
	}
	if info.Used > info.Total {
		t.Errorf("Used %d > Total %d", info.Used, info.Total)
	}
}

func TestReadGPU_RealHost(t *testing.T) {
	// Always exercise the wrapper; nvidia-smi may or may not exist.
	gpus, err := ReadGPU()
	if err != nil {
		t.Fatalf("ReadGPU should never error, got %v", err)
	}
	_ = gpus // length is host-dependent; we only care that the wrapper runs.
}

func TestParseProcStat_MalformedShortLine(t *testing.T) {
	// 'cpu ' followed by only one field: should report a malformed-cpu-line error.
	if _, _, err := parseProcStat("cpu 1\n"); err == nil {
		t.Errorf("expected error on short 'cpu' line")
	}
}

func TestParseKB_MissingValue(t *testing.T) {
	if _, err := parseKB("   "); err == nil {
		t.Errorf("expected error for empty value")
	}
}

func TestParseKB_SaturatesOnOverflow(t *testing.T) {
	// 2^54 kB would overflow uint64 when multiplied by 1024; we should saturate.
	huge := "18014398509481984 kB" // 2^54
	n, err := parseKB(huge)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if n != ^uint64(0) {
		t.Errorf("expected saturation to max uint64, got %d", n)
	}
}

func TestCPUDelta_ClampedAbove100(t *testing.T) {
	// If reported idle goes backwards but total still grew, our clamp keeps it at 100%.
	pct := cpuDelta(1000, 500, 2000, 0)
	if pct < 99 || pct > 101 {
		t.Errorf("expected ~100%%, got %.2f", pct)
	}
}
