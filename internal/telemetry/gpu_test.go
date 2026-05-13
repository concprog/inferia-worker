package telemetry

import (
	"errors"
	"testing"
)

func TestParseNvidiaSMI_Normal(t *testing.T) {
	out := "NVIDIA A100-SXM4-80GB, 81920, 12345\n" +
		"NVIDIA A100-SXM4-80GB, 81920, 9876\n"
	gpus, err := parseNvidiaSMI(out)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if len(gpus) != 2 {
		t.Fatalf("got %d gpus", len(gpus))
	}
	if gpus[0].Name != "NVIDIA A100-SXM4-80GB" {
		t.Errorf("name: %q", gpus[0].Name)
	}
	if gpus[0].MemoryTotalMiB != 81920 || gpus[0].MemoryUsedMiB != 12345 {
		t.Errorf("mem: %+v", gpus[0])
	}
}

func TestParseNvidiaSMI_EmptyMeansNoGPUs(t *testing.T) {
	gpus, err := parseNvidiaSMI("")
	if err != nil {
		t.Fatalf("expected ok for empty output, got %v", err)
	}
	if len(gpus) != 0 {
		t.Errorf("expected 0 gpus, got %d", len(gpus))
	}
}

func TestParseNvidiaSMI_WhitespaceOnly(t *testing.T) {
	gpus, err := parseNvidiaSMI("\n  \n\t\n")
	if err != nil {
		t.Fatalf("%v", err)
	}
	if len(gpus) != 0 {
		t.Errorf("expected 0 gpus, got %d", len(gpus))
	}
}

func TestParseNvidiaSMI_PartialLineRejected(t *testing.T) {
	// Only 2 fields instead of 3 — line is skipped, parser doesn't crash.
	out := "NVIDIA A100, 81920\n"
	gpus, _ := parseNvidiaSMI(out)
	if len(gpus) != 0 {
		t.Errorf("partial line should be skipped, got %d", len(gpus))
	}
}

func TestParseNvidiaSMI_NonNumericMiB(t *testing.T) {
	// One bad line in between good lines — bad one skipped, good ones kept.
	out := "NVIDIA A100, 81920, 12345\n" +
		"NVIDIA A100, NA, NA\n" +
		"NVIDIA A100, 81920, 6789\n"
	gpus, _ := parseNvidiaSMI(out)
	if len(gpus) != 2 {
		t.Errorf("got %d good gpus", len(gpus))
	}
}

func TestParseNvidiaSMI_NameWithComma(t *testing.T) {
	// Some driver versions emit names with embedded commas. We use a 3-field
	// rsplit so the name keeps its embedded commas.
	out := "NVIDIA RTX, A6000, 49152, 1000\n"
	gpus, _ := parseNvidiaSMI(out)
	if len(gpus) != 1 {
		t.Fatalf("got %d gpus", len(gpus))
	}
	if gpus[0].Name != "NVIDIA RTX, A6000" {
		t.Errorf("name: %q", gpus[0].Name)
	}
	if gpus[0].MemoryTotalMiB != 49152 || gpus[0].MemoryUsedMiB != 1000 {
		t.Errorf("mem: %+v", gpus[0])
	}
}

func TestReadGPU_RunnerError(t *testing.T) {
	// nvidia-smi missing or no GPUs: returns empty slice, no error.
	runner := func() (string, error) { return "", errors.New("exec: nvidia-smi not found") }
	gpus, err := readGPUFrom(runner)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if len(gpus) != 0 {
		t.Errorf("expected empty, got %d", len(gpus))
	}
}

func TestReadGPU_Success(t *testing.T) {
	runner := func() (string, error) {
		return "NVIDIA A100, 81920, 100\n", nil
	}
	gpus, err := readGPUFrom(runner)
	if err != nil || len(gpus) != 1 {
		t.Fatalf("got %d gpus, err=%v", len(gpus), err)
	}
}
