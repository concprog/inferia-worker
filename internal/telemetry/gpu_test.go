package telemetry

import (
	"errors"
	"strings"
	"testing"
)

// helpers ----------------------------------------------------------------

func emptyDevs() ([]string, error) { return nil, nil }

func devsWith(names ...string) devLister {
	return func() ([]string, error) { return names, nil }
}

func devsErroring(msg string) devLister {
	return func() ([]string, error) { return nil, errors.New(msg) }
}

// parseNvidiaSMI tests ---------------------------------------------------

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
	out := "NVIDIA A100, 81920\n"
	gpus, _ := parseNvidiaSMI(out)
	if len(gpus) != 0 {
		t.Errorf("partial line should be skipped, got %d", len(gpus))
	}
}

func TestParseNvidiaSMI_NonNumericMiB(t *testing.T) {
	out := "NVIDIA A100, 81920, 12345\n" +
		"NVIDIA A100, NA, NA\n" +
		"NVIDIA A100, 81920, 6789\n"
	gpus, _ := parseNvidiaSMI(out)
	if len(gpus) != 2 {
		t.Errorf("got %d good gpus", len(gpus))
	}
}

func TestParseNvidiaSMI_NameWithComma(t *testing.T) {
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

// nvidia-smi -> device-fallback flow ------------------------------------

func TestReadGPU_RunnerErrorTriesDevFallback(t *testing.T) {
	// nvidia-smi missing AND /dev empty: returns empty slice, no error.
	runner := func() (string, error) { return "", errors.New("exec: nvidia-smi not found") }
	gpus, err := readGPUFrom(runner, emptyDevs)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if len(gpus) != 0 {
		t.Errorf("expected empty, got %d", len(gpus))
	}
}

func TestReadGPU_NvidiaSMISuccess(t *testing.T) {
	runner := func() (string, error) {
		return "NVIDIA A100, 81920, 100\n", nil
	}
	gpus, err := readGPUFrom(runner, emptyDevs)
	if err != nil || len(gpus) != 1 {
		t.Fatalf("got %d gpus, err=%v", len(gpus), err)
	}
	// nvidia-smi must win — should NOT have been overridden by the empty dev listing.
	if gpus[0].Name != "NVIDIA A100" {
		t.Errorf("expected nvidia-smi name to win, got %q", gpus[0].Name)
	}
	if gpus[0].MemoryTotalMiB != 81920 {
		t.Errorf("expected memory from nvidia-smi, got %d", gpus[0].MemoryTotalMiB)
	}
}

func TestReadGPU_FallsBackToDevWhenNvidiaSMIMissing(t *testing.T) {
	runner := func() (string, error) { return "", errors.New("exec: nvidia-smi not found") }
	listDev := devsWith("nvidia0", "nvidiactl", "nvidia-uvm", "nvidia-modeset", "sda")
	gpus, err := readGPUFrom(runner, listDev)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(gpus) != 1 {
		t.Fatalf("expected 1 gpu from /dev/nvidia0, got %d", len(gpus))
	}
	if gpus[0].Name != "NVIDIA" {
		t.Errorf("fallback name=%q, want NVIDIA", gpus[0].Name)
	}
	if gpus[0].MemoryTotalMiB != 0 || gpus[0].MemoryUsedMiB != 0 {
		t.Errorf("fallback should report zero memory (can't query without driver), got %+v", gpus[0])
	}
}

func TestReadGPU_FallsBackToDevWhenNvidiaSMIReturnsEmpty(t *testing.T) {
	// nvidia-smi binary present but driver crashed / returned no gpus.
	// Should still try /dev fallback.
	runner := func() (string, error) { return "", nil }
	listDev := devsWith("nvidia0", "nvidia1", "nvidiactl")
	gpus, _ := readGPUFrom(runner, listDev)
	if len(gpus) != 2 {
		t.Fatalf("expected 2 gpus from /dev fallback, got %d", len(gpus))
	}
}

// readGPUFromDevices direct tests ---------------------------------------

func TestReadGPUFromDevices_CountsOnlyIndexedDevices(t *testing.T) {
	// /dev/nvidiactl, /dev/nvidia-uvm, etc. exist once per host regardless
	// of GPU count. Only nvidia[0-9]+ entries are real GPUs.
	listDev := devsWith(
		"nvidia0", "nvidia1", "nvidia2",
		"nvidiactl", "nvidia-uvm", "nvidia-uvm-tools", "nvidia-modeset",
		"nvidia-caps",
		"sda", "loop0", "tty",
	)
	gpus := readGPUFromDevices(listDev)
	if len(gpus) != 3 {
		t.Errorf("want 3, got %d", len(gpus))
	}
	for _, g := range gpus {
		if g.Name != "NVIDIA" {
			t.Errorf("want NVIDIA, got %q", g.Name)
		}
	}
}

func TestReadGPUFromDevices_NoNvidiaDevices(t *testing.T) {
	gpus := readGPUFromDevices(devsWith("sda", "tty", "null", "zero"))
	if len(gpus) != 0 {
		t.Errorf("want 0, got %d", len(gpus))
	}
}

func TestReadGPUFromDevices_EmptyDir(t *testing.T) {
	gpus := readGPUFromDevices(emptyDevs)
	if len(gpus) != 0 {
		t.Errorf("want 0, got %d", len(gpus))
	}
}

func TestReadGPUFromDevices_ListErrorReturnsNil(t *testing.T) {
	gpus := readGPUFromDevices(devsErroring("permission denied"))
	if gpus != nil {
		t.Errorf("error path should return nil, got %v", gpus)
	}
}

func TestReadGPUFromDevices_RejectsNearMatches(t *testing.T) {
	// Names that look-alike but aren't real GPU devices must NOT count.
	// Pattern is strictly ^nvidia[0-9]+$ — no prefix, no suffix.
	cases := []string{
		"nvidia",           // no index
		"nvidia0a",         // trailing letter
		"foo-nvidia0",      // prefix
		"nvidia-0",         // hyphen
		"nvidia0.bak",      // dot suffix
		"NVIDIA0",          // capitalisation mismatch
		"nvidia00000abc",   // mixed (trailing letters)
		"nvidia" + strings.Repeat("9", 32) + "x", // long digit run with non-digit tail
	}
	for _, c := range cases {
		gpus := readGPUFromDevices(devsWith(c))
		if len(gpus) != 0 {
			t.Errorf("name %q should not count, but produced %d gpus", c, len(gpus))
		}
	}
}

func TestReadGPUFromDevices_VeryLongDigitRunAccepted(t *testing.T) {
	// Hosts can't have this many GPUs but the regex must still match a
	// pure-digit suffix of arbitrary length — length-overflow safety.
	long := "nvidia" + strings.Repeat("9", 64)
	gpus := readGPUFromDevices(devsWith(long))
	if len(gpus) != 1 {
		t.Errorf("long digit suffix should still match, got %d", len(gpus))
	}
}

func TestReadGPUFromDevices_HighIndexedDevices(t *testing.T) {
	// 8-GPU host: nvidia0..nvidia7.
	names := []string{}
	for i := 0; i < 8; i++ {
		names = append(names, "nvidia"+string(rune('0'+i)))
	}
	gpus := readGPUFromDevices(devsWith(names...))
	if len(gpus) != 8 {
		t.Errorf("want 8, got %d", len(gpus))
	}
}
