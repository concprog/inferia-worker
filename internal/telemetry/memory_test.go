package telemetry

import "testing"

const (
	procMemInfoNormal = `MemTotal:       32856208 kB
MemFree:        16000000 kB
MemAvailable:   24000000 kB
Buffers:         500000 kB
Cached:         8000000 kB
SwapTotal:      8000000 kB
`
	procMemInfoMissingTotal = `MemFree:        16000000 kB
MemAvailable:   24000000 kB
`
	procMemInfoNoAvailable = `MemTotal:       1024 kB
MemFree:         512 kB
`
)

func TestParseMemInfo_Normal(t *testing.T) {
	total, avail, err := parseMemInfo(procMemInfoNormal)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if total != 32856208*1024 {
		t.Errorf("total: got %d", total)
	}
	if avail != 24000000*1024 {
		t.Errorf("avail: got %d", avail)
	}
}

func TestParseMemInfo_MissingTotal(t *testing.T) {
	if _, _, err := parseMemInfo(procMemInfoMissingTotal); err == nil {
		t.Errorf("expected error when MemTotal absent")
	}
}

func TestParseMemInfo_FallbackToFree(t *testing.T) {
	// Older kernels without MemAvailable: parser should fall back to MemFree.
	total, avail, err := parseMemInfo(procMemInfoNoAvailable)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if total != 1024*1024 {
		t.Errorf("total: got %d", total)
	}
	if avail != 512*1024 {
		t.Errorf("avail (fallback): got %d", avail)
	}
}

func TestParseMemInfo_OverflowSafe(t *testing.T) {
	// Values bigger than 2^53 still parse correctly as uint64.
	// MemTotal in kB; multiplying by 1024 must not overflow.
	// 16 EB total RAM is silly, but the parser must not panic.
	big := `MemTotal:       9007199254740992 kB
MemFree:        1 kB
`
	total, _, err := parseMemInfo(big)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if total == 0 {
		t.Errorf("expected non-zero")
	}
}

func TestReadMemory_FromTestData(t *testing.T) {
	r := func() (string, error) { return procMemInfoNormal, nil }
	info, err := readMemoryFrom(r)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if info.Total != 32856208*1024 {
		t.Errorf("Total: got %d", info.Total)
	}
	if info.Used == 0 || info.Used >= info.Total {
		t.Errorf("Used out of range: %d (total %d)", info.Used, info.Total)
	}
}

func TestReadMemory_ReaderError(t *testing.T) {
	r := func() (string, error) { return "", errBoom() }
	if _, err := readMemoryFrom(r); err == nil {
		t.Errorf("expected error")
	}
}
