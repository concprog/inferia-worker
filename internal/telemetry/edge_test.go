package telemetry

import (
	"errors"
	"testing"
)

func TestReadCPUFrom_SecondReaderError(t *testing.T) {
	first := func() (string, error) { return procStatNormal, nil }
	second := func() (string, error) { return "", errors.New("boom") }
	if _, err := readCPUFrom(first, second); err == nil {
		t.Errorf("expected error from second reader")
	}
}

func TestReadCPUFrom_SecondParseError(t *testing.T) {
	first := func() (string, error) { return procStatNormal, nil }
	second := func() (string, error) { return "garbage\n", nil }
	if _, err := readCPUFrom(first, second); err == nil {
		t.Errorf("expected parse error from second sample")
	}
}

func TestParseMemInfo_BadValue(t *testing.T) {
	bad := "MemTotal:       not-a-number kB\n"
	if _, _, err := parseMemInfo(bad); err == nil {
		t.Errorf("expected error on non-numeric MemTotal")
	}
}

func TestParseMemInfo_BadAvailable(t *testing.T) {
	bad := "MemTotal: 1024 kB\nMemAvailable: BOOM kB\n"
	if _, _, err := parseMemInfo(bad); err == nil {
		t.Errorf("expected error on bad MemAvailable")
	}
}

func TestParseMemInfo_BadFree(t *testing.T) {
	bad := "MemTotal: 1024 kB\nMemFree: BOOM kB\n"
	if _, _, err := parseMemInfo(bad); err == nil {
		t.Errorf("expected error on bad MemFree")
	}
}

func TestParseMemInfo_NoAvailableOrFree(t *testing.T) {
	// total only — used should be 0.
	s := "MemTotal: 2048 kB\n"
	total, avail, err := parseMemInfo(s)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if total != 2048*1024 || avail != 0 {
		t.Errorf("got total=%d avail=%d", total, avail)
	}
}

func TestReadMemoryFrom_ParseError(t *testing.T) {
	r := func() (string, error) { return "no MemTotal here\n", nil }
	if _, err := readMemoryFrom(r); err == nil {
		t.Errorf("expected error")
	}
}
