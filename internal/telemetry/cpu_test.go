package telemetry

import "testing"

const (
	procStatNormal = `cpu  3357 0 4313 1362393 0 0 0 0 0 0
cpu0 1112 0 1234 454126 0 0 0 0 0 0
cpu1 1234 0 1567 454133 0 0 0 0 0 0
cpu2 1011 0 1512 454134 0 0 0 0 0 0
intr 12345
ctxt 67890
`
	procStatNoCPU = `intr 12345
ctxt 67890
`
	procStatMalformedLine = `cpu  not numbers
cpu0 1 2 3 4 5 6 7 8 9 10
`
)

func TestParseProcStat_Normal(t *testing.T) {
	total, idle, err := parseProcStat(procStatNormal)
	if err != nil {
		t.Fatalf("%v", err)
	}
	// Sum of "cpu  3357 0 4313 1362393 ...": idle=1362393, total=sum=1370063
	if total != 1370063 {
		t.Errorf("total: got %d want 1370063", total)
	}
	if idle != 1362393 {
		t.Errorf("idle: got %d want 1362393", idle)
	}
}

func TestParseProcStat_NoAggregateCPULine(t *testing.T) {
	if _, _, err := parseProcStat(procStatNoCPU); err == nil {
		t.Errorf("expected error when no 'cpu ' line is present")
	}
}

func TestParseProcStat_MalformedLine(t *testing.T) {
	if _, _, err := parseProcStat(procStatMalformedLine); err == nil {
		t.Errorf("expected error on malformed 'cpu' values")
	}
}

func TestCPUDelta(t *testing.T) {
	cases := []struct {
		name               string
		t1, i1, t2, i2     uint64
		wantPctTimes100Min int
		wantPctTimes100Max int
	}{
		{"50pct", 1000, 500, 2000, 1000, 4900, 5100},     // delta=1000, idle=500 → busy=500/1000 = 50%
		{"100pct", 1000, 500, 2000, 500, 9900, 10100},    // idle didn't grow → 100% busy
		{"0pct", 1000, 500, 2000, 1500, -100, 100},       // all idle
		{"no_delta", 1000, 500, 1000, 500, -1, 1},        // returns 0 when no time elapsed
		{"backwards_total", 2000, 500, 1000, 500, -1, 1}, // monotonic violation → 0
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pct := cpuDelta(tc.t1, tc.i1, tc.t2, tc.i2)
			x100 := int(pct * 100)
			if x100 < tc.wantPctTimes100Min || x100 > tc.wantPctTimes100Max {
				t.Errorf("got %.2f%%, want in [%d/100, %d/100]", pct, tc.wantPctTimes100Min, tc.wantPctTimes100Max)
			}
		})
	}
}

func TestReadCPU_FromTestData(t *testing.T) {
	// readCPU is the package-level entry; we inject a reader.
	r := func() (string, error) { return procStatNormal, nil }
	info, err := readCPUFrom(r, r)
	if err != nil {
		t.Fatalf("%v", err)
	}
	// Same snapshot taken twice → ~0% usage.
	if info.UsedPercent < 0 || info.UsedPercent > 1 {
		t.Errorf("expected near-zero, got %.2f", info.UsedPercent)
	}
	if info.Cores != 3 {
		t.Errorf("expected 3 cores from sample, got %d", info.Cores)
	}
}

func TestReadCPU_ReaderError(t *testing.T) {
	r := func() (string, error) { return "", errBoom() }
	if _, err := readCPUFrom(r, r); err == nil {
		t.Errorf("expected error")
	}
}

func errBoom() error { return &boomErr{} }

type boomErr struct{}

func (*boomErr) Error() string { return "boom" }
