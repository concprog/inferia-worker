package telemetry

import "testing"

const sampleNetDev = `Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 1000      10    0    0    0     0          0         0    1000      10    0    0    0     0       0          0
  eth0: 5000      50    0    0    0     0          0         0    7000      70    0    0    0     0       0          0
 wlan0: 200        2    0    0    0     0          0         0     300       3    0    0    0     0       0          0
`

func TestParseNetDev_SumsNonLoopback(t *testing.T) {
	rx, tx, err := parseNetDev(sampleNetDev)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if rx != 5200 || tx != 7300 {
		t.Fatalf("rx=%d tx=%d, want rx=5200 tx=7300", rx, tx)
	}
}

func TestParseNetDev_LoopbackExcluded(t *testing.T) {
	in := "Inter-|\n face |\n    lo: 999 1 0 0 0 0 0 0 999 1 0 0 0 0 0 0\n"
	rx, tx, err := parseNetDev(in)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if rx != 0 || tx != 0 {
		t.Fatalf("loopback should be excluded, got rx=%d tx=%d", rx, tx)
	}
}

func TestParseNetDev_MalformedLinesSkipped(t *testing.T) {
	in := "garbage\neth0 no-colon line\neth0: 100 1 0 0 0 0 0 0 200 2 0 0 0 0 0 0\n"
	rx, tx, err := parseNetDev(in)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if rx != 100 || tx != 200 {
		t.Fatalf("rx=%d tx=%d, want 100/200", rx, tx)
	}
}

func TestParseNetDev_Empty(t *testing.T) {
	rx, tx, err := parseNetDev("")
	if err != nil || rx != 0 || tx != 0 {
		t.Fatalf("empty: rx=%d tx=%d err=%v", rx, tx, err)
	}
}
