package telemetry

import "testing"

func TestDeriveRate_Normal(t *testing.T) {
	if got := DeriveRate(2000, 1000, 2.0); got != 500 {
		t.Fatalf("got %v, want 500", got)
	}
}

func TestDeriveRate_CounterResetClampsZero(t *testing.T) {
	if got := DeriveRate(50, 100, 2.0); got != 0 {
		t.Fatalf("got %v, want 0", got)
	}
}

func TestDeriveRate_NonPositiveIntervalIsZero(t *testing.T) {
	if got := DeriveRate(2000, 1000, 0); got != 0 {
		t.Fatalf("dt=0 got %v, want 0", got)
	}
	if got := DeriveRate(2000, 1000, -1); got != 0 {
		t.Fatalf("dt<0 got %v, want 0", got)
	}
}

func TestDeriveRate_NoChange(t *testing.T) {
	if got := DeriveRate(1000, 1000, 5.0); got != 0 {
		t.Fatalf("got %v, want 0", got)
	}
}
