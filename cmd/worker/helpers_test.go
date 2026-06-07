package main

import "testing"

func TestToWS(t *testing.T) {
	cases := map[string]string{
		"https://x.com/path": "wss://x.com/path",
		"http://x.com":       "ws://x.com",
		"ws://x.com":         "ws://x.com",
		"wss://x.com":        "wss://x.com",
		"":                   "",
		"file:///x":          "file:///x",
	}
	for in, want := range cases {
		if got := toWS(in); got != want {
			t.Errorf("toWS(%q)=%q want %q", in, got, want)
		}
	}
}

func TestHostTelemetry_NeverErrors(t *testing.T) {
	used := (hostTelemetry{}).Read()
	if used == nil {
		t.Errorf("expected non-nil map")
	}
	// On any host we should have the three keys, even if their values reflect
	// "no GPU" (gpu=0) or zero CPU.
	for _, k := range []string{"cpu_pct", "mem_used", "gpu"} {
		if _, ok := used[k]; !ok {
			t.Errorf("missing key %q in %v", k, used)
		}
	}
}
