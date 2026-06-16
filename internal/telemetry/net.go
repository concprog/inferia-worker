package telemetry

import (
	"os"
	"strconv"
	"strings"
)

// NetCounters holds cumulative host network byte counters (summed over all
// non-loopback interfaces).
type NetCounters struct {
	RxBytes uint64
	TxBytes uint64
}

func readProcNetDev() (string, error) {
	b, err := os.ReadFile("/proc/net/dev")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// ReadNet returns cumulative rx/tx byte counters across every interface
// except loopback. Rate derivation (bytes/sec) is the caller's job.
func ReadNet() (NetCounters, error) {
	return readNetFrom(readProcNetDev)
}

func readNetFrom(r reader) (NetCounters, error) {
	s, err := r()
	if err != nil {
		return NetCounters{}, err
	}
	rx, tx, err := parseNetDev(s)
	if err != nil {
		return NetCounters{}, err
	}
	return NetCounters{RxBytes: rx, TxBytes: tx}, nil
}

// parseNetDev sums column 1 (rx bytes) and column 9 (tx bytes) of every
// "iface: ..." line except loopback. Lines without a colon (the two header
// lines, garbage) are skipped.
func parseNetDev(s string) (rx, tx uint64, err error) {
	for _, line := range strings.Split(s, "\n") {
		name, rest, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		name = strings.TrimSpace(name)
		if name == "" || name == "lo" {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) < 9 {
			continue
		}
		rxv, e1 := strconv.ParseUint(fields[0], 10, 64)
		txv, e2 := strconv.ParseUint(fields[8], 10, 64)
		if e1 != nil || e2 != nil {
			continue
		}
		rx += rxv
		tx += txv
	}
	return rx, tx, nil
}
