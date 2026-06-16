package telemetry

// DeriveRate converts two cumulative counter samples into a per-second rate.
// A non-positive interval, or a current value below the previous one (counter
// reset / wrap / reboot), yields 0 rather than a negative or infinite rate.
func DeriveRate(cur, prev uint64, dtSeconds float64) float64 {
	if dtSeconds <= 0 || cur < prev {
		return 0
	}
	return float64(cur-prev) / dtSeconds
}
