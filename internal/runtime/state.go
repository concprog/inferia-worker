package runtime

// State is the worker's view of a deployment's lifecycle.
type State int

const (
	StateAbsent State = iota
	StatePulling
	StateStarting
	StateRunning
	StateStopping
	StateFailed
)

func (s State) String() string {
	switch s {
	case StateAbsent:
		return "absent"
	case StatePulling:
		return "pulling"
	case StateStarting:
		return "starting"
	case StateRunning:
		return "running"
	case StateStopping:
		return "stopping"
	case StateFailed:
		return "failed"
	default:
		return "unknown"
	}
}
