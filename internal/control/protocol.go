// Package control speaks the worker side of the control-plane protocol:
// register + token exchange (HTTPS), then a long-lived WebSocket carrying
// Heartbeat / LoadModel / UnloadModel / CommandResult / Hello / Ping frames.
package control

import "time"

// MessageType is one of the strings in the catalogue.
type MessageType string

const (
	MsgHello         MessageType = "Hello"
	MsgHeartbeat     MessageType = "Heartbeat"
	MsgLoadModel     MessageType = "LoadModel"
	MsgUnloadModel   MessageType = "UnloadModel"
	MsgCommandResult MessageType = "CommandResult"
	MsgPing          MessageType = "Ping"

	// Shell + logs stream multiplexing over the worker→CP channel.
	// CP → worker (control side):
	MsgShellOpen   MessageType = "ShellOpen"
	MsgShellInput  MessageType = "ShellInput"
	MsgShellResize MessageType = "ShellResize"
	MsgShellClose  MessageType = "ShellClose"
	MsgLogsOpen    MessageType = "LogsOpen"
	MsgLogsClose   MessageType = "LogsClose"
	// worker → CP (data side):
	MsgShellOutput MessageType = "ShellOutput"
	MsgShellExit   MessageType = "ShellExit"
	MsgShellError  MessageType = "ShellError"
	MsgLogsLine    MessageType = "LogsLine"
	MsgLogsEnd     MessageType = "LogsEnd"
)

// Envelope wraps every frame. id is a UUIDv4 used for command/response
// correlation. ts is RFC3339Nano.
type Envelope struct {
	Type MessageType `json:"type"`
	ID   string      `json:"id"`
	TS   string      `json:"ts"`
	Body any         `json:"body,omitempty"`
}

// RegisterRequest is the bootstrap-time POST body.
type RegisterRequest struct {
	NodeName         string            `json:"node_name"`
	PoolID           string            `json:"pool_id"`
	AdvertiseURL     string            `json:"advertise_url,omitempty"`
	Allocatable      map[string]string `json:"allocatable"`
	RuntimeEnv       string            `json:"runtime_env,omitempty"`
	InstanceID       string            `json:"instance_id,omitempty"`
	Region           string            `json:"region,omitempty"`
	AvailabilityZone string            `json:"availability_zone,omitempty"`
	BootstrapToken   string            `json:"bootstrap_token,omitempty"`
}

// RegisterResponse is the control plane's reply.
type RegisterResponse struct {
	NodeID    string `json:"node_id"`
	WorkerJWT string `json:"worker_jwt"`
}

// HelloBody is the body of a Hello frame. The control plane sends it to the
// worker immediately after WS upgrade (carrying server_time and channel_id).
// The worker echoes cloud-env fields back so the CP can refresh
// compute_inventory.labels on every reconnect, not just on first register.
type HelloBody struct {
	ServerTime       *time.Time `json:"server_time,omitempty"`
	ChannelID        string    `json:"channel_id,omitempty"`
	RuntimeEnv       string    `json:"runtime_env,omitempty"`
	InstanceID       string    `json:"instance_id,omitempty"`
	Region           string    `json:"region,omitempty"`
	AvailabilityZone string    `json:"availability_zone,omitempty"`
}

// HeartbeatBody is what the worker sends every interval. used is a map of
// resource → opaque string (matches Python compute_node.proto for migration).
type HeartbeatBody struct {
	Used          map[string]string `json:"used"`
	LoadedModels  []string          `json:"loaded_models"`
	Events        []HeartbeatEvent  `json:"events,omitempty"`
	DeployMetrics []DeploymentMetric `json:"deploy_metrics,omitempty"`
}

// DeploymentMetric carries performance and lifecycle stats for a single model deployment.
type DeploymentMetric struct {
	DeploymentID        string `json:"deployment_id"`
	Recipe              string `json:"recipe"`
	Model               string `json:"model"`
	RequestsTotal       int64  `json:"requests_total"`
	ActiveRequests      int64  `json:"active_requests"`
	RequestLatencyP50Ms int64  `json:"request_latency_p50_ms"`
	RequestLatencyP95Ms int64  `json:"request_latency_p95_ms"`
	PullDurationMs      int64  `json:"pull_duration_ms"`
	StartDurationMs     int64              `json:"start_duration_ms"`
	Phase               string             `json:"phase"`
	EngineMetrics       map[string]float64 `json:"engine_metrics,omitempty"`
}

// HeartbeatEvent represents asynchronous lifecycle facts piggybacked on the

// heartbeat (rather than a separate WS frame). MVP only emits ModelExited.
type HeartbeatEvent struct {
	Type         string `json:"type"`
	DeploymentID string `json:"deployment_id"`
	ExitCode     int    `json:"exit_code,omitempty"`
	Reason       string `json:"reason,omitempty"`
}

// LoadModelBody is the command from CP to worker.
type LoadModelBody struct {
	DeploymentID string            `json:"deployment_id"`
	Recipe       string            `json:"recipe"`
	Model        ModelRef          `json:"model"`
	Config       map[string]any    `json:"config,omitempty"`
	GPUIndices   []int             `json:"gpu_indices"`
	Port         int               `json:"port,omitempty"`
	Env          map[string]string `json:"env,omitempty"`

	// PrefillReplicas and DecodeReplicas are used by disagg (prefill/decode
	// split) deployments. Zero means "single-container" and is compatible
	// with existing control planes that never send these fields.
	PrefillReplicas int `json:"prefill_replicas,omitempty"`
	DecodeReplicas  int `json:"decode_replicas,omitempty"`
}

// ModelRef points at an artifact.
type ModelRef struct {
	ArtifactURI string `json:"artifact_uri"`
	Format      string `json:"format,omitempty"`
	Backend     string `json:"backend,omitempty"`
}

// UnloadModelBody is the command to free a model.
type UnloadModelBody struct {
	DeploymentID string `json:"deployment_id"`
}

// CommandResultBody is the worker's response to a command.
type CommandResultBody struct {
	InReplyTo   string `json:"in_reply_to"`
	Status      string `json:"status"` // "ok" | "failed"
	Detail      string `json:"detail,omitempty"`
	EndpointURL string `json:"endpoint_url,omitempty"`
}

// --- Shell + logs stream multiplexing ---------------------------------------
//
// A single worker→CP control channel carries many concurrent shell + logs
// sessions, each identified by a stream_id minted by the CP. Frames flow in
// both directions; the channel read loop on each end dispatches by type and
// routes to the appropriate session. Field shapes mirror the Pydantic models
// in InferiaLLM's worker_controller/protocol.py — keep both sides in sync.

// ShellOpenBody is CP→worker: spawn an interactive shell session. The worker
// exec's shell in the target container (if given) or on the host. user
// switches uid via the same mechanism the legacy /v1/shell endpoint uses
// (e.g. "root" or "1000:1000"). cols/rows set the initial PTY window size.
type ShellOpenBody struct {
	StreamID     string `json:"stream_id"`
	Shell        string `json:"shell,omitempty"`
	User         string `json:"user,omitempty"`
	DeploymentID string `json:"deployment_id,omitempty"`
	ContainerID  string `json:"container_id,omitempty"`
	Cols         int    `json:"cols,omitempty"`
	Rows         int    `json:"rows,omitempty"`
}

// ShellInputBody is CP→worker: write bytes to the running shell's stdin.
// data is raw stdin (may include control chars like ^C).
type ShellInputBody struct {
	StreamID string `json:"stream_id"`
	Data     string `json:"data"`
}

// ShellResizeBody is CP→worker: resize the shell's PTY window.
type ShellResizeBody struct {
	StreamID string `json:"stream_id"`
	Cols     int    `json:"cols"`
	Rows     int    `json:"rows"`
}

// ShellCloseBody is CP→worker: kill the shell and tear down the session.
// Idempotent — worker ignores unknown stream_ids.
type ShellCloseBody struct {
	StreamID string `json:"stream_id"`
}

// ShellOutputBody is worker→CP: a chunk of PTY output (stdout+stderr merged).
type ShellOutputBody struct {
	StreamID string `json:"stream_id"`
	Data     string `json:"data"`
}

// ShellExitBody is worker→CP: the shell process exited cleanly.
type ShellExitBody struct {
	StreamID string `json:"stream_id"`
	ExitCode int    `json:"exit_code,omitempty"`
	Reason   string `json:"reason,omitempty"`
}

// ShellErrorBody is worker→CP: failed to spawn the shell or PTY died
// abnormally. Sent in lieu of (not in addition to) ShellExit.
type ShellErrorBody struct {
	StreamID string `json:"stream_id"`
	Message  string `json:"message"`
}

// LogsOpenBody is CP→worker: stream container logs for deployment_id /
// container_id. When both are empty the worker tails its first running
// container. Lines flow back as LogsLine envelopes until the dashboard sends
// LogsClose (or the container exits, which emits LogsEnd).
type LogsOpenBody struct {
	StreamID     string `json:"stream_id"`
	DeploymentID string `json:"deployment_id,omitempty"`
	ContainerID  string `json:"container_id,omitempty"`
}

// LogsLineBody is worker→CP: one line of container output. stream is
// "stdout" or "stderr".
type LogsLineBody struct {
	StreamID string `json:"stream_id"`
	Stream   string `json:"stream,omitempty"`
	Data     string `json:"data"`
}

// LogsEndBody is worker→CP: log stream ended (container stopped or follow
// timed out).
type LogsEndBody struct {
	StreamID string `json:"stream_id"`
	Reason   string `json:"reason,omitempty"`
}

// LogsCloseBody is CP→worker: stop streaming logs for this session.
// Idempotent — worker ignores unknown stream_ids.
type LogsCloseBody struct {
	StreamID string `json:"stream_id"`
}
