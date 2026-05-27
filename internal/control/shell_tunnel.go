package control

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/inferia/inferia-worker/internal/shellbridge"
)

// envelopeSender is the small surface the tunnel uses to write outbound
// envelopes onto the channel. Tests fake this with a slice-capturing impl
// so they can assert exactly which frames went out without spinning up a
// real websocket. Channel implements this via its serializing writer.
type envelopeSender interface {
	WriteEnvelope(ctx context.Context, env Envelope) error
}

// shellStarter / logsStarter are tiny indirection points so tests can
// inject a fake shellbridge backend per-tunnel without mutating the
// shellbridge package-level DefaultSpawn (which is process-global and
// would race across tests).
type shellStarter func(ctx context.Context, cfg shellbridge.ShellSessionConfig) (*shellbridge.ShellSession, error)
type logsStarter func(ctx context.Context, cfg shellbridge.LogsSessionConfig) (*shellbridge.LogsSession, error)

// ChannelShellTunnel multiplexes any number of concurrent shell + logs
// sessions over a single Channel. CP-originated frames (ShellOpen / ...)
// flow in via Handle; worker-originated frames (ShellOutput / ...) flow
// out via the envelopeSender.
//
// Lifecycle: instantiated once per connectOnce on the Channel, torn down
// (via CloseAll) when the WS connection drops. A fresh tunnel is created
// on reconnect — sessions don't survive disconnects because the CP-side
// pty/log consumer (the dashboard's WS) is also gone, so there's no peer
// to forward output to.
type ChannelShellTunnel struct {
	sender envelopeSender
	// startShell / startHostShell / startLogs default to the matching
	// shellbridge entry points in production. Tests override them via
	// NewChannelShellTunnelForTest so they can plug fake backends without
	// touching package globals.
	startShell     shellStarter
	startHostShell shellStarter
	startLogs      logsStarter

	mu       sync.Mutex
	shells   map[string]*shellbridge.ShellSession
	logs     map[string]*shellbridge.LogsSession
	disposed bool
}

// NewChannelShellTunnel returns a tunnel ready to receive CP envelopes.
// Production routing: ShellOpen frames with a deployment_id or
// container_id flow through shellbridge.StartShell (docker exec into the
// model container). Frames with neither flow through StartHostShell, which
// spawns a privileged sidecar that nsenters into the host's namespaces.
// The latter is the only path the dashboard's "Shell" tab uses when no
// deployment is selected.
func NewChannelShellTunnel(sender envelopeSender) *ChannelShellTunnel {
	return &ChannelShellTunnel{
		sender:         sender,
		startShell:     shellbridge.StartShell,
		startHostShell: shellbridge.StartHostShell,
		startLogs:      shellbridge.StartLogs,
		shells:         map[string]*shellbridge.ShellSession{},
		logs:           map[string]*shellbridge.LogsSession{},
	}
}

// NewChannelShellTunnelForTest constructs a tunnel with explicit starter
// callbacks so tests can plug a fake backend without touching package
// globals. Either starter may be nil; the tunnel routes accordingly. The
// host-shell starter defaults to the same fake as the docker starter so
// existing tests don't need to change — tests that exercise the routing
// distinction supply both explicitly via WithHostStarter.
func NewChannelShellTunnelForTest(sender envelopeSender, shellStart shellStarter, logsStart logsStarter) *ChannelShellTunnel {
	t := NewChannelShellTunnel(sender)
	if shellStart != nil {
		t.startShell = shellStart
		// Default the host starter to the same callback so existing tests
		// that pass empty targets still get their fake backend invoked.
		t.startHostShell = shellStart
	}
	if logsStart != nil {
		t.startLogs = logsStart
	}
	return t
}

// WithHostStarter overrides only the host-shell starter, leaving the
// docker starter untouched. Used by tests that need to assert "empty
// target routes to host, not docker" without bleeding the same fake into
// both paths.
func (t *ChannelShellTunnel) WithHostStarter(s shellStarter) *ChannelShellTunnel {
	if s != nil {
		t.startHostShell = s
	}
	return t
}

// Handle dispatches one inbound envelope. Returns true if the envelope
// was a shell/logs frame (consumed by this tunnel) so the channel's main
// switch can short-circuit and not log "unknown type". CP→worker frames
// only; worker→CP frames (ShellOutput / ShellExit / ShellError / LogsLine
// / LogsEnd) arriving here are logged at WARN and dropped — the worker
// must never receive its own data-plane frames on the control channel.
func (t *ChannelShellTunnel) Handle(ctx context.Context, env Envelope) bool {
	switch env.Type {
	case MsgShellOpen:
		var body ShellOpenBody
		if err := remarshal(env.Body, &body); err != nil {
			t.sendError("", "bad ShellOpen body")
			return true
		}
		t.handleShellOpen(ctx, body)
	case MsgShellInput:
		var body ShellInputBody
		if err := remarshal(env.Body, &body); err != nil {
			return true
		}
		t.handleShellInput(body)
	case MsgShellResize:
		var body ShellResizeBody
		if err := remarshal(env.Body, &body); err != nil {
			return true
		}
		t.handleShellResize(body)
	case MsgShellClose:
		var body ShellCloseBody
		if err := remarshal(env.Body, &body); err != nil {
			return true
		}
		t.handleShellClose(body)
	case MsgLogsOpen:
		var body LogsOpenBody
		if err := remarshal(env.Body, &body); err != nil {
			t.sendLogsEnd("", "bad LogsOpen body")
			return true
		}
		t.handleLogsOpen(ctx, body)
	case MsgLogsClose:
		var body LogsCloseBody
		if err := remarshal(env.Body, &body); err != nil {
			return true
		}
		t.handleLogsClose(body)
	// Worker→CP data-plane frames must never arrive on the worker's
	// inbound side. Log and drop so a misbehaving CP can't crash the loop.
	case MsgShellOutput, MsgShellExit, MsgShellError, MsgLogsLine, MsgLogsEnd:
		log.Printf("shell_tunnel: dropping worker→CP frame %s received on inbound side", env.Type)
		return true
	default:
		return false
	}
	return true
}

// CloseAll tears down every live session. Called from the channel's
// connectOnce teardown path so a network drop doesn't leak PTYs or
// hijacked docker connections. Safe to call from any goroutine; further
// Handle / Close calls after CloseAll are no-ops.
func (t *ChannelShellTunnel) CloseAll() {
	t.mu.Lock()
	if t.disposed {
		t.mu.Unlock()
		return
	}
	t.disposed = true
	shells := t.shells
	logs := t.logs
	t.shells = map[string]*shellbridge.ShellSession{}
	t.logs = map[string]*shellbridge.LogsSession{}
	t.mu.Unlock()
	// Close outside the mutex so a slow-closing session can't block other
	// goroutines trying to write to the tunnel.
	for _, s := range shells {
		_ = s.Close()
	}
	for _, l := range logs {
		_ = l.Close()
	}
}

// --- inbound handlers -------------------------------------------------------

func (t *ChannelShellTunnel) handleShellOpen(ctx context.Context, body ShellOpenBody) {
	if body.StreamID == "" {
		t.sendError("", "ShellOpen missing stream_id")
		return
	}
	t.mu.Lock()
	if t.disposed {
		t.mu.Unlock()
		t.sendError(body.StreamID, "tunnel closed")
		return
	}
	if _, exists := t.shells[body.StreamID]; exists {
		t.mu.Unlock()
		// Re-open of a live stream id — silently drop, the CP must mint
		// fresh IDs.
		return
	}
	t.mu.Unlock()
	streamID := body.StreamID
	cfg := shellbridge.ShellSessionConfig{
		Shell:       body.Shell,
		User:        body.User,
		Deployment:  body.DeploymentID,
		Container:   body.ContainerID,
		InitialCols: uint16safe(body.Cols),
		InitialRows: uint16safe(body.Rows),
		OnOutput: func(data []byte) {
			t.sendEnvelope(MsgShellOutput, ShellOutputBody{
				StreamID: streamID,
				Data:     string(data),
			})
		},
		OnExit: func(code int, reason string) {
			t.dropShell(streamID)
			t.sendEnvelope(MsgShellExit, ShellExitBody{
				StreamID: streamID,
				ExitCode: code,
				Reason:   reason,
			})
		},
		OnError: func(message string) {
			t.dropShell(streamID)
			t.sendEnvelope(MsgShellError, ShellErrorBody{
				StreamID: streamID,
				Message:  message,
			})
		},
	}
	// Route empty-target ShellOpen frames to the host-shell backend so a
	// fresh worker (with no deployments loaded) still exposes a usable
	// shell to operators. Any explicit deployment_id or container_id
	// keeps the existing docker-exec path. Picking the starter here, not
	// inside shellbridge, keeps the routing decision visible at the
	// protocol boundary — the package-level DefaultSpawn / DefaultHostSpawn
	// indirection only handles backend construction.
	starter := t.startShell
	if body.DeploymentID == "" && body.ContainerID == "" {
		if t.startHostShell != nil {
			starter = t.startHostShell
		}
	}
	sess, err := starter(ctx, cfg)
	if err != nil {
		t.sendError(streamID, err.Error())
		return
	}
	t.mu.Lock()
	if t.disposed {
		// A teardown raced us; close the just-spawned session.
		t.mu.Unlock()
		_ = sess.Close()
		return
	}
	t.shells[streamID] = sess
	t.mu.Unlock()
}

func (t *ChannelShellTunnel) handleShellInput(body ShellInputBody) {
	t.mu.Lock()
	sess := t.shells[body.StreamID]
	t.mu.Unlock()
	if sess == nil {
		return // unknown stream — drop silently per spec
	}
	if err := sess.WriteInput([]byte(body.Data)); err != nil {
		log.Printf("shell_tunnel: write input %s: %v", body.StreamID, err)
	}
}

func (t *ChannelShellTunnel) handleShellResize(body ShellResizeBody) {
	t.mu.Lock()
	sess := t.shells[body.StreamID]
	t.mu.Unlock()
	if sess == nil {
		return
	}
	_ = sess.Resize(uint16safe(body.Cols), uint16safe(body.Rows))
}

func (t *ChannelShellTunnel) handleShellClose(body ShellCloseBody) {
	t.mu.Lock()
	sess := t.shells[body.StreamID]
	delete(t.shells, body.StreamID)
	t.mu.Unlock()
	if sess == nil {
		return // idempotent
	}
	_ = sess.Close()
}

func (t *ChannelShellTunnel) handleLogsOpen(ctx context.Context, body LogsOpenBody) {
	if body.StreamID == "" {
		t.sendLogsEnd("", "LogsOpen missing stream_id")
		return
	}
	t.mu.Lock()
	if t.disposed {
		t.mu.Unlock()
		t.sendLogsEnd(body.StreamID, "tunnel closed")
		return
	}
	if _, exists := t.logs[body.StreamID]; exists {
		t.mu.Unlock()
		return
	}
	t.mu.Unlock()
	streamID := body.StreamID
	cfg := shellbridge.LogsSessionConfig{
		Deployment: body.DeploymentID,
		Container:  body.ContainerID,
		OnLine: func(stream, data string) {
			t.sendEnvelope(MsgLogsLine, LogsLineBody{
				StreamID: streamID,
				Stream:   stream,
				Data:     data,
			})
		},
		OnEnd: func(reason string) {
			t.dropLogs(streamID)
			t.sendEnvelope(MsgLogsEnd, LogsEndBody{
				StreamID: streamID,
				Reason:   reason,
			})
		},
	}
	sess, err := t.startLogs(ctx, cfg)
	if err != nil {
		t.sendLogsEnd(streamID, err.Error())
		return
	}
	t.mu.Lock()
	if t.disposed {
		t.mu.Unlock()
		_ = sess.Close()
		return
	}
	t.logs[streamID] = sess
	t.mu.Unlock()
}

func (t *ChannelShellTunnel) handleLogsClose(body LogsCloseBody) {
	t.mu.Lock()
	sess := t.logs[body.StreamID]
	delete(t.logs, body.StreamID)
	t.mu.Unlock()
	if sess == nil {
		return
	}
	_ = sess.Close()
}

// --- helpers ---------------------------------------------------------------

func (t *ChannelShellTunnel) dropShell(streamID string) {
	t.mu.Lock()
	delete(t.shells, streamID)
	t.mu.Unlock()
}

func (t *ChannelShellTunnel) dropLogs(streamID string) {
	t.mu.Lock()
	delete(t.logs, streamID)
	t.mu.Unlock()
}

func (t *ChannelShellTunnel) sendEnvelope(msgType MessageType, body any) {
	env := Envelope{
		Type: msgType,
		ID:   newID(),
		TS:   time.Now().UTC().Format(time.RFC3339Nano),
		Body: body,
	}
	// Bounded ctx so a stalled writer can't pin a session goroutine
	// indefinitely. 10s mirrors writeEnvelope's deadline.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := t.sender.WriteEnvelope(ctx, env); err != nil {
		// Best-effort logging only — the channel is probably down, in
		// which case CloseAll is about to run.
		log.Printf("shell_tunnel: write %s: %v", msgType, err)
	}
}

func (t *ChannelShellTunnel) sendError(streamID, message string) {
	t.sendEnvelope(MsgShellError, ShellErrorBody{
		StreamID: streamID,
		Message:  message,
	})
}

func (t *ChannelShellTunnel) sendLogsEnd(streamID, reason string) {
	t.sendEnvelope(MsgLogsEnd, LogsEndBody{
		StreamID: streamID,
		Reason:   reason,
	})
}

// uint16safe clamps an int into uint16 range so a malicious CP can't make
// us overflow. Negative or oversized values are treated as 0 (no resize).
func uint16safe(n int) uint16 {
	if n <= 0 || n > 0xffff {
		return 0
	}
	return uint16(n)
}
