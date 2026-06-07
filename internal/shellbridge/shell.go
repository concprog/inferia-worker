// Package shellbridge owns the worker-side machinery for interactive shells
// and container log streams. It exposes the same set of primitives to two
// surfaces:
//
//   - internal/admin       — operator-facing /v1/shell + /v1/logs HTTP-WS
//     endpoints (still mounted on the worker's Fiber app
//     for direct connections and backwards compatibility).
//   - internal/control     — the channel multiplexer that tunnels shell + logs
//     traffic over the long-lived worker→CP control
//     channel, so dashboards don't need to dial the
//     worker's port 8080 directly.
//
// Both consumers get the same lifecycle behaviour: spawn a session, pipe
// stdin bytes, forward stdout chunks via OnOutput, deliver OnExit or
// OnError once, idempotent Close. The Docker exec hijack is the only
// backend used in production; tests inject a local-PTY backend so they
// don't require a running docker daemon.
package shellbridge

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
)

// ShellBackend abstracts the transport that runs the actual shell. The
// production implementation talks to the Docker daemon via the SDK; tests
// inject a local PTY backend so the suite stays hermetic.
//
// Spawn returns an io.ReadWriteCloser that gives:
//   - Read: PTY output (stdout+stderr merged for tty mode).
//   - Write: stdin into the shell.
//   - Close: kill the process / detach.
//
// Resize is best-effort; backends that can't resize (e.g. a plain exec without
// a real pty) return nil and silently ignore. WaitExit blocks until the shell
// process terminates and returns its exit code; -1 if unknown.
type ShellBackend interface {
	Spawn(ctx context.Context, cfg ShellSessionConfig) (io.ReadWriteCloser, error)
	Resize(cols, rows uint16) error
	WaitExit(ctx context.Context) (int, error)
}

// SpawnBackend is the factory used by StartShell to build a ShellBackend. A
// non-nil DefaultSpawn is set at package init; tests swap it via
// StartShellWithBackend so they don't perturb global state.
type SpawnBackend func(cfg ShellSessionConfig) (ShellBackend, error)

// DefaultSpawn is the package-level factory used in production. Initialised
// by SetDockerClient; nil until then. ChannelShellTunnel guards against a
// nil DefaultSpawn so a misconfigured worker fails the ShellOpen frame
// cleanly rather than panicking.
var DefaultSpawn SpawnBackend

// SetDockerClient installs the docker-backed default spawner. Called once
// at worker boot. Safe to call from multiple goroutines but not designed
// for repeated re-configuration — callers should treat this as one-shot.
func SetDockerClient(cli *client.Client, rt Runtime) {
	DefaultSpawn = func(cfg ShellSessionConfig) (ShellBackend, error) {
		cid, err := ResolveContainer(cli, rt, cfg.Container, cfg.Deployment)
		if err != nil {
			return nil, err
		}
		return NewDockerShellBackend(cli, cid), nil
	}
}

// NewDockerShellBackend constructs a ShellBackend backed by the Docker
// SDK against an already-resolved containerID. Exposed (rather than
// internal) so the legacy admin /v1/shell handler can reuse it without
// going through ResolveContainer twice — admin already runs its own
// query-string resolution chain.
func NewDockerShellBackend(cli *client.Client, containerID string) ShellBackend {
	return &dockerShellBackend{cli: cli, containerID: containerID}
}

// Runtime is the small subset of runtime state shellbridge needs to map
// deployment IDs to docker container IDs.
type Runtime interface {
	ContainerForDeployment(deploymentID string) string
	LoadedDeployments() []string
}

// ShellSessionConfig fully describes one shell session. The caller is
// responsible for supplying non-nil callbacks for the events it cares
// about — nil callbacks are silently dropped, so a tunnel that doesn't
// need OnError can leave it nil.
type ShellSessionConfig struct {
	// Shell is the absolute path of the binary to exec. Defaults to
	// /bin/bash with a /bin/sh fallback when the image lacks bash.
	Shell string
	// User is forwarded to docker exec --user verbatim. Accepts "name",
	// "uid", "name:group", or "uid:gid". Empty means "use container default".
	User string
	// Deployment is the inferia deployment ID. The backend resolves it to
	// a container via the Runtime registry, with a docker-ps fallback for
	// post-restart recovery.
	Deployment string
	// Container is the raw container ID. When set, takes precedence over
	// Deployment so operators can pin to a specific container.
	Container string
	// InitialCols/InitialRows configure the PTY at spawn time. 0 disables
	// the initial resize call (the PTY still starts at its default size).
	InitialCols uint16
	InitialRows uint16

	// OnOutput receives raw PTY output. Called from the read-pump
	// goroutine; the callback must be safe to invoke concurrently with
	// WriteInput / Resize from other goroutines. Implementations that
	// marshal envelopes must serialize their writes themselves.
	OnOutput func(data []byte)
	// OnExit fires exactly once when the shell exits cleanly. Mutually
	// exclusive with OnError.
	OnExit func(code int, reason string)
	// OnError fires when spawn fails or the PTY dies abnormally. Mutually
	// exclusive with OnExit.
	OnError func(message string)
}

// ShellSession is one live shell. Returned by StartShell, terminated by
// Close (idempotent) or by the underlying process exiting.
type ShellSession struct {
	backend ShellBackend
	rw      io.ReadWriteCloser

	cfg ShellSessionConfig

	closed atomic.Bool

	// stdinMu serializes WriteInput calls so two concurrent goroutines
	// can't interleave writes to the same fd.
	stdinMu sync.Mutex

	cancel context.CancelFunc
	done   chan struct{}
}

// StartShell spawns a shell session using DefaultSpawn (production) or the
// backend explicitly set in cfg via WithBackend. The returned session has
// its read-pump goroutine already running; output flows into cfg.OnOutput
// until Close or until the shell exits.
func StartShell(ctx context.Context, cfg ShellSessionConfig) (*ShellSession, error) {
	return startShellWith(ctx, cfg, DefaultSpawn)
}

// StartShellWithBackend is the explicit form used by tests. Passing a nil
// factory returns ErrNoBackend so production paths that forget to call
// SetDockerClient fail loudly rather than silently.
func StartShellWithBackend(ctx context.Context, cfg ShellSessionConfig, factory SpawnBackend) (*ShellSession, error) {
	return startShellWith(ctx, cfg, factory)
}

// ErrNoBackend is returned when DefaultSpawn is nil and no explicit backend
// was supplied. Surfaces a misconfigured worker boot.
var ErrNoBackend = errors.New("shellbridge: no shell backend configured")

func startShellWith(ctx context.Context, cfg ShellSessionConfig, factory SpawnBackend) (*ShellSession, error) {
	if factory == nil {
		return nil, ErrNoBackend
	}
	if cfg.Shell == "" {
		cfg.Shell = "/bin/bash"
	}
	backend, err := factory(cfg)
	if err != nil {
		return nil, err
	}
	// The read-pump uses sessionCtx so Close can cancel any backend that
	// honors ctx.Done(). WaitExit uses the parent ctx, not sessionCtx, so
	// the inspect call survives the cancellation triggered by Close().
	sessionCtx, cancel := context.WithCancel(ctx)
	rw, err := backend.Spawn(sessionCtx, cfg)
	if err != nil {
		cancel()
		return nil, err
	}
	// Best-effort initial resize. Failure (e.g. backend doesn't support
	// resize) is ignored; the shell still runs at its default geometry.
	if cfg.InitialCols > 0 && cfg.InitialRows > 0 {
		_ = backend.Resize(cfg.InitialCols, cfg.InitialRows)
	}
	s := &ShellSession{
		backend: backend,
		rw:      rw,
		cfg:     cfg,
		cancel:  cancel,
		done:    make(chan struct{}),
	}
	go s.readPump(ctx)
	return s, nil
}

// WriteInput pipes raw stdin bytes to the running shell. Safe to call from
// multiple goroutines. Returns an error if the session is already closed.
func (s *ShellSession) WriteInput(data []byte) error {
	if s.closed.Load() {
		return errors.New("shellbridge: session closed")
	}
	s.stdinMu.Lock()
	defer s.stdinMu.Unlock()
	_, err := s.rw.Write(data)
	return err
}

// Resize updates the PTY window size. No-op on backends that don't support
// it.
func (s *ShellSession) Resize(cols, rows uint16) error {
	if s.closed.Load() {
		return errors.New("shellbridge: session closed")
	}
	return s.backend.Resize(cols, rows)
}

// Close terminates the session. Idempotent — subsequent calls are no-ops.
// Does not fire OnExit / OnError (the caller is initiating the teardown
// and doesn't need a callback to confirm it).
func (s *ShellSession) Close() error {
	if !s.closed.CompareAndSwap(false, true) {
		return nil
	}
	// Cancel first so backends watching ctx.Done() can wind up promptly.
	if s.cancel != nil {
		s.cancel()
	}
	err := s.rw.Close()
	// Wait for the read pump to exit so callbacks don't fire after Close
	// returns. Bounded so a wedged backend doesn't deadlock the caller —
	// the goroutine is owned by the backend and will eventually exit when
	// the OS process is reaped.
	select {
	case <-s.done:
	case <-time.After(2 * time.Second):
	}
	return err
}

// readPump owns the OnOutput/OnExit/OnError side of the session lifecycle.
// The parent ctx is used for WaitExit so we can still reap an exit code
// after Close() cancels the session ctx.
func (s *ShellSession) readPump(parentCtx context.Context) {
	defer close(s.done)
	buf := make([]byte, 32*1024)
	for {
		n, rerr := s.rw.Read(buf)
		if n > 0 && s.cfg.OnOutput != nil {
			// Copy out of the shared buffer so callbacks can hold the
			// slice past the next Read without racing with us.
			chunk := make([]byte, n)
			copy(chunk, buf[:n])
			s.cfg.OnOutput(chunk)
		}
		if rerr != nil {
			if s.closed.Load() {
				return // caller-initiated close; no callback
			}
			// Reap exit code via the parent ctx — the session ctx is
			// already cancelled if Close() ran, but the parent is still
			// alive (channel hasn't disconnected).
			code, werr := s.backend.WaitExit(parentCtx)
			if werr != nil && !errors.Is(werr, io.EOF) && !errors.Is(rerr, io.EOF) {
				// Hard error path: spawn worked but stream died abnormally.
				if s.cfg.OnError != nil {
					s.cfg.OnError(fmt.Sprintf("shell read: %v", rerr))
					return
				}
			}
			if s.cfg.OnExit != nil {
				reason := ""
				if !errors.Is(rerr, io.EOF) {
					reason = rerr.Error()
				}
				s.cfg.OnExit(code, reason)
			}
			return
		}
	}
}

// --- Docker-backed default backend ------------------------------------------

type dockerShellBackend struct {
	cli         *client.Client
	containerID string

	execID string
}

func (b *dockerShellBackend) Spawn(ctx context.Context, cfg ShellSessionConfig) (io.ReadWriteCloser, error) {
	shellPath := cfg.Shell
	if shellPath == "" {
		shellPath = "/bin/bash"
	}
	execCfg := container.ExecOptions{
		AttachStdin:  true,
		AttachStdout: true,
		AttachStderr: true,
		Tty:          true,
		Cmd:          []string{shellPath},
		User:         cfg.User,
	}
	created, err := b.cli.ContainerExecCreate(ctx, b.containerID, execCfg)
	if err != nil {
		// Bash fallback to /bin/sh for distroless-style images.
		if shellPath != "/bin/sh" {
			execCfg.Cmd = []string{"/bin/sh"}
			created, err = b.cli.ContainerExecCreate(ctx, b.containerID, execCfg)
		}
		if err != nil {
			return nil, fmt.Errorf("exec create: %w", err)
		}
	}
	b.execID = created.ID
	hijack, err := b.cli.ContainerExecAttach(ctx, created.ID, container.ExecStartOptions{Tty: true})
	if err != nil {
		return nil, fmt.Errorf("exec attach: %w", err)
	}
	return &hijackRWC{r: hijack.Reader, w: hijack.Conn, close: hijack.Close}, nil
}

func (b *dockerShellBackend) Resize(cols, rows uint16) error {
	if b.execID == "" || b.cli == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return b.cli.ContainerExecResize(ctx, b.execID, container.ResizeOptions{
		Height: uint(rows),
		Width:  uint(cols),
	})
}

func (b *dockerShellBackend) WaitExit(ctx context.Context) (int, error) {
	if b.execID == "" || b.cli == nil {
		return -1, nil
	}
	deadline, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	insp, err := b.cli.ContainerExecInspect(deadline, b.execID)
	if err != nil {
		return -1, err
	}
	return insp.ExitCode, nil
}

// hijackRWC bridges the Docker SDK hijack response (which has Reader +
// Conn + Close as three separate fields) to a single io.ReadWriteCloser.
type hijackRWC struct {
	r     io.Reader
	w     io.Writer
	close func()
}

func (h *hijackRWC) Read(p []byte) (int, error)  { return h.r.Read(p) }
func (h *hijackRWC) Write(p []byte) (int, error) { return h.w.Write(p) }
func (h *hijackRWC) Close() error {
	if c, ok := h.w.(io.Closer); ok {
		_ = c.Close()
	}
	if h.close != nil {
		h.close()
	}
	return nil
}
