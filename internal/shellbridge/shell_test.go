package shellbridge

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/creack/pty"
)

// localPTYBackend exec's a local process attached to a fresh PTY. Drives
// the test suite without requiring a docker daemon. The exec.Cmd is owned
// by the backend; Spawn returns the PTY (an io.ReadWriteCloser that gives
// us both ends of the stdin/stdout pipe via the master fd).
type localPTYBackend struct {
	cmd *exec.Cmd
	pty *os.File

	resizes    atomic.Int32
	waited     atomic.Bool
	exitCode   int
	waitErr    error
	waitOnce   sync.Once
	waitDoneCh chan struct{}
}

func (b *localPTYBackend) Spawn(ctx context.Context, cfg ShellSessionConfig) (io.ReadWriteCloser, error) {
	// Split the shell path into argv (first arg is the program). Local
	// PTY tests use a binary like /bin/echo with positional args supplied
	// via the User field as a shortcut — keeps the test API tiny.
	parts := strings.Fields(cfg.Shell)
	if len(parts) == 0 {
		return nil, errors.New("empty Shell")
	}
	// #nosec G204 — test code, args come from the test itself
	cmd := exec.CommandContext(ctx, parts[0], parts[1:]...)
	ptyFile, err := pty.Start(cmd)
	if err != nil {
		return nil, fmt.Errorf("pty.Start: %w", err)
	}
	b.cmd = cmd
	b.pty = ptyFile
	b.waitDoneCh = make(chan struct{})
	return ptyFile, nil
}

func (b *localPTYBackend) Resize(cols, rows uint16) error {
	b.resizes.Add(1)
	if b.pty == nil {
		return nil
	}
	return pty.Setsize(b.pty, &pty.Winsize{Cols: cols, Rows: rows})
}

func (b *localPTYBackend) WaitExit(ctx context.Context) (int, error) {
	b.waitOnce.Do(func() {
		err := b.cmd.Wait()
		b.waitErr = err
		if ee, ok := err.(*exec.ExitError); ok {
			b.exitCode = ee.ExitCode()
		} else if err != nil {
			b.exitCode = -1
		} else {
			b.exitCode = 0
		}
		close(b.waitDoneCh)
	})
	// Wait for the once-block to complete so concurrent callers see the
	// exit code, not the zero value.
	select {
	case <-b.waitDoneCh:
	case <-ctx.Done():
		return -1, ctx.Err()
	}
	b.waited.Store(true)
	return b.exitCode, nil
}

func newLocalBackend() *localPTYBackend { return &localPTYBackend{} }

// localFactory wraps a localPTYBackend so StartShellWithBackend can hand
// it to startShellWith. We new up one backend per Spawn so tests can hold
// references to the backend object for assertions.
func localFactory(holder **localPTYBackend) SpawnBackend {
	return func(cfg ShellSessionConfig) (ShellBackend, error) {
		b := newLocalBackend()
		*holder = b
		return b, nil
	}
}

func TestShellSession_EchoEmitsOutputAndExits(t *testing.T) {
	var backend *localPTYBackend
	var outputs [][]byte
	var outputsMu sync.Mutex
	exited := make(chan struct{}, 1)
	var gotCode int
	var gotReason string

	sess, err := StartShellWithBackend(context.Background(),
		ShellSessionConfig{
			Shell: "/bin/echo hello-from-shellbridge",
			OnOutput: func(data []byte) {
				outputsMu.Lock()
				outputs = append(outputs, append([]byte(nil), data...))
				outputsMu.Unlock()
			},
			OnExit: func(code int, reason string) {
				gotCode = code
				gotReason = reason
				select {
				case exited <- struct{}{}:
				default:
				}
			},
			OnError: func(message string) {
				t.Errorf("unexpected OnError: %s", message)
			},
		},
		localFactory(&backend),
	)
	if err != nil {
		t.Fatalf("StartShellWithBackend: %v", err)
	}
	defer sess.Close()

	select {
	case <-exited:
	case <-time.After(3 * time.Second):
		t.Fatalf("shell did not exit; outputs=%v", outputs)
	}

	// Echo writes "hello-from-shellbridge\r\n" (PTY adds \r). Concatenate
	// and check the substring.
	outputsMu.Lock()
	defer outputsMu.Unlock()
	var joined strings.Builder
	for _, b := range outputs {
		joined.Write(b)
	}
	if !strings.Contains(joined.String(), "hello-from-shellbridge") {
		t.Errorf("expected output to contain 'hello-from-shellbridge', got %q", joined.String())
	}
	if gotCode != 0 {
		t.Errorf("expected exit code 0, got %d", gotCode)
	}
	_ = gotReason
}

func TestShellSession_WriteInputToCat(t *testing.T) {
	var backend *localPTYBackend
	var seenMu sync.Mutex
	var seen string
	gotEcho := make(chan struct{}, 1)

	sess, err := StartShellWithBackend(context.Background(),
		ShellSessionConfig{
			Shell: "/bin/cat",
			OnOutput: func(data []byte) {
				seenMu.Lock()
				seen += string(data)
				if strings.Contains(seen, "PING") {
					select {
					case gotEcho <- struct{}{}:
					default:
					}
				}
				seenMu.Unlock()
			},
		},
		localFactory(&backend),
	)
	if err != nil {
		t.Fatalf("StartShellWithBackend: %v", err)
	}
	defer sess.Close()

	if err := sess.WriteInput([]byte("PING\n")); err != nil {
		t.Fatalf("WriteInput: %v", err)
	}
	select {
	case <-gotEcho:
	case <-time.After(2 * time.Second):
		seenMu.Lock()
		buf := seen
		seenMu.Unlock()
		t.Fatalf("did not receive echo of PING; got %q", buf)
	}
}

func TestShellSession_ResizeBeforeOutput(t *testing.T) {
	// Verifies Resize is callable immediately after Start without
	// crashing on a PTY that hasn't produced output yet. Uses /bin/cat
	// so the process stays alive until we close.
	var backend *localPTYBackend
	sess, err := StartShellWithBackend(context.Background(),
		ShellSessionConfig{
			Shell:       "/bin/cat",
			InitialCols: 80,
			InitialRows: 24,
		},
		localFactory(&backend),
	)
	if err != nil {
		t.Fatalf("StartShellWithBackend: %v", err)
	}
	// Resize should not crash even before any output has flowed.
	if err := sess.Resize(120, 40); err != nil {
		t.Fatalf("Resize: %v", err)
	}
	// Initial cols/rows are also a resize call.
	if r := backend.resizes.Load(); r < 1 {
		t.Errorf("expected at least one resize call (initial + explicit), got %d", r)
	}
	_ = sess.Close()
}

func TestShellSession_CloseIsIdempotent(t *testing.T) {
	var backend *localPTYBackend
	sess, err := StartShellWithBackend(context.Background(),
		ShellSessionConfig{Shell: "/bin/cat"},
		localFactory(&backend),
	)
	if err != nil {
		t.Fatalf("StartShellWithBackend: %v", err)
	}
	if err := sess.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	// Second Close must be a no-op (no panic, no error).
	if err := sess.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	// WriteInput / Resize on closed session must fail without crashing.
	if err := sess.WriteInput([]byte("x")); err == nil {
		t.Errorf("expected WriteInput after Close to error")
	}
	if err := sess.Resize(80, 24); err == nil {
		t.Errorf("expected Resize after Close to error")
	}
}

func TestShellSession_NoBackendReturnsErr(t *testing.T) {
	_, err := StartShellWithBackend(context.Background(),
		ShellSessionConfig{Shell: "/bin/cat"},
		nil,
	)
	if !errors.Is(err, ErrNoBackend) {
		t.Fatalf("expected ErrNoBackend, got %v", err)
	}
}

func TestShellSession_StartShellDefaultBackendNil(t *testing.T) {
	// DefaultSpawn is nil in tests unless SetDockerClient ran. StartShell
	// (the default-bound form) must surface ErrNoBackend rather than
	// panicking on a nil factory.
	old := DefaultSpawn
	DefaultSpawn = nil
	defer func() { DefaultSpawn = old }()
	_, err := StartShell(context.Background(), ShellSessionConfig{Shell: "/bin/echo x"})
	if !errors.Is(err, ErrNoBackend) {
		t.Fatalf("expected ErrNoBackend, got %v", err)
	}
}

func TestShellSession_SpawnError(t *testing.T) {
	failing := func(cfg ShellSessionConfig) (ShellBackend, error) {
		return nil, errors.New("boom")
	}
	_, err := StartShellWithBackend(context.Background(),
		ShellSessionConfig{Shell: "/bin/cat"},
		failing,
	)
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("expected spawn error to propagate, got %v", err)
	}
}

func TestShellSession_BackendSpawnError(t *testing.T) {
	// A backend whose Spawn returns an error must surface up through
	// StartShell rather than getting swallowed.
	failing := func(cfg ShellSessionConfig) (ShellBackend, error) {
		return &erroringBackend{}, nil
	}
	_, err := StartShellWithBackend(context.Background(),
		ShellSessionConfig{Shell: "/bin/cat"},
		failing,
	)
	if err == nil || !strings.Contains(err.Error(), "spawn-error") {
		t.Fatalf("expected spawn-error propagation, got %v", err)
	}
}

type erroringBackend struct{}

func (e *erroringBackend) Spawn(ctx context.Context, cfg ShellSessionConfig) (io.ReadWriteCloser, error) {
	return nil, errors.New("spawn-error")
}
func (e *erroringBackend) Resize(cols, rows uint16) error      { return nil }
func (e *erroringBackend) WaitExit(ctx context.Context) (int, error) { return -1, nil }

func TestShellSession_BackendReadErrorTriggersOnError(t *testing.T) {
	// A backend whose Read returns a non-EOF error (and whose WaitExit
	// also errors) must surface via OnError, not OnExit.
	gotError := make(chan string, 1)
	gotExit := make(chan struct{}, 1)
	failing := func(cfg ShellSessionConfig) (ShellBackend, error) {
		return &readFailBackend{}, nil
	}
	sess, err := StartShellWithBackend(context.Background(),
		ShellSessionConfig{
			Shell: "/bin/cat",
			OnOutput: func([]byte) {},
			OnExit: func(code int, reason string) {
				select {
				case gotExit <- struct{}{}:
				default:
				}
			},
			OnError: func(message string) {
				select {
				case gotError <- message:
				default:
				}
			},
		},
		failing,
	)
	if err != nil {
		t.Fatalf("StartShellWithBackend: %v", err)
	}
	defer sess.Close()
	select {
	case msg := <-gotError:
		if !strings.Contains(msg, "shell read") {
			t.Errorf("unexpected OnError message: %q", msg)
		}
	case <-gotExit:
		t.Errorf("expected OnError, not OnExit")
	case <-time.After(2 * time.Second):
		t.Fatalf("neither OnError nor OnExit fired")
	}
}

type readFailBackend struct{}

func (b *readFailBackend) Spawn(ctx context.Context, cfg ShellSessionConfig) (io.ReadWriteCloser, error) {
	return &errReader{}, nil
}
func (b *readFailBackend) Resize(cols, rows uint16) error { return nil }
func (b *readFailBackend) WaitExit(ctx context.Context) (int, error) {
	return -1, errors.New("inspect-failed")
}

type errReader struct{ done atomic.Bool }

func (e *errReader) Read(p []byte) (int, error) {
	if e.done.Swap(true) {
		return 0, errors.New("dead-pipe")
	}
	return 0, errors.New("dead-pipe")
}
func (e *errReader) Write(p []byte) (int, error) { return 0, errors.New("dead-pipe") }
func (e *errReader) Close() error                { return nil }
