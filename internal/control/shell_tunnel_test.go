package control

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/inferia/inferia-worker/internal/shellbridge"
	"nhooyr.io/websocket"
)

// fakeChannel captures every WriteEnvelope call so tests can assert
// exactly which frames went out and in what order.
type fakeChannel struct {
	mu        sync.Mutex
	envelopes []Envelope
	failNext  atomic.Int32
	closed    bool
}

func (f *fakeChannel) WriteEnvelope(ctx context.Context, env Envelope) error {
	if f.failNext.Add(-1) >= 0 {
		return errors.New("write rejected")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closed {
		return errors.New("closed")
	}
	f.envelopes = append(f.envelopes, env)
	return nil
}

func (f *fakeChannel) snapshot() []Envelope {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]Envelope, len(f.envelopes))
	copy(out, f.envelopes)
	return out
}

func (f *fakeChannel) typesIn(window time.Duration) []MessageType {
	deadline := time.Now().Add(window)
	for time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	envs := f.snapshot()
	out := make([]MessageType, len(envs))
	for i, e := range envs {
		out[i] = e.Type
	}
	return out
}

// fakeShellBackend is a deterministic test shell that emits scripted output,
// optionally records WriteInput, and returns a configurable exit code.
type fakeShellBackend struct {
	output       []byte
	exitCode     int
	exitErr      error
	resizeCalls  atomic.Int32
	writes       chan []byte
	pipeR, pipeW *io.PipeReader
	pipeWriter   *io.PipeWriter
	closeOnce    sync.Once
	startCh      chan struct{}
}

func newFakeShellBackend(output []byte, exitCode int) *fakeShellBackend {
	pr, pw := io.Pipe()
	return &fakeShellBackend{
		output:     output,
		exitCode:   exitCode,
		writes:     make(chan []byte, 16),
		pipeR:      pr,
		pipeWriter: pw,
		startCh:    make(chan struct{}),
	}
}

type fakeShellRWC struct {
	r   *io.PipeReader
	w   chan<- []byte
	b   *fakeShellBackend
	cmu sync.Mutex
	cl  bool
}

func (f *fakeShellRWC) Read(p []byte) (int, error)  { return f.r.Read(p) }
func (f *fakeShellRWC) Write(p []byte) (int, error) {
	f.w <- append([]byte(nil), p...)
	return len(p), nil
}
func (f *fakeShellRWC) Close() error {
	f.cmu.Lock()
	defer f.cmu.Unlock()
	if f.cl {
		return nil
	}
	f.cl = true
	_ = f.r.CloseWithError(io.EOF)
	return nil
}

func (b *fakeShellBackend) Spawn(ctx context.Context, cfg shellbridge.ShellSessionConfig) (io.ReadWriteCloser, error) {
	close(b.startCh)
	// Emit scripted output in a goroutine then close the writer so the
	// read pump sees EOF and calls WaitExit.
	go func() {
		if len(b.output) > 0 {
			_, _ = b.pipeWriter.Write(b.output)
		}
		if b.exitErr == nil {
			_ = b.pipeWriter.Close()
		} else {
			_ = b.pipeWriter.CloseWithError(b.exitErr)
		}
	}()
	return &fakeShellRWC{r: b.pipeR, w: b.writes, b: b}, nil
}
func (b *fakeShellBackend) Resize(cols, rows uint16) error { b.resizeCalls.Add(1); return nil }
func (b *fakeShellBackend) WaitExit(ctx context.Context) (int, error) {
	return b.exitCode, b.exitErr
}

// blockingShellBackend never produces output; lets tests exercise input/
// resize/close paths without a scripted EOF racing the assertion.
type blockingShellBackend struct {
	pipeR     *io.PipeReader
	pipeW     *io.PipeWriter
	writes    chan []byte
	resizes   atomic.Int32
	closeOnce sync.Once
}

func newBlockingShellBackend() *blockingShellBackend {
	pr, pw := io.Pipe()
	return &blockingShellBackend{
		pipeR:  pr,
		pipeW:  pw,
		writes: make(chan []byte, 16),
	}
}
func (b *blockingShellBackend) Spawn(ctx context.Context, cfg shellbridge.ShellSessionConfig) (io.ReadWriteCloser, error) {
	return &fakeShellRWC{r: b.pipeR, w: b.writes}, nil
}
func (b *blockingShellBackend) Resize(cols, rows uint16) error { b.resizes.Add(1); return nil }
func (b *blockingShellBackend) WaitExit(ctx context.Context) (int, error) {
	return 0, nil
}

// --- shell tunnel: open / output / exit -------------------------------------

func TestShellTunnel_OpenEmitsOutputAndExit(t *testing.T) {
	fc := &fakeChannel{}
	backend := newFakeShellBackend([]byte("hello"), 0)
	tunnel := NewChannelShellTunnelForTest(fc,
		func(ctx context.Context, cfg shellbridge.ShellSessionConfig) (*shellbridge.ShellSession, error) {
			return shellbridge.StartShellWithBackend(ctx, cfg,
				func(cfg shellbridge.ShellSessionConfig) (shellbridge.ShellBackend, error) { return backend, nil },
			)
		},
		nil,
	)
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellOpen,
		Body: ShellOpenBody{StreamID: "s1", Shell: "/bin/sh", Cols: 80, Rows: 24},
	})
	// Wait for output and exit to flow through.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		envs := fc.snapshot()
		if len(envs) >= 2 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	envs := fc.snapshot()
	if len(envs) < 2 {
		t.Fatalf("expected at least 2 envelopes (output + exit), got %d: %+v", len(envs), envs)
	}
	// First should be ShellOutput with "hello".
	if envs[0].Type != MsgShellOutput {
		t.Errorf("expected first envelope to be ShellOutput, got %s", envs[0].Type)
	}
	if body, ok := envs[0].Body.(ShellOutputBody); !ok || body.Data != "hello" || body.StreamID != "s1" {
		t.Errorf("unexpected first body: %+v", envs[0].Body)
	}
	if envs[1].Type != MsgShellExit {
		t.Errorf("expected second envelope to be ShellExit, got %s", envs[1].Type)
	}
}

// TestShellTunnel_InputAfterClose exercises the WriteInput error path:
// when the session is already closed (because tunnel.CloseAll ran), a
// subsequent ShellInput must be silently dropped rather than re-opening
// the session or crashing.
func TestShellTunnel_InputAfterClose(t *testing.T) {
	fc := &fakeChannel{}
	backend := newBlockingShellBackend()
	tunnel := NewChannelShellTunnelForTest(fc,
		func(ctx context.Context, cfg shellbridge.ShellSessionConfig) (*shellbridge.ShellSession, error) {
			return shellbridge.StartShellWithBackend(ctx, cfg,
				func(cfg shellbridge.ShellSessionConfig) (shellbridge.ShellBackend, error) { return backend, nil },
			)
		},
		nil,
	)
	tunnel.Handle(context.Background(), Envelope{Type: MsgShellOpen, Body: ShellOpenBody{StreamID: "z"}})
	// Close the session locally; input now hits the "session closed" path.
	tunnel.Handle(context.Background(), Envelope{Type: MsgShellClose, Body: ShellCloseBody{StreamID: "z"}})
	tunnel.Handle(context.Background(), Envelope{Type: MsgShellInput, Body: ShellInputBody{StreamID: "z", Data: "after-close"}})
	// No envelope should be emitted for an input on a closed session.
	envs := fc.snapshot()
	for _, e := range envs {
		if e.Type == MsgShellError {
			t.Errorf("input-after-close should not emit ShellError, got %+v", e)
		}
	}
}

func TestShellTunnel_InputRoutesToBackend(t *testing.T) {
	fc := &fakeChannel{}
	backend := newBlockingShellBackend()
	tunnel := NewChannelShellTunnelForTest(fc,
		func(ctx context.Context, cfg shellbridge.ShellSessionConfig) (*shellbridge.ShellSession, error) {
			return shellbridge.StartShellWithBackend(ctx, cfg,
				func(cfg shellbridge.ShellSessionConfig) (shellbridge.ShellBackend, error) { return backend, nil },
			)
		},
		nil,
	)
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellOpen,
		Body: ShellOpenBody{StreamID: "sx"},
	})
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellInput,
		Body: ShellInputBody{StreamID: "sx", Data: "echo hi\n"},
	})
	select {
	case got := <-backend.writes:
		if string(got) != "echo hi\n" {
			t.Errorf("expected 'echo hi\\n', got %q", got)
		}
	case <-time.After(time.Second):
		t.Fatalf("input did not reach backend")
	}
	// Unknown stream — silently dropped, no panic, no envelope.
	before := len(fc.snapshot())
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellInput,
		Body: ShellInputBody{StreamID: "missing", Data: "nope"},
	})
	if after := len(fc.snapshot()); after != before {
		t.Errorf("input to unknown stream produced %d new envelopes", after-before)
	}
	tunnel.CloseAll()
}

func TestShellTunnel_ResizeRoutesToBackend(t *testing.T) {
	fc := &fakeChannel{}
	backend := newBlockingShellBackend()
	tunnel := NewChannelShellTunnelForTest(fc,
		func(ctx context.Context, cfg shellbridge.ShellSessionConfig) (*shellbridge.ShellSession, error) {
			return shellbridge.StartShellWithBackend(ctx, cfg,
				func(cfg shellbridge.ShellSessionConfig) (shellbridge.ShellBackend, error) { return backend, nil },
			)
		},
		nil,
	)
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellOpen,
		Body: ShellOpenBody{StreamID: "r1"},
	})
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellResize,
		Body: ShellResizeBody{StreamID: "r1", Cols: 120, Rows: 40},
	})
	// Allow goroutine scheduling.
	time.Sleep(50 * time.Millisecond)
	if backend.resizes.Load() == 0 {
		t.Errorf("expected resize to reach backend")
	}
	// Resize on unknown stream → no-op.
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellResize,
		Body: ShellResizeBody{StreamID: "ghost", Cols: 80, Rows: 24},
	})
	tunnel.CloseAll()
}

func TestShellTunnel_CloseStopsSession(t *testing.T) {
	fc := &fakeChannel{}
	backend := newBlockingShellBackend()
	tunnel := NewChannelShellTunnelForTest(fc,
		func(ctx context.Context, cfg shellbridge.ShellSessionConfig) (*shellbridge.ShellSession, error) {
			return shellbridge.StartShellWithBackend(ctx, cfg,
				func(cfg shellbridge.ShellSessionConfig) (shellbridge.ShellBackend, error) { return backend, nil },
			)
		},
		nil,
	)
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellOpen,
		Body: ShellOpenBody{StreamID: "c1"},
	})
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellClose,
		Body: ShellCloseBody{StreamID: "c1"},
	})
	// Close is idempotent.
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellClose,
		Body: ShellCloseBody{StreamID: "c1"},
	})
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellClose,
		Body: ShellCloseBody{StreamID: "missing"},
	})
}

func TestShellTunnel_StartShellErrorEmitsShellError(t *testing.T) {
	fc := &fakeChannel{}
	tunnel := NewChannelShellTunnelForTest(fc,
		func(ctx context.Context, cfg shellbridge.ShellSessionConfig) (*shellbridge.ShellSession, error) {
			return nil, errors.New("spawn-broke")
		},
		nil,
	)
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellOpen,
		Body: ShellOpenBody{StreamID: "err1"},
	})
	envs := fc.snapshot()
	if len(envs) != 1 || envs[0].Type != MsgShellError {
		t.Fatalf("expected one ShellError envelope, got %+v", envs)
	}
	body, ok := envs[0].Body.(ShellErrorBody)
	if !ok || !strings.Contains(body.Message, "spawn-broke") || body.StreamID != "err1" {
		t.Errorf("unexpected ShellError body: %+v", envs[0].Body)
	}
}

func TestShellTunnel_OpenWithoutStreamIDErrors(t *testing.T) {
	fc := &fakeChannel{}
	tunnel := NewChannelShellTunnelForTest(fc,
		func(ctx context.Context, cfg shellbridge.ShellSessionConfig) (*shellbridge.ShellSession, error) {
			t.Fatalf("startShell must not be called for blank stream_id")
			return nil, nil
		},
		nil,
	)
	tunnel.Handle(context.Background(), Envelope{Type: MsgShellOpen, Body: ShellOpenBody{}})
	envs := fc.snapshot()
	if len(envs) != 1 || envs[0].Type != MsgShellError {
		t.Fatalf("expected single ShellError envelope, got %+v", envs)
	}
}

func TestShellTunnel_DuplicateOpenIsDropped(t *testing.T) {
	fc := &fakeChannel{}
	backend := newBlockingShellBackend()
	calls := 0
	tunnel := NewChannelShellTunnelForTest(fc,
		func(ctx context.Context, cfg shellbridge.ShellSessionConfig) (*shellbridge.ShellSession, error) {
			calls++
			return shellbridge.StartShellWithBackend(ctx, cfg,
				func(cfg shellbridge.ShellSessionConfig) (shellbridge.ShellBackend, error) { return backend, nil },
			)
		},
		nil,
	)
	tunnel.Handle(context.Background(), Envelope{Type: MsgShellOpen, Body: ShellOpenBody{StreamID: "dup"}})
	tunnel.Handle(context.Background(), Envelope{Type: MsgShellOpen, Body: ShellOpenBody{StreamID: "dup"}})
	if calls != 1 {
		t.Errorf("expected first-open-wins, got %d startShell calls", calls)
	}
	tunnel.CloseAll()
}

// --- logs tunnel ------------------------------------------------------------

// fakeLogsBackend scripts a sequence of lines and then ends.
type fakeLogsBackendCtl struct {
	lines     []struct{ stream, data string }
	endReason string
	blockUntilCancel bool
}

func (f *fakeLogsBackendCtl) Stream(ctx context.Context, onLine func(stream, data string)) (string, error) {
	for _, l := range f.lines {
		onLine(l.stream, l.data)
	}
	if f.blockUntilCancel {
		<-ctx.Done()
		return "", ctx.Err()
	}
	return f.endReason, nil
}

func TestLogsTunnel_OpenEmitsLinesAndEnd(t *testing.T) {
	fc := &fakeChannel{}
	backend := &fakeLogsBackendCtl{
		lines: []struct{ stream, data string }{
			{"stdout", "a"},
			{"stderr", "b"},
		},
		endReason: "exited",
	}
	tunnel := NewChannelShellTunnelForTest(fc,
		nil,
		func(ctx context.Context, cfg shellbridge.LogsSessionConfig) (*shellbridge.LogsSession, error) {
			return shellbridge.StartLogsWithBackend(ctx, cfg,
				func(cfg shellbridge.LogsSessionConfig) (shellbridge.LogsBackend, error) { return backend, nil },
			)
		},
	)
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgLogsOpen,
		Body: LogsOpenBody{StreamID: "l1", DeploymentID: "d-1"},
	})
	// Wait until we see three envelopes (LogsLine x2 + LogsEnd).
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if len(fc.snapshot()) >= 3 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	envs := fc.snapshot()
	if len(envs) < 3 {
		t.Fatalf("expected ≥3 envelopes, got %d: %+v", len(envs), envs)
	}
	if envs[0].Type != MsgLogsLine || envs[1].Type != MsgLogsLine {
		t.Errorf("expected two LogsLine, got %s, %s", envs[0].Type, envs[1].Type)
	}
	if envs[2].Type != MsgLogsEnd {
		t.Errorf("expected LogsEnd third, got %s", envs[2].Type)
	}
}

func TestLogsTunnel_CloseStopsStream(t *testing.T) {
	fc := &fakeChannel{}
	backend := &fakeLogsBackendCtl{blockUntilCancel: true}
	tunnel := NewChannelShellTunnelForTest(fc,
		nil,
		func(ctx context.Context, cfg shellbridge.LogsSessionConfig) (*shellbridge.LogsSession, error) {
			return shellbridge.StartLogsWithBackend(ctx, cfg,
				func(cfg shellbridge.LogsSessionConfig) (shellbridge.LogsBackend, error) { return backend, nil },
			)
		},
	)
	tunnel.Handle(context.Background(), Envelope{Type: MsgLogsOpen, Body: LogsOpenBody{StreamID: "lc1"}})
	tunnel.Handle(context.Background(), Envelope{Type: MsgLogsClose, Body: LogsCloseBody{StreamID: "lc1"}})
	tunnel.Handle(context.Background(), Envelope{Type: MsgLogsClose, Body: LogsCloseBody{StreamID: "lc1"}}) // idempotent
	tunnel.Handle(context.Background(), Envelope{Type: MsgLogsClose, Body: LogsCloseBody{StreamID: "ghost"}})
}

func TestLogsTunnel_OpenWithoutStreamIDEmitsLogsEnd(t *testing.T) {
	fc := &fakeChannel{}
	tunnel := NewChannelShellTunnelForTest(fc,
		nil,
		func(ctx context.Context, cfg shellbridge.LogsSessionConfig) (*shellbridge.LogsSession, error) {
			t.Fatalf("startLogs must not be called for blank stream_id")
			return nil, nil
		},
	)
	tunnel.Handle(context.Background(), Envelope{Type: MsgLogsOpen, Body: LogsOpenBody{}})
	envs := fc.snapshot()
	if len(envs) != 1 || envs[0].Type != MsgLogsEnd {
		t.Fatalf("expected one LogsEnd envelope, got %+v", envs)
	}
}

func TestLogsTunnel_StartErrorEmitsLogsEnd(t *testing.T) {
	fc := &fakeChannel{}
	tunnel := NewChannelShellTunnelForTest(fc,
		nil,
		func(ctx context.Context, cfg shellbridge.LogsSessionConfig) (*shellbridge.LogsSession, error) {
			return nil, errors.New("backend exploded")
		},
	)
	tunnel.Handle(context.Background(), Envelope{Type: MsgLogsOpen, Body: LogsOpenBody{StreamID: "leerr"}})
	envs := fc.snapshot()
	if len(envs) != 1 || envs[0].Type != MsgLogsEnd {
		t.Fatalf("expected single LogsEnd, got %+v", envs)
	}
	body, ok := envs[0].Body.(LogsEndBody)
	if !ok || !strings.Contains(body.Reason, "backend exploded") {
		t.Errorf("unexpected LogsEnd body: %+v", envs[0].Body)
	}
}

// --- frame routing ---------------------------------------------------------

func TestShellTunnel_DataPlaneFramesDropped(t *testing.T) {
	fc := &fakeChannel{}
	tunnel := NewChannelShellTunnelForTest(fc, nil, nil)
	for _, mt := range []MessageType{
		MsgShellOutput, MsgShellExit, MsgShellError, MsgLogsLine, MsgLogsEnd,
	} {
		consumed := tunnel.Handle(context.Background(), Envelope{Type: mt})
		if !consumed {
			t.Errorf("expected tunnel to consume %s (and log+drop), but it didn't", mt)
		}
	}
	if envs := fc.snapshot(); len(envs) != 0 {
		t.Errorf("unexpected outbound envelopes: %+v", envs)
	}
}

func TestShellTunnel_UnknownTypeNotConsumed(t *testing.T) {
	fc := &fakeChannel{}
	tunnel := NewChannelShellTunnelForTest(fc, nil, nil)
	if tunnel.Handle(context.Background(), Envelope{Type: MsgHeartbeat}) {
		t.Errorf("tunnel must not consume Heartbeat — leave it for the main switch")
	}
}

func TestShellTunnel_CloseAllStopsLiveSessions(t *testing.T) {
	fc := &fakeChannel{}
	backend := newBlockingShellBackend()
	logsBackend := &fakeLogsBackendCtl{blockUntilCancel: true}
	tunnel := NewChannelShellTunnelForTest(fc,
		func(ctx context.Context, cfg shellbridge.ShellSessionConfig) (*shellbridge.ShellSession, error) {
			return shellbridge.StartShellWithBackend(ctx, cfg,
				func(cfg shellbridge.ShellSessionConfig) (shellbridge.ShellBackend, error) { return backend, nil },
			)
		},
		func(ctx context.Context, cfg shellbridge.LogsSessionConfig) (*shellbridge.LogsSession, error) {
			return shellbridge.StartLogsWithBackend(ctx, cfg,
				func(cfg shellbridge.LogsSessionConfig) (shellbridge.LogsBackend, error) { return logsBackend, nil },
			)
		},
	)
	tunnel.Handle(context.Background(), Envelope{Type: MsgShellOpen, Body: ShellOpenBody{StreamID: "a"}})
	tunnel.Handle(context.Background(), Envelope{Type: MsgLogsOpen, Body: LogsOpenBody{StreamID: "b"}})
	tunnel.CloseAll()
	// After CloseAll, further opens must surface a ShellError /
	// LogsEnd with "tunnel closed".
	before := len(fc.snapshot())
	tunnel.Handle(context.Background(), Envelope{Type: MsgShellOpen, Body: ShellOpenBody{StreamID: "after"}})
	tunnel.Handle(context.Background(), Envelope{Type: MsgLogsOpen, Body: LogsOpenBody{StreamID: "after"}})
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if len(fc.snapshot()) >= before+2 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	envs := fc.snapshot()[before:]
	if len(envs) < 2 {
		t.Fatalf("expected ≥2 post-close envelopes, got %d: %+v", len(envs), envs)
	}
	// Order isn't guaranteed; just check both types appeared with the
	// expected "tunnel closed" reason.
	sawShellErr := false
	sawLogsEnd := false
	for _, e := range envs {
		switch b := e.Body.(type) {
		case ShellErrorBody:
			if strings.Contains(b.Message, "tunnel closed") {
				sawShellErr = true
			}
		case LogsEndBody:
			if strings.Contains(b.Reason, "tunnel closed") {
				sawLogsEnd = true
			}
		}
	}
	if !sawShellErr || !sawLogsEnd {
		t.Errorf("expected ShellError+LogsEnd post-close, got sawShellErr=%v sawLogsEnd=%v envs=%+v",
			sawShellErr, sawLogsEnd, envs)
	}
}

// --- write errors don't pin goroutines -------------------------------------

func TestShellTunnel_SendFailureDoesNotBlock(t *testing.T) {
	// Channel write returns an error — sendEnvelope must log+drop, not
	// pin the calling goroutine. We assert by completing within a
	// short deadline.
	fc := &fakeChannel{}
	fc.failNext.Store(10) // every write fails
	tunnel := NewChannelShellTunnelForTest(fc, nil, nil)
	done := make(chan struct{})
	go func() {
		tunnel.Handle(context.Background(), Envelope{Type: MsgShellOpen, Body: ShellOpenBody{}})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatalf("Handle blocked on a failing sender")
	}
}

// --- bad bodies surface cleanly --------------------------------------------

func TestShellTunnel_BadBodiesAreDroppedNotCrash(t *testing.T) {
	fc := &fakeChannel{}
	tunnel := NewChannelShellTunnelForTest(fc, nil, nil)
	// Each of these would normally route through remarshal then onto a
	// handler; pass an invalid body (a string) and assert no panic.
	for _, mt := range []MessageType{
		MsgShellInput, MsgShellResize, MsgShellClose, MsgLogsClose,
	} {
		tunnel.Handle(context.Background(), Envelope{Type: mt, Body: 12345})
	}
}

// badBody is a json-incompatible payload — remarshal of a chan can't
// produce a typed body, so the handler emits a ShellError / LogsEnd.
type badBody struct{ Ch chan int }

func TestShellTunnel_BadOpenBodyEmitsError(t *testing.T) {
	fc := &fakeChannel{}
	tunnel := NewChannelShellTunnelForTest(fc, nil, nil)
	tunnel.Handle(context.Background(), Envelope{Type: MsgShellOpen, Body: badBody{Ch: make(chan int)}})
	envs := fc.snapshot()
	if len(envs) != 1 || envs[0].Type != MsgShellError {
		t.Fatalf("expected one ShellError on bad ShellOpen, got %+v", envs)
	}
	tunnel.Handle(context.Background(), Envelope{Type: MsgLogsOpen, Body: badBody{Ch: make(chan int)}})
	envs = fc.snapshot()
	if len(envs) != 2 || envs[1].Type != MsgLogsEnd {
		t.Fatalf("expected LogsEnd on bad LogsOpen, got %+v", envs)
	}
}

// --- channelSender (the production WriteEnvelope bridge) ------------------

func TestChannelSender_NilStateErrors(t *testing.T) {
	cs := channelSender{state: nil}
	if err := cs.WriteEnvelope(context.Background(), Envelope{Type: MsgShellOutput}); err == nil {
		t.Fatalf("expected error with nil state")
	}
}

func TestChannelSender_NilConnErrors(t *testing.T) {
	cs := channelSender{state: &connState{}}
	if err := cs.WriteEnvelope(context.Background(), Envelope{Type: MsgShellOutput}); err == nil {
		t.Fatalf("expected error with nil conn")
	}
}

// TestChannelSender_RealWriteThroughChannel exercises the full path: a
// fake CP accepts the connection, the worker side opens a tunnel and
// emits a ShellError envelope, the fake CP reads the frame back. Proves
// the serializing writer actually pumps bytes onto the WS.
func TestChannelSender_RealWriteThroughChannel(t *testing.T) {
	disp := newFakeDispatcher()
	sawShellError := make(chan struct{}, 1)

	cp := newFakeCPServer(func(ctx context.Context, c *websocket.Conn, s *fakeCPServer) {
		// Send ShellOpen with an empty stream_id — that triggers a
		// ShellError reply via the tunnel's serializing sender.
		_ = writeJSON(ctx, c, Envelope{Type: MsgShellOpen, ID: "open-1", Body: ShellOpenBody{}})
		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			var env Envelope
			_ = json.Unmarshal(data, &env)
			if env.Type == MsgShellError {
				select {
				case sawShellError <- struct{}{}:
				default:
				}
				return
			}
		}
	})
	defer cp.srv.Close()

	ch := &Channel{
		ChannelURL:        wsURL(cp.srv.URL) + "/v1/workers/channel",
		Token:             func() string { return "jwt-x" },
		HeartbeatInterval: 1 * time.Hour, // suppress heartbeat noise
		Dispatcher:        disp,
		DedupTTL:          100 * time.Millisecond,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	go ch.Run(ctx)
	select {
	case <-sawShellError:
	case <-ctx.Done():
		t.Fatalf("did not observe ShellError flow back through channelSender")
	}
}

func TestUint16Safe(t *testing.T) {
	cases := []struct {
		in   int
		want uint16
	}{
		{0, 0},
		{-1, 0},
		{1, 1},
		{0xffff, 0xffff},
		{0x10000, 0}, // out of range → 0
	}
	for _, c := range cases {
		if got := uint16safe(c.in); got != c.want {
			t.Errorf("uint16safe(%d)=%d, want %d", c.in, got, c.want)
		}
	}
}
