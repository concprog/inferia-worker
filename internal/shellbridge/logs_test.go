package shellbridge

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeLogsBackend emits a scripted sequence of lines then either ends
// cleanly (with a configurable reason) or returns an error. Test-only;
// keeps the suite docker-free.
type fakeLogsBackend struct {
	scriptedLines []struct {
		stream string
		data   string
	}
	endReason string
	endError  error
	// blockUntilCancel waits on ctx after emitting the script — used to
	// exercise the Close path.
	blockUntilCancel bool
}

func (f *fakeLogsBackend) Stream(ctx context.Context, onLine func(stream, data string)) (string, error) {
	for _, l := range f.scriptedLines {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}
		onLine(l.stream, l.data)
	}
	if f.blockUntilCancel {
		<-ctx.Done()
		return "", ctx.Err()
	}
	return f.endReason, f.endError
}

func TestLogsSession_EmitsAllLinesAndEnds(t *testing.T) {
	var linesMu sync.Mutex
	var lines []string
	endCh := make(chan string, 1)

	backend := &fakeLogsBackend{
		scriptedLines: []struct{ stream, data string }{
			{"stdout", "first"},
			{"stderr", "boom"},
			{"stdout", "second"},
		},
		endReason: "container exited",
	}
	sess, err := StartLogsWithBackend(context.Background(),
		LogsSessionConfig{
			Container: "deadbeef",
			OnLine: func(stream, data string) {
				linesMu.Lock()
				lines = append(lines, stream+":"+data)
				linesMu.Unlock()
			},
			OnEnd: func(reason string) {
				select {
				case endCh <- reason:
				default:
				}
			},
		},
		func(cfg LogsSessionConfig) (LogsBackend, error) { return backend, nil },
	)
	if err != nil {
		t.Fatalf("StartLogsWithBackend: %v", err)
	}
	defer sess.Close()

	select {
	case reason := <-endCh:
		if reason != "container exited" {
			t.Errorf("expected OnEnd reason 'container exited', got %q", reason)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("OnEnd did not fire")
	}

	linesMu.Lock()
	defer linesMu.Unlock()
	want := []string{"stdout:first", "stderr:boom", "stdout:second"}
	if strings.Join(lines, "|") != strings.Join(want, "|") {
		t.Errorf("lines=%v, want=%v", lines, want)
	}
}

func TestLogsSession_CloseSuppressesOnEnd(t *testing.T) {
	endFired := make(chan struct{}, 1)
	backend := &fakeLogsBackend{blockUntilCancel: true}

	sess, err := StartLogsWithBackend(context.Background(),
		LogsSessionConfig{
			Container: "deadbeef",
			OnLine:    func(stream, data string) {},
			OnEnd: func(reason string) {
				select {
				case endFired <- struct{}{}:
				default:
				}
			},
		},
		func(cfg LogsSessionConfig) (LogsBackend, error) { return backend, nil },
	)
	if err != nil {
		t.Fatalf("StartLogsWithBackend: %v", err)
	}
	// Close before the backend ends naturally — OnEnd must NOT fire.
	if err := sess.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	select {
	case <-endFired:
		t.Errorf("OnEnd fired after caller-initiated Close")
	case <-time.After(150 * time.Millisecond):
		// expected — Close suppresses OnEnd
	}
}

func TestLogsSession_CloseIsIdempotent(t *testing.T) {
	backend := &fakeLogsBackend{blockUntilCancel: true}
	sess, err := StartLogsWithBackend(context.Background(),
		LogsSessionConfig{
			Container: "x",
			OnLine:    func(stream, data string) {},
		},
		func(cfg LogsSessionConfig) (LogsBackend, error) { return backend, nil },
	)
	if err != nil {
		t.Fatalf("StartLogsWithBackend: %v", err)
	}
	if err := sess.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := sess.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestLogsSession_NoBackend(t *testing.T) {
	_, err := StartLogsWithBackend(context.Background(),
		LogsSessionConfig{Container: "x"},
		nil,
	)
	if !errors.Is(err, ErrNoLogsBackend) {
		t.Fatalf("expected ErrNoLogsBackend, got %v", err)
	}
}

func TestLogsSession_StartLogsDefaultLogsSpawnNil(t *testing.T) {
	old := DefaultLogsSpawn
	DefaultLogsSpawn = nil
	defer func() { DefaultLogsSpawn = old }()
	_, err := StartLogs(context.Background(), LogsSessionConfig{Container: "x"})
	if !errors.Is(err, ErrNoLogsBackend) {
		t.Fatalf("expected ErrNoLogsBackend, got %v", err)
	}
}

func TestLogsSession_FactoryError(t *testing.T) {
	_, err := StartLogsWithBackend(context.Background(),
		LogsSessionConfig{Container: "x"},
		func(cfg LogsSessionConfig) (LogsBackend, error) {
			return nil, errors.New("resolve failed")
		},
	)
	if err == nil || !strings.Contains(err.Error(), "resolve failed") {
		t.Fatalf("expected factory error to propagate, got %v", err)
	}
}

func TestLogsSession_BackendErrorBecomesOnEndReason(t *testing.T) {
	endCh := make(chan string, 1)
	backend := &fakeLogsBackend{endError: errors.New("docker-disconnected")}
	sess, err := StartLogsWithBackend(context.Background(),
		LogsSessionConfig{
			Container: "x",
			OnLine:    func(stream, data string) {},
			OnEnd: func(reason string) {
				select {
				case endCh <- reason:
				default:
				}
			},
		},
		func(cfg LogsSessionConfig) (LogsBackend, error) { return backend, nil },
	)
	if err != nil {
		t.Fatalf("StartLogsWithBackend: %v", err)
	}
	defer sess.Close()
	select {
	case reason := <-endCh:
		if !strings.Contains(reason, "docker-disconnected") {
			t.Errorf("expected error to surface in OnEnd reason, got %q", reason)
		}
	case <-time.After(time.Second):
		t.Fatalf("OnEnd did not fire")
	}
}

func TestForwardLines_SplitsAndFlushesPartial(t *testing.T) {
	// Drives the internal line-splitter — it has to handle:
	//   - multi-line chunks
	//   - lines split across two reads
	//   - a trailing partial without newline at EOF
	r, w := io.Pipe()
	gotCh := make(chan string, 8)

	go func() {
		forwardLines(r, "stdout", func(stream, line string) {
			gotCh <- stream + ":" + line
		})
		close(gotCh)
	}()

	// Write "line1\npar" then "tial\nfinal-no-newline" then close.
	go func() {
		_, _ = w.Write([]byte("line1\npar"))
		_, _ = w.Write([]byte("tial\nfinal-no-newline"))
		_ = w.Close()
	}()

	var got []string
	for s := range gotCh {
		got = append(got, s)
	}
	want := []string{
		"stdout:line1",
		"stdout:partial",
		"stdout:final-no-newline",
	}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("got=%v, want=%v", got, want)
	}
}

func TestLogsSession_DefaultTailIsPopulated(t *testing.T) {
	// LogsSessionConfig.Tail==0 must be normalised to 200 (matches the
	// legacy /v1/logs behaviour). The factory inspects cfg.Tail to
	// configure the backend, so we capture it here.
	var seenTail int
	_, err := StartLogsWithBackend(context.Background(),
		LogsSessionConfig{
			Container: "x",
			OnLine:    func(stream, data string) {},
		},
		func(cfg LogsSessionConfig) (LogsBackend, error) {
			seenTail = cfg.Tail
			return &fakeLogsBackend{blockUntilCancel: true}, nil
		},
	)
	if err != nil {
		t.Fatalf("StartLogsWithBackend: %v", err)
	}
	if seenTail != 200 {
		t.Errorf("expected default Tail=200, got %d", seenTail)
	}
}
