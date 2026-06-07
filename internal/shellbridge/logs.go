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
	"github.com/docker/docker/pkg/stdcopy"
)

// LogsBackend abstracts the transport that streams container logs. Docker
// is the only production backend; tests inject a fake that emits scripted
// lines without touching a real daemon.
type LogsBackend interface {
	// Stream blocks until ctx is cancelled or the backing log stream ends,
	// emitting one OnLine call per newline-delimited record. Returns the
	// reason the stream ended (e.g. "container exited", "follow timeout",
	// ""=unknown).
	Stream(ctx context.Context, onLine func(stream, data string)) (reason string, err error)
}

// LogsSpawn is the factory used by StartLogs.
type LogsSpawn func(cfg LogsSessionConfig) (LogsBackend, error)

// DefaultLogsSpawn is the package-level factory installed at boot. Nil until
// SetDockerClient is called.
var DefaultLogsSpawn LogsSpawn

// LogsSessionConfig fully describes one logs session.
type LogsSessionConfig struct {
	Deployment string
	Container  string

	// Tail is the number of lines to replay before following. Defaults to
	// 200 — matches the legacy /v1/logs endpoint.
	Tail int

	// OnLine receives one decoded log record. stream is "stdout" or
	// "stderr". Concurrent with all other callbacks, the consumer must
	// serialize its own writes.
	OnLine func(stream, data string)
	// OnEnd fires once when the stream ends (container exited or follow
	// timed out). reason is a free-form explanation.
	OnEnd func(reason string)
}

// LogsSession is one live logs stream. Returned by StartLogs, terminated by
// Close or by the backing stream ending.
type LogsSession struct {
	backend LogsBackend
	cfg     LogsSessionConfig

	closed atomic.Bool

	cancel context.CancelFunc
	done   chan struct{}
}

// StartLogs spins up a logs session using DefaultLogsSpawn (production).
// Tests should use StartLogsWithBackend.
func StartLogs(ctx context.Context, cfg LogsSessionConfig) (*LogsSession, error) {
	return startLogsWith(ctx, cfg, DefaultLogsSpawn)
}

// StartLogsWithBackend is the explicit form used by tests.
func StartLogsWithBackend(ctx context.Context, cfg LogsSessionConfig, factory LogsSpawn) (*LogsSession, error) {
	return startLogsWith(ctx, cfg, factory)
}

// ErrNoLogsBackend is returned when DefaultLogsSpawn is nil and no explicit
// backend was supplied.
var ErrNoLogsBackend = errors.New("shellbridge: no logs backend configured")

func startLogsWith(ctx context.Context, cfg LogsSessionConfig, factory LogsSpawn) (*LogsSession, error) {
	if factory == nil {
		return nil, ErrNoLogsBackend
	}
	if cfg.Tail == 0 {
		cfg.Tail = 200
	}
	backend, err := factory(cfg)
	if err != nil {
		return nil, err
	}
	sessionCtx, cancel := context.WithCancel(ctx)
	s := &LogsSession{
		backend: backend,
		cfg:     cfg,
		cancel:  cancel,
		done:    make(chan struct{}),
	}
	go s.run(sessionCtx)
	return s, nil
}

// Close terminates the logs session. Idempotent. Does not fire OnEnd.
func (s *LogsSession) Close() error {
	if !s.closed.CompareAndSwap(false, true) {
		return nil
	}
	if s.cancel != nil {
		s.cancel()
	}
	select {
	case <-s.done:
	case <-time.After(2 * time.Second):
	}
	return nil
}

func (s *LogsSession) run(ctx context.Context) {
	defer close(s.done)
	reason, err := s.backend.Stream(ctx, func(stream, data string) {
		if s.closed.Load() {
			return
		}
		if s.cfg.OnLine != nil {
			s.cfg.OnLine(stream, data)
		}
	})
	if s.closed.Load() {
		return // caller-initiated close, no OnEnd
	}
	if err != nil && reason == "" {
		reason = err.Error()
	}
	if s.cfg.OnEnd != nil {
		s.cfg.OnEnd(reason)
	}
}

// SetDockerLogsBackend installs the docker-backed default logs spawner.
// Separate from SetDockerClient so callers can choose to wire only one
// surface (e.g. unit tests).
func SetDockerLogsBackend(cli *client.Client, rt Runtime) {
	DefaultLogsSpawn = func(cfg LogsSessionConfig) (LogsBackend, error) {
		cid, err := ResolveContainer(cli, rt, cfg.Container, cfg.Deployment)
		if err != nil {
			return nil, err
		}
		return NewDockerLogsBackend(cli, cid, cfg.Tail), nil
	}
}

// NewDockerLogsBackend constructs a LogsBackend backed by the Docker SDK
// against an already-resolved containerID. Exported so the admin /v1/logs
// handler can short-circuit ResolveContainer (admin runs its own query-
// string resolution and already knows the container ID).
func NewDockerLogsBackend(cli *client.Client, containerID string, tail int) LogsBackend {
	if tail <= 0 {
		tail = 200
	}
	return &dockerLogsBackend{cli: cli, containerID: containerID, tail: tail}
}

// --- Docker-backed logs backend --------------------------------------------

type dockerLogsBackend struct {
	cli         *client.Client
	containerID string
	tail        int
}

func (b *dockerLogsBackend) Stream(ctx context.Context, onLine func(stream, data string)) (string, error) {
	rc, err := b.cli.ContainerLogs(ctx, b.containerID, container.LogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Follow:     true,
		Tail:       fmt.Sprintf("%d", b.tail),
		Timestamps: false,
	})
	if err != nil {
		return "", fmt.Errorf("docker logs: %w", err)
	}
	defer rc.Close()
	return demuxAndForward(rc, onLine)
}

// demuxAndForward reads docker's multiplexed stdout/stderr stream, splits
// it line by line, and pumps each through onLine. Returns ("container
// exited", nil) at clean EOF.
func demuxAndForward(rc io.ReadCloser, onLine func(stream, data string)) (string, error) {
	outR, outW := io.Pipe()
	errR, errW := io.Pipe()
	defer outR.Close()
	defer errR.Close()

	// Forward both streams concurrently while StdCopy demuxes the
	// frame-prefixed docker stream.
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); forwardLines(outR, "stdout", onLine) }()
	go func() { defer wg.Done(); forwardLines(errR, "stderr", onLine) }()

	_, copyErr := stdcopy.StdCopy(outW, errW, rc)
	_ = outW.Close()
	_ = errW.Close()
	wg.Wait()
	if copyErr != nil && !errors.Is(copyErr, io.EOF) {
		return "", copyErr
	}
	return "container exited", nil
}

// forwardLines reads from r line-by-line and emits each via fn. EOF/errors
// silently end the loop — the caller wraps everything in one cancellable
// context so the goroutine winds up cleanly.
func forwardLines(r io.Reader, stream string, fn func(stream, data string)) {
	buf := make([]byte, 4096)
	var partial []byte
	for {
		n, err := r.Read(buf)
		if n > 0 {
			data := append(partial, buf[:n]...)
			partial = nil
			start := 0
			for i, b := range data {
				if b == '\n' {
					line := string(data[start:i])
					fn(stream, line)
					start = i + 1
				}
			}
			if start < len(data) {
				partial = append(partial, data[start:]...)
			}
		}
		if err != nil {
			if len(partial) > 0 {
				fn(stream, string(partial))
			}
			return
		}
	}
}
