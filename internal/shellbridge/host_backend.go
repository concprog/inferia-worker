// host_backend.go: spawn a shell on the EC2 host (or any docker host) by
// launching a privileged sidecar container that nsenters into PID 1's
// namespaces. Used by the channel tunnel and admin /v1/shell when the
// caller didn't pin a deployment or container — i.e. the dashboard's
// generic "web shell" tab.
//
// Why a container instead of os/exec? The worker itself runs in a
// distroless container that has neither `nsenter` nor `bash`; even an
// alpine-based worker wouldn't have the host's filesystem in scope. The
// canonical "host shell" pattern is to spin up a side container with the
// host's PID / mount / net / ipc / uts namespaces and exec there. The
// container is one-shot and AutoRemove'd on exit so we don't accumulate
// dead sidecars when sessions churn.
package shellbridge

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/client"
)

// DefaultHostShellImage is the OCI image used by the host backend when
// INFERIA_HOST_SHELL_IMAGE is unset. ubuntu:22.04 ships nsenter via
// util-linux in the base image, avoiding a first-run apk add and the
// associated startup latency. ~80 MB one-time pull is acceptable for
// admin-only shell flows.
const DefaultHostShellImage = "ubuntu:22.04"

// HostShellImage returns the image the host backend should launch.
// Reads INFERIA_HOST_SHELL_IMAGE and falls back to DefaultHostShellImage.
// Operators can pin a smaller / pre-cached image (e.g. a private mirror
// of busybox+util-linux) without rebuilding the worker.
func HostShellImage() string {
	if v := strings.TrimSpace(os.Getenv("INFERIA_HOST_SHELL_IMAGE")); v != "" {
		return v
	}
	return DefaultHostShellImage
}

// buildNsenterCmd assembles the argv for the sidecar's CMD. The container
// always runs `nsenter -t 1 -m -u -i -n -p --` to enter the host's mount,
// uts, ipc, net, and pid namespaces, then exec's the requested shell as
// the requested user.
//
// User="" or "root" → exec the shell directly with a -l login flag so
//
//	the user lands in a fully initialised environment ($PATH, $HOME).
//
// User="ubuntu" (or any non-root) → use `su - <user> -s <shell>` so the
//
//	host's pam/login machinery applies. We need a real login session,
//	not just setuid, so files in /etc/profile.d and ~/.bashrc fire.
//
// Shell="" defaults to /bin/bash. The caller can override (e.g. /bin/zsh,
// /bin/sh) for hosts that don't have bash.
func buildNsenterCmd(cfg ShellSessionConfig) []string {
	shell := cfg.Shell
	if shell == "" {
		shell = "/bin/bash"
	}
	base := []string{"nsenter", "-t", "1", "-m", "-u", "-i", "-n", "-p", "--"}
	if cfg.User == "" || cfg.User == "root" {
		return append(base, shell, "-l")
	}
	return append(base, "su", "-", cfg.User, "-s", shell)
}

// buildHostContainerConfigs constructs the Config + HostConfig pair the
// SDK needs to launch the sidecar. Extracted from StartShell so tests can
// inspect the spec without driving an httptest server through the full
// pull/create/attach choreography.
func buildHostContainerConfigs(cfg ShellSessionConfig, img string) (*container.Config, *container.HostConfig) {
	return &container.Config{
			Image:        img,
			Cmd:          buildNsenterCmd(cfg),
			OpenStdin:    true,
			AttachStdin:  true,
			AttachStdout: true,
			AttachStderr: true,
			Tty:          true,
			Env:          []string{"TERM=xterm-256color"},
		}, &container.HostConfig{
			Privileged:  true,
			PidMode:     "host",
			NetworkMode: "host",
			IpcMode:     "host",
			UTSMode:     "host",
			AutoRemove:  true,
			SecurityOpt: []string{"apparmor=unconfined", "seccomp=unconfined"},
		}
}

// hostBackend implements ShellBackend by launching a privileged sidecar
// container. One instance per shell session — the containerID is set
// during Spawn and torn down via Close() on the returned ReadWriteCloser
// (which AutoRemove handles for us on exit).
type hostBackend struct {
	cli         *client.Client
	image       string
	containerID string
}

// NewHostShellBackend constructs a ShellBackend backed by a privileged
// sidecar. The docker client is shared with the rest of the worker —
// the host backend doesn't need a separate daemon connection.
func NewHostShellBackend(cli *client.Client) ShellBackend {
	return &hostBackend{cli: cli, image: HostShellImage()}
}

// Spawn pulls the image (if not cached), creates the sidecar, attaches a
// hijacked stdin/stdout duplex, starts the container, and returns the
// duplex as the session's ReadWriteCloser. Order matters: attach BEFORE
// start so we don't miss the shell's prompt banner.
func (b *hostBackend) Spawn(ctx context.Context, cfg ShellSessionConfig) (io.ReadWriteCloser, error) {
	if b.cli == nil {
		return nil, errors.New("shellbridge: host backend has no docker client")
	}
	if err := b.ensureImage(ctx); err != nil {
		return nil, fmt.Errorf("host shell image: %w", err)
	}
	containerCfg, hostCfg := buildHostContainerConfigs(cfg, b.image)
	created, err := b.cli.ContainerCreate(ctx, containerCfg, hostCfg, nil, nil, "")
	if err != nil {
		return nil, fmt.Errorf("host shell create: %w", err)
	}
	b.containerID = created.ID
	hijack, err := b.cli.ContainerAttach(ctx, created.ID, container.AttachOptions{
		Stream: true,
		Stdin:  true,
		Stdout: true,
		Stderr: true,
	})
	if err != nil {
		// Created but never attached — best-effort kill so AutoRemove fires.
		_ = b.cli.ContainerRemove(context.Background(), created.ID, container.RemoveOptions{Force: true})
		return nil, fmt.Errorf("host shell attach: %w", err)
	}
	if err := b.cli.ContainerStart(ctx, created.ID, container.StartOptions{}); err != nil {
		hijack.Close()
		_ = b.cli.ContainerRemove(context.Background(), created.ID, container.RemoveOptions{Force: true})
		return nil, fmt.Errorf("host shell start: %w", err)
	}
	return &hostHijackRWC{conn: hijack.Conn, reader: hijack.Reader, tty: containerCfg.Tty}, nil
}

// Resize forwards a PTY size change to the running sidecar. The container
// runs with Tty=true so ContainerResize is supported. Errors are surfaced
// but rare in practice — the docker daemon doesn't synchronise resize
// with the in-flight stream.
func (b *hostBackend) Resize(cols, rows uint16) error {
	if b.containerID == "" || b.cli == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return b.cli.ContainerResize(ctx, b.containerID, container.ResizeOptions{
		Height: uint(rows),
		Width:  uint(cols),
	})
}

// WaitExit polls ContainerWait for the sidecar's exit status. Returns -1
// on uninitialised backends (Spawn failed before containerID was set) so
// the read pump can finish cleanly without surfacing a spurious error.
func (b *hostBackend) WaitExit(ctx context.Context) (int, error) {
	if b.containerID == "" || b.cli == nil {
		return -1, nil
	}
	// Use a bounded inner ctx so a wedged daemon can't pin the goroutine
	// forever. 5s is enough for the kernel to reap a process that's
	// already exited; if the daemon is broken we degrade to "unknown".
	deadline, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	statusC, errC := b.cli.ContainerWait(deadline, b.containerID, container.WaitConditionNotRunning)
	select {
	case s := <-statusC:
		return int(s.StatusCode), nil
	case err := <-errC:
		return -1, err
	case <-deadline.Done():
		return -1, deadline.Err()
	}
}

// ensureImage is a no-op if the image is already cached locally; otherwise
// it pulls and drains the progress stream. The first call on a fresh host
// can take a minute for ubuntu:22.04, but subsequent sessions reuse the
// cached image and start in <1s.
func (b *hostBackend) ensureImage(ctx context.Context) error {
	if b.cli == nil {
		return errors.New("nil docker client")
	}
	// Check for a cached image first so the happy path doesn't issue a
	// pull that the daemon would no-op anyway. Filter by reference so we
	// don't paginate the full image list.
	summaries, err := b.cli.ImageList(ctx, image.ListOptions{})
	if err == nil {
		for _, s := range summaries {
			for _, tag := range s.RepoTags {
				if tag == b.image {
					return nil
				}
			}
		}
	}
	rc, err := b.cli.ImagePull(ctx, b.image, image.PullOptions{})
	if err != nil {
		return fmt.Errorf("pull %s: %w", b.image, err)
	}
	defer rc.Close()
	// Drain the JSON progress stream so the pull actually completes
	// before we return.
	if _, err := io.Copy(io.Discard, rc); err != nil {
		return fmt.Errorf("drain pull: %w", err)
	}
	return nil
}

// hostHijackRWC adapts the Docker SDK hijack response to io.ReadWriteCloser.
//
// On TTY mode, the daemon writes raw bytes (no 8-byte frame header), so
// we read directly from the bufio.Reader. On non-TTY mode (which we
// never use for the host backend, but tests might exercise the helper),
// stdcopy.StdCopy would be needed to demultiplex. Keeping the code path
// tight: this implementation is TTY-only.
type hostHijackRWC struct {
	conn   io.WriteCloser
	reader io.Reader
	tty    bool
}

func (h *hostHijackRWC) Read(p []byte) (int, error) {
	if h.reader == nil {
		return 0, io.EOF
	}
	return h.reader.Read(p)
}

func (h *hostHijackRWC) Write(p []byte) (int, error) {
	if h.conn == nil {
		return 0, errors.New("hostHijackRWC: nil conn")
	}
	return h.conn.Write(p)
}

func (h *hostHijackRWC) Close() error {
	if h.conn != nil {
		return h.conn.Close()
	}
	return nil
}

// SetHostShellBackend installs the host-shell factory used by both the
// channel tunnel and the admin /v1/shell endpoint when the caller didn't
// pin a deployment/container. Mirrors SetDockerClient's one-shot,
// boot-time semantics — safe to call from main, not designed for repeated
// re-configuration. After this is called, StartHostShell returns a fresh
// session each time it's invoked.
func SetHostShellBackend(cli *client.Client) {
	DefaultHostSpawn = func(cfg ShellSessionConfig) (ShellBackend, error) {
		return NewHostShellBackend(cli), nil
	}
}

// DefaultHostSpawn is the package-level factory for host-mode sessions.
// nil until SetHostShellBackend is called; callers must guard against
// nil so a misconfigured worker fails the ShellOpen frame cleanly.
var DefaultHostSpawn SpawnBackend

// StartHostShell is the host-mode counterpart to StartShell. It uses the
// package-level DefaultHostSpawn rather than DefaultSpawn. Both
// StartShell paths route through the same startShellWith helper so the
// readPump / OnError / OnExit lifecycle is identical.
func StartHostShell(ctx context.Context, cfg ShellSessionConfig) (*ShellSession, error) {
	return startShellWith(ctx, cfg, DefaultHostSpawn)
}
