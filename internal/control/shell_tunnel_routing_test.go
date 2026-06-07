// Tests for ChannelShellTunnel's host-vs-docker routing. The decision
// happens in handleShellOpen: empty deployment_id AND empty container_id
// → host backend; either field set → docker backend. This file owns the
// routing assertions so the broader shell_tunnel_test.go can stay focused
// on lifecycle (open/input/exit/close) rather than routing details.
package control

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/inferia/inferia-worker/internal/shellbridge"
)

// routingProbe is a starter that records whether it was invoked. Pairing
// two probes (one as docker, one as host) lets a single test assert
// exactly which path the tunnel took without driving a real backend.
type routingProbe struct {
	calls   atomic.Int32
	lastCfg shellbridge.ShellSessionConfig
}

func (r *routingProbe) starter() shellStarter {
	return func(ctx context.Context, cfg shellbridge.ShellSessionConfig) (*shellbridge.ShellSession, error) {
		r.calls.Add(1)
		r.lastCfg = cfg
		return shellbridge.StartShellWithBackend(ctx, cfg,
			func(cfg shellbridge.ShellSessionConfig) (shellbridge.ShellBackend, error) {
				return newBlockingShellBackend(), nil
			},
		)
	}
}

// TestShellTunnel_EmptyTargetRoutesToHostBackend exercises the canonical
// case: a dashboard opens the Shell tab without selecting a deployment.
// Both DeploymentID and ContainerID are empty in the inbound ShellOpen
// envelope. The tunnel must invoke the host starter (and NOT the docker
// starter) so the operator gets a real shell on the EC2 host instead of
// a doomed docker-exec into the distroless worker container.
func TestShellTunnel_EmptyTargetRoutesToHostBackend(t *testing.T) {
	fc := &fakeChannel{}
	docker := &routingProbe{}
	host := &routingProbe{}
	tunnel := NewChannelShellTunnelForTest(fc, docker.starter(), nil).WithHostStarter(host.starter())
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellOpen,
		Body: ShellOpenBody{StreamID: "host-1"},
	})
	if got := host.calls.Load(); got != 1 {
		t.Errorf("expected host starter to be invoked once, got %d", got)
	}
	if got := docker.calls.Load(); got != 0 {
		t.Errorf("docker starter must not be invoked for empty target, got %d calls", got)
	}
	tunnel.CloseAll()
}

// TestShellTunnel_DeploymentTargetRoutesToDockerBackend verifies that any
// non-empty deployment_id still goes through the docker-exec path. This
// is the existing behaviour we mustn't break — the dashboard's per-
// container Shell tab supplies deployment_id and expects to land inside
// the model container.
func TestShellTunnel_DeploymentTargetRoutesToDockerBackend(t *testing.T) {
	fc := &fakeChannel{}
	docker := &routingProbe{}
	host := &routingProbe{}
	tunnel := NewChannelShellTunnelForTest(fc, docker.starter(), nil).WithHostStarter(host.starter())
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellOpen,
		Body: ShellOpenBody{StreamID: "dep-1", DeploymentID: "d1"},
	})
	if got := docker.calls.Load(); got != 1 {
		t.Errorf("expected docker starter to be invoked once, got %d", got)
	}
	if got := host.calls.Load(); got != 0 {
		t.Errorf("host starter must not be invoked when deployment_id is set, got %d calls", got)
	}
	if docker.lastCfg.Deployment != "d1" {
		t.Errorf("docker starter received cfg.Deployment=%q, want d1", docker.lastCfg.Deployment)
	}
	tunnel.CloseAll()
}

// TestShellTunnel_ContainerTargetRoutesToDockerBackend mirrors the
// deployment case but with container_id — operators sometimes pin a
// raw container ID via the dashboard's debug menu. Must also stay on
// the docker exec path.
func TestShellTunnel_ContainerTargetRoutesToDockerBackend(t *testing.T) {
	fc := &fakeChannel{}
	docker := &routingProbe{}
	host := &routingProbe{}
	tunnel := NewChannelShellTunnelForTest(fc, docker.starter(), nil).WithHostStarter(host.starter())
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellOpen,
		Body: ShellOpenBody{StreamID: "ctr-1", ContainerID: "c1"},
	})
	if got := docker.calls.Load(); got != 1 {
		t.Errorf("expected docker starter to be invoked once, got %d", got)
	}
	if got := host.calls.Load(); got != 0 {
		t.Errorf("host starter must not be invoked when container_id is set, got %d calls", got)
	}
	if docker.lastCfg.Container != "c1" {
		t.Errorf("docker starter received cfg.Container=%q, want c1", docker.lastCfg.Container)
	}
	tunnel.CloseAll()
}

// TestShellTunnel_BothTargetsSetStillUsesDockerBackend — defence in depth
// against a misbehaving CP that sends both. We prefer the docker path
// because container_id is more specific than the host fallback.
func TestShellTunnel_BothTargetsSetStillUsesDockerBackend(t *testing.T) {
	fc := &fakeChannel{}
	docker := &routingProbe{}
	host := &routingProbe{}
	tunnel := NewChannelShellTunnelForTest(fc, docker.starter(), nil).WithHostStarter(host.starter())
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellOpen,
		Body: ShellOpenBody{StreamID: "both-1", DeploymentID: "d1", ContainerID: "c1"},
	})
	if got := docker.calls.Load(); got != 1 {
		t.Errorf("expected docker starter to win when both targets are set, got %d calls", got)
	}
	if got := host.calls.Load(); got != 0 {
		t.Errorf("host starter must stay quiet when any target is set, got %d calls", got)
	}
	tunnel.CloseAll()
}

// TestShellTunnel_HostStarterFallsBackToDockerWhenNil — if no host
// starter is wired (e.g. the worker booted without a docker client and
// SetHostShellBackend was never called), the tunnel must not panic. The
// fallback is to invoke the docker starter, which will then return a
// clean ErrNoBackend if it's also unwired. The previous behaviour
// (docker-exec on "first container" via ResolveContainer) is preserved
// when only startShell is set.
func TestShellTunnel_HostStarterFallsBackToDockerWhenNil(t *testing.T) {
	fc := &fakeChannel{}
	docker := &routingProbe{}
	tunnel := NewChannelShellTunnelForTest(fc, docker.starter(), nil)
	// Note: do NOT call WithHostStarter — but the constructor defaults
	// the host starter to whatever was passed for shellStart in the
	// for-test constructor. To prove "nil host starter" we explicitly
	// nil it post-construction.
	tunnel.startHostShell = nil
	tunnel.Handle(context.Background(), Envelope{
		Type: MsgShellOpen,
		Body: ShellOpenBody{StreamID: "fallback"},
	})
	if got := docker.calls.Load(); got != 1 {
		t.Errorf("expected docker starter to be called as fallback, got %d", got)
	}
	tunnel.CloseAll()
}
