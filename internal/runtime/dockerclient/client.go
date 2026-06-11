package dockerclient

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/client"
	"github.com/docker/go-connections/nat"
)

// Client is the narrow surface the runtime needs from docker. Implemented by
// dockerEngine (real SDK) and by the fake in fake/fake.go for tests.
type Client interface {
	Ping(ctx context.Context) error
	EnsureNetwork(ctx context.Context, name string) error
	// Pull fetches the image. onProgress, when non-nil, receives throttled
	// human-readable progress lines (e.g. "Pulling image… 42% (14.7 GB / 35.0 GB)")
	// so callers can surface pull progress as logs. Pass nil to ignore progress.
	Pull(ctx context.Context, image string, onProgress func(line string)) error
	Create(ctx context.Context, spec *ContainerSpec) (containerID string, err error)
	Start(ctx context.Context, containerID string) error
	Stop(ctx context.Context, containerID string, timeoutSeconds int) error
	Remove(ctx context.Context, containerID string) error
	Inspect(ctx context.Context, containerID string) (*Inspect, error)
	Logs(ctx context.Context, containerID string, lines int) ([]byte, error)
}

// Inspect is the subset of container state we read.
type Inspect struct {
	ID       string
	Running  bool
	ExitCode int
	Status   string // running | exited | created | dead
}

// NewEngine returns a Client backed by the real Docker SDK.
func NewEngine(dockerHost string) (Client, error) {
	c, err := client.NewClientWithOpts(
		client.WithHost(dockerHost),
		client.WithAPIVersionNegotiation(),
	)
	if err != nil {
		return nil, err
	}
	return &dockerEngine{cli: c}, nil
}

// Raw returns the underlying Docker SDK client. Exposed for auxiliary
// endpoints (admin/logs, admin/shell) that need APIs beyond the narrow
// Client interface — specifically streaming ContainerLogs and exec
// hijack, which would clutter the test fake otherwise.
func (e *dockerEngine) Raw() *client.Client { return e.cli }

// RawAccessor is implemented by clients that wrap the real Docker SDK.
type RawAccessor interface {
	Raw() *client.Client
}

type dockerEngine struct {
	cli *client.Client
}

func (e *dockerEngine) Ping(ctx context.Context) error {
	_, err := e.cli.Ping(ctx)
	return err
}

func (e *dockerEngine) EnsureNetwork(ctx context.Context, name string) error {
	nets, err := e.cli.NetworkList(ctx, network.ListOptions{})
	if err != nil {
		return err
	}
	found := false
	for _, n := range nets {
		if n.Name == name {
			found = true
			break
		}
	}
	if !found {
		if _, err = e.cli.NetworkCreate(ctx, name, network.CreateOptions{
			Driver: "bridge",
			Labels: map[string]string{"inferia.managed_by": "inferia-worker"},
		}); err != nil {
			return err
		}
	}
	// Self-attach to the network so the worker can resolve sibling model
	// containers by name during readiness probes. The hostname of a Docker
	// container is the first 12 chars of its ID by default; ContainerInspect
	// accepts that as the ID.
	host, herr := os.Hostname()
	if herr != nil || host == "" {
		return nil // best-effort
	}
	info, ierr := e.cli.ContainerInspect(ctx, host)
	if ierr != nil {
		return nil // not running as a container, or short-id mismatch
	}
	if _, attached := info.NetworkSettings.Networks[name]; attached {
		return nil
	}
	if cerr := e.cli.NetworkConnect(ctx, name, info.ID, nil); cerr != nil {
		// Don't fail boot — operator can always attach manually if this
		// fails on their host.
		return nil
	}
	return nil
}

func (e *dockerEngine) Pull(ctx context.Context, img string, onProgress func(line string)) error {
	rc, err := e.cli.ImagePull(ctx, img, image.PullOptions{})
	if err != nil {
		return err
	}
	defer rc.Close()
	// The progress stream MUST be fully drained for the pull to complete. When
	// nobody wants progress, drain straight to /dev/null (cheapest path).
	if onProgress == nil {
		_, err = io.Copy(io.Discard, rc)
		return err
	}
	// Otherwise decode the per-line progress JSON, fold it into an overall
	// percentage, and emit throttled summary lines.
	tracker := newPullProgressTracker()
	dec := json.NewDecoder(rc)
	for {
		var m pullMessage
		if derr := dec.Decode(&m); derr != nil {
			if derr == io.EOF {
				return nil
			}
			// Malformed/partial frame — keep draining so the pull still
			// finishes; just stop reporting progress.
			_, _ = io.Copy(io.Discard, rc)
			return nil
		}
		if m.Error != "" {
			return fmt.Errorf("pull: %s", m.Error)
		}
		if line, ok := tracker.update(m); ok {
			onProgress(line)
		}
	}
}

func (e *dockerEngine) Create(ctx context.Context, spec *ContainerSpec) (string, error) {
	containerPort, err := nat.NewPort("tcp", trimAfter(spec.PortBinding.ContainerPort, '/'))
	if err != nil {
		return "", fmt.Errorf("port parse: %w", err)
	}

	exposed := nat.PortSet{containerPort: struct{}{}}
	portMap := nat.PortMap{
		containerPort: []nat.PortBinding{
			{HostIP: spec.PortBinding.HostIP, HostPort: spec.PortBinding.HostPort},
		},
	}

	envSlice := make([]string, 0, len(spec.Env))
	for k, v := range spec.Env {
		envSlice = append(envSlice, k+"="+v)
	}

	cfg := &container.Config{
		Image:       spec.Image,
		Cmd:         spec.Cmd,
		Entrypoint:  spec.Entrypoint,
		Env:         envSlice,
		ExposedPorts: exposed,
		Labels:      spec.Labels,
	}

	hostCfg := &container.HostConfig{
		PortBindings:  portMap,
		RestartPolicy: container.RestartPolicy{Name: container.RestartPolicyMode(spec.RestartPolicy)},
		ShmSize:       spec.ShmSize,
	}
	if len(spec.Mounts) > 0 {
		hostCfg.Mounts = make([]mount.Mount, len(spec.Mounts))
		for i, m := range spec.Mounts {
			hostCfg.Mounts[i] = mount.Mount{
				Type:     mount.Type(m.Type),
				Source:   m.Source,
				Target:   m.Target,
				ReadOnly: m.ReadOnly,
			}
		}
	}
	if len(spec.GPUDeviceIDs) > 0 {
		var req container.DeviceRequest
		if len(spec.GPUDeviceIDs) == 1 && spec.GPUDeviceIDs[0] == "all" {
			req = container.DeviceRequest{
				Driver:       "nvidia",
				Count:        -1,
				Capabilities: spec.GPUCapabilities,
			}
		} else {
			req = container.DeviceRequest{
				Driver:       "nvidia",
				DeviceIDs:    spec.GPUDeviceIDs,
				Capabilities: spec.GPUCapabilities,
			}
		}
		hostCfg.Resources.DeviceRequests = []container.DeviceRequest{req}
	}

	netCfg := &network.NetworkingConfig{
		EndpointsConfig: map[string]*network.EndpointSettings{
			spec.NetworkName: {NetworkID: spec.NetworkName},
		},
	}

	resp, err := e.cli.ContainerCreate(ctx, cfg, hostCfg, netCfg, nil, spec.Name)
	if err != nil {
		return "", err
	}
	return resp.ID, nil
}

func (e *dockerEngine) Start(ctx context.Context, id string) error {
	return e.cli.ContainerStart(ctx, id, container.StartOptions{})
}

func (e *dockerEngine) Stop(ctx context.Context, id string, timeoutSeconds int) error {
	t := timeoutSeconds
	return e.cli.ContainerStop(ctx, id, container.StopOptions{Timeout: &t})
}

func (e *dockerEngine) Remove(ctx context.Context, id string) error {
	return e.cli.ContainerRemove(ctx, id, container.RemoveOptions{Force: true})
}

func (e *dockerEngine) Inspect(ctx context.Context, id string) (*Inspect, error) {
	info, err := e.cli.ContainerInspect(ctx, id)
	if err != nil {
		return nil, err
	}
	state := info.State
	return &Inspect{
		ID:       info.ID,
		Running:  state.Running,
		ExitCode: state.ExitCode,
		Status:   state.Status,
	}, nil
}

func (e *dockerEngine) Logs(ctx context.Context, id string, lines int) ([]byte, error) {
	rc, err := e.cli.ContainerLogs(ctx, id, container.LogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Tail:       fmt.Sprintf("%d", lines),
	})
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	return io.ReadAll(rc)
}

func trimAfter(s string, sep byte) string {
	for i := 0; i < len(s); i++ {
		if s[i] == sep {
			return s[:i]
		}
	}
	return s
}

// Ensure encoding/json stays imported (used for debug helpers; keep dep stable
// across edits).
var _ = json.Marshal
