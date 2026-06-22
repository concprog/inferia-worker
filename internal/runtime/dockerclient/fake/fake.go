// Package fake provides an in-memory Client implementation used by the runtime
// package's unit tests. It is also useful for examples and ad-hoc dev runs
// where Docker is not available.
package fake

import (
	"context"
	"errors"
	"sync"

	"github.com/inferia/inferia-worker/internal/runtime/dockerclient"
)

// Client is an in-memory Client that records calls and lets tests inject errors.
type Client struct {
	mu sync.Mutex

	Pinged          int
	NetworksCreated []string
	Pulled          []string
	// PullProgressLines, when set, are replayed to Pull's onProgress callback.
	PullProgressLines []string
	ImagesRemoved   []string
	Created         []*dockerclient.ContainerSpec
	Started         []string
	Stopped         []string
	Removed         []string

	// containers maps containerID → inspect state, mutated by Start/Stop/Remove
	// and read by Inspect.
	containers map[string]*dockerclient.Inspect

	// next ID counter for synthesised containerIDs.
	nextID int

	// Error injectors: any non-nil makes the corresponding method return it.
	PingErr          error
	EnsureNetworkErr error
	PullErr          error
	CreateErr        error
	StartErr         error
	StopErr          error
	RemoveErr        error
	RemoveImageErr   error
	InspectErr       error
	LogsErr          error

	// FakeLogs returned by Logs (regardless of containerID).
	FakeLogs []byte
}

// New returns an empty fake.
func New() *Client { return &Client{containers: map[string]*dockerclient.Inspect{}} }

func (c *Client) Ping(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.Pinged++
	return c.PingErr
}

func (c *Client) EnsureNetwork(ctx context.Context, name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.EnsureNetworkErr != nil {
		return c.EnsureNetworkErr
	}
	c.NetworksCreated = append(c.NetworksCreated, name)
	return nil
}

func (c *Client) Pull(ctx context.Context, image string, onProgress func(line string)) error {
	c.mu.Lock()
	lines := append([]string(nil), c.PullProgressLines...)
	if c.PullErr != nil {
		c.mu.Unlock()
		return c.PullErr
	}
	c.Pulled = append(c.Pulled, image)
	c.mu.Unlock()
	// Replay any scripted progress lines so callers/tests can exercise the
	// progress callback path.
	if onProgress != nil {
		for _, l := range lines {
			onProgress(l)
		}
	}
	return nil
}

func (c *Client) Create(ctx context.Context, spec *dockerclient.ContainerSpec) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.CreateErr != nil {
		return "", c.CreateErr
	}
	c.nextID++
	id := "fake-id-" + spec.Name
	c.Created = append(c.Created, spec)
	c.containers[id] = &dockerclient.Inspect{ID: id, Running: false, Status: "created"}
	return id, nil
}

func (c *Client) Start(ctx context.Context, id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.StartErr != nil {
		return c.StartErr
	}
	c.Started = append(c.Started, id)
	if state, ok := c.containers[id]; ok {
		state.Running = true
		state.Status = "running"
	}
	return nil
}

func (c *Client) Stop(ctx context.Context, id string, timeoutSeconds int) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.StopErr != nil {
		return c.StopErr
	}
	c.Stopped = append(c.Stopped, id)
	if state, ok := c.containers[id]; ok {
		state.Running = false
		state.Status = "exited"
	}
	return nil
}

func (c *Client) Remove(ctx context.Context, id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.RemoveErr != nil {
		return c.RemoveErr
	}
	c.Removed = append(c.Removed, id)
	delete(c.containers, id)
	return nil
}

func (c *Client) RemoveImage(ctx context.Context, image string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.RemoveImageErr != nil {
		return c.RemoveImageErr
	}
	c.ImagesRemoved = append(c.ImagesRemoved, image)
	return nil
}

func (c *Client) Inspect(ctx context.Context, id string) (*dockerclient.Inspect, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.InspectErr != nil {
		return nil, c.InspectErr
	}
	s, ok := c.containers[id]
	if !ok {
		return nil, errors.New("not found")
	}
	cp := *s
	return &cp, nil
}

func (c *Client) Logs(ctx context.Context, id string, lines int) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.LogsErr != nil {
		return nil, c.LogsErr
	}
	return c.FakeLogs, nil
}

// MarkExited transitions a container to exited state with the given code.
// Used by tests to simulate a crash without a real docker events stream.
func (c *Client) MarkExited(id string, code int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if s, ok := c.containers[id]; ok {
		s.Running = false
		s.Status = "exited"
		s.ExitCode = code
	}
}
