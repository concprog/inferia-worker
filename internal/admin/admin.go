// Package admin exposes operator-only WS endpoints on the worker's Fiber
// app: live model-container log tailing and an interactive exec shell.
//
// Both endpoints are mounted under the same auth middleware as the inference
// proxy (INFERENCE_TOKEN bearer). The orchestration HTTP server proxies
// dashboard WebSocket connections through to these endpoints so operators
// can see what their model containers are doing without SSH'ing the host.
package admin

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/stdcopy"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/websocket/v2"
)

// Runtime is the subset of runtime.Runtime the admin handlers need.
type Runtime interface {
	// ContainerForDeployment returns the container ID currently running
	// the given deployment, or "" if none.
	ContainerForDeployment(deploymentID string) string

	// LoadedDeployments returns the set of deployment IDs whose containers
	// the worker considers running. Used as a fallback when the request
	// omits a target.
	LoadedDeployments() []string
}

// Register wires the admin endpoints onto the given Fiber app.
//
// dockerCli is the host docker client (already opened by the worker).
// rt is the runtime registry that maps deploymentID → container.
func Register(app *fiber.App, dockerCli *client.Client, rt Runtime) {
	app.Use("/v1/logs", websocketUpgrade)
	app.Use("/v1/shell", websocketUpgrade)

	app.Get("/v1/logs", websocket.New(func(c *websocket.Conn) {
		handleLogs(c, dockerCli, rt)
	}))
	app.Get("/v1/shell", websocket.New(func(c *websocket.Conn) {
		handleShell(c, dockerCli, rt)
	}))
}

func websocketUpgrade(c *fiber.Ctx) error {
	if websocket.IsWebSocketUpgrade(c) {
		c.Locals("allowed", true)
		return c.Next()
	}
	return fiber.ErrUpgradeRequired
}

// containerLookup is the narrow surface resolveContainerCore needs to
// query docker. Real callers use a docker-backed implementation; tests
// supply a fake. Both methods return "" on miss rather than an error so
// the core logic can keep the precedence chain in one place.
type containerLookup interface {
	ByDeploymentID(depID string) string
	MostRecent() string
}

// dockerLookup wires containerLookup to the real Docker SDK client.
type dockerLookup struct{ cli *client.Client }

func (d dockerLookup) ByDeploymentID(depID string) string {
	if d.cli == nil || depID == "" {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	list, err := d.cli.ContainerList(ctx, container.ListOptions{
		All:     false,
		Filters: filters.NewArgs(filters.KeyValuePair{Key: "name", Value: depID}),
	})
	if err != nil || len(list) == 0 {
		return ""
	}
	for _, c := range list {
		for _, n := range c.Names {
			if strings.Contains(n, depID) {
				return c.ID
			}
		}
	}
	return list[0].ID
}

func (d dockerLookup) MostRecent() string {
	if d.cli == nil {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	list, err := d.cli.ContainerList(ctx, container.ListOptions{
		Filters: filters.NewArgs(filters.KeyValuePair{Key: "name", Value: "inferia-"}),
	})
	if err != nil || len(list) == 0 {
		return ""
	}
	return list[0].ID
}

// resolveContainerCore picks the container to operate on. Preference:
//  1. query param `container` (raw container ID — operator override)
//  2. query param `deployment` resolved through the runtime
//  3. query param `deployment` resolved via docker ps (recovers after a
//     worker restart that drops the runtime's in-memory deployment map
//     while the model container is still up)
//  4. the first loaded deployment when the request omits both
//
// Returns ("", error string) if nothing can be resolved.
func resolveContainerCore(getQuery func(string) string, rt Runtime, look containerLookup) (string, string) {
	if raw := getQuery("container"); raw != "" {
		return raw, ""
	}
	depID := getQuery("deployment")
	if depID == "" {
		loaded := rt.LoadedDeployments()
		if len(loaded) == 0 {
			// Last-ditch fallback: pick the most recent inferia-managed
			// container the worker has spawned (helps when the runtime
			// registry is empty post-restart and no deployment id was
			// provided).
			if cid := look.MostRecent(); cid != "" {
				return cid, ""
			}
			return "", "no active deployment on this worker"
		}
		depID = loaded[0]
	}
	if cid := rt.ContainerForDeployment(depID); cid != "" {
		return cid, ""
	}
	if cid := look.ByDeploymentID(depID); cid != "" {
		return cid, ""
	}
	return "", fmt.Sprintf("deployment %q has no running container", depID)
}

// resolveContainer is the WS-facing wrapper around resolveContainerCore.
// Real handlers bind the docker client; tests exercise the core directly.
func resolveContainer(c *websocket.Conn, rt Runtime, docker *client.Client) (string, string) {
	getQuery := func(key string) string { return c.Query(key) }
	return resolveContainerCore(getQuery, rt, dockerLookup{cli: docker})
}

func sendErr(c *websocket.Conn, detail string) {
	_ = c.WriteJSON(map[string]string{"type": "error", "message": detail})
	_ = c.Close()
}

// --- /v1/logs ---------------------------------------------------------------

func handleLogs(c *websocket.Conn, dockerCli *client.Client, rt Runtime) {
	containerID, errMsg := resolveContainer(c, rt, dockerCli)
	if errMsg != "" {
		sendErr(c, errMsg)
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Stop the stream when the client disconnects.
	go func() {
		for {
			if _, _, err := c.ReadMessage(); err != nil {
				cancel()
				return
			}
		}
	}()

	rc, err := dockerCli.ContainerLogs(ctx, containerID, container.LogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Follow:     true,
		Tail:       "200",
		Timestamps: false,
	})
	if err != nil {
		sendErr(c, fmt.Sprintf("docker logs: %v", err))
		return
	}
	defer rc.Close()

	// docker multiplexes stdout/stderr in a framed protocol; demux into
	// two pipes and forward each line over the WS as a "log" JSON frame.
	outR, outW := io.Pipe()
	errR, errW := io.Pipe()
	defer outR.Close()
	defer errR.Close()

	demuxDone := make(chan struct{})
	go func() {
		defer close(demuxDone)
		_, _ = stdcopy.StdCopy(outW, errW, rc)
		_ = outW.Close()
		_ = errW.Close()
	}()

	// One mutex per WS — fasthttp/websocket cannot interleave writes.
	var wmu sync.Mutex
	emit := func(stream, line string) {
		wmu.Lock()
		defer wmu.Unlock()
		_ = c.WriteJSON(map[string]string{
			"type":   "log",
			"stream": stream,
			"data":   line,
		})
	}

	go forwardLines(outR, "stdout", emit)
	go forwardLines(errR, "stderr", emit)

	<-demuxDone
	wmu.Lock()
	_ = c.WriteJSON(map[string]string{"type": "end"})
	wmu.Unlock()
}

// forwardLines reads from r line-by-line and emits each via fn. EOF/errors
// silently end the loop — the caller (handleLogs) wraps everything in one
// cancellable context so the goroutine winds up cleanly.
func forwardLines(r io.Reader, stream string, fn func(stream, line string)) {
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

// --- /v1/shell --------------------------------------------------------------

func handleShell(c *websocket.Conn, dockerCli *client.Client, rt Runtime) {
	containerID, errMsg := resolveContainer(c, rt, dockerCli)
	if errMsg != "" {
		sendErr(c, errMsg)
		return
	}

	shellPath := c.Query("shell")
	if shellPath == "" {
		// Try /bin/bash first (most images), fall back to /bin/sh
		// at-runtime if exec fails on bash.
		shellPath = "/bin/bash"
	}
	// `user` is forwarded to docker exec --user verbatim. Accepts "name",
	// "uid", "name:group", or "uid:gid". Empty means "use container default".
	execUser := c.Query("user")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	execCfg := container.ExecOptions{
		AttachStdin:  true,
		AttachStdout: true,
		AttachStderr: true,
		Tty:          true,
		Cmd:          []string{shellPath},
		User:         execUser,
	}
	created, err := dockerCli.ContainerExecCreate(ctx, containerID, execCfg)
	if err != nil {
		// Fallback shell: caller asked for something the image doesn't
		// ship (e.g. /bin/bash on a distroless container). Retry once with
		// /bin/sh so the user gets *some* shell rather than a hard error.
		if shellPath != "/bin/sh" {
			execCfg.Cmd = []string{"/bin/sh"}
			created, err = dockerCli.ContainerExecCreate(ctx, containerID, execCfg)
		}
		if err != nil {
			sendErr(c, fmt.Sprintf("exec create: %v", err))
			return
		}
	}

	hijack, err := dockerCli.ContainerExecAttach(ctx, created.ID, container.ExecStartOptions{
		Tty: true,
	})
	if err != nil {
		sendErr(c, fmt.Sprintf("exec attach: %v", err))
		return
	}
	defer hijack.Close()

	var wmu sync.Mutex
	emit := func(payload map[string]any) error {
		wmu.Lock()
		defer wmu.Unlock()
		return c.WriteJSON(payload)
	}

	// Container → WS pump.
	go func() {
		buf := make([]byte, 4096)
		for {
			n, rerr := hijack.Reader.Read(buf)
			if n > 0 {
				if werr := emit(map[string]any{
					"type": "output",
					"data": string(buf[:n]),
				}); werr != nil {
					cancel()
					return
				}
			}
			if rerr != nil {
				_ = emit(map[string]any{"type": "exit", "reason": rerr.Error()})
				cancel()
				return
			}
		}
	}()

	// WS → container pump.
	for {
		_, msg, rerr := c.ReadMessage()
		if rerr != nil {
			break
		}
		// Two accepted shapes:
		//   raw bytes: forwarded verbatim
		//   JSON {"type":"stdin","data":"..."} / {"type":"resize","rows":..,"cols":..}
		var env map[string]any
		if json.Unmarshal(msg, &env) == nil {
			t, _ := env["type"].(string)
			switch t {
			case "stdin":
				data, _ := env["data"].(string)
				if data != "" {
					_, _ = hijack.Conn.Write([]byte(data))
				}
				continue
			case "resize":
				rows, _ := toInt(env["rows"])
				cols, _ := toInt(env["cols"])
				if rows > 0 && cols > 0 {
					_ = dockerCli.ContainerExecResize(ctx, created.ID, container.ResizeOptions{
						Height: uint(rows),
						Width:  uint(cols),
					})
				}
				continue
			}
		}
		_, _ = hijack.Conn.Write(msg)
	}

	// Best-effort: wait for exec to finish so we can report exit code.
	deadline, cdone := context.WithTimeout(context.Background(), 2*time.Second)
	defer cdone()
	if insp, ierr := dockerCli.ContainerExecInspect(deadline, created.ID); ierr == nil {
		_ = emit(map[string]any{"type": "exit", "code": insp.ExitCode})
	}
	log.Printf("shell session for %s ended", containerID)
}

func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case int64:
		return int(n), true
	}
	return 0, false
}
