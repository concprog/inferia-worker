// Package admin exposes operator-only WS endpoints on the worker's Fiber
// app: live model-container log tailing and an interactive exec shell.
//
// Both endpoints are mounted under the same auth middleware as the inference
// proxy (INFERENCE_TOKEN bearer). The orchestration HTTP server proxies
// dashboard WebSocket connections through to these endpoints so operators
// can see what their model containers are doing without SSH'ing the host.
//
// As of the channel-tunnel refactor, the heavy lifting (PTY pump,
// docker-logs demux) lives in internal/shellbridge so the same primitives
// also drive the control-channel multiplexer. These HTTP-WS endpoints
// remain mounted for direct connections and operator-issued WS dials.
package admin

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"

	"github.com/docker/docker/client"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/websocket/v2"
	"github.com/inferia/inferia-worker/internal/shellbridge"
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

// shellbridgeRuntime adapts admin.Runtime to shellbridge.Runtime — same
// surface, just a different package. Kept as a thin wrapper rather than
// re-exporting because admin's Runtime is the historical interface and
// shellbridge's is the new one; we don't want to bind one to the other.
type shellbridgeRuntime struct{ inner Runtime }

func (s shellbridgeRuntime) ContainerForDeployment(id string) string {
	return s.inner.ContainerForDeployment(id)
}
func (s shellbridgeRuntime) LoadedDeployments() []string { return s.inner.LoadedDeployments() }

// containerLookup retained for backwards compatibility with the existing
// test suite (admin_test.go exercises resolveContainerCore). The real
// resolver now lives in shellbridge.ResolveContainer; this surface only
// satisfies the older tests.
type containerLookup interface {
	ByDeploymentID(depID string) string
	MostRecent() string
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
//
// Functionally mirrors shellbridge.ResolveContainer; left in place for
// the existing test fixtures that exercise the precedence chain through
// a fake containerLookup.
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

func sendErr(c *websocket.Conn, detail string) {
	_ = c.WriteJSON(map[string]string{"type": "error", "message": detail})
	_ = c.Close()
}

// --- /v1/logs ---------------------------------------------------------------

func handleLogs(c *websocket.Conn, dockerCli *client.Client, rt Runtime) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// One mutex per WS — fasthttp/websocket cannot interleave writes.
	var wmu sync.Mutex

	// Stop the stream when the client disconnects.
	go func() {
		for {
			if _, _, err := c.ReadMessage(); err != nil {
				cancel()
				return
			}
		}
	}()

	// Build a per-call docker logs backend so we can keep resolveContainer
	// inline (admin's tests rely on the existing query-string precedence
	// chain). We bypass DefaultLogsSpawn deliberately so the admin path
	// stays independent of shellbridge package globals.
	cid, errMsg := shellbridge.ResolveContainer(dockerCli, shellbridgeRuntime{rt},
		c.Query("container"), c.Query("deployment"))
	if errMsg != nil {
		sendErr(c, errMsg.Error())
		return
	}
	emit := func(stream, line string) {
		wmu.Lock()
		defer wmu.Unlock()
		_ = c.WriteJSON(map[string]string{
			"type":   "log",
			"stream": stream,
			"data":   line,
		})
	}
	sess, err := shellbridge.StartLogsWithBackend(ctx,
		shellbridge.LogsSessionConfig{
			Container: cid,
			Tail:      200,
			OnLine:    emit,
			OnEnd: func(reason string) {
				wmu.Lock()
				defer wmu.Unlock()
				_ = c.WriteJSON(map[string]string{"type": "end", "reason": reason})
			},
		},
		func(cfg shellbridge.LogsSessionConfig) (shellbridge.LogsBackend, error) {
			return shellbridge.NewDockerLogsBackend(dockerCli, cid, cfg.Tail), nil
		},
	)
	if err != nil {
		sendErr(c, err.Error())
		return
	}
	defer sess.Close()
	<-ctx.Done()
}

// --- /v1/shell --------------------------------------------------------------

func handleShell(c *websocket.Conn, dockerCli *client.Client, rt Runtime) {
	rawContainer := c.Query("container")
	rawDeployment := c.Query("deployment")
	// hostMode mirrors the channel-tunnel routing: when the caller pinned
	// neither a container nor a deployment, fall through to a host-shell
	// session (privileged sidecar + nsenter) rather than trying to
	// resolve "first running container". The worker container itself is
	// distroless and would fail any subsequent /bin/sh exec.
	hostMode := rawContainer == "" && rawDeployment == ""

	var cid string
	if !hostMode {
		resolved, errMsg := shellbridge.ResolveContainer(dockerCli, shellbridgeRuntime{rt},
			rawContainer, rawDeployment)
		if errMsg != nil {
			sendErr(c, errMsg.Error())
			return
		}
		cid = resolved
	}
	shellPath := c.Query("shell")
	if shellPath == "" {
		// Try /bin/bash first (most images), fall back to /bin/sh
		// at-runtime if exec fails on bash.
		shellPath = "/bin/bash"
	}
	execUser := c.Query("user")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wmu sync.Mutex
	emit := func(payload map[string]any) error {
		wmu.Lock()
		defer wmu.Unlock()
		return c.WriteJSON(payload)
	}

	// Per-call factory so the admin path stays independent of shellbridge
	// package globals (and so the existing tests don't have to mutate
	// DefaultSpawn / DefaultHostSpawn). Same docker client either way —
	// the host backend wraps a sidecar around the same daemon connection.
	backendFactory := func(cfg shellbridge.ShellSessionConfig) (shellbridge.ShellBackend, error) {
		if hostMode {
			return shellbridge.NewHostShellBackend(dockerCli), nil
		}
		return shellbridge.NewDockerShellBackend(dockerCli, cid), nil
	}

	sess, err := shellbridge.StartShellWithBackend(ctx,
		shellbridge.ShellSessionConfig{
			Shell:      shellPath,
			User:       execUser,
			Container:  cid,
			Deployment: "", // already resolved into Container (docker mode) or empty (host mode)
			OnOutput: func(data []byte) {
				_ = emit(map[string]any{"type": "output", "data": string(data)})
			},
			OnExit: func(code int, reason string) {
				_ = emit(map[string]any{"type": "exit", "code": code, "reason": reason})
				cancel()
			},
			OnError: func(message string) {
				_ = emit(map[string]any{"type": "exit", "reason": message})
				cancel()
			},
		},
		backendFactory,
	)
	if err != nil {
		sendErr(c, err.Error())
		return
	}
	defer sess.Close()

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
					_ = sess.WriteInput([]byte(data))
				}
				continue
			case "resize":
				rows, _ := toInt(env["rows"])
				cols, _ := toInt(env["cols"])
				if rows > 0 && cols > 0 {
					_ = sess.Resize(uint16(cols), uint16(rows))
				}
				continue
			}
		}
		_ = sess.WriteInput(msg)
	}
	target := cid
	if hostMode {
		target = "host"
	}
	log.Printf("shell session for %s ended", target)
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
