# inferia-worker — repo structure

This document is **kept in sync with the codebase as it grows**. When a new package, file, or top-level directory is added, update the corresponding row below in the same commit.

## Top-level

| Path | Purpose |
|---|---|
| `cmd/worker/` | Process entry point. Wires deps, starts servers. Not coverage-gated (wiring only). |
| `internal/config/` | Env-driven configuration with validation. |
| `internal/telemetry/` | CPU / memory / GPU telemetry collection. |
| `internal/auth/` | Token store + inference-token middleware. |
| `internal/runtime/` | Model lifecycle: pull → run → readiness → unload. |
| `internal/runtime/recipes/` | Launch recipes (vllm, ollama, …) ported from Nosana job_builder. |
| `internal/runtime/dockerclient/` | Thin wrapper over the Docker SDK. |
| `internal/runtime/dockerclient/fake/` | In-memory Client for runtime tests. |
| `internal/control/` | WebSocket client to control plane: register, heartbeat, dispatch. |
| `internal/dispatcher/` | Adapter between control.Dispatcher and runtime.Runtime + telemetry. |
| `internal/inference/` | Fiber proxy for `/v1/*` to local model containers. |
| `internal/healthz/` | `/healthz` and `/readyz` handlers. |
| `docs/` | Project docs. `structure.md` (this file) is the living architecture map. |
| `Dockerfile` | Multi-stage build for the worker image. |
| `docker-compose.yml` | Single-command bring-up on a GPU host. |
| `.env.sample` | All env vars with comments + safe defaults. |
| `Makefile` | `test`, `test-integration`, `build`, `lint`, `coverage`. |

## Package responsibilities (short form)

### `internal/config`
Parses env vars at startup. Validates required fields, URL schemes, length bounds (`NODE_NAME` ≤ 255, tokens ≤ 4096 bytes). Returns a typed `Config` struct or a descriptive error. No I/O beyond env reads.

### `internal/telemetry`
- `cpu.go` — reads `/proc/stat`, derives a CPU-percent number across all cores.
- `memory.go` — reads `/proc/meminfo`, returns total/used bytes.
- `gpu.go` — shells out to `nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits`; returns empty slice on any error (no-GPU hosts remain usable).

### `internal/auth`
- Token store: in-memory + persisted to `${TOKEN_FILE}` (default `/var/lib/inferia-worker/token`).
- Inference middleware: `Authorization: Bearer <INFERENCE_TOKEN>`, constant-time compare via `crypto/subtle`.

### `internal/runtime`
- `Launcher` interface: `LoadModel`, `UnloadModel`, `LoadedDeployments`, `StatusOf`.
- `dockerLauncher` implementation: idempotent by `deployment_id`, manages container lifecycle, runs readiness probes, watches docker events for crashes.
- State machine: `absent → pulling → starting → running → stopping → absent` (or `→ failed`).

### `internal/runtime/recipes`
- `Registry` maps recipe name → `Recipe` (image, env, cmd template, container port, ready probe path, default timeouts).
- Recipes: `vllm`, `ollama`, `vllm-omni`, `infinity`, `triton`, `inferia-diffusion`.
- URI scheme allowlist and config-key allowlist enforced here (mirrors the Python side helper).

### `internal/runtime/dockerclient`
Wraps `github.com/docker/docker/client`. Surfaces only what the launcher needs: `Pull`, `Run`, `Stop`, `Rm`, `Inspect`, `Events`. Unit tests verify argument construction; integration tests behind `-tags integration` exercise a real Docker daemon.

### `internal/control`
- `Bootstrap` — POST `/v1/workers/register` with `BOOTSTRAP_TOKEN`, persists returned worker JWT.
- `Channel` — WebSocket loop: dial, send `Heartbeat` every 5s, dispatch inbound `LoadModel` / `UnloadModel` to the runtime, reply with `CommandResult`. Reconnect with exponential backoff (1s → 30s cap, jittered). Command dedup by id over a 5-minute window.

### `internal/inference`
- Fiber router with token middleware in front of `/v1/*`.
- `Proxy` looks up the deployment endpoint from the runtime registry, streams the request body to the local model container, streams the response (including SSE `text/event-stream`) back to the caller.
- `503` with `Retry-After: 5` when deployment is not yet `running`.

### `internal/healthz`
- `/healthz` — liveness, returns 200 if the process is running.
- `/readyz` — readiness, returns 200 only after the worker has successfully connected to the control plane at least once.

### `cmd/worker`
`main.go` reads config, builds the auth/token store, builds the runtime (docker client + recipes), starts the Fiber server (inference + healthz routes), starts the control channel, blocks on signal. Wiring only — the testable adapter logic lives in `internal/dispatcher`.

### `internal/dispatcher`
Adapter between `control.Dispatcher` (what the WS layer needs) and the concrete `runtime.Runtime` + telemetry source. Translates `LoadModelBody → recipes.Plan → runtime.LoadModel`. Composes the periodic heartbeat body. 100% unit-test coverage with a fake runtime.

## Test layout

- Unit tests live next to the code they test (`foo.go` + `foo_test.go`).
- Integration tests are tagged: `//go:build integration` and live in `*_integration_test.go` files.
- Shared test helpers live in `internal/testutil/` (added when needed; not yet created).
- Fake dependencies (e.g. fake docker client) live in `internal/runtime/dockerclient/fake/`.

## Coverage policy

`make coverage` runs `./internal/...` with `-race -tags=integration -coverprofile=coverage.out -covermode=atomic` and fails if total internal coverage drops below **95%**. `cmd/worker` is excluded from the gate because it is the wiring layer (its `main()` cannot be unit-tested). The `-tags=integration` flag enables real-Docker tests in `internal/runtime/dockerclient`; those tests skip cleanly when no Docker daemon is reachable.

## When to update this file

Update `docs/structure.md` in the same commit whenever you:
- add a new package under `internal/` or a new top-level dir
- meaningfully change a package's responsibility (not just internal refactors)
- introduce a new build tag, env var category, or test helper convention

Don't update it for routine edits to existing files.
