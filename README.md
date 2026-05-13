# inferia-worker

GPU-node agent for [InferiaLLM](https://github.com/inferia/inferiaLLM). Runs on each GPU host the operator controls directly (bare metal, self-hosted server, or a cloud VM). Connects back to the InferiaLLM control plane, accepts model-load commands, and serves OpenAI-compatible inference requests off its local GPU(s).

## Quick start

On a host with Docker + NVIDIA Container Toolkit installed:

```bash
cp .env.sample .env
# Edit .env: paste BOOTSTRAP_TOKEN + POOL_ID from InferiaLLM admin UI,
#            set CONTROL_PLANE_URL, WORKER_ADVERTISE_URL, NODE_NAME.
docker compose up -d
```

The worker registers with the control plane, exchanges the bootstrap token for a long-lived JWT, and starts heartbeating. It then waits for `LoadModel` commands. Inference traffic is served at `${WORKER_ADVERTISE_URL}`.

## Architecture

See [`docs/structure.md`](docs/structure.md) for the package layout and [`InferiaLLM/docs/specs/2026-05-13-inferia-worker-design.md`](https://github.com/inferia/inferiaLLM) for the full design spec.

## Development

```bash
make test           # unit tests with race detector + coverage
make test-integration  # integration tests (requires Docker)
make build          # build worker binary
make lint           # gofmt + go vet
```

## Configuration

All configuration is via environment variables. See `.env.sample` for the full list with defaults.

## License

Apache-2.0
