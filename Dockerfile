# syntax=docker/dockerfile:1.6

# ----- builder -----
FROM golang:1.26-alpine AS builder

WORKDIR /src

# Cache go mod deps.
COPY go.mod go.sum ./
RUN go mod download

# Source.
COPY . .

# Build a static binary.
ENV CGO_ENABLED=0 GOOS=linux GOARCH=amd64
RUN go build -trimpath -ldflags="-s -w" -o /out/worker ./cmd/worker

# ----- runtime -----
FROM gcr.io/distroless/static-debian12:nonroot

# The worker needs:
#   - read access to /var/run/docker.sock (mounted at runtime)
#   - write access to /var/lib/inferia-worker for the persisted token
#
# Distroless static is fine because the worker is statically linked. We do not
# include nvidia-smi here; telemetry/gpu.go tolerates its absence and reports
# zero GPUs in that case. For GPU-equipped hosts, host-mounted nvidia-smi is
# usually accessible via the docker socket integration, not from inside the
# worker container.

WORKDIR /app
COPY --from=builder /out/worker /app/worker

# Token persistence volume mount target.
VOLUME ["/var/lib/inferia-worker"]

USER nonroot:nonroot

EXPOSE 8080

ENTRYPOINT ["/app/worker"]
