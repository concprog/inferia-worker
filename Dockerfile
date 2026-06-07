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
ARG TARGETOS=linux
ARG TARGETARCH=amd64
ENV CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH}
RUN go build -trimpath -ldflags="-s -w" -o /out/worker ./cmd/worker

# ----- runtime -----
FROM gcr.io/distroless/static-debian12:latest

# The worker needs:
#   - read access to /var/run/docker.sock (mounted at runtime; the socket on
#     the host is typically owned by root:docker, gid varies by distro)
#   - write access to /var/lib/inferia-worker for the persisted token
#
# We run as root so the worker can read the docker.sock without operators
# having to figure out the host docker-group GID. This is acceptable because
# the worker container's sole purpose is to manage other Docker containers —
# anything that compromises the worker already has full docker socket access.
# Use Docker's userns-remap or rootless Docker to get host-side isolation.

WORKDIR /app
COPY --from=builder /out/worker /app/worker

# Token persistence volume mount target.
VOLUME ["/var/lib/inferia-worker"]

EXPOSE 8080

ENTRYPOINT ["/app/worker"]
