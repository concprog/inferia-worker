# mooncake-master — metadata coordination service for shared KV cache.
# No GPU / CUDA needed; mooncake-transfer-engine-non-cuda is sufficient.
# Runs the gRPC master + embedded HTTP metadata server on the inferia-models bridge.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal runtime deps: python3.12 is the Ubuntu 24.04 default.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# mooncake-transfer-engine-non-cuda ships the mooncake_master binary and
# the Python store bindings; no CUDA or PyTorch dependency required here.
RUN pip3 install --no-cache-dir --break-system-packages \
        mooncake-transfer-engine-non-cuda

# Confirm the binary landed on PATH.
RUN which mooncake_master

# /config is the shared named-volume mount point.
# entrypoint writes mooncake_config.json there before starting the master,
# so vLLM containers that mount the same volume get the config on first use.
RUN mkdir -p /config

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# gRPC master port
EXPOSE 50063
# Embedded HTTP metadata server (used by mooncake_config.json metadata_server field)
EXPOSE 8092
# Prometheus metrics
EXPOSE 9003

ENTRYPOINT ["/entrypoint.sh"]
