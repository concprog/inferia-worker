#!/bin/sh
# vLLM model containers mount mooncake-config:/etc/mooncake:ro and point
# MOONCAKE_CONFIG_PATH at /etc/mooncake/mooncake_config.json.
#
# DNS name "mooncake-master" resolves inside the inferia-models bridge network.

set -e

MOONCAKE_RPC_PORT="${MOONCAKE_RPC_PORT:-50063}"
MOONCAKE_HTTP_PORT="${MOONCAKE_HTTP_PORT:-8092}"
MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-9003}"
# local_buffer_size per vLLM instance in bytes (default 2 GiB).
# Override via compose environment if you have many instances or long contexts.
LOCAL_BUFFER_BYTES="${LOCAL_BUFFER_BYTES:-2147483648}"

CONFIG_FILE="/config/mooncake_config.json"

echo "[mooncake-master] writing ${CONFIG_FILE}"
cat > "${CONFIG_FILE}" <<EOF
{
  "metadata_server": "http://mooncake-master:${MOONCAKE_HTTP_PORT}/metadata",
  "master_server_address": "mooncake-master:${MOONCAKE_RPC_PORT}",
  "global_segment_size": "0",
  "local_buffer_size": "${LOCAL_BUFFER_BYTES}",
  "protocol": "tcp",
  "device_name": ""
}
EOF

echo "[mooncake-master] config written:"
cat "${CONFIG_FILE}"

echo "[mooncake-master] starting master (rpc=:${MOONCAKE_RPC_PORT} http=:${MOONCAKE_HTTP_PORT} metrics=:${MOONCAKE_METRICS_PORT})"
exec mooncake_master \
    --rpc_port="${MOONCAKE_RPC_PORT}" \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port="${MOONCAKE_HTTP_PORT}" \
    --metrics_port="${MOONCAKE_METRICS_PORT}"
