#!/usr/bin/env bash

# Run the Agent World Model (AWM) environment server.
# Serves envs/agent_world_model_env over HTTP/WebSocket via uvicorn.
#
# Overridable via environment:
#   HOST                (default 0.0.0.0)
#   PORT                (default 8899)
#   WS_PING_INTERVAL    (default 1800)
#   WS_PING_TIMEOUT     (default 1800)
# Extra args are passed through to uvicorn, e.g.:
#   scripts/run_awm_server.sh --reload

set -euo pipefail

# Run from the repo root so PYTHONPATH=src:envs resolves correctly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8899}"
WS_PING_INTERVAL="${WS_PING_INTERVAL:-1800}"
WS_PING_TIMEOUT="${WS_PING_TIMEOUT:-1800}"

# Raise the open-file limit so the server can hold many concurrent WebSocket /
# HTTP connections from parallel rollouts.
ulimit -n 65536

PYTHONPATH=src:envs uv run uvicorn \
    envs.agent_world_model_env.server.app:app \
    --host "$HOST" \
    --port "$PORT" \
    --ws-ping-interval "$WS_PING_INTERVAL" \
    --ws-ping-timeout "$WS_PING_TIMEOUT" \
    "$@"
