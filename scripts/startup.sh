#!/usr/bin/env bash
set -euo pipefail

uv venv
uv pip install -e ./envs/agent_world_model_env