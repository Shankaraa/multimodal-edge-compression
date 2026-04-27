#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

model_path="${1:-models/voxtral-realtime}"
config_path="${2:-configs/vllm/bf16.yaml}"
port="${3:-8080}"
venv_path="${VOXTRAL_VENV:-$HOME/.venvs/voxtral-baseline}"

# Run from the project root so relative paths in configs and scripts behave consistently.
source "${venv_path}/bin/activate"
cd "${project_root}"
exec python scripts/serve_model.py "${model_path}" --config "${config_path}" --port "${port}"
