#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-python3}"
venv_dir="${VENV_DIR:-${repo_root}/.venv}"
model_id="${MODEL_ID:-mistralai/Voxtral-Mini-4B-Realtime-2602}"
model_revision="${MODEL_REVISION:-2769294da9567371363522aac9bbcfdd19447add}"
model_dir="${MODEL_DIR:-models/voxtral-realtime}"
config_path="${CONFIG_PATH:-vllm_config.yaml}"
port="${PORT:-8115}"
label="${LABEL:-submission_fp8_runtime}"
lang="${LANG:-en_us}"
limit="${LIMIT:-20}"
dataset_source="${DATASET_SOURCE:-google_fleurs}"
skip_install="${SKIP_INSTALL:-0}"
install_vllm="${INSTALL_VLLM:-1}"
vllm_torch_backend="${VLLM_TORCH_BACKEND:-cu130}"
vllm_extra_index_url="${VLLM_EXTRA_INDEX_URL:-https://wheels.vllm.ai/nightly/cu130}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 1
fi

if [ ! -d "${venv_dir}" ]; then
  "${python_bin}" -m venv "${venv_dir}"
fi

source "${venv_dir}/bin/activate"

if [ "${skip_install}" != "1" ]; then
  python -m pip install -U pip
  python -m pip install -r requirements.txt

  if [ "${install_vllm}" != "0" ]; then
    python -m pip install -U uv
    uv pip install -U vllm --torch-backend="${vllm_torch_backend}" --extra-index-url "${vllm_extra_index_url}"
  fi
fi

python scripts/verify_claimed_reports.py --reports-dir reports --claims reports/claimed_results.json

download_args=(--repo-id "${model_id}" --local-dir "${model_dir}")
if [ -n "${model_revision}" ]; then
  download_args+=(--revision "${model_revision}")
fi
python scripts/download_model.py "${download_args[@]}"

export VOXTRAL_VENV="${venv_dir}"

python scripts/benchmark_vllm_variant.py \
  --model-path "${model_dir}" \
  --config "${config_path}" \
  --port "${port}" \
  --label "${label}" \
  --lang "${lang}" \
  --limit "${limit}" \
  --dataset-source "${dataset_source}" \
  --startup-timeout 900

echo
echo "Reproduction complete."
echo "Summary: reports/benchmark_${label}_${lang}_limit${limit}.json"
echo "Evaluation: reports/fleurs_${label}_${lang}_limit${limit}.json"
echo "Energy: reports/energy_fleurs_${label}_${lang}_limit${limit}.json"
