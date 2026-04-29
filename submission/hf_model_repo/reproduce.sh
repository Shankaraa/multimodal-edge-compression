#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-python3}"
venv_dir="${VENV_DIR:-${repo_root}/.venv}"
model_id="${MODEL_ID:-mistralai/Voxtral-Mini-4B-Realtime-2602}"
model_revision="${MODEL_REVISION:-2769294da9567371363522aac9bbcfdd19447add}"
model_dir="${MODEL_DIR:-${repo_root}}"
config_path="${CONFIG_PATH:-vllm_config.yaml}"
base_port="${BASE_PORT:-8145}"
dataset_source="${DATASET_SOURCE:-google_fleurs}"
skip_install="${SKIP_INSTALL:-0}"
install_vllm="${INSTALL_VLLM:-1}"
download_model="${DOWNLOAD_MODEL:-0}"
vllm_torch_backend="${VLLM_TORCH_BACKEND:-cu130}"
vllm_extra_index_url="${VLLM_EXTRA_INDEX_URL:-https://wheels.vllm.ai/nightly/cu130}"
run_slices="${RUN_SLICES:-en_us:500:submission_tracka_fp8_en500 fr_fr:100:submission_tracka_fp8_fr100 hi_in:100:submission_tracka_fp8_hi100 ja_jp:100:submission_tracka_fp8_ja100}"

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

if [ "${download_model}" != "0" ]; then
  download_args=(--repo-id "${model_id}" --local-dir "${model_dir}")
  if [ -n "${model_revision}" ]; then
    download_args+=(--revision "${model_revision}")
  fi
  python scripts/download_model.py "${download_args[@]}"
fi

export VOXTRAL_VENV="${venv_dir}"

port="${base_port}"
for spec in ${run_slices}; do
  IFS=":" read -r lang limit label <<< "${spec}"
  echo
  echo "Running ${label}: ${lang} limit ${limit} on port ${port}"
  python scripts/benchmark_vllm_variant.py \
    --model-path "${model_dir}" \
    --config "${config_path}" \
    --port "${port}" \
    --label "${label}" \
    --lang "${lang}" \
    --limit "${limit}" \
    --dataset-source "${dataset_source}" \
    --language-hint-mode fleurs_primary \
    --empty-retry-count 2 \
    --startup-timeout 900
  port=$((port + 1))
done

echo
echo "Track A reproduction complete."
echo "Generated benchmark summaries:"
for spec in ${run_slices}; do
  IFS=":" read -r lang limit label <<< "${spec}"
  echo "- reports/benchmark_${label}_${lang}_limit${limit}.json"
done
