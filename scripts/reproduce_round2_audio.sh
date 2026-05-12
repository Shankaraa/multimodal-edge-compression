#!/usr/bin/env bash
# Reproduce the Round-2 audio-lever evaluation locally (RTX 5080 / WSL).
#
# Outputs:
#   - reports/fleurs_fp8_<lang>_<limit>_lufs23_vadgate_smoke.json
#   - reports/energy_fleurs_fp8_<lang>_<limit>_lufs23_vadgate_smoke.json (CodeCarbon, relative)
#
# Pre-requisites already in place on this machine:
#   - WSL Ubuntu-22.04
#   - ~/.venvs/voxtral-baseline (vllm 0.19.1rc1.dev302+cu130, transformers 4.57.6,
#     pyloudnorm 0.2.0)
#   - models/voxtral-realtime/ on /mnt/c
#
# Pre-requisites you must arrange:
#   - An FP8 vLLM server already serving on http://localhost:8082/v1, e.g.:
#       python scripts/serve_model.py models/voxtral-realtime \
#         --config configs/vllm/fp8_round1.yaml --port 8082
#   - Server warm (1 dummy request) before timing matters.
#
# Energy numbers from this script are RELATIVE only (RTX 5080, CPU TDP estimate). The
# binding submission energy must be measured on the official L4 evaluation hardware
# with the same parameter set.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_PATH="${VOXTRAL_VENV:-$HOME/.venvs/voxtral-baseline}"
BASE_URL="${VOXTRAL_BASE_URL:-http://localhost:8082/v1}"
MODEL_NAME="${VOXTRAL_MODEL:-voxtral-realtime}"

SLICES=(
    "en_us:500"
    "hi_in:100"
    "fr_fr:100"
    "ja_jp:100"
)

VAD_GATE_FLAGS=(
    --target-lufs -23.0
    --lufs-max-gain-db 24.0
    --vad-trim --vad-aggressiveness 1 --vad-padding-ms 200
    --gate-silence
    --compress-internal-silence-to-ms 160
    --min-internal-silence-run-ms 320
)

source "${VENV_PATH}/bin/activate"

for slice in "${SLICES[@]}"; do
    lang="${slice%%:*}"
    limit="${slice##*:}"
    tag="lufs23_vadgate"
    energy_report="reports/energy_fleurs_fp8_${lang}_limit${limit}_${tag}.json"
    eval_report="reports/fleurs_fp8_${lang}_limit${limit}_${tag}.json"

    echo "===== ${lang} limit=${limit} ====="
    python scripts/measure_energy.py \
        --report "${energy_report}" \
        -- \
        python scripts/evaluate_fleurs.py \
            --lang "${lang}" --limit "${limit}" \
            --base-url "${BASE_URL}" --model "${MODEL_NAME}" \
            "${VAD_GATE_FLAGS[@]}" \
            --out "${eval_report}"
done

echo
echo "Summary:"
python scripts/summarize_round2_audio_runs.py \
    --variants "FP8 EN500 LUFS+VAD+gate=fleurs_fp8_en_us_limit500_lufs23_vadgate.json" \
    --variants "FP8 HI100 LUFS+VAD+gate=fleurs_fp8_hi_in_limit100_lufs23_vadgate.json" \
    --variants "FP8 FR100 LUFS+VAD+gate=fleurs_fp8_fr_fr_limit100_lufs23_vadgate.json" \
    --variants "FP8 JA100 LUFS+VAD+gate=fleurs_fp8_ja_jp_limit100_lufs23_vadgate.json"
