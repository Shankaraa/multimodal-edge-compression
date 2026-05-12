#!/usr/bin/env bash
# Run the Round-2 audio-lever multilingual sweep with the locked parameter set.
#
# Default flags assume LUFS+VAD+gate. Set ROUND2_NO_LUFS=1 if the EN500 ablation
# decides LUFS should be dropped.
#
# Pre-requisites:
#   - FP8 vLLM server already serving on http://localhost:8082/v1
#   - ~/.venvs/voxtral-baseline activated implicitly via the script

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_PATH="${VOXTRAL_VENV:-$HOME/.venvs/voxtral-baseline}"
BASE_URL="${VOXTRAL_BASE_URL:-http://localhost:8082/v1}"
MODEL_NAME="${VOXTRAL_MODEL:-voxtral-realtime}"

source "${VENV_PATH}/bin/activate"

LUFS_FLAGS=()
TAG="lufs23_vadgate"
if [ "${ROUND2_NO_LUFS:-0}" = "1" ]; then
    LUFS_FLAGS=()
    TAG="vadgate_nolufs"
else
    LUFS_FLAGS=(--target-lufs -23.0 --lufs-max-gain-db 24.0)
fi

VAD_GATE_FLAGS=(
    --vad-trim --vad-aggressiveness 1 --vad-padding-ms 200
    --gate-silence
    --compress-internal-silence-to-ms 320
    --min-internal-silence-run-ms 640
)

SLICES=(
    "hi_in:100"
    "fr_fr:100"
    "ja_jp:100"
)

for slice in "${SLICES[@]}"; do
    lang="${slice%%:*}"
    limit="${slice##*:}"
    energy_report="reports/energy_fleurs_fp8_${lang}_limit${limit}_${TAG}_smoke.json"
    eval_report="reports/fleurs_fp8_${lang}_limit${limit}_${TAG}_smoke.json"

    echo "===== ${lang} limit=${limit} ${TAG} ====="
    python scripts/measure_energy.py \
        --report "${energy_report}" \
        -- \
        python scripts/evaluate_fleurs.py \
            --lang "${lang}" --limit "${limit}" \
            --base-url "${BASE_URL}" --model "${MODEL_NAME}" \
            "${LUFS_FLAGS[@]}" "${VAD_GATE_FLAGS[@]}" \
            --out "${eval_report}"
done
