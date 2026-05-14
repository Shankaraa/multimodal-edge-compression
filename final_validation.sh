#!/bin/bash
set -euo pipefail

REPORT_ROOT="${REPORT_ROOT:-reports/l4_final}"
MODEL_PATH="${MODEL_PATH:-models/voxtral-realtime}"
MODEL_NAME="${MODEL_NAME:-voxtral-realtime}"
BF16_CONFIG="${BF16_CONFIG:-configs/vllm/bf16_current_harness.yaml}"
FP8_CONFIG="${FP8_CONFIG:-configs/vllm/fp8_round1.yaml}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Neither python nor python3 was found on PATH." >&2
        exit 127
    fi
fi
LIMIT="${LIMIT:-100}"
VARIANCE_LIMIT="${VARIANCE_LIMIT:-500}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
SERVER_SETTLE_SECONDS="${SERVER_SETTLE_SECONDS:-30}"
TAUS="${TAUS:-240 480}"
WINNING_TAU="${WINNING_TAU:-}"

mkdir -p "${REPORT_ROOT}/logs"
START_TIME=$(date +%s)
CURRENT_SERVER_PID=""
CURRENT_SERVER_NAME=""

log() {
    echo "[$(date -Is)] $*"
}

stop_server() {
    if [[ -z "${CURRENT_SERVER_PID}" ]]; then
        return 0
    fi

    log "Stopping ${CURRENT_SERVER_NAME} server pid=${CURRENT_SERVER_PID}"
    kill "${CURRENT_SERVER_PID}" 2>/dev/null || true
    wait "${CURRENT_SERVER_PID}" 2>/dev/null || true
    CURRENT_SERVER_PID=""
    CURRENT_SERVER_NAME=""
    sleep "${SERVER_SETTLE_SECONDS}"
}

cleanup() {
    stop_server || true
}

trap cleanup EXIT

run_server() {
    local name="$1"
    local config="$2"
    local port="$3"
    local log_path="${REPORT_ROOT}/logs/${name}_vllm.log"

    stop_server
    log "Starting ${name} on port ${port} with ${config}"
    "${PYTHON_BIN}" scripts/serve_model.py "${MODEL_PATH}" \
        --config "${config}" \
        --port "${port}" \
        >"${log_path}" 2>&1 &
    CURRENT_SERVER_PID=$!
    CURRENT_SERVER_NAME="${name}"

    "${PYTHON_BIN}" scripts/check_vllm_server.py \
        --base-url "http://localhost:${port}/v1" \
        --timeout "${STARTUP_TIMEOUT}"
}

run_energy_eval() {
    local label="$1"
    local port="$2"
    local config="$3"
    local out_dir="$4"
    shift 4

    mkdir -p "${out_dir}"
    "${PYTHON_BIN}" scripts/measure_energy.py \
        --report "${REPORT_ROOT}/${label}_energy.json" \
        -- \
        "${PYTHON_BIN}" scripts/evaluate_full_suite.py \
            --base-url "http://localhost:${port}/v1" \
            --model "${MODEL_NAME}" \
            --model-label "${label}" \
            --config "${config}" \
            --limit "${LIMIT}" \
            --output-dir "${out_dir}" \
            "$@"
}

write_summary() {
    local summary_path="${REPORT_ROOT}/summary.json"
    "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

root = Path("${REPORT_ROOT}")
summary = {
    "report_root": str(root),
    "limit_per_language": int("${LIMIT}"),
    "variance_limit": int("${VARIANCE_LIMIT}"),
    "taus": "${TAUS}".split(),
    "winning_tau": "${WINNING_TAU}",
    "runs": {},
}
for path in sorted(root.glob("**/summary.json")):
    if path == root / "summary.json":
        continue
    summary["runs"][str(path.parent.relative_to(root))] = json.loads(path.read_text(encoding="utf-8"))
for path in sorted(root.glob("*_energy.json")):
    summary.setdefault("energy", {})[path.stem] = json.loads(path.read_text(encoding="utf-8"))
summary_path = root / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {summary_path}")
PY
}

log "Step 1: Verify environment"
"${PYTHON_BIN}" - <<'PY'
import torch
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'}")
PY

log "Step 2: Sample 1985 sanity gate"
run_server "gate_fp8" "${FP8_CONFIG}" 8000
"${PYTHON_BIN}" scripts/run_sample_gate.py \
    --base-url "http://localhost:8000/v1" \
    --model "${MODEL_NAME}" \
    --sample-id 1985 \
    --out "${REPORT_ROOT}/sample_1985_gate.json"
stop_server

log "Step 3: BF16 baseline full-suite run"
run_server "bf16" "${BF16_CONFIG}" 8001
run_energy_eval "bf16" 8001 "${BF16_CONFIG}" "${REPORT_ROOT}/bf16" \
    --language-hint-mode fleurs_primary \
    --empty-retry-count 2
stop_server

log "Step 4: FP8 candidate full-suite tau sweep"
for TAU in ${TAUS}; do
    run_server "fp8_tau${TAU}" "${FP8_CONFIG}" 8002
    run_energy_eval "fp8_tau${TAU}" 8002 "${FP8_CONFIG}" "${REPORT_ROOT}/fp8_tau${TAU}" \
        --target-streaming-delay-ms "${TAU}" \
        --language-hint-mode fleurs_primary \
        --empty-retry-count 2
    stop_server
done

if [[ -z "${WINNING_TAU}" ]]; then
    WINNING_TAU="$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

root = Path("${REPORT_ROOT}")
best = None
for tau in "${TAUS}".split():
    summary_path = root / f"fp8_tau{tau}" / "summary.json"
    if not summary_path.exists():
        continue
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("empty_predictions", 0) != 0:
        continue
    score = summary.get("normalized_wer_percent")
    if score is None:
        continue
    if best is None or score < best[0]:
        best = (score, tau)
if best is None:
    raise SystemExit("No FP8 tau run had a usable summary with zero empty predictions.")
print(best[1])
PY
)"
fi
log "Step 5: Variance check on winning tau ${WINNING_TAU}"
for RUN_INDEX in 1 2 3; do
    run_server "fp8_tau${WINNING_TAU}_en500_run${RUN_INDEX}" "${FP8_CONFIG}" 8003
    "${PYTHON_BIN}" scripts/measure_energy.py \
        --report "${REPORT_ROOT}/fp8_tau${WINNING_TAU}_en500_run${RUN_INDEX}_energy.json" \
        -- \
        "${PYTHON_BIN}" scripts/evaluate_fleurs.py \
            --base-url "http://localhost:8003/v1" \
            --model "${MODEL_NAME}" \
            --model-label "fp8_tau${WINNING_TAU}_en500_run${RUN_INDEX}" \
            --config "${FP8_CONFIG}" \
            --lang en_us \
            --limit "${VARIANCE_LIMIT}" \
            --dataset-source google_fleurs \
            --target-streaming-delay-ms "${WINNING_TAU}" \
            --language-hint-mode fleurs_primary \
            --temperature 0.0 \
            --empty-retry-count 2 \
            --out "${REPORT_ROOT}/fp8_tau${WINNING_TAU}_en500_run${RUN_INDEX}.json"
    stop_server
done

log "Step 6: Summarize and package outputs"
write_summary
END_TIME=$(date +%s)
echo "Total elapsed: $((END_TIME - START_TIME)) seconds" | tee "${REPORT_ROOT}/elapsed.txt"
tar czf l4_final_results.tar.gz "${REPORT_ROOT}"
echo "Done. Results in l4_final_results.tar.gz"
