#!/usr/bin/env bash
# Reproduce the reported quality + energy numbers for voxtral-mini-4b-asr.
#
# Assumes a Linux GPU host with NVIDIA driver >= 565 (CUDA 12.7 forward-compat
# or newer) and a working Python 3.11 + pip. Installs all pinned dependencies
# into the active environment unless SKIP_INSTALL=1.
#
# Environment overrides:
#   SKIP_INSTALL=1                  skip pip install
#   BASE_PORT=8084                  vLLM port (default 8084)
#   RUN_SLICES="en_us:500 hi_in:100 ..."  custom slice list
#   REPORTS_DIR=reports             output directory
#
# Slice format: <lang>:<limit>

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PORT="${BASE_PORT:-8084}"
REPORTS_DIR="${REPORTS_DIR:-reports}"
mkdir -p "$REPORTS_DIR"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    echo "[reproduce] Installing pinned dependencies..."
    pip install -q --upgrade pip wheel
    # vllm 0.19.1 pulls in torch 2.10.0+cu128 automatically (compatible with
    # any NVIDIA driver >=565 via CUDA forward-compat). Do not pre-install
    # torch from the cu124 index — torch 2.10 is not available there.
    pip install -q \
        vllm==0.19.1 \
        compressed-tensors==0.15.0.1 \
        "datasets<4" librosa scipy soundfile webrtcvad \
        pyloudnorm jiwer codecarbon hf_transfer mistral-common \
        huggingface_hub
fi

echo "[reproduce] Starting vLLM server on port $PORT ..."
nohup vllm serve . --config vllm_config.yaml --port "$PORT" \
    > server.log 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[reproduce] Stopping vLLM (pid=$SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        sleep 3
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait up to 15 min for /health
for i in $(seq 1 180); do
    if curl -sf -m 3 "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[reproduce] Server healthy after $((i*5))s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[reproduce] Server died. tail of server.log:"
        tail -40 server.log
        exit 1
    fi
    sleep 5
done

# Warm with one dummy request
python3 - <<'PYEOF'
import sys, numpy as np
sys.path.insert(0, 'src')
from voxtral_project.asr import VLLMApiTranscriber
import os
port = os.environ.get('BASE_PORT', '8084')
t = VLLMApiTranscriber(base_url=f'http://localhost:{port}/v1',
                       model='voxtral-realtime',
                       temperature=0.0, max_tokens=200)
t.transcribe(audio_array=np.zeros(16000, dtype=np.float32),
             sample_rate=16000, lang_code='en_us')
print('[reproduce] warm OK')
PYEOF

# Default slice list: the 4 canonical FLEURS slices used for the energy claim.
SLICES="${RUN_SLICES:-en_us:500 hi_in:100 fr_fr:100 ja_jp:100}"

for slc in $SLICES; do
    LANG="${slc%%:*}"
    LIMIT="${slc#*:}"

    EVAL_OUT="$REPORTS_DIR/l4_${LANG}_limit${LIMIT}.json"
    ENERGY_OUT="$REPORTS_DIR/energy_l4_${LANG}_limit${LIMIT}.json"

    echo "[reproduce] ===== $LANG limit=$LIMIT ====="
    python scripts/measure_energy.py \
        --report "$ENERGY_OUT" \
        -- \
        python scripts/evaluate_fleurs.py \
            --lang "$LANG" --limit "$LIMIT" \
            --base-url "http://localhost:$PORT/v1" \
            --model voxtral-realtime \
            --target-lufs -23.0 --lufs-max-gain-db 24.0 \
            --vad-trim --vad-aggressiveness 1 --vad-padding-ms 200 \
            --gate-silence \
            --compress-internal-silence-to-ms 160 \
            --min-internal-silence-run-ms 320 \
            --out "$EVAL_OUT"
done

echo "[reproduce] All slices complete. Reports under $REPORTS_DIR/"
ls -lh "$REPORTS_DIR/"
