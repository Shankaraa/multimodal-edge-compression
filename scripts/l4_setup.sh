#!/usr/bin/env bash
# L4 Instance Setup — Round-2 Track A++ Submission Measurement
#
# Run this once on a fresh Ubuntu 22.04 L4 (24 GB VRAM) instance.
# Installs the exact venv stack from the dev machine, pulls the model, and
# leaves the machine ready to run reproduce_round2_audio.sh.
#
# Usage (from the repo root, as non-root user with sudo):
#   bash scripts/l4_setup.sh
#
# After this script completes:
#   1. Pull repo to ~/Fine_tuning:
#        git clone <your_repo> ~/Fine_tuning && cd ~/Fine_tuning
#   2. Start the FP8 server:
#        python scripts/serve_model.py ~/models/voxtral-realtime \
#          --config configs/vllm/fp8_round1.yaml --port 8082
#   3. Run the binding measurement:
#        bash scripts/reproduce_round2_audio.sh
#
# Pinned stack (must match voxtral-baseline on dev machine):
#   vllm==0.19.1rc1.dev302+g68be0f853.cu130
#   torch==2.11.0+cu130
#   transformers==4.57.6
#   compressed-tensors==0.14.0.1
#   flashinfer-python==0.6.7
#   pyloudnorm==0.2.0

set -euo pipefail

VENV_NAME="${VENV_NAME:-voxtral-l4}"
VENV_PATH="$HOME/.venvs/$VENV_NAME"
MODEL_DIR="${MODEL_DIR:-$HOME/models/voxtral-realtime}"

# ── System packages ──────────────────────────────────────────────────────────
echo "[l4_setup] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-venv python3-pip \
    git git-lfs \
    libsndfile1 ffmpeg \
    curl wget \
    nvtop  # optional: GPU monitoring

git lfs install --skip-repo

# ── Python venv ──────────────────────────────────────────────────────────────
echo "[l4_setup] Creating venv at $VENV_PATH ..."
mkdir -p "$(dirname "$VENV_PATH")"
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# Upgrade pip/wheel first
pip install -q --upgrade pip wheel setuptools

# ── CUDA-specific torch (pinned) ─────────────────────────────────────────────
echo "[l4_setup] Installing pinned torch 2.11.0+cu130 ..."
pip install -q \
    "torch==2.11.0" \
    --index-url https://download.pytorch.org/whl/cu130

# ── vLLM (pinned dev build matching dev machine) ────────────────────────────
echo "[l4_setup] Installing pinned vLLM 0.19.1rc1.dev302+cu130 ..."
# This exact wheel must be available; adjust URL if your team hosts it privately.
VLLM_WHEEL="vllm-0.19.1rc1.dev302+g68be0f853.cu130-cp310-cp310-linux_x86_64.whl"
if [[ -f "$HOME/$VLLM_WHEEL" ]]; then
    pip install -q "$HOME/$VLLM_WHEEL"
else
    echo "[l4_setup] WARNING: $VLLM_WHEEL not found at $HOME/$VLLM_WHEEL"
    echo "  Upload it first: scp voxtral-baseline-wheels/$VLLM_WHEEL user@l4:~/"
    echo "  Falling back to latest vLLM (NOT pinned — do not use for submission)."
    pip install -q vllm
fi

# ── Remaining pinned packages ────────────────────────────────────────────────
echo "[l4_setup] Installing pinned stack ..."
pip install -q \
    "transformers==4.57.6" \
    "compressed-tensors==0.14.0.1" \
    "flashinfer-python==0.6.7" \
    "pyloudnorm==0.2.0" \
    "soundfile" \
    "datasets" \
    "jiwer" \
    "codecarbon" \
    "webrtcvad" \
    "numpy" \
    "requests" \
    "tqdm"

echo "[l4_setup] Installed package versions:"
pip show vllm torch transformers compressed-tensors flashinfer-python pyloudnorm \
    2>/dev/null | grep -E '^(Name|Version)'

# ── Model download ───────────────────────────────────────────────────────────
echo "[l4_setup] Pulling model to $MODEL_DIR ..."
mkdir -p "$MODEL_DIR"

# The model is a private gated HF repo.
# Ensure HF_TOKEN is set in env or pass --token to huggingface-cli.
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[l4_setup] WARNING: HF_TOKEN not set. huggingface-cli login required."
    echo "  Run: huggingface-cli login"
fi

pip install -q huggingface_hub
python3 - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
token = os.environ.get("HF_TOKEN")
model_dir = os.path.expanduser(os.environ.get("MODEL_DIR", "~/models/voxtral-realtime"))
repo_id = "Shankara-A-S/voxtral-mini-realtime-fp8-runtime"
print(f"[l4_setup] Downloading {repo_id} -> {model_dir}")
snapshot_download(
    repo_id=repo_id,
    local_dir=model_dir,
    token=token,
    ignore_patterns=["*.md", "*.gitattributes"],
)
print(f"[l4_setup] Model ready at {model_dir}")
PYEOF

echo
echo "════════════════════════════════════════════════════════════"
echo " L4 setup complete."
echo ""
echo " Next steps:"
echo "   source $VENV_PATH/bin/activate"
echo "   cd ~/Fine_tuning"
echo "   # Start FP8 server:"
echo "   python scripts/serve_model.py $MODEL_DIR \\"
echo "     --config configs/vllm/fp8_round1.yaml --port 8082"
echo "   # Wait for /health 200, then run binding measurement:"
echo "   VOXTRAL_VENV=$VENV_PATH bash scripts/reproduce_round2_audio.sh"
echo "════════════════════════════════════════════════════════════"
