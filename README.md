# Voxtral Compression Project

This repo is the starter workspace for compressing `mistralai/Voxtral-Mini-4B-Realtime-2602`
for the Resilient AI Challenge audio-to-text track.

The project is built around one core constraint from the guide:

- Protect the audio encoder.
- Leave the adapter and output surfaces alone unless we have strong evidence.
- Compress the decoder first because it carries most of the parameter count and most of the
  energy opportunity.

## Current Goal

We are setting up a clean baseline workflow so we can:

- download the model,
- serve it through `vLLM`,
- run transcription smoke tests,
- measure WER on FLEURS,
- measure energy with CodeCarbon,
- compare baseline vs compressed experiments.

## Important Runtime Note

This machine is currently a Windows workspace, but the competition guide is Linux-oriented and
`vLLM` is most practical in Linux or WSL2. This repo is structured so we can manage the project
from Windows while running the heavy runtime pieces in Linux when needed.

## Repo Layout

- `configs/experiments.yaml` - named experiment matrix and module protection policy.
- `configs/vllm/` - starter `vLLM` configs for baseline and compression experiments.
- `docs/guide_notes.md` - distilled notes from the PDF guide.
- `scripts/download_model.py` - download the model from Hugging Face.
- `scripts/serve_model.py` - launch `vLLM serve` from a YAML config.
- `scripts/start_wsl_baseline.ps1` - start the BF16 baseline server in WSL from PowerShell.
- `scripts/check_vllm_server.py` - poll `/v1/models` until the server is ready.
- `scripts/smoke_test_hf_sample.py` - transcribe a known public sample audio file from Hugging Face.
- `scripts/transcribe_file.py` - send one audio file to the server and print the transcript.
- `scripts/evaluate_fleurs.py` - run WER evaluation on one or more FLEURS languages.
- `scripts/measure_energy.py` - wrap any command with CodeCarbon energy tracking.
- `src/voxtral_project/` - shared helpers for API calls, audio conversion, and report writing.

## Quick Start

1. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. If you are running the model in Linux or WSL2, install the GPU/runtime stack too:

```bash
pip install -r requirements-linux-gpu.txt
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

3. Download the baseline model:

```powershell
python scripts/download_model.py --local-dir models/voxtral-realtime
```

4. Serve the BF16 baseline:

```powershell
python scripts/serve_model.py models/voxtral-realtime --config configs/vllm/bf16.yaml
```

If you are launching from Windows into WSL, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_wsl_baseline.ps1
```

5. Wait for the server to become ready:

```powershell
python scripts/check_vllm_server.py --base-url http://localhost:8080/v1
```

6. Run a public sample smoke test:

```powershell
python scripts/smoke_test_hf_sample.py --model voxtral-realtime --out reports/smoke_test_transcript.txt
```

7. Smoke test one local audio file:

```powershell
python scripts/transcribe_file.py path\to\sample.wav --model voxtral-realtime
```

8. Run a small FLEURS evaluation:

```powershell
python scripts/evaluate_fleurs.py --lang en_us --limit 10 --out reports/fleurs_en_us.json
```

9. Measure energy for an evaluation run:

```powershell
python scripts/measure_energy.py --report reports/bf16_energy.json -- python scripts/evaluate_fleurs.py --lang en_us --limit 10
```

## Initial Experiment Order

- `bf16_baseline`
- `fp8_round1`
- `gptq8_round1`
- `gptq4_round2`
- `gptq4_fp8kv_round2`

That ordering follows the guide: get a stable baseline first, try the simplest hardware-friendly
compression next, then move into more aggressive decoder quantization.

## What Is Intentionally Missing

- No end-to-end quantization automation yet.
- No cloud deployment yet.
- No submission packaging yet.

Those come after we lock down a reliable local baseline and evaluation loop.
