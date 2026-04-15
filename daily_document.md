# Daily Document

## Date

April 15, 2026

## Project

Voxtral Real-Time 4B compression project for the Resilient AI Challenge audio-to-text track.

## Objective For Today

Set up the project workspace, understand the compression guide, prepare a baseline evaluation
workflow, and bootstrap the Linux runtime in WSL so the BF16 baseline can be run on the local
RTX 5080 before moving into compression experiments.

## What Was Done Today

### 1. Read and analyzed the PDF guide

- File reviewed: `voxtral realtime compression guide.pdf`
- The guide was inspected directly and its text was extracted from the PDF structure.
- The document was confirmed to be a 26-page guide, with 23 content pages and trailing PDF
  structure pages.
- The core strategy from the guide was identified:
  - protect the audio encoder,
  - leave the adapter mostly untouched,
  - compress the decoder first,
  - use `vLLM`,
  - optimize for energy while preserving WER.

### 2. Created the initial project scaffold

The workspace started nearly empty, with only the PDF present. The following project structure
was created:

- `README.md`
- `.gitignore`
- `requirements.txt`
- `requirements-linux-gpu.txt`
- `docs/guide_notes.md`
- `configs/experiments.yaml`
- `configs/vllm/bf16.yaml`
- `configs/vllm/fp8_round1.yaml`
- `configs/vllm/gptq_round1.yaml`
- `configs/vllm/gptq_round2.yaml`
- `configs/vllm/aggressive_round2.yaml`
- `src/voxtral_project/__init__.py`
- `src/voxtral_project/audio.py`
- `src/voxtral_project/api.py`
- `scripts/download_model.py`
- `scripts/serve_model.py`
- `scripts/transcribe_file.py`
- `scripts/evaluate_fleurs.py`
- `scripts/measure_energy.py`

### 3. Built the Python starter workflow

The project now includes baseline scripts for:

- downloading the model from Hugging Face,
- launching `vLLM` from YAML config,
- transcribing a local audio file through the server,
- evaluating WER on FLEURS,
- measuring energy usage with CodeCarbon.

### 4. Repaired and prepared the Windows-side virtual environment

- Created `.venv` in the project workspace.
- Fixed an initial `venv`/`ensurepip` issue where the environment existed but `pip` was missing.
- Installed the base Python dependencies from `requirements.txt`.
- Verified that the project entrypoints could parse `--help` correctly after import cleanup.

### 5. Verified local hardware on Windows

`nvidia-smi` on Windows confirmed:

- GPU: `NVIDIA GeForce RTX 5080`
- Driver version: `581.95`
- CUDA version: `13.0`
- Available VRAM: about `16 GB`

### 6. Verified WSL availability and Linux runtime access

WSL was initially blocked by sandbox permissions, then checked successfully with elevated access.

Confirmed:

- Default distro: `Ubuntu-22.04`
- WSL version: `2`
- The project workspace is accessible from WSL at:
  - `/mnt/c/Users/ASUS/Music/Fine_tuning`

### 7. Verified GPU visibility inside WSL

Inside WSL `Ubuntu-22.04`, the following were confirmed:

- Python available at `/usr/bin/python3`
- NVIDIA GPU visible through `nvidia-smi`
- Torch later confirmed CUDA access successfully

### 8. Installed Linux system packages in WSL

Using WSL root, the following Ubuntu packages were installed:

- `build-essential`
- `python3-pip`
- `python3-venv`
- `ffmpeg`
- `libsndfile1`

### 9. Created the Linux baseline virtual environment

Created WSL virtual environment:

- `~/.venvs/voxtral-baseline`

Installed into that environment:

- upgraded `pip`, `setuptools`, and `wheel`
- packages from `requirements.txt`
- `mistral_common[audio]`
- `transformers`
- `vllm` nightly

### 10. Verified the WSL baseline environment

The following checks passed inside WSL:

- `vllm` import worked
- `transformers` import worked
- Hugging Face connectivity worked for:
  - `mistralai/Voxtral-Mini-4B-Realtime-2602`
- Torch reported CUDA available
- Torch saw device:
  - `NVIDIA GeForce RTX 5080`

### 11. Resumed the model download and resolved Hugging Face lock contention

- The model download was restarted from the prepared WSL environment.
- A later inspection showed that multiple older WSL download processes were still running in the
  background at the same time.
- Those duplicate processes were competing for the same Hugging Face lock files:
  - `model.safetensors.lock`
  - `consolidated.safetensors.lock`
- The duplicate background jobs were stopped so that only the current foreground download could
  continue.
- Partial download progress was preserved because Hugging Face uses resumable `.incomplete`
  files.
- At the time of inspection, partial files included:
  - one file around `5.03 GB`
  - one file around `595 MB`

### 12. Added post-download baseline helper scripts

To reduce downtime while the model continues downloading, more helper tooling was added:

- `scripts/check_vllm_server.py`
  - waits for the `vLLM` server and lists models from `/v1/models`
- `scripts/smoke_test_hf_sample.py`
  - downloads a known public sample audio file and runs a transcription smoke test
- `scripts/start_wsl_baseline.ps1`
  - launches the BF16 baseline from Windows into WSL

Additional consistency update:

- All `vLLM` YAML configs were updated to use:
  - `served_model_name: voxtral-realtime`

This keeps the server-side model name aligned with the client-side scripts.

## Important Findings From Today

- The project is now structurally ready for baseline inference and evaluation.
- WSL is usable and already configured with Ubuntu 22.04.
- The Linux runtime is the right place to run Voxtral and `vLLM`.
- The baseline environment appears healthy enough to proceed directly to model download and
  serving.
- The guide strongly supports decoder-first compression and warns against aggressive encoder
  quantization.

## Model Download Status

- The model download was started from WSL more than once.
- Some earlier attempts were manually interrupted, but the local Hugging Face cache preserved
  partial progress.
- A later check showed multiple concurrent WSL download processes were blocking one another on
  lock files.
- Those duplicate background processes were stopped.
- The current foreground model download is now the intended active download.
- The download is still in progress as of the latest update.
- Because the Hugging Face downloader supports resuming, existing partial files should continue
  from their current state rather than restarting from zero.

Planned baseline model location:

- Windows path: `C:\Users\ASUS\Music\Fine_tuning\models\voxtral-realtime`
- WSL path: `/mnt/c/Users/ASUS/Music/Fine_tuning/models/voxtral-realtime`

## Commands Used Or Prepared

### Resume model download

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "source ~/.venvs/voxtral-baseline/bin/activate && cd /mnt/c/Users/ASUS/Music/Fine_tuning && python scripts/download_model.py --local-dir models/voxtral-realtime"
```

### Start BF16 baseline server after download completes

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "source ~/.venvs/voxtral-baseline/bin/activate && cd /mnt/c/Users/ASUS/Music/Fine_tuning && python scripts/serve_model.py models/voxtral-realtime --config configs/vllm/bf16.yaml"
```

### Alternative direct Hugging Face download command

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "source ~/.venvs/voxtral-baseline/bin/activate && huggingface-cli download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir /mnt/c/Users/ASUS/Music/Fine_tuning/models/voxtral-realtime --local-dir-use-symlinks False"
```

### Updated direct Hugging Face download command

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "source ~/.venvs/voxtral-baseline/bin/activate && hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir /mnt/c/Users/ASUS/Music/Fine_tuning/models/voxtral-realtime"
```

### Start BF16 baseline from PowerShell into WSL

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_wsl_baseline.ps1
```

### Wait for server readiness

```powershell
.\.venv\Scripts\python.exe scripts\check_vllm_server.py --base-url http://localhost:8080/v1
```

### Run public sample smoke test

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_hf_sample.py --model voxtral-realtime --out reports/smoke_test_transcript.txt
```

## Current Project State At End Of Today

- PDF guide understood and distilled into project notes.
- Repo scaffold created.
- Baseline scripts created.
- Windows `.venv` created and repaired.
- WSL Ubuntu runtime verified.
- WSL system dependencies installed.
- WSL Python baseline environment created.
- `vLLM` nightly installed in WSL.
- CUDA confirmed in WSL with Torch.
- Duplicate WSL Hugging Face download processes were identified and cleaned up.
- Post-download helper scripts were added for launch, health checks, and smoke testing.
- Model download currently in progress.
- BF16 server not yet launched.
- No transcription test run yet.
- No FLEURS baseline WER run yet.
- No compression experiment run yet.

## Recommended Next Step

Complete the baseline model download, then immediately:

1. start the BF16 `vLLM` server,
2. wait for the server readiness check to pass,
3. run a simple transcription smoke test,
4. run a small FLEURS evaluation on English,
5. capture the first baseline WER and energy numbers.
