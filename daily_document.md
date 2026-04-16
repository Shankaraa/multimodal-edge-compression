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

### 13. Prepared GitHub remote and pushed the project branch

- The repository remote was updated to:
  - `https://github.com/Shankaraa/multimodal-edge-compression.git`
- A working branch named `shankara` was created locally.
- The scaffold and helper scripts were committed and pushed successfully after GitHub CLI
  authentication was completed in WSL.
- Large model artifacts remain excluded from version control through `.gitignore`, including:
  - `*.pt`
  - `*.pth`
  - `*.bin`
  - `*.safetensors`

### 14. Completed the local Voxtral model download

- The full model download completed successfully in WSL.
- The model directory now contains the expected metadata files plus both large weight files:
  - `consolidated.safetensors`
  - `model.safetensors`
- Verified local model directory size:
  - about `17 GB`
- Confirmed model location:
  - Windows path: `C:\Users\ASUS\Music\Fine_tuning\models\voxtral-realtime`
  - WSL path: `/mnt/c/Users/ASUS/Music/Fine_tuning/models/voxtral-realtime`

### 15. Attempted the BF16 baseline launch and identified a runtime mismatch

- The BF16 baseline launch was attempted after the model became available locally.
- Torch in the WSL baseline environment reports:
  - version `2.11.0`
  - CUDA runtime `13.0`
  - CUDA available: `True`
- The current installed `vllm` package is:
  - `0.19.1rc1.dev300+g29e5d1020`
- The server did not start because the current `vllm` native extension is trying to load:
  - `libcudart.so.12`
- This indicates the installed `vllm` build does not match the CUDA 13 Torch/runtime stack in
  the current WSL environment.
- Result:
  - model download is complete,
  - baseline server launch is blocked until the `vllm` install is rebuilt or replaced with a
    CUDA 13 compatible wheel.

## Important Findings From Today

- The project is now structurally ready for baseline inference and evaluation.
- WSL is usable and already configured with Ubuntu 22.04.
- The Linux runtime is the right place to run Voxtral and `vLLM`.
- The model download has completed successfully and is available locally.
- The remaining blocker is not storage or download state, but a CUDA build mismatch inside the
  WSL `vllm` environment.
- The guide strongly supports decoder-first compression and warns against aggressive encoder
  quantization.

## Model Download Status

- The model download was started from WSL more than once earlier in the day.
- Some earlier attempts were manually interrupted, but the local Hugging Face cache preserved
  partial progress and allowed resuming.
- Duplicate download processes were cleaned up after they were found contending on lock files.
- The final download completed successfully.

Final baseline model location:

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
- GitHub remote updated and `shankara` branch pushed successfully.
- Model download completed successfully.
- BF16 server launch attempted.
- BF16 server currently blocked by `vllm` expecting `libcudart.so.12` while Torch is using CUDA
  `13.0`.
- No transcription test run yet.
- No FLEURS baseline WER run yet.
- No compression experiment run yet.

## Recommended Next Step

Repair the WSL `vllm` installation so it matches the CUDA 13 runtime, then immediately:

1. reinstall or replace `vllm` with a CUDA 13 compatible build,
2. start the BF16 `vLLM` server,
3. wait for the server readiness check to pass,
4. run a simple transcription smoke test,
5. run a small FLEURS evaluation on English,
6. capture the first baseline WER and energy numbers.

---

## Date

April 16, 2026

## Objective For Today

Get the BF16 baseline fully serving on the local RTX 5080 in WSL, make the transcription path work
end to end, and capture the first real baseline WER and energy numbers.

## What Was Done Today

### 1. Repaired the CUDA runtime mismatch in WSL

- The original `vllm` install was still linked against `libcudart.so.12`, while Torch in the WSL
  environment was running with CUDA `13.0`.
- The WSL baseline environment was repaired by installing a CUDA 13 compatible nightly `vllm`
  build.
- Verified working WSL stack after repair:
  - `torch 2.11.0+cu130`
  - `vllm 0.19.1rc1.dev302+g68be0f853.cu130`

### 2. Hardened local server launch for WSL

- `scripts/serve_model.py` was updated to inject the WSL venv Torch and NVIDIA shared-library
  paths into `LD_LIBRARY_PATH` before launching `vllm`.
- Added `scripts/start_vllm_server.sh` to provide a stable WSL-native launch path without fragile
  inline shell quoting.
- Port `8080` was already occupied locally, so the working baseline server was moved to:
  - `http://localhost:8081/v1`

### 3. Adjusted the BF16 config to fit the local 16 GB GPU budget

- The original baseline config used:
  - `max_model_len: 16384`
- That setting failed during KV-cache initialization on the RTX 5080 under WSL.
- The BF16 config was reduced to:
  - `max_model_len: 8192`
- After this change, the model loaded successfully and the server reached ready state.

### 4. Fixed the live transcription path

- The live server initially returned `500` errors because the WSL environment was missing:
  - `av`
- Installed the missing WSL audio runtime dependency:
  - `av`
- Later evaluation work also required:
  - `librosa`
- The project client in `src/voxtral_project/api.py` was switched from the chat-completions
  multimodal path to the dedicated OpenAI-style speech endpoint:
  - `/v1/audio/transcriptions`

### 5. Fixed evaluation compatibility issues

- `scripts/evaluate_fleurs.py` had a Python 3.10 issue using `datetime.UTC`.
- This was updated to use:
  - `datetime.now(timezone.utc)`
- The installed Hugging Face `datasets` package was too new for script-backed dataset loading.
- WSL `datasets` was downgraded from `4.8.4` to a script-compatible release:
  - `3.6.0`
- The repo requirement was updated to:
  - `datasets>=2.18.0,<4`
- FLEURS loading was updated to pass:
  - `trust_remote_code=True`

### 6. Captured the first successful smoke transcription

- The public sample smoke test completed successfully against:
  - `http://localhost:8081/v1`
- Transcript saved to:
  - `reports/smoke_test_transcript.txt`
- Example output:
  - `Yesterday it was 35 degrees in Barcelona, but today the temperature will go down to minus 20 degrees.`

### 7. Identified a concurrency-related engine crash

- A parallel smoke-test request and evaluation request were accidentally launched at the same time.
- Under concurrent transcription requests, the current `vllm` Voxtral realtime path crashed with
  tensor shape mismatch errors inside the engine core.
- Sequential requests worked correctly.
- Practical current rule for this local baseline:
  - run transcription and evaluation jobs one request at a time
  - avoid overlapping audio requests until this engine issue is better understood

### 8. Captured the first English baseline WER results

- A one-sample English FLEURS sanity check completed successfully:
  - `WER = 21.05%`
- Then a five-sample English FLEURS baseline completed successfully:
  - `WER = 34.95%`
- Reports written:
  - `reports/fleurs_en_us_limit1.json`
  - `reports/fleurs_en_us_limit5.json`

### 9. Captured the first energy measurement

- `scripts/measure_energy.py` was also patched for Python 3.10 compatibility.
- A five-sample English FLEURS evaluation was wrapped with CodeCarbon.
- Energy report written to:
  - `reports/energy_fleurs_en_us_limit5.json`
- Additional evaluation report written to:
  - `reports/fleurs_en_us_limit5_energy_run.json`
- Measured values:
  - `energy_joules: 4775.58`
  - `emissions_kg: 0.000946`

### 10. Expanded the baseline coverage

- A larger English FLEURS baseline was run sequentially with `20` samples.
- Result:
  - `en_us WER = 27.23%`
- Reports written:
  - `reports/fleurs_en_us_limit20.json`
  - `reports/energy_fleurs_en_us_limit20.json`
- Measured values for the 20-sample English run:
  - `energy_joules: 13782.59`
  - `emissions_kg: 0.002731`

### 11. Captured a first multilingual spot check

- A Hindi FLEURS spot check was run with `5` samples.
- Result:
  - `hi_in WER = 27.64%`
- Report written:
  - `reports/fleurs_hi_in_limit5.json`

### 12. Fixed the local concurrent-request crash at the client layer

- The earlier engine crash was traced to overlapping transcription requests hitting the local
  `vllm` server at the same time.
- A cross-process transcription lock was added in:
  - `src/voxtral_project/api.py`
- Because all project transcription scripts route through this shared helper, the lock now
  serializes requests from:
  - `scripts/smoke_test_hf_sample.py`
  - `scripts/evaluate_fleurs.py`
  - `scripts/transcribe_file.py`
- Two separate concurrent smoke-test jobs were launched deliberately after the fix.
- Both requests completed successfully with the expected transcript, and the BF16 server remained
  healthy on port `8081`.

### 13. Diagnosed and fixed the empty-prediction evaluation gap

- Two English FLEURS samples in the earlier `20`-sample run were returning deterministic empty
  transcripts:
  - `id 1776`
  - `id 1972`
- These blanks were reproduced directly three times each against the live BF16 server.
- Audio diagnostics showed the failing clips were unusually quiet compared with the successful
  samples.
- A quiet-audio preparation step was added in:
  - `src/voxtral_project/audio.py`
- `scripts/evaluate_fleurs.py` was updated to:
  - boost low-level samples before WAV export
  - record per-sample audio diagnostics
  - count empty predictions explicitly in the report
- The quiet-audio-aware BF16 rerun improved the English `20`-sample result to:
  - `WER = 22.20%`
  - `empty_prediction_count = 0`
- New reports written:
  - `reports/fleurs_bf16_en_us_limit20_quietfix.json`
  - `reports/energy_fleurs_bf16_en_us_limit20_quietfix.json`
- Measured values for the rerun:
  - `elapsed_seconds: 46.26`
  - `energy_joules: 8112.90`
  - `emissions_kg: 0.001608`

### 14. Brought up the first working FP8 compression run

- The first `fp8_round1` launch failed at first because the local GPU memory budget was too tight.
- `configs/vllm/fp8_round1.yaml` was updated to match the practical local serving envelope:
  - `max_model_len: 8192`
  - `gpu_memory_utilization: 0.85`
- The BF16 server on `8081` was stopped temporarily to free the GPU for the compression run.
- After that adjustment, the FP8 server launched successfully on:
  - `http://localhost:8082/v1`

### 15. Completed the first BF16 vs FP8 comparison

- The same English FLEURS `20`-sample evaluation was run against the FP8 server using the same
  quiet-audio-aware evaluator.
- FP8 result:
  - `WER = 21.97%`
  - `empty_prediction_count = 0`
- New reports written:
  - `reports/fleurs_fp8_en_us_limit20_quietfix.json`
  - `reports/energy_fleurs_fp8_en_us_limit20_quietfix.json`
- FP8 measured values:
  - `elapsed_seconds: 35.21`
  - `energy_joules: 4952.89`
  - `emissions_kg: 0.000982`
- Practical comparison against the quiet-audio-aware BF16 rerun:
  - WER stayed effectively unchanged and was slightly better under FP8
  - elapsed time dropped by about `24%`
  - energy dropped by about `39%`
  - emissions also dropped materially
- This makes `fp8_round1` the first compression configuration that has shown a clear efficiency
  gain without an obvious quality regression on the local English spot check.

### 16. Refreshed the project docs to match the verified checkpoint

- `README.md` was updated to reflect the current best-known local state.
- The README now includes:
  - the quiet-audio-aware BF16 reference numbers
  - the first working FP8 comparison numbers
  - the current practical runtime constraints for this machine
  - the current FP8 launch and evaluation commands
- This leaves the repo docs aligned with the actual validated benchmark state before pushing the
  latest code and notes to GitHub.

## Important Findings From Today

- The BF16 Voxtral baseline is now serving successfully in WSL on the local machine.
- The BF16 baseline was strong enough to expose a real evaluation blind spot:
  - very quiet samples could collapse to empty transcripts
- The empty-prediction issue was not random concurrency noise.
- Quiet-audio boosting plus better diagnostics removed the empty predictions in the English
  `20`-sample rerun.
- The current local 16 GB GPU budget supports the baseline reliably at:
  - `max_model_len: 8192`
- The transcription path works end to end for single requests.
- The local project scripts now avoid the known concurrent-request crash by serializing
  transcription calls through a shared lock.
- The stronger English BF16 reference is now:
  - `WER = 22.20%` over `20` FLEURS test samples with `0` empty predictions
- The first Hindi multilingual spot check is also encouraging:
  - `WER = 27.64%` over `5` FLEURS test samples
- The first FP8 compression run is now working locally on the same model.
- On the English `20`-sample comparison, FP8 preserved quality while reducing elapsed time and
  energy materially.

## Current Working State

- BF16 baseline server can be launched successfully in WSL when needed.
- The model is being served from local files in:
  - `/mnt/c/Users/ASUS/Music/Fine_tuning/models/voxtral-realtime`
- The stable launch path is:
  - `scripts/start_vllm_server.sh`
- The client now uses:
  - `/v1/audio/transcriptions`
- The evaluator now includes quiet-audio preparation and per-sample audio diagnostics.
- The currently active local server is:
  - `http://localhost:8082/v1`
- The currently active compression config is:
  - `configs/vllm/fp8_round1.yaml`
- Baseline reports currently available:
  - `reports/smoke_test_transcript.txt`
  - `reports/fleurs_en_us_limit1.json`
  - `reports/fleurs_en_us_limit5.json`
  - `reports/fleurs_en_us_limit5_energy_run.json`
  - `reports/energy_fleurs_en_us_limit5.json`
  - `reports/fleurs_en_us_limit20.json`
  - `reports/energy_fleurs_en_us_limit20.json`
  - `reports/fleurs_hi_in_limit5.json`
  - `reports/fleurs_bf16_en_us_limit20_quietfix.json`
  - `reports/energy_fleurs_bf16_en_us_limit20_quietfix.json`
  - `reports/fleurs_fp8_en_us_limit20_quietfix.json`
  - `reports/energy_fleurs_fp8_en_us_limit20_quietfix.json`

## Recommended Next Step

Now that the first FP8 result looks promising locally, the next most useful steps are:

1. run the same FP8 comparison on `hi_in` and one additional language for multilingual confidence,
2. decide whether BF16 should be restarted for more reference runs or whether the session should stay on FP8,
3. if FP8 continues to hold quality, move to the next compression branch such as `gptq8_round1`,
4. compare future compressed runs against the quiet-audio-aware BF16 reference instead of the older empty-containing reports.
