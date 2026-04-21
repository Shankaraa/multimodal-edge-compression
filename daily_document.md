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

### 17. Extended the FP8 multilingual check to Hindi

- The live FP8 server on `http://localhost:8082/v1` was used for a Hindi FLEURS spot check with
  `5` samples.
- FP8 Hindi result:
  - `WER = 26.83%`
  - `empty_prediction_count = 0`
- New reports written:
  - `reports/fleurs_fp8_hi_in_limit5_quietfix.json`
  - `reports/energy_fleurs_fp8_hi_in_limit5_quietfix.json`
- Measured values:
  - `energy_joules: 1620.28`
  - `emissions_kg: 0.000321`
- Compared with the earlier BF16 Hindi spot check:
  - BF16 `WER = 27.64%`
- FP8 was slightly better on this small sample and still produced no empty predictions
- The next multilingual follow-up is still to run one additional non-English language through the
  same FP8 check for broader confidence.

### 18. Extended the FP8 multilingual check to French

- The same live FP8 server on `http://localhost:8082/v1` was used for a French FLEURS spot check
  with `5` samples.
- FP8 French result:
  - `WER = 23.18%`
  - `empty_prediction_count = 0`
- New reports written:
  - `reports/fleurs_fp8_fr_fr_limit5_quietfix.json`
  - `reports/energy_fleurs_fp8_fr_fr_limit5_quietfix.json`
- Measured values:
  - `energy_joules: 2121.87`
  - `emissions_kg: 0.000421`
- This gives us a third language datapoint for FP8 after English and Hindi, and it continued to
  avoid the earlier empty-prediction failure mode.

### 19. Started the GPTQ round-1 branch and found the current blocker

- `configs/vllm/gptq_round1.yaml` was first brought in line with the local working runtime budget:
  - `max_model_len: 8192`
  - `gpu_memory_utilization: 0.85`
- The active FP8 server was stopped temporarily so GPTQ could get the full GPU.
- A foreground launch probe was then run for:
  - `configs/vllm/gptq_round1.yaml`
  - port `8083`
- The launch failed early with a clear `vLLM` validation error:
  - `Cannot find the config file for gptq`
- This means the local Voxtral checkpoint is not a ready-to-serve GPTQ checkpoint in its current
  form.
- In practical terms, `vLLM` is not performing GPTQ quantization on the fly here; it expects a
  compatible GPTQ quantization config/checkpoint to already exist.
- After confirming the blocker, the working FP8 server was restored successfully on:
  - `http://localhost:8082/v1`
- So the project remains in a usable state, but `gptq8_round1` is currently blocked on preparing
  or obtaining GPTQ-formatted weights/configs rather than just changing a `vLLM` flag.

### 20. Investigated the practical GPTQ preparation path

- The local WSL runtime was checked for GPTQ tooling support.
- Current state:
  - installed: `vllm`, `compressed_tensors`, `transformers`
  - missing: `llmcompressor`, `gptqmodel`, `auto_gptq`
- `transformers.AutoConfig.from_pretrained(...)` still fails on the local Voxtral checkpoint
  because `voxtral_realtime` is not recognized by the installed Transformers build.
- This reinforces the key constraint already hinted at by the model card:
  - Voxtral Realtime is currently practical in `vLLM`
  - it is not yet a normal Hugging Face Transformers model in this environment
- The local checkpoint layout was inspected directly and confirms a clean selective-compression
  boundary:
  - protect `audio_tower.*`
  - protect `multi_modal_projector.*`
  - protect `language_model.model.embed_tokens.*`
  - target `language_model.model.layers.*`
- A dedicated investigation note was added at:
  - `docs/gptq_investigation.md`
- Current conclusion:
- standard GPTQ serving is blocked because we do not yet have GPTQ-formatted artifacts
- standard calibration-first GPTQ tooling is also awkward because Voxtral Realtime is still not
  directly loadable through Transformers here
- the most realistic follow-up is to investigate `llmcompressor` model-free compression as a
  research branch while keeping FP8 as the main practical submission path

### 21. Extended the FP8 mainline to Japanese and found a metric caveat

- The live FP8 server on `http://localhost:8082/v1` was used for a Japanese FLEURS spot check
  with `5` samples.
- Raw report result:
  - `WER = 100.00%`
  - `empty_prediction_count = 0`
- New reports written:
  - `reports/fleurs_fp8_ja_jp_limit5_quietfix.json`
  - `reports/energy_fleurs_fp8_ja_jp_limit5_quietfix.json`
- Measured values:
  - `energy_joules: 2632.10`
  - `emissions_kg: 0.000522`
- The model outputs were non-empty and clearly Japanese, so this did not look like a total model
  failure.
- The likely issue is metric mismatch:
  - the FLEURS Japanese references contain spaces
  - the generated Japanese predictions are mostly unsegmented
  - standard word-based WER therefore becomes misleading
- A spacing-agnostic character-level check on the same `5` Japanese samples gave:
  - `CER ≈ 10.0%`
- So the Japanese run exposed a scoring caveat rather than a serving or empty-output failure.

### 22. Added a compact FP8 benchmark summary artifact

- A dedicated summary document was added at:
  - `docs/fp8_benchmark_summary.md`
- It captures:
  - the BF16 quiet-audio-aware English reference
  - the FP8 English comparison
  - the FP8 Hindi, French, and Japanese spot checks
  - the current Japanese metric caveat
- This gives the mainline track a reusable benchmark snapshot for future reporting and submission
  framing.

### 23. Added CER-aware scoring support for multilingual evaluation

- The Japanese FP8 spot check showed that raw word-based WER can be misleading for CJK-style text
  when references contain spaces but predictions are mostly unsegmented.
- `scripts/evaluate_fleurs.py` was updated to record:
  - `cer`
  - `cer_percent`
  - `cer_no_whitespace`
  - `cer_no_whitespace_percent`
- The Japanese FP8 run was then repeated with the updated evaluator.
- Updated Japanese result:
  - `WER = 100.00%`
  - `CER = 10.42%`
  - `CER(no-space) = 10.00%`
  - `empty_prediction_count = 0`
- New updated reports written:
  - `reports/fleurs_fp8_ja_jp_limit5_quietfix_v2.json`
  - `reports/energy_fleurs_fp8_ja_jp_limit5_quietfix_v2.json`
- This makes the Japanese result much more interpretable and gives the mainline track a better
  multilingual scoring story going forward.

### 24. Built the first submission-readiness docs around the FP8 mainline

- The mainline path is now explicitly framed as the first submission candidate.
- New submission-facing docs were added:
  - `docs/submission_candidate_summary.md`
  - `docs/submission_readiness_checklist.md`
- `README.md` was updated to point directly at:
  - the submission candidate summary
  - the submission readiness checklist
  - the submission benchmark table
- This means the repo now has:
  - an experimental track view
  - a benchmark-summary view
  - and a first-submission readiness view
- That is the right shape for a base acceptance-oriented submission path.

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
- The first FP8 Hindi spot check is also encouraging:
  - `WER = 26.83%` over `5` samples with `0` empty predictions
- The first FP8 French spot check is also encouraging:
  - `WER = 23.18%` over `5` samples with `0` empty predictions
- The next compression branch, `gptq8_round1`, is not immediately runnable from the current local
  checkpoint because the required GPTQ config/checkpoint artifacts are missing.
- GPTQ is now better understood as a preparation problem, not a simple `vLLM` runtime flag.
- FP8 is now holding up across English, Hindi, French, and a Japanese spot check, though Japanese
  needs better scoring treatment than raw WER.
- The evaluator now supports CER-aware reporting, which makes future CJK-style evaluations much
  more honest.
- The FP8 path is now not only benchmarked, but also explicitly packaged as the first submission
  candidate.

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
- `configs/vllm/gptq_round1.yaml` is now tuned to the local memory envelope, but the branch is
  still blocked by missing GPTQ artifacts.
- The GPTQ investigation summary now lives in:
  - `docs/gptq_investigation.md`
- The compact FP8 benchmark summary now lives in:
  - `docs/fp8_benchmark_summary.md`
- The submission-readiness docs now live in:
  - `docs/submission_candidate_summary.md`
  - `docs/submission_readiness_checklist.md`
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
  - `reports/fleurs_fp8_hi_in_limit5_quietfix.json`
  - `reports/energy_fleurs_fp8_hi_in_limit5_quietfix.json`
  - `reports/fleurs_fp8_fr_fr_limit5_quietfix.json`
  - `reports/energy_fleurs_fp8_fr_fr_limit5_quietfix.json`

## Recommended Next Step

Now that the first FP8 result looks promising locally, the next most useful steps are:

1. polish the actual first-submission narrative using the new FP8 submission docs,
2. keep using FP8 as the current working compression baseline until GPTQ-compatible artifacts exist,
3. compare future compressed runs against the quiet-audio-aware BF16 reference instead of the older empty-containing reports,
4. expand multilingual FP8 coverage further only if we still need more submission confidence.

---

## Date

April 20, 2026

## Project

Voxtral Real-Time 4B compression project for the Resilient AI Challenge audio-to-text track.

## Objective For Today

Anchor the current FP8 submission path against a serious external ASR baseline and fix the
metric framing so public benchmark comparisons are not misleading.

## What Was Done Today

### 1. Added an external-baseline backend to the FLEURS evaluator

- `scripts/evaluate_fleurs.py` was extended to support two interchangeable transcription
  backends:
  - `vllm_api`
  - `whisper_transformers`
- A new shared backend module was added:
  - `src/voxtral_project/asr.py`
- This means the same evaluation script can now benchmark:
  - local Voxtral through the `vLLM` API
  - local Hugging Face Whisper models through Transformers
- That matters because it keeps:
  - dataset selection
  - quiet-audio preparation
  - metric computation
  - report shape
  identical across the comparison.

### 2. Ran the first real external baseline probe with Whisper large-v3

- The WSL GPU environment was verified usable for the comparison:
  - Torch CUDA available: `True`
  - `transformers` already installed
- A first one-sample probe with:
  - `openai/whisper-large-v3`
  succeeded and downloaded the model into the WSL environment.
- After that, a full same-slice English run was executed on:
  - `google/fleurs`
  - `en_us`
  - `limit 20`
- New reports written:
  - `reports/fleurs_whisper_large_v3_en_us_limit20.json`
  - `reports/energy_fleurs_whisper_large_v3_en_us_limit20.json`
- Measured Whisper result on the local slice:
  - raw `WER = 20.59%`
  - raw `CER = 5.13%`
  - `0` empty predictions
  - `elapsed_seconds = 34.77`
  - `energy_joules = 3258.57`

### 3. Found and corrected a benchmark-framing mistake in our earlier comparison logic

- The earlier local WER comparisons were using raw string WER directly on:
  - lowercase punctuation-light FLEURS references
  - punctuated and capitalized model predictions
- That is internally consistent, but it is not a fair public benchmark comparison by itself.
- A new text-metric helper was added:
  - `src/voxtral_project/text.py`
- It now computes normalized ASR metrics using:
  - Unicode NFKC normalization
  - casefolding
  - punctuation and symbol stripping
  - control removal
  - whitespace collapsing
- `scripts/evaluate_fleurs.py` was updated to record:
  - raw WER and CER
  - raw whitespace-insensitive CER
  - normalized WER and CER
  - normalized whitespace-insensitive CER
- A new utility script was added:
  - `scripts/recompute_report_metrics.py`
- That utility was then used to recompute the English BF16 and FP8 report metrics in place so the
  old reports now carry normalized metrics too.

### 4. Produced the first honest same-slice Voxtral vs Whisper comparison

- After normalized metrics were added, the current English `20`-sample comparison now reads:
  - BF16 quietfix:
    - raw `WER = 22.20%`
    - normalized `WER = 6.36%`
    - `46.26 s`
    - `8112.90 J`
  - FP8 round 1:
    - raw `WER = 21.97%`
    - normalized `WER = 6.36%`
    - `35.21 s`
    - `4952.89 J`
  - Whisper large-v3:
    - raw `WER = 20.59%`
    - normalized `WER = 4.32%`
    - `34.77 s`
    - `3258.57 J`
- The key conclusion is:
  - FP8 is still clearly better than our BF16 Voxtral reference on efficiency
  - but Whisper large-v3 currently beats our local Voxtral setup on the same normalized English
    slice

### 5. Updated benchmark and submission docs with the corrected framing

- New note added:
  - `docs/global_benchmark_comparison.md`
- Updated docs:
  - `docs/submission_benchmark_table.md`
  - `docs/submission_candidate_summary.md`
  - `docs/fp8_benchmark_summary.md`
- These updates now distinguish clearly between:
  - the best compressed Voxtral path
  - and the strongest external same-slice baseline we have tested

### 6. Extended the external comparison to French and Hindi spot checks

- Whisper large-v3 was also run on:
  - `fr_fr limit5`
  - `hi_in limit5`
- New reports written:
  - `reports/fleurs_whisper_large_v3_fr_fr_limit5.json`
  - `reports/energy_fleurs_whisper_large_v3_fr_fr_limit5.json`
  - `reports/fleurs_whisper_large_v3_hi_in_limit5.json`
  - `reports/energy_fleurs_whisper_large_v3_hi_in_limit5.json`
- French Whisper result:
  - raw `WER = 21.85%`
  - normalized `WER = 8.07%`
  - `energy_joules = 3605.36`
- Hindi Whisper result:
  - raw `WER = 32.52%`
  - normalized `WER = 28.46%`
  - `energy_joules = 4679.43`
- Existing FP8 French and Hindi reports were recomputed with normalized metrics too so the
  spot-check comparisons are now fairer.

## Important Findings From Today

- The right comparison question was not "is our raw WER good?"
- The right comparison question is:
  - does FP8 beat BF16 within the Voxtral track
  - and how far are we from a strong external baseline under a fairer normalized metric
- Once normalized metrics are used, the local English picture is much clearer:
  - BF16 normalized `WER = 6.36%`
  - FP8 normalized `WER = 6.36%`
  - Whisper large-v3 normalized `WER = 4.32%`
- So the current FP8 path is a real compression success inside the Voxtral track, but it is not
  yet beating the strongest external baseline we checked.
- The multilingual external picture is mixed:
  - Whisper is ahead on the current French spot check
  - FP8 is ahead on the current Hindi spot check
- This is exactly the kind of finding we wanted early, because it sharpens the submission story
  instead of letting us over-claim.

## Current Working State

- The active serving path for compressed Voxtral remains:
  - `http://localhost:8082/v1`
- The current mainline compression config remains:
  - `configs/vllm/fp8_round1.yaml`
- The evaluator can now benchmark both:
  - Voxtral through `vLLM`
  - Whisper through Transformers
- The English comparison now has three useful report pairs:
  - `reports/fleurs_bf16_en_us_limit20_quietfix.json`
  - `reports/energy_fleurs_bf16_en_us_limit20_quietfix.json`
  - `reports/fleurs_fp8_en_us_limit20_quietfix.json`
  - `reports/energy_fleurs_fp8_en_us_limit20_quietfix.json`
  - `reports/fleurs_whisper_large_v3_en_us_limit20.json`
  - `reports/energy_fleurs_whisper_large_v3_en_us_limit20.json`
- The external spot-check comparison also now includes:
  - `reports/fleurs_whisper_large_v3_fr_fr_limit5.json`
  - `reports/energy_fleurs_whisper_large_v3_fr_fr_limit5.json`
  - `reports/fleurs_whisper_large_v3_hi_in_limit5.json`
  - `reports/energy_fleurs_whisper_large_v3_hi_in_limit5.json`
- Benchmark framing docs now include:
  - `docs/global_benchmark_comparison.md`
  - `docs/submission_benchmark_table.md`
  - `docs/submission_candidate_summary.md`

## Recommended Next Step

1. run Whisper on at least one more language from the current spot-check set so the external
   comparison is not English-only,
2. investigate why the local Voxtral setup still trails the published Voxtral model-card English
   numbers,
3. keep FP8 as the first submission path, but describe it honestly as the best compressed Voxtral
   path rather than a global ASR leader.

---

## Date

April 21, 2026

## Project

Voxtral Real-Time 4B compression project for the Resilient AI Challenge audio-to-text track.

## Objective For Today

Sync the completed global-benchmark comparison work into the repo cleanly and push only the
submission-relevant code and docs.

## What Was Done Today

### 1. Prepared the benchmark-comparison changes for a clean Git push

- Confirmed that the global-benchmark comparison work from April 20, 2026 is now documented in:
  - `README.md`
  - `daily_document.md`
  - `docs/global_benchmark_comparison.md`
  - `docs/submission_benchmark_table.md`
  - `docs/submission_candidate_summary.md`
  - `docs/fp8_benchmark_summary.md`
  - `scripts/evaluate_fleurs.py`
  - `scripts/recompute_report_metrics.py`
  - `src/voxtral_project/asr.py`
  - `src/voxtral_project/text.py`
- Confirmed that unrelated local GPTQ-side research files remain separate and should not be mixed
  into this push unless we explicitly decide to do that later.

## Important Findings From Today

- The benchmark-comparison work is now ready to be pushed as a coherent update.
- The right push for today is a selective one:
  - include the global-benchmark evaluation and submission-framing updates
  - exclude unrelated GPTQ-side experiments that are still local-only

## Recommended Next Step

1. push the benchmark-comparison update to `main`,
2. continue the FP8 submission track with one more carefully chosen external comparison only if it
   changes the submission story,
3. keep GPTQ work isolated until it is ready to stand on its own.
