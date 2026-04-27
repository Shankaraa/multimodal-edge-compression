# Daily Document

Compacted chronological log of the Voxtral Real-Time 4B compression project for the Resilient AI
Challenge audio-to-text track. Repetitive framing was removed; distinct facts, results, decisions,
blockers, file paths, and next steps were retained.

## April 15, 2026

- Objective: set up the workspace, understand the compression guide, prepare the baseline
  workflow, and bootstrap WSL so the BF16 baseline could run on the local RTX 5080 before
  compression experiments.
- Guide review: `voxtral realtime compression guide.pdf` was inspected directly; it is a 26-page
  guide with 23 content pages plus trailing PDF structure pages. Core strategy: protect the audio
  encoder, leave the adapter mostly untouched, compress the decoder first, use `vLLM`, and
  optimize for energy while preserving WER.
- Project scaffold created: `README.md`, `.gitignore`, `requirements.txt`,
  `requirements-linux-gpu.txt`, `docs/guide_notes.md`, `configs/experiments.yaml`,
  `configs/vllm/bf16.yaml`, `configs/vllm/fp8_round1.yaml`,
  `configs/vllm/gptq_round1.yaml`, `configs/vllm/gptq_round2.yaml`,
  `configs/vllm/aggressive_round2.yaml`, `src/voxtral_project/__init__.py`,
  `src/voxtral_project/audio.py`, `src/voxtral_project/api.py`,
  `scripts/download_model.py`, `scripts/serve_model.py`, `scripts/transcribe_file.py`,
  `scripts/evaluate_fleurs.py`, `scripts/measure_energy.py`.
- Starter workflow established: model download, `vLLM` launch from YAML, local audio
  transcription, FLEURS WER evaluation, and CodeCarbon energy measurement.
- Windows environment: `.venv` created and repaired after an `ensurepip`/missing-`pip` issue;
  base dependencies from `requirements.txt` installed; entrypoints verified with `--help`.
- Local hardware confirmed from Windows `nvidia-smi`: GPU `NVIDIA GeForce RTX 5080`, driver
  `581.95`, CUDA `13.0`, about `16 GB` VRAM.
- WSL runtime confirmed: distro `Ubuntu-22.04`, WSL `2`, workspace reachable at
  `/mnt/c/Users/ASUS/Music/Fine_tuning`.
- WSL GPU visibility confirmed: `/usr/bin/python3` available, `nvidia-smi` visible, Torch later
  confirmed CUDA.
- WSL system packages installed: `build-essential`, `python3-pip`, `python3-venv`, `ffmpeg`,
  `libsndfile1`.
- WSL baseline virtual environment created at `~/.venvs/voxtral-baseline`; installed upgraded
  `pip`, `setuptools`, `wheel`, packages from `requirements.txt`,
  `mistral_common[audio]`, `transformers`, and `vllm` nightly.
- WSL baseline environment verified: `vllm` import worked, `transformers` import worked, Hugging
  Face access to `mistralai/Voxtral-Mini-4B-Realtime-2602` worked, Torch saw CUDA and
  `NVIDIA GeForce RTX 5080`.
- Model download resumed; duplicate background WSL downloads were contending on
  `model.safetensors.lock` and `consolidated.safetensors.lock`. Duplicate jobs were stopped;
  resumable partial files preserved, including files around `5.03 GB` and `595 MB`.
- Post-download helper scripts added: `scripts/check_vllm_server.py`,
  `scripts/smoke_test_hf_sample.py`, `scripts/start_wsl_baseline.ps1`. All `vLLM` YAML configs
  were aligned to `served_model_name: voxtral-realtime`.
- GitHub remote updated to `https://github.com/Shankaraa/multimodal-edge-compression.git`; local
  branch `shankara` created; scaffold and helper scripts committed and pushed after GitHub CLI
  auth in WSL. `.gitignore` excludes large artifacts including `*.pt`, `*.pth`, `*.bin`,
  `*.safetensors`.
- Model download completed. Verified weight files: `consolidated.safetensors`,
  `model.safetensors`. Directory size about `17 GB`. Model paths:
  `C:\Users\ASUS\Music\Fine_tuning\models\voxtral-realtime` and
  `/mnt/c/Users/ASUS/Music/Fine_tuning/models/voxtral-realtime`.
- BF16 baseline launch attempted and blocked by runtime mismatch. Torch in WSL reported
  `2.11.0` / CUDA `13.0`; installed `vllm` was `0.19.1rc1.dev300+g29e5d1020`; server failed
  loading `libcudart.so.12`. Result: model download complete, baseline launch blocked until
  `vllm` matches CUDA 13.
- Important findings: project structure ready; WSL is the right runtime; model download is done;
  blocker is CUDA build mismatch in WSL `vllm`; guide strongly supports decoder-first compression
  and warns against aggressive encoder quantization.
- Prepared commands:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "source ~/.venvs/voxtral-baseline/bin/activate && cd /mnt/c/Users/ASUS/Music/Fine_tuning && python scripts/download_model.py --local-dir models/voxtral-realtime"
wsl -d Ubuntu-22.04 -- bash -lc "source ~/.venvs/voxtral-baseline/bin/activate && cd /mnt/c/Users/ASUS/Music/Fine_tuning && python scripts/serve_model.py models/voxtral-realtime --config configs/vllm/bf16.yaml"
wsl -d Ubuntu-22.04 -- bash -lc "source ~/.venvs/voxtral-baseline/bin/activate && huggingface-cli download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir /mnt/c/Users/ASUS/Music/Fine_tuning/models/voxtral-realtime --local-dir-use-symlinks False"
wsl -d Ubuntu-22.04 -- bash -lc "source ~/.venvs/voxtral-baseline/bin/activate && hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir /mnt/c/Users/ASUS/Music/Fine_tuning/models/voxtral-realtime"
powershell -ExecutionPolicy Bypass -File .\scripts\start_wsl_baseline.ps1
.\.venv\Scripts\python.exe scripts\check_vllm_server.py --base-url http://localhost:8080/v1
.\.venv\Scripts\python.exe scripts\smoke_test_hf_sample.py --model voxtral-realtime --out reports/smoke_test_transcript.txt
```

- End-of-day state: scaffold done, Windows `.venv` repaired, WSL runtime ready, model downloaded,
  helper scripts added, GitHub branch pushed, BF16 launch blocked by `libcudart.so.12` vs CUDA
  `13.0`, and no transcription/eval/compression run completed yet.
- Recommended next step: repair WSL `vllm` for CUDA 13, then start BF16, wait for readiness, run
  smoke transcription, run small English FLEURS eval, and capture first baseline WER and energy.

## April 16, 2026

- Objective: get BF16 fully serving in WSL, make transcription work end to end, and capture the
  first real baseline WER and energy numbers.
- CUDA/runtime mismatch fixed by installing CUDA 13 compatible nightly `vllm` in WSL. Verified
  stack: `torch 2.11.0+cu130`, `vllm 0.19.1rc1.dev302+g68be0f853.cu130`.
- Local server launch hardened: `scripts/serve_model.py` now injects WSL venv Torch/NVIDIA shared
  library paths into `LD_LIBRARY_PATH`; `scripts/start_vllm_server.sh` added for stable WSL
  launch. Port `8080` was occupied, so the working BF16 server moved to `http://localhost:8081/v1`.
- BF16 config reduced from `max_model_len: 16384` to `8192` so KV-cache initialization would fit
  on the local 16 GB GPU.
- Live transcription path fixed: installed missing WSL dependencies `av` and later `librosa`.
  Client switched from chat-completions multimodal path to `/v1/audio/transcriptions`.
- Evaluation compatibility fixed: `datetime.UTC` replaced with `datetime.now(timezone.utc)`;
  WSL `datasets` downgraded from `4.8.4` to `3.6.0`; repo requirement updated to
  `datasets>=2.18.0,<4`; FLEURS loading updated with `trust_remote_code=True`.
- First successful smoke transcription captured at `http://localhost:8081/v1`; saved to
  `reports/smoke_test_transcript.txt`. Example output:
  `Yesterday it was 35 degrees in Barcelona, but today the temperature will go down to minus 20 degrees.`
- Concurrency-related engine crash identified: overlapping transcription requests caused tensor
  shape mismatch errors in the Voxtral realtime path. Sequential requests worked; temporary rule
  was one audio request at a time.
- First English baseline WER results:
  - `en_us limit1`: `WER = 21.05%`, report `reports/fleurs_en_us_limit1.json`
  - `en_us limit5`: `WER = 34.95%`, report `reports/fleurs_en_us_limit5.json`
- First energy measurement captured on `en_us limit5`: reports
  `reports/energy_fleurs_en_us_limit5.json` and `reports/fleurs_en_us_limit5_energy_run.json`;
  `energy_joules = 4775.58`, `emissions_kg = 0.000946`.
- Larger English BF16 baseline run:
  - `en_us limit20`: `WER = 27.23%`
  - reports: `reports/fleurs_en_us_limit20.json`,
    `reports/energy_fleurs_en_us_limit20.json`
  - energy: `13782.59 J`, emissions `0.002731 kg`
- First multilingual BF16 spot check:
  - `hi_in limit5`: `WER = 27.64%`, report `reports/fleurs_hi_in_limit5.json`
- Client-layer fix for concurrency crash: cross-process transcription lock added in
  `src/voxtral_project/api.py`, covering `scripts/smoke_test_hf_sample.py`,
  `scripts/evaluate_fleurs.py`, and `scripts/transcribe_file.py`. Deliberate concurrent smoke
  tests then completed successfully while BF16 stayed healthy on `8081`.
- Empty-prediction gap diagnosed and fixed: `en_us limit20` samples `1776` and `1972` produced
  deterministic blank transcripts on the live BF16 server. Audio diagnostics showed unusually
  quiet clips. Added quiet-audio preparation in `src/voxtral_project/audio.py`; updated
  `scripts/evaluate_fleurs.py` to boost low-level samples, record audio diagnostics, and count
  empties explicitly. Quiet-audio-aware BF16 rerun:
  - `WER = 22.20%`
  - `empty_prediction_count = 0`
  - reports: `reports/fleurs_bf16_en_us_limit20_quietfix.json`,
    `reports/energy_fleurs_bf16_en_us_limit20_quietfix.json`
  - elapsed `46.26 s`, energy `8112.90 J`, emissions `0.001608 kg`
- First working FP8 compression run: `configs/vllm/fp8_round1.yaml` tuned to local envelope with
  `max_model_len: 8192`, `gpu_memory_utilization: 0.85`; BF16 server stopped temporarily; FP8
  launched successfully on `http://localhost:8082/v1`.
- First BF16 vs FP8 comparison on the same quiet-audio-aware `en_us limit20` slice:
  - FP8 `WER = 21.97%`, `empty_prediction_count = 0`
  - reports: `reports/fleurs_fp8_en_us_limit20_quietfix.json`,
    `reports/energy_fleurs_fp8_en_us_limit20_quietfix.json`
  - elapsed `35.21 s`, energy `4952.89 J`, emissions `0.000982 kg`
  - interpretation: WER effectively unchanged and slightly better under FP8; elapsed about `24%`
    lower; energy about `39%` lower; first compression configuration with clear efficiency gain
    and no obvious quality regression on the local English slice.
- Docs refreshed to match the verified checkpoint: `README.md` now reflects the quiet-audio-aware
  BF16 reference, the first working FP8 comparison, the local runtime constraints, and launch/eval
  commands.
- FP8 multilingual checks:
  - `hi_in limit5`: `WER = 26.83%`, `empty_prediction_count = 0`,
    energy `1620.28 J`, emissions `0.000321 kg`, reports
    `reports/fleurs_fp8_hi_in_limit5_quietfix.json`,
    `reports/energy_fleurs_fp8_hi_in_limit5_quietfix.json`
  - `fr_fr limit5`: `WER = 23.18%`, `empty_prediction_count = 0`,
    energy `2121.87 J`, emissions `0.000421 kg`, reports
    `reports/fleurs_fp8_fr_fr_limit5_quietfix.json`,
    `reports/energy_fleurs_fp8_fr_fr_limit5_quietfix.json`
- GPTQ round-1 branch started. `configs/vllm/gptq_round1.yaml` aligned to local memory envelope
  (`max_model_len: 8192`, `gpu_memory_utilization: 0.85`); FP8 server stopped temporarily; probe
  launch on port `8083` failed with `Cannot find the config file for gptq`. Conclusion: current
  local checkpoint is not a ready-to-serve GPTQ checkpoint; `vLLM` expects GPTQ artifacts to
  already exist. FP8 server restored on `http://localhost:8082/v1`.
- Practical GPTQ path investigated:
  - present packages: `vllm`, `compressed_tensors`, `transformers`
  - missing packages: `llmcompressor`, `gptqmodel`, `auto_gptq`
  - `transformers.AutoConfig.from_pretrained(...)` still failed on `voxtral_realtime`
  - protected boundaries confirmed: `audio_tower.*`, `multi_modal_projector.*`,
    `language_model.model.embed_tokens.*`
  - primary quantization target confirmed: `language_model.model.layers.*`
  - investigation note added: `docs/gptq_investigation.md`
  - conclusion: standard GPTQ serving blocked by missing GPTQ artifacts; calibration-first GPTQ
    also awkward because Voxtral Realtime is not directly loadable in Transformers here; most
    realistic follow-up is `llmcompressor` model-free compression as research while FP8 remains the
    practical path.
- Japanese FP8 spot check:
  - initial run on `ja_jp limit5`: raw `WER = 100.00%`, `empty_prediction_count = 0`,
    reports `reports/fleurs_fp8_ja_jp_limit5_quietfix.json`,
    `reports/energy_fleurs_fp8_ja_jp_limit5_quietfix.json`,
    energy `2632.10 J`, emissions `0.000522 kg`
  - outputs were non-empty and clearly Japanese; issue was metric mismatch due spaced references
    and mostly unsegmented predictions
  - initial spacing-agnostic character check showed `CER ≈ 10.0%`
  - evaluator extended to record `cer`, `cer_percent`, `cer_no_whitespace`,
    `cer_no_whitespace_percent`
  - rerun produced:
    - `WER = 100.00%`
    - `CER = 10.42%`
    - `CER(no-space) = 10.00%`
    - `empty_prediction_count = 0`
    - reports `reports/fleurs_fp8_ja_jp_limit5_quietfix_v2.json`,
      `reports/energy_fleurs_fp8_ja_jp_limit5_quietfix_v2.json`
- Compact benchmark and submission docs added:
  - `docs/fp8_benchmark_summary.md`
  - `docs/submission_candidate_summary.md`
  - `docs/submission_readiness_checklist.md`
  - `README.md` updated to point to the submission benchmark/candidate/checklist docs
- Important findings: BF16 baseline now serves in WSL; empty-prediction issue came from quiet
  audio rather than randomness; quiet-audio boosting fixed it; stable local envelope is
  `max_model_len: 8192`; single-request path works; scripts now serialize requests; trusted BF16
  English reference is `22.20%` WER over `20` samples with `0` empties; first FP8 English/Hindi/
  French checks are encouraging; GPTQ is a preparation problem, not a runtime flag; FP8 is now
  holding across English/Hindi/French/Japanese with CER-aware Japanese scoring; FP8 is the first
  submission candidate.
- Current state: BF16 launch path exists; model path
  `/mnt/c/Users/ASUS/Music/Fine_tuning/models/voxtral-realtime`; stable launcher
  `scripts/start_vllm_server.sh`; client uses `/v1/audio/transcriptions`; evaluator includes
  quiet-audio prep and per-sample diagnostics; active server `http://localhost:8082/v1`; active
  config `configs/vllm/fp8_round1.yaml`; GPTQ branch still blocked; key docs now include
  `docs/gptq_investigation.md`, `docs/fp8_benchmark_summary.md`,
  `docs/submission_candidate_summary.md`, `docs/submission_readiness_checklist.md`; reports include
  the smoke test, early English/Hindi baselines, quietfix BF16, FP8 English/Hindi/French, and
  Japanese v1/v2 runs.
- Recommended next step: polish the first-submission narrative, keep FP8 as the working baseline
  until GPTQ-compatible artifacts exist, compare future compressed runs against the quiet-audio
  BF16 reference, and expand multilingual FP8 coverage only if needed for submission confidence.

## April 20, 2026

- Objective: anchor the FP8 submission path against a serious external ASR baseline and fix the
  benchmark framing.
- Evaluator extended for external baselines:
  - `scripts/evaluate_fleurs.py` now supports `vllm_api` and `whisper_transformers`
  - `src/voxtral_project/asr.py` added to unify backends
  - same dataset selection, quiet-audio prep, metric computation, and report shape now apply to
    Voxtral via `vLLM` and Whisper via Transformers
- First external baseline probe with `openai/whisper-large-v3`:
  - WSL GPU environment confirmed usable (`torch.cuda.is_available() == True`,
    `transformers` already installed)
  - one-sample probe succeeded and downloaded the model
  - full same-slice English run on `google/fleurs`, `en_us`, `limit20`
  - reports: `reports/fleurs_whisper_large_v3_en_us_limit20.json`,
    `reports/energy_fleurs_whisper_large_v3_en_us_limit20.json`
  - result: raw `WER = 20.59%`, raw `CER = 5.13%`, `0` empty predictions,
    elapsed `34.77 s`, energy `3258.57 J`
- Benchmark-framing correction:
  - earlier comparisons used raw string WER on lowercase punctuation-light FLEURS references vs
    punctuated/capitalized predictions; internally consistent but not a fair public comparison
  - new metric helper added: `src/voxtral_project/text.py`
  - normalized metrics now use Unicode NFKC, casefolding, punctuation/symbol stripping, control
    removal, whitespace collapse
  - `scripts/evaluate_fleurs.py` now records raw and normalized WER/CER plus whitespace-insensitive
    variants
  - `scripts/recompute_report_metrics.py` added and used to recompute BF16/FP8 English reports
- First honest same-slice comparison after normalized metrics:
  - BF16 quietfix: raw `WER = 22.20%`, normalized `WER = 6.36%`, `46.26 s`, `8112.90 J`
  - FP8 round 1: raw `WER = 21.97%`, normalized `WER = 6.36%`, `35.21 s`, `4952.89 J`
  - Whisper large-v3: raw `WER = 20.59%`, normalized `WER = 4.32%`, `34.77 s`, `3258.57 J`
  - conclusion: FP8 remains clearly better than BF16 inside the Voxtral track on efficiency, but
    Whisper currently beats local Voxtral on the same normalized English slice
- Benchmark and submission docs updated:
  - new note `docs/global_benchmark_comparison.md`
  - updated `docs/submission_benchmark_table.md`,
    `docs/submission_candidate_summary.md`, `docs/fp8_benchmark_summary.md`
- External French/Hindi spot checks with Whisper:
  - `fr_fr limit5`: raw `WER = 21.85%`, normalized `WER = 8.07%`, energy `3605.36 J`
  - `hi_in limit5`: raw `WER = 32.52%`, normalized `WER = 28.46%`, energy `4679.43 J`
  - reports: `reports/fleurs_whisper_large_v3_fr_fr_limit5.json`,
    `reports/energy_fleurs_whisper_large_v3_fr_fr_limit5.json`,
    `reports/fleurs_whisper_large_v3_hi_in_limit5.json`,
    `reports/energy_fleurs_whisper_large_v3_hi_in_limit5.json`
  - existing FP8 French/Hindi reports recomputed with normalized metrics for fairer comparison
- Key findings: the right question is whether FP8 beats BF16 inside the Voxtral track and how far
  it is from a strong external baseline under normalized metrics; normalized English picture is
  BF16 `6.36%`, FP8 `6.36%`, Whisper `4.32%`; current FP8 path is a real compression success
  inside the Voxtral track, but it is not beating the strongest external baseline tested; external
  multilingual picture is mixed, with Whisper ahead on French and FP8 ahead on Hindi.
- Current state: active compressed serving path still `http://localhost:8082/v1`,
  config `configs/vllm/fp8_round1.yaml`, evaluator now benchmarks Voxtral and Whisper, English
  comparison now has BF16/FP8/Whisper report pairs, and docs now include benchmark-comparison and
  submission-framing notes.
- Recommended next step: add one more external comparison if it changes the submission story,
  investigate why local Voxtral still trails published Voxtral English numbers, and keep FP8 as
  the first submission path but describe it as the best compressed Voxtral path rather than a
  global ASR leader.

## April 21, 2026

- Objective: sync the completed benchmark-comparison work cleanly and push only the
  submission-relevant code/docs, while continuing high-signal benchmark and research follow-up.
- Prepared global-benchmark comparison changes for a clean push. Confirmed updates exist in
  `README.md`, `daily_document.md`, `docs/global_benchmark_comparison.md`,
  `docs/submission_benchmark_table.md`, `docs/submission_candidate_summary.md`,
  `docs/fp8_benchmark_summary.md`, `scripts/evaluate_fleurs.py`,
  `scripts/recompute_report_metrics.py`, `src/voxtral_project/asr.py`,
  `src/voxtral_project/text.py`. Unrelated GPTQ-side research stayed separate.
- First end-to-end GPTQ-side smoke test completed on narrowed artifact
  `models/voxtral-realtime-llmcompressor-consolidated-fp8dynamic-noada-test` using tokenizer
  `models/voxtral-realtime`, tokenizer mode `mistral`, config
  `configs/vllm/compressed_tensors_probe.yaml`, and reproducible helper
  `scripts/run_model_free_ptq.py`. It cleared earlier tokenizer/config and
  `ada_rms_norm_t_cond.*` compressed-weight blockers, booted on `http://127.0.0.1:8085`,
  exposed `/v1/models` as `voxtral-realtime-llmcompressor-probe`, and accepted a real request.
  The smoke input was a generated 1-second tone WAV, so the empty transcript was expected.
- Apples-to-apples GPTQ-side mini benchmark added through `scripts/benchmark_vllm_variant.py`.
  Summaries:
  `reports/benchmark_bf16_miniqf_en_us_limit5.json`,
  `reports/benchmark_fp8_round1_miniqf_en_us_limit5.json`,
  `reports/benchmark_ct_noada_miniqf_en_us_limit5.json`.
  Results:
  - BF16: startup `148.05 s`, first request `2.53 s`, normalized `WER 4.81%`,
    elapsed `18.54 s`, energy `2793.12 J`
  - FP8 round 1: startup `132.09 s`, first request `1.71 s`, normalized `WER 4.81%`,
    elapsed `14.69 s`, energy `1891.11 J`
  - narrowed compressed-tensors artifact: startup `93.83 s`, first request `10.14 s`,
    normalized `WER 100.00%`, `5/5` empty predictions, elapsed `59.55 s`, energy `8825.33 J`
  - conclusion: artifact boots and serves but ASR is functionally broken; startup/memory gains are
    irrelevant; repeated `empty multimodal embeddings` warnings line up with the all-empty speech
    outputs
- Push decision after the mini benchmark: benchmark-comparison update is ready to push; GPTQ-side
  branch is not benchmark-worthy for submission; FP8 remains the only serious compressed candidate.
- Published-gap hypothesis investigated on the FP8 mainline. Added explicit decode controls for the
  `vLLM` path (`temperature`, language hint mode) in
  `src/voxtral_project/api.py`, `src/voxtral_project/asr.py`,
  `scripts/evaluate_fleurs.py`, `scripts/transcribe_file.py`,
  `scripts/smoke_test_hf_sample.py`, `scripts/benchmark_vllm_variant.py`.
- Decode-side explanation ruled out:
  - `temperature = 0.0` on `en_us limit20` gave raw `WER = 22.43%`, normalized `WER = 7.05%`
  - `temperature = 0.0` plus `language_hint_mode = fleurs_primary` was identical
  - current default path remained raw `WER = 21.97%`, normalized `WER = 6.36%`
  - conclusion: neither missing `temperature = 0.0` nor explicit English hint explains the gap
- Larger English slice showed sample size and comparison frame matter more:
  - `en_us limit100`: raw `WER = 27.06%`, normalized `WER = 5.96%`,
    `0` empty predictions, elapsed `141.63 s`, energy `24348.88 J`
  - reports:
    `reports/fleurs_fp8_gap_temp0_en_us_limit20.json`,
    `reports/energy_fleurs_fp8_gap_temp0_en_us_limit20.json`,
    `reports/benchmark_fp8_gap_temp0_en_us_limit20.json`,
    `reports/fleurs_fp8_gap_temp0_langhint_en_us_limit20.json`,
    `reports/energy_fleurs_fp8_gap_temp0_langhint_en_us_limit20.json`,
    `reports/benchmark_fp8_gap_temp0_langhint_en_us_limit20.json`,
    `reports/fleurs_fp8_gap_limit100_en_us_limit100.json`,
    `reports/energy_fleurs_fp8_gap_limit100_en_us_limit100.json`,
    `reports/benchmark_fp8_gap_limit100_en_us_limit100.json`
  - interpretation: larger normalized slice moves closer to published Voxtral English numbers, so
    sample size and evaluation frame matter more than the decode flags tested
- Additional findings from gap diagnosis:
  - the apparent English gap is not mainly explained by explicit `temperature = 0.0` or English
    hint
  - bigger factor is the evaluation frame itself: normalized metrics and slice size
  - current progression: `limit20 = 6.36%`, `limit100 = 5.96%`
- Larger multilingual side-by-side run replaced more tiny spot checks:
  - kept existing `en_us limit20` anchor
  - expanded `fr_fr` and `hi_in` from `limit5` to `limit20`
  - FP8:
    - `fr_fr limit20`: raw `WER = 24.91%`, normalized `WER = 8.33%`,
      elapsed `41.38 s`, energy `7448.59 J`
    - `hi_in limit20`: raw `WER = 30.10%`, normalized `WER = 23.91%`,
      elapsed `47.68 s`, energy `9092.83 J`
  - Whisper:
    - `fr_fr limit20`: raw `WER = 23.04%`, normalized `WER = 6.73%`,
      elapsed `49.52 s`, energy `6447.85 J`
    - `hi_in limit20`: raw `WER = 29.33%`, normalized `WER = 25.43%`,
      elapsed `85.05 s`, energy `13767.02 J`
  - combined with `en_us limit20`:
    - `en_us`: FP8 `6.36%`, Whisper `4.32%`
    - `fr_fr`: FP8 `8.33%`, Whisper `6.73%`
    - `hi_in`: FP8 `23.91%`, Whisper `25.43%`
  - simple three-language normalized macro average:
    - FP8 `12.87%`
    - Whisper `12.16%`
  - total measured energy across those slices:
    - FP8 `21494.31 J`
    - Whisper `23473.44 J`
    - FP8 used about `8.43%` less energy
  - conclusion: Whisper still leads on English/French; FP8 still leads slightly on Hindi; earlier
    Hindi `limit5` edge was real but overstated
  - reports:
    `reports/fleurs_fp8_multilingual_fr_fr_limit20.json`,
    `reports/energy_fleurs_fp8_multilingual_fr_fr_limit20.json`,
    `reports/benchmark_fp8_multilingual_fr_fr_limit20.json`,
    `reports/fleurs_fp8_multilingual_hi_in_limit20.json`,
    `reports/energy_fleurs_fp8_multilingual_hi_in_limit20.json`,
    `reports/benchmark_fp8_multilingual_hi_in_limit20.json`,
    `reports/fleurs_whisper_large_v3_fr_fr_limit20.json`,
    `reports/energy_fleurs_whisper_large_v3_fr_fr_limit20.json`,
    `reports/fleurs_whisper_large_v3_hi_in_limit20.json`,
    `reports/energy_fleurs_whisper_large_v3_hi_in_limit20.json`
- Larger English FP8 slice run on `en_us limit500`:
  - raw `WER = 27.58%`
  - normalized `WER = 6.49%`
  - `1` empty prediction
  - elapsed `651.27 s`
  - energy `154679.76 J`
  - reports:
    `reports/fleurs_fp8_gap_limit500_en_us_limit500.json`,
    `reports/energy_fleurs_fp8_gap_limit500_en_us_limit500.json`,
    `reports/benchmark_fp8_gap_limit500_en_us_limit500.json`
  - interpretation: `limit100` now looks optimistic; sample size matters but is not the main
    remaining explanation for the published gap
  - empty-prediction note: sample id `1758`, reference beginning `the archipelago lies 120 km north of the peninsula...`,
    `audio_peak_abs_before = 0.014361`, `quiet_audio_boosted = False`
- Real Round 1 follow-up encoded in the repo:
  - `configs/vllm/fp8_round1.yaml` now sets `kv_cache_dtype: fp8_e4m3` and
    `enable_prefix_caching: true`
  - other `vLLM` configs also now set `enable_prefix_caching: true`
  - `scripts/warm_fleurs_prefix_cache.py` added
  - current speech-to-text path supports prefix caching in principle, but
    `/v1/audio/transcriptions` does not expose per-request `cache_salt`, so warmup is process-local
  - large `limit500` benchmark still logged `kv_cache_dtype=auto` and `Prefix cache hit rate: 0.0%`
  - conclusion: repo is better prepared, but the hoped-for FP8-KV or prefix-cache gains were not
    yet captured in measured results
- Decoder-skipping feasibility track opened:
  - new planning and experiment support in `docs/decoder_skipping_track.md`
    and `configs/experiments.yaml` as `decoder_skip_feasibility`
  - new script `scripts/profile_fleurs_silence.py`
  - new helper `analyze_audio_activity(...)` in `src/voxtral_project/audio.py`
  - environment issue surfaced: Windows `.venv` had `datasets==4.8.4`, which broke the
    `google/fleurs` script-backed path; FLEURS-facing scripts were switched to shared loader
    `src/voxtral_project/dataset_utils.py`, which now fails clearly when `datasets>=4` is present
  - affected scripts updated: `scripts/evaluate_fleurs.py`,
    `scripts/warm_fleurs_prefix_cache.py`, `scripts/benchmark_vllm_variant.py`,
    `scripts/profile_fleurs_silence.py`
  - Windows `.venv` repaired back to pinned `datasets==3.6.0`
  - first live silence-feasibility smoke test on `en_us limit1`:
    - average raw silent-frame ratio `61.36%`
    - clips with at least half silent frames `1/1`
    - quiet-audio boost applied `0`
    - report `reports/fleurs_silence_en_us_limit1.json`
- Recommended next step at end of April 21: rerun the English submission slice with the new FP8
  mainline config and warmup helper; verify FP8 KV and prefix-cache state from logs; move toward
  benchmark-stack alignment rather than more decode-flag sweeps or even larger English reruns; keep
  FP8 as the submission mainline and GPTQ isolated unless intentionally resumed.

## April 22, 2026

- Benchmark-aligned scoring profile added instead of guessing about leaderboard behavior:
  - evaluator now preserves current local metrics and adds `open_asr_like`
  - changes: `src/voxtral_project/text.py`, `scripts/evaluate_fleurs.py`,
    `scripts/recompute_report_metrics.py`, `scripts/benchmark_vllm_variant.py`,
    `requirements.txt` (`num2words`)
- Closer dataset-wrapper path added:
  - repo now supports dataset sources `google_fleurs` and `open_asr_multilingual`
  - loader work in `src/voxtral_project/dataset_utils.py`
  - `open_asr_multilingual` validated directly on `en_us`; fields included `audio`, `duration`,
    `file_name`, `source_lang`, `target_lang`, `text`
  - fixed integration bug where inherited `token=True` incorrectly required a Hugging Face login
- English reports recomputed under the new scorer:
  - `reports/fleurs_fp8_en_us_limit20_quietfix.json`
  - `reports/fleurs_fp8_gap_limit100_en_us_limit100.json`
  - `reports/fleurs_fp8_gap_limit500_en_us_limit500.json`
  - `reports/fleurs_whisper_large_v3_en_us_limit20.json`
  - English `open_asr_like` WER was identical to existing normalized English WER:
    - FP8 `limit20`: `6.36%`
    - FP8 `limit100`: `5.96%`
    - FP8 `limit500`: `6.49%`
    - Whisper `limit20`: `4.32%`
  - conclusion: remaining English difference is not mainly our text-normalization choice
- French and Hindi `limit20` reports recomputed under `open_asr_like`:
  - `fr_fr`
    - FP8 local normalized `8.33%`, `open_asr_like 8.11%`
    - Whisper local normalized `6.73%`, `open_asr_like 6.52%`
  - `hi_in`
    - FP8 local normalized `23.91%`, `open_asr_like 14.74%`
    - Whisper local normalized `25.43%`, `open_asr_like 13.82%`
  - English unchanged, French slightly changed, Hindi changed a lot and flipped the local FP8 edge
  - three-language `open_asr_like` macro average:
    - FP8 `9.74%`
    - Whisper `8.22%`
- Public `open_asr_multilingual` English comparison run:
  - FP8 on wrapper: raw `WER = 14.35%`, local normalized `7.01%`,
    `open_asr_like = 7.01%`, elapsed `56.40 s`, energy `8244.80 J`
  - Whisper on wrapper: raw `WER = 11.24%`, local normalized `4.21%`,
    `open_asr_like = 4.21%`, elapsed `92.13 s`, energy `8844.86 J`
  - compared with `google/fleurs` anchors:
    - FP8 moved `6.36% -> 7.01%`
    - Whisper moved `4.32% -> 4.21%`
    - FP8-minus-Whisper gap widened from `2.05` to `2.80` points
  - reports:
    `reports/fleurs_fp8_openasr_en_us_limit20.json`,
    `reports/energy_fleurs_fp8_openasr_en_us_limit20.json`,
    `reports/benchmark_fp8_openasr_en_us_limit20.json`,
    `reports/fleurs_whisper_large_v3_openasr_en_us_limit20.json`,
    `reports/energy_fleurs_whisper_large_v3_openasr_en_us_limit20.json`
  - conclusion: wrapper differences matter but do not help Voxtral; they make current English
    quality look slightly worse for FP8
- Benchmark helper inconsistency fixed: `scripts/benchmark_vllm_variant.py` now uses the selected
  dataset source for the `first_request` preview as well as the actual evaluation.
- Whisper wrapper path quirk recorded: run wrote valid report/energy files but exited with
  `return_code = -6` in late finalization; numbers are usable but the runner is not fully clean.
- Benchmark-alignment findings: full practical loop now done (rescoring, public-wrapper FP8,
  public-wrapper Whisper). Remaining mismatch is not a decode-flag problem, not mainly a local
  scorer problem, and not fixed by the public wrapper. Most plausible remaining causes are deeper
  benchmark procedure differences, manifest construction, or evaluation-stack details beyond the
  wrapper and transcript normalizer.
- Round-one positioning reframed as a selection problem rather than a tuning problem:
  - defended claim locked to the strongest validated FP8 anchor
  - FP8 is the strongest compressed Voxtral path in the repo
  - on trusted `en_us limit20`, FP8 matches BF16 normalized WER at `6.36%`
  - FP8 is about `24%` faster and about `39%` lower energy than BF16 on that slice
  - stronger public ASR baselines still lead on the benchmark-aligned quality view
  - prefix caching removed from the defended round-one story because `/v1/audio/transcriptions`
    still showed `Prefix cache hit rate: 0.0%` and a fresh TRITON-backed validation run landed at
    `6.82%` normalized WER and `7398 J`
  - submission docs updated: `README.md`, `docs/fp8_mainline_track.md`,
    `docs/submission_candidate_summary.md`, `docs/submission_benchmark_table.md`,
    `docs/submission_readiness_checklist.md`, `docs/fp8_benchmark_summary.md`,
    `docs/round1_submission_narrative.md`, `reports/team_status.md`
  - strategic verdict: strongest round-one frame is a reproducible edge-serving FP8 Voxtral
    system with real measured efficiency gains, cleaned-up evaluation, and multilingual credibility;
    not a benchmark-quality leader and not a story dependent on prefix cache
- Speech-gating path implemented as a benchmarkable contender lever:
  - `src/voxtral_project/audio.py` gained speech-aware gating
  - `scripts/evaluate_fleurs.py` gained gating controls and per-sample diagnostics
  - `scripts/benchmark_vllm_variant.py` forwards gating into first-request and evaluated runs
- First speech-gating probe (`fp8_gate_edgeprobe`) used edge-only silence trimming on
  `en_us limit5` with `preserve_leading_silence_ms = 160`,
  `preserve_trailing_silence_ms = 160`, no internal silence compression:
  - first sample removed `2.72 s` from a `10.56 s` clip
  - small-slice normalized WER worsened to `8.65%`
  - interpretation: removable edge silence exists, but naive edge trimming clips quiet trailing
    speech
- Second edge-trim probe (`fp8_gate_edgeprobe_v2b`) used `frame_ms = 40`,
  `peak_threshold = 0.005`, `rms_threshold = 0.0015`,
  `preserve_leading_silence_ms = 320`, `preserve_trailing_silence_ms = 480`:
  - first sample still removed `2.16 s` from `10.56 s`
  - small-slice normalized WER improved to `6.73%`
  - sample-level outputs looked better, but harness state became noisy because an earlier server
    process stayed alive and contaminated startup/energy comparisons
- Prepared-audio audit corrected the opportunity estimate:
  - raw-audio view over-classified several quiet clips as “all silence”
  - planning view should be prepared audio after quiet boosting
  - refreshed `en_us limit20` audit with `160/160 ms` preserve windows showed:
    - average prepared edge-trim candidate `1.944 s` per clip
    - average prepared edge-trim candidate ratio `20.4%`
    - median prepared edge-trim candidate ratio `22.7%`
    - clips with at least `20%` prepared edge-trim opportunity `12/20`
    - clips with at least `30%` prepared edge-trim opportunity `5/20`
  - conclusion: there is real application-layer compute to remove, but the current trimming rule is
    too sharp around clip endings
- Gating verdict after the audit:
  - speech-aware edge trimming is a legitimate Round 1 contender lever
  - current `160/160 ms` setting is not safe enough
  - likely failure mode is trailing-speech clipping rather than lack of removable silence
  - remaining upside still looks meaningful with larger tail windows:
    - `160/640 ms` implies roughly `16%` average trim headroom
    - `160/960 ms` implies roughly `13.5%` average trim headroom
  - next intended clean retune: `160/960 ms`, then `160/640 ms` if needed
- Practical blocker on that retune: GPU already occupied by another live FP8 benchmark. Decision:
  stop rather than stack another noisy run; update the record and make the next benchmark explicit.
- Decoder-skipping gate follow-through completed on the trusted `en_us limit20` slice:
  - report `reports/fleurs_silence_en_us_limit20.json`
  - average raw silent-frame ratio `68.27%`
  - median raw silent-frame ratio `96.02%`
  - clips with at least half silent frames `14/20`
  - average prepared edge-trim candidate `1.944 s`
  - clips with at least `20%` prepared edge-trim opportunity `12/20`
  - conclusion: PDF premise has real signal on the current English slice; worth at least one
    controlled benchmark
- Apparent multi-hour silence-profile runtime issue resolved:
  - interrupted run left orphaned Python processes
  - no report was written from that attempt
  - clean rerun completed in about `8 s`
  - direct dataset iteration for the first `20` FLEURS samples took about `4.75 s`
  - conclusion: long runtime was a stuck process / transient Hub stall, not intrinsic tool cost
- First clean `en_us limit20` control vs gating comparison:
  - fresh control `reports/benchmark_fp8_gate_control_en20_en_us_limit20.json`
  - aggressive boundary gate `reports/benchmark_fp8_gate_boundary_en20_en_us_limit20.json`
  - soft boundary gate `reports/benchmark_fp8_gate_boundary_soft_en20_en_us_limit20.json`
  - because the first fresh control included heavy cold-start/compile overhead, the fair comparison
    is against the later warm control:
    `reports/benchmark_fp8_gate_control_warm_en20_en_us_limit20.json`
- Aggressive edge trimming failed cleanly:
  - settings: `frame_ms = 80`, `peak_threshold = 0.01`, `rms_threshold = 0.003`,
    `preserve_leading_silence_ms = 160`, `preserve_trailing_silence_ms = 160`
  - result: normalized WER `15.45%`, energy `6651.24 J`, elapsed `40.84 s`
  - sample-level summary: gating changed `15/20` clips, average removed audio `1.935 s`,
    average removed fraction `20.3%`
  - interpretation: gain is real, but quality collapse makes it unusable
- Softer edge trimming survived quality but gain is modest:
  - settings: `peak_threshold = 0.005`, `rms_threshold = 0.0015`,
    `preserve_leading_silence_ms = 400`, `preserve_trailing_silence_ms = 400`
  - fair comparison:
    - warm control: normalized WER `6.82%`, energy `7102.34 J`, elapsed `43.48 s`
    - soft gate: normalized WER `6.82%`, energy `6588.71 J`, elapsed `41.78 s`
  - measured delta vs warm control:
    - normalized WER unchanged
    - elapsed about `1.70 s` lower, about `3.9%` lower
    - energy about `513.64 J` lower, about `7.2%` lower
  - sample-level summary: gating changed `14/20` clips, average removed audio `0.953 s`, total
    removed audio across the slice `19.06 s`
  - interpretation: viable as a small efficiency optimization, not the dramatic second lever
    implied by raw silence statistics; earlier giant win was cold-start noise
- Final decoder-skipping verdict:
  - true: PDF intuition is directionally right; current English slice is silence-heavy enough to
    justify investigation; conservative application-layer boundary trim can save a small amount of
    time and energy without hurting normalized WER
  - not true: naive edge trimming is a breakthrough; current boundary-only heuristic is strong
    enough to be the main differentiator; raw silence ratio maps directly to deployable decoder
    savings
  - next move is no longer more boundary-trim sweeps; it is to instrument actual Voxtral
    pad-token / decoder-step behavior on the same evaluation slice and only then decide whether
    deeper scheduling or speculative-decoding work is justified
  - classification of the idea: real signal, small audio-boundary win, still unproven as a major
    decoder-level optimization

## April 24, 2026

- Final consolidation and push:
  - concluded active workstreams and prepared the repo for one final push rather than leaving
    parallel chat output stranded locally
  - submission-facing documentation prepared around a conservative and defensible story:
    - FP8 is the best compressed Voxtral path currently working in the repo
    - FP8 improves efficiency versus the BF16 Voxtral reference on the trusted English slice
    - benchmark-aligned external comparisons are included as context, not hidden
    - prefix-cache and speech-gating findings are recorded as investigations, not overstated claims
  - final push prepared to include updated benchmark/evaluation tooling, audio preprocessing
    utilities, dataset helpers, FP8 serving configuration, README guidance, submission docs, and
    the new team instruction file
  - push intent: commit and push to `origin/main` so GitHub becomes the current source of truth
    after the chats conclude
