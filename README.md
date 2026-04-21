# multimodal-edge-compression

Compression workspace for `mistralai/Voxtral-Mini-4B-Realtime-2602`, focused on fast,
energy-aware audio transcription on local edge hardware for the Resilient AI Challenge.

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

## Current Verified Checkpoint

The current strongest local reference point is the quiet-audio-aware English FLEURS `20`-sample
comparison:

- BF16 quietfix baseline on `http://localhost:8081/v1`
  - `WER = 22.20%`
  - `normalized WER = 6.36%`
  - `empty_prediction_count = 0`
  - `elapsed_seconds = 46.26`
  - `energy_joules = 8112.90`
- FP8 round 1 on `http://localhost:8082/v1`
  - `WER = 21.97%`
  - `normalized WER = 6.36%`
  - `empty_prediction_count = 0`
  - `elapsed_seconds = 35.21`
  - `energy_joules = 4952.89`

So the first `fp8_round1` run is currently the best practical result we have: essentially flat
quality with materially lower time and energy on this machine.

The next low-risk submission stack is now explicit in the repo:

- `configs/vllm/fp8_round1.yaml` now includes `kv_cache_dtype: fp8_e4m3`
- the serving configs now set `enable_prefix_caching: true` explicitly
- `scripts/warm_fleurs_prefix_cache.py` can warm the shared speech-to-text prefix cache before a
  measured evaluation run

Important comparison note:

- raw WER in our reports is useful internally, but public-facing comparison should use normalized
  WER because FLEURS references are punctuation-light and model predictions are not
- the first external same-slice anchor is Whisper large-v3:
  - raw `WER = 20.59%`
  - normalized `WER = 4.32%`
  - `elapsed_seconds = 34.77`
  - `energy_joules = 3258.57`
- so FP8 is currently the best compressed Voxtral path here, but it is not yet beating the
  strongest external baseline we have checked

## Current Submission Path

The first submission path should be the FP8 mainline, not GPTQ.

Why:

- FP8 is already working and benchmarked
- FP8 already beats BF16 on efficiency on the core English comparison
- FP8 already has multilingual spot-check evidence
- GPTQ is still a research branch and is not yet artifact-ready

Alongside those two established tracks, the repo now has a low-risk decoder-skipping feasibility
track:

- `docs/decoder_skipping_track.md`
- `scripts/profile_fleurs_silence.py`

That track does not touch the working inference path yet. Its job is to measure whether the
paper's "skip decoder work on silence-heavy audio" premise is strong enough on our FLEURS slices
to justify deeper engineering.

Submission-facing docs:

- [docs/submission_candidate_summary.md](/C:/Users/ASUS/Music/Fine_tuning/docs/submission_candidate_summary.md)
- [docs/submission_readiness_checklist.md](/C:/Users/ASUS/Music/Fine_tuning/docs/submission_readiness_checklist.md)
- [docs/submission_benchmark_table.md](/C:/Users/ASUS/Music/Fine_tuning/docs/submission_benchmark_table.md)
- [docs/global_benchmark_comparison.md](/C:/Users/ASUS/Music/Fine_tuning/docs/global_benchmark_comparison.md)

## Important Runtime Note

This machine is currently a Windows workspace, but the competition guide is Linux-oriented and
`vLLM` is most practical in Linux or WSL2. This repo is structured so we can manage the project
from Windows while running the heavy runtime pieces in Linux when needed.

## Runtime Lessons So Far

- The local 16 GB GPU budget is happiest at:
  - `max_model_len: 8192`
- The first stable FP8 serving config on this machine is:
  - `configs/vllm/fp8_round1.yaml`
  - `gpu_memory_utilization: 0.85`
- Local transcription calls should be serialized through the shared API helper to avoid the known
  engine instability under overlapping audio requests.
- Some FLEURS clips are quiet enough to produce empty transcripts unless they are boosted first.
  The evaluator now applies quiet-audio preparation automatically and records per-sample audio
  diagnostics in the report JSON.

## Repo Layout

- `configs/experiments.yaml` - named experiment matrix and module protection policy.
- `configs/vllm/` - starter `vLLM` configs for baseline and compression experiments.
- `docs/guide_notes.md` - distilled notes from the PDF guide.
- `scripts/download_model.py` - download the model from Hugging Face.
- `scripts/serve_model.py` - launch `vLLM serve` from a YAML config.
- `scripts/start_wsl_baseline.ps1` - start the BF16 baseline server in WSL from PowerShell.
- `scripts/start_vllm_server.sh` - stable WSL-native launcher for the BF16 server.
- `scripts/check_vllm_server.py` - poll `/v1/models` until the server is ready.
- `scripts/smoke_test_hf_sample.py` - transcribe a known public sample audio file from Hugging Face.
- `scripts/transcribe_file.py` - send one audio file to the server and print the transcript.
- `scripts/evaluate_fleurs.py` - run WER evaluation on one or more FLEURS languages.
- `scripts/measure_energy.py` - wrap any command with CodeCarbon energy tracking.
- `scripts/profile_fleurs_silence.py` - measure silence-heavy structure as a proxy for decoder-skip opportunity.
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
python -m pip install -U uv
uv pip install -U vllm --torch-backend=cu130 --extra-index-url https://wheels.vllm.ai/nightly/cu130
```

3. Download the baseline model:

```powershell
python scripts/download_model.py --local-dir models/voxtral-realtime
```

4. Serve the BF16 baseline:

```powershell
python scripts/serve_model.py models/voxtral-realtime --config configs/vllm/bf16.yaml --port 8081
```

If you are launching from Windows into WSL, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_wsl_baseline.ps1
```

5. Wait for the server to become ready:

```powershell
python scripts/check_vllm_server.py --base-url http://localhost:8081/v1
```

6. Run a public sample smoke test:

```powershell
python scripts/smoke_test_hf_sample.py --base-url http://localhost:8081/v1 --model voxtral-realtime --out reports/smoke_test_transcript.txt
```

7. Run a small FLEURS evaluation:

```powershell
python scripts/evaluate_fleurs.py --lang en_us --limit 5 --base-url http://localhost:8081/v1 --out reports/fleurs_en_us_limit5.json
```

8. Measure energy for an evaluation run:

```powershell
python scripts/measure_energy.py --report reports/bf16_energy.json -- python scripts/evaluate_fleurs.py --lang en_us --limit 5 --base-url http://localhost:8081/v1
```

9. Launch the first FP8 compression server after stopping BF16:

```powershell
python scripts/serve_model.py models/voxtral-realtime --config configs/vllm/fp8_round1.yaml --port 8082
```

10. Run the current best apples-to-apples English comparison:

```powershell
python scripts/warm_fleurs_prefix_cache.py --lang en_us --base-url http://localhost:8082/v1 --model voxtral-realtime --out reports/prefix_warmup_fp8_en_us.json
python scripts/measure_energy.py --report reports/energy_fleurs_fp8_en_us_limit20_quietfix.json -- python scripts/evaluate_fleurs.py --lang en_us --limit 20 --base-url http://localhost:8082/v1 --model voxtral-realtime --out reports/fleurs_fp8_en_us_limit20_quietfix.json
```

The quiet-audio-aware BF16 comparison uses the same evaluation command, just pointed back to
`http://localhost:8081/v1`.

Important runtime note:

- the current WSL `vLLM` speech-to-text path supports prefix caching, but it does not expose
  per-request `cache_salt` on `/v1/audio/transcriptions`
- in practice that means warmup is process-local today: prime the server once, then run the
  measured evaluation against the same live process

## Initial Experiment Order

- `bf16_baseline`
- `fp8_round1`
- `gptq8_round1`
- `gptq4_round2`
- `gptq4_fp8kv_round2`

That ordering follows the guide: get a stable baseline first, try the simplest hardware-friendly
compression next, then move into more aggressive decoder quantization.

## Most Useful Reports Right Now

- `reports/fleurs_bf16_en_us_limit20_quietfix.json`
- `reports/energy_fleurs_bf16_en_us_limit20_quietfix.json`
- `reports/fleurs_fp8_en_us_limit20_quietfix.json`
- `reports/energy_fleurs_fp8_en_us_limit20_quietfix.json`
- `reports/fleurs_fp8_hi_in_limit5_quietfix.json`
- `reports/fleurs_fp8_fr_fr_limit5_quietfix.json`
- `reports/fleurs_fp8_ja_jp_limit5_quietfix_v2.json`

## Most Useful Docs Right Now

- `docs/submission_candidate_summary.md`
- `docs/submission_readiness_checklist.md`
- `docs/submission_benchmark_table.md`
- `docs/fp8_benchmark_summary.md`
- `docs/fp8_mainline_track.md`
- `docs/gptq_track_summary.md`
- `docs/decoder_skipping_track.md`

## What Is Intentionally Missing

- No end-to-end quantization automation yet.
- No cloud deployment yet.
- No submission packaging yet.

Those come after we lock down a reliable local baseline and evaluation loop.
