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

Two later findings matter for how this result should be presented:

- benchmark-aligned rescoring did not close the English gap to strong external baselines
- public-wrapper checks on `open_asr_multilingual` made the English gap look slightly worse for
  FP8, not better
- prefix-cache validation on `/v1/audio/transcriptions` still showed `0.0%` hit rate, so prefix
  caching is not part of the current submission claim

That changes the right framing:

- FP8 is still the strongest compressed Voxtral path in this repo
- FP8 is clearly better than our BF16 Voxtral reference on efficiency
- FP8 is credible and submission-worthy, but not honestly benchmark-leading against Whisper
  large-v3 on the current benchmark-aligned view

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

Important submission-scoping rule:

- treat the validated FP8 English anchor as the core claim
- treat benchmark-aligned rescoring as the honest external context
- do not claim a realized round-one prefix-cache gain until the speech path shows non-zero cache
  reuse in measured runs

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
- [docs/round1_submission_narrative.md](/C:/Users/ASUS/Music/Fine_tuning/docs/round1_submission_narrative.md)
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
python scripts/measure_energy.py --report reports/energy_fleurs_fp8_en_us_limit20_quietfix.json -- python scripts/evaluate_fleurs.py --lang en_us --limit 20 --base-url http://localhost:8082/v1 --model voxtral-realtime --out reports/fleurs_fp8_en_us_limit20_quietfix.json
```

The quiet-audio-aware BF16 comparison uses the same evaluation command, just pointed back to
`http://localhost:8081/v1`.

Important runtime note:

- the current WSL `vLLM` speech-to-text path supports prefix caching in principle, but the
  measured validation run on `/v1/audio/transcriptions` still showed `0.0%` hit rate
- so `scripts/warm_fleurs_prefix_cache.py` is an investigation tool right now, not part of the
  primary submission recipe

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

- `docs/round2_track_plan.md` &mdash; **master plan** for Round-2 tracks D, E, F, L4 with
  per-track venvs, kill switches, and decision gates.
- `docs/round2_candidate_snapshot.md` &mdash; current locked Round-2 candidate (audio lever,
  validated on 13 FLEURS languages on RTX 5080) and the L4 handoff briefing.
- `docs/round2_audio_lever.md` &mdash; rationale for the locked audio-prep stack.
- `docs/submission_candidate_summary.md`
- `docs/submission_readiness_checklist.md`
- `docs/submission_benchmark_table.md`
- `docs/fp8_benchmark_summary.md`
- `docs/fp8_mainline_track.md`
- `docs/gptq_track_summary.md`
- `docs/decoder_skipping_track.md`

## Round-2 Status (2026-05-08, end of session)

### Track A++ (audio lever) &mdash; LOCKED
Pinned audio-prep stack on top of Track A FP8:

```
--target-lufs -23.0 --lufs-max-gain-db 24.0
--vad-trim --vad-aggressiveness 1 --vad-padding-ms 200
--gate-silence
--compress-internal-silence-to-ms 160      # tightened from 320 in afternoon ablation
--min-internal-silence-run-ms 320          # tightened from 640
```

Validated on RTX 5080 across all 13 Voxtral-supported FLEURS languages at limit=100 (EN at
limit=500), zero empty predictions across 1700 samples. Energy total: **283.20 kJ** for the
4-language Track A canonical set on RTX 5080 (relative; binding L4 measurement still pending).

| Slice | norm WER | CER (no_ws) | empty | trim% | RTX 5080 kJ |
|---|---|---|---|---|---|
| es_419 (100) | **3.01%** | 5.33% | 0 | 14.71% | 30.32 |
| it_it (100) | **3.74%** | 5.95% | 0 | 15.70% | 46.63 |
| ru_ru (100) | **5.18%** | 6.04% | 0 | 12.17% | 26.96 |
| pt_br (100) | **5.29%** | 6.99% | 0 | 14.47% | 36.79 |
| de_de (100) | **5.64%** | 10.19% | 0 | 14.09% | 40.70 |
| en_us (500) | 5.69% | 7.90% | 0 | 12.59% | 129.79 |
| fr_fr (100) | 7.36% | 7.49% | 0 | 19.81% | 24.75 |
| nl_nl (100) | 8.71% | 7.52% | 0 | 11.22% | 27.68 |
| ar_eg (100) | 15.70% | 5.82% | 0 | 8.86% | 39.15 |
| ko_kr (100) | 16.02% | 8.72% | 0 | **27.20%** | 23.37 |
| hi_in (100) | 26.12% | 14.76% | 0 | 7.16% | 34.60 |
| ja_jp (100) | (WER moot) | **10.51%** | 0 | 14.83% | 44.52 |
| cmn_hans_cn (100) | (WER moot) | **12.41%** | 0 | **25.93%** | 25.90 |

Reports: `reports/fleurs_fp8_<lang>_limit{100,500}_lufs23_vadgate160_320_smoke.json`.

### Track B (SpinQuant W4A16 via llmcompressor) &mdash; DROPPED
Three different llmcompressor blockers (norm-fusion assertion, block-size mismatch, CUDA
index-OOB). Even plain GPTQ failed in the patched venv. See team_status.md and
`docs/round2_track_plan.md` for the full timeline.

### Track D1 (AutoRound W4A16) &mdash; PARTIALLY ALIVE, blocked on vLLM kernel side

What works:
- `~/.venvs/voxtral-trackd-autoround` venv (transformers 5.5.4 + auto-round 0.12.3)
  loads Voxtral cleanly. AutoRound v4 smoke quantized 234/235 modules in 44 s,
  packed 26 decoder layers x 7 projections = 234 modules in `auto_round:auto_gptq` format.
  Output size 2.27 GB (decoder only) vs 8.86 GB BF16; matches expected W4A16 ratio.
- `merge_autoround_into_voxtral.py` script splices the W4 decoder back into the original
  BF16 Voxtral checkpoint (audio_tower + multi_modal_projector kept BF16). Output 4.07 GB,
  patches `quant_method: auto-round -> gptq` in config.
- `~/.venvs/voxtral-trackd-serve` venv (vllm 0.19.1rc1.dev302 + transformers 5.5.4 force-installed
  with --no-deps + TORCH_INIT_FUNCTIONS shim). Loads the merged checkpoint, recognizes the
  architecture, accepts `--quantization gptq_marlin --dtype half`.

Where it's stuck:
- vLLM 0.19.1rc1's dedicated `voxtral_realtime.py` impl has **zero quantization references
  in source** &mdash; it can serve BF16 / FP8 KV but not GPTQ-prequantized weights.
- When `quantization_config` is present in config.json, vLLM falls back to
  `TransformersMultiModalForCausalLM` (the generic transformers backend), which assumes
  **images** (`return {"image": self.get_max_image_tokens()}`) and crashes loading the
  Voxtral audio processor.
- This is the actual architecture wall: vLLM has GPTQ for text models, audio for Voxtral,
  but no overlap.

Three forward paths from here (in `docs/round2_track_plan.md`):
1. Patch vLLM's `voxtral_realtime.py` to use parameterized linear factories from
   `vllm.model_executor.layers.linear` (which auto-handle GPTQ via QuantizationConfig).
2. Use `--model-impl vllm` to force the dedicated impl + see if it transparently handles
   GPTQ packed weights (probably not, but worth a 1-hour test).
3. Pivot to a non-vLLM serving stack (TGI, SGLang, or transformers + GPTQModel).

### Track E (EAGLE-3 spec decode) &mdash; UNTESTED
`~/.venvs/voxtral-tracke-eagle` is built. vLLM 0.19.1rc1 has `v1/spec_decode/eagle.py` and
`v1/worker/gpu/spec_decode/eagle/` so the kernel side exists. The smoke gate (does
`--speculative-config` route through `/v1/audio/transcriptions`?) hasn't been run yet.

## New Tooling This Session

- `~/models/voxtral-realtime/` &mdash; Voxtral relocated to native ext4. Loads in 0.6 s vs
  ~120 s from `/mnt/c` (9P).
- `~/.venvs/voxtral-trackd-autoround/` &mdash; AutoRound calibration venv.
- `~/.venvs/voxtral-tracke-eagle/` &mdash; EAGLE serving venv.
- `~/.venvs/voxtral-trackd-serve/` &mdash; vLLM 0.19 + transformers 5.5.4 force-pinned for
  serving quantized Voxtral.
- `scripts/run_round2_multilingual_sweep.sh`
- `scripts/reproduce_round2_audio.sh`
- `scripts/summarize_round2_audio_runs.py` (now supports `--full-fleurs`)
- `.claude/worktrees/.../merge_autoround_into_voxtral.py` &mdash; W4-decoder + BF16-audio splice.
- `.claude/worktrees/.../probe_autoround_smoke_v4.py` &mdash; AutoRound smoke with t_cond hook.
- `configs/vllm/track_d1_w4a16_autoround_smoke.yaml`

## Resume points (next session)

**2026-05-12 update**: Two more big steps cleared this session:

- **Track E1 (EAGLE-3 spec-decode) is ALIVE** — proven that vLLM's
  `--speculative-config` plumbs through `/v1/audio/transcriptions`. EAGLE-3
  needs a trained draft (Track E2), but the architectural path is open.
- **Track D1 (AutoRound W4 + vLLM serve)** cleared **9 of 10** walls. Wall #10:
  vLLM's GPTQ weight loader doesn't apply the runtime `attention_norm ->
  self_attn_layer_norm` rename that the BF16 path uses for the audio encoder.
  Fixing this means patching `vllm/model_executor/models/voxtral.py` to define
  a `stacked_params_mapping` / `packed_modules_mapping` for the audio encoder.
  Estimated 4-8 hours focused work.

Five workstreams now teed up for parallel threads:

1. **Track D1 wall #10**: patch vLLM `voxtral.py` audio encoder
   `packed_modules_mapping`. Or pre-rename audio encoder norm keys in our final
   checkpoint to match Whisper-internal names directly. Last attempted serve in
   `~/voxtral-w4a16-final-smoke/` with config in
   `configs/vllm/track_d1_w4a16_autoround_smoke.yaml`. Full wall-by-wall log in
   `reports/team_status.md` 2026-05-12 entry.
2. **Track E2**: train/wire EAGLE-3 draft model for Voxtral's text decoder.
3. **L4 cloud provisioning**: prerequisite for any binding energy claim. Run
   `scripts/reproduce_round2_audio.sh` against it for the locked
   Track A++ candidate immediately, regardless of D1/E2 progress.
4. **Send organizer email** about HI id 1985 idx 82 duplicate (drafted in
   `reports/sample_1985_investigation/hi_1985_findings.md`).
5. **Optional**: write `dequantize_ada_rms_norm` step into the AutoRound runner
   so future smoke runs don't need the post-hoc restore (small quality-of-life
   improvement on Track D1 iteration cycle).

## What Is Intentionally Missing

- No end-to-end quantization automation yet.
- No cloud deployment yet.
- No submission packaging yet.

Those come after we lock down a reliable local baseline and evaluation loop.
