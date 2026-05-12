# Round-2 Track Plan (Tracks D, E, F + L4)

This document defines the Round-2 work tracks beyond Track A (FP8) and Track A++
(Round-2 audio lever, locked 2026-05-08). Each track has its own venv, its own
mission, and its own kill condition so failures stay isolated.

The master goal: maximize P(win) on the Resilient AI Challenge submission due
2026-06-15. Today's audio lever bought us roughly +10-15% energy reduction over
Track A. The major remaining levers - W4 decoder, speculative decoding, and
encoder shrinking - are independent and stack multiplicatively.

## Track inventory

| Track | Mission | Status | P(win) impact | Effort |
|---|---|---|---|---|
| Track A | FP8 mainline, Round-1 submission | done & submitted | baseline | done |
| Track A++ | LUFS+VAD+gate(160/320) audio lever | done, locked 2026-05-08 | +10-15% energy | done |
| Track B | SpinQuant W4A16 via llmcompressor | dropped | n/a | dead (toolchain wall) |
| Track C | Final-validation script + L4 prep tooling | blocked | enables submission | medium |
| **Track D** | **W4 decoder via three sub-paths in parallel** | **NEW (today)** | **+15-25% energy on top of Track A** | **3 weeks** |
| **Track E** | **EAGLE-3 / speculative decoding** | **NEW (today)** | **+30-50% wall-clock = energy** | **1.5 weeks** |
| **Track F** | **Encoder shrink (top-2-layer prune / distil)** | **stretch** | **+1-3% energy** | **1 week, optional** |
| **Track L4** | **Cloud L4 binding measurement + submission package** | **NEW (final gate)** | **enables claim** | **5-7 days at end** |

## Per-track venv layout

Existing (do NOT mutate):
- `~/.venvs/voxtral-baseline` - Track A pinned stack
- `~/.venvs/voxtral-llmcompressor-research` - Track B legacy research
- `~/.venvs/voxtral-spinquant` - Track B retrospective (transformers 5 + llmcompressor 0.10 via .pth shim)
- `~/.venvs/voxtral-gptq-research` - reference for transformers 5 + torch 2.11+cu130

New today:
- `~/.venvs/voxtral-trackd-autoround` - **Track D1** (cloned from gptq-research, +auto-round 0.12.3)
- `~/.venvs/voxtral-tracke-eagle` - **Track E** (cloned from baseline, +speculators)

Pending (built only if D1 fails):
- `~/.venvs/voxtral-trackd-llmc5` - **Track D2** (transformers 5 + source-build llmcompressor)
- `~/.venvs/voxtral-trackd-rtn` - **Track D3** (custom model-free RTN W4)
- `~/.venvs/voxtral-trackf-encoder` - **Track F** (encoder pruning)

Shared, native ext4 (relocated from /mnt/c on 2026-05-08):
- `~/models/voxtral-realtime/` - 17 GB, both `consolidated.safetensors` and `model.safetensors` plus configs/tokenizer. **Loads in 0.6s vs ~120s from /mnt/c 9P.**

## Track D - W4 decoder (parallel sub-paths)

### Mission
Get any W4A16-quality decoder serving through vLLM 0.19.1rc1.dev302 with
normalized WER within 1.25x BF16 baseline on the canonical four-language slice
(EN500/HI100/FR100/JA100).

### Sub-paths (run in parallel, each with kill-switch)

**Track D1 - AutoRound (Intel)**

- Tool: `auto-round 0.12.3`
- Why first: has explicit `AutoRoundMLLM` class with `quant_nontext_module=False`
  default, which auto-skips the audio tower and multimodal projector that fought
  llmcompressor for three days
- Risk: medium (different framework, may have its own multimodal quirks)
- Effort: 1-2 days
- Smoke gate: 4 hours to first compressed-tensors artifact loading in vLLM /health 200
- Files: `.claude/worktrees/.../probe_autoround_smoke.py` (smoke runner)

**Track D2 - source-build llmcompressor**

- Tool: `llmcompressor` built from `main` branch with the transformers <=4.57.6
  pin removed
- Why: keeps our existing `scripts/run_track_b_llmcompressor_oneshot.py` runner
  reusable. The runner already knows the Voxtral protect/quantize layout.
- Risk: medium-high (the CUDA index-OOB during calibration forward we hit
  yesterday is likely a real API mismatch; source build may or may not fix it)
- Effort: 3-5 days
- Smoke gate: same as D1

**Track D3 - custom model-free RTN W4**

- Tool: write a ~150-line Python script that opens the safetensors directly,
  applies symmetric per-channel RTN W4 to the 26 decoder layers x 7 projection
  matrices (182 modules), packs to compressed-tensors format, emits config.json
  metadata
- Why: no transformers/llmcompressor dependency at all. Most patch-immune fallback.
- Risk: low engineering, ~1-2 pp WER penalty vs GPTQ-quality W4
- Effort: 2-3 days
- Smoke gate: same as D1

### Track D win condition
Any sub-path produces a serving artifact with:
- `vllm 0.19.1rc1.dev302+cu130 /health 200`
- Normalized WER <= 1.25x BF16 baseline on EN500 + HI100 + FR100 + JA100
- Empty count <= Track A baseline
- Stable through 500-sample evaluation (no engine crashes)

Expected energy improvement: **15-25% on top of Track A FP8** (so cumulative
**40-50% vs BF16 baseline**).

## Track E - Speculative decoding

### Mission
Determine whether vLLM 0.19.1rc1's EAGLE-3 path reaches `/v1/audio/transcriptions`
on Voxtral Realtime, and if so, train and wire a 200M draft model.

### Sub-steps

**Track E1 - kill-switch smoke (2 hours)**

Start an FP8 server with `--speculative-config '{"model": <small>, "method": "eagle3"}'`
pointing at any small Mistral-architecture decoder, send a single transcription
request. If vLLM rejects the speculative config for the audio path (most likely
outcome), Track E is dead and we don't sink more time.

**Track E2 - draft training (3-5 days, only if E1 passes)**

Train an EAGLE-3 draft for the Voxtral text decoder using:
- public ASR transcript pairs as base text
- the existing `data/calibration/track_b_audio_conditioned_fleurs_train_256_hi61/`
  audio-conditioned dataset for distillation

**Track E3 - benchmark (2 days, only if E2 passes)**

Stand up the full stack on L4. Measure draft-acceptance rate, end-to-end latency,
energy delta vs non-spec FP8.

### Track E win condition
- Spec-decode acceptance rate > 50% on FLEURS English transcription
- Same WER as non-spec FP8
- 1.4x or better wall-clock speedup on EN500

Expected energy improvement: **30-50% wall-clock = directly proportional energy reduction**.

## Track F - Encoder shrink (stretch only)

### Mission
Trim the Voxtral audio encoder by 1-2 transformer layers from the top, recover
WER if necessary with light fine-tuning.

### Sub-steps
- F1: produce a layer-pruned encoder checkpoint loadable via vLLM
- F2: measure WER delta on EN500 + multilingual

### Track F decision rule
Only attempt if Track D + Track E land with at least 5 days to spare before
2026-06-15 freeze.

Expected energy improvement: **1-3% total system energy** (smallest of the three
levers; the encoder is only ~16% of params).

## Track L4 - Binding measurement + submission package

### Mission
Provision an L4 24 GB cloud node, install the pinned stack, run the final
parameter set with measurement rigour, package the submission.

### Sub-steps

| Step | Day-of-month | Action |
|---|---|---|
| L4-1 | 28 (early June) | Rent L4 (Lambda Labs / RunPod / GCP / AWS), install pinned stack, /health 200 |
| L4-2 | 29 | Re-measure Track A++ baseline (sanity vs Round-1 submission) |
| L4-3 | 30 - June 5 | Measure Track D winner |
| L4-4 | June 6 - 8 | Measure Track E if it landed |
| L4-5 | June 9 - 11 | 3-seed averaging on the final stack, variance characterization |
| Submit | June 12 - 14 | Repackage `submission/hf_model_repo/`, push to HF, fill organizer form |
| Buffer | June 14 - 15 | Surprise reserve |

### Submission win condition
- Lowest energy of any qualifying submission (WER <= 1.25x BF16 baseline)
- Reproducible via `submission/hf_model_repo/reproduce.sh`
- Multi-seed-stable (variance < 3% on EN500 energy)

## Suggested thread split

| Thread | Track | Daily cadence |
|---|---|---|
| Thread 1 | Track A++ + L4 binding measurement | daily until June 1, then full-time |
| Thread 2 | Track D1 (AutoRound) | daily |
| Thread 3 | Track D2 (source-build llmcompressor) | every 2-3 days |
| Thread 4 | Track D3 (RTN fallback) | low-frequency, ready as backup |
| Thread 5 | Track E (EAGLE-3) | every 2-3 days |
| Thread 6 | Track F (stretch) | only after D+E green |

Each thread updates `reports/team_status.md` per the existing template when
material progress lands or a blocker appears.

## Decision gates (running)

- 2026-05-08: D1 smoke must produce a compressed artifact within 4 hours, OR D1 escalates to source-build path investigation.
- 2026-05-08: E1 smoke must show vLLM EAGLE-3 accepts `/v1/audio/transcriptions` config without rejection, OR Track E is dead.
- 2026-05-15: at least one Track D sub-path must have produced a /health 200 served artifact, OR escalate to D3 (RTN) as the floor.
- 2026-05-25: parameter-locked submission stack identified, L4 provisioning starts.
- 2026-06-08: final Track D + Track E parameter set frozen.
- 2026-06-15: submission deadline.

## P(win) targets along the path

| Milestone | P(win) before | P(win) after |
|---|---|---|
| Track A++ locked + 13-lang | ~15% | ~20% |
| Track D1 (AutoRound) lands W4 | ~20% | ~35% |
| Track E1 alive (EAGLE works for audio) | ~35% | ~45% |
| Track D + Track E both serving on L4 | ~45% | ~55% |
| 3-seed averaged, variance < 3% | ~55% | ~60% |
