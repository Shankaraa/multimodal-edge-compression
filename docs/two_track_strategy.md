# Two-Track Strategy For Winning The Competition

## Purpose

This document defines the two parallel tracks we should run from this point onward:

1. a practical mainline track that keeps producing stable, benchmarked results, and
2. a research track that investigates the more theoretically promising GPTQ path.

The goal is simple:

- do not risk the competition on an unstable branch
- do not abandon the branch that could produce a stronger compression result

## Why We Are Splitting Into Two Tracks

We now know two important things at the same time:

### What is already working

- The local Voxtral Realtime model serves successfully through `vLLM` in WSL.
- The evaluation pipeline is now much stronger than it was initially.
- The concurrency issue is controlled by serializing transcription requests in the shared API
  helper.
- The quiet-audio issue is controlled by boosting low-level samples before evaluation and
  recording audio diagnostics.
- `fp8_round1` is working and already looks genuinely competitive.

### What is still blocked

- `gptq8_round1` does not launch just by setting `--quantization gptq`.
- The direct `vLLM` failure is:

```text
Cannot find the config file for gptq
```

- The local checkpoint is a plain BF16/safetensors checkpoint, not a GPTQ-prepared checkpoint.
- The current local runtime still cannot load `voxtral_realtime` through standard
  `transformers.AutoConfig`.
- That means GPTQ is not a small serving tweak. It is a model-preparation problem.

So the right strategy is:

- keep shipping measurable progress on the working path
- keep researching the theoretically stronger path without letting it derail the project

## Current Findings

## Model And Runtime Findings

- The local model path is:
  - `models/voxtral-realtime`
- The local machine behaves best with:
  - `max_model_len: 8192`
- The practical memory setting that worked for FP8 is:
  - `gpu_memory_utilization: 0.85`
- The current working local server is:
  - `http://localhost:8082/v1`

## Evaluation Findings

- Empty transcripts were not random noise.
- They were deterministic on very quiet samples.
- The evaluator now:
  - boosts quiet samples before WAV export
  - records per-sample audio diagnostics
  - counts empty predictions explicitly

This makes our newer reports much more trustworthy than the older ones.

## Performance Findings

### BF16 quiet-audio-aware reference

- `en_us limit20`
  - `WER = 22.20%`
  - `empty_prediction_count = 0`
  - `elapsed_seconds = 46.26`
  - `energy_joules = 8112.90`

### FP8 current best practical path

- `en_us limit20`
  - `WER = 21.97%`
  - `empty_prediction_count = 0`
  - `elapsed_seconds = 35.21`
  - `energy_joules = 4952.89`

### FP8 multilingual spot checks

- `hi_in limit5`
  - `WER = 26.83%`
  - `empty_prediction_count = 0`
- `fr_fr limit5`
  - `WER = 23.18%`
  - `empty_prediction_count = 0`

## Interpretation

- FP8 is already producing meaningful efficiency gains.
- FP8 is not obviously hurting quality on our current checks.
- FP8 is the current best submission-grade path.
- GPTQ is still interesting because it aligns better with the decoder-first compression strategy,
  but it is not ready to benchmark yet.

## Track A: Mainline FP8 Track

## Mission

Turn the currently working FP8 path into the strongest reproducible competition submission we can
build quickly.

## Why This Track Exists

- It already works.
- It already has benchmark wins.
- It already has multilingual signal.
- It gives us a safe path to a good final result even if GPTQ research stalls.

## Main Objective

Strengthen the FP8 story until it is hard to beat:

- better quality evidence
- better multilingual evidence
- cleaner reports
- cleaner submission narrative

## What This Track Should Do

### 1. Expand evaluation coverage carefully

- Run additional languages beyond English, Hindi, and French.
- Prefer one harder language next for broader confidence.
- Keep sample counts small and targeted first, then grow where useful.

### 2. Keep reports apples-to-apples

- Compare everything against the quiet-audio-aware BF16 reference, not the older empty-containing
  reports.
- Use the same evaluation pipeline and the same energy wrapper.
- Do not mix old and new evaluator results in summary comparisons.

### 3. Strengthen the submission evidence

- Produce a concise benchmark matrix:
  - BF16 vs FP8
  - WER
  - energy
  - elapsed time
  - languages tested
- Capture any remaining runtime notes clearly:
  - WSL caveats
  - memory settings
  - current port usage

### 4. Keep the runtime stable

- Avoid overlapping transcription requests.
- Keep using the shared API helper.
- Preserve the current working `fp8_round1` config unless a change is clearly justified.

## Mainline Success Criteria

This track is succeeding if:

- FP8 keeps matching or nearly matching BF16 quality
- FP8 keeps beating BF16 on time or energy
- multilingual checks remain stable
- we can present the results as a clean, reproducible improvement story

## Track B: GPTQ Research Track

## Mission

Determine whether decoder-focused GPTQ or GPTQ-adjacent compression is realistically achievable for
Voxtral Realtime in this environment.

## Why This Track Exists

- GPTQ is still theoretically more promising than FP8 for aggressive decoder compression.
- It fits the original strategy better:
  - protect encoder
  - compress decoder
- If it works, it may outperform FP8 on size and possibly energy.

## Current GPTQ Blocker

The current blocker is not GPU memory.
The current blocker is not `vLLM` flags.

The current blocker is:

- we do not have GPTQ-formatted artifacts
- the normal GPTQ preparation toolchain is awkward because Voxtral Realtime is not directly
  loadable through standard Transformers in this environment

## What This Track Should Do

### 1. Treat GPTQ as a preparation workflow

Do not assume `vLLM` can create GPTQ weights from the plain checkpoint.

Assume instead that GPTQ requires:

- a quantization/preparation tool
- output artifacts/configs
- then a later serve step

### 2. Investigate `llmcompressor`

This is the highest-value next research step because it is the most plausible bridge between:

- a safetensors checkpoint
- missing normal Transformers support
- our need for selective compression

Specific questions to answer:

- Can `llmcompressor` be installed cleanly in the WSL env?
- Does its `model_free_ptq` path work on Voxtral-like checkpoints?
- Which compression schemes are actually supported there?
- Can we ignore:
  - `audio_tower.*`
  - `multi_modal_projector.*`
  - `language_model.model.embed_tokens.*`
- Can we target only:
  - `language_model.model.layers.*`

### 3. Keep the scope narrow

The research branch should not try to solve everything at once.

Start with:

- one installation step
- one minimal proof-of-compatibility
- one minimal recipe test

Do not jump straight into:

- full benchmark runs
- complex calibration datasets
- heavy custom architecture rewrites

### 4. Escalate only if the path is concrete

Only go deeper into custom GPTQ engineering if the tooling starts to look real.

Examples of “real” signs:

- `llmcompressor` can open the checkpoint
- selective ignore lists are accepted
- a compressed artifact is actually emitted
- `vLLM` can then read that artifact

If those signs do not appear, do not sink the schedule into a dead branch.

## Research Success Criteria

This track is succeeding if it answers one of these clearly:

1. yes, GPTQ or GPTQ-adjacent compression is feasible here and we can move to benchmarks
2. no, the tooling path is too immature for this checkpoint, and FP8 should stay primary

Either answer is valuable.

## Shared Rules For Both Tracks

These rules should stay true in both threads:

- Protect the audio encoder first.
- Protect the multimodal bridge unless strong evidence says otherwise.
- Treat the decoder as the main compression target.
- Use the quiet-audio-aware evaluator for all new comparisons.
- Keep the daily log honest and chronological.
- Do not overwrite the working FP8 runtime state casually.
- Prefer small, verifiable steps over large speculative changes.

## Recommended Thread Split

## Thread 1: Mainline Execution Thread

Use this thread for:

- FP8 evaluations
- multilingual checks
- energy comparisons
- report cleanup
- README and submission narrative work

Suggested mindset:

- optimize for stable wins
- optimize for reproducible evidence
- optimize for submission readiness

## Thread 2: GPTQ Research Thread

Use this thread for:

- `llmcompressor` installation
- compatibility tests
- model-free PTQ experiments
- selective compression recipe design
- GPTQ readiness investigation

Suggested mindset:

- optimize for technical truth
- optimize for fast go/no-go answers
- optimize for not breaking the mainline path

## Immediate Next Steps

### For The Mainline FP8 Track

1. Run one more carefully chosen multilingual check or start organizing the benchmark summary table.
2. Keep FP8 as the active working server.
3. Keep all new comparisons anchored to the quiet-audio-aware BF16 reference.

### For The GPTQ Research Track

1. Install `llmcompressor` in the WSL environment.
2. Verify whether the model-free pathway can even see the Voxtral checkpoint.
3. Test whether a decoder-only ignore/target pattern is expressible.
4. Decide quickly whether the branch is promising enough to continue.

## Final Decision Rule

If GPTQ becomes real and benchmarkable, we promote it.

If GPTQ stays blocked or vague, we do not chase it emotionally.
We keep FP8 as the main competition path and use the GPTQ work only if it becomes concrete enough
to help us win.
