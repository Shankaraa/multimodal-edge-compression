# GPTQ Track Summary

## Bottom Line

The GPTQ track is not a live submission candidate.

More precisely, it is not even a "true GPTQ" success story yet.

The best result so far is a GPTQ-adjacent `llmcompressor` plus `compressed-tensors` bridge that
can boot in `vLLM`, but it fails real ASR evaluation and should stay research-only.

## The Frame That Matters

The original question was slightly wrong.

The real problem was never:

- "Which `vLLM` flag enables GPTQ?"

The real problem was:

- "Can we produce a quantized artifact for Voxtral Realtime that `vLLM` can actually load and
  transcribe with correctly?"

That reframing matters because the direct `vLLM` error:

```text
Cannot find the config file for gptq
```

already told us the plain checkpoint was not enough.

## What Was Proven

### 1. Plain Voxtral is not serve-time GPTQ-ready

- `vLLM` does not convert the local BF16 checkpoint into GPTQ automatically.
- A plain `--quantization gptq` style launch is not the path.

### 2. The old "Transformers cannot load Voxtral" claim became stale

In a modern isolated environment, Voxtral Realtime does load through Transformers.

That means the problem is narrower than it first looked.

### 3. The ecosystem split is real

Two truths were established at the same time:

- a modern Voxtral-aware environment can recognize the checkpoint
- the current `llmcompressor` line depends on an older Transformers line

So the blocker became an ecosystem-bridge problem, not just a missing package problem.

### 4. A model-free compression bridge does exist

Using `llmcompressor.model_free_ptq`, the repo produced a real compressed artifact from the raw
Voxtral checkpoint.

Important point:

- this is not the same as proving end-to-end GPTQ
- it is evidence that a weights-only compressed-tensors path is technically possible

### 5. A narrowed artifact can load and serve

The strongest GPTQ-side result is the narrowed artifact:

- `models/voxtral-realtime-llmcompressor-consolidated-fp8dynamic-noada-test`

With the tokenizer override:

- `--tokenizer models/voxtral-realtime`
- `--tokenizer-mode mistral`

that artifact can:

- load in `vLLM`
- start the API server
- answer a transcription request

That crossed the minimum "is there a real artifact path at all?" threshold.

## What Failed

The artifact is technically servable but functionally broken for ASR.

The mini benchmark on the same `en_us limit 5` slice showed:

- BF16:
  - normalized `WER = 4.81%`
  - first request latency `2.53 s`
  - energy `2793.12 J`
- FP8 round 1:
  - normalized `WER = 4.81%`
  - first request latency `1.71 s`
  - energy `1891.11 J`
- narrowed compressed artifact:
  - normalized `WER = 100.00%`
  - empty predictions `5 / 5`
  - first request latency `10.14 s`
  - energy `8825.33 J`

That is not a near miss.

It means the current GPTQ-side artifact is unusable for the competition unless a specific failure
is fixed.

## Most Likely Current Failure

The strongest current clue is the repeated server warning:

```text
Realtime model received empty multimodal embeddings for 1 input tokens
```

So the current branch is best understood as:

- artifact creation: proven
- `vLLM` loading: proven
- transcription quality: broken

The bottleneck is no longer "can we quantize anything at all?"

It is "why does the compressed path collapse multimodal embeddings during real ASR requests?"

## What This Means For Project Strategy

The repo's two-track split is correct.

- FP8 is the mainline and submission path.
- GPTQ stays isolated as research.
- The current GPTQ-side result should be documented, not promoted.

This is why the honest repo-wide statement is:

- FP8 is already working, benchmarked, and reproducible.
- GPTQ-side work is only interesting again if someone wants to debug the empty-embedding failure.

## Exact Next Actions

### Default action

Do not spend more benchmark time on the current GPTQ-side artifact.

### If the GPTQ-side branch is resumed

Resume from the narrowed artifact path and the exact known-good probe setup:

- artifact:
  - `models/voxtral-realtime-llmcompressor-consolidated-fp8dynamic-noada-test`
- tokenizer override:
  - `--tokenizer models/voxtral-realtime`
  - `--tokenizer-mode mistral`
- reproducible builder:
  - `scripts/run_model_free_ptq.py`

Treat these surfaces as protected:

- `mm_streams_embeddings.*`
- `layers.*.ada_rms_norm_t_cond.*`

Then use this decision order:

1. Reproduce the empty-multimodal-embedding failure on a real speech clip.
2. Find whether the failure is caused by a specific compressed module group, not by the whole
   artifact.
3. Do not run broader benchmarks again until the branch produces non-empty speech transcripts.
4. Only compare against FP8 after the branch clears that quality gate.

### What not to do

- do not treat package installation as progress by itself
- do not call the current bridge "true GPTQ"
- do not mutate the working FP8 environment to keep this branch alive
- do not promote the branch because startup time or memory looks better while transcription is
  broken

## Decision Rule

This branch only becomes strategically relevant again if it produces all of the following:

- a reproducible compressed artifact
- a stable `vLLM` load path
- non-empty, correct speech transcriptions
- benchmark results that are competitive with FP8

Until then, GPTQ is a documented research outcome, not a competition path.

## Source Notes

This summary compresses the current state from:

- `docs/gptq_research_track.md`
- `docs/gptq_investigation.md`
- `docs/gptq_daily_summary.md`
- `docs/two_track_strategy.md`
- `README.md`
