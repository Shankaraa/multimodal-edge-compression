# GPTQ Research Track

## Purpose

This track is the research branch.

Its job is to determine whether GPTQ or GPTQ-adjacent decoder compression is realistically
achievable for Voxtral Realtime in this environment and, if yes, whether it can beat FP8.

## Why This Track Exists

GPTQ is still important because it is more promising in theory:

- it fits the decoder-first compression strategy better
- it could provide a stronger compression story than FP8
- it may beat FP8 on model size and possibly energy if the tooling path works

But right now GPTQ is not operational. It is a research problem.

## Current Blocker

The current blocker is not GPU memory and not port conflicts.

The current blocker is:

```text
Cannot find the config file for gptq
```

This tells us that `vLLM` expects GPTQ-prepared artifacts and is not converting the plain Voxtral
checkpoint into GPTQ automatically at serve time.

## What We Verified

## Checkpoint State

The local model directory contains plain BF16/safetensors weights, including:

- `model.safetensors`
- `consolidated.safetensors`
- `config.json`
- `params.json`

It does not contain GPTQ-specific config artifacts.

## Tooling State

In the current WSL runtime:

- installed:
  - `vllm`
  - `compressed_tensors`
  - `transformers`
- missing:
  - `llmcompressor`
  - `gptqmodel`
  - `auto_gptq`

## Transformers State

`transformers.AutoConfig.from_pretrained(...)` still fails on this checkpoint because
`voxtral_realtime` is not recognized by the installed Transformers build.

That matters because many standard GPTQ workflows assume the model can be loaded through normal
Transformers definitions for calibration.

## Checkpoint Layout Findings

The model structure is actually favorable for selective compression.

Important module boundaries:

- Audio encoder:
  - `audio_tower.*`
- Multimodal bridge:
  - `multi_modal_projector.*`
- Decoder:
  - `language_model.model.layers.*`
- Output and embedding surfaces:
  - `language_model.model.embed_tokens.*`
  - `language_model.model.norm.*`

This matches our intended strategy:

- protect the encoder
- protect the bridge
- protect embedding/output surfaces
- target decoder layers first

## What This Means

GPTQ is not something we can unlock with a better `vLLM` serve command.

It is a preparation pipeline problem that likely requires one of:

1. generating GPTQ-ready artifacts with a compatible quantization toolkit
2. using a model-free compression path that works directly on safetensors checkpoints
3. building a more custom calibration and export workflow

## Most Likely Path Forward

The most realistic next branch to investigate is:

- `llmcompressor`

Why:

- it has a model-free compression story
- it is the best candidate for checkpoints that are not normal Transformers models
- it may allow selective ignore/target patterns that match our decoder-only plan

## Research Questions

This track should answer these questions in order:

### 1. Can `llmcompressor` be installed cleanly in the WSL env?

If not, that is an immediate signal about the branch difficulty.

### 2. Can `llmcompressor` see the Voxtral checkpoint at all?

If it cannot open or reason about the checkpoint, the branch is probably not worth forcing.

### 3. Can we express a decoder-only compression recipe?

Ideal target split:

- ignore:
  - `audio_tower.*`
  - `multi_modal_projector.*`
  - `language_model.model.embed_tokens.*`
  - `language_model.model.norm.*`
- target:
  - `language_model.model.layers.*`

### 4. What schemes are actually available?

Important point:

- the model-free path may support useful schemes that are not true GPTQ
- even if strict GPTQ fails, a GPTQ-adjacent compression path could still help us win

### 5. Can the output artifact be served by `vLLM`?

This is the real gate.

If a tool emits an artifact that `vLLM` cannot load, the branch is not ready for benchmarking.

## Approach

## Phase 1: Compatibility

- install tooling
- test imports
- test checkpoint visibility
- do not benchmark anything yet

## Phase 2: Minimal recipe

- try the smallest controlled model-free compression experiment possible
- prefer one short, decoder-focused recipe
- do not attempt the full competition workflow immediately

## Phase 3: Artifact validation

- inspect output files
- determine whether they are GPTQ-like, compressed-tensors-like, or something else
- test whether `vLLM` can load them

## Phase 4: Only then benchmark

If and only if the artifact serves successfully:

- run a small smoke test
- run one short FLEURS check
- compare against FP8

## What Not To Do

- do not assume GPTQ is just a serve-time flag
- do not sink a lot of time into custom architecture rewrites too early
- do not break the working FP8 path while exploring this branch
- do not treat “interesting theory” as a reason to ignore weak tooling reality

## Success Criteria

This track is succeeding if it answers one of these clearly:

1. yes, a GPTQ or GPTQ-adjacent path is concretely viable and ready for benchmarks
2. no, the current tooling is too immature for this checkpoint and FP8 should remain primary

Both answers are valuable.

## Files That Matter Most

- `configs/vllm/gptq_round1.yaml`
- `docs/gptq_investigation.md`
- `models/voxtral-realtime/config.json`
- `models/voxtral-realtime/model.safetensors`
- `configs/experiments.yaml`

## Immediate Tasks For This Track

1. Install `llmcompressor` in the WSL environment.
2. Check whether the model-free compression path can see the Voxtral checkpoint.
3. Test whether a decoder-only ignore/target recipe is expressible.
4. Decide quickly whether the path is viable enough to keep investing in.

## Decision Rule

GPTQ stays a research branch until it produces a real, loadable, benchmarkable artifact.

Until that happens, FP8 remains the main competition path.
