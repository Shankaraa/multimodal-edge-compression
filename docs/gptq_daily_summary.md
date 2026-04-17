# GPTQ Daily Summary

## Date

April 17, 2026

## Track

GPTQ research track for Voxtral Realtime.

## Objective For Today

Take ownership of the GPTQ branch, separate it from the active FP8 mainline work, and verify
whether the current GPTQ blocker is still architectural or partly just an environment mismatch.

## What Was Done Today

### 1. Reviewed the repo-level and track-level docs

The following files were read together so the GPTQ branch could be framed against the real project
state rather than old assumptions:

- `README.md`
- `daily_document.md`
- `docs/two_track_strategy.md`
- `docs/gptq_research_track.md`
- `docs/gptq_investigation.md`
- `models/voxtral-realtime/README.md`

### 2. Reframed the current GPTQ blocker

The main conclusion still holds:

- GPTQ is not unlocked by a better `vLLM` serve flag
- the plain local checkpoint is not a GPTQ-ready checkpoint
- `vLLM` still expects GPTQ-prepared artifacts if we want to serve with `quantization: gptq`

So this remains a preparation-pipeline problem, not a port or GPU-memory problem.

### 3. Verified the actual current WSL runtime package state

The active WSL runtime for the project is still behind the model's stated expectation.

Current WSL package state that was verified directly:

- `transformers == 4.57.6`
- `vllm == 0.19.1rc1.dev302+g68be0f853.cu130`
- `compressed_tensors == 0.14.0.1`
- `mistral_common == 1.11.0`
- missing:
  - `llmcompressor`
  - `auto_gptq`
  - `gptqmodel`
  - `optimum`

This matters because the earlier "Transformers cannot load Voxtral Realtime here" finding may now
be partly stale rather than purely architectural.

### 4. Verified the checkpoint metadata against the model card

The local checkpoint metadata and model card point to a newer Transformers line than the active
research environment is using.

Important facts:

- `models/voxtral-realtime/config.json` declares:
  - `architectures: VoxtralRealtimeForConditionalGeneration`
  - `model_type: voxtral_realtime`
  - `transformers_version: 5.2.0.dev0`
- the local model card says Voxtral Realtime support starts at:
  - `transformers >= 5.2.0`

So before treating GPTQ as blocked by architecture alone, the first honest step is to test the
checkpoint under a modern Transformers build in a separate research environment.

### 5. Identified one likely stale selective-compression assumption

`configs/experiments.yaml` still protects legacy-looking module patterns such as:

- `whisper_encoder.*`
- `mm_streams_embeddings.*`

But the newer GPTQ notes and the checkpoint structure are expressed in terms of:

- `audio_tower.*`
- `multi_modal_projector.*`
- `language_model.model.layers.*`

This is not a cosmetic mismatch.

If the ignore/target patterns are stale, any future decoder-only compression recipe could be
silently wrong.

### 6. Established the environment-isolation rule for this track

The GPTQ branch should not mutate the active FP8 runtime.

Rule for this track:

- do not install GPTQ research packages into `~/.venvs/voxtral-baseline`
- treat the FP8 environment as mainline/stable
- use a separate WSL environment for GPTQ research

Recommended dedicated environment name:

- `~/.venvs/voxtral-gptq-research`

To support that split, this repo now includes:

- `requirements-linux-gptq-research.txt`

### 7. Created the isolated GPTQ research environment in WSL

A separate WSL environment was created successfully at:

- `~/.venvs/voxtral-gptq-research`

This environment is separate from the active FP8 runtime and does not modify:

- `~/.venvs/voxtral-baseline`

### 8. Installed the modern Voxtral-compatible stack in the isolated env

The following modern stack was installed successfully into the new GPTQ research environment:

- `torch == 2.11.0+cu130`
- `transformers == 5.5.4`
- `mistral_common == 1.11.0`
- `av == 17.0.0`
- `bitsandbytes == 0.49.2`
- `optimum == 2.1.0`

This gives the research track a clean environment for checkpoint visibility checks without
touching the FP8 branch.

### 9. Re-tested the old Transformers assumption in the new env

The earlier repo-era assumption that Voxtral Realtime is not loadable in Transformers is now
outdated for a modern environment.

The following checks passed in `~/.venvs/voxtral-gptq-research` against the local checkpoint:

- `AutoConfig.from_pretrained('models/voxtral-realtime')`
- `AutoProcessor.from_pretrained('models/voxtral-realtime')`
- `from transformers import VoxtralRealtimeForConditionalGeneration`

Observed results:

- `AutoConfig = VoxtralRealtimeConfig`
- `AutoProcessor = VoxtralRealtimeProcessor`
- `VoxtralRealtimeForConditionalGeneration = import ok`

So the correct statement now is:

- the old baseline environment could not load Voxtral through Transformers
- the new isolated modern environment can

### 10. Found a hard package conflict in the GPTQ ecosystem

The most important finding from the install work is that the original "one GPTQ research env"
idea is wrong.

There is a real dependency conflict:

- Voxtral Realtime support in the new env works with:
  - `transformers >= 5.2.0`
- `llmcompressor == 0.10.0.1` requires:
  - `transformers >= 4.56.1, <= 4.57.6`

That means:

- a modern Voxtral-compatible Transformers stack
- and the current `llmcompressor` line

do not coexist cleanly in the same environment.

### 11. Found two package-level GPTQ friction points

The classic GPTQ-side packages are also not drop-in clean in this modern environment.

Observed problems:

- `GPTQModel` failed during build isolation because its build step could not detect the already
  installed Torch package.
- `auto-gptq` fell into resolver backtracking toward older package combinations, which is a strong
  sign that the classic GPTQ stack is still tied to older Transformers-era assumptions.

### 12. Created a second isolated environment for the llmcompressor branch

Because the package constraints conflict directly, a second dedicated compatibility environment was
created at:

- `~/.venvs/voxtral-llmcompressor-research`

This environment is intentionally separate from:

- `~/.venvs/voxtral-baseline`
- `~/.venvs/voxtral-gptq-research`

### 13. Installed the llmcompressor compatibility stack

The llmcompressor environment installed successfully, but only by moving onto an older dependency
line than the modern Voxtral-compatible environment.

Verified package state in `~/.venvs/voxtral-llmcompressor-research`:

- `torch == 2.10.0`
- `transformers == 4.57.6`
- `llmcompressor == 0.10.0.1`
- `compressed_tensors == 0.14.0.1`
- `datasets == 4.6.0`
- `huggingface_hub == 0.36.2`
- `safetensors == 0.7.0`

Runtime checks also passed at a basic level:

- `torch.cuda.is_available() == True`
- `torch.version.cuda == 12.8`
- `import llmcompressor == ok`

### 14. Verified the split-env thesis directly

The split between the two environments is now proven by direct behavior rather than inference.

In the modern Voxtral-compatible env:

- `AutoConfig.from_pretrained('models/voxtral-realtime') == ok`
- `AutoProcessor.from_pretrained('models/voxtral-realtime') == ok`
- `VoxtralRealtimeForConditionalGeneration import == ok`

In the llmcompressor env:

- `AutoConfig.from_pretrained('models/voxtral-realtime') == fail`
- `AutoProcessor.from_pretrained('models/voxtral-realtime') == fail`
- `VoxtralRealtimeForConditionalGeneration import == fail`

So the current research reality is:

- the modern env can see the checkpoint but does not currently host `llmcompressor`
- the llmcompressor env can import `llmcompressor` but cannot natively recognize Voxtral

That is the actual bottleneck now.

## Current GPTQ State

- GPTQ remains a research-only branch.
- There is still no real GPTQ artifact to serve or benchmark.
- The direct `vLLM` blocker remains the same in spirit:
  - a plain Voxtral checkpoint is not enough for `quantization: gptq`
- The old Transformers incompatibility claim is no longer correct in a modern isolated env.
- The new blocker is sharper:
  - the modern Voxtral-compatible Transformers line
  - and the current `llmcompressor` line
  - conflict directly on required `transformers` versions.
- The second blocker is now experimentally confirmed:
  - the llmcompressor-compatible environment does not currently recognize `voxtral_realtime`
  - so installation alone is not enough to make the checkpoint usable there

## What This Means

The next GPTQ decision should be based on toolchain truth, not old failure messages.

The right order is:

1. use a separate research environment,
2. distinguish the modern Voxtral-compatible path from the legacy GPTQ-tooling path,
3. use the modern env to validate checkpoint visibility and model structure,
4. if we still want `llmcompressor`, give it its own compatibility env instead of forcing it into
   the modern env,
5. only then decide whether the branch is viable.

## Planned Next Steps

1. Keep `~/.venvs/voxtral-gptq-research` as the modern Voxtral-compatible env.
2. Keep `~/.venvs/voxtral-llmcompressor-research` as the old-line llmcompressor env.
3. Validate the actual module names before writing any decoder-only recipe.
4. Decide whether a bridge is possible between the modern checkpoint-loading env and the
   llmcompressor env.
5. Only attempt artifact creation if that bridge starts to look technically concrete.

## Decision Rule

GPTQ does not get promoted because it sounds better in theory.

It gets promoted only if this track produces:

- a concrete compressed artifact,
- that `vLLM` can load,
- that can be smoke-tested,
- and that is worth benchmarking against FP8.

Until then, FP8 remains the mainline path and this track stays isolated.
