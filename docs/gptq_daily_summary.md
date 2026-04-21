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

### 15. Validated the exact checkpoint key patterns against the current module policy

The local checkpoint is not represented in just one naming scheme.

It contains two parallel safetensors layouts:

- `model.safetensors`
  - modern Voxtral-style names such as:
    - `audio_tower.*`
    - `multi_modal_projector.*`
    - `language_model.model.layers.*`
    - `language_model.model.embed_tokens.*`
    - `language_model.model.norm.*`
- `consolidated.safetensors`
  - older Mistral-style names such as:
    - `mm_streams_embeddings.*`
    - `mm_streams_embeddings.embedding_module.whisper_encoder.*`
    - `layers.*`
    - `norm.weight`

The previous `configs/experiments.yaml` policy was only partially correct:

- on `model.safetensors`
  - `quantize_first: language_model.model.layers.*` matched the decoder correctly
  - but the protect list missed:
    - `audio_tower.*`
    - `multi_modal_projector.*`
    - `language_model.model.norm.*`
- on `consolidated.safetensors`
  - `mm_streams_embeddings.*` matched the encoder-side structure correctly
  - but `quantize_first: language_model.model.layers.*` matched nothing
  - the real decoder there is:
    - `layers.*`

This means the old policy was not just stale in wording.

It was structurally incomplete across the two checkpoint layouts.

### 16. Verified what `llmcompressor.model_free_ptq` actually supports

The installed `llmcompressor` source confirms that `model_free_ptq` is a real raw-safetensors
entrypoint.

Important details verified from the installed code:

- it accepts:
  - `model_stub`
  - `save_directory`
  - `scheme`
  - `ignore`
- it works directly on safetensors files
- it does not require a model definition or Transformers support
- it quantizes weight tensors that look like linear weights
- it automatically skips modules whose names end with `norm`

But it is not a generic GPTQ-calibration path.

Its own validation code makes two important constraints explicit:

- it is weights-only PTQ
- it cannot calibrate non-dynamic activations

So this bridge is most credible for data-free or weight-only schemes, not for assuming that full
classic GPTQ calibration is suddenly solved.

### 17. Verified a practical model-free caveat

`llmcompressor`'s checkpoint discovery on the local folder enumerates both:

- `model.safetensors`
- `consolidated.safetensors`

So a naive call like:

- `model_free_ptq('models/voxtral-realtime', ...)`

would try to process both checkpoint layouts in the same run.

That is important because the folder is not a single clean representation of the model.

Any future model-free experiment should be deliberate about whether it wants:

- the modern `model.safetensors` naming scheme,
- the older `consolidated.safetensors` naming scheme,
- or both.

### 18. Updated the repo policy to match the real checkpoint structure

`configs/experiments.yaml` has now been updated so the compression policy covers both naming
schemes honestly.

The updated policy now protects:

- `audio_tower.*`
- `multi_modal_projector.*`
- `language_model.model.embed_tokens.*`
- `language_model.model.norm.*`
- `mm_streams_embeddings.*`
- `mm_streams_embeddings.embedding_module.audio_language_projection.*`
- `mm_streams_embeddings.embedding_module.whisper_encoder.*`
- `norm.*`

And it now treats both decoder layouts as candidates:

- `language_model.model.layers.*`
- `layers.*`

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
- The third blocker is now sharper too:
  - the model folder contains two tensor naming schemes
  - and a model-free path must decide which representation it is actually compressing

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
3. Decide whether the first model-free attempt should operate on `model.safetensors` or
   `consolidated.safetensors`.
4. Build the ignore list from the actual key patterns, not from stale aliases.
5. Only attempt artifact creation if that bridge starts to look technically concrete.

## Decision Rule

GPTQ does not get promoted because it sounds better in theory.

It gets promoted only if this track produces:

- a concrete compressed artifact,
- that `vLLM` can load,
- that can be smoke-tested,
- and that is worth benchmarking against FP8.

Until then, FP8 remains the mainline path and this track stays isolated.

## April 20, 2026 Follow-Up

### 19. Built the first real model-free compression artifact

The first successful `llmcompressor` bridge run operated on a stub that exposed only:

- `consolidated.safetensors`
- `config.json`
- `generation_config.json`
- `params.json`

It used:

- `scheme = FP8_DYNAMIC`
- `ignore = ['re:^mm_streams_embeddings(\\.|$)']`

The output artifact was written to:

- `models/voxtral-realtime-llmcompressor-consolidated-fp8dynamic-test`

Important result:

- the output checkpoint was materially smaller than the source checkpoint
- decoder-side tensors gained `weight_scale` sidecars
- encoder-side `mm_streams_embeddings.*` tensors stayed untouched

This proved the raw-safetensors bridge was real for weight-only compression.

### 20. Verified the first vLLM probe was blocked by tokenizer packaging, not by the artifact alone

The first `vLLM` probe on that artifact did not fail in the quantized weight loader first.

It failed earlier because the baseline `vLLM` environment still carries:

- `transformers == 4.57.6`

and the compressed artifact folder did not include the original tokenizer assets such as:

- `tekken.json`
- `processor_config.json`

Without those files, `vLLM` fell back to Hugging Face `AutoTokenizer` on the compressed model
folder and hit the known old-env error:

- `voxtral_realtime` is not recognized by that Transformers build

The important correction is:

- this was not yet evidence that the compressed artifact itself was incompatible
- it was a model-folder packaging problem in the probe setup

### 21. Proved the tokenizer bridge with a separate tokenizer path

The probe config was updated to force:

- `tokenizer: models/voxtral-realtime`
- `tokenizer_mode: mistral`

That made `vLLM` use its native Mistral/Tekken tokenizer path from the original Voxtral folder
instead of trying to build a tokenizer from the compressed artifact folder.

Result:

- the old tokenizer/config failure stopped being the fatal blocker
- `vLLM` got far enough to start loading the compressed weights

So the tokenizer problem is now solved at probe level.

### 22. Identified the first actual compressed-weight incompatibility

With the tokenizer override in place, the first artifact then failed during weight loading with:

- `KeyError: 'layers.0.ada_rms_norm_t_cond.0.weight_scale'`

This was a much better failure.

It showed:

- `vLLM` could see the quantized artifact,
- the tokenizer bridge worked,
- but one decoder-side adapter branch had been quantized too aggressively for the current
  `vLLM` compressed-tensors loader path.

### 23. Built a narrowed second artifact that excludes the failing adapter branch

To test whether the incompatibility was narrow or fundamental, a second model-free artifact was
created with a tighter ignore list:

- `re:^mm_streams_embeddings(\\.|$)`
- `re:^layers\\.\\d+\\.ada_rms_norm_t_cond(\\.|$)`

That artifact was written to:

- `models/voxtral-realtime-llmcompressor-consolidated-fp8dynamic-noada-test`

To make these reruns reproducible, the repo now includes:

- `scripts/run_model_free_ptq.py`

### 24. Verified the narrowed artifact loads in vLLM

The second `vLLM` probe used:

- model:
  - `models/voxtral-realtime-llmcompressor-consolidated-fp8dynamic-noada-test`
- tokenizer:
  - `models/voxtral-realtime`
- tokenizer mode:
  - `mistral`

This probe got materially further than the first one:

- quantized weights loaded successfully
- model loading completed
- `vLLM` moved into encoder-cache setup and torch-compile / cache preparation
- no compressed-weight loading error was emitted

The remaining old-Transformers `voxtral_realtime` warning still appears during max-position
fallback logic, but it is no longer fatal once the tokenizer path is overridden.

### 25. Current research conclusion

The best current statement is now:

- the raw `llmcompressor` bridge is real
- the first tokenizer blocker was a packaging problem, not a proof of artifact incompatibility
- the first weight-loader blocker was a narrow decoder adapter branch, not a proof that all
  compressed-tensors artifacts fail
- a tightened decoder-only surface can be loaded by `vLLM` far enough to clear model-weight load

This is the strongest GPTQ-track progress so far, even though it is still not true GPTQ.

### 26. Best next move from here

The next clean continuation point is:

1. keep using the tokenizer override path for compressed probes
2. treat `ada_rms_norm_t_cond.*` as protected for now
3. run a full startup-and-request smoke test on the narrowed artifact
4. only then decide whether this branch is benchmark-worthy against FP8

### 27. Completed the first end-to-end smoke test

The narrowed artifact was launched on:

- `http://127.0.0.1:8085`

using:

- model:
  - `models/voxtral-realtime-llmcompressor-consolidated-fp8dynamic-noada-test`
- tokenizer:
  - `models/voxtral-realtime`
- tokenizer mode:
  - `mistral`

Observed results:

- `/v1/models` returned:
  - `voxtral-realtime-llmcompressor-probe`
- `vLLM` reported:
  - application startup complete
  - supported tasks:
    - `generate`
    - `transcription`
    - `realtime`
- a real `POST /v1/audio/transcriptions` request completed successfully

The smoke-test audio was a tiny generated 1-second pure-tone WAV, so the returned transcript was
empty.

That is still a successful smoke pass.

It proves:

- the narrowed compressed artifact can boot,
- the API server can come up,
- the model can survive request handling,
- and the transcription endpoint returns a normal response instead of crashing.

### 28. What this changes

The GPTQ research branch is now past the "can it even serve one request?" stage.

The next honest move is no longer another compatibility probe.

It is a small benchmark slice against the working baselines:

1. startup time
2. GPU memory footprint
3. first-request latency
4. short transcription throughput
