# GPTQ Investigation For Voxtral Realtime

## Current Status

`gptq8_round1` is currently blocked for the local `mistralai/Voxtral-Mini-4B-Realtime-2602`
checkpoint.

The direct `vLLM` launch failure is:

```text
Cannot find the config file for gptq
```

That error is not a simple runtime tuning problem. It means `vLLM` expects a GPTQ-prepared
checkpoint, not a plain BF16 checkpoint with `--quantization gptq` added at serve time.

## What We Verified Locally

- The local checkpoint is a plain BF16/safetensors model with no GPTQ metadata or GPTQ config
  files in the model directory.
- The local WSL runtime currently has:
  - `vllm`
  - `compressed_tensors`
  - `transformers`
- The local WSL runtime does not currently have:
  - `llmcompressor`
  - `gptqmodel`
  - `auto_gptq`
- `transformers.AutoConfig.from_pretrained(...)` still fails on this checkpoint because
  `voxtral_realtime` is not recognized by the installed Transformers build.

## Why The Standard GPTQ Paths Are Blocked

There are two normal GPTQ paths in the current ecosystem:

1. Serve an already-quantized GPTQ checkpoint with `vLLM`.
2. Create a GPTQ checkpoint first with a quantization toolkit, then serve that artifact.

For Voxtral Realtime, both standard pathways are awkward right now:

- The model card says Voxtral Realtime is currently supported in `vLLM`, but not in
  `Transformers`.
- Standard `GPTQModel` and many `llmcompressor` GPTQ examples assume the model can be loaded
  through a Hugging Face model definition for calibration.
- Our local runtime confirms that this checkpoint is still not directly loadable via
  `Transformers`, which blocks the normal calibration-first GPTQ workflow.

## What Changed In The New Research Environments

The older conclusion above is no longer fully true once the environment is modernized.

### Modern Voxtral-compatible environment

In:

- `~/.venvs/voxtral-gptq-research`

we verified:

- `torch == 2.11.0+cu130`
- `transformers == 5.5.4`
- `AutoConfig.from_pretrained('models/voxtral-realtime') == ok`
- `AutoProcessor.from_pretrained('models/voxtral-realtime') == ok`
- `VoxtralRealtimeForConditionalGeneration import == ok`

So the local checkpoint is now loadable through Transformers in a sufficiently new environment.

### llmcompressor compatibility environment

In:

- `~/.venvs/voxtral-llmcompressor-research`

we verified:

- `torch == 2.10.0`
- `transformers == 4.57.6`
- `llmcompressor == 0.10.0.1`
- `llmcompressor import == ok`

But the checkpoint still fails there:

- `AutoConfig.from_pretrained('models/voxtral-realtime') == fail`
- `AutoProcessor.from_pretrained('models/voxtral-realtime') == fail`
- `VoxtralRealtimeForConditionalGeneration import == fail`

## What This Now Means

The GPTQ branch is no longer blocked by a single statement like "Transformers cannot load
Voxtral."

The real problem is narrower and more important:

- the modern Transformers line can load Voxtral,
- but the current `llmcompressor` line pins us back to an older Transformers version that cannot.

So the remaining obstacle is now an ecosystem bridge problem between:

1. a modern Voxtral-aware environment, and
2. an older llmcompressor-compatible environment.

## Exact Checkpoint-Key Validation

We directly inspected the tensor keys in both local safetensors files.

### `model.safetensors`

This file uses modern Voxtral / Transformers-style names:

- `audio_tower.*`
- `multi_modal_projector.*`
- `language_model.model.layers.*`
- `language_model.model.embed_tokens.*`
- `language_model.model.norm.*`

Important concrete counts:

- `audio_tower.*`
  - `421` keys
- `multi_modal_projector.*`
  - `2` keys
- `language_model.model.layers.*`
  - `286` keys
- `language_model.model.embed_tokens.*`
  - `1` key
- `language_model.model.norm.*`
  - `1` key

### `consolidated.safetensors`

This file uses older Mistral-style names:

- `mm_streams_embeddings.*`
- `mm_streams_embeddings.embedding_module.whisper_encoder.*`
- `layers.*`
- `norm.weight`

Important concrete counts:

- `mm_streams_embeddings.*`
  - `424` keys
- `mm_streams_embeddings.embedding_module.audio_language_projection.*`
  - `2` keys
- `mm_streams_embeddings.embedding_module.whisper_encoder.*`
  - `421` keys
- `layers.*`
  - `286` keys
- `norm.*`
  - `1` key

## What This Means For The Existing Module Policy

The older `configs/experiments.yaml` policy was not consistent across the two layouts.

### On `model.safetensors`

The old policy:

- correctly matched:
  - `language_model.model.layers.*`
- but missed:
  - `audio_tower.*`
  - `multi_modal_projector.*`
  - `language_model.model.norm.*`

### On `consolidated.safetensors`

The old policy:

- correctly matched:
  - `mm_streams_embeddings.*`
- but missed the decoder entirely because it looked for:
  - `language_model.model.layers.*`

when the real decoder there is:

- `layers.*`

So the old module policy was not just imprecise.

It was only half-right depending on which checkpoint file a tool chose to read.

## What We Verified About `llmcompressor.model_free_ptq`

The installed `llmcompressor` source confirms a real model-free entrypoint:

- `model_free_ptq(model_stub, save_directory, scheme, ignore=..., max_workers=..., device=...)`

Important verified properties:

- it works directly on safetensors files
- it does not require a model definition or Transformers support
- it supports an `ignore` list by module name
- it automatically skips modules ending in `norm`
- it sets quantization targets to `Linear`

Important limitation:

- its validation code explicitly rejects non-dynamic activation calibration
- so this path is weights-only / data-free PTQ, not a generic full GPTQ calibration workflow

## Practical Bridge Caveat

`llmcompressor`'s checkpoint discovery on `models/voxtral-realtime` enumerates both:

- `model.safetensors`
- `consolidated.safetensors`

So a naive `model_free_ptq('models/voxtral-realtime', ...)` call would try to process both
checkpoint layouts in the same run.

That means the bridge question is not just:

- "Can llmcompressor process raw safetensors?"

It is also:

- "Which checkpoint representation should we deliberately give it first?"

## What The Checkpoint Layout Tells Us

The checkpoint structure is good for selective compression:

- Audio encoder:
  - `audio_tower.*`
- Audio-to-text bridge:
  - `multi_modal_projector.*`
- Text decoder:
  - `language_model.model.layers.*`
- Text embedding/output surfaces:
  - `language_model.model.embed_tokens.weight`
  - `language_model.model.norm.weight`

This matches the project strategy of protecting the encoder and output surfaces while targeting the
decoder first.

## Most Likely Practical Paths

### Path A: Stay On The Working FP8 Track

This is the lowest-risk path and the current best submission candidate.

- It is already working in `vLLM`.
- It already improves energy and latency without hurting our measured WER.
- It works across English, Hindi, and French in our local checks.

### Path B: Investigate `llmcompressor` Model-Free PTQ

This is the most realistic route for models without a normal Transformers definition.

Why it is interesting:

- `llmcompressor` documents `model_free_ptq` for models that do not have a Transformers model
  definition.
- The same docs call out "Mistral-format model compression (experimental)" and point to
  `model_free_ptq`.
- `model_free_ptq` works directly on safetensors checkpoints and accepts an `ignore` list, which
  fits our decoder-only strategy.

Important caveat:

- The official model-free PTQ docs emphasize data-free schemes such as FP8 Block.
- The installed validation code confirms it is weights-only and cannot do non-dynamic activation
  calibration.
- They do not clearly position `model_free_ptq` as the main GPTQ path.
- So this may become a good alternative compression route, but it is not yet strong evidence that
  true GPTQ on Voxtral will work end to end without custom integration.

### Path C: Build A Custom GPTQ Workflow

This is the highest-risk path.

It likely requires one of:

- adding or vendoring a Transformers-compatible Voxtral Realtime model definition,
- implementing a custom calibration loop for the decoder layers,
- exporting GPTQ-compatible artifacts that `vLLM` can load.

This is possible in theory, but it is much more engineering-heavy than the working FP8 route.

## Recommended Next Step

If the priority is winning with the highest chance of a strong, reproducible result:

1. Keep FP8 as the primary submission baseline.
2. Treat GPTQ as a research branch, not the mainline branch.
3. Investigate `llmcompressor` next, but start with the model-free pathway and verify which
   schemes it actually supports for this checkpoint.
4. Only pursue true GPTQ further if the tooling path looks concrete after that check.

## April 20 Bridge Update

The abstract bridge question is now partially answered.

We have a real compressed-tensors artifact path that can be produced from raw Voxtral
`consolidated.safetensors`, and a narrowed version of that artifact can be loaded by `vLLM`
past the model-weight stage.

That is not true GPTQ, but it is the first end-to-end sign that this research branch can produce
something more meaningful than install logs.

## What Was Proven About Artifact Creation

Using `llmcompressor.model_free_ptq` on a stub containing only the consolidated checkpoint and
its config files, we produced:

- `models/voxtral-realtime-llmcompressor-consolidated-fp8dynamic-test`

with:

- `scheme = FP8_DYNAMIC`
- `ignore = ['re:^mm_streams_embeddings(\\.|$)']`

This proved:

- `llmcompressor` can process the local Voxtral checkpoint in a model-free mode
- the encoder can be excluded cleanly
- the decoder receives compressed-tensors sidecars such as `weight_scale`

## What The First vLLM Probe Actually Showed

The first probe did not invalidate the artifact format.

It failed earlier than that because the compressed output folder did not include the original
tokenizer assets:

- `tekken.json`
- `processor_config.json`

The baseline `vLLM` env is still on:

- `transformers == 4.57.6`

so when `vLLM` tried to build a tokenizer directly from the compressed artifact folder, it hit the
old fatal error:

- `voxtral_realtime` is not recognized by this Transformers build

That means the first probe was blocked by tokenizer packaging, not yet by compressed decoder
weights.

## What Fixed The Tokenizer Blocker

The probe was then re-run with:

- `--tokenizer models/voxtral-realtime`
- `--tokenizer-mode mistral`

This matters because the original Voxtral folder contains:

- `consolidated.safetensors`
- `tekken.json`

which is enough for `vLLM` to auto-select its native Mistral/Tekken tokenizer path instead of
falling back to Hugging Face `AutoTokenizer`.

With that override in place, the old tokenizer failure stopped being the main blocker.

## First True Compressed-Weight Failure

Once the tokenizer path was fixed, the first artifact failed deeper in model loading with:

- `KeyError: 'layers.0.ada_rms_norm_t_cond.0.weight_scale'`

This is a much more useful failure than the earlier tokenizer/config issue.

It implies:

- `vLLM` can see the compressed artifact,
- the tokenizer bridge works,
- but one decoder-side adapter branch is not supported by the current compressed-tensors loading
  path as quantized.

The failing branch is the `ada_rms_norm_t_cond` sequential in each Mistral decoder layer.

## Narrowed Second Artifact

A second artifact was then created with a tighter ignore list:

- `re:^mm_streams_embeddings(\\.|$)`
- `re:^layers\\.\\d+\\.ada_rms_norm_t_cond(\\.|$)`

Output path:

- `models/voxtral-realtime-llmcompressor-consolidated-fp8dynamic-noada-test`

This experiment was made reproducible via:

- `scripts/run_model_free_ptq.py`

## What The Second vLLM Probe Proved

The second probe used:

- model:
  - `models/voxtral-realtime-llmcompressor-consolidated-fp8dynamic-noada-test`
- tokenizer:
  - `models/voxtral-realtime`
- tokenizer mode:
  - `mistral`

Observed behavior:

- the quantized weights loaded successfully
- `vLLM` reported model loading completed
- the engine advanced into encoder-cache setup
- the engine advanced into torch-compile / cache preparation
- no compressed-weight load error was emitted

The old `voxtral_realtime` warning still appears during a non-fatal config fallback path, but it
no longer stops startup once the tokenizer override is used.

## What This Means Now

The current GPTQ-track truth is:

- a raw model-free compression bridge exists
- the tokenizer issue is solvable without mutating the FP8 environment
- the first load failure was not fundamental
- `ada_rms_norm_t_cond.*` should currently be treated as protected
- a narrowed decoder-only compressed artifact is now viable enough for a real smoke test

## Best Next Step

The best next step is no longer "install more GPTQ packages."

It is:

1. use the narrowed `noada` artifact,
2. keep the tokenizer override,
3. run a full startup-and-request smoke test,
4. then decide whether this branch deserves benchmark time against FP8.

## Smoke Test Result

That smoke test has now been run.

Server configuration:

- model:
  - `models/voxtral-realtime-llmcompressor-consolidated-fp8dynamic-noada-test`
- tokenizer:
  - `models/voxtral-realtime`
- tokenizer mode:
  - `mistral`
- port:
  - `8085`

Observed end-to-end behavior:

- `/v1/models` returned the served model id:
  - `voxtral-realtime-llmcompressor-probe`
- `vLLM` completed application startup
- `vLLM` exposed:
  - `/v1/audio/transcriptions`
- a real transcription request completed successfully

The request audio was a tiny generated 1-second pure-tone WAV, so the returned transcript was
empty.

That empty transcript is expected for non-speech audio.

The important point is that the request path completed without:

- tokenizer failure
- compressed-weight loading failure
- engine crash
- API crash

## Updated Conclusion

The current branch has now crossed the minimum viability threshold for benchmarking.

It is still not "true GPTQ."

But it is now a working:

- model-free `llmcompressor`
- `compressed-tensors`
- decoder-narrowed
- `vLLM`-servable

artifact path for Voxtral Realtime.

## Real Next Step

The right next step is a small benchmark slice, not more compatibility spelunking:

1. startup time
2. memory footprint
3. first-request latency
4. short transcription throughput

Only after that should we decide whether this research branch earns more optimization work.
