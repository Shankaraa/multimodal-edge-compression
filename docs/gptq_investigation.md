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
