# Guide Notes

These notes capture the main ideas from `voxtral realtime compression guide.pdf` so the project
does not depend on rereading the PDF every time.

## Competition Facts

- Track: audio-to-text compression.
- Model: `mistralai/Voxtral-Mini-4B-Realtime-2602`.
- Evaluation hardware: NVIDIA L4 24 GB.
- Metric: lowest energy while staying within the WER quality floor.
- Quality rule: WER can grow to at most `1.25x` baseline.
- Round 1 window: April 27, 2026 to May 4, 2026.
- Round 2 window: May 20, 2026 to June 15, 2026.

## Architecture Summary

- Audio encoder: about `0.6B` params, 32 layers, dimension 1280.
- Adapter: about `0.05B` params, small bridge from audio to language space.
- Text decoder: about `3.4B` params, 26 layers, dimension 3072.
- Decoder uses GQA with 32 query heads and 8 KV heads.
- Default `max_model_len` is 131072, which is much larger than the benchmark clips need.

## Compression Thesis

- The encoder is sensitive. Avoid quantizing it first.
- The adapter is tiny. Leave it alone.
- The decoder holds most of the parameter count. Quantize this first.
- The challenge is mostly about lowering joules, not making the model fit in memory.

## Safe Module Policy

Protect:

- `whisper_encoder.*`
- `mm_streams_embeddings.*`
- `language_model.model.embed_tokens.*`
- `language_model.lm_head.*`
- `ada_*`

Quantize first:

- `language_model.model.layers.*`

## Round 1 Plan

- Establish BF16 baseline.
- Try FP8 in `vLLM` if supported on the target stack.
- Otherwise use GPTQ 8-bit, ideally decoder-first.
- Validate WER on FLEURS before caring about submission packaging.

## Round 2 Plan

- Move to GPTQ 4-bit decoder-only.
- Try FP8 KV cache.
- Reduce `max_model_len` to something closer to the benchmark needs, such as 16384.
- Consider AWQ, mixed precision, pruning, or delay tuning only after we have quality headroom.

## Risks Called Out By The Guide

- `vLLM` nightly is required.
- Encoder quantization can break WER badly.
- FP8 support may depend on runtime maturity.
- AWQ support may require extra validation.
- Some examples in the guide are strategy sketches, not guaranteed drop-in scripts.
