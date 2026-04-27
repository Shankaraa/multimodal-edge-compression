# Track B LLM-Compressor Recipe Proposal

## Conclusion

Use `GPTQModifier`, not another `model_free_ptq` artifact, for the Track B candidate.

The prior `model_free_ptq` run proved that compressed-tensors can boot, but it cannot express the
candidate we need: calibrated W4A16 decoder weights plus artifact-declared FP8 KV cache. It also
writes broad `targets: Linear` metadata without enough cross-layout protection for Voxtral's HF
calibration names and vLLM serving names.

## Weight Layout Used

The relevant layouts from `docs/gptq_investigation.md` are:

- HF / `model.safetensors`
  - protect `audio_tower.*`
  - protect `multi_modal_projector.*`
  - protect `language_model.model.embed_tokens.*`
  - protect `language_model.model.norm.*`
  - quantize only decoder projection weights under `language_model.model.layers.*`
- Consolidated / `consolidated.safetensors`
  - protect `mm_streams_embeddings.*`
  - protect `mm_streams_embeddings.embedding_module.audio_language_projection.*`
  - protect `mm_streams_embeddings.embedding_module.whisper_encoder.*`
  - protect `mm_streams_embeddings.embedding_module.tok_embeddings.*`
  - protect `norm.*`
  - quantize only decoder projection weights under `layers.*`

The extra realtime-specific protection is mandatory:

- HF name: `language_model.model.layers.<n>.ada_rms_norm.*`
- vLLM / consolidated name: `layers.<n>.ada_rms_norm_t_cond.*`

Those layers condition the decoder on realtime timing. They are small, not worth compressing, and
already caused a compressed-weight load failure when quantized in the first artifact.

## Proposed Recipe

Review file:

- `configs/llmcompressor/voxtral_track_b_w4a16_decoder_fp8kv_proposed.yaml`

Core policy:

```yaml
targets:
  - re:^language_model\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$

ignore:
  - re:^audio_tower(\.|$)
  - re:^multi_modal_projector(\.|$)
  - re:^language_model\.model\.embed_tokens$
  - re:^language_model\.model\.norm$
  - re:^language_model\.lm_head$
  - re:^.*ada_[^.]*($|\.)
  - re:^.*rms_norm[^.]*($|\.)
  - re:^.*layer_?norm[^.]*($|\.)
  - re:^whisper_encoder(\.|$)
  - re:^audio_language_adapter(\.|$)
  - re:^language_model\.model\.layers\.\d+\.ada_rms_norm_t_cond(\.|$)
  - re:^mm_streams_embeddings(\.|$)
  - re:^layers\.\d+\.ada_rms_norm_t_cond(\.|$)
  - re:^layers\.\d+\.(attention_norm|ffn_norm)$
  - re:^norm$
```

This is the `llmcompressor`-valid form of the requested boundary. A literal
`language_model.model.layers.*` target is not safe because the matching helper treats non-`re:`
strings as exact names, not shell globs.

Effective quantized tensors:

- decoder attention projections:
  - HF: `language_model.model.layers.<n>.self_attn.{q_proj,k_proj,v_proj,o_proj}`
  - consolidated: `layers.<n>.attention.{wq,wk,wv,wo}`
- decoder MLP projections:
  - HF: `language_model.model.layers.<n>.mlp.{gate_proj,up_proj,down_proj}`
  - consolidated: `layers.<n>.feed_forward.{w1,w2,w3}`

Everything else stays BF16.

## Environment Constraint

The recipe assumes a single environment where both are true:

- `transformers` can load `VoxtralRealtimeForConditionalGeneration`
- `llmcompressor` can run `GPTQModifier`

The existing `~/.venvs/voxtral-llmcompressor-research` environment is not enough because it pins
Transformers to `<=4.57.6`, which cannot load Voxtral Realtime. Calibration should not start until
that environment bridge is resolved and the recipe above is reviewed.

## Serve-Time Expectation

The compressed artifact should contain:

```json
"kv_cache_scheme": {
  "num_bits": 8,
  "type": "float",
  "symmetric": true,
  "strategy": "tensor",
  "dynamic": false
}
```

vLLM recognizes this compressed-tensors field and sets the cache dtype to FP8 for attention layers.
The serving config should still use `quantization: compressed-tensors`, `tokenizer:
models/voxtral-realtime`, and `tokenizer_mode: mistral`.

## Review Gate

Do not run calibration yet.

Before calibration, verify from the loaded model's `named_modules()` that the ignore list leaves
exactly these quantized `Linear` modules:

- 26 layers x 4 attention projections
- 26 layers x 3 MLP projections

Expected total: 182 quantized decoder projection modules.

I checked the proposed policy against both safetensors headers. It leaves 182 targeted weight
modules in `model.safetensors`, matching the intended 26 x 7 decoder projection set. It leaves 0
targeted modules in `consolidated.safetensors` because this recipe deliberately targets the HF
layout only; do not feed the consolidated stub to this recipe unless the target names are remapped
in a separate reviewed recipe.

## Step 1 Boundary Audit

Directory listing:

- present: `config.json`
- present: `model.safetensors`
- present: `consolidated.safetensors`
- absent: `model.safetensors.index.json`

`config.json` confirms the relevant config fields:

- `audio_config.model_type = voxtral_realtime_encoder`
- `projector_hidden_act = gelu`
- text config has `num_attention_heads`, `num_key_value_heads`, `rms_norm_eps`, and
  `tie_word_embeddings = true`

Since there is no safetensors index file, I used the safetensors headers as the source of truth for
module names.

Audit result:

- `model.safetensors`: 175 modules contain `ada`, `encoder`, `projector`, `embed`, `head`, or
  `norm`; all are covered by the modern protect rules; targeted-after-ignore count is 182.
- `consolidated.safetensors`: the modern protect rules alone miss 347 suspect modules, mostly
  `mm_streams_embeddings.*`, `layers.<n>.attention_norm`, `layers.<n>.ffn_norm`, and
  `audio_language_projection` / `whisper_encoder` paths.
- With the consolidated/vLLM aliases in the proposed recipe, uncovered suspect modules: 0; targeted
  after ignore is 0 because consolidated names are not the calibration target for this recipe.

## Step 2 Calibration Data

Prepared text-only calibration data is documented in `docs/track_b_calibration_data.md`.

- JSONL: `data/calibration/track_b_multilingual_text_256.jsonl`
- Summary: `data/calibration/track_b_multilingual_text_256.summary.json`
- Tokenizer check: `data/calibration/track_b_multilingual_text_256.tokenizer_check.json`
- Source: `google/fleurs`
- Total records: 256
- Languages: EN, ZH, HI, ES, AR, FR, PT, RU, DE, JA, KO, IT, NL
- Passage mix: 126 short, 130 medium
- Tokenizer: exact local Voxtral `MistralCommonBackend` from `models/voxtral-realtime`
- Tokenization errors: 0

Calibration has not been run.
