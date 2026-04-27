# Submission Form Draft

## Submission Name

Voxtral Mini Realtime Runtime-FP8

## Track

Fill with final organizer track: Track A or Track B.

## Hugging Face Model Repo

Fill after upload:

```text
https://huggingface.co/<org-or-user>/<repo-name>
```

Upload source:

```text
submission/hf_model_repo
```

## Base Model

```text
mistralai/Voxtral-Mini-4B-Realtime-2602
revision 2769294da9567371363522aac9bbcfdd19447add
```

## Compression Method

The submitted artifact uses the base Voxtral Mini Realtime checkpoint served through vLLM runtime
FP8 quantization. The pinned serving config is `vllm_config.yaml` in the uploaded repo:

```yaml
quantization: fp8
kv_cache_dtype: fp8_e4m3
attention_backend: TRITON_ATTN
max_model_len: 8192
gpu_memory_utilization: 0.85
compilation_config:
  cudagraph_mode: PIECEWISE
```

This is a base-checkpoint-plus-runtime-config submission, not a static quantized safetensors
checkpoint. Prefix caching is enabled in the exact config but is not claimed as a performance win,
because the measured speech path did not show positive prefix-cache reuse.

## Reproduction Command

```bash
git clone https://huggingface.co/<org-or-user>/<repo-name>
cd <repo-name>
bash reproduce.sh
```

The script installs dependencies, downloads the pinned base model revision, starts vLLM, evaluates
FLEURS, measures energy, and writes fresh reports under `reports/`.

## Primary Claim

On the submitted local English FLEURS 20-sample slice, runtime FP8 preserved normalized WER versus
the BF16 Voxtral reference while reducing wall time and measured energy.

| Run | Language | Samples | Raw WER | Norm WER | Empty preds | Time | Energy | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 Voxtral reference | `en_us` | 20 | 22.20% | 6.36% | 0 | 46.26 s | 8112.90 J | `reports/fleurs_bf16_en_us_limit20_quietfix.json`, `reports/energy_fleurs_bf16_en_us_limit20_quietfix.json` |
| FP8 Voxtral candidate | `en_us` | 20 | 21.97% | 6.36% | 0 | 35.21 s | 4952.89 J | `reports/fleurs_fp8_en_us_limit20_quietfix.json`, `reports/energy_fleurs_fp8_en_us_limit20_quietfix.json` |

Derived from the report JSON, FP8 is 23.89% faster and uses 38.95% less measured energy than the
BF16 Voxtral reference on this slice.

## Multilingual Checks

| Language | Samples | Quality read | Empty preds | Evidence |
| --- | ---: | --- | ---: | --- |
| `fr_fr` | 5 | raw WER 23.18%, normalized WER 10.56% | 0 | `reports/fleurs_fp8_fr_fr_limit5_quietfix.json`, `reports/energy_fleurs_fp8_fr_fr_limit5_quietfix.json` |
| `hi_in` | 5 | raw WER 26.83%, normalized WER 23.58% | 0 | `reports/fleurs_fp8_hi_in_limit5_quietfix.json`, `reports/energy_fleurs_fp8_hi_in_limit5_quietfix.json` |
| `ja_jp` | 5 | CER 10.42%, no-whitespace CER 10.00% | 0 | `reports/fleurs_fp8_ja_jp_limit5_quietfix_v2.json`, `reports/energy_fleurs_fp8_ja_jp_limit5_quietfix_v2.json` |

Japanese word-level WER is not used as the defended quality metric because segmentation dominates
that read; the report includes it for transparency.

## External Context

Whisper large-v3 is included only as an external anchor, not as the submitted model. On the same
English 20-sample slice it produced raw WER 20.59%, normalized WER 4.32%, 34.77 seconds, and
3258.57 joules. Evidence:

```text
reports/fleurs_whisper_large_v3_en_us_limit20.json
reports/energy_fleurs_whisper_large_v3_en_us_limit20.json
```

## Hardware Used For Submitted Numbers

```text
NVIDIA GeForce RTX 5080
16 GB VRAM
Linux/WSL2 runtime
```

The later runtime validation report is
`reports/benchmark_fp8_gate_control_warm_en20_en_us_limit20.json`.

## Known Caveats

- The defended claim is efficiency versus BF16 Voxtral, not beating every external ASR baseline.
- GPTQ/static quantized safetensors branches are not part of this submission.
- Prefix caching is not claimed as a realized speedup.
- Full clean-env reproduction still needs to be run before final form submission.
