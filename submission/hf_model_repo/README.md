---
license: apache-2.0
base_model: mistralai/Voxtral-Mini-4B-Realtime-2602
library_name: vllm
pipeline_tag: automatic-speech-recognition
tags:
- voxtral
- vllm
- fp8
- resilient-ai-challenge
---

# Voxtral Mini Realtime FP8 Runtime Package

This repository packages `mistralai/Voxtral-Mini-4B-Realtime-2602` with the exact vLLM serving
configuration used for the reported benchmark results. The model is served with vLLM runtime FP8
quantization and FP8 E4M3 KV cache.

The repository root includes `consolidated.safetensors`, so the serving config resolves the model
locally:

```bash
vllm serve --config vllm_config.yaml
```

Single-command benchmark reproduction after cloning:

```bash
bash reproduce.sh
```

The reproduction script serves this package with `vllm_config.yaml`, runs the configured FLEURS
benchmark slices, records energy, and writes benchmark JSON files under `reports/`.

## Package Contents

- Base model: `mistralai/Voxtral-Mini-4B-Realtime-2602`
- Base revision: `2769294da9567371363522aac9bbcfdd19447add`
- Packaged weights: `consolidated.safetensors`
- Serving config: `vllm_config.yaml`
- Local model path in serving config: `.`
- Runtime quantization: `fp8`
- KV cache dtype: `fp8_e4m3`
- Max model length: `4096`
- Benchmark policy: `--language-hint-mode fleurs_primary --empty-retry-count 2`
- VAD trimming: disabled

This is a runtime-quantized serving package. The checkpoint weights are the packaged BF16 base
weights; compression is applied by the pinned vLLM runtime configuration.

## Reported Results

Every value in this table is cross-referenced through `reports/claimed_results.json` and the
committed benchmark reports in `reports/`.

| Language | Samples | Metric | Value | 95% CI low | 95% CI high | Empty predictions | Retry requests | Energy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English (`en_us`) | 500 | normalized WER | 6.1456% | 5.4996% | 6.7794% | 0 | 0 | 189,442.10 J |
| French (`fr_fr`) | 100 | normalized WER | 8.4548% | 6.7809% | 10.2486% | 0 | 0 | 37,882.64 J |
| Hindi (`hi_in`) | 100 | normalized WER | 25.4309% | 22.4806% | 28.6336% | 0 | 0 | 44,502.93 J |
| Japanese (`ja_jp`) | 100 | no-space CER | 7.0919% | 5.5534% | 8.6900% | 0 | 0 | 73,906.48 J |

`reports/claimed_results.json` lists the source report file for each row in the table.

## Energy Summary

- Runtime-FP8 total energy across the reported slices: `345,734.14 J`
- BF16 reference total energy under the same benchmark policy: `474,614.96 J`
- Measured energy reduction: `27.15%`

These values are derived in `reports/claimed_results.json` from the FP8 reports and the BF16
reference reports committed in `reports/`.

## Reproduction

Default full reproduction:

```bash
bash reproduce.sh
```

Expected output: language-specific benchmark JSON files written under `reports/`.

Quick smoke run:

```bash
RUN_SLICES="en_us:1:packaged_smoke_en1" DOWNLOAD_MODEL=0 MODEL_DIR=/path/to/voxtral bash reproduce.sh
```

Useful environment overrides:

```bash
SKIP_INSTALL=1 bash reproduce.sh
INSTALL_VLLM=0 bash reproduce.sh
BASE_PORT=8200 bash reproduce.sh
MODEL_DIR=/path/to/local/voxtral bash reproduce.sh
```

Before running benchmarks, the script verifies the committed claims:

```bash
python scripts/verify_claimed_reports.py --reports-dir reports --claims reports/claimed_results.json
```

That check fails if reported values drift from the committed JSON reports.

## Logs

Server logs from the reported FP8 runs are included under `logs/`.

## Notes

- Energy measurements are hardware- and harness-dependent; the reported values are tied to the
  committed benchmark reports.
- The benchmark uses FLEURS primary language hints: `en_us -> en`, `fr_fr -> fr`, `hi_in -> hi`,
  and `ja_jp -> ja`.
- Prefix caching is enabled in `vllm_config.yaml`, but the reported results do not attribute any
  efficiency gain to prefix-cache reuse.
