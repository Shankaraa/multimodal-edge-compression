# Voxtral Runtime-FP8 Submission Artifact

This is a runtime-FP8 Voxtral submission, not a static post-training quantized checkpoint.
The artifact downloads `mistralai/Voxtral-Mini-4B-Realtime-2602` at revision
`2769294da9567371363522aac9bbcfdd19447add` and serves it with the exact vLLM config in
`vllm_config.yaml`.

Single-command reproduction after cloning this repo:

```bash
bash reproduce.sh
```

That command creates a Python environment, installs the benchmark harness, downloads the base
model, starts vLLM, runs the canonical English FLEURS benchmark, measures energy, writes fresh JSON
reports, and stops the server.

## Compression Method

The defended compression path is vLLM runtime FP8:

- base checkpoint: `mistralai/Voxtral-Mini-4B-Realtime-2602`
- base revision: `2769294da9567371363522aac9bbcfdd19447add`
- serving config: `vllm_config.yaml`
- served model name: `voxtral-realtime`
- vLLM quantization: `fp8`
- KV cache dtype: `fp8_e4m3`
- attention backend: `TRITON_ATTN`
- max model length: `8192`
- GPU memory utilization: `0.85`
- CUDA graph mode: `PIECEWISE`

`enable_prefix_caching` is present in the pinned config because it was part of the runtime envelope,
but prefix-cache speedup is not part of the claimed result. It was attempted and did not activate
on this vLLM build for the measured speech transcription path.

## Claimed Numbers

All numbers below are copied from committed JSON in `reports/`. The machine for the submitted
runs used an NVIDIA GeForce RTX 5080 with 16 GB VRAM.

| Run | Language | Samples | Raw WER | Norm WER | Empty preds | Time | Energy | Report files |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 Voxtral reference | `en_us` | 20 | 22.20% | 6.36% | 0 | 46.26 s | 8112.90 J | `reports/fleurs_bf16_en_us_limit20_quietfix.json`, `reports/energy_fleurs_bf16_en_us_limit20_quietfix.json` |
| FP8 Voxtral candidate | `en_us` | 20 | 21.97% | 6.36% | 0 | 35.21 s | 4952.89 J | `reports/fleurs_fp8_en_us_limit20_quietfix.json`, `reports/energy_fleurs_fp8_en_us_limit20_quietfix.json` |
| Whisper large-v3 anchor | `en_us` | 20 | 20.59% | 4.32% | 0 | 34.77 s | 3258.57 J | `reports/fleurs_whisper_large_v3_en_us_limit20.json`, `reports/energy_fleurs_whisper_large_v3_en_us_limit20.json` |

Against the BF16 Voxtral reference, the FP8 candidate is 23.89% faster and uses 38.95% less measured
energy on this 20-sample English slice. Those derived values are recorded in
`reports/claimed_results.json`.

## Per-Language FP8 Checks

| Language | Samples | Defended quality read | Empty preds | Time | Energy | Report files |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `en_us` | 20 | raw WER 21.97%, normalized WER 6.36% | 0 | 35.21 s | 4952.89 J | `reports/fleurs_fp8_en_us_limit20_quietfix.json`, `reports/energy_fleurs_fp8_en_us_limit20_quietfix.json` |
| `fr_fr` | 5 | raw WER 23.18%, normalized WER 10.56% | 0 | 21.74 s | 2121.87 J | `reports/fleurs_fp8_fr_fr_limit5_quietfix.json`, `reports/energy_fleurs_fp8_fr_fr_limit5_quietfix.json` |
| `hi_in` | 5 | raw WER 26.83%, normalized WER 23.58% | 0 | 16.93 s | 1620.28 J | `reports/fleurs_fp8_hi_in_limit5_quietfix.json`, `reports/energy_fleurs_fp8_hi_in_limit5_quietfix.json` |
| `ja_jp` | 5 | CER 10.42%, no-whitespace CER 10.00% | 0 | 19.62 s | 2692.89 J | `reports/fleurs_fp8_ja_jp_limit5_quietfix_v2.json`, `reports/energy_fleurs_fp8_ja_jp_limit5_quietfix_v2.json` |

Japanese word-level WER is not the quality claim because whitespace segmentation dominates that
metric. The committed Japanese report includes raw WER for transparency and CER for the defended
read.

## Reproduction Details

Default command:

```bash
bash reproduce.sh
```

Useful overrides:

```bash
SKIP_INSTALL=1 bash reproduce.sh
INSTALL_VLLM=0 bash reproduce.sh
PORT=8120 LABEL=my_rerun bash reproduce.sh
LANG=fr_fr LIMIT=5 LABEL=fp8_fr_rerun bash reproduce.sh
MODEL_DIR=/path/to/local/voxtral bash reproduce.sh
```

Expected output files for the default run:

- `reports/benchmark_submission_fp8_runtime_en_us_limit20.json`
- `reports/fleurs_submission_fp8_runtime_en_us_limit20.json`
- `reports/energy_fleurs_submission_fp8_runtime_en_us_limit20.json`

The script also runs:

```bash
python scripts/verify_claimed_reports.py --reports-dir reports --claims reports/claimed_results.json
```

That verifier proves the README numbers are traceable to the committed report JSON.

## Environment Notes

This workflow is intended for Linux or WSL2 with an NVIDIA GPU and a working CUDA stack. The default
vLLM install path uses the CUDA 13.0 nightly wheel path used for the submitted runs:

```bash
uv pip install -U vllm --torch-backend=cu130 --extra-index-url https://wheels.vllm.ai/nightly/cu130
```

For a different CUDA stack, set `VLLM_TORCH_BACKEND` and `VLLM_EXTRA_INDEX_URL` before running
`reproduce.sh`.

## Manual Serve Command

If the environment is already prepared and the base model is already downloaded:

```bash
python scripts/serve_model.py models/voxtral-realtime --config vllm_config.yaml --port 8115
python scripts/check_vllm_server.py --base-url http://127.0.0.1:8115/v1
```

Then evaluate:

```bash
python scripts/measure_energy.py --report reports/manual_energy.json -- \
  python scripts/evaluate_fleurs.py \
    --lang en_us \
    --limit 20 \
    --base-url http://127.0.0.1:8115/v1 \
    --model voxtral-realtime \
    --out reports/manual_fleurs.json
```
