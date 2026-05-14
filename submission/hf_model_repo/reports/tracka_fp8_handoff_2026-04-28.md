# Track A FP8 Handoff

Last updated: 2026-04-28 13:35 IST

## Verdict

Track A is packaging-ready as the verified safety submission floor.

Final uniform client policy:

- Do not use pre-endpoint VAD trimming.
- Use `--language-hint-mode fleurs_primary`.
- Use `--empty-retry-count 2` as an auditable guard. The final FP8 run needed `0` retries on all slices.

Total FP8 energy across the four canonical slices: `345,734.14 J`.

Same-policy BF16 energy across the four canonical slices: `474,614.96 J`.

Measured FP8 energy reduction vs same-policy BF16: `27.15%`.

This `27.15%` is a measured local/harness result, not a promised cross-hardware ratio. The
direction is the defended claim; the magnitude can move on organizer hardware.

## Runtime

- Final YAML: `configs/vllm/fp8_round1.yaml`
- Model path: `models/voxtral-realtime`
- Runtime mode: base BF16 Voxtral served through vLLM runtime FP8
- Required evaluator flags: `--language-hint-mode fleurs_primary --empty-retry-count 2`
- VAD: disabled for the final package
- Prefix cache: attempted, enabled in config/logs, but no speedup claimed because hit rate stayed `0.0%`
- Tau-fold: rejected, not part of Track A

## FP8 10-Column Report

| Slice | Limit | Gate metric | Value % | CI low % | CI high % | Ceiling % | Margin pp | Empty predictions | Energy J |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| en_us | 500 | normalized WER | 6.1456 | 5.4996 | 6.7794 | 7.56 | 1.4144 | 0 | 189,442.099 |
| fr_fr | 100 | normalized WER | 8.4548 | 6.7809 | 10.2486 | 10.30 | 1.8452 | 0 | 37,882.636 |
| hi_in | 100 | normalized WER | 25.4309 | 22.4806 | 28.6336 | 32.84 | 7.4091 | 0 | 44,502.932 |
| ja_jp | 100 | no-space CER | 7.0919 | 5.5534 | 8.6900 | 11.08 | 3.9881 | 0 | 73,906.476 |

## Same-Policy BF16 Baseline

| Slice | Limit | Gate metric | Value % | Empty predictions | Retry requests | Energy J |
|---|---:|---|---:|---:|---:|---:|
| en_us | 500 | normalized WER | 6.0563 | 1 | 2 | 250,404.795 |
| fr_fr | 100 | normalized WER | 8.2726 | 0 | 0 | 55,218.915 |
| hi_in | 100 | normalized WER | 26.2735 | 1 | 2 | 71,785.646 |
| ja_jp | 100 | no-space CER | 6.7353 | 0 | 0 | 97,205.604 |

## FP8 Reports

- EN500: `reports/benchmark_fp8_tracka_novad_hint_retry2_en500_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_en_us_limit500.json`
- FR100: `reports/benchmark_fp8_tracka_novad_hint_retry2_fr100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_fr_fr_limit100.json`
- HI100: `reports/benchmark_fp8_tracka_novad_hint_retry2_hi100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_hi_in_limit100.json`
- JA100: `reports/benchmark_fp8_tracka_novad_hint_retry2_ja100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_ja_jp_limit100.json`

## FP8 Server Logs

- EN500: `logs/fp8_tracka_novad_hint_retry2_en500_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_benchmark_server.log`
- FR100: `logs/fp8_tracka_novad_hint_retry2_fr100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_benchmark_server.log`
- HI100: `logs/fp8_tracka_novad_hint_retry2_hi100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_benchmark_server.log`
- JA100: `logs/fp8_tracka_novad_hint_retry2_ja100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_benchmark_server.log`

## RCA Notes

- Mixed VAD is rejected. It is not reproducible enough for the organizer.
- Uniform conservative VAD is rejected. HI produced `1` empty prediction.
- Uniform no-VAD without language hints is rejected. JA produced `1` empty prediction.
- The final fix is explicit language hints with no VAD. This cleared all FP8 slices with `0` empty predictions and `0` retry requests.
- The empty-output issue is not FP8-specific: same-policy BF16 also produced persistent empty predictions on EN and HI despite two retries.
- Successful runs still emit Voxtral `empty multimodal embeddings` warnings in server logs. The accepted FP8 reports, not the warning count alone, are the pass/fail evidence.
- `num_gpu_blocks_override=2` appears in vLLM logs, while the actual KV capacity line reports `GPU KV cache size: 5,936 tokens` for FP8.

## Submission Caveats

- The BF16 ceilings used here are same-harness internal gates: EN `7.56%`, FR `10.30%`, HI `32.84%`, JA no-space CER `11.08%`. Before upload, re-read the organizer scoring spec and confirm whether the WER gate is applied against the organizer's own baseline or the submitted same-harness BF16 baseline.
- The language-hint policy is validated against the FLEURS language IDs used by the current evaluator (`en_us`, `fr_fr`, `hi_in`, `ja_jp`) and `--dataset-source google_fleurs`. If the organizer uses a different wrapper or manifest, the language-hint dispatch must still map to the primary language code (`en`, `fr`, `hi`, `ja`).
- `--empty-retry-count 2` is not dead code: the guard fired in failed/non-accepted diagnostics, including `reports/benchmark_fp8_tracka_novad_retry2_ja100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_ja_jp_limit100.json`, and in same-policy BF16 EN/HI baselines. The accepted final FP8 run needed `0` retries, which is the preferred outcome.
