# Track A Submission Form Draft

## Submission Name

Track A Voxtral Mini Realtime Runtime-FP8

## Hugging Face Model Repo

Fill after upload:

```text
https://huggingface.co/Shankara-A-S/voxtral-mini-realtime-fp8-runtime
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

## Final Serving Config

```text
vllm_config.yaml
packaged model field: .
benchmark runtime fields sha256: 4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4
organizer entrypoint: vllm serve --config vllm_config.yaml
```

## Final Client Policy

```text
--language-hint-mode fleurs_primary --empty-retry-count 2
VAD disabled
dataset source: google_fleurs
```

## Reproduction Command

```bash
git clone https://huggingface.co/Shankara-A-S/voxtral-mini-realtime-fp8-runtime
cd voxtral-mini-realtime-fp8-runtime
bash reproduce.sh
```

## Primary Claim

Track A uses base Voxtral served through vLLM runtime FP8. Across the four canonical FLEURS slices,
the submitted FP8 policy used `345,734.14 J` versus `474,614.96 J` for the same-policy BF16
reference, a measured local energy reduction of `27.15%`.

The final FP8 run cleared every gate with `0` empty predictions and `0` retry requests.

| Slice | Limit | Gate metric | Value | CI low | CI high | Ceiling | Margin | Energy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `en_us` | 500 | normalized WER | 6.1456% | 5.4996% | 6.7794% | 7.56% | 1.4144 pp | 189,442.10 J |
| `fr_fr` | 100 | normalized WER | 8.4548% | 6.7809% | 10.2486% | 10.30% | 1.8452 pp | 37,882.64 J |
| `hi_in` | 100 | normalized WER | 25.4309% | 22.4806% | 28.6336% | 32.84% | 7.4091 pp | 44,502.93 J |
| `ja_jp` | 100 | no-space CER | 7.0919% | 5.5534% | 8.6900% | 11.08% | 3.9881 pp | 73,906.48 J |

Every number above is cross-referenced in:

```text
reports/claimed_results.json
```

## Evidence Files

```text
reports/tracka_fp8_handoff_2026-04-28.md
reports/benchmark_fp8_tracka_novad_hint_retry2_en500_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_en_us_limit500.json
reports/benchmark_fp8_tracka_novad_hint_retry2_fr100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_fr_fr_limit100.json
reports/benchmark_fp8_tracka_novad_hint_retry2_hi100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_hi_in_limit100.json
reports/benchmark_fp8_tracka_novad_hint_retry2_ja100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_ja_jp_limit100.json
```

## Required Caveats

- Portability: the energy ratio is measured locally and may move on organizer hardware.
- Organizer baseline rule: local same-policy BF16 gates are internal; organizer baseline rules
  supersede them if different.
- FLEURS language-hint coupling: `fleurs_primary` must map the manifest language IDs to primary
  language codes.
- No prefix-cache claim: prefix caching was attempted/enabled, but hit rate stayed `0.0%`.
- No tau-fold: tau-fold is rejected for Track A.
