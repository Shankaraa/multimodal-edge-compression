# Round-2 Candidate Snapshot

This document is the structured handoff for the Round-2 candidate. It is filled in
locally on the RTX 5080 development machine and is the read-once briefing for the
person who provisions the L4 cloud node and runs the binding submission measurement.

## Status

- Local validation: in progress (RTX 5080 / WSL Ubuntu 22.04 / vllm 0.19.1rc1.dev302).
- L4 binding measurement: not yet started.

## Candidate parameter set (locked)

The Round-2 submission stack is the existing FP8 path plus a deterministic audio
preprocessing chain. No model-side changes vs Track A FP8.

```
# vLLM serve config (unchanged from Track A FP8)
configs/vllm/fp8_round1.yaml

# evaluator audio-prep flags (tight gate — matches 13-language sweep 2026-05-08)
--target-lufs -23.0
--lufs-max-gain-db 24.0
--vad-trim --vad-aggressiveness 1 --vad-padding-ms 200
--gate-silence
--compress-internal-silence-to-ms 160
--min-internal-silence-run-ms 320
```

> **Note (2026-05-12):** The earlier 4-language canonical sweep (2026-05-07) used
> `320/640` gate parameters. The 13-language sweep (2026-05-08) tightened to `160/320`
> and produced slightly better EN500 norm WER (`5.69%` vs `5.80%`). The locked
> candidate and `scripts/reproduce_round2_audio.sh` now use `160/320` throughout.

**LUFS is mandatory.** The EN500 ablation showed that without LUFS, `--gate-silence`
catastrophically over-trims the 69% of FLEURS EN samples that have input RMS below 0.003
(empty rate jumps from 0 to 5.2%, normalized WER blows up from 6.15% to 25.93%, audio
trim balloons from a healthy ~10% to 31%). With LUFS the activity thresholds are
calibrated correctly and quiet samples decode normally.

## EN20 development-machine evidence

| Stack | raw WER | norm WER | empty | wall-clock | trim% |
|---|---|---|---|---|---|
| BF16 baseline | 22.20% | 6.36% | 0 | &mdash; | 0% |
| BF16 + LUFS-23 | 22.20% | 6.14% | 0 | 37.23 s | 0% |
| BF16 + LUFS + VAD-trim | 22.20% | 6.14% | 0 | 40.59 s | 5.13% |
| BF16 + LUFS + VAD + gate(320/640) | 22.20% | **5.68%** | 0 | 39.43 s | **8.68%** |
| FP8 baseline | 21.97% | 6.36% | 0 | &mdash; | 0% |
| FP8 + LUFS-23 | 22.20% | 6.59% | 0 | 31.19 s | 0% |
| FP8 + LUFS + VAD-trim | 22.20% | 6.59% | 0 | 28.57 s | 5.13% |
| **FP8 + LUFS + VAD + gate(320/640)** | **21.97%** | 6.59% | 0 | **26.32 s** | **8.68%** |

Reports: `reports/fleurs_{bf16,fp8}_en_us_limit20_lufs23{_vadtrim,_vadtrim_gate}_smoke.json`.

## EN500 ablation evidence (RTX 5080, 2026-05-07)

| Variant | raw WER | norm WER | empty | sum_lat | audio_out | trim% |
|---|---|---|---|---|---|---|
| FP8 EN500 baseline (existing 2026-04-28) | 27.32% | 6.15% | 0 | n/a | 4939.32 s | 0% |
| FP8 EN500 + VAD+gate (no LUFS) | 42.54% | 25.93% | **26** | 1296.15 s | 3392.76 s | **31.31%** (broken) |
| **FP8 EN500 + LUFS + VAD+gate** | **27.08%** | **5.80%** | **0** | **732.30 s** | **4426.02 s** | **10.39%** |

Headlines:
- Normalized WER **improved** by 0.35 pp (6.15% -> 5.80%) under LUFS+VAD+gate.
- Zero empty predictions; LUFS rescued every quiet sample (FLEURS EN500 is 69%
  RMS-below-0.003).
- 10.39% audio reduction (513 seconds out of 4939) translates directly to encoder
  energy on Voxtral Realtime (variable-length encoder).
- Wall-clock latency is healthy: variant B's 732 s sum-of-per-sample latencies vs
  variant A's 1296 s shows the no-LUFS variant was actually slower because the
  trimmed-empty path adds error/retry overhead.

Reports: `reports/fleurs_fp8_en500_{vadgate_nolufs,lufs23_vadgate}_smoke.json`.
Energy: `reports/energy_fleurs_fp8_en500_{vadgate_nolufs,lufs23_vadgate}_smoke.json`.

CodeCarbon EN500 energy (RTX 5080, **relative only**, not the binding L4 number):

| Variant | elapsed | total kJ | emissions kg CO2 |
|---|---|---|---|
| FP8 EN500 + VAD+gate (no LUFS) | 1326.05 s | 278.17 | 0.0551 |
| **FP8 EN500 + LUFS + VAD+gate** | **756.02 s** | **154.15** | **0.0305** |

The locked candidate is **45% lower energy and 1.75x faster** than the broken-without-LUFS
variant on this hardware. The directly-comparable Track A FP8 EN500 number was measured on
the L4 (`345.7 kJ` total across four languages, EN portion only `189.4 kJ`); the RTX 5080
number cannot be directly compared to that. The L4 re-measurement is required for any
absolute energy claim.

## Multilingual canonical evidence (RTX 5080, 2026-05-07)

Apples-to-apples vs the existing FP8 Track A baseline reports
(`reports/fleurs_fp8_tracka_novad_hint_retry2_<lang><limit>_*.json`).

| Slice | Baseline norm WER | Locked norm WER | &Delta; norm WER | Baseline CER (no_ws) | Locked CER (no_ws) | &Delta; CER | empty | trim% |
|---|---|---|---|---|---|---|---|---|
| EN500 | 6.15% (older baseline) | **5.80%** | **&minus;0.35** | &mdash; | &mdash; | &mdash; | 0 | 10.39% |
| HI100 | 25.43% | 25.55% | +0.12 | 14.05% | 14.13% | +0.08 | 0 | 5.93% |
| FR100 | 8.45% | **7.51%** | **&minus;0.94** | 7.72% | 7.56% | **&minus;0.16** | 0 | 17.95% |
| JA100 | (WER moot) | (WER moot) | &mdash; | 11.44% | **10.53%** | **&minus;0.91** | 0 | 12.64% |

CodeCarbon energy on RTX 5080 (RELATIVE only, NOT the binding L4 number):

| Slice | elapsed | total kJ | emissions kg CO2 |
|---|---|---|---|
| EN500 | 756.0 s | 154.15 | 0.0305 |
| HI100 | 210.8 s | 41.91 | 0.0083 |
| FR100 | 139.6 s | 28.65 | 0.0057 |
| JA100 | 281.3 s | 58.49 | 0.0116 |
| **Total** | **1387.7 s** | **283.20** | **0.0561** |

For context: Track A FP8 final submission reported `345.7 kJ` total on the L4 evaluation
hardware. RTX 5080 is a different SKU and the local energy number cannot stand in for the
submission claim. The directional signal (locked candidate has lower energy than Track A on
this dev machine, in addition to the audio-second reduction) is consistent with what we
should see on the L4 re-measurement.

Reports: `reports/fleurs_fp8_{en500_lufs23_vadgate,hi_in,fr_fr,ja_jp}_limit{500,100,100,100}_lufs23_vadgate_smoke.json`.
Energy reports: same paths with `energy_` prefix.

## Full 13-language FLEURS coverage (locked stack, tight gate 160/320, 2026-05-08)

This is the broader sweep covering every Voxtral-supported FLEURS language at limit=100 (EN
at limit=500), under the updated locked parameter set (`160 ms` compress / `320 ms` min run).
Use CER (no whitespace) as the primary metric for the no-word-boundary languages (zh, ja).

| FLEURS slice | norm WER | CER | CER (no_ws) | empty | trim% | RTX 5080 kJ |
|---|---|---|---|---|---|---|
| en_us limit=500 | 5.69% | 6.91% | 7.90% | 0 | 12.59% | 129.79 |
| es_419 limit=100 | **3.01%** | 4.60% | 5.33% | 0 | 14.71% | 30.32 |
| it_it limit=100 | **3.74%** | 5.30% | 5.95% | 0 | 15.70% | 46.63 |
| ru_ru limit=100 | **5.18%** | 5.36% | 6.04% | 0 | 12.17% | 26.96 |
| pt_br limit=100 | **5.29%** | 6.21% | 6.99% | 0 | 14.47% | 36.79 |
| de_de limit=100 | **5.64%** | 8.90% | 10.19% | 0 | 14.09% | 40.70 |
| fr_fr limit=100 | 7.36% | 6.87% | 7.49% | 0 | 19.81% | 24.75 |
| nl_nl limit=100 | 8.71% | 6.75% | 7.52% | 0 | 11.22% | 27.68 |
| ar_eg limit=100 | 15.70% | 5.33% | 5.82% | 0 | 8.86% | 39.15 |
| ko_kr limit=100 | 16.02% | 8.15% | 8.72% | 0 | **27.20%** | 23.37 |
| hi_in limit=100 | 26.12% | 12.89% | 14.76% | 0 | 7.16% | 34.60 |
| ja_jp limit=100 | (WER moot) | 12.35% | **10.51%** | 0 | 14.83% | 44.52 |
| cmn_hans_cn limit=100 | (WER moot) | 52.28% | **12.41%** | 0 | **25.93%** | 25.90 |
| **TOTAL (13 langs)** | &mdash; | &mdash; | &mdash; | **0** | &mdash; | **531.16** |

Reports: `reports/fleurs_fp8_<lang>_limit{100,500}_lufs23_vadgate160_320_smoke.json` and
matching `energy_*.json`.

Headlines:
- **Zero empty predictions across 13 languages and 1700 total samples.** The locked stack
  doesn't break on any Voxtral-supported language.
- **es_419 norm WER `3.01%` is the strongest score** in the entire matrix; it_it `3.74%`
  is second-best. Voxtral genuinely shines on Iberian + Italic speech.
- **Trim% varies from 7.16% (hi_in) to 27.20% (ko_kr)** &mdash; Korean recordings carry the
  longest internal pauses and benefit most from gating; Hindi the least.
- Total RTX 5080 energy across all 13 slices is `531.16 kJ` over `~25 minutes` of wall
  clock. Track A FP8 reported `345.7 kJ` for the 4-language submission set on the L4. Adding
  9 new languages on the RTX 5080 yields +185 kJ relative to the 4-lang RTX 5080 number
  (`283.2 -> 531.2`); on the L4 the absolute total will differ but the per-language ratios
  should hold.

## HI sample 1985 idx 82 status under the locked stack

Under FP8 + locked audio-prep, **idx 82 now produces a non-empty Hindi prediction** in HI100
(stream index 82, id `1985`, original RMS `0.008`, post-LUFS RMS `0.071`). The prediction
content is wrong for both duplicates (idx 9 partly translated to English; idx 82 garbled
Hindi), but neither row contributes to `empty_prediction_count` and neither hurts the
slice-level normalized WER beyond what the row would have contributed at any precision
under the existing baseline. Day-2 BF16+LUFS evidence still showed empty on idx 82 - the
combination of LUFS *and* the gating pipeline is what tips the FP8 path over the empty
boundary on this row. We still recommend sending the organizer email
(`reports/sample_1985_investigation/hi_1985_findings.md`) before final submission so the
duplicate-row anomaly is documented externally.

## L4 handoff checklist

1. **Provision** an L4 24 GB node (Lambda Labs / RunPod / GCP — matches official eval hardware).
2. **Install stack** using `scripts/l4_setup.sh` (created 2026-05-12). Verify versions match:
   - `vllm==0.19.1rc1.dev302+g68be0f853.cu130`
   - `torch==2.11.0+cu130`
   - `transformers==4.57.6`
   - `compressed-tensors==0.14.0.1`
   - `flashinfer-python==0.6.7`
   - `pyloudnorm==0.2.0`
3. **Pull model** from HF (private gated repo `Shankara-A-S/voxtral-mini-realtime-fp8-runtime`).
4. **Start server**: `python scripts/serve_model.py ~/models/voxtral-realtime --config configs/vllm/fp8_round1.yaml --port 8082`. Verify `/health` 200 and warm with one dummy transcription.
5. **Run binding measurement**: `VOXTRAL_VENV=~/.venvs/voxtral-l4 bash scripts/reproduce_round2_audio.sh`.
   The script uses the **locked `160/320` gate flags** and wraps each call with
   `scripts/measure_energy.py` for CodeCarbon.
6. **Quality gate** per language: normalized WER ≤ 1.25 × BF16 baseline on same slice.
   **Energy gate**: total measured kJ must beat Round-1 total `345.7 kJ`.
7. **Repackage** `submission/hf_model_repo/` with the new audio-prep flags in `reproduce.sh`
   and the new per-language reports. Push to HF as a new commit with a Round-2 tag.
8. **Send organizer email** about HI sample 1985 (`reports/sample_1985_investigation/hi_1985_findings.md`)
   before final submission.

## Things that are *not* in this candidate

- No SpinQuant / QuIP / rotation-based weight transforms. The integration was
  attempted in `~/.venvs/voxtral-spinquant` and hit three different llmcompressor
  blockers; even plain GPTQ in that venv crashes with a CUDA index-OOB during
  calibration forward. See the corresponding entry in `reports/team_status.md`.
- No EAGLE-3 speculative decoding. Worth a follow-up Round-2 sub-track once vLLM's
  EAGLE-3 support for the `/v1/audio/transcriptions` endpoint is verified.
- No HI sample `id=1985` exclusion. Day-2 measurement showed BF16 itself produces
  empty on idx 82, so the row is treated as a real evaluation failure mode at all
  precisions. The pre-drafted organizer email (`reports/sample_1985_investigation/
  hi_1985_findings.md`) should be sent so we have a documented external check
  before submission.
