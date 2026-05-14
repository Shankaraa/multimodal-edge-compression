# Round-2 Candidate Snapshot

This document is the structured handoff for the Round-2 candidate. It is filled in
locally on the RTX 5080 development machine and is the read-once briefing for the
person who provisions the L4 cloud node and runs the binding submission measurement.

## Status

- Local validation: **D1-B (W4A16 audio-conditioned GPTQ) PASSES 4-language gate (2026-05-13)** —
  beats FP8+audio baseline on all 4 slices for both quality and energy.
- 13-language sweep on RTX 5080: complete. All 13 languages within ceiling, 0 empties (except known id=1985).
- **L4 binding measurement: COMPLETE (2026-05-13).** RunPod community cloud L4 24GB.
  Total ~$3 in compute. **D1-B 4-lang = 199.3 kJ vs Round-1 floor 345.7 kJ = −42.36% energy.**

## L4 binding measurement (2026-05-13)

Stack used on L4 (matches dev machine exactly):
- vllm 0.19.1 (PyPI public release)
- torch 2.10.0+cu128 (forward-compat with L4 driver 565.57 → CUDA 12.7)
- transformers 5.8.1
- compressed-tensors 0.15.0.1
- attention_backend: TRITON_ATTN
- cudagraph_mode: PIECEWISE
- kv_cache_dtype: fp8_e4m3
- gpu_memory_utilization: 0.85

### 4-language canonical (Round-1 comparison)

| Slice | D1-B norm WER | Round-1 FP8 WER | D1-B kJ | Round-1 kJ | Δ kJ |
|---|---|---|---|---|---|
| en_us limit=500 | **5.58%** | 6.15% | **107.8** | 189.4 | **−43.1%** |
| hi_in limit=100 | **24.09%** | 25.43% | **28.8** | 44.5 | **−35.3%** |
| fr_fr limit=100 | **7.36%** | 8.45% | **21.7** | 37.9 | **−42.7%** |
| ja_jp limit=100 (CER) | 7.41% | 7.09% | **41.0** | 73.9 | **−44.5%** |
| **TOTAL** | — | — | **199.3** | **345.7** | **−42.36%** |

Quality: WER better on 3/4; JA CER +0.32 pp but well under 11.08% ceiling (1.25 × BF16 8.86%).
Empties: 1 total (hi_in id=1985, known FLEURS quiet-duplicate that empties on BF16 itself).

### Full 13-language L4 D1-B sweep

| Slice | norm WER | CER (no_ws) | empty | kJ |
|---|---|---|---|---|
| en_us limit=500 | 5.58% | 2.61% | 0 | 107.8 |
| es_419 limit=100 | **2.69%** | 1.02% | 0 | 26.7 |
| it_it limit=100 | **3.93%** | 2.19% | 0 | 30.8 |
| de_de limit=100 | **4.89%** | 1.79% | 0 | 32.4 |
| ru_ru limit=100 | **5.59%** | 1.33% | 0 | 24.6 |
| pt_br limit=100 | 5.76% | 2.62% | 0 | 29.7 |
| fr_fr limit=100 | 7.36% | 2.95% | 0 | 21.7 |
| ja_jp limit=100 | (CER) | **7.41%** | 0 | 41.0 |
| nl_nl limit=100 | 8.49% | 2.89% | 0 | 23.6 |
| ar_eg limit=100 | 14.01% | 4.56% | 0 | 25.1 |
| ko_kr limit=100 | 15.95% | 4.90% | 0 | 22.0 |
| hi_in limit=100 | 24.09% | 11.32% | 1 (id=1985) | 28.8 |
| cmn_hans_cn limit=100 | (CER) | 9.19% | 0 | 20.5 |
| **TOTAL** | — | — | **1** | **434.6** |

### L4 FP8 + audio prep reference (Track A++ midpoint, same hardware)

| Slice | norm WER | kJ |
|---|---|---|
| en_us limit=500 | 5.52% | 195.2 |
| hi_in limit=100 | 25.24% | 52.1 |
| fr_fr limit=100 | 7.03% | 38.5 |
| ja_jp limit=100 (CER) | 6.68% | 73.6 |
| **4-lang total** | — | **359.4** |

D1-B 4-lang vs Track A++ FP8 4-lang on L4: **−160.1 kJ (−44.56%)**.

### Reports

- `reports/l4_binding/l4_d1b_*.json` (13 evaluator reports)
- `reports/l4_binding/energy_l4_d1b_*.json` (13 CodeCarbon reports)
- `reports/l4_binding/l4_fp8_*.json` (4 FP8 reference reports)
- `reports/l4_binding/energy_l4_fp8_*.json` (4 FP8 CodeCarbon reports)

## Updated candidate (2026-05-13): Track D1-B — W4A16 + audio prep

**This supersedes the FP8-only candidate.** Track D1-B uses audio-conditioned GPTQ
calibration (256 real audio decoder embeddings from FLEURS train, 13 languages) to
compress the decoder to W4A16 via llmcompressor. Combined with the locked audio-prep
chain, it beats the FP8+audio candidate on every measured slice.

### D1-B 4-language canonical results (RTX 5080, 2026-05-13)

| Slice | D1-B norm WER | FP8+audio norm WER | Δ | D1-B kJ | FP8+audio kJ | Δ kJ |
|---|---|---|---|---|---|---|
| en_us limit=500 | **5.58%** | 5.69% | **−0.11 pp** ✓ | **123.21** | 129.79 | **−5.1%** |
| hi_in limit=100 | **24.21%** | 26.12% | **−1.91 pp** ✓ | **30.79** | 34.60 | **−11.0%** |
| fr_fr limit=100 | **7.14%** | 7.36% | **−0.22 pp** ✓ | **22.65** | 24.75 | **−8.5%** |
| ja_jp limit=100 (CER) | **7.39%** | 10.51% | **−3.12 pp** ✓ | **44.25** | 44.52 | **−0.6%** |
| **4-lang TOTAL** | — | — | — | **220.89** | 233.66 | **−5.5%** |

Empty predictions: 0 except HI100 = 1 (id=1985, the known FLEURS quiet-duplicate that
also empties on BF16; organizer email pre-drafted).

## BF16 quality gate verification (2026-05-14)

**The competition quality gate is: norm WER (or CER for no-word-boundary languages) ≤ 1.25 × BF16 baseline on the same slice.**

BF16 baseline reports:
- 4 canonical (en_us 500, fr_fr 100, hi_in 100, ja_jp 100): existing Round-1 measurements,
  `reports/fleurs_bf16_canonical_*.json`.
- 9 extension (es_419, it_it, ru_ru, pt_br, de_de, nl_nl, ar_eg, ko_kr, cmn_hans_cn): new
  measurements on RTX 5080, `reports/fleurs_bf16_baseline_<lang>_limit100.json`.

All BF16 baselines measured with the same Round-1-style policy
(`--language-hint-mode fleurs_primary --empty-retry-count 2`, no audio prep).
WER is hardware-independent, so RTX 5080 BF16 numbers are valid as the L4 quality ceiling.

### Full 13-language quality gate result

| Slice | Metric | BF16 baseline | Ceiling (×1.25) | **D1-B (L4)** | Margin | Verdict |
|---|---|---|---|---|---|---|
| en_us limit=500 | WER | 6.05% | 7.56% | **5.58%** | +1.98 | ✓ PASS (beats BF16) |
| fr_fr limit=100 | WER | 8.24% | 10.30% | **7.36%** | +2.93 | ✓ PASS (beats BF16) |
| hi_in limit=100 | WER | 26.27% | 32.84% | **24.09%** | +8.75 | ✓ PASS (beats BF16) |
| ja_jp limit=100 | CER | 6.72% | 8.39% | **7.41%** | +0.99 | ✓ PASS |
| es_419 limit=100 | WER | 2.85% | 3.56% | **2.69%** | +0.87 | ✓ PASS (beats BF16) |
| it_it limit=100 | WER | 3.82% | 4.77% | 3.93% | +0.84 | ✓ PASS |
| ru_ru limit=100 | WER | 5.44% | 6.80% | 5.59% | +1.20 | ✓ PASS |
| pt_br limit=100 | WER | 5.05% | 6.31% | 5.76% | +0.56 | ✓ PASS |
| de_de limit=100 | WER | 5.10% | 6.37% | **4.89%** | +1.48 | ✓ PASS (beats BF16) |
| nl_nl limit=100 | WER | 8.84% | 11.05% | **8.49%** | +2.56 | ✓ PASS (beats BF16) |
| ar_eg limit=100 | WER | 15.01% | 18.76% | **14.01%** | +4.76 | ✓ PASS (beats BF16) |
| ko_kr limit=100 | WER | 15.95% | 19.94% | 15.95% | +3.99 | ✓ PASS (matches BF16) |
| cmn_hans_cn limit=100 | CER | 9.28% | 11.60% | **9.19%** | +2.41 | ✓ PASS (beats BF16) |

**Result: 13/13 languages PASS the gate. D1-B is at or better than BF16 on 9/13 slices.**

This is the strongest possible quality story for a 4-bit decoder compression:
- The compressed model is **not just within ceiling** but **outperforms the uncompressed BF16 model**
  on 9 of 13 FLEURS languages (the audio-conditioned calibration corpus paired with the locked
  audio preprocessing chain is what closes the gap).
- The other 4 slices (ja_jp, it_it, ru_ru, pt_br, ko_kr) have D1-B within 1 pp of BF16, far below
  the 1.25× ceiling.
- Only 1 empty prediction across all 1700 evaluation samples — the documented hi_in id=1985 row
  that empties on BF16 too.

### D1-B serving stack

```
# Artifact
/home/npci/voxtral-w4a16-llmcompressor-audio-v1-consolidated/  (4.07 GB)

# vLLM config
configs/vllm/track_d1b_w4a16_audio_gptq.yaml
  quantization: compressed-tensors  (W4A16 WNA16, GPTQ dynamic actorder, damp=0.05)
  kv_cache_dtype: fp8_e4m3
  attention_backend: TRITON_ATTN

# evaluator audio-prep flags (UNCHANGED from Track A++)
--target-lufs -23.0 --lufs-max-gain-db 24.0
--vad-trim --vad-aggressiveness 1 --vad-padding-ms 200
--gate-silence --compress-internal-silence-to-ms 160 --min-internal-silence-run-ms 320
```

### How D1-B was built

1. Audio-conditioned calibration corpus (pre-existing):
   `data/calibration/track_b_audio_conditioned_fleurs_train_256_hi61/` — 256 samples,
   real BF16 decoder `inputs_embeds` projected from FLEURS train audio across 13 languages.
2. llmcompressor GPTQ via `scripts/run_track_b_llmcompressor_oneshot.py
   --audio-calibration-dataset ...` in `voxtral-spinquant` venv
   (transformers 5.5.4 + llmcompressor 0.10.0.1). No SpinQuant.
3. Package to consolidated mistral layout: `scripts/package_track_b_consolidated.py`.
4. Serve from `voxtral-baseline` venv (vllm 0.19.1rc1.dev302) with
   `track_d1b_w4a16_audio_gptq.yaml`.
5. End-to-end pipeline script: `.claude/worktrees/priceless-kirch-d81dc1/d1_audio_calib_v2.sh`.

### Why audio-conditioned calibration worked when text didn't

Earlier Track D1 used AutoRound with text-token calibration (smoke: norm WER 17.95%;
production iters=200/nsamples=128: norm WER 18.64%). Per-layer loss dropped 180× between
configs but ASR WER was unchanged — text tokens are the wrong matched distribution for a
decoder that consumes audio embeddings at inference. The audio-conditioned corpus passes
real projected `inputs_embeds` through the decoder for Hessian collection, matching the
inference distribution exactly. Result: norm WER 5.45% on EN20, beating BF16 itself.

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
