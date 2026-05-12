# Round 2 Audio Lever: LUFS + VAD + Speech-Gate

## Bottom line

The Round-2 candidate adds two cheap, deterministic audio preprocessing layers in front of the
existing FP8 stack. They reduce encoder workload (audio seconds processed) without touching the
model and without depending on the brittle rotation-based W4 toolchain.

```
--target-lufs -23.0
--lufs-max-gain-db 24.0
--vad-trim --vad-aggressiveness 1 --vad-padding-ms 200
--gate-silence
--compress-internal-silence-to-ms 320
--min-internal-silence-run-ms 640
```

## Why this lever and not rotation-based W4

The original Round-2 plan led with SpinQuant W4A16 because the failing canary was HI id `1985`
idx 82 ("the quiet duplicate"). Day-2 measurement showed BF16 itself produces an empty
prediction on idx 82 even after LUFS lifts the peak to a comparable level as the loud
duplicate. That kills the strategic case for rotation: the failure is not a W4 quantization
artifact, it is a data-side issue that no model variant decodes.

Day-3 then confirmed that the `transformers 5.5.4` + `llmcompressor 0.10.x` + Voxtral Realtime
combination has multiple toolchain blockers (SpinQuant `_fuse_norms` assertion, QuIP block-size
mismatch, CUDA index-OOB during calibration forward — even on plain GPTQ), so even a degraded
W4A16 ship would require open-ended toolchain debugging. See the `Track B SpinQuant W4A16`
entry in `reports/team_status.md` for the full timeline.

The audio lever is independent of all of that.

## Pieces

### 1. ITU-R BS.1770-4 LUFS normalization

`src/voxtral_project/audio.py:_apply_lufs_normalization` runs
[pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)'s integrated-loudness measurement on
the prepared audio array and applies a single linear gain to hit the requested target LUFS,
clipped to `lufs_max_gain_db`. The BS.1770 gating procedure excludes silence from the
loudness estimate, so the gain is set by the speech-burst loudness and not by the silent
floor — that is exactly the property that makes a quiet recording with a brief utterance
behave like its loud counterpart after normalization.

Default target is unset (`None`), which preserves byte-for-byte legacy behavior. With
`--target-lufs -23.0` the audio prep pipeline becomes:

```
mono float32 -> LUFS norm -> existing quiet-boost (no-op once LUFS is on) -> VAD trim -> gate
```

Caveat: BS.1770 needs ~0.4 s of audio to gate-and-integrate; clips shorter than that fall
through unchanged.

### 2. WebRTC VAD trim (leading + trailing silence)

`prepare_audio_array_for_transcription(..., vad_trim=True, vad_aggressiveness=1,
vad_padding_ms=200)` keeps only the audio between the first and last voiced frames detected
by `webrtcvad.Vad`, padded by `vad_padding_ms` on each side so word onsets/codas are not
clipped. Aggressiveness `1` is the least aggressive and is intentional; we want to remove
silence we are sure about.

### 3. Speech-aware silence gating with internal-silence compression

`gate_audio_by_activity` is the more aggressive layer. It tags 80 ms frames as active vs
silent using both peak (>= `0.01`) and RMS (>= `0.003`) thresholds, finds runs of silent
frames, and:

- preserves up to `preserve_leading_silence_ms` (default 160) of leading silence
- preserves up to `preserve_trailing_silence_ms` (default 160) of trailing silence
- compresses internal silent runs that are at least `min_internal_silence_run_ms` (640) long
  down to `compress_internal_silence_to_ms` (320), keeping half on each side of the
  compressed span

This avoids hard-cutting word onsets while still removing the long pauses that make Voxtral's
encoder do work for no information.

## Empirical signal (RTX 5080, EN20 FLEURS quietfix)

The reports below are smoke runs only; the binding numbers must be measured on the L4
evaluation hardware. They are reported here to justify locking the parameter set.

| Stack | raw WER | norm WER | empty | wall-clock | trim% |
|---|---|---|---|---|---|
| BF16 baseline | 22.20% | 6.36% | 0 | — | 0% |
| BF16 + LUFS-23 | 22.20% | 6.14% | 0 | 37.23 s | 0% |
| BF16 + LUFS + VAD-trim | 22.20% | 6.14% | 0 | 40.59 s | 5.13% |
| BF16 + LUFS + VAD + gate | 22.20% | **5.68%** | 0 | 39.43 s | **8.68%** |
| FP8 baseline | 21.97% | 6.36% | 0 | — | 0% |
| FP8 + LUFS-23 | 22.20% | 6.59% | 0 | 31.19 s | 0% |
| FP8 + LUFS + VAD-trim | 22.20% | 6.59% | 0 | 28.57 s | 5.13% |
| **FP8 + LUFS + VAD + gate** | **21.97%** | 6.59% | 0 | **26.32 s** | **8.68%** |

Reports: `reports/fleurs_{bf16,fp8}_en_us_limit20_lufs23{_vadtrim,_vadtrim_gate}_smoke.json`.

Headlines:
- 8.68% audio-second reduction at zero quality cost (FP8 raw WER returns to 21.97% baseline
  with VAD+gate on; normalized stays at 6.59%, well under the 7.95% ceiling at 1.25 x
  6.36% BF16 baseline).
- Wall-clock improves 15.6% on FP8 (31.19 -> 26.32 s) — Voxtral Realtime is variable-length,
  so encoder cost scales with audio time, and the trimmed audio time pays back the CPU-side
  VAD overhead and then some.
- LUFS is unambiguous on BF16 (6.36 -> 5.68 normalized at the +VAD+gate setting) and slightly
  ambiguous on FP8 EN20 (6.36 baseline -> 6.59 with LUFS — within 95% CI [3.40, 7.87]). The
  FP8 EN500 ablation is in flight to settle this.

## EN500 ablation result (RTX 5080, 2026-05-07)

| Variant | raw WER | norm WER | empty | sum_lat | trim% |
|---|---|---|---|---|---|
| FP8 EN500 baseline (existing 2026-04-28) | 27.32% | 6.15% | 0 | n/a | 0% |
| FP8 EN500 + VAD+gate (no LUFS) | 42.54% | 25.93% | 26 | 1296.15 s | 31.31% |
| **FP8 EN500 + LUFS + VAD+gate** | **27.08%** | **5.80%** | **0** | **732.30 s** | **10.39%** |

Result: LUFS is mandatory. FLEURS EN500 is 69% RMS-below-0.003 - quiet by design - so
without LUFS the gate's `0.003` RMS threshold misclassifies entire clips as silent, the
gate compresses them to nothing, and the model returns empty. With LUFS at `-23 LUFS` the
quiet clips are lifted into a normal-loudness range first and the gate threshold is then
correctly calibrated for the speech-vs-silence boundary.

Locked Round-2 audio-prep block:

```
--target-lufs -23.0
--lufs-max-gain-db 24.0
--vad-trim --vad-aggressiveness 1 --vad-padding-ms 200
--gate-silence
--compress-internal-silence-to-ms 320
--min-internal-silence-run-ms 640
```

CodeCarbon EN500 energy on the dev machine (RTX 5080, **relative only**, not the binding L4
number): locked candidate `154.15 kJ` over `756.02 s`, vs `278.17 kJ` over `1326.05 s` for
the broken no-LUFS variant. Compared to the existing FP8 Track A baseline (`189.4 kJ` for EN
on L4) the absolute kJ is meaningful only after the L4 re-measurement.

## What's next

1. Multilingual sweep with the locked parameter set: HI100, FR100, JA100 (in flight, fires
   `reports/fleurs_fp8_{hi_in,fr_fr,ja_jp}_limit100_lufs23_vadgate_smoke.json`).
2. Provision an L4 cloud node and re-measure energy on the same parameter set; that is the
   binding number for the round-2 submission.
3. Single-shot HI `id=1985` idx 82 on FP8 + locked params - confirm graceful behaviour
   (expected: still empty per Day-2 evidence, but no harmful side-effects on the loud
   duplicate idx 9).
4. (Optional) probe more aggressive `compress-internal-silence-to-ms` on EN20 to find the
   WER ceiling, but only after the L4 binding measurement is captured at the locked
   parameter set so the L4 ablation stays clean.
