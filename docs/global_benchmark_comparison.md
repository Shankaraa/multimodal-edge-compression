# Global Benchmark Comparison

## Purpose

This note anchors the current local Voxtral compression work against a serious external ASR
baseline.

The right question is not "does our raw WER look good?" The right question is:

- are we evaluating with a metric that is even comparable to public ASR benchmarks,
- does the current FP8 Voxtral path beat our own BF16 reference,
- and how far are we from a strong external baseline on the same local slice.

## Metric Correction

Our earlier local reports used raw string WER and CER directly on:

- lowercase punctuation-light FLEURS references
- punctuated, capitalized model predictions

That is fine for internal consistency, but it is not a good public-comparison metric by itself.

We now also compute normalized ASR metrics with:

- Unicode NFKC normalization
- casefolding
- punctuation and symbol stripping
- control removal
- whitespace collapsing

Use normalized WER when comparing to public baselines. Keep raw WER in the report for transparency.

## Same-Slice English Comparison

All rows below use the same local `google/fleurs` `en_us` test streaming slice with `20` samples.

| System | Backend | Raw WER | Normalized WER | Empty preds | Elapsed (s) | Energy (J) | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Voxtral BF16 quietfix | `vLLM` | 22.20% | 6.36% | 0 | 46.26 | 8112.90 | trusted local reference |
| Voxtral FP8 round 1 | `vLLM` | 21.97% | 6.36% | 0 | 35.21 | 4952.89 | best current compressed Voxtral path |
| Whisper large-v3 | `Transformers` | 20.59% | 4.32% | 0 | 34.77 | 3258.57 | strongest external baseline on this slice so far |

## Multilingual Spot-Check Anchors

These are still small-sample checks, but they are useful because they show the external comparison
is not uniformly one-sided.

| Language | Voxtral FP8 raw WER | Voxtral FP8 norm WER | Whisper raw WER | Whisper norm WER | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `en_us` | 21.97% | 6.36% | 20.59% | 4.32% | Whisper ahead |
| `fr_fr` | 23.18% | 10.56% | 21.85% | 8.07% | Whisper ahead |
| `hi_in` | 26.83% | 23.58% | 32.52% | 28.46% | FP8 ahead |

## What This Actually Means

- FP8 is still a real win over our BF16 Voxtral reference because it preserves normalized English
  WER while cutting time and energy materially.
- Whisper large-v3 currently beats our local Voxtral setup on the same normalized English slice.
- The multilingual picture is mixed:
  - Whisper is ahead on the current English and French spot checks
  - FP8 is ahead on the current Hindi spot check
- So the current FP8 submission path is credible, but not yet globally leading on this benchmark.

## Public Reference Anchors

These are useful public numbers to keep in mind while reading our local comparison:

- Whisper large-v3 model page:
  - [https://huggingface.co/openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- Voxtral Mini 4B Realtime model page:
  - [https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- Open ASR Leaderboard overview:
  - [https://huggingface.co/blog/open-asr-leaderboard](https://huggingface.co/blog/open-asr-leaderboard)

Those public numbers are still not perfectly apples-to-apples with our local run because they use
their own standardized evaluation stack, but they are the right direction for calibration.

## Current Honest Framing

The current submission story should be:

- we have a working compressed Voxtral path
- FP8 is materially better than our BF16 Voxtral serving reference on this hardware
- our evaluation is now more honest because it includes normalized ASR metrics
- strong external baselines like Whisper large-v3 still set a higher bar than our current local
  Voxtral setup

## Best Next Step

1. expand the external same-harness comparison to one or two more languages only if it changes
   submission strategy
2. investigate why the local Voxtral setup still trails the published Voxtral model-card English
   numbers
3. keep FP8 as the submission mainline, but avoid any claim that it beats strong public ASR
   baselines yet
