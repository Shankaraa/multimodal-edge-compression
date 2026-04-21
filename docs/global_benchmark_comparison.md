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

## Larger Multilingual Side-by-Side

We then upgraded the multilingual comparison from tiny `limit5` checks to a broader same-harness
view:

- keep the existing `en_us limit20` anchor
- expand `fr_fr` from `5` to `20`
- expand `hi_in` from `5` to `20`
- keep the same local evaluator and normalized scoring path for both systems

| Language | Voxtral FP8 raw WER | Voxtral FP8 norm WER | Whisper raw WER | Whisper norm WER | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `en_us` | 21.97% | 6.36% | 20.59% | 4.32% | Whisper clearly ahead |
| `fr_fr` | 24.91% | 8.33% | 23.04% | 6.73% | Whisper ahead |
| `hi_in` | 30.10% | 23.91% | 29.33% | 25.43% | FP8 slightly ahead on normalized WER |

This broader multilingual check changed the interpretation in an important way:

- the earlier Hindi `limit5` edge was directionally right, but it overstated how large the gap was
- Whisper still leads on English and French
- FP8 still holds Hindi on normalized WER, but only narrowly
- on a simple three-language macro average, normalized WER is now close:
  - FP8: `12.87%`
  - Whisper: `12.16%`
- across those same three evaluation slices, FP8 also used less total measured evaluation energy:
  - FP8: `21494.31 J`
  - Whisper: `23473.44 J`

## Gap Diagnosis Follow-Up

We tested the most obvious decoding-side hypotheses directly on the local FP8 Voxtral path.

### Controlled English `limit20` checks

| Setting | Raw WER | Norm WER | Read |
| --- | ---: | ---: | --- |
| Current default path | 21.97% | 6.36% | baseline |
| `temperature = 0.0` | 22.43% | 7.05% | slightly worse |
| `temperature = 0.0` + English hint | 22.43% | 7.05% | no improvement |

So the easy explanation was wrong:

- the published-gap is not mainly caused by forgetting `temperature = 0.0`
- the explicit English language hint did not help either

### Larger English slice checks

We then tested whether the gap was mostly a sample-size artifact by expanding the FP8 English run
twice, first to `100` samples and then to `500`.

| Setting | Samples | Raw WER | Norm WER | Empty preds | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| FP8 default | 20 | 21.97% | 6.36% | 0 | smaller slice |
| FP8 default | 100 | 27.06% | 5.96% | 0 | intermediate slice, looked closer to the published reference |
| FP8 default | 500 | 27.58% | 6.49% | 1 | much larger slice, moves back away from the published reference |

This changes the diagnosis again:

- the `limit100` result was probably the optimistic slice, not the stable local number
- the much larger `limit500` run lands at `6.49%` normalized WER, not near the published Voxtral
  English reference of roughly `4.9%`
- raw WER stays noisy because punctuation and formatting effects are still substantial
- there was `1` empty prediction in the `limit500` run, but not enough to explain the full gap by
  itself

So the remaining gap now looks less like "we just needed a bigger English slice" and more like:

- benchmark-stack differences
- official normalization or evaluation differences
- with sample-size variance still present, but not the main explanation

than a single missing decoding flag.

## What This Actually Means

- FP8 is still a real win over our BF16 Voxtral reference because it preserves normalized English
  WER while cutting time and energy materially.
- Whisper large-v3 currently beats our local Voxtral setup on the same normalized English slice.
- The multilingual picture is still mixed, but now on a less flimsy frame:
  - Whisper is ahead on English and French at `limit20`
  - FP8 is still ahead on Hindi normalized WER at `limit20`, but only slightly
  - the simple three-language macro average is close enough that the story is no longer
    "Whisper dominates everywhere"
- The published-gap diagnosis is now clearer:
  - decoding knobs we tested did not explain it
  - one intermediate larger English slice narrowed it materially
  - the much larger English slice did not hold that gain
  - a larger multilingual side-by-side showed the external comparison is mixed and closer than the
    tiny spot checks implied
- So the current FP8 submission path is credible, efficient, and competitive, but not yet the
  strongest overall ASR result on our current local benchmark frame, and the published English gap
  now looks more like a benchmarking-frame issue than a slice-size issue.

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
- strong external baselines like Whisper large-v3 still set a slightly higher bar on the current
  multilingual quality view
- FP8 remains attractive because it stays close on normalized WER while using less evaluation
  energy across the current three-language side-by-side

## Best Next Step

1. align more closely to the official benchmark stack and normalization pipeline rather than
   continuing local flag sweeps or even larger English reruns
2. keep FP8 as the submission mainline, but describe it honestly as efficient and competitive,
   not clearly ahead of strong public ASR baselines
3. only run another large English slice if we need a variance estimate, not as the main gap
   explanation anymore
