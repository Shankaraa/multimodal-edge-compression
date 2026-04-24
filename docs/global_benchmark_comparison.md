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

## Benchmark-Alignment Rescoring

We then added a second scoring profile to the repo:

- keep the current local normalized metrics for continuity
- add an `open_asr_like` profile inspired by `huggingface/open_asr_leaderboard`
- reuse the exact same saved predictions and references so the rescoring isolates scoring-frame
  effects instead of mixing in new inference variance

### English rescoring check

This was the most important check for the published-gap question.

| FP8 English slice | Current norm WER | Open-ASR-like WER | Read |
| --- | ---: | ---: | --- |
| `limit20` | 6.36% | 6.36% | no change |
| `limit100` | 5.96% | 5.96% | no change |
| `limit500` | 6.49% | 6.49% | no change |

This is the key result:

- the published English gap does not move at all under the new benchmark-aligned rescoring path
- so the remaining English difference is not explained by our current text-normalization choice
- the problem is farther out in the benchmark frame:
  - dataset wrapper differences
  - manifest procedure differences
  - or other evaluation-stack details beyond text normalization alone

### Three-language rescoring check

We also rescored the current same-harness `en_us` / `fr_fr` / `hi_in` side-by-side with the new
profile.

| Language | FP8 open-ASR-like WER | Whisper open-ASR-like WER | Read |
| --- | ---: | ---: | --- |
| `en_us` | 6.36% | 4.32% | Whisper ahead |
| `fr_fr` | 8.11% | 6.52% | Whisper ahead |
| `hi_in` | 14.74% | 13.82% | Whisper slightly ahead |

That changed the multilingual read in an important way:

- English stayed exactly the same
- French moved only slightly
- Hindi moved a lot for both systems, and the slight local FP8 edge flipped to a slight Whisper edge
- the three-language macro average under `open_asr_like` scoring is no longer especially close:
  - FP8: `9.74%`
  - Whisper: `8.22%`

## Public Wrapper Comparison

We then ran the next direct test on the public `open_asr_multilingual` wrapper itself instead of
staying inside `google/fleurs`.

This used the current English `limit20` comparison for both systems.

| Dataset wrapper | System | Raw WER | Open-ASR-like WER | Energy (J) | Read |
| --- | --- | ---: | ---: | ---: | --- |
| `google_fleurs` | FP8 | 21.97% | 6.36% | 4952.89 | current local anchor |
| `google_fleurs` | Whisper | 20.59% | 4.32% | 3258.57 | current local anchor |
| `open_asr_multilingual` | FP8 | 14.35% | 7.01% | 8244.80 | public wrapper check |
| `open_asr_multilingual` | Whisper | 11.24% | 4.21% | 8844.86 | public wrapper check |

The important comparison is the benchmark-aligned WER:

- FP8 moved from `6.36%` to `7.01%`
- Whisper moved from `4.32%` to `4.21%`
- so the English FP8-minus-Whisper gap widened from `2.05` points to `2.80` points

This is the key verdict from the wrapper run:

- dataset-wrapper differences do matter
- but they do not move the English story in Voxtral's favor
- the public wrapper makes the current local English quality gap look slightly worse for FP8, not
  better

One runtime caveat is worth being explicit about:

- the Whisper `open_asr_multilingual` run wrote valid output files, but the Python process still
  ended with a late finalization crash (`return_code = -6`) after saving the report and energy
  JSON
- so the quality numbers are usable, but that environment quirk should not be mistaken for a clean
  runner state yet

## What This Actually Means

- FP8 is still a real win over our BF16 Voxtral reference because it preserves normalized English
  WER while cutting time and energy materially.
- Whisper large-v3 currently beats our local Voxtral setup on the same normalized English slice.
- The multilingual picture now depends on which scoring frame we are using:
  - under our local normalized scorer, FP8 still holds a slight Hindi edge
  - under the new `open_asr_like` scorer, Whisper is ahead on all three current languages
- The published-gap diagnosis is now clearer:
  - decoding knobs we tested did not explain it
  - one intermediate larger English slice narrowed it materially
  - the much larger English slice did not hold that gain
  - a larger multilingual side-by-side showed the external comparison is mixed and closer than the
    tiny spot checks implied
  - benchmark-aligned rescoring did not improve the English gap at all
  - the public `open_asr_multilingual` wrapper widened the English FP8-vs-Whisper gap instead of
    narrowing it
- So the current FP8 submission path is credible and efficient, but not yet the strongest overall
  ASR result on our current benchmark frames, and the published English gap now looks much more
  like a full evaluation-stack issue than a slice-size or scorer issue.

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
- strong external baselines like Whisper large-v3 still set a higher bar on quality under the
  current benchmark-aligned rescoring view
- FP8 remains attractive because it is still the strongest compressed Voxtral path here and it uses
  less evaluation energy across the current three-language local side-by-side

## Best Next Step

1. move from the wrapper check to a closer manifest-style benchmark procedure rather than
   continuing local scorer experiments
2. keep FP8 as the submission mainline, but describe it honestly as efficient and practical,
   not quality-leading against strong public baselines
3. only run another large English slice if we need variance estimates, not as the main explanation
   for the published gap
