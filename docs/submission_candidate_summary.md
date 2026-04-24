# Submission Candidate Summary

## Current Candidate

The current first-submission candidate is the FP8 mainline path:

- config:
  - `configs/vllm/fp8_round1.yaml`
- active local server:
  - `http://localhost:8082/v1`
- runtime stack:
  - `quantization: fp8`
  - `kv_cache_dtype: fp8_e4m3`
  - `enable_prefix_caching: true` in config, but not part of the defended performance claim

## Why This Is The Candidate

This is the current best candidate because it already gives us three things at the same time:

- reliable serving
- measurable efficiency improvement
- no obvious quality regression on the strongest English comparison we trust

## Core Claim

Our current claim is:

FP8 is the first compression path in this project that already delivers reproducible efficiency
wins against the BF16 Voxtral reference while holding the same normalized English score on the
trusted local slice.

That is the round-one claim. It is strong enough to defend, and narrow enough not to break under
review.

## Evidence

### English reference comparison

- BF16 quietfix:
  - `WER = 22.20%`
  - `0` empty predictions
  - `46.26 s`
  - `8112.90 J`
- FP8 round 1:
  - `WER = 21.97%`
  - `0` empty predictions
  - `35.21 s`
  - `4952.89 J`

Interpretation:

- FP8 is about `24%` faster
- FP8 uses about `39%` less energy
- FP8 did not regress on the trusted English comparison
- normalized English WER is `6.36%` for both BF16 and FP8 on this slice
- this is still the strongest clean result in the repo for a first submission

### External benchmark anchor

- Whisper large-v3 on the same local `en_us limit20` slice:
  - raw `WER = 20.59%`
  - normalized `WER = 4.32%`
  - `0` empty predictions
  - `34.77 s`
  - `3258.57 J`

Interpretation:

- FP8 remains the best compressed Voxtral path we have
- Whisper large-v3 is currently stronger than our local Voxtral setup on the same normalized
  English slice
- the newer benchmark-aligned rescoring pass does not change that English result at all
- that means our current first submission is credible, but not yet a claim of beating strong
  public ASR baselines
- the multilingual external picture is more favorable to Whisper than our earlier local
  normalization suggested:
  - under the benchmark-aligned `open_asr_like` scorer, Whisper is slightly ahead on English,
    French, and Hindi
  - the three-language macro average is `9.74%` for FP8 versus `8.22%` for Whisper

### Multilingual support signal

- `hi_in limit5`
  - `WER = 26.83%`
  - `0` empty predictions
- `fr_fr limit5`
  - `WER = 23.18%`
  - `0` empty predictions
- `ja_jp limit5`
  - `CER = 10.42%`
  - `CER(no-space) = 10.00%`
  - `0` empty predictions

Interpretation:

- FP8 is not only working on English
- the path remains stable across multiple languages
- Japanese requires CER-aware interpretation rather than raw WER

## Why This Can Still Get Through Round One

The right frame is not "did we beat Whisper?" The right frame is "did we build a credible,
measured, deployment-minded compression result?"

Why the answer is still yes:

- the serving path is real and reproducible, not a notebook-only artifact
- the efficiency gain against the BF16 Voxtral reference is material and measured
- the evaluation pipeline was hardened instead of padded:
  - quiet audio is handled
  - empty predictions are counted
  - requests are serialized to avoid engine artifacts
  - benchmark-aligned rescoring was added even though it made the external comparison harsher
- multilingual behavior is stable enough to show the result is not an English-only fluke

## Honest Caveats

- GPTQ is still a research branch, not a submission-ready branch
- Japanese raw word-based WER is misleading because of segmentation mismatch
- our strongest comparison is still local and relatively small, especially outside English
- strong external baselines like Whisper large-v3 still outperform our current local Voxtral
  setup on the normalized English slice
- the benchmark-aligned rescoring pass strengthened that external gap story rather than weakening
  it
- the recent prefix-cache validation run did not show realized cache reuse on
  `/v1/audio/transcriptions`, so prefix caching should not be sold as a validated round-one gain

These are acceptable caveats for a first submission as long as we describe them honestly.

## What We Should Emphasize

- we fixed real evaluation issues instead of hiding them
- the current benchmark path is stronger and more honest than the earlier repo state
- FP8 is already a practical compression improvement, not just a theoretical one
- the round-one edge should come from disciplined measurement and deployment realism, not from
  speculative runtime additions

## What We Should Avoid Over-Claiming

- do not claim GPTQ results
- do not present Japanese raw WER as the true quality signal
- do not imply large-scale multilingual validation beyond the spot checks we actually ran
- do not claim we are beating strong global ASR baselines yet
- do not claim a prefix-cache speedup or cache reuse result that we have not actually measured

## One-Sentence Submission Framing

We built a reproducible FP8 Voxtral serving path, fixed important evaluation blind spots, and
showed a material efficiency win over the BF16 reference while keeping a credible multilingual
quality bar, even though stronger public ASR baselines still lead on the current benchmark-aligned
view.
