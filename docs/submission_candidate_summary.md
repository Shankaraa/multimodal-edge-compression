# Submission Candidate Summary

## Current Candidate

The current first-submission candidate is the FP8 mainline path:

- config:
  - `configs/vllm/fp8_round1.yaml`
- active local server:
  - `http://localhost:8082/v1`

## Why This Is The Candidate

This is the current best candidate because it already gives us three things at the same time:

- reliable serving
- measurable efficiency improvement
- no obvious quality regression on the strongest English comparison we trust

## Core Claim

Our current claim is:

FP8 is the first compression path in this project that already delivers reproducible efficiency
wins without an obvious quality loss on the current local benchmark suite.

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

## Honest Caveats

- GPTQ is still a research branch, not a submission-ready branch
- Japanese raw word-based WER is misleading because of segmentation mismatch
- our strongest comparison is still local and relatively small, especially outside English

These are acceptable caveats for a first submission as long as we describe them honestly.

## What We Should Emphasize

- we fixed real evaluation issues instead of hiding them
- the current benchmark path is stronger and more honest than the earlier repo state
- FP8 is already a practical compression improvement, not just a theoretical one

## What We Should Avoid Over-Claiming

- do not claim GPTQ results
- do not present Japanese raw WER as the true quality signal
- do not imply large-scale multilingual validation beyond the spot checks we actually ran

## One-Sentence Submission Framing

We established a reliable BF16 reference, fixed evaluation blind spots, and showed that an FP8
decoder-focused serving path already improves efficiency materially while maintaining competitive
transcription quality across our current multilingual checks.
