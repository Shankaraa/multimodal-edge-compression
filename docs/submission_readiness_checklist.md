# Submission Readiness Checklist

## Purpose

This checklist is for the first important submission on the current FP8 mainline path.

It is intentionally practical:

- what is already ready
- what still needs to be tightened
- what should not block the first acceptance-oriented submission

## Current Submission Candidate

- Primary submission path:
  - `configs/vllm/fp8_round1.yaml`
- Active working server:
  - `http://localhost:8082/v1`
- Core benchmark reference:
  - [submission_benchmark_table.md](/C:/Users/ASUS/Music/Fine_tuning/docs/submission_benchmark_table.md)

## Must-Have Submission Elements

- [x] Working local runtime for the chosen path
- [x] Trusted BF16 reference point
- [x] Measured FP8 improvement on the core English comparison
- [x] Quiet-audio and empty-prediction issues addressed in the evaluator
- [x] Multilingual spot checks beyond English
- [x] Honest caveat handling for Japanese scoring
- [x] Submission-facing benchmark table
- [x] Mainline track doc
- [ ] Short final submission narrative polished
- [ ] One final acceptance-oriented checklist pass before sending

## Current Evidence We Can Already Stand Behind

### English apples-to-apples comparison

- BF16 quietfix:
  - `WER = 22.20%`
  - `elapsed_seconds = 46.26`
  - `energy_joules = 8112.90`
- FP8 round 1:
  - `WER = 21.97%`
  - `elapsed_seconds = 35.21`
  - `energy_joules = 4952.89`

### Multilingual FP8 checks

- `hi_in limit5`
  - `WER = 26.83%`
  - `empty_prediction_count = 0`
- `fr_fr limit5`
  - `WER = 23.18%`
  - `empty_prediction_count = 0`
- `ja_jp limit5`
  - `CER = 10.42%`
  - `CER(no-space) = 10.00%`
  - `empty_prediction_count = 0`

## What We Should Say In The First Submission

- The current submission path is FP8, not GPTQ.
- FP8 is chosen because it is already reproducible and benchmarked.
- The core claim is efficiency improvement without obvious quality regression on the current
  English reference.
- Multilingual spot checks show the path is not English-only.
- Japanese is included honestly with a CER-style reading because raw word WER is misleading there.

## What Should Not Block The First Submission

- GPTQ not being ready
- full automation of every experiment branch
- cloud deployment
- perfect packaging polish
- large multilingual sweeps

These are important later, but they should not stop the first strong submission.

## What Still Needs Tightening Before We Feel Fully Comfortable

- [ ] write or polish the short submission narrative
- [ ] decide whether one more multilingual spot check is worth the time
- [ ] make sure README points clearly to the current candidate docs
- [ ] keep the benchmark table and daily log aligned

## Recommended Final Pre-Submission Pass

Before the actual first submission:

1. Re-read the benchmark table.
2. Re-read the candidate summary.
3. Confirm the FP8 config and runtime values one last time.
4. Make sure the submission text does not over-claim beyond the current evidence.

## Decision Rule

If nothing clearly stronger is benchmark-ready, the first submission should go out on the FP8
mainline path.
