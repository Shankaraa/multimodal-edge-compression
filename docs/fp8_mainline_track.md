# FP8 Mainline Track

## Purpose

This track is the practical execution path for the competition.

Its job is to turn the currently working FP8 result into the strongest reproducible submission
story we can build on this machine.

## Why This Track Exists

FP8 is currently the best working compression path we have because:

- it already serves successfully through `vLLM`
- it already improves energy and latency
- it is already holding quality on our current checks
- it already works across multiple languages

This means FP8 is the safest path to a strong competition result while the GPTQ branch is still
research.

## Current Findings

## Runtime State

- Active working server:
  - `http://localhost:8082/v1`
- Current working config:
  - `configs/vllm/fp8_round1.yaml`
- Practical local runtime envelope:
  - `max_model_len: 8192`
  - `gpu_memory_utilization: 0.85`
  - `kv_cache_dtype: fp8_e4m3`
  - `enable_prefix_caching: true`

## Evaluation State

The evaluation pipeline is much stronger than it was initially.

Important fixes already in place:

- transcription requests are serialized through the shared API helper to avoid overlapping-engine
  crashes
- quiet audio is boosted before evaluation to avoid deterministic empty transcripts
- per-sample diagnostics are written into the report JSON
- empty predictions are now counted explicitly

All new comparisons should use this fixed evaluator, not the older reports.

## Best Reference Numbers So Far

### BF16 quiet-audio-aware reference

- `en_us limit20`
  - `WER = 22.20%`
  - `empty_prediction_count = 0`
  - `elapsed_seconds = 46.26`
  - `energy_joules = 8112.90`

### FP8 current best result

- `en_us limit20`
  - `WER = 21.97%`
  - `empty_prediction_count = 0`
  - `elapsed_seconds = 35.21`
  - `energy_joules = 4952.89`

### FP8 multilingual spot checks

- `hi_in limit5`
  - `WER = 26.83%`
  - `empty_prediction_count = 0`
- `fr_fr limit5`
  - `WER = 23.18%`
  - `empty_prediction_count = 0`
- `ja_jp limit5`
  - `WER = 100.00%`
  - `CER = 10.42%`
  - `CER(no-space) = 10.00%`
  - `empty_prediction_count = 0`

## What These Results Mean

- FP8 is already materially better than BF16 on efficiency.
- FP8 is not showing an obvious quality regression in our current English, Hindi, and French
  checks.
- Japanese exposed a scoring caveat, not a serving failure.
- FP8 is currently the best submission-grade path in the repo.

Interpretation of the Japanese run:

- the raw word-based WER is misleading
- the outputs are still non-empty and clearly Japanese
- CER-style scoring is the better way to read that run

## Main Goal

Strengthen the FP8 evidence until it becomes very hard to beat:

- better multilingual coverage
- better benchmark clarity
- cleaner comparison tables
- cleaner reporting for the final story

## Approach

## 1. Expand multilingual evidence carefully

Add languages one by one instead of jumping to large broad sweeps.

Reason:

- we want signal, not noise
- we want to catch failure modes early
- we want each added report to be interpretable

## 2. Keep comparisons apples-to-apples

Every new compression result should be compared against the quiet-audio-aware BF16 reference, not
the older empty-containing runs.

Reason:

- the older runs understate the true baseline quality
- mixing old and new evaluator behavior would make the story weaker

## 3. Keep the runtime stable

Rules for this track:

- do not overlap transcription requests
- keep using the shared API helper
- do not casually change the working FP8 config
- do not stop the FP8 server unless another controlled experiment needs the GPU

## 4. Build the benchmark story as we go

Do not wait until the end to organize results.

As new evaluations are added, keep the following clean:

- WER
- empty prediction count
- elapsed time
- energy
- emissions
- language
- active config

## Immediate Tasks For This Track

1. Re-run the English submission slice with the updated FP8 KV cache config.
2. Warm prefix cache before measured runs with `scripts/warm_fleurs_prefix_cache.py`.
3. Add one more useful multilingual check if broader confidence is still needed.
4. Build a simple benchmark matrix from the current BF16 and FP8 reports.
5. Keep Japanese-like languages interpreted with CER-aware metrics instead of raw WER alone.
6. Keep the daily log and README aligned with the latest validated checkpoint.

## Prefix-Caching Constraint

The current WSL `vLLM` build does support prefix caching for the speech-to-text path, but the
`/v1/audio/transcriptions` request model in this runtime does not expose `cache_salt`.

Implication:

- warmup is still worth doing
- the warmup should happen against the same live server process that will handle the measured run
- multi-tenant salted cache partitioning is not available on this path today

## Success Criteria

This track is succeeding if:

- FP8 keeps matching or nearly matching BF16 quality
- FP8 keeps outperforming BF16 on efficiency
- multilingual checks remain stable
- the results are easy to explain and defend

## Files That Matter Most

- `configs/vllm/fp8_round1.yaml`
- `scripts/evaluate_fleurs.py`
- `scripts/measure_energy.py`
- `src/voxtral_project/api.py`
- `src/voxtral_project/audio.py`
- `reports/fleurs_bf16_en_us_limit20_quietfix.json`
- `reports/energy_fleurs_bf16_en_us_limit20_quietfix.json`
- `reports/fleurs_fp8_en_us_limit20_quietfix.json`
- `reports/energy_fleurs_fp8_en_us_limit20_quietfix.json`
- `reports/fleurs_fp8_hi_in_limit5_quietfix.json`
- `reports/fleurs_fp8_fr_fr_limit5_quietfix.json`
- `reports/fleurs_fp8_ja_jp_limit5_quietfix_v2.json`
- `reports/energy_fleurs_fp8_ja_jp_limit5_quietfix_v2.json`

## Decision Rule

Unless another branch produces clearly better evidence, FP8 stays the primary competition path.
