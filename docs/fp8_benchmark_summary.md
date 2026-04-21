# FP8 Benchmark Summary

## Purpose

This document is the compact benchmark view for the current FP8 mainline track.

It is meant to answer three questions quickly:

1. Is FP8 better than the BF16 reference on this machine?
2. Is FP8 holding up across multiple languages?
3. Are there any evaluation caveats we need to explain honestly?

## Current Runtime

- Active working server:
  - `http://localhost:8082/v1`
- Active config:
  - `configs/vllm/fp8_round1.yaml`
- Proven local runtime envelope:
  - `max_model_len: 8192`
  - `gpu_memory_utilization: 0.85`

## Baseline Reference

### BF16 quiet-audio-aware reference

| Run | Samples | WER | Empty preds | Elapsed (s) | Energy (J) |
| --- | --- | ---: | ---: | ---: | ---: |
| `en_us` | 20 | 22.20% | 0 | 46.26 | 8112.90 |

## FP8 Results

### English comparison

| Run | Samples | Raw WER | Norm WER | Empty preds | Elapsed (s) | Energy (J) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `en_us` | 20 | 21.97% | 6.36% | 0 | 35.21 | 4952.89 |

### External same-slice anchor

| System | Samples | Raw WER | Norm WER | Empty preds | Elapsed (s) | Energy (J) | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Whisper large-v3 | 20 | 20.59% | 4.32% | 0 | 34.77 | 3258.57 | stronger external baseline on this English slice |

### External multilingual anchors

| Language | Whisper raw WER | Whisper norm WER | Notes |
| --- | ---: | ---: | --- |
| `fr_fr` | 21.85% | 8.07% | ahead of the current FP8 French spot check |
| `hi_in` | 32.52% | 28.46% | behind the current FP8 Hindi spot check |

### Multilingual spot checks

| Run | Samples | WER | Empty preds | Energy (J) | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `hi_in` | 5 | 26.83% | 0 | 1620.28 | Strong spot check, no empty outputs |
| `fr_fr` | 5 | 23.18% | 0 | 2121.87 | Strong spot check, no empty outputs |
| `ja_jp` | 5 | 100.00% | 0 | 2692.89 | Raw WER is not trustworthy here due tokenization mismatch |

## Japanese Metric Caveat

The Japanese run should not be read as “the model completely failed.”

What happened:

- the predictions are non-empty and clearly Japanese
- the FLEURS Japanese references contain spaces
- the generated Japanese predictions are mostly unsegmented
- standard word-based WER therefore becomes a poor metric here

Useful interpretation:

- reported `WER = 100.00%`
- character error rate on the same 5 samples is about:
  - `CER = 10.42%`
- spacing-agnostic character error rate on the same 5 samples is about:
  - `CER(no-space) = 10.00%`

So Japanese currently exposes an evaluation-metric caveat, not a simple FP8 collapse.

## Main Findings

- FP8 is already better than BF16 on English efficiency:
  - about `24%` faster
  - about `39%` lower energy
- FP8 is holding normalized quality on the current English Voxtral reference.
- FP8 is also producing stable non-empty outputs on Hindi, French, and Japanese.
- Japanese needs better scoring treatment before it should be used as a headline quality number.
- Whisper large-v3 currently beats our local Voxtral setup on the same normalized English slice.
- The external multilingual comparison is mixed rather than uniformly against FP8.

## Submission-Relevant Interpretation

Right now, FP8 is the strongest practical compression path because:

- it works reliably in the current runtime
- it improves efficiency materially
- it has multilingual evidence
- its biggest new issue is a metric caveat, not a serving failure
- it is the best current compressed Voxtral path, even though it is not yet beating the strongest
  external baseline we checked

## Best Next Step

The next most useful FP8 mainline improvements are:

1. expand the benchmark matrix with one more carefully chosen language only if broader confidence is still needed
2. keep comparing all new compressed runs against the quiet-audio-aware BF16 reference
3. turn the current benchmark snapshot into a cleaner submission-facing table or chart
