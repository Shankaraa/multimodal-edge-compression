# Submission Benchmark Table

## Purpose

This is the clean, submission-facing benchmark snapshot for the current mainline path.

Use this file when we need:

- a compact summary for decision-making
- a cleaner narrative for README or submission notes
- a quick comparison without reading the raw JSON reports

## Active Compression Baseline

- Mainline compression path:
  - `configs/vllm/fp8_round1.yaml`
- Active working server:
  - `http://localhost:8082/v1`
- Stable local runtime envelope:
  - `max_model_len: 8192`
  - `gpu_memory_utilization: 0.85`
  - `kv_cache_dtype: fp8_e4m3`

## Core Comparison

### English reference comparison

| Config | Backend | Language | Samples | Raw WER | Norm WER | Empty preds | Elapsed (s) | Energy (J) | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 quietfix | `vLLM` | `en_us` | 20 | 22.20% | 6.36% | 0 | 46.26 | 8112.90 | trusted internal reference |
| FP8 round 1 | `vLLM` | `en_us` | 20 | 21.97% | 6.36% | 0 | 35.21 | 4952.89 | best current compressed Voxtral path |
| Whisper large-v3 | `Transformers` | `en_us` | 20 | 20.59% | 4.32% | 0 | 34.77 | 3258.57 | strongest external same-slice baseline so far |

## Benchmark-Aligned External Read

| View | FP8 | Whisper | Interpretation |
| --- | ---: | ---: | --- |
| `open_asr_like` macro WER on `en_us` / `fr_fr` / `hi_in` | 9.74% | 8.22% | Whisper currently leads on benchmark-aligned quality |
| `open_asr_multilingual en_us limit20` | 7.01% | 4.21% | public wrapper did not close the English gap |

This matters for positioning:

- FP8 is a strong compressed Voxtral submission
- FP8 is not the strongest quality baseline overall on the current benchmark-aligned view
- the submission should therefore lead with efficiency, reproducibility, and deployment realism

## Multilingual FP8 Snapshot

| Config | Language | Samples | Primary read | Empty preds | Energy (J) | Interpretation |
| --- | --- | ---: | --- | ---: | ---: | --- |
| FP8 round 1 | `hi_in` | 5 | `WER = 26.83%` | 0 | 1620.28 | good spot check |
| FP8 round 1 | `fr_fr` | 5 | `WER = 23.18%` | 0 | 2121.87 | good spot check |
| FP8 round 1 | `ja_jp` | 5 | `CER = 10.42%`, `CER(no-space) = 10.00%` | 0 | 2692.89 | raw WER is misleading here |

## Main Message

The current evidence says:

- FP8 is already a better practical path than BF16 on this machine
- FP8 is materially better on efficiency
- FP8 preserves normalized English quality against our BF16 Voxtral reference
- FP8 is producing stable multilingual outputs across Hindi, French, and Japanese
- Whisper large-v3 currently beats our local Voxtral setup on the same normalized English slice
- the benchmark-aligned external multilingual picture currently favors Whisper overall
- Japanese needs CER-aware interpretation, not raw WER alone

## What We Should Say Honestly

- FP8 is the strongest current submission path.
- FP8 is the strongest current compressed Voxtral path, not the strongest model overall.
- GPTQ is still research, not a ready benchmark branch.
- Use normalized WER for public-facing English comparisons.
- Use the benchmark-aligned external view as context, not as something to hide from.
- The evaluation pipeline is much stronger now than it was earlier because it includes:
  - serialized transcription requests
  - quiet-audio preparation
  - explicit empty-prediction counting
  - CER-aware reporting support
  - normalized ASR metrics for fairer external comparison
  - an `open_asr_like` rescoring path that keeps our comparisons honest
- Do not claim prefix-cache uplift in the round-one table because the measured speech path still
  shows `0.0%` realized cache-hit rate

## Recommended Competition Framing

If we had to describe the current state in one sentence:

FP8 is the first compression path that is already giving us reproducible efficiency wins within the
Voxtral track, while stronger external baselines still give us a clear quality target to chase.
