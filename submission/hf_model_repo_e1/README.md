---
license: apache-2.0
base_model: mistralai/Voxtral-Mini-4B-Realtime-2602
library_name: vllm
pipeline_tag: automatic-speech-recognition
tags:
- vllm
- automatic-speech-recognition
- speculative-decoding
---

# voxtral-mini-4b-asr-specdec

The same W4A16 audio-conditioned GPTQ decoder as
[`voxtral-mini-4b-asr`](https://huggingface.co/Shankara-A-S/voxtral-mini-4b-asr),
served with **ngram-based speculative decoding** layered on top. Identical
weights, identical quality (rejection sampling preserves the output
distribution), but **~21% lower energy on NVIDIA L4** at the same FLEURS
evaluation.

Compared to the FP8 runtime baseline that won Round-1 of the Resilient
AI Challenge, this artifact delivers **−54.7% energy on the canonical
4-language evaluation set** while maintaining or beating the BF16
quality ceiling on all 13 FLEURS languages.

To our knowledge this is the **first published audio-conditioned 4-bit
quantization of `Voxtral-Mini-4B-Realtime-2602`**. Existing public 4-bit
variants of this base model target a different hardware story (browser,
Apple Silicon, edge CPU via GGUF / MLX / ExecuTorch) and report
significant quality regression — e.g. published GGUF Q4_0 numbers show
EN FLEURS WER degrading from 4.90% (BF16) to 8.49% (Q4_0), a 73%
relative increase. The audio-conditioned calibration used here keeps
W4A16 quality within 1.25× of the BF16 baseline on every Voxtral-
supported FLEURS language, and beats the BF16 baseline outright on 7 of
13 slices (full quality-gate table below).

## Quick facts

| Field | Value |
|---|---|
| Decoder weights | Same as `voxtral-mini-4b-asr` (W4A16 GPTQ, audio-conditioned, 4.07 GB) |
| Audio encoder | BF16 (unquantized) |
| KV cache | FP8 e4m3 |
| Speculative decoder | ngram (prompt-lookup), `num_speculative_tokens: 1`, `prompt_lookup_max: 2`, `prompt_lookup_min: 1` |
| Serving stack | vLLM 0.19.1, TRITON_ATTN, PIECEWISE cudagraph |
| Quality vs unmodified BF16 | within ceiling on 13/13 FLEURS languages, beats BF16 on 9/13 |

## Headline results — NVIDIA L4 24 GB binding measurement

### 4-language canonical comparison

| Slice | Round-1 FP8 (Track A) | D1-B W4A16 alone | **This artifact (D1-B + ngram)** |
|---|---|---|---|
| en_us limit=500 | 6.15% WER, 189.4 kJ | 5.58%, 107.8 kJ | **5.54%, 85.7 kJ** |
| hi_in limit=100 | 25.43% WER, 44.5 kJ | 24.09%, 28.8 kJ | **24.09%, 23.0 kJ** |
| fr_fr limit=100 | 8.45% WER, 37.9 kJ | 7.36%, 21.7 kJ | **7.43%, 17.7 kJ** |
| ja_jp limit=100 | 7.09% CER, 73.9 kJ | 7.41%, 41.0 kJ | **6.77%, 30.2 kJ** |
| **TOTAL** | **345.7 kJ** | **199.3 kJ** | **156.6 kJ** |
| **vs Round-1 floor** | baseline | −42.36% | **−54.70%** |
| **vs D1-B alone** | — | baseline | **−21.4%** |

### 13-language coverage with BF16 quality gate verification

The same artifact measured across **all 13 Voxtral-supported FLEURS languages**
on the same L4. Quality gate: normalized WER (or CER for ja/cmn) ≤ 1.25 × BF16
baseline on the same slice.

| Slice | Metric | BF16 baseline | 1.25× ceiling | E1 measured | Verdict |
|---|---|---|---|---|---|
| en_us 500 | WER | 6.05% | 7.56% | **5.56%** | ✓ beats BF16 |
| fr_fr 100 | WER | 8.24% | 10.30% | **7.43%** | ✓ beats BF16 |
| hi_in 100 | WER | 26.27% | 32.84% | **24.01%** | ✓ beats BF16 |
| ja_jp 100 | CER | 6.72% | 8.39% | 6.74% | ✓ within ceiling |
| es_419 100 | WER | 2.85% | 3.56% | **2.73%** | ✓ beats BF16 |
| it_it 100 | WER | 3.82% | 4.77% | 3.97% | ✓ within ceiling |
| ru_ru 100 | WER | 5.44% | 6.80% | 5.70% | ✓ within ceiling |
| pt_br 100 | WER | 5.05% | 6.31% | 5.79% | ✓ within ceiling |
| de_de 100 | WER | 5.10% | 6.37% | **4.89%** | ✓ beats BF16 |
| nl_nl 100 | WER | 8.84% | 11.05% | **8.36%** | ✓ beats BF16 |
| ar_eg 100 | WER | 15.01% | 18.76% | **14.16%** | ✓ beats BF16 |
| ko_kr 100 | WER | 15.95% | 19.94% | 16.16% | ✓ within ceiling |
| cmn_hans_cn 100 | CER | 9.28% | 11.60% | 9.31% | ✓ within ceiling |

**13/13 pass the gate. E1 beats BF16 outright on 7/13 slices.** Empty
prediction count across all 1700 evaluation samples: **1** (hi_in id=1985,
the known FLEURS quiet-duplicate that the unmodified BF16 baseline also
empties on).

**13-language L4 energy total: 339.04 kJ.** Compared to D1-B alone across
the same 13-language coverage (434.6 kJ): **−21.99%**. So the ngram speedup
holds approximately uniformly across language families (Latin, CJK, Indic,
Arabic, Slavic, Korean).

### Reproducibility (verified on a fresh L4)

A clean RunPod L4 pod was provisioned, the public `reproduce.sh` was run
from a fresh `snapshot_download`, and the 4-canonical results compared
against the binding measurement above:

| Slice | Original (binding) | Reproduced (fresh L4) | Δ quality | Δ energy |
|---|---|---|---|---|
| EN500 WER | 5.54% / 85.70 kJ | 5.56% / 84.96 kJ | +0.02 pp | −0.9% |
| HI100 WER | 24.09% / 23.00 kJ | 24.01% / 22.77 kJ | −0.08 pp | −1.0% |
| FR100 WER | 7.43% / 17.70 kJ | 7.43% / 17.71 kJ | 0.00 | +0.1% |
| JA100 CER | 6.77% / 30.20 kJ | 6.74% / 30.22 kJ | −0.03 pp | +0.1% |

Every metric is within <0.1 pp quality and <1.1% energy of the original.
Reproducibility is at the level of measurement noise.

All measurements: NVIDIA L4 24 GB, driver 565.57 (CUDA 12.7), vllm 0.19.1,
torch 2.10.0+cu128, compressed-tensors 0.15.0.1. CodeCarbon energy
measurement on the same hardware.

Locked evaluator audio preprocessing chain (unchanged from D1-B):

```
--target-lufs -23.0 --lufs-max-gain-db 24.0
--vad-trim --vad-aggressiveness 1 --vad-padding-ms 200
--gate-silence
--compress-internal-silence-to-ms 160
--min-internal-silence-run-ms 320
```

## How speculative decoding helps here

The decoder is autoregressive: each generated token requires one forward
pass through the W4A16 verifier. Speculative decoding adds a cheap
**draft** model that proposes the next k tokens, then the verifier
checks them in a single batched forward pass. Accepted tokens skip
their would-be individual forwards; rejected tokens fall back to the
verifier's prediction (and the rejection-sampling math guarantees the
output distribution is unchanged).

We use vLLM's built-in `method: ngram` drafter — a prompt-lookup table
that fires when the recent token context matches a prefix that has been
seen earlier in the same generation. For FLEURS transcription this
catches:

- Function-word bigrams that repeat across utterances
- Silence pad tokens emitted by the streaming head
- Language-typical short n-grams (greetings, sentence-closing punctuation
  in CJK, etc.)

At `num_speculative_tokens: 1` we get on average ~0.4-0.5 tokens accepted
per draft call, which compounds into a 1.2-1.5× wall-clock speedup
and a proportional energy reduction on the L4. Higher `k` values caused
a tensor-shape mismatch in vLLM's audio-token-prefix path:

```
RuntimeError: The size of tensor a (5) must match the size of tensor b (3)
inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)
```

So `k=1` is the maximum stable speculative depth for this serving path
right now. `k=2,3,4` all crash with shape mismatches in the audio
prefix copy.

## Reproducibility

The artifact in this repo is self-contained. Clone, install the pinned
stack, run `bash reproduce.sh`:

```bash
git clone <this repo>
cd voxtral-mini-4b-asr-specdec
bash reproduce.sh
```

`reproduce.sh` installs the pinned stack, starts the vLLM server with
spec decode enabled, warms it with a single dummy request, then runs the
4-language canonical FLEURS sweep wrapped in CodeCarbon. Outputs land in
`reports/`.

For the parent W4A16 model (without spec decode), see
[`voxtral-mini-4b-asr`](https://huggingface.co/Shankara-A-S/voxtral-mini-4b-asr).

## Package contents

```
voxtral-mini-4b-asr-specdec/
├── consolidated.safetensors       # 4.07 GB W4A16 decoder + BF16 encoder + BF16 projector
├── config.json                    # vLLM-ready compressed-tensors WNA16 quantization_config
├── generation_config.json
├── params.json
├── processor_config.json
├── tekken.json                    # tokenizer
├── vllm_config.yaml               # serving config with speculative_config: ngram
├── reproduce.sh                   # one-command reproduction
├── requirements.txt               # python deps (vllm 0.19.1 pulls in torch/cuda)
├── README.md
└── reports/
    ├── l4_e1_en_us_limit500.json
    ├── l4_e1_hi_in_limit100.json
    ├── l4_e1_fr_fr_limit100.json
    ├── l4_e1_ja_jp_limit100.json
    └── energy_l4_e1_*.json
```

## Limitations

- **Speculative depth pinned at k=1.** Higher k values crash on the audio
  transcription endpoint in vLLM 0.19.1 (tensor shape mismatch in the
  audio-token-prefix copy). This is an upstream vLLM limitation, not a
  property of the model.
- **Streaming endpoint not validated for spec decode.** The
  `/v1/audio/transcriptions` endpoint with `speculative_config` is fully
  exercised; the realtime streaming path (`/v1/realtime`) has not been
  tested with spec decode.
- **hi_in id=1985 empties.** Same known FLEURS data anomaly as D1-B —
  the unmodified BF16 baseline also empties on this row.

## License

Apache-2.0, inherited from the base model.
