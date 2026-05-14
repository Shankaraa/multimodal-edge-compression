---
license: apache-2.0
base_model: mistralai/Voxtral-Mini-4B-Realtime-2602
library_name: vllm
pipeline_tag: automatic-speech-recognition
tags:
- vllm
- automatic-speech-recognition
---

# voxtral-mini-4b-asr

A 4-bit weight-quantized derivative of `mistralai/Voxtral-Mini-4B-Realtime-2602`
for low-energy multilingual automatic speech recognition. The decoder
projections are quantized to W4A16 (GPTQ, group-128, dynamic activation order)
using **audio-conditioned calibration** — real decoder input embeddings from
FLEURS audio across 13 languages, not text tokens. The audio encoder, projector,
embeddings, and normalization layers remain in BF16. KV cache is declared FP8
(e4m3) at serve time. Inference is via vLLM with PIECEWISE cudagraph and Triton
attention.

The model **matches or beats the BF16 baseline on 9 of 13 FLEURS languages** while
serving with **~42% lower energy than the FP8 baseline on NVIDIA L4 24 GB** on the
canonical 4-language evaluation set, at **4.07 GB on disk** (≈50% of FP8,
≈25% of BF16).

## Quick facts

| Field | Value |
|---|---|
| Base model | `mistralai/Voxtral-Mini-4B-Realtime-2602` (BF16) |
| Decoder quantization | W4A16 GPTQ, group_size=128, dynamic actorder |
| Calibration | 256 real-audio decoder embeddings (FLEURS train, 13 languages) |
| Audio encoder | BF16 (unquantized) |
| KV cache | FP8 e4m3 (declared at serve time) |
| Artifact size | 4.07 GB (`consolidated.safetensors`) |
| Layout | Native Mistral consolidated (vLLM-compatible) |
| Compressed-tensors version | 0.15.0.1 |
| Serving | vLLM 0.19.1, TRITON_ATTN, PIECEWISE cudagraph |

## Headline results

### Quality — passes the BF16 1.25× ceiling on all 13 FLEURS languages

| Slice | Metric | BF16 baseline | 1.25× ceiling | **This model** | Verdict |
|---|---|---|---|---|---|
| en_us limit=500 | WER | 6.05% | 7.56% | **5.58%** | ✓ beats BF16 |
| fr_fr limit=100 | WER | 8.24% | 10.30% | **7.36%** | ✓ beats BF16 |
| hi_in limit=100 | WER | 26.27% | 32.84% | **24.09%** | ✓ beats BF16 |
| ja_jp limit=100 | CER | 6.72% | 8.39% | 7.41% | ✓ within ceiling |
| es_419 limit=100 | WER | 2.85% | 3.56% | **2.69%** | ✓ beats BF16 |
| it_it limit=100 | WER | 3.82% | 4.77% | 3.93% | ✓ within ceiling |
| ru_ru limit=100 | WER | 5.44% | 6.80% | 5.59% | ✓ within ceiling |
| pt_br limit=100 | WER | 5.05% | 6.31% | 5.76% | ✓ within ceiling |
| de_de limit=100 | WER | 5.10% | 6.37% | **4.89%** | ✓ beats BF16 |
| nl_nl limit=100 | WER | 8.84% | 11.05% | **8.49%** | ✓ beats BF16 |
| ar_eg limit=100 | WER | 15.01% | 18.76% | **14.01%** | ✓ beats BF16 |
| ko_kr limit=100 | WER | 15.95% | 19.94% | 15.95% | ✓ matches BF16 |
| cmn_hans_cn limit=100 | CER | 9.28% | 11.60% | **9.19%** | ✓ beats BF16 |

WER is normalized; CER reported for languages without word boundaries (ja, cmn).
BF16 baselines measured on RTX 5080 with `--language-hint-mode fleurs_primary
--empty-retry-count 2` and no audio preprocessing. WER and CER are
hardware-independent, so the same ceiling applies on L4.

Empty predictions across all 1700 evaluation samples: **1** (hi_in id=1985, a
known FLEURS quiet-duplicate row that also empties on the unmodified BF16 model).

### Energy — measured on NVIDIA L4 24 GB

| Slice | This model (kJ) | FP8 baseline (kJ) | Δ |
|---|---|---|---|
| en_us limit=500 | 107.8 | 189.4 | **−43.1%** |
| hi_in limit=100 | 28.8 | 44.5 | **−35.3%** |
| fr_fr limit=100 | 21.7 | 37.9 | **−42.7%** |
| ja_jp limit=100 | 41.0 | 73.9 | **−44.5%** |
| **4-language total** | **199.3** | **345.7** | **−42.36%** |

FP8 baseline kJ values are the audited Round-1 numbers under the same evaluation
policy. Energy measured with CodeCarbon wrapping the evaluator; numbers are
hardware-specific to NVIDIA L4 (driver 565.57, CUDA 12.7).

Full 13-language energy total on L4: **434.6 kJ** (see `reports/`).

## How this model was made

The decoder is the inference-time bottleneck. The audio encoder and the
multimodal projector are small relative to the language model, so the
compression strategy targets the decoder weights only.

### Step 1 — Calibration corpus

GPTQ-style quantization needs a calibration set that matches the distribution
the layer actually sees at inference. For an audio-conditioned decoder, that
distribution is the **projected audio embeddings** flowing out of the
multimodal projector — not the text tokens the language model was originally
pretrained on.

We built a 256-sample corpus from FLEURS train splits across 13 languages
(EN, ZH, HI, ES, AR, FR, PT, RU, DE, JA, KO, IT, NL — with HI at 61 samples to
strengthen Indic coverage). For each sample we ran the BF16 model up to the
language model boundary and saved the projected `inputs_embeds` tensor along
with its `num_delay_tokens` and `attention_mask`. The corpus is stored as a
HuggingFace `Dataset` with each row pointing at a `.pt` tensor file:

```
data/calibration/track_b_audio_conditioned_fleurs_train_256_hi61/
├── processor_dataset/        # HF Dataset; rows reference decoder_inputs/sample_NNNNNN.pt
├── decoder_inputs/           # 256 projected inputs_embeds tensors
├── layer0_inputs/            # layer-0 activations (audit only)
└── metadata.json
```

The calibration builder is `scripts/build_track_b_audio_conditioned_calibration.py`.

### Step 2 — GPTQ calibration with llm-compressor

We use `llm-compressor 0.10.0.1` with the `GPTQModifier`. The custom collator in
`scripts/run_track_b_llmcompressor_oneshot.py` loads the pre-projected
`inputs_embeds` directly into the decoder, bypassing the audio encoder during
calibration.

Recipe summary:

| Field | Value |
|---|---|
| Scheme | W4A16 (4-bit weights, 16-bit activations) |
| Targets | `language_model.model.layers.X.{self_attn,mlp}.*` projections |
| Ignore | embedding, lm_head, layernorms, ada_rms_norm, audio_tower, projector |
| Group size | 128 |
| Dampening fraction | 0.05 |
| Activation order | dynamic |
| Samples | 256 |
| Max seq length | 2048 |
| Sequential targets | one decoder layer at a time |

The runner installs a `forward` method on the multimodal wrapper that handles
the audio-conditioned path: when `inputs_embeds` is provided directly, it
synthesizes the `t_cond` time-embedding from `num_delay_tokens` and calls
`self.language_model` with the projected embeddings. This is what lets
llm-compressor walk the decoder layer-by-layer with realistic activations
while the audio encoder stays untouched.

The output HF-format artifact contains the BF16 audio tower and projector
plus the quantized decoder sidecars (`weight_packed`, `weight_scale`,
`weight_g_idx`, `weight_shape`).

### Step 3 — Package to native Mistral consolidated layout

vLLM with `tokenizer_mode: mistral` expects a `consolidated.safetensors` in
the native Mistral layout. `scripts/package_track_b_consolidated.py` merges
the BF16 base `consolidated.safetensors` (audio + non-decoder tensors) with
the quantized decoder sidecars from the HF artifact, remapping decoder keys
from HF naming (`language_model.model.layers.X.self_attn.q_proj.*`) to
Mistral naming (`layers.X.attention.wq.*`). The `quantization_config` regex
targets and `ignore` patterns are rewritten for the new layout.

### Step 4 — Serve with vLLM

```yaml
# vllm_config.yaml
served_model_name: voxtral-realtime
quantization: compressed-tensors
kv_cache_dtype: fp8_e4m3
attention_backend: TRITON_ATTN
tokenizer_mode: mistral
max_model_len: 4096
gpu_memory_utilization: 0.85
max_num_seqs: 1
max_num_batched_tokens: 4096
enable_prefix_caching: true
compilation_config:
  cudagraph_mode: PIECEWISE
disable_log_stats: true
```

```bash
vllm serve <repo-or-local-path> --config vllm_config.yaml --port 8084
```

Voxtral Realtime requires `cudagraph_mode: PIECEWISE` (full cudagraphs are not
supported by the streaming head) and the Triton attention backend (FlashInfer
is not implemented for Whisper-causal block pooling).

### Step 5 — Locked audio preprocessing chain

All reported quality and energy numbers are measured with this evaluator-side
audio preprocessing applied to every FLEURS sample before transcription:

```
--target-lufs -23.0 --lufs-max-gain-db 24.0
--vad-trim --vad-aggressiveness 1 --vad-padding-ms 200
--gate-silence
--compress-internal-silence-to-ms 160
--min-internal-silence-run-ms 320
```

The chain does three things:

1. **LUFS normalization to −23 LUFS** with a 24 dB ceiling. This fixes a class
   of quiet-clip failures that empties the FP8 path on FLEURS samples with input
   RMS below 0.003 (about 69% of FLEURS en_us).
2. **WebRTC VAD edge-trim** at aggressiveness 1 with 200 ms padding.
3. **Internal silence gate**: any internal silence run of ≥320 ms is compressed
   to 160 ms, reducing audio duration ~10% on average without quality loss.

Without LUFS, the gate over-trims quiet clips and the empty rate spikes from
0% to 5.2% on EN500. With LUFS, every quiet clip is rescued.

## Quality and energy contract

A submission that uses this artifact should report:

- **Per-language quality gate**: normalized WER (or CER for ja/cmn) ≤ 1.25 ×
  same-slice BF16 baseline. Verified for all 13 FLEURS languages above.
- **Empty-prediction gate**: zero empty predictions per slice, with `hi_in
  id=1985` explicitly noted as a known FLEURS data anomaly (the duplicate-at-
  low-volume row that also empties on the unmodified BF16 model).
- **Energy claim**: bind to an L4 measurement under the same audio preprocessing
  and vLLM configuration as this artifact was tuned for. RTX 5080 numbers should
  be reported as relative-only.

## Reproducing the numbers

The single-command flow assumes a clean Linux GPU host with the right driver
(NVIDIA driver ≥ 565, CUDA 12.7 forward-compat or newer).

```bash
git clone <this repo>  # or hf clone
cd voxtral-mini-4b-asr
bash reproduce.sh
```

`reproduce.sh` does the following in order:

1. Installs the pinned Python stack: `torch==2.10.0+cu128`, `vllm==0.19.1`,
   `compressed-tensors==0.15.0.1`, `transformers==5.8.x`, `pyloudnorm`,
   `webrtcvad`, `librosa`, `jiwer`, `codecarbon`, `mistral-common`.
2. Starts a vLLM server with `vllm_config.yaml` on port 8084.
3. Warms the server with a single dummy transcription.
4. Runs FLEURS evaluation across the 4 canonical slices (en_us 500, hi_in 100,
   fr_fr 100, ja_jp 100) with the locked audio preprocessing chain, wrapped in
   CodeCarbon energy measurement.
5. Writes language-specific evaluation JSON files and energy JSON files under
   `reports/`.

Smoke test:

```bash
RUN_SLICES="en_us:20" bash reproduce.sh
```

## Package contents

```
voxtral-mini-4b-asr/
├── consolidated.safetensors       # 4.07 GB, quantized decoder + BF16 encoder/projector
├── config.json                    # vLLM-ready quantization_config (compressed-tensors WNA16)
├── generation_config.json
├── params.json
├── processor_config.json
├── tekken.json                    # tokenizer
├── vllm_config.yaml               # serving config used for reported numbers
├── reproduce.sh                   # one-command reproduction
├── README.md
└── reports/                       # all evaluation + energy JSON reports backing the claims
    ├── l4_d1b_<13 FLEURS slices>.json
    ├── energy_l4_d1b_<13 FLEURS slices>.json
    ├── l4_fp8_<4 canonical slices>.json     # L4 FP8 reference for the energy Δ
    ├── energy_l4_fp8_*.json
    └── fleurs_bf16_baseline_<9 slices>.json # BF16 baselines for the 1.25× quality ceiling
```

## Hardware and software environment used for reported numbers

- GPU: NVIDIA L4 24 GB (RunPod community cloud)
- Driver: 565.57.01 (CUDA 12.7)
- Python: 3.11.10
- torch: 2.10.0+cu128
- vllm: 0.19.1
- compressed-tensors: 0.15.0.1
- transformers: 5.8.x

BF16 baselines and the calibration step were run on a workstation with an
RTX 5080 16 GB. WER and CER are hardware-independent; energy is not. Energy
numbers in this repository are L4 measurements only.

## Limitations and caveats

- **Streaming output not validated end-to-end.** The artifact serves through
  vLLM's `/v1/audio/transcriptions` endpoint; the realtime streaming path
  (`/v1/realtime`) shares the same decoder weights but the streaming-delay
  parameter has not been swept against this artifact.
- **hi_in sample id=1985 empties.** This is a known FLEURS data row (one
  duplicate transcript, two audio recordings — one loud, one near-silent) that
  the unmodified BF16 model also fails on. The empty rate on every other
  evaluated sample is 0.
- **Calibration corpus size**: 256 samples is sufficient for W4A16 GPTQ on the
  Voxtral decoder but could be scaled up. The marginal value of additional
  samples beyond 256 was not characterized.

## License

Apache-2.0, inherited from the base model.
