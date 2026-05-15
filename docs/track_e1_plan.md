# Track E1 — Speculative Decoding Plan

Stack ngram speculative decoding on top of the locked D1-B W4A16 model.
Goal: 1.3-1.5× wall-clock speedup → proportional energy reduction.
Risk-free on quality (verifier is unchanged D1-B; rejection sampling
preserves the output distribution).

See `docs/track_e1_isolation_contract.md` for what this work is allowed
to touch.

## Hypothesis

vLLM 0.19.1 supports `speculative_config.method: ngram` natively. The
ngram drafter is an in-process prompt-lookup table — no separate model,
no training. For audio transcription:

- The token stream contains predictable bigrams (language-typical
  function words, repeated subword fragments) that ngram lookup gets
  right at >50% acceptance rate.
- Silence pad tokens emitted by the streaming head are highly
  predictable.
- High acceptance rate × cheap draft compute = fewer verifier forward
  passes per generated token = fewer joules.

Expected effect: 1.2-1.5× tokens/second improvement on FLEURS clips,
which translates almost 1:1 to energy reduction because the L4 is
compute-bound on the W4 GEMMs.

## Sub-phases

### Phase 0 — Sanity (dev machine, free, ~30 min)

- Confirm the existing `configs/vllm/track_e1_fp8_ngram_smoke.yaml`
  (FP8 + ngram) actually works on RTX 5080.
- Measure FP8 EN20 with and without ngram. Expect ≥1.2× faster on
  ngram. If not, the spec-decode path is broken before we layer it
  on D1-B.

### Phase 1 — D1-B + ngram, RTX 5080 EN20 (free, ~30 min)

- Build `configs/vllm/track_e1_d1b_ngram.yaml` — same as
  `track_d1b_w4a16_audio_gptq.yaml` but with `speculative_config:
  method: ngram, num_speculative_tokens: 4, prompt_lookup_max: 4,
  prompt_lookup_min: 2`.
- Serve, measure EN20 wall-clock vs the D1-B baseline (same model,
  no spec decode).
- Quality must match exactly (rejection sampling preserves
  distribution). Wall-clock must improve.
- If quality drifts (>0.1 pp norm WER), the spec-decode path has a
  bug — diagnose before going further.

**Gate**: ≥1.15× speedup on EN20. Below that, the L4 measurement isn't
worth the cost.

### Phase 2 — D1-B + ngram, RTX 5080 4-language sweep (free, ~30 min)

- EN500, HI100, FR100, JA100 with spec-decode enabled.
- Compare wall-clock per slice against the existing
  `reports/l4_binding/l4_d1b_*` numbers.
- Different languages have different ngram hit rates (English has
  the strongest prompt-lookup signal; CJK languages may not benefit
  as much).

**Gate**: aggregate wall-clock ≥1.15× faster across the 4 slices.

### Phase 3 — L4 binding measurement (~$1, ~30 min)

- Spin up an L4 (same recipe as the D1-B dry run).
- Install pinned stack via the same reproduce path.
- Serve D1-B + ngram on L4.
- 4-language binding sweep with CodeCarbon.
- Compare total kJ against D1-B's 199.3 kJ.

**Gate**: L4 total ≤ 175 kJ (i.e. at least −12% vs D1-B baseline →
−49% vs Round-1).

### Phase 4 — Decide

- **Pass** → package as a separate HF repo
  (`voxtral-mini-4b-asr-specdec` private), commit on the E1 branch,
  open a new PR. Final submission switches to D1-B + ngram.
- **Fail** → discard E1 work, ship D1-B. Annotate the compression
  story with "E1 was attempted; did not exceed the D1-B floor by
  enough to justify."

## Cost budget

| Phase | Compute | Cash |
|---|---|---|
| Phase 0 | RTX 5080 (free) | $0 |
| Phase 1 | RTX 5080 (free) | $0 |
| Phase 2 | RTX 5080 (free) | $0 |
| Phase 3 | L4 (RunPod secure cloud, ~$0.39/hr × ~1 hr) | $0.50 |
| **Total** | | **<$1** |

Walk-away cost if everything fails: $0.50. We can afford this.

## Stretch: EAGLE-3 draft (only if ngram disappoints)

- Train a small (~50M param) EAGLE-3 draft model on Voxtral decoder
  traces.
- 3-5 days of dev-machine GPU time.
- Higher draft accuracy than ngram → larger speedup.
- Same plumbing (`speculative_config.method: eagle` in vLLM 0.19.1).

Do this only if Phase 1/2 ngram speedup is below 1.2× and we still
think the underlying spec-decode path is worth pursuing.

## What does NOT change

- The D1-B model artifact (`consolidated.safetensors`, `config.json`,
  etc.) is read-only for this entire track. We do not re-quantize, do
  not re-calibrate, do not edit the safetensors.
- The Round-1 FP8 submission repo and its files are untouched.
- The audio preprocessing chain (`--target-lufs -23.0 --vad-trim
  --gate-silence --compress-internal-silence-to-ms 160
  --min-internal-silence-run-ms 320`) is locked. Spec decoding does
  not interact with the audio side.

## Success metric

A single number: `total_kJ_e1 / total_kJ_round1_floor`. We aim for
≤0.55 (i.e. >45% reduction). The D1-B-only candidate is at 0.576.
Anything under 0.50 is a "story upgrade" worth the submission swap.
