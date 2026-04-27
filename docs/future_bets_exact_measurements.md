# Exact Wins And Measurement Status

## Conclusion

Prefix KV seeding is not currently working on the measured vLLM speech path. The exact AdaRMSNorm
fold is implemented as a checkpoint transformation script and dry-run validated against the real
Voxtral tensors, but WER parity is not claimed yet because the folded checkpoint has not been
served and evaluated.

## Status Block

Prefix KV seeding working:     [ ]  [method: vLLM prefix cache]
AdaRMSNorm fold implemented:   [x]  WER delta: not measured
Pad-token rate measured:       [ ]  EN: 0.00%  FR: 0.00%  HI: 0.00%  JA: 0.00% visible lower bound
Audio length distribution:     [x]  EN p50: 9.48s  p95: 16.77s

## Evidence

- Prefix cache:
  - Tool: `scripts/check_prefix_cache_logs.py`
  - Report: `reports/prefix_cache_all_logs_measurement.json`
  - Parsed logs: 34
  - Prefix caching enabled seen: true
  - Max prefix-cache hit rate: 0.0%
  - Positive hit-rate samples: 0
  - Decision: do not claim prefix KV seeding as a working optimization.

- AdaRMSNorm fold:
  - Tool: `scripts/fold_ada_rmsnorm.py`
  - Dry-run report: `reports/ada_rmsnorm_fold_dry_run.json`
  - Proof report: `reports/ada_rmsnorm_fold_proof.json`
  - Real tensors found and fold factors computed for all 26 decoder layers.
  - Algebraic proof: grouped float64 max absolute difference is 0.0.
  - WER parity is still required before this becomes a defended runtime claim.

- Pad-token rate:
  - Tool: `scripts/analyze_fleurs_report_stats.py`
  - Report: `reports/future_bets_measurements.json`
  - EN source: `reports/fleurs_fp8_gap_limit500_en_us_limit500.json`
  - FR source: `reports/fleurs_fp8_multilingual_fr_fr_limit20.json`
  - HI source: `reports/fleurs_fp8_multilingual_hi_in_limit20.json`
  - JA source: `reports/fleurs_fp8_ja_jp_limit5_quietfix_v2.json`
  - Result: 0 visible `<pad>` or `[STREAMING_PAD]` markers in decoded predictions.
  - Caveat: saved reports do not contain raw generated token IDs, so this is a decoded-text lower
    bound, not a true decoder pad-token emission measurement.

- Audio length:
  - Same measurement report: `reports/future_bets_measurements.json`
  - EN limit500: p50 9.48s, p95 16.77s, max 29.30s
  - Estimated at 12.5 realtime audio tokens/s, EN p95 is about 210 audio tokens and max is about
    367 audio tokens, far below `max_model_len=4096`.

## Implementation Notes

The Ada fold is valid because the current implementations multiply post-attention RMSNorm output
by a delay-conditioned vector that is constant for the request:

```text
RMSNorm(x, w) * g(tau) == RMSNorm(x, w * g(tau))
```

The script targets the vLLM Mistral-format checkpoint:

- folds `layers.N.ada_rms_norm_t_cond` into `layers.N.ffn_norm.weight`
- sets `ada_rms_norm_t_cond` to false in copied JSON config files
- omits the folded-away Ada tensors from the output safetensors stream

The actual folded checkpoint was not written in this pass to avoid adding a multi-GB artifact
without immediate WER validation.

## Sources

- Hugging Face Transformers Voxtral Realtime source shows `hidden_states = hidden_states * (1 + self.ada_rms_norm(t_cond))` in the decoder layer and constructs `t_cond` from `num_delay_tokens`: <https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/voxtral_realtime/modeling_voxtral_realtime.py>
- vLLM Mistral source shows the same AdaRMSNorm conditional path and disables it when `ada_rms_norm_t_cond` is false: <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/mistral.py>
- vLLM Voxtral Realtime source shows the fixed `n_delay_tokens` time embedding path used by the realtime model: <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/voxtral_realtime.py>
