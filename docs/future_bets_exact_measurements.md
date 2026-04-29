# Exact Wins And Measurement Status

## Conclusion

Prefix KV seeding is not currently working on the measured vLLM speech path. The AdaRMSNorm fold
is algebraically valid but failed the project exactness bar in the served checkpoint test because
WER moved. Pad-token emission is not truly measurable from the saved FLEURS reports because they
do not store raw generated token IDs.

## Status Block

Prefix KV seeding working:     [ ]  [method: vLLM prefix cache; manual KV memcpy not implemented]
AdaRMSNorm fold implemented:   [ ]  WER delta: +0.2330 pp normalized EN100, failed exactness
Pad-token rate measured:       [ ]  EN: 0.00%  FR: 0.00%  HI: 0.00%  JA: 0.00% visible lower bound only
Audio length distribution:     [x]  EN p50: 9.48s  p95: 16.77s

## Evidence

- Prefix cache:
  - Tool: `scripts/check_prefix_cache_logs.py`
  - Report: `reports/prefix_cache_all_logs_measurement_20260428.json`
  - Parsed logs: 137
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
  - Served checkpoint test: `reports/fleurs_fp8_step5_adarms_folded_en100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_en_us_limit100.json`
  - Comparable pre-fold report: `reports/fleurs_fp8_step3_vad_off_en100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_en_us_limit100.json`
  - Pre-fold normalized WER: 5.7316%; folded normalized WER: 5.9646%.
  - WER delta: +0.2330 percentage points, so the fold is not accepted as exact.

- Pad-token rate:
  - Tool: `scripts/analyze_fleurs_report_stats.py`
  - Report: `reports/future_bets_measurements_20260428.json`
  - EN source: `reports/fleurs_fp8_tracka_novad_hint_retry2_en500_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_en_us_limit500.json`
  - FR source: `reports/fleurs_fp8_tracka_novad_hint_retry2_fr100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_fr_fr_limit100.json`
  - HI source: `reports/fleurs_fp8_tracka_novad_hint_retry2_hi100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_hi_in_limit100.json`
  - JA source: `reports/fleurs_fp8_tracka_novad_hint_retry2_ja100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_ja_jp_limit100.json`
  - Result: 0 visible `<pad>`, `[STREAMING_PAD]`, `[P]`, `[STREAMING_WORD]`, or `[W]` markers in decoded predictions.
  - Caveat: saved reports do not contain raw generated token IDs, so this is a decoded-text lower
    bound, not a true decoder pad-token emission measurement.
  - Decision: do not invest in decoder-skipping from current artifacts; require token-ID capture
    before reopening that bet.

- Audio length:
  - Same measurement report: `reports/future_bets_measurements_20260428.json`
  - EN limit500: p50 9.48s, p95 16.77s, max 29.30s
  - FR limit100: p50 10.35s, p95 16.16s, max 27.78s
  - HI limit100: p50 11.25s, p95 20.91s, max 31.44s
  - JA limit100: p50 12.54s, p95 18.60s, max 24.48s
  - Estimated at 12.5 realtime audio tokens/s, the largest p95 is HI at about 262 audio tokens and
    the largest max is HI at about 393 audio tokens, far below `max_model_len=4096` after normal
    text-output margin.

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

The actual served folded-checkpoint run drifted WER, so this fold is not part of the defended
submission path.

## Sources

- Hugging Face Transformers Voxtral Realtime source shows `hidden_states = hidden_states * (1 + self.ada_rms_norm(t_cond))` in the decoder layer and constructs `t_cond` from `num_delay_tokens`: <https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/voxtral_realtime/modeling_voxtral_realtime.py>
- vLLM Mistral source shows the same AdaRMSNorm conditional path and disables it when `ada_rms_norm_t_cond` is false: <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/mistral.py>
- vLLM Voxtral Realtime source shows the fixed `n_delay_tokens` time embedding path used by the realtime model: <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/voxtral_realtime.py>
