# Track E1 — Isolation Contract

Track E1 (ngram speculative decoding stacked on D1-B) is an additive
swing bet. If it fails — for any reason — we fall back to the
D1-B-only Round-2 submission, which is already locked, measured,
quality-gate verified, and L4-reproducibility verified.

This file documents the isolation guarantees and the rollback path.

## The D1-B safe state

| Layer | What's frozen | Where |
|---|---|---|
| Git tag | `d1b-submission-ready` at commit `ded2e42` | `origin/refs/tags/d1b-submission-ready` |
| Git branch (PR'd) | `codex/track-c-final-validation` | `origin/codex/track-c-final-validation` |
| HF repo | `Shankara-A-S/voxtral-mini-4b-asr` (private, 69 files) | huggingface.co |
| Local artifact | `/home/npci/voxtral-w4a16-llmcompressor-audio-v1-consolidated/` (4.07 GB) | dev machine WSL ext4 |
| Local submission staging | `submission/hf_model_repo_w4a16/` | this repo |
| L4 binding reports | `reports/l4_binding/` (34 JSON files) | this repo |
| L4 dry-run reports | `reports/l4_dry_run/` (8 JSON files) | this repo |
| BF16 baselines | `reports/fleurs_bf16_baseline_*.json` (9 files) + `submission/hf_model_repo/reports/fleurs_bf16_canonical_*.json` (4 files) | this repo |
| Round-1 FP8 reports | `submission/hf_model_repo/reports/*` | this repo |
| Round-1 FP8 HF repo | `Shankara-A-S/voxtral-mini-realtime-fp8-runtime` (public, gated) | huggingface.co |
| Round-1 submission docs | `submission/hf_model_repo/{README.md,vllm_config.yaml,reproduce.sh,scripts/,src/,LICENSE,requirements.txt}` | this repo |
| Round-2 candidate doc | `docs/round2_candidate_snapshot.md` | this repo |
| Compression story | `compression_story.md` | this repo |

**None of these are allowed to change while Track E1 work is in progress.**

## Where Track E1 work is allowed to land

| Path | What it holds |
|---|---|
| Git branch | `track-e1-spec-decode` (this branch) |
| `configs/vllm/track_e1_*.yaml` | E1 vLLM configs (FP8 ngram already present; W4 + ngram is the new one) |
| `scripts/build_ngram_draft.py` (new) | ngram-table construction script (if needed) |
| `scripts/run_e1_*.sh` (new) | E1 launcher scripts |
| `submission/hf_model_repo_e1/` (new) | E1 submission staging (only created if E1 passes the gate) |
| `reports/e1_*/` (new) | E1 evaluation + energy reports |
| `docs/track_e1_*.md` | E1 design + result notes |
| New HF repo: `voxtral-mini-4b-asr-specdec` (created only if needed) | E1 submission HF mirror |

E1 work must **only** touch these paths. Anything outside is read-only.

## Failure modes and rollback

| If E1 fails because... | Rollback action |
|---|---|
| Quality regresses (any slice exceeds 1.25× BF16) | Abandon E1; submit from `d1b-submission-ready` tag |
| Energy is not lower than D1-B baseline (199.3 kJ) | Abandon E1; submit from `d1b-submission-ready` tag |
| vLLM `speculative_config` doesn't work with `compressed-tensors` quantization on Voxtral Realtime | Try EAGLE-3 draft instead; if that also fails, ship D1-B |
| L4 driver/torch incompatibility | Same vllm 0.19.1 + torch 2.10+cu128 stack as D1-B — this should not be a new failure mode |
| Test budget exhausted | Stop. Ship D1-B. Annotate the story doc with "E1 was attempted; did not complete in budget." |

## Rollback procedure (if needed)

```bash
# 1. Switch back to the safe branch
git checkout codex/track-c-final-validation

# 2. Verify HEAD matches the safe tag
git rev-parse HEAD
# expected: ded2e42... (the d1b-submission-ready tag)

# 3. Confirm HF repo voxtral-mini-4b-asr is unchanged (file count = 69)
huggingface-cli repo info Shankara-A-S/voxtral-mini-4b-asr

# 4. Submit using submission/hf_model_repo_w4a16/ contents and the
#    voxtral-mini-4b-asr HF URL as the model pointer.
```

## What E1 is allowed to publish

If E1 passes its gates (quality + lower energy than D1-B baseline) and
we choose to ship it:

- A new commit on `track-e1-spec-decode` with the verified E1 artifacts
  and reports.
- A separate HF repo (`voxtral-mini-4b-asr-specdec` or similar) containing
  the spec-decode-aware serving config and a `reproduce.sh` adapted for
  it. **Not pushed to `voxtral-mini-4b-asr`** — keep that repo as the
  D1-B-only fallback.
- A new PR for review, merged or held depending on the final submission
  choice.

D1-B remains shippable independently of any E1 outcome until the
official submission moment.
