# Decoder-Skipping Feasibility Track

## Bottom Line

This track exists to test the PDF's central claim without destabilizing the working submission
path.

The claim is not "quantize harder."

The claim is:

- the decoder is memory-bound,
- silence-heavy streaming ASR wastes many forward passes on low-information steps, and
- skipping those steps may save more energy than further bit-width reduction.

That might be true.
It is not yet proven in this repo.

So this track starts with measurement, not decoder surgery.

## Why This Is A Separate Track

We already have two established paths:

- `fp8_round1` as the practical submission path
- GPTQ and `llmcompressor` work as higher-risk quantization research

The decoder-skipping idea should not be mixed into either one yet.

Reason:

- FP8 is already working and should remain stable.
- GPTQ is already risky enough and is solving a different problem.
- Decoder skipping likely needs custom inference logic, which is a separate engineering surface.

So the correct first move is to estimate whether the opportunity is even large enough to matter.

## What We Can Measure Right Now

The PDF talks about Voxtral emitting many pad tokens during delayed streaming.

We are not measuring pad-token rate yet.

What we can measure immediately is the closest cheap proxy:

- how much of FLEURS audio is acoustically inactive at an `80 ms` frame scale
- how much leading and trailing silence the clips contain
- how often quiet-audio boosting changes the activity picture

This does not prove decoder skip viability.

It does answer the gating question:

- if the silence proxy is weak, a custom decoder path is probably not worth our time
- if the silence proxy is strong, deeper instrumentation becomes rational

## Current Tooling

New script:

- `scripts/profile_fleurs_silence.py`

What it does:

- streams FLEURS test samples
- profiles raw audio frame activity
- profiles prepared audio after the repo's quiet-audio boost path
- summarizes silent-frame ratio, leading silence, trailing silence, and longest silent run
- writes a JSON report for later comparison

Important interpretation rule:

- this script estimates an upper bound on possible skip opportunity from audio inactivity
- it is not a direct measurement of decoder pad-token emission

## Recommended Experiment Order

1. Run the profiler on `en_us limit20` so it lines up with the current FP8 benchmark slice.
2. Check whether the average raw silent-frame ratio is actually large.
3. Compare raw vs prepared activity to see how much the quiet-audio fix changes the proxy.
4. If the signal is strong, instrument the serving path next.
5. Only after that consider custom gating or speculative decoder changes.

## Decision Rule

This track becomes strategically important only if at least one of these turns out to be true:

- a large share of clips are silence-heavy at the `80 ms` frame scale
- leading and trailing silence are consistently large enough to skip safely
- prepared-audio profiling suggests the evaluator is currently paying decoder cost on many low-information regions

If those do not show up, the PDF's idea may still be interesting in theory, but it is probably not
the best use of project time here.

## Next Step After A Positive Signal

Do not jump straight into a `vLLM` fork.

The next technical step should be one of these:

- instrument Voxtral outputs to estimate actual pad-token frequency on our evaluation slice
- test a cheap application-layer boundary skip for obvious leading and trailing silence
- verify whether warmed prefix reuse already captures part of the "constant prefix" opportunity

That keeps the track empirical instead of speculative.
