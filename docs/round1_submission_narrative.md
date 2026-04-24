# Round 1 Submission Narrative

## Positioning Rule

Do not sell this as a benchmark-winning ASR system. Sell it as a disciplined, reproducible edge
compression result with measured efficiency gains and honest evaluation.

That is the strongest truthful frame, and it is the one most likely to survive round-one review.

## Short Version

We built a reproducible local serving pipeline for `mistralai/Voxtral-Mini-4B-Realtime-2602`,
hardened the evaluation stack to handle quiet audio and empty predictions correctly, and showed
that an FP8 serving path materially improves efficiency over the BF16 Voxtral reference on the
same trusted English slice. On `en_us limit20`, FP8 matches the BF16 normalized WER at `6.36%`
while running about `24%` faster and using about `39%` less measured energy. The system also
remains stable across Hindi, French, and Japanese spot checks, though stronger public ASR
baselines such as Whisper large-v3 still lead on the current benchmark-aligned quality view.

## Longer Version

Our round-one contribution is not a claim that we beat the strongest public ASR baselines. The
contribution is that we turned Voxtral into a cleaner, measurable compression candidate on local
hardware and validated a practical FP8 path that already outperforms the BF16 reference on
efficiency without degrading the trusted normalized English score on our main slice.

Just as important, we improved the credibility of the evaluation itself. The repo now serializes
audio requests to avoid engine-side overlap artifacts, boosts quiet clips before scoring, counts
empty predictions explicitly, and keeps both local normalized metrics and a benchmark-aligned
`open_asr_like` rescoring path. Those changes made the external comparison harsher, not easier,
and we kept them anyway. That makes the current result easier to trust and easier to defend.

## Claims We Can Support

- FP8 is the strongest compressed Voxtral path currently working in this repo.
- FP8 is materially more efficient than the BF16 Voxtral reference on the trusted `en_us limit20`
  comparison.
- The serving and evaluation pipeline is reproducible and substantially cleaner than the earlier
  repo state.
- The result is multilingual enough to show it is not an English-only accident.

## Claims To Avoid

- Do not claim we beat Whisper large-v3 or the published Voxtral benchmark numbers.
- Do not claim large-scale multilingual validation beyond the spot checks and current side-by-side
  reports.
- Do not claim a realized prefix-cache gain for `/v1/audio/transcriptions`.
- Do not imply GPTQ is submission-ready.
