# Organizer Email — Multiple Round-2 Submissions

**Subject**: Round-2 submission — quick question about submitting two variants

---

Hi,

Quick procedural question before the Round-2 window opens — apologies if
this is covered somewhere I missed.

For the audio-to-text track I've ended up with two versions of my
compressed model that I think are both worth showing. They're closely
related: the second is just the first with one extra serving-time
optimisation layered on. Both pass the BF16 quality ceiling across all
the FLEURS languages I tested, and both come in well under my Round-1
energy on an L4.

Would it be alright to submit both for evaluation, or should I pick
one? Happy to send just the better of the two if multiple submissions
aren't the norm. I just wanted to ask first rather than assume.

If it helps to know: the two variants share the same weights, so the
review effort is mostly the same. The split is really just to make the
comparison clean — one with the serving optimisation, one without.

I have both packaged as Hugging Face repos with a single-command
reproduction script, so however you'd like to handle it, I can share
the link(s) on request.

Thanks!

[Your name]

---

## Notes to self (don't include in the email)

- If they say "one submission, pick one" → send the spec-decode variant
  (`voxtral-mini-4b-asr-specdec`). It's strictly better on every slice.
- If they say "multiple OK, scored independently" → send both. The
  no-spec-decode variant is a safer fallback if their judging environment
  has any trouble with the spec-decode config.
- If they say "multiple OK but only the best counts" → still send both.
  No downside.
- If they ask for technical detail on the variants, then volunteer it.
  Otherwise keep it light in the first email.
