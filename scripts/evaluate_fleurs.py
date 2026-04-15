from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    from voxtral_project.api import DEFAULT_PROMPT

    parser = argparse.ArgumentParser(description="Evaluate WER on one or more FLEURS languages.")
    parser.add_argument(
        "--lang",
        action="append",
        required=True,
        help="Language code such as en_us, fr_fr, hi_in, ja_jp.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Samples per language.")
    parser.add_argument("--base-url", default="http://localhost:8080/v1", help="Server base URL.")
    parser.add_argument("--model", default="voxtral-realtime", help="Model name exposed by the server.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction prompt.")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max output tokens.")
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def evaluate_language(
    *,
    lang_code: str,
    limit: int,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    import jiwer
    from datasets import load_dataset

    from voxtral_project.api import transcribe_audio_bytes
    from voxtral_project.audio import audio_array_to_wav_bytes

    fleurs = load_dataset("google/fleurs", lang_code, split="test", streaming=True)

    predictions: list[str] = []
    references: list[str] = []
    samples: list[dict[str, str]] = []

    for index, sample in enumerate(fleurs):
        if index >= limit:
            break

        audio_bytes = audio_array_to_wav_bytes(
            audio_array=sample["audio"]["array"],
            sample_rate=sample["audio"]["sampling_rate"],
        )
        prediction = transcribe_audio_bytes(
            base_url=base_url,
            model=model,
            audio_bytes=audio_bytes,
            mime_type="audio/wav",
            prompt=prompt,
            max_tokens=max_tokens,
        )
        reference = sample["transcription"]

        predictions.append(prediction)
        references.append(reference)
        samples.append(
            {
                "id": str(sample.get("id", index)),
                "reference": reference,
                "prediction": prediction,
            }
        )

    wer_value = jiwer.wer(references, predictions)
    return {
        "language": lang_code,
        "samples_evaluated": len(samples),
        "wer": wer_value,
        "wer_percent": wer_value * 100.0,
        "samples": samples,
    }


def main() -> int:
    from voxtral_project.audio import write_json

    args = parse_args()

    results = [
        evaluate_language(
            lang_code=lang_code,
            limit=args.limit,
            base_url=args.base_url,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
        for lang_code in args.lang
    ]

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "limit_per_language": args.limit,
        "results": results,
    }

    for result in results:
        print(
            f"{result['language']}: WER={result['wer']:.4f} "
            f"({result['wer_percent']:.2f}%) over {result['samples_evaluated']} samples"
        )

    if args.out:
        write_json(Path(args.out), payload)
        print(f"Saved report to: {Path(args.out).resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
