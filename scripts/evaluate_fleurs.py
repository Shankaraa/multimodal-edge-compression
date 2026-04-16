from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
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
    parser.add_argument(
        "--quiet-audio-peak-threshold",
        type=float,
        default=0.01,
        help="If the absolute peak is below this level, boost quiet samples before transcription.",
    )
    parser.add_argument(
        "--quiet-audio-target-peak",
        type=float,
        default=0.02,
        help="Target absolute peak after boosting quiet samples.",
    )
    parser.add_argument(
        "--max-audio-gain",
        type=float,
        default=8.0,
        help="Maximum gain multiplier used for quiet-sample boosting.",
    )
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
    quiet_audio_peak_threshold: float,
    quiet_audio_target_peak: float,
    max_audio_gain: float,
) -> dict:
    import jiwer
    from datasets import load_dataset

    from voxtral_project.api import transcribe_audio_bytes
    from voxtral_project.audio import audio_array_to_wav_bytes, prepare_audio_array_for_transcription

    fleurs = load_dataset(
        "google/fleurs",
        lang_code,
        split="test",
        streaming=True,
        trust_remote_code=True,
    )

    predictions: list[str] = []
    references: list[str] = []
    samples: list[dict[str, str]] = []
    empty_prediction_count = 0

    for index, sample in enumerate(fleurs):
        if index >= limit:
            break

        prepared_audio_array, audio_diagnostics = prepare_audio_array_for_transcription(
            sample["audio"]["array"],
            sample["audio"]["sampling_rate"],
            quiet_peak_threshold=quiet_audio_peak_threshold,
            target_peak=quiet_audio_target_peak,
            max_gain=max_audio_gain,
        )
        audio_bytes = audio_array_to_wav_bytes(
            audio_array=prepared_audio_array,
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
        is_empty_prediction = not prediction.strip()
        if is_empty_prediction:
            empty_prediction_count += 1

        predictions.append(prediction)
        references.append(reference)
        samples.append(
            {
                "id": str(sample.get("id", index)),
                "reference": reference,
                "prediction": prediction,
                "audio_duration_seconds": round(float(audio_diagnostics["duration_seconds"]), 6),
                "audio_peak_abs_before": round(float(audio_diagnostics["peak_abs_before"]), 6),
                "audio_peak_abs_after": round(float(audio_diagnostics["peak_abs_after"]), 6),
                "audio_rms_before": round(float(audio_diagnostics["rms_before"]), 6),
                "audio_rms_after": round(float(audio_diagnostics["rms_after"]), 6),
                "audio_gain_applied": round(float(audio_diagnostics["gain_applied"]), 6),
                "quiet_audio_boosted": bool(audio_diagnostics["quiet_audio_boosted"]),
                "empty_prediction": is_empty_prediction,
            }
        )

    wer_value = jiwer.wer(references, predictions)
    return {
        "language": lang_code,
        "samples_evaluated": len(samples),
        "wer": wer_value,
        "wer_percent": wer_value * 100.0,
        "empty_prediction_count": empty_prediction_count,
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
            quiet_audio_peak_threshold=args.quiet_audio_peak_threshold,
            quiet_audio_target_peak=args.quiet_audio_target_peak,
            max_audio_gain=args.max_audio_gain,
        )
        for lang_code in args.lang
    ]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "limit_per_language": args.limit,
        "results": results,
    }

    for result in results:
        print(
            f"{result['language']}: WER={result['wer']:.4f} "
            f"({result['wer_percent']:.2f}%) over {result['samples_evaluated']} samples "
            f"with {result['empty_prediction_count']} empty predictions"
        )

    if args.out:
        write_json(Path(args.out), payload)
        print(f"Saved report to: {Path(args.out).resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
