from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile acoustic silence on one or more FLEURS languages as a proxy for "
            "decoder-skipping opportunity."
        )
    )
    parser.add_argument(
        "--lang",
        action="append",
        required=True,
        help="Language code such as en_us, fr_fr, hi_in, or ja_jp.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Samples per language.")
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=80.0,
        help="Frame size used for activity profiling. The PDF idea assumes 80 ms decoder steps.",
    )
    parser.add_argument(
        "--active-peak-threshold",
        type=float,
        default=0.01,
        help="Minimum frame peak amplitude treated as active audio.",
    )
    parser.add_argument(
        "--active-rms-threshold",
        type=float,
        default=0.003,
        help="Minimum frame RMS treated as active audio.",
    )
    parser.add_argument(
        "--quiet-audio-peak-threshold",
        type=float,
        default=0.01,
        help="If the absolute peak is below this level, boost quiet samples for the prepared-audio view.",
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many most-silent samples to keep in each language summary.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def profile_language(
    *,
    lang_code: str,
    limit: int,
    frame_ms: float,
    active_peak_threshold: float,
    active_rms_threshold: float,
    quiet_audio_peak_threshold: float,
    quiet_audio_target_peak: float,
    max_audio_gain: float,
    top_k: int,
) -> dict:
    from voxtral_project.audio import (
        analyze_audio_activity,
        prepare_audio_array_for_transcription,
    )
    from voxtral_project.dataset_utils import load_fleurs_streaming

    fleurs = load_fleurs_streaming(lang_code=lang_code, split="test")

    samples: list[dict] = []

    for index, sample in enumerate(fleurs):
        if index >= limit:
            break

        raw_audio = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]
        prepared_audio, preparation = prepare_audio_array_for_transcription(
            raw_audio,
            sample_rate,
            quiet_peak_threshold=quiet_audio_peak_threshold,
            target_peak=quiet_audio_target_peak,
            max_gain=max_audio_gain,
        )
        raw_activity = analyze_audio_activity(
            raw_audio,
            sample_rate,
            frame_ms=frame_ms,
            active_peak_threshold=active_peak_threshold,
            active_rms_threshold=active_rms_threshold,
        )
        prepared_activity = analyze_audio_activity(
            prepared_audio,
            sample_rate,
            frame_ms=frame_ms,
            active_peak_threshold=active_peak_threshold,
            active_rms_threshold=active_rms_threshold,
        )

        samples.append(
            {
                "id": str(sample.get("id", index)),
                "reference": sample["transcription"],
                "audio_duration_seconds": round(float(preparation["duration_seconds"]), 6),
                "quiet_audio_boosted": bool(preparation["quiet_audio_boosted"]),
                "audio_gain_applied": round(float(preparation["gain_applied"]), 6),
                "raw_activity": {
                    key: round(value, 6) if isinstance(value, float) else value
                    for key, value in raw_activity.items()
                },
                "prepared_activity": {
                    key: round(value, 6) if isinstance(value, float) else value
                    for key, value in prepared_activity.items()
                },
            }
        )

    raw_silence = [float(sample["raw_activity"]["silent_frame_ratio"]) for sample in samples]
    prepared_silence = [float(sample["prepared_activity"]["silent_frame_ratio"]) for sample in samples]
    leading_silence = [float(sample["raw_activity"]["leading_silent_seconds"]) for sample in samples]
    trailing_silence = [float(sample["raw_activity"]["trailing_silent_seconds"]) for sample in samples]
    longest_silence = [float(sample["raw_activity"]["longest_silent_run_seconds"]) for sample in samples]
    boosted_count = sum(1 for sample in samples if sample["quiet_audio_boosted"])
    mostly_silent_count = sum(1 for ratio in raw_silence if ratio >= 0.5)

    most_silent_samples = sorted(
        samples,
        key=lambda sample: (
            float(sample["raw_activity"]["silent_frame_ratio"]),
            float(sample["raw_activity"]["longest_silent_run_seconds"]),
        ),
        reverse=True,
    )[: max(0, top_k)]

    return {
        "language": lang_code,
        "samples_profiled": len(samples),
        "summary": {
            "frame_ms": frame_ms,
            "active_peak_threshold": active_peak_threshold,
            "active_rms_threshold": active_rms_threshold,
            "average_raw_silent_frame_ratio": round(_mean(raw_silence), 6),
            "median_raw_silent_frame_ratio": round(_median(raw_silence), 6),
            "average_prepared_silent_frame_ratio": round(_mean(prepared_silence), 6),
            "median_prepared_silent_frame_ratio": round(_median(prepared_silence), 6),
            "average_leading_silence_seconds": round(_mean(leading_silence), 6),
            "average_trailing_silence_seconds": round(_mean(trailing_silence), 6),
            "average_longest_silent_run_seconds": round(_mean(longest_silence), 6),
            "clips_with_at_least_half_silent_frames": int(mostly_silent_count),
            "clips_with_at_least_half_silent_frames_ratio": round(
                mostly_silent_count / len(samples), 6
            )
            if samples
            else 0.0,
            "quiet_audio_boosted_count": int(boosted_count),
            "quiet_audio_boosted_ratio": round(boosted_count / len(samples), 6) if samples else 0.0,
        },
        "most_silent_samples": most_silent_samples,
        "samples": samples,
    }


def main() -> int:
    from voxtral_project.audio import write_json

    args = parse_args()
    results = [
        profile_language(
            lang_code=lang_code,
            limit=args.limit,
            frame_ms=args.frame_ms,
            active_peak_threshold=args.active_peak_threshold,
            active_rms_threshold=args.active_rms_threshold,
            quiet_audio_peak_threshold=args.quiet_audio_peak_threshold,
            quiet_audio_target_peak=args.quiet_audio_target_peak,
            max_audio_gain=args.max_audio_gain,
            top_k=args.top_k,
        )
        for lang_code in args.lang
    ]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Estimate the upper bound of decoder-skipping opportunity from acoustic inactivity. "
            "This is a silence proxy, not a direct measurement of Voxtral pad-token emission."
        ),
        "limit_per_language": args.limit,
        "results": results,
    }

    for result in results:
        summary = result["summary"]
        print(
            f"{result['language']}: raw silent frames avg={summary['average_raw_silent_frame_ratio']:.2%}, "
            f"median={summary['median_raw_silent_frame_ratio']:.2%}, "
            f"clips>=50% silent={summary['clips_with_at_least_half_silent_frames']}/"
            f"{result['samples_profiled']}, boosted={summary['quiet_audio_boosted_count']}"
        )

    if args.out:
        out_path = Path(args.out)
        write_json(out_path, payload)
        print(f"Saved report to: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
