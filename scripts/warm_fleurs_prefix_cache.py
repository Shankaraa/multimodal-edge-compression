from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    from voxtral_project.api import DEFAULT_PROMPT
    from voxtral_project.dataset_utils import FLEURS_DATASET_SOURCES

    parser = argparse.ArgumentParser(
        description=(
            "Prime the vLLM speech-to-text prefix cache with one FLEURS sample "
            "before a measured evaluation run."
        )
    )
    parser.add_argument("--lang", required=True, help="FLEURS language code such as en_us.")
    parser.add_argument("--sample-index", type=int, default=0, help="Zero-based FLEURS test sample index.")
    parser.add_argument(
        "--dataset-source",
        choices=FLEURS_DATASET_SOURCES,
        default="google_fleurs",
        help="Dataset wrapper used for the warmup sample.",
    )
    parser.add_argument("--base-url", default="http://localhost:8080/v1", help="Server base URL.")
    parser.add_argument("--model", default="voxtral-realtime", help="Model name exposed by the server.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction prompt.")
    parser.add_argument(
        "--language-hint-mode",
        choices=("none", "fleurs_primary"),
        default="none",
        help="Optionally send the FLEURS primary language code to the transcription endpoint.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature for the vLLM API backend.",
    )
    parser.add_argument(
        "--target-streaming-delay-ms",
        type=int,
        default=None,
        help=(
            "Optional Voxtral Realtime target delay tau in milliseconds. "
            "Defaults to the served model's configured delay."
        ),
    )
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
    parser.add_argument(
        "--gate-silence",
        action="store_true",
        help="Apply speech-aware silence gating before warmup transcription.",
    )
    parser.add_argument(
        "--vad-trim",
        action="store_true",
        help="Strip leading/trailing silence with conservative WebRTC VAD before warmup.",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        choices=(0, 1, 2, 3),
        default=1,
        help="WebRTC VAD aggressiveness for --vad-trim.",
    )
    parser.add_argument(
        "--vad-padding-ms",
        type=float,
        default=200.0,
        help="Silence preserved before first voiced frame and after last voiced frame.",
    )
    parser.add_argument(
        "--gate-frame-ms",
        type=float,
        default=80.0,
        help="Frame size used for speech-aware silence gating.",
    )
    parser.add_argument(
        "--gate-peak-threshold",
        type=float,
        default=0.01,
        help="Peak threshold used when classifying active audio for gating.",
    )
    parser.add_argument(
        "--gate-rms-threshold",
        type=float,
        default=0.003,
        help="RMS threshold used when classifying active audio for gating.",
    )
    parser.add_argument(
        "--preserve-leading-silence-ms",
        type=float,
        default=160.0,
        help="Silence preserved immediately before speech onset when gating is enabled.",
    )
    parser.add_argument(
        "--preserve-trailing-silence-ms",
        type=float,
        default=160.0,
        help="Silence preserved immediately after speech offset when gating is enabled.",
    )
    parser.add_argument(
        "--compress-internal-silence-to-ms",
        type=float,
        default=None,
        help="If set, long internal silent spans are compressed to this duration.",
    )
    parser.add_argument(
        "--min-internal-silence-run-ms",
        type=float,
        default=640.0,
        help="Only compress internal silent spans at least this long.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def get_fleurs_sample(*, lang_code: str, sample_index: int, dataset_source: str) -> dict:
    from voxtral_project.dataset_utils import load_transcription_dataset_streaming

    fleurs = load_transcription_dataset_streaming(
        lang_code=lang_code,
        split="test",
        dataset_source=dataset_source,
    )

    for index, sample in enumerate(fleurs):
        if index == sample_index:
            return sample

    raise IndexError(f"FLEURS sample index {sample_index} is out of range for {lang_code}.")


def main() -> int:
    from voxtral_project.asr import build_transcriber
    from voxtral_project.audio import prepare_audio_array_for_transcription, write_json
    from voxtral_project.dataset_utils import get_sample_text

    args = parse_args()
    sample = get_fleurs_sample(
        lang_code=args.lang,
        sample_index=args.sample_index,
        dataset_source=args.dataset_source,
    )
    prepared_audio_array, audio_diagnostics = prepare_audio_array_for_transcription(
        sample["audio"]["array"],
        sample["audio"]["sampling_rate"],
        quiet_peak_threshold=args.quiet_audio_peak_threshold,
        target_peak=args.quiet_audio_target_peak,
        max_gain=args.max_audio_gain,
        gate_silence=args.gate_silence,
        vad_trim=args.vad_trim,
        vad_aggressiveness=args.vad_aggressiveness,
        vad_padding_ms=args.vad_padding_ms,
        gate_frame_ms=args.gate_frame_ms,
        gate_peak_threshold=args.gate_peak_threshold,
        gate_rms_threshold=args.gate_rms_threshold,
        preserve_leading_silence_ms=args.preserve_leading_silence_ms,
        preserve_trailing_silence_ms=args.preserve_trailing_silence_ms,
        compress_internal_silence_to_ms=args.compress_internal_silence_to_ms,
        min_internal_silence_run_ms=args.min_internal_silence_run_ms,
    )

    transcriber = build_transcriber(
        backend="vllm_api",
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        language_hint_mode=args.language_hint_mode,
        temperature=args.temperature,
        target_streaming_delay_ms=args.target_streaming_delay_ms,
        max_tokens=args.max_tokens,
        hf_model_id="openai/whisper-large-v3",
        hf_device="auto",
        hf_torch_dtype="auto",
        hf_attn_implementation=None,
        hf_language_hint_mode="known_if_supported",
    )
    prediction = transcriber.transcribe(
        audio_array=prepared_audio_array,
        sample_rate=sample["audio"]["sampling_rate"],
        lang_code=args.lang,
    )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "prefix_cache_warmup",
        "backend_details": transcriber.describe(),
        "language": args.lang,
        "dataset_source": args.dataset_source,
        "sample_index": args.sample_index,
        "sample_id": str(sample.get("id", args.sample_index)),
        "reference": get_sample_text(sample),
        "prediction": prediction,
        "audio_duration_seconds": round(float(audio_diagnostics["duration_seconds"]), 6),
        "audio_peak_abs_before": round(float(audio_diagnostics["peak_abs_before"]), 6),
        "audio_peak_abs_after": round(float(audio_diagnostics["peak_abs_after"]), 6),
        "audio_rms_before": round(float(audio_diagnostics["rms_before"]), 6),
        "audio_rms_after": round(float(audio_diagnostics["rms_after"]), 6),
        "audio_gain_applied": round(float(audio_diagnostics["gain_applied"]), 6),
        "quiet_audio_boosted": bool(audio_diagnostics["quiet_audio_boosted"]),
        "vad_trim_applied": bool(audio_diagnostics["vad_trim_applied"]),
        "vad_trim_changed_audio": bool(audio_diagnostics["vad_trim_changed_audio"]),
        "vad_trim_seconds_removed": round(float(audio_diagnostics["vad_trim_seconds_removed"]), 6),
        "vad_trim_fraction_removed": round(float(audio_diagnostics["vad_trim_fraction_removed"]), 6),
        "speech_gating_applied": bool(audio_diagnostics["speech_gating_applied"]),
        "speech_gating_changed_audio": bool(audio_diagnostics["speech_gating_changed_audio"]),
        "speech_gating_seconds_removed": round(
            float(audio_diagnostics["speech_gating_seconds_removed"]), 6
        ),
    }

    print(
        f"Warmed prefix cache with {args.lang} sample {args.sample_index} "
        f"(id={payload['sample_id']}, quiet_boosted={payload['quiet_audio_boosted']})."
    )
    print("Transcript:")
    print(prediction)

    if args.out:
        out_path = Path(args.out)
        write_json(out_path, payload)
        print(f"Saved warmup report to: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
