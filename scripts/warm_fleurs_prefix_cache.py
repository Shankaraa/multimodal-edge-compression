from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    from voxtral_project.api import DEFAULT_PROMPT

    parser = argparse.ArgumentParser(
        description=(
            "Prime the vLLM speech-to-text prefix cache with one FLEURS sample "
            "before a measured evaluation run."
        )
    )
    parser.add_argument("--lang", required=True, help="FLEURS language code such as en_us.")
    parser.add_argument("--sample-index", type=int, default=0, help="Zero-based FLEURS test sample index.")
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


def get_fleurs_sample(*, lang_code: str, sample_index: int) -> dict:
    from voxtral_project.dataset_utils import load_fleurs_streaming

    fleurs = load_fleurs_streaming(lang_code=lang_code, split="test")

    for index, sample in enumerate(fleurs):
        if index == sample_index:
            return sample

    raise IndexError(f"FLEURS sample index {sample_index} is out of range for {lang_code}.")


def main() -> int:
    from voxtral_project.asr import build_transcriber
    from voxtral_project.audio import prepare_audio_array_for_transcription, write_json

    args = parse_args()
    sample = get_fleurs_sample(lang_code=args.lang, sample_index=args.sample_index)
    prepared_audio_array, audio_diagnostics = prepare_audio_array_for_transcription(
        sample["audio"]["array"],
        sample["audio"]["sampling_rate"],
        quiet_peak_threshold=args.quiet_audio_peak_threshold,
        target_peak=args.quiet_audio_target_peak,
        max_gain=args.max_audio_gain,
    )

    transcriber = build_transcriber(
        backend="vllm_api",
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        language_hint_mode=args.language_hint_mode,
        temperature=args.temperature,
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
        "sample_index": args.sample_index,
        "sample_id": str(sample.get("id", args.sample_index)),
        "reference": sample["transcription"],
        "prediction": prediction,
        "audio_duration_seconds": round(float(audio_diagnostics["duration_seconds"]), 6),
        "audio_peak_abs_before": round(float(audio_diagnostics["peak_abs_before"]), 6),
        "audio_peak_abs_after": round(float(audio_diagnostics["peak_abs_after"]), 6),
        "audio_rms_before": round(float(audio_diagnostics["rms_before"]), 6),
        "audio_rms_after": round(float(audio_diagnostics["rms_after"]), 6),
        "audio_gain_applied": round(float(audio_diagnostics["gain_applied"]), 6),
        "quiet_audio_boosted": bool(audio_diagnostics["quiet_audio_boosted"]),
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
