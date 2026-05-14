from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    from voxtral_project.api import DEFAULT_PROMPT
    from voxtral_project.dataset_utils import FLEURS_DATASET_SOURCES

    parser = argparse.ArgumentParser(description="Fail-fast transcription gate for a specific FLEURS sample id.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="voxtral-realtime")
    parser.add_argument("--sample-id", required=True, help="FLEURS sample id to locate.")
    parser.add_argument("--lang", default="hi_in", help="FLEURS language code.")
    parser.add_argument(
        "--occurrence",
        type=int,
        default=2,
        help="1-based occurrence of the sample id in the streamed split. HI id 1985 canary is occurrence 2.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-source", choices=FLEURS_DATASET_SOURCES, default="google_fleurs")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--language-hint-mode", choices=("none", "fleurs_primary"), default="fleurs_primary")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--target-streaming-delay-ms", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--empty-retry-count", type=int, default=2)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def find_sample(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    from voxtral_project.dataset_utils import load_transcription_dataset_streaming

    seen = 0
    dataset = load_transcription_dataset_streaming(
        lang_code=args.lang,
        split=args.split,
        dataset_source=args.dataset_source,
    )
    for index, sample in enumerate(dataset):
        if str(sample.get("id", "")) != str(args.sample_id):
            continue
        seen += 1
        if seen == args.occurrence:
            return index, sample
    raise RuntimeError(
        f"Could not find occurrence {args.occurrence} of sample id {args.sample_id} "
        f"in {args.dataset_source}/{args.lang}/{args.split}; saw {seen} occurrence(s)."
    )


def main() -> int:
    from voxtral_project.asr import build_transcriber
    from voxtral_project.audio import prepare_audio_array_for_transcription, write_json
    from voxtral_project.dataset_utils import get_sample_text
    from voxtral_project.text import normalize_asr_text, word_error_rate

    args = parse_args()
    index, sample = find_sample(args)
    audio = sample["audio"]
    prepared_audio, audio_diagnostics = prepare_audio_array_for_transcription(
        audio["array"],
        audio["sampling_rate"],
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
        hf_model_id="",
        hf_device="auto",
        hf_torch_dtype="auto",
        hf_attn_implementation=None,
        hf_language_hint_mode="known_if_supported",
    )

    attempts: list[dict[str, Any]] = []
    prediction = ""
    for attempt_index in range(args.empty_retry_count + 1):
        started = time.perf_counter()
        prediction = transcriber.transcribe(
            audio_array=prepared_audio,
            sample_rate=audio["sampling_rate"],
            lang_code=args.lang,
        )
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "latency_seconds": round(time.perf_counter() - started, 6),
                "empty": prediction.strip() == "",
                "prediction": prediction,
            }
        )
        if prediction.strip():
            break

    reference = get_sample_text(sample)
    wer = word_error_rate(
        normalize_asr_text(reference),
        normalize_asr_text(prediction),
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "dataset_source": args.dataset_source,
        "lang": args.lang,
        "split": args.split,
        "sample_id": str(args.sample_id),
        "occurrence": args.occurrence,
        "stream_index": index,
        "target_streaming_delay_ms": args.target_streaming_delay_ms,
        "empty_retry_count": args.empty_retry_count,
        "reference": reference,
        "prediction": prediction,
        "empty_prediction": prediction.strip() == "",
        "normalized_wer": wer,
        "audio_duration_seconds": round(float(audio_diagnostics["duration_seconds"]), 6),
        "audio_peak_abs_before": round(float(audio_diagnostics["peak_abs_before"]), 6),
        "audio_rms_before": round(float(audio_diagnostics["rms_before"]), 6),
        "attempts": attempts,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.out:
        write_json(Path(args.out), payload)

    if payload["empty_prediction"]:
        raise SystemExit(f"Sample gate failed: sample {args.sample_id} returned an empty prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
