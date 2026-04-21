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
        "--backend",
        choices=("vllm_api", "whisper_transformers"),
        default="vllm_api",
        help="Transcription backend. Use vLLM API for Voxtral or local Transformers for Whisper.",
    )
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
    parser.add_argument(
        "--hf-model-id",
        default="openai/whisper-large-v3",
        help="Transformers model id used when --backend whisper_transformers.",
    )
    parser.add_argument(
        "--hf-device",
        default="auto",
        help="Torch device for the Transformers backend, such as auto, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--hf-torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Torch dtype used when loading the Transformers backend.",
    )
    parser.add_argument(
        "--hf-attn-implementation",
        default=None,
        help="Optional Transformers attention implementation, such as sdpa or flash_attention_2.",
    )
    parser.add_argument(
        "--hf-language-hint-mode",
        choices=("known_if_supported", "auto"),
        default="known_if_supported",
        help="Pass the known FLEURS language to Whisper when supported, or let the model auto-detect.",
    )
    return parser.parse_args()


def evaluate_language(
    *,
    lang_code: str,
    limit: int,
    quiet_audio_peak_threshold: float,
    quiet_audio_target_peak: float,
    max_audio_gain: float,
    transcriber: object,
) -> dict:
    from datasets import load_dataset

    from voxtral_project.audio import prepare_audio_array_for_transcription
    from voxtral_project.text import summarize_transcript_metrics

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
        prediction = transcriber.transcribe(
            audio_array=prepared_audio_array,
            sample_rate=sample["audio"]["sampling_rate"],
            lang_code=lang_code,
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

    metrics = summarize_transcript_metrics(
        references=references,
        predictions=predictions,
    )
    return {
        "language": lang_code,
        "samples_evaluated": len(samples),
        "empty_prediction_count": empty_prediction_count,
        **metrics,
        "samples": samples,
    }


def main() -> int:
    from voxtral_project.asr import build_transcriber
    from voxtral_project.audio import write_json

    args = parse_args()
    transcriber = build_transcriber(
        backend=args.backend,
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        hf_model_id=args.hf_model_id,
        hf_device=args.hf_device,
        hf_torch_dtype=args.hf_torch_dtype,
        hf_attn_implementation=args.hf_attn_implementation,
        hf_language_hint_mode=args.hf_language_hint_mode,
    )

    results = [
        evaluate_language(
            lang_code=lang_code,
            limit=args.limit,
            quiet_audio_peak_threshold=args.quiet_audio_peak_threshold,
            quiet_audio_target_peak=args.quiet_audio_target_peak,
            max_audio_gain=args.max_audio_gain,
            transcriber=transcriber,
        )
        for lang_code in args.lang
    ]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "backend_details": transcriber.describe(),
        "limit_per_language": args.limit,
        "results": results,
    }

    for result in results:
        print(
            f"{result['language']}: WER={result['wer']:.4f} "
            f"({result['wer_percent']:.2f}%), CER={result['cer_percent']:.2f}%, "
            f"CER(no-space)={result['cer_no_whitespace_percent']:.2f}%, "
            f"norm WER={result['wer_normalized_percent']:.2f}%, "
            f"norm CER={result['cer_normalized_percent']:.2f}% "
            f"over {result['samples_evaluated']} samples with "
            f"{result['empty_prediction_count']} empty predictions"
        )

    if args.out:
        write_json(Path(args.out), payload)
        print(f"Saved report to: {Path(args.out).resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
