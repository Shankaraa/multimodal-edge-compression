from __future__ import annotations

import argparse
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

    parser = argparse.ArgumentParser(
        description="Probe Voxtral Realtime target-delay tau on one EN sample and the HI 1985 canary."
    )
    parser.add_argument("--base-url", default="http://localhost:8080/v1", help="Server base URL.")
    parser.add_argument("--model", default="voxtral-realtime", help="Model name exposed by the server.")
    parser.add_argument(
        "--dataset-source",
        choices=FLEURS_DATASET_SOURCES,
        default="google_fleurs",
        help="Dataset wrapper used for probe samples.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction prompt.")
    parser.add_argument(
        "--language-hint-mode",
        choices=("none", "fleurs_primary"),
        default="fleurs_primary",
        help="Optionally send the FLEURS primary language code.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max output tokens.")
    parser.add_argument(
        "--tau-ms",
        type=int,
        nargs="+",
        default=[240, 480, 2400],
        help="Target delay tau values in milliseconds.",
    )
    parser.add_argument("--en-lang", default="en_us", help="Normal English probe language.")
    parser.add_argument("--en-sample-index", type=int, default=0, help="Normal English sample index.")
    parser.add_argument("--hi-lang", default="hi_in", help="Hindi canary language.")
    parser.add_argument("--hi-sample-id", default="1985", help="Hindi canary sample id.")
    parser.add_argument(
        "--hi-sample-occurrence",
        type=int,
        default=2,
        help="1-based occurrence for duplicate HI sample ids; 2 targets the known low-volume canary.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def get_sample_by_index(*, lang_code: str, sample_index: int, dataset_source: str) -> dict[str, Any]:
    from voxtral_project.dataset_utils import load_transcription_dataset_streaming

    dataset = load_transcription_dataset_streaming(
        lang_code=lang_code,
        split="test",
        dataset_source=dataset_source,
    )
    for index, sample in enumerate(dataset):
        if index == sample_index:
            return dict(sample, _probe_index=index)
    raise IndexError(f"Sample index {sample_index} not found for {lang_code}.")


def get_sample_by_id(
    *,
    lang_code: str,
    sample_id: str,
    occurrence: int,
    dataset_source: str,
) -> dict[str, Any]:
    from voxtral_project.dataset_utils import load_transcription_dataset_streaming

    if occurrence < 1:
        raise ValueError("--hi-sample-occurrence must be 1 or greater.")

    seen = 0
    fallback: dict[str, Any] | None = None
    dataset = load_transcription_dataset_streaming(
        lang_code=lang_code,
        split="test",
        dataset_source=dataset_source,
    )
    for index, sample in enumerate(dataset):
        if str(sample.get("id", "")) != sample_id:
            continue
        seen += 1
        sample_with_index = dict(sample, _probe_index=index)
        if fallback is None:
            fallback = sample_with_index
        if seen == occurrence:
            return sample_with_index

    if fallback is not None:
        return fallback
    raise LookupError(f"Sample id {sample_id!r} not found for {lang_code}.")


def transcribe_probe_sample(
    *,
    sample: dict[str, Any],
    lang_code: str,
    tau_ms: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from voxtral_project.asr import build_transcriber
    from voxtral_project.audio import prepare_audio_array_for_transcription
    from voxtral_project.dataset_utils import get_sample_text

    prepared_audio, audio_diagnostics = prepare_audio_array_for_transcription(
        sample["audio"]["array"],
        sample["audio"]["sampling_rate"],
    )
    transcriber = build_transcriber(
        backend="vllm_api",
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        language_hint_mode=args.language_hint_mode,
        temperature=args.temperature,
        target_streaming_delay_ms=tau_ms,
        max_tokens=args.max_tokens,
        hf_model_id="openai/whisper-large-v3",
        hf_device="auto",
        hf_torch_dtype="auto",
        hf_attn_implementation=None,
        hf_language_hint_mode="known_if_supported",
    )
    started = time.perf_counter()
    prediction = transcriber.transcribe(
        audio_array=prepared_audio,
        sample_rate=sample["audio"]["sampling_rate"],
        lang_code=lang_code,
    )
    elapsed = time.perf_counter() - started
    return {
        "language": lang_code,
        "sample_index": int(sample["_probe_index"]),
        "sample_id": str(sample.get("id", sample["_probe_index"])),
        "reference": get_sample_text(sample),
        "tau_ms": tau_ms,
        "prediction": prediction,
        "empty_prediction": not prediction.strip(),
        "latency_seconds": round(elapsed, 6),
        "audio_duration_seconds": round(float(audio_diagnostics["duration_seconds"]), 6),
        "audio_peak_abs_before": round(float(audio_diagnostics["peak_abs_before"]), 6),
        "audio_rms_before": round(float(audio_diagnostics["rms_before"]), 6),
        "quiet_audio_boosted": bool(audio_diagnostics["quiet_audio_boosted"]),
        "backend_details": transcriber.describe(),
    }


def main() -> int:
    from voxtral_project.audio import write_json

    args = parse_args()
    en_sample = get_sample_by_index(
        lang_code=args.en_lang,
        sample_index=args.en_sample_index,
        dataset_source=args.dataset_source,
    )
    hi_sample = get_sample_by_id(
        lang_code=args.hi_lang,
        sample_id=args.hi_sample_id,
        occurrence=args.hi_sample_occurrence,
        dataset_source=args.dataset_source,
    )

    results: list[dict[str, Any]] = []
    for tau_ms in args.tau_ms:
        for lang_code, sample_name, sample in (
            (args.en_lang, "normal_en", en_sample),
            (args.hi_lang, "hi_canary", hi_sample),
        ):
            result = transcribe_probe_sample(
                sample=sample,
                lang_code=lang_code,
                tau_ms=tau_ms,
                args=args,
            )
            result["probe_sample"] = sample_name
            results.append(result)
            status = "empty" if result["empty_prediction"] else "non-empty"
            print(
                f"{sample_name} {lang_code} id={result['sample_id']} "
                f"tau={tau_ms}ms latency={result['latency_seconds']:.3f}s {status}"
            )
            print(result["prediction"])

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "track_c_tau_sweep_probe",
        "base_url": args.base_url,
        "model": args.model,
        "dataset_source": args.dataset_source,
        "tau_ms": args.tau_ms,
        "results": results,
    }
    if args.out:
        out_path = Path(args.out)
        write_json(out_path, payload)
        print(f"Saved tau probe report to: {out_path.resolve()}")

    return 1 if any(result["empty_prediction"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
