from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HARNESS_VERSION = "2026-04-24.bootstrap-ci-v1"


def parse_args() -> argparse.Namespace:
    from voxtral_project.api import DEFAULT_PROMPT
    from voxtral_project.dataset_utils import FLEURS_DATASET_SOURCES

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
    parser.add_argument(
        "--dataset-source",
        choices=FLEURS_DATASET_SOURCES,
        default="google_fleurs",
        help="Transcription dataset wrapper to evaluate against.",
    )
    parser.add_argument("--base-url", default="http://localhost:8080/v1", help="Server base URL.")
    parser.add_argument("--model", default="voxtral-realtime", help="Model name exposed by the server.")
    parser.add_argument(
        "--model-label",
        default=None,
        help="Measurement label such as bf16_baseline or fp8_round1.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional serving config path; used to hash and stamp the report.",
    )
    parser.add_argument("--config-hash", default=None, help="Optional precomputed config SHA-256.")
    parser.add_argument("--harness-git-sha", default=None, help="Optional git commit SHA.")
    parser.add_argument(
        "--normalization-version",
        default=None,
        help="Optional SHA-256 of src/voxtral_project/text.py.",
    )
    parser.add_argument("--server-log-path", default=None, help="Server log path for this run.")
    parser.add_argument("--elapsed-seconds", type=float, default=None, help="Measured eval wall time.")
    parser.add_argument("--energy-joules", type=float, default=None, help="Measured eval energy.")
    parser.add_argument("--emissions-kg", type=float, default=None, help="Measured eval emissions.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction prompt.")
    parser.add_argument(
        "--language-hint-mode",
        choices=("none", "fleurs_primary"),
        default="none",
        help="When using the vLLM API backend, optionally send the FLEURS primary language code.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature for the vLLM API backend. The model card recommends 0.0.",
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
        "--empty-retry-count",
        type=int,
        default=0,
        help="Retry a sample this many times when the transcription endpoint returns empty text.",
    )
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
        help="Apply speech-aware silence gating before transcription.",
    )
    parser.add_argument(
        "--vad-trim",
        action="store_true",
        help="Strip leading/trailing silence with conservative WebRTC VAD before transcription.",
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
        help=(
            "If set, long internal silent spans are compressed to this duration instead of being "
            "kept in full."
        ),
    )
    parser.add_argument(
        "--min-internal-silence-run-ms",
        type=float,
        default=640.0,
        help="Only compress internal silent spans at least this long.",
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


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def evaluate_language(
    *,
    lang_code: str,
    limit: int,
    quiet_audio_peak_threshold: float,
    quiet_audio_target_peak: float,
    max_audio_gain: float,
    gate_silence: bool,
    vad_trim: bool,
    vad_aggressiveness: int,
    vad_padding_ms: float,
    gate_frame_ms: float,
    gate_peak_threshold: float,
    gate_rms_threshold: float,
    preserve_leading_silence_ms: float,
    preserve_trailing_silence_ms: float,
    compress_internal_silence_to_ms: float | None,
    min_internal_silence_run_ms: float,
    dataset_source: str,
    transcriber: object,
    empty_retry_count: int,
) -> dict:
    from voxtral_project.audio import prepare_audio_array_for_transcription
    from voxtral_project.dataset_utils import (
        get_sample_text,
        load_transcription_dataset_streaming,
    )
    from voxtral_project.text import summarize_transcript_metrics
    from voxtral_project.text import (
        character_error_rate_no_whitespace,
        normalize_asr_text,
        word_error_rate,
    )

    fleurs = load_transcription_dataset_streaming(
        lang_code=lang_code,
        split="test",
        dataset_source=dataset_source,
    )

    predictions: list[str] = []
    references: list[str] = []
    samples: list[dict[str, object]] = []
    empty_prediction_count = 0
    empty_retry_sample_count = 0
    empty_retry_request_count = 0
    total_hyp_chars = 0
    total_ref_chars = 0
    latency_total_seconds: list[float] = []

    for index, sample in enumerate(fleurs):
        if index >= limit:
            break

        prepared_audio_array, audio_diagnostics = prepare_audio_array_for_transcription(
            sample["audio"]["array"],
            sample["audio"]["sampling_rate"],
            quiet_peak_threshold=quiet_audio_peak_threshold,
            target_peak=quiet_audio_target_peak,
            max_gain=max_audio_gain,
            gate_silence=gate_silence,
            vad_trim=vad_trim,
            vad_aggressiveness=vad_aggressiveness,
            vad_padding_ms=vad_padding_ms,
            gate_frame_ms=gate_frame_ms,
            gate_peak_threshold=gate_peak_threshold,
            gate_rms_threshold=gate_rms_threshold,
            preserve_leading_silence_ms=preserve_leading_silence_ms,
            preserve_trailing_silence_ms=preserve_trailing_silence_ms,
            compress_internal_silence_to_ms=compress_internal_silence_to_ms,
            min_internal_silence_run_ms=min_internal_silence_run_ms,
        )
        sample_started = time.perf_counter()
        attempt_latencies: list[float] = []
        attempt_started = time.perf_counter()
        prediction = transcriber.transcribe(
            audio_array=prepared_audio_array,
            sample_rate=sample["audio"]["sampling_rate"],
            lang_code=lang_code,
        )
        attempt_latencies.append(time.perf_counter() - attempt_started)
        retry_attempts = 0
        while not prediction.strip() and retry_attempts < empty_retry_count:
            retry_attempts += 1
            empty_retry_request_count += 1
            attempt_started = time.perf_counter()
            prediction = transcriber.transcribe(
                audio_array=prepared_audio_array,
                sample_rate=sample["audio"]["sampling_rate"],
                lang_code=lang_code,
            )
            attempt_latencies.append(time.perf_counter() - attempt_started)
        sample_latency_total = time.perf_counter() - sample_started
        latency_total_seconds.append(sample_latency_total)
        if retry_attempts:
            empty_retry_sample_count += 1
        reference = get_sample_text(sample)
        normalized_reference = normalize_asr_text(reference)
        normalized_prediction = normalize_asr_text(prediction)
        hyp_chars = len(prediction)
        ref_chars = len(reference)
        total_hyp_chars += hyp_chars
        total_ref_chars += ref_chars
        is_empty_prediction = not prediction.strip()
        if is_empty_prediction:
            empty_prediction_count += 1

        sample_id = str(sample.get("id", index))
        predictions.append(prediction)
        references.append(reference)
        samples.append(
            {
                "id": sample_id,
                "reference": reference,
                "prediction": prediction,
                "hyp_chars": hyp_chars,
                "ref_chars": ref_chars,
                "wer_raw": word_error_rate(reference, prediction),
                "wer_normalized": word_error_rate(normalized_reference, normalized_prediction),
                "cer_normalized_no_whitespace": character_error_rate_no_whitespace(
                    normalized_reference,
                    normalized_prediction,
                ),
                "latency_total_seconds": round(sample_latency_total, 6),
                "latency_attempt_seconds": [
                    round(attempt_latency, 6) for attempt_latency in attempt_latencies
                ],
                "ttft_seconds": None,
                "streaming_tokens_per_second": None,
                "audio_duration_seconds": round(float(audio_diagnostics["duration_seconds"]), 6),
                "audio_peak_abs_before": round(float(audio_diagnostics["peak_abs_before"]), 6),
                "audio_peak_abs_after": round(float(audio_diagnostics["peak_abs_after"]), 6),
                "audio_rms_before": round(float(audio_diagnostics["rms_before"]), 6),
                "audio_rms_after": round(float(audio_diagnostics["rms_after"]), 6),
                "audio_gain_applied": round(float(audio_diagnostics["gain_applied"]), 6),
                "quiet_audio_boosted": bool(audio_diagnostics["quiet_audio_boosted"]),
                "vad_trim_applied": bool(audio_diagnostics["vad_trim_applied"]),
                "vad_trim_changed_audio": bool(audio_diagnostics["vad_trim_changed_audio"]),
                "vad_trim_duration_before_seconds": round(
                    float(audio_diagnostics["vad_trim_duration_before_seconds"]), 6
                ),
                "vad_trim_duration_after_seconds": round(
                    float(audio_diagnostics["vad_trim_duration_after_seconds"]), 6
                ),
                "vad_trim_seconds_removed": round(
                    float(audio_diagnostics["vad_trim_seconds_removed"]), 6
                ),
                "vad_trim_fraction_removed": round(
                    float(audio_diagnostics["vad_trim_fraction_removed"]), 6
                ),
                "vad_trim_leading_trimmed_seconds": round(
                    float(audio_diagnostics["vad_trim_leading_trimmed_seconds"]), 6
                ),
                "vad_trim_trailing_trimmed_seconds": round(
                    float(audio_diagnostics["vad_trim_trailing_trimmed_seconds"]), 6
                ),
                "vad_trim_frame_count": int(audio_diagnostics["vad_trim_frame_count"]),
                "vad_trim_voiced_frame_count": int(
                    audio_diagnostics["vad_trim_voiced_frame_count"]
                ),
                "speech_gating_applied": bool(audio_diagnostics["speech_gating_applied"]),
                "speech_gating_changed_audio": bool(audio_diagnostics["speech_gating_changed_audio"]),
                "speech_gating_duration_before_seconds": round(
                    float(audio_diagnostics["speech_gating_duration_before_seconds"]), 6
                ),
                "speech_gating_duration_after_seconds": round(
                    float(audio_diagnostics["speech_gating_duration_after_seconds"]), 6
                ),
                "speech_gating_seconds_removed": round(
                    float(audio_diagnostics["speech_gating_seconds_removed"]), 6
                ),
                "speech_gating_fraction_removed": round(
                    float(audio_diagnostics["speech_gating_fraction_removed"]), 6
                ),
                "speech_gating_leading_trimmed_seconds": round(
                    float(audio_diagnostics["speech_gating_leading_trimmed_seconds"]), 6
                ),
                "speech_gating_trailing_trimmed_seconds": round(
                    float(audio_diagnostics["speech_gating_trailing_trimmed_seconds"]), 6
                ),
                "speech_gating_internal_trimmed_seconds": round(
                    float(audio_diagnostics["speech_gating_internal_trimmed_seconds"]), 6
                ),
                "speech_gating_internal_spans_compressed": int(
                    audio_diagnostics["speech_gating_internal_spans_compressed"]
                ),
                "empty_prediction": is_empty_prediction,
                "empty_retry_attempts": retry_attempts,
            }
        )

    metrics = summarize_transcript_metrics(
        references=references,
        predictions=predictions,
        lang_code=lang_code,
    )
    verbosity_ratio = total_hyp_chars / total_ref_chars if total_ref_chars else None
    verbosity_drift_warning = (
        verbosity_ratio is not None and not (0.95 <= verbosity_ratio <= 1.05)
    )
    return {
        "language": lang_code,
        "dataset_source": dataset_source,
        "samples_evaluated": len(samples),
        "hyp_chars_total": total_hyp_chars,
        "ref_chars_total": total_ref_chars,
        "verbosity_ratio": verbosity_ratio,
        "verbosity_drift_warning": verbosity_drift_warning,
        "verbosity_drift_warning_range": [0.95, 1.05],
        "empty_prediction_count": empty_prediction_count,
        "empty_retry_sample_count": empty_retry_sample_count,
        "empty_retry_request_count": empty_retry_request_count,
        "latency_total_seconds_p50": percentile(latency_total_seconds, 0.50),
        "latency_total_seconds_p95": percentile(latency_total_seconds, 0.95),
        "ttft_seconds_p50": None,
        "ttft_seconds_p95": None,
        "streaming_tokens_per_second_p50": None,
        "streaming_tokens_per_second_p95": None,
        "realtime_failure_threshold_note": (
            "This non-streaming /v1/audio/transcriptions harness would fail any "
            f"total-utterance p95 constraint below {percentile(latency_total_seconds, 0.95):.6f}s. "
            "TTFT and streaming tokens/sec are unavailable unless a streaming endpoint is used."
            if latency_total_seconds
            else "No latency threshold can be derived because no samples were evaluated."
        ),
        **metrics,
        "samples": samples,
    }


def main() -> int:
    from voxtral_project.asr import build_transcriber
    from voxtral_project.audio import write_json
    from voxtral_project.reporting import (
        attach_measurement_contract,
        ensure_config_hash_in_filename,
        file_sha256,
        get_git_head_sha,
        normalization_version,
    )

    args = parse_args()
    config_hash = args.config_hash
    if config_hash is None and args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_hash = file_sha256(config_path)
    harness_git_sha = args.harness_git_sha or get_git_head_sha(PROJECT_ROOT)
    normalization_version_hash = args.normalization_version or normalization_version(PROJECT_ROOT)

    transcriber = build_transcriber(
        backend=args.backend,
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        language_hint_mode=args.language_hint_mode,
        temperature=args.temperature,
        target_streaming_delay_ms=args.target_streaming_delay_ms,
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
            dataset_source=args.dataset_source,
            transcriber=transcriber,
            empty_retry_count=args.empty_retry_count,
        )
        for lang_code in args.lang
    ]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "harness_version": HARNESS_VERSION,
        "backend": args.backend,
        "backend_details": transcriber.describe(),
        "target_streaming_delay_ms": args.target_streaming_delay_ms,
        "limit_per_language": args.limit,
        "speech_gating": {
            "enabled": args.gate_silence,
            "frame_ms": args.gate_frame_ms,
            "peak_threshold": args.gate_peak_threshold,
            "rms_threshold": args.gate_rms_threshold,
            "preserve_leading_silence_ms": args.preserve_leading_silence_ms,
            "preserve_trailing_silence_ms": args.preserve_trailing_silence_ms,
            "compress_internal_silence_to_ms": args.compress_internal_silence_to_ms,
            "min_internal_silence_run_ms": args.min_internal_silence_run_ms,
        },
        "vad_trim": {
            "enabled": args.vad_trim,
            "aggressiveness": args.vad_aggressiveness,
            "padding_ms": args.vad_padding_ms,
        },
        "empty_retry_count": args.empty_retry_count,
        "results": results,
        "verbosity_drift_warnings": [
            {
                "language": result["language"],
                "dataset_source": result.get("dataset_source"),
                "verbosity_ratio": result.get("verbosity_ratio"),
                "hyp_chars_total": result.get("hyp_chars_total"),
                "ref_chars_total": result.get("ref_chars_total"),
                "warning_range": result.get("verbosity_drift_warning_range"),
            }
            for result in results
            if result.get("verbosity_drift_warning")
        ],
    }
    attach_measurement_contract(
        payload,
        limit=args.limit,
        model_label=args.model_label or args.model,
        config_hash=config_hash,
        harness_git_sha=harness_git_sha,
        normalization_version_hash=normalization_version_hash,
        server_log_path=args.server_log_path,
        elapsed_seconds=args.elapsed_seconds,
        energy_joules=args.energy_joules,
        emissions_kg=args.emissions_kg,
    )

    for result in results:
        normalized_wer_ci = result["wer_normalized_bootstrap_ci"]
        print(
            f"{result['language']}: WER={result['wer']:.4f} "
            f"({result['wer_percent']:.2f}%), CER={result['cer_percent']:.2f}%, "
            f"CER(no-space)={result['cer_no_whitespace_percent']:.2f}%, "
            f"norm WER={result['wer_normalized_percent']:.2f}%, "
            f"norm WER 95% CI="
            f"[{normalized_wer_ci['low_percent']:.2f}, {normalized_wer_ci['high_percent']:.2f}]%, "
            f"open-asr-like WER={result['metric_profiles']['open_asr_like']['wer_percent']:.2f}%, "
            f"norm CER={result['cer_normalized_percent']:.2f}% "
            f"over {result['samples_evaluated']} samples with "
            f"{result['empty_prediction_count']} empty predictions"
        )

    if args.out:
        out_path = ensure_config_hash_in_filename(Path(args.out), config_hash)
        payload["report_path"] = str(out_path)
        write_json(out_path, payload)
        print(f"Saved report to: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
