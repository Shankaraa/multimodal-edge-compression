from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODE_CONFIGS = {
    "bf16": "configs/vllm/bf16_current_harness.yaml",
    "fp8": "configs/vllm/fp8_round1.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a vLLM-served Voxtral variant on a small comparable slice."
    )
    parser.add_argument("--model-path", required=True, help="Model path to serve.")
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_CONFIGS),
        default=None,
        help=(
            "Serve with the repo's same-harness BF16 or FP8 config. "
            "Use --config to override the selected config explicitly."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="vLLM YAML config path. Optional when --mode is supplied.",
    )
    parser.add_argument("--port", type=int, required=True, help="Local server port.")
    parser.add_argument("--label", required=True, help="Short label for output files.")
    parser.add_argument(
        "--model-label",
        default=None,
        help="Stable label stamped into report contracts, such as bf16_baseline.",
    )
    parser.add_argument("--lang", default="en_us", help="FLEURS language code.")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to evaluate.")
    parser.add_argument(
        "--dataset-source",
        choices=("google_fleurs", "open_asr_multilingual"),
        default="google_fleurs",
        help="Dataset wrapper used for evaluation.",
    )
    parser.add_argument(
        "--prompt",
        default="Transcribe this audio.",
        help="Prompt passed to the transcription endpoint.",
    )
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
        help="Optional sampling temperature. The model card recommends 0.0.",
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
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=900,
        help="Seconds to wait for server readiness.",
    )
    parser.add_argument(
        "--gate-silence",
        action="store_true",
        help="Apply speech-aware silence gating during the benchmark evaluation.",
    )
    parser.add_argument(
        "--vad-trim",
        action="store_true",
        help="Strip leading/trailing silence with conservative WebRTC VAD during the benchmark.",
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
        "--empty-retry-count",
        type=int,
        default=0,
        help="Retry a sample this many times when the transcription endpoint returns empty text.",
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
    return parser.parse_args()


def get_gpu_snapshot() -> dict[str, str | int]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    first_line = completed.stdout.strip().splitlines()[0]
    name, memory_used, memory_total, utilization = [part.strip() for part in first_line.split(",")]
    return {
        "gpu_name": name,
        "memory_used_mib": int(memory_used),
        "memory_total_mib": int(memory_total),
        "utilization_gpu_percent": int(utilization),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_first_dataset_sample(*, lang_code: str, dataset_source: str) -> dict[str, Any]:
    from voxtral_project.dataset_utils import load_transcription_dataset_streaming

    dataset = load_transcription_dataset_streaming(
        lang_code=lang_code,
        split="test",
        dataset_source=dataset_source,
    )
    return next(iter(dataset))


def benchmark_first_request(
    *,
    base_url: str,
    model: str,
    lang_code: str,
    dataset_source: str,
    prompt: str,
    language_hint_mode: str,
    temperature: float | None,
    target_streaming_delay_ms: int | None,
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
) -> dict[str, Any]:
    from voxtral_project.api import transcribe_audio_bytes
    from voxtral_project.audio import (
        audio_array_to_wav_bytes,
        prepare_audio_array_for_transcription,
    )
    from voxtral_project.dataset_utils import get_sample_text

    sample = get_first_dataset_sample(
        lang_code=lang_code,
        dataset_source=dataset_source,
    )
    prepared_audio, audio_diagnostics = prepare_audio_array_for_transcription(
        sample["audio"]["array"],
        sample["audio"]["sampling_rate"],
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
    audio_bytes = audio_array_to_wav_bytes(
        audio_array=prepared_audio,
        sample_rate=sample["audio"]["sampling_rate"],
    )

    started = time.perf_counter()
    transcript = transcribe_audio_bytes(
        base_url=base_url,
        model=model,
        audio_bytes=audio_bytes,
        mime_type="audio/wav",
        prompt=prompt,
        language=(lang_code.split("_", 1)[0].lower() if language_hint_mode == "fleurs_primary" else None),
        temperature=temperature,
        target_streaming_delay_ms=target_streaming_delay_ms,
        max_tokens=1000,
        timeout=300,
    )
    elapsed = time.perf_counter() - started

    return {
        "sample_id": str(sample.get("id", sample.get("file_name", ""))),
        "reference": get_sample_text(sample),
        "prediction": transcript,
        "dataset_source": dataset_source,
        "prompt": prompt,
        "language_hint_mode": language_hint_mode,
        "temperature": temperature,
        "target_streaming_delay_ms": target_streaming_delay_ms,
        "latency_seconds": elapsed,
        "audio_duration_seconds": float(audio_diagnostics["duration_seconds"]),
        "vad_trim_applied": bool(audio_diagnostics["vad_trim_applied"]),
        "vad_trim_changed_audio": bool(audio_diagnostics["vad_trim_changed_audio"]),
        "vad_trim_seconds_removed": float(audio_diagnostics["vad_trim_seconds_removed"]),
        "vad_trim_fraction_removed": float(audio_diagnostics["vad_trim_fraction_removed"]),
        "vad_trim_duration_after_seconds": float(
            audio_diagnostics["vad_trim_duration_after_seconds"]
        ),
        "gated_audio_duration_seconds": float(audio_diagnostics["speech_gating_duration_after_seconds"]),
        "speech_gating_seconds_removed": float(audio_diagnostics["speech_gating_seconds_removed"]),
        "speech_gating_fraction_removed": float(audio_diagnostics["speech_gating_fraction_removed"]),
        "quiet_audio_boosted": bool(audio_diagnostics["quiet_audio_boosted"]),
        "audio_gain_applied": float(audio_diagnostics["gain_applied"]),
        "speech_gating_applied": bool(audio_diagnostics["speech_gating_applied"]),
        "speech_gating_changed_audio": bool(audio_diagnostics["speech_gating_changed_audio"]),
    }


def run_eval(
    *,
    base_url: str,
    model: str,
    model_label: str,
    lang: str,
    limit: int,
    dataset_source: str,
    prompt: str,
    language_hint_mode: str,
    temperature: float | None,
    target_streaming_delay_ms: int | None,
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
    empty_retry_count: int,
    eval_report: Path,
    energy_report: Path,
    config_hash: str,
    harness_git_sha: str | None,
    normalization_version_hash: str,
    server_log_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        "scripts/measure_energy.py",
        "--report",
        str(energy_report),
        "--",
        sys.executable,
        "scripts/evaluate_fleurs.py",
        "--lang",
        lang,
        "--limit",
        str(limit),
        "--dataset-source",
        dataset_source,
        "--base-url",
        base_url,
        "--model",
        model,
        "--model-label",
        model_label,
        "--config-hash",
        config_hash,
        "--normalization-version",
        normalization_version_hash,
        "--server-log-path",
        str(server_log_path),
        "--prompt",
        prompt,
        "--language-hint-mode",
        language_hint_mode,
        "--out",
        str(eval_report),
        "--empty-retry-count",
        str(empty_retry_count),
    ]
    if target_streaming_delay_ms is not None:
        command.extend(["--target-streaming-delay-ms", str(target_streaming_delay_ms)])
    if harness_git_sha:
        command.extend(["--harness-git-sha", harness_git_sha])
    if temperature is not None:
        command.extend(["--temperature", str(temperature)])
    if vad_trim:
        command.extend(
            [
                "--vad-trim",
                "--vad-aggressiveness",
                str(vad_aggressiveness),
                "--vad-padding-ms",
                str(vad_padding_ms),
            ]
        )
    if gate_silence:
        command.extend(
            [
                "--gate-silence",
                "--gate-frame-ms",
                str(gate_frame_ms),
                "--gate-peak-threshold",
                str(gate_peak_threshold),
                "--gate-rms-threshold",
                str(gate_rms_threshold),
                "--preserve-leading-silence-ms",
                str(preserve_leading_silence_ms),
                "--preserve-trailing-silence-ms",
                str(preserve_trailing_silence_ms),
                "--min-internal-silence-run-ms",
                str(min_internal_silence_run_ms),
            ]
        )
        if compress_internal_silence_to_ms is not None:
            command.extend(
                [
                    "--compress-internal-silence-to-ms",
                    str(compress_internal_silence_to_ms),
                ]
            )
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    eval_payload = json.loads(eval_report.read_text(encoding="utf-8"))
    energy_payload = json.loads(energy_report.read_text(encoding="utf-8"))
    from voxtral_project.audio import write_json
    from voxtral_project.reporting import attach_measurement_contract

    attach_measurement_contract(
        eval_payload,
        limit=limit,
        model_label=model_label,
        config_hash=config_hash,
        harness_git_sha=harness_git_sha,
        normalization_version_hash=normalization_version_hash,
        server_log_path=str(server_log_path),
        elapsed_seconds=energy_payload.get("elapsed_seconds"),
        energy_joules=energy_payload.get("energy_joules"),
        emissions_kg=energy_payload.get("emissions_kg"),
    )
    write_json(eval_report, eval_payload)
    return eval_payload, energy_payload


def run_warmup(
    *,
    base_url: str,
    model: str,
    lang: str,
    dataset_source: str,
    prompt: str,
    language_hint_mode: str,
    temperature: float | None,
    target_streaming_delay_ms: int | None,
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
    warmup_report: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/warm_fleurs_prefix_cache.py",
        "--lang",
        lang,
        "--sample-index",
        "0",
        "--dataset-source",
        dataset_source,
        "--base-url",
        base_url,
        "--model",
        model,
        "--prompt",
        prompt,
        "--language-hint-mode",
        language_hint_mode,
        "--out",
        str(warmup_report),
    ]
    if target_streaming_delay_ms is not None:
        command.extend(["--target-streaming-delay-ms", str(target_streaming_delay_ms)])
    if temperature is not None:
        command.extend(["--temperature", str(temperature)])
    if vad_trim:
        command.extend(
            [
                "--vad-trim",
                "--vad-aggressiveness",
                str(vad_aggressiveness),
                "--vad-padding-ms",
                str(vad_padding_ms),
            ]
        )
    if gate_silence:
        command.extend(
            [
                "--gate-silence",
                "--gate-frame-ms",
                str(gate_frame_ms),
                "--gate-peak-threshold",
                str(gate_peak_threshold),
                "--gate-rms-threshold",
                str(gate_rms_threshold),
                "--preserve-leading-silence-ms",
                str(preserve_leading_silence_ms),
                "--preserve-trailing-silence-ms",
                str(preserve_trailing_silence_ms),
                "--min-internal-silence-run-ms",
                str(min_internal_silence_run_ms),
            ]
        )
        if compress_internal_silence_to_ms is not None:
            command.extend(
                [
                    "--compress-internal-silence-to-ms",
                    str(compress_internal_silence_to_ms),
                ]
            )

    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return json.loads(warmup_report.read_text(encoding="utf-8"))


def build_summary(
    *,
    label: str,
    model_path: str,
    config_path: str,
    base_url: str,
    served_model: str,
    startup_seconds: float,
    gpu_snapshot: dict[str, Any],
    first_request: dict[str, Any],
    eval_payload: dict[str, Any],
    energy_payload: dict[str, Any],
    config_sha256: str,
    warmup_payload: dict[str, Any],
    warmup_report: Path,
    log_path: Path,
) -> dict[str, Any]:
    result = eval_payload["results"][0]
    open_asr_like_profile = result.get("metric_profiles", {}).get("open_asr_like", {})
    total_audio_seconds = sum(sample["audio_duration_seconds"] for sample in result["samples"])
    elapsed_eval_seconds = float(energy_payload["elapsed_seconds"])
    measurement_summary = (
        eval_payload.get("measurement_summaries", [{}])[0]
        if eval_payload.get("measurement_summaries")
        else {}
    )
    summary = {
        "label": label,
        "model_path": model_path,
        "config_path": config_path,
        "config_sha256": config_sha256,
        "base_url": base_url,
        "served_model": served_model,
        "evaluator_harness_version": eval_payload.get("harness_version"),
        "prompt": first_request.get("prompt"),
        "language_hint_mode": first_request.get("language_hint_mode"),
        "temperature": first_request.get("temperature"),
        "target_streaming_delay_ms": first_request.get("target_streaming_delay_ms"),
        "vad_trim": eval_payload.get("vad_trim"),
        "speech_gating": eval_payload.get("speech_gating"),
        "startup_seconds": startup_seconds,
        "gpu_snapshot": gpu_snapshot,
        "first_request": first_request,
        "warmup": {
            "script": "scripts/warm_fleurs_prefix_cache.py",
            "ran_before_first_request": True,
            "ran_before_timed_eval": True,
            "report_path": str(warmup_report),
            "sample_id": warmup_payload.get("sample_id"),
            "language": warmup_payload.get("language"),
            "dataset_source": warmup_payload.get("dataset_source"),
            "empty_prediction": not str(warmup_payload.get("prediction", "")).strip(),
        },
        "evaluation": {
            "language": result["language"],
            "dataset_source": result.get("dataset_source"),
            "samples_evaluated": result["samples_evaluated"],
            "empty_prediction_count": result["empty_prediction_count"],
            "empty_retry_sample_count": result.get("empty_retry_sample_count", 0),
            "empty_retry_request_count": result.get("empty_retry_request_count", 0),
            "wer_percent": result["wer_percent"],
            "wer_bootstrap_ci": result.get("wer_bootstrap_ci"),
            "wer_normalized_percent": result["wer_normalized_percent"],
            "wer_normalized_bootstrap_ci": result.get("wer_normalized_bootstrap_ci"),
            "wer_open_asr_like_percent": open_asr_like_profile.get("wer_percent"),
            "wer_open_asr_like_bootstrap_ci": result.get("wer_open_asr_like_bootstrap_ci"),
            "cer_percent": result["cer_percent"],
            "cer_normalized_percent": result["cer_normalized_percent"],
            "cer_no_whitespace_normalized_percent": result.get(
                "cer_no_whitespace_normalized_percent"
            ),
            "cer_no_whitespace_normalized_bootstrap_ci": result.get(
                "cer_no_whitespace_normalized_bootstrap_ci"
            ),
            "hyp_chars_total": result.get("hyp_chars_total"),
            "ref_chars_total": result.get("ref_chars_total"),
            "verbosity_ratio": result.get("verbosity_ratio"),
            "verbosity_drift_warning": result.get("verbosity_drift_warning"),
            "verbosity_drift_warning_range": result.get("verbosity_drift_warning_range"),
            "elapsed_seconds": elapsed_eval_seconds,
            "energy_joules": energy_payload["energy_joules"],
            "emissions_kg": energy_payload.get("emissions_kg"),
            "ttft_seconds_p50": result.get("ttft_seconds_p50"),
            "ttft_seconds_p95": result.get("ttft_seconds_p95"),
            "latency_total_seconds_p50": result.get("latency_total_seconds_p50"),
            "latency_total_seconds_p95": result.get("latency_total_seconds_p95"),
            "streaming_tokens_per_second_p50": result.get("streaming_tokens_per_second_p50"),
            "streaming_tokens_per_second_p95": result.get("streaming_tokens_per_second_p95"),
            "realtime_failure_threshold_note": result.get("realtime_failure_threshold_note"),
            "total_audio_seconds": total_audio_seconds,
            "audio_seconds_per_wall_second": (
                total_audio_seconds / elapsed_eval_seconds if elapsed_eval_seconds else None
            ),
            "samples_per_second": (
                result["samples_evaluated"] / elapsed_eval_seconds if elapsed_eval_seconds else None
            ),
            "report_path": str(Path(eval_payload.get("report_path", ""))) if eval_payload.get("report_path") else None,
            "energy_report_path": str(energy_payload.get("report_path", "")) if energy_payload.get("report_path") else None,
        },
        "log_path": str(log_path),
    }
    if measurement_summary:
        summary["measurement_summary"] = measurement_summary
        summary.update(measurement_summary)
    return summary


def build_failed_summary(
    *,
    label: str,
    model_path: str,
    config_path: str,
    base_url: str,
    served_model: str,
    startup_seconds: float,
    gpu_snapshot: dict[str, Any],
    first_request: dict[str, Any],
    config_sha256: str,
    warmup_report: Path,
    error: str,
    eval_report: Path,
    energy_report: Path,
    log_path: Path,
) -> dict[str, Any]:
    return {
        "label": label,
        "model_path": model_path,
        "config_path": config_path,
        "config_sha256": config_sha256,
        "base_url": base_url,
        "served_model": served_model,
        "prompt": first_request.get("prompt"),
        "language_hint_mode": first_request.get("language_hint_mode"),
        "temperature": first_request.get("temperature"),
        "target_streaming_delay_ms": first_request.get("target_streaming_delay_ms"),
        "startup_seconds": startup_seconds,
        "gpu_snapshot": gpu_snapshot,
        "first_request": first_request,
        "warmup": {
            "script": "scripts/warm_fleurs_prefix_cache.py",
            "ran_before_first_request": True,
            "ran_before_timed_eval": True,
            "report_exists": warmup_report.exists(),
            "report_path": str(warmup_report),
        },
        "evaluation": {
            "error": error,
            "report_exists": eval_report.exists(),
            "energy_report_exists": energy_report.exists(),
            "report_path": str(eval_report),
            "energy_report_path": str(energy_report),
        },
        "log_path": str(log_path),
    }


def main() -> int:
    from voxtral_project.api import wait_for_server_ready
    from voxtral_project.audio import write_json
    from voxtral_project.reporting import get_git_head_sha, normalization_version

    args = parse_args()
    config_arg = args.config or (MODE_CONFIGS[args.mode] if args.mode else None)
    if config_arg is None:
        raise ValueError("Provide either --config or --mode {bf16,fp8}.")

    base_url = f"http://127.0.0.1:{args.port}/v1"
    report_dir = PROJECT_ROOT / "reports"
    log_dir = PROJECT_ROOT / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_sha256 = file_sha256(config_path)
    config_hash_tag = f"cfg{config_sha256}"
    harness_git_sha = get_git_head_sha(PROJECT_ROOT)
    normalization_version_hash = normalization_version(PROJECT_ROOT)
    model_label = args.model_label or args.label

    log_path = log_dir / f"{args.label}_{config_hash_tag}_{args.lang}_limit{args.limit}_benchmark_server.log"
    eval_report = report_dir / f"fleurs_{args.label}_{config_hash_tag}_{args.lang}_limit{args.limit}.json"
    energy_report = report_dir / f"energy_fleurs_{args.label}_{config_hash_tag}_{args.lang}_limit{args.limit}.json"
    summary_report = report_dir / f"benchmark_{args.label}_{config_hash_tag}_{args.lang}_limit{args.limit}.json"
    warmup_report = report_dir / f"warmup_{args.label}_{config_hash_tag}_{args.lang}_limit{args.limit}.json"

    start_time = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                "bash",
                "scripts/start_vllm_server.sh",
                args.model_path,
                config_arg,
                str(args.port),
            ],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
        )

        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Server exited early with code {process.returncode}. See {log_path}."
                    )
                try:
                    models = wait_for_server_ready(
                        base_url=base_url,
                        timeout=5,
                        interval=1.0,
                    )
                    break
                except TimeoutError:
                    if time.monotonic() >= deadline:
                        raise

            startup_seconds = time.perf_counter() - start_time
            served_model = models[0]["id"] if models else args.label
            gpu_snapshot = get_gpu_snapshot()
            warmup_payload = run_warmup(
                base_url=base_url,
                model=served_model,
                lang=args.lang,
                dataset_source=args.dataset_source,
                prompt=args.prompt,
                language_hint_mode=args.language_hint_mode,
                temperature=args.temperature,
                target_streaming_delay_ms=args.target_streaming_delay_ms,
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
                warmup_report=warmup_report,
            )
            first_request = benchmark_first_request(
                base_url=f"http://127.0.0.1:{args.port}",
                model=served_model,
                lang_code=args.lang,
                dataset_source=args.dataset_source,
                prompt=args.prompt,
                language_hint_mode=args.language_hint_mode,
                temperature=args.temperature,
                target_streaming_delay_ms=args.target_streaming_delay_ms,
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
            try:
                eval_payload, energy_payload = run_eval(
                    base_url=base_url,
                    model=served_model,
                    model_label=model_label,
                    lang=args.lang,
                    limit=args.limit,
                    dataset_source=args.dataset_source,
                    prompt=args.prompt,
                    language_hint_mode=args.language_hint_mode,
                    temperature=args.temperature,
                    target_streaming_delay_ms=args.target_streaming_delay_ms,
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
                    empty_retry_count=args.empty_retry_count,
                    eval_report=eval_report,
                    energy_report=energy_report,
                    config_hash=config_sha256,
                    harness_git_sha=harness_git_sha,
                    normalization_version_hash=normalization_version_hash,
                    server_log_path=log_path,
                )
            except subprocess.CalledProcessError as exc:
                summary = build_failed_summary(
                    label=args.label,
                    model_path=args.model_path,
                    config_path=config_arg,
                    base_url=base_url,
                    served_model=served_model,
                    startup_seconds=startup_seconds,
                    gpu_snapshot=gpu_snapshot,
                    first_request=first_request,
                    config_sha256=config_sha256,
                    warmup_report=warmup_report,
                    error=str(exc),
                    eval_report=eval_report,
                    energy_report=energy_report,
                    log_path=log_path,
                )
                summary["mode"] = args.mode
                write_json(summary_report, summary)
                print(f"Benchmark summary written to: {summary_report.resolve()}")
                print(json.dumps(summary, indent=2, ensure_ascii=False))
                raise

            summary = build_summary(
                label=args.label,
                model_path=args.model_path,
                config_path=config_arg,
                base_url=base_url,
                served_model=served_model,
                startup_seconds=startup_seconds,
                gpu_snapshot=gpu_snapshot,
                first_request=first_request,
                eval_payload=eval_payload,
                energy_payload=energy_payload,
                config_sha256=config_sha256,
                warmup_payload=warmup_payload,
                warmup_report=warmup_report,
                log_path=log_path,
            )
            summary["mode"] = args.mode
            write_json(summary_report, summary)
            print(f"Benchmark summary written to: {summary_report.resolve()}")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
