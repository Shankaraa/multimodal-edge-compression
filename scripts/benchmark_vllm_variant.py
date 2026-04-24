from __future__ import annotations

import argparse
import contextlib
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a vLLM-served Voxtral variant on a small comparable slice."
    )
    parser.add_argument("--model-path", required=True, help="Model path to serve.")
    parser.add_argument("--config", required=True, help="vLLM YAML config path.")
    parser.add_argument("--port", type=int, required=True, help="Local server port.")
    parser.add_argument("--label", required=True, help="Short label for output files.")
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
    gate_silence: bool,
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
        "latency_seconds": elapsed,
        "audio_duration_seconds": float(audio_diagnostics["duration_seconds"]),
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
    lang: str,
    limit: int,
    dataset_source: str,
    prompt: str,
    language_hint_mode: str,
    temperature: float | None,
    gate_silence: bool,
    gate_frame_ms: float,
    gate_peak_threshold: float,
    gate_rms_threshold: float,
    preserve_leading_silence_ms: float,
    preserve_trailing_silence_ms: float,
    compress_internal_silence_to_ms: float | None,
    min_internal_silence_run_ms: float,
    eval_report: Path,
    energy_report: Path,
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
        "--prompt",
        prompt,
        "--language-hint-mode",
        language_hint_mode,
        "--out",
        str(eval_report),
    ]
    if temperature is not None:
        command.extend(["--temperature", str(temperature)])
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
    return eval_payload, energy_payload


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
    log_path: Path,
) -> dict[str, Any]:
    result = eval_payload["results"][0]
    open_asr_like_profile = result.get("metric_profiles", {}).get("open_asr_like", {})
    total_audio_seconds = sum(sample["audio_duration_seconds"] for sample in result["samples"])
    elapsed_eval_seconds = float(energy_payload["elapsed_seconds"])
    return {
        "label": label,
        "model_path": model_path,
        "config_path": config_path,
        "base_url": base_url,
        "served_model": served_model,
        "prompt": first_request.get("prompt"),
        "language_hint_mode": first_request.get("language_hint_mode"),
        "temperature": first_request.get("temperature"),
        "speech_gating": eval_payload.get("speech_gating"),
        "startup_seconds": startup_seconds,
        "gpu_snapshot": gpu_snapshot,
        "first_request": first_request,
        "evaluation": {
            "language": result["language"],
            "dataset_source": result.get("dataset_source"),
            "samples_evaluated": result["samples_evaluated"],
            "empty_prediction_count": result["empty_prediction_count"],
            "wer_percent": result["wer_percent"],
            "wer_normalized_percent": result["wer_normalized_percent"],
            "wer_open_asr_like_percent": open_asr_like_profile.get("wer_percent"),
            "cer_percent": result["cer_percent"],
            "cer_normalized_percent": result["cer_normalized_percent"],
            "elapsed_seconds": elapsed_eval_seconds,
            "energy_joules": energy_payload["energy_joules"],
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
    error: str,
    eval_report: Path,
    energy_report: Path,
    log_path: Path,
) -> dict[str, Any]:
    return {
        "label": label,
        "model_path": model_path,
        "config_path": config_path,
        "base_url": base_url,
        "served_model": served_model,
        "prompt": first_request.get("prompt"),
        "language_hint_mode": first_request.get("language_hint_mode"),
        "temperature": first_request.get("temperature"),
        "startup_seconds": startup_seconds,
        "gpu_snapshot": gpu_snapshot,
        "first_request": first_request,
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

    args = parse_args()

    base_url = f"http://127.0.0.1:{args.port}/v1"
    report_dir = PROJECT_ROOT / "reports"
    log_dir = PROJECT_ROOT / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.label}_benchmark_server.log"
    eval_report = report_dir / f"fleurs_{args.label}_{args.lang}_limit{args.limit}.json"
    energy_report = report_dir / f"energy_fleurs_{args.label}_{args.lang}_limit{args.limit}.json"
    summary_report = report_dir / f"benchmark_{args.label}_{args.lang}_limit{args.limit}.json"

    start_time = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                "bash",
                "scripts/start_vllm_server.sh",
                args.model_path,
                args.config,
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
            first_request = benchmark_first_request(
                base_url=f"http://127.0.0.1:{args.port}",
                model=served_model,
                lang_code=args.lang,
                dataset_source=args.dataset_source,
                prompt=args.prompt,
                language_hint_mode=args.language_hint_mode,
                temperature=args.temperature,
                gate_silence=args.gate_silence,
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
                    lang=args.lang,
                    limit=args.limit,
                    dataset_source=args.dataset_source,
                    prompt=args.prompt,
                    language_hint_mode=args.language_hint_mode,
                    temperature=args.temperature,
                    gate_silence=args.gate_silence,
                    gate_frame_ms=args.gate_frame_ms,
                    gate_peak_threshold=args.gate_peak_threshold,
                    gate_rms_threshold=args.gate_rms_threshold,
                    preserve_leading_silence_ms=args.preserve_leading_silence_ms,
                    preserve_trailing_silence_ms=args.preserve_trailing_silence_ms,
                    compress_internal_silence_to_ms=args.compress_internal_silence_to_ms,
                    min_internal_silence_run_ms=args.min_internal_silence_run_ms,
                    eval_report=eval_report,
                    energy_report=energy_report,
                )
            except subprocess.CalledProcessError as exc:
                summary = build_failed_summary(
                    label=args.label,
                    model_path=args.model_path,
                    config_path=args.config,
                    base_url=base_url,
                    served_model=served_model,
                    startup_seconds=startup_seconds,
                    gpu_snapshot=gpu_snapshot,
                    first_request=first_request,
                    error=str(exc),
                    eval_report=eval_report,
                    energy_report=energy_report,
                    log_path=log_path,
                )
                write_json(summary_report, summary)
                print(f"Benchmark summary written to: {summary_report.resolve()}")
                print(json.dumps(summary, indent=2, ensure_ascii=False))
                raise

            summary = build_summary(
                label=args.label,
                model_path=args.model_path,
                config_path=args.config,
                base_url=base_url,
                served_model=served_model,
                startup_seconds=startup_seconds,
                gpu_snapshot=gpu_snapshot,
                first_request=first_request,
                eval_payload=eval_payload,
                energy_payload=energy_payload,
                log_path=log_path,
            )
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
