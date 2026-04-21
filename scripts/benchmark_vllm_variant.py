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


def get_first_fleurs_sample(lang_code: str) -> dict[str, Any]:
    from voxtral_project.dataset_utils import load_fleurs_streaming

    fleurs = load_fleurs_streaming(lang_code=lang_code, split="test")
    return next(iter(fleurs))


def benchmark_first_request(
    *,
    base_url: str,
    model: str,
    lang_code: str,
    prompt: str,
    language_hint_mode: str,
    temperature: float | None,
) -> dict[str, Any]:
    from voxtral_project.api import transcribe_audio_bytes
    from voxtral_project.audio import (
        audio_array_to_wav_bytes,
        prepare_audio_array_for_transcription,
    )

    sample = get_first_fleurs_sample(lang_code)
    prepared_audio, audio_diagnostics = prepare_audio_array_for_transcription(
        sample["audio"]["array"],
        sample["audio"]["sampling_rate"],
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
        "sample_id": str(sample.get("id", "")),
        "reference": sample["transcription"],
        "prediction": transcript,
        "prompt": prompt,
        "language_hint_mode": language_hint_mode,
        "temperature": temperature,
        "latency_seconds": elapsed,
        "audio_duration_seconds": float(audio_diagnostics["duration_seconds"]),
        "quiet_audio_boosted": bool(audio_diagnostics["quiet_audio_boosted"]),
        "audio_gain_applied": float(audio_diagnostics["gain_applied"]),
    }


def run_eval(
    *,
    base_url: str,
    model: str,
    lang: str,
    limit: int,
    prompt: str,
    language_hint_mode: str,
    temperature: float | None,
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
        "startup_seconds": startup_seconds,
        "gpu_snapshot": gpu_snapshot,
        "first_request": first_request,
        "evaluation": {
            "language": result["language"],
            "samples_evaluated": result["samples_evaluated"],
            "empty_prediction_count": result["empty_prediction_count"],
            "wer_percent": result["wer_percent"],
            "wer_normalized_percent": result["wer_normalized_percent"],
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
                prompt=args.prompt,
                language_hint_mode=args.language_hint_mode,
                temperature=args.temperature,
            )
            try:
                eval_payload, energy_payload = run_eval(
                    base_url=base_url,
                    model=served_model,
                    lang=args.lang,
                    limit=args.limit,
                    prompt=args.prompt,
                    language_hint_mode=args.language_hint_mode,
                    temperature=args.temperature,
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
