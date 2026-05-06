from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


LANGUAGES = (
    "en_us",
    "fr_fr",
    "hi_in",
    "ja_jp",
    "es_419",
    "de_de",
    "pt_br",
    "cmn_hans_cn",
    "ar_eg",
    "ru_ru",
    "it_it",
    "nl_nl",
    "ko_kr",
)

DATASET_SOURCES = (
    "google_fleurs",
    "open_asr_multilingual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all 13 FLEURS languages across both repo corpora.")
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--model", default="voxtral-realtime")
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-streaming-delay-ms", type=int, default=None)
    parser.add_argument("--language-hint-mode", choices=("none", "fleurs_primary"), default="fleurs_primary")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--empty-retry-count", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--dataset-source", action="append", choices=DATASET_SOURCES, default=None)
    parser.add_argument("--lang", action="append", default=None)
    return parser.parse_args()


def newest_matching_report(path: Path) -> Path:
    matches = sorted(
        path.parent.glob(f"{path.stem}*.json"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No report produced for expected stem {path.stem} in {path.parent}")
    return matches[0]


def run_eval(args: argparse.Namespace, *, dataset_source: str, languages: list[str], output_dir: Path) -> Path:
    raw_out = output_dir / f"{dataset_source}_limit{args.limit}.json"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_fleurs.py"),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--limit",
        str(args.limit),
        "--dataset-source",
        dataset_source,
        "--language-hint-mode",
        args.language_hint_mode,
        "--temperature",
        str(args.temperature),
        "--empty-retry-count",
        str(args.empty_retry_count),
        "--max-tokens",
        str(args.max_tokens),
        "--out",
        str(raw_out),
    ]
    if args.model_label:
        command.extend(["--model-label", args.model_label])
    if args.config:
        command.extend(["--config", args.config])
    if args.target_streaming_delay_ms is not None:
        command.extend(["--target-streaming-delay-ms", str(args.target_streaming_delay_ms)])
    for lang in languages:
        command.extend(["--lang", lang])

    subprocess.run(command, check=True)
    return newest_matching_report(raw_out)


def aggregate_report(report_paths: list[Path]) -> dict[str, Any]:
    source_summaries: list[dict[str, Any]] = []
    weighted_wer_sum = 0.0
    total_samples = 0
    total_empty = 0
    total_retry_requests = 0
    verbosity_warnings: list[dict[str, Any]] = []

    for report_path in report_paths:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_empty = 0
        report_weighted_wer_sum = 0.0
        report_samples = 0
        for result in payload["results"]:
            samples = int(result["samples_evaluated"])
            report_samples += samples
            report_weighted_wer_sum += float(result["wer_normalized"]) * samples
            report_empty += int(result["empty_prediction_count"])
            total_retry_requests += int(result.get("empty_retry_request_count", 0))
        weighted_wer_sum += report_weighted_wer_sum
        total_samples += report_samples
        total_empty += report_empty
        verbosity_warnings.extend(payload.get("verbosity_drift_warnings", []))
        source_summaries.append(
            {
                "report_path": str(report_path),
                "dataset_sources": sorted({result["dataset_source"] for result in payload["results"]}),
                "languages": [result["language"] for result in payload["results"]],
                "samples": report_samples,
                "normalized_wer_sample_weighted": (
                    report_weighted_wer_sum / report_samples if report_samples else None
                ),
                "normalized_wer_percent": (
                    100.0 * report_weighted_wer_sum / report_samples if report_samples else None
                ),
                "empty_predictions": report_empty,
            }
        )

    normalized_wer = weighted_wer_sum / total_samples if total_samples else None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reports": source_summaries,
        "normalized_wer_sample_weighted": normalized_wer,
        "normalized_wer_percent": 100.0 * normalized_wer if normalized_wer is not None else None,
        "samples": total_samples,
        "empty_predictions": total_empty,
        "empty_retry_request_count": total_retry_requests,
        "verbosity_drift_warnings": verbosity_warnings,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    languages = args.lang or list(LANGUAGES)
    dataset_sources = args.dataset_source or list(DATASET_SOURCES)

    report_paths = [
        run_eval(args, dataset_source=dataset_source, languages=languages, output_dir=output_dir)
        for dataset_source in dataset_sources
    ]
    summary = aggregate_report(report_paths)
    summary.update(
        {
            "base_url": args.base_url,
            "model": args.model,
            "model_label": args.model_label,
            "config": args.config,
            "limit_per_language": args.limit,
            "target_streaming_delay_ms": args.target_streaming_delay_ms,
            "dataset_sources": dataset_sources,
            "languages": languages,
        }
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
