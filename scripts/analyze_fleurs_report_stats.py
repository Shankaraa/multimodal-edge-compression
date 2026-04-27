from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REPORT_CANDIDATES = {
    "EN": [
        "reports/fleurs_fp8_gap_limit500_en_us_limit500.json",
        "reports/fleurs_fp8_gap_limit100_en_us_limit100.json",
        "reports/fleurs_fp8_en_us_limit20_quietfix.json",
    ],
    "FR": [
        "reports/fleurs_fp8_multilingual_fr_fr_limit20.json",
        "reports/fleurs_fp8_fr_fr_limit5_quietfix.json",
    ],
    "HI": [
        "reports/fleurs_fp8_multilingual_hi_in_limit20.json",
        "reports/fleurs_fp8_hi_in_limit5_quietfix.json",
    ],
    "JA": [
        "reports/fleurs_fp8_ja_jp_limit5_quietfix_v2.json",
        "reports/fleurs_fp8_ja_jp_limit5_quietfix.json",
    ],
}

DEFAULT_PAD_MARKERS = ("<pad>", "[STREAMING_PAD]")
TOKEN_PATTERN = re.compile(r"\S+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure visible pad markers and audio-duration distributions from saved "
            "FLEURS report JSON files."
        )
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Report to analyze. May be repeated. If omitted, the script uses the "
            "largest existing default report for EN, FR, HI, and JA."
        ),
    )
    parser.add_argument(
        "--pad-marker",
        action="append",
        default=[],
        help="Decoded marker string to count as a visible pad token.",
    )
    parser.add_argument(
        "--audio-token-rate",
        type=float,
        default=12.5,
        help="Estimated Voxtral realtime audio tokens per second.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Context length to compare against the measured audio-token distribution.",
    )
    parser.add_argument(
        "--out",
        default="reports/future_bets_measurements.json",
        help="Path for the JSON measurement report.",
    )
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def first_existing_default_reports() -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for label, candidates in DEFAULT_REPORT_CANDIDATES.items():
        for candidate in candidates:
            path = Path(candidate)
            if path.exists():
                selected[label] = path
                break
    return selected


def parse_report_specs(report_specs: list[str]) -> dict[str, Path]:
    if not report_specs:
        return first_existing_default_reports()

    selected: dict[str, Path] = {}
    for spec in report_specs:
        if "=" not in spec:
            raise ValueError(f"Expected LABEL=PATH, got: {spec}")
        label, path_text = spec.split("=", 1)
        label = label.strip().upper()
        path = Path(path_text.strip())
        if not label or not path:
            raise ValueError(f"Expected non-empty LABEL=PATH, got: {spec}")
        selected[label] = path
    return selected


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_samples(report: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for result in report.get("results", []):
        for sample in result.get("samples", []):
            rows.append((result, sample))
    return rows


def count_visible_pad_markers(text: str, markers: tuple[str, ...]) -> int:
    return sum(text.count(marker) for marker in markers)


def analyze_one_report(
    *,
    label: str,
    path: Path,
    pad_markers: tuple[str, ...],
    audio_token_rate: float,
    max_model_len: int,
) -> dict[str, Any]:
    report = load_report(path)
    rows = iter_samples(report)

    visible_pad_marker_count = 0
    decoded_text_token_count = 0
    samples_with_visible_pad_marker = 0
    durations: list[float] = []

    languages = sorted({str(result.get("language", "")) for result, _ in rows if result.get("language")})
    dataset_sources = sorted(
        {str(result.get("dataset_source", "")) for result, _ in rows if result.get("dataset_source")}
    )

    for _, sample in rows:
        prediction = str(sample.get("prediction", ""))
        marker_count = count_visible_pad_markers(prediction, pad_markers)
        visible_pad_marker_count += marker_count
        samples_with_visible_pad_marker += int(marker_count > 0)
        decoded_text_token_count += len(TOKEN_PATTERN.findall(prediction))

        duration = sample.get("audio_duration_seconds")
        if duration is not None:
            durations.append(float(duration))

    denominator = decoded_text_token_count + visible_pad_marker_count
    visible_pad_marker_rate = (
        visible_pad_marker_count / denominator if denominator else 0.0
    )
    p50 = percentile(durations, 0.50)
    p95 = percentile(durations, 0.95)
    max_duration = max(durations) if durations else None

    p95_audio_tokens = math.ceil(p95 * audio_token_rate) if p95 is not None else None
    max_audio_tokens = (
        math.ceil(max_duration * audio_token_rate) if max_duration is not None else None
    )

    return {
        "label": label,
        "path": str(path),
        "languages": languages,
        "dataset_sources": dataset_sources,
        "sample_count": len(rows),
        "pad_marker_measurement": {
            "measurement_kind": "decoded_text_visible_marker_lower_bound",
            "raw_generated_token_ids_available": False,
            "pad_markers": list(pad_markers),
            "visible_pad_marker_count": visible_pad_marker_count,
            "decoded_text_token_count": decoded_text_token_count,
            "samples_with_visible_pad_marker": samples_with_visible_pad_marker,
            "visible_pad_marker_rate": visible_pad_marker_rate,
            "visible_pad_marker_rate_percent": visible_pad_marker_rate * 100.0,
            "caveat": (
                "Saved reports contain decoded transcript text, not raw generated token IDs. "
                "This is a lower bound on visible pad markers, not a true decoder pad-token "
                "emission measurement."
            ),
        },
        "audio_duration_seconds": {
            "count": len(durations),
            "min": min(durations) if durations else None,
            "p50": p50,
            "p95": p95,
            "max": max_duration,
        },
        "audio_token_estimate": {
            "audio_token_rate_per_second": audio_token_rate,
            "p95_audio_tokens": p95_audio_tokens,
            "max_audio_tokens": max_audio_tokens,
            "max_model_len": max_model_len,
            "p95_fraction_of_max_model_len": (
                p95_audio_tokens / max_model_len
                if p95_audio_tokens is not None and max_model_len
                else None
            ),
            "max_fraction_of_max_model_len": (
                max_audio_tokens / max_model_len
                if max_audio_tokens is not None and max_model_len
                else None
            ),
        },
    }


def main() -> int:
    args = parse_args()
    selected_reports = parse_report_specs(args.report)
    pad_markers = tuple(args.pad_marker or DEFAULT_PAD_MARKERS)

    measurements = []
    missing = []
    for label, path in selected_reports.items():
        if not path.exists():
            missing.append({"label": label, "path": str(path)})
            continue
        measurements.append(
            analyze_one_report(
                label=label,
                path=path,
                pad_markers=pad_markers,
                audio_token_rate=args.audio_token_rate,
                max_model_len=args.max_model_len,
            )
        )

    status_block = {
        item["label"]: {
            "visible_pad_marker_rate_percent": item["pad_marker_measurement"][
                "visible_pad_marker_rate_percent"
            ],
            "audio_p50_seconds": item["audio_duration_seconds"]["p50"],
            "audio_p95_seconds": item["audio_duration_seconds"]["p95"],
        }
        for item in measurements
    }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "measurement_scope": "saved_fleurs_report_json",
        "status_block_values": status_block,
        "missing_reports": missing,
        "measurements": measurements,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for item in measurements:
        pad = item["pad_marker_measurement"]["visible_pad_marker_rate_percent"]
        durations = item["audio_duration_seconds"]
        print(
            f"{item['label']}: visible pad marker rate={pad:.4f}% "
            f"over {item['sample_count']} samples; "
            f"audio p50={durations['p50']:.2f}s p95={durations['p95']:.2f}s"
        )
    if missing:
        print(f"Missing reports: {len(missing)}")
    print(f"Saved measurement report to: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
