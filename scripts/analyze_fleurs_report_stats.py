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
        "reports/fleurs_fp8_tracka_novad_hint_retry2_en500_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_en_us_limit500.json",
        "reports/fleurs_fp8_gap_limit500_en_us_limit500.json",
        "reports/fleurs_fp8_gap_limit100_en_us_limit100.json",
        "reports/fleurs_fp8_en_us_limit20_quietfix.json",
    ],
    "FR": [
        "reports/fleurs_fp8_tracka_novad_hint_retry2_fr100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_fr_fr_limit100.json",
        "reports/fleurs_fp8_multilingual_fr_fr_limit20.json",
        "reports/fleurs_fp8_fr_fr_limit5_quietfix.json",
    ],
    "HI": [
        "reports/fleurs_fp8_tracka_novad_hint_retry2_hi100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_hi_in_limit100.json",
        "reports/fleurs_fp8_multilingual_hi_in_limit20.json",
        "reports/fleurs_fp8_hi_in_limit5_quietfix.json",
    ],
    "JA": [
        "reports/fleurs_fp8_tracka_novad_hint_retry2_ja100_cfg4413da57d3e41bc1270ef423f9d6ad0b2295d305f1da6b387c130cbd7a9d10b4_ja_jp_limit100.json",
        "reports/fleurs_fp8_ja_jp_limit5_quietfix_v2.json",
        "reports/fleurs_fp8_ja_jp_limit5_quietfix.json",
    ],
}

DEFAULT_PAD_MARKERS = ("<pad>", "[STREAMING_PAD]", "[P]")
DEFAULT_WORD_BOUNDARY_MARKERS = ("[STREAMING_WORD]", "[W]")
RAW_TOKEN_ID_FIELDS = (
    "token_ids",
    "output_token_ids",
    "generated_token_ids",
    "raw_token_ids",
    "tokens",
)
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
        "--word-boundary-marker",
        action="append",
        default=[],
        help="Decoded marker string to count as a visible word-boundary token.",
    )
    parser.add_argument(
        "--tekken",
        default="models/voxtral-realtime/tekken.json",
        help="Tokenizer JSON used to identify special token IDs when raw token IDs are present.",
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


def load_special_token_ids(tekken_path: Path) -> dict[str, int | None]:
    if not tekken_path.exists():
        return {
            "pad_token_id": None,
            "streaming_pad_token_id": None,
            "streaming_word_token_id": None,
        }

    data = json.loads(tekken_path.read_text(encoding="utf-8"))
    by_string = {
        item.get("token_str"): int(item["rank"])
        for item in data.get("special_tokens", [])
        if "rank" in item
    }
    return {
        "pad_token_id": by_string.get("<pad>"),
        "streaming_pad_token_id": by_string.get("[STREAMING_PAD]"),
        "streaming_word_token_id": by_string.get("[STREAMING_WORD]"),
    }


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


def get_raw_token_ids(sample: dict[str, Any]) -> list[int] | None:
    for field_name in RAW_TOKEN_ID_FIELDS:
        value = sample.get(field_name)
        if not isinstance(value, list):
            continue
        if all(isinstance(item, int) for item in value):
            return list(value)
    return None


def decoded_content_count(
    text: str,
    *,
    pad_markers: tuple[str, ...],
    word_boundary_markers: tuple[str, ...],
) -> int:
    cleaned = text
    for marker in pad_markers + word_boundary_markers:
        cleaned = cleaned.replace(marker, " ")
    return len(TOKEN_PATTERN.findall(cleaned))


def analyze_one_report(
    *,
    label: str,
    path: Path,
    pad_markers: tuple[str, ...],
    word_boundary_markers: tuple[str, ...],
    special_token_ids: dict[str, int | None],
    audio_token_rate: float,
    max_model_len: int,
) -> dict[str, Any]:
    report = load_report(path)
    rows = iter_samples(report)

    pad_token_count = 0
    word_boundary_token_count = 0
    content_token_count = 0
    samples_with_pad = 0
    samples_with_word_boundary = 0
    samples_with_raw_token_ids = 0
    durations: list[float] = []

    languages = sorted({str(result.get("language", "")) for result, _ in rows if result.get("language")})
    dataset_sources = sorted(
        {str(result.get("dataset_source", "")) for result, _ in rows if result.get("dataset_source")}
    )

    for _, sample in rows:
        raw_token_ids = get_raw_token_ids(sample)
        if raw_token_ids is not None:
            samples_with_raw_token_ids += 1
            pad_ids = {
                value
                for value in (
                    special_token_ids["pad_token_id"],
                    special_token_ids["streaming_pad_token_id"],
                )
                if value is not None
            }
            word_boundary_id = special_token_ids["streaming_word_token_id"]
            sample_pad_count = sum(1 for token_id in raw_token_ids if token_id in pad_ids)
            sample_word_count = (
                sum(1 for token_id in raw_token_ids if token_id == word_boundary_id)
                if word_boundary_id is not None
                else 0
            )
            sample_content_count = len(raw_token_ids) - sample_pad_count - sample_word_count
        else:
            prediction = str(sample.get("prediction", ""))
            sample_pad_count = count_visible_pad_markers(prediction, pad_markers)
            sample_word_count = count_visible_pad_markers(prediction, word_boundary_markers)
            sample_content_count = decoded_content_count(
                prediction,
                pad_markers=pad_markers,
                word_boundary_markers=word_boundary_markers,
            )

        pad_token_count += sample_pad_count
        word_boundary_token_count += sample_word_count
        content_token_count += sample_content_count
        samples_with_pad += int(sample_pad_count > 0)
        samples_with_word_boundary += int(sample_word_count > 0)

        duration = sample.get("audio_duration_seconds")
        if duration is not None:
            durations.append(float(duration))

    denominator = pad_token_count + word_boundary_token_count + content_token_count
    pad_rate = pad_token_count / denominator if denominator else 0.0
    word_boundary_rate = word_boundary_token_count / denominator if denominator else 0.0
    content_rate = content_token_count / denominator if denominator else 0.0
    p50 = percentile(durations, 0.50)
    p95 = percentile(durations, 0.95)
    max_duration = max(durations) if durations else None

    p95_audio_tokens = math.ceil(p95 * audio_token_rate) if p95 is not None else None
    max_audio_tokens = (
        math.ceil(max_duration * audio_token_rate) if max_duration is not None else None
    )

    raw_token_ids_available = samples_with_raw_token_ids == len(rows) and len(rows) > 0

    return {
        "label": label,
        "path": str(path),
        "languages": languages,
        "dataset_sources": dataset_sources,
        "sample_count": len(rows),
        "token_rate_measurement": {
            "measurement_kind": (
                "generated_token_ids" if raw_token_ids_available else "decoded_text_visible_marker_lower_bound"
            ),
            "raw_generated_token_ids_available": raw_token_ids_available,
            "samples_with_raw_token_ids": samples_with_raw_token_ids,
            "pad_markers": list(pad_markers),
            "word_boundary_markers": list(word_boundary_markers),
            "special_token_ids": special_token_ids,
            "pad_token_count": pad_token_count,
            "word_boundary_token_count": word_boundary_token_count,
            "content_token_count": content_token_count,
            "samples_with_pad": samples_with_pad,
            "samples_with_word_boundary": samples_with_word_boundary,
            "pad_rate": pad_rate,
            "pad_rate_percent": pad_rate * 100.0,
            "word_boundary_rate": word_boundary_rate,
            "word_boundary_rate_percent": word_boundary_rate * 100.0,
            "content_rate": content_rate,
            "content_rate_percent": content_rate * 100.0,
            "caveat": (
                None
                if raw_token_ids_available
                else (
                    "Saved reports contain decoded transcript text, not raw generated token IDs. "
                    "Pad and word-boundary rates are visible-marker lower bounds, not true "
                    "decoder control-token emission rates."
                )
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
    word_boundary_markers = tuple(
        args.word_boundary_marker or DEFAULT_WORD_BOUNDARY_MARKERS
    )
    special_token_ids = load_special_token_ids(Path(args.tekken))

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
                word_boundary_markers=word_boundary_markers,
                special_token_ids=special_token_ids,
                audio_token_rate=args.audio_token_rate,
                max_model_len=args.max_model_len,
            )
        )

    status_block = {
        item["label"]: {
            "pad_rate_percent": item["token_rate_measurement"]["pad_rate_percent"],
            "word_boundary_rate_percent": item["token_rate_measurement"][
                "word_boundary_rate_percent"
            ],
            "content_rate_percent": item["token_rate_measurement"][
                "content_rate_percent"
            ],
            "audio_p50_seconds": item["audio_duration_seconds"]["p50"],
            "audio_p95_seconds": item["audio_duration_seconds"]["p95"],
        }
        for item in measurements
    }
    true_token_rates_available = all(
        item["token_rate_measurement"]["raw_generated_token_ids_available"]
        for item in measurements
    )
    max_pad_rate = max(
        (
            item["token_rate_measurement"]["pad_rate_percent"]
            for item in measurements
        ),
        default=None,
    )
    if not true_token_rates_available:
        decoder_skipping_decision = {
            "decision": "do_not_invest_from_current_artifacts",
            "reason": (
                "Current reports do not contain raw generated token IDs, so true "
                "pad-token emission cannot be measured. Visible pad/control markers are "
                "0.00% in decoded text, which is not evidence for a >60% pad-heavy future "
                "decoder-skipping bet."
            ),
        }
    elif max_pad_rate is not None and max_pad_rate > 60.0:
        decoder_skipping_decision = {
            "decision": "worth_future_investment",
            "reason": "At least one measured language has true pad-token rate above 60%.",
        }
    elif max_pad_rate is not None and max_pad_rate < 40.0:
        decoder_skipping_decision = {
            "decision": "not_worth_future_investment",
            "reason": "All measured true pad-token rates are below the 40% dead-idea threshold.",
        }
    else:
        decoder_skipping_decision = {
            "decision": "inconclusive_across_languages",
            "reason": "True pad-token rates are mixed or between the requested thresholds.",
        }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "measurement_scope": "saved_fleurs_report_json",
        "status_block_values": status_block,
        "decoder_skipping_decision": decoder_skipping_decision,
        "missing_reports": missing,
        "measurements": measurements,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for item in measurements:
        token_rates = item["token_rate_measurement"]
        durations = item["audio_duration_seconds"]
        print(
            f"{item['label']}: pad_rate={token_rates['pad_rate_percent']:.4f}% "
            f"word_boundary_rate={token_rates['word_boundary_rate_percent']:.4f}% "
            f"content_rate={token_rates['content_rate_percent']:.4f}% "
            f"over {item['sample_count']} samples; "
            f"audio p50={durations['p50']:.2f}s p95={durations['p95']:.2f}s"
        )
    if missing:
        print(f"Missing reports: {len(missing)}")
    print(f"Saved measurement report to: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
