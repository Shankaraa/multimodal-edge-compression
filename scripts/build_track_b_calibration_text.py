from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SUPPORTED_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("EN", "en_us"),
    ("ZH", "cmn_hans_cn"),
    ("HI", "hi_in"),
    ("ES", "es_419"),
    ("AR", "ar_eg"),
    ("FR", "fr_fr"),
    ("PT", "pt_br"),
    ("RU", "ru_ru"),
    ("DE", "de_de"),
    ("JA", "ja_jp"),
    ("KO", "ko_kr"),
    ("IT", "it_it"),
    ("NL", "nl_nl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the reviewed Track B multilingual GPTQ text calibration set."
    )
    parser.add_argument(
        "--dataset-source",
        choices=("google_fleurs", "open_asr_multilingual"),
        default="google_fleurs",
        help="Transcript dataset source.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to stream.")
    parser.add_argument(
        "--out",
        default="data/calibration/track_b_multilingual_text_256.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--summary-out",
        default="data/calibration/track_b_multilingual_text_256.summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=256,
        help="Total calibration records. The Track B requirement is 256.",
    )
    return parser.parse_args()


def allocation(total_samples: int) -> dict[str, dict[str, int]]:
    if total_samples < len(SUPPORTED_LANGUAGES):
        raise ValueError("total_samples must cover every supported language at least once")

    base = total_samples // len(SUPPORTED_LANGUAGES)
    extra = total_samples % len(SUPPORTED_LANGUAGES)
    result: dict[str, dict[str, int]] = {}

    for index, (label, _config) in enumerate(SUPPORTED_LANGUAGES):
        count = base + (1 if index < extra else 0)
        short = count // 2
        medium = count - short
        result[label] = {"total": count, "short": short, "medium": medium}

    return result


def compact_text(value: str) -> str:
    return " ".join(value.replace("\n", " ").split())


def take_transcripts(
    *,
    lang_config: str,
    dataset_source: str,
    split: str,
    minimum_count: int,
) -> list[dict[str, str]]:
    from voxtral_project.dataset_utils import (
        get_sample_text,
        load_transcription_dataset_streaming,
    )

    dataset = load_transcription_dataset_streaming(
        lang_code=lang_config,
        split=split,
        dataset_source=dataset_source,
    )
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for index, sample in enumerate(dataset):
        text = compact_text(get_sample_text(sample))
        if not text or text in seen:
            continue

        seen.add(text)
        rows.append(
            {
                "source_id": str(sample.get("id", index)),
                "text": text,
            }
        )
        if len(rows) >= minimum_count:
            return rows

    raise RuntimeError(
        f"Only found {len(rows)} usable transcripts for {lang_config}; "
        f"needed {minimum_count}"
    )


def build_records(
    *,
    dataset_source: str,
    split: str,
    total_samples: int,
) -> list[dict[str, object]]:
    plan = allocation(total_samples)
    records: list[dict[str, object]] = []

    for language_label, lang_config in SUPPORTED_LANGUAGES:
        counts = plan[language_label]
        medium_sizes = [3 + (i % 3) for i in range(counts["medium"])]
        needed = counts["short"] + sum(medium_sizes)
        transcripts = take_transcripts(
            lang_config=lang_config,
            dataset_source=dataset_source,
            split=split,
            minimum_count=needed,
        )

        cursor = 0
        for short_index in range(counts["short"]):
            item = transcripts[cursor]
            cursor += 1
            records.append(
                {
                    "id": f"{language_label.lower()}_short_{short_index:03d}",
                    "language": language_label,
                    "dataset_config": lang_config,
                    "source": dataset_source,
                    "split": split,
                    "kind": "short",
                    "source_ids": [item["source_id"]],
                    "text": item["text"],
                }
            )

        for medium_index, size in enumerate(medium_sizes):
            chunk = transcripts[cursor : cursor + size]
            cursor += size
            records.append(
                {
                    "id": f"{language_label.lower()}_medium_{medium_index:03d}",
                    "language": language_label,
                    "dataset_config": lang_config,
                    "source": dataset_source,
                    "split": split,
                    "kind": "medium",
                    "source_ids": [item["source_id"] for item in chunk],
                    "text": " ".join(item["text"] for item in chunk),
                }
            )

    return records


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def summarize(records: list[dict[str, object]], *, dataset_source: str, split: str) -> dict:
    by_language = Counter(str(record["language"]) for record in records)
    by_kind = Counter(str(record["kind"]) for record in records)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Track B Voxtral W4A16 decoder GPTQ text calibration",
        "dataset_source": dataset_source,
        "split": split,
        "total_records": len(records),
        "languages": [label for label, _config in SUPPORTED_LANGUAGES],
        "dataset_configs": {label: config for label, config in SUPPORTED_LANGUAGES},
        "records_by_language": dict(sorted(by_language.items())),
        "records_by_kind": dict(sorted(by_kind.items())),
        "min_records_per_language": min(by_language.values()),
        "max_records_per_language": max(by_language.values()),
        "schema": {
            "id": "stable calibration record id",
            "language": "one of EN, ZH, HI, ES, AR, FR, PT, RU, DE, JA, KO, IT, NL",
            "dataset_config": "FLEURS-style dataset language config",
            "source": "dataset source name",
            "split": "dataset split",
            "kind": "short or medium",
            "source_ids": "source transcript ids included in this record",
            "text": "text passed to tokenizer/collator for calibration",
        },
    }


def main() -> int:
    args = parse_args()
    records = build_records(
        dataset_source=args.dataset_source,
        split=args.split,
        total_samples=args.total_samples,
    )

    out_path = Path(args.out)
    summary_path = Path(args.summary_out)
    write_jsonl(out_path, records)

    summary = summarize(records, dataset_source=args.dataset_source, split=args.split)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(records)} records to {out_path}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
