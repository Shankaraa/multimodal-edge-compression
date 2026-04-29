#!/usr/bin/env python3
"""Build Track B calibration v2 with real FLEURS-heavy HI coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


FLEURS_COUNTS = {
    "EN": ("en_us", 24),
    "FR": ("fr_fr", 24),
    "HI": ("hi_in", 45),
    "JA": ("ja_jp", 28),
}

SUPPORTED_LANGS = [
    "EN",
    "ZH",
    "HI",
    "ES",
    "AR",
    "FR",
    "PT",
    "RU",
    "DE",
    "JA",
    "KO",
    "IT",
    "NL",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-jsonl",
        default="data/calibration/track_b_multilingual_text_256.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/calibration/track_b_multilingual_text_256_v2_fleurs_hi.jsonl",
    )
    return parser.parse_args()


def load_base_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_fleurs_rows(language: str, fleurs_code: str, count: int) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(
        "google/fleurs",
        fleurs_code,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    rows: list[dict] = []
    for sample in dataset:
        text = sample.get("transcription") or sample.get("raw_transcription")
        if not text:
            continue
        text = " ".join(str(text).split())
        if not text:
            continue
        rows.append(
            {
                "language": language,
                "kind": "fleurs_train",
                "source": f"google/fleurs:{fleurs_code}:train",
                "sample_id": str(sample.get("id", "")),
                "text": text,
            }
        )
        if len(rows) >= count:
            break

    if len(rows) != count:
        raise RuntimeError(
            f"Expected {count} FLEURS rows for {language}, loaded {len(rows)}"
        )
    return rows


def main() -> int:
    args = parse_args()
    base_path = Path(args.base_jsonl)
    output_path = Path(args.output_jsonl)

    base_rows = load_base_rows(base_path)
    rows: list[dict] = []

    for language, (fleurs_code, count) in FLEURS_COUNTS.items():
        rows.extend(load_fleurs_rows(language, fleurs_code, count))

    existing_by_language: dict[str, list[dict]] = {lang: [] for lang in SUPPORTED_LANGS}
    for row in base_rows:
        language = str(row["language"]).upper()
        if language in existing_by_language:
            existing_by_language[language].append(row)

    for language in SUPPORTED_LANGS:
        if language in FLEURS_COUNTS:
            continue
        rows.extend(existing_by_language[language][:15])

    if len(rows) != 256:
        raise RuntimeError(f"Expected 256 rows, built {len(rows)}")

    counts = Counter(row["language"] for row in rows)
    missing = [lang for lang in SUPPORTED_LANGS if counts[lang] < 15]
    if missing:
        raise RuntimeError(f"Insufficient rows for languages: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "rows": len(rows),
                "counts": dict(counts),
                "sources": dict(Counter(row.get("source", "base") for row in rows)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
