#!/usr/bin/env python3
"""Build Track B calibration v3 with stronger clean HI FLEURS-train coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


FLEURS_COUNTS = {
    "EN": ("en_us", 20),
    "FR": ("fr_fr", 20),
    "HI": ("hi_in", 61),
    "JA": ("ja_jp", 20),
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
        default="data/calibration/track_b_multilingual_text_256_v3_hi61_fleurs_train.jsonl",
    )
    parser.add_argument("--candidate-pool", type=int, default=240)
    return parser.parse_args()


def load_base_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_fleurs_rows(
    language: str,
    fleurs_code: str,
    count: int,
    candidate_pool: int,
) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(
        "google/fleurs",
        fleurs_code,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    candidates: list[dict] = []
    for sample in dataset:
        text = sample.get("transcription") or sample.get("raw_transcription")
        if not text:
            continue
        text = " ".join(str(text).split())
        if not text:
            continue
        candidates.append(
            {
                "language": language,
                "kind": "fleurs_train",
                "source": f"google/fleurs:{fleurs_code}:train",
                "sample_id": str(sample.get("id", "")),
                "text": text,
            }
        )
        if len(candidates) >= candidate_pool:
            break

    if len(candidates) < count:
        raise RuntimeError(
            f"Expected at least {count} FLEURS rows for {language}, loaded {len(candidates)}"
        )

    candidates.sort(key=lambda row: len(row["text"]), reverse=True)
    long_half = candidates[: max(count // 2, 1)]
    remaining = [row for row in candidates if row not in long_half]
    mixed = []
    mixed.extend(long_half)
    mixed.extend(remaining[:: max(len(remaining) // max(count - len(long_half), 1), 1)])
    return mixed[:count]


def main() -> int:
    args = parse_args()
    base_rows = load_base_rows(Path(args.base_jsonl))
    rows: list[dict] = []

    for language, (fleurs_code, count) in FLEURS_COUNTS.items():
        rows.extend(load_fleurs_rows(language, fleurs_code, count, args.candidate_pool))

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

    output_path = Path(args.output_jsonl)
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
                "min_chars": min(len(row["text"]) for row in rows),
                "max_chars": max(len(row["text"]) for row in rows),
                "avg_chars": round(sum(len(row["text"]) for row in rows) / len(rows), 1),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
