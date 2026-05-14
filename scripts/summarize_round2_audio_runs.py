"""Print a wide comparison table across the saved Round-2 audio-lever runs.

The Round-2 audio lever stacks LUFS-23 + VAD-trim + speech-gate(320/640) on top of
the FP8 vLLM stack. This helper aggregates the per-language reports into one table:
raw WER, normalized WER, empties, sum-of-per-sample latency, audio in / out, trim%.

Usage::

    python scripts/summarize_round2_audio_runs.py
        [--reports-dir reports]
        [--variants <name>=<path> ...]

Each `--variants` arg is repeated, e.g.::

    --variants "FP8 + VAD+gate (no LUFS)=fleurs_fp8_en500_vadgate_nolufs_smoke.json"

When invoked with no `--variants`, the script prints the EN20 + EN500 ablation tables
that are referenced from `docs/round2_audio_lever.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_VARIANTS = [
    # EN20 BF16
    ("BF16 baseline (no LUFS, no VAD)",        "fleurs_bf16_en_us_limit20_quietfix.json"),
    ("BF16 + LUFS-23",                          "fleurs_bf16_en_us_limit20_lufs23_smoke.json"),
    ("BF16 + LUFS + VAD-trim",                  "fleurs_bf16_en_us_limit20_lufs23_vadtrim_smoke.json"),
    ("BF16 + LUFS + VAD + gate(320/640)",       "fleurs_bf16_en_us_limit20_lufs23_vadtrim_gate_smoke.json"),
    ("FP8  baseline (no LUFS, no VAD)",         "fleurs_fp8_en_us_limit20_quietfix.json"),
    ("FP8  + LUFS-23",                          "fleurs_fp8_en_us_limit20_lufs23_smoke.json"),
    ("FP8  + LUFS-23 + VAD-trim",               "fleurs_fp8_en_us_limit20_lufs23_vadtrim_smoke.json"),
    ("FP8  + LUFS-23 + VAD + gate(320/640)",    "fleurs_fp8_en_us_limit20_lufs23_vadtrim_gate_smoke.json"),
    # EN500 ablation
    ("FP8  EN500 + VAD+gate (no LUFS)",         "fleurs_fp8_en500_vadgate_nolufs_smoke.json"),
    ("FP8  EN500 + LUFS-23 + VAD+gate(320/640)","fleurs_fp8_en500_lufs23_vadgate_smoke.json"),
    ("FP8  EN500 + LUFS-23 + VAD+gate(160/320)","fleurs_fp8_en500_lufs23_vadgate160_320_smoke.json"),
]


# 13 Voxtral-supported FLEURS languages and their codes (per
# voxtral_project.dataset_utils.COMMON_VOICE_17_CONFIGS).
VOXTRAL_FLEURS_LANGUAGES = [
    ("ar_eg",       "ar"),
    ("cmn_hans_cn", "zh"),
    ("de_de",       "de"),
    ("en_us",       "en"),
    ("es_419",      "es"),
    ("fr_fr",       "fr"),
    ("hi_in",       "hi"),
    ("it_it",       "it"),
    ("ja_jp",       "ja"),
    ("ko_kr",       "ko"),
    ("nl_nl",       "nl"),
    ("pt_br",       "pt"),
    ("ru_ru",       "ru"),
]


def full_fleurs_variants(tag: str = "lufs23_vadgate160_320") -> list[tuple[str, str]]:
    """Locked-stack reports for all 13 Voxtral-supported FLEURS languages.

    Default tag matches the 2026-05-08 tight-gate Round-2 candidate.
    """
    rows: list[tuple[str, str]] = []
    for lang_code, primary in VOXTRAL_FLEURS_LANGUAGES:
        # EN uses limit=500, others limit=100.
        limit = 500 if lang_code == "en_us" else 100
        if lang_code == "en_us" and tag == "lufs23_vadgate160_320":
            # The EN500 file naming differs slightly because it predates the per-language naming.
            name = f"fleurs_fp8_en500_lufs23_vadgate160_320_smoke.json"
        else:
            name = f"fleurs_fp8_{lang_code}_limit{limit}_{tag}_smoke.json"
        rows.append((f"FP8 {lang_code:>11s} limit={limit:<3d} ({primary})", name))
    return rows


def summarize(reports_dir: Path, variants: list[tuple[str, str]]) -> None:
    rows = []
    for label, name in variants:
        path = reports_dir / name
        if not path.exists():
            rows.append((label, "(no report)"))
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            rows.append((label, f"(read error: {exc!r})"))
            continue
        result = data["results"][0]
        samples = result.get("samples", [])
        sum_lat = sum(s.get("latency_total_seconds", 0) or 0 for s in samples)
        audio_in = sum(s.get("audio_duration_seconds", 0) or 0 for s in samples)
        vad_t = sum(s.get("vad_trim_seconds_removed", 0) or 0 for s in samples)
        gate_t = sum(s.get("speech_gating_seconds_removed", 0) or 0 for s in samples)
        trim_total = vad_t + gate_t
        audio_out = audio_in - trim_total
        trim_pct = (trim_total / audio_in * 100.0) if audio_in else 0.0
        rows.append(
            (
                label,
                {
                    "raw": result.get("wer_percent"),
                    "norm": result.get("wer_normalized_percent"),
                    "empty": result.get("empty_prediction_count"),
                    "sum_lat": sum_lat,
                    "audio_in": audio_in,
                    "audio_out": audio_out,
                    "trim_pct": trim_pct,
                    "samples": len(samples),
                },
            )
        )

    width = max(len(label) for label, _ in rows)
    header = (
        f"{'variant':>{width}}  {'samples':>7}  {'raw':>6}  {'norm':>6}  "
        f"{'empty':>5}  {'sum_lat':>8}  {'audio_in':>9}  {'audio_out':>9}  {'trim%':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, row in rows:
        if isinstance(row, str):
            print(f"{label:>{width}}  {row}")
            continue
        print(
            f"{label:>{width}}  {row['samples']:>7}  "
            f"{row['raw']:>5.2f}%  {row['norm']:>5.2f}%  {row['empty']:>5}  "
            f"{row['sum_lat']:>7.2f}s  {row['audio_in']:>8.2f}s  "
            f"{row['audio_out']:>8.2f}s  {row['trim_pct']:>5.2f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="reports", type=Path)
    parser.add_argument(
        "--variants",
        action="append",
        default=None,
        help="Repeated --variants \"label=path\" (relative to --reports-dir).",
    )
    parser.add_argument(
        "--full-fleurs",
        action="store_true",
        help=(
            "Print the all-13-Voxtral-language tight-gate sweep table instead of the "
            "default ablation table."
        ),
    )
    parser.add_argument(
        "--full-fleurs-tag",
        default="lufs23_vadgate160_320",
        help="Report-name tag for --full-fleurs mode (defaults to the 160/320 tight-gate runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_fleurs:
        variants = full_fleurs_variants(args.full_fleurs_tag)
    elif args.variants is None:
        variants = DEFAULT_VARIANTS
    else:
        variants = []
        for entry in args.variants:
            if "=" not in entry:
                raise ValueError(f"--variants entry must be 'label=path', got {entry!r}")
            label, name = entry.split("=", 1)
            variants.append((label.strip(), name.strip()))
    summarize(args.reports_dir, variants)


if __name__ == "__main__":
    main()
