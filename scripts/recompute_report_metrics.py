from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute transcript quality metrics for an existing evaluation report.",
    )
    parser.add_argument("report", help="Existing JSON evaluation report.")
    parser.add_argument("--out", default=None, help="Optional output path. Defaults to in-place update.")
    return parser.parse_args()


def main() -> int:
    from voxtral_project.audio import write_json
    from voxtral_project.text import summarize_transcript_metrics

    args = parse_args()
    report_path = Path(args.report)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    for result in payload.get("results", []):
        samples = result.get("samples", [])
        references = [str(sample.get("reference", "")) for sample in samples]
        predictions = [str(sample.get("prediction", "")) for sample in samples]
        metrics = summarize_transcript_metrics(
            references=references,
            predictions=predictions,
            lang_code=result.get("language"),
        )
        result.update(metrics)

    output_path = Path(args.out) if args.out else report_path
    write_json(output_path, payload)
    print(f"Updated report metrics: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
