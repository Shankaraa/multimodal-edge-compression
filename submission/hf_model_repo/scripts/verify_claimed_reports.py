from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that claimed_results.json matches the committed report JSON files."
    )
    parser.add_argument("--reports-dir", default="reports", help="Directory containing report JSON files.")
    parser.add_argument(
        "--claims",
        default="reports/claimed_results.json",
        help="Claim manifest to verify.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_result(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Expected report to contain a non-empty `results` list.")
    return results[0]


def metric_value(metric_report: dict[str, Any], energy_report: dict[str, Any] | None, key: str) -> Any:
    if key in {"elapsed_seconds", "energy_joules"}:
        if energy_report is not None:
            return energy_report[key]
        evaluation = metric_report.get("evaluation", {})
        return evaluation[key]

    if "evaluation" in metric_report:
        return metric_report["evaluation"][key]

    return first_result(metric_report)[key]


def assert_close(*, claim_id: str, key: str, expected: Any, actual: Any) -> None:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if abs(float(expected) - float(actual)) <= TOLERANCE:
            return
    elif expected == actual:
        return
    raise AssertionError(
        f"{claim_id}: {key} expected {expected!r}, got {actual!r}"
    )


def main() -> int:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    claims_path = Path(args.claims)
    claims = load_json(claims_path)

    verified = 0
    loaded_claim_metrics: dict[str, dict[str, Any]] = {}
    for claim in claims["claims"]:
        claim_id = claim["id"]
        metric_report = load_json(reports_dir / claim["metric_file"])
        energy_file = claim.get("energy_file")
        energy_report = load_json(reports_dir / energy_file) if energy_file else None

        for key, expected in claim["metrics"].items():
            actual = metric_value(metric_report, energy_report, key)
            assert_close(claim_id=claim_id, key=key, expected=expected, actual=actual)
            verified += 1
        loaded_claim_metrics[claim_id] = claim["metrics"]

    if "derived_claims" in claims:
        bf16 = loaded_claim_metrics["bf16_en_us_limit20_reference"]
        fp8 = loaded_claim_metrics["fp8_en_us_limit20_core"]
        derived = {
            "fp8_vs_bf16_elapsed_reduction_percent": (
                1.0 - fp8["elapsed_seconds"] / bf16["elapsed_seconds"]
            )
            * 100.0,
            "fp8_vs_bf16_energy_reduction_percent": (
                1.0 - fp8["energy_joules"] / bf16["energy_joules"]
            )
            * 100.0,
        }
        for key, actual in derived.items():
            assert_close(
                claim_id="derived_claims",
                key=key,
                expected=claims["derived_claims"][key],
                actual=actual,
            )
            verified += 1

    print(f"Verified {verified} claimed metrics against {claims_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
