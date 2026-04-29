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


def value_at_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing path {path!r} at {part!r}")
        current = current[part]
    return current


def assert_close(*, claim_id: str, key: str, expected: Any, actual: Any) -> None:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if abs(float(expected) - float(actual)) <= TOLERANCE:
            return
    elif expected == actual:
        return
    raise AssertionError(f"{claim_id}: {key} expected {expected!r}, got {actual!r}")


def verify_claim(
    *,
    reports_dir: Path,
    claim: dict[str, Any],
) -> int:
    claim_id = claim["id"]
    report = load_json(reports_dir / claim["metric_file"])
    verified = 0

    for path, expected in claim.get("report_values", {}).items():
        actual = value_at_path(report, path)
        assert_close(claim_id=claim_id, key=path, expected=expected, actual=actual)
        verified += 1

    gate = claim.get("gate")
    if gate:
        actual_value = float(value_at_path(report, gate["metric_path"]))
        margin = float(gate["ceiling_percent"]) - actual_value
        assert_close(
            claim_id=claim_id,
            key="gate.margin_pp",
            expected=gate["margin_pp"],
            actual=margin,
        )
        verified += 1

    return verified


def sum_metric(
    *,
    claims_by_id: dict[str, dict[str, Any]],
    reports_by_file: dict[str, dict[str, Any]],
    claim_ids: list[str],
    metric_path: str,
) -> float:
    total = 0.0
    for claim_id in claim_ids:
        claim = claims_by_id[claim_id]
        report = reports_by_file[claim["metric_file"]]
        total += float(value_at_path(report, metric_path))
    return total


def main() -> int:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    claims_path = Path(args.claims)
    claims = load_json(claims_path)

    verified = 0
    claims_by_id = {claim["id"]: claim for claim in claims["claims"]}
    reports_by_file = {
        claim["metric_file"]: load_json(reports_dir / claim["metric_file"])
        for claim in claims["claims"]
    }

    for claim in claims["claims"]:
        verified += verify_claim(reports_dir=reports_dir, claim=claim)

    derived = claims.get("derived_claims", {})
    if derived:
        fp8_total = sum_metric(
            claims_by_id=claims_by_id,
            reports_by_file=reports_by_file,
            claim_ids=derived["fp8_claim_ids"],
            metric_path="evaluation.energy_joules",
        )
        bf16_total = sum_metric(
            claims_by_id=claims_by_id,
            reports_by_file=reports_by_file,
            claim_ids=derived["bf16_claim_ids"],
            metric_path="evaluation.energy_joules",
        )
        reduction = (1.0 - fp8_total / bf16_total) * 100.0
        expected_values = {
            "fp8_total_energy_joules": fp8_total,
            "bf16_total_energy_joules": bf16_total,
            "fp8_vs_bf16_energy_reduction_percent": reduction,
        }
        for key, actual in expected_values.items():
            assert_close(
                claim_id="derived_claims",
                key=key,
                expected=derived[key],
                actual=actual,
            )
            verified += 1

    print(f"Verified {verified} claimed values against {claims_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
