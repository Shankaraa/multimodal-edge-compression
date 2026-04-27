from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HIT_RATE_PATTERN = re.compile(r"Prefix cache hit rate:\s*([0-9.]+)%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract vLLM prefix-cache evidence from server logs."
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="Log file to parse. May be repeated.",
    )
    parser.add_argument(
        "--all-logs",
        action="store_true",
        help="Parse all logs/*.log files.",
    )
    parser.add_argument(
        "--out",
        default="reports/prefix_cache_log_measurement.json",
        help="Path for the JSON evidence report.",
    )
    return parser.parse_args()


def selected_logs(args: argparse.Namespace) -> list[Path]:
    paths = [Path(item) for item in args.log]
    if args.all_logs:
        paths.extend(sorted(Path("logs").glob("*.log")))
    if not paths:
        paths.append(Path("logs/fp8_round1_kvprefix_validate_triton_benchmark_server.log"))
    return sorted(dict.fromkeys(paths))


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    hit_rates = [float(match.group(1)) for match in HIT_RATE_PATTERN.finditer(text)]
    enabled = "enable_prefix_caching=True" in text or "--enable-prefix-caching" in text
    return {
        "path": str(path),
        "exists": True,
        "enable_prefix_caching_seen": enabled,
        "hit_rate_samples": len(hit_rates),
        "first_hit_rate_percent": hit_rates[0] if hit_rates else None,
        "last_hit_rate_percent": hit_rates[-1] if hit_rates else None,
        "max_hit_rate_percent": max(hit_rates) if hit_rates else None,
        "positive_hit_rate_samples": sum(1 for value in hit_rates if value > 0.0),
    }


def main() -> int:
    args = parse_args()
    rows = []
    for path in selected_logs(args):
        if not path.exists():
            rows.append({"path": str(path), "exists": False})
            continue
        rows.append(parse_log(path))

    existing = [row for row in rows if row.get("exists")]
    max_hit_rate = max(
        (
            row["max_hit_rate_percent"]
            for row in existing
            if row.get("max_hit_rate_percent") is not None
        ),
        default=None,
    )
    positive_samples = sum(int(row.get("positive_hit_rate_samples", 0)) for row in existing)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "logs_parsed": len(existing),
            "any_prefix_caching_enabled": any(
                bool(row.get("enable_prefix_caching_seen")) for row in existing
            ),
            "max_prefix_cache_hit_rate_percent": max_hit_rate,
            "positive_hit_rate_samples": positive_samples,
            "prefix_kv_seeding_working": bool(max_hit_rate and max_hit_rate > 0.0),
        },
        "logs": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = payload["summary"]
    print(
        "prefix cache: "
        f"enabled_seen={summary['any_prefix_caching_enabled']} "
        f"max_hit_rate={summary['max_prefix_cache_hit_rate_percent']} "
        f"positive_samples={summary['positive_hit_rate_samples']}"
    )
    print(f"Saved prefix-cache evidence to: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
