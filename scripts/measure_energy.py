from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure energy for an arbitrary command.")
    parser.add_argument(
        "--report",
        required=True,
        help="Path to the JSON report to write.",
    )
    parser.add_argument(
        "--measure-power-secs",
        type=int,
        default=1,
        help="Sampling interval for CodeCarbon.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute. Put `--` before the command.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from codecarbon import EmissionsTracker

    from voxtral_project.audio import write_json

    command = args.command

    if command and command[0] == "--":
        command = command[1:]

    if not command:
        raise SystemExit("No command provided. Example: -- python scripts/evaluate_fleurs.py --lang en_us")

    tracker = EmissionsTracker(
        measure_power_secs=args.measure_power_secs,
        tracking_mode="process",
        log_level="warning",
    )

    start = time.perf_counter()
    tracker.start()
    completed = subprocess.run(command, check=False)
    emissions = tracker.stop()
    elapsed = time.perf_counter() - start

    total_energy = getattr(tracker, "_total_energy", None)
    joules = None
    if total_energy is not None:
        joules = total_energy.kWh * 3_600_000.0

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": command,
        "return_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "energy_joules": joules,
        "emissions_kg": emissions,
    }

    write_json(Path(args.report), payload)
    print(f"Energy report written to: {Path(args.report).resolve()}")
    if joules is not None:
        print(f"Energy: {joules:.2f} J")
    if emissions is not None:
        print(f"Emissions: {emissions:.6f} kg CO2eq")

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
