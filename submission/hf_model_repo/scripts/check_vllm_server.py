from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for a vLLM server and print the exposed models.")
    parser.add_argument("--base-url", default="http://localhost:8080/v1", help="Server base URL.")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds to wait before failing.")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds.")
    return parser.parse_args()


def main() -> int:
    from voxtral_project.api import wait_for_server_ready

    args = parse_args()
    models = wait_for_server_ready(
        base_url=args.base_url,
        timeout=args.timeout,
        interval=args.interval,
    )

    print(f"Server ready at: {args.base_url}")
    if not models:
        print("No models were returned by /v1/models.")
        return 0

    print("Models:")
    for model in models:
        print(f"- {model.get('id', '<unknown>')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
