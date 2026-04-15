from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch vLLM with a YAML config.")
    parser.add_argument("model_path", help="Path to the local model or model id.")
    parser.add_argument(
        "--config",
        default="configs/vllm/bf16.yaml",
        help="Path to the YAML config.",
    )
    parser.add_argument("--host", default=None, help="Optional host override.")
    parser.add_argument("--port", type=int, default=8080, help="Port for the server.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it.",
    )
    return parser.parse_args()


def flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def append_option(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return

    name = flag_name(key)

    if isinstance(value, bool):
        if value:
            command.append(name)
        return

    if isinstance(value, (dict, list)):
        command.extend([name, json.dumps(value)])
        return

    command.extend([name, str(value)])


def build_command(model_path: str, config: dict[str, Any], host: str | None, port: int) -> list[str]:
    command = ["vllm", "serve", model_path]

    if host:
        command.extend(["--host", host])

    command.extend(["--port", str(port)])

    for key, value in config.items():
        append_option(command, key, value)

    return command


def main() -> int:
    args = parse_args()
    import yaml

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    command = build_command(args.model_path, config, args.host, args.port)
    print("Launching:")
    print(" ".join(command))

    if args.dry_run:
        return 0

    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
