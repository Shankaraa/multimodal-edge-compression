from __future__ import annotations

import argparse
import json
import os
import site
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


def build_launch_env() -> dict[str, str]:
    env = os.environ.copy()

    if os.name != "posix":
        return env

    candidate_dirs: list[str] = []
    seen: set[str] = set()

    for root in site.getsitepackages():
        site_root = Path(root)
        direct_paths = [
            site_root / "torch" / "lib",
            site_root / "nvidia" / "cu12" / "lib",
            site_root / "nvidia" / "cu13" / "lib",
        ]
        nvidia_root = site_root / "nvidia"
        if nvidia_root.is_dir():
            direct_paths.extend(sorted(nvidia_root.glob("*/lib")))

        for path in direct_paths:
            if not path.is_dir():
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            candidate_dirs.append(resolved)

    existing = [entry for entry in env.get("LD_LIBRARY_PATH", "").split(":") if entry]
    merged: list[str] = []
    for entry in candidate_dirs + existing:
        if entry in merged:
            continue
        merged.append(entry)

    if merged:
        env["LD_LIBRARY_PATH"] = ":".join(merged)

    return env


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

    completed = subprocess.run(command, check=False, env=build_launch_env())
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
