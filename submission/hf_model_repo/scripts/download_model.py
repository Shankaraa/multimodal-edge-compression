from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REPO_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Voxtral model locally.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id.")
    parser.add_argument(
        "--local-dir",
        default="models/voxtral-realtime",
        help="Local directory for the snapshot.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision or commit hash.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from huggingface_hub import snapshot_download

    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
    )

    print(f"Model downloaded to: {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
