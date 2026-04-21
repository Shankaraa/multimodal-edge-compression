from __future__ import annotations

import argparse

from llmcompressor.entrypoints import model_free_ptq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run llmcompressor model-free PTQ on a local checkpoint stub."
    )
    parser.add_argument("model_stub", help="Directory containing the checkpoint stub.")
    parser.add_argument("save_directory", help="Directory to write the compressed model.")
    parser.add_argument("--scheme", default="FP8_DYNAMIC", help="Compression scheme name.")
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Regex or glob pattern to exclude. Repeat for multiple patterns.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of workers for model-free PTQ.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Target device for the compression run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Running model_free_ptq")
    print(f"  model_stub={args.model_stub}")
    print(f"  save_directory={args.save_directory}")
    print(f"  scheme={args.scheme}")
    print(f"  ignore={args.ignore}")
    print(f"  max_workers={args.max_workers}")
    print(f"  device={args.device}")

    model_free_ptq(
        model_stub=args.model_stub,
        save_directory=args.save_directory,
        scheme=args.scheme,
        ignore=args.ignore,
        max_workers=args.max_workers,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
