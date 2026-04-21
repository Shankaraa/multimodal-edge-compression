from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    from voxtral_project.api import DEFAULT_PROMPT

    parser = argparse.ArgumentParser(description="Run a smoke-test transcription on a known Hugging Face sample audio file.")
    parser.add_argument("--base-url", default="http://localhost:8080/v1", help="Server base URL.")
    parser.add_argument("--model", default="voxtral-realtime", help="Model name exposed by the server.")
    parser.add_argument(
        "--repo-id",
        default="patrickvonplaten/audio_samples",
        help="Hugging Face dataset repo containing the sample audio.",
    )
    parser.add_argument(
        "--filename",
        default="bcn_weather.mp3",
        help="Sample audio filename inside the repo.",
    )
    parser.add_argument("--repo-type", default="dataset", help="Hugging Face repo type.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction prompt.")
    parser.add_argument("--language", default=None, help="Optional language hint such as en or fr.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. The official recommendation is 0.0.",
    )
    parser.add_argument("--max-tokens", type=int, default=500, help="Max output tokens.")
    parser.add_argument("--out", default=None, help="Optional path to save the transcript.")
    return parser.parse_args()


def main() -> int:
    from huggingface_hub import hf_hub_download

    from voxtral_project.api import transcribe_audio_bytes
    from voxtral_project.audio import guess_audio_mime_type

    args = parse_args()
    sample_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type=args.repo_type,
        )
    )

    transcript = transcribe_audio_bytes(
        base_url=args.base_url,
        model=args.model,
        audio_bytes=sample_path.read_bytes(),
        mime_type=guess_audio_mime_type(sample_path),
        prompt=args.prompt,
        language=args.language,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"Sample audio: {sample_path}")
    print("Transcript:")
    print(transcript)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(transcript, encoding="utf-8")
        print(f"Saved transcript to: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
