from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    from voxtral_project.api import DEFAULT_PROMPT

    parser = argparse.ArgumentParser(description="Transcribe one audio file via the vLLM API.")
    parser.add_argument("audio_file", help="Path to a local audio file.")
    parser.add_argument("--base-url", default="http://localhost:8080/v1", help="Server base URL.")
    parser.add_argument("--model", default="voxtral-realtime", help="Model name exposed by the server.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction prompt.")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max output tokens.")
    return parser.parse_args()


def main() -> int:
    from voxtral_project.api import transcribe_audio_bytes
    from voxtral_project.audio import guess_audio_mime_type

    args = parse_args()
    audio_path = Path(args.audio_file)
    transcript = transcribe_audio_bytes(
        base_url=args.base_url,
        model=args.model,
        audio_bytes=audio_path.read_bytes(),
        mime_type=guess_audio_mime_type(audio_path),
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    print(transcript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
