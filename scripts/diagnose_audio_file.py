from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voxtral_project.audio import (  # noqa: E402
    audio_array_to_wav_bytes,
    prepare_audio_array_for_transcription,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and optionally prepare a WAV file.")
    parser.add_argument("audio_file", help="Input WAV file.")
    parser.add_argument("--prepared-out", default=None, help="Optional prepared WAV output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio_file)
    sample_rate, data = wavfile.read(audio_path)

    array = data.astype("float32")
    if np.issubdtype(data.dtype, np.integer):
        array = array / float(np.iinfo(data.dtype).max)

    prepared, diagnostics = prepare_audio_array_for_transcription(array, sample_rate)

    if args.prepared_out:
        out_path = Path(args.prepared_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio_array_to_wav_bytes(prepared, sample_rate))
    else:
        out_path = None

    stats = {
        "audio_file": str(audio_path),
        "sample_rate": int(sample_rate),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "duration_seconds": float(array.shape[0] / sample_rate) if sample_rate else 0.0,
        "peak_abs": float(np.max(np.abs(array))) if array.size else 0.0,
        "rms": float(np.sqrt(np.mean(array.astype("float64") ** 2))) if array.size else 0.0,
        "prepared_out": str(out_path) if out_path else None,
        "prepare_diagnostics": diagnostics,
    }
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
