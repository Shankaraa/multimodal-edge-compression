from __future__ import annotations

import base64
import io
import json
import mimetypes
from pathlib import Path
from typing import Any


def audio_array_to_wav_bytes(audio_array: Any, sample_rate: int) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    return buffer.getvalue()


def guess_audio_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "audio/wav"


def encode_bytes_as_data_url(data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def encode_file_as_data_url(path: Path) -> str:
    mime_type = guess_audio_mime_type(path)
    return encode_bytes_as_data_url(path.read_bytes(), mime_type)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
