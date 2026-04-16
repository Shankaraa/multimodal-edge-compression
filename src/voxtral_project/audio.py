from __future__ import annotations

import base64
import io
import json
import mimetypes
import math
from pathlib import Path
from typing import Any


def audio_array_to_wav_bytes(audio_array: Any, sample_rate: int) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    return buffer.getvalue()


def prepare_audio_array_for_transcription(
    audio_array: Any,
    sample_rate: int,
    *,
    quiet_peak_threshold: float = 0.01,
    target_peak: float = 0.02,
    max_gain: float = 8.0,
) -> tuple[Any, dict[str, float | bool]]:
    import numpy as np

    prepared = np.asarray(audio_array, dtype=np.float32)
    if prepared.ndim > 1:
        prepared = np.squeeze(prepared)

    if prepared.size == 0:
        return prepared, {
            "duration_seconds": 0.0,
            "rms_before": 0.0,
            "rms_after": 0.0,
            "peak_abs_before": 0.0,
            "peak_abs_after": 0.0,
            "gain_applied": 1.0,
            "quiet_audio_boosted": False,
        }

    prepared = prepared.copy()
    peak_before = float(np.max(np.abs(prepared)))
    rms_before = float(math.sqrt(float(np.mean(np.square(prepared.astype(np.float64))))))
    duration_seconds = float(prepared.size / sample_rate) if sample_rate else 0.0

    gain = 1.0
    boosted = False
    if 0.0 < peak_before < quiet_peak_threshold:
        gain = min(max_gain, target_peak / peak_before)
        if gain > 1.0:
            prepared = np.clip(prepared * gain, -1.0, 1.0)
            boosted = True

    peak_after = float(np.max(np.abs(prepared)))
    rms_after = float(math.sqrt(float(np.mean(np.square(prepared.astype(np.float64))))))

    return prepared, {
        "duration_seconds": duration_seconds,
        "rms_before": rms_before,
        "rms_after": rms_after,
        "peak_abs_before": peak_before,
        "peak_abs_after": peak_after,
        "gain_applied": gain,
        "quiet_audio_boosted": boosted,
    }


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
