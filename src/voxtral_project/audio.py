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


def analyze_audio_activity(
    audio_array: Any,
    sample_rate: int,
    *,
    frame_ms: float = 80.0,
    active_peak_threshold: float = 0.01,
    active_rms_threshold: float = 0.003,
) -> dict[str, float | int]:
    import numpy as np

    prepared = np.asarray(audio_array, dtype=np.float32)
    if prepared.ndim > 1:
        prepared = np.squeeze(prepared)

    if prepared.size == 0 or sample_rate <= 0:
        return {
            "frame_ms": float(frame_ms),
            "frame_count": 0,
            "active_frame_count": 0,
            "silent_frame_count": 0,
            "active_frame_ratio": 0.0,
            "silent_frame_ratio": 0.0,
            "leading_silent_frames": 0,
            "trailing_silent_frames": 0,
            "leading_silent_seconds": 0.0,
            "trailing_silent_seconds": 0.0,
            "longest_silent_run_frames": 0,
            "longest_silent_run_seconds": 0.0,
            "longest_active_run_frames": 0,
            "longest_active_run_seconds": 0.0,
            "active_span_count": 0,
            "peak_abs": 0.0,
            "rms": 0.0,
        }

    frame_size = max(1, int(round(sample_rate * (frame_ms / 1000.0))))
    frame_count = int(math.ceil(prepared.size / frame_size))
    padded = np.pad(prepared, (0, frame_count * frame_size - prepared.size))
    frames = padded.reshape(frame_count, frame_size)

    frame_peaks = np.max(np.abs(frames), axis=1)
    frame_rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
    active_mask = (frame_peaks >= active_peak_threshold) | (frame_rms >= active_rms_threshold)

    active_frame_count = int(np.sum(active_mask))
    silent_frame_count = int(frame_count - active_frame_count)

    if active_frame_count == 0:
        leading_silent_frames = frame_count
        trailing_silent_frames = frame_count
    else:
        first_active = int(np.argmax(active_mask))
        last_active = int(frame_count - 1 - np.argmax(active_mask[::-1]))
        leading_silent_frames = first_active
        trailing_silent_frames = frame_count - 1 - last_active

    longest_silent_run = 0
    longest_active_run = 0
    active_span_count = 0
    current_run = 0
    current_value = None
    for is_active in active_mask.tolist():
        if current_value is None or is_active != current_value:
            if current_value is True:
                longest_active_run = max(longest_active_run, current_run)
            elif current_value is False:
                longest_silent_run = max(longest_silent_run, current_run)

            current_value = is_active
            current_run = 1
            if is_active:
                active_span_count += 1
        else:
            current_run += 1

    if current_value is True:
        longest_active_run = max(longest_active_run, current_run)
    elif current_value is False:
        longest_silent_run = max(longest_silent_run, current_run)

    return {
        "frame_ms": float(frame_ms),
        "frame_count": frame_count,
        "active_frame_count": active_frame_count,
        "silent_frame_count": silent_frame_count,
        "active_frame_ratio": float(active_frame_count / frame_count),
        "silent_frame_ratio": float(silent_frame_count / frame_count),
        "leading_silent_frames": int(leading_silent_frames),
        "trailing_silent_frames": int(trailing_silent_frames),
        "leading_silent_seconds": float(leading_silent_frames * frame_size / sample_rate),
        "trailing_silent_seconds": float(trailing_silent_frames * frame_size / sample_rate),
        "longest_silent_run_frames": int(longest_silent_run),
        "longest_silent_run_seconds": float(longest_silent_run * frame_size / sample_rate),
        "longest_active_run_frames": int(longest_active_run),
        "longest_active_run_seconds": float(longest_active_run * frame_size / sample_rate),
        "active_span_count": int(active_span_count),
        "peak_abs": float(np.max(np.abs(prepared))),
        "rms": float(math.sqrt(float(np.mean(np.square(prepared.astype(np.float64)))))),
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
