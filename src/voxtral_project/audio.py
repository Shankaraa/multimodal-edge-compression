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


def _as_mono_float32_audio(audio_array: Any) -> Any:
    import numpy as np

    prepared = np.asarray(audio_array, dtype=np.float32)
    if prepared.ndim > 1:
        prepared = np.squeeze(prepared)
    return prepared


def _frame_activity_mask(
    audio_array: Any,
    sample_rate: int,
    *,
    frame_ms: float,
    active_peak_threshold: float,
    active_rms_threshold: float,
) -> tuple[Any, int]:
    import numpy as np

    prepared = _as_mono_float32_audio(audio_array)
    if prepared.size == 0 or sample_rate <= 0:
        return np.zeros(0, dtype=bool), 0

    frame_size = max(1, int(round(sample_rate * (frame_ms / 1000.0))))
    frame_count = int(math.ceil(prepared.size / frame_size))
    padded = np.pad(prepared, (0, frame_count * frame_size - prepared.size))
    frames = padded.reshape(frame_count, frame_size)

    frame_peaks = np.max(np.abs(frames), axis=1)
    frame_rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
    active_mask = (frame_peaks >= active_peak_threshold) | (frame_rms >= active_rms_threshold)
    return active_mask, frame_size


def gate_audio_by_activity(
    audio_array: Any,
    sample_rate: int,
    *,
    frame_ms: float = 80.0,
    active_peak_threshold: float = 0.01,
    active_rms_threshold: float = 0.003,
    preserve_leading_silence_ms: float = 160.0,
    preserve_trailing_silence_ms: float = 160.0,
    compress_internal_silence_to_ms: float | None = None,
    min_internal_silence_run_ms: float = 640.0,
) -> tuple[Any, dict[str, float | int | bool | None]]:
    import numpy as np

    prepared = _as_mono_float32_audio(audio_array)
    duration_seconds = float(prepared.size / sample_rate) if sample_rate and prepared.size else 0.0
    diagnostics: dict[str, float | int | bool | None] = {
        "speech_gating_applied": True,
        "speech_gating_changed_audio": False,
        "speech_gating_duration_before_seconds": duration_seconds,
        "speech_gating_duration_after_seconds": duration_seconds,
        "speech_gating_seconds_removed": 0.0,
        "speech_gating_fraction_removed": 0.0,
        "speech_gating_leading_trimmed_seconds": 0.0,
        "speech_gating_trailing_trimmed_seconds": 0.0,
        "speech_gating_internal_trimmed_seconds": 0.0,
        "speech_gating_internal_spans_compressed": 0,
        "speech_gating_frame_ms": float(frame_ms),
        "speech_gating_peak_threshold": float(active_peak_threshold),
        "speech_gating_rms_threshold": float(active_rms_threshold),
        "speech_gating_preserve_leading_silence_ms": float(preserve_leading_silence_ms),
        "speech_gating_preserve_trailing_silence_ms": float(preserve_trailing_silence_ms),
        "speech_gating_compress_internal_silence_to_ms": (
            float(compress_internal_silence_to_ms)
            if compress_internal_silence_to_ms is not None
            else None
        ),
        "speech_gating_min_internal_silence_run_ms": float(min_internal_silence_run_ms),
    }

    if prepared.size == 0 or sample_rate <= 0:
        return prepared, diagnostics

    active_mask, frame_size = _frame_activity_mask(
        prepared,
        sample_rate,
        frame_ms=frame_ms,
        active_peak_threshold=active_peak_threshold,
        active_rms_threshold=active_rms_threshold,
    )

    if active_mask.size == 0 or not bool(np.any(active_mask)):
        # Stay conservative if the thresholds classify the whole clip as silent.
        return prepared, diagnostics

    preserve_leading_frames = max(
        0, int(math.ceil(preserve_leading_silence_ms / frame_ms))
    )
    preserve_trailing_frames = max(
        0, int(math.ceil(preserve_trailing_silence_ms / frame_ms))
    )
    compress_internal_frames = (
        max(0, int(math.ceil(compress_internal_silence_to_ms / frame_ms)))
        if compress_internal_silence_to_ms is not None
        else None
    )
    min_internal_run_frames = max(1, int(math.ceil(min_internal_silence_run_ms / frame_ms)))

    spans: list[tuple[bool, int, int]] = []
    run_start = 0
    current_value = bool(active_mask[0])
    for frame_index in range(1, int(active_mask.size)):
        next_value = bool(active_mask[frame_index])
        if next_value != current_value:
            spans.append((current_value, run_start, frame_index))
            run_start = frame_index
            current_value = next_value
    spans.append((current_value, run_start, int(active_mask.size)))

    keep_segments: list[tuple[int, int]] = []
    leading_removed_samples = 0
    trailing_removed_samples = 0
    internal_removed_samples = 0
    internal_spans_compressed = 0
    total_samples = int(prepared.size)

    for span_index, (is_active, start_frame, end_frame) in enumerate(spans):
        span_start = min(total_samples, start_frame * frame_size)
        span_end = min(total_samples, end_frame * frame_size)
        if span_end <= span_start:
            continue

        if is_active:
            keep_segments.append((span_start, span_end))
            continue

        span_frame_count = end_frame - start_frame
        is_leading = span_index == 0
        is_trailing = span_index == len(spans) - 1

        if is_leading:
            keep_frames = min(span_frame_count, preserve_leading_frames)
            keep_start = max(span_start, span_end - keep_frames * frame_size)
            if keep_start < span_end:
                keep_segments.append((keep_start, span_end))
            leading_removed_samples += keep_start - span_start
            continue

        if is_trailing:
            keep_frames = min(span_frame_count, preserve_trailing_frames)
            keep_end = min(span_end, span_start + keep_frames * frame_size)
            if span_start < keep_end:
                keep_segments.append((span_start, keep_end))
            trailing_removed_samples += span_end - keep_end
            continue

        if (
            compress_internal_frames is None
            or span_frame_count < min_internal_run_frames
            or compress_internal_frames >= span_frame_count
        ):
            keep_segments.append((span_start, span_end))
            continue

        internal_spans_compressed += 1
        first_frames = compress_internal_frames // 2
        last_frames = compress_internal_frames - first_frames
        first_end = min(span_end, span_start + first_frames * frame_size)
        last_start = max(span_start, span_end - last_frames * frame_size)

        if first_end > span_start:
            keep_segments.append((span_start, first_end))
        if last_start < span_end:
            if keep_segments and keep_segments[-1][1] >= last_start:
                keep_segments[-1] = (keep_segments[-1][0], span_end)
            else:
                keep_segments.append((last_start, span_end))

        internal_removed_samples += max(0, last_start - first_end)

    if not keep_segments:
        return prepared, diagnostics

    merged_segments: list[tuple[int, int]] = []
    for start, end in sorted(keep_segments):
        if not merged_segments or start > merged_segments[-1][1]:
            merged_segments.append((start, end))
        else:
            merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], end))

    gated = np.concatenate([prepared[start:end] for start, end in merged_segments])
    kept_samples = int(sum(end - start for start, end in merged_segments))
    removed_samples = max(0, total_samples - kept_samples)
    gated_duration_seconds = float(gated.size / sample_rate)

    diagnostics.update(
        {
            "speech_gating_changed_audio": bool(removed_samples > 0),
            "speech_gating_duration_after_seconds": gated_duration_seconds,
            "speech_gating_seconds_removed": float(removed_samples / sample_rate),
            "speech_gating_fraction_removed": float(removed_samples / total_samples)
            if total_samples
            else 0.0,
            "speech_gating_leading_trimmed_seconds": float(leading_removed_samples / sample_rate),
            "speech_gating_trailing_trimmed_seconds": float(trailing_removed_samples / sample_rate),
            "speech_gating_internal_trimmed_seconds": float(internal_removed_samples / sample_rate),
            "speech_gating_internal_spans_compressed": int(internal_spans_compressed),
        }
    )
    return gated, diagnostics


def prepare_audio_array_for_transcription(
    audio_array: Any,
    sample_rate: int,
    *,
    quiet_peak_threshold: float = 0.01,
    target_peak: float = 0.02,
    max_gain: float = 8.0,
    gate_silence: bool = False,
    gate_frame_ms: float = 80.0,
    gate_peak_threshold: float = 0.01,
    gate_rms_threshold: float = 0.003,
    preserve_leading_silence_ms: float = 160.0,
    preserve_trailing_silence_ms: float = 160.0,
    compress_internal_silence_to_ms: float | None = None,
    min_internal_silence_run_ms: float = 640.0,
) -> tuple[Any, dict[str, float | bool]]:
    import numpy as np

    prepared = _as_mono_float32_audio(audio_array)

    if prepared.size == 0:
        return prepared, {
            "duration_seconds": 0.0,
            "rms_before": 0.0,
            "rms_after": 0.0,
            "peak_abs_before": 0.0,
            "peak_abs_after": 0.0,
            "gain_applied": 1.0,
            "quiet_audio_boosted": False,
            "speech_gating_applied": bool(gate_silence),
            "speech_gating_changed_audio": False,
            "speech_gating_duration_before_seconds": 0.0,
            "speech_gating_duration_after_seconds": 0.0,
            "speech_gating_seconds_removed": 0.0,
            "speech_gating_fraction_removed": 0.0,
            "speech_gating_leading_trimmed_seconds": 0.0,
            "speech_gating_trailing_trimmed_seconds": 0.0,
            "speech_gating_internal_trimmed_seconds": 0.0,
            "speech_gating_internal_spans_compressed": 0,
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

    gating_diagnostics: dict[str, float | int | bool | None]
    if gate_silence:
        prepared, gating_diagnostics = gate_audio_by_activity(
            prepared,
            sample_rate,
            frame_ms=gate_frame_ms,
            active_peak_threshold=gate_peak_threshold,
            active_rms_threshold=gate_rms_threshold,
            preserve_leading_silence_ms=preserve_leading_silence_ms,
            preserve_trailing_silence_ms=preserve_trailing_silence_ms,
            compress_internal_silence_to_ms=compress_internal_silence_to_ms,
            min_internal_silence_run_ms=min_internal_silence_run_ms,
        )
    else:
        gating_diagnostics = {
            "speech_gating_applied": False,
            "speech_gating_changed_audio": False,
            "speech_gating_duration_before_seconds": duration_seconds,
            "speech_gating_duration_after_seconds": duration_seconds,
            "speech_gating_seconds_removed": 0.0,
            "speech_gating_fraction_removed": 0.0,
            "speech_gating_leading_trimmed_seconds": 0.0,
            "speech_gating_trailing_trimmed_seconds": 0.0,
            "speech_gating_internal_trimmed_seconds": 0.0,
            "speech_gating_internal_spans_compressed": 0,
            "speech_gating_frame_ms": float(gate_frame_ms),
            "speech_gating_peak_threshold": float(gate_peak_threshold),
            "speech_gating_rms_threshold": float(gate_rms_threshold),
            "speech_gating_preserve_leading_silence_ms": float(preserve_leading_silence_ms),
            "speech_gating_preserve_trailing_silence_ms": float(preserve_trailing_silence_ms),
            "speech_gating_compress_internal_silence_to_ms": (
                float(compress_internal_silence_to_ms)
                if compress_internal_silence_to_ms is not None
                else None
            ),
            "speech_gating_min_internal_silence_run_ms": float(min_internal_silence_run_ms),
        }

    peak_after = float(np.max(np.abs(prepared)))
    rms_after = float(math.sqrt(float(np.mean(np.square(prepared.astype(np.float64))))))

    diagnostics: dict[str, float | bool | int | None] = {
        "duration_seconds": duration_seconds,
        "rms_before": rms_before,
        "rms_after": rms_after,
        "peak_abs_before": peak_before,
        "peak_abs_after": peak_after,
        "gain_applied": gain,
        "quiet_audio_boosted": boosted,
    }
    diagnostics.update(gating_diagnostics)
    return prepared, diagnostics


def analyze_audio_activity(
    audio_array: Any,
    sample_rate: int,
    *,
    frame_ms: float = 80.0,
    active_peak_threshold: float = 0.01,
    active_rms_threshold: float = 0.003,
) -> dict[str, float | int]:
    import numpy as np

    prepared = _as_mono_float32_audio(audio_array)

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

    active_mask, frame_size = _frame_activity_mask(
        prepared,
        sample_rate,
        frame_ms=frame_ms,
        active_peak_threshold=active_peak_threshold,
        active_rms_threshold=active_rms_threshold,
    )
    frame_count = int(active_mask.size)

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
