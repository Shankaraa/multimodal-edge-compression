from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from voxtral_project.text import normalize_asr_text, word_error_rate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_head_sha(project_root: Path) -> str | None:
    head_path = project_root / ".git" / "HEAD"
    if not head_path.exists():
        return None

    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head

    ref = head[5:]
    ref_path = project_root / ".git" / ref
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()

    packed_refs = project_root / ".git" / "packed-refs"
    if not packed_refs.exists():
        return None

    for line in packed_refs.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and line.endswith(f" {ref}"):
            return line.split()[0]
    return None


def normalization_version(project_root: Path) -> str:
    return file_sha256(project_root / "src" / "voxtral_project" / "text.py")


def ensure_config_hash_in_filename(path: Path, config_hash: str | None) -> Path:
    if not config_hash or config_hash in path.name:
        return path
    return path.with_name(f"{path.stem}_cfg{config_hash}{path.suffix}")


def _ci_pair(interval: dict[str, Any] | None) -> list[float | None]:
    if not interval:
        return [None, None]
    return [interval.get("low"), interval.get("high")]


def _ci_percent_pair(interval: dict[str, Any] | None) -> list[float | None]:
    if not interval:
        return [None, None]
    return [interval.get("low_percent"), interval.get("high_percent")]


def _sample_wer(sample: dict[str, Any]) -> dict[str, Any]:
    reference = str(sample.get("reference", ""))
    prediction = str(sample.get("prediction", ""))
    normalized_reference = normalize_asr_text(reference)
    normalized_prediction = normalize_asr_text(prediction)
    return {
        "id": str(sample.get("id", "")),
        "wer_raw": word_error_rate(reference, prediction),
        "wer_normalized": word_error_rate(normalized_reference, normalized_prediction),
    }


def build_measurement_summary(
    result: dict[str, Any],
    *,
    limit: int,
    model_label: str,
    config_hash: str | None,
    harness_git_sha: str | None,
    normalization_version_hash: str | None,
    server_log_path: str | None,
    elapsed_seconds: float | None = None,
    energy_joules: float | None = None,
    emissions_kg: float | None = None,
) -> dict[str, Any]:
    raw_profile = result.get("metric_profiles", {}).get("raw", {})
    normalized_profile = result.get("metric_profiles", {}).get("local_asr", {})
    raw_wer_ci = raw_profile.get("bootstrap_ci", {}).get("wer", result.get("wer_bootstrap_ci"))
    normalized_wer_ci = normalized_profile.get(
        "bootstrap_ci", {}
    ).get("wer", result.get("wer_normalized_bootstrap_ci"))
    normalized_no_space_ci = normalized_profile.get("bootstrap_ci", {}).get(
        "cer_no_whitespace",
        result.get("cer_no_whitespace_normalized_bootstrap_ci"),
    )

    empty_prediction_ids = [
        str(sample.get("id", ""))
        for sample in result.get("samples", [])
        if sample.get("empty_prediction")
    ]

    summary = {
        "slice": result.get("language"),
        "limit": limit,
        "model_label": model_label,
        "config_hash": config_hash,
        "harness_git_sha": harness_git_sha,
        "normalization_version": normalization_version_hash,
        "wer_raw": result.get("wer"),
        "wer_raw_percent": result.get("wer_percent"),
        "wer_raw_ci95": _ci_pair(raw_wer_ci),
        "wer_raw_ci95_percent": _ci_percent_pair(raw_wer_ci),
        "wer_normalized": result.get("wer_normalized"),
        "wer_normalized_percent": result.get("wer_normalized_percent"),
        "wer_normalized_ci95": _ci_pair(normalized_wer_ci),
        "wer_normalized_ci95_percent": _ci_percent_pair(normalized_wer_ci),
        "cer_normalized_no_space": result.get("cer_no_whitespace_normalized"),
        "cer_normalized_no_space_percent": result.get(
            "cer_no_whitespace_normalized_percent"
        ),
        "cer_normalized_no_space_ci95": _ci_pair(normalized_no_space_ci),
        "cer_normalized_no_space_ci95_percent": _ci_percent_pair(normalized_no_space_ci),
        "hyp_chars_total": result.get("hyp_chars_total"),
        "ref_chars_total": result.get("ref_chars_total"),
        "verbosity_ratio": result.get("verbosity_ratio"),
        "verbosity_drift_warning": result.get("verbosity_drift_warning"),
        "verbosity_drift_warning_range": result.get("verbosity_drift_warning_range"),
        "empty_prediction_count": result.get("empty_prediction_count"),
        "empty_prediction_ids": empty_prediction_ids,
        "elapsed_seconds": elapsed_seconds,
        "energy_joules": energy_joules,
        "emissions_kg": emissions_kg,
        "ttft_seconds_p50": result.get("ttft_seconds_p50"),
        "ttft_seconds_p95": result.get("ttft_seconds_p95"),
        "latency_total_seconds_p50": result.get("latency_total_seconds_p50"),
        "latency_total_seconds_p95": result.get("latency_total_seconds_p95"),
        "streaming_tokens_per_second_p50": result.get("streaming_tokens_per_second_p50"),
        "streaming_tokens_per_second_p95": result.get("streaming_tokens_per_second_p95"),
        "realtime_failure_threshold_note": result.get("realtime_failure_threshold_note"),
        "per_sample_wer": [_sample_wer(sample) for sample in result.get("samples", [])],
        "server_log_path": server_log_path,
    }
    return summary


def attach_measurement_contract(
    payload: dict[str, Any],
    *,
    limit: int,
    model_label: str,
    config_hash: str | None,
    harness_git_sha: str | None,
    normalization_version_hash: str | None,
    server_log_path: str | None,
    elapsed_seconds: float | None = None,
    energy_joules: float | None = None,
    emissions_kg: float | None = None,
) -> dict[str, Any]:
    summaries = [
        build_measurement_summary(
            result,
            limit=limit,
            model_label=model_label,
            config_hash=config_hash,
            harness_git_sha=harness_git_sha,
            normalization_version_hash=normalization_version_hash,
            server_log_path=server_log_path,
            elapsed_seconds=elapsed_seconds,
            energy_joules=energy_joules,
            emissions_kg=emissions_kg,
        )
        for result in payload.get("results", [])
    ]
    payload["measurement_summaries"] = summaries
    if len(summaries) == 1:
        payload.update(summaries[0])
    return payload
