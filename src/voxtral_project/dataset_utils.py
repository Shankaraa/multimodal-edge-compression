from __future__ import annotations

TRANSCRIPT_TEXT_FIELDS = (
    "text",
    "sentence",
    "normalized_text",
    "transcript",
    "transcription",
)

FLEURS_DATASET_SOURCES = (
    "google_fleurs",
    "open_asr_multilingual",
)


def fleurs_primary_language(lang_code: str) -> str:
    return lang_code.split("_", 1)[0].lower()


def get_sample_text(sample: dict) -> str:
    for field_name in TRANSCRIPT_TEXT_FIELDS:
        value = sample.get(field_name)
        if value is not None:
            return str(value)

    available_fields = ", ".join(sorted(sample.keys()))
    raise ValueError(
        "Expected one of the transcript fields "
        f"{TRANSCRIPT_TEXT_FIELDS}, but only found: {available_fields}"
    )


def open_asr_fleurs_config_name(lang_code: str) -> str:
    return f"fleurs_{fleurs_primary_language(lang_code)}"


def load_transcription_dataset_streaming(
    *,
    lang_code: str,
    split: str = "test",
    dataset_source: str = "google_fleurs",
):
    import datasets
    from datasets import load_dataset

    if dataset_source == "google_fleurs":
        major_version = int(str(datasets.__version__).split(".", 1)[0])
        if major_version >= 4:
            raise RuntimeError(
                "FLEURS loading in this repo requires `datasets<4` because the current "
                "`google/fleurs` packaging still depends on a dataset script. "
                f"Installed version: {datasets.__version__}. "
                "Reinstall the pinned workspace requirements before running FLEURS-based scripts."
            )

        return load_dataset(
            "google/fleurs",
            lang_code,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )

    if dataset_source == "open_asr_multilingual":
        return load_dataset(
            "nithinraok/asr-leaderboard-datasets",
            open_asr_fleurs_config_name(lang_code),
            split=split,
            streaming=True,
        )

    raise ValueError(f"Unsupported dataset source: {dataset_source}")


def load_fleurs_streaming(*, lang_code: str, split: str = "test"):
    return load_transcription_dataset_streaming(
        lang_code=lang_code,
        split=split,
        dataset_source="google_fleurs",
    )
