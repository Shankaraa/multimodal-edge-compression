from __future__ import annotations

import os

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
    "common_voice_17",
)

COMMON_VOICE_17_DEFAULT_REPO_ID = "fsicoli/common_voice_17_0"

COMMON_VOICE_17_CONFIGS = {
    "ar_eg": "ar",
    "cmn_hans_cn": "zh-CN",
    "de_de": "de",
    "en_us": "en",
    "es_419": "es",
    "fr_fr": "fr",
    "hi_in": "hi",
    "it_it": "it",
    "ja_jp": "ja",
    "ko_kr": "ko",
    "nl_nl": "nl",
    "pt_br": "pt",
    "ru_ru": "ru",
}


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


def common_voice_17_config_name(lang_code: str) -> str:
    try:
        return COMMON_VOICE_17_CONFIGS[lang_code]
    except KeyError as exc:
        raise ValueError(f"No Common Voice 17 config mapped for language: {lang_code}") from exc


def common_voice_17_repo_id() -> str:
    return os.environ.get("COMMON_VOICE_17_REPO_ID", COMMON_VOICE_17_DEFAULT_REPO_ID)


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

    if dataset_source == "common_voice_17":
        from datasets import Audio, Features, Value

        return load_dataset(
            common_voice_17_repo_id(),
            common_voice_17_config_name(lang_code),
            split=split,
            streaming=True,
            features=Features(
                {
                    "client_id": Value("string"),
                    "path": Value("string"),
                    "sentence_id": Value("string"),
                    "sentence": Value("string"),
                    "sentence_domain": Value("string"),
                    "up_votes": Value("string"),
                    "down_votes": Value("string"),
                    "age": Value("string"),
                    "gender": Value("string"),
                    "variant": Value("string"),
                    "locale": Value("string"),
                    "segment": Value("string"),
                    "accent": Value("string"),
                    "audio": Audio(sampling_rate=48_000, mono=True, decode=True),
                }
            ),
        )

    raise ValueError(f"Unsupported dataset source: {dataset_source}")


def load_fleurs_streaming(*, lang_code: str, split: str = "test"):
    return load_transcription_dataset_streaming(
        lang_code=lang_code,
        split=split,
        dataset_source="google_fleurs",
    )
