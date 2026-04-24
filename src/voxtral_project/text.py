from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Sequence


ASR_NORMALIZATION_DESCRIPTION = (
    "Unicode NFKC, casefold, strip punctuation and symbols, remove controls, collapse whitespace"
)

OPEN_ASR_LIKE_NORMALIZATION_DESCRIPTION = (
    "Inspired by huggingface/open_asr_leaderboard: Whisper EnglishTextNormalizer for English "
    "when available; multilingual symbol stripping with diacritics preserved and optional "
    "digit-to-words normalization for non-English"
)

NUM2WORDS_LANGUAGE_OVERRIDES = {
    "ar": "ar",
    "de": "de",
    "en": "en",
    "es": "es",
    "fr": "fr",
    "hi": "hi",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "nl": "nl",
    "pt": "pt",
    "ru": "ru",
    "zh": "zh",
}


def _primary_language(lang_code: str | None) -> str | None:
    if not lang_code:
        return None
    return lang_code.split("_", 1)[0].lower()


def normalize_asr_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    characters: list[str] = []

    for character in normalized:
        category = unicodedata.category(character)
        if category.startswith("P") or category.startswith("S"):
            characters.append(" ")
            continue
        if category.startswith("C"):
            continue
        characters.append(character)

    return " ".join("".join(characters).split())


def remove_all_whitespace(text: str) -> str:
    return "".join(text.split())


@lru_cache(maxsize=1)
def _load_open_asr_normalizers():
    try:
        from transformers.models.whisper.english_normalizer import (
            BasicMultilingualTextNormalizer,
            EnglishTextNormalizer,
        )
    except Exception:
        return None, None

    return EnglishTextNormalizer(), BasicMultilingualTextNormalizer(remove_diacritics=False)


def _remove_bracketed_metadata(text: str) -> str:
    without_square_brackets = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
    return re.sub(r"\(([^)]+?)\)", "", without_square_brackets)


def _fallback_open_asr_multilingual_normalize(text: str) -> str:
    cleaned = _remove_bracketed_metadata(text).lower()
    cleaned = unicodedata.normalize("NFKC", cleaned)
    characters: list[str] = []
    for character in cleaned:
        if unicodedata.category(character)[0] in "MSP":
            characters.append(" ")
            continue
        characters.append(character)

    collapsed = "".join(characters)
    collapsed = re.sub(r"[^\w\s]", "", collapsed)
    return " ".join(collapsed.split())


def _normalize_digit_sequences(text: str, *, lang_code: str | None) -> str:
    primary_language = _primary_language(lang_code)
    if primary_language is None:
        return text

    num2words_language = NUM2WORDS_LANGUAGE_OVERRIDES.get(primary_language)
    if num2words_language is None:
        return text

    try:
        import num2words
    except Exception:
        return text

    joined_thousands = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)

    def replace_digits(match: re.Match[str]) -> str:
        try:
            return str(num2words.num2words(int(match.group()), lang=num2words_language))
        except Exception:
            return match.group()

    return re.sub(r"\d+", replace_digits, joined_thousands)


def normalize_open_asr_like_text(text: str, *, lang_code: str | None) -> str:
    english_normalizer, multilingual_normalizer = _load_open_asr_normalizers()
    primary_language = _primary_language(lang_code)

    if primary_language == "en":
        if english_normalizer is not None:
            normalized = english_normalizer(text)
        else:
            normalized = normalize_asr_text(text)
        return " ".join(str(normalized).split())

    if multilingual_normalizer is not None:
        normalized = multilingual_normalizer(text)
    else:
        normalized = _fallback_open_asr_multilingual_normalize(text)

    normalized = _normalize_digit_sequences(normalized, lang_code=lang_code)
    return " ".join(str(normalized).split())


def _compute_metric_profile(
    *,
    references: Sequence[str],
    predictions: Sequence[str],
    description: str,
) -> dict[str, float | str]:
    import jiwer

    references_no_whitespace = [remove_all_whitespace(text) for text in references]
    predictions_no_whitespace = [remove_all_whitespace(text) for text in predictions]

    wer_value = jiwer.wer(references, predictions)
    cer_value = jiwer.cer(references, predictions)
    cer_no_whitespace_value = jiwer.cer(
        references_no_whitespace,
        predictions_no_whitespace,
    )

    return {
        "description": description,
        "wer": wer_value,
        "wer_percent": wer_value * 100.0,
        "cer": cer_value,
        "cer_percent": cer_value * 100.0,
        "cer_no_whitespace": cer_no_whitespace_value,
        "cer_no_whitespace_percent": cer_no_whitespace_value * 100.0,
    }


def summarize_transcript_metrics(
    *,
    references: Sequence[str],
    predictions: Sequence[str],
    lang_code: str | None = None,
) -> dict[str, float | str]:
    normalized_references = [normalize_asr_text(text) for text in references]
    normalized_predictions = [normalize_asr_text(text) for text in predictions]

    normalized_references_no_whitespace = [
        remove_all_whitespace(text) for text in normalized_references
    ]
    normalized_predictions_no_whitespace = [
        remove_all_whitespace(text) for text in normalized_predictions
    ]

    open_asr_like_references = [
        normalize_open_asr_like_text(text, lang_code=lang_code) for text in references
    ]
    open_asr_like_predictions = [
        normalize_open_asr_like_text(text, lang_code=lang_code) for text in predictions
    ]

    local_profile = _compute_metric_profile(
        references=references,
        predictions=predictions,
        description="No normalization; direct raw transcript comparison",
    )
    normalized_profile = _compute_metric_profile(
        references=normalized_references,
        predictions=normalized_predictions,
        description=ASR_NORMALIZATION_DESCRIPTION,
    )
    open_asr_like_profile = _compute_metric_profile(
        references=open_asr_like_references,
        predictions=open_asr_like_predictions,
        description=OPEN_ASR_LIKE_NORMALIZATION_DESCRIPTION,
    )

    import jiwer

    normalized_cer_no_whitespace_value = jiwer.cer(
        normalized_references_no_whitespace,
        normalized_predictions_no_whitespace,
    )

    return {
        "metric_normalization": ASR_NORMALIZATION_DESCRIPTION,
        "wer": local_profile["wer"],
        "wer_percent": local_profile["wer_percent"],
        "cer": local_profile["cer"],
        "cer_percent": local_profile["cer_percent"],
        "cer_no_whitespace": local_profile["cer_no_whitespace"],
        "cer_no_whitespace_percent": local_profile["cer_no_whitespace_percent"],
        "wer_normalized": normalized_profile["wer"],
        "wer_normalized_percent": normalized_profile["wer_percent"],
        "cer_normalized": normalized_profile["cer"],
        "cer_normalized_percent": normalized_profile["cer_percent"],
        "cer_no_whitespace_normalized": normalized_cer_no_whitespace_value,
        "cer_no_whitespace_normalized_percent": normalized_cer_no_whitespace_value * 100.0,
        "metric_profiles": {
            "raw": local_profile,
            "local_asr": normalized_profile,
            "open_asr_like": open_asr_like_profile,
        },
    }
