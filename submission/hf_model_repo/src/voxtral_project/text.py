from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from math import floor
from random import Random
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

BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 1729


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


def _edit_distance(reference: Sequence[str], prediction: Sequence[str]) -> int:
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)

    previous = list(range(len(prediction) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index] + [0] * len(prediction)
        for prediction_index, prediction_item in enumerate(prediction, start=1):
            substitution_cost = 0 if reference_item == prediction_item else 1
            current[prediction_index] = min(
                previous[prediction_index] + 1,
                current[prediction_index - 1] + 1,
                previous[prediction_index - 1] + substitution_cost,
            )
        previous = current
    return previous[-1]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator:
        return numerator / denominator
    return 0.0 if numerator == 0 else 1.0


def _sample_metric_counts(
    *,
    references: Sequence[str],
    predictions: Sequence[str],
) -> list[dict[str, int]]:
    counts: list[dict[str, int]] = []
    for reference, prediction in zip(references, predictions, strict=True):
        reference_words = reference.split()
        prediction_words = prediction.split()
        reference_chars = list(reference)
        prediction_chars = list(prediction)
        reference_no_whitespace = list(remove_all_whitespace(reference))
        prediction_no_whitespace = list(remove_all_whitespace(prediction))

        counts.append(
            {
                "wer_errors": _edit_distance(reference_words, prediction_words),
                "wer_units": len(reference_words),
                "cer_errors": _edit_distance(reference_chars, prediction_chars),
                "cer_units": len(reference_chars),
                "cer_no_whitespace_errors": _edit_distance(
                    reference_no_whitespace,
                    prediction_no_whitespace,
                ),
                "cer_no_whitespace_units": len(reference_no_whitespace),
            }
        )
    return counts


def _percent_interval(values: list[float], *, confidence_level: float) -> dict[str, float]:
    sorted_values = sorted(values)
    if not sorted_values:
        return {
            "low": 0.0,
            "high": 0.0,
            "low_percent": 0.0,
            "high_percent": 0.0,
        }

    alpha = max(0.0, min(1.0, 1.0 - confidence_level))
    low_index = int(floor((alpha / 2.0) * (len(sorted_values) - 1)))
    high_index = int(floor((1.0 - alpha / 2.0) * (len(sorted_values) - 1)))
    low = sorted_values[low_index]
    high = sorted_values[high_index]
    return {
        "low": low,
        "high": high,
        "low_percent": low * 100.0,
        "high_percent": high * 100.0,
    }


def _bootstrap_metric_ci(
    sample_counts: Sequence[dict[str, int]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    rng = Random(seed)
    sample_count = len(sample_counts)
    if sample_count == 0:
        empty_interval = {
            "low": 0.0,
            "high": 0.0,
            "low_percent": 0.0,
            "high_percent": 0.0,
        }
        return {
            "method": "utterance_resampling_with_replacement",
            "unit": "sample",
            "iterations": iterations,
            "confidence_level": confidence_level,
            "seed": seed,
            "wer": empty_interval,
            "cer": empty_interval,
            "cer_no_whitespace": empty_interval,
        }

    wer_values: list[float] = []
    cer_values: list[float] = []
    cer_no_whitespace_values: list[float] = []

    for _ in range(iterations):
        wer_errors = 0
        wer_units = 0
        cer_errors = 0
        cer_units = 0
        cer_no_whitespace_errors = 0
        cer_no_whitespace_units = 0

        for _ in range(sample_count):
            sample = sample_counts[rng.randrange(sample_count)]
            wer_errors += sample["wer_errors"]
            wer_units += sample["wer_units"]
            cer_errors += sample["cer_errors"]
            cer_units += sample["cer_units"]
            cer_no_whitespace_errors += sample["cer_no_whitespace_errors"]
            cer_no_whitespace_units += sample["cer_no_whitespace_units"]

        wer_values.append(_safe_ratio(wer_errors, wer_units))
        cer_values.append(_safe_ratio(cer_errors, cer_units))
        cer_no_whitespace_values.append(
            _safe_ratio(cer_no_whitespace_errors, cer_no_whitespace_units)
        )

    return {
        "method": "utterance_resampling_with_replacement",
        "unit": "sample",
        "iterations": iterations,
        "confidence_level": confidence_level,
        "seed": seed,
        "wer": _percent_interval(wer_values, confidence_level=confidence_level),
        "cer": _percent_interval(cer_values, confidence_level=confidence_level),
        "cer_no_whitespace": _percent_interval(
            cer_no_whitespace_values,
            confidence_level=confidence_level,
        ),
    }


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
) -> dict[str, float | str | dict[str, object]]:
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
        "bootstrap_ci": _bootstrap_metric_ci(
            _sample_metric_counts(references=references, predictions=predictions)
        ),
    }


def summarize_transcript_metrics(
    *,
    references: Sequence[str],
    predictions: Sequence[str],
    lang_code: str | None = None,
) -> dict[str, object]:
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
        "wer_bootstrap_ci": local_profile["bootstrap_ci"]["wer"],
        "cer": local_profile["cer"],
        "cer_percent": local_profile["cer_percent"],
        "cer_no_whitespace": local_profile["cer_no_whitespace"],
        "cer_no_whitespace_percent": local_profile["cer_no_whitespace_percent"],
        "wer_normalized": normalized_profile["wer"],
        "wer_normalized_percent": normalized_profile["wer_percent"],
        "wer_normalized_bootstrap_ci": normalized_profile["bootstrap_ci"]["wer"],
        "cer_normalized": normalized_profile["cer"],
        "cer_normalized_percent": normalized_profile["cer_percent"],
        "cer_no_whitespace_normalized": normalized_cer_no_whitespace_value,
        "cer_no_whitespace_normalized_percent": normalized_cer_no_whitespace_value * 100.0,
        "cer_no_whitespace_normalized_bootstrap_ci": normalized_profile["bootstrap_ci"][
            "cer_no_whitespace"
        ],
        "wer_open_asr_like_bootstrap_ci": open_asr_like_profile["bootstrap_ci"]["wer"],
        "metric_profiles": {
            "raw": local_profile,
            "local_asr": normalized_profile,
            "open_asr_like": open_asr_like_profile,
        },
    }
