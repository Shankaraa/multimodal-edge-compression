from __future__ import annotations

import unicodedata
from typing import Sequence


ASR_NORMALIZATION_DESCRIPTION = (
    "Unicode NFKC, casefold, strip punctuation and symbols, remove controls, collapse whitespace"
)


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


def summarize_transcript_metrics(
    *,
    references: Sequence[str],
    predictions: Sequence[str],
) -> dict[str, float | str]:
    import jiwer

    references_no_whitespace = [remove_all_whitespace(text) for text in references]
    predictions_no_whitespace = [remove_all_whitespace(text) for text in predictions]

    normalized_references = [normalize_asr_text(text) for text in references]
    normalized_predictions = [normalize_asr_text(text) for text in predictions]

    normalized_references_no_whitespace = [
        remove_all_whitespace(text) for text in normalized_references
    ]
    normalized_predictions_no_whitespace = [
        remove_all_whitespace(text) for text in normalized_predictions
    ]

    wer_value = jiwer.wer(references, predictions)
    cer_value = jiwer.cer(references, predictions)
    cer_no_whitespace_value = jiwer.cer(
        references_no_whitespace,
        predictions_no_whitespace,
    )

    normalized_wer_value = jiwer.wer(normalized_references, normalized_predictions)
    normalized_cer_value = jiwer.cer(normalized_references, normalized_predictions)
    normalized_cer_no_whitespace_value = jiwer.cer(
        normalized_references_no_whitespace,
        normalized_predictions_no_whitespace,
    )

    return {
        "metric_normalization": ASR_NORMALIZATION_DESCRIPTION,
        "wer": wer_value,
        "wer_percent": wer_value * 100.0,
        "cer": cer_value,
        "cer_percent": cer_value * 100.0,
        "cer_no_whitespace": cer_no_whitespace_value,
        "cer_no_whitespace_percent": cer_no_whitespace_value * 100.0,
        "wer_normalized": normalized_wer_value,
        "wer_normalized_percent": normalized_wer_value * 100.0,
        "cer_normalized": normalized_cer_value,
        "cer_normalized_percent": normalized_cer_value * 100.0,
        "cer_no_whitespace_normalized": normalized_cer_no_whitespace_value,
        "cer_no_whitespace_normalized_percent": normalized_cer_no_whitespace_value * 100.0,
    }
