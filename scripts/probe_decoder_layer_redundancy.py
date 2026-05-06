#!/usr/bin/env python3
"""Probe Voxtral decoder-layer redundancy on a small mixed-language FLEURS slice.

This is intentionally non-destructive. It ranks consecutive decoder layers by
hidden-state cosine similarity, then can run a small generation WER check with
one or more layers patched to identity in memory.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import MethodType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


DEFAULT_LANGS = (
    "en_us",
    "fr_fr",
    "hi_in",
    "ja_jp",
    "es_419",
    "de_de",
    "pt_br",
    "cmn_hans_cn",
    "ar_eg",
    "ru_ru",
    "it_it",
    "nl_nl",
    "ko_kr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/voxtral-realtime")
    parser.add_argument("--dataset-source", choices=("google_fleurs", "open_asr_multilingual"), default="google_fleurs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--lang", action="append", default=None, help="FLEURS language code. Repeat to customize the mix.")
    parser.add_argument("--limit", type=int, default=50, help="Total mixed-language samples.")
    parser.add_argument("--rank-only", action="store_true", help="Only compute cosine ranking; skip generation WER checks.")
    parser.add_argument(
        "--skip-layer",
        action="append",
        type=int,
        default=None,
        help="Layer index to patch to identity for WER check. Repeat for multiple layers. Defaults to top ranked layer.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--torch-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or cpu.")
    parser.add_argument("--out", default="reports/decoder_layer_redundancy_probe.json")
    return parser.parse_args()


def runtime_deps():
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "This probe requires torch and transformers>=5.2 with Voxtral Realtime support. "
            "Run it in the pinned Linux/vLLM environment, not the lightweight Windows .venv."
        ) from exc
    return torch, functional, AutoProcessor, VoxtralRealtimeForConditionalGeneration


def resolve_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resolve_dtype(torch: Any, requested: str) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[requested]


def move_batch_to_device(batch: dict[str, Any], device: Any, dtype: Any) -> dict[str, Any]:
    import torch

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        moved[key] = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
    return moved


def select_samples(*, languages: list[str], total_limit: int, split: str, dataset_source: str) -> list[dict[str, Any]]:
    from voxtral_project.dataset_utils import get_sample_text, load_transcription_dataset_streaming

    per_language = {lang: total_limit // len(languages) for lang in languages}
    for lang in languages[: total_limit % len(languages)]:
        per_language[lang] += 1

    selected: list[dict[str, Any]] = []
    for lang in languages:
        needed = per_language[lang]
        if needed <= 0:
            continue
        dataset = load_transcription_dataset_streaming(
            lang_code=lang,
            split=split,
            dataset_source=dataset_source,
        )
        count = 0
        for index, sample in enumerate(dataset):
            audio = sample.get("audio")
            if not audio or audio.get("array") is None:
                continue
            selected.append(
                {
                    "language": lang,
                    "index": index,
                    "sample_id": str(sample.get("id", "")),
                    "reference": get_sample_text(sample),
                    "sample": sample,
                }
            )
            count += 1
            if count >= needed:
                break
        if count < needed:
            raise RuntimeError(f"Only found {count}/{needed} usable samples for {lang}.")
    return selected


def get_layers(model: Any) -> Any:
    try:
        return model.language_model.model.layers
    except AttributeError as exc:
        raise RuntimeError("Could not find decoder layers at model.language_model.model.layers.") from exc


def first_tensor(value: Any) -> Any:
    if isinstance(value, tuple):
        return value[0]
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    return value


def layer_output_hook(layer_outputs: dict[int, Any], layer_index: int):
    def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        layer_outputs[layer_index] = first_tensor(output).detach().float().cpu()

    return hook


def rank_layer_pairs(
    *,
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    device: Any,
    dtype: Any,
    torch: Any,
    functional: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layers = get_layers(model)
    pair_sums = [0.0 for _ in range(len(layers) - 1)]
    pair_counts = [0 for _ in range(len(layers) - 1)]
    sample_reports: list[dict[str, Any]] = []

    handles = []
    captured: dict[int, Any] = {}
    for layer_index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(layer_output_hook(captured, layer_index)))

    try:
        for sample_row in samples:
            sample = sample_row["sample"]
            audio = sample["audio"]
            encoded = processor(
                audio["array"],
                sampling_rate=int(audio["sampling_rate"]),
                return_tensors="pt",
            )
            num_delay_tokens = int(encoded["num_delay_tokens"].item())
            captured.clear()
            with torch.no_grad():
                batch = move_batch_to_device({"input_features": encoded["input_features"]}, device, dtype)
                audio_outputs = model.get_audio_features(
                    input_features=batch["input_features"],
                    use_cache=False,
                    return_dict=True,
                )
                inputs_embeds = audio_outputs.pooler_output.to(device=device, dtype=dtype)
                seq_len = int(inputs_embeds.shape[1])
                attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)
                time_tensor = torch.full((1,), num_delay_tokens, device=device, dtype=dtype)
                t_cond = model.time_embedding(time_tensor)[None, ...]
                model.language_model(
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    use_cache=False,
                    logits_to_keep=1,
                    t_cond=t_cond,
                )

            missing = [idx for idx in range(len(layers)) if idx not in captured]
            if missing:
                raise RuntimeError(f"Missing layer outputs for sample {sample_row['sample_id']}: {missing}")

            pair_values: list[float] = []
            for layer_index in range(len(layers) - 1):
                left = captured[layer_index].reshape(-1)
                right = captured[layer_index + 1].reshape(-1)
                cosine = float(functional.cosine_similarity(left, right, dim=0).item())
                pair_sums[layer_index] += cosine
                pair_counts[layer_index] += 1
                pair_values.append(cosine)

            sample_reports.append(
                {
                    "language": sample_row["language"],
                    "index": sample_row["index"],
                    "sample_id": sample_row["sample_id"],
                    "num_delay_tokens": num_delay_tokens,
                    "sequence_tokens": int(captured[0].shape[1]),
                    "max_consecutive_cosine": max(pair_values),
                    "max_pair_start_layer": int(max(range(len(pair_values)), key=pair_values.__getitem__)),
                }
            )
    finally:
        for handle in handles:
            handle.remove()

    pair_reports = [
        {
            "layer": layer_index,
            "successor_layer": layer_index + 1,
            "mean_cosine": pair_sums[layer_index] / pair_counts[layer_index],
            "samples": pair_counts[layer_index],
        }
        for layer_index in range(len(pair_sums))
        if pair_counts[layer_index]
    ]
    pair_reports.sort(key=lambda row: row["mean_cosine"], reverse=True)
    return pair_reports, sample_reports


@contextlib.contextmanager
def identity_skipped_layers(model: Any, layer_indices: list[int]) -> Iterator[None]:
    layers = get_layers(model)
    originals = {idx: layers[idx].forward for idx in layer_indices}

    def identity_forward(self: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> tuple[Any]:
        return (hidden_states,)

    try:
        for idx in layer_indices:
            layers[idx].forward = MethodType(identity_forward, layers[idx])
        yield
    finally:
        for idx, original in originals.items():
            layers[idx].forward = original


def generate_predictions(
    *,
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    device: Any,
    dtype: Any,
    torch: Any,
    max_new_tokens: int,
    skip_layers: list[int] | None = None,
) -> list[dict[str, Any]]:
    context = identity_skipped_layers(model, skip_layers or [])
    rows: list[dict[str, Any]] = []
    with context:
        for sample_row in samples:
            sample = sample_row["sample"]
            audio = sample["audio"]
            encoded = processor(
                audio["array"],
                sampling_rate=int(audio["sampling_rate"]),
                return_tensors="pt",
            )
            encoded = move_batch_to_device(dict(encoded), device, dtype)
            with torch.no_grad():
                output_ids = model.generate(**encoded, max_new_tokens=max_new_tokens)
            prediction = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            rows.append(
                {
                    "language": sample_row["language"],
                    "index": sample_row["index"],
                    "sample_id": sample_row["sample_id"],
                    "reference": sample_row["reference"],
                    "prediction": prediction,
                    "empty_prediction": prediction == "",
                }
            )
    return rows


def summarize_wer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from voxtral_project.text import normalize_asr_text, word_error_rate

    per_sample = []
    total_errors = 0
    total_words = 0
    for row in rows:
        reference = normalize_asr_text(row["reference"])
        prediction = normalize_asr_text(row["prediction"])
        result = word_error_rate(reference, prediction)
        total_errors += int(result["errors"])
        total_words += int(result["reference_words"])
        per_sample.append(
            {
                "language": row["language"],
                "index": row["index"],
                "sample_id": row["sample_id"],
                "normalized_wer": result["wer"],
                "errors": result["errors"],
                "reference_words": result["reference_words"],
                "empty_prediction": row["empty_prediction"],
            }
        )
    return {
        "normalized_wer": (total_errors / total_words) if total_words else None,
        "errors": total_errors,
        "reference_words": total_words,
        "empty_predictions": sum(1 for row in rows if row["empty_prediction"]),
        "per_sample": per_sample,
    }


def main() -> int:
    args = parse_args()
    torch, functional, AutoProcessor, VoxtralRealtimeForConditionalGeneration = runtime_deps()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.torch_dtype)
    languages = args.lang or list(DEFAULT_LANGS)

    samples = select_samples(
        languages=languages,
        total_limit=args.limit,
        split=args.split,
        dataset_source=args.dataset_source,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False

    pair_reports, sample_reports = rank_layer_pairs(
        model=model,
        processor=processor,
        samples=samples,
        device=device,
        dtype=dtype,
        torch=torch,
        functional=functional,
    )
    candidate_layer = int(pair_reports[0]["layer"])
    skip_layers = args.skip_layer if args.skip_layer is not None else [candidate_layer]

    payload: dict[str, Any] = {
        "model": args.model,
        "device": str(device),
        "torch_dtype": args.torch_dtype,
        "dataset_source": args.dataset_source,
        "split": args.split,
        "languages": languages,
        "sample_count": len(samples),
        "layer_count": len(get_layers(model)),
        "ranked_consecutive_layer_pairs": pair_reports,
        "candidate_removed_layer": candidate_layer,
        "sample_similarity_reports": sample_reports,
        "wer_check": None,
    }

    if not args.rank_only:
        baseline_rows = generate_predictions(
            model=model,
            processor=processor,
            samples=samples,
            device=device,
            dtype=dtype,
            torch=torch,
            max_new_tokens=args.max_new_tokens,
        )
        skipped_rows = generate_predictions(
            model=model,
            processor=processor,
            samples=samples,
            device=device,
            dtype=dtype,
            torch=torch,
            max_new_tokens=args.max_new_tokens,
            skip_layers=skip_layers,
        )
        baseline = summarize_wer(baseline_rows)
        skipped = summarize_wer(skipped_rows)
        delta = (
            skipped["normalized_wer"] - baseline["normalized_wer"]
            if skipped["normalized_wer"] is not None and baseline["normalized_wer"] is not None
            else None
        )
        payload["wer_check"] = {
            "skip_layers": skip_layers,
            "baseline": baseline,
            "skipped": skipped,
            "normalized_wer_delta": delta,
            "normalized_wer_delta_percentage_points": (delta * 100.0) if delta is not None else None,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"sample_similarity_reports"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
