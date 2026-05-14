#!/usr/bin/env python3
"""Build audio-conditioned Track B calibration tensors from FLEURS audio.

The saved dataset contains one row per real audio sample plus paths to the
projected audio embeddings that feed the decoder. The script runs the BF16
audio tower/projector and stores decoder layer-0 input activations as the audit
artifact for the calibration distribution.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


FLEURS_CODES = {
    "EN": "en_us",
    "ZH": "cmn_hans_cn",
    "HI": "hi_in",
    "ES": "es_419",
    "AR": "ar_eg",
    "FR": "fr_fr",
    "PT": "pt_br",
    "RU": "ru_ru",
    "DE": "de_de",
    "JA": "ja_jp",
    "KO": "ko_kr",
    "IT": "it_it",
    "NL": "nl_nl",
}


FLEURS_COUNTS = {
    "EN": 20,
    "ZH": 15,
    "HI": 61,
    "ES": 15,
    "AR": 15,
    "FR": 20,
    "PT": 15,
    "RU": 15,
    "DE": 15,
    "JA": 20,
    "KO": 15,
    "IT": 15,
    "NL": 15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/voxtral-realtime")
    parser.add_argument(
        "--output-dir",
        default="data/calibration/track_b_audio_conditioned_fleurs_train_256_hi61",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--candidate-pool", type=int, default=240)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--skip-activation-capture",
        action="store_true",
        help="Only save processor outputs; skip BF16 layer-0 activation audit.",
    )
    return parser.parse_args()


def select_fleurs_samples(language: str, fleurs_code: str, count: int, split: str, candidate_pool: int) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(
        "google/fleurs",
        fleurs_code,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )
    candidates: list[dict] = []
    for sample in dataset:
        audio = sample.get("audio")
        if not audio or audio.get("array") is None:
            continue
        text = sample.get("transcription") or sample.get("raw_transcription") or ""
        candidates.append(
            {
                "language": language,
                "fleurs_code": fleurs_code,
                "sample_id": str(sample.get("id", "")),
                "text": " ".join(str(text).split()),
                "sample": sample,
            }
        )
        if len(candidates) >= candidate_pool:
            break

    if len(candidates) < count:
        raise RuntimeError(
            f"Expected at least {count} samples for {language}/{fleurs_code}, got {len(candidates)}"
        )

    # Mix longer and normal utterances without leaking test data or adding randomness.
    candidates.sort(key=lambda row: len(row["text"]), reverse=True)
    long_count = max(count // 2, 1)
    long_rows = candidates[:long_count]
    remaining = candidates[long_count:]
    step = max(len(remaining) // max(count - long_count, 1), 1)
    selected = long_rows + remaining[::step]
    return selected[:count]


def tensor_to_list(tensor):
    return tensor.detach().cpu().tolist()


def move_batch_to_device(batch: dict, device, dtype):
    import torch

    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        if value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    dataset_dir = output_dir / "processor_dataset"
    activation_dir = output_dir / "layer0_inputs"

    if output_dir.exists():
        if not args.overwrite_output:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite-output")
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    activation_dir.mkdir(parents=True, exist_ok=True)
    decoder_input_dir = output_dir / "decoder_inputs"
    decoder_input_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    model = None
    if not args.skip_activation_capture:
        model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            args.model,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        model.config.use_cache = False

    rows: list[dict] = []
    activation_summaries: list[dict] = []
    sample_index = 0

    for language, count in FLEURS_COUNTS.items():
        fleurs_code = FLEURS_CODES[language]
        samples = select_fleurs_samples(
            language,
            fleurs_code,
            count,
            args.split,
            args.candidate_pool,
        )
        for selected in samples:
            sample = selected["sample"]
            audio = sample["audio"]
            audio_array = np.asarray(audio["array"], dtype=np.float32)
            sample_rate = int(audio["sampling_rate"])
            encoded = processor(
                audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )

            input_features = encoded["input_features"][0]
            num_delay_tokens = int(encoded["num_delay_tokens"].item())

            captured = {}
            with torch.no_grad():
                batch = move_batch_to_device(
                    {"input_features": encoded["input_features"]},
                    device,
                    dtype,
                )
                audio_outputs = model.get_audio_features(
                    input_features=batch["input_features"],
                    use_cache=False,
                    return_dict=True,
                )
                inputs_embeds = audio_outputs.pooler_output.to(dtype)

            seq_len = int(inputs_embeds.shape[1])
            attention_mask = torch.ones(seq_len, dtype=torch.long)
            input_ids = torch.zeros(seq_len, dtype=torch.long)
            decoder_input_path = f"decoder_inputs/sample_{sample_index:06d}.pt"
            torch.save(
                {
                    "inputs_embeds": inputs_embeds.detach().cpu().to(torch.bfloat16),
                    "attention_mask": attention_mask,
                    "num_delay_tokens": num_delay_tokens,
                    "language": language,
                    "fleurs_code": fleurs_code,
                    "sample_id": selected["sample_id"],
                    "input_features_shape": list(input_features.shape),
                },
                output_dir / decoder_input_path,
            )

            activation_path = None
            if model is not None:
                def capture_layer0_input(_module, hook_args, hook_kwargs):
                    hidden_states = hook_kwargs.get("hidden_states")
                    if hidden_states is None:
                        hidden_states = hook_args[0]
                    captured["layer0_input"] = hidden_states.detach().cpu().to(torch.bfloat16)

                handle = model.language_model.model.layers[0].register_forward_pre_hook(
                    capture_layer0_input,
                    with_kwargs=True,
                )
                try:
                    with torch.no_grad():
                        time_tensor = torch.full(
                            (1,),
                            num_delay_tokens,
                            device=device,
                            dtype=dtype,
                        )
                        t_cond = model.time_embedding(time_tensor)[None, ...]
                        model.language_model(
                            attention_mask=attention_mask[None, :].to(device),
                            inputs_embeds=inputs_embeds.to(device),
                            use_cache=False,
                            logits_to_keep=1,
                            t_cond=t_cond,
                        )
                finally:
                    handle.remove()

                layer0_input = captured["layer0_input"]
                activation_path = f"layer0_inputs/sample_{sample_index:06d}.pt"
                torch.save(
                    {
                        "layer0_input": layer0_input,
                        "language": language,
                        "fleurs_code": fleurs_code,
                        "sample_id": selected["sample_id"],
                        "input_ids_length": int(input_ids.numel()),
                        "input_features_shape": list(input_features.shape),
                    },
                    output_dir / activation_path,
                )
                activation_summaries.append(
                    {
                        "index": sample_index,
                        "language": language,
                        "sample_id": selected["sample_id"],
                        "shape": list(layer0_input.shape),
                        "dtype": str(layer0_input.dtype),
                        "abs_max": float(layer0_input.abs().max().float().item()),
                        "rms": float(layer0_input.float().pow(2).mean().sqrt().item()),
                    }
                )

            rows.append(
                {
                    "input_ids": tensor_to_list(input_ids),
                    "attention_mask": tensor_to_list(attention_mask),
                    "inputs_embeds_path": decoder_input_path,
                    "num_delay_tokens": num_delay_tokens,
                    "language": language,
                    "fleurs_code": fleurs_code,
                    "split": args.split,
                    "sample_id": selected["sample_id"],
                    "text": selected["text"],
                    "audio_duration_seconds": float(audio_array.size / sample_rate),
                    "audio_peak_abs": float(np.max(np.abs(audio_array))) if audio_array.size else 0.0,
                    "audio_rms": float(np.sqrt(np.mean(np.square(audio_array.astype(np.float64)))))
                    if audio_array.size
                    else 0.0,
                    "layer0_input_path": activation_path,
                }
            )
            sample_index += 1

    if len(rows) != 256:
        raise RuntimeError(f"Expected 256 rows, built {len(rows)}")

    dataset = Dataset.from_list(rows)
    dataset.save_to_disk(str(dataset_dir))

    counts = Counter(row["language"] for row in rows)
    metadata = {
        "output_dir": str(output_dir),
        "dataset_dir": str(dataset_dir),
        "rows": len(rows),
        "counts": dict(counts),
        "split": args.split,
        "model": args.model,
        "processor_class": processor.__class__.__name__,
        "tokenizer_class": processor.tokenizer.__class__.__name__,
        "feature_extractor_class": processor.feature_extractor.__class__.__name__,
        "activation_capture": not args.skip_activation_capture,
        "activation_count": len(activation_summaries),
        "min_input_ids": min(len(row["input_ids"]) for row in rows),
        "max_input_ids": max(len(row["input_ids"]) for row in rows),
        "min_decoder_tokens": min(len(row["input_ids"]) for row in rows),
        "max_decoder_tokens": max(len(row["input_ids"]) for row in rows),
        "activation_abs_max_max": max((row["abs_max"] for row in activation_summaries), default=None),
        "activation_rms_avg": (
            sum(row["rms"] for row in activation_summaries) / len(activation_summaries)
            if activation_summaries
            else None
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "layer0_activation_summary.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in activation_summaries),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
