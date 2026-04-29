#!/usr/bin/env python3
"""Run Track B Voxtral W4A16 GPTQ with llm-compressor.

This runner exists because llm-compressor 0.10.x still assumes a CausalLM
AutoModel loader, while Voxtral Realtime is loaded through
VoxtralRealtimeForConditionalGeneration in the newer Transformers stack.
"""

from __future__ import annotations

import argparse
import json
import shutil
from types import MethodType
from pathlib import Path


TORCH_INIT_FUNCTION_NAMES = [
    "uniform_",
    "normal_",
    "trunc_normal_",
    "constant_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "uniform",
    "normal",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
]


TARGETS = [
    r"re:^language_model\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$",
]


SEQUENTIAL_TARGETS = [
    r"re:^language_model\.model\.layers\.\d+$",
]


IGNORE = [
    r"re:^audio_tower(\.|$)",
    r"re:^multi_modal_projector(\.|$)",
    r"re:^language_model\.model\.embed_tokens$",
    r"re:^language_model\.model\.norm$",
    r"re:^language_model\.lm_head$",
    r"re:^.*ada_[^.]*($|\.)",
    r"re:^.*rms_norm[^.]*($|\.)",
    r"re:^.*layer_?norm[^.]*($|\.)",
    r"re:^whisper_encoder(\.|$)",
    r"re:^audio_language_adapter(\.|$)",
    r"re:^language_model\.model\.layers\.\d+\.ada_rms_norm_t_cond(\.|$)",
    r"re:^mm_streams_embeddings(\.|$)",
    r"re:^layers\.\d+\.ada_rms_norm_t_cond(\.|$)",
    r"re:^layers\.\d+\.(attention_norm|ffn_norm)$",
    r"re:^norm$",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/voxtral-realtime")
    parser.add_argument(
        "--calibration-jsonl",
        default="data/calibration/track_b_multilingual_text_256.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="models/voxtral-w4a16-llmcompressor-v2",
    )
    parser.add_argument("--num-calibration-samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--dampening-frac", type=float, default=0.01)
    parser.add_argument(
        "--actorder",
        choices=("static", "dynamic", "weight", "group"),
        default="static",
        help="GPTQ activation ordering. static/weight is faster; dynamic/group stores g_idx.",
    )
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def patch_transformers_for_llmcompressor() -> None:
    """Restore the symbol llm-compressor imports from older Transformers."""
    import transformers.modeling_utils as modeling_utils

    if not hasattr(modeling_utils, "TORCH_INIT_FUNCTIONS"):
        modeling_utils.TORCH_INIT_FUNCTIONS = {
            name: None for name in TORCH_INIT_FUNCTION_NAMES
        }


def load_records(path: Path, limit: int | None) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def build_tokenized_dataset(records: list[dict], tokenizer, max_seq_length: int):
    from datasets import Dataset

    tokenized_rows = []
    for record in records:
        encoded = tokenizer(
            text=record["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokenized_rows.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "language": record["language"],
                "kind": record["kind"],
            }
        )

    return Dataset.from_list(tokenized_rows)


def copy_runtime_files(model_dir: Path, output_dir: Path) -> None:
    for name in [
        "tekken.json",
        "processor_config.json",
        "generation_config.json",
        "params.json",
        "README.md",
    ]:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def inject_fp8_kv_metadata(output_dir: Path) -> None:
    """Add compressed-tensors FP8 KV metadata after GPTQ weight calibration.

    llm-compressor 0.10.x initializes `kv_cache_scheme` by walking every attention
    module, including the Voxtral audio tower. Track A already applies FP8 KV at
    serving time through vLLM, so we keep calibration decoder-only and only add
    metadata for loaders that inspect the artifact config.
    """

    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization_config = config.setdefault("quantization_config", {})
    quantization_config["kv_cache_scheme"] = {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "tensor",
        "dynamic": False,
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    logits_to_keep=0,
    num_delay_tokens=None,
    **kwargs,
):
    """Calibration-only forward that bypasses the audio encoder wrapper."""
    import torch

    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if num_delay_tokens is None:
        num_delay_tokens = self.config.default_num_delay_tokens

    time_tensor = torch.full(
        (1,),
        num_delay_tokens,
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    t_cond = self.time_embedding(time_tensor)[None, ...]

    return self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        logits_to_keep=logits_to_keep,
        t_cond=t_cond,
        **kwargs,
    )


def main() -> None:
    args = parse_args()
    patch_transformers_for_llmcompressor()

    import torch
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoTokenizer, VoxtralRealtimeForConditionalGeneration

    model_dir = Path(args.model)
    calibration_jsonl = Path(args.calibration_jsonl)
    output_dir = Path(args.output_dir)

    if output_dir.exists() and args.overwrite_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    records = load_records(calibration_jsonl, args.limit_records)
    dataset = build_tokenized_dataset(records, tokenizer, args.max_seq_length)

    print(
        json.dumps(
            {
                "records_loaded": len(records),
                "dataset_rows": len(dataset),
                "min_tokens": min(len(row["input_ids"]) for row in dataset),
                "max_tokens": max(len(row["input_ids"]) for row in dataset),
                "tokenizer_class": tokenizer.__class__.__name__,
                "tokenizer_module": tokenizer.__class__.__module__,
                "vocab_size": getattr(tokenizer, "vocab_size", None),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()
    model.forward = MethodType(forward, model)

    recipe = GPTQModifier(
        scheme="W4A16",
        targets=TARGETS,
        ignore=IGNORE,
        dampening_frac=args.dampening_frac,
        actorder=args.actorder,
        offload_hessians=True,
    )

    oneshot(
        model=model,
        processor=tokenizer,
        recipe=recipe,
        dataset=dataset,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        data_collator="truncation",
        pad_to_max_length=False,
        shuffle_calibration_samples=False,
        output_dir=str(output_dir),
        sequential_targets=SEQUENTIAL_TARGETS,
        save_compressed=True,
        precision="bfloat16",
    )

    copy_runtime_files(model_dir, output_dir)
    inject_fp8_kv_metadata(output_dir)


if __name__ == "__main__":
    main()
