#!/usr/bin/env python3
"""Run Track B Voxtral AWQ with llm-compressor.

AutoAWQ's public loader is CausalLM-shaped, while Voxtral Realtime needs
VoxtralRealtimeForConditionalGeneration. This runner keeps the same bridge used
for GPTQ: load Voxtral with the modern Transformers stack, patch a text-only
calibration forward, and let llm-compressor run AWQ only on decoder projections.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import MethodType


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
        default="models/voxtral-w4a16-awq-llmcompressor-v1",
    )
    parser.add_argument("--num-calibration-samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def patch_transformers_for_llmcompressor() -> None:
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
    from llmcompressor.modifiers.awq import AWQMapping, AWQModifier
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
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

    mappings = [
        AWQMapping(
            smooth_layer=r"re:^language_model\.model\.layers\.\d+\.input_layernorm$",
            balance_layers=[
                r"re:^language_model\.model\.layers\.\d+\.self_attn\.q_proj$",
                r"re:^language_model\.model\.layers\.\d+\.self_attn\.k_proj$",
                r"re:^language_model\.model\.layers\.\d+\.self_attn\.v_proj$",
            ],
        ),
        AWQMapping(
            smooth_layer=r"re:^language_model\.model\.layers\.\d+\.self_attn\.v_proj$",
            balance_layers=[
                r"re:^language_model\.model\.layers\.\d+\.self_attn\.o_proj$",
            ],
        ),
        AWQMapping(
            smooth_layer=r"re:^language_model\.model\.layers\.\d+\.post_attention_layernorm$",
            balance_layers=[
                r"re:^language_model\.model\.layers\.\d+\.mlp\.gate_proj$",
                r"re:^language_model\.model\.layers\.\d+\.mlp\.up_proj$",
            ],
        ),
        AWQMapping(
            smooth_layer=r"re:^language_model\.model\.layers\.\d+\.mlp\.up_proj$",
            balance_layers=[
                r"re:^language_model\.model\.layers\.\d+\.mlp\.down_proj$",
            ],
        ),
    ]

    recipe = AWQModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=TARGETS,
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=False,
                    strategy="group",
                    group_size=128,
                ),
                input_activations=None,
                output_activations=None,
                format="pack-quantized",
            )
        },
        targets=TARGETS,
        ignore=IGNORE,
        mappings=mappings,
        n_grid=args.n_grid,
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

    print(f"Saved AWQ artifact to {output_dir}")


if __name__ == "__main__":
    main()
