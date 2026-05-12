#!/usr/bin/env python3
"""Run Track B Voxtral GPTQ with llm-compressor.

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


PROJECTION_SUFFIX_RE = (
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))"
)


TARGETS = [
    rf"re:^language_model\.model\.layers\.\d+\.{PROJECTION_SUFFIX_RE}$",
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
        "--audio-calibration-dataset",
        default=None,
        help=(
            "Path to a Dataset.save_to_disk directory built by "
            "build_track_b_audio_conditioned_calibration.py. When set, "
            "calibration loads real-audio projected decoder inputs from disk."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="models/voxtral-w4a16-llmcompressor-v2",
    )
    parser.add_argument("--num-calibration-samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--smoothquant", action="store_true")
    parser.add_argument("--smoothing-strength", type=float, default=0.8)
    parser.add_argument(
        "--spinquant",
        action="store_true",
        help=(
            "Inject SpinQuant Hadamard rotations (R1+R2) before GPTQ. Spreads "
            "outliers across channels so W4A16 stays robust on low-magnitude "
            "decoder activations (e.g. quiet-audio inputs). Note: SpinQuant "
            "fuses rotations into RMSNorms and may not handle Voxtral's extra "
            "ada_rms_norm_t_cond per-layer norm; use --quip if calibration "
            "fails inside SpinQuantModifier._fuse_norms."
        ),
    )
    parser.add_argument(
        "--spinquant-rotations",
        default="R1,R2",
        help="Comma-separated SpinQuant rotation set (subset of R1,R2,R3,R4).",
    )
    parser.add_argument(
        "--spinquant-transform-type",
        default="hadamard",
        choices=("hadamard", "random-hadamard"),
        help="Transform family used by SpinQuantModifier.",
    )
    parser.add_argument(
        "--quip",
        action="store_true",
        help=(
            "Inject QuIP-style Hadamard rotations (v+u) before GPTQ. Architecture-"
            "agnostic alternative to SpinQuant that does NOT fuse norms, so it "
            "works on multimodal models with extra norms like Voxtral."
        ),
    )
    parser.add_argument(
        "--quip-rotations",
        default="v,u",
        help="Comma-separated QuIP rotation set (subset of v,u).",
    )
    parser.add_argument(
        "--quip-transform-block-size",
        type=int,
        default=128,
        help="QuIP block size for Hadamard transforms.",
    )
    parser.add_argument(
        "--scheme",
        default="W4A16",
        choices=("W4A16", "W8A16"),
        help="Single-precision GPTQ scheme for all decoder projection targets.",
    )
    parser.add_argument(
        "--w8-layers",
        default=None,
        help=(
            "Optional mixed-precision layer spec for W8A16, e.g. '0-2,23-25'. "
            "When set, these layers use W8A16 and all remaining decoder layers "
            "use --scheme."
        ),
    )
    parser.add_argument("--num-decoder-layers", type=int, default=26)
    parser.add_argument("--dampening-frac", type=float, default=0.01)
    parser.add_argument(
        "--actorder",
        choices=("static", "dynamic", "weight", "group"),
        default="static",
        help="GPTQ activation ordering. static/weight is faster; dynamic/group stores g_idx.",
    )
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


SMOOTHQUANT_MAPPINGS = [
    [
        [
            r"re:^language_model\.model\.layers\.\d+\.self_attn\.q_proj$",
            r"re:^language_model\.model\.layers\.\d+\.self_attn\.k_proj$",
            r"re:^language_model\.model\.layers\.\d+\.self_attn\.v_proj$",
        ],
        r"re:^language_model\.model\.layers\.\d+\.input_layernorm$",
    ],
    [
        [
            r"re:^language_model\.model\.layers\.\d+\.mlp\.gate_proj$",
            r"re:^language_model\.model\.layers\.\d+\.mlp\.up_proj$",
        ],
        r"re:^language_model\.model\.layers\.\d+\.post_attention_layernorm$",
    ],
]


SMOOTHQUANT_IGNORE = [
    r"re:^audio_tower(\.|$)",
    r"re:^multi_modal_projector(\.|$)",
    r"re:^language_model\.model\.embed_tokens$",
    r"re:^language_model\.model\.norm$",
    r"re:^language_model\.lm_head$",
    r"re:^.*ada_[^.]*($|\.)",
    r"re:^whisper_encoder(\.|$)",
    r"re:^audio_language_adapter(\.|$)",
    r"re:^mm_streams_embeddings(\.|$)",
    r"re:^norm$",
]


def parse_layer_spec(spec: str, *, num_layers: int) -> list[int]:
    layers: set[int] = set()
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending layer range: {item}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))

    invalid = sorted(layer for layer in layers if layer < 0 or layer >= num_layers)
    if invalid:
        raise ValueError(f"Layer indices out of range for {num_layers} layers: {invalid}")
    return sorted(layers)


def target_for_layers(layers: list[int]) -> list[str]:
    if not layers:
        return []
    layer_alt = "|".join(str(layer) for layer in layers)
    return [
        rf"re:^language_model\.model\.layers\.({layer_alt})\.{PROJECTION_SUFFIX_RE}$"
    ]


def build_recipe(args: argparse.Namespace, modifier_cls):
    if args.w8_layers is None:
        return modifier_cls(
            scheme=args.scheme,
            targets=TARGETS,
            ignore=IGNORE,
            dampening_frac=args.dampening_frac,
            actorder=args.actorder,
            offload_hessians=True,
        )

    w8_layers = parse_layer_spec(args.w8_layers, num_layers=args.num_decoder_layers)
    remaining_layers = [
        layer for layer in range(args.num_decoder_layers) if layer not in set(w8_layers)
    ]
    if args.scheme == "W8A16":
        raise ValueError("--w8-layers is only meaningful when --scheme is W4A16")
    if not remaining_layers:
        raise ValueError("--w8-layers covers every layer; use --scheme W8A16 instead")

    return [
        modifier_cls(
            scheme="W8A16",
            targets=target_for_layers(w8_layers),
            ignore=IGNORE,
            dampening_frac=args.dampening_frac,
            actorder=args.actorder,
            offload_hessians=True,
        ),
        modifier_cls(
            scheme=args.scheme,
            targets=target_for_layers(remaining_layers),
            ignore=IGNORE,
            dampening_frac=args.dampening_frac,
            actorder=args.actorder,
            offload_hessians=True,
        ),
    ]


def recipe_summary(recipe) -> list[dict]:
    modifiers = recipe if isinstance(recipe, list) else [recipe]
    summary = []
    for modifier in modifiers:
        item = {
            "modifier": modifier.__class__.__name__,
        }
        if hasattr(modifier, "scheme"):
            item.update(
                {
                    "scheme": modifier.scheme,
                    "targets": modifier.targets,
                    "ignore": modifier.ignore,
                    "dampening_frac": modifier.dampening_frac,
                    "actorder": str(modifier.actorder),
                }
            )
        if hasattr(modifier, "smoothing_strength"):
            item.update({
                "smoothing_strength": modifier.smoothing_strength,
                "mappings": modifier.mappings,
                "ignore": modifier.ignore,
            })
        if hasattr(modifier, "rotations") and hasattr(modifier, "transform_type"):
            item.update({
                "rotations": list(modifier.rotations),
                "transform_type": str(modifier.transform_type),
            })
        summary.append(item)
    return summary


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


def make_audio_conditioned_collator(calibration_root: Path):
    def collator(features: list[dict]):
        import torch

        if len(features) != 1:
            raise ValueError("Audio-conditioned calibration requires batch_size=1")

        feature = features[0]
        tensor_path = calibration_root / feature["inputs_embeds_path"]
        payload = torch.load(tensor_path, map_location="cpu")

        inputs_embeds = payload["inputs_embeds"].to(dtype=torch.bfloat16)
        if inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        attention_mask = payload["attention_mask"].to(dtype=torch.long)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "num_delay_tokens": int(payload["num_delay_tokens"]),
        }

    return collator


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
    from datasets import load_from_disk
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
    if args.spinquant:
        from llmcompressor.modifiers.transform import SpinQuantModifier
    if args.quip:
        from llmcompressor.modifiers.transform import QuIPModifier
    from transformers import AutoProcessor, AutoTokenizer, VoxtralRealtimeForConditionalGeneration

    model_dir = Path(args.model)
    calibration_jsonl = Path(args.calibration_jsonl)
    output_dir = Path(args.output_dir)
    audio_calibration_dataset = (
        Path(args.audio_calibration_dataset)
        if args.audio_calibration_dataset is not None
        else None
    )

    if output_dir.exists() and args.overwrite_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = tokenizer
    records = []
    if audio_calibration_dataset is None:
        records = load_records(calibration_jsonl, args.limit_records)
        dataset = build_tokenized_dataset(records, tokenizer, args.max_seq_length)
        data_collator = "truncation"
    else:
        if args.batch_size != 1:
            raise ValueError("--audio-calibration-dataset requires --batch-size 1")
        processor = AutoProcessor.from_pretrained(model_dir)
        dataset = load_from_disk(str(audio_calibration_dataset))
        if args.limit_records is not None:
            dataset = dataset.select(range(min(args.limit_records, len(dataset))))
        data_collator = make_audio_conditioned_collator(audio_calibration_dataset)

    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()
    model.forward = MethodType(forward, model)

    # SpinQuantModifier (and a few llm-compressor utilities) call helpers like
    # get_head_dim() on `model.config` and assume a flat text-LLM config. Voxtral
    # Realtime nests the relevant fields under `text_config`, so we expose
    # the text-side fields at the top level. This is metadata-only: the model
    # weights and forward path are unchanged.
    if hasattr(model.config, "text_config"):
        text_cfg = model.config.text_config
        for attr in (
            "head_dim",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_hidden_layers",
            "intermediate_size",
            "vocab_size",
            "max_position_embeddings",
            "rms_norm_eps",
        ):
            if not hasattr(model.config, attr) and hasattr(text_cfg, attr):
                setattr(model.config, attr, getattr(text_cfg, attr))

    gptq_recipe = build_recipe(args, GPTQModifier)
    recipe = gptq_recipe if isinstance(gptq_recipe, list) else [gptq_recipe]
    if args.smoothquant:
        recipe = [
            SmoothQuantModifier(
                smoothing_strength=args.smoothing_strength,
                mappings=SMOOTHQUANT_MAPPINGS,
                ignore=SMOOTHQUANT_IGNORE,
            )
        ] + recipe
    if args.spinquant and args.quip:
        raise ValueError("Use either --spinquant or --quip, not both.")
    if args.spinquant:
        rotations = [r.strip() for r in args.spinquant_rotations.split(",") if r.strip()]
        recipe = [
            SpinQuantModifier(
                rotations=rotations,
                transform_type=args.spinquant_transform_type,
            )
        ] + recipe
    if args.quip:
        rotations = [r.strip() for r in args.quip_rotations.split(",") if r.strip()]
        # Constrain QuIP rotations to the same decoder projection set GPTQ
        # quantizes. Without this QuIP also tries to rotate the small Linears
        # in the audio adapter, which fails with "block_size must divide N"
        # because adapter projections aren't multiples of 128.
        recipe = [
            QuIPModifier(
                rotations=rotations,
                transform_block_size=args.quip_transform_block_size,
                transform_type="hadamard",
                targets=TARGETS,
                ignore=IGNORE,
            )
        ] + recipe
    if len(recipe) == 1:
        recipe = recipe[0]

    print(
        json.dumps(
            {
                "calibration_mode": "audio_conditioned"
                if audio_calibration_dataset is not None
                else "text",
                "records_loaded": len(records) if records else len(dataset),
                "dataset_rows": len(dataset),
                "min_tokens": min(len(row["input_ids"]) for row in dataset),
                "max_tokens": max(len(row["input_ids"]) for row in dataset),
                "audio_calibration_dataset": str(audio_calibration_dataset)
                if audio_calibration_dataset is not None
                else None,
                "tokenizer_class": tokenizer.__class__.__name__,
                "tokenizer_module": tokenizer.__class__.__module__,
                "vocab_size": getattr(tokenizer, "vocab_size", None),
                "output_dir": str(output_dir),
                "recipe": recipe_summary(recipe),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    oneshot(
        model=model,
        processor=processor,
        recipe=recipe,
        dataset=dataset,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        data_collator=data_collator,
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
