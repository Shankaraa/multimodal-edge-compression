from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


ATTN_MAP = {
    "q_proj": "wq",
    "k_proj": "wk",
    "v_proj": "wv",
    "o_proj": "wo",
}

MLP_MAP = {
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
}

HF_ATTN_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)\."
    r"(weight_packed|weight_scale|weight_shape|weight_zero_point|weight_g_idx)$"
)
HF_MLP_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.mlp\."
    r"(gate_proj|down_proj|up_proj)\."
    r"(weight_packed|weight_scale|weight_shape|weight_zero_point|weight_g_idx)$"
)
HF_DECODER_NORM_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\."
    r"(input_layernorm|post_attention_layernorm)\.weight$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the Track B HF GPTQ output as a vLLM Mistral-layout artifact."
    )
    parser.add_argument("--source-model", default="models/voxtral-realtime")
    parser.add_argument("--hf-gptq-artifact", default="models/voxtral-w4a16-llmcompressor-v2")
    parser.add_argument(
        "--output-dir",
        default="models/voxtral-w4a16-llmcompressor-v2-consolidated",
    )
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def mistral_weight_name_for_hf_sidecar(name: str) -> str | None:
    attn_match = HF_ATTN_RE.match(name)
    if attn_match:
        layer, proj, suffix = attn_match.groups()
        return f"layers.{layer}.attention.{ATTN_MAP[proj]}.{suffix}"

    mlp_match = HF_MLP_RE.match(name)
    if mlp_match:
        layer, proj, suffix = mlp_match.groups()
        return f"layers.{layer}.feed_forward.{MLP_MAP[proj]}.{suffix}"

    return None


def mistral_weight_name_for_hf_decoder_norm(name: str) -> str | None:
    norm_match = HF_DECODER_NORM_RE.match(name)
    if not norm_match:
        return None

    layer, norm_name = norm_match.groups()
    mapped = {
        "input_layernorm": "attention_norm",
        "post_attention_layernorm": "ffn_norm",
    }[norm_name]
    return f"layers.{layer}.{mapped}.weight"


def quantized_mistral_weight_keys(num_layers: int = 26) -> set[str]:
    keys: set[str] = set()
    for layer in range(num_layers):
        for proj in ATTN_MAP.values():
            keys.add(f"layers.{layer}.attention.{proj}.weight")
        for proj in MLP_MAP.values():
            keys.add(f"layers.{layer}.feed_forward.{proj}.weight")
    return keys


def native_target_names(num_layers: int = 26) -> list[str]:
    targets: list[str] = []
    for layer in range(num_layers):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            targets.append(
                f"language_model.model.layers.{layer}.self_attn.{proj}"
            )
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            targets.append(f"language_model.model.layers.{layer}.mlp.{proj}")
    return targets


def native_ignore_names() -> list[str]:
    return [
        r"re:^whisper_encoder(\.|$)",
        r"re:^audio_language_adapter(\.|$)",
        r"re:^language_model\.model\.embed_tokens$",
        r"re:^language_model\.lm_head$",
        r"re:^language_model\.model\.norm$",
        r"re:^language_model\.model\.layers\.\d+\.ada_rms_norm_t_cond(\.|$)",
        r"re:^language_model\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)$",
        r"re:^mm_streams_embeddings(\.|$)",
        r"re:^layers\.\d+\.ada_rms_norm_t_cond(\.|$)",
        r"re:^layers\.\d+\.(attention_norm|ffn_norm)$",
        r"re:^norm$",
    ]


def update_config(source_config: Path, hf_artifact_config: Path, output_config: Path) -> None:
    config = json.loads(hf_artifact_config.read_text(encoding="utf-8"))
    source = json.loads(source_config.read_text(encoding="utf-8"))

    # Keep source serving defaults and the GPTQ quantization metadata, but drop
    # calibration-only mutations such as use_cache=False.
    for key, value in source.items():
        config.setdefault(key, value)
    config.pop("use_cache", None)

    quantization_config = config["quantization_config"]
    group = quantization_config["config_groups"]["group_0"]
    group["targets"] = native_target_names()
    quantization_config["ignore"] = native_ignore_names()
    quantization_config["kv_cache_scheme"] = {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "tensor",
        "dynamic": False,
    }

    output_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_runtime_files(source_model: Path, hf_artifact: Path, output_dir: Path) -> None:
    for name in [
        "params.json",
        "generation_config.json",
        "processor_config.json",
        "tekken.json",
        "README.md",
    ]:
        src = hf_artifact / name
        if not src.exists():
            src = source_model / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def main() -> int:
    args = parse_args()
    source_model = Path(args.source_model)
    hf_artifact = Path(args.hf_gptq_artifact)
    output_dir = Path(args.output_dir)

    if output_dir.exists() and args.overwrite_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = source_model / "consolidated.safetensors"
    hf_path = hf_artifact / "model.safetensors"
    output_path = output_dir / "consolidated.safetensors"

    skip_weights = quantized_mistral_weight_keys()
    tensors = {}

    with safe_open(base_path, framework="pt", device="cpu") as base_file:
        for key in base_file.keys():
            if key in skip_weights:
                continue
            tensors[key] = base_file.get_tensor(key)

    remapped_count = 0
    source_sidecar_count = 0
    smoothed_norm_count = 0
    with safe_open(hf_path, framework="pt", device="cpu") as hf_file:
        for key in hf_file.keys():
            mapped = mistral_weight_name_for_hf_sidecar(key)
            if mapped is not None:
                source_sidecar_count += 1
                if mapped in tensors:
                    raise ValueError(f"Refusing to overwrite existing tensor: {mapped}")
                tensors[mapped] = hf_file.get_tensor(key)
                remapped_count += 1
                continue

            mapped = mistral_weight_name_for_hf_decoder_norm(key)
            if mapped is None:
                continue
            if mapped not in tensors:
                raise ValueError(f"Expected decoder norm tensor missing from base: {mapped}")
            # AWQ smoothing changes these BF16 norm weights. They remain
            # unquantized, but the consolidated artifact must keep the smoothed
            # values or the balanced quantized projections no longer match.
            tensors[mapped] = hf_file.get_tensor(key)
            smoothed_norm_count += 1

    expected_sidecars = source_sidecar_count
    if remapped_count != expected_sidecars:
        raise ValueError(
            f"Expected {expected_sidecars} quantized sidecars, got {remapped_count}"
        )

    save_file(tensors, output_path)
    copy_runtime_files(source_model, hf_artifact, output_dir)
    update_config(
        source_model / "config.json",
        hf_artifact / "config.json",
        output_dir / "config.json",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "output_safetensors": str(output_path),
                "base_weights_skipped": len(skip_weights),
                "quantized_sidecars_remapped": remapped_count,
                "smoothed_decoder_norms_copied": smoothed_norm_count,
                "total_output_tensors": len(tensors),
                "targets": len(native_target_names()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
