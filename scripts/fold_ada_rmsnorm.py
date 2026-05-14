from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ADA_W1_TEMPLATE = "layers.{layer}.ada_rms_norm_t_cond.0.weight"
ADA_W2_TEMPLATE = "layers.{layer}.ada_rms_norm_t_cond.2.weight"
FFN_NORM_TEMPLATE = "layers.{layer}.ffn_norm.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fold Voxtral Realtime's constant delay AdaRMSNorm modulation into "
            "Mistral-format ffn_norm weights for vLLM."
        )
    )
    parser.add_argument("--model-dir", required=True, help="Input model directory.")
    parser.add_argument("--out-dir", required=True, help="Output folded model directory.")
    parser.add_argument(
        "--source-file",
        default="consolidated.safetensors",
        help="Mistral-format safetensors file inside --model-dir.",
    )
    parser.add_argument("--delay-tokens", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--cond-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=26)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write only the JSON report.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path. Defaults to <out-dir>/ada_rmsnorm_fold_report.json.",
    )
    return parser.parse_args()


def require_runtime_deps():
    try:
        import torch
        import torch.nn.functional as functional
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit(
            "This script requires torch and safetensors. Use the Linux/WSL vLLM "
            "environment, for example ~/.venvs/voxtral-baseline."
        ) from exc
    return torch, functional, safe_open


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return header_len, header


def tensor_data_length(entry: dict[str, Any]) -> int:
    start, end = entry["data_offsets"]
    return int(end) - int(start)


def build_time_embedding(*, torch: Any, delay_tokens: int, hidden_size: int) -> Any:
    inv_freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(hidden_size // 2, dtype=torch.float32)
        / (hidden_size // 2)
    )
    time_tensor = torch.full((1, 1), float(delay_tokens), dtype=torch.float32)
    emb = time_tensor * inv_freq[None, :]
    return torch.cat((emb.cos(), emb.sin()), dim=-1)


def compute_folded_weights(
    *,
    source_path: Path,
    num_layers: int,
    hidden_size: int,
    delay_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch, functional, safe_open = require_runtime_deps()
    folded: dict[str, Any] = {}
    layer_reports: list[dict[str, Any]] = []
    t_cond = build_time_embedding(
        torch=torch,
        delay_tokens=delay_tokens,
        hidden_size=hidden_size,
    )

    with safe_open(source_path, framework="pt", device="cpu") as tensors:
        for layer in range(num_layers):
            w1_key = ADA_W1_TEMPLATE.format(layer=layer)
            w2_key = ADA_W2_TEMPLATE.format(layer=layer)
            norm_key = FFN_NORM_TEMPLATE.format(layer=layer)
            missing = [key for key in (w1_key, w2_key, norm_key) if key not in tensors.keys()]
            if missing:
                raise KeyError(f"Missing tensors for layer {layer}: {missing}")

            original_norm = tensors.get_tensor(norm_key)
            original_norm_dtype = str(original_norm.dtype).replace("torch.", "")
            w1 = tensors.get_tensor(w1_key).float()
            w2 = tensors.get_tensor(w2_key).float()
            norm_weight = original_norm.float()

            hidden = functional.linear(t_cond, w1)
            hidden = functional.gelu(hidden)
            ada = functional.linear(hidden, w2).squeeze(0)
            modulation = 1.0 + ada
            folded_weight = (norm_weight * modulation).to(original_norm.dtype)
            folded[norm_key] = folded_weight.contiguous()

            layer_reports.append(
                {
                    "layer": layer,
                    "norm_key": norm_key,
                    "compute_dtype": "float32",
                    "source_norm_dtype": original_norm_dtype,
                    "saved_folded_dtype": str(folded_weight.dtype).replace("torch.", ""),
                    "modulation_min": float(modulation.min().item()),
                    "modulation_max": float(modulation.max().item()),
                    "modulation_mean": float(modulation.mean().item()),
                }
            )
    return folded, layer_reports


def copy_non_weight_files(model_dir: Path, out_dir: Path) -> None:
    for path in model_dir.iterdir():
        if path.is_dir():
            if path.name == ".cache":
                continue
            target = out_dir / path.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
            continue
        if path.suffix == ".safetensors":
            continue
        shutil.copy2(path, out_dir / path.name)


def update_json_configs(out_dir: Path) -> None:
    params_path = out_dir / "params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text(encoding="utf-8"))
        params["ada_rms_norm_t_cond"] = False
        params_path.write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    config_path = out_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["ada_rms_norm_t_cond"] = False
        if isinstance(config.get("text_config"), dict):
            config["text_config"]["ada_rms_norm_t_cond"] = False
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tensor_to_bytes(tensor: Any) -> bytes:
    import torch

    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype is torch.bfloat16:
        return tensor.view(torch.uint16).numpy().tobytes()
    return tensor.numpy().tobytes()


def write_streamed_safetensors(
    *,
    source_path: Path,
    target_path: Path,
    old_header_len: int,
    old_header: dict[str, Any],
    folded_tensors: dict[str, Any],
) -> dict[str, Any]:
    old_data_start = 8 + old_header_len
    skipped_suffixes = (".ada_rms_norm_t_cond.0.weight", ".ada_rms_norm_t_cond.2.weight")
    tensor_names = [name for name in old_header.keys() if name != "__metadata__"]
    output_names = [
        name for name in tensor_names if not name.endswith(skipped_suffixes)
    ]

    new_header: dict[str, Any] = {}
    if "__metadata__" in old_header:
        new_header["__metadata__"] = dict(old_header["__metadata__"])
        new_header["__metadata__"]["ada_rmsnorm_folded"] = "true"

    offset = 0
    for name in output_names:
        old_entry = old_header[name]
        entry = {
            "dtype": old_entry["dtype"],
            "shape": old_entry["shape"],
            "data_offsets": [offset, offset + tensor_data_length(old_entry)],
        }
        if name in folded_tensors:
            entry["data_offsets"] = [offset, offset + len(tensor_to_bytes(folded_tensors[name]))]
        new_header[name] = entry
        offset = int(entry["data_offsets"][1])

    header_bytes = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    with source_path.open("rb") as source, target_path.open("wb") as target:
        target.write(struct.pack("<Q", len(header_bytes)))
        target.write(header_bytes)
        for name in output_names:
            if name in folded_tensors:
                target.write(tensor_to_bytes(folded_tensors[name]))
                continue
            start, end = old_header[name]["data_offsets"]
            source.seek(old_data_start + int(start))
            remaining = int(end) - int(start)
            while remaining:
                chunk = source.read(min(1024 * 1024 * 32, remaining))
                if not chunk:
                    raise IOError(f"Unexpected EOF while copying tensor {name}")
                target.write(chunk)
                remaining -= len(chunk)

    return {
        "output_tensors": len(output_names),
        "skipped_ada_tensors": len(tensor_names) - len(output_names),
        "output_bytes": target_path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    source_path = model_dir / args.source_file
    report_path = Path(args.report) if args.report else out_dir / "ada_rmsnorm_fold_report.json"

    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if out_dir.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{out_dir} exists. Use --overwrite to replace it.")

    old_header_len, old_header = read_safetensors_header(source_path)
    folded_tensors, layer_reports = compute_folded_weights(
        source_path=source_path,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        delay_tokens=args.delay_tokens,
    )

    write_summary: dict[str, Any] | None = None
    if not args.dry_run:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        copy_non_weight_files(model_dir, out_dir)
        update_json_configs(out_dir)
        write_summary = write_streamed_safetensors(
            source_path=source_path,
            target_path=out_dir / args.source_file,
            old_header_len=old_header_len,
            old_header=old_header,
            folded_tensors=folded_tensors,
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "out_dir": str(out_dir),
        "source_file": str(source_path),
        "delay_tokens": args.delay_tokens,
        "fold_compute_dtype": "float32",
        "fold_save_policy": "cast folded ffn_norm.weight back to each source tensor dtype at serialization time",
        "num_layers": args.num_layers,
        "folded_norm_keys": sorted(folded_tensors.keys()),
        "dry_run": bool(args.dry_run),
        "write_summary": write_summary,
        "layer_reports": layer_reports,
        "exactness_claim": (
            "Algebraic fold for constant t_cond: post_attention_rmsnorm(x, w) * "
            "(1 + ada(t_cond)) is equivalent to post_attention_rmsnorm(x, "
            "w * (1 + ada(t_cond))). Final checkpoint parity must still be "
            "validated with WER or logits because folded weights are serialized "
            "back to checkpoint dtype."
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Prepared AdaRMSNorm fold for {len(folded_tensors)} layers "
        f"(dry_run={args.dry_run})."
    )
    print(f"Saved fold report to: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
