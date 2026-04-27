from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Numerically prove the AdaRMSNorm tau-fold algebra on synthetic tensors."
    )
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--cond-dim", type=int, default=32)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=7)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument(
        "--out",
        default="reports/ada_rmsnorm_fold_proof.json",
        help="Path for the JSON proof report.",
    )
    return parser.parse_args()


def gelu(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(np.square(x), axis=-1, keepdims=True)
    return x * np.reciprocal(np.sqrt(variance + eps)) * weight


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x = rng.normal(size=(args.batch, args.seq_len, args.hidden_size)).astype(np.float64)
    weight = rng.normal(size=(args.hidden_size,)).astype(np.float64)
    t_cond = rng.normal(size=(args.hidden_size,)).astype(np.float64)
    w1 = rng.normal(size=(args.cond_dim, args.hidden_size)).astype(np.float64)
    w2 = rng.normal(size=(args.hidden_size, args.cond_dim)).astype(np.float64)

    ada = gelu(t_cond @ w1.T) @ w2.T
    modulation = 1.0 + ada
    folded_weight = weight * modulation

    normalized = rmsnorm(x, np.ones_like(weight), args.eps)
    legacy_grouped = normalized * (weight * modulation)
    folded = rmsnorm(x, folded_weight, args.eps)
    legacy_sequential = rmsnorm(x, weight, args.eps) * modulation

    grouped_diff = np.abs(legacy_grouped - folded)
    sequential_diff = np.abs(legacy_sequential - folded)

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formula": "RMSNorm(x, w) * g(tau) == RMSNorm(x, w * g(tau)) when g(tau) is constant for the utterance/layer.",
        "hidden_size": args.hidden_size,
        "cond_dim": args.cond_dim,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "eps": args.eps,
        "seed": args.seed,
        "grouped_float64_max_abs_diff": float(grouped_diff.max()),
        "grouped_float64_allclose_zero": bool(np.array_equal(legacy_grouped, folded)),
        "sequential_float64_max_abs_diff": float(sequential_diff.max()),
        "sequential_float64_mean_abs_diff": float(sequential_diff.mean()),
        "interpretation": (
            "Grouped multiplication is bit-identical in this proof because the folded path "
            "uses the same product w * g(tau). Sequential floating-point multiplication can "
            "differ at roundoff scale, so model-level validation should use WER-identical "
            "or logit-tolerance evidence after writing folded weights."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        "AdaRMSNorm fold proof: "
        f"grouped_max_abs_diff={payload['grouped_float64_max_abs_diff']:.3g}, "
        f"sequential_max_abs_diff={payload['sequential_float64_max_abs_diff']:.3g}"
    )
    print(f"Saved proof report to: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
