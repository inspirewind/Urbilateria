#!/usr/bin/env python3
"""Generate an independent real-payload oracle for one Qwen3.8 FP8 expert.

This reads the official safetensors payload with Python safetensors and implements
the Transformers fine-grained FP8 contract directly in PyTorch:

  weight = float8_e4m3fn_payload * BF16 weight_scale_inv
  activation_group_scale = max(abs(x[group])) / 448
  y = BF16(weight @ dequantize(float8(x / activation_group_scale)))

It never imports Urbilateria or any Rust-produced output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


BLOCK = 128
HIDDEN = 8192
INTERMEDIATE = 2048


def shard_for(model_dir: Path, name: str) -> Path:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return model_dir / index["weight_map"][name]


def load(model_dir: Path, name: str) -> torch.Tensor:
    with safe_open(shard_for(model_dir, name), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def dynamic_fp8(input_: torch.Tensor) -> torch.Tensor:
    groups = input_.reshape(-1, BLOCK)
    maximum = groups.abs().amax(dim=1, keepdim=True)
    scale = maximum / 448.0
    quantized = torch.where(
        maximum == 0,
        torch.zeros_like(groups),
        (groups / torch.where(scale == 0, torch.ones_like(scale), scale))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
        * scale,
    )
    return quantized.reshape(-1)


def fp8_matvec(weight: torch.Tensor, scale: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
    assert weight.ndim == 2 and scale.ndim == 2
    rows, columns = weight.shape
    assert rows % BLOCK == 0 and columns % BLOCK == 0
    assert tuple(scale.shape) == (rows // BLOCK, columns // BLOCK)
    quantized = dynamic_fp8(input_.to(torch.float32))
    output = torch.empty(rows, dtype=torch.float32)
    blocks = weight.reshape(rows // BLOCK, BLOCK, columns // BLOCK, BLOCK)
    for row_block in range(rows // BLOCK):
        dequantized = (
            blocks[row_block].to(torch.float32)
            * scale[row_block].to(torch.float32).reshape(1, -1, 1)
        ).reshape(BLOCK, columns)
        output[row_block * BLOCK : (row_block + 1) * BLOCK] = dequantized @ quantized
    return output.to(torch.bfloat16).to(torch.float32)


def projection(model_dir: Path, prefix: str, name: str, input_: torch.Tensor) -> tuple[torch.Tensor, dict]:
    weight_name = f"{prefix}.{name}.weight"
    scale_name = f"{prefix}.{name}.weight_scale_inv"
    weight = load(model_dir, weight_name)
    scale = load(model_dir, scale_name)
    output = fp8_matvec(weight, scale, input_)
    evidence = {
        "name": weight_name,
        "dtype": str(weight.dtype),
        "shape": list(weight.shape),
        "scale_dtype": str(scale.dtype),
        "scale_shape": list(scale.shape),
        "first_weight_codes": weight.reshape(-1)[:16].view(torch.uint8).tolist(),
        "first_scales": scale.to(torch.float32).reshape(-1)[:16].tolist(),
    }
    return output, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    prefix = f"model.layers.{args.layer}.mlp.experts.{args.expert}"
    input_ = torch.tensor(
        [(((index * 37) % 257) - 128) / 37.0 for index in range(HIDDEN)],
        dtype=torch.float32,
    )
    gate, gate_evidence = projection(args.model_dir, prefix, "gate_proj", input_)
    up, up_evidence = projection(args.model_dir, prefix, "up_proj", input_)
    activated = (torch.nn.functional.silu(gate.to(torch.bfloat16)) * up.to(torch.bfloat16))
    activated = activated.to(torch.bfloat16).to(torch.float32)
    expert, down_evidence = projection(args.model_dir, prefix, "down_proj", activated)

    artifact = {
        "format_version": 1,
        "model_family": "qwen3_8",
        "checkpoint": "Qwen/Qwen3.8-2.4T-A95B-FP8",
        "scope": "real layer-0 expert-0 FP8 payload and complete SwiGLU expert",
        "generator": "tools/generate_qwen3_8_fp8_oracle.py",
        "oracle": {
            "implementation": "independent PyTorch block dequantization",
            "torch_version": torch.__version__,
            "block": [BLOCK, BLOCK],
            "activation_group": BLOCK,
        },
        "layer": args.layer,
        "expert": args.expert,
        "input_formula": "(((index * 37) % 257) - 128) / 37",
        "payload_evidence": [gate_evidence, up_evidence, down_evidence],
        "expected": {
            "gate": gate.tolist(),
            "up": up.tolist(),
            "activated": activated.tolist(),
            "expert": expert.tolist(),
        },
    }
    serialized = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
