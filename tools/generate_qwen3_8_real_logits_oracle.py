#!/usr/bin/env python3
"""Stream one real Qwen3.8 token to an independent PyTorch logits oracle.

The implementation follows the upstream Transformers equations while reading only
one decoder trunk and the ten selected FP8 experts at a time.  It therefore avoids
constructing the 2.4T-parameter model.  This is deliberately separate from the Rust
runtime and is intended as a slow, opt-in release gate rather than an inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open


BLOCK = 128


class Checkpoint:
    def __init__(self, directory: Path):
        self.directory = directory
        index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]
        self.handles: dict[str, object] = {}

    def handle(self, name: str):
        shard = self.weight_map[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(self.directory / shard, framework="pt", device="cpu")
        return self.handles[shard]

    def tensor(self, name: str) -> torch.Tensor:
        return self.handle(name).get_tensor(name)

    def row(self, name: str, row: int) -> torch.Tensor:
        return self.handle(name).get_slice(name)[row]


def bf16(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16)


def linear(checkpoint: Checkpoint, name: str, input_: torch.Tensor, chunk: int) -> torch.Tensor:
    """BF16 weight/input, F32 accumulation, BF16 materialization."""
    weight = checkpoint.tensor(name)
    assert weight.dtype == torch.bfloat16 and weight.ndim == 2
    input_f32 = input_.to(torch.float32)
    output = torch.empty(weight.shape[0], dtype=torch.bfloat16)
    for start in range(0, weight.shape[0], chunk):
        stop = min(start + chunk, weight.shape[0])
        output[start:stop] = (weight[start:stop].to(torch.float32) @ input_f32).to(torch.bfloat16)
    return output


def zero_centered_norm(input_: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    value = input_.to(torch.float32)
    value = value * torch.rsqrt(value.square().mean() + eps)
    return (value * (1.0 + weight.to(torch.float32))).to(torch.bfloat16)


def dynamic_fp8(input_: torch.Tensor) -> torch.Tensor:
    groups = input_.to(torch.float32).reshape(-1, BLOCK)
    maximum = groups.abs().amax(dim=1, keepdim=True)
    scale = maximum / 448.0
    denominator = torch.where(scale == 0, torch.ones_like(scale), scale)
    value = (groups / denominator).clamp(-448.0, 448.0)
    value = value.to(torch.float8_e4m3fn).to(torch.float32) * scale
    return torch.where(maximum == 0, torch.zeros_like(value), value).reshape(-1)


def fp8_linear(
    checkpoint: Checkpoint, weight_name: str, input_: torch.Tensor, chunk_blocks: int
) -> torch.Tensor:
    scale_name = weight_name.removesuffix(".weight") + ".weight_scale_inv"
    weight = checkpoint.tensor(weight_name)
    scale = checkpoint.tensor(scale_name)
    rows, columns = weight.shape
    assert weight.dtype == torch.float8_e4m3fn
    assert rows % BLOCK == 0 and columns % BLOCK == 0
    assert tuple(scale.shape) == (rows // BLOCK, columns // BLOCK)
    input_f32 = dynamic_fp8(input_)
    output = torch.empty(rows, dtype=torch.bfloat16)
    blocks = weight.reshape(rows // BLOCK, BLOCK, columns // BLOCK, BLOCK)
    for first in range(0, rows // BLOCK, chunk_blocks):
        last = min(first + chunk_blocks, rows // BLOCK)
        dequantized = (
            blocks[first:last].to(torch.float32)
            * scale[first:last].to(torch.float32)[:, None, :, None]
        ).reshape((last - first) * BLOCK, columns)
        output[first * BLOCK : last * BLOCK] = (dequantized @ input_f32).to(torch.bfloat16)
    return output


def fp8_expert(checkpoint: Checkpoint, prefix: str, input_: torch.Tensor, chunk_blocks: int) -> torch.Tensor:
    gate = fp8_linear(checkpoint, f"{prefix}.gate_proj.weight", input_, chunk_blocks)
    up = fp8_linear(checkpoint, f"{prefix}.up_proj.weight", input_, chunk_blocks)
    activated = torch.nn.functional.silu(gate) * up
    return fp8_linear(
        checkpoint,
        f"{prefix}.down_proj.weight",
        activated.to(torch.bfloat16),
        chunk_blocks,
    )


def dense_expert(checkpoint: Checkpoint, prefix: str, input_: torch.Tensor, chunk: int) -> torch.Tensor:
    gate = linear(checkpoint, f"{prefix}.gate_proj.weight", input_, chunk)
    up = linear(checkpoint, f"{prefix}.up_proj.weight", input_, chunk)
    activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    return linear(checkpoint, f"{prefix}.down_proj.weight", activated, chunk)


def delta_net(
    checkpoint: Checkpoint, prefix: str, input_: torch.Tensor, config: dict, chunk: int
) -> torch.Tensor:
    key_heads = config["linear_num_key_heads"]
    value_heads = config["linear_num_value_heads"]
    key_dim = config["linear_key_head_dim"]
    value_dim = config["linear_value_head_dim"]
    key_width = key_heads * key_dim
    value_width = value_heads * value_dim

    mixed = linear(checkpoint, f"{prefix}.linear_attn.in_proj_qkv.weight", input_, chunk)
    convolution = checkpoint.tensor(f"{prefix}.linear_attn.conv1d.weight")[:, 0, -1]
    convolved = torch.nn.functional.silu((mixed * convolution).to(torch.bfloat16)).to(torch.bfloat16)
    query = convolved[:key_width].reshape(key_heads, key_dim)
    key = convolved[key_width : 2 * key_width].reshape(key_heads, key_dim)
    value = convolved[2 * key_width : 2 * key_width + value_width].reshape(value_heads, value_dim)
    beta = torch.sigmoid(
        linear(checkpoint, f"{prefix}.linear_attn.in_proj_b.weight", input_, chunk)
    ).to(torch.bfloat16)
    z = linear(checkpoint, f"{prefix}.linear_attn.in_proj_z.weight", input_, chunk).reshape(
        value_heads, value_dim
    )

    # Match the upstream l2norm call before its recurrent kernel converts to F32.
    query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
    repeats = value_heads // key_heads
    query = query.repeat_interleave(repeats, dim=0).to(torch.float32)
    key = key.repeat_interleave(repeats, dim=0).to(torch.float32)
    value_f32 = value.to(torch.float32)
    beta_f32 = beta.to(torch.float32)[:, None]
    dot = (query * key).sum(dim=-1, keepdim=True) / math.sqrt(key_dim)
    core = (value_f32 * beta_f32 * dot).to(torch.bfloat16)

    # Qwen3_5MoeRMSNormGated: norm, direct weight, then SiLU(z).
    normalized = core.to(torch.float32)
    normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + config["rms_norm_eps"])
    norm_weight = checkpoint.tensor(f"{prefix}.linear_attn.norm.weight")
    normalized = norm_weight * normalized.to(torch.bfloat16)
    gated = (normalized * torch.nn.functional.silu(z.to(torch.float32))).to(torch.bfloat16)
    return linear(checkpoint, f"{prefix}.linear_attn.out_proj.weight", gated.reshape(-1), chunk)


def full_attention_position_zero(
    checkpoint: Checkpoint, prefix: str, input_: torch.Tensor, config: dict, chunk: int
) -> torch.Tensor:
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    projected = linear(checkpoint, f"{prefix}.self_attn.q_proj.weight", input_, chunk)
    projected = projected.reshape(heads, 2 * head_dim)
    gate = projected[:, head_dim:].reshape(-1)
    value = linear(checkpoint, f"{prefix}.self_attn.v_proj.weight", input_, chunk)
    value = value.reshape(kv_heads, head_dim)
    context = value.repeat_interleave(heads // kv_heads, dim=0).reshape(-1)
    context = (context * torch.sigmoid(gate)).to(torch.bfloat16)
    return linear(checkpoint, f"{prefix}.self_attn.o_proj.weight", context, chunk)


def moe(
    checkpoint: Checkpoint,
    prefix: str,
    input_: torch.Tensor,
    config: dict,
    chunk: int,
    fp8_chunk_blocks: int,
) -> tuple[torch.Tensor, list[dict]]:
    logits = linear(checkpoint, f"{prefix}.mlp.gate.weight", input_, chunk)
    probabilities = torch.softmax(logits.to(torch.float32), dim=-1)
    route_weights, route_experts = torch.topk(probabilities, config["num_experts_per_tok"])
    route_weights = (route_weights / route_weights.sum()).to(torch.bfloat16)
    routes = [
        {
            "expert": int(expert),
            "weight": float(weight.to(torch.float32)),
            "router_logit": float(logits[int(expert)].to(torch.float32)),
        }
        for expert, weight in zip(route_experts.tolist(), route_weights)
    ]

    # The upstream expert loop visits active expert IDs in ascending order and index_add_
    # materializes every addition into the BF16 accumulator.
    routed = torch.zeros_like(input_)
    weights_by_expert = {int(expert): weight for expert, weight in zip(route_experts.tolist(), route_weights)}
    for expert in sorted(weights_by_expert):
        expert_output = fp8_expert(
            checkpoint,
            f"{prefix}.mlp.experts.{expert}",
            input_,
            fp8_chunk_blocks,
        )
        routed.add_((expert_output * weights_by_expert[expert]).to(torch.bfloat16))

    shared = dense_expert(checkpoint, f"{prefix}.mlp.shared_expert", input_, chunk)
    shared_gate = torch.sigmoid(
        linear(checkpoint, f"{prefix}.mlp.shared_expert_gate.weight", input_, chunk)
    )
    return (routed + shared * shared_gate).to(torch.bfloat16), routes


def trace_hidden(hidden: torch.Tensor) -> dict:
    bits = hidden.contiguous().view(torch.uint16).numpy().tobytes()
    value = hidden.to(torch.float32)
    return {
        "sha256_bf16_le": hashlib.sha256(bits).hexdigest(),
        "first_16": value[:16].tolist(),
        "l2_norm": float(torch.linalg.vector_norm(value)),
        "sum": float(value.to(torch.float64).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--token", type=int, default=9419)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--matrix-row-chunk", type=int, default=256)
    parser.add_argument("--fp8-row-block-chunk", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = Checkpoint(args.model_dir)
    started = time.monotonic()
    hidden = checkpoint.row("model.embed_tokens.weight", args.token).to(torch.bfloat16)
    layer_traces = []
    all_routes = []

    with torch.no_grad():
        for layer, layer_type in enumerate(config["layer_types"]):
            prefix = f"model.layers.{layer}"
            layer_started = time.monotonic()
            normalized = zero_centered_norm(
                hidden,
                checkpoint.tensor(f"{prefix}.input_layernorm.weight"),
                config["rms_norm_eps"],
            )
            if layer_type == "linear_attention":
                mixed = delta_net(checkpoint, prefix, normalized, config, args.matrix_row_chunk)
            else:
                mixed = full_attention_position_zero(
                    checkpoint, prefix, normalized, config, args.matrix_row_chunk
                )
            after_mixer = (hidden + mixed).to(torch.bfloat16)
            normalized = zero_centered_norm(
                after_mixer,
                checkpoint.tensor(f"{prefix}.post_attention_layernorm.weight"),
                config["rms_norm_eps"],
            )
            moe_output, routes = moe(
                checkpoint,
                prefix,
                normalized,
                config,
                args.matrix_row_chunk,
                args.fp8_row_block_chunk,
            )
            hidden = (after_mixer + moe_output).to(torch.bfloat16)
            trace = trace_hidden(hidden)
            trace.update(
                {
                    "layer": layer,
                    "type": layer_type,
                    "seconds": time.monotonic() - layer_started,
                }
            )
            layer_traces.append(trace)
            all_routes.append(routes)
            print(
                f"layer {layer + 1:02d}/{config['num_hidden_layers']} {layer_type} "
                f"{trace['seconds']:.2f}s routes={[route['expert'] for route in routes]}",
                flush=True,
            )

        final_hidden = zero_centered_norm(
            hidden, checkpoint.tensor("model.norm.weight"), config["rms_norm_eps"]
        )
        logits = linear(checkpoint, "lm_head.weight", final_hidden, args.matrix_row_chunk)

    logits_f32 = logits.to(torch.float32)
    top_values, top_indices = torch.topk(logits_f32, 20)
    artifact = {
        "format_version": 1,
        "model_family": "qwen3_8",
        "checkpoint": "Qwen/Qwen3.8-2.4T-A95B-FP8",
        "scope": "one real token through all 92 base layers and the complete LM head",
        "generator": "tools/generate_qwen3_8_real_logits_oracle.py",
        "oracle": {
            "implementation": "independent streaming PyTorch implementation of upstream Transformers equations",
            "torch_version": torch.__version__,
            "threads": args.threads,
            "matrix_row_chunk": args.matrix_row_chunk,
            "fp8_row_block_chunk": args.fp8_row_block_chunk,
        },
        "token": args.token,
        "elapsed_seconds": time.monotonic() - started,
        "layer_traces": layer_traces,
        "routes_by_layer": all_routes,
        "final_hidden": trace_hidden(final_hidden),
        "logits": logits_f32.tolist(),
        "argmax": int(logits_f32.argmax()),
        "top_20": [
            {"token": int(token), "logit": float(value)}
            for token, value in zip(top_indices.tolist(), top_values.tolist())
        ],
    }
    serialized = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
