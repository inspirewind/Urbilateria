#!/usr/bin/env python3
"""Generate the independent Qwen3.8 tiny full-stack oracle.

The fixture is produced by the upstream Hugging Face Transformers implementation,
not by Urbilateria.  It covers both Qwen3.8 mixer kinds, zero-centered RMSNorm,
top-k routed and shared experts, the final norm, and the LM head.

Example:
  python tools/generate_qwen3_8_tiny_oracle.py \
    --output tests/fixtures/qwen3_8_transformers_tiny.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig


def tensor(values: list) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def copy(parameter: torch.Tensor, values: object) -> None:
    value = tensor(values).reshape(parameter.shape)
    parameter.copy_(value)


def install_experts(layer: torch.nn.Module) -> None:
    # gate_up_proj is [expert, gate-or-up, intermediate, hidden].
    copy(
        layer.mlp.experts.gate_up_proj,
        [
            [[0.5, 0.1], [0.3, -0.2]],
            [[-0.4, 0.6], [0.2, 0.5]],
        ],
    )
    copy(
        layer.mlp.experts.down_proj,
        [
            [[0.4], [-0.1]],
            [[-0.3], [0.7]],
        ],
    )
    copy(layer.mlp.gate.weight, [[0.8, -0.3], [-0.2, 0.7]])
    copy(layer.mlp.shared_expert.gate_proj.weight, [[0.25, -0.15]])
    copy(layer.mlp.shared_expert.up_proj.weight, [[0.4, 0.2]])
    copy(layer.mlp.shared_expert.down_proj.weight, [[0.6], [-0.2]])
    copy(layer.mlp.shared_expert_gate.weight, [[0.3, -0.4]])


def build_model() -> Qwen3_5MoeForCausalLM:
    config = Qwen3_5MoeTextConfig(
        vocab_size=3,
        hidden_size=2,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=1,
        linear_value_head_dim=1,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        moe_intermediate_size=1,
        shared_expert_intermediate_size=1,
        num_experts_per_tok=1,
        num_experts=2,
        full_attention_interval=2,
        partial_rotary_factor=1.0,
        rms_norm_eps=1e-6,
        output_hidden_states=True,
        output_router_logits=True,
        use_cache=False,
    )
    model = Qwen3_5MoeForCausalLM(config).eval()
    with torch.no_grad():
        copy(model.model.embed_tokens.weight, [[1.0, -0.5], [-0.25, 0.75], [0.4, 0.9]])

        linear = model.model.layers[0]
        copy(linear.input_layernorm.weight, [0.1, -0.05])
        copy(linear.post_attention_layernorm.weight, [0.02, -0.03])
        copy(linear.linear_attn.in_proj_qkv.weight, [[1.0, 0.0], [0.0, 1.0], [0.6, 0.4]])
        copy(linear.linear_attn.in_proj_z.weight, [[0.5, -0.25]])
        copy(linear.linear_attn.in_proj_b.weight, [[0.2, 0.1]])
        copy(linear.linear_attn.in_proj_a.weight, [[-0.1, 0.3]])
        copy(linear.linear_attn.conv1d.weight, [[0.25, 0.75], [-0.2, 0.8], [0.1, 0.9]])
        copy(linear.linear_attn.dt_bias, [0.1])
        copy(linear.linear_attn.A_log, [-0.2])
        copy(linear.linear_attn.norm.weight, [1.1])
        copy(linear.linear_attn.out_proj.weight, [[0.7], [-0.3]])
        install_experts(linear)

        full = model.model.layers[1]
        copy(full.input_layernorm.weight, [-0.04, 0.08])
        copy(full.post_attention_layernorm.weight, [0.06, -0.02])
        copy(
            full.self_attn.q_proj.weight,
            [[1.0, 0.0], [0.0, 1.0], [0.25, 0.0], [0.0, -0.25]],
        )
        copy(full.self_attn.k_proj.weight, [[0.8, 0.1], [-0.2, 0.9]])
        copy(full.self_attn.v_proj.weight, [[0.7, -0.3], [0.2, 0.6]])
        copy(full.self_attn.o_proj.weight, [[0.5, 0.1], [-0.2, 0.8]])
        copy(full.self_attn.q_norm.weight, [0.0, 0.0])
        copy(full.self_attn.k_norm.weight, [0.0, 0.0])
        install_experts(full)

        copy(model.model.norm.weight, [0.05, -0.05])
        copy(model.lm_head.weight, [[0.8, -0.2], [-0.4, 0.9], [0.3, 0.7]])
    return model


def floats(value: torch.Tensor) -> list[float]:
    return value.detach().cpu().to(torch.float32).reshape(-1).tolist()


def execute(model: Qwen3_5MoeForCausalLM) -> dict:
    layer_hidden_states: list[torch.Tensor] = []

    def capture_layer(_module: torch.nn.Module, _inputs: tuple, output: object) -> None:
        value = output[0] if isinstance(output, tuple) else output
        assert isinstance(value, torch.Tensor)
        layer_hidden_states.append(value)

    handles = [layer.register_forward_hook(capture_layer) for layer in model.model.layers]
    with torch.no_grad():
        output = model(
            input_ids=torch.tensor([[0]], dtype=torch.long),
            output_hidden_states=True,
            output_router_logits=True,
            use_cache=False,
            return_dict=True,
        )
    for handle in handles:
        handle.remove()

    routes: list[dict] = []
    for logits in output.router_logits:
        probabilities = torch.softmax(logits.to(torch.float32), dim=-1)
        weights, experts = torch.topk(probabilities, 1, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        routes.append(
            {
                "router_logits": floats(logits),
                "experts": experts.reshape(-1).tolist(),
                "weights": floats(weights),
            }
        )

    return {
        "embedding": floats(output.hidden_states[0]),
        "layer_hidden_states": [floats(value) for value in layer_hidden_states],
        "final_hidden": floats(output.hidden_states[-1]),
        "logits": floats(output.logits),
        "argmax": int(output.logits.argmax(dim=-1).item()),
        "routes": routes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = build_model()
    artifact = {
        "format_version": 1,
        "model_family": "qwen3_8",
        "scope": "one token; Gated DeltaNet layer; full GQA layer; routed/shared MoE; LM head",
        "generator": "tools/generate_qwen3_8_tiny_oracle.py",
        "oracle": {
            "implementation": "transformers.Qwen3_5MoeForCausalLM",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        },
        "token": 0,
        "geometry": {
            "hidden_size": 2,
            "layer_types": model.config.layer_types,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "vocab_size": 3,
        },
        "expected": execute(model),
        "expected_bf16": execute(build_model().to(torch.bfloat16)),
    }
    serialized = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
