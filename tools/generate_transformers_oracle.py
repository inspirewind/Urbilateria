#!/usr/bin/env python3
"""Generate the official-class tiny GLM oracle used to close the M1 parity gate.

Run this with Transformers 5.12.0 on ``PYTHONPATH``. The model is tiny, dense-only, CPU/F32,
and uses deterministic in-source weights shared with ``generate_tiny_oracle.py``. Its DSA
top-k covers the complete three-token sequence, so the known v5.12.0 indexer-RoPE bug cannot
change attention output; DSA itself remains a separate, later gate.
"""

import json

import torch
import transformers
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM


EXPECTED_TRANSFORMERS = "5.12.0"
HIDDEN = 6
VOCAB = 7
NUM_HEADS = 2
Q_LORA = 5
KV_LORA = 3
QK_NOPE = 2
QK_ROPE = 4
V_HEAD = 3
INTERMEDIATE = 5
TOKENS = [1, 5, 2]


def values(count: int, seed: int) -> torch.Tensor:
    raw = [((index * 17 + seed * 13) % 29) - 14 for index in range(count)]
    return torch.tensor(raw, dtype=torch.float32) / 100.0


def matrix(rows: int, columns: int, seed: int) -> torch.Tensor:
    return values(rows * columns, seed).reshape(rows, columns)


def norm(length: int, seed: int) -> torch.Tensor:
    return 1.0 + values(length, seed) / 10.0


def assign(model: GlmMoeDsaForCausalLM, name: str, value: torch.Tensor) -> None:
    parameters = dict(model.named_parameters())
    if name not in parameters:
        raise KeyError(f"official model has no parameter {name!r}")
    target = parameters[name]
    if target.shape != value.shape:
        raise ValueError(f"{name}: official shape {tuple(target.shape)} != {tuple(value.shape)}")
    target.copy_(value)


def build_model() -> GlmMoeDsaForCausalLM:
    if transformers.__version__ != EXPECTED_TRANSFORMERS:
        raise RuntimeError(
            f"requires transformers=={EXPECTED_TRANSFORMERS}, got {transformers.__version__}"
        )
    config = GlmMoeDsaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        moe_intermediate_size=3,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts=4,
        routed_scaling_factor=2.5,
        kv_lora_rank=KV_LORA,
        q_lora_rank=Q_LORA,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_HEAD,
        qk_nope_head_dim=QK_NOPE,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=16,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=6,
        tie_word_embeddings=False,
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "default"},
        mlp_layer_types=["dense"],
        attention_bias=False,
        attention_dropout=0.0,
        index_topk=len(TOKENS),
        index_head_dim=QK_ROPE,
        index_n_heads=2,
        mlp_bias=False,
        head_dim=QK_ROPE,
        first_k_dense_replace=1,
        layer_types=["deepseek_sparse_attention"],
        indexer_types=["full"],
        dtype="float32",
    )
    config._attn_implementation = "eager"
    model = GlmMoeDsaForCausalLM(config).to(dtype=torch.float32, device="cpu")
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        assign(model, "model.embed_tokens.weight", matrix(VOCAB, HIDDEN, 1))
        assign(model, "model.layers.0.input_layernorm.weight", norm(HIDDEN, 2))
        assign(model, "model.layers.0.self_attn.q_a_proj.weight", matrix(Q_LORA, HIDDEN, 3))
        assign(model, "model.layers.0.self_attn.q_a_layernorm.weight", norm(Q_LORA, 4))
        assign(
            model,
            "model.layers.0.self_attn.q_b_proj.weight",
            matrix(NUM_HEADS * (QK_NOPE + QK_ROPE), Q_LORA, 5),
        )
        assign(
            model,
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            matrix(KV_LORA + QK_ROPE, HIDDEN, 6),
        )
        assign(model, "model.layers.0.self_attn.kv_a_layernorm.weight", norm(KV_LORA, 7))
        assign(
            model,
            "model.layers.0.self_attn.kv_b_proj.weight",
            matrix(NUM_HEADS * (QK_NOPE + V_HEAD), KV_LORA, 8),
        )
        assign(
            model,
            "model.layers.0.self_attn.o_proj.weight",
            matrix(HIDDEN, NUM_HEADS * V_HEAD, 9),
        )
        assign(model, "model.layers.0.post_attention_layernorm.weight", norm(HIDDEN, 10))
        assign(model, "model.layers.0.mlp.gate_proj.weight", matrix(INTERMEDIATE, HIDDEN, 11))
        assign(model, "model.layers.0.mlp.up_proj.weight", matrix(INTERMEDIATE, HIDDEN, 12))
        assign(model, "model.layers.0.mlp.down_proj.weight", matrix(HIDDEN, INTERMEDIATE, 13))
        assign(model, "model.norm.weight", norm(HIDDEN, 14))
        assign(model, "lm_head.weight", matrix(VOCAB, HIDDEN, 15))
        assign(
            model,
            "model.layers.0.self_attn.indexer.k_norm.weight",
            torch.ones(QK_ROPE, dtype=torch.float32),
        )
    return model


def rows(tensor: torch.Tensor) -> list[list[float]]:
    return [[float(value) for value in row] for row in tensor.detach().cpu()]


def assert_complete_topk(topk: torch.Tensor, key_count: int) -> None:
    if topk.shape[-1] != key_count:
        raise AssertionError(f"indexer returned top-{topk.shape[-1]}, expected all {key_count} keys")
    expected = torch.arange(key_count, dtype=topk.dtype).expand(*topk.shape[:-1], key_count)
    torch.testing.assert_close(topk.sort(dim=-1).values.cpu(), expected, rtol=0.0, atol=0.0)


def main() -> None:
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    model = build_model()
    captured: list[torch.Tensor] = []
    captured_topk: list[torch.Tensor] = []

    def capture_layer(_module, _inputs, output) -> None:
        captured.append(output[0].detach().clone())

    hook = model.model.layers[0].register_forward_hook(capture_layer)
    indexer_hook = model.model.layers[0].self_attn.indexer.register_forward_hook(
        lambda _module, _inputs, output: captured_topk.append(output.detach().clone())
    )
    token_tensor = torch.tensor([TOKENS], dtype=torch.long)
    with torch.inference_mode():
        full = model(input_ids=token_tensor, use_cache=False, logits_to_keep=0)
    full_layer = captured.pop()[0]
    full_logits = full.logits[0]
    assert_complete_topk(captured_topk.pop(), len(TOKENS))

    incremental_layer = []
    incremental_logits = []
    past = None
    with torch.inference_mode():
        for position, token in enumerate(TOKENS):
            step = model(
                input_ids=torch.tensor([[token]], dtype=torch.long),
                past_key_values=past,
                use_cache=True,
                logits_to_keep=0,
            )
            past = step.past_key_values
            incremental_layer.append(captured.pop()[0, 0])
            incremental_logits.append(step.logits[0, 0])
            assert_complete_topk(captured_topk.pop(), position + 1)
    hook.remove()
    indexer_hook.remove()
    incremental_layer = torch.stack(incremental_layer)
    incremental_logits = torch.stack(incremental_logits)
    torch.testing.assert_close(full_layer, incremental_layer, rtol=0.0, atol=3e-8)
    torch.testing.assert_close(full_logits, incremental_logits, rtol=0.0, atol=3e-8)

    print(
        json.dumps(
            {
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "tokens": TOKENS,
                "full_layer_hidden": rows(full_layer),
                "full_logits": rows(full_logits),
                "incremental_layer_hidden": rows(incremental_layer),
                "incremental_logits": rows(incremental_logits),
                "argmax": [int(value) for value in full_logits.argmax(dim=-1)],
                "indexer_topk_covers_all": True,
                "max_layer_parity_error": float((full_layer - incremental_layer).abs().max()),
                "max_logits_parity_error": float((full_logits - incremental_logits).abs().max()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
