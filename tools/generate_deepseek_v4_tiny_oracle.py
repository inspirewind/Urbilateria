#!/usr/bin/env python3
"""Generate a tiny independent DeepSeek-V4 base-layer oracle.

This script uses only the Python standard library and does not import Urbilateria, PyTorch,
Transformers, or the checkpoint's reference implementation.  It intentionally covers exactly
one token through one ratio-0 layer:

    HC attention pre -> local attention -> HC post
    HC FFN pre -> hash-routed MoE -> HC post -> HC head -> logits

The tensor values and all rounding points are deterministic.  Matrices are stored as plain F32
in the tiny fixture; BF16 vectors and intermediate materializations follow the documented model
dtype contract.  Compression, indexing, DSpark, and quantized matrix kernels are outside this
oracle's scope.
"""

import json
import math
import struct


HIDDEN = 2
HC_MULT = 2
MIX_COUNT = (2 + HC_MULT) * HC_MULT
VOCAB = 4
TOKEN = 2
HEADS = 1
HEAD_DIM = 2
Q_LORA = 2
O_GROUPS = 1
O_RANK = 2
INTERMEDIATE = 2
EXPERTS = 3
TOP_K = 2
ROUTE_SCALE = 1.5
NORM_EPS = 1e-6
HC_EPS = 1e-6
SINKHORN_ITERS = 3
SWIGLU_LIMIT = 10.0


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def add(left: float, right: float) -> float:
    return f32(f32(left) + f32(right))


def mul(left: float, right: float) -> float:
    return f32(f32(left) * f32(right))


def div(left: float, right: float) -> float:
    return f32(f32(left) / f32(right))


def bf16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded = (bits + rounding_bias) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def bf16_vec(values: list[float]) -> list[float]:
    return [bf16(value) for value in values]


def values(count: int, seed: int, denominator: float = 100.0) -> list[float]:
    return [f32((((index * 17 + seed * 13) % 31) - 15) / denominator) for index in range(count)]


def matrix(rows: int, columns: int, seed: int, denominator: float = 100.0) -> list[float]:
    return values(rows * columns, seed, denominator)


def matvec(weight: list[float], rows: int, columns: int, input_: list[float]) -> list[float]:
    assert len(weight) == rows * columns and len(input_) == columns
    # Accumulate reference dot products in Python binary64, then materialize F32.
    return [
        f32(sum(float(weight[row * columns + column]) * float(input_[column])
                for column in range(columns)))
        for row in range(rows)
    ]


def rms_norm(input_: list[float], weight: list[float]) -> list[float]:
    mean_square = sum(float(value) * float(value) for value in input_) / len(input_)
    inverse = f32(1.0 / math.sqrt(mean_square + NORM_EPS))
    return bf16_vec([mul(mul(value, inverse), scale) for value, scale in zip(input_, weight)])


def unit_rms_norm(input_: list[float]) -> list[float]:
    # This unweighted q normalization operates directly on a BF16 tensor in the standalone
    # model, unlike the explicit-FP32 RMSNorm module above.
    square_sum = f32(0.0)
    for value in input_:
        square_sum = add(square_sum, bf16(mul(value, value)))
    mean_square = bf16(div(square_sum, len(input_)))
    shifted = bf16(add(mean_square, NORM_EPS))
    inverse = bf16(f32(1.0 / math.sqrt(shifted)))
    return [bf16(mul(value, inverse)) for value in input_]


def sigmoid(value: float) -> float:
    value = f32(value)
    if value >= 0.0:
        return div(1.0, add(1.0, f32(math.exp(-value))))
    exponential = f32(math.exp(value))
    return div(exponential, add(1.0, exponential))


def softplus(value: float) -> float:
    value = f32(value)
    if value > 0.0:
        return f32(value + math.log1p(math.exp(-value)))
    return f32(math.log1p(math.exp(value)))


def normalize_rows(values_: list[float], width: int) -> None:
    for start in range(0, len(values_), width):
        denominator = f32(0.0)
        for value in values_[start:start + width]:
            denominator = add(denominator, value)
        denominator = add(denominator, HC_EPS)
        for index in range(start, start + width):
            values_[index] = div(values_[index], denominator)


def normalize_columns(values_: list[float], width: int) -> None:
    for column in range(width):
        denominator = f32(0.0)
        for row in range(width):
            denominator = add(denominator, values_[row * width + column])
        denominator = add(denominator, HC_EPS)
        for row in range(width):
            index = row * width + column
            values_[index] = div(values_[index], denominator)


def hc_pre(
    hidden: list[float], function: list[float], scale: list[float], base: list[float]
) -> tuple[list[float], dict[str, list[float]]]:
    width = HIDDEN * HC_MULT
    mean_square = sum(float(value) * float(value) for value in hidden) / width
    reciprocal_rms = f32(1.0 / math.sqrt(mean_square + NORM_EPS))
    mixes = [mul(value, reciprocal_rms) for value in matvec(function, MIX_COUNT, width, hidden)]
    pre = [add(sigmoid(add(mul(mixes[i], scale[0]), base[i])), HC_EPS)
           for i in range(HC_MULT)]
    post = [mul(2.0, sigmoid(add(mul(mixes[HC_MULT + i], scale[1]),
                                base[HC_MULT + i])))
            for i in range(HC_MULT)]
    offset = 2 * HC_MULT
    combination = [add(mul(mixes[offset + i], scale[2]), base[offset + i])
                   for i in range(HC_MULT * HC_MULT)]
    for start in range(0, len(combination), HC_MULT):
        row = combination[start:start + HC_MULT]
        maximum = max(row)
        denominator = f32(0.0)
        for index, value in enumerate(row):
            row[index] = f32(math.exp(f32(value - maximum)))
            denominator = add(denominator, row[index])
        for index, value in enumerate(row):
            combination[start + index] = add(div(value, denominator), HC_EPS)
    normalize_columns(combination, HC_MULT)
    for _ in range(1, SINKHORN_ITERS):
        normalize_rows(combination, HC_MULT)
        normalize_columns(combination, HC_MULT)

    collapsed = []
    for feature in range(HIDDEN):
        value = sum(float(pre[copy]) * float(hidden[copy * HIDDEN + feature])
                    for copy in range(HC_MULT))
        collapsed.append(bf16(value))
    return collapsed, {"pre": pre, "post": post, "combination": combination}


def hc_post(branch: list[float], residual: list[float], mix: dict[str, list[float]]) -> list[float]:
    output = []
    for target in range(HC_MULT):
        for feature in range(HIDDEN):
            value = float(mix["post"][target]) * float(branch[feature])
            value += sum(
                float(mix["combination"][target * HC_MULT + source])
                * float(residual[source * HIDDEN + feature])
                for source in range(HC_MULT)
            )
            output.append(bf16(value))
    return output


def sparse_attention(query: list[float], key_value: list[float], sink: float) -> list[float]:
    scale = f32(1.0 / math.sqrt(HEAD_DIM))
    dot = f32(0.0)
    for left, right in zip(query, key_value):
        dot = add(dot, mul(left, right))
    score = mul(dot, scale)
    maximum = score
    probability = div(1.0, add(1.0, f32(math.exp(f32(sink - maximum)))))
    return bf16_vec([mul(probability, value) for value in key_value])


def bounded_swiglu(gate: list[float], up: list[float]) -> list[float]:
    output = []
    for gate_value, up_value in zip(gate, up):
        gate_value = min(gate_value, SWIGLU_LIMIT)
        up_value = max(-SWIGLU_LIMIT, min(up_value, SWIGLU_LIMIT))
        silu = div(gate_value, add(1.0, f32(math.exp(-gate_value))))
        output.append(mul(silu, up_value))
    return output


def expert(
    input_: list[float],
    prefix: str,
    tensors: dict[str, dict],
    route_weight: float | None = None,
) -> list[float]:
    def weight(name: str) -> list[float]:
        return tensors[f"{prefix}.{name}.weight"]["values"]

    gate = matvec(weight("w1"), INTERMEDIATE, HIDDEN, input_)
    up = matvec(weight("w3"), INTERMEDIATE, HIDDEN, input_)
    activated = bounded_swiglu(gate, up)
    if route_weight is not None:
        activated = [mul(route_weight, value) for value in activated]
    activated = bf16_vec(activated)
    return matvec(weight("w2"), HIDDEN, INTERMEDIATE, activated)


def add_tensor(
    tensors: dict[str, dict], name: str, dtype: str, shape: list[int], tensor_values: list
) -> None:
    count = math.prod(shape)
    assert len(tensor_values) == count
    if dtype == "BF16":
        tensor_values = bf16_vec(tensor_values)
    elif dtype == "F32":
        tensor_values = [f32(value) for value in tensor_values]
    tensors[name] = {"dtype": dtype, "shape": shape, "values": tensor_values}


def build_weights() -> dict[str, dict]:
    tensors: dict[str, dict] = {}
    add_tensor(tensors, "embed.weight", "F32", [VOCAB, HIDDEN], matrix(VOCAB, HIDDEN, 1, 20.0))
    for sublayer, seed in [("attn", 2), ("ffn", 7)]:
        add_tensor(tensors, f"layers.0.hc_{sublayer}_fn", "F32",
                   [MIX_COUNT, HC_MULT * HIDDEN], matrix(MIX_COUNT, HC_MULT * HIDDEN, seed))
        add_tensor(tensors, f"layers.0.hc_{sublayer}_base", "F32", [MIX_COUNT],
                   values(MIX_COUNT, seed + 1, 50.0))
        add_tensor(tensors, f"layers.0.hc_{sublayer}_scale", "F32", [3],
                   [0.75, 0.625, 0.5])
    for name, tensor_values in [
        ("layers.0.attn_norm.weight", [1.0, 0.875]),
        ("layers.0.ffn_norm.weight", [0.9375, 1.0625]),
        ("layers.0.attn.q_norm.weight", [1.0, 0.8125]),
        ("layers.0.attn.kv_norm.weight", [0.875, 1.0]),
        ("norm.weight", [1.0, 0.9375]),
    ]:
        add_tensor(tensors, name, "BF16", [2], tensor_values)
    add_tensor(tensors, "layers.0.attn.attn_sink", "F32", [1], [-0.125])
    for name, seed in [
        ("layers.0.attn.wq_a.weight", 11),
        ("layers.0.attn.wq_b.weight", 12),
        ("layers.0.attn.wkv.weight", 13),
        ("layers.0.attn.wo_a.weight", 14),
        ("layers.0.attn.wo_b.weight", 15),
    ]:
        add_tensor(tensors, name, "F32", [2, 2], matrix(2, 2, seed, 20.0))
    add_tensor(tensors, "layers.0.ffn.gate.weight", "F32", [EXPERTS, HIDDEN],
               matrix(EXPERTS, HIDDEN, 16, 25.0))
    # Token 2 intentionally routes to experts 2 then 0.  Other rows keep the table valid.
    add_tensor(tensors, "layers.0.ffn.gate.tid2eid", "I64", [VOCAB, TOP_K],
               [0, 1, 1, 2, 2, 0, 0, 2])
    for expert_id in range(EXPERTS):
        prefix = f"layers.0.ffn.experts.{expert_id}"
        for projection, seed in [("w1", 20), ("w2", 21), ("w3", 22)]:
            add_tensor(tensors, f"{prefix}.{projection}.weight", "F32", [2, 2],
                       matrix(2, 2, seed + 3 * expert_id, 20.0))
    for projection, seed in [("w1", 30), ("w2", 31), ("w3", 32)]:
        add_tensor(tensors, f"layers.0.ffn.shared_experts.{projection}.weight", "F32", [2, 2],
                   matrix(2, 2, seed, 20.0))
    add_tensor(tensors, "hc_head_fn", "F32", [HC_MULT, HC_MULT * HIDDEN],
               matrix(HC_MULT, HC_MULT * HIDDEN, 35))
    add_tensor(tensors, "hc_head_base", "F32", [HC_MULT], [0.0625, -0.03125])
    add_tensor(tensors, "hc_head_scale", "F32", [1], [0.75])
    add_tensor(tensors, "head.weight", "F32", [VOCAB, HIDDEN], matrix(VOCAB, HIDDEN, 36, 20.0))
    return tensors


def main() -> None:
    tensors = build_weights()
    tensor = lambda name: tensors[name]["values"]

    embedding = tensor("embed.weight")[TOKEN * HIDDEN:(TOKEN + 1) * HIDDEN]
    expanded = embedding * HC_MULT
    attn_collapsed, attn_mix = hc_pre(
        expanded,
        tensor("layers.0.hc_attn_fn"),
        tensor("layers.0.hc_attn_scale"),
        tensor("layers.0.hc_attn_base"),
    )
    attn_input = rms_norm(attn_collapsed, tensor("layers.0.attn_norm.weight"))
    qr = matvec(tensor("layers.0.attn.wq_a.weight"), Q_LORA, HIDDEN, attn_input)
    qr = rms_norm(qr, tensor("layers.0.attn.q_norm.weight"))
    query = unit_rms_norm(matvec(tensor("layers.0.attn.wq_b.weight"), HEAD_DIM, Q_LORA, qr))
    # position=0 ratio-0 paired RoPE is identity followed by a BF16 materialization.
    query = bf16_vec(query)
    key_value = matvec(tensor("layers.0.attn.wkv.weight"), HEAD_DIM, HIDDEN, attn_input)
    key_value = bf16_vec(rms_norm(key_value, tensor("layers.0.attn.kv_norm.weight")))
    attention_value = sparse_attention(query, key_value, tensor("layers.0.attn.attn_sink")[0])
    attention_value = bf16_vec(attention_value)  # inverse position-0 RoPE materialization
    attention_low_rank = bf16_vec(matvec(
        tensor("layers.0.attn.wo_a.weight"), O_RANK, HEAD_DIM, attention_value
    ))
    attention_branch = matvec(
        tensor("layers.0.attn.wo_b.weight"), HIDDEN, O_RANK, attention_low_rank
    )
    after_attention = hc_post(attention_branch, expanded, attn_mix)

    ffn_collapsed, ffn_mix = hc_pre(
        after_attention,
        tensor("layers.0.hc_ffn_fn"),
        tensor("layers.0.hc_ffn_scale"),
        tensor("layers.0.hc_ffn_base"),
    )
    ffn_input = rms_norm(ffn_collapsed, tensor("layers.0.ffn_norm.weight"))
    router_logits = matvec(tensor("layers.0.ffn.gate.weight"), EXPERTS, HIDDEN, ffn_input)
    hash_experts = tensor("layers.0.ffn.gate.tid2eid")[TOKEN * TOP_K:(TOKEN + 1) * TOP_K]
    scores = [f32(math.sqrt(softplus(value))) for value in router_logits]
    denominator = f32(0.0)
    for expert_id in hash_experts:
        denominator = add(denominator, scores[expert_id])
    routes = [
        {
            "expert": expert_id,
            "weight": mul(div(scores[expert_id], denominator), ROUTE_SCALE),
            "selection_score": scores[expert_id],
        }
        for expert_id in hash_experts
    ]
    shared = expert(ffn_input, "layers.0.ffn.shared_experts", tensors)
    routed = []
    for route in routes:
        output = expert(
            ffn_input,
            f"layers.0.ffn.experts.{route['expert']}",
            tensors,
            route["weight"],
        )
        routed.append(output)
    # Preserve the public route/table order, but match the reference's expert-ID execution order.
    moe_branch = [f32(0.0)] * HIDDEN
    for route_index in sorted(range(len(routes)), key=lambda index: routes[index]["expert"]):
        output = routed[route_index]
        moe_branch = [add(value, contribution)
                      for value, contribution in zip(moe_branch, output)]
    moe_branch = [add(value, contribution)
                  for value, contribution in zip(moe_branch, shared)]
    moe_branch = bf16_vec(moe_branch)
    after_ffn = hc_post(moe_branch, after_attention, ffn_mix)

    head_fn = tensor("hc_head_fn")
    head_base = tensor("hc_head_base")
    head_scale = tensor("hc_head_scale")[0]
    width = HC_MULT * HIDDEN
    mean_square = sum(float(value) * float(value) for value in after_ffn) / width
    inverse = f32(1.0 / math.sqrt(mean_square + NORM_EPS))
    head_mixes = matvec(head_fn, HC_MULT, width, after_ffn)
    head_weights = [add(sigmoid(add(mul(mul(head_mixes[i], inverse), head_scale), head_base[i])), HC_EPS)
                    for i in range(HC_MULT)]
    head_hidden = bf16_vec([
        sum(float(head_weights[copy]) * float(after_ffn[copy * HIDDEN + feature])
            for copy in range(HC_MULT))
        for feature in range(HIDDEN)
    ])
    final_hidden = rms_norm(head_hidden, tensor("norm.weight"))
    logits = matvec(tensor("head.weight"), VOCAB, HIDDEN, final_hidden)

    result = {
        "format_version": 1,
        "generator": "tools/generate_deepseek_v4_tiny_oracle.py",
        "model_family": "deepseek_v4",
        "scope": "single token; one ratio-0 layer; HC attention; hash MoE; HC head",
        "geometry": {
            "hidden_size": HIDDEN,
            "hc_mult": HC_MULT,
            "mix_count": MIX_COUNT,
            "vocab_size": VOCAB,
            "num_attention_heads": HEADS,
            "head_dim": HEAD_DIM,
            "q_lora_rank": Q_LORA,
            "o_groups": O_GROUPS,
            "o_lora_rank": O_RANK,
            "moe_intermediate_size": INTERMEDIATE,
            "n_routed_experts": EXPERTS,
            "num_experts_per_tok": TOP_K,
            "routed_scaling_factor": ROUTE_SCALE,
            "rms_norm_eps": NORM_EPS,
            "hc_eps": HC_EPS,
            "hc_sinkhorn_iters": SINKHORN_ITERS,
            "swiglu_limit": SWIGLU_LIMIT,
        },
        "token": TOKEN,
        "weights": tensors,
        "expected": {
            "embedding": embedding,
            "expanded_hidden": expanded,
            "attention_hc_collapsed": attn_collapsed,
            "attention_hc_mix": attn_mix,
            "attention_input": attn_input,
            "query": query,
            "key_value": key_value,
            "attention_value": attention_value,
            "attention_low_rank": attention_low_rank,
            "attention_branch": attention_branch,
            "after_attention": after_attention,
            "ffn_hc_collapsed": ffn_collapsed,
            "ffn_hc_mix": ffn_mix,
            "ffn_input": ffn_input,
            "router_logits": router_logits,
            "routes": routes,
            "shared_expert": shared,
            "routed_experts": routed,
            "moe_branch": moe_branch,
            "after_ffn": after_ffn,
            "head_weights": head_weights,
            "head_hidden": head_hidden,
            "final_hidden": final_hidden,
            "logits": logits,
            "argmax": max(range(VOCAB), key=lambda index: logits[index]),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
