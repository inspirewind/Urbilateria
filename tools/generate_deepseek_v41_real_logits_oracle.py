#!/usr/bin/env python3
"""Independent, slow position-zero oracle for native DeepSeek-V4.1-Flash.

This follows DeepSeek's published ``inference/model.py`` and ``kernel.py`` equations, but reads
the unconverted Hugging Face shards directly.  The stock safetensors Python package cannot open
shards containing F8_E8M0, so the small reader below parses the format without using Urbilateria.
Only one decoder layer and seven experts are materialized at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sympy import isprime
from tokenizers import Regex, Tokenizer, normalizers


MX_BLOCK = 32
FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def e4m3_table() -> torch.Tensor:
    values = []
    for code in range(256):
        sign = 1.0 if code & 0x80 == 0 else -1.0
        exponent = (code >> 3) & 0x0F
        mantissa = code & 7
        if exponent == 0:
            value = sign * mantissa * 2.0**-9
        elif exponent == 15 and mantissa == 7:
            value = math.nan
        else:
            value = sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        values.append(value)
    return torch.tensor(values, dtype=torch.float32)


E4M3_VALUES = e4m3_table()


@dataclass(frozen=True)
class TensorSpec:
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


class Checkpoint:
    def __init__(self, directory: Path):
        self.directory = directory
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.headers: dict[str, tuple[int, dict]] = {}
        self.specs: dict[str, TensorSpec] = {}

    def spec(self, name: str) -> TensorSpec:
        cached = self.specs.get(name)
        if cached is not None:
            return cached
        shard_name = self.weight_map[name]
        if shard_name not in self.headers:
            shard = self.directory / shard_name
            with shard.open("rb") as handle:
                header_len = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_len))
            self.headers[shard_name] = (8 + header_len, header)
        data_start, header = self.headers[shard_name]
        item = header[name]
        begin, end = item["data_offsets"]
        spec = TensorSpec(
            self.directory / shard_name,
            item["dtype"],
            tuple(item["shape"]),
            data_start + begin,
            end - begin,
        )
        self.specs[name] = spec
        return spec

    def raw(self, name: str, first: int = 0, count: int | None = None) -> np.ndarray:
        spec = self.spec(name)
        item_bytes = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}[spec.dtype]
        count = spec.elements - first if count is None else count
        if first < 0 or count < 0 or first + count > spec.elements:
            raise IndexError((name, first, count, spec.elements))
        dtype = {2: np.uint16, 4: np.float32, 1: np.uint8}[item_bytes]
        return np.asarray(
            np.memmap(spec.shard, mode="r", dtype=dtype, offset=spec.offset + first * item_bytes, shape=(count,))
        ).copy()

    def values(self, name: str) -> torch.Tensor:
        spec = self.spec(name)
        raw = self.raw(name)
        if spec.dtype == "BF16":
            bits = torch.from_numpy(raw.astype(np.uint16)).to(torch.int32) << 16
            value = bits.view(torch.float32)
        elif spec.dtype == "F32":
            value = torch.from_numpy(raw.astype(np.float32))
        elif spec.dtype == "F8_E4M3":
            value = E4M3_VALUES[torch.from_numpy(raw.astype(np.uint8)).long()]
        elif spec.dtype == "F8_E8M0":
            value = decode_e8m0(torch.from_numpy(raw.astype(np.uint8)))
        else:
            raise TypeError(f"{name}: values() does not unpack {spec.dtype}")
        return value.reshape(spec.shape)

    def rows_f32(self, name: str, start: int, count: int) -> torch.Tensor:
        spec = self.spec(name)
        assert len(spec.shape) == 2
        columns = spec.shape[1]
        raw = self.raw(name, start * columns, count * columns)
        if spec.dtype == "BF16":
            bits = torch.from_numpy(raw.astype(np.uint16)).to(torch.int32) << 16
            return bits.view(torch.float32).reshape(count, columns)
        if spec.dtype == "F32":
            return torch.from_numpy(raw.astype(np.float32)).reshape(count, columns)
        if spec.dtype == "F8_E4M3":
            return E4M3_VALUES[torch.from_numpy(raw.astype(np.uint8)).long()].reshape(count, columns)
        raise TypeError(f"unsupported matrix dtype {spec.dtype} for {name}")

    def row_bf16(self, name: str, row: int) -> torch.Tensor:
        return self.rows_f32(name, row, 1)[0].to(torch.bfloat16)


def decode_e8m0(codes: torch.Tensor) -> torch.Tensor:
    exponents = codes.to(torch.int32) - 127
    return torch.ldexp(torch.ones(codes.shape, dtype=torch.float32), exponents)


def bf16(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16)


def dynamic_e4m3(value: torch.Tensor) -> torch.Tensor:
    groups = value.to(torch.float32).reshape(-1, MX_BLOCK)
    maximum = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(maximum / 448.0)))
    quantized = (groups / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (quantized.to(torch.float32) * scale).reshape(value.shape)


def dynamic_fp4(value: torch.Tensor, block: int, e4_scale: bool) -> torch.Tensor:
    groups = value.to(torch.float32).reshape(-1, block)
    minimum = 6.0 * (2.0**-9 if e4_scale else 2.0**-126)
    maximum = groups.abs().amax(dim=-1, keepdim=True).clamp_min(minimum)
    raw_scale = maximum / 6.0
    if e4_scale:
        scale = raw_scale.to(torch.float8_e4m3fn).to(torch.float32)
    else:
        scale = torch.pow(2.0, torch.ceil(torch.log2(raw_scale)))
    normalized = (groups / scale).clamp(-6.0, 6.0)
    distances = (normalized.unsqueeze(-1) - FP4_VALUES).abs()
    codes = distances.argmin(dim=-1)
    return (FP4_VALUES[codes] * scale).reshape(value.shape).to(torch.bfloat16)


def plain_linear(
    checkpoint: Checkpoint,
    name: str,
    input_: torch.Tensor,
    row_chunk: int,
    output_bf16: bool = True,
) -> torch.Tensor:
    spec = checkpoint.spec(name)
    rows, columns = spec.shape
    assert input_.numel() == columns and spec.dtype in ("BF16", "F32")
    x = input_.to(torch.float32)
    output = torch.empty(rows, dtype=torch.float32)
    for start in range(0, rows, row_chunk):
        count = min(row_chunk, rows - start)
        output[start : start + count] = checkpoint.rows_f32(name, start, count) @ x
    return output.to(torch.bfloat16) if output_bf16 else output


def mx8_linear(checkpoint: Checkpoint, name: str, input_: torch.Tensor, row_blocks: int) -> torch.Tensor:
    spec = checkpoint.spec(name)
    rows, columns = spec.shape
    assert spec.dtype == "F8_E4M3" and input_.numel() == columns
    scale_name = name.removesuffix(".weight") + ".scale"
    scales = checkpoint.values(scale_name).to(torch.float32)
    assert tuple(scales.shape) == (math.ceil(rows / 32), math.ceil(columns / 32))
    x = dynamic_e4m3(input_).to(torch.float32)
    output = torch.empty(rows, dtype=torch.bfloat16)
    chunk = row_blocks * 32
    for start in range(0, rows, chunk):
        count = min(chunk, rows - start)
        encoded = checkpoint.raw(name, start * columns, count * columns)
        weight = E4M3_VALUES[torch.from_numpy(encoded).long()].reshape(count, columns)
        block_scales = scales[start // 32 : math.ceil((start + count) / 32)]
        block_scales = block_scales.repeat_interleave(32, 0)[:count].repeat_interleave(32, 1)[:, :columns]
        output[start : start + count] = ((weight * block_scales) @ x).to(torch.bfloat16)
    return output


def mx8_rows_no_activation_quant(
    checkpoint: Checkpoint, name: str, start: int, count: int, input_: torch.Tensor
) -> torch.Tensor:
    """Dequantize selected MXFP8 rows and apply them to BF16 input, as convert.py does for wo_a."""
    spec = checkpoint.spec(name)
    rows, columns = spec.shape
    assert spec.dtype == "F8_E4M3" and start % 32 == 0 and count % 32 == 0
    assert start + count <= rows and input_.numel() == columns
    scales = checkpoint.values(name.removesuffix(".weight") + ".scale").to(torch.float32)
    encoded = checkpoint.raw(name, start * columns, count * columns)
    weight = E4M3_VALUES[torch.from_numpy(encoded).long()].reshape(count, columns)
    block_scales = scales[start // 32 : (start + count) // 32]
    block_scales = block_scales.repeat_interleave(32, 0).repeat_interleave(32, 1)[:, :columns]
    return ((weight * block_scales) @ input_.to(torch.float32)).to(torch.bfloat16)


def mx4_linear(checkpoint: Checkpoint, name: str, input_: torch.Tensor, row_chunk: int) -> torch.Tensor:
    spec = checkpoint.spec(name)
    rows, packed_columns = spec.shape
    columns = packed_columns * 2
    assert spec.dtype == "I8" and input_.numel() == columns
    scale_name = name.removesuffix(".weight") + ".scale"
    scales = checkpoint.values(scale_name).to(torch.float32)
    assert tuple(scales.shape) == (rows, columns // 32)
    x = dynamic_e4m3(input_).to(torch.float32)
    output = torch.empty(rows, dtype=torch.bfloat16)
    for start in range(0, rows, row_chunk):
        count = min(row_chunk, rows - start)
        packed = torch.from_numpy(checkpoint.raw(name, start * packed_columns, count * packed_columns)).reshape(count, -1)
        codes = torch.empty((count, columns), dtype=torch.long)
        codes[:, 0::2] = (packed & 15).long()
        codes[:, 1::2] = (packed >> 4).long()
        weight = FP4_VALUES[codes]
        weight *= scales[start : start + count].repeat_interleave(32, 1)
        output[start : start + count] = (weight @ x).to(torch.bfloat16)
    return output


def linear(checkpoint: Checkpoint, name: str, input_: torch.Tensor, args) -> torch.Tensor:
    dtype = checkpoint.spec(name).dtype
    if dtype == "F8_E4M3":
        return mx8_linear(checkpoint, name, input_, args.mx8_row_blocks)
    if dtype == "I8":
        return mx4_linear(checkpoint, name, input_, args.matrix_row_chunk)
    return plain_linear(checkpoint, name, input_, args.matrix_row_chunk, output_bf16=dtype == "BF16")


def rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = value.to(torch.float32)
    return (weight.to(torch.float32) * x * torch.rsqrt(x.square().mean() + eps)).to(torch.bfloat16)


def hc_mixes(checkpoint: Checkpoint, prefix: str, sublayer: str, hidden: torch.Tensor, config: dict, args):
    x = hidden.flatten().to(torch.float32)
    function = f"{prefix}.hc_{sublayer}_fn"
    mixes = plain_linear(checkpoint, function, x, args.matrix_row_chunk, output_bf16=False)
    mixes *= torch.rsqrt(x.square().mean() + config["rms_norm_eps"])
    scale = checkpoint.values(f"{prefix}.hc_{sublayer}_scale").to(torch.float32)
    base = checkpoint.values(f"{prefix}.hc_{sublayer}_base").to(torch.float32)
    hc = config["hc_mult"]
    pre = torch.sigmoid(mixes[:hc] * scale[0] + base[:hc]) + config["hc_eps"]
    post = 2 * torch.sigmoid(mixes[hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[2 * hc :] * scale[2] + base[2 * hc :]).reshape(hc, hc)
    comb = torch.softmax(comb, dim=-1) + config["hc_eps"]
    comb /= comb.sum(dim=0, keepdim=True) + config["hc_eps"]
    for _ in range(1, config["hc_sinkhorn_iters"]):
        comb /= comb.sum(dim=1, keepdim=True) + config["hc_eps"]
        comb /= comb.sum(dim=0, keepdim=True) + config["hc_eps"]
    return pre, post, comb


def hc_pre(hidden: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    return (pre[:, None] * hidden.to(torch.float32)).sum(dim=0).to(torch.bfloat16)


def hc_post(branch: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    value = post[:, None] * branch.to(torch.float32)[None, :]
    value += torch.einsum("ts,sd->td", comb, residual.to(torch.float32))
    return value.to(torch.bfloat16)


def build_hashes(model_dir: Path, token_id: int, config: dict) -> list[list[int]]:
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel), normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    keys: dict[str, int] = {}
    token_map = []
    for current in range(tokenizer.get_vocab_size(with_added_tokens=True)):
        decoded = tokenizer.decode([current], skip_special_tokens=False)
        key = tokenizer.id_to_token(current) if "\ufffd" in decoded else normalizer.normalize_str(decoded)
        if not key:
            key = decoded
        token_map.append(keys.setdefault(key, len(keys)))
    assert len(keys) == config["engram_compressed_vocab_size"]
    pad = token_map[config["engram_pad_token_id"]]
    tokens = [token_map[token_id], pad, pad, pad]
    seen: set[int] = set()
    layouts = []
    for layer_id in config["engram_layer_ids"]:
        planes = []
        offset = 0
        generator = np.random.default_rng(10007 * layer_id)
        bound = max(1, (np.iinfo(np.int64).max // len(keys)) // 2)
        multipliers = generator.integers(0, bound, size=4, dtype=np.int64) * 2 + 1
        rolling = (tokens[0] * int(multipliers[0])) & ((1 << 64) - 1)
        for ngram in range(3):
            rolling ^= (tokens[ngram + 1] * int(multipliers[ngram + 1])) & ((1 << 64) - 1)
            current_prime = config["engram_vocab_size"] - 1
            row = []
            for _ in range(config["engram_n_heads"]):
                candidate = current_prime + 1
                while not isprime(candidate) or candidate in seen:
                    candidate += 1
                current_prime = candidate
                seen.add(candidate)
                row.append(rolling % candidate + offset)
                offset += candidate
            planes.extend(row)
        layouts.append(planes)
    return layouts


def engram(checkpoint: Checkpoint, prefix: str, hidden: torch.Tensor, hashes: list[int], config: dict, args):
    rows = []
    table = f"{prefix}.embed.weight"
    scales = f"{prefix}.embed.scale"
    width = config["engram_head_dim"]
    for row_id in hashes:
        encoded = checkpoint.raw(table, row_id * width, width)
        value = E4M3_VALUES[torch.from_numpy(encoded).long()]
        scale = checkpoint.raw(scales, row_id * (width // 32), width // 32)
        value *= decode_e8m0(torch.from_numpy(scale)).repeat_interleave(32)
        rows.append(value.to(torch.bfloat16))
    kv = linear(checkpoint, f"{prefix}.wkv.weight", torch.cat(rows), args)
    hc, dim = config["hc_mult"], config["hidden_size"]
    key, value = kv[: hc * dim].reshape(hc, dim), kv[hc * dim :]
    q_weight = checkpoint.values(f"{prefix}.q_weight").reshape(hc, dim).to(torch.float32)
    k_weight = checkpoint.values(f"{prefix}.k_weight").reshape(hc, dim).to(torch.float32)
    h = hidden.to(torch.float32)
    k = key.to(torch.float32)
    rstd = torch.rsqrt(h.square().mean(-1) + config["rms_norm_eps"])
    rstd *= torch.rsqrt(k.square().mean(-1) + config["rms_norm_eps"])
    dot = (h * q_weight * k_weight * k).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    return (h + gate[:, None] * value.to(torch.float32)[None, :]).to(torch.bfloat16)


def attention_position_zero(checkpoint: Checkpoint, prefix: str, input_: torch.Tensor, layer: int, config: dict, source, args):
    dim, heads, head_dim = config["hidden_size"], config["num_attention_heads"], config["head_dim"]
    qr = linear(checkpoint, f"{prefix}.attn.wq_a.weight", input_, args)
    qr = rms_norm(qr, checkpoint.values(f"{prefix}.attn.q_norm.weight"), config["rms_norm_eps"])
    query = linear(checkpoint, f"{prefix}.attn.wq_b.weight", qr, args).reshape(heads, head_dim)
    local = linear(checkpoint, f"{prefix}.attn.wkv.weight", input_, args)
    local = rms_norm(local, checkpoint.values(f"{prefix}.attn.kv_norm.weight"), config["rms_norm_eps"])
    local = dynamic_e4m3(local).to(torch.bfloat16)
    kv = [local]
    ratio = config["compress_ratios"][layer]
    if layer == 20:
        latent = plain_linear(checkpoint, f"{prefix}.attn.compressor.wkv.weight", input_, args.matrix_row_chunk)
        latent = rms_norm(latent, checkpoint.values(f"{prefix}.attn.compressor.norm.weight"), config["rms_norm_eps"])
        source["main"] = dynamic_fp4(latent, 16, True)
        index_key = plain_linear(checkpoint, f"{prefix}.attn.indexer.wk.weight", latent, args.matrix_row_chunk)
        index_key = rms_norm(index_key, checkpoint.values(f"{prefix}.attn.indexer.k_norm.weight"), config["rms_norm_eps"])
        source["index"] = dynamic_fp4(index_key, 32, False)
    if ratio == 1:
        assert "main" in source
        kv.append(source["main"])
        if layer in config["index_source_layer_ids"]:
            # With one visible compressed position, top-k is deterministically position zero.
            index_q = linear(checkpoint, f"{prefix}.attn.indexer.wq_b.weight", qr, args)
            index_q = dynamic_fp4(index_q, 32, False)
            weights = plain_linear(checkpoint, f"{prefix}.attn.indexer.weights_proj.weight", input_, args.matrix_row_chunk)
            _ = (index_q.reshape(config["index_n_heads"], -1) @ source["index"]) * weights

    sinks = checkpoint.values(f"{prefix}.attn.attn_sink").to(torch.float32)
    output = torch.empty_like(query)
    keys = torch.stack(kv).to(torch.float32)
    for head in range(heads):
        scores = (keys @ query[head].to(torch.float32)) * head_dim**-0.5
        maximum = torch.maximum(scores.max(), sinks[head])
        exponentials = torch.exp(scores - maximum)
        denominator = exponentials.sum() + torch.exp(sinks[head] - maximum)
        # The published sparse kernel casts exponentials to BF16 before its value GEMM.
        numerator = exponentials.to(torch.bfloat16).to(torch.float32) @ keys
        output[head] = (numerator / denominator).to(torch.bfloat16)

    groups, rank = config["o_groups"], config["o_lora_rank"]
    grouped = output.reshape(groups, -1)
    wo_a_name = f"{prefix}.attn.wo_a.weight"
    low = torch.stack([
        mx8_rows_no_activation_quant(checkpoint, wo_a_name, group * rank, rank, grouped[group])
        for group in range(groups)
    ])
    return linear(checkpoint, f"{prefix}.attn.wo_b.weight", low.flatten(), args)


def expert(checkpoint: Checkpoint, prefix: str, input_: torch.Tensor, config: dict, args, route_weight=None):
    gate = linear(checkpoint, f"{prefix}.w1.weight", input_, args).to(torch.float32).clamp_max(config["swiglu_limit"])
    up = linear(checkpoint, f"{prefix}.w3.weight", input_, args).to(torch.float32).clamp(-config["swiglu_limit"], config["swiglu_limit"])
    value = F.silu(gate) * up
    if route_weight is not None:
        value *= route_weight
    return linear(checkpoint, f"{prefix}.w2.weight", value.to(torch.bfloat16), args)


def moe(checkpoint: Checkpoint, prefix: str, input_: torch.Tensor, config: dict, args):
    router_name = f"{prefix}.ffn.gate.weight"
    logits = plain_linear(checkpoint, router_name, input_, args.matrix_row_chunk, output_bf16=False)
    scores = F.softplus(logits).sqrt()
    bias = checkpoint.values(f"{prefix}.ffn.gate.bias").to(torch.float32)
    _, ids = torch.topk(scores + bias, config["num_experts_per_tok"])
    weights = scores[ids]
    weights = weights / (weights.sum() + 1e-20) * config["routed_scaling_factor"]
    output = torch.zeros(config["hidden_size"], dtype=torch.float32)
    for expert_id in sorted(ids.tolist()):
        slot = (ids == expert_id).nonzero()[0, 0]
        value = expert(checkpoint, f"{prefix}.ffn.experts.{expert_id}", input_, config, args, weights[slot])
        output += value.to(torch.float32)
    output += expert(checkpoint, f"{prefix}.ffn.shared_experts", input_, config, args).to(torch.float32)
    routes = [{"expert": int(i), "weight": float(w), "selection_score": float(scores[i] + bias[i])}
              for i, w in zip(ids, weights)]
    return output.to(torch.bfloat16), routes


def trace(value: torch.Tensor) -> dict:
    value = value.contiguous().to(torch.bfloat16)
    bits = value.view(torch.int16).numpy().tobytes()
    f32 = value.to(torch.float32)
    return {
        "sha256_bf16_le": hashlib.sha256(bits).hexdigest(),
        "first_16": f32.flatten()[:16].tolist(),
        "sum": float(f32.to(torch.float64).sum()),
        "l2_norm": float(torch.linalg.vector_norm(f32)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--token", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--matrix-row-chunk", type=int, default=256)
    parser.add_argument("--mx8-row-blocks", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.bfloat16)
    config = json.loads((args.model_dir / "config.json").read_text())["text_config"]
    checkpoint = Checkpoint(args.model_dir)
    started = time.monotonic()
    hashes = build_hashes(args.model_dir, args.token, config)
    hidden = checkpoint.row_bf16("embed.weight", args.token).repeat(config["hc_mult"], 1)
    incoming_pre = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    source: dict[str, torch.Tensor] = {}
    routes_by_layer = []
    layer_traces = []
    for layer in range(config["num_hidden_layers"]):
        layer_started = time.monotonic()
        prefix = f"layers.{layer}"
        if layer in config["engram_layer_ids"]:
            plane = config["engram_layer_ids"].index(layer)
            hidden = engram(checkpoint, f"{prefix}.engram", hidden, hashes[plane], config, args)
        residual = hidden
        attn_pre, attn_post, attn_comb = hc_mixes(checkpoint, prefix, "attn", hidden, config, args)
        collapsed = hc_pre(hidden, incoming_pre)
        normalized = rms_norm(collapsed, checkpoint.values(f"{prefix}.attn_norm.weight"), config["rms_norm_eps"])
        branch = attention_position_zero(checkpoint, prefix, normalized, layer, config, source, args)
        hidden = hc_post(branch, residual, attn_post, attn_comb)

        residual = hidden
        ffn_pre, ffn_post, ffn_comb = hc_mixes(checkpoint, prefix, "ffn", hidden, config, args)
        collapsed = hc_pre(hidden, attn_pre)
        normalized = rms_norm(collapsed, checkpoint.values(f"{prefix}.ffn_norm.weight"), config["rms_norm_eps"])
        branch, routes = moe(checkpoint, prefix, normalized, config, args)
        hidden = hc_post(branch, residual, ffn_post, ffn_comb)
        incoming_pre = ffn_pre
        routes_by_layer.append(routes)
        layer_traces.append(trace(hidden))
        print(f"layer {layer:02d}: {time.monotonic() - layer_started:.2f}s routes={[r['expert'] for r in routes]}", flush=True)

    final = hc_pre(hidden, incoming_pre)
    final = rms_norm(final, checkpoint.values("norm.weight"), config["rms_norm_eps"])
    logits = plain_linear(checkpoint, "head.weight", final, args.matrix_row_chunk, output_bf16=False)
    top_values, top_ids = torch.topk(logits, 16)
    result = {
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "token": args.token,
        "source": "independent port of official inference/model.py and kernel.py position-zero graph",
        "elapsed_seconds": time.monotonic() - started,
        "layer_traces": layer_traces,
        "routes_by_layer": routes_by_layer,
        "final_hidden": trace(final),
        "logits": {
            "sha256_f32_le": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
            "sum": float(logits.to(torch.float64).sum()),
            "l2_norm": float(torch.linalg.vector_norm(logits)),
            "top_16": [{"token": int(i), "logit": float(v)} for i, v in zip(top_ids, top_values)],
        },
    }
    output = args.output or Path("tests/fixtures/deepseek_v41_bos_real_oracle.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output} in {result['elapsed_seconds']:.2f}s")


if __name__ == "__main__":
    main()
