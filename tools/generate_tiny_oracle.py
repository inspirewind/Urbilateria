#!/usr/bin/env python3
"""Generate the independent tiny dense-GLM oracle embedded in Rust tests.

This script mirrors the public GLM-5.2 equations with PyTorch primitives; it does not import
Urbilateria or share Rust kernels. It prints JSON only and never touches the full checkpoint.
"""

import json
import math

import torch
import torch.nn.functional as F


torch.set_grad_enabled(False)
torch.set_num_threads(1)

HIDDEN = 6
VOCAB = 7
NUM_HEADS = 2
Q_LORA = 5
KV_LORA = 3
QK_NOPE = 2
QK_ROPE = 4
V_HEAD = 3
INTERMEDIATE = 5
THETA = 10_000.0


def values(count: int, seed: int) -> torch.Tensor:
    raw = [((index * 17 + seed * 13) % 29) - 14 for index in range(count)]
    return torch.tensor(raw, dtype=torch.float32) / 100.0


def matrix(rows: int, columns: int, seed: int) -> torch.Tensor:
    return values(rows * columns, seed).reshape(rows, columns)


def norm(length: int, seed: int) -> torch.Tensor:
    return 1.0 + values(length, seed) / 10.0


WEIGHTS = {
    "embedding": matrix(VOCAB, HIDDEN, 1),
    "input_norm": norm(HIDDEN, 2),
    "q_a": matrix(Q_LORA, HIDDEN, 3),
    "q_a_norm": norm(Q_LORA, 4),
    "q_b": matrix(NUM_HEADS * (QK_NOPE + QK_ROPE), Q_LORA, 5),
    "kv_a": matrix(KV_LORA + QK_ROPE, HIDDEN, 6),
    "kv_a_norm": norm(KV_LORA, 7),
    "kv_b": matrix(NUM_HEADS * (QK_NOPE + V_HEAD), KV_LORA, 8),
    "o": matrix(HIDDEN, NUM_HEADS * V_HEAD, 9),
    "post_norm": norm(HIDDEN, 10),
    "gate": matrix(INTERMEDIATE, HIDDEN, 11),
    "up": matrix(INTERMEDIATE, HIDDEN, 12),
    "down": matrix(HIDDEN, INTERMEDIATE, 13),
    "final_norm": norm(HIDDEN, 14),
    "lm_head": matrix(VOCAB, HIDDEN, 15),
}


def rms_norm(vector: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    return vector * torch.rsqrt(vector.square().mean() + epsilon) * weight


def interleaved_rope(vector: torch.Tensor, position: int) -> torch.Tensor:
    half = vector.numel() // 2
    output = torch.empty_like(vector)
    for pair in range(half):
        inverse_frequency = THETA ** (-2.0 * pair / vector.numel())
        angle = position * inverse_frequency
        even = vector[2 * pair]
        odd = vector[2 * pair + 1]
        output[pair] = even * math.cos(angle) - odd * math.sin(angle)
        output[half + pair] = odd * math.cos(angle) + even * math.sin(angle)
    return output


def main() -> None:
    latent_cache = []
    rope_cache = []
    steps = []
    for position, token in enumerate([1, 5, 2]):
        hidden = WEIGHTS["embedding"][token].clone()
        normalized = rms_norm(hidden, WEIGHTS["input_norm"], 1e-5)

        q_low = F.linear(normalized, WEIGHTS["q_a"])
        q_low = rms_norm(q_low, WEIGHTS["q_a_norm"], 1e-6)
        query = F.linear(q_low, WEIGHTS["q_b"]).reshape(
            NUM_HEADS, QK_NOPE + QK_ROPE
        )
        query = torch.stack(
            [
                torch.cat(
                    [head[:QK_NOPE], interleaved_rope(head[QK_NOPE:], position)]
                )
                for head in query
            ]
        )

        compressed = F.linear(normalized, WEIGHTS["kv_a"])
        latent = rms_norm(compressed[:KV_LORA], WEIGHTS["kv_a_norm"], 1e-6)
        rope_key = interleaved_rope(compressed[KV_LORA:], position)
        latent_cache.append(latent)
        rope_cache.append(rope_key)

        reconstructed_cache = [
            F.linear(cached_latent, WEIGHTS["kv_b"]).reshape(
                NUM_HEADS, QK_NOPE + V_HEAD
            )
            for cached_latent in latent_cache
        ]
        contexts = []
        for head_index in range(NUM_HEADS):
            scores = torch.stack(
                [
                    (
                        torch.dot(
                            query[head_index, :QK_NOPE],
                            reconstructed[head_index, :QK_NOPE],
                        )
                        + torch.dot(query[head_index, QK_NOPE:], cached_rope)
                    )
                    / math.sqrt(QK_NOPE + QK_ROPE)
                    for reconstructed, cached_rope in zip(
                        reconstructed_cache, rope_cache
                    )
                ]
            )
            probabilities = torch.softmax(scores, dim=0)
            contexts.append(
                sum(
                    probability * reconstructed[head_index, QK_NOPE:]
                    for probability, reconstructed in zip(
                        probabilities, reconstructed_cache
                    )
                )
            )
        context = torch.cat(contexts)
        hidden = hidden + F.linear(context, WEIGHTS["o"])

        normalized = rms_norm(hidden, WEIGHTS["post_norm"], 1e-5)
        mlp = F.linear(
            F.silu(F.linear(normalized, WEIGHTS["gate"]))
            * F.linear(normalized, WEIGHTS["up"]),
            WEIGHTS["down"],
        )
        hidden = hidden + mlp
        logits = F.linear(
            rms_norm(hidden, WEIGHTS["final_norm"], 1e-5),
            WEIGHTS["lm_head"],
        )
        steps.append(
            {
                "layer_hidden": [float(value) for value in hidden],
                "logits": [float(value) for value in logits],
                "argmax": int(torch.argmax(logits)),
            }
        )

    print(json.dumps({"tokens": [1, 5, 2], "steps": steps}, indent=2))


if __name__ == "__main__":
    main()
