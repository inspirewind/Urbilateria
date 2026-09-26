# Qwen3.6-35B-A3B BF16 adapter

This experimental **text-only CPU runtime** reads the unmodified native checkpoint.
It detects `qwen3_5_moe` / `Qwen3_5MoeForConditionalGeneration`, validates the nested
text/vision config, and runs the base text decoder. It is separate from the pinned
Qwen3.8 FP8 adapter; both reuse the Qwen attention, normalization, router, and expert kernels.

## Run

```bash
cargo build --release
./target/release/urb inspect /mnt/990pro/Qwen3.6-35B-A3B
./target/release/urb preflight /mnt/990pro/Qwen3.6-35B-A3B --context 128 --expert-slots 0
./target/release/urb generate /mnt/990pro/Qwen3.6-35B-A3B \
  --prompt '只回答数字：1+1等于几？' --no-thinking \
  --ram-gib 4 --max-new-tokens 16 --allow-large-model
```

Thinking is enabled by default. `--no-thinking` renders the release's empty thinking
block; it does not inject Qwen3.8 reasoning-effort instructions. CLI chat and the TUI
use the same renderer and support persistent session state. As with Qwen3.8, use
`preflight` for hybrid memory accounting; the generic MLA-oriented `plan` is unsupported.

## Checkpoint and memory contract

- 26 shards, 1,045 BF16 tensors, 71,903,645,408 payload bytes including vision/MTP.
- 40 decoder blocks: 30 Gated DeltaNet and 10 gated GQA, hidden width 2,048.
- Each block selects 8 of 256 routed experts, plus a gated shared expert.
- Packed `gate_up_proj [256,1024,2048]` and `down_proj [256,2048,512]` are read by
  selected expert byte ranges. One expert stays in BF16 and occupies 6 MiB.
- Trunk loading peaks at approximately 71.4 MiB. Recurrent and convolution state
  occupy 63.75 MiB; GQA KV is 40 KiB per context token. The planner reserves an
  additional 512 MiB scratch allowance plus context-dependent hidden snapshots.
- Public generation uses zero retained expert slots. Every selected expert is
  reread, making this a correctness-oriented path, not a throughput-optimized one.
  Library users can request a bounded per-layer expert cache through `RuntimeLoadOptions`.
- Vision and MTP names, shapes, dtypes, and shard assignments are validated. Image,
  video, tool execution, and speculative decoding are not implemented.

## Validation

```bash
cargo test --all-targets
QWEN36_MODEL_DIR=/mnt/990pro/Qwen3.6-35B-A3B \
  cargo test --release --test qwen3_6_real -- --ignored --nocapture --test-threads=1
```

The committed oracle was generated with PyTorch 2.6.0 and Transformers 5.14.1 by
`tools/generate_qwen3_6_real_oracle.py`. It executes actual Transformers decoder,
attention, cache, norm, router, and shared-expert modules. Routed experts are streamed
from selected native slices, using the upstream BF16 expert equation and accumulation
order. No complete 35B model allocation is needed. The source hash is recorded in the
fixture. To regenerate in an environment with those dependencies:

```bash
python tools/generate_qwen3_6_real_oracle.py /mnt/990pro/Qwen3.6-35B-A3B \
  --output tests/fixtures/qwen3_6_real_oracle.json
```

The two-token `9419, 11` check covers all 40 layers, nonzero-position RoPE, recurrent
state, and GQA cache. Both next-token argmaxes match upstream (`11`, then `271`).
Sampled-logit RMSE was 0.1111 and 0.0725; maximum absolute error was 0.4375 and 0.265625.
The largest layer norm relative discrepancy was 3.54%. BF16 rounding and the different
recurrent/chunked execution order perturb near-tied expert choices. Tests bound sampled
logit maximum error to 0.5, RMSE to 0.125, layer norm drift to 4%, and require at least
87.5% expert-set overlap. These checks **do not establish bitwise parity or long-context
quality**. Layer-wise prefill and incremental Rust decode agree exactly in the real
three-token test.

Additional tests cover exact release detection, malformed configs/manifests, packed
expert offset/budget boundaries, and official Jinja prompt fixtures in both thinking
modes. A local 22-token Chinese arithmetic prompt returned `2` and stopped at EOS;
the zero-expert-cache run took about 39 seconds (20 CPU workers), mostly prefill.
