# Qwen3.8-Flash-Next

Native BF16 text adapter for the `qwen4_exp` / `Qwen4ExpForConditionalGeneration`
checkpoint. This is a separate model family from the existing Qwen3.8-2.4T FP8
adapter. Config and tensor declarations are pinned to the inspected release.

## Run

```bash
cargo build --release --locked
./target/release/urb inspect /mnt/990pro/Qwen3.8-Flash-Next
./target/release/urb preflight /mnt/990pro/Qwen3.8-Flash-Next --context 2048
./target/release/urb generate /mnt/990pro/Qwen3.8-Flash-Next \
  --prompt '只回复“你好”。' --no-thinking \
  --ram-gib 16 --threads 8 --max-new-tokens 16 --allow-large-model
./target/release/urb chat /mnt/990pro/Qwen3.8-Flash-Next \
  --ram-gib 16 --threads 8 --max-new-tokens 512 --allow-large-model
./target/release/urb ui /mnt/990pro/Qwen3.8-Flash-Next
```

CLI and TUI share model detection, tokenizer, prompt rendering and runtime.
Thinking defaults to `xhigh`; `--no-thinking` is supported. Historical reasoning
is preserved by default, matching the shipped Jinja template. The prompt API also
supports medium/low reasoning and explicit history stripping. Generation currently
uses the shared greedy sampler, rather than the checkpoint's sampling defaults.
`plan` directs hybrid models to `preflight` for their state and cache estimates.

## Architecture and storage

- 48 layers: 36 Gated DeltaNet and 12 QSA gated-GQA blocks; hidden width 2,560.
- Four Hyper-Connection streams, grouped zero-centered RMSNorm, rank-320 mixing,
  and learned block injection. A final HC mixer feeds the LM head directly.
- Top-10 of 512 routed experts per layer, plus a sigmoid-gated shared expert.
  Packed BF16 gate/up and down tensors are sliced by expert, without loading all
  512 experts. The bounded per-layer LRU uses the remaining planned RAM.
- DeltaNet uses sigmoid output gating and FP32 Q/K L2 normalization and recurrence.
  Existing Qwen3.6/3.8 retain their original SiLU and BF16 normalization behavior.
- QSA mean-pools blocks of four **raw** index keys before normalization and RoPE.
  Four index query heads select 512 blocks; the incomplete tail is always retained.
  GQA and the indexer share 64 rotary dimensions. KV remains available for the
  complete context, while index keys are stored as completed blocks plus a tail.
- PLE is inserted at decoder index 1 (the second layer). EOS-aware 2/3-gram hashing
  uses the checkpoint's I64 multipliers, moduli and offsets. Sixteen embedding rows
  are gathered from 128 split tables; the roughly 102 GB table stays on disk.
  Gated values receive a four-tap depthwise convolution with dilation three.

The checkpoint contains 1,658 tensors in 131 shards, with 359,999,963,128 payload
bytes and 179,999,981,424 BF16 parameters (plus 35 I64 metadata entries).
`inspect` checks manifest metadata; `preflight` additionally validates every header,
including vision and MTP. Header validation checks declarations and assignments,
not every payload value or a cryptographic checksum of the weights.

The runtime streams embedding rows, PLE rows and experts. It retains as much of
the LM head and decoder backbone as the RAM budget allows, with a streamed-layer
reserve. Prefill runs up to 32 tokens layer by layer to reuse loaded trunks;
expert computations currently remain tokenwise. State budgets include recurrent
matrices, convolution history, QSA KV/index caches, PLE history and session rollback.
Append-only KV/index blocks are truncated on rollback; recurrent/PLE states are
copied. Failed forwards poison the state until reset or valid checkpoint restore.

## Validation

```bash
cargo test --locked
QWEN38_FLASH_NEXT_MODEL_DIR=/mnt/990pro/Qwen3.8-Flash-Next \
  cargo test --release --locked --test qwen3_8_flash_next_real -- --ignored --nocapture
```

Default tests include independent BF16 PyTorch fixtures for HC, QSA selection and
incomplete blocks, all PLE convolution taps, and sigmoid DeltaNet. Additional tests
check signed n-gram hashing, EOS boundaries, strict config/manifest rejection, and
12 combinations of thinking/preservation/effort against the shipped Jinja template.
Opt-in gates exercise real headers, tokenizer parity, RAM rejection, two complete
48-layer forwards, streamed/resident equivalence, prompt prefill and conversation checkpoint replay.

Local validation on 2026-11-03: both reference steps selected the same greedy token;
logit sample MAE was 0.06396 / 0.07070 and maximum absolute error 0.25 / 0.28125.
These are BF16 numerical comparisons, not bitwise equality with PyTorch. Rust
prefill and checkpoint replay are checked for exact equality. A CLI smoke request
`只回复“你好”。` returned `你好` followed by EOS, using a 16 GiB budget and eight CPU
workers: 17 input tokens, two output IDs, 51.68 s TTFT, 54.40 s total. This single
short run includes weight loading and is not a sustained throughput benchmark.

The oracle generator uses mathematical modules extracted from the official
[Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py),
Git blob `1d80822ffa26327b348ccd57607197379dec6162` (source SHA256 is recorded in
both fixtures). Framework decorators are removed; cache plumbing and checkpoint
storage are replaced so Python also streams the selected weights. It does not
call the Rust runtime. To regenerate with PyTorch, safetensors, Jinja2 and tokenizers:

```bash
python tools/generate_qwen3_8_flash_next_oracle.py MODEL_DIR \
  --reference-source /path/to/modeling_qwen4_exp.py \
  --output tests/fixtures/qwen3_8_flash_next_real.json
python tools/generate_qwen3_8_flash_next_tiny_oracle.py \
  --reference-source /path/to/modeling_qwen4_exp.py --model-dir MODEL_DIR \
  --output tests/fixtures/qwen3_8_flash_next_tiny.json
```

The public path supports text input. Vision/video, tools and MTP speculation are
not implemented. The schema validates their weights but base text execution does
not load them. Long-context quality and throughput have not been qualified; the
262,144-token config limit is not a performance claim. Memory budgets estimate
runtime allocations and include working reserves, not OS page cache or a hard RSS
limit. Equal QSA block scores use the lower block index as a deterministic tie
break; upstream `topk` does not guarantee tie order.
