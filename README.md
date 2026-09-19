<div align="center">

<p>
<img src="assets/urbilateria-mark.svg" width="760" alt="Urbilateria logo and wordmark, highlighting the urb CLI command">
</p>

<h3>Frontier MoE checkpoints, made runnable.</h3>

<p>
English · <a href="README_CN.md">中文</a>
</p>

<p>
A pure-Rust research runtime for understanding and running frontier models such as<br>
GLM-5.2, DeepSeek-V4/V4.1, Kimi-K3, Qwen3.8, and Hy4 on bounded-memory, CPU-only machines.
</p>

<p>
<img src="https://img.shields.io/badge/Rust-1.83%2B-b7410e?style=flat-square&logo=rust&logoColor=white" alt="Rust 1.83+">
<img src="https://img.shields.io/badge/runtime-CPU--only-3d6b5d?style=flat-square" alt="CPU-only runtime">
<img src="https://img.shields.io/badge/model_families-5-247ba0?style=flat-square" alt="Five model families">
<img src="https://img.shields.io/badge/tests-338_passing-2e7d32?style=flat-square" alt="338 tests passing">
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-6c5ce7?style=flat-square" alt="MIT License"></a>
</p>

<table>
<tr>
<td align="center"><b>2.78T</b><br><sub>maximum parameter count</sub></td>
<td align="center"><b>6</b><br><sub>model adapters</sub></td>
<td align="center"><b>6</b><br><sub>public generation paths</sub></td>
<td align="center"><b>338</b><br><sub>default tests passing</sub></td>
<td align="center"><b>0</b><br><sub>GPUs required</sub></td>
</tr>
</table>

<p>
<a href="#quick-start">Quick start</a> ·
<a href="#how-it-works">How it works</a> ·
<a href="#model-coverage">Model coverage</a> ·
<a href="#validation">Validation</a> ·
<a href="#cli-reference">CLI</a> ·
<a href="#development">Development</a>
</p>

</div>

> [!WARNING]
> Urbilateria is an experimental learning and model-forensics project. Its scalar runtimes favor
> readable, runnable behavior over production throughput. Large-checkpoint generation works for
> GLM-5.2, DeepSeek-V4/V4.1, Kimi-K3, Qwen3.8, and Hy4, but numerical accuracy, output quality, memory
> use, and speed carry no production guarantees. DeepSeek-V4.1 includes a scalar base-text
> generation path validated against an independent real-weight BOS oracle. Its tokenwise prompt
> prefill is intended for correctness work and is slow.

## Why Urbilateria?

Most inference frameworks need staggering GPU resources to run frontier models. Urbilateria lets
you experience these checkpoints with just a CPU, a fast SSD, and a reasonable amount of RAM.

It is implemented entirely in Rust. No Python framework participates in the inference path, and no
model weights are included in this repository.

## Quick Start

Build the release binary and inspect the available commands:

```bash
cargo build --release --locked
./target/release/urb help
```

Start with metadata. These commands do not need to read the full tensor payload:

```bash
# Static parameter and quantization X-ray
./target/release/urb inspect /path/to/model

# RAM, context, KV, scratch, and expert-cache plan
./target/release/urb plan /path/to/model --ram-gib 32 --context 2048

# Exact runtime tensor and memory validation
./target/release/urb preflight /path/to/model --context 2048 --expert-slots 8
```

Try the model-native tokenizer and chat template:

```bash
./target/release/urb tokenize /path/to/model "Explain sparse MoE routing" --chat
```

Run the `generate` command:

```bash
./target/release/urb generate /path/to/model \
  --prompt "Explain sparse MoE routing" \
  --ram-gib 32 \
  --max-new-tokens 8 \
  --no-thinking \
  --allow-large-model
```

`generate` requires an explicit RAM limit and `--allow-large-model`; a short generation can still
read many gigabytes from storage. The public command supports GLM-5.2, DeepSeek-V4,
DeepSeek-V4.1, text-only Kimi-K3, text-only Qwen3.8, and Hy4 today. Qwen3.8 uses the release's
always-thinking chat template, so omit `--no-thinking` for that model. Hy4 generation is exact through 2,048 total
prompt+generation tokens, where its top-2,048 DSA selection contains the complete causal history.

## How It Works

```mermaid
flowchart LR
    A[(Native checkpoint<br/>shards + metadata)] --> B[Strict config<br/>and schema gate]
    B --> C[inspect · list · probe]
    B --> D[plan · preflight]
    B --> E[Model-native<br/>tokenizer + chat]
    E --> F[Layer-streamed<br/>CPU runtime]
    D --> F
    F <--> G[(Bounded per-layer<br/>expert cache)]
    F --> H[Shared greedy<br/>token loop]
    H --> I[Streamed UTF-8<br/>and profile JSON]

    J[KDA · MLA · DSA<br/>DeltaNet · GQA] -. family math .-> F
    K[INT8 · INT4 · MXFP8<br/>MXFP4 · BF16 · FP8] -. native weights .-> F
```

The runtime keeps the working set explicit rather than pretending the checkpoint is resident:

| Memory class | What lives there | How Urbilateria controls it |
| --- | --- | --- |
| **Resident** | configuration, tokenizer, root vectors, model state | validated once against the requested RAM ceiling |
| **Layer-streamed** | decoder trunks, large rows, vocabulary head chunks | loaded in execution order and released at layer boundaries |
| **Cache-bounded** | routed MoE experts | deterministic per-layer LRU with an explicit slot budget |
| **Sequence state** | MLA KV, KDA/DeltaNet recurrence, convolution, GQA KV | calculated from exact family geometry and requested context |
| **Scratch** | activations, quantization blocks, batched prompt work | included in preflight before payload reads begin |

Large matvec output rows run on one persistent CPU worker pool. Prompt ingestion is layer-wise for
DeepSeek-V4, Kimi-K3, Hy4, and the Qwen3.8 library runtime: one decoder layer is loaded for the entire
prompt, and only the final prompt token reaches the vocabulary-sized LM head. DeepSeek-V4.1
currently uses its exact token-at-a-time path for prompt prefill as well as decode.

## Model Coverage

| Capability | GLM-5.2 | DeepSeek-V4 | DeepSeek-V4.1 | Kimi-K3 | Qwen3.8 | Hy4 |
| --- | --- | --- | --- | --- | --- | --- |
| **Checkpoint ABI** | converted Colibri | native 48-shard | native 48-shard; 96,085 tensors | native 96-shard | native 213-shard | native 130-shard |
| **Tokenizer / chat** | byte-BPE | native text chat | native text chat + numeric effort | TikToken + XTML | always-thinking ChatML | native reasoning/no-think |
| **Native weights** | INT8/INT4 | 128×128 MXFP8 + MXFP4 | 32×32 MXFP8 + MXFP4 | BF16 + MXFP4 | block FP8 | ModelOpt MXFP8 |
| **Attention path** | MLA | local + compressed | CED + CSA2 + Engram + mHC | KDA + MLA | DeltaNet + GQA | iHC + MLA/DSA |
| **Public CLI generation** | Experimental | Experimental | Experimental, text base runtime | Experimental | Experimental | Experimental, ≤2,048 tokens |
| **Multimodal execution** | No | No | Schema only | Schema only | No | No |

### What “native” means

**GLM-5.2.** The runtime expects a converted Colibri-style checkpoint rather than the official FP8
release. It supports resident core weights plus streamed experts in the converted container.

**DeepSeek-V4.** Urbilateria reads the release's 48-shard layout directly in Rust: config,
tokenizer, safetensors headers, E8M0 scale sidecars, MXFP8 matrices, and packed low-nibble-first
MXFP4 experts. DSpark tensors are schema-validated but speculative decoding is disabled.

**DeepSeek-V4.1.** The separate adapter validates the 40-layer CED graph, CSA2 ownership, 32×32
MXFP8 ABI, 384-way experts, two Engram tables, vision tower, and three DSpark stages. Public scalar
text generation executes the complete layer-streamed CED/CSA2/Engram/mHC base path with trained
compressed caches, exact bounded Engram reads, top-6 MoE, and a streamed LM head. An independent
PyTorch real-weight BOS oracle gates routing across all 40 layers and the final top-16 logits.
Prompt prefill remains token-at-a-time; vision and DSpark remain schema-only.

**Kimi-K3.** The text runtime streams 92 sparse layers of 896-way native MXFP4 routed experts,
keeps the compact BF16 trunk bounded by layer, and preserves recurrent KDA plus compressed MLA
state. XTML is assembled from trusted structural segments and escaped untrusted content. Generation
stops on `<|end_of_msg|>` (163586), not the tokenizer metadata's `[EOS]` token (163585).

**Qwen3.8.** The adapter is pinned to `model_type="qwen3_5_moe_text"` and the released
`Qwen/Qwen3.8-2.4T-A95B-FP8` ABI. It implements the base 92-layer text forward, top-10/512 routed
experts, gated shared experts, final norm, and LM head. MTP tensors are schema-validated but are not
part of the base autoregressive forward. The official checkpoint has passed a complete header gate
and an independent 92-layer plus full-LM-head oracle. Tiny FP32/BF16 graph fixtures and a bounded
real E4M3 expert payload oracle additionally pin dtype boundaries and quantized projection math.

**Hy4.** The adapter is pinned to `model_type="hy_v4"` and `HYV4ForCausalLM`. It validates the
complete 78-layer base plus native MTP payload, reads ModelOpt MXFP8 weights without conversion,
and slices a single routed expert from the consolidated 256-expert tensors. The tokenizer/chat,
identity-HC reference math, header/probe/preflight paths, and bounded-memory planner are wired.
The checkpoint advertises 1,048,576 positions, but exact execution remains capped at 2,048 until
Gated DSA/IndexCache and activation-quantization semantics pass an independent logits oracle.

## Checkpoint Setup

Install the Hugging Face CLI only for downloading checkpoints; it is not an inference dependency:

```bash
python3 -m pip install --upgrade huggingface_hub
```

<details>
<summary><b>GLM-5.2 converted checkpoint</b></summary>

The compatible converted checkpoint is
[`mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp`](https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp).

```bash
hf download mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp --dry-run

hf download mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp \
  --local-dir /path/to/glm-5.2-colibri
```

</details>

<details>
<summary><b>Hy4-preview-FP8 native checkpoint</b></summary>

```bash
hf download tencent/Hy4-preview-FP8 --local-dir /path/to/hy4-preview-fp8

./target/release/urb inspect /path/to/hy4-preview-fp8
./target/release/urb preflight /path/to/hy4-preview-fp8 --context 2048 --expert-slots 1
```

The strict release gate expects all 130 shards.

</details>

<details>
<summary><b>DeepSeek-V4/V4.1 and Kimi-K3 native checkpoints</b></summary>

Point Urbilateria at the model release directory after its native shards, `config.json`, tokenizer,
and index files have finished downloading. No Transformers/Python conversion is performed.

Kimi transfers can be checked before all 96 shards arrive. Dot-prefixed rsync temporary files are
ignored, and every decoder layer fully represented by completed exact `.safetensors` files is
validated:

```bash
./target/release/urb preflight /path/to/Kimi-K3 --partial
```

</details>

<details>
<summary><b>Qwen3.8 metadata-first setup</b></summary>

Before downloading roughly 2.5 TB of tensor payload, fetch the five small release files needed for
configuration, tokenizer, and manifest development:

```bash
hf download Qwen/Qwen3.8-2.4T-A95B-FP8 \
  --revision d2dc35658bcf77e66643428cb52e774cc3b5bd29 \
  --include config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors.index.json \
  --local-dir /path/to/qwen3.8-metadata
```

After all 213 shards arrive, preflight and run the text-only always-thinking path:

```bash
./target/release/urb preflight /path/to/Qwen3.8-2.4T-A95B-FP8 --context 64 --expert-slots 0
./target/release/urb generate /path/to/Qwen3.8-2.4T-A95B-FP8 \
  --prompt "Hello" --ram-gib 4 --max-new-tokens 1 --allow-large-model
```

The 4 GiB example is above the measured ~2.04 GiB scalar-runtime peak for a 64-token context, but
the native 2.496 TB checkpoint still makes even short generation storage-bound and very slow.

</details>

Large checkpoints need ample local storage and NVMe bandwidth. Re-running the same `hf download`
command resumes completed work. Use `hf auth login` only when Hugging Face requests authentication.

## Validation

Urbilateria separates “the file looks plausible” from “the model produces the right logits.” Each
layer of confidence has its own gate:

```text
1  Config gate       exact geometry, token IDs, quantization contract
2  Header gate       tensor names, dtypes, shapes, byte lengths, shard assignment
3  Tiny oracle       deterministic end-to-end graph against independent fixtures
4  Payload gate      bounded reads decode real quantized bytes exactly
5  Runtime gate      full real layer/token execution under the memory contract
6  Output gate       independent full logits and known-token continuation
```

| Family | Strongest completed evidence represented in the repository | Remaining public gate |
| --- | --- | --- |
| GLM-5.2 | tiny full-model oracle; real converted-checkpoint token regressions | throughput optimization |
| DeepSeek-V4 | independent tiny oracle; real layer/token and tokenizer regressions | sustained performance work |
| DeepSeek-V4.1 | exact headers, native MX payloads, complete Engram/CSA2/mHC base forward and public generation, real 40-layer BOS routing and top-16-logit oracle | layer-wise prefill, vision, and DSpark execution |
| Kimi-K3 | independent full-stack logits parity and known `Paris` continuation | sustained performance and revision-by-revision validation |
| Qwen3.8 | independent tiny FP32/BF16 graph, real FP8 expert payload, and real 92-layer full-logits oracles | sustained performance and multi-token release validation |
| Hy4 | upstream iHC/Gated-MLA semantics; exact schema/MXFP8/expert payload gates; real 78-layer token and public CLI generation smoke | independent full-logits parity and >2,048-token IndexCache execution |

For the Qwen3.8 release token `Hello`, the independent PyTorch streamer and Rust runtime select the
same argmax and the same top 20 tokens. Across all 248,320 logits, cosine similarity is
`0.9999224`, mean absolute error is `0.028373`, and maximum absolute error is `0.1875`. Twenty-three
layers exchange 24 of 920 expert positions at the top-10 routing boundary because the two backends
use different GEMM reduction trees; the gate bounds aggregate replacements to 32. The isolated
real FP8 expert gate remains exact to one BF16 ULP.

The model-free suite is fast and does not need a checkpoint:

```bash
cargo test --all-targets --locked
```

Current result: **338 passed, 0 failed**, with real-checkpoint tests explicitly ignored unless
their model directory is supplied.

<details>
<summary><b>Run representative real-checkpoint gates</b></summary>

```bash
URB_DEEPSEEK_V4_DIR=/path/to/DeepSeek-V4-Flash-0731 \
  cargo test --release --locked --test deepseek_v4_real -- --ignored

URB_GLM52_DIR=/path/to/colibri-glm-5.2 \
  cargo test --release --locked --test glm_real -- --ignored

KIMI_K3_MODEL_DIR=/path/to/Kimi-K3 \
KIMI_K3_REFERENCE_LOGITS=/path/to/kimi-k3-in-c/tests/fixtures/golden/ref_logits.json \
  cargo test --release --locked --test kimi_k3_real \
  real_checkpoint_logits_match_the_independent_pytorch_golden -- --ignored --exact --nocapture

KIMI_K3_MODEL_DIR=/path/to/Kimi-K3 \
  cargo test --release --locked --test kimi_k3_real \
  real_checkpoint_completes_france_with_paris -- --ignored --exact --nocapture

HY4_MODEL_DIR=/path/to/hy4-preview-fp8 \
  cargo test --release --locked --test hy4_real -- --ignored --nocapture
```

</details>

## CLI Reference

| Command | Purpose | Model support |
| --- | --- | --- |
| `inspect` | Build a static parameter, quantization, tensor, and routing X-ray | All six adapters |
| `plan` | Estimate resident, KV, scratch, and expert-cache budgets | GLM, DeepSeek, Kimi, Hy4 |
| `preflight` | Validate exact runtime tensors and memory without payload reads | All six adapters |
| `list` | Search exact tensor names | All safetensors checkpoints |
| `probe` | Sample one tensor without loading the checkpoint | All safetensors checkpoints |
| `tokenize` | Encode raw text or a model-native chat turn | All six adapters |
| `decode` | Decode comma-separated token IDs | All six adapters |
| `generate` | Run RAM-planned greedy generation | All six adapters |
| `explain` | Print the token path and tensor geometry | All six adapters |

Every analysis command that supports it can emit JSON for automation. Run `urb help` for the exact
flags and defaults.

### Prompt safety

Ordinary user text is rendered through the model-native chat protocol. Use `--no-thinking` to
disable reasoning where that family permits it. `--raw-prompt` is a trusted escape hatch only for
a fully rendered prompt beginning with the model's native BOS/protocol prefix; bare text is rejected
because it can otherwise produce immediate EOS or unrelated output.

Kimi-K3 makes the trust boundary explicit: protocol markers are emitted only by typed structural
segments, while user, assistant, and tool content remain ordinary escaped segments. Qwen3.8's
official template always requires thinking, so `--no-thinking` is invalid for that family.

## Profiling

Human-readable profiles go to stderr. Stable-schema JSON goes only to the requested file, never into
streamed model output. Timings are inclusive, so nested stages intentionally overlap. On Linux,
profile schema v2 also samples process CPU, RSS/swap, page faults, thread count, system memory, and
process I/O every 100 ms, with an exact final sample. `storage_read_bytes`/`storage_write_bytes` are
Linux storage-layer accounting; `read_char_bytes`/`write_char_bytes` include page-cache traffic.
Neither should be confused with a stage's logical checkpoint payload or whole-device utilization.

```bash
./target/release/urb generate /path/to/model \
  --prompt "Hello" --ram-gib 32 --max-new-tokens 1 --no-thinking \
  --allow-large-model --threads 20 --profile \
  --profile-json profile-20.json --profile-trace profile-20.trace.json
```

`--profile-trace` enables bounded per-span recording and writes Chrome Trace Event JSON. Open the
trace directly in [Perfetto UI](https://ui.perfetto.dev/), or open
[`tools/profile_viewer.html`](tools/profile_viewer.html) locally and drop both JSON files onto it.
The dependency-free viewer provides stage ranking, resource charts, thread filtering, search,
zoom/pan, and a time-ordered flame chart. Trace collection is disabled unless the flag is present.
DeepSeek-V4 trace spans attach token position/ID, layer ID, expert ID, pre-batch cache residency,
batch size, and a stable per-layer flow ID where those values are meaningful.

Use `--threads 1` as the serial baseline. Without `--threads`, the persistent worker pool uses the
platform's available logical parallelism. More threads are not always faster on SMT or hybrid-core
CPUs, so compare several values on the same prompt and cache state.

A model-free release benchmark exercises the same F32 kernel and profiler:

```bash
URB_BENCH_THREADS=1 cargo run --release --example profile_matvec --locked
URB_BENCH_THREADS=20 cargo run --release --example profile_matvec --locked
```

It reports dimensions, requested and effective workers, latency percentiles, GMAC/s, checksum, and
the full profile. `URB_BENCH_ROWS`, `URB_BENCH_COLS`, and `URB_BENCH_ITERATIONS` override defaults.

<details>
<summary><b>Real-checkpoint performance harness</b></summary>

Ignored `inference_performance` tests measure the same fixed-length greedy workload for GLM-5.2,
DeepSeek-V4, and Kimi-K3. Reports include engine TTFT, observed and model-only decode tokens/s,
setup stages, expert-cache telemetry, logical attention/KV growth, planned memory, and inclusive
profiles.

Run each model in a separate release test process so persistent worker pools, memory pressure, and
page-cache state do not overlap. Machine-local paths and defaults can live in the git-ignored
`.env.performance` file:

```bash
./tests/run_inference_performance.sh
```

Reports are written under `target/perf/`. Compare them only on a fixed host with identical
checkpoint, prompt, thread count, and expert-slot settings.

</details>

## Project Map

| Path | Responsibility |
| --- | --- |
| `src/analysis/` | checkpoint reports, tensor classification, probes, traces, and resource plans |
| `src/storage/` | strict safetensors indexing, bounded payload reads, and native weight loaders |
| `src/math/` | readable scalar quantization, routing, MXFP, and numerical reference operations |
| `src/models/` | family-private config, schema, prompt, attention, MoE, weights, and runtime code |
| `src/runtime/` | model-neutral runtime contracts and deterministic expert-cache mechanics |
| `src/generation.rs` | backend-independent autoregressive token loop and stop semantics |
| `src/profiling.rs` | opt-in inclusive stage metrics and stable JSON reports |
| `tests/` | real-checkpoint correctness gates and cross-model performance harnesses |
| `tools/` | independent Python oracle generation used during validation, not inference |

The small shared boundaries are deliberate. A new model family should reuse storage, generation,
profiling, and runtime contracts without forcing one architecture's tensor ABI or prompt semantics
onto another.

## Development

```bash
cargo fmt --all --check
cargo test --all-targets --locked
cargo clippy --all-targets --locked -- -D warnings
cargo build --release --locked
```

Behavior changes should include focused tests. Optimized kernels must remain checked against the
readable scalar reference; a faster path is not complete until its numerical contract is explicit.

## Current Limitations

- CPU-only correctness runtimes with parallel scalar matvec kernels; sustained large-model
  generation still needs SIMD/fused kernels and more I/O optimization.
- No server API, Web UI, CUDA, or Metal backend.
- The GLM path supports the converted Colibri container, not arbitrary official FP8 releases.
- DeepSeek-V4/V4.1 DSpark speculative decoding is schema-validated but disabled; the V4.1 vision
  tower is also schema-only, and its prompt prefill remains token-at-a-time.
- Kimi-K3 generation is text-only; MoonViT and projector tensors are validated but not executed.
- A Kimi checkpoint revision other than the tested release needs its own logits and performance
  validation before inheriting output-quality claims.
- Qwen3.8 MTP tensors are schema-validated but speculative decoding is disabled; the scalar
  generation path is suitable for correctness work, not interactive throughput.
- Hy4 generation is exact only through 2,048 total tokens; longer-context IndexCache execution and
  independent full-logits parity remain open validation milestones.
- Arbitrary Transformers architectures and quantization layouts are intentionally rejected.

## The Name

**Colibri + Ferris = Urbilateria.**

A hummingbird is a chordate and deuterostome; a crab is an arthropod and protostome. *Urbilateria*
is the name commonly given to their hypothetical last shared bilaterian ancestor, before those two
lineages diverged. It is not a known fossil species, and its actual form remains debated.

The name joins the project's two inspirations: Colibri's pursuit of low-memory inference and
Ferris's Rust ecosystem. Urbilateria brings those ideas together in an independent,
learning-oriented implementation.

## Contributing

Issues, experiments, and focused pull requests are welcome. Keep reference behavior readable,
include tests for behavioral changes, and document any new checkpoint or numerical assumptions.

## Acknowledgements

- [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2-FP8) for the model architecture
- [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) for its release configuration and standalone inference reference
- [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) for the native CED/CSA2/Engram/mHC model and checkpoint
- [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) for the released model, tokenizer, and architecture specification
- [Qwen3.8-2.4T-A95B-FP8](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8) for the release-pinned hybrid-attention MoE model
- [Hy4-preview-FP8](https://huggingface.co/tencent/Hy4-preview-FP8) for the release-pinned iHC/Gated-DSA MoE model
- [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) for an independent Apache-2.0 behavioral reference used to cross-check scalar formulas and checkpoint conventions
- [Colibri](https://github.com/JustVugg/colibri) for inspiring low-memory expert streaming
- [Rabbit](https://github.com/ferrumox/rabbit) for inspiring a Rust learning-oriented inference path

Urbilateria is an independent implementation and does not include code or model weights from these
projects.

## License

Urbilateria is licensed under the [MIT License](LICENSE).
