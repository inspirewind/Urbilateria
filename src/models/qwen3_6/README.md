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
  --ram-gib 90 --max-new-tokens 16 --allow-large-model
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
- Generation reserves causal state, scratch, and session-checkpoint memory first.
  Remaining RAM retains the LM head and decoder layers, then a per-layer expert LRU.
  At about 65 GiB total budget in a short context all 256 expert slots per layer fit;
  weights populate lazily as they are used. Increasing `--ram-gib` now changes actual
  weight residency. Smaller budgets retain a prefix of layers or fall back to streaming.
- Weight caches survive session checkpoint restore; recurrent and GQA state are still
  rewound separately. A cache miss retains the original transient expert/layer reserve.
  Library callers can use `Qwen36RuntimeRequirements::plan_load_options` or set explicit
  budgets with `RuntimeLoadOptions`.
- With multiple CPU workers, an uncached next layer inside the planned resident prefix
  loads while the current layer computes. Both jobs share the existing worker pool and
  finish before either error is returned; the current layer's error takes precedence.
  At most one next-layer payload is in flight, already charged to its cache reservation.
  Cached layers and the streamed tail skip prefetch. With multiple workers, forward,
  traced forward, and prefill enter the CPU pool once, including the final LM head;
  nested kernels reuse that pool without repeated dispatch from the caller thread.
  Profiling context follows the work, and single-worker execution stays direct.
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
three-token test and across 32/64/128-token chunk boundaries with an existing prefix.

Additional tests cover exact release detection, malformed configs/manifests, packed
expert offset/budget boundaries, and official Jinja prompt fixtures in both thinking
modes. A local 22-token Chinese arithmetic prompt returned `2` and stopped at EOS;
the zero-expert-cache run took about 39 seconds (20 CPU workers), mostly prefill.

## Reproducible CPU performance comparison

```bash
cargo build --release --example profile_qwen36
# Minimum-memory reference, then RAM-budgeted resident execution.
./target/release/examples/profile_qwen36 /mnt/990pro/Qwen3.6-35B-A3B 0 20 8
./target/release/examples/profile_qwen36 /mnt/990pro/Qwen3.6-35B-A3B 90 20 8
# Same-process crossover: warm up once, then compare tokenwise and batched prefill.
./target/release/examples/profile_qwen36 /mnt/990pro/Qwen3.6-35B-A3B 90 20 32 5 32,1,32,32,1
# Optional UTF-8 prompt file for longer-context measurements.
python3 -c 'print("Explain what a compiler does. " * 80, end="")' > /tmp/qwen36-long-prompt.txt
./target/release/examples/profile_qwen36 /mnt/990pro/Qwen3.6-35B-A3B 90 20 32 3 32 /tmp/qwen36-long-prompt.txt
# Time the same runtime without stage collection or background resource sampling.
./target/release/examples/profile_qwen36 /mnt/990pro/Qwen3.6-35B-A3B 90 20 32 3 128 /tmp/qwen36-long-prompt.txt off
```

Arguments are `MODEL_DIR RAM_GIB THREADS TOKENS [PASSES] [BATCH_SIZES] [PROMPT_FILE] [PROFILE]`.
Batch sizes cycle through comma-separated values in 1..=128; defaults are two
passes using the RAM plan: 128 when all 256 experts per layer fit, otherwise 32.
Size 1 retains tokenwise projections and expert execution
while sharing the same layer-major schedule, resident weights, and final LM head.
The optional file replaces the built-in user prompt before applying the same ChatML
template. Its path and the rendered token IDs are included in the JSON report.
`PROFILE` accepts `on` (the default) or `off`. Off runs without a `ProfileSession`,
sets `profile` to null and leaves stage arrays empty. On Linux, `resources_before`
and `resources_after` record RSS, process-lifetime peak RSS, swap and cumulative
storage reads outside the timed inference calls. These endpoints do not measure
per-pass peak RSS or transient swap. They are null when procfs is unavailable;
profiling-on reports continue to use the existing sampled resource report.
The harness reports model-load time, prompt prefill, per-token decode latency, expert
I/O/cache counters, output tokens, and hashes over all logits. Subsequent passes replay
identical inputs after rewinding causal state while retaining weight caches. It describes
fully warmed weights for that workload, not guaranteed throughput on arbitrary new text.
Run benchmarks serially without concurrent builds or other inference. Qwen3.6 profiling
separates layer loading, attention, MoE, expert loading/computation, and the LM head.
With layer prefetch, loading overlaps attention/MoE; those stage durations are not additive.
Shared `qwen.delta.*` stages additionally separate input projections, state updates
(including convolution and output normalization), and output projections.
Per-pass `prefill_stages` and `decode_stage_totals` separate completed-span counters
at the prefill boundary without restarting the resource sampler. Decode totals subtract
additive counters only; cumulative minima and maxima cannot be subtracted by phase.

Bulk BF16 activation casts round eight values at a time on AVX2 CPUs, using the same
integer ties-to-even rule. Both input and rounded exponent bits are checked; invalid
chunks replay scalarly so the first error and already-written prefix match the scalar
contract. Other architectures and short tails keep the scalar path. Boundary tests
cover every finite BF16 high word around rounding ties, unaligned slices, signed zero,
subnormals, NaN/infinity, and finite values that round to infinity.

Attention and MoE BF16 projections preserve their FP32 accumulation order. DeltaNet
state projections and the LM head retain ordered F64 reductions, and expert outputs
retain sorted accumulation order.
On AVX2 CPUs, the LM head's ordered BF16/F64 projection can advance eight rows
together, transposing four-column tiles only in registers. Each SIMD lane retains
one row's original column-order sum, separate F64 multiply/add operations, and
negative-zero identity. Weights remain in their native row-major BF16 layout.
This kernel is selected for serial work, or for at most four CPU workers when
there is at least one complete eight-row tile per worker. Larger pools keep the
original row schedule; incomplete tiles and non-AVX2 CPUs keep their existing path.
On AVX2/FMA CPUs, batched BF16 projections process two output rows and up to four
input tokens together: each input load feeds both rows, while each decoded weight
feeds four independent accumulators. Every output keeps the original eight-lane FMA
tree, horizontal sum, and scalar column tail. An odd final weight row keeps the
single-row kernel; other CPUs keep their original scalar scheduling, and one-token
decode keeps its existing GEMV path. No packed weight copy or extra activation buffer
is introduced. Exact GEMV comparisons cover paired/odd rows and input-group tails.
DeltaNet reuses normalized keys/queries across value heads, updates independent heads
in parallel, and uses stack scratch for the native 128-wide heads. Its AVX2 state
projection uses separate multiply/add operations to retain reference rounding.
During Qwen3.6 prefill, one 2.125 MiB DeltaNet transaction workspace is shared across
all recurrent layers, within the existing 512 MiB scratch reservation. Each chunk
copies the committed state into reusable buffers; tokens within the chunk update
that copy in place. The new state is published only after the complete output
projection passes validation. Failed work is overwritten before reuse, and checkpoints
contain only committed causal state. Incremental decode retains per-call transactions.
The `qwen.delta.state_update` span includes the transaction copy and all recurrence/gating
work, once per token or chunk.
For multi-worker prefill with enough recurrent work to parallelize heads, convolution
windows and normalized Q/K vectors are prepared in token order for the chunk. Each
worker then advances one value head through all chunk tokens, including finite checks,
BF16 casts, and output gating, before the outputs return to token-major layout. This
keeps each head's state local and dispatches heads once per chunk. Every head retains
the original token order, FP32 materialization points, and ordered F64 reductions.
Single-token decode, single-worker execution, and small geometries keep tokenwise
scheduling. When decode parallelizes heads, each worker now checks its updated state,
casts its core output, and applies the head's normalization and gate before returning.
This avoids the serial scan of all recurrent state and removes one core-output buffer.
Collected failures retain the original global phase priority: head update, state
finiteness, core finiteness, BF16 cast, then normalization/gating. Within a phase,
head order is retained. The complete output projection still must succeed before
state is committed.
State validation uses AVX2 exponent-bit scans on supported CPUs, retaining rejection
of every NaN/infinity encoding and a scalar fallback. During resident decode, shared
and routed expert projections use the worker pool concurrently; the shared contribution
is still accumulated after routed experts. Cache budgets and streaming behavior stay
the same.
On AVX2 CPUs, GQA scores eight cached keys together using four-column transposes
in registers. Each lane keeps one key's original dimension-order F64 sum and
negative-zero identity, with separate multiply/add operations. The F32 score cast,
scaling, BF16 rounding, softmax and token order remain unchanged. The current key,
incomplete groups of cached keys, and other CPUs retain scalar dot products;
column tails retain scalar additions. The KV cache stays unexpanded; score tiles
use registers and small stack buffers.
GQA value aggregation retains each dimension's original token-order F64 sum and
positive-zero identity, with separate multiply/add operations. On AVX2 with multiple
workers, the native 16-query/2-KV/256-wide geometry groups the eight queries sharing
each KV head once at least 128 tokens are attended. Each KV group computes its
query scores, packs probabilities by token, and distributes 16-column value tiles
over the same worker pool. Within 32-token blocks, every loaded value vector serves
all eight queries. The two KV groups can progress independently. Contexts return to
query-head order, with BF16 casts and score/context failures checked in the original
head order before committing KV state.
Other cases retain contiguous value-row traversal with independent output lanes
and scalar column tails. The head path uses the worker pool when query heads × head
width × attended tokens reaches 256K; native contexts below 64 attended tokens and
single-worker execution retain serial head scheduling. Native value sums use stack
storage. At maximum context, head score buffers and packed probabilities total at
most 32 MiB, within the existing 512 MiB fixed scratch reservation.

Prefill uses 128-token chunks when the planned cache can hold all 256 experts per
layer, and retains 32-token chunks for partial caches. Explicit library/profiler batch
sizes range from 1 to 128. Larger chunks group more token inputs by expert; attention
input and output projections share BF16 weight loads across input groups and reuse inputs
across pairs of output rows. DeltaNet
recurrence and GQA cache updates still follow token order. MoE projections group tokens
by routed expert when the complete selected set fits the cache without eviction.
Cache accesses replay the original token/expert sequence, including repeated hits;
each token retains sorted expert accumulation and the same BF16 rounding boundaries.
Smaller caches fall back to tokenwise expert execution. The bounded projection and
grouping buffers fit the fixed scratch reserve; prompt hidden snapshots remain
accounted for separately. Failed attention batches restore their prior state,
including failures in the final output projection.

For the fixed native geometry, a conservative simultaneous activation estimate at
128 tokens is 62 MiB: layer/attention/shared buffers use `8H + 4P + 2C` F32 values per
token, and routed gathers, projection layouts, and intermediates use `K(3H + 3I)`.
Here `H=2048`, `I=512`, `K=8`, and the largest projection width `P=8192`.
The additional `2C`, with convolution width `C=8192`, covers chunk-wide convolved
QKV, normalized Q/K, and the head-major output buffer; it adds 8 MiB at 128 tokens.
Doubling that estimate for vector capacities, adding 32 MiB of parallel attention
scores and packed probabilities at maximum context, the 2.125 MiB DeltaNet
transaction, and 16 MiB for metadata and small scratch gives 174.125 MiB, within the
unchanged 512 MiB fixed reserve.
A regression checks this bound against the bundled release geometry. Expert weights,
committed causal state, and full-prompt hidden snapshots retain their separate budgets.

Local measurements on an i7-14700K with 128 GiB RAM, 20 workers, and a 90 GiB runtime
budget used `Explain what a compiler does.`, no thinking, a 19-token rendered prompt,
and 8 generated tokens. These are single-run measurements with a populated OS page
cache, not cold-storage benchmarks. The CLI runs used identical arguments:

| CLI version | Decode tok/s | Time to first token | Total time |
| --- | ---: | ---: | ---: |
| Initial adaptation (`9a551f7`, zero retained experts) | 0.42 | 22.51 s | 39.03 s |
| Resident weights and tokenwise projections | 2.57 | 9.21 s | 12.22 s |
| Resident weights and batched attention projections | 2.78 | 7.50 s | 10.22 s |
| Resident weights, batched attention, and grouped MoE | 2.27 | 5.12 s | 8.43 s |
| Above plus vectorized state validation and shared-expert overlap | 2.53 | 5.35 s | 8.46 s |
| Above plus contiguous GQA value aggregation | 2.10 | 3.92 s | 7.45 s |
| Above plus GQA head parallelism above the work threshold | 2.08 | 4.17 s | 7.77 s |
| Above plus reusable DeltaNet prefill transactions | 1.91 | 4.01 s | 7.89 s |
| Above plus retained-layer prefetch on the CPU pool | 2.16 | 3.05 s | 6.57 s |

The grouped-MoE run preserves all 2,847 expert misses and 17,911,775,232 payload bytes
read from the previous resident runs. First-use decode still includes expert misses;
these short samples vary and do not demonstrate a steady decode improvement.

For a 32-token continuation, the crossover command above compares two warmed passes
per path in one process. Prefill is the arithmetic mean and decode is total decode
steps divided by their total time:

| Prefill path | Prefill time | Decode tok/s | Expert misses per pass |
| --- | ---: | ---: | ---: |
| Tokenwise projections and MoE, replay | 3.36 s | 4.52 | 0 |
| Batched attention and grouped MoE, replay | 1.21 s | 4.46 | 0 |

Harness prefill timing excludes model loading; it is not CLI time to first token.
The initial batched pass took 4.47 s for prefill and decoded at 2.98 tok/s with 3,924
expert misses. Peak sampled RSS was about 26.8 GiB because untouched experts were not
allocated. An earlier comparison between separate attention-batching binaries showed
decode falling from 4.75 to 4.40 tok/s. A subsequent same-process attention-only
crossover (four warmed passes per path) measured 4.69 versus 4.76 tok/s, so the
earlier difference does not establish a batching-caused decode regression. The
established benefit of batching and grouping is prefill latency. Other or longer
continuations can encounter new expert misses, and these short runs should not be
treated as universal speed limits.

Subsequent decode tuning overlaps the shared expert with the retained routed experts.
Two executables with the same vectorized state validation were run serially in
overlap/control/control/overlap order, with one initial pass and four warmed passes
per process. Across eight warmed passes per variant, on the same 32-token workload:

| Shared-expert schedule | MoE time per decode step | Overall decode tok/s |
| --- | ---: | ---: |
| After routed experts | 74.51 ms | 4.83 |
| Concurrent with routed experts | 65.82 ms | 4.90 |

MoE elapsed time decreased by 11.7%. Overall throughput varied from 4.75–4.93 tok/s
in the control and 4.72–5.12 tok/s with overlap; their overlapping ranges do not
establish a robust end-to-end gain from the 1.4% aggregate difference. Attention and
LM-head timings also varied between processes. Separately, a fixed-core microbenchmark
of a 2 MiB finite-value scan measured about 107 µs scalar versus 36 µs with AVX2;
the full-model comparisons did not isolate an end-to-end benefit from that scan alone.

The subsequent GQA value-layout comparison used the prompt-file command above:
493 rendered input tokens and a fixed 32-token continuation. Two groups reversed
the executable order. Initial passes were excluded; the first group had two warmed
passes per executable and the reverse group had one:

| Group | Prefill, columnwise → contiguous | Decode tok/s, columnwise → contiguous |
| --- | ---: | ---: |
| Control then candidate | 30.66 → 26.66 s | 3.68 → 3.91 |
| Candidate then control | 23.56 → 21.22 s | 4.87 → 5.13 |

These paired runs reduced warmed prefill by 10–13% and increased decode throughput
by 5–6%. Total attention time per decode step decreased from 159.25 to 132.67 ms
in the first group and from 119.68 to 94.66 ms in the reverse group. A fixed-core
microbenchmark of 16 query heads over 513 cached tokens measured value aggregation
alone at about 1.34 ms columnwise versus 0.19 ms contiguous; it excludes projections,
score computation, softmax, and the rest of the model.

For the original 19-token prompt, paired warmed decode measured 3.65 → 3.73 tok/s
(four passes each) and 5.99 → 6.16 tok/s in reverse order (two passes each). Absolute
timings changed substantially between groups even for unchanged executables, and
unmodified MoE/LM-head stages also varied. The 6.16 tok/s result describes this warmed
workload under those run conditions; the entire change from earlier measurements
must not be attributed to GQA optimization. The benchmark did not change power settings.
All paired runs preserved output tokens, every full-logit hash, expert hit/miss counts,
payload bytes read, and resident expert bytes.

Independent GQA-head parallelism uses the final 256K work threshold described above.
A subsequent paired run used one first-use pass and two warmed passes per executable,
with the same 20-worker, 90 GiB settings and 32-token continuation:

| Rendered prompt | Prefill, serial → parallel threshold | Decode tok/s, serial → parallel threshold |
| --- | ---: | ---: |
| 19 tokens | 0.811 → 0.835 s | 6.131 → 6.128 |
| 493 tokens | 20.789 → 18.259 s | 5.864 → 6.207 |

The longer input reduced warmed prefill by 12.2% and improved decode by 5.9% in this
comparison. The short workload stays below the head-parallelism threshold throughout
its continuation and showed essentially unchanged decode throughput. Both comparisons
preserved every output token, full-logit hash, and expert-cache counter. These are
paired workload measurements, not hardware-independent or long-context speed limits.

A separate resident LM-head experiment interleaved BF16 rows by column to keep
independent ordered F64 sums in SIMD lanes. Four-row tiles measured 6.120 → 6.121
and 6.165 → 6.190 tok/s in opposite execution orders; a 16-row variant measured
6.145 → 6.172 tok/s. All used two warmed passes on the 19-token/32-output workload,
with identical logits and cache counters. The four-row variant added about 65–70 ms
to model loading. These sub-0.5% overall differences did not establish a robust
benefit, so neither layout was retained; the LM head keeps its row-major format.

DeltaNet transaction reuse is enabled for batched prefill. The final comparison
retained per-call decode transactions and ran candidate then control, each with one
first-use pass and two warmed passes, using the same 20-worker/90 GiB configuration:

| Rendered prompt | First-use prefill, before → after | Warm prefill, before → after | Warm decode tok/s, before → after |
| --- | ---: | ---: | ---: |
| 19 tokens | 3.301 → 3.484 s | 0.825 → 0.764 s | 6.143 → 6.134 |
| 493 tokens | 24.666 → 22.620 s | 18.034 → 16.740 s | 6.123 → 6.181 |

Warm prefill decreased by 7.2–7.4%. For 493 input tokens, the state-update stage
(including transaction preparation) decreased from 5.793 to 4.641 s, accounting for
most of the overall reduction. Peak sampled RSS stayed approximately 26.81 GiB for
the short workload and 41.74 GiB for the long workload. All output tokens, full-logit
hashes, and expert-cache counters matched. First-use measurements still include
expert-cache misses and a populated OS page cache; the short first-use prefill did
not improve, and the small warmed decode differences do not establish a decode gain.
A variant retaining this workspace during decode showed inconsistent throughput and
slower first-use comparisons, so persistent reuse is limited to prefill.

Retained-layer prefetch was then compared with the same 20-worker/90 GiB settings,
32 generated tokens, and three passes per process. Each short-input group used the
opposite executable order; the longer input ran control then candidate:

| Workload | First-use prefill, synchronous → shared-pool prefetch | Warm decode tok/s, synchronous → shared-pool prefetch |
| --- | ---: | ---: |
| 19 input tokens, candidate then control | 3.326 → 2.342 s | 6.151 → 6.140 |
| 19 input tokens, control then candidate | 3.376 → 2.317 s | 6.208 → 6.053 |
| 493 input tokens, control then candidate | 22.227 → 19.288 s | 6.206 → 6.189 |

First-use prefill decreased by 30–31% on the short input and 13% on the long input.
Warm short-input prefill ranged from 0.75–0.79 s across variants; longer-input warm
prefill averaged 16.370 → 16.046 s. These warm differences do not establish a steady
throughput improvement; the optimization targets uncached retained-layer loading.
Every comparison preserved output tokens, full-logit hashes, expert hit/miss counts,
payload bytes, and resident expert bytes. All processes loaded 40 layers initially
and zero layers on either warmed replay. Long-input peak sampled RSS remained about
41.73 GiB, and both executables peaked at 22 threads including profiling.

An earlier variant used the separate I/O pool. It improved short first-use prefill
by only 3–4% while increasing layer-load and MoE elapsed times, so it was replaced by
the shared compute-pool schedule. As elsewhere, first-use means empty runtime weight
caches with a populated OS page cache, not a cold-storage measurement.

Complete forward/prefill execution was subsequently moved into one CPU-pool entry,
including the LM head. The comparison baseline already has retained-layer prefetch;
both variants use the same 20 workers, 90 GiB budget, 32 generated tokens, and three
passes per process. Group 1 runs candidate then control; group 2 reverses that order.
Warm rows exclude the initial pass, using mean prefill and total steps / total decode time:

| Input / group | First-use decode tok/s, before → after | Warm prefill seconds, before → after | Warm decode tok/s, before → after |
| --- | ---: | ---: | ---: |
| 19 tokens / 1 | 2.422 → 4.385 | 0.725 → 0.719 | 6.534 → 6.694 |
| 19 tokens / 2 | 2.744 → 4.418 | 0.719 → 0.710 | 6.543 → 6.795 |
| 493 tokens / 1 | 3.485 → 4.899 | 16.135 → 15.332 | 6.460 → 6.733 |
| 493 tokens / 2 | 3.367 → 4.917 | 16.315 → 15.446 | 6.527 → 6.646 |

First-use decode improved by 61–81% on the short workload and 41–46% on the long
workload; warmed decode improved by 1.8–4.2% across these pairs. Long warmed prefill
fell by about 5%. Short first-use prefill increased by 60–70 ms (about 3%), while
long first-use prefill decreased from 19.05–19.08 s to 18.65–18.70 s. The largest
benefit is during decode with expert-cache misses, not a general TTFT reduction.
These remain populated-OS-page-cache measurements of fixed replayed workloads.
All paired output tokens, full-logit hashes, expert-cache counters, and stage
call/work/byte counters matched. Peak sampled RSS stayed around 26.8 GiB for short
inputs and 41.8 GiB for long inputs; peak thread count stayed at 22, including profiling.
An intermediate variant entered the pool separately for each layer; long warmed
decode regressed by 4–8%, so that variant was replaced by the complete-forward entry.

A separate CLI comparison disabled profiling and used fresh runtime caches for each
process. The same OS page-cache caveat applies:

```bash
./target/release/urb generate /mnt/990pro/Qwen3.6-35B-A3B \
  --prompt 'Explain what a compiler does.' --no-thinking --ram-gib 90 \
  --max-new-tokens 8 --threads 20 --allow-large-model
```

| Execution order | Decode tok/s, before → after | TTFT, before → after | Total time, before → after |
| --- | ---: | ---: | ---: |
| Control then candidate | 1.88 → 4.04 | 3.02 → 3.01 s | 6.99 → 5.01 s |
| Candidate then control | 2.22 → 4.06 | 3.01 → 3.02 s | 6.52 → 4.96 s |

CLI stdout matched byte for byte in both pairs. These eight-token decode samples
include expert-cache misses and are separate from the warmed 32-token harness results.
The reduction in total time was 24–28%; TTFT remained about 3.0 seconds.

After the complete-forward pool entry, lowering GQA head parallelism from 256K to
64K work units was retested in opposite executable orders. Warm decode changed
6.516 → 6.568 tok/s and 6.513 → 6.482 tok/s; attention changed 70.223 → 69.799 ms
and 70.909 → 71.000 ms per step. This did not establish a stable benefit, so the
256K threshold was retained.

AVX2 bulk BF16 rounding was then compared against the complete-forward baseline,
with the 256K GQA threshold unchanged. A standalone cache-resident kernel benchmark
pinned to CPU 0 used 512, 2,048, 65,536, and 248,320 elements, about 50 million
values per timed sample, and scalar/vector/vector/scalar order. After initial warm-up,
scalar rounding took about 0.48–0.49 ns/value and AVX2 about 0.107–0.111 ns/value,
roughly a 4.4× kernel speedup. The whole-model measurements below use the same
20-worker/90 GiB setup, 32 generated tokens, and three passes per process. Group 1
runs control then candidate; group 2 reverses that order, excluding the initial pass:

| Input / group | Warm prefill seconds, scalar → AVX2 casts | Warm decode tok/s, scalar → AVX2 casts |
| --- | ---: | ---: |
| 19 tokens / 1 | 0.735 → 0.712 | 6.526 → 6.579 |
| 19 tokens / 2 | 0.707 → 0.685 | 6.972 → 7.043 |
| 493 tokens / 1 | 15.515 → 14.900 | 6.490 → 6.925 |
| 493 tokens / 2 | 15.256 → 14.954 | 6.836 → 6.915 |

Warm prefill decreased by 3.0–3.2% on short inputs and 2.0–4.0% on long inputs.
Three decode pairs improved by about 1%; the 6.7% long-input difference in group 1
did not repeat in reverse order and should not be generalized. Output tokens,
full-logit hashes, expert hit/miss/read/residency counters, and stage call/work/byte
counters matched in every pair. Peak sampled RSS remained about 26.8/41.8 GiB for
short/long inputs, with at most 22 threads including profiling. These measurements
also use a populated OS page cache and fixed replayed workloads.

Larger prefill chunks were evaluated in same-process crossover runs with eight
generated tokens. Both runs begin with a 32-token-chunk warm-up, then replay cleared
causal state with resident weights. The 493-token repeated prompt used
`32,32,64,128,256,256,128,64,32`; a 587-token technical design prompt used
`32,32,128,256,256,128,32`. An experimental build allowed 256 for this sweep:

| Chunk size | Warm prefill, 493-token prompt | Warm prefill, 587-token design prompt |
| --- | ---: | ---: |
| 32 | 15.216 s | 19.547 s |
| 64 | 13.153 s | — |
| 128 | 11.967 s | 15.505 s |
| 256 | 11.915 s | 15.476 s |

At 128 tokens, warmed prefill decreased by about 21% on both inputs. The additional
improvement at 256 was below 0.5%, so the retained maximum/default for full expert
caches is 128, with half the batch activation storage of 256. Partial caches keep
the 32-token default to preserve their existing opportunities for grouped execution.
Every warmed replay preserved all output/logit hashes and expert-cache counters.
The design prompt is available at `examples/prompts/qwen36_design.txt`; the retained
32/128 comparison can be reproduced with:

```bash
./target/release/examples/profile_qwen36 /mnt/990pro/Qwen3.6-35B-A3B \
  90 20 32 5 32,32,128,128,32 examples/prompts/qwen36_design.txt
```

The final implementation was also compared in separate processes with 32 generated
tokens and three passes each. The 493-token comparison ran control then candidate;
the design-prompt comparison reversed that order:

| Input | First-use prefill, 32 → 128 | Warm prefill, 32 → 128 | Warm decode tok/s, 32 → 128 |
| --- | ---: | ---: | ---: |
| 493 tokens | 19.958 → 15.758 s | 16.083 → 12.626 s | 5.920 → 5.899 |
| 587-token design prompt | 23.754 → 20.072 s | 19.582 → 15.731 s | 6.071 → 5.953 |

First-use prefill decreased by 15–21% and warmed prefill by 20–21%. Warm decode
was 0.4–1.9% lower in these pairs; no decode improvement is claimed. On the
493-token workload, warm MoE time decreased from 8.253 to 4.568 s while attention
changed from 7.631 to 7.852 s. Peak sampled RSS was 41.73 → 41.77 GiB and
43.92 → 43.90 GiB respectively; peak thread count remained 22 including profiling.
All output tokens, full-logit hashes, expert-cache counters, and decode stage
call/work/byte counters matched. First-use continues to mean empty runtime caches
with a populated OS page cache. Native-checkpoint coverage now spans 129 input
tokens after an existing prefix, 32/64/128 chunk sizes, a singleton tail, rewind,
and two continuation steps compared with incremental execution.

An unprofiled CLI comparison used the 493-token input, eight generated tokens,
20 workers, and 90 GiB, letting each executable select its default prefill chunk.
Control then candidate measured TTFT 19.31 → 15.90 s and total time 21.34 → 17.79 s;
decode was 4.67 → 4.69 tok/s. CLI stdout matched byte for byte. A separate profiler
smoke run without `BATCH_SIZES` confirmed automatic selection of 128 at 90 GiB.

Resident harness runs preserved output tokens and every full-logit hash from the original
streaming benchmark. Grouped and tokenwise MoE runs also preserved all 32-token logit
hashes, expert hit/miss counts, bytes read, and resident expert bytes. CLI stdout matched
byte for byte. Seven native-checkpoint tests passed,
including partial-prefix load counts, resident/streamed exact parity, checkpoint replay,
chunk-boundary continuation,
and the existing Transformers accuracy gates without changing their tolerances.
Tests also compare tokenwise versus grouped cache access and subsequent LRU evictions.
The complete default
and no-UI suites and Clippy checks also passed.

### Paired-row BF16 batch projections

The paired-row kernel was compared with commit `14937cd` using separate release
executables, 20 CPU workers, a 90 GiB RAM budget, 128-token prefill chunks, 16 generated
tokens, and three passes per process on the same i7-14700K. Runs were serial, with no
concurrent builds or tests. Warm results below average the final two passes after
rewinding causal state and retaining the workload's weights. The OS page cache was
populated for warm runs; the first design control additionally performed storage
reads, so its initial prefill is excluded from the comparison.

| Input / order | Warm prefill seconds, control → paired rows | Reduction | Warm decode tok/s, control → paired rows |
| --- | ---: | ---: | ---: |
| 587-token design / control then candidate | 16.395 → 14.672 | 10.5% | 5.900 → 7.323 |
| 587-token design / candidate then control | 15.337 → 14.246 | 7.1% | 7.125 → 7.316 |
| 19-token short prompt / control then candidate | 0.693 → 0.670 | 3.3% | 7.256 → 7.315 |
| 493-token repeated prompt / control then candidate | 12.181 → 11.409 | 6.3% | 6.973 → 7.004 |

The reverse-order design comparison reduced DeltaNet input/output projection time
from 1.884/0.590 s to 1.443/0.431 s. Its recurrent-state stage remained about 5.3 s;
total attention time decreased from 9.309 to 8.505 s. This leaves recurrence as a
larger fraction of prefill work. Single-token GEMV was unchanged, and control decode
throughput varied substantially between the design runs, so these results do not
establish a decode speedup.

Every pair matched all generated tokens, full-logit hashes, expert hit/miss/read and
resident-byte counters, and prefill/decode stage call/work/byte counters. Peak RSS
was about 22.76 GiB for the short input, 39.35–39.36 GiB for the repeated prompt, and
43.60–43.62 GiB for the design prompt; paired differences were under 0.02 GiB.
Peak sampled thread count remained 22 including profiling. Weight residency, the
prefill chunk policy, and the scratch budget were unchanged.

Use the same profiler invocation for each executable under comparison:

```bash
./target/release/examples/profile_qwen36 /mnt/990pro/Qwen3.6-35B-A3B \
  90 20 16 3 128 examples/prompts/qwen36_design.txt
```

The short case omits the prompt file. The repeated case uses the 80-repetition
prompt above. These are fixed replayed workloads, not a throughput guarantee for
new text or other hardware.

Validation passed all 453 default-feature tests, 414 no-UI tests, seven native
Qwen3.6 checkpoint gates, both Clippy configurations, and the release CLI build.
The Transformers tolerances and expected native argmaxes were unchanged.

### DeltaNet prefill by value head

The next comparison uses the paired-row implementation above as its control. Both
executables use 20 workers, a 90 GiB budget, 128-token chunks, 16 generated tokens,
and three passes per process on the same i7-14700K. Each process starts with an
initial pass, then rewinds causal state for two warmed replays. Runs are serial,
with no concurrent builds or tests. The table reports the mean warmed prefill time
and aggregate decode throughput across those two replays:

| Input / order | Warm prefill seconds, control → head-major | Reduction | Warm decode tok/s, control → head-major |
| --- | ---: | ---: | ---: |
| 587-token design / candidate then control | 14.708 → 12.389 | 15.8% | 6.335 → 6.295 |
| 587-token design / control then candidate | 14.769 → 12.431 | 15.8% | 6.292 → 6.232 |
| 493-token repeated prompt / candidate then control | 11.549 → 9.703 | 16.0% | 6.160 → 6.197 |
| 19-token short prompt / control then candidate | 0.758 → 0.727 | 4.0% | 5.832 → 5.892 |

For the two design pairs, the DeltaNet state-update stage decreased from
5.452 → 3.130 s and 5.342 → 3.044 s, a 42.6–43.0% reduction. Both design initial
passes had no measured storage reads; initial prefill decreased by 11.1–12.3%.
The repeated-prompt candidate and short-prompt control additionally read about
1.42 GB and 74 MB from storage on their initial passes, so those initial timings
are excluded from performance comparisons. All warmed passes had zero expert
misses and zero measured storage reads. Decode changed by approximately -1% to
+1%; these runs establish a prefill improvement, not a decode improvement.

All three passes in every pair matched output tokens, full-logit hashes, expert
hit/miss/read/residency counters, and prefill/decode stage call/work/byte counters.
Peak sampled RSS was about 22.8, 39.3, and 43.6 GiB for the short, repeated, and
design inputs; paired differences were below 0.07 GiB. Peak thread count remained
22 including profiling. The additional bounded activation storage is covered by
the 158.125 MiB conservative scratch estimate above; cache budgets and the prefill
chunk policy are unchanged.

Before changing the schedule, a separate AVX2 experiment fused state decay/update
with their projections. Opposite-order design pairs measured prefill changes of
14.910 → 14.906 s and 14.786 → 14.925 s with matching logits. It did not establish
a benefit, so the fusion was removed. The retained implementation uses the
existing staged FP32 updates and ordered F64 projection kernel.

The profiler commands in the paired-row section also reproduce this comparison.
Local raw reports and verified counter comparisons are saved as
`profiles/qwen36-headbatch-*.json`; `profiles/qwen36-delta-verified-*.json` records
the rejected fusion experiment. Results describe fixed replayed workloads and
exclude model loading from prefill latency.

Validation passed 456 default-feature tests, 417 no-UI tests, all seven native
Qwen3.6 checkpoint gates, both Clippy configurations, formatting checks, and the
release CLI build. Head-major regression tests cover FP32/BF16, 128-wide and wider
heads, an existing prefix and continuation, late-token failure, and reuse of a
failed transaction workspace. The upstream accuracy tolerances are unchanged.

### Ordered LM-head reductions with fewer workers

A same-binary worker sweep on the head-major baseline used the 19-token prompt,
32 generated tokens, a 90 GiB budget, and two warmed replays per process. In run
order, 20/8/12/16/28/20 workers achieved 6.273/6.338/6.254/6.202/5.534/6.232 tok/s.
Outputs, full-logit hashes, and expert-cache counters matched across the sweep.
The small differences among 8–20 workers did not justify changing the physical-core
default; enabling all 28 logical CPUs was slower on this host.

The ordered BF16/F64 kernel was then changed to keep eight independent row sums
in AVX2 lanes, with four-column transposes in registers. A CPU-0-pinned standalone
test of 248,320 × 2,048 synthetic BF16 weights measured 188.550 → 93.755 ms for the
projection, with identical outputs. This measures one core, not whole-model speed.
An unrestricted whole-model version showed no benefit at eight workers and slight
regressions at twenty workers, so the retained policy limits parallel row tiles to
at most four workers and requires enough tiles to occupy the pool.

Low-worker comparisons use the same 19-token prompt, 16 generated tokens, three
passes per process, 128-token prefill chunks, and a 90 GiB budget. Both variants
have the preceding head-major prefill optimization. Initial passes are excluded;
decode throughput is total warmed decode steps divided by their elapsed time:

| Workers / comparison order | Warm decode tok/s, control → tiled | Increase |
| --- | ---: | ---: |
| 1 / initial control then candidate | 2.249 → 2.846 | 26.6% |
| 1 / final candidate then control | 2.228 → 2.830 | 27.0% |
| 4 / initial candidate then control | 5.186 → 5.628 | 8.5% |
| 4 / final control then candidate | 5.185 → 5.620 | 8.4% |

The final policy uses the same tiled kernel at one and four workers. In its final
pairs, LM-head time per decode step fell from 189.486 → 93.336 ms with one worker
and from 48.472 → 32.532 ms with four. The final 20-worker policy was separately
checked with 32 generated tokens in both executable orders: 6.196 → 6.240 and
6.202 → 6.216 tok/s. Those sub-1% differences do not establish a high-worker decode
gain. Prefill timings varied across the paired processes, including unchanged
attention stages, so no prefill improvement is attributed to this kernel.

Every compared pass matched output tokens, full-logit hashes, expert-cache
hit/miss/read/residency counters, and stage call/work/byte counters. No weight copy,
extra persistent allocation, lower precision, cache-budget change, or thread-count
change is introduced. Raw final reports are `profiles/qwen36-ordered-final-*.json`;
`profiles/qwen36-ordered-rows-*.json` preserves the unrestricted experiment, and
`profiles/qwen36-thread-sweep-*.json` records the worker sweep.

Reproduce the one-worker final comparison with each executable:

```bash
./target/release/examples/profile_qwen36 /mnt/990pro/Qwen3.6-35B-A3B 90 1 16 3 128
```

Use `4` in place of `1` for the four-worker case. These are fixed, resident replay
measurements; they do not guarantee the same speedup on other hardware or new text.

Validation passed 458 default-feature tests, 419 no-UI tests, all seven native
checkpoint gates, both Clippy configurations, formatting checks, and the release
CLI build. New scalar-oracle comparisons cover row ranges, complete and partial
tiles, column tails, cancellation, signed zero, subnormal inputs/weights, and finite
inputs whose output overflows F32. Upstream tolerances and cache budgets are unchanged.

### Rejected lossless expert-cache experiment

A subsequent trial stored each 32-weight block with a shared four-bit exponent
prefix and twelve bits per BF16 value, retaining raw blocks when prefixes differed.
All weight bits were preserved, including very small nonzero values in layer 0.
Sampled expert matrices used about 80% of their original bytes including block
metadata. A 256 MiB standalone matrix test with 20 workers improved from about
8.1 to 6.1 ms; that improvement did not carry through to whole-model inference.

The final experimental version used AVX2 packing and a bounded stack buffer to
reuse decoded rows across batched inputs. It was compared against the preceding
implementation on the same host, with a 90 GiB budget, 128-token chunks, 32
generated tokens and three passes per process. Runs were serial; warm metrics
exclude the initial pass and aggregate both replay passes:

| Input / workers / executable order | Warm prefill seconds, control → trial | Warm decode tok/s, control → trial |
| --- | ---: | ---: |
| 587-token design / 20 / trial then control | 12.082 → 11.982 | 6.846 → 6.991 |
| 587-token design / 20 / control then trial | 12.170 → 11.903 | 6.780 → 6.991 |
| 19-token short / 20 / control then trial | 0.642 → 0.613 | 6.821 → 6.925 |
| 19-token short / 8 / trial then control | 0.639 → 0.610 | 6.826 → 6.989 |

The resulting warm decode gains were only 1.5–3.1%. In the eight-worker short
comparison, neither initial pass performed measured storage reads, yet initial
prefill plus decode increased from 8.767 to 9.458 s, a 7.9% regression. The first
design trial and short 20-worker control had additional storage reads, so their
initial timings are not used to isolate packing overhead. Logical expert storage
fell about 21%, but peak RSS fell by less than 1 GiB in these final comparisons;
allocator retention prevented comparable physical-memory savings.

All paired passes matched generated tokens, full-logit hashes, expert access/read
counters and stage call/work/byte counters. Actual resident expert byte counts
intentionally differed. Every warm pass had zero expert misses and measured
storage reads. The prototype passed 462 default-feature tests and Clippy, including
exhaustive finite-BF16 round trips and exact projection comparisons. Earlier heap
buffers for decoded rows caused a large prefill regression and were replaced in
the measured final trial.

This tradeoff did not justify enabling the new cache encoding by default. The
runtime changes were removed and the preceding validated sources restored
exactly. Local reports are `profiles/qwen36-lossless-final-*.json`; the survey,
microbenchmarks and earlier trials use `profiles/qwen36-lossless-*`. Experimental
sources and binaries remain under `target/qwen36-lossless-session-20261003/`.

A separate batch-kernel microbenchmark compared four, five, six and eight input
tokens per two-row tile at 8 and 20 workers. Larger tiles did not consistently
improve representative 512/4,096-row projections; eight-input tiles regressed
substantially. The existing four-input tile was retained. Outputs matched bit for
bit; local measurements are saved in `profiles/qwen36-tile-micro.log`.

A final scheduling trial kept expert batches of at most eight tokens on one
worker when the outer expert loop already had at least one group per worker.
The 587-token, 20-worker comparison measured 12.302 → 12.224 s warm prefill and
6.926 → 6.810 tok/s decode, with identical logits and all cache/stage counters.
That did not establish a useful overall improvement, so the scheduling change
was also removed. Reports are `profiles/qwen36-granularity-*.json`; the attempted
scope guard and runtime changes are archived under
`target/qwen36-granularity-session-20261003/rejected-source/`.

### Ordered GQA key-score tiles

The following comparison adds eight-key AVX2 score tiles to the preceding validated
implementation. Both executables use 20 workers on the same i7-14700K, a 90 GiB
budget, 128-token chunks, and three passes per process. Long and design inputs
generate 16 tokens; the short input generates 32. Runs are serial, without concurrent
builds or tests. Warm prefill is the mean of the two replay passes; decode throughput
is their total decode steps divided by total decode time:

| Input / executable order | Warm prefill seconds, control → tiled | Reduction | Warm decode tok/s, control → tiled |
| --- | ---: | ---: | ---: |
| 1,933-token repeated prompt / candidate then control | 53.847 → 43.144 | 19.9% | 6.386 → 6.529 |
| 1,933-token repeated prompt / control then candidate | 51.273 → 43.598 | 15.0% | 6.413 → 6.541 |
| 587-token design / candidate then control | 12.487 → 11.901 | 4.7% | 6.562 → 6.689 |
| 19-token short prompt / control then candidate | 0.651 → 0.632 | 3.0% | 6.715 → 6.695 |

Most of the long-input prefill reduction was in attention: 36.179 → 26.453 s and
34.022 → 26.588 s in the two orders. The short-input difference includes variation
in unchanged DeltaNet stages and does not establish a short-input speedup. Decode
improved about 2% for the long/design workloads and was essentially unchanged for
the short prompt. These are fixed replayed workloads, not a throughput guarantee.

The reversed long comparison had no measured storage reads on either initial pass;
initial prefill decreased from 56.842 to 47.874 s (15.8%). The first long candidate
additionally read about 1.68 GB from storage, so its initial timing is excluded from
that comparison. Model loading is outside the reported prefill interval. Every warm
pass had zero expert misses and zero measured storage reads.

All paired passes matched generated tokens, full-logit hashes, expert hit/miss/read/
residency counters, and stage call/work/byte counters. Peak thread count stayed at
22 including profiling. Peak sampled RSS was about 43.2 GiB for the long input,
43.6 GiB for design, and 26.8 GiB for short; paired differences were below 0.07 GiB.
No weight conversion, expanded KV cache, precision change, cache-budget change,
or prefill-policy change is introduced.

Reproduce the long workload with each executable:

```bash
python3 -c 'print("Explain what a compiler does. " * 320, end="")' > /tmp/qwen36-gqa-long.txt
./target/release/examples/profile_qwen36 /mnt/990pro/Qwen3.6-35B-A3B 90 20 16 3 128 /tmp/qwen36-gqa-long.txt
```

Raw reports and verified comparisons are `profiles/qwen36-gqa-scores-*.json`.
`profiles/qwen36-gqa-scores-micro.log` records an isolated score benchmark with 16
query heads and two KV heads, running inside the persistent worker pool. At 20
workers, 512/2,048/8,192 cached keys took 0.448/1.149/3.331 → 0.203/0.418/1.513 ms,
respectively, with identical outputs. Whole-model results are reported separately
above because kernel speedups do not directly determine generation throughput.

Validation passed 460 default-feature tests, 421 no-UI tests, all seven native
checkpoint gates, both Clippy configurations, formatting checks, and the release
CLI build. New scalar-oracle tests cover KV strides, native/wide heads, column tails,
cancellation, subnormal inputs, signed zero and finite inputs whose score overflows
F32. Existing attention tests retain exact serial/parallel and batched/incremental
comparisons and cache rollback on failure. Upstream accuracy tolerances are unchanged.

### Rejected register-blocked GQA value aggregation

Two subsequent prototypes kept several dimension sums in AVX2 registers across
small token blocks, preserving the original ordered F64 multiply/add operations.
The first used 32 columns × 64 tokens; the second used 16 columns × 8 tokens.
Both kept the existing KV layout, precision, cache policy and short-context fallback.
Isolated value kernels improved: one 20-worker, 512-token measurement decreased from
0.077 to 0.042 ms with identical sums. Complete attention microbenchmarks also
improved, but neither result predicted the native model's prefill behavior.

Serial native comparisons used the preceding validated score-tile implementation
as control, 20 workers, a 90 GiB budget, 128-token chunks, and three passes per
process. The table averages the two warm passes. Long/design runs generated 16
tokens; short runs generated 32:

| Prototype / input / executable order | Warm prefill seconds, control → prototype | Increase |
| --- | ---: | ---: |
| 32 × 64 / 1,933 tokens / control then candidate | 43.200 → 45.277 | 4.8% |
| 32 × 64 / 1,933 tokens / candidate then control | 42.971 → 45.794 | 6.6% |
| 16 × 8 / 1,933 tokens / candidate then control | 43.299 → 44.340 | 2.4% |
| 32 × 64 / 587-token design / candidate then control | 11.793 → 11.871 | 0.7% |
| 32 × 64 / 19-token short / control then candidate | 0.663 → 0.677 | 2.0% |

The larger prototype's long-input attention stage increased from 26.244 to
28.025 s and from 26.183 to 28.651 s in the two orders. The smaller prototype
also increased attention time, from 26.566 to 27.406 s. Decode changes were
inconsistent: the larger prototype's long pairs changed from 6.410 to 6.442 and
from 6.677 to 6.474 tok/s; the smaller prototype changed from 6.634 to 6.747 tok/s.
Neither implementation was retained.

All 30 paired passes matched generated tokens, full-logit hashes, expert hit/miss/
read/residency counters, and stage call/work/byte counters. Every warm pass had zero
expert misses; every measured pass had zero storage reads. Paired peak RSS
differences stayed below 0.09 GiB, and peak thread count remained 22.

Additional instrumentation checked over 607 million probability values across two
long-input passes and found no subnormal values, while retaining output and cache
parity. Synthetic subnormal tests also did not support repeated probability
conversion as the explanation. A separate instrumented pair accumulated per-head
elapsed times: in its warm pass, score/softmax time rose approximately 42.842 →
52.120 s and value aggregation rose 30.302 → 31.569 s. These sums overlap across
workers and include scheduling; they are diagnostics, not end-to-end latency or
cache-miss measurements. They show that an isolated value-loop speedup can be lost
when combined with adjacent work in the real model.

The prototype scalar-oracle tests passed, including signed zero, cancellation,
subnormal inputs, column tails and token-block boundaries. After rejection, the
tracked source was restored exactly to the round's initial diff, and rebuilt
release CLI and profiler binaries matched their control SHA256 hashes exactly.
That restored runtime became the control for the shared-value experiment below,
which measured the native attention path before proceeding with kernel tuning.

Raw reports are `profiles/qwen36-gqa-value-blocks-*.json` and
`profiles/qwen36-gqa-small-blocks-*.json`. Microbenchmarks, prototype sources,
instrumentation, executable hashes and restoration evidence are archived in
`target/qwen36-gqa-values-session-20261003/` and the corresponding profile logs.

### GQA value loads shared across query heads

The runtime now shares each loaded value vector across the eight query heads of
one KV group, retaining the ordered F64 sums described above. The grouped path
starts at 128 attended tokens. It was compared with the preceding
score-tile implementation on the same i7-14700K, with a 90 GiB budget and 128-token
prefill chunks. Long/design runs generate 16 tokens; short runs generate 32. Runs
are serial without concurrent builds or tests. Each process has three passes,
except the reversed design comparison, which has five. Warm prefill excludes the
first pass; warm decode divides total replay decode steps by total decode time:

| Input / workers / executable order | Warm prefill seconds, control → shared values | Reduction | Warm decode tok/s, control → shared values |
| --- | ---: | ---: | ---: |
| 1,933 tokens / 20 / candidate then control | 43.219 → 38.397 | 11.2% | 6.689 → 6.783 |
| 1,933 tokens / 20 / control then candidate | 42.237 → 37.795 | 10.5% | 6.663 → 6.808 |
| 1,933 tokens / 8 / control then candidate | 42.957 → 42.641 | 0.7% | 6.670 → 6.692 |
| 587-token design / 20 / candidate then control | 11.865 → 12.216 | -3.0% | 6.900 → 6.838 |
| 587-token design / 20 / control then candidate, five passes | 11.904 → 11.931 | -0.2% | 6.893 → 6.808 |
| 19-token short / 20 / control then candidate | 0.612 → 0.627 | -2.5% | 6.977 → 6.959 |

The long-input improvement at 20 workers repeated in both executable orders.
Its attention stage decreased 26.405 → 22.166 s and 25.686 → 21.755 s. Initial
prefill also decreased 47.176 → 42.375 s and 46.933 → 42.984 s; model loading is
outside these intervals, and every measured pass had zero storage reads, so these
are not cold-storage measurements. The eight-worker comparison is nearly flat.

The first design-prefill regression mostly appeared in the unchanged DeltaNet
input projection; reversing execution order and using four warm passes reduced
the total difference to 27 ms. The short-prefill difference was 15 ms and appeared
in unchanged DeltaNet state updates. Neither comparison establishes a short or
medium input improvement. Medium-input decode remained 0.9–1.2% slower in both
orders, with increased DeltaNet state-update time. These timings do not identify
the underlying cause; the retained long-input benefit is not a universal speedup.

A separate candidate raised the crossover to 1,024 attended tokens. Its five-pass
design comparison measured 11.956 → 11.957 s warm prefill and 6.834 → 6.704 tok/s
decode, despite retaining the original head path throughout that input. Its long
comparison measured 42.604 → 37.783 s and 6.443 → 6.448 tok/s. Raising the threshold
did not resolve the medium-input decode cost or establish an improvement over the
more broadly measured 128-token version, which is retained. All 16 paired passes
in this threshold experiment also matched tokens, full-logit hashes and cache/stage
counters. Reports are `profiles/qwen36-gqa-shared-threshold-*.json`.

All 40 paired passes matched generated tokens, full-logit hashes, expert hit/miss/
read/residency counters, and stage call/work/byte counters. Every warm pass had zero
expert misses. Paired peak sampled RSS differed by less than 0.07 GiB; peak thread
counts were 22 with 20 workers and 10 with eight workers. Cache budgets, prefill
policy, weight representation, and numerical tolerances are unchanged.

Validation of the retained implementation passed 462 default-feature tests, 423 no-UI
tests, all seven native checkpoint gates, both Clippy configurations, formatting,
and the release CLI build. New tests compare grouped values with scalar F64 sums
bit for bit across column tiles, token-block boundaries, signed zero, cancellation,
subnormals and extreme finite inputs. Serial/grouped contexts match in FP32 and
BF16, including the dispatch boundary. Tests also check score/context error order
and unchanged KV state after output failure. Release assembly uses no YMM stack
spills or fused multiply/add in the grouped value helper.

Raw reports and verified comparisons are `profiles/qwen36-gqa-shared-values-*.json`.
The long workload is the same repeated prompt and profiler command shown above;
set the worker argument to `8` for that comparison. Executables, source snapshots,
SHA256 hashes and validation logs are archived in
`target/qwen36-gqa-shared-values-session-20261003/`.

### DeltaNet decode validation and gating on each head's worker

The decode path previously joined all recurrent head updates, scanned their complete
state on the calling worker, and normalized their outputs serially. The retained
candidate completes those checks and normalization alongside each head update, then
selects errors in the original phase/head order. Numerical reductions, BF16 casts,
cache policies and transactional publication are unchanged.

The earlier GQA comparison's medium-input timing difference did not correspond to
different normalized machine instructions in DeltaNet's input preparation, head
update or AVX2 state projection. That observation does not rule out effects from
instruction placement or scheduling. This change targets the remaining serial
decode work; it does not establish the cause of that earlier difference.

On the same i7-14700K, two 20-worker, 90 GiB comparisons used the 587-token design
input, 64 generated tokens and five passes per executable. Excluding the first
pass, control-then-candidate decode changed 6.725 → 6.868 tok/s (+2.1%), with
DeltaNet state-update time 15.117 → 12.694 ms/token. Reversing execution order
gave 6.656 → 6.691 tok/s (+0.5%) and 17.192 → 15.728 ms/token. Warm prefill changed
12.010 → 11.881 s and 12.036 → 12.146 s, respectively; the batched prefill schedule
is unchanged.

Additional 90 GiB checks used three passes and 32 generated tokens. At 20 workers,
the 1,933-token input measured 38.790 → 37.891 s warm prefill and 6.386 → 6.715 tok/s
decode in candidate-then-control order. Several unchanged projection stages were
also faster in that pair, so its 5.2% decode difference is not treated as a stable
gain from this change alone. At eight workers, a control-then-candidate comparison
with the 19-token input measured 0.675 → 0.671 s and 6.367 → 6.478 tok/s.

Performance acceptance also includes several RAM plans, so a favorable full-cache
result alone is insufficient. The following paired runs use the first two paragraphs
of `examples/prompts/qwen36_design.txt` (132 rendered tokens), 20 workers, and each
plan's default chunk size. There are three passes per executable for 16 generated
tokens and five for 64. Replay metrics exclude the first pass, even when the expert
working set exceeds cache capacity. Decode throughput divides total replay decode
steps by total decode time. Request time is prefill plus those decode steps and
excludes initial model loading:

| RAM budget | Cached layers / expert slots per layer | Chunk | Generated tokens | Replay prefill seconds, control → candidate | Replay decode tok/s, control → candidate | Replay request seconds, control → candidate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 GiB | 35 / 0 | 32 | 16 | 36.055 → 36.124 | 2.180 → 2.173 | 42.936 → 43.027 |
| 16 GiB | 40 / 49 | 32 | 16 | 17.874 → 17.754 | 4.111 → 4.126 | 21.523 → 21.390 |
| 32 GiB | 40 / 118 | 32 | 16 | 9.325 → 9.428 | 5.994 → 5.976 | 11.828 → 11.938 |
| 90 GiB | 40 / 256 | 128 | 16 | 2.808 → 2.845 | 7.037 → 7.020 | 4.939 → 4.982 |
| 32 GiB, reversed order | 40 / 118 | 32 | 64 | 10.077 → 10.016 | 6.025 → 6.084 | 20.533 → 20.371 |
| 90 GiB, reversed order | 40 / 256 | 128 | 64 | 2.753 → 2.772 | 7.023 → 7.114 | 11.724 → 11.627 |

The 16-token comparisons are nearly flat; small prefill differences occur in the
unchanged batch path. Increasing the decode sample and reversing executable order
showed approximately 1.0% and 1.3% decode improvements at 32 and 90 GiB. The
4/16 GiB cases do not establish a throughput improvement. No memory budget trades
away expert capacity or expands its prefill scratch to obtain these results.

Replay expert misses were 47,040 / 11,089 / 2,942 / 0 in the four 16-token cases.
The extended 32 GiB case had 4,150 misses per replay; 90 GiB had none. Every paired
pass matched token IDs, full-logit hashes, expert hit/miss/read/residency counters,
and stage call/work/byte counters. Across all four budgets, tokens and full-logit
hashes also matched for each continuation length despite different chunk policies.

These are runtime allocation budgets on a 128 GiB host, not measurements on machines
with those physical RAM capacities. Every replay had zero measured storage reads;
the first 4 GiB control pass read about 60 MiB from storage. Nevertheless, the 4 GiB
runtime read 275.625 GiB of logical expert payload per request from the OS page
cache. The measurements therefore do not establish throughput under cold-storage
or low-physical-memory page-cache pressure.

The 32 GiB plan also exposed an existing memory-accounting limit: sampled RSS rose
from about 31.8 GiB after the first request to 33.1 GiB after replay in the control,
while retained expert payload stayed at 27.586 GiB. The candidate also reached
about 33.0 GiB. Thus the logical plan does not ensure a process RSS cap. The
expert-buffer-reuse comparison below investigates recurring allocations without
reducing expert capacity; it does not turn the planner into a hard RSS limit.

Validation passed 464 default-feature tests, 425 no-UI tests, all seven native
checkpoint gates, both Clippy configurations, formatting and the release CLI build.
The final profiler binary matches the measured candidate's SHA256. New tests
compare decode against the original staged schedule bit for bit in FP32/BF16,
including native/wide heads, repeated steps, existing convolution/recurrent state,
signed zero and subnormal inputs. Mixed failures verify that late update/state/core
cast errors still precede earlier heads' normalization failures. Existing tests
cover transaction rollback and reuse after failure; numerical tolerances are unchanged.
The complete set of 76 measured passes preserved paired tokens, full-logit hashes
and cache/stage counters.

### Reusing native expert buffers within the transient reserve

The Qwen3.6 store now retains at most one displaced or uncached native expert in
the existing 6 MiB transient-expert reservation. On the next miss it recovers the
three owned BF16 projection buffers only if no caller still holds that expert.
Positional reads overwrite the same allocations, followed by the original shape
and finite-value checks. Loading and validation still precede LRU eviction, so a
short or corrupt read cannot discard a usable cached expert. Shared buffers are
never overwritten; cache hits, eviction order, slot counts and logical read
counters retain their original semantics. Cache telemetry continues to count LRU
payload, while the spare is charged to the existing transient reserve.

Sparse-fixture tests verify exact allocation reuse and replacement contents,
outstanding readers in zero-slot and one-slot stores, and cache preservation and
recovery after a failed reused-buffer load. No weight packing, precision change,
extra expert slots or larger scratch budget is involved.

The retained candidate reuses expert buffers only; incremental DeltaNet decode
keeps its existing per-call transaction. On the same i7-14700K/128 GiB host,
paired comparisons used the same 132-token input, 20 workers and default chunks
(32 for partial expert caches, 128 for the full-cache plan). Three passes generated
16 tokens each; the reversed-order comparisons used five passes and 64 tokens.
Replay metrics exclude the first pass and model loading. RSS is the maximum
100 ms sample across all passes, including the first:

| RAM budget | Generated tokens | Replay prefill seconds, control → reuse | Replay decode tok/s, control → reuse | Replay request seconds, control → reuse | Peak RSS GiB, control → reuse |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 GiB | 16 | 37.772 → 36.206 | 1.973 → 2.131 | 45.377 → 43.245 | 3.652 → 3.646 |
| 16 GiB | 16 | 18.234 → 18.278 | 3.902 → 3.885 | 22.079 → 22.139 | 15.577 → 15.310 |
| 32 GiB | 16 | 9.876 → 9.457 | 5.668 → 5.666 | 12.523 → 12.105 | 36.011 → 31.415 |
| 90 GiB | 16 | 2.930 → 2.948 | 6.697 → 6.606 | 5.170 → 5.219 | 35.843 → 35.845 |
| 32 GiB, reversed order | 64 | 10.273 → 10.066 | 5.828 → 5.771 | 21.082 → 20.984 | 42.824 → 31.547 |
| 90 GiB, reversed order | 64 | 2.822 → 2.896 | 6.741 → 6.618 | 12.169 → 12.416 | 38.258 → 38.188 |

This is a memory-stability improvement with a measured speed tradeoff, not a
uniform throughput gain. The 4 GiB request was 4.7% faster, 16 GiB was nearly
flat, and 32 GiB requests were 3.3% and 0.5% faster in the two comparisons.
The 90 GiB decoder was 1.4% and 1.8% slower, with request time increasing 1.0%
and 2.0%; that cost remains an optimization target. Unchanged kernels also
contributed to the timing differences. For example, the ordered LM-head dot
product has identical machine instructions after normalizing relocation addresses
between the control and buffer-only binaries; its placement differs. These data
do not establish which code-placement, allocation or scheduling effects caused
the full-cache slowdown.

At 32 GiB, both versions retain exactly 27.586 GiB of expert payload in 118 slots
per layer. The control's RSS varies substantially between processes and grows on
replay; the reuse candidate stayed below 32 GiB in both measured processes without
sacrificing slots. In the first comparison, first-replay minor faults fell from
942,652 to 521. In the extended comparison, the control grew from 36.812 to
42.824 GiB, while the candidate went from 31.521 to 31.547 GiB across five requests.
The remaining small allocations and allocator behavior are not subject to a hard
RSS cap, so these observations are specific to the measured workloads.

All 44 final-candidate/control passes matched token IDs, full-logit hashes,
expert hit/miss/read/residency counters, and stage call/work/byte counters. Outputs
also matched across RAM plans and repeated requests; the first 16 generated
tokens and their logits hashes matched the 64-token continuations. Every measured
pass had zero storage reads and zero process swap. These remain runtime-budget
comparisons on a 128 GiB host with warm OS page cache, rather than measurements
under the page-cache pressure of physically smaller machines.

An additional decode-workspace-reuse variant reduced allocation churn but did not
provide a consistently favorable speed tradeoff. An entirely-resident-batch
shortcut also failed to remove the full-cache timing loss. Neither is retained;
the final source keeps the narrower expert-buffer change. Reproducible commands,
binary/source hashes and all measurements are recorded in
`target/qwen36-expert-buffer-only-session-20261004/metadata.json` and
`profiles/qwen36-expert-buffer-only-memory-*.json`; the other variants have separate
`decode-workspace`, `expert-buffer-reuse` and `expert-buffer-hit` experiment records.

Final validation passed 464 default-feature tests, 425 no-UI tests, all seven native
checkpoint gates, both Clippy configurations, formatting and diff checks, and the
release CLI build. The rebuilt profiler exactly matches the measured candidate's
SHA256. Throughput figures above come from the profiling example; unprofiled CLI
throughput has not been measured for this change.

Raw results are `profiles/qwen36-delta-decode-heads-*.json` and
`profiles/qwen36-delta-memory-balanced-*.json`. The latter summaries intentionally
allow replay misses, record logical reads separately from storage reads, and include
sampled RSS. Benchmark commands, source/binary hashes, numerical comparisons and
validation logs are archived in `target/qwen36-delta-decode-session-20261003/`.

### Checking expert-buffer reuse without active profiling

The benchmark's optional `PROFILE=off` mode permits timing the native runtime
without stage collection or a resource-sampling thread. Rebuilt controls use the
same driver with and without expert-buffer reuse; no other runtime code changes.
The same 132-token input, 20 workers, 64 generated tokens and five passes were used
at 32 and 90 GiB, with default chunks. Replay metrics exclude the first pass.
RSS below is Linux's process-lifetime high-water mark, including loading and the
first request, rather than the sampled peak in the preceding table:

| RAM budget | Profiling | Execution order | Replay prefill seconds, control → reuse | Replay decode tok/s, control → reuse | Lifetime peak RSS GiB, control → reuse |
| --- | --- | --- | ---: | ---: | ---: |
| 90 GiB | off | control, reuse | 2.790 → 2.834 | 6.690 → 6.706 | 38.193 → 38.191 |
| 90 GiB | on | reuse, control | 2.906 → 2.858 | 6.653 → 6.622 | 38.257 → 38.204 |
| 32 GiB | off | reuse, control | 10.404 → 10.131 | 5.764 → 5.778 | 40.219 → 31.545 |
| 32 GiB | on | control, reuse | 10.296 → 10.199 | 5.766 → 5.774 | 36.923 → 31.547 |

The memory reduction remains substantial without profiling. Decode differences
are within 0.5% in these pairs, so the preceding 1–2% full-cache slowdown is not
stable across these builds and runs. This does not establish that profiling caused
the earlier slowdown: changing the driver also changes executable placement, and
timing varies between processes. These measurements use the native runtime
benchmark, not the CLI, and retain the same warm-page-cache/128 GiB host limitation.

All 40 passes matched token IDs, complete-logit hashes and expert cache counters
across profiling modes and RAM budgets; enabled profiles also matched stage
call/work/byte counters. Every pass had zero measured storage reads and zero swap
at the available sampling points. Commands, source/binary hashes and validation
records are in `target/qwen36-profile-overhead-session-20261004/metadata.json`;
reports are `profiles/qwen36-profile-overhead-*.json`.

A subsequent two-row ordered-F64 kernel shared input conversion and interleaved
two independent scalar sums. Cache-sized microbenchmarks improved, but native
LM-sized microbenchmarks differed by less than 0.6% at 8 and 20 workers. Two
90 GiB native comparisons measured +2.60% and −0.52% decode changes when executable
order was reversed; unrelated prefill timing also varied. The minimum-memory
streamed plan changed decode by −0.08% with a 19-token prompt. All 26 native passes
preserved tokens, complete-logit hashes and paired cache counters. The candidate
is not retained because an end-to-end benefit was not established. Measurements
and rejected source are archived in
`target/qwen36-lm-paired-f64-session-20261004/` and
`profiles/qwen36-lm-paired-f64-*.json`.

Two bounded expert-read-ahead variants were also rejected. They computed the
current streamed expert while reading the next miss, deferred the next LRU access
until current execution succeeded, and recycled owned buffers. A second transient
expert, when needed by the zero-slot plan, fit the existing scratch reservation.
Scoped compute-pool and dedicated I/O-pool versions both passed focused ownership,
cache-order and error-draining checks, but their native performance was worse.

At 16 GiB with the 132-token input, 20 workers, 16 generated tokens and three
passes, the compute-pool variant changed replay prefill from 17.399 to 45.746 s and
decode from 4.357 to 1.928 tok/s. The I/O-pool variant measured 17.448 to 40.976 s
and 4.368 to 2.246 tok/s. A profiled I/O-pool repeat confirmed the regression:
summed expert-read time rose 9.904 to 22.241 s per replay prefill, and summed
expert-compute time rose 12.106 to 25.182 s. Concurrent stage totals are not
additive. All reads were served from page cache; these measurements do not
establish the specific scheduling or memory-contention cause, or predict cold
storage behavior.

All 18 paired passes across these two variants preserved tokens, complete-logit
hashes and cache counters; enabled profiles also preserved stage counters. Neither
variant progressed to balanced-RAM acceptance, and both were fully reverted.
Source, commands and measurements remain under
`target/qwen36-expert-prefetch-session-20261004/`,
`target/qwen36-expert-io-prefetch-session-20261004/`, and their matching
`profiles/qwen36-expert*-prefetch-*.json` reports.

### Reusing prepared routes on a streamed-prefill fallback

Batched MoE first projects and validates every token's routing decisions. When
its selected experts cannot all remain charged to the cache, it now reuses those
decisions in the existing tokenwise expert schedule. Previously that fallback
repeated each router projection, top-k selection and BF16 route-weight cast.
The shared ordered-execution helper preserves expert-ID accumulation order,
every output cast, the final probability-ordered routes and the same cache
accesses. Expert capacity, scratch reservations and prefill chunk sizes are unchanged.

For the 132-token input at 16 GiB, profiling confirms exactly 5,280 fewer router
projections per prefill: 2,768,240,640 fewer scalar multiply-accumulates. All other
stage call/work/byte counters match, including every decode counter. Complete
logit hashes, token IDs and cache hit/miss/read/residency counters also match.

The following comparisons use 20 workers, profiling disabled, the same input and
each RAM plan's default chunk. Sixteen-token cases have three passes and the
64-token case has five; replay metrics exclude the first pass and model loading.
RSS is the process-lifetime high-water mark, including loading and the first pass:

| RAM budget | Generated tokens | Execution order | Replay prefill seconds, control → reuse | Replay decode tok/s, control → reuse | Replay request seconds, control → reuse | Lifetime peak RSS GiB, control → reuse |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 4 GiB | 16 | reuse, control | 35.787 → 35.819 | 2.317 → 2.219 | 42.262 → 42.578 | 3.675 → 3.658 |
| 4 GiB, reversed | 16 | control, reuse | 36.076 → 35.661 | 2.290 → 2.333 | 42.627 → 42.091 | 3.698 → 3.571 |
| 16 GiB | 16 | control, reuse | 17.339 → 17.288 | 4.327 → 4.312 | 20.806 → 20.767 | 15.370 → 15.310 |
| 32 GiB | 16 | control, reuse | 8.972 → 9.028 | 6.253 → 6.286 | 11.370 → 11.414 | 31.475 → 31.414 |
| 90 GiB | 64 | reuse, control | 2.766 → 2.795 | 7.391 → 7.470 | 11.290 → 11.229 | 38.199 → 38.193 |

A profiled 16 GiB comparison in reversed executable order measured 17.445 →
17.280 s prefill and 4.345 → 4.357 tok/s decode. The deterministic work reduction
is established, but these timings do not establish a uniform end-to-end speedup.
At 4 GiB, the initial −4.2% decode difference reversed to +1.9%; the initial loss
did not persist in the reverse-order pair. Other timing changes were small, and
the fully retained batch path eliminates no router work. No cache capacity was
traded for these results.

All 40 paired passes preserved tokens, complete-logit hashes and cache counters,
including common continuation prefixes across the 16- and 64-token requests.
Enabled profiles differed only by the expected removed router work. Every pass
had zero measured storage reads and swap. These remain runtime-budget comparisons
on the 128 GiB host with warm OS page cache, not physical 4/16/32 GiB machine or
cold-storage measurements. Source/binary hashes, commands and reports are archived
in `target/qwen36-prepared-routes-session-20261004/` and
`profiles/qwen36-prepared-routes-*.json`.

Final validation passed 464 default-feature tests, 425 no-UI tests, all seven
native checkpoint gates, both Clippy configurations, formatting and diff checks,
and the release CLI build. The native three-token prefill gate now compares the
complete step, including final routes, against incremental decoding at batch
sizes 1/2/32/128. The rebuilt profiler is byte-identical to the measured candidate.

### Rejected expert gate/up projection scheduling

Two prototypes combined each BF16 expert's gate and up projections into one
Rayon phase. The first task evaluated four gate rows followed by four up rows
using the existing kernels. The second task evaluated two gate and two up rows
with four independent AVX2/FMA accumulators sharing each input load. Both kept
all per-row reduction trees, BF16 casts, projection-error priority, weight bytes,
cache capacity and scratch reservations unchanged. Focused bitwise/tail/error
checks and default-feature all-target Clippy passed for both prototypes.

Native comparisons used 20 workers, the 132-token input, profiling disabled and
the existing RAM-plan chunk sizes. Sixteen-token requests had three passes;
64-token requests had five. Replay metrics exclude the first pass and loading.
RSS is the process-lifetime high-water mark, including loading and the first pass.
The 16/90 GiB runs used control then candidate; the finer prototype's 4/32 GiB
runs used candidate then control.

| Task | RAM GiB | Generated tokens | Replay prefill seconds, control → candidate | Replay decode tok/s, control → candidate | Replay request seconds, control → candidate | Lifetime peak RSS GiB, control → candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 gate + 4 up rows | 16 | 16 | 17.377 → 17.150 | 4.113 → 4.115 | 21.024 → 20.795 | 15.371 → 15.368 |
| 4 gate + 4 up rows | 90 | 64 | 2.731 → 2.768 | 7.022 → 6.981 | 11.703 → 11.792 | 38.197 → 38.263 |
| 2 gate + 2 up rows | 4 | 16 | 36.019 → 35.775 | 2.209 → 2.038 | 42.811 → 43.136 | 3.572 → 3.665 |
| 2 gate + 2 up rows | 16 | 16 | 17.395 → 17.186 | 4.142 → 4.160 | 21.017 → 20.792 | 15.370 → 15.370 |
| 2 gate + 2 up rows | 32 | 16 | 8.978 → 8.905 | 5.886 → 5.929 | 11.527 → 11.435 | 31.415 → 31.476 |
| 2 gate + 2 up rows | 90 | 64 | 2.855 → 2.858 | 6.882 → 6.892 | 12.009 → 11.999 | 38.205 → 38.193 |

At 16 GiB both prototypes shortened replay prefill by about 0.2 seconds, while
90 GiB decode was essentially flat (−0.57% and +0.15%). The finer prototype's
4 GiB decode fell from 2.209 to 2.038 tok/s (−7.73%), outweighing its small prefill
saving in total request time. That loss was observed in one execution order;
it is not a proven diagnosis of the kernel or host behavior. The limited gains
did not justify retaining a candidate that failed this small-budget comparison.
Neither prototype was accepted, and both code and tests were fully reverted.
No cache or chunk-size tradeoff was introduced to compensate.

All 44 paired passes preserved generated tokens, complete-logit hashes and cache
counters, including shared continuation prefixes across both prototypes. Measured
storage reads and swap were zero. These remain warm-page-cache budget comparisons
on the 128 GiB host, not physical small-memory or cold-storage measurements.
Artifacts are under `target/qwen36-gate-up-session-20261004/`,
`target/qwen36-gate-up-pair2-session-20261004/`, and matching
`profiles/qwen36-gate-up*.json` reports. The restored source was checked against
the prior accepted snapshot; formatting, diff checks and both release executable
rebuilds passed, and their hashes match the prior accepted binaries. No full
regression acceptance is claimed for the discarded prototypes.

### Bounded grouped expert streaming with parallel initial cache fill

When a prefill chunk cannot retain all selected experts without eviction, the
runtime can now gather each expert's token inputs and compute them together while
streaming one expert at a time. It processes distinct experts in last-use order,
stores their outputs, then performs each token's original expert-ID-ordered
accumulation and BF16 casts, followed by that token's shared expert. The existing
batched BF16 kernels retain each output's GEMV reduction tree.

A read-only LRU simulation compares the original and grouped access sequences.
Grouping is rejected if it would add misses, including the warm-cache case where
an early hit protects an expert from an intervening miss. Successful grouped loads
restore the original logical clock and every surviving entry's last-use timestamp,
including behavior at timestamp saturation. Eliminated accesses are not reported
as synthetic cache hits: telemetry counts the actual grouped cache acquisitions.
The final cache contents and subsequent decode accesses remain unchanged.
The partial-cache policy in [Preserving cached experts during grouped prefill](#preserving-cached-experts-during-grouped-prefill)
below supersedes this original rejection guard.

An empty cache that can hold one complete token's route keeps the existing parallel
initial-fill path when multiple workers are available. This avoids serializing
initial allocation/loading of a partially retained cache; later chunks can group
their requests. Zero-capacity and smaller-than-route caches can group immediately.
This condition addresses a prototype's first-prefill regression on the second
input, despite its faster replay; it does not change cache capacity or load policy
after a failed read. Each expert is still validated before a usable cache entry
can be evicted.

Streaming keeps at most one 6 MiB transient expert payload beyond the cache. Saved routed
outputs are at most 128 × 8 × 2,048 FP32 values (8 MiB); gathered inputs and temporary
outputs remain inside the existing scratch bound. No weight format, resident-slot
budget, 512 MiB scratch reservation or default 32/128-token chunk policy changes.
Load errors are deferred to their first original request position. A failed expert
batch is retried tokenwise to resolve its error locations before ordered reduction;
tests cover earlier shared, load and computation failures in the streaming path.

The following native comparisons use 20 workers on the 14700K, the 132-token input,
profiling disabled and each RAM plan's default chunk. The 4/16/32 GiB requests
generate 16 tokens over three passes; 90 GiB generates 64 tokens over five passes.
Replay metrics exclude the first pass and loading. First-prefill measurements use
a fresh expert cache but warm OS page cache. RSS is the process-lifetime high-water
mark, including loading and the first pass. All arrows compare control → grouped:

| RAM budget | First prefill s | Replay prefill s | Replay decode tok/s | Replay request s | Lifetime peak RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 GiB | 36.268 → 13.098 | 36.062 → 12.899 | 2.226 → 2.246 | 42.800 → 19.577 | 3.677 → 3.673 |
| 16 GiB | 17.462 → 16.158 | 17.452 → 15.413 | 4.112 → 4.112 | 21.100 → 19.060 | 15.370 → 15.372 |
| 32 GiB | 8.331 → 7.996 | 8.957 → 7.748 | 5.856 → 5.925 | 11.519 → 10.280 | 31.414 → 31.477 |
| 90 GiB | 5.859 → 5.642 | 2.752 → 2.779 | 6.944 → 6.971 | 11.825 → 11.817 | 38.208 → 38.190 |

The 4/32 GiB runs used candidate then control; 16/90 GiB used control then
candidate. A reversed-order 16 GiB profiling run measured 17.458 → 15.335 s replay
prefill. Its replay prefill
expert-compute calls fell from 42,240 to 31,304, and actual expert
loads from 9,787 to 9,029. BF16 matrix work stayed at
322,154,332,160 multiply-accumulates. Every decode call/work/byte counter matched.
The reduction comes from grouped dispatch and weight reuse, not fewer mathematical
outputs or a precision change. All other prefill stage counters matched except
the expected MoE, expert-load/compute and BF16 projection call counts.

Per-request expert logical reads at 4 GiB were 275.625 → 100.705 GiB,
and at 16 GiB 64.975 → 60.533 GiB. These include decode. The
fully retained 90 GiB path does not use the new fallback; its small timing
differences do not establish a separate full-cache speedup.

A second, 71-token input about expert caching used two passes and 16 generated
tokens at each budget, with candidate then control. Each replay value below has
one sample, so small decode differences are not stable speedup/regression claims:

| RAM budget, second input | First prefill s | Replay prefill s | Replay decode tok/s | Replay request s | Lifetime peak RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 GiB | 9.719 → 9.220 | 9.772 → 8.563 | 4.181 → 4.222 | 13.359 → 12.116 | 15.367 → 15.370 |
| 32 GiB | 4.853 → 4.819 | 3.278 → 3.193 | 6.314 → 6.215 | 5.654 → 5.606 | 29.423 → 29.425 |

Two additional 4 GiB runs explicitly used 128-token chunks. Prefill took
7.913/7.847 s, lifetime peak RSS was 3.702 GiB, and complete-logit hashes
and generated tokens matched the default-chunk runs. This checks the larger
supported batch without changing the default policy. Decode measured
1.990/1.997 tok/s in those two runs, below the default-chunk pair.
These unpaired observations motivated the controlled chunk comparisons below;
they did not justify promoting the larger batch to a default.

All 42 paired passes and the two additional chunk checks preserved generated
tokens and complete-logit hashes. Expert residency matched the controls, actual
miss/read counts never increased, and enabled profiles preserved all decode
counters. Measured storage reads and swap were zero. These are runtime-budget
comparisons on a 128 GiB host with warm OS page cache, not measurements on physical
4/16/32 GiB machines or cold storage. Exact-prompt replay and a second short input
do not establish gains for every routing distribution or long-running workload.

Validation passed 467 default-feature tests, 428 no-UI tests, all seven native
checkpoint gates, both Clippy configurations, formatting and the release build.
The new cache test checks 81,920 sequence/capacity/seed combinations and exact
recency restoration; sparse native-shape expert fixtures cover tiny/warm caches,
future hits, error order and preservation of parallel initial filling. The final
profiler is byte-identical to the measured candidate. Source, executables, checks
and commands are archived in `target/qwen36-grouped-fill-session-20261004/`, with
reports in `profiles/qwen36-grouped-fill-*.json`. Earlier prototypes remain under
`target/qwen36-grouped-stream-session-20261004/` and
`target/qwen36-grouped-replay-session-20261004/`.

### Controlled chunk and allocator probes: defaults retained

The zero-slot expert plan still defaults to 32-token chunks. A proposed change to
128, together with glibc allocation settings for streamed decoder trunks, was
rejected after its actual runtime implementation reduced decode throughput by
9.0%. The preceding grouped-streaming optimization remains in place. Nonzero
partial caches retain 32 and fully retained expert caches retain 128.

The accepted executable first ran a same-process crossover at a 4 GiB runtime
budget: 20 workers, the same 132-token input, 64 generated tokens, profiling off,
and chunks `32,32,128,128,32,128,32,32,128`. Causal state was restored after each
request while backbone weights remained cached. Excluding the first pass leaves
four samples per size, in counterbalanced order:

| Chunk size | Replay prefill s | Replay decode tok/s | Replay inference s |
| --- | ---: | ---: | ---: |
| 32 | 13.032 | 2.162 | 42.167 |
| 128 | 7.763 | 2.034 | 38.736 |

Inference time here and below is prefill plus the timed decode calls; loading,
tokenizer setup and diagnostic work are excluded. The larger chunk reduced
prefill by 40.4%, but decode was 5.9% slower, including the later decode steps.
A five-pass, 32-generated-token crossover with profiling enabled did not reproduce
that gap: replay decode was 2.204/2.210 tok/s for chunks 32/128, with identical
decode call/work/byte counters. Profiled timing alone therefore did not support
changing the default.

A temporary diagnostic build recorded the root Rayon worker and call duration.
All 576 forward/prefill calls in its nine-pass crossover used worker 0, pinned to
CPU 0, a P core on this host. Replay decode still measured 2.163/1.992 tok/s for
32/128. This rules out root-worker migration within that diagnostic; it does not
rule out allocation effects or scheduling of the other workers. The diagnostic
source was restored immediately after compilation.

Separate processes then tested glibc 2.35 startup settings, using the unchanged
accepted executable. Each used `32,32,128,128,32`; replay values exclude the first
pass. `M64` means `MALLOC_MMAP_THRESHOLD_=67108864`; `T64` additionally sets
`MALLOC_TRIM_THRESHOLD_=67108864`. The minimum plan streams every decoder trunk
and the LM head with no expert slots (`RAM_GIB=0` in the benchmark, not zero RAM).
Peak RSS is the process lifetime high-water value across the entire crossover,
so it cannot be assigned independently to either chunk size:

| Plan / startup settings | Generated tokens | Decode tok/s, 32 / 128 | Lifetime peak RSS GiB |
| --- | ---: | ---: | ---: |
| Minimum / default | 32 | 1.027 / 1.039 | 0.413 |
| Minimum / M64 | 32 | 1.017 / 1.031 | 0.411 |
| Minimum / M64 + T64 | 32 | 1.054 / 1.053 | 0.412 |
| 4 GiB / M64 | 64 | 2.168 / 2.184 | 3.658 |
| 4 GiB / M64 + T64 | 64 | 2.210 / 2.221 | 3.582 |

Those favorable crossovers all started with a 32-token chunk. The actual candidate
instead selected 128 from the first prompt and set both thresholds during model
loading when decoder trunks streamed. It was compared with the accepted default,
candidate first, with five passes and 64 generated tokens per process at 4 GiB:

| Metric | Accepted default | Rejected candidate |
| --- | ---: | ---: |
| First prefill s | 13.097 | 7.800 |
| Replay prefill s | 12.939 | 7.744 |
| Replay decode tok/s | 2.198 | 2.000 |
| Replay inference s | 41.604 | 39.248 |
| Lifetime peak RSS GiB | 3.671 | 3.596 |

Prefill improved 40.2%, but the repeated 9.0% decode loss reduced the 64-token
request benefit to 5.7%; longer decoding can offset that saving. Expert logical
reads decreased from 190.705 to 152.309 GiB per request and expert residency
remained zero. Neither output correctness nor reduced logical reads alone makes
this an acceptable default performance tradeoff.

Two follow-ups used that same candidate binary. Setting both parameters at
process startup while retaining 128 from the first prompt still produced
1.993 tok/s replay decode over three passes. Starting with a 32-token warm-up
and using `32,32,128,128,32` instead produced 2.210/2.208 tok/s in its two
128-token passes. Thus the favorable warmed crossover was not representative of
the proposed first-request default. Allocation history is a plausible explanation;
the precise mechanism has not been established. No allocator setting or chunk
policy from this candidate is retained, and broader promotion tests at
16/32/90 GiB were not pursued after the 4 GiB regression.

All 66 passes preserved 3,584 generated-token/complete-logit-hash pairs against
the accepted continuation. Every zero-cache miss/read count matched the expected
chunk schedule, and enabled profiles preserved decode call/work/byte counters.
Measured storage reads and process swap were zero. These remain warm OS-page-cache
tests on the 128 GiB host, not physical small-RAM or cold-storage measurements.

Runtime and test sources were restored byte for byte to the previously validated
version (467 default-feature tests, 428 no-UI tests and seven native gates).
Those suites were not rerun for the documentation-only retained change; formatting,
diff checks and release rebuilding verify the restored state, with final executable
hashes checked against the accepted binaries. Reports and commands are archived in
`target/qwen36-chunk-crossover-session-20261004/`,
`target/qwen36-worker-diagnostic-session-20261004/`,
`target/qwen36-allocation-probe-session-20261004/` and
`target/qwen36-zero-cache-prefill-session-20261004/`. The last archive retains the
rejected candidate separately from the restored final source and executables.

### Preserving cached experts during grouped prefill

Partial expert caches now compute selected resident experts before any missing
expert can displace them. A read-only simulation determines the original ordered
sequence's final cache contents and timestamps. Initially absent experts are
loaded once each: non-final values use the transient buffer, while final values
are admitted by evicting only entries outside that final set. Once every load
succeeds, the runtime publishes the original timestamps and global logical clock.
This avoids extra intermediate reloads while preserving the exact final LRU state
and subsequent decode accesses.

For example, with resident experts A/B and requests A/C/A, ordinary last-use
grouping C/A would evict A and require two loads. The new schedule computes A
first, loads C once, and restores the original final C/A recency. It can therefore
group warm-cache sequences that the earlier no-extra-misses guard rejected.

The initial parallel cache-fill guard, zero-slot schedule, fully retained path,
default 32/128-token chunks and resident-slot budgets remain in place. The new
admission policy uses the existing one-expert transient reserve and recycles
exclusively owned buffers. Saved outputs still reduce in the original token and
expert order with the same BF16 casts; scratch remains within the existing
512 MiB reservation. A failed expert load does not evict a usable entry. Sparse
native-shape fixtures check outstanding readers, corrupt loads, ordered error
selection and subsequent cache behavior.

The following paired native runs use the 14700K's 20 workers, the same 132-token
input, profiling disabled and each budget's default chunk. The 4/16/32 GiB runs
generate 16 tokens over three passes; 90 GiB generates 64 tokens over five passes.
Replay metrics exclude the first pass. Inference is prefill plus timed decode,
excluding loading, tokenizer setup and diagnostic hashing. First-prefill values
start with empty expert caches and warm OS page cache. RSS includes the entire
process lifetime. Arrows compare the previously accepted runtime → new schedule:

| RAM budget | First prefill s | Replay prefill s | Replay decode tok/s | Replay inference s | Lifetime peak RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 GiB | 13.175 → 13.397 | 12.964 → 13.071 | 2.023 → 2.212 | 20.378 → 19.854 | 3.677 → 3.571 |
| 16 GiB | 16.079 → 12.325 | 15.417 → 10.006 | 4.108 → 4.106 | 19.068 → 13.659 | 15.368 → 15.372 |
| 32 GiB | 8.133 → 7.693 | 7.740 → 6.369 | 5.882 → 5.920 | 10.291 → 8.903 | 31.476 → 31.476 |
| 90 GiB | 5.660 → 5.660 | 2.796 → 2.706 | 6.993 → 6.964 | 11.805 → 11.752 | 38.243 → 38.263 |

The 4/16/32 GiB comparisons ran candidate then control; 90 GiB ran control then
candidate. The substantial prefill gains are at the partial-cache budgets. Small
full-cache timing differences do not establish a separate 90 GiB speedup.

At 16 GiB, replay prefill decreased 35.1%, and timed
inference decreased 28.4%. Expert logical reads, including decode,
fell from 60.533 to 44.396 GiB per request; expert residency matched.

At 32 GiB, replay prefill decreased 17.7%, and timed
inference decreased 13.5%. Expert logical reads, including decode,
fell from 17.068 to 13.207 GiB per request; expert residency matched.

A paired 16 GiB run with profiling enabled measured replay prefill
15.465 → 10.086 s. Prefill expert loads decreased
9,029 → 6,275, expert-compute calls 31,304 → 12,387, and BF16
projection calls 116,183 → 59,432. Matrix work remained
322,154,332,160 multiply-accumulates. Every decode call/work/byte counter
matched, as did other prefill counters outside the expected MoE/expert/projection
call changes. Telemetry counts actual acquisitions; collapsed requests do not
create synthetic cache hits.

A second, 71-token input about expert caching used two passes and 16 generated
tokens at each partial-cache budget, candidate then control. Each replay value
has one sample, so small decode differences are not stable speedup claims:

| RAM budget, second input | First prefill s | Replay prefill s | Replay decode tok/s | Replay inference s | Lifetime peak RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 GiB | 9.254 → 7.939 | 8.636 → 5.660 | 4.153 → 4.191 | 12.248 → 9.240 | 15.367 → 15.369 |
| 32 GiB | 4.923 → 4.861 | 3.276 → 2.663 | 6.285 → 6.321 | 5.663 → 5.035 | 29.426 → 29.486 |

The short 4 GiB control's decode samples were 2.211, 2.051 and 1.996 tok/s,
while the candidate stayed near 2.21. That apparent 9.3% advantage is not treated
as an established kernel speedup: the zero-slot scheduling policy is unchanged,
and earlier chunk experiments also showed sensitivity to run history.
A follow-up reversed the process order, control first, and generated 64 tokens
over five passes per executable:

| RAM budget, extended check | First prefill s | Replay prefill s | Replay decode tok/s | Replay inference s | Lifetime peak RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 GiB | 13.120 → 13.149 | 12.923 → 13.004 | 2.222 → 2.216 | 41.279 → 41.428 | 3.689 → 3.576 |

Extended replay decode changed -0.24%. Both executables made
the same expert accesses and logical reads at 4 GiB. These measurements do not
identify the cause of the short-run timing variation.

All 52 native passes preserved 1,792 generated-token/complete-logit-hash pairs
within each input across RAM budgets and profiling modes. Expert residency matched,
actual miss/read counts never increased, and all profiled decode counters matched.
Measured storage reads and process swap were zero. These are runtime-budget tests
on a 128 GiB host with warm OS page cache, not physical small-RAM or cold-storage
measurements. Short prompt replays do not establish performance for every routing
distribution or long-running session.

Validation passed 469 default-feature tests, 430 no-UI tests, all seven native
checkpoint gates, both Clippy configurations, formatting and release rebuilding.
The new cache tests cover 65,536 sequence/capacity/seed combinations, the lower
bound on loads, exact final recency and future accesses, clock saturation,
transient ownership and load failure. Native-shape MoE tests cover tiny and warm
caches, exact output bits and ordered errors. The final profiler is byte-identical
to the measured candidate. Source snapshots, both executables, commands and checks
are archived in `target/qwen36-protected-groups-session-20261004/`; native reports
are under `profiles/qwen36-protected-groups-*.json`.
