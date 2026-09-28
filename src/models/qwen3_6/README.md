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
```

Arguments are `MODEL_DIR RAM_GIB THREADS TOKENS [PASSES] [BATCH_SIZES] [PROMPT_FILE]`.
Batch sizes cycle through comma-separated values in 1..=128; defaults are two
passes using the RAM plan: 128 when all 256 experts per layer fit, otherwise 32.
Size 1 retains tokenwise projections and expert execution
while sharing the same layer-major schedule, resident weights, and final LM head.
The optional file replaces the built-in user prompt before applying the same ChatML
template. Its path and the rendered token IDs are included in the JSON report.
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
State validation uses AVX2 exponent-bit scans on supported CPUs, retaining rejection
of every NaN/infinity encoding and a scalar fallback. During resident decode, shared
and routed expert projections use the worker pool concurrently; the shared contribution
is still accumulated after routed experts. Cache budgets and streaming behavior stay
the same.
GQA value aggregation reads each cached value row contiguously and uses AVX2 F64
lanes for independent output dimensions. Each dimension retains the original
token-order sum and positive-zero identity, with separate multiply/add operations;
the native 256-wide heads use stack scratch and other widths retain a bounded fallback.
When query heads × head width × attended tokens reaches 256K, independent
heads share the CPU pool for score computation, softmax, and value aggregation.
Each head retains its token/reduction order and owns a separate output slice;
errors are checked in head order before committing KV state. Native contexts below
64 attended tokens and single-worker execution retain the serial path, avoiding
head-scheduling overhead on short inputs. At the native maximum context,
the 16 concurrent score buffers total at most 16 MiB, within the existing 512 MiB
fixed scratch reservation.

Prefill uses 128-token chunks when the planned cache can hold all 256 experts per
layer, and retains 32-token chunks for partial caches. Explicit library/profiler batch
sizes range from 1 to 128. Larger chunks group more token inputs by expert; attention
input and output projections still share BF16 weight loads across up to eight inputs. DeltaNet
recurrence and GQA cache updates still follow token order. MoE projections group tokens
by routed expert when the complete selected set fits the cache without eviction.
Cache accesses replay the original token/expert sequence, including repeated hits;
each token retains sorted expert accumulation and the same BF16 rounding boundaries.
Smaller caches fall back to tokenwise expert execution. The bounded projection and
grouping buffers fit the fixed scratch reserve; prompt hidden snapshots remain
accounted for separately. Failed attention batches restore their prior state,
including failures in the final output projection.

For the fixed native geometry, a conservative simultaneous activation estimate at
128 tokens is 54 MiB: layer/attention/shared buffers use `8H + 4P` F32 values per
token, and routed gathers, projection layouts, and intermediates use `K(3H + 3I)`.
Here `H=2048`, `I=512`, `K=8`, and the largest projection width `P=8192`.
Doubling that estimate for vector capacities, adding 16 MiB of parallel attention
scores at maximum context, the 2.125 MiB DeltaNet transaction, and 16 MiB for metadata
and small scratch gives 142.125 MiB, within the unchanged 512 MiB fixed reserve.
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
