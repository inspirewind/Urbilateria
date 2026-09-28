# Changelog

## Unreleased

- Use 128-token Qwen3.6 prefill chunks when all routed experts fit in the planned
  cache, retaining 32-token defaults for partial caches and the same scratch budget.
- Extend prefill parity checks through 128-token boundaries, singleton tails,
  checkpoint replay, and continued decoding; make profiler defaults follow the RAM plan.
- Vectorize Qwen BF16 activation rounding with AVX2 while preserving ties-to-even,
  finite-overflow rejection, the first failing element, and partial-write behavior.
- Enter the CPU pool once per Qwen3.6 forward or prefill, including the LM head,
  reducing repeated caller-to-pool dispatch while preserving profiling context,
  single-worker execution, and deterministic results.
- Overlap Qwen3.6 current-layer computation with one retained next-layer load on
  the existing CPU pool; preserve streaming budgets and wait for both jobs on errors.
- Reuse one DeltaNet transaction workspace across Qwen3.6 prefill layers and copy
  recurrent state once per chunk, preserving numerical order and output-failure rollback.
- Use Qwen3.6 RAM budgets for persistent decoder layers, LM head, and lazy routed-expert
  caches, preserving the minimum-memory streaming fallback and session rewind behavior.
- Vectorize BF16 payload validation, interleave four BF16 matrix rows without
  reassociation, and traverse DeltaNet state contiguously with ordered reductions.
- Reuse normalized DeltaNet keys/queries across value heads, remove per-head heap
  scratch for native shapes, and parallelize independent head updates with ordered
  AVX2 F64 projections while preserving state on errors.
- Load and compute selected Qwen3.6 experts concurrently when the cache can retain
  the complete set within budget; preserve deterministic BF16 accumulation.
- Add numerical parity gates, cache-rewind checks, and a real-model CPU profiler.
- Batch Qwen3.6 attention projections in bounded 32-token prefill chunks with
  shared BF16 weight loads and unchanged per-token accumulation and causal order.
- Cover batch tails, continuation across chunk boundaries, and rollback after
  attention output-projection failures.
- Group prefill tokens by routed expert when the full selection fits the RAM budget;
  batch shared-expert projections while preserving BF16 rounding, cache counters,
  and subsequent LRU eviction order. Keep the tokenwise fallback for smaller caches.
- Add configurable same-process prefill crossover runs and separate prefill/decode
  profiling totals without restarting resource sampling.
- Vectorize DeltaNet finite-value validation without changing NaN/infinity rejection.
- Compute shared and retained routed experts concurrently during Qwen3.6 decode,
  keeping the shared contribution last and preserving the streaming fallback.
- Traverse GQA values contiguously with ordered AVX2 F64 reductions and scalar tails;
  preserve token-order sums and use stack scratch for native head widths.
- Allow a UTF-8 prompt file in the Qwen3.6 profiler for longer-context comparisons.
- Parallelize independent GQA heads above a 256K work threshold, preserving
  per-head reductions, error order, and KV rollback while retaining the serial fallback.

- Add experimental Qwen3.6-35B-A3B native BF16 text inference, with strict nested
  configuration and complete 26-shard/1,045-tensor validation.
- Stream selected experts from packed gate/up/down tensors; reuse hybrid Qwen
  attention kernels and support bounded recurrent/GQA state and session reuse.
- Wire CLI/TUI inspect, preflight, tokenize, chat, and generate; support the shipped
  thinking/no-thinking template. Vision and MTP remain schema-only.
- Add packed-expert offset/budget tests, official prompt fixtures, two-step real
  Transformers numerical gates, and real prefill/decode consistency checks.

## [0.2.2]

### Persistent chat sessions

- Add interactive `urb chat` with bounded conversation history, `/clear`, and `/quit`.
- Keep the TUI inference process, model, and runtime state resident between completed turns.
- Automatically reuse matching token-prefix KV caches across GLM, Hy4, Qwen3.8, Kimi-K3,
  DeepSeek-V4, and DeepSeek-V4.1. Report reused tokens, remaining prefill, and context capacity.
- Restore model-specific checkpoints when chat templates rewrite earlier reasoning or turn
  boundaries; rebuild sequence state when no saved prefix matches the next prompt.
- Budget context capacity and checkpoint memory alongside model weights and expert caches.
  Replan when context grows beyond capacity, and retain the initial automatic RAM ceiling
  so resident model memory does not shrink the next turn's budget.
- Release resident sessions on conversation clearing, model or RAM/thread/thinking changes,
  cancellation, and exit. Keep streamed text and diagnostics synchronized across turns.

### Terminal input

- Fix intermittent TUI input stalls when window resize and keyboard events arrive together.
  Use Crossterm's file-descriptor polling backend and cover both event arrival orders in PTY tests.

### Validation and scope

- Add cached-versus-fresh generation and checkpoint replay checks, plus CLI/TUI terminal tests
  for multi-turn process reuse and cache cleanup.
- Caches remain in memory for the current session; they are not persisted across restarts.
  `urb generate` remains a single-request command.
- This release does not establish new real-model quality, memory-use, or throughput guarantees.

## [0.2.1]

### Interactive generation

- Move TUI runtime diagnostics beneath the model card; add an expandable F2 log view.
- Add CLI generation metrics for prompt/output/total token counts, request TTFT and decode
  throughput, with optional live text or JSON snapshots. Reuse them in the TUI footer.
- Pin inspected model metadata at the upper right; retain model identity in narrow terminals.
- Generate from plain text with model-native multi-turn context. Add `/settings` for session RAM,
  token, thread and thinking defaults; preserve completed turns independently of display clipping.
- Bound conversation memory and drop oldest complete turns to fit the runtime token limit.
  Clear context with `/clear` or a successful model change; exclude cancelled/failed and raw replies.
- Pass structured conversations over stdin to the existing generation process, avoiding argv limits.
- Add `/generate` to the TUI with streamed Unicode text, separate runtime diagnostics,
  explicit RAM/weight-loading authorization, and the existing CLI's generation/profile options.
- Stop generation with Esc; preserve partial output and reap the model process on cancellation,
  normal UI exit, handled termination signals, or UI errors. Each request uses an independent process.
- Bound streaming queues and displayed text/logs; keep the terminal responsive during loading
  and generation. Add tiny-model PTY tests for generation, errors, cancellation and process cleanup.
- Parse CLI generation options without treating literal prompt values as flags.
- Detect terminal size changes while idle even when a resize notification is lost.

### DeepSeek-V4.1 runtime

- Batch prompt prefill by layer and cache decoder layers and the LM head within the RAM budget.
- Batch attention, router and grouped expert projections; overlap adjacent layer reads with
  computation and add bit-exact F32, MXFP8 and MXFP4 batch kernels.

## [0.2.0]

### Interactive terminal

- Add `urb ui [MODEL_DIR]`, built with Ratatui and Crossterm for Linux and Apple Silicon macOS.
- Expose `/inspect`, `/plan`, `/preflight`, `/list`, `/explain`, `/probe`, `/tokenize`,
  `/decode`, `/help`, `/version`, `/clear`, and `/quit` inside the TUI.
- Add multiline Unicode editing, bracketed paste, command completion, session history,
  transcript scrolling, elapsed-time indicators, and background model operations.
- Reuse the selected model between commands; support quoted paths and command-specific options.
- Restore terminal state after normal exit, signals, and panic, including when a filesystem
  operation is blocked.
- Share checkpoint analysis and tokenizer services with the existing CLI while preserving
  its text and JSON output formats.

### Runtime improvements since v0.1.0

- Pipeline DeepSeek-V4 decoder-layer, expert, and BF16 LM-head reads alongside CPU computation.
- Adapt decoder and LM-head residency to the RAM budget and reuse streamed weight,
  activation, and small-vector buffers.
- Add bit-exact batched MXFP4/MXFP8 kernels and avoid redundant quantization work.
- Share the bounded LM-head pipeline with DeepSeek-V4.1, Hy4, Kimi-K3, and Qwen3.8.
- Improve CPU worker placement, glibc allocation reuse, profiling examples, and resource accounting.

### Toolchain, compatibility, and distribution

- Raise the minimum Rust version from **1.83 to 1.88**, pin development to **1.88.0**,
  and retain compatible dependency versions in `Cargo.lock`.
- Enable the `ui` Cargo feature by default; retain `--no-default-features` for CLI-only builds.
- Fix platform-specific compilation and tests for Apple Silicon macOS.
- Test Rust 1.88.0 and stable on Linux and Apple Silicon macOS, with and without `ui`;
  add terminal interaction, signal, and panic-restoration checks.
- Add tag-triggered release preparation: validate the version, run CI, build and test native
  binaries, produce `.tar.gz` archives and `SHA256SUMS`, then create a GitHub Release draft.
- Provide Linux x86_64 GNU binaries built on Ubuntu 22.04 (glibc 2.35 or newer) and Apple
  Silicon binaries with a macOS 13.0 deployment target, tested on macOS 15.
- Include English/Chinese documentation, installation instructions, build information,
  and the MIT license in each archive. Running a binary needs no Rust, Python, or Node.js.

### Scope

- TUI model operations cover analysis and tokenization. Inference remains available through
  the existing `urb generate` command in the shell.
- `/probe` reads bounded samples of tensor payloads; other TUI model operations read
  metadata or tokenizer files. No checkpoint weights are distributed.
- On macOS, `/plan` requires an explicit `--ram-gib` value. Model-specific limitations
  documented in the README still apply.
- Intel Mac, Linux ARM64, and Windows release binaries are outside this release's scope.
- Generation remains experimental; this release does not establish new real-model quality
  or throughput guarantees.

## [0.1.0] - 2026-09-19

- Initial public release of the pure-Rust CPU inference and checkpoint analysis framework.
- Include model adapters, bounded-memory execution, tokenizer and chat-template support,
  runtime validation, profiling, and an interactive profile-trace viewer.

[0.2.2]: https://github.com/inspirewind/Urbilateria/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/inspirewind/Urbilateria/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/inspirewind/Urbilateria/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/inspirewind/Urbilateria/releases/tag/v0.1.0
