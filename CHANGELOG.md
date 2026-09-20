# Changelog

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

[0.2.0]: https://github.com/inspirewind/Urbilateria/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/inspirewind/Urbilateria/releases/tag/v0.1.0
