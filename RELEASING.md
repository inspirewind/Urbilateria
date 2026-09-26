# Releasing Urbilateria

The release workflow runs when a stable `vMAJOR.MINOR.PATCH` tag is pushed. It validates the
tag against Cargo and the changelog, reuses the full CI workflow on that commit, builds native
archives, tests each packaged TUI in a pseudo-terminal, and creates a **draft** GitHub Release.
It does not publish the draft automatically. No crates.io publication is configured.

## Prepare a version

1. Update the root `Cargo.toml` version and run `cargo check --all-targets` to synchronize
   `Cargo.lock`. Keep `rust-toolchain.toml`, `rust-version`, and CI toolchains consistent.
2. Add a nonempty `## [VERSION]` section to `CHANGELOG.md`, including all changes since the
   preceding tag. A ` - YYYY-MM-DD` suffix may be added when the publication date is known.
   Update both READMEs if commands or compatibility change.
3. Run the checks and validate the proposed tag locally (this does not create a tag):

   ```bash
   cargo fmt --all -- --check
   cargo test --locked --all-targets
   cargo test --locked --no-default-features --all-targets
   cargo clippy --locked --all-targets -- -D warnings
   cargo clippy --locked --no-default-features --all-targets -- -D warnings
   python3 -B -m unittest discover -s .github -p 'test_release.py' -v
   python3 .github/release.py validate --tag v0.2.2
   cargo build --release --locked --features ui --bin urb
   python3 tests/tui_smoke.py target/release/urb
   ```

4. Review and commit the prepared changes, then push the commit and wait for CI. Do this only
   when ready to release; the preparation scripts do not commit or push anything.
5. Tag the reviewed commit and push that tag:

   ```bash
   git tag -a v0.2.2 -m "Urbilateria 0.2.2"
   git push origin v0.2.2
   ```

6. Wait for the **Release** workflow and inspect its draft in GitHub Releases. Check the
   changelog notes, both archives, `BUILD_INFO.json`, and `SHA256SUMS`. Test downloaded
   binaries on Linux x86_64 and an Apple Silicon Mac before publishing the draft manually.

The workflow requires GitHub Actions to be enabled and permits `contents: write` only in
the draft job. Re-running a failed workflow can update an existing draft. It refuses to
replace assets on an already published release. Correct published releases with a new version.

## Artifacts and compatibility

| Archive | Native builder | Runtime baseline |
| --- | --- | --- |
| `urb-v0.2.2-x86_64-unknown-linux-gnu.tar.gz` | Ubuntu 22.04 | Linux x86_64 with glibc 2.35+ |
| `urb-v0.2.2-aarch64-apple-darwin.tar.gz` | macOS 15, Apple Silicon | Apple Silicon macOS 13+ |

Archives include `urb` with the default `ui` feature, the MIT license, English/Chinese READMEs,
the changelog, this release guide, installation instructions, and build information. The
binary needs no Rust, Python, or Node.js at runtime. Python is used only for release tooling
and tests. Model weights are never bundled. Intel Mac, Linux ARM64, Windows, and musl/Alpine
binaries are not built by this workflow.

Linux builds reject GLIBC symbol requirements above 2.35. macOS builds set
`MACOSX_DEPLOYMENT_TARGET=13.0` and check the Mach-O architecture and recorded minimum OS;
runtime tests use macOS 15. macOS 13 is a deployment target, not a separately tested CI runner.
Mac archives are not Developer ID signed or notarized. On macOS, memory planning needs an
explicit `--ram-gib` value.

The published release has three attachments: the two archives and `SHA256SUMS`. Download the
checksum file beside your archive and verify before extraction:

```bash
# Linux: ignore the missing Mac archive, but require the Linux archive to report OK.
sha256sum --ignore-missing -c SHA256SUMS

# macOS: the missing Linux archive may be reported; require the Mac archive to report OK.
shasum -a 256 -c SHA256SUMS
```

For a local packaging rehearsal on a supported native host:

```bash
cargo build --release --locked --features ui --bin urb
python3 .github/release.py package --tag v0.2.2 \
  --target x86_64-unknown-linux-gnu --binary target/release/urb
```

On Apple Silicon, use `--target aarch64-apple-darwin` and set
`MACOSX_DEPLOYMENT_TARGET=13.0` before building. A Linux build on a newer distribution can
require newer glibc and fail the baseline check; use Ubuntu 22.04 for official archives.
Packages go to `target/release-assets/`, which is ignored by Git. A local rehearsal with
uncommitted changes records `source_dirty: true` in `BUILD_INFO.json` and is not an official
release artifact. The `collect` subcommand requires both native archives and their generated
`.sha256` sidecars; it verifies them before generating combined checksums and release notes.

Workflow references: [reusable workflows](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows),
[draft release creation](https://cli.github.com/manual/gh_release_create), and
[Rust Apple deployment targets](https://doc.rust-lang.org/rustc/platform-support/apple-darwin.html).
