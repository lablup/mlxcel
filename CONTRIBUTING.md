# Contributing to mlxcel

Thank you for your interest in contributing to mlxcel! This document covers the basics for getting started. The deeper working contract lives in the `docs/` directory: [`docs/architecture.md`](docs/architecture.md) for the runtime and module map, [`docs/code-guidelines.md`](docs/code-guidelines.md) for the shared-function conventions and the file-size and module-split thresholds, and [`docs/adding-models.md`](docs/adding-models.md) for the model-porting checklist.

## Quick links

| You want to... | Read |
|----------------|------|
| Report a security vulnerability | [`SECURITY.md`](SECURITY.md) — **do not** open a public issue |
| File a bug or feature request | [GitHub Issues](https://github.com/lablup/mlxcel/issues) (use the templates) |
| Build and test locally | [`docs/installation.md`](docs/installation.md) |
| Understand the architecture | [`docs/architecture.md`](docs/architecture.md) |
| Add a new model family | [`docs/adding-models.md`](docs/adding-models.md) |
| Add or validate an embedding or reranker family (`/v1/embeddings`, `/v1/rerank`, `mlxcel embed`, `mlxcel rerank`) | [`docs/embeddings.md`](docs/embeddings.md) |
| Understand the code guidelines (shared functions, file size and module splits) | [`docs/code-guidelines.md`](docs/code-guidelines.md) |

## How to contribute

### Reporting issues

- Search existing issues first.
- Use the bug-report or feature-request template — they prompt for the information we need to act on the issue.
- Include the mlxcel version (`mlxcel --version`), platform (macOS Apple Silicon / Linux CUDA + GPU model), and the checkpoint you were running.
- For inference correctness or performance reports, also include the prompt and seed so the run is reproducible.

### Submitting pull requests

1. Fork the repository and create a feature branch off `main`:
   ```bash
   git checkout -b feat/short-description
   ```
2. Make your changes. Keep one PR scoped to **one logical change** — a model port, an MLX bump, and a CLI rename are three PRs.
3. Build and test for your target:
   ```bash
   # macOS (Apple Silicon)
   cargo build --release --features metal,accelerate
   cargo test --workspace --profile test-fast --features metal,accelerate

   # Linux / CUDA
   cargo build --release --features cuda
   cargo test --workspace --profile test-fast --features cuda -- --test-threads=1
   ```
4. Run the local quality gates:
   ```bash
   cargo fmt --all -- --check   # gated at PR time; fmt violations block merge
   cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings   # NOT gated at PR time; yours to run
   cargo test --workspace --profile test-fast --features metal,accelerate --no-fail-fast -- --test-threads=1   # NOT gated at PR time; yours to run
   cargo deny check             # gated at PR time (advisories + licenses + sources)
   ```
   PR-time CI runs only the cheap gates: `cargo fmt`, `cargo deny` and a workspace crate-version consistency check in [`ci.yml`](.github/workflows/ci.yml), plus a path-filtered clippy and a `distributed::`-scoped `cargo test` in [`pipeline-parallel-ci.yml`](.github/workflows/pipeline-parallel-ci.yml) when you touch pipeline-parallel code. Clippy and the general unit suite are **not** enforced on your PR. They were moved in #21 and removed in #23 because ~30 min per run on the shared self-hosted Apple Silicon runner blocked PRs and releases for failures that `make verify` catches locally in a fraction of the time. [`nightly-verify.yml`](.github/workflows/nightly-verify.yml) runs the full `make verify` once a day on `self-hosted-macos-26-arm64` and files an issue when `main` goes red or when the run does not finish, so a broken suite surfaces within a day rather than on the next contributor's `make verify`. Treat that as a backstop, not a substitute: run the two commands above yourself before you push. CUDA verification is not gated at PR time either; that stays exclusive to `release.yml`.

   `make verify` runs six prerequisite targets: `verify-versions`, `verify-kernel-dtype-keys`, `verify-llama-compat`, `verify-fmt`, `verify-clippy`, and `verify-test`. The last three are the fmt, clippy, and test commands above; `cargo deny check`, the fourth command in that block, is not one of them, and PR CI is what gates it. The first three are cheap consistency gates: `verify-versions` asserts every version-tracking workspace crate carries the root `mlxcel` version, `verify-kernel-dtype-keys` asserts every CUDA JIT kernel launch keys its cache on the input dtypes (#1053, #1054), and `verify-llama-compat` validates the checked-in llama-server b10621 compatibility manifest under `compat/llama-server/b10621/` structurally, with no network access required (#1443). The nightly invokes those same Makefile targets, so the local gate and CI cannot drift apart. Tests build under `[profile.test-fast]` rather than `--release`: `opt-level = 3` is kept so MLX numerics stay representative, while cross-crate LTO and the single codegen unit that make full `--release` test links expensive are dropped. `make test-fast` / `make test-fast-cuda` are the same profile with `--test-threads=1` and a `FILTER` hook for narrowing the run while you iterate. Reach for `cargo test --release --features metal,accelerate` by hand only when you suspect a defect specific to release codegen. See [`docs/installation.md`](docs/installation.md#fast-iteration-builds) for the measured comparison.

   The plain `make test`, `make test-verbose`, and `make test-lib` targets stay on Cargo's default `dev` profile and now export `RUST_MIN_STACK=16777216` for you. That stack bump is there for `server::reasoning_effort_tests`: its repeated renders of the pinned 8,952-byte Qwen3.8 chat template can overflow libtest's default 2 MiB per-test thread stack under unoptimized `dev`, even though the same suite passes under `test-fast` and `release`. If you bypass the Makefile and run the `dev`-profile server tests directly, keep the same prefix yourself: `RUST_MIN_STACK=16777216 cargo test --lib --features metal,accelerate server::`. `make verify-test` is intentionally unaffected because it already builds under `[profile.test-fast]`.

   `--workspace` is not optional, and leaving it off is how the gate went blind before (#1007). This repository's workspace root is itself the `mlxcel` package, so a bare `cargo test` or `cargo clippy` resolves to `-p mlxcel` and never builds `mlxcel-core`, `mlxcel-mlx-pin`, `mlxcel-surgery` or `mlxcel-xla` at all, let alone their test targets. That hid 1754 tests, 1354 of them in `mlxcel-core`, which is the crate holding the MLX `cxx` bridge, `layers.rs`, the KV cache and the quantization loaders. It also hid test-only lint debt, since `--all-targets` without `--workspace` does not compile a member's test target either. `cargo fmt --all` was already workspace-wide, which is why the fmt gate never had the hole. Each member builds at the feature set the root selects: `mlxcel-core` gets `metal` and `accelerate` through the root's forwarding, `mlxcel-mlx-pin`, `mlxcel-surgery` and `mlxcel-xla` get their empty defaults. `mlxcel-xla`'s `iree` feature stays off, so the gate needs no IREE distribution and the code behind `iree`, `diagnostics` and `micro-oracle` remains ungated.

   `--no-fail-fast` goes with it. Once the run covers five members, the first failing test binary would otherwise end it and hide the other four; cargo still exits non-zero, so the gate is no weaker. Cargo runs the test binaries one at a time, so the `mlxcel-core` suite never shares the Metal device with the root suite, which is the condition that corrupts results in #1008.

   **`--test-threads=1` goes with it too, on macOS as well as on CUDA (#1092).** Cargo's one-at-a-time sequencing bounds concurrency between binaries and says nothing about concurrency inside one, and libtest defaults to one test thread per logical CPU. That is what took `main` red on 2026-08-16: the `mlxcel-core` binary died with `signal: 11, SIGSEGV` and published no panic and no `test result` line, so cargo reported a failed target with nothing to read. The crash report from the local repro on an 18-core M5 Max has 18 libtest workers live at the fault, all in MLX-backed cache tests, two inside `iokit_user_client_trap`, faulting on an unmapped address. Do not reach for `--jobs 1`: `--jobs` bounds the build, which has finished before any test runs. Serializing costs +7.2s on the whole workspace (69.17s to 76.39s, 101 binaries, 8128 tests, warm), because the work already serializes on the one Metal device; `mlxcel-core` costs +23s while the root suite gains 12s. `make test-fast` has serialized on macOS since #809, so the gate now agrees with the edit-test loop. Unlike CUDA there is no guard test, because a parallel `cargo test -p mlxcel-core --lib` is three times faster and nearly always succeeds; narrowed hand-runs are meant to stay parallel.

   **On Linux/NVIDIA the gate is `make verify-test-cuda`, and it must run single threaded.** That target is `cargo test --workspace --profile test-fast --features cuda --no-fail-fast -- --test-threads=1`. CUDA tests are gated nowhere in CI, not even nightly (`release.yml` builds under `--features cuda` but runs no tests), so this local run is the entire gate for the backend. `--test-threads=1` is not a formality: driving MLX from the many host threads libtest spawns by default takes the process down with SIGABRT partway through the suite, at a different test and with a different CUDA error each run, so the abort reads as if whichever test was running is broken. Measured on GB10 at MLX pin `2c46b953`, the default 20-thread run dies at `cudaStreamEndCapture ... previous error during capture` while the same binary serialized reports a verdict on 1410 tests in 88 seconds (#1048). Do not reach for `MLX_USE_CUDA_GRAPHS=0`: with graph capture disabled the 20-thread run still aborts, as `cuLaunchKernelEx ... invalid argument`, so capture only selects the symptom and concurrency is the cause. Capture stays fully on under the gate. `mlxcel-core` carries a `the_cuda_test_suite_must_run_single_threaded` guard, so a hand-run `cargo test --workspace --features cuda` that forgets the flag fails by name with the right command rather than aborting anonymously; narrowed runs whose filter does not match the guard's own name filter it out and stay parallel, which is fine, because it is whole-suite runs that abort. A filter that does match it, `--lib cuda` for one, trips the guard on a run that would have been safe; set `MLXCEL_ALLOW_PARALLEL_CUDA_TESTS=1` for that case.

   **If you touch the video path, also run `make verify-test-video`.** That target is `MLXCEL_TEST_VIDEO=1 cargo test --profile test-fast --features metal,accelerate -p mlxcel --lib -- --include-ignored --skip bench_single_pass_768_frames multimodal::video vision::processors::gemma4::tests::process_videos_pixel_values_match_input_color`, and it needs `ffmpeg` 5.0 or newer on `PATH`. The two positional filters are ORed by libtest, and they sit after the `--` because `cargo test` itself takes only one TESTNAME. The second one is the Gemma 4 processor's pixel-content test: it decodes a synthetic clip through the same `load_video` entry point before running it through `process_videos`, so it is ffmpeg-backed too even though it lives outside `multimodal::video`. It is deliberately not part of `make verify`, so a contributor without ffmpeg still gets a green gate. The ffmpeg-backed tests carry `#[ignore]`, which is why `--include-ignored` is required to select them at all: stable libtest counts a test that inspects `PATH` and returns early as a **pass**, so the older runtime skip made a host with no ffmpeg report `34 passed; 0 failed` while a host with ffmpeg reported `31 passed; 3 failed`. The broken host was the one that looked healthy, and every video path in the runtime stayed broken for as long as it took ffmpeg 8 to remove `-vsync` (#1172). `MLXCEL_TEST_VIDEO=1` closes the other half: on a run that is supposed to exercise video, a missing ffmpeg is a hard failure rather than a skip. `nightly-verify.yml` installs ffmpeg and runs this target as its own reported step.

   `make test-fast` and `make test-fast-cuda` are edit-test-loop targets, not gates. They stay on the root package, so `test-fast-cuda` runs none of `mlxcel-core`'s 1410 tests.
5. **If the change moves the numbers, measure what it moved.** Quantization, kernel selection, fused ops, block widths and anything else that changes arithmetic needs a measurement rather than an assertion, and the two obvious measurements are both traps on their own. Byte-identity is a yes/no that says nothing once the answer is no, and on Apple GPU generation 15 and newer it is already no for reasons you did not choose. Perplexity is a corpus-level scalar that a kernel reordering can leave unmoved while flipping percents of the greedy tokens a user sees. Use the teacher-forced logit trace instead (`examples/logit_trace` with `scripts/compare_logit_traces.py`, documented under [Judging a change that moves the numbers](docs/benchmarks.md#judging-a-change-that-moves-the-numbers)) and put its numbers in the PR body. Two things decide whether the answer means anything: gate on disagreement at **decided** positions, because a position the reference was indifferent about has no right answer to get wrong; and trace at the **width the code under test actually runs at, behind a realistic context**, because the forward width selects which quantized-matmul kernel MLX dispatches and therefore selects what is being measured. The same comparison has read 20.6% disagreement at width 8 and 0.0% at width 32.
6. For inference changes, validate against a real checkpoint. Synthetic or build-only validation is not enough: a shape-compatible change can compile, pass unit tests, and still produce wrong logits on an actual quantized checkpoint. Fetch one with `mlxcel download mlx-community/<model-id>`, and see [`docs/supported-models.md`](docs/supported-models.md) for the families each code path covers. A change to a shared component should be smoke-tested against at least two families.
7. Commit with a conventional prefix (see below) and a clear message.
8. Push to your fork and open a Pull Request. The PR template will prompt for a summary, test plan, and linked issues.

### Commit and PR conventions

Write commits, PR titles, and issue comments in **English**. Use Conventional Commits prefixes:

| Prefix | When |
|--------|------|
| `feat:` | New user-visible feature |
| `fix:` | Bug fix |
| `perf:` | Performance improvement with measurable evidence |
| `refactor:` | Internal restructuring without behavior change |
| `chore:` | Build, CI, dependencies, release infrastructure |
| `docs:` | Documentation |
| `test:` | Tests only |

### Code standards

- Follow standard Rust conventions: `rustfmt`, `clippy -D warnings`, idiomatic ownership and error handling.
- Tests live next to the code (`_tests.rs` files) for unit tests, and under `tests/` for end-to-end integration.
- When modifying a function shared by multiple models, update the `// Used by: Model1, Model2, …` comment above it. See [`docs/code-guidelines.md`](docs/code-guidelines.md).
- Keep files under the size and module-split thresholds in [`docs/code-guidelines.md`](docs/code-guidelines.md#file-size-and-module-structure); move inline tests to a sibling `_tests.rs` file once they outgrow the guidance there.
- Do not introduce Python on the inference request path. Python is acceptable only for benchmarks and out-of-band tooling.

### Cross-repository issue references

`#NNN` auto-links to `lablup/mlxcel`, so use a bare `#NNN` **only** for issues and PRs in this repository. Any reference to another repository must be qualified so it resolves correctly and never leaks a private-repo number:

- Upstream references are written `org/repo#NNN` — `ml-explore/mlx-lm#1240`, `Blaizzy/mlx-vlm#1181`, `ml-explore/mlx#3475`, `huggingface/transformers#NNN`.
- `mlxcel-internal` (private) numbers must never appear anywhere — code comments, docs, commit subjects, or PR bodies. Map an internal reference to its public-equivalent PR/issue when one demonstrably exists; otherwise describe the change without a number.

Pre-flight before pushing — review every bare 3+-digit reference you add:

```bash
git diff origin/main...HEAD | grep -nE '#[0-9]{3,}'
# or, scoped and classified (advisory by default; STRICT=1 to gate):
python3 scripts/ci/check_cross_repo_refs.py
```

When `GH_TOKEN` or `GITHUB_TOKEN` is available, the helper asks GitHub for the
current highest issue/PR number in `lablup/mlxcel` and treats any larger bare
ref as likely cross-repository. Lines that explicitly name an upstream project
are still flagged regardless of the number. Offline, unauthenticated, or
failed-API runs stay advisory, print the fallback reason, and leave non-upstream
bare refs in the manual-review bucket.

CI runs the same check on every pull request (advisory). Same-repository pull
requests pass `github.token` so the live boundary is exercised there too; fork
pull requests intentionally use the offline fallback rather than exposing the
base repository's token to PR-controlled code. The same CI step also runs
`scripts/ci/check_cross_repo_refs_test.sh`, the companion shell test for the
classifier.

### Adding a new model family

See [`docs/adding-models.md`](docs/adding-models.md) for the full checklist. The short version: land one working checkpoint plus tests before broadening, mirror the `mlx-lm` / `mlx-vlm` directory shape where it helps, and update [`docs/supported-models.md`](docs/supported-models.md) plus the detection table in `src/models/detection.rs`.

### Bumping the MLX upstream pin

Edit one line: `GIT_TAG` in the `FetchContent_Declare(mlx ...)` block of [`src/lib/mlx-cpp/CMakeLists.txt`](src/lib/mlx-cpp/CMakeLists.txt). That is the commit CMake fetches, and since #1047 it is the only place the value is written down. It must be a full 40-character lowercase hex SHA; a branch or tag name is rejected, because the build-cache marker and the fetched-HEAD check both compare against an exact commit.

Everything else derives from it:

| Consumer | How it reads the pin |
|----------|----------------------|
| `src/lib/mlxcel-core/build.rs` | Parses the `GIT_TAG` line at build time via `build_support/mlx_pin.rs`, then drives `_deps/` purging, the `_deps/.mlx-build-commit` marker, the post-build check that the fetched `_deps/mlx-src` HEAD really is that commit, and the `MLXCEL_MLX_COMMIT` value baked into the binary |
| `.github/workflows/release.yml` | Runs [`scripts/ci/mlx_pinned_commit.sh`](scripts/ci/mlx_pinned_commit.sh) inside each "Validate MLX build cache" step |

Both parsers scope themselves to the declaration whose `GIT_REPOSITORY` names the MLX repository, so a second `FetchContent_Declare` cannot supply the pin by accident, and both fail loudly rather than guess when the line is missing, duplicated, or malformed. The Rust half is unit-tested by `cargo test -p mlxcel-mlx-pin`, which runs in seconds because it does not compile `mlxcel-core`.

This used to be three literals in three files with nothing checking that they agreed. A partial bump left `_deps/` looking valid, so `FetchContent` never re-ran and the build linked the previous MLX while reporting the new commit. The workflow's copy had in fact already fallen a bump behind by the time #1047 was filed.

After bumping the pin, re-validate the in-tree fused Metal kernel launchers in `src/lib/mlx-cpp/turbo/`, which are runtime-JIT paths a breaking MLX API change can silently regress:

- `sparse_v_sdpa.cpp`, test `sparse_v_kernel_threshold_zero_matches_graph`.
- `turbo4_delegated_sdpa.cpp::turbo4_delegated_cold_weighted_sum`, test `delegated_fused_kernel_matches_reference_over_200_steps`.
- `turbo4_delegated_sdpa.cpp::turbo4_delegated_steel_sdpa`, test `delegated_steel_envelope_matches_cold_only_fused_over_200_steps`.

All three should produce output within RMS < 5e-3 of the graph reference on Apple Silicon.

## Development environment

Detailed setup instructions are in [`docs/installation.md`](docs/installation.md).

Minimum:

- Rust **1.97+** (project uses edition 2024)
- macOS: Apple Silicon Mac on macOS Sonoma+; Xcode Command Line Tools
- Linux: CUDA 13+ toolchain, OpenBLAS, LAPACK (see [`docs/installation.md`](docs/installation.md) for the package list)

Recommended local tooling:

```bash
cargo install cargo-deny --locked
cargo install cargo-audit --locked
```

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to abide by its terms.

## Questions

- General questions, design discussion: open a [GitHub Discussion](https://github.com/lablup/mlxcel/discussions) (when enabled) or a `question` issue.
- Security: see [`SECURITY.md`](SECURITY.md).

## License

By contributing to mlxcel, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
