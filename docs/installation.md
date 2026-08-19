# Installation

`mlxcel` builds two native executables from the root Rust package:

- `mlxcel` — command-line generation, model listing, and downloads.
- `mlxcel-server` — HTTP server with OpenAI/llama-server-style endpoints.

The binaries do not require Python or Node.js at runtime. They are not fully
static binaries: platform GPU/runtime libraries are still required.

## Supported platforms

| Platform | Status | Typical feature flags | Notes |
|----------|--------|-----------------------|-------|
| macOS on Apple Silicon | primary | `metal,accelerate` | Main development and validation target. |
| Linux with NVIDIA CUDA | secondary | `cuda` | Release builds currently target CUDA 13-era systems; other versions depend on MLX/CUDA compatibility. |
| Linux CPU-only | not a release target | none | May compile in limited configurations, but it is not a useful or validated inference target for this project. |
| Windows | not documented here | — | The current public installation path is macOS/Linux. |

## Cargo feature flags

Both binaries (`mlxcel` and `mlxcel-server`) build from the same root package, so
one feature set applies to both. Pass them with `cargo build --features <a,b>`.
Shipping builds enable only the platform backend flags; the rest are opt-in seams
or test scaffolding.

| Feature | Default | Effect |
|---------|---------|--------|
| `surgery` | **on** | Axis A weight-load surgery. Exposes `--surgery <config.yaml>` and `MLXCEL_SURGERY` for `scale` / `add` / `prune` / `replace` / `interpolate` weight-space edits at load time, and pulls in the `mlxcel-surgery` crate. When no surgery config is supplied the load path is byte-for-byte identical to a build without the feature. |
| `metal` | off | Apple Silicon Metal GPU backend (delegates to `mlxcel-core/metal`). Standard on macOS. |
| `accelerate` | off | Apple Accelerate CPU BLAS backend (delegates to `mlxcel-core/accelerate`). Standard on macOS. |
| `cuda` | off | NVIDIA CUDA GPU backend (delegates to `mlxcel-core/cuda`). Required on NVIDIA hosts; a plain build is CPU-only (see the footgun note below). |
| `experimental-backend` | off | Reserves the non-MLX compute-backend seam slot (issue #338). Ships no kernels and adds no runtime dispatch; it only compiles the plug-in boundary where a future non-MLX engine (e.g. FuriosaAI RNGD) would implement `ComputeBackend`. `select_backend()` still folds to MLX. |
| `xla-backend` | off | OpenXLA / StableHLO backend seam (issue #449, [ADR 0004](adr/0004-compute-backend-session-seam-and-stablehlo-family.md)). Pulls in `mlxcel-xla` and compiles the `Backend::Xla` / `Session::Xla` arms and the `MLXCEL_BACKEND=xla` selector, but no native execution engine: the crate is pure-Rust stubs plus the StableHLO graph emitter, so CI builds it unchanged. |
| `xla-iree` | off | `xla-backend` plus real IREE execution (`mlxcel-xla/iree`). Compiles a C shim against a prebuilt IREE runtime and drives the bundled prefill / decode_step graphs. Needs `IREE_DIST` (or the source-build vars below) at build time, so it is a local / opt-in build, not a CI or release default. |
| `test-utils` | off | Test-only helpers. Required to build the `distributed_integration`, `pipeline_e2e`, and `paged_handoff_parity` integration tests (`cargo test --features test-utils`). Not needed for the binaries. |

`default = ["surgery"]`, so a plain `cargo build` enables surgery only. A real
build always adds a platform backend on top, e.g. `--features metal,accelerate` on
Apple Silicon or `--features cuda` on NVIDIA. Build with `--no-default-features`
to drop the `mlxcel-surgery` crate entirely (CI parity tests against pre-surgery
behavior, or constrained embedded targets):

```bash
# Metal + Accelerate, no surgery crate.
cargo build --release --no-default-features --features metal,accelerate
```

### OpenXLA / StableHLO backend (`xla-backend`, `xla-iree`)

The XLA path is a two-tier opt-in and never enters Apple-Silicon or CUDA shipping
builds, so those binaries compile none of it and the seam folds to MLX:

- `xla-backend` compiles only the seam: the `Backend::Xla` / `Session::Xla` arms,
  the `MLXCEL_BACKEND=xla` selection, and the StableHLO graph emitter. It needs no
  native toolchain, so CI builds it unchanged.
- `xla-iree` adds the executing runtime. Its build script compiles a C shim
  against a prebuilt IREE distribution, so one of these must be set at build time:
  - `IREE_DIST`: the extracted `iree-dist-<ver>-linux-<arch>` tree (CPU / Vulkan
    dist). The dist's own `bin/iree-compile` lowers the bundled graphs.
  - `IREE_CUDA_HOME` (+ `IREE_CUDA_COMPILE`): a source-built CUDA-enabled IREE
    runtime and a matching cuda-capable `iree-compile`, for the GB10-class GPU
    path. `scripts/iree/setup-cuda.sh` produces this tree.
  - `IREE_MACOS_HOME` (+ `IREE_MACOS_COMPILE`): a source-built macOS runtime and
    a Metal-capable `iree-compile`, for the Apple Silicon dev path.
    `scripts/iree/setup-macos.sh` produces this tree and prints the matching
    environment.

At runtime, select the backend with `MLXCEL_BACKEND=xla` and tune it with the
`MLXCEL_XLA_*` variables (device, precision, packed quant). See
[Environment variables](environment-variables.md#openxla--stablehlo-backend-variables)
for the full list and [ADR 0004](adr/0004-compute-backend-session-seam-and-stablehlo-family.md)
for the design.

## macOS on Apple Silicon

Prerequisites:

- Apple Silicon Mac.
- Rust toolchain compatible with the Rust 2024 edition.
- Xcode Command Line Tools (`xcode-select --install`).
- Metal toolchain component.
- CMake available on `PATH`.
- `ffmpeg` 5.0 or newer, only if you need video input (`brew install ffmpeg`).
  It is a runtime dependency, not a build one: the build and every text, image,
  and audio path work without it, and `--video` reports a named error when it
  is absent. See [Video input and ffmpeg](#video-input-and-ffmpeg).

```bash
# One-time: install the Metal shader compiler if it is not already present.
xcodebuild -downloadComponent MetalToolchain

git clone https://github.com/lablup/mlxcel.git
cd mlxcel
cargo build --release --features metal,accelerate
```

The build outputs:

```text
target/release/mlxcel
target/release/mlxcel-server
```

The macOS release workflow also packages a `mlx.metallib` artifact when needed.
If you distribute binaries manually, verify the runtime package layout against the
release workflow rather than assuming the executable alone is always sufficient.

## Linux with CUDA

Prerequisites vary by distribution and CUDA version. At minimum you need:

- Rust toolchain compatible with the Rust 2024 edition.
- CMake and a C++20-capable compiler.
- CUDA toolkit with `nvcc`.
- NVIDIA driver compatible with the selected CUDA toolkit.
- cuDNN and CUDA runtime libraries required by the pinned MLX build.
- BLAS and LAPACK development packages, including the C headers. MLX's CMake
  resolves `cblas.h` and `lapacke.h`, so the `lapacke` headers must be present,
  not only the runtime libraries.
- `ffmpeg` 5.0 or newer, only if you need video input
  (`sudo apt-get install -y ffmpeg`). Runtime only, same as on macOS; see
  [Video input and ffmpeg](#video-input-and-ffmpeg).

On Debian/Ubuntu (x86_64 or aarch64) the build packages are:

```bash
sudo apt-get install -y \
    build-essential cmake git \
    libopenblas-dev liblapack-dev liblapacke-dev
# CUDA toolkit (nvcc) and cuDNN come from NVIDIA's apt repository, e.g.
#   cuda-toolkit-13-0  cudnn9-cuda-13
```

`liblapacke-dev` is the package that ships `lapacke.h`; `liblapack-dev` alone
omits it and the MLX CMake configure step fails with `LAPACK_INCLUDE_DIRS` set
to `NOTFOUND`.

Example build shape:

```bash
git clone https://github.com/lablup/mlxcel.git
cd mlxcel
cargo build --release --features cuda
```

> **CPU-only build footgun.** A plain `cargo build --release` on Linux uses the
> default features (no `cuda`) and produces a CPU-only binary. It still loads and
> generates, but silently runs MLX on the host CPU at a fraction of GPU
> throughput (single-digit tok/s on GB10 instead of hundreds), so the mistake is
> easy to miss. Always pass `--features cuda` on an NVIDIA host.

If CUDA is not installed under `/usr/local/cuda`, set `CUDA_HOME`:

```bash
CUDA_HOME=/opt/cuda cargo build --release --features cuda
```

### CUDA architecture selection

`src/lib/mlxcel-core/build.rs` reads `MLX_CUDA_ARCHITECTURES`. If it is unset,
the build script tries to detect the compute capability with `nvidia-smi` and
falls back to `90a` when detection fails. For SM 90 and above it appends CUDA's
architecture-specific `a` suffix (so `90` becomes `90a`), because the dedicated
Hopper quantized kernel (`qmm_sm90`) is only compiled when `90a` is in the arch
list. An explicitly set `MLX_CUDA_ARCHITECTURES` is used verbatim, so include the
suffix yourself for Hopper (`90a`).

```bash
# Hopper / GH200-style target. The `a` suffix is required for the Hopper
# quantized kernel; plain `90` builds without it.
MLX_CUDA_ARCHITECTURES=90a cargo build --release --features cuda

# GB10 / DGX Spark-style target used by the release workflow.
MLX_CUDA_ARCHITECTURES=121 cargo build --release --features cuda

# Multiple targets, if your MLX/CUDA toolchain supports them.
MLX_CUDA_ARCHITECTURES="90a;121" cargo build --release --features cuda
```

The repository release workflow builds two Linux CUDA targets on self-hosted
runners, each as one fat binary: aarch64 covering GH200 (`90a`), GB200 (`100`),
and GB10 (`121`) in a single build (`90a;100;121`), and x86_64 covering Ampere
through Blackwell (`80;86;89;90a;100;120`). For each target the `mlxcel` CLI and the
`mlxcel-server` are published as separate archives (`mlxcel-...` and
`mlxcel-server-...`, each roughly 347 MB) so a consumer downloads only the one
it needs. Every published release also ships a CycloneDX SBOM named
`sbom-<version>.cyclonedx.json.gz` for supply-chain transparency and
vulnerability scanning. Treat other GPU/OS combinations as source builds that
need local validation.

### Prebuilt CUDA artifact: runtime requirements

MLX's CUDA backend compiles some kernels at runtime with NVRTC the first time
they run (gather and other indexing kernels, and since the 2026-07 MLX pin
also the quantized matmul kernels), so a prebuilt binary needs CUDA headers
available on the deployment host, not only the runtime libraries:

- **CCCL (libcu++) headers** are bundled inside the prebuilt Linux CUDA
  archives (both aarch64 and x86_64). Each unpacks to `bin/` + `include/cccl/`,
  the layout MLX's JIT looks for relative to the executable
  (`<exe-dir>/../include/cccl`). Keep `mlxcel`/`mlxcel-server` under `bin/` and
  the `include/cccl/` directory beside it; do not flatten them. The runtime
  resolves the bundled headers from the executable's canonical path
  (`/proc/self/exe`), so any launch style works, including a relative
  `./mlxcel`. Set `MLXCEL_CCCL_DIR` to point the JIT at the CCCL headers
  explicitly, e.g. when embedding mlxcel and keeping a flat binary layout.
- **CUTLASS/CuTe headers** are bundled the same way (`include/cute/` and
  `include/cutlass/` beside `bin/`). The MLX pin from 2026-07 on JIT-compiles
  the quantized matmul kernels (`qmm`, `gather_gemm`) with NVRTC, and those
  kernels include `<cute/...>`/`<cutlass/...>`. The JIT resolves them from
  `<exe-dir>/../include`; set `MLXCEL_CUTLASS_DIR` to a directory containing
  `cute/` and `cutlass/` to override, e.g. for a flat embedded layout. Source
  builds fall back to the build tree automatically. Without these headers the
  first quantized-model run fails with
  `cannot open source file "cute/numeric/numeric_types.hpp"`.
- **CUDA toolkit headers** (`cuda_runtime.h` and friends) come from the host.
  Install the CUDA toolkit and set `CUDA_HOME` (or `CUDA_PATH`) if it is not at
  `/usr/local/cuda`. Without them the first NVRTC compile fails with
  `cannot open source file` errors.
- An NVIDIA driver matching the CUDA toolkit must be present to run on the GPU.

Compiled kernels are cached on disk (`MLX_PTX_CACHE_DIR`, default under the
system temp dir), so only the first run of each kernel variant pays the NVRTC
cost. Point `MLX_PTX_CACHE_DIR` at a persistent path to keep the cache across
sessions.

### C++ ISA baseline (`MLXCEL_CXX_MARCH`)

In release builds the C++ bridge defaults to `-march=native`, which tunes for
(and only runs on) the build host's CPU. That is correct for builds that run
where they are built (developer machines, the per-machine GB10/GH200 release
assets). For a binary that must run on other machines, set `MLXCEL_CXX_MARCH`
to a portable baseline; the release workflow's x86-64 assets use `x86-64-v3`
(AVX2):

```bash
# Portable x86-64 build (any AVX2-capable CPU, ~2013+).
MLXCEL_CXX_MARCH=x86-64-v3 cargo build --release --features cuda

# Omit -march entirely (compiler default baseline).
MLXCEL_CXX_MARCH=none cargo build --release --features cuda
```

## Runtime environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_HOME` | CUDA toolkit root, build-time and for runtime NVRTC headers | `/usr/local/cuda` when present |
| `MLX_CUDA_ARCHITECTURES` | CUDA SM target list, build-time | auto-detect via `nvidia-smi`, then `90a` fallback |
| `MLXCEL_CXX_MARCH` | C++ bridge `-march` value, build-time; `none` omits the flag | `native` |
| `MLXCEL_CCCL_DIR` | Override for the bundled CCCL (libcu++) header dir used by the CUDA NVRTC JIT | bundled `<exe-dir>/../include/cccl`, then build-time fallback |
| `MLXCEL_CUTLASS_DIR` | Override for the bundled CUTLASS/CuTe header dir used by the CUDA NVRTC JIT for quantized matmul kernels | bundled `<exe-dir>/../include`, then build-time fallback |
| `MLX_PTX_CACHE_DIR` | On-disk cache for JIT-compiled CUDA kernels | system temp dir |
| `MLXCEL_QUIET_JIT` | Suppress the one-time "compiling CUDA kernels" notice on a cold first run | unset (notice shown) |
| `MLXCEL_DEVICE` | Runtime device hint (`gpu` default; `metal`, `cpu`) | `gpu` |
| `MLXCEL_WIRED_LIMIT` | Apple Silicon wired-memory ceiling, e.g. `64GB`; `0`/`none` disables it | `max` |
| `LLAMA_ARG_*` | Environment-backed server options accepted by clap | unset |

For the complete `MLXCEL_*` reference, see
[Environment variables](environment-variables.md).

## Video input and ffmpeg

Video frame extraction shells out to the system `ffmpeg` and `ffprobe`. Both
must be on `PATH`, and both must come from **ffmpeg 5.0 (2022) or newer**.
Neither is a build-time dependency: a build without ffmpeg is complete and
every text, image, and audio path works, and `--video` (CLI) or a `video_url`
content block (server) returns a named error rather than failing obscurely.

```bash
# macOS
brew install ffmpeg
# Debian / Ubuntu
sudo apt-get install -y ffmpeg

ffmpeg -version | head -1   # must report 5.0 or newer
```

The floor is set by one flag. Extraction passes `-fps_mode vfr`, which ffmpeg
added in 5.0 at the same time it deprecated the older `-vsync`; ffmpeg 8
removed `-vsync` outright. On 4.x and older, `-fps_mode` is unrecognized and
video input is unsupported, so upgrade the system binary rather than trying to
work around it. There is no upper bound; releases through 9.x work unchanged.

A wrong-version ffmpeg fails at argument parsing, before any frame is decoded,
so the error names the option rather than the video:

```text
Unrecognized option 'fps_mode'.
Error splitting the argument list: Option not found
```

Contributors touching the video path should run `make verify-test-video`, which
runs the ffmpeg-backed tests for real. They are `#[ignore]` in the normal suite,
so a machine without ffmpeg reports them as ignored instead of silently passing
(#1172).

## Verifying the build

```bash
./target/release/mlxcel --version
./target/release/mlxcel-server --version

# `download` defaults to the global store at
# ${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>.
./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit
./target/release/mlxcel generate \
    -m ~/.cache/mlxcel/models/mlx-community/Qwen3-0.6B-4bit \
    -p "Hello" -n 1
```

On CUDA hosts, run the test suite single threaded. Since the 2026-07 MLX pin
the quantized kernels are JIT-compiled and module-loaded on first use, and
those first-use paths are not safe against concurrent test threads, so the
default parallel run aborts. The measured signatures are in the table below.
Inference binaries are unaffected; this is a test-parallelism artifact.

```bash
make verify-test-cuda
# which is:
cargo test --workspace --profile test-fast --features cuda --no-fail-fast -- --test-threads=1
```

Three runs of `cargo test --release --features cuda -p mlxcel-core --lib` on an
idle GB10 (sm_121) at MLX pin `2c46b953` put numbers on that (#1048):

| Threads | `MLX_USE_CUDA_GRAPHS` | Outcome |
|---|---|---|
| default (20) | on | SIGABRT, `cudaStreamEndCapture ... previous error during capture` |
| `--test-threads=1` | on | ran to a verdict, 1410 tests in 88s |
| default (20) | `0` | SIGABRT, `cuLaunchKernelEx ... invalid argument` |

The third row is why `MLX_USE_CUDA_GRAPHS=0` is not the workaround it looks
like: disabling capture does not rescue the parallel run, it only changes which
CUDA call reports the failure, from a module load racing another thread's
stream capture to a kernel-configure race. Serializing addresses the cause;
capture stays fully on under the gate, so the suite keeps exercising it. The
same command under `[profile.test-fast]`, which is what `make verify-test-cuda`
actually builds, behaves the same way: 1411 passed, 4 failed, 89.35s, no abort.
The abort site and the error text both move between runs, which is what makes
the raw SIGABRT expensive to read: it looks like whichever test happened to be
running is broken.

`mlxcel-core` carries a `the_cuda_test_suite_must_run_single_threaded`
guard (`src/lib/mlxcel-core/src/cuda_test_serialization_tests.rs`) so an
invocation that forgets the flag fails by name with the right command instead.
Being an ordinary test, the guard is filtered out of any narrowed run whose
filter does not match its name, and those runs stay parallel; scoped subsets
pass parallel and it is whole-suite runs that abort. A filter that does match
it, `--lib cuda` for one, trips the guard on a run that would have been safe;
set `MLXCEL_ALLOW_PARALLEL_CUDA_TESTS=1` to downgrade it to a warning there.

`make verify-test-cuda` is the Linux/NVIDIA counterpart of `make verify-test`.
Before #1048 there was no CUDA target that ran `mlxcel-core`'s tests at all:
`verify-test` pins `--features metal,accelerate`, and `make test-fast-cuda`
serializes but stays on the root package, so a bare `cargo test` under it
resolves to `-p mlxcel` and never builds `mlxcel-core`. That is the same
blindness #1007 removed on macOS, on the other backend.

## Fast iteration builds

`cargo build --release` (and a hand-run `cargo test --release`) use
`[profile.release]`: fat LTO across all ~439 locked crates plus
`codegen-units = 1` for the ~390k-line main crate. That is the right tradeoff
for anything you ship, but it is expensive for the day-to-day edit-test loop:
measured at 4 to 6 minutes per incremental rebuild, so a typical issue cycle of
several edit-test iterations pays 20+ minutes of pure compile time.

For local and agent development, use `[profile.test-fast]` instead (thin LTO,
`codegen-units = 16`, incremental compilation, `strip = false`; still
`opt-level = 3` so MLX-heavy numerics stay representative):

```bash
# CPU / Metal / Accelerate (macOS adds metal,accelerate automatically)
make test-fast

# Linux / CUDA
make test-fast-cuda

# Narrow to a subset while iterating
make test-fast-cuda FILTER=server::chat_request
```

or invoke cargo directly:

```bash
cargo test --profile test-fast --features cuda -- --test-threads=1
```

Measured on the Linux/CUDA development machine (2026-07): a cold `test-fast`
build (all dependencies plus the MLX C++ tree) takes about 4m53s, and an
incremental rebuild after touching one main-crate source file takes about 19s,
versus 4 to 6 minutes under `[profile.release]`, roughly a 13x to 19x
iteration speedup. A representative narrow test set (139 tests across model,
server, sampling, and cache modules) passes identically under both profiles.

Use `[profile.release]` (`make release*`, or plain `cargo build --release`) for
anything you ship, benchmark, or quote as representative performance:
`test-fast` trades link time and binary size for rebuild speed and is not tuned
for either.

Running *tests* is the exception. `make verify-test`, and therefore the
[nightly workflow](https://github.com/lablup/mlxcel/blob/main/.github/workflows/nightly-verify.yml)
that invokes it, builds the test binaries under `test-fast` as well. Linking
roughly 77 test binaries under fat LTO was costing that job its entire
180-minute budget before a single test ran. `opt-level = 3` is unchanged, so
the optimised MLX numerics the suite depends on are the same; what is no longer
covered is a defect that reproduces only under fat LTO or `codegen-units = 1`.
Reach for `cargo test --release --features metal,accelerate` by hand when you
are chasing one of those.

## Why the gate says `--workspace`

`make verify-clippy` and `make verify-test` pass `--workspace`, and dropping it
changes what they cover rather than only how fast they run. The workspace root
in this repository is itself the `mlxcel` package, so a bare `cargo test` or
`cargo clippy` here resolves to `-p mlxcel` and never builds `mlxcel-core`,
`mlxcel-surgery` or `mlxcel-xla`. Until #1007 the gate did exactly that, which
left 1754 tests unrun, 1354 of them in `mlxcel-core`, the crate holding the MLX
`cxx` bridge, `layers.rs`, the KV cache and the quantization loaders. The lint
half of the hole is easier to miss: `--all-targets` without `--workspace` does
not compile a member's *test* target either, so test-only lint errors and
test-only compile errors both passed the gate.

There is deliberately no `default-members` in `Cargo.toml` doing this instead.
It would re-scope every bare cargo invocation in the repository at once,
including the `cargo build --release --target aarch64-apple-darwin --locked`
that `release.yml` runs, which would start compiling the default-off
`mlxcel-xla` into every release build.

Each member builds at the feature set the root selects. `mlxcel-core` resolves
to `metal` and `accelerate` through the root package's forwarding, so there is
one build of it shared by every member. `mlxcel-mlx-pin`, `mlxcel-surgery` and
`mlxcel-xla` resolve to their empty defaults; in particular `mlxcel-xla`'s
`iree` feature stays off, so its build script skips the native shim and the gate
needs no IREE distribution. The code behind `iree`, `diagnostics` and
`micro-oracle` is still outside the gate for that reason, and needs a local IREE
dist to check (see `scripts/iree/setup-macos.sh`). `mlxcel-mlx-pin` is a leaf
with no production role, holding the unit tests for the MLX-pin logic in
`mlxcel-core/build_support/mlx_pin.rs`; it deliberately does not depend on
`mlxcel-core`, so `cargo test -p mlxcel-mlx-pin` runs in seconds instead of
triggering an MLX C++ build.

`make verify-test-cuda` (#1048) says `--workspace` for the same reason and
resolves the same way, with `cuda` in place of `metal,accelerate`. Pulling in
`mlxcel-surgery` and `mlxcel-xla` on the CUDA path is intended and close to
free. Both depend on `mlxcel-core`, and one `cargo test --workspace --features
cuda` unifies that into a single `cuda`-enabled build of it, so neither triggers
a second MLX compile; `mlxcel-xla`'s `iree` feature stays off there too, so it
stays pure Rust and needs no IREE distribution. What their test targets contain
is backend-agnostic Rust, so excluding them would only mean the two crates are
gated on macOS and nowhere else. `mlxcel-mlx-pin` does not depend on
`mlxcel-core` at all and costs the run seconds.

`make verify-test` also passes `--no-fail-fast`, which matters only now that
the run covers five members: without it the first failing test binary ends the
run and hides the other four behind whatever failed first. Cargo still exits
non-zero, so the gate is no weaker for it. `make verify-test-cuda` passes it
too.

Widening the scope does not put the run into the concurrency hazard of #1008,
where two `mlxcel-core` suites sharing one Metal device aborted 7 of 12 runs.
Cargo builds every test binary and then runs them one at a time, so the
`mlxcel-core` suite never overlaps the root suite on the device. That is
measured, not assumed: on cargo 1.97.1 a three-crate probe workspace completes
its entire build before the first test binary starts, and finishes each binary
before starting the next. Anything that changes it, a parallel test runner such
as `cargo nextest` for instance, has to re-establish it: the
`no_other_mlxcel_core_test_binary_is_sharing_the_gpu` guard in
`src/lib/mlxcel-core/src/gpu_exclusivity_tests.rs` detects a second
`mlxcel-core` binary, not a root-suite binary competing for the same device.

**`make verify-test` also passes `--test-threads=1` (#1092).** The sequencing
above bounds concurrency *between* binaries and nothing else, and the failure
that took `main` red on 2026-08-16 was inside one: the `mlxcel-core` binary
died with `signal: 11, SIGSEGV`, publishing no panic and no `test result` line,
so `--no-fail-fast` had nothing to collect and cargo reported a failed target
with no explanation. libtest defaults to one test thread per logical CPU, and
the macOS crash report from the local repro on an 18-core M5 Max shows what
that means here: 18 libtest workers live at the fault, all of them running
MLX-backed cache tests, two inside `iokit_user_client_trap` and two inside the
allocator, faulting on an address in no mapped region. It is the CUDA abort of
#1048 on the other backend. `--jobs 1` does not address it, because `--jobs`
bounds the build and the build has already finished by the time any test runs.

Serializing is close to free, because the work serializes on the one Metal
device whether or not the host threads do. Measured on an M5 Max at `5dfcb390`,
warm cache, whole workspace, 101 binaries and 8128 tests:

| test threads | wall clock |
|---|---|
| default (18) | 69.17s |
| `--test-threads=1` | 76.39s |

+7.2s, against a `cargo test` step the nightly budgets 180 minutes for and
whose time goes to the build rather than to running tests. The two large
members pull in opposite directions and nearly cancel: `mlxcel-core` costs
+23s serialized (10.2s to 33.2s), while the root suite *gains* 12s (23.5s to
11.6s), because thread contention across 5695 tests is worse than running them
in a row. Order matters when reproducing these numbers: a cold first run pays
roughly 50s of one-time Metal shader compilation, which is enough to invert the
comparison if the two arms are not both warm.

There is deliberately no macOS counterpart to the
`the_cuda_test_suite_must_run_single_threaded` guard. The CUDA suite aborts
every time it runs parallel, so failing by name costs nothing; the Metal suite
crashes rarely, and a hard guard would break `cargo test -p mlxcel-core --lib`,
which is three times faster parallel and succeeds nearly always. Narrowed
hand-runs are meant to stay parallel. It is the whole-suite gate that
serializes.

## Troubleshooting

**Missing Metal toolchain on macOS** — run
`xcodebuild -downloadComponent MetalToolchain` and rebuild.

**`Cannot find CUDA library directory` on Linux** — set `CUDA_HOME` to the CUDA
toolkit root and rebuild.

**`nvidia-smi` is unavailable on the build host** — set `MLX_CUDA_ARCHITECTURES`
explicitly.

**CUDA/cuDNN linker errors** — confirm that the libraries expected by the pinned
MLX version are installed and discoverable by the linker. The root build script
links CUDA runtime/math libraries directly and relies on the system driver for
`libcuda`.

**`gmake: *** Error 137` (SIGKILL) while compiling `qmm_*.cu`** — the build ran
out of memory. The CUTLASS-heavy quantized-matmul kernels peak at ~4-5 GB of
compiler memory per parallel job, so a default `-j$(nproc)` build needs roughly
`5 GB × cores`. Cap the parallelism with `cargo build -j N ...` (cargo forwards
`N` to the CMake subbuild); pick `N ≈ available_RAM_GB / 5`.

**CMake error: `LAPACK_INCLUDE_DIRS ... NOTFOUND`** — install `liblapacke-dev`
(MLX needs `lapacke.h`, which `liblapack-dev` alone does not provide) and
`libopenblas-dev`.
