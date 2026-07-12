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
| `MLXCEL_DEVICE` | Runtime device hint (`gpu` or `cpu`) | auto |
| `MLXCEL_WIRED_LIMIT` | Apple Silicon wired-memory ceiling, e.g. `64GB`; `0`/`none` disables it | `max` |
| `LLAMA_ARG_*` | Environment-backed server options accepted by clap | unset |

For the complete `MLXCEL_*` reference, see
[Environment variables](environment-variables.md).

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
those first-use paths are not safe against concurrent test threads: the
default parallel run can abort with
`cudaStreamEndCapture ... previous error during capture` (a module load
racing another thread's stream capture) or, with graphs disabled, with
`cuLaunchKernelEx ... invalid argument` (a kernel-configure race). Inference
binaries are unaffected; this is a test-parallelism artifact.

```bash
cargo test --release --features cuda -- --test-threads=1
```

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
