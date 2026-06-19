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
it needs. Treat other GPU/OS combinations as source builds that need local
validation.

### Prebuilt CUDA artifact: runtime requirements

MLX's CUDA backend compiles some kernels (gather and other indexing kernels)
at runtime with NVRTC the first time they run, so a prebuilt binary needs CUDA
headers available on the deployment host, not only the runtime libraries:

- **CCCL (libcu++) headers** are bundled inside the prebuilt Linux CUDA
  archives (both aarch64 and x86_64). Each unpacks to `bin/` + `include/cccl/`,
  the layout MLX's JIT looks for relative to the executable
  (`<exe-dir>/../include/cccl`). Keep `mlxcel`/`mlxcel-server` under `bin/` and
  the `include/cccl/` directory beside it; do not flatten them. The runtime
  resolves the bundled headers from the executable's canonical path
  (`/proc/self/exe`), so any launch style works, including a relative
  `./mlxcel`. Set `MLXCEL_CCCL_DIR` to point the JIT at the CCCL headers
  explicitly, e.g. when embedding mlxcel and keeping a flat binary layout.
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
