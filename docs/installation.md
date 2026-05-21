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
- OpenBLAS and LAPACK development/runtime packages.

Example build shape:

```bash
git clone https://github.com/lablup/mlxcel.git
cd mlxcel
cargo build --release --features cuda
```

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

The repository release workflow currently builds Linux ARM64 CUDA artifacts for
GB10 (`121`) and GH200 (`90a`) on a self-hosted runner. Treat other GPU/OS
combinations as source builds that need local validation.

## Runtime environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_HOME` | CUDA toolkit root used by the build script | `/usr/local/cuda` when present |
| `MLX_CUDA_ARCHITECTURES` | CUDA SM target list, build-time | auto-detect via `nvidia-smi`, then `90a` fallback |
| `MLXCEL_DEVICE` | Runtime device hint (`gpu` or `cpu`) | auto |
| `MLXCEL_WIRED_LIMIT` | Apple Silicon wired-memory ceiling, e.g. `64GB`; `0`/`none` disables it | `max` |
| `LLAMA_ARG_*` | Environment-backed server options accepted by clap | unset |

For the complete `MLXCEL_*` reference, see
[Environment variables](environment-variables.md).

## Verifying the build

```bash
./target/release/mlxcel --version
./target/release/mlxcel-server --version

./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit
./target/release/mlxcel generate \
    -m models/Qwen3-0.6B-4bit \
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
