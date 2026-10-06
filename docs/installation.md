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
| Linux with AMD ROCm | experimental | `rocm` | Source build only. Validated on RDNA 3.5 (`gfx1151`, Strix Halo) with ROCm 10.0 / HIP 7.15; tracked in lablup/mlxcel#1801. See [Linux with AMD ROCm](#linux-with-amd-rocm-experimental). |
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
| `rocm` | off | AMD GPU backend on Linux (delegates to `mlxcel-core/rocm`), built from the ROCm overlay in `src/lib/mlx-cpp/patches-rocm/`. Cannot be combined with `cuda` or `metal`. Experimental; see [Linux with AMD ROCm](#linux-with-amd-rocm-experimental). |
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

On macOS, MLX also enables Metal and Accelerate for a plain `cargo build --release`; the explicit features above document the intended backend. Runtime controls such as the MTP `qmv_wide` retry follow the backend actually built, including plain builds. `MLXCEL_BUILD_METAL=OFF` disables the Metal backend and its controls even if the Cargo `metal` feature is selected.

To verify the runtime controls without loading a checkpoint, run `cargo run -p mlxcel-core --profile test-fast --example metal_runtime_switch_probe`. It checks both QMV selection and the decode command-buffer override, then restores their previous values. It reports a skip when no Metal device is available.

For an older plain macOS build affected by [#1988](https://github.com/lablup/mlxcel/issues/1988), start a fresh process with `MLXCEL_QMV_WIDE=0` before the existing server command. This pins the narrower Metal projection kernel at startup while keeping the exactness gate enabled. It may reduce throughput, applies only to Metal, and still requires the loaded model's probe to pass.

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

> **Volta (sm_70) requires a CUDA 12.x toolchain.** CUDA 13 removed support for Volta, so its `nvcc` rejects `compute_70` outright with `nvcc fatal : Unsupported gpu architecture 'compute_70'` before compiling anything. The published release archives are built against CUDA 13 and therefore contain no sm_70 code at all, and the project's CUDA CI runners carry CUDA 13, so the `cuda-sm70-compile` gate skips there rather than failing. Building for a V100 or any other Volta card means a source build on a host with CUDA 12.x installed. This was verified on CUDA 12.9.41, which compiles sm_70 without complaint.


`src/lib/mlxcel-core/build.rs` reads `MLX_CUDA_ARCHITECTURES`. If it is unset,
the build script detects the compute capability with `nvidia-smi` and spells it
the way the release workflow spells it: Hopper gets CUDA's architecture-specific
`a` suffix (`90` becomes `90a`), and Blackwell (`sm_100`, `sm_120`, `sm_121`)
stays plain. Detection failure falls back to `90a`. An auto-detected build and a
published one therefore differ in which architectures they cover and never in
what machine code those architectures get, which is the point: before issue
#1943 the rule suffixed everything from SM 90 up, so a default build on a GB10
produced `121a` while the release archives for the same card carried `121`.

An explicitly set `MLX_CUDA_ARCHITECTURES` is used verbatim, so spell it the same
way by hand: `90a` for Hopper, plain for Blackwell. The rest of this section is
why Blackwell is plain.

On Blackwell the `a` suffix decides something else. MLX compiles the hardware
block-float converters in `mlx/backend/cuda/quantized/nvfp4_quantize.cuh`, the
ones that issue `cvt.rn.satfinite.e2m1x2.f32`, only when nvcc is compiling for
an architecture-specific target, which it signals with `__CUDA_ARCH_SPECIFIC__`.
A plain `121` build therefore compiles them out, and both NVFP4 and MXFP4
quantization fall back to a scalar CUTLASS conversion sequence with nothing in
the build output saying so.

Putting `121a` in the list would fix that by charging every translation unit
for it. CUTLASS derives `CUTLASS_ARCH_MMA_SM121A_ENABLED` from the same macro,
so the decode kernels compile differently too, and on a device the suffix
matches, the architecture-specific image is the one the driver loads rather
than an alternative it can ignore. On this project's GB10 that is 12.7 MB of
extra archive and a different `qmv` on every Blackwell machine, to fix a
converter no decode kernel calls.

So the list stays plain and `src/lib/mlx-cpp/CMakeLists.txt` adds the
architecture-specific image to `fp_quantize.cu` alone, which is the only
translation unit that can reach those converters. You get the hardware path
without asking for it, and without it reaching anything else. Building with
`121a` is a step backwards, not a step forwards, and it is also self-defeating:
the injection skips a capability the list already names architecture-specific,
because asking nvcc for the same `--generate-code` twice is an error rather than
a no-op. `121f` is worse still, because it satisfies the converters' dispatcher
gate but not their own gate and fails to compile outright.

Hopper's `90a` is a different case: it is what both release lists ship, so the
auto-detected spelling matches it. At the current MLX pin the suffix no longer
gates anything on its own. Upstream commit `44540d12` moved `qmm_sm80`,
`qmm_sm90` and `gather_gemm` to runtime NVRTC compilation and removed the
`MLX_CUDA_SM90A_ENABLED` definition an earlier version of this section cited,
and `jit_module.cpp` now derives the NVRTC `--gpu-architecture` from the running
device, appending `a` itself from compute capability 9 up. Cross-compiling the
pinned tree at `90` and at `90a` agrees: `qmm_sm90.cu`, `qmm.cu` and
`qmm_sm80.cu` emit no device function at either spelling, and `qmv.cu` and
`fp_qmv.cu` emit identical SASS apart from the `EF_CUDA_ACCELERATORS` header
flag that marks a cubin architecture-specific.
`CUTLASS_ARCH_MMA_SM90A_ENABLED` still keys on `__CUDA_ARCH_FEAT_SM90_ALL`, so
a later pin can make it matter again.

```bash
# Hopper / GH200-style target, spelled the way the release workflow spells it.
MLX_CUDA_ARCHITECTURES=90a cargo build --release --features cuda

# GB10 / DGX Spark-style target used by the release workflow. Plain: the
# hardware NVFP4/MXFP4 converter is added to one translation unit by CMake.
MLX_CUDA_ARCHITECTURES=121 cargo build --release --features cuda

# Multiple targets, if your MLX/CUDA toolchain supports them.
MLX_CUDA_ARCHITECTURES="90a;121" cargo build --release --features cuda
```

To confirm what a finished build carries, read the archive rather than the
environment variable. A correct Blackwell build has the fp4 instruction in
`fp_quantize.cu.o`, exactly one architecture-specific image in the whole
archive, and its forward-JIT PTX untouched:

```bash
A=$(ls -d target/release/build/mlxcel-core-*/out/build/lib/libmlx.a | head -1)
mkdir -p /tmp/mlxq && (cd /tmp/mlxq && ar x "$A" fp_quantize.cu.o qmv.cu.o)
cuobjdump --dump-sass /tmp/mlxq/fp_quantize.cu.o | grep -c F2FP.SATFINITE.E2M1  # > 0
cuobjdump --list-elf  /tmp/mlxq/qmv.cu.o | grep -c 'sm_[0-9]*a'                 # 0: decode untouched
cuobjdump --list-elf  "$A" | grep -c 'sm_[0-9]*a'                               # 1
cuobjdump --dump-ptx  "$A" | grep -c '.target sm_121$'                          # > 0
```

One checkout can hold several of those build directories, one per feature set
and architecture list, so confirm the one you are reading is the one you meant:
`grep MLX_CUDA_ARCHITECTURES "$(dirname "$A")/../CMakeCache.txt"` names the list
it was configured with.

Do not look for `cvt.rn.satfinite.e2m1x2` in the PTX. The injection emits a
cubin and no PTX, and the PTX that is emitted comes from the plain `compute_121`
pass, which takes the fallback arm, so that grep returns zero on a correct
build. The SASS count above is the one that distinguishes the two builds.

If the architecture list a binary was built with does not cover the GPU it is
started on, it refuses to start and says so, naming both the list it carries and
the compute capability it found, instead of failing later with an opaque CUDA
load error at the first kernel launch (issue #1537). Two cases produce that: a
published x86_64 archive, whose matrix starts at `80`, on a pre-Ampere card such
as a V100; and a source build made on a host where `nvidia-smi` was unavailable,
which falls back to `90a` and so cannot run on its own build machine. The fix in
both cases is the rebuild above with `MLX_CUDA_ARCHITECTURES` set to the target
device. Set `MLXCEL_TRACE_ARCH` (see
[Environment variables](environment-variables.md)) to print the running
capability, the compiled list, and whether the device is served by a cubin or by
JIT-compiled PTX; the same summary appears next to the `Detected N GPU(s)` line
at startup. `MLXCEL_DEVICE=cpu` bypasses the refusal, so a binary built for the
wrong architecture can still be run on the CPU while a correct one is built.

The repository release workflow builds two Linux CUDA targets on self-hosted
runners, each as one fat binary: aarch64 covering GH200 (`90a`), GB200 (`100`),
and GB10 (`121`) in a single build (`90a;100;121`), and x86_64 covering Ampere
through Blackwell (`80;86;89;90a;100;120`). Blackwell is plain in both for the
reason above, with the hardware converter added to one translation unit by
CMake. For each target the `mlxcel` CLI and the
`mlxcel-server` are published as separate archives (`mlxcel-...` and
`mlxcel-server-...`, each roughly 347 MB) so a consumer downloads only the one
it needs. Every published release also ships a CycloneDX SBOM named
`sbom-<version>.cyclonedx.json.gz` for supply-chain transparency and
vulnerability scanning. Treat other GPU/OS combinations as source builds that
need local validation.

### ROCm architecture selection

`src/lib/mlxcel-core/build.rs` reads `MLX_ROCM_ARCHITECTURES`. If it is unset, the build script asks `rocminfo` for the GPU agent's `gfx` target and fails with a named error when no agent is reported, which is what happens in a container that cannot reach `/dev/kfd`. Setting the variable bypasses detection entirely and is used verbatim, so a build host with no visible GPU can still produce a binary for one.

```bash
# The Radeon 8060S (Ryzen AI MAX+ 395) used for ROCm validation.
MLX_ROCM_ARCHITECTURES=gfx1151 cargo build --release --features rocm

# Multiple targets, semicolon-separated.
MLX_ROCM_ARCHITECTURES="gfx1100;gfx1151" cargo build --release --features rocm
```

HIP coverage is not the ordering CUDA coverage is. A CUDA cubin runs on a higher minor revision and its PTX JITs forward across majors, so a list can cover a device it does not name. A HIP code object is built for one `gfx` target and runs on that target only, with no JIT fallback, so `gfx1151` and `gfx1150` are unrelated despite the adjacent numbers and a list covers exactly the targets it names. Target-feature suffixes (`gfx90a:xnack+`) select code-object features on one target rather than naming another, so they are ignored when the list is compared against the device.

A binary whose compiled `gfx` list does not contain the device it is started on refuses to start and names both, rather than failing later with an opaque HIP error at the first kernel launch (issue #1805). Set `MLXCEL_TRACE_ARCH` (see [Environment variables](environment-variables.md)) to print the running target, the compiled list and whether it is covered; the same summary appears next to the `Detected N GPU(s)` line at startup, alongside the device name and its memory. `MLXCEL_DEVICE=cpu` bypasses the refusal, so a binary built for the wrong target can still run on the CPU while a correct one is built.

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
- **CUDA shared libraries** come from the host toolkit too. The binary links
  `cudart`, `cublas`, `cublasLt`, `cufft`, `cusolver`, `nvrtc` and cuDNN
  dynamically. cuSOLVER joined that list when the MLX pin moved to `81ba1c6a`
  (MLX's CUDA backend now uses it for Cholesky and initializes it at startup),
  so a runtime-only install that omits it fails at launch with
  `libcusolver.so...: cannot open shared object file`. The full CUDA toolkit
  package ships all of them.
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

## Linux with AMD ROCm (experimental)

**Experimental.** AMD GPU support on Linux was added in lablup/mlxcel#1802 and
is tracked by lablup/mlxcel#1801. It is a source build only: there is no
release artifact and no ROCm CI job running yet, and the gaps listed below are open.

The backend is not part of upstream MLX. mlxcel vendors the ROCm backend from
the `rocm-support` branch of
[NripeshN/mlx](https://github.com/NripeshN/mlx/tree/rocm-support) (MIT) into
`src/lib/mlx-cpp/patches-rocm/` and applies it on top of the same pinned MLX
commit that the Metal and CUDA builds use. See
[mlxcelverse](architecture.md#mlxcelverse-the-mlx-side-layer) for how the
overlay is organized, and `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md` for the
changes mlxcel carries on top of the fork.

Tested configuration: AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`,
RDNA 3.5) and a 96 GiB VRAM carve-out, Debian 13, ROCm 10.0.0 packages
(HIP 7.15, AMD clang 23). Other RDNA 3, 3.5 and 4 parts are expected to build;
CDNA parts (for example MI300) compile but carry no tuning, and on them the
fused kernels that reduce across a 32-lane wavefront fall back to the MLX
graph until validated (see "Wave64 devices" below).

### Prerequisites

- Rust toolchain compatible with the Rust 2024 edition, CMake and a C++20
  compiler.
- `pkg-config` and the OpenSSL headers, which the Rust dependencies need on
  Linux.
- BLAS and LAPACK development packages, including `lapacke.h` (see
  [Linux with CUDA](#linux-with-cuda)).
- A ROCm installation that provides the `hip`, `rocblas`, `rocthrust`,
  `rocprim`, `hiprand`, `rocwmma`, `hipblaslt`, `hipfft` and `hiprtc` CMake packages
  (`ls "${ROCM_PATH:-/opt/rocm}"/lib/cmake` lists them), plus `hipcc` and
  `rocminfo`.
- Access to the GPU device nodes: the build and run user must be in the `video`
  and `render` groups.

On Debian/Ubuntu:

```bash
sudo apt-get install -y \
    build-essential cmake git pkg-config libssl-dev \
    libopenblas-dev liblapack-dev liblapacke-dev
# ROCm itself comes from AMD's repository (repo.radeon.com).
rocminfo | grep -E '^\s+Name:\s+gfx'   # should list your GPU target
```

### Build

```bash
cargo build --release --features rocm
# or
make release-rocm
```

`rocm` cannot be combined with `cuda` or `metal`; the build fails with a message
naming the conflict. The first build compiles the MLX device code with `hipcc`,
which takes a few minutes on top of the Rust build.

Incremental builds track headers: `hipcc` writes a dependency file next to
each HIP object, so after you change a header under
`src/lib/mlx-cpp/patches-rocm/`, the next build recompiles exactly the `.hip`
files that include it, directly or transitively, and a build with no change
recompiles none. The overlay reaches the build tree through CMake's
`configure_file`, which rewrites a copy only when its content differs, so a
bare `touch` of an overlay header rebuilds nothing; its content has to change.
The dependency files also list ROCm and system headers, so updating those
rebuilds every HIP object on its own. They do not list headers that only the
device compilation includes (today only rocWMMA's), nor the compiler itself.

To force a clean HIP rebuild anyway (for example after upgrading `hipcc` or
rocWMMA, or to rule out a stale object while bisecting), delete the HIP objects
of the build profile you use and touch any overlay file so that Cargo reruns
the build script:

```bash
rm -rf target/release/build/mlxcel-core-*/out/build/_deps/mlx-build/mlx/backend/rocm/hip_objs
touch src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/CMakeLists.txt
cargo build --release --features rocm
```

Replace `release` with the profile directory you build (`debug`, `test-fast`,
and so on). Without the `touch`, Cargo sees no changed input, skips the build
script and keeps linking the old kernels. `cargo clean -p mlxcel-core
--release` (or `--profile <name>`) also forces it, but rebuilds the whole MLX
C++ library too.

### HIP architecture selection

The build compiles MLX device code for the `gfx` targets that `rocminfo`
reports on the build host. Set `MLX_ROCM_ARCHITECTURES` to choose them
explicitly, for example to build on a host without a GPU or for several
targets:

```bash
MLX_ROCM_ARCHITECTURES=gfx1151 cargo build --release --features rocm
MLX_ROCM_ARCHITECTURES="gfx1100;gfx1151" cargo build --release --features rocm
```

If neither the variable nor `rocminfo` yields a target, the build fails and
names the variable. The chosen list is recorded in the binary as
`MLXCEL_ROCM_ARCHITECTURES`. Set `ROCM_PATH` when ROCm is not installed under
`/opt/rocm`; the build reads its CMake packages, `hipcc` and `rocminfo` from
there.

### Running

The binaries carry `$ROCM_PATH/lib` as an rpath, so `LD_LIBRARY_PATH` is not
needed even when the ROCm libraries are not registered with the dynamic loader.
A GPU run prints `Runtime device: GPU` at startup, and the process appears in
`rocm-smi --showpids`:

```bash
./target/release/mlxcel generate -m models/mlx/Qwen3-0.6B-4bit -p "Hello" -n 50 --temp 0
```

On a UMA host the GPU shares memory with the operating system and with any
other GPU process, so check `rocm-smi --showpids` for other tenants before
loading a large model.

To check the HTTP server rather than the CLI, start it and run the chat smoke
script against it. Both a dense and an affine MoE checkpoint pass on `gfx1151`;
the results are in
[`docs/benchmark_results/rocm-correctness-gfx1151-2026-09-12.md`](benchmark_results/rocm-correctness-gfx1151-2026-09-12.md).
A sliding-window model, two SSM hybrids and a VLM were added, and compared
against an Apple M5 Max Metal reference, in
[`docs/benchmark_results/rocm-correctness-gfx1151-2026-09-30.md`](benchmark_results/rocm-correctness-gfx1151-2026-09-30.md).

```bash
./target/release/mlxcel-server -m models/mlx/Qwen3-30B-A3B-4bit --port 8080 &
./scripts/server_chat_smoke.sh --port 8080
```

The script checks `/health`, `/v1/models`, and `/v1/chat/completions` both
streaming and non-streaming. It counts the reasoning channel as output, so a
thinking model that spends its whole budget inside the thinking block reports
`channel=reasoning only` rather than looking like an empty response.

Decode and prefill throughput on `gfx1151`, against mlx-lm on the same host, is
in
[`docs/benchmark_results/rocm-baseline-gfx1151-2026-09-30.md`](benchmark_results/rocm-baseline-gfx1151-2026-09-30.md);
`scripts/bench_decode.sh` recognises a ROCm host on its own (see
[Benchmarks](benchmarks.md#rocm-hosts-issue-1810)).
Where that decode time goes, per kernel, and the order the #1814 kernel ports
should land in, is in
[`docs/benchmark_results/rocm-decode-profile-gfx1151-2026-09-30.md`](benchmark_results/rocm-decode-profile-gfx1151-2026-09-30.md).

#### Memory footprint

Two defaults keep the allocator's footprint close to the weights
(lablup/mlxcel#2062). Measured on `gfx1151` at pp512/tg128, the MLX peak for
Meta-Llama-3.1-8B-Instruct-4bit went from 20.60 GB to 6.14 GB and for
Qwen3-30B-A3B-4bit from 23.56 GB to 18.58 GB, with decode throughput
unchanged and prefill within 2%; the breakdown and every knob compared are in
[`docs/benchmark_results/rocm-memory-gfx1151-2026-09-30.md`](benchmark_results/rocm-memory-gfx1151-2026-09-30.md).

- **In-flight bound, `MLX_ROCM_MAX_INFLIGHT_MB` (default 1024).** Most of
  the old peak was not cache: a prefill allocates a transient for every
  operation (for an f16 4-bit model on the dequantize-and-GEMM path, an f16
  copy of each weight matrix), each one is released only when its command batch
  finishes on the GPU, and the host could encode far ahead of the GPU. The
  backend now commits a batch once it has allocated a quarter of this budget
  and waits for the oldest batch while more than the budget is in flight, so
  the transients stay within roughly this many MiB (the count is of
  allocations, so it is approximate). `0` restores the unbounded behavior.
- **Buffer-cache bound, `MLXCEL_CACHE_LIMIT` (default 2 GiB on ROCm).** Freed
  buffers are cached for reuse, and the ROCm allocator only reuses a buffer of
  exactly the requested size, with a cache limit that defaulted to its memory
  limit (76.8 GiB here). A ROCm build now caps the cache at 2 GiB unless
  `MLXCEL_CACHE_LIMIT` says otherwise; `0` or `none` removes the cap. Before
  this change the allocator ignored the cache limit altogether, so
  `MLXCEL_CACHE_LIMIT` had no effect on ROCm. Decode throughput did not move
  between 128 MiB and no cap on either model; prefill lost 16% on the 8B at
  128 MiB, because each f16 weight copy is reallocated once the cache cannot
  hold it, and nothing measurable from 512 MiB up. 2 GiB is four times that
  smallest free value, which leaves room for the larger weight copies of
  bigger models.

The allocator's peak counts live buffers only, so the most the allocator
holds is the peak plus the cache limit. The memory limit the pre-load
estimate reads (`memory_limit()`, 76.80 GiB) is unchanged by either default,
so `mlxcel inspect` and `--estimate-memory` give the same answers as before.

### Current status

| Area | Status on ROCm |
|------|----------------|
| Affine 4-bit / 8-bit checkpoints | Run natively. |
| mxfp8 and mxfp4 checkpoints | Run natively, including MoE experts through `gather_qmm` (for example gpt-oss-20b-MXFP4-Q4). The load log says so once per mode (`Quantization mode mxfp4: running on native ROCm kernels, no load-time conversion ...`). mxfp4 `quantize`, `quantized_matmul` and `gather_qmm` are checked against CPU references by `tests/rocm_mxfp4_quant.rs` (lablup/mlxcel#1808). Vendor FP8 block checkpoints (`quant_method: fp8`, 128x128 blocks) are requantized to mxfp8 at load and run natively; verified end to end on a dense 0.8B Qwen3.5 FP8 checkpoint, with mxfp8 `gather_qmm` and `quantized_matmul` checked by `models::switch_layers::mxfp_tests`. The official Qwen3.5 FP8 MoE releases have not been run on ROCm (lablup/mlxcel#1807). |
| NVFP4 checkpoints | No native kernel. ModelOpt NVFP4 checkpoints (for example the Gemma 4 NVFP4 exports) are converted to affine 4-bit at load with no environment variables, and the load log names the route and the reason. A layer that cannot be converted fails the load with its name and the reason. MLX-native NVFP4 exports (`"mode": "nvfp4"` in `config.json`, such as `mlx-community/*-nvfp4`) have no load-time conversion and are refused at load; use an affine export instead (lablup/mlxcel#1806). |
| Affine MoE models (for example Qwen3-30B-A3B) | Run natively. Single-token decode takes the fused two-kernel MoE path (gate-up, then down) through its HIP port (lablup/mlxcel#2065), as on Metal and CUDA: Qwen3-30B-A3B-4bit decode on `gfx1151` went from 61.3 to 62.5 tok/s, with logits matching Metal's fused kernel at every decided position ([results](benchmark_results/rocm-fused-moe-gfx1151-2026-10-05.md)); `MLXCEL_FUSED_MOE=0` selects the `gather_qmm` graph path instead. Experts wider than `MLXCEL_FUSED_MOE_MAX_DFF` (8192 by default, as on CUDA; Mixtral-8x7B's 14336) stay on `gather_qmm`. |
| Model-specific fused kernels (Apertus xIELU, Cohere2 add3 + LayerNorm, Mamba1 selective scan, Nemotron-H's opt-in `MLXCEL_FUSED_MOE_RELU2`) | Run as HIP ports (lablup/mlxcel#2069). xIELU and the add3 LayerNorm are byte-identical to ROCm's own unfused graph; the Mamba1 scan keeps its state in f32 as on Metal; the relu2 path is performance-neutral on Nemotron-H and stays opt-in. Apertus, Cohere2 and the Mamba1 families have no checkpoint on the tested host, so they are covered by kernel tests only ([results](benchmark_results/rocm-metal-only-ports-gfx1151-2026-10-05.md)). |
| Fused residual-add RMSNorm and RoPE + KV append (`MLXCEL_FUSED_ADD_RMSNORM`, `MLXCEL_FUSED_ROPE_APPEND`) | Off by default, as on Metal and CUDA. Opted in, both run as HIP ports (lablup/mlxcel#2063) whose output is byte-identical to the graph they replace, so turning them on changes no logits; decode on `gfx1151` moved by +0.3% on Llama 3.1 8B and +1.7% on Qwen2.5 7B, not clearly above the noise, so the defaults stay off ([results](benchmark_results/rocm-fused-norm-rope-gfx1151-2026-10-05.md)). |
| Paged attention (the server's batched paged decode, MLA split-KV, sparse paged decode) | The v1 decode, v2 partial and merge kernels run as HIP ports (lablup/mlxcel#2068), so these paths take the fused kernels instead of gather-then-SDPA, as on Metal and CUDA; the kernels' tests match the gather and host references on `gfx1151`. Server decode on `gfx1151` with Meta-Llama-3.1-8B-Instruct-4bit, against the gather path on the same binary: 1.28x per-request decode at batch 4 and ~16K tokens, 1.40x and 1.77x for one sequence at ~16K and ~32K, no measurable change at batch 4 and ~4K ([results](benchmark_results/rocm-paged-attention-gfx1151-2026-10-05.md)). `MLXCEL_PAGED_ATTENTION_NATIVE=0` forces the gather path. |
| Sampled decode (`temperature > 0`) | The fused Gumbel-max sampler (no filter) and the dual-pivot rejection sampler (top-p, and the other filter combinations its routing admits) run as HIP kernels, as on Metal and CUDA (lablup/mlxcel#2064). On `gfx1151` with Meta-Llama-3.1-8B-Instruct-4bit, decode at `--temperature 0.8 --top-p 0.95` went from 33.98 to 37.56 tok/s (medians of three; the before runs drifted from 36.26 down to 32.81, the after runs held 37.06 to 37.82); at `--temperature 0.8` alone the change is inside the noise ([results](benchmark_results/rocm-samplers-gfx1151-2026-10-05.md)). `MLXCEL_SAMPLING_GUMBEL=0` and `MLXCEL_SAMPLING_REJECTION=0` force the graph. |
| Hybrid SSM decode (granite-4.0-h, falcon-h1, plamo-2, Nemotron-H) | The single-token Mamba2 SSM update runs the fused HIP kernel instead of the ~55-op SSD graph (lablup/mlxcel#2067): decode on `gfx1151` went from 60.4 to 88.4 tok/s on granite-4.0-h-tiny-4bit and from 51.4 to 74.4 tok/s on Nemotron-3-Nano-30B-A3B-4bit, with model logits matching the graph and Metal's kernel at every decided position ([results](benchmark_results/rocm-ssm-update-kernel-gfx1151-2026-10-04.md)). Measured on granite and Nemotron-H; falcon-h1 and plamo-2 take the same gate but have not been run on ROCm. `MLXCEL_SSM_KERNEL=0` forces the graph. |
| GPU faults | Reported as errors (lablup/mlxcel#1804). A launch HIP rejects (an oversized block, no code object for the device) fails the evaluation that issued it and the device stays usable. An asynchronous fault (an out-of-bounds access) fails the evaluation waiting on it within about a second instead of hanging or returning NaN; the server fails that request and answers later ones with the same error, because after a queue fault the HIP runtime rejects every call for the rest of the process, so the process has to restart, and its shutdown may need a SIGKILL (HIP's teardown waits on callbacks the faulted queue never runs). A kernel that never finishes is not detected; `MLX_ROCM_GPU_WATCHDOG_SECS` (default off) fails any single host wait that outlives it. |
| Memory estimation on UMA hosts | Correct. Measured on the tested configuration: the ROCm allocator reports a nonzero cap (76.80 GiB of the 96 GiB carve-out), which the estimator reads before it would ever reach host RAM, so nothing that fits the carve-out is refused for that reason (lablup/mlxcel#1805). |
| Diagnostics | Report the AMD vendor, device name, `gfx` target and device memory, and no longer print a CUDA compute capability for it (lablup/mlxcel#1805). A binary whose compiled `gfx` list does not cover the device refuses to start rather than failing at the first kernel launch. |
| CPU device on a ROCm build (`MLXCEL_DEVICE=cpu`) | Runs, but slowly: on the tested host a Qwen3-0.6B-4bit decode step takes about two minutes, so it is a correctness reference and an escape hatch for a mismatched `gfx` build, not a serving mode. Before lablup/mlxcel#1807 every attention model aborted at the first token with `NYI`. BLAS work on this device (f32 matmul, convolution, linear algebra) runs on one OpenBLAS thread: with OpenBLAS's default thread count it intermittently wrote wrong output columns into the ROCm allocator's fine-grained memory (lablup/mlxcel#2072, `patches-rocm/LOCAL_FIXES.md` item 27). |
| `mlxcel-server` chat completions | Work for dense and affine MoE checkpoints, streaming and non-streaming; verified with `scripts/server_chat_smoke.sh`. |
| Audio (speech to text, text to speech) | Works. The FFT primitive runs on hipFFT; plans are cached up to `MLX_ROCM_FFT_CACHE_SIZE` (default 128, as on CUDA; lablup/mlxcel#1825, #1876); a value that is not a positive integer is ignored with a warning (#2051). |
| Wave64 devices (CDNA: MI200 `gfx90a`, MI300 `gfx942`) | Not run. The fused HIP kernels that reduce across lanes (BitLinear, the two fused MoE kernels and the squared-ReLU one, the Mamba2 SSM update, the Mamba1 scan, the add3 LayerNorm, the fused residual-add RMSNorm, and the paged-attention v1 decode and v2 partial) have only run on 32-lane wavefronts, and their compile-time wavefront guards do nothing with AMD clang 23. So mlxcel selects them only when the device reports a 32-lane wavefront (the hardware value, which `MLX_ROCM_FORCE_WARP_SIZE` does not change); on a 64-lane device those paths use their MLX graph fallbacks, and stderr says so once per process (lablup/mlxcel#2147). The kernels with no cross-lane operation (xIELU, RoPE + KV append, the paged-attention merge, both samplers) are selected at any width. BitLinear has no graph fallback, so BitNet checkpoints are refused at load on a wave64 device. |
| Windows, multiple GPUs, distributed inference | Not supported. |

Decode throughput measured on the tested configuration, for orientation only
(greedy, short prompt): Qwen3-0.6B-4bit about 250 tok/s, Qwen3-30B-A3B-4bit
about 55 tok/s (measured with `MLXCEL_FUSED_MOE=0`, before lablup/mlxcel#1803
made that unnecessary; not re-measured since), gpt-oss-20b-MXFP4-Q4 about 8 tok/s
(generic gather kernel; 8.07 tok/s in the baseline page linked above, up from the
3.6 tok/s lablup/mlxcel#1818 measured).

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
| `MLXCEL_DEVICE` | Runtime device hint (`gpu`, `metal`, or `cpu`) | `gpu` |
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

For local and agent development, use `[profile.test-fast]` instead (no cross-crate LTO,
`codegen-units = 16`, incremental compilation, `strip = false`; still
`opt-level = 3` so MLX-heavy numerics stay representative):

```bash
# CPU / Metal / Accelerate (macOS adds metal,accelerate automatically)
make test-fast

# Linux / CUDA
make test-fast-cuda

# Linux / ROCm (experimental): no make target yet
cargo test --profile test-fast --features rocm -- --test-threads=1

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
covered is a defect that reproduces only under release LTO or `codegen-units = 1`.
Reach for `cargo test --release --features metal,accelerate` by hand when you
are chasing one of those.

### CPU-only link check

A Linux build with no GPU feature compiles MLX without any GPU backend, so MLX's GPU-side helpers (for example `copy_gpu_inplace`) do not exist in it. Bridge C++ under `src/lib/mlx-cpp/turbo/` and `src/lib/mlxcel-core/cpp/` that calls one must sit behind `MLXCEL_BRIDGE_GPU_BACKEND`, which `mlxcel-core/build.rs` defines exactly when MLX builds Metal, CUDA or ROCm. A missed guard is a link error that only this configuration shows: `cargo check` does not link, and every GB10 CI job passes `--features cuda`. Check it locally with a separate target directory so it does not evict a CUDA cache:

```bash
CARGO_TARGET_DIR=target/cpu cargo test -p mlxcel-core --profile test-fast --lib --no-run
```

Measured on GB10 (2026-10, lablup/mlxcel#2108): about 2.5 minutes cold, including the CPU-only MLX tree, and about 6 seconds to rebuild and relink after a one-file change in the bridge or in `mlxcel-core`. CI runs the same command as the `CPU-only link` job in `.github/workflows/ci.yml`, on the GB10 runner with its own persistent target directory (`$HOME/.cargo-target/mlxcel-cpu-link-ci`), whenever a pull request changes Rust sources, manifests, or the bridge C++ under `src/lib/mlx-cpp/` or `src/lib/mlxcel-core/cpp/` (lablup/mlxcel#2111).

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

**ROCm: `Could NOT find hip` (or `rocblas`, `rocwmma`, ...) during the MLX
configure**: install the ROCm development packages that ship those CMake
configs, or point `ROCM_PATH` at the ROCm root that has them under `lib/cmake`.

**ROCm: `openssl-sys` fails because `pkg-config` could not be found**: install
`pkg-config` and `libssl-dev`, or set `OPENSSL_LIB_DIR` and
`OPENSSL_INCLUDE_DIR` for the build.

**ROCm: `rocminfo reported no GPU agent`**: set `MLX_ROCM_ARCHITECTURES`, and
check that the user is in the `video` and `render` groups so `rocminfo` can open
`/dev/kfd`.

**ROCm: `[cuda_kernel] No CUDA back-end.` followed by an abort**: a fused path
without a ROCm fallback was reached. lablup/mlxcel#1803 routed the fused paths
that had a graph fallback, MoE included, so this should no longer happen on a
routed path; report the model and the traceback. A direct call to a fused
kernel entry point that has no ROCm port returns an error naming the kernel and
its fallback rather than aborting (lablup/mlxcel#1885); the Gumbel-max and
rejection samplers have HIP ports (lablup/mlxcel#2064).
