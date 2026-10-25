# PR #2599: Build and run mlxcel natively on Windows with CUDA

**Date**: 2026-10-25
**Status**: Complete locally (Linux/macOS CI pending at this snapshot)
**Languages**: Rust, C++, CMake, Markdown
**Risk Level**: Medium

## Executive Summary

PR #2599 closes the issue #58 spike with a working native Windows build: `cargo build --release --features cuda` under MSVC produces `mlxcel.exe` and `mlxcel-server.exe`, and both run single-GPU inference with no CUDA environment variables set and no toolkit directory on `PATH`. The only NVIDIA Windows host available was ARM64 (NVIDIA RTX Spark N1X, `sm_121`), so the spike ran on `aarch64-pc-windows-msvc` with the issue owner's agreement; the build glue resolves `x64` and `arm64` toolkit layouts alike, but no x86_64 GPU host has run it.

The Phase 1 gate passed without patching MLX: the pinned upstream tree plus mlxcel's CPU and CUDA overlays compiled under MSVC 14.51 and nvcc 13.4 once CMake was told where the arm64 libraries live. The remaining work was in what cargo does that CMake normally does for MLX consumers (delay-load link options, the delay-load hook object, OpenBLAS, static-library defines), in runtime DLL resolution, and in a handful of latent defects that only MSVC's linker reports.

## 1. Problem Statement

Before this PR, Windows was undocumented and the crate did not compile there. The issue split the blockers into three layers:

- **Upstream MLX**: Windows x86_64 CUDA is CI-built upstream but never executed, and Windows on Arm CUDA had no upstream coverage at all.
- **mlxcel build glue**: `build.rs` assumed Linux (`lib64`, `/usr/local/cuda/bin/nvcc`, GCC flags, system OpenBLAS), and the static-link delay-load that MLX attaches to its CMake target as PUBLIC options never reaches a cargo link.
- **Rust surface**: Unix APIs (`tokio::net::UnixListener`, `libc` errno constants, fd-anchored file operations) and a distributed stack that the issue scoped out for Windows.

The acceptance criterion was a native `--features cuda` build that passes the Phase 5 smoke tests (generate on the GPU, serve chat completions) on an NVIDIA Windows host, single node.

## 2. Change Summary

| Area | Change |
|---|---|
| Build glue | New `build_support/windows_cuda.rs`, shared by `mlxcel-core/build.rs` and the root `build.rs`: toolkit and cuDNN layout (`lib\<arch>`, `bin\<arch>`), Ninja generator, forward-slash CMake paths, `/DELAYLOAD` for the CUDA and cuDNN DLLs, `/INCLUDE:__pfnDliNotifyHook2`. Unit tests run in `mlxcel-mlx-pin`. |
| C++ bridge | MSVC branch: `/std:c++20`, `MLX_STATIC`, `/bigobj`, `/Zc:preprocessor`, `/fp:fast`, and `MLXCEL_CXX_MARCH` passed as `/arch:<value>`. |
| CPU backend | Links the OpenBLAS import library MLX's CMake downloads and copies `openblas.dll` next to the built executables. |
| CUDA overlays | New `delayload.cpp` overlay; `jit_module.cpp` header lookup; `matmul.cpp` gains `BlockMaskedMM::eval_gpu`. |
| Dependencies | `tokenizers` without its default `esaxx_fast` feature; dead C++ `split()` removed from the bridge. |
| Single node | `src/distributed/platform.rs` and `ServerDistributedFlags` refuse distributed flags in `generate`, `serve` and `mlxcel-server` before model resolution. |
| Unix gating | `UnixListener` path, Unix-only helpers and tests are `cfg`-gated; Windows listeners no longer set `SO_REUSEADDR`; download tempfiles use exclusive create. |
| Docs | Windows section in `docs/installation.md`, README, `docs/distributed.md`, `docs/environment-variables.md`. |

## 3. Technical Decisions

### 3.1 Replicate the delay-load link step in cargo instead of building `mlx.dll`

MLX's CUDA backend links Windows binaries with `/DELAYLOAD` for every CUDA and cuDNN DLL and compiles `delayload.cpp` into the consumer; with a static `mlx`, CMake attaches both as PUBLIC usage requirements. Cargo links `mlx.lib` directly, so neither reaches the final link, and without them the executable imports the DLLs eagerly and fails to start unless the toolkit is on `PATH`.

The PR keeps the static library and reproduces the link step. A dependency's `rustc-link-arg` does not reach the binary that links it, so the `/DELAYLOAD` list is emitted by both build scripts from one shared module, the same pattern `rocm_rpath.rs` already uses for the ROCm rpath. The list is restricted to the DLLs behind linked import libraries; `cudart64_` and `cublas64_` stay listed although nothing imports them at this pin (nvcc links the static cudart and MLX calls only cuBLASLt), so a later pin that imports them still delay-loads them.

The hook object lives in `mlx.lib`, but nothing references `__pfnDliNotifyHook2`, so the linker would take the null default from `delayimp.lib`. `/INCLUDE` forces the symbol, and `mlx` is linked ahead of `delayimp`. The first version also compiled `delayload.cpp` into the bridge; the implementation review found the duplicate definition in `mlx.lib`, and it was removed.

Building MLX as `mlx.dll` was the alternative the issue offered. It was not taken because it adds a DLL to every deployment, changes the link model the other platforms share, and does not remove the runtime DLL-resolution work below.

### 3.2 Runtime DLL resolution belongs in the hook, not in `PATH`

`dumpbin` shows the final executable delay-loads `cublasLt64_13`, `cufft64_12`, `cusolver64_12`, `cudnn64_9` and `nvrtc64_130_0`, and imports no `nvcuda.dll` or `cudart` directly. Three runtime failures appeared in sequence, each with a distinct cause:

1. `0xC06D007E` (delay-load module not found): the bin directory reached CMake with backslashes and became a C++ string literal, so `\t`, `\b` and `\a` turned into control characters. All paths handed to CMake are now written with forward slashes (`windows_cuda::cmake_path`, with a test using exactly that path).
2. `failed to open nvrtc-builtins64_134.dll`: NVRTC opens its builtins DLL by plain name, and `AddDllDirectory` only affects `LoadLibraryEx` calls that request user directories.
3. `Could not locate cudnn_engines_runtime_compiled64_9.dll`, then `cublasLt64_13.dll` with a fail-fast `0xC0000409`: cuDNN opens its sub-libraries and cuBLAS the same way.

The `delayload.cpp` overlay therefore preloads, by full path, the DLLs NVRTC and cuDNN open by name (cuBLAS and NVRTC before the cuDNN sub-libraries), so their later by-name loads find modules already in the process. Each DLL is loaded with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`, which resolves its imports from its own directory and never from the current directory or `PATH`; the first version used `LOAD_WITH_ALTERED_SEARCH_PATH`, which the security review flagged as still reaching both.

### 3.3 Leave MLX's CMake untouched

`FindCUDAToolkit` sets Windows library search suffixes only when the host processor is `AMD64`, MLX's `FindCUDNN` searches `lib/x64`, and MLX defaults `MLX_CUDA_BIN_DIR` to `bin/x64`. All three are configurable from outside (`CMAKE_LIBRARY_PATH`, `CUDNN_LIBRARY_PATH`, `CUDNN_INCLUDE_PATH`, `MLX_CUDA_BIN_DIR`), so `build.rs` passes them and no CMake overlay was added. NVRTC header lookup was the one place that needed code: the overlay's `default_cuda_toolkit_path()` threw from `directory_iterator` when the installer directory did not exist, and on Windows it now falls back to the build toolkit's `include\` (`MLXCEL_CUDA_INCLUDE_DIR`), placed after `CUDA_HOME` / `CUDA_PATH`. The fallback is Windows-only so Linux keeps `/usr/local/cuda`.

### 3.4 Refuse distributed modes at startup rather than compiling them out

`src/distributed/` compiles on Windows (its only Unix API use is the `target_os`-gated RDMA probe), so a `cfg` gate would add complexity without removing a compile error. What is missing is validation of multi-host transport on Windows. The PR refuses every distributed flag (`--tp-size` above 1, PP, cluster and disaggregated flags) with an error naming the flag. Both server CLIs check before resolving the model reference, so a repo id is not downloaded only to be refused, and `start_server` checks again for programmatic callers.

## 4. Latent Defects MSVC Exposed

GNU ld with `--gc-sections` discards unreferenced sections before reporting undefined symbols inside them; MSVC reports first. Three defects that predate this PR surfaced that way:

- The CUDA `matmul.cpp` overlay was based on upstream `9a795735` and lacked `BlockMaskedMM::eval_gpu`, which upstream added later. The vtable in `ops.cpp` referenced it on every CUDA build; Linux linked only because the vtable was unused. The function is now taken verbatim from the pin; the overlay's declared base stays `9a795735` because the rest of the file was not rebased.
- A C++ `split()` returning `rust::Vec<std::unique_ptr<MlxArray>>` had no cxx declaration since the initial release, so its `rust::Vec` methods were never instantiated.
- `tokenizers`' default `esaxx_fast` compiles esaxx C++ with the static CRT (`/MT`), which cannot link with the `/MD` objects of sentencepiece and MLX. It accelerates only Unigram training, which mlxcel does not do.

## 5. Validation

On the RTX Spark N1X (driver 616.62, CUDA 13.4.3, cuDNN 9.27, VS 2026), with only `System32` on `PATH`, no `CUDA_PATH`, and an empty JIT cache:

| Check | Result |
|---|---|
| gemma-4-e4b-it-qat-4bit | Correct answer; 6.33 GB resident; about 16 tok/s warm |
| gemma-4-26b-a4b-it-4bit (MoE) | Correct answer; 14.29 GB resident; 10.07 tok/s including first-run JIT |
| qwen3_5 27B 4-bit (exercises cuDNN) | Correct answer; 14.10 GB resident; 4.3 to 4.5 tok/s warm |
| `mlxcel-server` | `/health`, `/v1/models`, non-streaming (`finish_reason: stop`) and streaming chat (`[DONE]`) |
| Distributed refusal | `generate`, `serve`, `mlxcel-server` refuse in 0.2 s with a nonexistent repo id |
| Static checks | Windows clippy `--lib --tests --bins -D warnings` and fmt clean; 17 targeted lib tests and 7 `windows_cuda` unit tests pass |

Linux, ROCm and macOS builds were checked by code reading in review (the shared Linux link path keeps the same library list and order; every new CMake and overlay branch is behind `WIN32` / `_WIN32`) and are otherwise left to CI.

## 6. Known Gaps

- No x86_64 Windows GPU run. The `x64` layout follows the CUDA 13 installer (`bin\x64`), but it is untested.
- CUDA 12.x is not handled: its DLLs live in `bin\`. The cuDNN installer's `lib\13.x\x64` layout is not searched either; the redistributable layout is required.
- The executables depend on the build machine's toolkit directory, and `openblas.dll` is copied only into the cargo profile directories. Packaging and a Windows CI job are Phase 6 work.
- Qwen 3.5 MTP speculative decoding is refused on CUDA by existing design (Metal-only exactness kernel), on Linux as well.
- Decode throughput was not investigated. 27B dense at about 4.4 tok/s reads roughly 63 GB/s of weights per second, well below the host's memory bandwidth.
- CPU-only generation on Windows runs MLX's scalar quantized loops; a 24-token run did not finish in 30 minutes.

## 7. Learning Points

- A static C++ library consumed by cargo loses every CMake PUBLIC usage requirement: link options, interface sources and compile definitions (`MLX_STATIC` here). Each has to be reproduced explicitly, and link options must be emitted from the build script of the crate that owns the final binary.
- On Windows, `AddDllDirectory` does not affect plain `LoadLibrary` calls made inside third-party DLLs. Preloading by full path is the dependable way to satisfy a library that opens its dependencies by name.
- Strings that cross from Rust into CMake into a C++ literal need a separator that survives every layer; forward slashes do.
- MSYS reports some Windows crash codes as exit 127; `cmd` and `%ERRORLEVEL%` give the actual NTSTATUS (`0xC06D007E`, `0xC0000409`).
