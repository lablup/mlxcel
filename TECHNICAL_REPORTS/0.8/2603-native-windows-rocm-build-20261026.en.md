# PR #2603: Build and run mlxcel natively on Windows with ROCm

**Date**: 2026-10-26
**Status**: Complete on Windows; Linux ROCm build not exercised (CI ROCm job parked)
**Languages**: Rust, C++, HIP, CMake, Markdown
**Risk Level**: Medium

## Executive Summary

PR #2603 (issue #2601) makes `cargo build --release --features rocm` work under MSVC on Windows x86_64 with a ROCm 10 SDK, and `mlxcel generate` produce correct text on the GPU. It was validated on an AMD Ryzen AI MAX+ 395 (Radeon 8060S, `gfx1151`) with the ROCm 10.1.0 TheRock tarball: `mlx-community/Qwen3-0.6B-4bit` at about 113 tok/s and `mlx-community/Qwen3-1.7B-bf16` at about 28.7 tok/s, greedy, with correct answers.

The build work follows the Windows CUDA path from #2599 closely. The part that is new is the memory model: Windows does not map device memory into the CPU address space even on an APU, so the backend's Linux assumption that an integrated GPU's allocations are host-dereferenceable does not hold. The PR routes every CPU access on Windows through the backend's existing host-shadow path and, in doing so, fixes two latent defects in that path that also affect Linux discrete GPUs.

Running still requires the ROCm `bin` directory on `PATH` and `ROCM_PATH` (or `HIP_PATH`) set; #2602 removes both.

## 1. Problem Statement

The `rocm` feature (#1801/#1802) was Linux-only: the build script panicked on any other target, the vendored ROCm backend CMake assumed GCC (libstdc++ include probing, `-fPIC`, `ar rcs libmlx_rocm_kernels.a`), and a few backend sources used POSIX calls. #2599 had already proven the MSVC toolchain for CUDA, so the open question was whether AMD's Windows stack fits the same mold, and what breaks at run time once it builds.

Two facts settled the toolchain question before any code changed. ROCm 10.1's clang defaults to `x86_64-pc-windows-msvc`, and its libraries (`rocblas.dll`, `hipblas.dll`, `libhipblaslt.dll`) import MSVCP140, VCRUNTIME140 and the UCRT, with no MinGW runtime. `libhipblaslt.dll.a`, despite its GNU-style name, is a COFF import library. So the whole build can stay MSVC ABI: MLX's host C++ under `cl`, HIP kernels under ROCm clang, `/MD` throughout.

## 2. Change Summary

| Area | Change |
|---|---|
| Feature gate | `rocm` accepted on Linux and on Windows x86_64 MSVC; still exclusive with `cuda` and `metal`. |
| SDK layout | New `build_support/windows_rocm.rs`: root from `ROCM_PATH` then `HIP_PATH` (empty means unset), `RocmLayout`, `parse_amdgpu_arch`; reuses `windows_cuda::cmake_path`, so `windows_cuda` now compiles for `any(cuda, rocm)`. Tests in `mlxcel-mlx-pin`. |
| CMake | Ninja generator, forward-slash `ROCM_PATH`/`CMAKE_PREFIX_PATH`/`CMAKE_HIP_COMPILER`, and `/EHsc` for MLX's C++ on every MSVC build. |
| Link | `mlx_rocm_kernels.lib`, ROCm import libraries, hipBLASLt's `.dll.a` copied to `OUT_DIR/hipblaslt.lib`; no rpath. |
| Overlay CMake | WIN32 branch: no libstdc++ probing, no `-fPIC`, `--rocm-path` and `--rocm-device-lib-path`, `-fms-runtime-lib=dll`, `.obj`, `lib.exe` archive. Linux branch byte-identical. |
| Overlay sources | `sysconf` to `GlobalMemoryStatusEx`, weak symbol, `_dup2`/`NUL`, `<iterator>`, `__builtin_clz` shim, LLP64-safe hiprtc `size_t`, `HIP_PATH` fallback and empty-value handling in `rocm_home()`. |
| Memory model | Host shadow for non-host-visible memory, dirty flag set by `raw_ptr()`, shadow lifetime tied to its buffer. |
| Duplicate symbol | `binary_two.hip` removed; `binary.hip` keeps `DivMod::eval_gpu`. |
| Docs | "Windows with AMD ROCm (experimental)" in `docs/installation.md`, environment-variable rows, feature comments. |

## 3. Technical Decisions

### 3.1 Keep device memory and shadow it, rather than allocate host-visible memory

The first build that linked crashed in `mlx::core::array::init` with an access violation in `memmove`: the allocator had handed out fine-grained device memory, which Linux's KFD maps into the process on an APU, and the CPU wrote into it directly. A probe on the host measured each allocation kind:

| Allocation | CPU write | GPU read | Largest single allocation |
|---|---|---|---|
| `hipMalloc`, fine-grained, uncached | faults | about 233 GB/s | 92 GiB (`hipMalloc`) |
| `hipHostMalloc` | works | about 233 GB/s | about 28 GiB |
| `hipMallocManaged` | works | about 233 GB/s up to OS RAM | beyond OS RAM: 2.9 GB/s CPU, GPU read did not finish in 6 minutes |

On this host the BIOS UMA carve-out (about 96 GB) is device memory and the OS sees about 32 GB. Host-visible allocations would have capped models at the OS-visible RAM, so the PR keeps `hipMalloc` and uses the shadow path the backend already had for discrete GPUs: `raw_ptr()` returns a pinned host copy, and `gpu_ptr` flushes it back before a kernel reads the buffer. A `device_memory_host_visible()` predicate (constexpr, false only on `_WIN32`) selects the path, so Linux compiles to the same code as before.

### 3.2 Mark the shadow dirty on every hand-out

The shadow path had never been exercised: nothing in the backend set `host_dirty = true`, so CPU writes through `raw_ptr()` were never flushed, on Windows or on Linux discrete GPUs with the opt-in async pool. Since MLX's `Buffer::raw_ptr()` does not say whether the caller will read or write, the PR marks the shadow dirty whenever it hands one out. That is correct but conservative: a CPU read followed by a kernel costs one whole-buffer copy back to the device. Separating reads from writes is a performance follow-up.

The implementation review then found the shadow outliving its buffer's identity: a scalar-pool slot returned to the pool leaked its pinned shadow, a buffer-cache hit gave the next owner the previous owner's dirty shadow, and freeing a shadow left the dirty flag set. The fix frees the shadow with its buffer and clears the flag on free and on reuse.

### 3.3 Copy the hipBLASLt import library instead of passing it as a link argument

`cargo:rustc-link-arg` from a dependency's build script does not reach the final binaries, while `cargo:rustc-link-lib` does. The build script copies `libhipblaslt.dll.a` to `OUT_DIR/hipblaslt.lib` and links it by name. The cost is a stale copy if the SDK is upgraded in place at the same path without the build script rerunning.

### 3.4 Remove the duplicate `DivMod::eval_gpu` by keeping the definition Linux used

`binary.hip` (two passes, FloorDivide then Remainder) and `binary_two.hip` (one fused kernel) both defined it. GNU ld pulls the first archive member that defines a symbol, and archive order follows `HIP_SOURCES`, which lists `binary.hip` first; had both objects been pulled, Linux would have failed with a multiple-definition error. MSVC `lib.exe` warned (LNK4006) and kept the first. Since `DivMod::eval_gpu` was the only non-template definition in `binary_two.hip` and nothing else referenced its templates, the file is removed. This was reasoned, not traced on a Linux link.

## 4. Latent Defects Found Along the Way

- **`/EHsc` missing for MLX on MSVC.** cmake-rs passes the cc crate's flags as `CMAKE_CXX_FLAGS`, which replaces CMake's MSVC defaults, so MLX compiled without unwind semantics (C4530) and destructors were skipped on exception propagation. This also affects the Windows CUDA build from #2599.
- **LP64 assumption in the hiprtc preamble.** `using size_t = unsigned long;` is 32-bit on Windows and conflicts with hiprtc's own typedef, failing every fused-kernel JIT. `decltype(sizeof(0))` matches the compiler on both ABIs.
- **Empty environment values.** `rocm_home()` accepted an empty `ROCM_PATH`, producing `-I/include`: on Windows, the root of the current drive, where any local user can plant a header that a JIT kernel includes. Empty values are now unset, matching the build script.
- **Device bitcode in a relocatable SDK.** `hipcc.exe` does not find the device libraries of a tarball SDK on its own; every `.hip` failed until `--rocm-path` and `--rocm-device-lib-path` were passed.

## 5. Validation

On Windows 11 with ROCm 10.1.0 and VS 18: a full `cargo build --release --features rocm` built both binaries with no C4530 or LNK4006 in the log, `amdgpu-arch` detection returned `gfx1151`, and both models generated correct text on the final head (about 113 and 28.7 tok/s). A JIT run with `ROCM_HOME` and `ROCM_PATH` empty and `HIP_PATH` set succeeded. `cargo test -p mlxcel-mlx-pin --lib` (44 tests, 11 new), fmt, clippy on `mlxcel-core` with `--features rocm`, and the overlay `drift` and `records` checks passed.

## 6. Known Gaps

- **Linux ROCm build not exercised.** CI's ROCm job is parked (no `rocm-gfx1151` runner, #1811). Both reviews checked that every Linux-reachable change is identical or one of the intended fixes (dirty flag, shadow lifetime, DivMod), but nothing compiled the Linux ROCm link.
- **Windows CUDA with `/EHsc`** needs a CUDA host.
- **Run-time environment**: ROCm `bin` on `PATH` and `ROCM_PATH`/`HIP_PATH` are still required, and the binaries import ROCm DLLs by name, so a DLL planted in the working directory would win. #2602 adds delay-load from a recorded directory and a recorded include path.
- **Shadow cost and accounting**: shadows are pinned memory outside `active_memory_`, capped near the OS-visible RAM, and every CPU read costs a flush. Weight loading is unaffected (it stages through the GPU).
- **Pre-existing**: the MoE workspace deleter in `naive_gemm.hip` deletes a `RocmBuffer` without freeing its shadow.

## 7. Learning Points

- On Windows, "integrated GPU" does not imply host-visible device memory; WDDM decides, and on a UMA carve-out machine the large pool is device memory only.
- A backend path written for hardware nobody tests (here, the discrete-GPU shadow) can be wrong in ways that only show up when a new platform starts depending on it.
- When a build has to cross ABIs, checking the import tables of the vendor DLLs (`llvm-readobj --coff-imports`) answers the toolchain question faster than trying builds.
