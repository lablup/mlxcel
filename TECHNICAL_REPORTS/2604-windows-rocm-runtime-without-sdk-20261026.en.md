# PR #2604: Run Windows ROCm builds without the SDK on PATH or ROCM_PATH

**Date**: 2026-10-26
**Status**: Complete on Windows; Linux ROCm not exercised (CI ROCm job parked)
**Languages**: Rust, C++, CMake, Markdown
**Risk Level**: Medium

## Executive Summary

PR #2604 (issue #2602) removes the two run-time requirements that #2601 left on Windows ROCm builds: the ROCm `bin` directory on `PATH` and `ROCM_PATH`/`HIP_PATH` set. The ROCm DLLs are now delay-loaded by full path from the SDK recorded at build time, the hiprtc JIT finds its headers through a build-time include directory, and the Windows integrated-GPU memory limit follows the device pool instead of OS RAM. `mlxcel generate` and `mlxcel-server` were validated from a plain `cmd.exe` with `PATH=C:\Windows\System32;C:\Windows`, no ROCm variables, a cold JIT cache, and decoy ROCm DLLs in the working directory.

The work also exposed a correctness bug in #2601 that nobody had noticed: the AMD display driver installs older `amdhip64_7.dll` and `amd_comgr.dll` in System32, and the #2601 binaries were running the driver's HIP runtime rather than the SDK's.

## 1. Problem Statement

After #2601 a Windows `--features rocm` binary imported five ROCm DLLs by name (`amdhip64_7`, `rocblas`, `libhipblaslt`, `hipfft`, `hiprtc0716`) and found its hiprtc headers only through `ROCM_HOME`/`ROCM_PATH`/`HIP_PATH`. That made the binary depend on a configured shell, and it left a DLL search-order hole: with SafeDllSearchMode, the working directory is searched before `PATH`, so a DLL planted next to a model file or in a download folder would be loaded instead of the SDK copy. #2599 had solved the same two problems for CUDA with delay-load and a baked include directory; this PR mirrors that design and closes the ROCm-specific gaps the CUDA design does not have to face.

## 2. Change Summary

| Area | Change |
|---|---|
| Link | `windows_rocm::select_delayload_dlls` and `emit_delayload_link_args` emit `/DELAYLOAD` for the five imported DLLs (excluding `hiprtc-builtins*` and `hipfftw`, which match a prefix but are not imported), link `delayimp`, and are called from both `mlxcel-core/build.rs` and the root `build.rs`. |
| Delay-load hook | New overlay `delayload.cpp`: on `dliNotePreLoadLibrary` for a ROCm name, loads it by full path from `MLXCEL_ROCM_BIN_DIR`; once per process calls `AddDllDirectory(<bin>)` and preloads `amd_comgr*.dll` and `hiprtc-builtins*.dll` by full path. |
| hiprtc headers | `jit_module.cpp` resolves the include directory through a pure `choose_rocm_include_dir()`: `ROCM_HOME`, `ROCM_PATH`, `HIP_PATH` (empty means unset), then the build-time `MLXCEL_ROCM_INCLUDE_DIR` if it exists, then `/opt/rocm`. |
| Memory | Under `_WIN32` the integrated-GPU allocator limit is `total * 0.8`, with no system-RAM term. Rust gains a `GlobalMemoryStatusEx` probe so Windows CPU-only estimates are not zero. |
| Build guards | A relative `ROCM_PATH`/`HIP_PATH` and an SDK `bin` without `amdhip64_*.dll` now fail the build instead of producing a binary that would search the working directory. |
| Docs | Run-time limits removed; DLL resolution described; the SDK directory must be writable only by administrators; `ROCM_PATH`/`HIP_PATH` are build-time only on Windows. `LOCAL_FIXES.md` item 57. |

## 3. Technical Decisions

### 3.1 Preload the run-time-loaded DLLs instead of restricting the process search path

The five imported DLLs are covered by the delay-load hook, but hiprtc and the HIP runtime also call `LoadLibrary` themselves for `amd_comgr` and the hiprtc builtins, and the hook never sees those calls. The obvious fix, `SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`, changes the search order for the whole process, including the display driver's own loads, which this project does not control and cannot test across driver versions. The PR instead loads those DLLs first, by full path, from the recorded `bin`. Windows checks the list of already-loaded modules before any directory, so the later plain-name loads bind to the SDK copies. The cost is that a DLL named by plain name that the hook does not know about still follows the normal search order. The security review scanned the SDK DLLs for embedded names: three optional ones (`hsa-amd-aqlprofile64`, `hsa-ext-finalize64`, `dtif64a`) exist in neither the SDK nor System32, are not loaded by generation or serving, and are documented as the remaining exposure.

### 3.2 Raise the helper's exception instead of returning null

The issue asked the hook to return null when a DLL is missing from the recorded `bin`, so the loader would report its normal error. The developer found that `__delayLoadHelper2` treats null as "no opinion" and retries with a plain `LoadLibraryExA(name, NULL, 0)`, which searches the working directory and `PATH`: the exact path this PR closes. The hook therefore prints the expected path and raises the same `ERROR_MOD_NOT_FOUND` exception the helper would raise after a failed retry, with the helper's own `DelayLoadInfo`. The implementation review checked this against the helper's source behavior and confirmed it.

### 3.3 Cap the Windows APU limit at the device pool

The Linux formula, `max(total * 0.8, sys_ram * 0.8)`, assumes the APU pool draws from system RAM. On Windows WDDM the large pool is the BIOS carve-out reported as device memory. A probe allocated 1 GiB `hipMalloc` chunks until failure: it stopped at 105 GiB with 107.87 GiB reported and 31.65 GiB of OS RAM. Since allocations do not exceed the reported total, the Windows limit is `total * 0.8` (86.29 GiB on this host), which `--estimate-memory` now reports.

### 3.4 Fail closed in the build

Two review findings were cases where a misconfiguration would silently produce a less safe binary rather than a failed build. A relative `ROCM_PATH` would be baked into `MLXCEL_ROCM_BIN_DIR` and resolved against the run-time working directory; an unreadable or wrong `bin` would yield no `/DELAYLOAD` arguments and fall back to by-name imports. Both now panic with the offending value. The CUDA helper keeps its existing shape.

## 4. Latent Defect Found: the System32 HIP runtime

The Adrenalin driver ships `amdhip64_7.dll` and `amd_comgr.dll` in System32, older than ROCm 10.1. System32 comes before `PATH` in the search order, so the #2601 binaries, run with the SDK on `PATH`, loaded the driver's HIP runtime and only took the SDK's math libraries from `PATH`. They worked because the driver's runtime and comgr matched each other. A probe that loaded the SDK's `amdhip64_7.dll` beside the driver's `amd_comgr.dll` found zero devices. The full-path delay-load plus the comgr preload makes the pairing deterministic: both come from the SDK.

## 5. Validation

On Windows 11, Radeon 8060S (`gfx1151`), ROCm 10.1.0, VS 18:

- `llvm-readobj --coff-imports` on both binaries: the five ROCm DLLs are `DelayImport`; only `libopenblas.dll` remains an eager non-system import.
- Clean `cmd.exe` (`PATH` = System32 only, no ROCm variables, fresh `MLX_HSACO_CACHE_DIR`): Qwen3-0.6B-4bit about 117 to 121 tok/s and Qwen3-1.7B-bf16 about 28.6 to 30.4 tok/s, correct text, including the first JIT compile that the #2601 binary fails without `ROCM_PATH`.
- Decoys in the working directory (renamed `version.dll` as `amdhip64_7`, `amd_comgr`, `hiprtc0716`, `hiprtc-builtins0716`, `rocblas`, and in a second run the three optional names above plus `amd_comgr_3` and `atidxx64`): none loaded; every ROCm module came from `C:\rocm\10.1.0\bin`. The #2601 binary exits at start with the `rocblas.dll` decoy.
- `mlxcel-server` from the same environment: `/v1/chat/completions` (streaming and non-streaming) and `/v1/messages` return "The capital of France is **Paris**." with the expected stop reasons.
- `cargo test -p mlxcel-mlx-pin --lib` (49), the `memory_estimate` tests (64), fmt, clippy with `--features rocm`, overlay `drift` and `records`.

## 6. Known Gaps

- **Linux ROCm** not built or run (no host, CI job parked, #1811). Every change is behind a Windows gate; reviews confirmed the Linux `-I` string and allocator formula are unchanged.
- **Windows CUDA** not rebuilt (no NVIDIA GPU); its root `build.rs` call is unchanged.
- **Missing-DLL error path** not exercised (it needs the SDK moved away).
- **SDK directory permissions**: a directory under the drive root inherits write access for every authenticated user. The docs now require an admin-only directory and give an `icacls` command, but the build does not check it.
- **Run-time environment variables** still override the build-time include directory, matching Linux and CUDA; they need the same protection as the SDK directory.
- **Optional plain-name DLLs** listed in 3.1 would follow the normal search order if a profiling path ever loads them.
- **Pre-existing**: the JIT kernel cache defaults to the shared temp directory (`/tmp` on Linux), where another user could create it first.

## 7. Learning Points

- On Windows, a GPU driver can put an older copy of a vendor runtime in System32, and System32 outranks `PATH`. "It runs with the SDK on PATH" does not mean it runs the SDK.
- Delay-load hooks cover only the DLLs the binary imports. Libraries that call `LoadLibrary` themselves need a separate answer, and preloading by full path is a narrower one than changing the process default search order.
- Read the delay-load helper's fallback behavior before relying on "return null": the fallback is a plain search.
