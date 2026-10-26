# PR #2607: Load the Windows ROCm runtime from a relocatable directory

**Date**: 2026-10-26
**Status**: Complete on Windows (gfx1151, ROCm 10.1.0); other GPUs and drivers not exercised
**Languages**: C++, Python, Markdown
**Risk Level**: Medium

## Executive Summary

PR #2607 (issue #2606) lets a Windows ROCm build of mlxcel run on a machine with no ROCm SDK. The delay-load hook from #2604 used to load every ROCm DLL from the SDK path recorded at build time. It now chooses one runtime directory per process: `MLXCEL_ROCM_RUNTIME_DIR`, else the executable's own directory, else the build-time SDK `bin`. A new script, `scripts/package_windows_rocm_runtime.py`, builds the runtime bundle that goes in that directory: the DLL closure, the HIP JIT headers and the arch-filtered kernel data, 76 MiB zipped for gfx1151. This follows the llama.cpp model of a separately distributed runtime, but with mlxcel's own bundle, because llama.cpp's ROCm zip lacks rocBLAS, hipBLASLt and hipRTC.

With the SDK folder renamed away, a fresh folder holding the bundle plus `mlxcel.exe` ran `generate` (cold JIT) and all three server endpoints on the GPU. Every ROCm module came from the bundle; none came from System32, the working directory or the missing SDK.

## 1. Problem Statement

After #2604 a Windows ROCm binary no longer needed the SDK on `PATH`, but it still needed the SDK at the exact path recorded at build time. That ties a binary to the build machine's layout. Distributing mlxcel to ordinary Windows users needs the binary to find a runtime that ships with it. It must do so without reopening the DLL search-order hole #2604 closed (working directory and `PATH`), and without mixing DLLs from two ROCm installations. #2604 found that mixing the SDK's HIP runtime with the display driver's System32 `amd_comgr.dll` reports zero devices.

## 2. Change Summary

| Area | Change |
|---|---|
| Runtime directory | New header-only `runtime_dir.h` with a pure `choose_rocm_runtime_dir()`. `delayload.cpp` calls it once and loads every delay-loaded ROCm DLL, the `amd_comgr*`/`hiprtc-builtins*` preloads and the `AddDllDirectory` entry from that one directory. It prints `[mlx-rocm] ROCm runtime: <dir> (<source>)`. |
| Override semantics | When `MLXCEL_ROCM_RUNTIME_DIR` is set it is the only candidate. A relative path, or a folder without the HIP runtime, is an error rather than a fall-back. |
| Candidate test | A folder qualifies when it holds the HIP runtime DLL named in the binary's own delay-import table (`amdhip64_7.dll` today). The binary's directory comes from `GetModuleFileNameW(&__ImageBase)`. |
| Preload presence check | The chosen folder must contain an `amd_comgr*.dll` and a `hiprtc-builtins*.dll`, otherwise startup fails naming the folder and pattern. A DLL missing at load time fails with its expected path. |
| JIT headers | `choose_rocm_include_dir()` gains a candidate: `<runtime dir>\include` or `<runtime dir>\..\include` when it holds `hip/hip_runtime.h`. It sits after `ROCM_HOME`/`ROCM_PATH`/`HIP_PATH` and before the baked SDK include. |
| Bundle script | `package_windows_rocm_runtime.py` (stdlib only): `--rocm-path`, repeatable `--arch`, `--out` (zip) or `--stage-dir`, `--add-binary`, `--rocm-version`. It fails on any missing file and writes deterministic output. |
| Tests | Host C++ test `scripts/mlxcelverse/rocm_runtime_dir_test.{cpp,sh}` (in `make verify-rocm-overlay`), covering order, override errors, the preload check and include order. `tests/test_package_windows_rocm_runtime.py` has 18 tests. |
| Docs | `docs/installation.md` "Runtime directory and bundle", plus a `MLXCEL_ROCM_RUNTIME_DIR` row in `docs/environment-variables.md`. Install rules cover a separate folder from llama.cpp's runtime, admin-only folders, a path length of at most 130 characters, and the `HIP_PATH` precedence note. `LOCAL_FIXES.md` item 58. |

## 3. Technical Decisions

### 3.1 One directory per process, no fall-back

The hook could try each candidate DLL by DLL. It does not: the choice is made once, and a DLL missing from the chosen folder is an error. Falling back per DLL would produce exactly the cross-installation pairing #2604 showed to fail silently (zero devices, then CPU). For the same reason a set-but-wrong `MLXCEL_ROCM_RUNTIME_DIR` is an error. A typo must not quietly run the SDK or a bundle beside the exe instead.

### 3.2 The executable directory is a candidate

Windows already searches the application directory first for implicit imports, so trusting it adds no new planting surface. It is also the natural place for an unzipped app. The candidate test uses the binary's own delay-import name, so a future build against a different HIP major version checks for the right file.

### 3.3 Keep the SDK layout instead of a flat folder

The issue proposed a flat folder beside `mlxcel.exe`. Testing with `ROCM_KPACK_DEBUG=1` showed rocBLAS, hipBLASLt and rocFFT load device code from `<dll dir>\..\.kpack\<lib>_<arch>.kpack`, so a flat folder breaks every GEMM. The bundle therefore keeps `bin\`, `.kpack\`, `include\` and `licenses\`. The exes go into `bin\`, or the override names that `bin\`. This is also why the JIT looks at `<runtime dir>\..\include`.

### 3.4 Require the plain-name DLLs to be present

The orchestrator's re-check removed `amd_comgr.dll` from a bundle. The old preload silently loaded nothing, the HIP runtime's own `LoadLibrary("amd_comgr.dll")` resolved to `C:\Windows\System32\amd_comgr.dll`, and the process fell back to CPU and crashed with 0xC0000409. The hook now lists the chosen folder before loading anything. If either plain-name family is missing it raises the delay-load helper's `ERROR_MOD_NOT_FOUND` exception with a message, as #2604 does for a missing imported DLL. `hiprtc-builtins` is required too, because a missing one would be searched for in the working directory and `PATH`.

### 3.5 Print paths as UTF-8

The MSVC CRT stops a `%ls` conversion at the first non-ASCII character in the default C locale, so a bundle under a Korean user name printed a truncated path. Paths are converted with `WideCharToMultiByte(CP_UTF8)` and printed with `%s`. Directory listing uses a non-throwing conversion, so no exception can escape the delay-load helper.

## 4. Validation

All runs on Windows 11, Radeon 8060S (`gfx1151`), ROCm 10.1.0, final build at eb1c47af. `C:\rocm\10.1.0` was renamed away, `PATH` was System32 only, no ROCm variables were set, the JIT cache started empty, and the working directory held `version.dll` copies named like 14 ROCm DLLs.

- **Bundle staged by the committed script** with `--add-binary`: `generate` with Qwen3-1.7B-bf16 answered "Paris". The module list showed all 10 ROCm DLLs from the bundle's `bin\`.
- **Server** from the same folder: chat, streaming (`[DONE]`) and `/v1/messages` all returned "Paris".
- **`MLXCEL_ROCM_RUNTIME_DIR`** naming the bundle, with the exe in the build directory: "Paris" at 70 tok/s.
- **Bundles missing `amd_comgr.dll`, `hiprtc-builtins*.dll` or `rocblas.dll`:** each exits 0xC06D007E naming the file and folder, and nothing loads from System32.
- **Runtime folder reached through a junction named `정규`:** the path printed intact and cold-JIT `generate` worked, which also shows hiprtc accepts a UTF-8 non-ASCII `-I`.
- **No bundle and no SDK:** the error names both candidates. A developer build with the SDK present still uses `C:\rocm\10.1.0\bin`.
- **Checks:** `cargo test -p mlxcel-mlx-pin --lib` (49), fmt, clippy with `--features rocm`, overlay drift and records, `check_env_var_docs.py`, the host C++ test and the Python packaging tests all pass.
- **Informational llama.cpp test:** llama.cpp b11889's ROCm 10.0 `amdhip64_7.dll`, `amd_comgr.dll` and `rocm_kpack.dll` with the 10.1 math libraries ran both test models correctly. That is one host, so the docs still forbid sharing a folder.

## 5. Known Gaps

- **`HIP_PATH` precedence.** The issue specified that `ROCM_HOME`/`ROCM_PATH`/`HIP_PATH` come before the bundle's headers, matching Linux and CUDA. The AMD HIP SDK installer sets `HIP_PATH` system-wide, so on such a machine the bundle compiles JIT kernels against that SDK's headers. This is documented, not changed in code; preferring the bundle's headers when the runtime came from a bundle is a candidate follow-up.
- **Not exercised:**
  - other GPUs, drivers and architectures (the AOT arch list is still gfx1151 only);
  - Windows long-path mode;
  - FFT (hipFFT and rocFFT ship but no FFT ran);
  - Linux, which these Windows-only code paths should not affect.
- **The host C++ test is not run by CI.** Only `make verify-rocm-overlay` runs it, and on this host it needs `CXX` set to ROCm's clang++.
- **Missing licenses.** The SDK tarball has no license file for the HIP runtime, hipRTC or rocm_kpack, so the bundle cannot ship them.
- **Path length.** A runtime path over 130 characters breaks rocBLAS/hipBLASLt kernel loading when long paths are off. The hook warns about it; it does not fix it.

## 6. Learning Points

- Vendor math libraries can locate data relative to their own DLL (`..\.kpack`), which constrains bundle layout more than the import table suggests. Trace real file opens before designing a flat layout.
- A preload that silently does nothing turns into a fall-through to the system search order. "Load X by full path" needs "and fail if X is not there".
- On MSVC, `%ls` with non-ASCII text silently truncates in the default C locale. Convert paths to UTF-8 before printing.
