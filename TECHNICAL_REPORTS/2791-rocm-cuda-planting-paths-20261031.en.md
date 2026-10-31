# PR #2791: Close the ROCm and CUDA DLL and kernel-cache planting paths

**Date**: 2026-10-31 (generated before merge)
**Status**: In review
**Languages**: C++, Rust, Shell
**Risk Level**: Medium (changes DLL resolution in every Windows binary and the ROCm cache location; the ROCm paths ran only as host tests)

## Executive Summary

Two local code-planting paths remained after #2604 and #2607. On Windows, libraries that open DLLs by plain name searched the current directory, on ROCm and on CUDA. On ROCm, kernels were cached in a shared temp directory and loaded into the GPU from whatever was found there. This PR removes the current directory from the Windows DLL search, links binaries so static imports skip it too, moves the ROCm cache to a per-user location, and refuses a directory another user could write. On an RTX Spark N1X, decoys under ten plain names in the working directory never loaded, and the module sets match the pre-change binary. No ROCm binary ran.

## 1. Problem Statement

`amdhip64_7.dll` names three DLLs that no SDK or driver ships, and the CUDA libraries name others (`zlibwapi.dll`, `nvtileiras64.dll`, `cask_profile_interface.dll`). The preload hooks cover only the DLLs they know. A same-named file in the current directory would load into the process.

`hsaco_cache_dir()` defaulted to `<tmp>/mlx/<version>/hsaco/<arch>` with default permissions, and `read_cached_hsaco()` loaded what it found. On Linux another user could create that directory first and plant blobs.

## 2. Technical Decisions

### 2.1 `SetDllDirectoryW(L"")`, and a link flag for what it cannot reach

`SetDefaultDllDirectories` stays rejected because it changes every later load in the process, including the display driver's. `SetDllDirectoryW(L"")` removes only the current directory from the standard search. It runs first in each delay-load hook and, on CUDA, from a static initializer pinned with `#pragma init_seg(lib)`, because nvcc links the static cudart, whose first API call can open `nvcuda.dll` without reaching the hook.

Security review measured that the setting cannot help static imports: a decoy `openblas.dll` in the working directory loaded before `main`. `/DEPENDENTLOADFLAG:0xA00` restricts those imports to the application directory and System32. With it, that case exits `0xC0000135` (not found) where the old binary exits `0xC0000139`.

A consequence users can notice: a DLL kept only in the working directory no longer loads, and a Visual C++ runtime found only on `PATH` no longer works.

### 2.2 The cache follows the CUDA trust rules

The POSIX walk matches #2539: every component up to `/` must be owned by the user or root and not group- or other-writable, since the owner of a parent can swap a child. Review found the first version's repair of `umask 002` directories could chmod through an intermediate symlink, ran before the trust walk, and silently accepted world-writable directories. It now runs after the ancestors pass, walks the managed tail with `openat(O_NOFOLLOW)`, clears group write only, and refuses other-write.

### 2.3 Windows needed its own check

The first version had no ownership check on Windows, on the reasoning that `%TEMP%` is per-user. Review rated that HIGH: setting `MLXCEL_CACHE_DIR=D:\mlxcel` for model storage moved a GPU-loaded cache to a folder that usually inherits write access for Authenticated Users. `hsaco_cache_dir_win.h` now reads each component's owner and DACL below the drive root and refuses a foreign owner, a null DACL, or a write-class allow entry for another SID. The drive root is excluded because its default ACL lets Authenticated Users create folders on every drive. What it grants to children appears on the first component below it.

### 2.4 Logic in a header with no HIP dependency

The ROCm overlay does not compile on most hosts, and CI skips `ROCm build and generate`. Keeping the path and trust logic in a header let `overlay-host-tests` run the POSIX cases on Linux and let this host run the Windows cases with real ACLs.

## 3. Change Summary

- DLL search: both `delayload.cpp` overlays and the two `build.rs` files.
- ROCm cache: new `hsaco_cache_dir.h` and `hsaco_cache_dir_win.h`, called on the read and write paths in `jit_module.cpp`. Entries publish through a temporary file and rename, and an empty `.hsaco` or malformed `.txt` is a miss.
- Tests: `rocm_hsaco_cache_test`, `tests/windows_dll_search_cwd.rs`, and source guards in both runtime-dir scripts so CI fails if a call is removed.
- Docs: `docs/installation.md`, `docs/environment-variables.md`, `LOCAL_FIXES.md` item 71.

## 4. Results

| Check | Where | Result |
|---|---|---|
| Decoys under 10 plain names in the working directory; generate and server, developer build and bundle | N1X | none loaded |
| Module sets before the review fixes vs after | N1X | identical on all four runs |
| Pre-change vs PR binary, warm and cold cache, 8 runs | N1X | same modules except the executable name |
| `nvcuda.dll` | N1X | from System32 |
| Exe alone, decoy `openblas.dll` | N1X | `0xC0000135`, old binary `0xC0000139` |
| Write ACE for Users, Everyone or Authenticated Users on leaf or parent | N1X, `icacls` | refused, no blob loaded |
| World- or group-writable, other owner, symlinks | CI, Linux | refused, no blob loaded |

## 5. Trade-offs and Residuals

- **ROCm not run:** Linux `make verify-rocm`, the Windows ROCm module-list check, and the DLL test on a ROCm build need AMD hosts. `jit_module.cpp` has not been compiled for Linux ROCm, and no ROCm binary has linked with the new `#pragma comment(lib)` lines. CI's Windows job compiles the overlay only.
- **No real request path:** the DLL test calls `LoadLibraryW` directly; nothing drove the paths that request the absent ROCm DLLs.
- **Open by design:** `PATH`, a relative `PATH` entry, the executable's own directory, and `SearchPathW` all still resolve. `ffmpeg`, `ffprobe` and the IREE compiler do not inherit the setting, so they keep the current directory in their own search.
- **Drive root:** this host has one drive, so the inherited-ACE case was set with `icacls` under `%TEMP%`, not on a real `D:\`.
- **Entries on Windows:** no per-file ACL check; entries rely on the directory check.
- **Not fixed here:** the CUDA PTX cache has the same missing Windows check and the same `ptx_cache_dirs.rs` repair rule. An empty `MLXCEL_CACHE_DIR` means unset in the header but a relative path in `lang_analyzer/cache.rs`.

## 6. Learning Points

- `SetDllDirectoryW` acts too late for static imports. A fix placed in a hook needs the question of what loads before the hook, and measuring it with a decoy answered that.
- A check that only uses Unix rules leaves the Windows branch with an assumption (`%TEMP%` is private) that a separate variable can quietly invalidate.
- A module list that loses an entry needs an A/B with the old binary before anyone blames the change. `nvptxJitCompilera64.dll` went missing against an earlier run but was absent from the old binary too.
- A guard script's `grep -v | grep -q` under `pipefail` reads an early exit as not found, which failed CI on the first push of the guards.
