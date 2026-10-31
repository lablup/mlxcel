# PR #2799: Check the CUDA PTX cache directory on Windows

**Date**: 2026-10-31 (generated before merge)
**Status**: In review
**Languages**: C++, Rust, Shell
**Risk Level**: Medium (moves the default cache of `mlxcel` and `mlxcel-server` on Windows, and turns the cache off for directories that used to be accepted)

## Executive Summary

`jit_module.cpp` loads cached modules into the GPU, and its `_WIN32` branches created the cache directory without checking it. This PR applies the owner and DACL rules of #2791's ROCm cache to the CUDA cache, for `MLX_PTX_CACHE_DIR`, the persistent default and the temp fallback. It also refuses relative cache paths on all platforms, stops clearing other-write in the POSIX repair, and treats an empty `MLXCEL_CACHE_DIR` as unset. Measuring the starting point showed that the persistent default had never taken effect on Windows, which this PR also fixes.

## 1. Problem Statement

A user who set `MLXCEL_CACHE_DIR=D:\mlxcel` for model storage would, by the documented behavior, move the cache to a folder that usually inherits write access for Authenticated Users, and another local account could plant modules there.

The binary from `main` did something else. With `MLXCEL_CACHE_DIR` set, `ensure_persistent_ptx_cache` created `cuda-ptx/<commit>`, that directory stayed empty, and the entries went to `%TEMP%\mlx`. Rust's `set_var` calls `SetEnvironmentVariableW`, while `std::getenv` reads the C runtime's copy of the environment, taken at startup. On this host `~/.cache/mlxcel/cuda-ptx/*` held no files from any earlier run.

## 2. Technical Decisions

### 2.1 Read the process environment on Windows

The overlay now reads both variables with `GetEnvironmentVariableW`. Checking only the temp fallback would have left the check on a path that did not match the documentation, and #2797 names the persistent default as a location to check. Users see one cold compile after upgrading. `%TEMP%\mlx` is no longer used by `mlxcel` and `mlxcel-server`.

### 2.2 Same rules as ROCm, in a header of the overlay's own

`ptx_cache_dir_win.h` keeps the rules of `hsaco_cache_dir_win.h`: owner must be the user, SYSTEM or Administrators, a null DACL is refused, any write-class allow entry for another SID is refused, and the drive root is excluded because its default grant is the right to create subdirectories, not to delete or rename another account's. Overlays share no code, so this is a copy by design.

### 2.3 The `\\?\` prefix is removed for the walk and restored for the calls

`std::filesystem` parses `\\?\C:\x` with `\\?` as the root name, so the root is not the drive and `parent_path` runs past `C:\`. The walk uses the plain form, which is also what the notice prints. Every file system call adds the prefix back, so a path past `MAX_PATH` still works, and the returned path keeps it as upstream's code did. `lexically_normal` runs before the prefix goes on, because the prefix disables parsing of `/` and `..`, and `cache_root` joins `.cache/mlxcel` with a forward slash.

### 2.4 Relative paths stop in both layers

`ptx_cache_env_refusal` refuses a relative `MLX_PTX_CACHE_DIR`, or a relative `MLXCEL_CACHE_DIR` when the first is unset. `ensure_persistent_ptx_cache` also returns early for a relative root. Without that it would create `rel-root\cuda-ptx\<commit>` under the current directory and set `MLX_PTX_CACHE_DIR` to it, and the notice would name a variable the user never set.

### 2.5 Other-write stays for the overlay to refuse

A `umask` of 002 explains group write. It does not explain other write, and a world-writable directory may already hold someone else's modules, so clearing the bit would accept them. `make_private_dir` now clears group write only.

### 2.6 Testing with real ACLs

`tests/cuda_ptx_cache_trust_windows.rs` copies entries a first run compiled into the directory under test, so a run that trusted it would load them without compiling. `planted_entries_load_from_a_private_dir` is the control for that. `ptx_cache_env_refusal` also runs in `overlay-host-tests` on Linux, the only place CI executes the relative-path rule, and that script has source guards on the calls.

## 3. Change Summary

- Overlay: new `ptx_cache_dir.h` and `ptx_cache_dir_win.h`, called from `ptx_cache_dir()` and `device_cache_dir()` in `jit_module.cpp`.
- Rust: `cache_root` ignores an empty value, `persistent_ptx_cache_dir` returns `None` for a relative root, `make_private_dir` clears group write only.
- Tests: `tests/cuda_ptx_cache_trust_windows.rs` and `location.rs`, two relative-path cases in `tests/cuda_ptx_cache_trust/entries.rs`, `cuda_runtime_dir_test`, and unit tests in `ptx_cache_dirs.rs` and `lang_analyzer/cache.rs`.
- Docs: `docs/installation.md`, `docs/environment-variables.md`, the overlay `README.md`. The `windows_build` CI filter gains the two headers.

## 4. Results

| Check | Where | Result |
|---|---|---|
| New Windows test, overlay from `main` | N1X | 15 of 19 fail; the 4 passes are the child, the private hit, the control and read-only entries |
| New Windows test, this PR | N1X | 19 of 19 pass |
| Write entry for Users, Everyone, Authenticated Users on leaf, parent or grandparent, planted entries | N1X, `icacls` | refused with one notice, compiled, entries unchanged |
| Inherited, inherit-only, junction at a managed name, shared temp directory, `\\?\` spelling, past `MAX_PATH` | N1X, `icacls`, `mklink /J` | refused |
| Relative `MLX_PTX_CACHE_DIR` (`rel`, `.\rel`, `\rel`, `C:rel`) and `MLXCEL_CACHE_DIR` | N1X | refused, nothing created in the current directory |
| `mlxcel generate`, default directory | N1X | first run compiles, second run hits, 4 `.ptx` in `~\.cache\mlxcel\cuda-ptx\7ca40033132d\sm121-nvrtc13.4` |
| `mlxcel generate`, `MLXCEL_CACHE_DIR` inheriting Authenticated Users write, warm entries planted | N1X | `... is writable by S-1-5-11`, compiled |
| `mlxcel-server`, one chat completion, shared root; private root twice | N1X | refused and compiled; compiled, then hit |

## 5. Trade-offs and Residuals

- **POSIX C++ not compiled here:** this host has no Linux toolchain, so the `#else` branches of `jit_module.cpp` and the Linux tests were not built or run on it. CI compiles the former. The two new `entries.rs` cases and the `ptx_cache_dirs.rs` tests that cover other-write need a run on GB10.
- **Paths not exercised:** a foreign owner needs a second account or a privilege this test does not assume, and a null DACL needs a FAT or exFAT volume. Both come from the ROCm header unchanged.
- **Drive root:** this host has one drive, so the inherited case used `icacls` under `%TEMP%`, not a real `D:\`.
- **Entries on Windows:** no per-file check, as in ROCm. A directory that was shared and then made private still loads what it holds, and the documentation says to delete those entries.
- **Empty `MLX_PTX_CACHE_DIR`:** still turns the cache off silently, as it did on both platforms. ROCm treats its variable as unset when empty. The issue did not ask to change this.

## 6. Learning Points

- The issue described a path the binary did not take. A run with the variable set and a look at where the files landed took one command and changed the scope of the fix.
- `std::getenv` and `SetEnvironmentVariableW` do not meet on Windows. Any Rust `set_var` meant for C++ code has this problem, and it fails without an error.
- A comment line ending in `\\?\` continues onto the next line in C++. Neither MSVC at `/W4 /WX` nor a test noticed; a grep did.
