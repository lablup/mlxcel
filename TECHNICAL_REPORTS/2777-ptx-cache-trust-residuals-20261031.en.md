# PR #2777: Close the PTX cache trust residuals from #2743

**Date**: 2026-10-31 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: C++ (CUDA overlay), Rust, CMake
**Risk Level**: Low (defense in depth on a cache whose directories #2743 already protects; the visible change is that a group-writable or symlinked cache entry is recompiled once and replaced, and that a `--release` binary ignores `MLXCEL_PTX_CACHE_TEST_TMP_SUFFIX`)

## Executive Summary

PR #2743 (closing #2539) made the CUDA JIT PTX disk cache trust only directories that the user or root owns and that no one else can write. Its implementation and security reviews left four LOW residuals, collected in #2765, plus a fifth from the #2743 review: a test-only hook compiled into every build. This PR closes all five. `..` in `MLX_PTX_CACHE_DIR` is resolved before any directory is created; each cache entry file (`.ptx` and `.txt`) is opened without following symlinks and loaded only if `fstat` on the descriptor shows a private regular file; the Rust-side directory repair takes the uid from `geteuid()` instead of a probe file and opens with `O_DIRECTORY|O_NOFOLLOW|O_NONBLOCK`, so a FIFO swapped in cannot hang startup; and the suffix hook is compiled in only for the `debug` and `test-fast` Cargo profiles. None of the residuals allowed code injection on its own. Closes #2765.

## 1. Problem Statement

### `..` left a stray directory behind

`ptx_cache_dir()` in the `jit_module.cpp` overlay creates missing components one by one with `make_private_directories` and only canonicalizes the path afterwards, in `cache_dir_is_trusted`. For `<dir>/missing/../cache` the walk created `<dir>/missing`, stepped back out through `..`, and created `<dir>/cache`. The cache worked, but `<dir>/missing` stayed on disk.

### The uid probe could fail or leak

`mlxcel-core` had no `libc` dependency, so `current_euid` in `ptx_cache_dirs.rs` learned the effective uid by creating `.mlxcel-uid-probe-<pid>` in the scope directory and reading its owner. A crash between create and remove left the file behind, and on a scope directory the owner could not write (for example mode 0575) the probe failed, so the 0700 repair silently did nothing. That is the case #2743's own report listed as a residual.

### Cache entries were not checked per file

`read_cached_ptx` read `<module>.ptx` and `<module>.txt` with `std::ifstream`, which follows symlinks and ignores ownership and mode. #2743's directory checks stop other users from planting entries, but a file can still be writable by others inside a trusted directory, from a restored backup, a copied cache, or an earlier umask. The entry would then load silently.

### The 0700 repair could block startup

`repair_dir` checked the path with `symlink_metadata`, then called `std::fs::File::open(path)` and compared device and inode. Anyone who can write the parent could swap the directory for a FIFO between the two calls; opening a FIFO for reading blocks until a writer appears. `ensure_persistent_ptx_cache` runs at `mlxcel` and `mlxcel-server` startup, so the process would hang before loading a model.

### A test hook shipped in release binaries

#2743 added `MLXCEL_PTX_CACHE_TEST_TMP_SUFFIX`, which fixes the suffix of the publish temporaries so a test can plant a symlink at the exact path a publish opens. It was validated to the `.tmp<hex>_<hex>` form and could not carry a path separator, but it was compiled into every build, so a shipped binary honored an environment variable that exists only for tests. The issue originally marked it out of scope; the orchestrator added it to this run from the #2743 review.

## 2. Technical Decisions

### Normalize with `weakly_canonical` before creating anything

`ptx_cache_dir()` now replaces the absolute cache path with `std::filesystem::weakly_canonical(cache, error)` before `make_private_directories`. That call resolves the existing prefix the way `canonical()` does in the trust check and normalizes the nonexistent remainder lexically, so the directory created is the one the trust check later approves, and the check still runs on the result. An error or an empty result returns the empty path, the existing "cache off" value, like every other failure branch. Rejecting `..` outright was the alternative; it would break a configuration that works today and gains nothing, since the trust check never saw the unnormalized path anyway.

### Check the entry on the descriptor that is read

A new POSIX-only helper, `read_trusted_cache_entry`, opens each entry file with `O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC`, `fstat`s the descriptor, and reads through it only when the file is a regular file, owned by `geteuid()` or root, with neither `S_IWGRP` nor `S_IWOTH`. Checking the descriptor rather than the path means the file read is the file checked; a separate `stat` then `ifstream` would reopen the window. `O_NOFOLLOW` makes a symlinked entry fail the open, and `O_NONBLOCK` keeps a FIFO at the entry path from blocking it (the `S_ISREG` test then rejects it). The read loop retries `EINTR`, and a short read, an `fstat` failure, or a `std::bad_alloc` from `resize` on an absurd `st_size` is a miss. An RAII `FdCloser` closes the descriptor on every path, including the exception; that and the `.txt` coverage came from the follow-up commit d39ce2a3.

Both files of an entry must pass, and an empty `.ptx` is still a miss as in #2492. A miss needs no new code: `JitModule` recompiles and `write_cached_ptx` republishes through its existing `O_CREAT|O_EXCL|O_NOFOLLOW` temporary (mode 0600) and `rename`, which replaces a symlink or a group-writable file at the entry path without writing through it. The Windows branch keeps its `file_size` plus `ifstream` read.

### Accept root-owned entries, against the issue text

The issue asked for "owned by `geteuid()`". Commit 7237b94b relaxed this to the current user or root. `cache_dir_is_trusted` already accepts root-owned directory components, and root can write any file regardless, so rejecting a root-owned file adds no protection. Without it, a root-owned prebuilt cache (one baked into a container image, for example) would be a permanent and silent miss for a non-root user: every process would recompile every kernel and try to republish into a directory it may not be able to write. The PR body records the deviation.

### Use `libc` directly in `mlxcel-core`

`libc = "0.2"` is added under `[target.'cfg(unix)'.dependencies]` of `mlxcel-core`, matching the root crate's pin, so `Cargo.lock` gains one dependency edge and no new package. `tighten_persistent_ptx_dirs` calls `libc::geteuid()`, which cannot fail, and the probe file and `current_euid` are deleted. The repair therefore no longer needs a writable scope directory. The unit tests read the uid the same way.

### `O_DIRECTORY|O_NOFOLLOW|O_NONBLOCK` on the repair open

`repair_dir` now opens through `OpenOptions::custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK)`. A non-directory or a symlink fails the open, and `O_NONBLOCK` ensures that even the FIFO case returns immediately. The device and inode comparison with the earlier `lstat` stays as the second guard, and the mode change still goes through the descriptor (`fchmod`).

### Gate the test hook on the Cargo profile directory, not on `PROFILE` or a feature

The hook is now inside `#if defined(MLXCEL_PTX_CACHE_TEST_HOOKS)`. Deciding when to define it took the most care, and the module comment in `build_support/ptx_cache_test_hooks.rs` records why the obvious signals fail:

| Option | Problem |
|--------|---------|
| `PROFILE` env var | Cargo sets it to `release` for every profile that inherits from release, `test-fast` included, so test builds would lose the hook |
| `debug_assertions` | `test-fast` turns them off |
| A Cargo feature | Changes the package metadata hash and with it `OUT_DIR`, so alternating test and binary builds would rebuild MLX from scratch each time |
| **Profile directory from `OUT_DIR`** (chosen) | Depends on Cargo's `<target>/[<triple>/]<profile>/build/<pkg>-<hash>/out` layout; an unrecognized layout fails closed |

`profile_dir_name` walks four ancestors of `OUT_DIR`, requires the `out` and `build` names, and returns the profile directory; `test_hooks_enabled` returns true only for `debug` and `test-fast`. Any other profile (`release`, `bench`, `dist`, a custom one) and any unrecognized layout leave the hook out. `build.rs` combines this with `cfg!(feature = "cuda")`, passes `MLXCEL_PTX_CACHE_TEST_HOOKS=ON|OFF` to CMake, and emits `cfg(mlxcel_ptx_cache_test_hooks)`, which `cuda_arch::PTX_CACHE_TEST_HOOKS` exposes so `symlinked_temporary_is_not_written_through` skips with a message when the hook is absent. In CMake the definition is set as a source property on `jit_module.cpp` alone (`set_property(SOURCE ... TARGET_DIRECTORY mlx ...)`), so flipping it recompiles one file, not MLX.

The detection is a pure function included by path into both `build.rs` and `mlxcel-mlx-pin`, the same pattern `mlx_pin.rs` uses, so its unit tests run without building MLX. The CI `mlx_pin` path filter now includes `build_support/ptx_cache_test_hooks*.rs`, and `scripts/ci/check_env_var_docs.py` allowlists `MLXCEL_PTX_CACHE_TEST_HOOKS` as a build-internal CMake option rather than an environment variable.

## 3. Change Summary

16 files, +676/-35. Three commits plus two merges of origin/main (77dff247, f04452d8):

| Commit | Content |
|--------|---------|
| `9cca24d4` | All five items: `weakly_canonical`, per-entry `.ptx` check, `libc::geteuid`, non-blocking repair open, profile-gated test hook; tests and docs |
| `7237b94b` | Accept root-owned entries, matching the directory rule |
| `d39ce2a3` | Check the `.txt` kernel map too, close the descriptor through an RAII guard on `resize` throw, document the entry rule |

By area:

- `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/jit_module.cpp`: `weakly_canonical` in `ptx_cache_dir()`; new `read_trusted_cache_entry`; `read_cached_ptx` split into a `_WIN32` branch (unchanged) and a POSIX branch reading both files through the helper; `test_temp_suffix` guarded by `MLXCEL_PTX_CACHE_TEST_HOOKS`; header note tagged `[mlxcel #2765]`.
- `src/lib/mlxcel-core/src/ptx_cache_dirs.rs`: probe removed, `geteuid`, flagged open, 2 new unit tests (6 total).
- `src/lib/mlxcel-core/build.rs`, `build_support/ptx_cache_test_hooks.rs` (+ tests), `src/cuda_arch.rs`, `src/lib/mlxcelverse/CMakeLists.txt`, `src/lib/mlxcel-mlx-pin/src/lib.rs`: the profile gate end to end.
- `src/lib/mlxcel-core/Cargo.toml`, `Cargo.lock`: unix-only `libc`.
- `tests/cuda_ptx_cache_trust.rs` and new `tests/cuda_ptx_cache_trust/entries.rs`: 4 new child-process scenarios (`dotdot_in_cache_dir_is_normalized`, `group_writable_cache_entry_is_not_loaded`, `symlinked_cache_entry_is_not_loaded`, `group_writable_kernel_map_is_not_loaded`), each asserting a JIT recompile, output identical to the first run, and a republished regular file without group or other write; the hook test now skips when the hook is compiled out.
- `.github/workflows/ci.yml`, `scripts/ci/check_env_var_docs.py`: path filter and allowlist.
- `docs/environment-variables.md`, `docs/installation.md`: `..` resolution, the per-entry rule, and the hook's profile restriction.

## 4. Trade-offs and Residuals

- **Untrusted entries cost one recompile.** A group-writable, symlinked, foreign-owned, or non-regular entry is a miss; the kernel is recompiled and the entry replaced with a 0600 file, so the cost is paid once per bad entry. `docs/installation.md` says entries an earlier release wrote under umask 002 are recompiled once. In practice the affected set is small: since #2743, publishes create 0600 temporaries inside the device subdirectory, and the pre-#2743 flat entries are no longer read at all. The realistic triggers are caches copied, restored, or prepared by hand.
- **A root-owned entry in a directory the user cannot write stays a miss if it is bad.** The republish cannot replace it, so that kernel recompiles on every run. This is unchanged from before and only matters for a broken prebuilt cache.
- **Hook coverage depends on the test profile.** A test run under `release` or a custom profile skips `symlinked_temporary_is_not_written_through` with a printed message instead of failing. CI and `make verify-test*` use `test-fast`, so the gate still runs it. The detection also depends on Cargo's target directory layout; a future layout change would turn the hook off (fail closed) and surface as that skip.
- **No release build was run.** That a `--release` binary ignores the variable rests on the profile-detection unit tests (including a mutation that adding `release` to the allowlist fails `release_builds_leave_the_hooks_out`) and on the CMake define being per-source; no release binary was built and probed.
- **Windows and ROCm unchanged.** The `_WIN32` read path still uses `ifstream` without per-file checks; the ROCm overlay (#2641) was out of scope.
- **Mode bits only.** As in #2743, POSIX ACLs are not inspected beyond what the group bits reflect.
- **The overlay README entry was not updated.** The issue listed it in scope; the diff does not touch it, and the source header comment carries the `[mlxcel #2765]` description instead.

## 5. Validation

As reported in the PR body, on GB10 with `--profile test-fast --features cuda`.

**Failing first on 5c8ef80a**, with the tests written before the fix: `dotdot_in_cache_dir_is_normalized` (the stray `missing` directory existed), `group_writable_cache_entry_is_not_loaded` and `symlinked_cache_entry_is_not_loaded` (silent cache hit, no recompile), `repair_runs_on_a_scope_dir_the_owner_cannot_write` (mode stayed 0575), and `a_directory_swapped_for_a_fifo_does_not_block` (did not return within 2 s; the test opened the FIFO for writing itself to release the blocked thread and fail instead of hang). For the hook gate, a mutation adding `release` to `PROFILES_WITH_TEST_HOOKS` fails `release_builds_leave_the_hooks_out`.

**After the fix.** `cuda_ptx_cache_trust` 14/14, including the hook test and `group_writable_kernel_map_is_not_loaded`; `CMakeCache` shows `MLXCEL_PTX_CACHE_TEST_HOOKS:BOOL=ON` and only `jit_module.cpp` receives the define. `cuda_custom_kernel_disk_cache` 1/1, `ptx_cache_dirs` 6/6, `mlxcel-mlx-pin` `ptx_cache_test_hooks` 4/4. `cargo fmt`, `clippy -D warnings` on mlxcel-core, mlxcel and mlxcel-mlx-pin lib and tests, `make verify-env-var-docs`, and `cargo deny check` passed.

**Not verified.** No release build. The issue's real-binary check (two `mlxcel generate` runs on granite-4.0-h-tiny-4bit with the home default, the second printing no JIT notice) is not listed in the PR body. The root-owned entry branch is not exercised by a test, since creating a root-owned file needs privileges; it shares the predicate with the tested user-owned branch. `repair_runs_on_a_scope_dir_the_owner_cannot_write` skips when run as root. No performance measurement; the added work per entry is one `open` and one `fstat` per file, replacing a `file_size` and an `ifstream` open.

## 6. Learning Points

- Validate the object you read, not the name you looked up. `open(O_NOFOLLOW)` plus `fstat` on the descriptor gives a check and a read on the same inode; `stat(path)` followed by a second open does not.
- `O_NONBLOCK` belongs on any open whose target might have been swapped for a FIFO, even when the expected object is a regular file or directory: a plain `open` for reading on a FIFO blocks until a writer appears, which turns a type confusion into a hang.
- Normalize a user path before acting on it component by component. A walk that creates directories executes `..` literally, so side effects happen before any later canonicalization can catch up.
- A trust rule for files should match the rule for the directories that hold them. Accepting root for directories but not for files would make a legitimate prebuilt cache a permanent miss without adding protection.
- Cargo's `PROFILE` variable is the inherited base profile, not the profile name, and a feature flag changes `OUT_DIR`. When a build script needs the actual profile, the `OUT_DIR` layout is the available signal; parse it strictly and fail closed.
- Scope a CMake definition to the one source that needs it (`set_property(SOURCE ... TARGET_DIRECTORY ...)`) so toggling it does not invalidate a large dependency's build.
