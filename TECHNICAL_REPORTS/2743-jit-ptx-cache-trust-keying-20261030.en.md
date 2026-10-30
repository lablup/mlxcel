# PR #2743: Harden the JIT PTX disk cache's trust and keying

**Date**: 2026-10-30 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: C++ (CUDA overlay), Rust
**Risk Level**: Medium (closes a cross-user code-injection path; a group-writable explicit cache directory, or a group-writable `~/.cache`, now turns the disk cache off with a notice, which costs a full NVRTC compile on every run until the mode is fixed)

## Executive Summary

The CUDA JIT overlay (`jit_module.cpp`) loads compiled modules, including mlxcel's fused MoE and SSM decode kernels since #2538, from a PTX disk cache whose directory it never checked. On a multi-user host whoever created `/tmp/mlx` first, or could write any parent of the cache directory, could plant modules under derivable names that load into other users' processes. This PR makes the overlay refuse a cache directory unless every component of its canonical path, from the leaf up to `/`, is owned by the user or root and is not group- or other-writable (root-owned sticky directories such as `/tmp` excepted). It also moves the POSIX fallback to a per-user `$TMPDIR/mlx-<euid>/` root, keys entries by an `sm<cc>-nvrtc<major>.<minor>` subdirectory, sweeps orphaned publish temporaries, and opens temporaries with `O_CREAT|O_EXCL|O_NOFOLLOW`. On the Rust side, `ensure_persistent_ptx_cache` creates its directories 0700 and repairs the 0775 directories earlier releases created under umask 002, so existing installs keep their persistent cache after the upgrade. Closes #2539, including the two items added from the PR #2552 security review.

## 1. Problem Statement

### The cache directory was trusted unconditionally

`ptx_cache_dir()` was a once-per-process static: `MLX_PTX_CACHE_DIR`, else `temp_directory_path() / "mlx" / version() / "ptx"`, created with `create_directories` and never checked for ownership or mode. The fallback was shared by every user on the host. Only `mlxcel` and `mlxcel-server` call `ensure_persistent_ptx_cache`, so `bench_decode`, `bench_engine`, `engine_parity`, `speculative_bench`, every test binary, and every embedder used that shared fallback. Module file names are derivable (kernel name, template arguments, source hash), so the first user to create `/tmp/mlx` could pre-populate entries that other users' processes would hand to `cuModuleLoadDataEx`. PR #2538 raised the stakes by making custom kernels cacheable: the fused decode kernels now come from this cache on every warm run.

### Entries were not keyed by device or compiler

`compile()` targets `--gpu-architecture=sm_<major><minor>[a]` and may emit SASS (`compiler_supports_device_sass`), but neither the module name nor the directory carried the compute capability or the NVRTC version. A cache directory shared across GPUs or toolkits (an NFS home, a container image layer, a machine with two GPU generations) could hand one device a module built for another.

### Orphaned temporaries and a symlink-following publish

`write_cached_ptx` publishes by writing a temporary and renaming it. A process killed between the two left the temporary forever. After the PR #2552 rename, temporaries are dotfiles named `.tmp<pid hex>_<tick hex>.<ext>` without the module name, so a sweep has to match that form, not a module-name prefix. The temporary was opened with `std::ofstream`, which follows a symlink planted at the path and writes through it.

### The retry branch was untested

`JitModule::JitModule` has a retry: if the driver rejects a cached module, it removes the entry, recompiles, and republishes. `tests/cuda_custom_kernel_disk_cache.rs` reached it only through an empty `.ptx` and a malformed map, both of which `read_cached_ptx` already rejects as misses, so the `cuModuleLoadDataEx` failure path itself never ran in a test.

## 2. Technical Decisions

### All enforcement in the overlay, not in a Rust init

Every binary and embedder shares `jit_module.cpp`. A Rust-side check would reach only callers that run it, which is exactly the set (`mlxcel`, `mlxcel-server`) that was already off the shared temp fallback. Putting the check in `ptx_cache_dir()` and `device_cache_dir()` covers bench binaries, test binaries and embedders without touching them. The `_WIN32` branch keeps its earlier behavior; the issue scoped the change to POSIX.

### Check every component of the canonical path, then use the canonical path

`cache_dir_is_trusted` calls `std::filesystem::canonical(dir)` and then `lstat`s each component from the leaf up to `/`. A component passes when it is owned by `geteuid()` or root and is not group- or other-writable, unless it is root-owned with the sticky bit (`/tmp`, where others cannot rename or delete your entries). The leaf must also be a directory. Checking only the leaf was rejected: the owner of a writable parent can rename the leaf away and put their own in its place. Using the canonical path afterwards means a symlink component swapped in after the check cannot redirect later reads and writes, and because every component is then owned by the user or root and closed to others, no other non-root user can change the resolved path for the rest of the process. A `canonical()` failure counts as untrusted.

The check runs on whatever the source is: an explicit `MLX_PTX_CACHE_DIR`, the persistent default the mlxcel binaries set through that variable, or the temp fallback. Missing components are created with `::mkdir(..., 0700)` first (`make_private_directories`); existing ones are never chmod-ed by the overlay, only judged.

### Refuse names only mlxcel creates when they are symlinks

The per-user root `$TMPDIR/mlx-<euid>` sits in a shared directory, where another user can create the name first, and the device subdirectory `sm<cc>-nvrtc<ver>` is a name only the overlay creates. For these two, `make_owned_directory` `lstat`s the path after `mkdir` and refuses a symlink or a non-directory. Following such a link would let someone else choose where this user's cache lives, and the trust check would then inspect the target, which may pass while still holding modules written before it was made private (for example a directory that was group-writable until the Rust-side repair tightened it). A user-supplied `MLX_PTX_CACHE_DIR` may still contain symlinks; `canonical()` resolves them and the resolved components are checked. This refinement came from review of the previous PR head (commit 43e3d073).

### Refusal turns the cache off, not the process

On refusal `ptx_cache_dir()` or `device_cache_dir()` returns an empty path, the existing "cache off" value that `read_cached_ptx` and `write_cached_ptx` already honor, so kernels still compile and run. One line per process goes to stderr, deliberately not muted by `MLXCEL_QUIET_JIT` because a refused cache is a configuration problem rather than progress noise: `mlxcel: CUDA kernel disk cache disabled: <component> <reason>`. The reasons are `is not owned by the current user`, `is group- or other-writable`, `is a symbolic link`, `is not a directory`, and `cannot be resolved (<error>)`. Failing the process was not considered: the cache is an optimization, and a hard failure would turn a mode bit into an outage.

### A device and compiler level beneath the existing scoping

`device_cache_dir(device)` returns `ptx_cache_dir() / "sm<major><minor>-nvrtc<major>.<minor>"` (for example `sm121-nvrtc13.0`), created 0700, refused if it is a symlink, and trust-checked like the root. It is resolved once per key per process under a leaked mutex and map (leaked like `get_jit_module`'s cache because modules can be built during static teardown). `JitModule::JitModule` uses it for `read_cached_ptx`, `write_cached_ptx` and the retry's `get_ptx_path`. The Rust commit scoping in `ensure_persistent_ptx_cache` (`cuda-ptx/<mlx-commit>`) is unchanged; this adds a level beneath it. Flat entries already in a root are no longer read and are not migrated; they leave with the next pin-scoped directory.

### Exclusive, non-following temporaries as defense in depth

Once the trust check passes, no other user can create files in the cache directory, so the planted-symlink case mainly covers files left before a directory was made private and same-user mistakes. The POSIX publish now opens the temporary with `O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC`, mode 0600, and writes through `write_all` (retrying short writes and `EINTR`). Any failure (`EEXIST`, `ELOOP`) leaves that entry unpublished for this process and the cache cold for that kernel; the compiled code is still used, and the colliding path is not removed because it is not ours.

To test this without predicting a nanosecond tick, the overlay reads a test-only hook, `MLXCEL_PTX_CACHE_TEST_TMP_SUFFIX`, that fixes the suffix. It is honored only when the value has the generated form `.tmp<hex>_<hex>`, so it cannot carry a path separator; any other value is ignored. It is documented as test-only in `docs/environment-variables.md`.

### Sweep stale temporaries by name, type and age

On the first resolve of a device directory per process, `sweep_stale_temporaries` removes top-level entries whose name matches `.tmp<hex>_<hex>.{ptx,txt,cu}`, that are regular files judged by `symlink_status` (so a symlink is never followed or removed), and whose mtime is more than one hour old (`kStaleTempAge`). A live publish renames within milliseconds, so an hour separates orphans from in-flight writes by a wide margin. Errors are ignored; the sweep is housekeeping. Nested directories from module names over 245 characters are out of scope.

### Repair the directories older releases created, through a descriptor

Under umask 002 (user private groups, the Ubuntu default for regular users), earlier releases created `~/.cache/mlxcel/cuda-ptx/<commit>` as 0775. Without a repair, the new check would turn the persistent cache off on every such existing install, the hosts the cache was built for. `ensure_persistent_ptx_cache` now creates with `DirBuilder` mode 0700 and calls `ptx_cache_dirs::tighten_persistent_ptx_dirs`, which clears group and other write on `cuda-ptx/<scope>` and `cuda-ptx`, and on the root only when `MLXCEL_CACHE_DIR` is unset. An explicit `MLXCEL_CACHE_DIR`, an explicit `MLX_PTX_CACHE_DIR`, and anything above the root (such as `~/.cache`) are left as the user set them and refused with the notice if they fail the check.

The repair touches only a real directory owned by the euid with a write bit to clear. It opens the path, compares the opened object's device and inode with the earlier `lstat`, and changes the mode through the descriptor (`fchmod` via `File::set_permissions`), keeping every bit except group and other write (setgid survives). A chmod by path would follow a symlink swapped in by anyone who can write the parent and apply 0755 to whatever it points at, such as a private file. `mlxcel-core` has no libc dependency, so the euid is read from the owner of a probe file created in the scope directory and removed immediately; when the directory is not writable the repair is skipped.

## 3. Change Summary

9 files, +1520/-140, four commits plus a merge of origin/main (c242a692):

| Commit | Content |
|--------|---------|
| `3a08816b` | Trust check, per-user fallback, device keying, stale-temporary sweep, `O_EXCL\|O_NOFOLLOW` publish with the test hook, retry-branch test, shared test helpers |
| `7031ca5f` | `ensure_persistent_ptx_cache` creates 0700 and repairs group-writable default directories |
| `43e3d073` | Refuse symlinked `mlx-<euid>` root and device directory; descriptor-based repair with a device and inode match |
| `a8fc4840` | Document every refusal reason and the sweep |

- `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/jit_module.cpp`: `report_cache_disabled`, `make_owned_directory`, `make_private_directories`, `cache_dir_is_trusted`, `is_temp_suffix`, `is_publish_temporary`, `test_temp_suffix`, `write_all`, `sweep_stale_temporaries`, `device_cache_dir`; `ptx_cache_dir()` POSIX branch rewritten; `write_cached_ptx` opens temporaries exclusively and creates nested directories private; `JitModule::JitModule` uses `device_cache_dir(device)` for read, publish and retry.
- `src/lib/mlxcel-core/src/lib.rs`: `ensure_persistent_ptx_cache` uses a 0700 `DirBuilder` and calls the repair.
- `src/lib/mlxcel-core/src/ptx_cache_dirs.rs` (new, unix only): `make_private_dir`, `repair_dir`, `current_euid`, `tighten_persistent_ptx_dirs`, and 4 unit tests.
- `tests/common/cuda_kernels.rs` (new): the fused MoE launch, `emit`/`parse_out`, `private_tempdir` (tempfile leaves the mode to the umask), and device-directory helpers, shared by both CUDA cache tests; `tests/common/mod.rs` exposes it under `cfg(feature = "cuda")`.
- `tests/cuda_ptx_cache_trust.rs` (new, 10 tests): group-writable dir refused, 0700 dir under a 0777 parent refused, private dir accepted, per-user fallback under `TMPDIR` (and `$TMPDIR/mlx` not created), device and NVRTC keying, stale-temporary sweep, symlinked temporary not written through (plus an invalid hook value ignored), symlinked device dir refused, symlinked fallback root refused, and the child entry point. Each scenario re-runs the test binary as a child, because `MLX_PTX_CACHE_DIR` and `TMPDIR` are read once per process; the parent never touches CUDA.
- `tests/cuda_custom_kernel_disk_cache.rs`: entries found in the device subdirectory; a new phase overwrites the gate-up `.ptx` with `b"not a cuda module\n"` and keeps its map, then requires a successful child, the JIT notice, bit-identical output, a republished full-length `.ptx`, and an unchanged map.
- `docs/installation.md` and `docs/environment-variables.md`: the per-user fallback, the device subdirectory, the trust rule, every refusal reason, the repair, the sweep, and the test hook.

## 4. Trade-offs and Residuals

- **Behavior change for group-writable directories.** A group-writable explicit `MLX_PTX_CACHE_DIR` or `MLXCEL_CACHE_DIR`, or a group-writable `~/.cache` or home directory, is refused. The process still works but compiles every kernel on every run (the custom fused decode kernels alone cost about a second per `mlxcel generate` on GB10, per #2492). The notice names the component; the fix is `chmod go-w` on it or a private `MLX_PTX_CACHE_DIR`. A deployment that deliberately shares one cache across users of a group loses the cache; that sharing is the attack this PR closes.
- **Foreign ownership is not tested.** It uses the same predicate as group-writability, but a directory owned by another user cannot be created unprivileged, so only the mode branch is exercised.
- **Mode bits only.** The check reads `st_mode` and `st_uid`; POSIX ACLs are not inspected. On Linux an ACL that grants write to a named user or group normally shows up as the group-write bit (the ACL mask), so it would be refused, but no test covers ACLs.
- **Root is trusted.** Root-owned components pass, and root can always change anything; this is by design.
- **Windows and ROCm unchanged.** The `_WIN32` branch keeps the shared `%TEMP%\mlx\<version>\ptx` without an ownership check. The ROCm overlay was out of scope.
- **Old flat entries stay on disk.** They are not read and not deleted; they occupy space until the pin-scoped directory is replaced.
- **The repair depends on a writable scope directory.** `current_euid` creates a probe file; on a read-only or squashed filesystem the uid cannot be read or does not match, and the repair silently does nothing, leaving the overlay to refuse with a notice.
- **The overlay README was not updated.** Its `jit_module.cpp` line is shared with #2696, so it was left to avoid a conflict.
- **Out of scope per the issue:** calling `ensure_persistent_ptx_cache` from bench binaries, fsync on publish, and sweeping nested directories for long module names.

## 5. Validation

All on GB10 under `gpu-lock`, as reported in the PR body.

**Suites.** `cuda_ptx_cache_trust` 10/10, `cuda_custom_kernel_disk_cache` 1/1, `ptx_cache_dirs` unit tests 4/4. `cargo fmt`, `clippy -D warnings` (mlxcel and mlxcel-core lib and tests, both test targets) and `make verify-env-var-docs` passed.

**Failing first on main 8385f40b.** The 0770 dir, the 0777 parent, the per-user fallback, the device keying, the sweep, and the updated disk-cache test all failed; the private-dir control passed. On the previous PR head, the symlinked device dir, the symlinked fallback root, and the swapped-path chmod failed, which is what commit 43e3d073 fixed.

**Mutation evidence for the two tests main cannot fail.** The test hook is new and the retry already existed, so neither test can fail on main. With the temporary open reverted to a following `O_CREAT|O_TRUNC` and `throw;` added at the top of the retry catch, the symlink test failed and the garbage-`.ptx` phase aborted.

**Real binary.** `mlxcel generate` on granite-4.0-h-tiny-4bit printed no JIT notice on its second run, both with a fresh cache and with the real home default. On the home default the 0775 directories became 0755, and `sm121-nvrtc13.0` was created 0700.

**Not verified.** Only Linux on GB10 was run; other distributions with different umask or `TMPDIR` conventions were not. A home directory on NFS, a cache shared by two GPU generations on one host, and the Windows branch were not exercised. No performance measurement was taken; the added work is one `canonical()` plus one `lstat` per path component per resolved directory, once per process.

## 6. Learning Points

- A trust check on a path has to cover every component up to `/`, not the leaf: write access to any parent is the ability to replace the leaf. The root-owned sticky exception is what lets `/tmp` itself pass while a foreign `/tmp/mlx` fails.
- Check, then use the canonical path the check approved. Re-resolving the original path later reopens the window the check closed.
- For a name only your code creates in a shared directory, a symlink at that name is evidence of tampering, not a configuration to follow. Resolving it and checking the target can approve a directory that still holds stale planted content.
- Changing a mode safely means acting on a descriptor whose device and inode match what you inspected. `chmod(path)` follows whatever sits at the path at that instant.
- A security tightening that existing installs fail by default needs a migration in the same PR. Here the repair of umask-002 directories is what keeps the persistent cache working on upgrade; without it the fix would have silently disabled the feature it protects.
- When a code path cannot be reached deterministically (a nanosecond-tick temporary name), a narrowly validated test hook plus a mutation run is stronger evidence than a test that passes on both old and new code.
