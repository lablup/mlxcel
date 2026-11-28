# PR #3289: Make the XLA artifact caches private to the user

**Date**: 2026-11-23
**Status**: PR open (closes #3248). Validated on ROCm gfx1151 with IREE `hip`. The GB10 / CUDA run of the same gates and the greedy comparison is pending.

## Executive Summary

Every OpenXLA/IREE compiled-artifact cache (text VMFB bundles, Phi4MM and Gemma3n audio, six vision-tower caches, the ROCm tuning spec, the native GEMV HSACO objects, the aux smoke and numeric probes) was a fixed-name directory under the shared temp directory, and cached artifacts were loaded with no ownership check. Another local user could create the directory first or plant a VMFB next to a manifest that matches, because every manifest field is derivable from public inputs. A CPU-target VMFB is native code, so this was code execution as the victim.

The caches now live under `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/xla/<cache>`. An artifact is reused only when the cache directories and the file are private to the current user (checked on an `O_NOFOLLOW` descriptor) and no directory above the cache root lets another user swap the root out. Anything that fails is left untouched, named in one stderr warning, and the artifact is compiled into a fresh `mkdtemp` directory. Files are published `0600` by atomic rename. Two new `mlxcel-core` modules carry the checks (`private_dir`, `private_group`) and a new `mlxcel-xla` module (`cache_dir`) routes every production site through them.

The PR went through five commits. Review added the ancestor check, the user chose a fail-closed policy for group-writable ancestors, and the security review found and fixed a HIGH (fallback root never re-verified) and a MEDIUM (nsswitch parser fail-open).

## 1. Problem Statement

#3248 (found during the security review of #3242, pre-existing) listed the sites: `iree.rs`, `tuning.rs` and `phi4_audio.rs` used `temp_dir()/mlxcel-xla-vmfb`; the vision and audio runtimes used per-family names under the same temp directory; `native_qmv.rs` used `temp_dir()/mlxcel-xla-qmv4`. Each site called `create_dir_all` and then loaded whatever it found if the name existed.

The auxiliary manifests (`aux_manifest.rs`) prove only that an artifact matches a contract built from the checkpoint config, compiler version, flags and MLIR text. None of that is secret, so a manifest says nothing about who wrote the file. The JIT PTX cache had the same class of problem and was hardened in #2743 and #2777; the XLA caches never were.

Two smaller defects sat next to it. `compile_one` and the tuning spec wrote straight to the final name, so a crash left a partial VMFB that the next load reused. And under a `umask` of `002` every cache file would have been group-writable.

## 2. Change Summary

| File | Change |
|---|---|
| `mlxcel-core/src/private_dir.rs` (new, public) | `ensure_private_dir`, `verify_private_dir`, `verify_trusted_ancestors`, `open_private_file`, `seal_private_file`, `create_fresh_private_dir`, `current_euid`, `clear_group_write` |
| `mlxcel-core/src/private_group.rs` (new) | Proves the process egid is a sole-member group from `/etc/nsswitch.conf`, `/etc/group`, `/etc/passwd` |
| `mlxcel-xla/src/cache_dir.rs` (new) | `XlaCache` enum, `xla_cache_dir`, `artifact_slot`, `read_private`, `write_private_atomic`, `publish_private`, the `mkdtemp` fallback |
| `iree.rs`, `tuning.rs`, `native_qmv.rs`, `aux_manifest.rs` | Cache lookup through `artifact_slot`; compile to a temporary sibling and publish `0600` by rename; the GEMV HSACO is read from the checked descriptor |
| Vision and audio runtimes, `aux_smoke.rs`, `numeric_probe.rs` | `temp_dir().join(...)` plus `create_dir_all` replaced by `xla_cache_dir(XlaCache::...)` |
| `production_tests.rs` | Fixture directory created `0700`, as production does; a `create_dir_all` fixture under umask 002 would now be refused |
| `docs/installation.md`, `docs/environment-variables.md`, `CHANGELOG.md` | Cache location, the ancestor and group rules, the refusal message, how to clear the cache |
| `scripts/xla/fusion_spike.sh` | Looks for the decode MLIR in the new location |

### Directory and file checks

`verify_private_dir` and `open_private_file` open the path with `O_NOFOLLOW | O_NONBLOCK` and `fstat` the descriptor: it must be a real directory or regular file, owned by `euid`, with no group or other write bit. `O_NONBLOCK` keeps a FIFO swapped in for a file from hanging the open (the same lesson as #2765). The GEMV cache reads its bytes from that same descriptor, so there is no window between check and read.

`ensure_private_dir(root, subpath, repair_root, euid)` refuses a relative root or one containing `..`, creates the chain `0700`, clears group write on the directories mlxcel manages (and on the root only when it is the default, not an explicit `MLXCEL_CACHE_DIR`), and verifies every directory in the chain. Group write is cleared with the PTX cache's existing `make_private_dir` (fchmod through a descriptor with a dev/inode re-check), reused rather than copied, and PTX behavior is unchanged.

### Ancestors

IREE opens a VMFB by path, so the descriptor check alone is not enough: whoever can write a directory above the root can rename the verified root away between the check and the load. `ensure_private_dir` canonicalizes the root's parent once, runs `verify_trusted_ancestors` from there to `/` (each component a real directory owned by the user or root, not other-writable unless root-owned and sticky like `/tmp`), and returns the leaf spelled through the canonical parent, so a symlink above the root swapped after the check cannot redirect later opens. Nothing above the root is modified. This is the PTX overlay's ancestor rule from #2539.

### Group-writable ancestors

With a `umask` of `002` and user private groups, `~/.cache` is `0775` and its group contains only its owner. The strict rule would refuse it, which would push every such host onto the fallback. `private_group` accepts a group-writable ancestor only when all of these are proven, and otherwise refuses it exactly like the strict rule:

- the host is Linux, the directory's group is the process egid, and it is not other-writable;
- every `/etc/nsswitch.conf` line for `passwd`, `group`, `initgroups`, `passwd_compat` and `group_compat` lists only `files`, `systemd` or `compat`, parsed the way glibc's `process_line` does (colon optional, a later line overriding an earlier one); `passwd` and `group` must be present, and a missing file refuses;
- no `+` / `-` compat line or malformed line in `/etc/passwd` or `/etc/group`, leading white space included;
- `/etc/group` lists no other member for that gid, and no other `/etc/passwd` account uses it as its primary group.

The files are read as plain text with a 4 MiB cap, once per process, never through `getpwent`. A refused ancestor produces a `GroupWritable` error whose message names the path and the fix (`chmod g-w <path>` or a private `MLXCEL_CACHE_DIR`).

### Fallback

A refused location is never read, moved or deleted. `artifact_slot` returns the same file name in a fresh subdirectory of a per-process `mkdtemp` root (`mlxcel-xla-XXXXXX` under the temp directory), after the temp directory itself passes the ancestor check. That root is held in a `Mutex` and re-verified before and after each subdirectory is created; a root that is missing, foreign, writable or a symlink is dropped and recreated.

## 3. Technical Decisions

**Reuse the existing cache root, not `XDG_CACHE_HOME`.** The issue suggested `$XDG_CACHE_HOME/mlxcel`. The PR uses `mlxcel_core::cache_root()`, which the tokenizer-script cache and the autotune tactic store already use, so all mlxcel caches stay under one root with one override (`MLXCEL_CACHE_DIR`). No new environment variable was added.

**Refuse and recompile, never repair a foreign location.** A refused directory may hold another user's files. Deleting or rewriting it would mean acting on paths an attacker controls, so the code leaves it alone and compiles elsewhere. The cost is one compile per refused load plus a warning; the load still works.

**Check on the descriptor, then guard the path.** Files are checked with `fstat` on an `O_NOFOLLOW` descriptor, which removes symlink and swap races for the file itself. Because IREE reopens by path, the remaining race is at the directory level, which the ancestor walk closes. The two parts depend on each other: neither alone is sufficient.

**Group-write policy chosen by the user, on the condition that it never fails open.** The first review fix (9ab6dbf0) accepted any group-writable ancestor whose group was the process egid. On hosts with a shared primary group (for example `users`), that trusts every member of the group. Three options were on the table: keep the egid shortcut, apply the strict PTX rule and refuse every group-writable ancestor, or accept one only when a sole-member group is proven from local NSS files. The user chose the third, on the condition that any doubt refuses rather than accepts. That condition is why a missing `nsswitch.conf` refuses although glibc would default to `files`, why any remote source (`sss`, `ldap`, `nis`, `winbind`) refuses, and why an unparsable line refuses where glibc would skip it. The strict rule is the floor, and the relaxation only ever adds hosts that are provably safe.

**Parse NSS files as plain text.** `getpwent` / `getgrent` go through NSS and can enumerate remote sources or block on the network. Reading the three files directly is bounded, deterministic, and testable: the parsers take contents, uid, gid and name as arguments, so tests run without root and without depending on the host's accounts. `verify_trusted_ancestors_with` takes the group decision as a closure for the same reason.

**Atomic `0600` publish everywhere.** VMFBs, HSACOs, MLIR inputs, the tuning spec and manifests are written to a unique `O_EXCL` temporary sibling, sealed `0600`, and renamed. This fixes the partial-VMFB reuse after a crash and makes published files pass the private-file check under any umask.

## 4. Review Outcomes

**Implementation review** (PR comment 7088994696): one HIGH finding. The first commit checked only the cache root and the directories below it, while IREE opens the VMFB by path, so a writable or foreign-owned parent (or a symlinked parent pointing at one) allowed swapping the root after the check. Fixed in 9ab6dbf0 by the ancestor walk and the canonical-parent respelling. Two LOW items were left: `emitter/gemma3n_qmv.rs::run_cuda_stablehlo_probe` still writes predictable per-pid files under the temp directory, but it is behind the `diagnostics` feature on a CUDA IREE build and never in production; and fallback directories are not removed (documented). The reviewer also confirmed that stubbing out the ownership, mode, symlink and ancestor checks makes 7 of the 11 `cache_dir` tests fail.

**Group policy** (85d1f434, 951faf46): the egid shortcut from 9ab6dbf0 was replaced by the proven sole-member rule described in section 3.

**Security and performance review** (PR comment 7089410163), reviewed as a local multi-user attacker. The ancestor walk, the respelled leaf, the descriptor checks, the paired VMFB/manifest check and the GEMV descriptor read held up. Hard links need no `nlink` check, because only the user can create entries in a verified directory and a linked inode keeps its owner and mode.

- HIGH, fixed in 42f96db5: the `mkdtemp` fallback root was cached in a `OnceLock` and never re-checked. If a temp cleaner such as systemd-tmpfiles deleted it under a long-running server, another user could recreate the listable name in `/tmp` and swap the per-load subdirectories between compile and load. The root is now re-verified before and after each subdirectory is created and replaced when it fails.
- MEDIUM, fixed in 951faf46: `nsswitch_is_local` required a colon and looked at each line on its own, so `passwd sss` was ignored and `passwd: files` followed by `passwd sss` passed. It also ignored `initgroups`, which can add supplementary members from a remote source. The parser now follows glibc's `process_line` and requires every checked line to be local. The review rated this MEDIUM because it fails open only on root-written configuration, but under the user's "never fail open" condition it had to be fixed.
- LOW, fixed in 951faf46: `+` / `-` compat lines with leading white space were not caught.
- MEDIUM, documented: fallback directories are never removed, so every refused run leaves compiled artifacts in `/tmp` (tmpfs on Debian 13).
- LOW, not fixed: `write_private_atomic` and `write_owner_only_new` delete the target name when the `O_EXCL` open fails. The name is a unique sibling inside a verified private directory, so only the user could have created it.

Performance: the ancestor walk runs once per cache resolution at model load and the group decision once per process. Nothing runs per token.

## 5. Tests and Verification

New tests:

- `mlxcel-xla` `cache_dir::tests` (11): default location created owner-only and reused, another uid (mismatched euid), group- and world-writable directories, a symlinked cache directory, a root below an other-writable directory, a relative or missing root, a symlinked artifact, a group-writable artifact in a private directory, a planted VMFB with a valid manifest (world-writable directory and other uid), owner reuse with exactly one compile, atomic owner-only writes, and a fallback root replaced after creation.
- `mlxcel-core` `private_dir` tests: relative root and escaping subpath, foreign owner, writable directories, managed chain losing group write while an explicit root does not, symlinks anywhere in the chain, descriptor checks on files, fresh directories, other-writable ancestors, group-writable ancestors with the decision driven both ways, and the canonical-parent leaf.
- `mlxcel-core` `private_group` tests: sole member accepted (including `compat`); another member, another primary account, remote sources, override lines, the optional colon, `initgroups`, bracketed actions, compat lines with leading white space, malformed lines and missing files refused.

Results on the final head:

- `cargo test --profile test-fast -p mlxcel-xla --lib`: 447 passed.
- `cargo test --profile test-fast -p mlxcel-core --lib -- private_dir private_group ptx_cache_dirs`: 28 passed.
- Clippy `-D warnings` on `mlxcel-xla` (no features, `iree`, `iree,micro-oracle,diagnostics`) and `mlxcel-core`, lib and tests; `cargo fmt --check`; `scripts/ci/check_used_by.py`; `scripts/ci/check_env_var_docs.py`.
- ROCm gfx1151 with IREE `hip`: `xla_greedy_timing` 3/3 on the final head with no refusal warning. This host has `~/.cache` at `0775 devops:devops`, a sole-member group, and `passwd: files systemd` / `group: files systemd`, so it exercises the relaxed rule in production. `xla_decode_scalar_reuse` 4/4 and `xla_prepared_prefill` 4/4 after 9ab6dbf0.
- Real-model greedy token ids (`MLXCEL_BACKEND=xla MLXCEL_PRINT_TOKEN_IDS=1 mlxcel generate --temp 0 -n 24`): llama-3.2-1b-instruct-4bit, Qwen3-VL-2B-Instruct-4bit and Molmo-7B-D-0924-4bit (image, `MLXCEL_XLA_VISION_BACKEND=iree`) identical to main from a cold and a warm cache at c362e290. llama-3.2-1b re-checked after 85d1f434 with a cleared cache: identical, cold 11 s, warm 4 s, no refusal.
- `xla_gemma3n_prefill` fails at line 367 (`[20, 20, 20]` against `[5, 5, 5]`) identically on unmodified main (a9860ece). It is pre-existing and not caused by this PR.

## 6. Pending Work and Limitations

- **GB10 / CUDA validation is pending.** The last acceptance criterion of #3248 (text and multimodal real-model loads unchanged on the GB10 host) is checked only on ROCm. The CUDA IREE path and the CUDA GEMV object path have not been run with the new cache code.
- **Fallback directories are not removed at exit.** Each refused location leaves a `mlxcel-xla-XXXXXX` directory under the temp directory. This is documented in `docs/installation.md`; on a host where the cache is refused every run, those directories accumulate until a temp cleaner removes them.
- **First load after upgrade recompiles.** The old `/tmp/mlxcel-xla-*` directories are no longer read and can be deleted by hand.
- **Left as is:** `#[cfg(test)]` per-pid scratch paths, and `run_cuda_stablehlo_probe` (diagnostics feature, CUDA IREE build only).
- **Hosts that will see a refusal:** a group-writable ancestor on a host with a shared primary group, a remote name service, or NIS compat lines. The warning names the path and the fix; the load still works through the fallback.

## 7. Learning Points

- A content manifest is not an authenticity check when every field comes from public inputs. Trust in a cache has to come from who can write it.
- A descriptor check protects the file but not a path that another component reopens later. When a library (here the IREE runtime) opens by path, every directory up to `/` is part of the trust boundary.
- A one-time check of a temp directory decays: temp cleaners delete aged directories, and a listable name in a shared `/tmp` can be recreated by anyone. State held for the life of a long-running process needs re-verification at each use.
- Reimplementing a config parser for a security decision means matching the real parser's quirks (optional colon, last line wins). Where the reimplementation could differ, refusing every ambiguous case keeps the difference fail-closed.
- Passing uid, gid, file contents and policy decisions as parameters lets tests model another user's directory and other hosts' account databases without root.
