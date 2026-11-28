# PR #2652: Add a Windows build CI job and run the runtime-dir host tests

**Date**: 2026-10-27
**Status**: Complete; both new jobs pass on the PR, the first `main` run happens at merge
**Languages**: GitHub Actions YAML, Python, Shell, Markdown
**Risk Level**: Low

## Executive Summary

PR #2652 (issue #2642) puts the Windows build glue and the Windows runtime-directory logic under CI for the first time.
- **`overlay-host-tests`** (ubuntu-latest, about 20 s, unconditional) runs `make verify-rocm-overlay verify-cuda-runtime-dir`: the ROCm overlay record checks and the host tests for both overlays' runtime and include directory choice.
- **`windows-build`** (windows-latest, compile-only, about 20 min on a cache hit) runs the `mlxcel-mlx-pin` lib tests, the `tests/` Python suites and `cargo check -p mlxcel-core --features rocm` against a digest-pinned, cached ROCm 10.1.0 TheRock tarball.
- **Windows test hygiene:** the bash-driven test classes skip on `win32`, and two encoding bugs that broke the suites on Windows consoles are fixed.

## 1. Problem Statement

The Windows CUDA and ROCm build paths (#2599, #2603, #2604) and the relocatable runtime logic (#2607, #2648, #2651) were verified by hand only. The two host tests of the selection logic ran only from `make verify-*` targets that no CI job invoked, and nothing ran a real Windows `build.rs`. A regression in the include order or a breaking MLX pin bump could merge green.

## 2. Change Summary

| Area | Change |
|---|---|
| Host tests | New `overlay-host-tests` job. Unconditional, because several files it runs (`rocm_overlay.py`, `allocator_ownership_test.*`, Makefile targets) match no path filter and the job takes seconds. The record checks build their own synthetic git history, so a shallow clone suffices. |
| Windows job | New `windows-build` job on a hosted runner. `cargo check` hides build-script output, so a final step asserts `mlx_rocm_kernels.lib` exists, which proves the HIP build ran. |
| SDK tarball | The `gfx1151` TheRock 10.1.0 tarball (1.65 GB) is pinned by a self-computed sha256, since AMD publishes none. Cache key: version, family, digest. The digest is checked after download and after every restore, and the cache is saved right after verification so a failed build keeps it. Extraction excludes composable-kernel libraries, MIOpen and test plugins (3.8 of 8.4 GB). |
| Gating | New `windows_build` filter: root and core `build.rs`, `build_support/**`, `mlxcel-mlx-pin/**`, the MLX pin, the ROCm and CUDA overlays' `delayload.cpp`/`runtime_dir.h`/`jit_module.cpp`, the runtime-dir test scripts, the packaging scripts and their tests. Hosted runners only, `contents: read`, no credentials persisted. |
| Tests | Bash-driven classes of five suites (plus `test_bench_serving_concurrency.py`) skip on `win32`; pure-Python classes still run there. Locale code-page reads and emoji prints in `insert_apache_header.py` fixed. |
| Docs | `docs/installation.md` limits table, `CONTRIBUTING.md` job lists, `docs/mlxcelverse/rocm-overlay.md`. |

## 3. Technical Decisions

### 3.1 Full ROCm check instead of a subset

The issue left the depth of the Windows check open. It was measured on a scratch branch before committing: about 20 min on a cache hit (12 to 17.5 min of it `cargo check`), 14 min on a miss with a 5 s download, a 1.6 GB cache entry. That fits the bounds set for the issue (under 45 min, under 4 GB), so the job compiles the full MLX ROCm build rather than configure-only. The CUDA variant was optional and is not added.

### 3.2 Making the host-test job unconditional

The first version gated `overlay-host-tests` on the `rocm` and `windows_build` filters. The review found files the job runs that neither filter matches. At about 20 s, gating costs more in missed coverage than it saves, so it now runs on every PR and push, like `python-tooling`.

### 3.3 Proving the gate catches a regression

A commit swapped the `HIP_PATH` and bundle branches of `choose_rocm_include_dir()`; CI failed with `FAIL: include: bundle beats HIP_PATH`, and the next commit reverted it. After the revert the PR diff touched no gated path and both jobs skipped. With no `ci:full` label in the repository to force them, a comment naming the CI job was added to both host test scripts, which keeps them in the diff. The squash commit carries those paths, so `windows-build` also runs on `main` once and creates the `main`-scoped cache entry.

### 3.4 Cache scope

A PR can only write to its own `refs/pull/N/merge` cache scope, and the digest check runs after every restore, so neither a fork PR nor a corrupted entry can feed another run a different SDK.

## 4. Validation

- Windows 11 host: `python -m unittest discover -s tests -p 'test_*.py'` (188 run, 76 skipped), `cargo test -p mlxcel-mlx-pin --lib` (49), both host tests with ROCm's `clang++`, actionlint 1.7.7 clean.
- CI: the deliberate break failed (run 40406365843); the final tree passed both jobs (run 40415050086: host tests 21 s, Windows 19 min 50 s on a cache hit).
- Security review: no injection path from event data into `run:` scripts, read-only permissions, pinned actions in the workflow's tag style.

## 5. Known Gaps

- **`ci:full` label missing:** `ci.yml` refers to it, but the repository does not define it. Until a maintainer creates it, a PR that edits only the `windows-build` definition (for example a digest bump) cannot run that job before merge, and the label opt-in for self-hosted jobs has no effect either.
- **Python suites on Windows are gated:** a `tests/` change that breaks only on Windows fails later, on the next PR that touches a gated path. A small ungated Windows job for the suites would close this.
- **Partial overlay coverage:** the filter lists three files per overlay, while the check compiles the whole HIP overlay. Widening it adds a 20 min run to every ROCm overlay PR.
- **No link step:** `cargo check` does not link, so the root `build.rs` delay-load link arguments remain unexercised.
- **Cache pressure:** the repository already used about 10.5 GB of the 10 GB cache limit, so LRU eviction can drop the SDK entry; a miss costs one extra download, not a failure.
- **Over-broad skips:** a few tests that need no bash (Makefile text checks, one hardware-tag test) skip on Windows; Linux still runs them.

## 6. Learning Points

- **Gating a cheap job costs coverage:** a path filter has to list every input of the job, and for a job that takes seconds the filter usually saves less than the coverage it loses.
- **`cargo check` runs build scripts silently:** assert an artifact the build script must produce, otherwise a skipped native build looks like success.
- **Proving a negative needs the gate to fire:** after reverting a deliberate break, check that the jobs actually ran on the revert, not just that the run is green.
