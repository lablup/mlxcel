# Technical Report: Issue #2111 - WebUI Activity gate waits for the host GPU lock; CPU-only link job

**Date**: 2026-10-05

**Status**: Implemented; control flow validated locally against the real `gpu-lock` script on a private lock directory. Based on origin/main `6ddbb579`, pending the PR's own GB10 CI run and merge.

**Languages**: GitHub Actions YAML and bash (`.github/workflows/ci.yml`), Markdown (`docs/webui-integration-matrix.md`, `docs/installation.md`)

**Risk Level**: Low (CI-only; the gate keeps failing closed, and a runner without `gpu-lock` runs the unchanged body)

## Executive Summary

The `WebUI installed artifact` job on the GB10 runner failed for two environmental reasons. Its Activity gate ignored the `gpu-lock` that development sessions on the same host use, so a session's GPU job tripped the gate's fail-closed `nvidia-smi` check. Its persistent `test-fast` target directory also kept incremental state that linked with undefined `serde_json` generics until cleared by hand. The gate now waits up to 600 seconds for `gpu-lock` between the cooperative CI flock and the `nvidia-smi` check, and the job builds with `CARGO_INCREMENTAL=0`. Folded in from #2108: a new `CPU-only link` job links the `mlxcel-core` test binary with no GPU feature, the configuration in which #2108's missing guard surfaced.

## 1. Problem Statement

- Lock collision: run 36982300183 attempt 1 (PR #2093) failed at the `nvidia-smi` check while a development session held the GPU; a rerun made while holding `gpu-lock` passed. The runner runs as uid 1000 with `PrivateTmp=no`, so it sees `/tmp/gpu-lock-1000/lock`.
- Stale incremental state: run 36988516810 (PR #2094) failed to link on attempts 1 to 5 (undefined `serde_json` generics in one codegen unit of `$HOME/.cargo-target/mlxcel-webui-installed-ci`) and passed on attempt 6 only after the incremental state was cleared.
- No CI job linked a CPU-only build: every GB10 job passes `--features cuda`, and `cargo check` does not link (#2108).

## 2. Change Summary

| Area | Change |
|---|---|
| `webui-installed-artifact` job env | `CARGO_INCREMENTAL: "0"` for this job only |
| `Run Activity performance gate...` step | Body (nvidia-smi check and harness, unchanged) written to `$RUNNER_TEMP/webui-activity-gate.sh`; inside the existing flock it runs under `gpu-lock run --tag webui-activity --wait 600 -- bash <script>` when `gpu-lock` is on `PATH`, directly otherwise; step timeout 45 to 65 minutes to cover both 600-second waits |
| `changes` job | New `cpu_link` filter: `**/*.rs`, `**/Cargo.toml`, `Cargo.lock`, `build.rs`, `rust-toolchain.toml`, `src/lib/mlx-cpp/**`, `src/lib/mlxcel-core/cpp/**`, `src/lib/mlxcel-core/build_support/**` |
| New `cpu-link` job (`CPU-only link`) | GB10, repository guard, `cpu_link == 'true'` or `ci:full`; target dir `$HOME/.cargo-target/mlxcel-cpu-link-ci`; runs `cargo test -p mlxcel-core --profile test-fast --lib --no-run` |
| `concurrency` comment | Lists the seven GB10 jobs instead of four |
| Docs | Runner notes in `docs/webui-integration-matrix.md`; `docs/installation.md` points at the `CPU-only link` job |

## 3. Design Decisions

- Lock order is CI flock, then `gpu-lock`, then `nvidia-smi`. The cooperative fd 9 is inherited through `gpu-lock`'s `exec`, so both locks are held for the whole body. A quiet-host wait (#1949) slots in at the top of the body script without re-plumbing.
- A lock timeout is told apart from a body failure by a marker file the body writes first, not by exit code 75: the body's own exit status is passed through unchanged, and a nonzero exit with no marker fails the step with `::error::` and `gpu-lock status`.
- The log prints the holder and a timestamp before the wait and the seconds waited when the body starts, so a wait is visible in the job log.
- `CARGO_INCREMENTAL=0` instead of clean-and-retry on link failure, which the issue rejected because it hides the cause and pays a full rebuild. Other GB10 target directories keep the profile default.
- `cpu-link` gets its own filter because the code #2108 fixed (`src/lib/mlx-cpp/turbo/kv_inplace_write.cpp`) is not matched by `rust`. No overlay excludes: the job costs seconds warm. It keeps incremental compilation, since the 6-second warm figure was measured with it.

## 4. Validation

- `actionlint` 1.7.7 (the version `workflow-lint` pins) with `-shellcheck='shellcheck --severity=error'`: clean. At the default severity the new job and step add no findings.
- YAML parses; seven jobs have `runs-on: GB10`.
- The step script, extracted from the YAML with `--wait` shortened to 3 seconds and with stub `nvidia-smi` and `python3`, was run in seven cases: no `gpu-lock` (body runs, exit 0, no oom adjustment); a stub `gpu-lock` that times out (exit 1 with the holder); the real `gpu-lock` script pointed at a private lock directory when free (exit 0, `oom_score_adj=1000`), held for 2 seconds (waits about 1 second, then passes), and held past the wait (exit 1 with the holder); a body that itself exits 75 under the lock (exit 75 passed through, not reported as a lock timeout); and a body failure without `gpu-lock` (exit 3 passed through).
- Not validated locally: the GB10 run itself and the warm build time with `CARGO_INCREMENTAL=0`. The first run after the change rebuilds the workspace crates, so the warm figure comes from the second run.

## 5. Coordination

#1949 proposes a quiet-host wait in the same precondition block, and #1925 is related. Neither is implemented here; the ordering above leaves #1949 a single insertion point.
