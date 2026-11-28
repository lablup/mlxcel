# Technical Report: PR #2021 - Correct CONTRIBUTING's PR-time CI Gate List Against ci.yml

**Date**: 2026-09-28
**Status**: Completed
**Languages**: Markdown
**Risk Level**: Low

## Executive Summary

`CONTRIBUTING.md` told contributors that clippy was "NOT gated at PR time" and that PR-time CI ran "only the cheap gates": `cargo fmt`, `cargo deny`, and a crate-version check. That description predates `#1285`, which re-added a `clippy` job to `.github/workflows/ci.yml` and gated it on Rust-touching pull requests, alongside roughly a dozen further jobs the paragraph never named. PR #2021 rewrites the inline comment on the local clippy command and the PR-time CI paragraph to match the current workflow file, enumerating which jobs run unconditionally, which are path-filtered, which run on the shared self-hosted `GB10` runner, and which are advisory-only.

## 1. Problem Statement

### 1.1 Background

`ci.yml` grew substantially since the CONTRIBUTING.md paragraph was last accurate: a `clippy` job was re-added at PR time (`#1285`, after `#21`/`#23` had dropped it for cost reasons), and separate jobs now cover CUDA/MLX overlay compilation, the WebUI bundle and its installed-artifact smoke test, the pinned MLX commit parsers, a parked ROCm gate, and several toolchain-free consistency checks. None of that growth was reflected back into the contributor-facing doc, so a contributor who trusted the paragraph would be surprised by a red `clippy` check on their PR.

### 1.2 Existing Issue

- **Stale gate list**: the paragraph named three gates (`fmt`, `deny`, crate-version check) where the workflow now runs roughly twenty jobs, several of them gating.
- **Wrong clippy claim**: the doc said clippy was not enforced; `ci.yml`'s `clippy` job enforces `cargo clippy -p mlxcel --lib --tests -- -D warnings` on Rust-touching PRs.
- **No visibility into path-conditional and advisory jobs**: a contributor reading the old paragraph had no way to know which jobs would actually run against their diff, or that some jobs (`cross-repo-refs`, `rocm-ci-status`, `self-hosted-gate-advisory`) report warnings without ever failing a PR.

### 1.3 Risk Assessment

Low. This is a documentation-only change; no source, build script, or workflow file is touched, and no gate's actual behavior changes. The risk is entirely in the write-up being wrong again, which is why the change was verified against the live workflow file and went through a full review cycle.

## 2. Change Summary

| Item | Value |
|------|-------|
| Files changed | 1 |
| Lines added | 2 |
| Lines deleted | 2 |

- `CONTRIBUTING.md:46`: the inline comment on `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings` now says the narrower `-p mlxcel --lib --tests` slice is gated in CI, and that the fuller invocation stays the contributor's own responsibility.
- `CONTRIBUTING.md:50`: the PR-time CI paragraph was rewritten to enumerate, from the current `ci.yml`: the `changes` path-filter job; seven toolchain-free checks that gate every PR unconditionally (`crate-versions`, `kernel-dtype-keys`, `binary-assets`, `cuda-arch-lists`, `license-headers`, `llama-compat-manifest`, `webui-contract`); `fmt`/`deny`/`clippy` gated on Rust changes; the further `GB10` jobs, each gated on its own path filter (`xla-compile` on Rust changes, `xla-link` on the IREE link recipe, `cuda-sm70-compile`/`cuda-blockfloat` on the CUDA-architecture paths, `webui-installed-artifact` on Rust or WebUI changes); the hosted `webui-bundle` and `mlx-pin` path filters; the parked `rocm-build` gate and its `rocm-ci-status` advisory; `workflow-lint` and the advisory `self-hosted-gate-advisory`; and the always-running, advisory `cross-repo-refs`. The existing `pipeline-parallel-ci.yml` and `nightly-verify.yml` mentions were preserved and re-verified as still accurate.
- The closing sentence, previously "CUDA verification is not gated at PR time either; that stays exclusive to `release.yml`", was corrected: `release.yml` has no `cargo test` step, so the exhaustive CUDA and ROCm test suites are gated by `make verify-test-cuda` and `make verify-rocm` locally, not by `release.yml`.

## 3. Technical Decisions

### 3.1 Verify against the live workflow file, not the issue's own enumeration

**Context**: Issue #1702's "Implementation Notes" already listed several jobs (`crate-versions`, `kernel-dtype-keys`, `llama-compat-manifest`, `mlx-pin`, `cross-repo-refs`, `xla-compile`, `xla-link`, `cuda-sm70-compile`), but the issue was filed 2026-09-08 and recent commits changed CI gating in the meantime (GB10 runner scoping, a new ROCm gate).

**Rationale**: The task explicitly called out that the issue's job list might be stale. Every job name, `if:` condition, `runs-on` value, and path filter was re-derived directly from `.github/workflows/ci.yml` on current `main`, including the `dorny/paths-filter` definitions in the `changes` job, rather than trusted from the issue body.

**Outcome**: The first draft still got one thing wrong by extrapolation rather than direct verification: it grouped `cuda-sm70-compile` and `cuda-blockfloat` under "Rust-path-gated" jobs alongside `clippy` and `xla-compile`. `pr-reviewer` caught this against the workflow file: both jobs actually gate on the `cuda_arch` filter (`src/lib/mlx-cpp/**` minus its ROCm and Metal subtrees, plus the MLX build scripts), which contains no `*.rs` glob, so an ordinary Rust source change does not trigger them. This was fixed in a follow-up commit before the PR body was refreshed to match.

### 3.2 Enumerate rather than summarize

**Context**: A shorter rewrite could have said "CI now runs many more gates than before, including a Rust-triggered clippy job" without naming each one.

**Rationale**: The paragraph rotted once already because a summary went stale silently. Naming each job by its actual identifier makes a future drift (a renamed job, a changed path filter) detectable by grep against `ci.yml`, and matches the file's own existing style of naming jobs and `#N` issue references inline rather than only describing them.

**Trade-off**: The rewritten paragraph is roughly 2.5x the length of the original (1104 to about 2530 characters as one physical line, consistent with the file's no-hard-wrap convention). This was weighed against brevity during review and kept, since every named job is a fact a contributor can act on (which check will run on their diff) rather than ceremony.

## 4. Validation

- `python3 scripts/ci/check_cross_repo_refs.py`: passes; flags `#1285` for manual verification (no `GH_TOKEN` in this environment), independently confirmed real via `gh pr view 1285` (merged, title "fix(lint): clear the err_expect on main and gate clippy at PR time").
- `python3 "$HOME/.claude/skills/commit-conventions/scripts/validate_body.py" CONTRIBUTING.md`: no new violations; both edited lines remain single physical lines. Pre-existing hard-wrap findings elsewhere in the file (lines 109-121) are unrelated and out of scope.
- `gh api repos/lablup/mlxcel/actions/variables/ROCM_CI_ENABLED` returned 404 and `gh api repos/lablup/mlxcel/actions/runners` returned zero runners, confirming the "parked" ROCm gate claim.
- Full review cycle: `pr-reviewer` found and fixed one HIGH finding (the `cuda-sm70-compile`/`cuda-blockfloat` trigger, above); `pr-security-checker` passed with no findings (prose-only diff, no secrets, links, or gate-weakening claims); `pr-finalizer` confirmed no non-English counterpart of `CONTRIBUTING.md` exists and no other doc restates the stale claim.
- No Rust build or test suite applies; the PR's own CI run shows every toolchain-free check passing and every Rust/WebUI/CUDA/ROCm job `SKIPPED`, matching what the corrected doc now predicts for a non-Rust-touching PR.

## 5. Related Work

- Issue #1702: source issue for this change.
- PR #1285: re-added the `clippy` job to PR-time CI; the event this documentation update reflects.
- Issues #21 / #23: the original removal of clippy and the test suite from PR-time CI, referenced in the corrected paragraph for historical context.
- Issue #1992: introduced the `changes` job's `predicate-quantifier: some-with-excludes` path-filter behavior that several of the enumerated jobs depend on.
