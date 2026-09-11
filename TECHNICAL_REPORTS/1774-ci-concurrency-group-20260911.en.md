# Technical Report: PR #1780 - chore(ci): add a concurrency group so superseded runs stop stacking

**Date**: 2026-09-11
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed
**Languages**: YAML (GitHub Actions workflow)
**Risk Level**: Medium (shared CI infrastructure every contributor depends on; no product code touched)

---

## Executive Summary

`.github/workflows/ci.yml` carried no workflow-level `concurrency` key, so every push queued a full CI run and nothing superseded anything. That is ordinarily a minutes-and-money problem, but four jobs in this file run on `GB10`, the single self-hosted Linux runner that also serves the release build. A run for a commit nobody is waiting on any more therefore did not merely sit idle, it held a place in line ahead of the runs that mattered.

This change adds one workflow-level `concurrency` block keyed on workflow plus ref, with `cancel-in-progress` expressed conditionally so that pull-request refs supersede their predecessors while an in-flight run on `main` is never cancelled. It touches no other workflow and no other job.

---

## Problem Statement

Measured over 2026-09-10 KST from `gh run list`: 113 CI runs were created in 24 hours, and at 2026-09-10T03:18:40Z fourteen runs were open at once, eleven of them stacked on just four branches. Six runs in that window ended `cancelled`, every one superseded by a newer push on its own branch and cancelled by hand to free the runner.

The contention had a second-order effect worth recording, because it is what makes this a correctness concern and not only a throughput one. On 2026-09-10 at 02:15:54 KST the runner service was OOM-killed mid-job, and the `cargo-clippy` job of run `34381499951` is recorded by GitHub with zero steps and no logs. That is indistinguishable from a real lint result until someone reads the service journal. Its clippy job did not start until 8h38m after the run was created, with four branches waiting behind it.

Two jobs in the same file had already solved the problem locally, which is what established the pattern as accepted here: `xla-link` and `cuda-sm70-compile` each carry a per-PR `concurrency` group, on the reasoning that they are the jobs long enough to hold GB10. Nothing covered `clippy` or `xla-compile`, and nothing covered the run as a whole.

---

## Change Summary

A single block inserted between `on:` and `permissions:`, plus the rationale comment the file's heavily-commented style calls for:

- **Group**: `ci-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}`. The `pull_request.number || ref` shape is what `pipeline-parallel-ci.yml` and both per-job groups in this file already use, rather than a third invented form. For a `push` to `main` the left operand is null, so the fallback `github.ref` applies and a `main` push run lands in a group distinct from every PR group.
- **Cancellation**: `cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}`. Expression-valued `cancel-in-progress` is documented behavior, the sanctioned example in GitHub's own docs being `${{ !contains(github.ref, 'release/') }}`.
- **Untouched**: the per-job groups in `xla-link` and `cuda-sm70-compile`. Workflow-level and job-level groups compose, and those two are keyed differently on purpose. Reconciling the schemes is a separate change.

---

## Technical Decisions

**Not cancelling in-progress runs on `main`, decided explicitly rather than by default.** This was the one open question the issue raised, and the expression takes a side. A push-to-main run is the run that catches whatever a stale PR base hid, and it is the last gate before a release builds from that commit; cancelling it because the next merge landed ninety seconds later would discard the only signal nothing else produces. The argument on the other side is real, namely that this repository merges frequently enough for back-to-back merges to queue GB10 jobs, but the cost is bounded rather than unbounded. Cancelling a *pending* run is inherent to grouping, while `cancel-in-progress` governs only the *running* one, so `main` holds one in-flight run plus at most one pending run regardless of merge rate.

The residual trade is recorded in the comment so a future reader does not rediscover it as a surprise: an intermediate `main` commit can lose its run while still pending. Whatever commit sits at the head of `main` is always the newest in the group, so it always gets a run, which is the property that matters before a release is cut. `nightly-verify.yml` made the same trade for the same reason.

**Leaving `release.yml` alone, stated rather than silently taken.** `release.yml` is the only other contributor to the GB10 queue, through its `build-linux-cuda` job. It triggers only on `release: published` and `workflow_dispatch`, so it does not stack from pushes, and grouping an artifact-producing release build risks cancelling a run whose output is shipped. It is also not starved by this change, because cancellation only ever removes CI work from the queue ahead of it.

**Leaving `python.yml` and `update_homebrew_formula.yml` alone.** Neither has a concurrency group, but neither reaches GB10, so neither participates in the contention this issue describes. Grouping them would be a hosted-runner-minutes change judged on its own merits.

---

## Workflow Inventory

| Workflow | Workflow-level `concurrency` before this change | GB10 jobs |
|---|---|---|
| `ci.yml` | none (job-level only, in `xla-link` and `cuda-sm70-compile`) | 4: `clippy`, `xla-compile`, `xla-link`, `cuda-sm70-compile` |
| `release.yml` | none | 1: `build-linux-cuda` |
| `nightly-verify.yml` | `group: nightly-verify`, `cancel-in-progress: false` | none (self-hosted macOS) |
| `pipeline-parallel-ci.yml` | `group: pp-ci-...`, `cancel-in-progress: true` | none (`pp-three-host`) |
| `python.yml` | none | none (ubuntu-latest) |
| `update_homebrew_formula.yml` | none | none (macos-latest) |

---

## Validation

Verified locally:

- The file parses and the block resolves to the intended group string and expression. The four `GB10` jobs and both surviving per-job groups were re-read out of the parsed tree to confirm nothing else moved.
- `actionlint` 1.7.7 reports byte-identical findings before and after the change: 8 pre-existing (unknown self-hosted runner labels, shellcheck `info` notices inside `run:` scripts), 0 introduced. The comparison was made against `origin/main:.github/workflows/ci.yml` with line numbers stripped, since the change is a pure insertion.
- That clean result was confirmed to be load-bearing rather than vacuous. A negative control containing the same two expressions with `github.reff` and a bogus context field makes `actionlint` fail on the concurrency block, so its expression type-checker does inspect this construct.
- `git diff --name-only origin/main` returns only `.github/workflows/ci.yml`, confirming the "no change to `release.yml`, `nightly-verify.yml`, or `pipeline-parallel-ci.yml`" criterion.

**Not verified, and deliberately not claimed.** That a newer push actually supersedes an in-flight run is cross-branch behavior that only two racing pushes against live GitHub infrastructure can demonstrate, and a green CI run on this PR would not show it. A staged double-push was considered and rejected on cost: `.github/workflows/ci.yml` appears in all four `changes` path filters (`rust`, `mlx_pin`, `xla_link`, `cuda_arch`), so every push to this branch starts all four GB10 jobs, two of them carrying `timeout-minutes: 120`. Manufacturing the heaviest available load on the single shared runner in order to prove a point about relieving load on it is the wrong trade, and the issue's proposed verification recipe did not account for that filter coverage.

The behavior is observable without staging anything, because this PR's branch received a second push carrying this report as ordinary required work. The resulting supersession, or its absence, is visible in `gh run list --workflow ci.yml --branch chore/issue-1774-ci-concurrency` and is reported on the PR rather than asserted here in advance.

---

## Follow-ups

- Bounding the runner service with `MemoryMax=` and `MemoryAccounting=yes` in a systemd drop-in for `actions.runner.lablup.lablup-dgxspark21.service` would turn an OOM into a job that dies inside the runner, with logs, instead of the whole agent being reaped. This is host configuration rather than a repository file, so it belongs in the runner runbook. The issue called it out and explicitly did not block on it.
- Whether `clippy` should run on GB10 at PR time at all is tracked separately in #1283.
- Reconciling the workflow-level group with the two per-job groups into one scheme remains available as a later cleanup.
