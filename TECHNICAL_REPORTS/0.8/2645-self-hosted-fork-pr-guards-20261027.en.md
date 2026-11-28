# PR #2645: Skip self-hosted CI jobs for pull requests from forks

**Date**: 2026-10-27
**Status**: Completed
**Languages**: GitHub Actions YAML, Markdown
**Risk Level**: Medium

## Executive Summary

Eight `ci.yml` jobs on the shared self-hosted runners (seven on `GB10`, plus `rocm-build` on `rocm-gfx1151`) ran the code of pull requests opened from forks as the runner's own user. Each now also requires the PR's head repository to be the base repository, the guard `pipeline-parallel-ci.yml` already used, so a fork PR that leaves the workflow alone skips them. The guard does not bind a fork PR that edits the workflow itself; that case is held back only by the repository's Actions fork-PR approval policy, which the PR documents but cannot change (issue #2638).

## 1. Problem Statement

Every self-hosted job was gated on `github.repository == 'lablup/mlxcel'`. On a `pull_request` event `github.repository` is the base repository, so the condition is true for a PR opened from a fork. The guard only keeps the jobs from queueing forever inside a fork of the repository, which has no `GB10` runner.

With the guard passing, checkout fetches the PR merge commit and the jobs run code the PR author controls: `cargo clippy`, `cargo check` and `cargo test --no-run` execute every crate's `build.rs` (`mlxcel-core`'s builds MLX through cmake), `xla-compile` and `xla-link` run `scripts/iree/setup-cuda.sh`, and `webui-installed-artifact` runs `scripts/webui/*.py` and `pnpm --dir webui install`. The runner is long-lived and shares a uid with gpu-lock-managed development sessions and with the release build. The release job in `release.yml` reuses `$HOME/.cargo-target/mlxcel-cuda-*`, the shared rustup and `~/.cargo/bin`, and its purges are keyed on marker files a same-uid process can write. A build script from a fork could therefore plant artifacts that a later release links.

The trigger is `pull_request`, not `pull_request_target`, so a fork run already gets a read-only `GITHUB_TOKEN` and no secrets. The exposure was the runner host, not the repository credentials.

## 2. Technical Decisions

**Guard shape.** Each of the eight jobs gains

```yaml
&& (github.event_name != 'pull_request'
    || github.event.pull_request.head.repo.full_name == github.repository)
```

ANDed after the repository guard and before the existing path-filter and `ci:full` clause. This is the expression `pipeline-parallel-ci.yml` uses for its own jobs, so the repository now has one fork-guard idiom. The `event_name` disjunct exists because `push` (the only other `ci.yml` trigger today), `schedule` and `workflow_dispatch` have a null `github.event.pull_request`; without it, the comparison would be `null == 'lablup/mlxcel'` and the jobs would stop running on pushes to `main`.

**Ordering against `ci:full`.** The label clause is inside the final `&&` term, so the label can force a same-repository PR's self-hosted jobs but cannot opt a fork PR in. Anyone can ask for a label in a review thread; the guard does not depend on a maintainer remembering not to apply it.

**Deleted head repository.** If the fork is deleted while the PR is open, `head.repo` is null and `full_name` compares unequal to the base, so the jobs skip. The guard fails closed.

**No cascade.** A YAML sweep of `ci.yml` finds exactly the eight jobs whose `runs-on` names `GB10` or `self-hosted`. Each depends only on the hosted `changes` job, no job lists a guarded job in `needs`, and no job aggregates with `if: always()`, so skipping them on a fork PR changes nothing else. Hosted jobs (fmt, deny, the contract checkers, `webui-bundle`, `workflow-lint`) still run for fork PRs. Outside `ci.yml`, the only PR-triggered self-hosted job is `three-host-real-model` in `pipeline-parallel-ci.yml`, which already had the guard.

**Advisory messages.** `rocm-ci-status` used to say the ROCm job was the authority; on a fork PR that job never runs, so it now emits a warning that a maintainer must verify the ROCm build separately, and exits 0. The event name, head repository and base repository reach the script through `env:` rather than `${{ }}` interpolation inside `run:`. The head repository name is fork-controlled text, and environment variables keep it out of the shell source. `self-hosted-gate-advisory` and `CONTRIBUTING.md` now say that `ci:full` applies to same-repository PRs only.

**What the guard does not cover.** A `pull_request` run takes its workflow files from the PR's merge commit, not from `main`. A hostile fork PR can delete these `if:` lines, or add a new job with `runs-on: GB10`, and its run will execute that edited workflow. Nothing written in `ci.yml` can prevent this. The only barrier is the repository's Actions fork-PR approval policy, which the REST API (`actions/permissions/fork-pr-contributor-approval`) reports as `first_time_contributors`. That setting holds a fork PR's runs only until one of the contributor's contributions has merged; after that, their fork PRs run without approval, including ones that rewrite the workflow. `all_external_contributors` would hold every fork PR's runs for approval. Changing it needs a repository admin and was not part of this PR. The rewritten `clippy` comment states this limit and asks reviewers to read a fork PR's `.github/workflows/**` diff before approving its run. An earlier commit in the PR had dropped the approval-policy reference from that comment and claimed fork code never reached the runner; a follow-up commit (4f5beac6) restored the accurate statement.

**Acceptance item 3 documented, not enforced.** Isolating PR-time CI from release inputs would need a separate uid or `HOME` per job class on the runner host, a host change rather than a workflow edit. Same-repository PRs (authors with write access) still run as the runner's uid; the per-job `$HOME/.cargo-target/mlxcel-*-ci` directories are separated from the release build's `mlxcel-cuda-*` ones only by naming, and rustup and `~/.cargo/bin` are shared. The `clippy` comment records this as remaining exposure.

## 3. Change Summary

- `.github/workflows/ci.yml`: head-repository guard on `clippy`, `cpu-link`, `webui-installed-artifact`, `xla-compile`, `xla-link`, `cuda-sm70-compile`, `cuda-blockfloat` and `rocm-build` (which keeps `vars.ROCM_CI_ENABLED`); fork branch in `rocm-ci-status`; `clippy` comment rewritten to describe both guards, the hostile-PR limit and the same-uid exposure, with `cpu-link` pointing to it; `self-hosted-gate-advisory` comment and warning text scoped to same-repository PRs.
- `CONTRIBUTING.md`: one sentence stating that self-hosted jobs never run for a fork PR and `ci:full` does not change that.

## 4. Trade-offs

A fork PR now merges without the self-hosted signal: no clippy, no CUDA or XLA link check, no installed-artifact WebUI check. A job skipped by its job-level `if:` reports as skipped, which GitHub treats as passing for a required check, so branch protection will not block such a PR either. A maintainer who wants that coverage has to push the fork's commits to a branch of this repository and run CI there. The PR accepts that cost; the alternative was running untrusted build scripts on a host shared with release builds.

## 5. Validation

- `yaml.safe_load` parses `ci.yml`.
- A per-job script confirms every job whose `runs-on` names `GB10` or `self-hosted` carries both the `event_name != 'pull_request'` disjunct and the head-repository comparison (OK for all eight).
- `actionlint` 1.7.7 with `.github/actionlint.yaml` reports nothing on `ci.yml`, the same as `main`.

Unverified: no real fork PR was opened, so the skip on a fork PR and the `rocm-ci-status` fork branch are established by reading the expressions, not by an observed run. The approval-policy recommendation is not applied, so the hostile-PR path described above is still open until an admin changes it.

## 6. Learning Points

- On `pull_request`, `github.repository` names the base repository. A guard written against it distinguishes "running in a fork of this repo" from "running here", not "PR from a fork" from "PR from a branch". The head repository is `github.event.pull_request.head.repo.full_name`.
- An `if:` in a workflow file is not a security boundary against the PR that runs it, because a `pull_request` run uses the PR's copy of the workflow. Protection against a hostile workflow edit has to come from outside the file: the fork-PR approval policy, runner groups restricted to selected workflows or repositories, or ephemeral runners.
- Fork-controlled event fields (head repository name, branch name, PR title) should reach shell steps through `env:`, never through `${{ }}` inside `run:`.
