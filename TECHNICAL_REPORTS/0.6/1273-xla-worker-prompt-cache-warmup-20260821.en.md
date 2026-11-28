# Technical Report: PR #1273 - Handle PromptCacheWarmup in the OpenXLA worker

## Executive Summary

`main` did not compile under `--features xla-iree`. PR #1154 added the `ModelRequest::PromptCacheWarmup` variant and updated the three workers it knew about, but not the OpenXLA one, leaving a non-exhaustive `match`. The fix is a five-line arm that drops the job.

The change is trivial. What is worth recording is that a compile error sat in `main` undetected, and why the repository's CI cannot see that class of breakage at all.

## 1. Problem Statement

```text
error[E0004]: non-exhaustive patterns: `model_provider::ModelRequest::PromptCacheWarmup { .. }` not covered
  --> src/server/batch/xla_worker_admission.rs:482:15
```

`src/server/batch/mod.rs` gates `xla_preprocess` and `xla_worker` behind `#[cfg(feature = "xla-iree")]`, and no CI workflow enables any XLA feature. `ci.yml` runs `deny`, `fmt`, crate-version, kernel-dtype-key, MLX-pin and cross-repo-ref jobs with no workspace `cargo check`. `pipeline-parallel-ci.yml` runs clippy on default features. `nightly-verify.yml` runs clippy with `metal,accelerate`. None compile the module, so the break was invisible to every check run that reported green on #1154.

It surfaced only when PR #916 was rebased onto `main` and its branch was built with the XLA features.

## 2. Technical Decisions

### 2.1 Drop the job rather than forward or error

Only `BatchScheduler` owns a prompt cache. The OpenXLA worker has no snapshot state to warm, so there is nothing to forward the request to and nothing has gone wrong when one arrives. `diffusion_worker.rs` and `florence2_worker.rs` already reached the same conclusion for the same reason, so this arm matches theirs, including a comment stating why rather than leaving a silent empty block.

### 2.2 Do not touch admission state

The arm deliberately does nothing at all. `PromptCacheWarmup` carries no `response_tx` and no queue reservation, so no caller is waiting and no gauge needs to be released. Touching pending-image or pending-audio state to "clean up" would be inventing work that the variant's contract says does not exist.

### 2.3 Split from PR #916

This was found while rebasing #916 and was initially fixed on that branch, bundled with an unrelated re-export gate. It was separated out because #916 is a large feature PR and this is a compile fix for `main` that other branches also need. Splitting also exposed a misattribution: the re-export gate that travelled with it was not a `main` defect at all, but something #916 itself introduced, and it stayed on that branch.

## 3. Change Summary

| File | Change |
| --- | --- |
| `src/server/batch/xla_worker_admission.rs` | `ModelRequest::PromptCacheWarmup { .. }` arm in `XlaServeWorker::handle`, dropping the job with the reason recorded |

## 4. Review Findings

No review findings. The change is one match arm with no behavioral surface beyond making the crate compile.

## 5. Validation

| command | before | after |
| --- | --- | --- |
| `cargo check --features cuda,xla-iree --lib` | E0004 | passes |
| `cargo check --features cuda --all-targets` | passes | passes, no warnings |

Both were run on this branch alone, with no other build sharing the working tree. That detail mattered: an earlier verification attempt in this session produced a false failure because two scripts checked out different revisions in the same tree concurrently.

## 6. Related Work

Issue #1270 tracks the durable half of this problem, which is that nothing in CI compiles an XLA feature combination. It was closed by this PR's `Closes` keyword and then reopened and retitled, because only the first of its four acceptance criteria was met here. The remaining three are the CI job itself.

The coverage gap has now let two defects reach `main`: this one, and the link failure in #1274, which `cargo check` cannot catch because it never links. That pair is the argument for the job.
