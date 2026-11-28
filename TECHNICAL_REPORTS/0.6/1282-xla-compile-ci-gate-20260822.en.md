# Technical Report: PR #1282 - Compile the OpenXLA feature combinations in CI

## Executive Summary

No CI job compiled any XLA feature, and two defects reached `main` through that gap. This adds a compile gate on the self-hosted GB10 runner covering `cuda,xla-iree` and `xla-diagnostics`.

Two decisions carry the report. The gate denies `unused_imports` rather than every warning, because measured on the current tree a full `-D warnings` policy fails immediately and a job that is red on arrival stops being a gate. And the gate was observed failing on a deliberately reintroduced defect before being merged, because a job that only ever passes is not evidence that it catches anything.

## 1. Problem Statement

`ci.yml` runs `deny`, `fmt`, crate-version, kernel-dtype-key, MLX-pin and cross-repo-ref jobs and never builds the crate. `pipeline-parallel-ci.yml` runs clippy on default features. `nightly-verify.yml` runs clippy with `metal,accelerate`. The OpenXLA serve worker sits behind `#[cfg(feature = "xla-iree")]`, so no check run had ever compiled it.

| defect | class | found by |
| --- | --- | --- |
| `ModelRequest::PromptCacheWarmup` unhandled in the OpenXLA worker | E0004 | hand, during an unrelated rebase |
| dead `load_weights_from_dir_with_filter` re-export | `unused_imports` | hand, during the same rebase |
| integration tests could not link under `cuda,xla-iree` | link error | hand, while validating another PR |

## 2. Technical Decisions

### 2.1 The provisioning question the issue raised does not exist here

The issue treated IREE provisioning as the real decision: an actions cache keyed on the pinned revision, a container image, or a self-hosted runner that already has one. Inspection settled it. The GB10 runner is the same host that carries `~/.cache/mlxcel/iree-cuda-<version>`, and `scripts/iree/setup-cuda.sh` is idempotent, logging "reusing runtime build" when the tree is present. So the job provisions by calling the script: warm runners pay nothing and a fresh one pays the one-time build. `xla-diagnostics` implies `cuda`, which points at the same runner anyway.

### 2.2 Deny the lint that let a defect through, not every lint

Measured before choosing:

```text
cargo clippy --features cuda,xla-iree --lib --tests -- -D warnings
  -> 4 errors in mlxcel-xla alone, before reaching mlxcel
cargo check --features cuda,xla-iree --all-targets
  -> ~10 dead-code warnings across both crates
```

`-D warnings` would therefore have landed red. A gate that is red on arrival is routed around and then ignored, which is worse than no gate because it also produces false confidence in a green summary elsewhere.

`unused_imports` is the exact class of the dead re-export, and compile errors cover the other defect, so both historical breaks are caught with a policy that is green today. Broadening needs the dead-code backlog cleared and is left as separate work rather than smuggled in here.

### 2.3 A `$GITHUB_ENV` format bug worth recording

`setup-cuda.sh --env` emits shell `export VAR=value` lines. `$GITHUB_ENV` expects bare `VAR=value`, so appending the output directly would have defined variables literally named `export IREE_CUDA_HOME`. The failure mode is misleading rather than obvious: `build.rs` would abort claiming no IREE distribution is configured, on a runner that has one. The prefix is stripped with `sed` and the reason is recorded at the call site.

### 2.4 State what the gate does not cover, in the workflow file

A gate that closes part of a gap invites the assumption that the gap is shut. Three exclusions are written where someone editing the job will see them: no test execution, since running the XLA suites needs a GPU not contended with development work on the same host; no full warning policy, for the reason above; and no macOS or `IREE_DIST` build, since neither has a runner. The third is why the link failure above would still not be caught: `cargo check` never links.

## 3. Change Summary

| File | Change |
| --- | --- |
| `.github/workflows/ci.yml` | `xla-compile` job on GB10, `RUSTFLAGS: -D unused_imports`, separate persistent `CARGO_TARGET_DIR`, `MLX_CUDA_ARCHITECTURES: 121`, path filter extended to `build.rs` and `scripts/iree/**` |
| `tests/molmo2_xla_vision_parity.rs` | Two dead-code warnings this repository's own parity test introduced under the non-diagnostics feature set, silenced so the gate starts from a clean tree |

## 4. Review Findings

The `$GITHUB_ENV` bug in 2.3 was caught by reading the emitted format rather than by a run, which matters because the job would still have failed, just for a reason pointing at the wrong subsystem.

`MLX_CUDA_ARCHITECTURES` is pinned to `121` rather than left to auto-detection, which yields `121a` on this host. The two are numerically identical here, but releases build `121`, and a gate should compile what ships.

## 5. Validation

Locally, with the job's own commands and `RUSTFLAGS`:

```text
RUSTFLAGS="-D unused_imports" cargo check --features cuda,xla-iree --all-targets                        -> 0
RUSTFLAGS="-D unused_imports" cargo check --no-default-features --features xla-diagnostics --all-targets -> 0
```

On the PR, the job was driven through all three states rather than only observed passing:

| commit | change | verdict |
| --- | --- | --- |
| `763bc8a0` | the job as proposed | SUCCESS |
| `6f193333` | `PromptCacheWarmup` arm removed | FAILURE |
| `f88709bc` | revert | SUCCESS |

The failing run reported `error[E0004]: non-exhaustive patterns: ModelRequest::PromptCacheWarmup { .. } not covered`, the same error the gap let through. The middle commit and its revert stay in the branch history deliberately: they cancel out in the squash merge, and until then they are the evidence.

## 6. Related Work

Issue #1270 is closed by this PR. It was originally filed as a bug titled after the compile break, closed by the fix for that break, then reopened and retitled once it was clear only one of its four acceptance criteria had been met. The remaining three were this job, its documented feature matrix and provisioning strategy, and the demonstration in section 5.

The uncovered classes named in 2.4 are the honest remainder. The link failure in particular needs a job that actually links a binary, which is a different cost profile from a compile check and deserves its own decision rather than being folded in here.
