# Technical Report: PR #1129 - fix(cli): name the real --tp-size flag in the 2D-parallelism error

**Date**: 2026-08-14
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Low

---

## Executive Summary

PR #1129 closes issue #1112. The 2D (pipeline x tensor) parallelism validator rejected an under-specified topology by instructing the user to pass `--tensor-parallel-size > 1`. No mlxcel binary has ever accepted that spelling, so a user who followed the instruction was answered with `error: unexpected argument '--tensor-parallel-size' found`. The message now names `--tp-size`.

The finding that carries the most weight is not the message itself. The same wrong spelling was also in `tests/pp_tp_2d_real_models.rs`, where the non-ignored test `pp_tp_2d_validator_accepts_combination` had been passing **vacuously**: clap rejected the flag at argument parsing, the process died before the validator ran, and the test's only assertion (that a specific old rejection string is absent from the output) held trivially. The test that existed to guard this code path could not have caught this defect, and its green result was evidence of nothing.

---

## 1. Problem Statement

### 1.1 Background

`validate_pipeline_parallel_args` in `src/commands/generate.rs` guards the 2D composition: a tensor-parallel run with `tp_size > 1` must also carry a pipeline topology, either `--pp-size >= 2` or an explicit `--pp-layers` spec. The guard itself is correct. Its message was not:

```rust
"2D parallelism requires --pp-size >= 2 (or an explicit --pp-layers spec) \
 alongside --tensor-parallel-size > 1"
```

The tensor-parallel rank count is `--tp-size`, defined in `src/main.rs` and `src/bin/mlx_server.rs`. `--tensor-parallel-size` exists nowhere as a clap flag or alias.

### 1.2 Existing Issues

- **The one wrong name was more misleading than a vague message.** `--pp-size` and `--pp-layers`, named in the same sentence, are both real. A reader who spot-checked the message found two of three names correct, which is exactly the pattern that makes the third look trustworthy.
- **The test surface reproduced the same error and hid it.** Both tests in `tests/pp_tp_2d_real_models.rs` passed `--tensor-parallel-size` on the command line. The parity test is `#[ignore]`d and needs real weights, so it never ran. `pp_tp_2d_validator_accepts_combination` is not ignored and did run, and passed for the wrong reason: it asserts only that the string `pipeline parallelism does not support tensor parallelism` is absent from stdout and stderr, which is trivially true of a process that died inside clap.
- **A comment above the validator pointed at files that have never existed.** It cited `docs/en/distributed/pipeline-parallelism.md` and `docs/en/distributed/tensor-parallelism.md`. `git log --all -- docs/en` is empty; the directory has never been part of this repository. These were the only two `docs/en/` references anywhere in `src/`.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| User follows the error message and hits a second, unrelated error | Medium | High |
| A future 2D-validator change is "verified" by a test that never reaches the validator | High | Medium |
| Reader follows the comment to a nonexistent operator guide | Low | Medium |

---

## 2. Technical Review

### 2.1 Scope of the Message Change

The `ensure!` condition, the surrounding `tp_size > 1` branch, the `total_ranks` sanity check, and every other arm of `validate_pipeline_parallel_args` are byte-identical. Only the string literal changed, satisfying the issue's fourth acceptance criterion directly.

### 2.2 The Sweep

The issue asked for a sweep of the neighbouring messages on the reasoning that one wrong name suggests checking the rest. Reading them is weaker evidence than asking the binary, so every `--flag` token appearing anywhere in `src/commands/generate.rs` was diffed against the long flags the built binary actually advertises:

```
$ grep -o -- '--[a-z][a-z0-9-]*' src/commands/generate.rs | sort -u \
    | comm -23 - <(mlxcel generate --help | grep -o -- '--[a-z][a-z0-9-]*' | sort -u)
--tensor-parallel-size
```

One line out. `--pp-micro-batch-size`, `--pp-size`, `--pp-layers`, `--estimate-memory`, `--no-memory-check`, `--max-tokens`, `--recommend-quant`, `--surgery` and the rest of the 64 advertised flags all resolve. The sweep is therefore complete for this file rather than sampled.

### 2.3 The Vacuous Test

The failure mode is worth stating precisely, because it is a shape that recurs. The test asserts a **negative**: that a particular rejection string does not appear. A negative assertion over process output is satisfied by every path that produces different output, including every path that never reaches the code under test. Adding the flag-name defect to the command line moved the process's death from the validator to the argument parser, and the assertion did not notice.

The fix is to make the precondition explicit. The test now asserts, before the original check, that the arguments were not rejected at parse time:

```rust
assert!(
    !stderr.contains("unexpected argument") && !stderr.contains("unrecognized"),
    "the 2D flags were rejected by the argument parser, so the validator \
     was never reached:\nstdout={stdout}\nstderr={stderr}"
);
```

This was verified against the real binary rather than assumed: clap emits `error: unexpected argument '--tensor-parallel-size' found`, so the guard's substring matches the observed text. `unrecognized` is carried as a defensive second spelling.

### 2.4 Compatibility

No behavior changes for any command line that worked before. The validator accepts and rejects exactly the same inputs; only the text of one rejection differs. There is no CLI surface change, no serialization change, and no API change.

---

## 3. Technical Decisions

### 3.1 Fix the Test Rather Than Only the Message

| Option | Pros | Cons |
|---|---|---|
| Fix the `ensure!` string only | Smallest possible diff; satisfies the issue as literally written (its criterion is scoped to `src/`) | Leaves a non-ignored test passing a flag the binary rejects, and leaves the vacuous pass in place for the next person |
| **Chosen: fix the message, the test's flag, and the test's assertion** | The code path the test names is actually exercised; a reintroduction of the same defect now fails | Slightly wider diff than the issue's literal scope |

The test is not incidental to this issue. It is the artifact that should have caught the defect and did not, and it failed for the same root cause. Fixing the message while leaving the test unable to detect the message being wrong again would close the issue without closing the gap.

### 3.2 Repoint the Comment Rather Than Delete It

The issue offered both: repoint at `docs/distributed.md`, or drop the references if that document does not cover 2D composition. `docs/distributed.md` was read to decide. It documents tensor parallelism and pipeline parallelism in separate sections, with `--tp-size`, `--pp-size`, `--pp-layers` and `--pp-micro-batch-size` all present, but it contains no section on composing the two.

Deleting the reference would lose a genuinely useful pointer; repointing it silently would overstate the document. The comment now points at `docs/distributed.md` and states that the 2D composition is not written up there yet, which is the accurate version of both.

### 3.3 Keep the Historical Flag Name Out of the Tree

The new guard's comment explains why it exists by referring to the old spelling. It is worded as "a tensor-parallel flag spelling the binary has never accepted" rather than quoting the literal string, so `grep -rn -- "--tensor-parallel-size"` stays empty across the whole repository, not just under `src/`. A future sweep of the same kind will not rediscover this as a live hit.

---

## 4. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 3 |
| Lines added | +27 |
| Lines deleted | -14 |
| Behavior changes | 0 |

### Changes by Area

| Area | File | Summary |
|---|---|---|
| CLI | `src/commands/generate.rs` | `ensure!` message names `--tp-size`; validator comment repointed from two nonexistent `docs/en/` paths to `docs/distributed.md` with an accurate note on 2D coverage |
| Tests | `tests/pp_tp_2d_real_models.rs` | Both tests use `--tp-size`; `pp_tp_2d_validator_accepts_combination` gains a parse-rejection guard so it cannot pass without reaching the validator; stale line-number reference replaced with the function name |
| Documentation | `CHANGELOG.md` | Entry under `## [Unreleased]` / `### Fixed` |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `1593eb8a` | fix | fix(cli): name the real --tp-size flag in the 2D-parallelism error |

---

## 5. Validation and Follow-up

### Passed

- `MLX_CUDA_ARCHITECTURES=121 cargo test --profile test-fast --features cuda --test pp_tp_2d_real_models`: 1 passed, 0 failed, 1 ignored.
- Old spelling rejected, as the issue predicted: `mlxcel generate -m /nonexistent -p x -n 1 --pp-size 2 --tensor-parallel-size 2` prints `error: unexpected argument '--tensor-parallel-size' found`.
- New spelling accepted: the same command with `--tp-size 2` parses and proceeds to model resolution, failing there on the deliberately nonexistent path.
- `grep -rn -- "--tensor-parallel-size" src/` empty; `grep -rn "docs/en/" src/` empty.
- `cargo fmt --all -- --check` clean.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings` clean.

### Follow-up Candidates

- The vacuous-negative-assertion shape is not unique to this test. Any test whose only assertion is that a string is absent from process output passes when the process dies early for an unrelated reason. A sweep for that pattern across `tests/` would be a self-contained follow-up.
- `docs/distributed.md` has no section on 2D (PP x TP) composition, which is why the repointed comment has to qualify itself. Writing that section would let the qualification be dropped.
