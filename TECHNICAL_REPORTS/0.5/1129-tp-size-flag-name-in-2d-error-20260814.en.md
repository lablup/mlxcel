# Technical Report: PR #1129 - fix(cli): name the real --tp-size flag in the 2D-parallelism error

**Date**: 2026-08-14
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

PR #1129 closes issue #1112. The 2D (pipeline x tensor) parallelism guard in `validate_pipeline_parallel_args` carried a message naming `--tensor-parallel-size`, a spelling no mlxcel binary accepts. The message now names `--tp-size`.

Two findings from review materially changed the shape of this PR, and both are worth more than the original defect.

**The guard is unreachable.** The `ensure!` condition is the exact logical negation of the early return eight lines above it, so it can never fail and the message can never be emitted. No user has ever seen the wrong flag name. This is a latent text defect, not a user-facing bug, which is why the change ships with no CHANGELOG entry.

**The `docs/en/` paths the issue called nonexistent are real.** `mkdocs.yml` sets `docs_dir: docs/en` and its `nav:` names both files the comment cited. They are pages of the published operator manual whose sources live in a separate documentation repository, a split `docs/README.md` documents explicitly as "deliberate, not drift". An earlier draft of this PR deleted those references on the issue's false premise, which is precisely the failure mode PR #1122 recorded as a risk. They are retained and explained instead.

---

## 1. Problem Statement

### 1.1 Background

`validate_pipeline_parallel_args` in `src/commands/generate.rs` guards the 2D composition. Its structure is:

```rust
if pp.pp_layers.is_none() && pp.pp_size <= 1 {
    return Ok(());
}
// ...
if tp_size > 1 {
    ensure!(
        pp.pp_size >= 2 || pp.pp_layers.is_some(),
        "2D parallelism requires --pp-size >= 2 (or an explicit --pp-layers spec) \
         alongside --tensor-parallel-size > 1"
    );
```

The tensor-parallel rank count is `--tp-size`, defined in `src/main.rs` and `src/bin/mlx_server.rs`. `--tensor-parallel-size` exists nowhere as a clap flag or alias.

### 1.2 Existing Issues

- **The message named a flag the binary rejects.** `--pp-size` and `--pp-layers`, named in the same sentence, are both real, which is the pattern that makes a third name look trustworthy.
- **The guard cannot fire.** Past the early return, `pp_layers.is_some() || pp_size >= 2` holds by construction, and that is exactly the `ensure!` condition. The repository's own unit test records the consequence: `validate_pipeline_parallel_args_rejects_2d_without_pp_enabled` (`src/commands/generate_tests.rs`) sets `pp_size = 1, tp_size = 2` and asserts `is_ok()`, with a comment noting the validator returns early. The test name says "rejects" and the assertion says the opposite.
- **The integration test passed the same nonexistent flag, and passed anyway.** Both tests in `tests/pp_tp_2d_real_models.rs` used `--tensor-parallel-size`. The parity test is `#[ignore]`d. `pp_tp_2d_validator_accepts_combination` was not, and passed vacuously: its only assertion was that a particular old rejection string is absent from the output, which is trivially true of a process clap killed at argument parsing.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| A future change makes the guard reachable and ships the wrong flag name to users | Medium | Medium |
| A 2D-validator change is "verified" by a test that never reaches the validator | High | Medium |
| Someone deletes the cross-tree manual references believing them dangling | Medium | Medium |

---

## 2. Technical Review

### 2.1 Scope of the Message Change

The `ensure!` condition, the `tp_size > 1` branch, the `total_ranks` sanity check, and every other arm of the function are byte-identical. Only the string literal changed, satisfying the issue's fourth acceptance criterion directly.

### 2.2 The Sweep

Every `--flag` token appearing anywhere in `src/commands/generate.rs` was diffed against the long flags the built binary advertises, rather than read by eye:

```
$ grep -o -- '--[a-z][a-z0-9-]*' src/commands/generate.rs | sort -u \
    | comm -23 - <(mlxcel generate --help | grep -o -- '--[a-z][a-z0-9-]*' | sort -u)
--tensor-parallel-size
```

One line out of 64 advertised flags. `--pp-micro-batch-size`, `--pp-size`, `--pp-layers`, `--estimate-memory`, `--no-memory-check`, `--max-tokens`, `--recommend-quant` and `--surgery` all resolve.

### 2.3 Why the Integration Test Is Scoped to the Parser

`run_generate` resolves `-m` **before** it calls the validators, because the validators read the resolved model directory:

```rust
args.model.model =
    resolve_model_source_with_override(&args.model.model, args.model.models_dir.as_deref())?;

validate_tensor_parallel_args(&args)?;
validate_pipeline_parallel_args(&args)?;
```

A subprocess invocation therefore cannot reach `validate_pipeline_parallel_args` without a real model on disk. Worse, the value the test used, `nonexistent-model-path-for-validator-only-check`, is a valid bare repo segment, so the resolver expands it against `$MLXCEL_DEFAULT_ORG` and goes to the network. Running the test's exact argv:

```
[mlxcel] 'nonexistent-model-path-for-validator-only-check' -> mlx-community/nonexistent-...
[mlxcel] model '...' not found locally; downloading into the mlxcel store...
Error: failed to download model '...': authentication failed (HTTP 401).
```

Exit 1, before either validator. An intermediate draft of this PR fixed the flag name and added a parse-rejection guard while keeping this argv, which would have put an outbound HuggingFace request into the non-ignored CI surface on every run, timing out rather than failing on an offline runner. That draft was wrong and is not what shipped.

The shipped test appends `--help`, so clap exits as soon as parsing succeeds. It is hermetic (no runtime initialization, no network) and asserts **positively** on exit status, so it cannot pass by dying early:

- `--pp-size 2 --tp-size 2 --help` exits 0.
- `--pp-size 2 --tensor-parallel-size 2 --help` exits 2 with a clap usage error.

Nothing is lost by narrowing the scope, because the validator has direct unit coverage at `validate_pipeline_parallel_args_accepts_2d_pp_tp` in `src/commands/generate_tests.rs`. This also restores what the test's own comment had always claimed it did ("We invoke `mlxcel generate --help` plus the 2D flags"); the code had drifted from its comment.

### 2.4 The `docs/en/` References Are Not Broken

The issue asserted that `docs/en/` "has never been part of this repository" and made "no `docs/en/` reference remains in `src/`" an acceptance criterion. The first half is true of this git tree and the conclusion drawn from it is not:

```
mkdocs.yml:8:docs_dir: docs/en
mkdocs.yml:160:      - Tensor Parallelism: distributed/tensor-parallelism.md
mkdocs.yml:161:      - Pipeline Parallelism: distributed/pipeline-parallelism.md
```

Those two nav entries resolve, under `docs_dir: docs/en`, to exactly the two paths the comment cited. `docs/README.md` states that `docs/en`, `docs/ko` and `docs/shared` are maintained in a separate documentation repository, that the root mkdocs configs name paths into that tree, and, verbatim, "That is deliberate, not drift." `Makefile` carries a `docs-guard` target built on the same fact, added by PR #1122, whose own report lists "someone fixes the dangling navs by pointing them at `docs/*.md`, breaking the tree that owns them" as a high-impact risk.

So the comment was pointing at real manual pages. The acceptance criterion is therefore not honoured as written, deliberately, and that is called out on the issue and in the PR description rather than being satisfied silently.

### 2.5 Compatibility

No behavior changes. The validator accepts and rejects exactly the same inputs, and the one message whose text changed cannot be emitted at all. No CLI surface change, no serialization change, no API change, no new dependency.

---

## 3. Technical Decisions

### 3.1 No CHANGELOG Entry

An earlier draft added one describing users hitting `error: unexpected argument '--tensor-parallel-size' found` after following the message. That cannot happen: a `--tp-size > 1` run without a pipeline topology returns `Ok(())` at the early return and never reaches the guard. The changelog is for user-visible change, and there is none here, so the entry was removed rather than reworded into something technically true but immaterial.

### 3.2 Keep the Manual References, Add the In-Checkout One

| Option | Pros | Cons |
|---|---|---|
| Delete the two `docs/en/` paths | Satisfies the issue's criterion literally | Deletes live pointers to the canonical operator manual on a false premise |
| Repoint them at `docs/distributed.md` only | Short; every path is in-tree | Same information loss, and `docs/distributed.md` has no 2D section, so it does not replace them |
| **Chosen: keep both manual pages, name `docs/distributed.md` as the in-checkout summary, and say why the sources are not here** | Nothing is lost; the next reader is told the paths are cross-tree by design | Leaves a `docs/en` string in `src/`, against the issue's criterion |

The comment now explains the split, so the next sweep of this kind does not rediscover these as dangling.

### 3.3 Fix the Test Rather Than Only the Message

The test is not incidental to this issue. It is the artifact that should have caught the wrong flag name and could not, and it failed for the same root cause. Fixing the message while leaving a test that cannot detect the message being wrong again would close the issue without closing the gap.

### 3.4 Leave the Unreachable Guard Alone

The dead `ensure!` is a real defect, but removing it or widening the early return is a logic change, and "message text only; the validation logic is unchanged" is an explicit acceptance criterion of this issue. It is filed as a follow-up instead, together with the misnamed unit test that documents the same tautology.

---

## 4. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 2 |
| Behavior changes | 0 |
| User-visible changes | 0 |

### Changes by Area

| Area | File | Summary |
|---|---|---|
| CLI | `src/commands/generate.rs` | `ensure!` message names `--tp-size`; the validator comment keeps both operator-manual pages, explains that their sources are in the separate documentation repository by design, and adds `docs/distributed.md` as the in-checkout summary |
| Tests | `tests/pp_tp_2d_real_models.rs` | Parity test uses `--tp-size`; the non-ignored test is rescoped to the argument parser, made hermetic with `--help`, and asserts positively on exit status; module doc reflowed |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `1593eb8a` | fix | fix(cli): name the real --tp-size flag in the 2D-parallelism error |

---

## 5. Validation and Follow-up

### Passed

- `MLX_CUDA_ARCHITECTURES=121 cargo test --profile test-fast --features cuda --test pp_tp_2d_real_models`: passing, 1 ignored.
- Hermetic: the non-ignored test performs no network request. Its argv exits inside clap.
- Non-vacuous, verified in both directions against the built binary: `--pp-size 2 --tp-size 2 --help` exits 0; substituting `--tensor-parallel-size` exits 2 with `error: unexpected argument '--tensor-parallel-size' found`.
- `grep -rn -- "--tensor-parallel-size" src/` empty.
- `cargo fmt --all -- --check` clean.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings` clean.

### Follow-up Candidates

- **The unreachable guard.** The `ensure!` at the top of the `tp_size > 1` branch cannot fail, and `validate_pipeline_parallel_args_rejects_2d_without_pp_enabled` asserts `is_ok()` under a name saying it rejects. Either the early return is too broad or the guard is redundant; deciding which is a logic change and out of scope here.
- **Machine-check the flag names.** The sweep in section 2.2 exists only as a shell pipeline. `tests/cli_help_consistency.rs` is the established home for CLI-surface invariants; a test asserting that every `--flag` named in an error-message literal appears in `--help` would close the class rather than this instance.
- **Vacuous negative assertions.** Any test whose only assertion is that a string is absent from process output passes when the process dies early for an unrelated reason. A sweep of `tests/` for that shape is self-contained.
- **2D composition is undocumented.** Neither `docs/distributed.md` nor, per its nav, the manual has a PP x TP section, which is why the comment has to qualify itself.
