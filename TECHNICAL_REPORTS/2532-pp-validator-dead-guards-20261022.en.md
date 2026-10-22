# PR #2532: drop unreachable guards in validate_pipeline_parallel_args

**Date**: 2026-10-22
**Status**: Completed
**Risk Level**: Low
**Languages**: Rust

---

## Executive Summary

Issue #1131 reported that `validate_pipeline_parallel_args` in `src/commands/generate.rs` carried error branches that could never fire, plus a test whose name claimed a rejection while it asserted success. This PR implements reading (a) from the maintainer comment: TP-only runs (`--tp-size > 1` with pipeline parallelism off) are a shipped path and stay accepted. Two unreachable `ensure!`s are removed, a third is simplified to an equivalent form, two misnamed tests are renamed, and tests are added for every remaining check. The set of accepted and rejected inputs is unchanged.

---

## 1. Problem Statement

### 1.1 Background

The validator returns early when PP is off (`pp_layers` is `None` and `pp_size <= 1`). Everything after that return runs only when `pp_layers.is_some() || pp_size >= 2`. Two later checks contradicted that invariant, so their error messages could never be emitted.

### 1.2 Existing Issues

- The "2D parallelism requires --pp-size >= 2" check was the exact negation of the early return, so it never failed.
- The trailing `--pp-size must be at least 2` arm ran only with `pp_layers == None`, where the early return already guarantees `pp_size >= 2`. The issue did not name it, but it falls under the "no unreachable branch" criterion.
- `..._rejects_2d_without_pp_enabled` and `..._requires_two_stages_without_manual_ranges` asserted `is_ok()` while their names promised a rejection.
- Several live checks had no test: `--pp-micro-batch-size 0`, an empty `--pp-layers`, multimodal input under PP.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Dead messages mislead readers about what the CLI rejects | Low (maintainability) | Deterministic |
| Removing a guard that was actually reachable | Low (a bad topology reaches later code) | Ruled out by the reachability analysis and tests |

---

## 2. Technical Review

### 2.1 Security

No new input paths. The validator still rejects the same inputs; only unreachable code is deleted. No unsafe code.

### 2.2 Performance

Not applicable. The validator runs once at CLI startup.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. Accepted and rejected inputs are identical before and after.
- **New Dependencies**: none.

### 2.4 Code Quality

- The `total_ranks > 0` check is kept, simplified to the equivalent `pp_size > 0` with the same message. It is reachable: `--pp-size 0 --pp-layers "0-3,4-7" --tp-size 2` passes the early return. The issue body assumed `pp_size >= 1` there, but `--pp-size` is an unrestricted `usize`.
- Comments now state the invariant that holds after the early return and why the topology guard is still reachable.
- The flag-names-in-error-strings invariant already exists on main as `every_flag_named_in_command_source_exists_in_some_help` in `tests/cli_help_consistency.rs`, so no duplicate test was added. A stricter per-command variant was evaluated with a throwaway script and found no gaps.

---

## 3. Technical Decisions

### 3.1 Keep TP-only accepted

The maintainer chose reading (a) on #1131: TP-only is dispatched on `tp_size` elsewhere and is a supported path. Rejecting it would have been a behavior change, so the dead check is deleted rather than made live.

### 3.2 Keep the topology guard, narrow it

Deleting the guard with the other two would have let `--pp-size 0` with a layer spec and TP reach sharding code. The `pp_size > 0` form keeps the same error and message at lower cost.

---

## 4. Implementation Details

- `src/commands/generate.rs`: removed the `pp_size >= 2 || pp_layers.is_some()` ensure and the trailing `else` ensure; replaced `total_ranks` with `pp.pp_size > 0`; added explanatory comments.
- `src/commands/generate_tests.rs`: renamed two tests to `..._accepts_tp_only_when_pp_disabled` and `..._accepts_single_stage_and_manual_ranges`; added `..._rejects_zero_pp_size_with_layers_and_tp` and `..._rejects_zero_micro_batch_empty_layers_and_media`.

---

## 5. Verification

| Check | Result |
|---|---|
| `cargo test -p mlxcel --bin mlxcel validate_pipeline_parallel` (test-fast, default features) | 7 passed |
| `cargo test --test cli_help_consistency` | 27 passed |
| `cargo clippy -p mlxcel --bin mlxcel --tests -- -D warnings` | clean |
| `cargo fmt --check` on touched files | clean |
| CI | not verified at report time |

---

## 6. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 2 |
| Lines | +61 / -22 |
| Tests added | 2 (plus 2 renamed) |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `b71368c3` | fix | fix(cli): drop unreachable guards in validate_pipeline_parallel_args |
| `84776b93` | style | style: rustfmt the new reachability test |
| `68f95915` | test | test(cli): rename pipeline validator test that named a removed check |
| `2664098f` | test | test(cli): cover the untested pipeline-parallel validator checks |

---

## 7. Follow-up Actions

### Future Improvements

- None required. The `pp_size * tp_size == nodes` check for remote topologies stays at the cluster-TOML validator layer.

### Monitoring Required

- None.
