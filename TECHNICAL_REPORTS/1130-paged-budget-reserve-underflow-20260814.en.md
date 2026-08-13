# Technical Report: PR #1130 - fix(core): stop the paged budget test underflowing past the v2 reserve

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low (a test expectation and an operator log message; no budget resolves to a different block count)

---

## Executive Summary

`execution::memory_estimate::tests::resolve_block_budget_explicit_bytes_floors_to_block_count` failed deterministically in 0.00s on GB10 / CUDA sm_121 / Linux aarch64, at `assertion failed: shortfall < 100`. The panic was not flaky and not a TF32 numeric artifact: the test computed its own expected value with `(per_block * 100 - workspace) / per_block`, an unsigned subtraction of a device-derived quantity from a fixed one.

The issue's evidence said the defect was confined to the test, and the acceptance criteria required that be confirmed or refuted rather than assumed. It is confirmed, by an exhaustive audit of every site where the reserve meets a budget and by a new regression test that probes the implementation directly at budgets below the reserve. The implementation was already correct.

A second finding: `Some(0)` is reachable in production on any non-Metal host and is handled correctly by the caller, but the warning it emits blamed the wrong cause. That message now names the reserve.

---

## 1. Problem Statement

### 1.1 Background

#899 charges the fused paged decode v2 workspace to the KV byte budget before the remainder is divided into blocks, so admission only hands out blocks it can actually back with memory. The reserve is computed by `paged_v2_workspace_reserve_bytes`, and its dominant term is `2 * device_target_ctas() * n_rep * (head_dim + 1) * 4`.

`device_target_ctas()` (`src/lib/mlxcel-core/src/paged_v2/plan.rs`) returns `DEFAULT_TARGET_CTAS = 512` whenever `metal_is_available()` is false, which is every Linux and CUDA host. On Apple parts it is `gpu_core_count * 8`, floored at 64. The reserve is therefore a property of the machine the test runs on, not of the model geometry the test writes into its temporary directory.

### 1.2 Existing Issues

- **The test hardcoded an assumption about the host.** `per_block` is 131072 for the minimal config, so the budget under test was `per_block * 100 = 13107200` bytes. With `PAGED_V2_MAX_N_REP = 16` and `PAGED_V2_CONCURRENT_LAUNCHES = 2`, a 512-CTA target gives a reserve of 17040384 bytes. The reserve exceeded the whole budget by 3933184 bytes, the `u64` subtraction wrapped, and `shortfall` became `2^47 - 31`, which still converts cleanly through `usize::try_from` on a 64-bit host. The guard `assert!(shortfall < 100)` was the only thing standing between the wrap and a silently wrong `assert_eq!`.
- **The crossover is inside the Apple range, so macOS CI could not be trusted to catch it either.** At a reported core count of 48 the reserve is 12813312 bytes and the test passes; at 50 it is 13341696 and the test fails. The test's correctness depended on which Mac ran it.
- **The assertion that failed was not the assertion that mattered.** `shortfall < 100` is a sanity check on the test's own arithmetic. The property worth pinning, that a budget which ignored the workspace comes back short by exactly the reserve, was carried only by the `assert_eq!` below it.
- **The `Some(0)` contract was documented but never covered.** `resolve_paged_block_budget` documents that a budget rounding below one block returns `Some(0)`, and the `saturating_sub` is what makes a sub-reserve budget land there rather than near `2^47` blocks. No test asked for a budget below the reserve, so the saturation was covered only incidentally.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| The failing test is dismissed as another host-specific numeric artifact and the module stops being trusted | Medium | High |
| A future edit replaces the implementation's `saturating_sub` with `-`, minting roughly `2^47` blocks for a small budget | High | Low |
| An operator sets a low `--kv-cache-budget`, gets an unbounded pool, and reads a warning that points at model size | Medium | Medium |

---

## 2. Technical Review

### 2.1 Where the Underflow Lives: Test-Only, Confirmed

The audit is small enough to be exhaustive. `paged_v2_workspace_reserve_bytes` has four call sites in the tree:

- `src/execution/memory_estimate.rs:875`, the implementation, which subtracts with `saturating_sub`.
- `src/execution/memory_estimate.rs:1697`, the failing test, which subtracted with `-`.
- Two sites in `the_v2_workspace_reserve_is_small_and_scales_with_the_head_dim` that measure the reserve and never subtract it.

There is exactly one subtraction of the reserve from a budget in non-test code, and it already saturated. The claim is not left resting on the grep, though: the new `resolve_block_budget_below_the_workspace_reserve_is_zero_blocks` calls the production function with budgets of `0`, `1`, `workspace / 2`, `workspace - 1` and `workspace`, and asserts `Some(0)` for each. A wrapping subtraction would return a block count near `2^47` for the last three: a budget near `u64::MAX` divided by a 128 KiB block is not `usize::MAX` on a 64-bit host, which is worth stating precisely because the comment guarding this line exists to stop a future "simplification" back to `-`. The test is green, so the implementation is empirically clean, not merely clean by inspection.

### 2.2 Why the Environment Override Was Rejected

Pinning `MLXCEL_PAGED_DECODE_V2_TARGET_CTAS` inside the test would make the reserve deterministic, and the issue lists it last for a reason: `device_target_ctas()` memoizes through a `OnceLock`. Under a shared test binary the first caller wins, so whether the override takes effect depends on test ordering. That turns a deterministic failure into an order-dependent one, which is strictly worse than what was there before.

### 2.3 Stating the Contract Without Depending on the Device

The fix expresses the expectation the way the implementation computes it, then states the resulting property twice so neither statement is vacuous on any host:

- Capped: `100 - shortfall == min(ceil(workspace / per_block), 100)`. For `workspace <= 100 * per_block` this is an identity, since `floor((100 * pb - w) / pb) == 100 - ceil(w / pb)`. Above it, the saturation pins both sides at 100.
- Uncapped: asking for `per_block * (100 + ceil(workspace / per_block))` bytes resolves to exactly 100 blocks. The remainder after the reserve is `100 * per_block + (ceil(w / pb) * pb - w)`, and the parenthesised term is in `[0, per_block)`, so it floors away.

On GB10 the capped statement runs in its saturated branch, which is why the uncapped one is there: it exercises the exact-reserve arithmetic on large-CTA devices too, rather than leaving that path covered only on small Apple parts.

### 2.4 The `Some(0)` Path Is Reachable and Handled

The issue asked whether the documented `Some(0)` behavior is actually reachable and handled by callers, rather than assumed. Both halves check out. `resolve_worker_paged_block_budget` (`src/server/model_worker.rs`) matches `Some(n) if n > 0` first and falls through to `Some(_)`, where it warns and returns `None`, leaving the pool unbounded. It never installs a zero budget, which would wedge every request.

What did not check out is the diagnosis. The warning read "model too large for a meaningful paged budget at this batch / available memory", which is a reasonable description of the `Auto` path and a wrong one for an explicit byte budget: on a non-Metal host any `--kv-cache-budget` under roughly 16.25 MiB lands here regardless of model size.

---

## 3. Technical Decisions

### 3.1 Mirror the Implementation Rather Than Re-Derive It

The expectation now uses `saturating_sub`, the same operation the implementation uses. Re-deriving an expected value with different arithmetic is what created the defect: the test and the code disagreed about what happens below zero, and only the test was wrong.

### 3.2 Keep the 100-Block Framing

Scaling the whole test off the measured reserve was the issue's second option. It was not taken for the first three assertions, which read naturally in whole blocks and are unaffected by the reserve's size because they add it to the request. Only the fourth case, which deliberately omits the reserve, needed the device-independent treatment.

### 3.3 A Dedicated Regression Test, Not Another Case in the Existing One

The below-reserve path is a different contract from "a byte budget floors to a block count", and it is the one that pins the `saturating_sub`. It gets its own test with the boundary pinned from both sides: `workspace + per_block - 1` yields `Some(0)` and `workspace + per_block` yields `Some(1)`.

### 3.4 Split the Warning by Directive Instead of Writing One Accurate Sentence

The first attempt gave both directives the reserve-naming message, on the grounds that it is accurate for `Auto` too: a budget too small to cover the reserve plus one block is literally what both paths ran into. Review rejected that, correctly. Accuracy is not the bar a diagnostic has to clear. `Auto` is the shipped default on both binaries, and it reaches zero blocks when the model leaves no room for KV at all, so naming a 16 MiB reserve when the real shortfall is tens of gigabytes points the operator at `MLXCEL_PAGED_DECODE_V2_TARGET_CTAS`, which is the wrong knob entirely.

The arm now matches on the directive. `Auto` keeps its original wording, which was the right diagnosis for that case all along. `Bytes` reports the requested budget, the reserve, and the smallest budget that would mint one block, all rendered through the module's own `format_bytes` so they read as sizes rather than as raw byte counts. `PagedBudgetDirective` is `Copy` and the directive is still in scope, so the split costs two lines. No resolution logic changed on either path.

---

## 4. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 3 |
| Production behavior changes | 1 (log message text only) |
| Tests added | 1 |
| Tests repaired | 1 |
| Assertions added to the repaired test | 2 |

### Changes by Area

**`src/execution/memory_estimate.rs`**
- `resolve_paged_block_budget`: a comment recording why the subtraction must saturate, with the numbers that make a sub-reserve budget reachable on any non-Metal host.
- `resolve_block_budget_explicit_bytes_floors_to_block_count`: the wrapping `-` becomes `saturating_sub`, the arithmetic sanity check `shortfall < 100` is replaced by the capped exact-reserve identity, and an uncapped case is added.
- `resolve_block_budget_below_the_workspace_reserve_is_zero_blocks`: new regression test.

**`src/server/model_worker.rs`**
- `resolve_worker_paged_block_budget`: the zero-block arm matches on the directive instead of emitting one message for both. `Auto` keeps its original wording; `Bytes` reports the requested budget, the workspace reserve, and the smallest budget that would mint one block, all through `format_bytes`.

**`CHANGELOG.md`**
- One entry under `## [Unreleased]` / `### Fixed` for the warning text, stating explicitly that no budget resolves to a different block count.

---

## 5. Validation and Follow-up

### Passed

- `cargo test --profile test-fast --features cuda --lib -- --exact execution::memory_estimate::tests::resolve_block_budget_explicit_bytes_floors_to_block_count` on GB10, the case that failed before.
- `cargo test --profile test-fast --features cuda --lib execution::memory_estimate` on GB10: 36 passed, 0 failed.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`.
- `cargo fmt --all -- --check`.

### Not Covered

- The Apple small-CTA branch was not executed on hardware. It is covered by construction rather than by a run: every assertion is now either an identity over `workspace` or a request computed from it, and the capped statement is exact on both sides of the crossover.
- `--test-threads=1` was not used. The full lib suite aborts under parallel execution on this CUDA host for reasons unrelated to this change, and a serialized full-suite run is not a gate this box can deliver; the module filter above is the equivalent evidence.
- No end-to-end run was made with a sub-reserve `--kv-cache-budget` against a real model. The path is covered at the unit level and the warning text was read, not observed in a server log.

### Follow-up

- `PAGED_V2_MAX_N_REP = 16` over-reserves by roughly 4x for the common case (Llama 3 is 4). The reserve is deliberately generous, but it is now large enough relative to a small explicit budget that tightening it would widen the range of budgets that resolve to a usable block count. Out of scope here; worth a separate look.
