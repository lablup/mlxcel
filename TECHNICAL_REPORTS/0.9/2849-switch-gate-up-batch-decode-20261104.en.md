# Technical Report: PR #2849 - perf(moe): gate the switch gate/up kernel to single-token decode

**Date**: 2026-11-04
**Status**: In review (open PR; closes #2515)
**Languages**: Rust, Markdown, CSV
**Risk Level**: Low

---

## Executive Summary

PR #2849 re-measures the `moe_switch_gate_up` Metal kernel at true batch-decode shapes on `models/olmoe-1b-7b-0125-4bit` and changes the gate that decides when `SwitchGLU::forward` attempts it. The 2026-10-21 record that kept the kernel on for M3 and later ran the server at its default `--parallel 4`, so its concurrency 5 to 8 levels queued behind a four-row decode batch, and its "batch 8 control" was two waves of four rows rather than one 64-route step. With `--parallel 8`, occupancy confirmed at exactly B, and three alternating measured runs per arm, the kernel loses to `gather_qmm` at every B from 2 to 7 by 3.6% to 7.6% per stream, while the B=8 sorted-path control differs by +0.6%. Teacher-forced parity traces report 0/256 top-1 and 0/32 decided-position disagreements with a zero logit delta, so the kernel buys no measured accuracy at multi-token shapes. The gate is now `tokens == 1` for every generation, and Inkling's `forward_activated` path is unchanged.

---

## 1. Problem Statement

### 1.1 Background

PR #2514 added `moe_switch_gate_up`, a direct-dequant Metal kernel for the routed gate/up pair, to the shared `SwitchGLU`. It is wired into Inkling's decode through `forward_activated` and into `SwitchGLU::forward`, the path the other MoE families take at multi-token batch decode. At B >= 2 the #268 fused single-token kernel never fires, so the switch gate/up kernel is the only custom candidate and `MLXCEL_FUSED_MOE=0` isolates it cleanly.

Issue #2515 asked for a real-checkpoint batch-decode measurement, a parity statement for the two paths at batch shapes, and a gate adjustment if the kernel regressed. It stays open until this PR merges.

### 1.2 Existing Issues

- **Batch geometry was not measured.** The 2026-10-21 A/B ran `mlxcel-server` at its default `--parallel 4`. Above concurrency 4, requests queue behind a four-row decode batch, so the rows labelled B=5 to B=8 did not measure B-row batches, and the batch-8 control compared two waves of four rows (32 routes per wave), not one 64-route step.
- **The evidence was inside the noise floor.** That record's batch-3 and batch-4 cells could not be separated from run-to-run variation with two measured passes per arm. Only B=2 showed a clear loss, at -4.0% per stream.
- **The gate premise was that record.** The shared gate admitted the kernel at B >= 2 for M3, M4, and M5 on the strength of those numbers. M1 and M2 already declined at B >= 2 after their measured ~10% loss (#2587).
- **No real-checkpoint batch parity existed.** Batch-shape parity had been checked only on synthetic fixtures.
- **The proposed refinement was too narrow.** After the 2026-10-21 run, `tokens >= 2 && total < 24` was proposed as a carve-out that would keep the kernel at B=2 only. That proposal rested on the one batch-2 cell.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| Keeping the M3/M4/M5 exception leaves a measured 3.6% to 7.6% per-stream decode loss on M3 | Medium | High without this change |
| M4 and M5 lose a possible multi-token win because they are now declined without measurement | Low (performance only) | Certain until a positive cell is measured |
| Another checkpoint shows a different result at these shapes | Low | Low, the parity result is checkpoint-specific |
| The single-token path changes behavior | Low | Very low, the path is unchanged except for the gate arity |

---

## 2. Technical Review

### 2.1 Performance

The measurement corrected the batch geometry instead of reusing the earlier configuration. `mlxcel-server` ran with `--parallel 8 --ctx-size 8192`, which splits the context into eight 1024-token slots, so client concurrency B reaches a decode batch of B rows. Prompt cache and APC were off so the eight clients did not start decoding at different times after shared-prefix reuse. The scheduler's own occupancy counter (`mlxcel_batch_decode_tokens_total / mlxcel_batch_decode_steps_total`, exposed by `--metrics`) reported exactly B in both arms at every level, with 127 decode steps and 127 * B advanced tokens per level. That counter, not the `--parallel` value, is the check that the ladder measured the batch sizes it names.

The harness booted six fresh server processes in alternating order (A, B, A, B, A, B), discarded one full-ladder warm-up pass per boot, then recorded one measured pass. That gives three measured passes per arm. The ladder was B = 2 through 8 with nominal 512-token prompts (445 actual), 128 generated tokens, temperature 0, and greedy sampling. Arm A is the default environment, where the kernel is attempted at B = 2 to 7 (`tokens * top_k` = 16 to 56, below the sorted-path threshold). Arm B sets `MLXCEL_FUSED_MOE=0`, so both projections take the indexed `gather_qmm` path. At B = 8 (64 routes) both arms take the sorted path, so that row is the control where both arms run identical code.

Per-stream decode throughput, median of the three measured runs with the spread `(max - min) / median`:

| B | kernel median tok/s | kernel spread | gather median tok/s | gather spread | kernel vs gather |
|---:|---:|---:|---:|---:|---:|
| 2 | 233.0 | 0.5% | 241.8 | 6.4% | -3.6% |
| 3 | 180.4 | 0.7% | 195.2 | 0.4% | -7.6% |
| 4 | 154.6 | 0.5% | 162.2 | 0.5% | -4.7% |
| 5 | 128.6 | 0.8% | 137.9 | 0.1% | -6.7% |
| 6 | 114.5 | 0.3% | 122.3 | 0.2% | -6.4% |
| 7 | 101.8 | 0.2% | 108.9 | 0.0% | -6.5% |
| 8 (control) | 95.0 | 1.9% | 94.4 | 0.1% | +0.6% |

The gather arm's 6.4% spread at B = 2 is one low first run (226.7 tok/s) against 241.8 and 242.2 for the other two. The kernel arm is 232.1 to 233.2 across all three. Every other cell is under 2%. Aggregate throughput follows the same shape: -3.0% to -6.3% at B = 2 to 7 and +0.4% at the B = 8 control.

Mean decode step time and occupancy:

| B | kernel step ms | gather step ms | kernel vs gather | kernel occupancy | gather occupancy |
|---:|---:|---:|---:|---:|---:|
| 2 | 4.30 | 4.14 | +3.9% | 2.00 | 2.00 |
| 3 | 5.56 | 5.14 | +8.2% | 3.00 | 3.00 |
| 4 | 6.50 | 6.20 | +4.8% | 4.00 | 4.00 |
| 5 | 7.82 | 7.30 | +7.1% | 5.00 | 5.00 |
| 6 | 8.80 | 8.25 | +6.7% | 6.00 | 6.00 |
| 7 | 9.90 | 9.27 | +6.8% | 7.00 | 7.00 |
| 8 (control) | 10.63 | 10.69 | -0.6% | 8.00 | 8.00 |

`1000 / step_ms` reproduces the per-stream rate at every cell, so the two views are independent readings of the same decode. Time to first token agrees within 0.6% at every level: 109.0/109.0 ms at B=2, 137.0/137.7 at B=3, 172.0/172.1 at B=4, 206.7/206.9 at B=5, 240.8/241.0 at B=6, 273.7/275.1 at B=7, and 309.6/308.7 at B=8. The throughput difference is therefore decode arithmetic, not admission or prefill.

The B = 8 control bounds the protocol's bias at +0.6% per stream, +0.4% aggregate, and -0.6% step time. Every loss at B = 2 to 7 is several times that floor. The loss is not monotonic in B, since B = 3 is the worst cell and B = 4 the mildest, so it is not a single scheduling threshold. The kernel's threadgroup grid and route pairing cost more than the indexed gather at these route counts.

### 2.2 Correctness and Parity

`examples/logit_trace` with `MLXCEL_TRACE_BATCH_ROWS=B` runs each traced position as a width-one `forward_batched_with_ids` decode over B independently prefixed rows, so the shared `SwitchGLU` receives `x [B, hidden]` and `indices [B, top_k]`, exactly the shape `forward` reduces with either path. The corpus is `docs/CONTINUOUS_BATCHING.md`, sixteen chunks of sixteen teacher-forced positions (256 positions per width), top-k 8, in the same two environment arms as the throughput run. `scripts/compare_logit_traces.py` treats the gather arm as the reference, the kernel arm as the candidate, and defines a decided position as a reference top-two logit gap of at least 2.0.

| B | top-1 disagreement | decided-position disagreement | logit delta (p50 / p90 / p99 / max) | perplexity |
|---:|---|---|---|---|
| 2 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 3 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 4 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 5 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 6 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 7 | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |
| 8 (control) | 0 / 256 | 0 / 32 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | +0.000% |

A shorter 64-position pass agreed at every level as well, so the result does not depend on the sample size. The B = 8 control confirms the comparison itself does not manufacture a difference. This is a parity statement for OLMoE at these shapes and this corpus, not a general bit-identity claim. The kernel accumulates in f32 with a different reduction order from `gather_qmm`, and another checkpoint can flip near-tie greedy tokens for that reason.

### 2.3 Compatibility and Dependencies

- **Breaking changes**: none in a public API. `switch_gate_up_preferred_in_forward` is `pub(crate)`, and its signature loses the `AppleSiliconGen` argument.
- **New dependencies**: none.
- **Behavior by generation**: every generation now takes the gather path at B >= 2. M1 and M2 are unchanged there because they already declined. M4 and M5 now decline with M3 instead of inheriting its superseded exception. Inkling's `forward_activated` path still calls the kernel directly at its measured single-token win.
- **Toolchain pins**: unchanged. The measurement used the MLX pin `7ca40033132dae444d45f7fc3daae57e24c13f6c` and `mlxcel 0.8.1`.

### 2.4 Code Quality

- **Test coverage**: the generation-matrix unit test is replaced by `switch_gate_up_in_forward_is_single_token_only`, which asserts true at one token and false for every `tokens` in `2..=8`.
- **Complexity**: the hot-path condition drops from two inputs and a hardware query to one integer comparison.
- **Documentation**: the function comment now records both the M1 Ultra and M3 Ultra measurements, the geometry error in the earlier record, the focused replication, and the M4/M5 gap. The kernel's own comment records that the second wire-in is single-token only.
- **Technical debt**: unchanged. One historical document remains, explicitly marked as superseded for its concurrency 5 to 8 rows.

### 2.5 Security

The change touches kernel selection inside the MoE forward pass. It adds no input parsing, authentication, authorization, or logging surface, and it does not change the data the server accepts or returns.

---

## 3. Technical Decisions

### 3.1 Corrected Batch Geometry With Alternating Boots

**Context:** The earlier record's concurrency 5 to 8 cells were not the batch sizes they named, because the server's default `--parallel 4` caps the native batch at four rows. Its batch-8 control was therefore not a control at all.

**Alternatives Considered:**

| Option | Pros | Cons |
|---|---|---|
| Reuse the 2026-10-21 protocol | Comparable to the old numbers with no new harness work | Repeats the geometry error and cannot measure B > 4 |
| Raise `--parallel` to 8 and confirm with the occupancy counter | Measures true B-row batches; the counter detects a mis-sized ladder | Requires a larger context split and fresh boots |
| **Chosen: `--parallel 8 --ctx-size 8192`, occupancy check, alternating boots** | Directly falsifies the geometry; three passes per arm separate the loss from boot order | Longer run; occupancy becomes a required protocol step |

**Rationale:** The occupancy counter is the only evidence that client concurrency became a B-row decode batch. Alternating boots removes the systematic order effect that two sequential passes could not detect. The control row at B = 8, where both arms execute identical sorted-path code, bounds the residual protocol bias.

**Trade-offs:** The corrected configuration raises memory pressure per context slot and disables prompt cache and APC. Those choices make the measurement cleaner but mean the absolute throughput is not directly comparable to a production default configuration.

### 3.2 Gate Decision: Single-Token Only

**Context:** The gate must choose between the custom kernel and the reference gather path for every multi-token decode on every Apple generation.

**Alternatives Considered:**

| Option | Pros | Cons |
|---|---|---|
| Keep the M3+ exception | No code churn | Rests on invalid geometry; leaves a measured 3.6% to 7.6% loss on M3 |
| `tokens >= 2 && total < 24` | Removes the B = 3 to 7 cells | Still admits B = 2, the cell it was meant to protect, which loses 3.6% per stream; leaves a route-count special case |
| Per-generation table for M4/M5 | Could preserve an unmeasured win | No positive multi-token measurement exists; more code and a larger test matrix for no evidence |
| **Chosen: `tokens == 1` for every generation** | Matches every measured cell; one rule; uniform behavior | Gives up any unmeasured multi-token win on M4 and M5 |

**Rationale:** The kernel loses at every measured multi-token size, not only at the cell the earlier proposal targeted. At B = 2 it is 233.0 against 241.8 tok/s per stream, a 3.6% loss, and the focused `--parallel 4` replication reads -4.3% (271.6 against 283.9). The loss extends through B = 7, so a `total < 24` carve-out would have to exclude nearly the whole multi-token range to be useful. Once those cells are excluded, the remaining rule is one token. The corrected geometry also shows that route count is not the axis: B = 3 has 24 routes and is the worst cell at -7.6%, while the 64-route B = 8 control sits at the noise floor. Inkling is unaffected because `forward_activated` calls the kernel directly and does not pass through this gate.

**Trade-offs:** M4 and M5 are now declined without their own measurement. A future host with a positive multi-token cell would need the gate reopened for that generation, which is a small, local change.

### 3.3 Why `tokens >= 2 && total < 24` Was Rejected

The proposal came from the 2026-10-21 record, where B = 2 was the only clear loss and B = 3 and B = 4 were inside the noise floor. The corrected measurement invalidates that premise in two ways. First, B = 2 itself is a real loss in the new run, so the carve-out admits the exact cell it was supposed to remove. Second, the loss is present at B = 3 through 7 as well, so there is no measured region above one token where the kernel wins. A route-count threshold of 24 would exclude B = 3 and above and leave only B = 2, which is the one cell that should not keep the kernel. The focused `--parallel 4` run rules out the corrected configuration as the cause: it reproduces -4.3% at B = 2 and -7.5% at B = 4 under the earlier geometry.

### 3.4 Why the Earlier Record's Batch-4 Cell Changed Sign

The 2026-10-21 record read +0.4% per stream at B = 4, and the focused replication reads -7.5% (172.9 against 186.9 tok/s). The kernel arm moved by +2.2% (172.9 against 169.2), so the difference is on the gather side: `gather_qmm` is about 11% faster at the current head than at `0f43e98e`. The earlier batch-3 and batch-4 cells did measure true 3- and 4-row batches, but both sat inside its noise floor, so they could not support keeping the kernel. Its batch-5 to 8 rows were queued behind four rows and did not measure the printed sizes.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
[Before]
SwitchGLU::forward
  total >= 64  -> sorted path
  else if switch_gate_up_preferred_in_forward(n_tokens, silicon_gen)
               -> switch gate/up kernel (M3/M4/M5 at B >= 2)
  else         -> indexed gather_qmm

[After]
SwitchGLU::forward
  total >= 64  -> sorted path
  else if switch_gate_up_preferred_in_forward(n_tokens)
               -> switch gate/up kernel only when n_tokens == 1
  else         -> indexed gather_qmm
```

Inkling's path is separate and unchanged: `SwitchGLU::forward_activated` calls `forward_switch_gate_up_kernel` directly at its measured single-token win.

### 4.2 Key Code Changes

**File: `src/models/switch_layers.rs`**

```rust
// Before
pub(crate) fn switch_gate_up_preferred_in_forward(
    tokens: i32,
    silicon_gen: mlxcel_core::hardware::AppleSiliconGen,
) -> bool {
    tokens == 1 || silicon_gen.wide_quantized_projections()
}

// After
pub(crate) fn switch_gate_up_preferred_in_forward(tokens: i32) -> bool {
    tokens == 1
}
```

The call site no longer reads `mlxcel_core::hardware::get_hardware().silicon_gen`:

```rust
// Before
} else if let Some(activated) = switch_gate_up_preferred_in_forward(
    n_tokens,
    mlxcel_core::hardware::get_hardware().silicon_gen,
)
.then(|| self.forward_switch_gate_up_kernel(x, indices))
.flatten()

// After
} else if let Some(activated) = switch_gate_up_preferred_in_forward(n_tokens)
    .then(|| self.forward_switch_gate_up_kernel(x, indices))
    .flatten()
```

**Reason for change:** The measured multi-token loss applies to every generation with a positive measurement, and the only generation-specific exception rested on a batch geometry that did not measure its named sizes. Removing the hardware query also removes a global hardware access from this hot path.

**Test change:**

```rust
#[test]
fn switch_gate_up_in_forward_is_single_token_only() {
    assert!(switch_gate_up_preferred_in_forward(1));
    for tokens in 2..=8 {
        assert!(
            !switch_gate_up_preferred_in_forward(tokens),
            "multi-token batch decode of {tokens} tokens loses to gather_qmm"
        );
    }
}
```

### 4.3 Data Model Changes

None. The PR adds two committed result files, `decode.csv` (42 rows, three runs per arm per batch size) and `focus-parallel4.csv` (12 rows), under `docs/benchmark_results/data/switch-gate-up-batch-m3ultra-2026-11-04/`.

---

## 5. Learning Points

### 5.1 Occupancy Is the Geometry Check, Not `--parallel`

**Concept:** A server's concurrency setting is a request-admission limit, not proof that a decode step advanced that many rows. The scheduler's occupancy counter measures the rows a step actually advanced.

**Application in this PR:** The old protocol labelled cells "batch 5" through "batch 8" while the server's native batch was capped at four. Reading `mlxcel_batch_decode_tokens_total / mlxcel_batch_decode_steps_total` at exactly B, and confirming 127 steps with 127 * B advanced tokens, is what makes the new ladder credible.

**Common use cases:** Any serving benchmark that claims per-batch-size results, especially when the server has separate concurrency and max-batch settings.

### 5.2 A Control Cell Where Both Arms Run Identical Code

**Concept:** A/B protocols accumulate bias from boot order, thermal state, and background load. A cell where both arms execute the same code measures that bias directly.

**Application in this PR:** At B = 8 both arms take the sorted path, so the +0.6% per-stream delta is the protocol's noise floor. Every measured loss at B = 2 to 7 is several times larger, which makes the sign of the result robust even though individual cells differ in magnitude.

### 5.3 Teacher-Forced Parity at Batch Widths

**Concept:** `MLXCEL_TRACE_BATCH_ROWS=B` runs each traced position as a width-one decode over B independently prefixed rows. That produces the `x [B, hidden]` and `indices [B, top_k]` shapes the batch path sees without requiring B live clients with identical prompt timing.

**Application in this PR:** 256 teacher-forced positions per width, 32 decided positions per width, zero top-1 and decided disagreements, and a zero logit delta at every B from 2 to 8. The B = 8 control guards against a comparison artifact.

### 5.4 Median of Alternating Boots

**Concept:** Two sequential passes cannot separate an arm effect from boot order. Three passes with alternating arm order give a median and a spread, and the spread shows whether a delta exceeds run-to-run variation.

**Application in this PR:** Six boots in A, B, A, B, A, B order produced three measured passes per arm. The only large spread, the gather arm at B = 2 (6.4%), is traceable to one low first run, and the report says so rather than hiding it behind the median.

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---|---|---|
| `moe_switch_gate_up` | Direct-dequant Metal kernel for the routed gate/up pair in `SwitchGLU` | The kernel whose batch behavior this PR measures and gates |
| `gather_qmm` | MLX indexed quantized matmul path | The reference arm and the path every multi-token decode now takes |
| Occupancy | Advanced decode tokens divided by decode steps | Proves client concurrency B produced a B-row batch |
| Sorted path | `SwitchGLU` path taken when `tokens * top_k >= 64` | The B = 8 control where both arms run identical code |
| `MLXCEL_FUSED_MOE` | Environment kill switch for fused MoE kernels | `=0` isolates the switch gate/up kernel at B >= 2 |
| `MLXCEL_TRACE_BATCH_ROWS` | Logit-trace option that replays a position over B rows | Produces batch-shape parity without live concurrency |
| `forward_activated` | Inkling's decode entry into the kernel | Unchanged; does not pass through the shared gate |

### Related Technologies

- **MLX**: Apple's array framework. `gather_qmm` and the sorted-path kernels are its indexed MoE primitives.
- **OLMoE**: The 1B-active, 7B-total sparse MoE checkpoint used here, with 64 experts, top-k 8, hidden 2048, and expert intermediate 1024.

### Related PRs and Issues

- PR #2514: added the switch gate/up kernel and wired it into `SwitchGLU::forward`.
- Issue #2515: requested this batch-decode measurement, parity statement, and gate adjustment.
- Issue #2587: measured the ~10% M1 Ultra multi-token loss that already declined M1 and M2.
- Issue #268: the fused single-token MoE kernel, which cannot fire at B >= 2.
- Issue #2479: the fused decode-MoE down-projection work this kernel sits beside.
- `docs/benchmark_results/versions/v0.8.0.md#switch-gate-up-batch-m3ultra-2026-10-21`: the superseded record.
- `docs/benchmark_results/versions/v0.8.0.md#switch-gate-up-baseline-m3ultra-2026-10-21`: the single-token baseline this PR does not re-measure.

---

## 7. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 4 |
| Lines added | +232 |
| Lines deleted | -34 |
| Tests added | 1 rewritten unit test |
| Net | +198 |

### Changes by Category

| Category | Count | Summary |
|---|---|---|
| Performance | 1 | Gate narrowed to single-token decode; hardware query removed from the decision |
| Documentation | 3 | Corrected-geometry benchmark report and two per-run CSV files |
| Code quality | 1 | Unit test replaced; function and kernel comments updated |
| Security | 0 | No change |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `ff898648` | perf | perf(moe): gate the switch gate/up kernel to single-token decode |

---

## 8. Follow-up Actions

### Unvalidated Surface

- **M4 and M5 are not measured.** They now decline the kernel at B >= 2 as M3 does. Reopening requires a positive multi-token cell on one of those hosts and a new gate branch.
- **The `tokens == 1` path of `SwitchGLU::forward` was not re-measured.** This A/B covered B = 2 to 8. The single-token win that keeps the first branch alive comes from the Inkling measurement in `switch-gate-up-baseline-m3ultra-2026-10-21.md`; the `forward` one-token cell is not independently re-confirmed here.
- **Parity is checkpoint-specific.** The zero delta applies to OLMoE at these shapes and this corpus. The kernel's f32 accumulation order differs from `gather_qmm`, so another checkpoint can flip near-tie greedy tokens.
- **CI clippy was pending when the PR's evidence was assembled.** The PR cited the targeted gate test and `cargo fmt --check` as its local validation. The remote `cargo-clippy` check has since passed on run `41666406529`, but the recorded evidence does not depend on it.

### Monitoring Required

- Watch multi-token decode throughput on M3 Ultra after the gate change. It should improve by the measured 3.6% to 7.6% per stream at B = 2 to 7, with no change at B = 8.
- Watch single-token decode throughput for any regression. No measured surface predicts one, because the single-token branch is unchanged.

### Future Improvements

- Re-run this A/B on M4 and M5 when either host is available. A positive multi-token cell would reopen the gate for that generation.
- Re-measure the `SwitchGLU::forward` one-token cell so the first branch has a current, local number rather than an inherited one.
- Keep the 2026-10-21 document as a historical record. Its batch-2 and batch-4 cells are superseded by this report, and its concurrency 5 to 8 rows should be read as queued behind four rows.
- The raw sweep text and server logs remain under the gitignored `bench-results/switch-gate-up-batch/20261104-002134/` and `20261104-focus-p4/`.

---

## Appendix

### A. Test Results

| Check | Result |
|---|---|
| `cargo test --offline --profile test-fast --features metal,accelerate -p mlxcel switch_gate_up_in_forward_is_single_token_only` | Pass |
| `cargo fmt --check` | Pass |
| Remote `cargo-clippy` | Pass (run `41666406529`; pending when the evidence was assembled) |
| Other PR checks | Pass or skipped at report time |

### B. Measurement Environment

| Field | Value |
|---|---|
| Host | Mac Studio, Apple M3 Ultra, 512 GB unified memory, macOS 26.6.2 (25G83) |
| GPU generation | 15, `AppleSiliconGen::M3` |
| Build | `cargo build --release --features metal,accelerate --bin mlxcel-server --example logit_trace`; `mlxcel 0.8.1`; fat LTO, `codegen-units = 1` |
| Measurement binary | `main` at `79f0a447`, with no source changes |
| Server flags | `--parallel 8 --ctx-size 8192 --metrics --offline --no-prompt-cache --apc-enabled=false` |
| Target | `models/olmoe-1b-7b-0125-4bit`, 64 experts, top-k 8, hidden 2048, expert intermediate 1024, affine q4 group 64 |
| Sampling | 512 nominal (445 actual) prompt tokens, 128 generated, temperature 0, greedy |
| MLX pin | `7ca40033132dae444d45f7fc3daae57e24c13f6c` |
| Wrapper | `caffeinate -i`, indexers paused with `scripts/with_indexers_paused.sh` |

### C. References

- `docs/benchmark_results/versions/unreleased.md#switch-gate-up-batch-m3ultra-2026-11-04`
- `docs/benchmark_results/data/switch-gate-up-batch-m3ultra-2026-11-04/decode.csv`
- `docs/benchmark_results/data/switch-gate-up-batch-m3ultra-2026-11-04/focus-parallel4.csv`
- `docs/benchmark_results/versions/v0.8.0.md#switch-gate-up-batch-m3ultra-2026-10-21`
- `docs/benchmark_results/versions/v0.8.0.md#switch-gate-up-baseline-m3ultra-2026-10-21`
- `src/models/switch_layers.rs`
