# Technical Report: PR #1254 - B=1 MTP gate on GPU generation

**Date**: 2026-08-20
**Status**: Partial
**Languages**: Rust, Shell, Markdown
**Risk Level**: Medium

## Executive Summary

PR #1254 re-measures the batch-capable singleton MTP pairing on M3 Ultra and moves the static gate `mtp_b1_default` off the `has_neural_accelerator` proxy onto Apple GPU generation 15, the `use_qmv_wide` split. The gate was declining a pairing that measures 1.95x to 2.65x on that host. The change is deliberately partial: it does not close issue #1217, whose M1 Ultra acceptance criteria need hardware the measuring host is not.

## 1. Problem Statement

The B=1 MTP burst for batch-capable targets ran only where `has_neural_accelerator` held, which is M5 and nothing else. That policy rested on the founding measurement in #165: about 1.2 to 1.4x on M5 Max against a 0.75 to 0.96x regression on M1 Ultra. Both numbers predate #1194, #1199, #1203, #1208 and #1215. M3 Ultra had never been run on the pairing, so a generation-15 part was classified with generation 13 by a binary proxy that does not track the mechanism.

A second, quieter problem made the first one durable. Neither `scripts/bench_speculative.sh` nor `scripts/bench_block_width.sh` had a case for the Gemma 4 31B + bf16 assistant pairing, so the pairing the gate governs had no path through the #1215 measurement protocol. The founding numbers could not be reproduced or refreshed by the harness that exists to keep such numbers honest.

## 2. Technical Decisions

### 2.1 Discriminate on GPU generation rather than the Neural Accelerator

`AppleSiliconGen::wide_quantized_projections` encodes MLX's `use_qmv_wide` predicate reduced to its chip-dependent half: from generation 15 an affine-quantized projection at `M >= 2` runs as one wide pass, and on generation 13 the verify block runs as `K` narrow passes whose cost grows with the block. This is the mechanism the published round-cost model already credited for the host ordering, so the gate now reads the mechanism instead of a correlate of it. `Unknown` reads false, preserving today's decline for non-Apple hosts where a K-wide verify does not amortize at all (#638) and for Apple generations newer than the enumerated ones.

### 2.2 Leave generation 13 declining, and say why that is a conclusion

The width sweep made the conservative choice defensible rather than merely cautious. Round cost fits `0.83 + 0.170 K` classic steps for this pairing on M3 Ultra against `1.14 + 0.090 K` for the 12B pairing on the same host: the bf16 drafter costs about 1.9x as much per extra block position, and the two lines cross at K = 4. A naive transfer of M1 Ultra's published block-4 round cost of 2.71 across pairings would have predicted that generation 13 now clears break-even; carrying the slope ratio instead puts a block-4 round near 3.6 classic steps there, which the emitted tokens would only just cover, consistent with the founding regression.

### 2.3 Make the safety argument a test rather than a claim

The new predicate is strictly more permissive than the one it replaced, so no host loses a path it previously had. That is what makes it legitimate to change the gate on evidence from one host without re-measuring the others, so a unit test asserts the implication over every enumerated generation rather than leaving it as prose.

## 3. Change Summary

| Area | Change |
|---|---|
| `src/lib/mlxcel-core/src/hardware.rs` | New `AppleSiliconGen::wide_quantized_projections` predicate plus two unit tests (generation split, and the weaker-than-NA implication). |
| `src/server/batch/speculative_burst.rs` | `mtp_b1_default` third parameter becomes `wide_quantized_projections`; `mtp_b1_burst_enabled` reads it from `silicon_gen`; docstring rewritten to the new measurements with the two evidence limits stated; two new unit tests. |
| `src/server/batch/mtp_policy.rs` | Field renamed and re-sourced, so the adaptive policy's ambiguous-window fallback follows the same predicate. |
| `src/server/batch/scheduler.rs` | Decline-path comment updated to the new policy. |
| `scripts/bench_speculative.sh`, `scripts/bench_block_width.sh` | New `gemma31b` case in both, so the protocol can reach the pairing the gate governs. |
| `docs/benchmarks.md`, `docs/environment-variables.md` | 31B section rewritten with the measured rows; the claim that the pairing's speedup comes from B>1 windows and that its single-stream acceptance is too low is removed as falsified. |
| `docs/benchmark_results/mtp-b1-gate-m3ultra-2026-08-20.md` | New dated record: environment, three-prompt table, width sweep, round-cost fits, dispatch-path verification, and an explicit list of what was not measured. |

## 4. Review Findings

Two corrections were made during the work rather than after it.

The first was an analytical error caught by the width sweep. An earlier draft of the docstring and the env-var reference asserted that the round-cost model predicted post-#1203 M1 Ultra would now clear break-even on this pairing. That inference transferred a round cost between two pairings with different drafter dtypes, which the sweep then showed have slopes differing by 1.9x. Both statements were corrected before commit, and the results record now warns explicitly against transferring a round cost between these pairings at any width other than the crossing point.

The second was a PR-body defect. The sentence stating that the PR does not close #1217 itself contained the substring `close #1217`, which GitHub's closing-keyword parser matched without regard to the negation, so the PR was initially registered as auto-closing the issue it was written to leave open. The body was reworded and `closingIssuesReferences` re-verified as empty.

## 5. Validation

`cargo fmt --all --check` clean. `cargo clippy --profile test-fast --features metal,accelerate --all-targets` clean. 361 `server::batch` unit tests pass, plus the new `hardware` and `mtp_b1_default` tests.

Beyond the pure test seam, the gate was exercised through `Scheduler::mtp_b1_should_run` on the real checkpoints on the measuring host, with `MLXCEL_MTP_ADAPTIVE=0` so the static gate decides. `main` at `9e2c6675` declined the burst; this branch ran it (block 4, 80 tokens over 21 rounds, acceptance 0.921); `MLXCEL_ENABLE_MTP_B1=0` still declined. That before-and-after is on one host with one pair of checkpoints, which is what the acceptance criterion asked for and no more.

Recorded as part of validation: the offline `mlxcel generate` path does not consult this gate at all. `mtp_b1_default` has exactly one caller and `MtpPolicy` is built only in the server worker, so the bench harness runs the burst unconditionally. This is why the harness is the right instrument for deciding the gate, and why the two env vars the issue prescribes are inert there.

## 6. Related Work

Issue #1217 remains open by design. Its M1 Ultra rows and its M5 Max re-measurement need hardware that was not available, and the M3 Ultra Qwen rows it also asks for were already measured on current main under this protocol in #1215. Predecessors: #165 (the founding gate), #333 (adaptive policy, which falls back to this static default on an ambiguous window), #1203 (drafter projections quantized at load), #1215 (the measurement protocol and its guards).
