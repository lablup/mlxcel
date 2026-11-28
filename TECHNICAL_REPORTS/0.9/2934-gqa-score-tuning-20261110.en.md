# PR #2934: Tune the GQA decode score matmul on ROCm

**Date**: 2026-11-10
**Status**: Completed (PR open as a draft; closes #2921)

## Executive Summary

Decode attention on the single-sequence XLA graph was dominated by one dispatch: the GQA score contraction took 1187.17 us of a 1509.75 us attention block, in 16 dispatches at 74.20 us each, while the context matmul with the same MAC count ran at 6.42 us. A per-dispatch `rocprofv3` table ruled out elementwise folding and dispatch-count reduction, because IREE canonicalizes those graph rewrites before dispatch formation. The shipped fix is a narrow second match in the ROCm tuning spec: batch 8, group 4, `head_dim` 64, and 128 or more keys get a 128-thread `[32, 4, 1]` workgroup with the full 64-wide reduction. Attention per step falls from 1509.75 to 337.64 us, the score dispatch from 74.20 to 1.47 us per call, and the decode step from 8.80 ms to 7.33 ms median (7.07 ms best) at n=128 and capacity 256. Greedy output is byte-identical to the previous native binary and to the f16 path on two prompts, `xla_batch_bench` stays REFERENCE-EXACT at B=4 and B=8, and no committed golden was refreshed. 6 files, +328/-14.

## 1. Problem Statement

After #2890, #2892 and #2907, the projection block fell from about 21 ms to 3.7 ms per step while attention stayed at about 1.31 ms, making it the largest remaining kernel block. The issue asked for a per-dispatch table first and for the fix the table selects, not a guess.

The measurement at n=1 and n=128 (127 decode steps, subtracted) attributes 1509.75 us per step to attention. Within that, the GQA score contraction alone is 1187.17 us in 16 dispatches at 74.20 us each. The context matmul does the same number of multiply-accumulates in 6.42 us per call. The gap is therefore in how the score matmul is lowered to a dispatch, not in the math, the masks, or the number of kernels: attention was 10 dispatches per layer before and after the investigation.

## 2. What Was Tried and Rejected

- Folding the attention scale into q once before the layer loop, and emitting K head-major so the contraction axis is contiguous. Both were implemented and measured, and both are compiled-graph no-ops because IREE canonicalizes them before dispatch formation. They were reverted and are not in this PR.
- Reducing the per-layer dispatch count or changing the KV layout. Ruled out by the same table: the elementwise and reduction dispatches are small next to the score contraction.

## 3. Shipped Change

`src/lib/mlxcel-xla/assets/rocm/bigk_vecmat_tuning.mlir` gains `match_gqa_score`, a second match after the existing long-K vecmat rule:

- Condition: batch 8, group 4, `head_dim` 64, keys >= 128.
- Lowering: 128-thread workgroup with shape `[32, 4, 1]` and the full 64-wide reduction, instead of the generic lowering that was forming the 74 us dispatch.

Supporting changes are a tuning test in `src/lib/mlxcel-xla/src/tuning.rs`, a focused emitter test asserting the single decode score contraction keeps the tuning spec shape (`single_decode_score_contraction_keeps_the_tuning_spec_shape`), and a comment update in `src/lib/mlxcel-xla/src/iree.rs`. `AttnLayout::Ragged` and `AttnLayout::Prefill` emitted text stays byte-identical, no golden was refreshed, and the model math is unchanged.

## 4. Measurements

All runs are on `gfx1151`, llama-3.2-1b-instruct-4bit, native quant, `MLXCEL_XLA_CONTEXT_CAPACITY=256`, under `scripts/rocm_gpu_guard.sh --hold`.

| Metric | Before | After |
|--------|--------|-------|
| Attention per step | 1509.75 us | 337.64 us |
| GQA score dispatch | 74.20 us/call | 1.47 us/call |
| Whole-graph kernel time | 5291 us | 4099 us |
| Decode step, n=128 | 8.80 ms (113.6 tok/s) | 7.33 ms median, 7.07 ms best (136.4 / 141.5 tok/s) |

The A/B is three interleaved rounds on one binary. Greedy output is byte-identical to the pre-change native binary and to the f16 path on two prompts. `xla_batch_bench` prints REFERENCE-EXACT PASS at B=4 and B=8.

The issue target of a step below 7.1 ms is met at the best sample (7.07 ms) but not at the median (7.33 ms). Attention is no longer the constraint; the remainder is host-side per-token work.

## 5. Technical Decisions

### 5.1 Change the tuning spec, not the graph

The profile showed the graph was already canonical and the cost lived in dispatch formation. Editing the emitter would have added code that the compiler discards. A tuning match is the only lever in this path that changes the formed dispatch, and it keeps the graph, the math, and the goldens untouched.

### 5.2 Narrow match instead of a general one

The match is limited to batch 8, group 4, `head_dim` 64, and 128 or more keys, the configuration that was traced end to end. The batch predicate is load-bearing rather than decorative: IREE's StableHLO input lowering collapses the ragged and uniform-B `dot_general` to the same rank-3 contraction with the same indexing maps, using `bsz * n_kv` as the batch, so the first version of the match also applied the one-row config to the ragged score matmul. Pinning the batch to the measured `n_kv = 8` keeps the single-sequence graph matched and sends every multi-row graph back to the compiler default; the matcher gives 16 matches on single decode, 0 on prefill, 0 on ragged B=4 and 0 on ragged B=8. Widening it later is a one-line condition change backed by a new trace.

## 6. Verification and Limits

- `cargo fmt --all -- --check`, `cargo clippy -p mlxcel-xla --lib --tests -- -D warnings`: clean.
- `cargo test -p mlxcel-xla --lib` focused runs: `validation::tests::emit_all_reproduces_registered_goldens`, both `tuning::tests`, and the new `single_decode_score_contraction_keeps_the_tuning_spec_shape` test pass.
- The implementation review caught that the first version of the match also fired on the ragged and uniform-B score contractions, which the issue keeps out of scope. The batch predicate above fixes that; the matcher was re-run on the lowered goldens to confirm 16 matches on single decode and none on prefill, ragged B=4 or ragged B=8. The `pr-security-checker` and `pr-finalizer` passes were not run under the coordinator's stop instruction; the bounded gates above are the verification that ran. The change surface is a compiler tuning spec plus tests, with no runtime input handling or credential surface.
- Only capacity 256 was traced end to end. The KV read convert (103 us per step) and the context matmul (103 us per step) are now the largest attention items and are candidates for a follow-up.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 6 |
| Lines added | +328 |
| Lines deleted | -14 |
| Tests added | 2 tuning assertions, 1 emitter test |

| Category | Summary |
|----------|---------|
| Performance | Narrow GQA score tuning cuts attention from 1509.75 to 337.64 us per step |
| Correctness | Token-identical output, batch reference-exact, goldens untouched |
| Documentation | Comment update in `iree.rs` |

## 8. Follow-ups

1. Trace and address the KV read convert and context matmul, now about 103 us each per step.
2. Widen the tuning match only with a trace for the new shape (other head dimensions or key ranges).
3. Confirm the n=128 result at other capacities, since only capacity 256 was measured.
