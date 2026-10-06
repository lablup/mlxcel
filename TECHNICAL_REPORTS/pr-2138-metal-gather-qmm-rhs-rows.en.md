# PR #2138: Size Metal `gather_qmm_rhs` by the broadcast x rows

**Date**: 2026-10-06
**Status**: Draft. Root cause proven from source; the fix is unverified on Metal (authored on a CUDA host where the Metal overlay is not compiled)
**Risk**: Low (no-op for every production caller; changes behavior only for an x that `gather_qmm_rhs` has to broadcast)

## Summary

The Metal nightly on the M1 Ultra runner has failed since 2026-09-30 on `models::switch_layers::mxfp_tests::{mxfp4,mxfp8}_gather_qmm_matches_host_reference`, both in the `SortedSharedActivation` case. `GatherQMM::eval_gpu` in MLX's Metal backend passes `x.size() / K` to `gather_qmm_rhs` as the row count. That is the row count of x before `gather_qmm_rhs` broadcasts x against the indices. For one activation row gathered against 32 sorted slots it is 1, so the grid and the kernel's `M` bound cover one row and output rows 1..31 are never written. The overlay now passes `B * M`.

Refs #1599.

## 1. Nightly history

| Dates | Failure | Status |
|---|---|---|
| 2026-09-28 | `verify-clippy`: unused `Result` in `fused_norm_parity_tests.rs:158` and `fused_rope_parity_tests.rs:133` | fixed by #2029 (d8d34e2b) |
| 2026-09-29 | `tests::family_order_is_exhaustive`: `FAMILY_ORDER` missing `Speech` | fixed by #2080 (5486e404) |
| 2026-09-30 to 10-05 | the two mxfp gather tests | this PR |

The mxfp tests arrived with #2071 (2cee9cf4) on 2026-09-30, so the last signature is a new test exposing a latent upstream edge case, not a regression.

## 2. Root cause

- The case uses x `[1, 1, K]`, indices `[32]` over 8 experts, `sorted = true`. With M == 1, B = 32, B / E = 4 and `right_sorted_`, `eval_gpu` takes `gather_qmm_rhs`.
- `gather_qmm_rhs` broadcasts x to 32 rows (`broadcast_with_indices`), but its `M` argument was computed by the caller from the original x: 1. `grid_dims.y = ceil(M / 16)` and the kernel's `tgp_bm = min(BM, M - y_row)` then write row 0 only. `out` comes from `allocator::malloc`, so the other rows hold whatever was there: 7.4e32 relative error (mxfp4) and non-finite values (mxfp8).
- The same runs pass the three earlier gather cases in bf16, f16 and f32, and `mxfp_matmuls_match_host_reference_on_cpu_device`, which runs `SortedSharedActivation` on MLX's CPU backend. The reference and the test are correct; "bf16" in the message is just the first dtype tried.
- The defect is mode-independent (affine would hit it too) and is still present on ml-explore/mlx main.

## 3. Change

`B * M` (the output row count) replaces `x.size() / K` at the one call site. It equals the old value whenever x is already expanded, which is every production caller (`SwitchLinear::forward(.., true)` always receives `gather_sort` output), and it also fixes the NAX variant, which receives the same argument. The overlay header and the CMake overlay notes list the new hunk.

## 4. Validation

- CUDA (GB10): no regression; see the PR body for the run.
- Metal: not run. Needs a Mac: `cargo test --profile test-fast --features metal,accelerate -p mlxcel --lib models::switch_layers::mxfp_tests -- --test-threads=1`, plus a MoE prefill smoke over 64 slots.

## 5. Follow-up

Report the call to ml-explore/mlx, and drop the hunk when a pin bump carries an upstream fix.
