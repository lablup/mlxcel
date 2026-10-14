# PR #2138: Size Metal `gather_qmm_rhs` by the broadcast x rows

**Date**: 2026-10-15
**Status**: Integrated with main `9bfabbab` after #2304; the remaining delta is attribution and reports, with production source identical to validated main
**Risk**: Low (no-op for every production caller; changes behavior only for an x that `gather_qmm_rhs` has to broadcast)

## Summary

The Metal nightly on the M1 Ultra runner has failed since 2026-09-30 on `models::switch_layers::mxfp_tests::{mxfp4,mxfp8}_gather_qmm_matches_host_reference`, both in the `SortedSharedActivation` case. `GatherQMM::eval_gpu` in MLX's Metal backend passes `x.size() / K` to `gather_qmm_rhs` as the row count. That is the row count of x before `gather_qmm_rhs` broadcasts x against the indices. For one activation row gathered against 32 sorted slots it is 1, so the grid and the kernel's `M` bound cover one row and output rows 1..31 are never written. The overlay now passes `B * M`. The PR also makes two split-MTP tests portable across macOS path canonicalization and filesystems that reject invalid-byte names; the production resolver and its symlink protections are unchanged.

Refs #1599.

## 1. Nightly history

| Dates | Failure | Status |
|---|---|---|
| 2026-09-28 | `verify-clippy`: unused `Result` in `fused_norm_parity_tests.rs:158` and `fused_rope_parity_tests.rs:133` | fixed by #2029 (d8d34e2b) |
| 2026-09-29 | `tests::family_order_is_exhaustive`: `FAMILY_ORDER` missing `Speech` | fixed by #2080 (5486e404) |
| 2026-09-30 to 10-11 | the two mxfp gather tests | fixed by this PR; focused Metal validation passed at `25202198` |
| 2026-10-12 (`1e8c7022`) | the same mxfp tests; split-MTP `/var` canonicalization and invalid-byte filename assumptions; three YAML `err_expect` lints | kernel and split-MTP tests fixed by this PR; YAML lint already fixed on current main |

The mxfp tests arrived with #2071 (2cee9cf4) on 2026-09-30, so the last signature is a new test exposing a latent upstream edge case, not a regression.

## 2. Root cause

- The case uses x `[1, 1, K]`, indices `[32]` over 8 experts, `sorted = true`. With M == 1, B = 32, B / E = 4 and `right_sorted_`, `eval_gpu` takes `gather_qmm_rhs`.
- `gather_qmm_rhs` broadcasts x to 32 rows (`broadcast_with_indices`), but its `M` argument was computed by the caller from the original x: 1. `grid_dims.y = ceil(M / 16)` and the kernel's `tgp_bm = min(BM, M - y_row)` then write row 0 only. `out` comes from `allocator::malloc`, so the other rows hold whatever was there: 7.4e32 relative error (mxfp4) and non-finite values (mxfp8).
- The same runs pass the three earlier gather cases in bf16, f16 and f32, and `mxfp_matmuls_match_host_reference_on_cpu_device`, which runs `SortedSharedActivation` on MLX's CPU backend. The reference and the test are correct; "bf16" in the message is just the first dtype tried.
- The defect is mode-independent (affine would hit it too) and is still present on ml-explore/mlx main.

## 3. Change

`B * M` (the output row count) replaces `x.size() / K` at the one call site. It equals the old value whenever x is already expanded, which is every production caller (`SwitchLinear::forward(.., true)` always receives `gather_sort` output), and it also fixes the NAX variant, which receives the same argument. The overlay header and the CMake overlay notes list the new hunk.

The split-MTP test for `.` and `..` outputs now compares the staging parent with the canonical temporary-directory path, matching the resolver's intentional behavior on macOS where `/var` resolves to `/private/var`. The non-UTF-8 test first probes the temporary filesystem: filesystems that accept the byte name retain the successful round trip, while rejecting filesystems must return the same I/O error and leave no output or staging entry.

The normal merge of current main moved the Xcode 27 NAX compatibility file into `src/lib/mlxcelverse/overlays/metal/`. Because that overlay is a whole-file derivative of MLX, its preserved Apple header now has the required MLX source URL, MIT-license statement, and top-level `NOTICE` pointer. `NOTICE` covers the whole overlay and carries MLX's copyright and MIT terms, while the README acknowledgment identifies the licensed reuse.

## 4. Validation

- Historical exact-head Metal gate: Apple M1 Ultra (128 GB), macOS 27.0 (26A428), full Xcode 27 RC, PR head `25202198` on base `b34d517b`. All five `models::switch_layers::mxfp_tests` passed with the Metal overlay rebuilt, and the release build passed.
- Historical exact-head model smoke: Qwen3-30B-A3B-4bit, greedy 64 tokens, produced byte-identical text on base, repeated base, and PR; SHA-256 `d2f40b2d5831c3d7790d467fb51123f0d1f615fb7300aa2f92d86a9d8831a0de`.
- CUDA (GB10): the original PR branch built and all 32 `models::switch_layers` tests passed. This does not validate Metal.
- Current-main integration: main `59974283` was merged normally into the existing PR branch at `44045522`. The sole merge conflict was the NAX overlay's move from `src/lib/mlx-cpp` to `src/lib/mlxcelverse`; the quantized-kernel hunk, split-MTP tests, and overlay registration merged without semantic conflict. The five focused `models::switch_layers::mxfp_tests` pass under `--profile test-fast --features metal,accelerate`, and both focused split-MTP portability tests pass. Formatting, diff, license-header, MLX-pin parser, and cross-repository-reference checks pass.
- Fresh release real-model A/B: the compatibility baseline was built from `44045522` with only `quantized.cpp` restored from main `59974283`, excluding the `B * M` change while holding the Xcode 27 NAX compatibility overlay and all other source equal. It is explicitly not a pristine-main artifact. Its SHA-256 is `d76febfc9adb23ca6bc627310994c108cc4e721c26341ef7f49ab092a89db651`; the exact `44045522` candidate SHA-256 is `ee83df3db3cea7159579487718419d4329c7fbeb147655cb1847a6867b13a872`.
- Both release artifacts ran twice on Qwen3-30B-A3B-4bit with `MLXCEL_FUSED_MOE=0`, greedy decoding, no chat template, a 96-token repeated-word prompt, and 64 generated tokens. Qwen3 routes each token to 8 of 128 experts, so prefill covers about 768 expert-route slots per MoE layer, comfortably above the 64-slot boundary. All four generated texts are byte-identical: 962 bytes, SHA-256 `cbf1670d0b554a35c029ec4104ae3ecfc8a46e07296d7e33a5f3b2c214a5fa8f`. Timings varied and are not used for a performance claim.
- The exact-source candidate Metal nightly [38582194425](https://github.com/lablup/mlxcel/actions/runs/38582194425) passed at `44045522` on `lablup-macmini01`: formatting, Clippy, 12,316 general tests (390 ignored), and 51 video tests, with zero failures.
- Review found no remaining functional or security issue in the #1599 implementation. It identified a high-severity attribution gap in the whole-file MLX NAX overlay; the header, `NOTICE`, and README changes described above fix it, and the attribution patch passed follow-up review and the repository license-header check.
- PR CI's `cargo-deny` job exposed four rustls 0.23.45 advisories published on 2026-10-14. Exact main `59974283` locks the same version, so this is a base-wide prerequisite rather than a regression from #1599. It is not waived: issue #2312 and lock-only PR #2313 update rustls to 0.23.46 separately.

## 5. Follow-up

Report the call to ml-explore/mlx, and drop the hunk when a pin bump carries an upstream fix. Merge the separate rustls security prerequisite rather than suppressing its advisory gate. Keep #1599 open through merge and close it only after a green post-merge Metal nightly.

## 6. Final integration after #2304

Main `9bfabbab` already carries this PR's `B * M` correction and portable split-MTP tests through #2304. Resolve the CMake and Metal quantized-file conflicts by retaining those main versions. Upstream MLX `243bdf09` includes the Xcode 27 correction from ml-explore/mlx#4594, so remove the obsolete whole-file NAX overlay and its registration instead of restoring the old `59d600b5e` source. Remove only the obsolete overlay-specific NOTICE wording; retain the expanded MIT terms and the expm1f attribution.

The resulting production tree is identical to main. Its exact pre-squash source `4befdbc7` passed Metal nightly [38583586205](https://github.com/lablup/mlxcel/actions/runs/38583586205): formatting, Clippy, 12,315 workspace tests (390 ignored), and 51 FFmpeg video tests. The earlier release-model evidence above remains attributed to its original source and is not a fresh run on the new pin. #2304 also already upgraded rustls to 0.23.46, resolving the security prerequisite described above.

Keep #1599 open until a post-merge main Metal nightly succeeds. M5-specific execution remains unverified.
