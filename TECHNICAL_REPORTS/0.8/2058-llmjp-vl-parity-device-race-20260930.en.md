# Technical Report: Issue #2058 - llmjp_vl conv-layout parity test raced the default device

## Summary

`vision::llmjp_vl::tests::either_patch_embedding_conv_layout_produces_the_same_features` failed in 2 of 15 full `cargo test --release -p mlxcel --lib` runs on an M1 Ultra, always with `max diff 0.000000018626451`. It passed alone. The test took no lock, so a sibling test that moved MLX's process-global default device to the CPU (`DefaultDeviceGuard::cpu()` under `lock_default_device()`) could overlap it and leave one tower on the CPU and the other on Metal. CPU and Metal differ in the last bits, which broke the exact `== 0.0` comparison.

## Cause proof

With the fix in place, a probe `let _cpu = DefaultDeviceGuard::cpu();` was inserted between the two forward passes. The test then failed deterministically with `max diff 0.000000018626451`, the same value seen in the soak failures. The probe was removed before commit.

The weight layout was ruled out as the cause: both layouts reach the conv as the same strided view (MLX `copy` shares the buffer, and the Metal conv makes the weight row-contiguous before dispatch).

## Change

- `src/vision/llmjp_vl_tests.rs`: the test takes `mlx_test_guard()` first, which holds the default-device lock and asserts the default device is the GPU. The assertion stays `max_diff == 0.0`; no tolerance was added.
- `src/vision/encoders/siglip.rs`: new test `patch_embed_both_conv_layouts_yield_identical_weight` builds `VisionEmbeddings::from_weights` from an HF-layout and an MLX-layout map of the same values and asserts both weights have shape `[O, kH, kW, I]` and are element-wise exactly equal. Mutating the transpose in `from_weights` from `[0, 2, 3, 1]` to `[0, 3, 2, 1]` makes it fail.

## Verification

- 20 consecutive full `cargo test --release -p mlxcel --lib` runs: 20 of 20 passed (8726 passed each), no failures.
- `cargo fmt --all -- --check`, `cargo clippy --release -p mlxcel --lib --tests -- -D warnings` and the `cli_help_consistency`, `dead_doc_pointers` and `llama_compat_manifest` contract tests pass.

## Not changed

Other exact-compare vision tests were swept and left alone (out of scope per the issue); see the PR description.
