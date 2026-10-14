# PR #2310: Attribute the MoE Decode Gap at the M = 1 Expert Gather

- **Date**: 2026-10-14
- **Status**: Implemented, reviewed, formatted, and verified
- **Risk**: Low (benchmark attribution document, Inkling single-token fused MoE routing, and DeepSeek-V4 gather-index sharing)

## Summary

PR #2310 resolves issue #1859 by recording a kernel- and graph-level attribution of the MoE single-token decode gap (`M = 1, B = top_k = 6, E = 256`) for `deepseek-v4-flash-4bit`, `inkling-small-mlx-4bit`, and `inkling-small-nvfp4` in `docs/benchmark_results/moe-m1-gather-attribution-2026-10-14.md`, and then landing the two targeted improvements justified by that evidence: routing `InklingSparseMoe::forward` through `SwitchGLU::forward_fused_kernel` on unscaled affine checkpoints (`inkling-small-mlx-4bit`) and sharing gather indices across `gate_proj`, `up_proj`, and `down_proj` inside `LimitedSwitchGlu::forward` in `src/models/deepseek_v4_moe.rs`.

## Problem

An Apple Silicon benchmark on `b8fdddfa` reported single-token decode on `deepseek-v4-flash-4bit`, `inkling-small-mlx-4bit`, and `inkling-small-nvfp4` at 16% to 23% of the `mlx-vlm` baseline while prefill reached 55% to 82%, and hypothesized that the `M = 1` expert gather shape (`x_exp` expanded to `[n_tokens, top_k, 1, D]`) forced `gather_qmm` / `gather_mm` into low-occupancy `bm = 32` (Metal) or `tile_m = 16` (CUDA `qmm_sm80`) tiled matrix kernels. Issue #1859 required first establishing the exact kernel dispatch and graph-level attribution before modifying `src/models/switch_layers.rs`, `src/models/deepseek_v4_moe.rs`, or `src/models/inkling/mlp.rs`.

## Design and Implementation

- **Kernel & Graph Attribution (`docs/benchmark_results/moe-m1-gather-attribution-2026-10-14.md`)**:
  - Traced `GatherQMM::eval_gpu` across Metal (`src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/quantized.cpp`), CUDA (`src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/quantized/qmm.cu`), and ROCm (`src/lib/mlxcelverse/overlays/rocm/mlx/backend/rocm/quantized/qmm.hip`). At `M = 1, B = 6`, all three backends dispatch specialized vector-matrix kernels (`qmv_quad` / `qmv_fast` / `fp_qmv_quad` on Metal since `M = 1 < get_qmv_batch_limit >= 6`; `qmv` / `fp_qmv` on CUDA since `M * B = 6 < 8` and `!supports_qmm_sm80(nvfp4)`; `qmv_gather` on ROCm since `M = 1 < 4`). Neither Metal's `bm = 32` tiled `gather_qmm` nor CUDA's `tile_m = 16` `qmm_sm80` is ever entered at single-token decode.
  - Reconciled the `forward_fused_kernel` family count across baseline and current `main` (`17` at `b8fdddfa`, `18` after `#1739` added `laguna_layers.rs`, `19` after `#2283` added `granitemoehybrid.rs`, and `20` after wiring `inkling/mlp.rs` in this PR).
  - Identified the actual per-family bottlenecks:
    1. `deepseek-v4-flash-4bit`: Over 85% of the decode gap comes from `HyperConnection::depth_connection` running an unfused 20-iteration `f32` Sinkhorn-Knopp normalization loop twice per layer across 43 layers (86 loops/token, ~12,470 graph nodes/token), while `LimitedSwitchGlu::forward` also synthesized redundant gather index subgraphs (`expand_dims` + per-projection `Arange`/`Broadcast`/`Reshape`) because it bypassed `#2137`.
    2. `inkling-small-mlx-4bit`: `InklingSparseMoe::forward` unconditionally called `SwitchGLU::forward` + `moe_weighted_sum` even when `gate_scale` and `out_scale` were `None`, missing `SwitchGLU::forward_fused_kernel`.
- **Inkling Single-Token Fused MoE Routing (`src/models/inkling/mlp.rs`)**: When `flat.shape()[0] == 1 && self.gate_scale.is_none() && self.out_scale.is_none() && fused_moe_enabled()`, `InklingSparseMoe::forward` now calls `self.experts.forward_fused_kernel(&flat, &top_k_indices, &expert_weights)` beforefalling back to `SwitchGLU::forward` + `moe_weighted_sum`. NVFP4 checkpoints with `gate_scale` / `out_scale` remain on `forward_with_expert_scales`.
- **DeepSeek-V4 Gather-Index Sharing (`src/models/deepseek_v4_moe.rs`)**: Updated `LimitedSwitchGlu::forward` to use `expand_dims_multi(x, &[-3, -2])`, `prepare_gather_indices(indices, sorted_indices)`, and `SwitchLinear::forward_indexed`, eliminating 6 graph nodes per layer (258 nodes/token across 43 layers) while preserving bit-identical arithmetic.

## Review Findings and Fixes

- `pr-reviewer`: Passed with 0 CRITICAL/HIGH/MEDIUM findings.
- `pr-security-checker`: Passed with 0 vulnerabilities or performance regressions.
- `pr-finalizer`: Verified test coverage, `Used by:` doc comments in `src/models/switch_layers.rs`, and `cargo fmt --check`.

## Verification

- `cargo fmt --all -- --check`: Passed.
- `MLXCEL_DEVICE=cpu cargo test --profile test-fast --lib models::deepseek_v4::tests::limited_switch_glu -- --test-threads=1`: 1 passed.
- `MLXCEL_DEVICE=cpu cargo test --profile test-fast --lib models::inkling::mlp::tests -- --test-threads=1`: 3 passed.
- `MLXCEL_DEVICE=cpu cargo test --profile test-fast --lib models::inkling::tests -- --test-threads=1`: 17 passed.
- `MLXCEL_DEVICE=cpu cargo test --profile test-fast --lib models::switch_layers::gather_index_tests -- --test-threads=1`: 5 passed.
- `cargo clippy --profile test-fast -p mlxcel --lib --tests -- -D warnings`: Passed with 0 warnings.
