# PR #3413: Mamba1 and Falcon-Mamba selective scan

**Date**: 2026-11-25

**Status**: Integrated CUDA acceptance passed; ROCm acceptance passed at the pre-integration source SHA

## Executive Summary

This PR adds native OpenXLA/IREE execution for Mamba1 and Falcon-Mamba models. It covers configuration parsing, checkpoint validation, weight ordering, f32 recurrent state, chunk-bounded prefill, batched decode, independent Transformers references, and a registered whole-model decode golden.

The prefill mixer constructs recurrence operands inside 128-row chunks and applies a log-depth affine prefix scan within each chunk. A 4096-token graph therefore never creates a full `[4096, d_inner, state_size]` tensor, while the carried state remains `[d_inner, state_size]` per layer.

## Problem Statement

The existing recurrent path supported Mamba2's grouped state-space layout, but Mamba1 uses a different diagonal recurrence. Each channel has its own decay, input projection, and skip parameter; B and C vary by token; the convolutional history and recurrent state must both survive prefill and batched decode. Falcon-Mamba additionally applies weightless RMS normalization to the B, C, and dt projections.

A direct row-by-row StableHLO lowering scales graph depth and concatenation count with the prompt capacity. Materializing all recurrence operands before slicing them into chunks also leaves a full `[T,D,N]` value in the graph, which is unacceptable for a 4096-token, 1536-channel, 16-state model.

## Change Summary

The configuration layer recognizes `mamba` and `falcon_mamba`, including Hugging Face names and the original state-spaces aliases. Canonical `use_bias` and `use_conv_bias` keys take precedence over `bias` and `conv_bias`. Unsupported LayerNorm configs are rejected, and active block and mixer epsilons must be finite and positive.

The emitter adds Mamba1 checkpoint names, tensor shapes, state allocation, convolution carry, prefill, and decode. The recurrence is evaluated in f32 even when surrounding values use another dtype. The shared recurrent ABI remains compatible with Mamba2 and Nemotron-H, including compact KV indices, grouped normalization, hybrid layer schedules, and MoE configuration added by the integrated base.

The Python oracle and Rust model-check harness now cover an official Mamba checkpoint and a seeded synthetic Falcon-Mamba model. The compact registered fixture exercises whole-model Mamba decode and freezes its StableHLO bytes.

## Technical Decisions

### Compose affine recurrence pairs in log depth

Each step is represented as `(a_t, u_t)` for `h_t = a_t * h_(t-1) + u_t`. The inclusive scan composes pairs as `(a_r * a_l, u_r + a_r * u_l)` at doubling offsets. This reduces each bounded block from a linear chain to logarithmic depth while preserving the recurrence order.

### Construct recurrence operands inside the mixer chunk loop

The model-specific prefill path first slices `x`, `dt`, B, and C to one chunk. Only then does it broadcast and multiply them into `log_a` and `bx [chunk,D,N]`. The generic scan receives only those bounded values. This producer boundary, rather than post-hoc slicing of full tensors, is what prevents a capacity-sized `[T,D,N]` allocation.

### Keep recurrent state and sensitive arithmetic in f32

Decay, affine composition, state carry, B/C/dt normalization, and readout accumulation use f32. The threaded state shape remains `[layers,d_inner,state_size]`, with convolution history stored separately, so split prefill and one-token decode follow the same numerical contract as whole prefill.

### Fail closed on unsupported configuration variants

`rms_norm = false`, non-SiLU activation, malformed dimensions, and non-finite or non-positive active epsilons are rejected before graph construction. Legacy bias aliases are accepted without allowing them to override canonical keys.

## Validation

The integrated executable source is `abd1520fd669624cb668bd558b98e332d0955093`, based on the merged PR #3411. `cargo test -p mlxcel-xla --lib -- --test-threads=4` passed 555 tests with 21 ignored. The registered byte-exact golden test passed for Mamba1, Mamba2, and Nemotron-H. Formatting and focused clippy with `-D warnings` passed. The environment-variable documentation checker passed with 396 discovered names, 265 documented names, and 131 allowlisted names; its Python suite passed all 15 tests.

On NVIDIA GB10, the exact integrated source compiled the capacity-4096 CUDA probe in 5.03 seconds and ran it in 0.319 seconds under `gpu-lock`. Process peak RSS was 203200 KiB. The independent CPU oracle reported maximum absolute differences of `3.651e-7` for output, `1.141e-8` for final recurrent state, and `1.371e-6` for convolution state. The timing log is `/tmp/epic564-plan/3348-cuda-cap4096-integrated-abd1520f-time.log`; the result bundle is `/tmp/issue3348-cap4096-integrated-abd1520f`. These are correctness and compile-footprint observations, not throughput claims.

The structural capacity probe confirmed that the 4096-token graph has no `[4096,D,N]` tensor. It produced 3,116 StableHLO lines (320 KiB); a direct CPU compile took 10.64 seconds at 228.2 MiB peak RSS and ran in 0.862 seconds.

On IREE local-task, official `state-spaces/mamba-130m-hf` revision `1e76775f628fbf1350fbe4dbb3d971ba64af25a1` matched an independent Transformers 5.12.1 CPU fp32 reference for whole prefill, split prefill, and decode. The worst maximum absolute difference was `1.488e-4`, with the same argmax. The seeded synthetic `FalconMambaForCausalLM` covered its B/C/dt RMS normalization and matched whole, split, and decode paths with worst maximum absolute difference `3.725e-7` and the same argmax.

ROCm gfx1151 acceptance was captured at exact source `3a1688f7daaf9bae8bea09997b804e6509357c25`: 551 library tests passed with 21 ignored; local-task and HIP mixer probes covered 12 cases; the real Mamba and synthetic Falcon reference and CLI checks matched 40/40 tokens; B4/N8 batching, cancellation, slot recycling, and four concurrent server requests passed; capacity 4096 compiled; five hybrid tests and Mamba2 regressions passed; and the integrated real Nemotron-H checks passed. All ROCm device work used the ROCm guard. HTTP checks compare rendered text because that API does not expose token IDs, and optimized `test-fast` binaries make no performance claim. The [ROCm acceptance record](https://github.com/lablup/mlxcel/pull/3413#issuecomment-7147199610) contains commands and raw-evidence provenance.

After that ROCm run, the branch was rebased onto PR #3411. The Mamba1 scan, parser, recurrent emitter, and Python oracle blobs are byte-identical to `3a1688f7`; the integration adds disjoint Nemotron-H MoE paths and the CI allowlist correction. CUDA, full library tests, and registered goldens were rerun on the integrated source. ROCm was not rerun after this source-neutral integration, so the two exact-SHA evidence boundaries remain explicit.

## Learning Points

Bounding a consumer is insufficient when the producer already creates a capacity-sized tensor. The safe boundary must begin before broadcasting the token projections into recurrence operands.

Recurrent model validation needs both numerical references and state-transition coverage. Whole-prefill parity alone cannot detect a wrong carried convolution tail, a stale recycled slot, or a discrepancy between split prefill and decode.

## Follow-up Actions

- [x] Validate the integrated capacity-4096 graph on CUDA with an independent oracle.
- [x] Validate real Mamba, synthetic Falcon-Mamba, batching, serving, cancellation, and recurrent regressions on ROCm gfx1151.
- [x] Rebase onto the merged Nemotron-H MoE work without changing Mamba1 numerical-source blobs, then rerun integrated library, golden, lint, and CUDA gates.

## References

- Issue #3348
- Epic #564
- PR #3413
- `spike/openxla/FINDINGS_ssm.md`
- [ROCm acceptance record](https://github.com/lablup/mlxcel/pull/3413#issuecomment-7147199610)
