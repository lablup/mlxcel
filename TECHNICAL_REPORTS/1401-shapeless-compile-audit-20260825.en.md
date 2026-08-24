# Technical Report: PR #1401 - Audit shapeless MLX compile sites

**Date**: 2026-08-25
**Status**: Completed
**Languages**: Rust, C++, Bash
**Risk Level**: Medium

## Executive Summary

PR #1401 turns the previously ad hoc investigation of `mlx::core::compile(..., shapeless=true)` into a repeatable hardware audit. Every production shapeless compile construction in `mlx_cxx_bridge.cpp` now passes through an opt-in eager oracle, while permanent regressions protect the quiet no-op shapes that could otherwise return plausible but wrong tensors.

The final implementation preserves all compiled fusion. Local NVIDIA CUDA and Apple Silicon Metal both cleared all 20 current callables across f32, bf16, and f16 where supported, including first-use and warmed-cache calls; no model checkpoint was read or downloaded.

## 1. Problem Statement

PR #1391 removed a compiled min-p filter after it silently returned its input on Metal. The same C++ bridge contained many other shapeless compile constructions, and reasoning from output shape alone could not clear same-shape operations such as softcap and residual clipping.

The issue's original line/function table had also drifted by the time implementation began: `softplus` was already eager, a cited QKV location had become the quantized MoE expert path, and masked versus unmasked softcap SDPA were two separate compiled callables. The current inventory is 20 callables.

If left unaudited, a future MLX pin could silently disable an activation, attention transform, or state update while returning a shape- and dtype-valid tensor. The highest-risk cases are filters and bounded transforms whose no-op output remains numerically plausible.

## 2. Technical Decisions

### 2.1 Central opt-in eager oracle

Every shapeless construction calls `compile_shapeless_audited(site, eager_fn)`. With `MLXCEL_SHAPELESS_COMPILE_AUDIT` unset, it returns the same compiled callable and adds no per-call comparison or registry work. When enabled before the first compiled function is initialized, it runs the compiled and eager graphs on identical inputs, checks output count, shape, dtype, and numeric closeness, and records calls by dtype/shape signature.

This design avoids maintaining a second handwritten reference graph for the 20 production callables. The exact lambda passed to MLX compile is also the eager oracle, so later implementation changes cannot update one side and forget the other.

### 2.2 Preserve fusion after measured clearance

An intermediate implementation removed compile from softcap, clip-residual, and softcap SDPA proactively. Review rejected that approach because it would discard attention fusion without evidence of divergence. The final CUDA and Metal audits cleared those sites, so the optimized paths remain in production and permanent independent-reference tests guard their semantics.

### 2.3 Hardware audit, not a model CI expansion

`scripts/audit_shapeless_compile.sh` uses only small synthetic tensors. On Linux it requires `nvidia-smi`, reads the attached GPU's compute capability, and exports `MLX_CUDA_ARCHITECTURES`; on macOS it selects `metal,accelerate`. It rejects direct shapeless compile constructions outside the central wrapper and falls back to `grep` when a minimal Apple runner does not have `rg`.

No permanent GitHub Actions job was added. The Apple audit ran through a temporary branch-only workflow that was removed before PR creation, leaving no Actions diff. This preserves the existing project boundary: hosted Actions may use the established approximately 0.6B fixtures, while larger checkpoint validation belongs on local hardware with existing weights.

## 3. Implementation Details

### 3.1 Audit flow

```text
Production (environment unset)
call site -> compile_shapeless_audited -> original compiled callable

On-demand audit (MLXCEL_SHAPELESS_COMPILE_AUDIT=1)
call site -> compiled graph ----+
             eager graph -------+-> shape/dtype/allclose -> per-site report
                                      first call + warmed call per signature
```

### 3.2 Permanent quiet-site regressions

- `compiled_softcap`: nonuniform logits above and below the cap, explicit `tanh(scores / cap) * cap` reference, and an assertion that the result is not the input.
- `compiled_clip_residual`: f16 overflow-boundary values, explicit f32 widen/add/clip/f16 reference, and an assertion that the result is not the first input.
- Masked and unmasked softcap SDPA: nonuniform Q/K/V values and an independent eager attention reference across f32, bf16, and f16.
- GQA softcap SDPA: explicit repeated-K/V eager attention reference, preventing a shape-only test from accepting a plausible no-op.

## 4. Hardware Audit Baseline

The CUDA host was an NVIDIA GB10 with driver 580.173.02, CUDA 13 runtime visibility, and compute capability 12.1. The Metal result is GitHub Actions run [32746852051](https://github.com/lablup/mlxcel/actions/runs/32746852051) on `self-hosted-macos-26-arm64`.

`6 / 3 / 3` means six calls, three distinct dtype/shape signatures, and all three signatures warmed by a second call. Clip-residual is intentionally f16-only and therefore reports `2 / 1 / 1`.

| Site | Coverage (calls / signatures / warmed) | CUDA | Metal |
|---|---:|---|---|
| `compiled_swiglu_activation` | 6 / 3 / 3 | PASS | PASS |
| `compiled_relu_squared` | 6 / 3 / 3 | PASS | PASS |
| `compiled_silu` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gpt_oss_swiglu_activation` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gelu` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gelu_approx` | 6 / 3 / 3 | PASS | PASS |
| `compiled_geglu_activation` | 6 / 3 / 3 | PASS | PASS |
| `compiled_geglu_approx_activation` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gelu_topk` | 6 / 3 / 3 | PASS | PASS |
| `compiled_softcap` | 6 / 3 / 3 | PASS | PASS |
| `compiled_clip_residual` | 2 / 1 / 1 | PASS | PASS |
| `compiled_softcap_sdpa_nomask` | 6 / 3 / 3 | PASS | PASS |
| `compiled_softcap_sdpa_masked` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gelu_mlp_forward` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gelu_approx_mlp_forward` | 6 / 3 / 3 | PASS | PASS |
| `compiled_gelu_approx_mlp_forward_global_scale` | 6 / 3 / 3 | PASS | PASS |
| `compiled_per_layer_input_gate` | 6 / 3 / 3 | PASS | PASS |
| `compiled_moe_expert_forward` | 6 / 3 / 3 | PASS | PASS |
| `fused_gated_delta_decode_step_scalar_gate` | 6 / 3 / 3 | PASS | PASS |
| `fused_gated_delta_decode_step_dim_gate` | 6 / 3 / 3 | PASS | PASS |

## 5. Compatibility, Security, and Performance

- **Breaking changes**: None. Existing bridge function names and production compiled behavior remain unchanged.
- **New dependencies**: None. The audit script uses tools already expected on the selected backend and has an `rg`/`grep` inventory fallback.
- **Security**: The audit is opt-in and synthetic. It reads no prompts, tokens, credentials, or model weights, and the report contains only site names and tensor signatures.
- **Performance**: Audit-disabled calls return the original compiled function. The quiet-site fusion is retained after direct CUDA and Metal clearance.
- **PR #1395**: Already included in the branch base, but unrelated; it fixes MLA KV-cache/Turbo4 safety rather than MLX compile behavior.

## 6. Change Summary

Implementation diff before adding this report:

| Item | Value |
|---|---:|
| Files changed | 5 |
| Lines added | 832 |
| Lines deleted | 114 |
| Permanent regression tests | 4 |
| Audited shapeless callables | 20 |
| New runtime dependencies | 0 |
| Permanent workflow changes | 0 |

Key files:

- `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`: central wrapper, registry, report, and complete production-site routing.
- `src/lib/mlxcel-core/src/ffi_tests.rs`: multi-dtype hardware harness and quiet-site regressions.
- `scripts/audit_shapeless_compile.sh`: backend selection, source inventory enforcement, and isolated ignored-test invocation.
- `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.h`, `src/lib/mlxcel-core/src/lib.rs`: audit report bridge.

## 7. Validation

- `scripts/audit_shapeless_compile.sh` on local CUDA: 20/20 PASS; 19 sites at `6/3/3`, clip-residual at `2/1/1`.
- Apple Silicon Metal run 32746852051: the same 20/20 table; quiet-site regressions 4/4 PASS.
- `cargo test -p mlxcel-core --profile test-fast --features cuda --lib eager_regression -- --nocapture --test-threads=1`: 4 passed.
- `cargo test -p mlxcel-core --profile test-fast --features cuda --lib compiled_ -- --test-threads=1`: 30 passed.
- `cargo clippy -p mlxcel-core --features cuda --all-targets -- -D warnings`: passed.
- `cargo fmt --all -- --check`: passed.

## 8. Follow-up Actions

- Re-run `scripts/audit_shapeless_compile.sh` on CUDA and Apple Silicon after every MLX pin bump.
- Treat a missing site, an unwarmed signature, or a `FAIL` row as a hard qualification failure; remove compile only from a measured divergent site or document the exact safe precondition.
- Keep real-checkpoint validation separate: use existing local weights for checkpoints larger than the established hosted fixture boundary.

## References

- Issue #1392: shapeless compile audit request.
- PR #1391: motivating compiled min-p no-op fix.
- PR #1395: adjacent but unrelated MLA KV-cache/Turbo4 safety fix already present in the base.
