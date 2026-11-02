# PR #2744: Clear the 11 CUDA lib test failures on GB10

**Date**: 2026-10-30 (report written pre-merge; the PR was open and in draft at the time)
**Status**: Open PR, validated on CUDA (GB10, sm_121) only. No Metal host was available, so the Metal run requested for items 1, 2 and 4 did not happen
**Languages**: Rust, C++ (CUDA overlays), Python, JSON, Markdown
**Risk Level**: Medium (two production changes on shared paths: the Qwen4-Exp gated-delta precise policy and the SAM 3D Body casts; two CUDA-only MLX overlays that change conv and compiled-kernel behavior for every CUDA build)

## Executive Summary

PR #2744 (closes #2739, epic #2348) clears eleven lib test failures found when the epic's post-merge gate ran the CUDA lib suite on GB10 for the first time (aarch64, sm_121, `--profile test-fast --features cuda`, `--test-threads=1`). No test was ignored and no tolerance was loosened without a recorded measured CUDA value.

The failures had four causes. Production code diverged from upstream MLX where mlxcel's CUDA-only bf16+f32 promotion overlay yields bf16 (Qwen4-Exp gated-delta state, SAM 3D Body linear, norm, layer scale, RoPE and backbone stem, and the SAM 3D Objects head norm). Backend-specific oracles were captured on ROCm or x86_64 and are now keyed on CUDA with values measured from upstream `mlx` 0.32.4 CUDA (SAM 3D Objects modulate, tanh GELU and bicubic resize, Sapiens2 RoPE) or on host architecture (the MHR ULP bound). Two real CUDA backend bugs were fixed with new MLX source overlays: a cuDNN conv with no execution plan now falls back to the gemm conv, and fused `AsType` now uses the eager cast so complex64 to real compiles.

The Kokoro finding is a production bug, not only a test failure: `src/lib.rs` defaults `MLX_ENABLE_TF32=0`, so Kokoro synthesis threw on GB10 in normal use.

## 1. Problem Statement

Issue #2739 lists eight items, eleven tests. None is in image generation code or code #2348 changed; the CUDA lib suite had simply never been run against them (the issue cites "there is no CUDA data point" in #2613, Metal-only validation in #2631, and "CUDA has no capture and is not keyed" in the Sapiens2 test). All reproduced on rerun and with `MLX_ENABLE_TF32=0`.

The dtype failures come from `overlays/cuda/mlx/dtype.cpp`, which makes bf16 with f32 promote to bf16 (to avoid `copy_v` kernel overhead). The contract from #2087 and #2093 applies: keep the overlay, cast explicitly in production code that needs f32, and branch on `cfg!(feature = "cuda")` only in tests that do nothing but pin promotion.

## 2. Change Summary

19 files, +1278 / -36, three commits (`06d145af` main fix, `d9493e97` merge of origin/main `060b5051`, `2862e24d` narrowed conv fallback).

| Item | Area | Files | Change |
|---|---|---|---|
| 1 | qwen4_exp state | `src/models/gated_delta.rs` | On the non-Metal T=1 and sequential paths the precise policy casts q, k, v, g, beta and the state to f32 and casts the output back to the query dtype. New `compute_g_float32` used only by `gated_delta_update_precise`. |
| 2 | sam3d_body | `sam3d_body/layers.rs`, `backbone.rs`, `rope.rs`, `dense/mod.rs` | `Linear`, `Norm`, `LayerScale`, `apply_rope` and the backbone stem cast explicitly to upstream's promoted dtype. Weights stay bf16. |
| 3 | MHR geometry | `sam3d_body/mhr_tests.rs` | Bit-exact only on x86_64 Linux; other hosts use the 2 ULP bound, now documented with the GB10 measurement. |
| 4 | Objects layers | `mlx_cxx_objects.cpp`, `sam3d_objects/layers_tests.rs`, `cuda_oracle_tests.rs` | Head norm f32 cast (overlay bug); modulate, residual and tanh GELU bf16 grid keyed on CUDA. |
| 5 | Sapiens2 RoPE | `sapiens2/backbone_tests.rs` | CUDA keyed with the ROCm value `-4.559755325317383e-6`. |
| 6 | bicubic resize | `interpolate_objects_tests.rs` | CUDA keyed, 12 of 48 f32 values differ by 1 to 3 ULP. |
| 4/6 data | fixtures | `tests/fixtures/sam3d_objects/cuda-oracle.json`, `generate_cuda_oracle.py`, `README.md` | Sparse overrides (index and value per key) captured from upstream mlx 0.32.4 CUDA. |
| 7 | Kokoro | `overlays/cuda/mlx/backend/cuda/conv.cpp`, `conv_no_plan_tests.rs` | cuDNN no-plan fallback to the gemm conv. |
| 8 | complex64 cast | `overlays/cuda/mlx/backend/cuda/compiled.cpp`, README | Fused `AsType` emits eager `cast_to<T>()`. README file count 27 to 29. |

## 3. Technical Decisions

### 3.1 Item 1: cast operands, not just the state

**Context:** off Metal, T=1 went to a fused decode step that ignored `preserve_precision`, and `add(state_f32, k * delta_bf16)` is bf16 under the overlay. The issue's hypothesis (state dtype only) was incomplete: `compute_g` also returned a bf16 gate on CUDA because `exp(A_log)` (f32) times a bf16 softplus gives bf16.

**Decision:** a `PreciseOperands` helper casts q, k, v, g and beta to f32 before the step, the state is f32, and `cast_output` returns the output in the query dtype. `compute_g_float32` promotes the softplus term first. Each cast is exact and is the cast MLX inserts itself on Metal, so Metal is bit-identical by construction. Qwen3Next, Qwen3.5 and KimiLinear call `compute_g` with `preserve_precision = false` and are unchanged.

**Alternative ruled out:** changing `compute_g` for everyone. It would alter three other model families on CUDA for no failing test.

### 3.2 Items 4 to 6: fix production first, key the oracle only when forced

Per the acceptance criteria, f32 intermediates were tried first. They do not reproduce the ROCm oracle (13 of 32 modulate elements differ against 11 of 32 for the compiled graph), so modulate, residual and the tanh GELU bf16 grid are keyed on CUDA. The Objects head norm was different: it multiplied an f32 activation by bf16 gamma, an overlay-induced bug fixed with an f32 cast in `mlx_cxx_objects.cpp`, and its fixture then matches unchanged.

CUDA values are stored as sparse overrides over the pinned oracle (`cuda_oracle_tests.rs::keyed`), asserting element count and index/value length, so a stale fixture fails loudly. The generator records its provenance (mlx 0.32.4 CUDA, mlx-vlm 0.7.8, GB10 12.1, driver 580.178.04, `MLX_ENABLE_TF32=0`). The test contract is: mlxcel must equal upstream CUDA bit for bit on CUDA and the pinned oracle everywhere else.

### 3.3 Item 3: key on architecture, keep the bound

Measured on GB10: 41 of 266 `body_params` and 6 of 54 `hand_params` elements differ by exactly 1 ULP, every other field is bit-exact. The cause is aarch64 glibc `atan2`/`sin`/`cos` rounding, not CUDA. The gate changed from `target_os = "linux"` to `all(linux, x86_64)`; the 2 ULP bound is kept (M1 Ultra 2, GB10 1).

### 3.4 Item 7: narrow cuDNN fallback

**Root cause:** with TF32 off MLX deselects cuDNN's tensor-core plans. On sm_121 that leaves no plan for f32 dilated conv1d (Kokoro generator shapes `x=(1,1200,256) w=(256,3,256) d=3` and `k=11 d=5` at 256 and 128 channels), and upstream `graph.build()` throws "No execution plans support the graph". Reproduced with upstream mlx CUDA; TF32 on passes.

**Decision:** the overlay returns `nullopt` only when `build()` fails with `GRAPH_EXECUTION_PLAN_CREATION_FAILED`, so the conv takes upstream's existing `gemm_conv` path, cached per shape as `CONV_FALLBACK`. The first commit caught more; `2862e24d` narrowed it so any other cuDNN build error still throws as upstream does and supported convs take the unchanged cuDNN path.

**Rejected:** flipping the `MLX_ENABLE_TF32=0` default (changes numerics for every model) and rewriting Kokoro's conv layout (fixes one family, leaves every other f32 dilated conv exposed).

### 3.5 Item 8: mirror the ROCm fix

Upstream CUDA compiled codegen emitted `static_cast<float>(complex64_t)` for the `AsType` after `Abs`. The overlay emits `cast_to<T>()` (the eager `CastOp`, which keeps the real part and tests both parts for bool) and adds `cast_op.cuh` to the JIT includes. This is the CUDA analog of the ROCm `compiled.cpp` from #2613.

## 4. Validation

All on GB10 with `--test-threads=1` under `gpu-lock`.

- All 11 target tests pass; the two Kokoro tests ran with `models/kokoro-82m` present.
- mlxcel lib filters, all passing: `vision::dense` 193, `models::qwen4_exp` 70, `models::gated_delta` 10, `models::qwen3_next` 14, `models::qwen3_5` 32, `kimi` 173, `models::kokoro` 10. mlxcel_core: `compiled_dtype_cast_tests` 10, `compiled` 46, `conv` 16, `conv_no_plan_tests` 1 (pins the three Kokoro shapes against the CPU stream).
- Item 1 regression guard: a logit trace on `qwen3.5-0.8b-4bit` (`MLX_ENABLE_TF32=0`) between main's `gated_delta.rs` without the new overlays and this branch is byte-identical (256 positions, ppl 28.6184 in both).
- Clippy `-D warnings` for `mlxcel` and `mlxcel-core`, `make verify-*` (versions, kernel-dtype-keys, kernel-port-dispatch, llama-compat, rocm-overlay, python-tooling, env-var-docs, fmt) and `check_binary_assets.py` pass. The full set was re-run on the final head after merging origin/main `060b5051` and narrowing the conv fallback.
- A grep confirmed both new overlays are copied into `mlx-src` at build.

## 5. Learning Points

- The CUDA promotion overlay is a standing source of bf16 where upstream has f32. When a CUDA dtype assertion fails, check for an f32 operand meeting a bf16 one before suspecting the kernel, and look at every operand, not only the state (the gate here).
- TF32 off is not a neutral switch on cuDNN. It removes plan candidates, and a shape can end up with none. The error text alone points at cuDNN, not at the TF32 default.
- Match the fallback trigger to the failure. Catching only the plan-creation error code keeps all other cuDNN diagnostics intact.
- A pinned oracle captured on another backend is a host-specific value. Sparse overrides plus a recorded generator keep CUDA keying auditable without duplicating whole tensors.
- Rule out the cheap explanation first (f32 intermediates) and record that it failed, so the next reader does not retry it.

## 6. Limitations and Unverified Points

- No Metal run. The issue asked for `vision::dense` and `models::qwen4_exp` on Metal for items 1, 2 and 4. The argument for safety is structural: each change is an explicit `astype` to the dtype Metal's unpatched promotion table already produces, and the Metal precise gated-delta kernel path never reaches the changed fallback.
- The overlays compile only with `MLX_BUILD_CUDA`, so they were not exercised on other backends.
- qwen4_exp has no checkpoint on this host. Item 1's evidence is the parity and synthetic tests plus the Qwen3.5 logit trace, which shows the other families unchanged but is not a Qwen4-Exp end-to-end run.
- The CUDA oracle values come from one GPU (GB10, one driver). Other CUDA architectures could round differently and would need their own keys.
- The aarch64 MHR bound rests on two measured hosts.
- The optional step3p7 TF32 note (`tests/step3p7_parity.rs:566`) was not addressed.

## References

- Issue #2739 (closes), epic #2348, PR #2613 (ROCm compiled cast), PR #2631, #2580, #2578, #2550, #2475, #374 (introducing changes), #2087 and #2093 (promotion overlay contract).
- Related: PR #2740 and PR #2730 reports.
- Upstream: `mlx` 0.32.4 CUDA, mlx-vlm 0.7.8.
- Files: `src/lib/mlxcelverse/overlays/cuda/README.md`, `tests/fixtures/sam3d_objects/README.md`.
