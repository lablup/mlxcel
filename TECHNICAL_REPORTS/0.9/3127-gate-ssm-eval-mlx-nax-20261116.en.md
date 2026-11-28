# Technical Report: PR #3127 - gate the SSM eval boundary on MLX's NAX flag

**Date**: 2026-11-16
**Status**: PR open (closes #3123), verified on M3 Ultra and M5 Max
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

Nemotron-H and PLaMo 2 force an `eval` after each SSM mixer to stop the M5 Max NAx GEMM from fusing a lazy float32/float16 graph into NaN. That workaround was gated on mlxcel's own reading of the chip (`is_m5_neural_accelerator()`, from the sysctl brand string), while MLX decides whether to run NAx kernels from `metal::is_nax_available()`. This PR gates the workaround, and the finite-decode test that replaced an older boundary, on MLX's flag through a cached helper `hardware::mlx_nax_available()`.

---

## 1. Problem Statement

#2806 showed the two views disagreeing: the window SDPA dispatch test failed on every nightly run on two runners because the test used the brand string while MLX ran NAx kernels. PR #3110 fixed only that test. Production code still keyed the NaN workaround on the brand string, so on a host where MLX runs NAx but the brand string is not recognised as M5, the workaround stayed off and NaN logits were possible. In the other direction (an M5 told by `MLX_METAL_GPU_ARCH` or a `MLX_METAL_NO_NAX` build to skip NAx), the workaround paid a per-layer sync for nothing.

---

## 2. Technical Decisions

### 2.1 Gate on what MLX dispatches

The NaN belongs to the kernel MLX picks, so the gate follows MLX's flag. `is_m5_neural_accelerator()` stays as the chip-level view, and its doc comment lists when the two can differ (`MLX_METAL_GPU_ARCH`, `MLX_METAL_NO_NAX`, non-Metal backends, unrecognised brand strings).

### 2.2 Cached helper instead of FFI in model code

`mlx_nax_available()` wraps the bridge from #3110 in a `OnceLock<bool>`. MLX caches the answer itself, so the value cannot change within a process, and the per-forward cost is one atomic load. The bridge returns false off Metal. It reports GPU capability, not the default device, so a CPU-device run on an NAx host still takes the `eval`; that costs a sync and is never wrong.

### 2.3 No "views agree" test

The issue allowed either a test asserting agreement or a documented difference. A test would turn nightly red on the #2806 runners with no code defect, so the doc comment route was taken.

---

## 3. Change Summary

| Item | Value |
|---|---|
| Files changed | 4 |
| Tests added | 2 (`hardware::tests`) |

- `hardware.rs`: `mlx_nax_available()`; expanded doc on `is_m5_neural_accelerator()`.
- `nemotron_h.rs`, `plamo2.rs`: the `eval` boundary gates on `mlx_nax_available()`.
- `tests/mamba2_hybrid_decode_finite.rs`: NOT APPLICABLE only where MLX runs no NAx kernels.

### Validation

- M3 Ultra: `hardware::` 45 passed, printing `MLX NAX false; sysctl view M3`; the finite-decode tests report NOT APPLICABLE, as expected. Clippy, fmt and contract checks pass.
- M5 Max: `hardware::` 45 passed, printing `MLX NAX true; sysctl view M5, Neural Accelerator true, macOS NA support true`, so on M5 Max the gate is on, as before. `mamba2_hybrid_decode_finite` ran (not NOT APPLICABLE) and passed 2/2: Falcon-H1 tiny and Granite 4.0-H tiny each decoded 64 finite, coherent tokens.

---

## 4. Follow-up Actions

- Watch nightly on the two runners from #2806: the workaround now turns on there if MLX runs NAx kernels.
