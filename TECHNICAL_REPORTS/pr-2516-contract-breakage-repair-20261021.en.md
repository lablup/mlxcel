# PR #2516: repair the three contract breakages left by PR #2505

**Date**: 2026-10-21
**Status**: Completed
**Risk Level**: High (contract/FFI)
**Languages**: Rust, C++

---

## Executive Summary

PR #2505 (merged while CI was frozen) left three repo-wide contract breakages. This PR repairs all of them in one change: the two kernel-port dispatch violations in `mlx_cxx_kernels.cpp` are routed through `mlxcel::select_kernel_port` with Metal-only port tables and a `-> Result<()>` bridge contract (so a backend without a port errors instead of aborting), the new `MLXCEL_INKLING_MOE_KERNEL` env var is documented as an operator rollback toggle, and the fmt violations in `inkling/mlp.rs` and `switch_layers.rs` are fixed mechanically. Every contract target now exits 0.

---

## 1. Problem Statement

### 1.1 Background

PR #2505 (`perf(inkling): decode via fast SDPA and fused MoE kernels`) merged while GitHub Actions was frozen (no runs repo-wide since 2026-10-20 15:03 UTC), so nothing checked its contract surfaces. The full gate on the next PR head caught all three breakages at once (CT_EXIT=2).

### 1.2 Existing Issues

- `make verify-kernel-port-dispatch` failed: `mlx_cxx_kernels.cpp:1660` and `:1773` reached kernel holders directly instead of through `mlxcel::select_kernel_port`, so a backend without a port silently ran the Metal arm (the crash class the pattern exists to prevent, #1801/#1803/#1885/#2018/#2147).
- `make verify-python-tooling` failed: the new env var `MLXCEL_INKLING_MOE_KERNEL` (`inkling/mlp.rs:142`) was neither documented on `docs/environment-variables.md` nor allowlisted.
- `cargo fmt --all -- --check` failed: 4 diffs in `inkling/mlp.rs`, 1 in `switch_layers.rs`.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A C++ throw crossing a `noexcept` extern aborts the process | High (every non-Metal MoE request on the routed kernels) | Deterministic on non-Metal hosts before this fix |
| Operators cannot discover the rollback toggle | Low | High |

---

## 2. Technical Review

### 2.1 Security

Clean. The two new port tables are `static const` function-local (immutable after magic-static init), captureless lambdas as plain function pointers, no shared mutable state added. A systematic audit of every `select_kernel_port`-reachable throw confirmed all bridge-declared launchers are `-> Result<...>` (fused_xielu, bitlinear_matmul, ssm_update_kernel, mamba1_selective_scan, fused_add3_layer_norm, fused_moe_forward, fused_moe_expert/geglu, plus the two fixed here) and C++-internal helpers are only reachable under Result-declared entries or `has_kernel_port` guards. C++ signatures stayed `void` with no `noexcept`, so the cxx trampoline converts throws to `Err`.

### 2.2 Performance

Negligible. The `select_kernel_port` success path is one read of the once-resolved static backend enum plus a switch — the same holder access the old direct call performed; `std::string` construction exists only on the throw path. Side benefit: non-Metal hosts no longer touch the Metal holder at all (the old direct call registered `fast::metal_kernel(...)` unconditionally, which was both the crash source and wasted work).

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. The routed kernels are Metal-only; on a correctly-gated Metal host behavior is unchanged. Non-Metal hosts now take the graph path from the start (previously: abort).
- **New Dependencies**: none.

### 2.4 Code Quality

- Test Coverage: inkling filter 92 passed / 0 failed on CUDA (the Metal-gated parity tests take their designed skip paths here; real-kernel validation is Metal-host work, the pre-existing pattern). All contract targets re-verified exit 0 on this tree.
- fmt: 5 pure line-rewrapping hunks, no semantic change.

---

## 3. Technical Decisions

### 3.1 `-> Result<()>` bridge declarations (required, not stylistic)

**Context:** `select_kernel_port` throws when no port names the entry point and fallback. A `void noexcept` extern would let the throw cross the FFI boundary and end the process.

**Chosen:** both bridge declarations changed to `-> Result<()>` (`fn inkling_moe_route_kernel`, kept `unsafe fn inkling_moe_down_combine_kernel`; cxx allows unsafe+Result, precedent `fused_moe_forward`). C++ signatures stay `void` (matching the in-repo `try_eval` pattern), so cxx wraps the call and converts any thrown MLX exception into a Rust `Err`. Documented in `kernel_port.h` terms (the gate `verify-kernel-port-dispatch` demands this after a launcher signature change).

### 3.2 Call sites fall back on `Err`

Route: `is_ok().then_some((idx, wk, g))` with the `None` arm falling back to `route_weights` (the pre-kernel graph routing chain); null UniquePtrs are moved into an eagerly-built tuple that `then_some` discards — no dereference, no leak. Down-combine: `.is_ok().then(|| reshape(&out, ...))` is lazy, so `reshape` never evaluates on `Err`; falls back to `forward_with_expert_scales` + `moe_weighted_sum`. The direct-call test gate-then-expects per the `kernel_port.h:81-82` contract.

### 3.3 Down-combine gates on actual fast-route success (review adoption)

The down-combine dispatch initially gated on the env+backend gate (`route_kernel`) rather than the actual fast-route success; in the degraded gate-true-but-launch-errs state it would attempt a launch that refuses the same way. Adopted from review: gate on `fast_route_ok` (captured before the match consumes the tuple). On a correctly-gated host behavior is unchanged; the degraded state goes straight to the graph path. The two launcher doc comments now state that production consumes the `Err` through the fallback rather than "only from a direct call".

### 3.4 Documented, not allowlisted

`MLXCEL_INKLING_MOE_KERNEL` is an operator-facing rollback toggle (0 disables; unset/other enables; default on), the same category as `MLXCEL_SSM_KERNEL`, so it is documented on `docs/environment-variables.md` (next to the `MLXCEL_FUSED_MOE` family) rather than allowlisted. The doc row's values/default/effect were verified against the code.

---

## 4. Implementation Details

**File: `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp`** — `inkling_route_ports()` and `inkling_down_combine_ports()` (`.metal` set, `.cuda = nullptr, .rocm = nullptr` explicit, `turbo4_delegated_ports` template); both direct holder calls replaced with `mlxcel::select_kernel_port`.

**File: `src/lib/mlxcel-core/src/lib.rs`** — both bridge declarations `-> Result<()>`; launcher docs state the Err-reaches-production-through-fallback contract.

**File: `src/models/inkling/mlp.rs`** — route call restructured with the `route_weights` fallback; down-combine falls back to the gather path; `down_parts` gates on `fast_route_ok`; the direct-call test gate-then-expects.

**File: `docs/environment-variables.md`** — `MLXCEL_INKLING_MOE_KERNEL` row in "Hardware and kernel diagnostic variables".

**File: `src/models/switch_layers.rs`** — whitespace-only reflow.

---

## 5. Verification

| Check | Result |
|---|---|
| make verify-kernel-port-dispatch | exit 0 (12 launcher files route, 0 exempt; negative-coverage cases pass) |
| make verify-python-tooling | exit 0 (139 tests OK) |
| make verify-fmt / cargo fmt --all -- --check | exit 0 / clean |
| Combined contract targets | exit 0 |
| scoped clippy (lib+tests, -D warnings) | exit 0 |
| inkling filter (CUDA, gpu-lock, TF32=0, --test-threads=1) | 92 passed / 0 failed (Metal-gated parity tests skip by design) |
| Full gate (161 bins, HEAD 60c744eb) | build 0, clippy 0, doc 0, CT_EXIT=0; 160/161 (mllama_parity 7/1 = known #2467 flake, control-proven environmental on main) |
| Review findings adopted | `a2ca2104`: down-combine gates on `fast_route_ok`, both launcher docs precise; re-verified fmt 0 / clippy 0 / inkling 92/0 |
| Security review | CLEAN; FFI audit clean; fallbacks safe; per-call cost negligible |

The routed kernels themselves are not exercisable on this CUDA host (Metal-only); graph-path tests pass and real-kernel validation remains Metal-host work.

---

## 6. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 5 (+2 in follow-up commit) |
| Lines | ~+120/-30 |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `60c744eb` | fix | fix: repair the three contract breakages left by PR #2505 |
| `a2ca2104` | fix | fix(inkling): gate down-combine on fast-route success, precise Err docs |

---

## 7. Follow-up Actions

### Monitoring Required

- First Metal-host run of the routed kernels (route parity, down-combine) in the next release cycle.
- Watch for further contract drift while CI remains frozen: each unit's full gate substitutes for it.

### Future Improvements

- `mod gemma3n_math` (src/lib/mlxcel-xla/src/emitter/mod.rs:59) dead-code cleanup (from the #2513 review, separate surface).

---

## Appendix

### A. Review verdicts

pr-reviewer: PASS all six focus areas, no CRITICAL/HIGH; one LOW doc-wording observation (adopted in `a2ca2104`). pr-security-checker: CLEAN, no CRITICAL/HIGH/MEDIUM; one optional LOW efficiency nit (the down_parts tightening, adopted in `a2ca2104`); the FFI audit found every `select_kernel_port`-reachable throw already Result-declared.
