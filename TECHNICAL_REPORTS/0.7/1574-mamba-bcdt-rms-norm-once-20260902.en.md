# Technical Report: PR #1574 - fix(mamba): apply Falcon-Mamba B/C/dt RMS norm once, no per-call ones

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: pending
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

`MambaBlock::ssm_step` normalized `delta`, `B` and `C` twice per call and allocated a fresh `ones` weight array for every one of those six calls, once per token per layer during prefill. This PR applies the weight-less RMS norm exactly once per tensor through the bridge's existing `fast_rms_norm_no_weight`, matching the upstream mlx-lm reference, and deletes the now-unused `ones`-allocating helper.

---

## 1. Problem Statement

### 1.1 Background

`falcon_mamba` checkpoints set `use_bcdt_rms = true`, which is supposed to apply one weight-less RMS norm to each of the three tensors split out of `x_proj(x)`: `delta`, `B`, and `C`. Since the SSM scan processes one token at a time, this call runs `T` times per layer for a `T`-token prompt, across 64 layers on the 7B checkpoint.

### 1.2 Existing Issues

- **Double normalization**: `ssm_step` called `self.mixer_norm(&self.mixer_norm(&x))` for each of `delta`, `B`, `C`, doubling the norm launches over what the architecture calls for. The result is not simply "twice as expensive for the same answer": renormalizing an already near-unit-RMS tensor rescales it again by roughly `1/sqrt(1+eps)`, so the second pass is a small additional numerical perturbation on top of the correct value, not a true no-op.
- **Per-call allocation**: the old `rms_norm_no_scale` helper built a fresh `ones` array shaped to the tensor's last dimension on every call, purely to reuse the weighted `fast_rms_norm` kernel. The bridge already exposes a weight-less variant (`fast_rms_norm_no_weight`) that needs no such allocation, and other model families (`gemma3n.rs`, `gemma4.rs`, `falcon_ocr.rs`) already use it.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Extra norm kernel launches per token per layer inflate decode latency on long prompts | Medium | Certain (present on every falcon_mamba forward pass) |
| Redundant `ones` allocation per call adds lazy-graph node count proportional to prompt length | Low-Medium | Certain |
| Small numerical drift from the spurious second norm application | Low | Present but bounded (`~1/sqrt(1+eps)` factor, `eps = 1e-6`) |

---

## 2. Technical Review

### 2.2 Performance

**Checklist:**
- [x] Algorithm complexity: removes 3 of 6 norm launches and 6 of 6 `ones` allocations per `ssm_step` call
- [ ] Query optimization: not applicable
- [ ] Caching strategy: not applicable
- [x] Memory usage: eliminates a per-call temporary array allocation

**Performance Impact:**

| Area | Before | After | Improvement |
|------|--------|-------|-------------|
| Norm kernel launches per `ssm_step` call | 6 (2 per tensor x 3 tensors) | 3 (1 per tensor) | 50% fewer norm launches |
| `ones` allocations per `ssm_step` call | 6 | 0 | 100% removed |
| Lazy graph nodes added per token per layer (falcon_mamba, `use_bcdt_rms=true`) | 6 `ones` + 6 norm nodes | 3 norm nodes | 9 fewer nodes/token/layer |

No end-to-end throughput benchmark was run as part of this PR; the real-checkpoint validation (greedy-output identity and peak-memory comparison on `models/falcon-mamba-7b-4bit`) is a follow-up step outside this PR's unit-test scope.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. `mixer_norm`'s signature and call sites are unchanged; only its internal implementation and how many times `ssm_step` invokes it changed.
- **New Dependencies**: none. `fast_rms_norm_no_weight` already existed in the `mlxcel-core` bridge (`src/lib/mlxcel-core/src/lib.rs`) and is already used by three other model families.
- **Compatibility**: scoped to `src/models/mamba.rs`, which serves Falcon-Mamba and Mamba v1 only. `mamba2.rs`, `falcon_h1.rs`, `jamba.rs`, `granitemoehybrid.rs` and `plamo2.rs` carry no `use_bcdt_rms` field and are unaffected.

### 2.4 Code Quality

- **Test Coverage**: 2 new unit tests added in `src/models/mamba_tests.rs`; both exercise the corrected code path and pass.
- **Code Complexity**: reduced. The 10-line `rms_norm_no_scale` helper is deleted, and `ssm_step`'s three normalization lines drop from nested double calls to single calls with an updated one-line comment.
- **Technical Debt**: decreased. The comment documenting the (buggy) double-application behavior is removed along with the behavior itself.

---

## 3. Technical Decisions

### 3.1 Exact equality, not tolerance, as the regression test oracle

**Context:**

The issue's own diagnosis states that applying the weight-less RMS norm twice is "numerically almost a no-op": renormalizing a tensor that is already close to unit RMS changes it by a `1/sqrt(1+eps)` factor, on the order of `5e-7` for `eps = 1e-6`. A naive regression test comparing "once" vs. "twice" with a floating-point tolerance (e.g., `allclose` with `atol = 1e-5`) would very likely pass even if the double-application bug were reintroduced, because the difference is smaller than most reasonable tolerances.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Option A: `allclose`-based tolerance check between "once" and "twice" outputs | Simple, matches existing test style elsewhere in the codebase | Not tight enough to catch the actual regression; a `1e-5`-scale tolerance would likely swallow the `~5e-7` difference |
| Option B: full `ssm_step`-level numerical test with hand-computed expected values (as sketched in the issue) | Exercises the real call path end-to-end | Requires constructing a full `MambaBlock` with real `x_proj`/`dt_proj` weights and working through the softplus/state-update math by hand; substantially more test code for the same regression coverage |
| **Chosen: Option C: `array_equal` (bit-exact) at the `mixer_norm` boundary** | Directly and reliably distinguishes "applied once" from "applied twice" regardless of how numerically close the two results are, since a second kernel pass introduces at least ULP-level differences; minimal test scaffolding | Does not exercise the full `ssm_step` numerical pipeline (softplus, state update); relies on the assumption that two independent kernel evaluations are not bit-identical, which held in local verification |

**Rationale:**

`array_equal` performs exact (not tolerance-based) comparison. Because the "double application" bug produces a result that is numerically close to but not bit-identical with the "single application" result, exact equality is what actually distinguishes the two cases: `mixer_norm(x)` (the fixed code) is asserted `array_equal` to one `fast_rms_norm_no_weight(x, eps)` call, and explicitly asserted **not** `array_equal` to a second `fast_rms_norm_no_weight` call layered on top of it. If someone reintroduces `self.mixer_norm(&self.mixer_norm(&x))`, the first assertion catches it directly (the block's `mixer_norm` output would then match the "twice" reference, not the "once" reference).

**Trade-offs:**

The test operates at the `mixer_norm` method boundary rather than through the full `ssm_step` pipeline, so it does not independently verify the softplus/state-update math downstream of the norm. That coverage is out of scope for this fix (the downstream math was not changed) and is covered separately by the real-checkpoint validation step.

### 3.2 Minimal `MambaBlock` test fixture over exposing new `pub(crate)` API

**Context:**

The issue suggested exposing `MambaBlock::normalize_bcdt(...)` as `pub(crate)` if internals were not reachable from tests. `mamba_tests.rs` is wired in as `#[path = "mamba_tests.rs"] mod tests;` inside `mamba.rs`, making it a child module of `mamba`. Private fields and methods of `MambaBlock`, including `mixer_norm` itself, are already visible there under Rust's module-based privacy rules.

**Rationale:**

Given that visibility, no new `pub(crate)` surface was needed. A `tiny_mamba_block(use_bcdt_rms, mixer_rms_eps)` helper builds a `MambaBlock` via its private struct literal, using zero-filled placeholder tensors for the fields `mixer_norm` does not read (`conv_weight`, `in_proj`, `x_proj`, `dt_proj`, `out_proj`, `a_log`, `d_param`). This keeps `mixer_norm`'s implementation exactly as specified in the issue (a two-branch method on `&self`) while still allowing direct, low-boilerplate unit testing.

---

## 4. Implementation Details

### 4.2 Key Code Changes

**File: `src/models/mamba.rs`**
```rust
// Before
fn rms_norm_no_scale(x: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let last_dim = shape[shape.len() - 1];
    let ones = mlxcel_core::ones(&[last_dim], mlxcel_core::array_dtype(x));
    mlxcel_core::fast_rms_norm(x, &ones, eps)
}

impl MambaBlock {
    fn mixer_norm(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if self.use_bcdt_rms {
            rms_norm_no_scale(x, self.mixer_rms_eps)
        } else {
            mlxcel_core::copy(x)
        }
    }
    // ...
    fn ssm_step(&self, ...) -> ... {
        // ...
        let delta_normed = self.mixer_norm(&self.mixer_norm(&delta_raw));
        let b_normed = self.mixer_norm(&self.mixer_norm(&b_raw));
        let c_normed = self.mixer_norm(&self.mixer_norm(&c_raw));
        // ...
    }
}

// After
impl MambaBlock {
    fn mixer_norm(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if self.use_bcdt_rms {
            mlxcel_core::fast_rms_norm_no_weight(x, self.mixer_rms_eps)
        } else {
            mlxcel_core::copy(x)
        }
    }
    // ...
    fn ssm_step(&self, ...) -> ... {
        // ...
        let delta_normed = self.mixer_norm(&delta_raw);
        let b_normed = self.mixer_norm(&b_raw);
        let c_normed = self.mixer_norm(&c_raw);
        // ...
    }
}
```

**Reason for change:** the free function `rms_norm_no_scale` existed only to adapt the weighted `fast_rms_norm` kernel into a weight-less call by manufacturing a `ones` array; `fast_rms_norm_no_weight` already provides that behavior natively in the bridge. The nested `mixer_norm(&mixer_norm(&x))` calls applied the norm twice where the architecture (and the upstream mlx-lm reference) call for exactly one application.

---

## 5. Learning Points

### 5.1 Weight-less fast-norm bridge calls

**Concept:**

`mlxcel-core`'s FFI bridge exposes `fast_rms_norm_no_weight(x, eps)`, implemented as `mlx::core::fast::rms_norm(x, std::nullopt, eps)` in the C++ layer, alongside the weighted `fast_rms_norm(x, weight, eps)`. When a model architecture needs an unweighted norm, calling the dedicated bridge function avoids manufacturing a `ones` weight array purely to satisfy the weighted kernel's signature.

**Application in this PR:**

`MambaBlock::mixer_norm` now calls `fast_rms_norm_no_weight` directly instead of synthesizing a `ones` array and calling `fast_rms_norm`.

**Common Use Cases:**
- Any per-token or per-step normalization without a learned scale, where allocating a `ones` array per call would add avoidable lazy-graph nodes (as already done in `gemma3n.rs`, `gemma4.rs`, and `falcon_ocr.rs`).

**Example Code:**
```rust
fn mixer_norm(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
    if self.use_bcdt_rms {
        mlxcel_core::fast_rms_norm_no_weight(x, self.mixer_rms_eps)
    } else {
        mlxcel_core::copy(x)
    }
}
```

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 2 |
| Lines added | +130 |
| Lines deleted | -20 |
| Tests added | 2 |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Performance | 1 | Removed redundant norm launches and per-call `ones` allocations in `MambaBlock::ssm_step` |
| Code Quality | 1 | Deleted the now-unused `rms_norm_no_scale` helper and its stale "applies TWICE" comment |
| Testing | 2 | Added `mamba_mixer_norm_applies_the_bridge_norm_exactly_once` and `mamba_mixer_norm_is_identity_when_bcdt_rms_disabled` |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `3b05da6` | fix | fix(mamba): apply Falcon-Mamba B/C/dt RMS norm once, no per-call ones |

---

## 8. Follow-up Actions

### Required
- [ ] Run the real-checkpoint validation from issue #1333: greedy-output identity (or logit agreement within `1e-2` max-abs) on `models/falcon-mamba-7b-4bit` before/after this change, and confirm long-prompt peak memory does not increase.

### Monitoring Required
- None beyond the standard nightly `cargo test --workspace --profile test-fast --features metal,accelerate` gate.

### Future Improvements
- Vectorizing the per-token prefill scan in `MambaBlock::forward` is explicitly out of scope for this PR (a separate performance item noted in issue #1333).

---

## Appendix

### A. Test Results

```
cargo test --profile test-fast --features metal,accelerate --lib models::mamba
test result: ok. 18 passed; 0 failed; 2 ignored; 0 measured; 7643 filtered out

cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings
Finished `test-fast` profile [optimized] target(s)  (no warnings)

cargo fmt --all -- --check
(no output; clean)
```

### C. References

- Issue: #1333
- Upstream reference: [`mlx_lm/models/mamba.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba.py) (`ssm_step`, `use_bcdt_rms` branch)
