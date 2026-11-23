# Technical Report: PR #3335 - Refuse mismatched experts in family-local MoE stacking

**Date**: 2026-11-23
**Status**: PR open (closes #3320). Unit suites pass; four local checkpoints load and generate unchanged
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

Six MoE families (DeepSeek-V3, Mistral4, PhiMoE, OLMoE, MiMo-V2, Mellum) stack per-expert tensors in their own sanitize code. A checkpoint where one expert's plane had a different shape reached `mlxcel_core::stack`, and MLX's `[stack] All arrays must have the same shape` exception crossed the cxx bridge and aborted the process. Each site now checks the parts with the shared `refuse_mismatched_expert_parts` helper from #3304 before stacking, so the same input fails the load with an error that names the offending key and both shapes.

---

## 1. Problem Statement

### 1.1 Background

#3304 (issue #3297) added `refuse_mismatched_expert_parts` to `src/models/switch_layers.rs` and used it in the shared stacking paths (`drain_numbered_expert_planes`, `stack_individual_experts_with_count`). Families that stack experts in their own loops were left out.

### 1.2 Existing Issues

- **Process abort instead of a load error**: `mlxcel_core::stack` and `stack_owned` are not fallible across the FFI. The C++ `std::invalid_argument` is uncaught, so `libc++abi` terminates the process (SIGABRT). A server loading such a checkpoint dies instead of reporting the bad file.
- **Non-fallible signatures**: Mistral4 `sanitize_weights`, OLMoE `sanitize_weights`, and Mellum `sanitize` / `stack_experts` returned `()` or a bare `WeightMap`, so they had no way to report the problem.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Corrupt or mis-converted checkpoint kills the serving process | High | Low |
| Error gives no hint about which tensor is wrong | Medium | Low |

---

## 2. Technical Review

### 2.3 Compatibility

- **Breaking changes**: three crate-internal signatures became fallible (`mistral4::sanitize_weights` is `pub`, used only inside the crate). Every caller propagates the error: `Mistral4Model::load`, `load_from_text_config`, `src/loading/mod.rs`, `src/loading/vlm_pixtral.rs`, `OlmoeModel::load`, `MellumModel::load`.
- **Well-formed checkpoints**: unchanged. The check reads shape and dtype metadata only, with no evaluation and no copy.

### 2.4 Code Quality

- Six new tests, one per family, each failing first (SIGABRT with the check disabled) and passing with it.
- The `Used by` list on the helper names the new callers; `scripts/ci/check_used_by.py` passes.

---

## 3. Technical Decisions

### 3.1 Call the shared helper at each site instead of adding a checked stacking wrapper

| Option | Pros | Cons |
|--------|------|------|
| A: new `checked_stack_expert_leaf` wrapper in `switch_layers.rs` | One call per site | Only fits the three sites that use `stack_expert_leaf`; MiMo-V2 and Mellum use `stack_owned`, OLMoE uses `stack` |
| **Chosen: B: call `refuse_mismatched_expert_parts` directly** | Same helper and message for every family, as the issue specifies | A few lines of `(expert, &array)` collection at each site |

### 3.2 Exempt `global_scale`

`stack_expert_leaf` flattens and concatenates the ModelOpt NVFP4 sidecar rather than stacking it, matching #3304. DeepSeek-V3, Mistral4 and PhiMoE skip the check for that leaf; the other three never stack it.

### 3.3 PhiMoE test location

The issue asked for the PhiMoE test in `phimoe_tests.rs`. `PhiMoeModel::sanitize_weights` is private and `phimoe_tests.rs` is a sibling module, so the test sits in `phimoe.rs`'s inline `sanitize_tests` module beside the existing stacking test instead of widening the function's visibility.

---

## 4. Implementation Details

Each loop now collects the parts with their expert indices, checks them, and only then builds the pointer list for the FFI. DeepSeek-V3, as an example:

```rust
let expert_key = |e: usize| format!("{}.mlp.experts.{}.{}.{}", prefix, e, m, k);
let parts: Vec<(usize, &MlxArray)> = (0..n_routed)
    .filter_map(|e| weights.get(&expert_key(e)).map(|w| (e, w.as_ref().expect("..."))))
    .collect();
if !parts.is_empty() {
    if k != "global_scale" {
        refuse_mismatched_expert_parts(&parts, expert_key)?;
    }
    let expert_ptrs: Vec<*const MlxArray> = parts.iter().map(|(_, w)| *w as *const _).collect();
    let (leaf, stacked) = stack_expert_leaf(k, &expert_ptrs);
    ...
}
```

The error names expert 1's full key, its shape `[4, 6]` and dtype, and expert 0's key, shape `[4, 8]` and dtype, in the format #3304 introduced.

On error nothing is inserted for the refused leaf. OLMoE and Mellum have already removed that leaf's expert keys from the map, but every caller drops the map on `Err`.

---

## 5. Learning Points

### 5.1 C++ exceptions across cxx

A C++ function bound through `cxx` without a `Result` return cannot report an exception to Rust; an uncaught exception terminates the process. Any FFI call with an input precondition (here, equal shapes for `stack`) needs the precondition checked on the Rust side first, or the binding has to be made fallible.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 12 source files |
| Tests added | 6 |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `05cf5444` | fix | refuse mismatched experts in family-local MoE stacking |

### Related PRs/Issues

- #3304 / #3297: shared expert shape checks this PR reuses
- #3331 / #3319: DeepSeek-V3 unquantized MoE loading, which this builds on

---

## 8. Follow-up Actions

### Future Improvements

- The `global_scale` sidecar is still unchecked. If two experts' sidecars ever differ in rank, the concatenate inside `stack_expert_leaf` could still abort. This matches #3304's scope.
- Mellum still returns early without error when an expert is missing (explicitly out of scope in #3320).

---

## Appendix

### A. Test Results

`cargo test --profile test-fast --features metal,accelerate --lib <selector>`, under the GPU lock:

| Selector | Result |
|----------|--------|
| `mismatched_expert` | 9 passed (6 new) |
| `models::deepseek_v3` | 73 passed |
| `models::mistral4` | 4 passed |
| `models::phimoe` | 15 passed |
| `models::olmoe` | 2 passed, 2 ignored |
| `models::mimo_v2` | 20 passed |
| `models::mellum` | 7 passed |
| `models::switch_layers` | 66 passed, 1 ignored |
| `coverage_tests` | 18 passed |

Failing first: with the helper short-circuited, each of the six new tests aborted with `[stack] All arrays must have the same shape` (SIGABRT).

### B. Checkpoint Regressions

`mlxcel generate -p "The capital of France is" -n 24 --temp 0` (release build):

| Checkpoint | Family | Expert layout | Output |
|------------|--------|---------------|--------|
| `gigachat3-10b-a1.8b-bf16` | DeepSeek-V3 | per-expert (stacked by this code) | "Paris." |
| `olmoe-1b-7b-0125-4bit` | OLMoE | per-expert (stacked by this code) | "Paris." |
| `gigachat3-10b-a1.8b-mlx-8bit` | DeepSeek-V3 | quantized | "Paris." |
| `moonlight-16b-a3b-instruct-4bit` | DeepSeek-V3 | quantized | "Paris." |

No local Mistral4, PhiMoE or Mellum checkpoint exists. `MiMo-V2-Flash-4bit` (162 GB) was skipped under the 100 GB limit.
