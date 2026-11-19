# Technical Report: PR #3065 - pass the decode scalars as ABI i32 arguments

**Date**: 2026-11-16
**Status**: Completed
**Languages**: Rust, C
**Risk Level**: Medium

---

## Executive Summary

The single-sequence OpenXLA decode step took token, position and cache_len as rank-0 `tensor<i32>` buffer arguments. Because the graph uses those values as host-side slice offsets and mask bounds, IREE materialized a host-visible 192-byte `!stream.resource<staging>` and three `stream.resource.load`s per invocation, which on `gfx1151` costs a pinned `hipHostMalloc`/`hipHostFree` pair per decode step. Passing the three values as ABI-level `i32` arguments removes the staging resource from the compiled graph and lifts decode from 195.56 to 200.25 tok/s (+2.40%, six of six paired rounds, identical greedy output).

---

## 1. Problem Statement

### 1.1 Background

After #3048 moved the decode KV cache into two module globals, the decode step on `gfx1151` measured 5.10 ms against MLX ROCm's roughly 4.2 ms. A kernel trace of the earlier step split it into 4.32 ms of GPU kernel time and 1.51 ms of idle, and the idle half had a measured composition: 0.79 ms of launch gaps across 309 to 312 dispatches, 0.35 ms of step-start stall next to a 163 us `hipHostFree`, 0.23 ms of KV copies, and 0.14 ms of other. #3048 removed the copies and cut the dispatch count to 277; the staging churn was documented as a remainder.

### 1.2 Existing Issues

- **Issue 1**: every decode invocation allocated and freed a 192-byte pinned staging buffer, because the graph needed the three scalar values on the host to compute slice offsets.
- **Issue 2**: the same values were uploaded to the device first as rank-0 tensor arguments and then read back, so the round trip existed only to satisfy the argument type.
- **Issue 3**: the allocation sat at the step boundary, where its free showed up as GPU idle rather than as kernel time.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Argument count or type drift between the emitter and the C shim | High | Low (single ordering, both sides reviewed) |
| An older compiled bundle breaks on the new ABI | High | None (capability marker keeps the buffer path) |
| M-RoPE's 3-axis position is pushed as a scalar | High | None (marker is not emitted on that path) |
| Numerics change | High | Low (op sequence after the prologue unchanged; greedy output byte-identical) |

---

## 2. Technical Review

### 2.1 Security

The change is confined to how three integers reach the graph. No new host input, no new file or network surface. The scalar values keep the bounds checks they had (`position` and `cache_len` against `[0, context_capacity)`), and the removed buffer-view assertion never validated values, only shape and type.

**Checklist:**
- [x] Input validation (position and cache_len bounds unchanged)
- [x] Authentication/Authorization (not applicable)
- [x] Data encryption (not applicable)
- [x] Logging (one new error path for an impossible ABI mismatch, prints no model data)

**Issues Found:** none.

### 2.2 Performance

| Area | Before | After | Improvement |
|------|--------|-------|-------------|
| Decode rate (n=128, capacity 512, native quant) | 195.56 tok/s | 200.25 tok/s | +2.40% |
| Decode step | 5.09 ms | 5.01 ms | about -0.09 ms |
| `!stream.resource<staging>` in the compiled decode graph | 1 (12 IR mentions) | 0 | removed |
| `stream.resource.load` per invocation | 3 | 0 | removed |

The step gain is smaller than the 0.35 ms of step-start idle that #3048 associated with the staging pair, so part of that idle was not the pair itself, most likely the surrounding host work at the step boundary. The round record is clean: all six paired rounds favor the new arm.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none at the Rust or C API level. The compiled-artifact ABI changes for the single-sequence decode only, which is why the shim probes for `decode_step.scalar_inputs` instead of assuming it.
- **New Dependencies**: none.
- **Compatibility**: prefill, embeddings prefill, ragged and lockstep batch decode, Gemma3n decode and the `phi4` `adapter_mode` argument keep their full argument ABI. M-RoPE keeps its `tensor<3xi32>` position and therefore the buffer-argument ABI, with its residual staging.

### 2.4 Code Quality

- **Test Coverage**: one emitter unit test (signature, prologue rebuilds, marker, and the unchanged signatures of every other graph), and a device gate that runs a scalar-ABI artifact and a legacy buffer-view artifact in one process and compares their token streams.
- **Code Complexity**: one flag on `ArgDecl`, one signature-rendering branch, one `Builder` helper, and one probe plus one push branch in the shim.
- **Technical Debt**: the scalar path still calls `xla_step_scalars_ensure`, which allocates the three input maps it no longer uses, to obtain the readback slot. One allocation per context, not per step.

---

## 3. Technical Decisions

### 3.1 ABI-level scalars instead of a graph with static indices

**Context:** the staging resource exists because the graph reads its own arguments on the host. There were two ways to stop that.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Make the readback indices static | No ABI change | A decode graph's position and mask bounds are dynamic by construction; not general |
| Move the token scan to the device | Removes the readback entirely | An extra dispatch per step and a new sampling path |
| `--iree-stream-resource-memory-model=unified`, `--iree-stream-resource-alias-mutable-bindings` | Zero code change | Measured in #3048: identical or unchanged IR |
| **Chosen: pass the three values as ABI `i32`** | Staging gone, no extra dispatch, no numerics change | Artifact ABI changes; the shim needs two paths |

**Rationale:** a VM scalar argument is host data at the boundary, so there is nothing to materialize and nothing to read back. IREE lowers `util.func public @main(..., i32, i32, i32)` with no staging resource, verified on the pinned compiler and then on the real graph.

**Trade-offs:** two live ABIs in one decode function. The shim decides per session from the marker, and the buffer path stays as it was.

### 3.2 A probed marker instead of a bundle version bump

**Context:** a shim compiled against the old signature must not push scalars into a graph that expects buffer views.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Version the bundle or the cache key | Explicit | Invalidates every compiled cache |
| **Chosen: a no-op `decode_step.scalar_inputs` entry, probed** | Old artifacts keep working; the shim decides at runtime | One probe per session |

**Rationale:** the presence of the entry is exactly the capability the shim needs, and probing it after the bundle loads costs one call per session. The pattern already existed for `decode_step.kv_import`.

**Trade-offs:** the marker must be kept in sync with the signature it stands for. The emitter unit test pins both, and the device gate exercises both sides.

### 3.3 M-RoPE keeps the buffer ABI

**Context:** the M-RoPE decode path carries a `tensor<3xi32>` position rather than a rank-0 scalar.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Three separate scalar arguments for the axes | Removes M-RoPE staging too | A second scalar shape and a two-way capability signal |
| **Chosen: leave M-RoPE on the buffer ABI** | One on/off marker; smaller blast radius | Residual staging on that path |

**Rationale:** the measured win is on the plain-RoPE decode, which is every current single-sequence path except M-RoPE models. A two-valued marker for an unmeasured path would trade clarity for nothing.

**Trade-offs:** M-RoPE keeps its per-step staging. The shim rejects a marked graph meeting an M-RoPE call rather than pushing the wrong type, so the mismatch cannot silently corrupt a decode.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
[Before]
host: token, pos, cache_len -> device rank-0 tensors -> graph
graph: stream.resource.load x3 (192 B staging) -> slice offsets and mask bounds

[After]
host: token, pos, cache_len -> VM i32 arguments
graph: tensor.from_elements x3 -> slice offsets and mask bounds  (no staging)
```

### 4.2 Key Code Changes

**File: `src/lib/mlxcel-xla/src/emitter/model.rs`**
```rust
// Before: every decode argument is a buffer view
let token = take_arg(&mut decls, &mut idx, Ty::scalar("i32"), "token".into());

// After: the 1-D decode marks the three scalars, and the prologue rebuilds the tensor
let scalar_abi = !c.uses_mrope() && decode_scalar_abi_enabled();
let token = if scalar_abi {
    take_abi_scalar_arg(b, &mut decls, &mut idx, "token".into())
} else { ... };
```

**Reason for change:** the signature is the only thing that decides whether IREE stages the value, so the argument type itself has to change.

**File: `src/lib/mlxcel-xla/csrc/xla_iree.c`**
```c
// Probe after the bundle loads, exactly like decode_step.kv_import
iree_status_t s = iree_runtime_call_initialize_by_name(
    c->session, iree_make_cstring_view("decode_step.scalar_inputs"), &probe);
if (iree_status_is_ok(s)) { c->scalar_inputs = 1; iree_runtime_call_deinitialize(&probe); }
else { iree_status_ignore(s); }

// At the call site, the scalar path pushes values instead of views
XLA_CHECK(iree_vm_list_push_value(iree_runtime_call_inputs(&call), &token_value));
```

**Reason for change:** the shim cannot know the artifact's ABI at compile time, and the buffer-view path has to keep working for artifacts compiled before this change.

### 4.3 Data Model Changes

Not applicable (no persistent state).

---

## 5. Learning Points

### 5.1 An argument's type is a lowering decision

**Concept:** `tensor<i32>` and `i32` carry the same value, but on a stream-oriented runtime only the tensor form has to exist as memory. When the graph needs that value on the host, the compiler must bring it back, and the round trip is a real allocation.

**Application in this PR:** the staging resource was not a defect in the graph's arithmetic; it was the cost of the argument type. Changing the type removed it with no change to any op that followed the prologue.

**Common Use Cases:**
- Any decode or sampling graph on IREE, CUDA graphs or another runtime with explicit buffer lifetimes
- Graph parameters that only feed indices, shapes or bounds

**Example Code:**
```mlir
// The staging-free prologue
func.func public @main(%weights: tensor<...>, %arg146: i32, %arg147: i32, %arg148: i32) {
  %token = tensor.from_elements %arg146 : tensor<i32>
  %slice = stablehlo.dynamic_slice %embed, %token, ...  // host value, no readback
```

### 5.2 A capability marker keeps one compiled-bundle format

**Concept:** when a graph's ABI changes, either every cached artifact is invalidated or the caller can ask what ABI it is holding.

**Application in this PR:** a no-op exported entry is a cheap, name-based answer, and it turns an incompatible-artifact crash into a runtime branch. The same pattern already covered the KV-globals change in #3048.

**Common Use Cases:**
- IREE module bundles, ONNX graph caches, any ahead-of-time compiled artifact consumed by a newer host

---

## 6. Further Learning

### Key Terms
| Keyword | Description | Relevance |
|---------|-------------|-----------|
| `!stream.resource<staging>` | IREE's host-visible buffer class for values the host must read | The allocation this PR removes |
| `stream.resource.load` | The op that reads such a buffer | Went from three per step to zero |
| `tensor.from_elements` | Builds a tensor from scalar values | Rebuilds the rank-0 operands at the prologue |
| `iree_vm_list_push_value` | Pushes a scalar VM value into a call | The shim's new input path |

### Related Technologies/Frameworks
- **IREE**: compiler and runtime the XLA backend targets. Official docs: https://iree.dev/

### Related PRs/Issues
- Issue #3060: the performance issue this PR closes.
- PR #3048: the module-global KV cache that this PR follows and shares its probe pattern with.
- PR #3031: the persistent mapped decode scalars that removed the earlier host churn.
- PR #3078: regenerated the decode goldens for the #3048 ABI, which is why this PR's goldens conflicted and were rebased.

---

## 7. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 22 |
| Lines added | +5357 |
| Lines deleted | -4915 |
| Tests added | 2 |

The line counts are dominated by the regenerated decode goldens.

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Performance | 3 | Emitter ABI scalars, shim probe and push path, M-RoPE guard |
| Code Quality | 2 | Emitter unit test, device gate for both ABIs |
| Documentation | 2 | `src/lib/mlxcel-xla/README.md` subsection, this report |
| Tests | 15 | Regenerated `assets/*/decode.mlir` goldens |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `3f9279b3` | perf | pass the decode scalars as ABI i32 arguments |
| `9246d11a` | perf | (rebased onto main) pass the decode scalars as ABI i32 arguments |
| `385ad4a3` | docs | document the decode scalar ABI and marker |

---

## 8. Follow-up Actions

### Required
- [ ] None.

### Monitoring Required
- The next full `gfx1151` sweep: decode rows should move by roughly the measured 2.4 percent over `6e2e6810`.

### Future Improvements
- Allocate only the `decode_result` slot in the scalar path instead of calling `xla_step_scalars_ensure` for its side effect.
- If an M-RoPE model ever becomes performance-relevant on this backend, extend the marker to carry the position shape so that path loses its staging too.
- HIP graph capture of the decode step remains the candidate for the launch-gap time.

---

## Appendix

### A. Test Results

- `cargo test -p mlxcel-xla --lib`: 417 passed, 0 failed, 19 ignored.
- `tests/xla_decode_scalar_reuse.rs -- --ignored`: 4 passed.
- `tests/xla_greedy_timing.rs -- --ignored`: 3 passed.
- `cargo fmt --all -- --check`, `cargo clippy -p mlxcel --features xla-iree --lib --tests -- -D warnings`, `cargo clippy -p mlxcel-xla --all-targets -- -D warnings`: clean.
- `gcc -fsyntax-only -std=gnu11 -DMLXCEL_XLA_TEST_SUPPORT=1 csrc/xla_iree.c`: clean.

### B. Performance Benchmarks

Paired interleaved A/B, `gfx1151`, llama-3.2-1b-instruct-4bit, native quant, capacity 512, n=128, six rounds:

| Round | before (tok/s) | after (tok/s) |
|-------|----------------|---------------|
| 1 | 198.55 | 200.64 |
| 2 | 195.51 | 199.87 |
| 3 | 196.46 | 197.89 |
| 4 | 195.61 | 198.39 |
| 5 | 190.17 | 205.10 |
| 6 | 192.84 | 212.58 |
| median | 195.56 | 200.25 |

### C. References
- Issue #3060
- IREE stream dialect documentation: https://iree.dev/developers/design-docs/streaming/
