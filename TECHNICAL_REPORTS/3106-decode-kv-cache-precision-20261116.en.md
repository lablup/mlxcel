# Technical Report: PR #3106 - keep the decode KV cache in the contraction precision

**Date**: 2026-11-16
**Status**: Completed
**Languages**: Rust (MLIR emitter), Bash (measurement)
**Risk Level**: Medium

---

## Executive Summary

The single-sequence IREE decode step stored its KV cache in f32 module globals while every consumer contraction ran in f16, so each layer converted the whole cache slab on every step: 32 layout-transpose dispatches per step and 10.5 MB of cache traffic where 5.3 MB suffice. The globals now carry the contraction precision (f16 on hip and cuda, bf16 under `MLXCEL_XLA_PRECISION=bf16`, f32 on the CPU), the newly written row is narrowed once at the update site, and `decode_step.kv_import` keeps its f32 signature with one conversion per request. The saving scales with the KV slab: at capacity 512 it is inside the paired protocol's noise (four runs, 50 rounds, 27 won by the new arm), and at capacity 2048 it measures +2.85% (9 of 12 rounds) with identical greedy output.

---

## 1. Problem Statement

### 1.1 Background

Issue #2886 asked for what remains of the decode gap on the OpenXLA/IREE backend. After #3048 moved the KV cache into module globals and PR #3065 removed the per-step host staging readback, the native-quant decode step on `gfx1151` measured about 5.0 ms against MLX ROCm's about 4.2 ms. A kernel trace of the native path (281 dispatches, 3.976 ms of kernel time, 1.077 ms of launch gaps) showed the weight-streaming kernels already at the practical roofline: gate/up and the LM head ran at about 230 to 240 GB/s, and the step streamed 695 MB of packed weights at 199 GB/s aggregate.

### 1.2 Existing Issues

- **Issue 1**: the decode KV globals were `f32` while the score and context contractions ran in `f16`, so IREE inserted a whole-slab `convert` before every contraction, fused into 32 layout-transpose dispatches per step.
- **Issue 2**: that conversion moved 10.5 MB of cache per step where the narrowed form needs 5.3 MB.
- **Issue 3**: the conversion cost scales with the KV slab, so it grows with the context capacity the decode actually runs at.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A graph writes an f32 row into the narrowed global (StableHLO element-type mismatch) | High (graph does not compile) | Occurred: the MLA single-decode path, caught in review and fixed |
| Numerics change in a way greedy output exposes | High | Low (measured: `text EQUAL` in every paired run, batch `REFERENCE-EXACT`) |
| The printed precision no longer matches what the runtime compiles | Medium | Low (`resolve_precision` keys the vmfb cache on the emitted MLIR) |

---

## 2. Technical Review

### 2.1 Security

No new input, network or file surface. The change narrows a device-resident buffer and keeps the C shim ABI unchanged; `xla_validate_kv_pair` still validates the f32 prefill caches it receives.

**Checklist:**
- [x] Input validation (unchanged; prefill caches still validated as f32 rank-4)
- [x] Authentication/Authorization (not applicable)
- [x] Data encryption (not applicable)
- [x] Logging (no model data)

**Issues Found:** the MLA dtype mismatch (fixed in `044eded6`); no CRITICAL, HIGH or MEDIUM findings in the security review.

### 2.2 Performance

| Area | Before | After |
|------|--------|-------|
| Traced decode step | 5.054 ms | 4.895 ms |
| Dispatches per step | 281 | 263 |
| Launch gaps per step | 1.077 ms | 0.982 ms |
| Cache traffic per step | 10.5 MB | 5.3 MB |
| Paired A/B, capacity 512 | 197.01 / 199.43 / 200.07 / 201.01 tok/s | 200.14 / 200.20 / 201.87 / 199.78 tok/s |
| Paired A/B, capacity 2048 | 178.02 tok/s | 183.09 tok/s (+2.85%) |

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. The compiled decode graph changes its global element type, but the bundle has no version and the C shim never touches the globals directly; `decode_step.kv_import` keeps its f32 signature.
- **New Dependencies**: none.
- **Compatibility**: prefill, ragged and uniform-B batched caches stay f32 buffer arguments. MLA keeps its own default (f32) through `effective_precision`, and under an explicit f16/bf16 it now narrows its latent and PE rows like the ordinary K/V path.

### 2.4 Code Quality

- **Test Coverage**: one emitter test pins the global element type for f32, f16 and bf16 and asserts the absence of a whole-slab convert; a review test emits the MLA decode at f16 and asserts the row conversion; `single_decode_threads_kv_through_module_globals` is pinned to f32 so the documented precision override does not break it.
- **Code Complexity**: one shared `narrow_cache_row` helper used by both write sites, one crate-visible `Precision::dot_elt`.
- **Technical Debt**: decreased. The duplicated narrowing closure that let the MLA path drift is gone.

---

## 3. Technical Decisions

### 3.1 Store the cache in the contraction precision instead of converting per read

**Context:** the cache was f32 and every contraction narrowed it. Either the reads or the storage had to change.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Keep f32 and convert per read | No ABI or dtype change | Whole-slab conversion per step, scaling with capacity |
| Store f16 and widen on read | Also narrow storage | A whole-slab widen per step, the same cost moved |
| **Chosen: store in the contraction precision, narrow the new row** | Conversion runs once per written row instead of per slab | The globals' element type changes; every writer must narrow |

**Rationale:** the row written each step is 1 of `capacity` rows, so narrowing at the write site is the only form whose cost does not scale with capacity.

**Trade-offs:** precision is now applied once at write time instead of at each read. The security review confirmed this is bit-identical to the previous behavior for every consumer, because `dot_general` demotes f32 operands and leaves already-narrow ones untouched.

### 3.2 Keep `kv_import` on f32 and convert once per request

**Context:** the host imports the KV cache on a context switch or a cache reset.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Narrow `kv_import` to the cache element type | No conversion in the graph | The C shim's f32 buffer ABI breaks for every caller |
| **Chosen: keep f32 import and convert once** | No shim change; one conversion per request | One full-slab conversion per import |

**Rationale:** a request imports once and decodes many steps, so a per-import conversion is not on the per-step path, and the shim keeps one ABI.

### 3.3 One `narrow_cache_row` helper for both write sites

**Context:** the ordinary K/V path and the MLA latent/PE path both write into the same single-decode globals.

**Rationale:** the review found the MLA path had been missed, and the duplicated closure is what let the two drift. A single helper makes a future write site fail visibly rather than produce a graph that will not compile.

---

## 4. Implementation Details

### 4.1 Key Code Changes

**File: `src/lib/mlxcel-xla/src/emitter/model.rs`**
```rust
let cache_elt = b.precision().dot_elt().unwrap_or("f32");
let cache_ty = Ty::new(vec![c.n_layers, c.context_capacity, c.cache_heads(), c.cache_dim()], cache_elt);
// The new row is narrowed once, at the update site.
let k_upd = narrow_cache_row(b, b.reshape(kk, vec![1, 1, nkv, d]));
```
**Reason for change:** the global's element type decides whether IREE converts the slab per read; narrowing the row keeps the conversion off the per-step path.

**File: `src/lib/mlxcel-xla/src/emitter/model.rs`, MLA path (`044eded6`)**
```rust
let latent_update = narrow_cache_row(b, latent_update);
let pe_update = narrow_cache_row(b, pe_update);
```
**Reason for change:** without this, `stablehlo.dynamic_update_slice` mixed an f16 operand with an f32 update and the MLA decode graph could not compile.

### 4.2 Data Model Changes

The decode KV globals change from `tensor<n_layers x capacity x heads x dim x f32>` to the same shape in f16 or bf16. Nothing persists between runs; the vmfb cache is keyed on the emitted MLIR.

---

## 5. Learning Points

### 5.1 Storage precision is a lowering decision

**Concept:** with an element-wise conversion between a buffer and every reader, the conversion cost is proportional to the buffer size times the read count. Moving the precision into the buffer turns that into one conversion per write.

**Application in this PR:** 32 dispatches per step and 5.2 MB of traffic per step disappeared, and the saving scaled with the KV capacity, which is why the effect is invisible at capacity 512 and measurable at 2048.

### 5.2 A paired protocol bounds small effects from above and below

**Concept:** interleaved paired rounds cancel drift but not noise; the resolution is set by the host's round-to-round spread, not by the number of pairs alone.

**Application in this PR:** the first claim for this change was a single traced step time. Fifty paired rounds at capacity 512 could not separate a 1 percent effect from zero, and reporting that honestly is what led to measuring the capacity where the mechanism is large enough to see.

---

## 6. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 5 (the two report files not counted) |
| Lines added | +342 |
| Lines deleted | -12 |
| Tests added | 4 |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Performance | 2 | Cache global precision, row narrowing at the write site |
| Correctness | 2 | MLA row narrowing, shared `narrow_cache_row` |
| Tests | 4 | Global-type pin (f32/f16/bf16, import signature), MLA narrowing, native and ragged graph kinds, precision-independent globals test |
| Documentation | 3 | README breakdown, `docs/environment-variables.md`, PR measurements |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `6ac7fd37` | perf | keep the decode KV cache in the contraction precision |
| `044eded6` | fix | narrow the MLA single-decode cache row too |
| `0e4374b7` | test | pin the KV globals test to f32 |
| `efaf4335` | test | cover the narrowed decode cache on every graph kind |

---

## 7. Follow-up Actions

### Required
- [ ] None.

### Monitoring Required
- Long-context decode (capacity 2048 and above): the cache traffic per step should stay flat rather than scale with capacity.

### Future Improvements
- The down projection runs at 151 GB/s against gate/up's 230; vectorizing it changes the f32 reduction grouping and needs its own token-identity gate.
- Launch gaps over about 263 small dispatches remain; only further graph fusion moves them.
- #2873 owns small-M vecmats and the serial argmax, which are absent from the native path but still apply to the default f16 graph.

---

## Appendix

### A. Test Results

- `cargo test -p mlxcel-xla --lib emitter::tests::`: 67 passed, 5 ignored.
- `validation::tests::registered_fixtures_are_byte_exact`: passed (goldens unchanged).
- `cargo clippy -p mlxcel-xla --all-targets -- -D warnings` and with `--features iree`: clean.
- Batch gate on hip: `xla_batch_bench --batch 8 --requests 16 --maxcap 48` printed `RESULT: REFERENCE-EXACT PASS`, 16 of 16 requests matching their single-sequence reference.
- `iree-compile` lowering of the llama f16, MLA f16 and MLA bf16 decode graphs: all succeeded.

### B. Performance Benchmarks

Paired interleaved A/B through `scripts/xla/ab_decode_pair.sh`, `gfx1151`, llama-3.2-1b-instruct-4bit, native quant, n=128, `text EQUAL` in every run.

| Capacity | Rounds | old median (tok/s) | new median (tok/s) | delta | rounds won by new |
|----------|--------|--------------------|--------------------|-------|-------------------|
| 512 | 6 | 197.01 | 200.14 | +1.59% | 3/6 |
| 512 | 8 | 199.43 | 200.20 | +0.39% | 4/8 |
| 512 | 12 | 200.07 | 201.87 | +0.90% | 8/12 |
| 512 | 24 | 201.01 | 199.78 | -0.61% | 12/24 |
| 2048 | 12 | 178.02 | 183.09 | +2.85% | 9/12 |

### C. References
- Issue #2886, PR #3048 (module-global KV cache), PR #3065 (decode scalar ABI), issue #2873 (remaining decode kernels).
