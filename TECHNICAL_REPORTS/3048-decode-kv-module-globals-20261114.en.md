# Technical Report: PR #3048 - update the decode KV cache in place through module globals

**Date**: 2026-11-14
**Status**: Completed
**Languages**: Rust, C
**Risk Level**: Medium

---

## Executive Summary

The single-sequence OpenXLA decode step used the KV cache as a functional input and output, which forced IREE to copy both 16 MiB cache tensors on every step. Moving the cache into two module-level mutable globals lets the step update it in place: the copies disappear, the dispatch count and the numerics are unchanged, and a new `kv_import` entry seeds the globals from the prefill result. Measured on `gfx1151`, the decode rate rises from 179.78 to 196.06 tok/s (+9.05%, six of six paired rounds) with identical greedy output.

---

## 1. Problem Statement

### 1.1 Background

The XLA backend keeps its per-chip benchmark protocol identical to the MLX backend so the two can be compared directly. After the host-side work in #3028 (persistent mapped scalars, device-to-device token readback) the remaining gap to MLX on `gfx1151` was about 160 against 239 tok/s, a step of 5.9 ms against MLX's roughly 4.2 ms. A kernel trace split that step into 4.32 ms of GPU kernel time and 1.51 ms of idle, and the idle half had a measured composition: 0.79 ms of per-dispatch launch gaps across 309 to 312 dispatches, 0.35 ms of step-start stall next to a 163 µs `hipHostFree`, 0.23 ms of KV copies, and 0.14 ms of other.

### 1.2 Existing Issues

- **Issue 1**: The decode module returned the two cache tensors as results, so every step emitted two `stream.cmd.copy` ops of 16,777,216 bytes each, 32 MiB of device traffic that the 4-bit 1B decode does not otherwise move.
- **Issue 2**: The shim had to adopt a fresh cache buffer-view pair after every step and release the previous pair, which is bookkeeping that only exists to serve the copy.
- **Issue 3**: The graph carried 309 dispatches, about 2.5 µs of launch latency each, for a chain of small elementwise and reduction work.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| The in-place update changes numerics or token output | High | Low (verified bit-identical) |
| Module state leaks across requests that reuse a session | High | Medium (mitigated by the import flag) |
| A graph that does not use globals breaks on the new ABI | High | Low (capability probe keeps the old path) |
| The aggressive-fusion flag changes prefill behaviour | Medium | None (decode compile only) |

---

## 2. Technical Review

### 2.1 Security

No new inputs, no host-visible surface, no changes to model loading or request parsing. The module globals are private to the module and unreachable from the host.

**Checklist:**
- [x] Input validation (unchanged; the graph's own index arithmetic is unchanged)
- [x] Authentication/Authorization (not applicable)
- [x] Data encryption (not applicable)
- [x] Logging (one error path added, prints no model data)

**Issues Found:** none.

### 2.2 Performance

| Area | Before | After | Improvement |
|------|--------|-------|-------------|
| Decode rate (n=128, cap 512) | 179.78 tok/s | 196.06 tok/s | +9.05% |
| Decode step | 5.56 ms | 5.10 ms | -0.46 ms |
| Decode flow dispatches | 309 | 277 | -32 |
| KV cache copies per step | 2 x 16 MiB | 0 | -32 MiB |

The copy removal and the fusion each contribute; this PR does not separate them into two A/Bs, so the 9.05% is the combined figure.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none at the Rust API level. The compiled-artifact ABI changes, which is why the shim probes for `decode_step.kv_import` instead of assuming.
- **New Dependencies**: none.
- **Compatibility**: the ragged batch engine, prefill, embedded-input prefill, deepstack prefill and Gemma3n's fixed-schedule CUDA decode all keep the full-cache argument ABI.

### 2.4 Code Quality

- **Test Coverage**: a new emitter unit test and a new ignored device gate (fresh session, second session, re-prefill).
- **Code Complexity**: one helper pair in the emitter (`global_load`, `global_store`) and one shim helper; the decode call path gains two branches on `c->kv_globals`.
- **Technical Debt**: a pre-existing clippy error in `src/server/batch/xla_worker_tests.rs` was fixed because it only compiles with `xla-iree`, so the default-feature clippy job never sees it.

---

## 3. Technical Decisions

### 3.1 Mutable module globals instead of a row-return ABI

**Context:** the cache must stop travelling through the ABI, but the shim still needs the prefill result to reach the decode graph.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Return only the written row, scatter in the shim | Small transfer; no module state | About 3 dispatches per layer (roughly 120 µs) against roughly 130 µs of copy, a wash, plus a rank-5 ABI and a scatter kernel |
| `--iree-stream-resource-alias-mutable-bindings` | Zero code change | IR unchanged, measured |
| **Chosen: `util.global private mutable` pair** | Zero copies, zero extra dispatches, same ops | Module state; needs a seeding entry point |

**Rationale:** IREE lowers a global read-update-store to a read-write binding into one persistent `!stream.resource<variable>` buffer. The update is in place, so there is nothing to copy and nothing to add.

**Trade-offs:** the graph is no longer a pure function of its arguments, so a second consumer of the same session must not interleave a different sequence. The single-sequence decode path already had one cache pair per context in the shim, so this restates an existing constraint rather than adding one.

### 3.2 Seeding through a probed entry point

**Context:** a private global is not host-visible, so the prefill result needs an entry point.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Version the bundle or the cache key | Explicit | Every existing compiled cache is invalidated |
| **Chosen: `@kv_import` plus a probe** | Old artifacts keep working; the shim decides at runtime | One probe per session |

**Rationale:** the presence of `decode_step.kv_import` is exactly the capability the shim needs, and it is cheap to test after the bundle is appended. A graph without it keeps the argument ABI, which is the code path that already existed.

**Trade-offs:** two live ABIs in one function. The cost is the two `if (!c->kv_globals)` branches in the decode call, and the risk that a future change edits only one side; the device gates cover the global side and the ragged/prefill assertions cover the other.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
[Before]
prefill  ->  kcache, vcache (results)
decode   ->  (kcache, vcache) -> (token, kcache', vcache')   2 copies per step

[After]
prefill  ->  kcache, vcache (results)
kv_import ->  globals @kcache, @vcache                        once per prefill
decode   ->  () -> token                                     updates globals in place
```

### 4.2 Key Code Changes

**File: `src/lib/mlxcel-xla/src/emitter/model.rs`**
```rust
// Before
let (decls, a) = build_arg_schema(&mut b, c);
...
"{alias}module @decode_step {{\n{native}  func.func public @main({sig}) -> ({out_ty}, {cache_ty}, {cache_ty}) {{ ... }}"
```

```rust
// After
let (mut decls, mut a) = build_arg_schema(&mut b, c);
decls.truncate(decls.len() - 2);
a.kcache = b.global_load("kcache", cache_ty.clone());
a.vcache = b.global_load("vcache", cache_ty.clone());
...
"util.global private mutable @kcache : {cache_ty_s}" ...
"func.func public @kv_import(%k: {cache_ty_s}, %v: {cache_ty_s}) { util.global.store %k, @kcache ... }"
```

**Reason for change:** the cache leaves the argument list and the result list, and the step reads and writes the module global instead.

**File: `src/lib/mlxcel-xla/csrc/xla_iree.c`**
```c
// After: capability probe after the bundle loads
iree_status_t s = iree_runtime_call_initialize_by_name(
    c->session, iree_make_cstring_view("decode_step.kv_import"), &probe);
if (iree_status_is_ok(s)) { c->kv_globals = 1; iree_runtime_call_deinitialize(&probe); }
```

**Reason for change:** the shim decides per session whether the graph keeps its cache in globals, and the one-time import runs at the prefill boundary only.

### 4.3 Data Model Changes

Not applicable (no persistent state).

---

## 5. Learning Points

### 5.1 Functional cache semantics are a lowering decision, not a graph requirement

**Concept:** a graph that takes a tensor and returns an updated copy of it is a pure function, but on a streaming runtime that purity is lowered into a real copy.

**Application in this PR:** the graph already wrote exactly one row; the copy existed because of the signature, not the computation. Changing the state model removed 32 MiB of traffic per step without touching a single op.

**Common Use Cases:**
- Any decode or KV-cached graph on a stream-oriented runtime
- Runtimes with explicit buffer lifetimes (IREE/HAL, CUDA graphs) where aliasing means a copy unless the compiler proves otherwise

**Example Code:**
```mlir
// The copy-free form
%k = util.global.load @kcache : tensor<16x512x8x64xf32>
%k2 = ...update at the current position...
util.global.store %k2, @kcache : tensor<16x512x8x64xf32>
```

---

## 6. Further Learning

### Key Terms
| Keyword | Description | Relevance |
|---------|-------------|-----------|
| `util.global private mutable` | MLIR module-level mutable state, not host-visible | The mechanism that removes the copy |
| `!stream.resource<variable>` | IREE's persistent buffer class for module globals | Why the update is in place |
| `--iree-dispatch-creation-enable-aggressive-fusion` | Compiler pass that fuses small elementwise dispatches | 309 to 277 dispatches |

### Related Technologies/Frameworks
- **IREE**: compiler and runtime the XLA backend targets. Official docs: https://iree.dev/

### Related PRs/Issues
- Issue #3043: the performance issue this PR closes.
- PR #3042: the phase-accurate `--profile` accounting that made the decode figures comparable.
- PR #3031: the host-side scalar work that this PR builds on.

---

## 7. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 8 |
| Lines added | +275 |
| Lines deleted | -27 |
| Tests added | 2 |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Performance | 4 | Emitter globals, shim import path, decode-only fusion flag, changelog entry |
| Code Quality | 2 | Emitter unit test, device gate |
| Documentation | 1 | CHANGELOG `Changed` entry |
| Code Quality | 1 | Pre-existing clippy fix in `xla_worker_tests.rs` |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `cd0c026` | perf | update the decode KV cache in place through module globals (#3043) |

---

## 8. Follow-up Actions

### Required
- [ ] None.

### Monitoring Required
- Decode rate on the ROCm benchmark page after the next full sweep; the `gfx1151` rows should move by roughly the measured 9%.

### Future Improvements
- The 192-byte pinned staging allocation per invocation remains (about 0.35 ms of GPU idle). `--iree-stream-resource-memory-model=unified` produces byte-identical IR and `--iree-stream-resource-alias-mutable-bindings` leaves the IR unchanged, so removing it needs a graph whose readback indices are static, or a device-side token scan.
- HIP graph capture of the decode step remains the candidate for the remaining launch-gap time.

---

## Appendix

### A. Test Results

- `cargo test -p mlxcel --features xla-iree,test-utils --test xla_greedy_timing -- --ignored`: 3 passed.
- Emitter unit test `single_decode_threads_kv_through_module_globals`: passed.
- `cargo fmt --all -- --check`: clean.
- `cargo clippy -p mlxcel --features xla-iree --lib --tests -- -D warnings`: clean.
- `cargo test -p mlxcel-core --profile test-fast --lib --no-run`: linked.

### B. Performance Benchmarks

Paired interleaved A/B, `gfx1151`, llama-3.2-1b-instruct-4bit, native quant, capacity 512, n=128, six rounds:

| Round | before (tok/s) | after (tok/s) |
|-------|----------------|---------------|
| 1 | 184.97 | 193.49 |
| 2 | 173.26 | 198.17 |
| 3 | 179.49 | 193.94 |
| 4 | 182.33 | 198.23 |
| 5 | 180.07 | 198.61 |
| 6 | 177.21 | 193.41 |
| median | 179.78 | 196.06 |

### C. References
- Issue #3043
- IREE stream dialect documentation: https://iree.dev/developers/design-docs/streaming/
