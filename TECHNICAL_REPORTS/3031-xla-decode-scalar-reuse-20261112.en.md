# Technical Report: PR #3031 - perf(xla): reuse decode step scalars and token readback

**Date**: 2026-11-12
**Status**: Completed
**Languages**: C, Rust
**Risk Level**: Low

---

## Executive Summary

The single-sequence OpenXLA decode step allocated four host-visible buffers per step (three scalar inputs and one token readback staging buffer) and paid a pinned `hipHostMalloc`/`hipHostFree` pair for each, which on gfx1151 costs about 129 us per pair and showed up as GPU-idle gaps at every step boundary. The buffers are now allocated and persistently mapped once per context, written in place, and the 4-byte token readback travels device-to-device into a preallocated slot. Decode throughput rises from a combined median 140.7 to 150.1 tok/s (7.11 to 6.66 ms per step) with byte-identical greedy output, and a forced-fallback device test proves the old path still works on a device that cannot serve the heap.

---

## 1. Problem Statement

### 1.1 Background

The OpenXLA/IREE backend decodes at far below the MLX backend on the same ROCm host: 143 tok/s against 239 on llama-3.2-1b-instruct-4bit. A `rocprofv3 --kernel-trace` of the decode loop attributed 4.31 ms of the 6.97 ms step to GPU kernels, leaving about 2.2 ms of GPU idle plus host time. One `rocprofv3 --hip-trace` run localized it: 2.06 pinned `hipHostMalloc`/`hipHostFree` pairs per steady-state step, with the free at a p50 of 147.9 us, and the kernel trace showed 148.3, 21.1 and 147.0 us idle gaps at the step boundary.

### 1.2 Existing Issues

- The three scalar inputs (token id, positions, cache length) were each allocated through `xla_alloc_bv`, which allocates a device buffer and copies bytes in, and IREE backs the host side of that copy with its own staging allocation.
- The 4-byte token readback used `iree_hal_device_transfer_d2h`, which allocates and frees a host staging buffer per call.
- `xla_ctx` had no place to keep step state, so everything was per-call by construction.

A standalone HIP microbenchmark on the host measured the primitives directly: `hipHostMalloc(32 B)` 11.7 us, `hipHostFree` 122.4 us, a pair 128.9 us, while `hipMalloc` + `hipFree` costs 0.9 us. The whole cost was allocator churn on 4 to 16 bytes of payload.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Persistent mapping is read before the device wrote it (missing sync) | Wrong tokens | Low: the d2d transfer is synchronous, the same ordering the old d2h readback relied on |
| A device cannot serve the host-local device-visible heap | Decode fails outright | Medium: mitigated by the state machine and the fallback test |
| Write-combined upload memory is not visible to the device in time | Wrong tokens | Low: the same heap already backs every IREE h2d staging copy |

---

## 2. Technical Review

### 2.1 Security

No input validation, authentication or data-handling surface changes. The new buffers hold four int32 values of model state. The fallback path logs the IREE status code and message to stderr once; it carries no user data.

### 2.2 Performance

| Area | Before | After | Improvement |
|------|--------|-------|-------------|
| Decode step (gfx1151, n=256) | 7.11 ms | 6.66 ms | -6.3% |
| Decode throughput | 140.7 tok/s | 150.1 tok/s | +6.7% |
| Pinned malloc/free pairs per step | 2.06 | 0 | -100% |
| Per-step allocations | 4 | 0 | -100% |

The GPU kernel count and the emitted graphs are unchanged; the saving is entirely host-side. Two interleaved four-round A/Bs (eight runs per arm) were used because run-to-run drift on this host is larger than the effect: the new path was ahead in 7 of 8 paired rounds.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. The C ABI and the public Rust API are unchanged.
- **New Dependencies**: none.
- **Compatibility**: any IREE HAL device that can serve `HOST_LOCAL | DEVICE_VISIBLE` memory takes the fast path; everything else keeps the previous behaviour through `step_scalars_state = -1`.

### 2.4 Code Quality

Three new device tests (reuse counters, forced fallback with an identical token stream, prefill-only context) plus the existing four prepared-prefill and six ragged-hardening gates pass. `cargo clippy` reports no new findings.

---

## 3. Technical Decisions

### 3.1 One persistent buffer set instead of one block with subspan views

**Context:** the issue sketched a single 32-byte allocation with four subspan views.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Single 32-byte block plus subspans | One allocation, one mapping | The readback word and the inputs need different heaps (see below), so one block forces the wrong heap on one of them |
| **Chosen: four separate allocations** | Each slot gets the heap its access pattern needs | Four allocation calls at context setup instead of one, on first decode only |

**Rationale:** the HIP allocator selects a memory heap per allocation from the requested type bits. The three inputs are written by the host and read by the device, which is the write-combined upload heap; the readback word is written by the device and read by the host, where write-combined memory would bypass the CPU cache. Splitting the allocations costs three extra `hipHostMalloc` calls once per context and lets each slot use the right heap.

**Trade-offs:** slightly more setup code and four pointers instead of one base pointer.

### 3.2 Device-to-device readback instead of a cached d2h

**Context:** the readback target must be device-writable and host-readable.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Keep `transfer_d2h` | No new code | Allocates and frees a staging buffer per call, which is the cost being removed |
| **Chosen: `transfer_d2d` into a mapped host-local slot** | No allocation, one 4-byte copy | Depends on a mapped pinned pointer being a valid d2d target, which holds on ROCm's unified address space |

**Rationale:** the transfer functions are synchronous and wait for the copy, so a plain read of the mapped pointer immediately after is the same ordering the existing d2h readback relies on. `HOST_COHERENT` removes the need for an explicit invalidate.

### 3.3 A three-state field instead of a hard requirement

`step_scalars_state` is `0` before the first decode, `1` when reuse is active, and `-1` when allocation or mapping failed. A `-1` context runs the untouched per-step code forever, which keeps a driver or allocator limitation a performance regression rather than a decode failure. `xla_test_force_step_scalars_unsupported` drives that state directly so the fallback is testable, and the test asserts the fallback decodes the same token stream.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
[Before]
decode step:
  xla_alloc_bv(token)    -> device buffer + h2d staging (pinned malloc/free)
  xla_alloc_bv(positions)-> device buffer + h2d staging (pinned malloc/free)
  xla_alloc_bv(cache_len)-> device buffer + h2d staging (pinned malloc/free)
  invoke, pop outputs
  transfer_d2h(token_out) -> staging (pinned malloc/free)

[After]
first decode:
  xla_step_scalars_ensure -> 4 host-local device-visible buffers, mapped once
decode step:
  write token/positions/cache_len into the mappings
  invoke, pop outputs
  transfer_d2d(token_out -> persistent slot), read mapped pointer
```

### 4.2 Key Code Changes

**File: `src/lib/mlxcel-xla/csrc/xla_iree.c`**

```c
static const iree_hal_buffer_params_t kStepScalarParams = {
    .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
            IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
    .access = IREE_HAL_MEMORY_ACCESS_ALL,
    .usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING |
             IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
};

if (xla_step_scalars_ensure(c, position_count) == 1) {
    owns_scalars = false;
    *(int32_t*)c->step_token_map.contents.data = token;
    *(int32_t*)c->step_len_map.contents.data = cache_len;
    memcpy(c->step_pos_map.contents.data, positions,
           (size_t)position_count * sizeof(int32_t));
    tok_bv = c->step_token; pos_bv = c->step_pos; len_bv = c->step_len;
} else {
    /* the previous per-step allocation path, unchanged */
}
```

**Reason for change:** the decode step stops allocating and starts writing into memory it already owns.

The readback moved into a decode-only helper so prefill and ragged decode keep the old readback path:

```c
static iree_status_t xla_read_token_decode(xla_ctx* c,
                                           iree_hal_buffer_view_t* out,
                                           int32_t* out_token) {
  if (c->step_scalars_state != 1 || !c->decode_result) {
    return xla_read_token(c, out, out_token);
  }
  if (iree_hal_buffer_view_element_count(out) != 1 ||
      iree_hal_buffer_view_element_type(out) != IREE_HAL_ELEMENT_TYPE_INT_32) {
    return xla_read_token(c, out, out_token);
  }
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2d(
      c->device, iree_hal_buffer_view_buffer(out), 0,
      iree_hal_buffer_view_buffer(c->decode_result), 0, sizeof(int32_t),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  *out_token = *(const int32_t*)c->decode_result_map.contents.data;
  return iree_ok_status();
}
```

`xla_llama_free` unmaps and releases the four views before the device goes away, and the ownership flag (`owns_scalars`) keeps the error path from releasing views it does not own.

### 4.3 Data Model Changes

None.

---

## 5. Learning Points

### 5.1 Pinned allocation cost on an APU

**Concept:** `hipHostMalloc` creates page-locked memory for DMA or mapped device access. Page-locking and unregistering pages is a kernel operation whose cost scales with system state, not with the requested size.

**Application in this PR:** the 4-byte readback staging buffer cost as much as a multi-megabyte one, 128.9 us per pair, which on a 6.97 ms step is 1.8% per pair and about 3.7% for the two pairs traced.

**Example:** a 30-line HIP microbenchmark that allocates and frees the primitives in isolation separated this from GPU work that the end-to-end numbers alone could not.

### 5.2 IREE memory heaps and persistent mapping

`IREE_HAL_MEMORY_TYPE_HOST_LOCAL` is a composite that already contains `HOST_VISIBLE` and `HOST_COHERENT`. Adding `HOST_CACHED` selects the cached page-locked heap; omitting it selects the write-combined upload heap. A persistent mapping (`IREE_HAL_MAPPING_MODE_PERSISTENT`) requires the buffer to have been allocated with `MAPPING_PERSISTENT` usage and keeps the caller responsible for releasing the buffer, which is why the four views are held on `xla_ctx`.

### 5.3 Measure the pairing, not just the medians

Two A/B sessions with the same old binary produced old medians four percent apart, larger than the effect being measured. Alternating old and new within each round, and reporting per-round pairs, keeps host drift from being mistaken for a regression.

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| `hipHostMalloc` | Page-locked host allocation | The per-step cost this PR removes |
| `HOST_LOCAL` | IREE composite memory type (host visible, coherent) | The type requested for the persistent scalars |
| `IREE_HAL_MAPPING_MODE_PERSISTENT` | Mapping that lives until explicitly unmapped | Keeps a writable pointer for the context's lifetime |
| `transfer_d2d` | Device-to-device copy through the transfer queue | Replaces the staging d2h readback |

### Related PRs/Issues

- Issue #3028: the filed performance issue with the measurements this change acts on.
- PR #2976: split ragged load timing and isolated decode-only throughput, which made the baseline comparable.
- #2886 and #2873: the remaining decode-path gaps (narrow-weight projections, small-M vecmats and the serial argmax), tracked separately.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 6 |
| Lines added | +544 |
| Lines deleted | -17 |
| Tests added | 3 device tests |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Performance | 1 | Persistent decode scalars and d2d readback |
| Tests | 1 | New `xla_decode_scalar_reuse` gate, `[[test]]` entry |
| Documentation | 1 | CHANGELOG entry |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `5bea811d` | perf | perf(xla): reuse decode step scalars and token readback (#3028) |

---

## 8. Follow-up Actions

### Required

- [x] Device tests, generated-text identity, A/B benchmark.
- [ ] None outstanding.

### Monitoring Required

- Decode throughput after merge; the change should not alter kernel counts or GPU busy time, so any movement there indicates an unrelated regression.

### Future Improvements

- The remaining gap to MLX (150 against 239 tok/s) is GPU-side: 4.31 ms of kernel time for 1B of 4-bit weights. Specialized narrow-weight projections (#2886) and small-M vecmats plus a parallel argmax (#2873) are the next candidates.
- If step pipelining lands, the three input slots need double buffering; the readback slot is a single trailing value and stays valid.

---

## Appendix

### A. Test Results

```
cargo test --features "xla-iree test-utils" --test xla_decode_scalar_reuse -- --ignored --test-threads=1   # 3 passed
cargo test --features "xla-iree test-utils" --test xla_ragged_hardening -- --ignored --test-threads=1      # 6 passed
cargo test --features "xla-iree test-utils" --test xla_prepared_prefill -- --ignored --test-threads=1      # 4 passed
cargo check --features xla-iree                                                                            # pass
cargo fmt --all -- --check                                                                                 # pass
```

### B. Performance Benchmarks

Two interleaved four-round A/Bs, llama-3.2-1b-instruct-4bit, native quant, context capacity 512, n=256, greedy:

| round | old tok/s | old ms | new tok/s | new ms |
|-------|-----------|--------|-----------|--------|
| 1 | 138.72 | 1845.50 | 145.80 | 1755.82 |
| 2 | 142.66 | 1794.51 | 145.87 | 1755.02 |
| 3 | 137.69 | 1859.25 | 150.76 | 1698.03 |
| 4 | 127.83 | 2002.66 | 155.15 | 1650.06 |
| 5 | 144.98 | 1765.80 | 142.78 | 1792.96 |
| 6 | 136.68 | 1873.01 | 151.72 | 1687.33 |
| 7 | 144.73 | 1768.77 | 153.15 | 1671.58 |
| 8 | 143.60 | 1782.73 | 149.48 | 1712.58 |

Combined medians: 150.1 against 140.7 tok/s, 6.66 against 7.11 ms per step.

### C. References

- IREE HAL memory types and mapping: `iree/hal/buffer.h` in the pinned ROCm distribution.
- HIP pinned allocation behaviour: `hip_allocator.c` in the same distribution.
