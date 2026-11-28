# Technical Report: PR #2220 - Give Every Scalar-Pool Slot Its Own Cache Line

**Date**: 2026-10-07

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); code head `a7c85a73` on main `8ac65bd8`, PR open, pending merge. Closes #2213.

**Languages**: C++ (ROCm overlay `allocator.cpp`, `allocator.h`), Rust (new `tests/rocm_scalar_pool_concurrency.rs`, edited `tests/rocm_device_map_concurrency.rs`), Markdown (`patches-rocm/LOCAL_FIXES.md`)

**Risk Level**: Low. The change moves the 8192 scalar slots from 8 bytes apart to 128 bytes apart and grows the pool from 64 KB to 1 MB. No kernel, argument or allocator locking changes. Metal and CUDA code paths are untouched.

## Executive Summary

Issue #2213 came out of the test work for #2197 (PR #2214). That test evaluates `arange * 2 + 1` with `multiply_scalar` and `full_like` on 8 threads, each on its own stream. With its own fix applied it failed 9 of 10 runs, and the wrong outputs were not the device map's doing: an elementwise kernel read a scalar constant that another scalar had held earlier. #2214 worked around it with full-length constant arrays and filed #2213.

Any value of at most 8 bytes that the host writes and a kernel reads comes from `SmallSizePool`: `array(2.0f)`, `full(shape, value)`, `multiply_scalar`, `full_like`, the softcap and scaling helpers, and the one-token index of a decode step. The server evaluates on several threads that each own a stream (batch scheduler, embedding, rerank, audio workers), so a wrong scalar there is a silently wrong result, not a crash.

The cause was isolated with a probe that draws every scalar from one process-wide increasing counter. A wrong value then names the thread that wrote it and whether it was written before or after the value the kernel should have read. On main 10 of 10 runs were wrong, 75 wrong scalars, every one written earlier. A live-bit check in the allocator never fired, serializing kernels or using uncached memory made the failure disappear, and fences made no difference or made it worse. The conclusion is a stale GPU-cached 128-byte line installed by a kernel on another queue that was reading a neighbouring slot, not an allocator race and not CPU store ordering. Which cache level holds the stale line was not identified.

The fix places each slot 128 bytes apart (`small_block_stride`), so no two live scalars share a line. With it the probe fails 0 of 20 runs; with the stride set back to 8 it fails 18 of 20. Decode throughput on Llama-3.1-8B-4bit is unchanged within its spread. `make verify-rocm` on the head passes (161 suites, 12,131 passed, 0 failed). Metal and CUDA were not verified, and whether upstream CUDA's managed-memory pool has the same hazard is unknown.

## 1. Problem Statement

### 1.1 How it was found

The #2197 unit wrote `tests/rocm_device_map_concurrency.rs` with `multiply_scalar` and `full_like` for the operands. With the device-map fix it failed 9 of 10 runs, and with `device.cpp` restored to main it failed 10 of 10, so the fix was not the cause. The #2214 report records the narrowing: one stream per thread still failed, the same constants as full-length host arrays passed 10 of 10, and an op with no host data passed 10 of 10. Scalars of at most 8 bytes are the only operands served from `SmallSizePool`, so the pool was the suspect. The two candidate explanations were an allocator race (one slot handed to two live arrays) and a coherence effect (the GPU reads a stale value from a reused slot).

### 1.2 Why it matters

`SmallSizePool` backs the scalars of every elementwise op, the scaling and softcap helpers and the token index of each decode step. A server that runs workers concurrently, each on its own stream, shares this one pool across all of them. The failure produces plausible but wrong numbers with no error, only under concurrent evaluation, which is why single-stream benchmarks and the existing suites do not see it.

### 1.3 The pool before the change

As in upstream CUDA, `SmallSizePool` carved one 64 KB `unified_malloc` into 8192 slots 8 bytes apart (`small_pool_size = 4 * page_size`, `small_block_size = 8`), served through a freelist under the allocator's mutex. Sixteen slots therefore share one 128-byte line.

## 2. Change Summary

| Area | Change |
|---|---|
| `allocator.cpp` | New `small_block_count` (8192, unchanged) and `small_block_stride = 128`. `SmallSizePool` allocates `small_block_count * small_block_stride` bytes and places slot `i` at `i * small_block_stride`. `in_pool` uses `small_block_count`. |
| `allocator.h` | Comment on `SmallSizePool`. |
| `tests/rocm_scalar_pool_concurrency.rs` (new) | The probe as a regression test (section 4.1). |
| `tests/rocm_device_map_concurrency.rs` | Back on `multiply_scalar` and `full_like`, as #2213 asked. |
| `patches-rocm/LOCAL_FIXES.md` | Item 40. |

One commit, 5 files, 357 insertions, 15 deletions.

## 3. Diagnosis

### 3.1 The probe

The child body of `tests/rocm_scalar_pool_concurrency.rs`: 8 threads, each on its own stream, released together by a barrier, 16 rounds of 4 evaluations each (64 per thread). Each evaluation computes `arange_f32(0, 256, 1) * k + c`, with `k` from `multiply_scalar` and `c` from `full_like`, and every element is checked exactly.

The one change from the issue's original probe: every `k` and `c` is drawn from a single process-wide `AtomicU32` counter, and a table records which thread, round, eval and role (`k` or `c`) drew each value. The kernel's output reveals the scalars it read (`c` is element 0, `k` is element 1 minus element 0), so a wrong value is traced to its writer and compared with the expected one by counter order. A stale read shows an earlier value. A slot handed to two live arrays, or a kernel still reading a recycled slot, would show a later one.

### 3.2 Measurements

Run on gfx1151 (ROCm 7.15). "Wrong values" counts the scalars the kernel read that were not the expected ones, taken from the provenance line of each probe log.

| Configuration | Runs wrong | Wrong values | Direction |
|---|---:|---:|---|
| main (8-byte slots) | 10 of 10 | 75 | all earlier, 0 later |
| fix branch, stride set back to 8 | 18 of 20 | 93 | all earlier, 0 later |
| 8-byte slots plus a live bit per pool slot | 5 of 5 | 28 | all earlier; the live-bit abort never fired |
| `AMD_SERIALIZE_KERNEL=3` | 0 of 5 | 0 | |
| `HSA_DISABLE_CACHE=1` | 0 of 5 | 0 | |
| slots 128 bytes apart | 0 of 5 | 0 | |
| pool from `hipDeviceMallocUncached` | 0 of 5 | 0 | |
| `GPU_FLUSH_ON_EXECUTION=1` | 5 of 5 | 238 | all earlier (worse: 221 mismatching evaluations) |
| `MLX_ROCM_FINEGRAINED=0` (coarse `hipMalloc` pool) | 5 of 5 | 31 | all earlier |
| `__sync_synchronize()` before every eager `hipLaunchKernel` | 5 of 5 | 34 | all earlier |
| fix (128-byte stride), full probe | 0 of 20 | 0 | |

The first two rows have a second reading. In the 75 and 93, most wrong scalars are the slot's previous occupant, often the same thread's previous evaluation (for example thread 2 round 10 eval 0 expected `k = 612` and the kernel read 596, thread 2 round 9 eval 3's `k`).

### 3.3 Reading the table

- **Direction.** Every wrong value is an earlier write, none a later one, in every configuration that fails. That is a stale read of a slot's old contents.
- **Not an allocator race.** The live-bit experiment set a bit under the allocator mutex when a pool slot left `malloc_async` and cleared it in `free`, aborting on a second live owner or a free of a dead slot. It never fired in 5 of 5 wrong runs. No slot was live twice and no freed slot was still being read. The pool's locking also matches upstream CUDA `allocator.cpp` at pin 81ba1c6a, checked against that file: `malloc_async` takes `std::unique_lock lock(mutex_)` before `scalar_pool_.malloc()`, and `free_cuda_buffer` is documented as called with `mutex_` held and calls `scalar_pool_.free`.
- **Not CPU store ordering.** A full fence before every launch changes nothing.
- **Concurrent kernels matter.** `AMD_SERIALIZE_KERNEL=3` removes the failure.
- **A GPU cache is involved.** `HSA_DISABLE_CACHE=1` and an uncached pool remove the failure.
- **Line sharing is the trigger.** Giving each slot its own 128-byte line removes the failure, so what is stale is a line shared with a neighbouring slot.
- **The L2 fences do not help.** `GPU_FLUSH_ON_EXECUTION=1` writes back and invalidates L2 at system scope on every dispatch and fails more often, and the coarse-grained pool fails too. The stale copy is not cured by flushing L2.

### 3.4 Cause

A GPU-cached 128-byte line holding a slot's old value, installed by a kernel on another queue that was reading a neighbouring slot in the same line while this thread rewrote its slot and launched. The launch then read that line's old contents. The cache level that holds the stale line is not identified: the L2 fences do not cure it, so it is something else below the host write, and no experiment here separated the remaining levels. The fix does not depend on the answer. The 128-byte stride matches the line size of gfx11 per LOCAL_FIXES item 40; that figure was not measured in this work.

## 4. Design

### 4.1 The fix

`SmallSizePool` keeps its 8192 slots and its freelist. Each slot's data now sits `small_block_stride = 128` bytes after the previous one, so the pool allocates 8192 * 128 = 1 MB through `unified_malloc` instead of 64 KB. Each `RocmBuffer` still reports `size = small_block_size` (8). A line that no other live scalar shares cannot be held by a running kernel when its owner's host write lands.

### 4.2 Rejected

- **An uncached pool** (`hipDeviceMallocUncached`) passed 5 of 5, but every wave's scalar load would then go to memory.
- **Serializing kernels or disabling the cache** (`AMD_SERIALIZE_KERNEL=3`, `HSA_DISABLE_CACHE=1`) are diagnostics, not options for a server.
- **Fences** (CPU or L2) do not cure it.
- **Locking changes** are moot, since the allocator was not the cause.

## 5. Verification

### 5.1 The test

`tests/rocm_scalar_pool_concurrency.rs` (`#![cfg(feature = "rocm")]`) has a parent test that spawns an `#[ignore]` child 10 times in fresh processes. The child is the probe of section 3.1. Each child must exit 0 and print its done marker, so a child that skipped or died before the check cannot pass. A failure message names the element, the expected value, and for each wrong scalar its writer and whether it was written earlier or later than the expected one. The module doc records the measurements.

`tests/rocm_device_map_concurrency.rs` is back on `multiply_scalar` and `full_like`, with its comment updated.

### 5.2 Results

On gfx1151 (one GPU, ROCm 7.15):

| Build | Child runs wrong |
|---|---:|
| main | 10 of 10 |
| fix branch, stride 8 | 18 of 20 |
| This PR (stride 128) | 0 of 20 |

The parent test (10 children) and `tests/rocm_device_map_concurrency.rs` pass.

### 5.3 Performance

Decode on Meta-Llama-3.1-8B-Instruct-4bit under `rocm_gpu_guard.sh`, medians of 3 runs: 37.71 tok/s before (range 37.45 to 38.09) and 37.95 after (range 37.43 to 37.96). The difference is inside the run-to-run spread, so no change is claimed. The extra 1 MB of pool is allocated once.

### 5.4 Gates

- `cargo clippy` on both tests: clean. `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay`: pass.
- Orchestrator's `make verify-rocm` on `a7c85a73` (on main `8ac65bd8`): 161 suites, 12,131 passed, 0 failed, 395 ignored, smoke OK.

Not verified: Metal and CUDA (not available on this host; the change touches only the ROCm overlay allocator and two ROCm-gated tests).

## 6. LOCAL_FIXES Item 40

The entry, "Every scalar-pool slot on its own cache line", records the pool and its users, the measurements and what each ruled out, the new stride and pool size, why the uncached pool was not taken, the test results, and the upstream note: upstream CUDA at 81ba1c6a keeps the 8-byte packing on managed memory and locks the pool the same way, and whether the hazard is reachable there is not known. It ends with the fork-policy wording: "Applies to the fork; kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there (lablup/mlxcel#2213)."

## 7. Technical Decisions

- **Stride, not a different memory type.** It removes the sharing at no per-load cost, and it keeps the pool in the same host-visible memory as before.
- **Keep 8192 slots.** The count is CUDA's and nothing showed it was a limit. Only the spacing changed.
- **Counter-ordered probe.** Provenance by writer and time turned "wrong value" into a direction, which is what ruled out the allocator.
- **Leave the cause's cache level open.** The fix works at line granularity, and an experiment to name the level would not change it.

## 8. Residual Risks

- **The cache level is unknown.** If the stale copy were held at a granularity larger than 128 bytes, the stride would be too small. The 0 of 20 result says it is not on this host.
- **One host.** All runs are on one gfx1151 with ROCm 7.15. Other ROCm architectures with different line sizes are untested.
- **Other small host-written buffers.** Only the scalar pool was examined. Another allocation that a host rewrites while a neighbour in the same line is being read would behave the same way.
- **CUDA is unexamined.** Upstream's pool packs 8-byte slots in managed memory. Whether a GPU cache can hold a stale line there is unknown, and no CUDA host was available.
- **Performance is claimed unchanged only within noise.**

## 9. Learning Points

- **Provenance beats a pass/fail count.** Drawing every value from one counter made the failure say "stale", which a failure count alone would not have.
- **A cheap assertion can rule a hypothesis out.** The live bit took a few lines and removed the allocator from the list.
- **Two null results and a worse one are still evidence.** Fences that do nothing or make it worse narrowed the cause as much as the stride that fixed it.
- **A test written with realistic operands found it.** The #2197 test used scalar ops and hit a bug unrelated to its subject.

## 10. The Series

This closes one line of work on shared state on the ROCm launch path. Each concurrency test, once it checked every element exactly across threads, reached the next shared global that nobody had looked at.

| Step | Issue / PR | What it fixed |
|---|---|---|
| 1 | #2196 | JIT module cache and HIP event pool |
| 2 | #2202 | `Device`'s rocBLAS handle, created once under a lock |
| 3 | #2208 | hipBLASLt handle, per-stream workspace, thread-local pipe cache |
| 4 | #2214 | The device map, plus leaking the event pool |
| 5 | #2220 (this PR, #2213) | Host-written 8-byte scalars sharing cache lines in `SmallSizePool` |

Open after the series:

- **Which cache level holds the stale line.** An experiment that separates the levels (for example by varying the distance between slots, or reading with different cache-control bits) would name it. The fix does not need it.
- **Whether upstream CUDA's managed-memory pool has the same hazard.** It needs a CUDA host and the same probe. If it does, it is an upstream question, and under the 2026-10-06 fork policy nothing is proposed there from this work.
