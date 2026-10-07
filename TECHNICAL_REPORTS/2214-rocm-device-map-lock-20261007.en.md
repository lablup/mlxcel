# Technical Report: PR #2214 - Lock the Device Map in rocm::device()

**Date**: 2026-10-07

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); code head `05368e26` on main `454500b9`, PR open, pending merge. Closes #2197 (part of #1801).

**Languages**: C++ and HIP (ROCm overlay `device.cpp`, `device.h`, `event.hip`), Rust (new `tests/rocm_device_map_concurrency.rs`), Markdown (`patches-rocm/LOCAL_FIXES.md`)

**Risk Level**: Low. The race is latent on every host that exists today, the change adds one uncontended shared lock to the per-primitive lookup, and no kernel, argument or result changes. The one new behavior is that `Device` objects and the HIP event pool are now leaked, so their destructors no longer run at exit. Metal and CUDA code paths are untouched.

## Executive Summary

Issue #2197 came out of the review of #2196, which locked the JIT module cache and the HIP event pool. The review found the same pattern in `rocm::device()`: it kept one `Device` per HIP index in a function-static `std::unordered_map<int, Device>` and read it with no lock on every primitive launch, from whichever thread evaluates, with an unlocked `try_emplace` on a miss. `record_stream_error`, `clear_all_encoders` and `Device::clear_encoders` had the same gaps.

The race is not reachable today. Every insert comes from a stream's creation and `mlx::core::new_stream` serializes those, and on a one-GPU host the only key is 0, inserted before any thread can evaluate. It becomes reachable when a first stream on GPU 1 overlaps evaluation on GPU 0, which the multi-GPU work in #486 and #488 will do. The PR fixes it ahead of that.

The map is now a leaked `DeviceTable` (a `std::shared_mutex` plus the map) reachable only through two helpers in an anonymous namespace. `device()` looks up under a shared lock and builds a missing `Device` under the unique lock after a re-check. The event pool's map and mutex are leaked as well. The fork keeps lazy per-device construction instead of upstream CUDA's eager construction.

The test cannot reproduce the insert race on this one-GPU host: 0 of 10 runs failed with the fix and 0 of 10 with `device.cpp` restored to main. It guards the lookup path, the one-`Device`-per-index property and a clean exit. The second-GPU test skips below two GPUs and has not run anywhere. Writing the test also exposed a separate bug, filed as #2213: 8-byte host-written scalars read values other scalars held earlier under multi-threaded evaluation. The test uses full-length constants until #2213 is fixed.

## 1. Problem Statement

### 1.1 Shared-state inventory

Read on main at `84d3a7bc`, all in `device.cpp` unless noted:

| State | Guard before | Problem |
|---|---|---|
| function-static `unordered_map<int, Device>` behind `get_devices()` | none | `device()` did `find` and, on a miss, `hipSetDevice`, `ensure_device_flags` and an unlocked `try_emplace`. |
| `record_stream_error` | none | Read the same map unlocked from the event error path. |
| `clear_all_encoders()` | none | Iterated the map unlocked. |
| `Device::clear_encoders()` | none | Cleared `encoders_` without `encoders_mtx_`, which `get_command_encoder` and `find_encoder` take. |
| `HipEventPool` map and mutex (`event.hip`) | locked by #2196, but destructible | A `HipEvent` released during static teardown pushes into a destroyed map; the map's destructor ran `hipEventDestroy` on every pooled handle, which fails on a faulted device (LOCAL_FIXES item 7). |

`device()` runs for every primitive `gpu::eval` launches (`get_command_encoder`, `finalize`, `synchronize`, `new_stream`) and is called directly in the `eval_gpu` of qmm, SDPA, flash attention, conv, compiled and fp8 convert. The server's scheduler, embedding, rerank and audio workers each evaluate on their own thread-local stream.

### 1.2 Reachability

- Every insert comes from a stream's creation, and `mlx::core::new_stream` holds the `all_streams()` unique lock around it, so two first inserts never overlap on any host.
- On a one-GPU host the only key is 0, inserted by the process's first stream before any thread can evaluate. Every later call is a `find` on a map that no longer changes.
- The race needs two GPUs: a first stream on GPU 1 (the insert) while another thread evaluates on GPU 0 (a `find` walking the same buckets). That is a data race and undefined behavior.
- Nothing in production targets index 1. `--main-gpu` other than 0 is rejected, and `new_stream_on_gpu`, `set_default_gpu_device` and `new_thread_local_stream_on_gpu` have no callers outside tests. The multi-GPU work in #486 and #488 would make it reachable.

## 2. Change Summary

| Area | Change |
|---|---|
| `device.cpp` | Leaked `DeviceTable` behind `device_table()` and `find_device()`; `get_devices()` removed; `device()` with shared lookup and unique-lock construction; `record_stream_error` and `clear_all_encoders` through the same lock; `Device::clear_encoders` swaps under `encoders_mtx_` and destroys outside it. |
| `device.h` | Comments on `device()` and `clear_all_encoders()`. |
| `event.hip` | `HipEventPool` map and mutex leaked. |
| `tests/rocm_device_map_concurrency.rs` (new) | Two fresh-process tests (section 4.1). |
| `patches-rocm/LOCAL_FIXES.md` | Item 39. |

One commit, 5 files, 533 insertions, 21 deletions.

## 3. Design

### 3.1 The device table

`DeviceTable` holds a `std::shared_mutex` and the `unordered_map<int, Device>`. It is allocated with `new` and never destroyed. `device_table()` and `find_device()` live in an anonymous namespace, so nothing outside `device.cpp` can reach the map, and `get_devices()` and its forward declaration are gone (`grep -rn get_devices src/lib/mlx-cpp/patches-rocm` is empty).

`find_device(index)` takes a shared lock, looks the index up and returns a `Device*` or null. The lock is released before the caller uses the result. Nothing is ever erased, so a `Device&` stays valid for the rest of the process.

### 3.2 `device()`

- **Hit.** `find_device` and return. One shared lock, no HIP call. This is the path every primitive takes.
- **Miss.** Take the unique lock, find again, and only if the index is still missing run `hipSetDevice(index)`, `ensure_device_flags(index)` and `try_emplace(index, index)`. Several threads that first use one index build one `Device`. The construction touches only the requested index.
- **Failure.** If the constructor or HIP throws, nothing is inserted, the unwind releases the lock and the next call retries.

### 3.3 Callers that reach into a `Device`

`record_stream_error` calls `find_device`, then `find_encoder` on the result with the table unlocked. `clear_all_encoders` copies the `Device*` values under the shared lock into a vector, releases the lock and calls `clear_encoders()` on each. The rule is that the table lock is never held while calling into a `Device`, except for the construction in `device()`. An encoder destructor that reaches `record_stream_error` or `device()` therefore cannot deadlock on the non-recursive lock. Lock order is the table, then `encoders_mtx_`; nothing holding the table lock waits on anything else.

`Device::clear_encoders()` swaps `encoders_` into a local map under `encoders_mtx_`, and the local map destroys the encoders after the lock is released, so `~CommandEncoder` (which waits on in-flight work and releases HIP resources) never runs under that mutex.

### 3.4 The event pool

`HipEventPool::cache_for` now returns from a `new`-allocated `std::map` and `mutex()` from a `new`-allocated `std::mutex`. A `HipEvent` released during static teardown can no longer push into a destroyed map, and no `hipEventDestroy` runs at exit. The pooled handles go away with the process. This is the same treatment the JIT module cache received (items 35 and 36) and is what the #2196 review asked for.

### 3.5 Lazy versus eager construction

Upstream CUDA at the pin 81ba1c6a (`mlx/backend/cuda/device.cpp`, `device(int)`) leaks a `std::vector<Device>` filled in one magic-static initializer that constructs a `Device` for every `gpu::device_count()` index, so its lookups are read-only afterwards. That was verified against the pin. The fork keeps lazy, per-device construction, for the reason given in the issue: `Device::Device` binds the device with `hipSetDevice`, and on a multi-GPU host a context or queue on a GPU nobody asked for wedges the discrete GPU's queue over a TB5 link (see `ensure_device_flags`). Eager construction is out of scope for the PR.

### 3.6 Rejected

- **Eager construction of all devices** (upstream): see section 3.5.
- **Per-key locking:** out of scope per the issue.
- **Editing the CUDA overlay:** out of scope per the issue.

## 4. Verification

### 4.1 The test

`tests/rocm_device_map_concurrency.rs` (`#![cfg(feature = "rocm")]`) has two parent tests, each spawning an `#[ignore]` child 10 times in fresh processes, so the process's first stream (the index-0 insert) happens with every thread already released from the barrier.

- `streams_created_while_others_evaluate`: 8 threads behind a barrier, 16 rounds. In each round every thread creates and installs a new thread-local stream, then evaluates `arange * 2 + 1` four times, so `device()` lookups overlap other threads' stream creation and evaluation. Every element is checked exactly.
- `second_gpu_first_stream_while_first_gpu_evaluates`: skipped unless `gpu_device_count() >= 2`. 8 threads keep evaluating on GPU 0 while one thread creates and installs the first stream on GPU 1, fills `ones` on it and runs the same check.

Each child must exit 0, print its done marker, print `[mlx-rocm] bound HIP device N:` exactly once per index it used (one `Device` per index however many threads raced for it), and print no `releasing a HIP handle failed` line, which is what a failing `hipEventDestroy` or handle release at static teardown looks like. The constants are full-length host-written arrays (section 4.3).

### 4.2 What the test can and cannot show

On this host (one gfx1151) the insert race cannot run, as section 1.2 predicts. The test is a crash guard for the lookup path under concurrent stream creation and a check of the teardown behavior and of the one-`Device`-per-index property. It is not a reproduction. The module doc in the test says so.

### 4.3 Results

On gfx1151 (one GPU, ROCm 7.15), 10 runs of `tests/rocm_device_map_concurrency.rs` per build, 10 children each:

| Build | Runs failed |
|---|---:|
| `device.cpp` restored to main (fix removed) | 0 of 10 |
| This PR | 0 of 10 |

The two builds are indistinguishable here, so the numbers say nothing about whether the fix works; they say the new code does not break the single-GPU path. The second test printed its skip line (this host reports one GPU), so the index-1 insert under index-0 lookups has not run anywhere. No multi-GPU host was available. The teardown check on `releasing a HIP handle failed` passes with the fix; the report makes no claim that it fails without it.

### 4.4 Discovery: #2213

The issue's original test body used `multiply_scalar` and `full_like` for the `arange * 2 + 1` check. That version failed 9 of 10 runs with the fix and 10 of 10 without it, with wrong constants in the output. The cause is not the device table.

A probe (8 threads behind a barrier, 16 rounds of 4 evaluations of `arange * k + c`, every element compared exactly, in one process) isolated it:

- Host-written 8-byte scalars read a value that another scalar held earlier: another thread's constant, or the same thread's constant from its previous evaluation. Scalars of at most 8 bytes come from `SmallSizePool` in `allocator.cpp`.
- One stream per thread for the whole run, instead of a new stream per round, still failed (3 of 5 runs; with per-thread distinct values 5 of 5). Stream creation is not a factor.
- The same constants as full-length host-written arrays (1 KB each) passed 10 of 10 runs, and an op with no host data (`a + a + arange(1, 257)`) passed 10 of 10.

The cause is not isolated. Either one 8-byte slot is handed to two live arrays (an allocator race) or the GPU reads a stale value from a reused slot (a coherence effect); no measurement yet separates them. #2213 (priority high) is being implemented now and lists the discriminating experiments. Until it lands, the #2197 test uses full-length host-written constants, with a comment pointing at #2213, and #2213's acceptance criteria include moving this test back to `multiply_scalar` and `full_like`.

### 4.5 Performance

The hit path adds one uncontended shared lock per primitive and no HIP call. This was not measured and the report claims nothing about throughput.

### 4.6 Gates

- `tests/rocm_jit_module_concurrency.rs` and `tests/rocm_gpu_faults.rs`: pass. `cargo clippy --features rocm --test rocm_device_map_concurrency -- -D warnings`: clean. `grep -rn get_devices src/lib/mlx-cpp/patches-rocm`: empty.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay`: pass (39 LOCAL_FIXES entries).
- Orchestrator's `make verify-rocm` on `05368e26` (on main `454500b9`): 159 suites, 12,116 passed, 0 failed, 389 ignored, smoke OK.

Not verified: Metal and CUDA (not available on this host; the change touches only files under `patches-rocm/` and a `rocm`-gated test).

## 5. LOCAL_FIXES Item 39

The entry, "Device table locked and leaked, event pool leaked, encoders cleared outside the locks", records the unlocked sites, the reachability argument, the new structure and lock order, the upstream comparison and why eager construction was rejected, the leak and its effect on static teardown, the test with its single-GPU result, and the #2213 note. It ends with the fork-policy wording: "Applies to the fork; kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there (lablup/mlxcel#2197)." Nothing is added under `docs/mlxcelverse/upstream/`.

## 6. Technical Decisions

- **Shared lock on the hit path, unique lock only to construct.** Every primitive reads the table; only a first use of an index writes.
- **Re-check under the unique lock.** Several threads that miss together build one `Device`.
- **Never hold the table lock while calling into a `Device`.** It removes the lock-inversion and recursion cases with encoder destructors.
- **Leak the table and the event pool.** It matches upstream CUDA, matches the JIT cache, and removes teardown-time HIP calls on a faulted device.
- **Keep lazy construction.** The fork must not touch a GPU it was not asked for.
- **Fix it before it is reachable.** The multi-GPU work will otherwise hit an undefined-behavior race with no test pointing at it.

## 7. Follow-ups

- **#2213, scalar-pool constants under multi-threaded eval.** Next in the series. Isolate the cause, fix it, add `tests/rocm_scalar_pool_concurrency.rs`, and return the #2197 test to scalar constants.
- **Run the second-GPU test on a multi-GPU host** when one is available.

## 8. Residual Risks

- **The insert race was never exercised.** The fix rests on the reachability argument and the lock structure, not on a failing-then-passing test.
- **The second-GPU test has not run anywhere.** A mistake in its harness would show first on a two-GPU host.
- **Leaked `Device` objects.** Their destructors no longer run at exit, which is the point, but any cleanup that relied on them now needs to be explicit.
- **Hit-path cost unmeasured.** One uncontended shared lock per primitive is expected to be small.
- **One host.** All runs are on one gfx1151.

## 9. Learning Points

- **A guard test is not a reproduction.** The 0 of 10 and 0 of 10 are reported as they are; the test's value here is the crash and teardown checks.
- **Write the test with realistic operands.** Using scalar ops in the first draft found a bug the device-table change had nothing to do with.
- **Do not attribute a failure to the change under test without a control.** The 10 of 10 failures without the fix looked like a reproduction and were not.

## 10. The Series

This is the fifth step of one line of work on shared globals on the ROCm launch path, and each lock fix exposed the next global.

| Step | Issue / PR | What it fixed |
|---|---|---|
| 1 | #2196 | JIT module cache and HIP event pool |
| 2 | #2202 | `Device`'s rocBLAS handle, created once under a lock |
| 3 | #2208 | hipBLASLt handle, per-stream workspace, thread-local pipe cache |
| 4 | #2214 (this PR) | The device map, plus leaking the event pool |
| 5 | #2213 | 8-byte scalar allocation under multi-threaded evaluation (in progress) |

The #2196 review found the device map. The #2208 test work found the bf16 residue (#2206), and the #2214 test work found #2213. In each case the concurrency test, once it checks every element exactly across threads, reaches the next shared global that nobody had looked at.
