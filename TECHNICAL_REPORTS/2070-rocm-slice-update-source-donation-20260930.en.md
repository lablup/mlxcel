# Technical Report: PR #2070 - Stop ROCm SliceUpdate from donating a still-referenced source

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: C++ and HIP (ROCm overlay), C++ (cxx bridge), Rust (tests), Markdown

**Risk Level**: Medium (two overlay primitives change their buffer-reuse rule on a path every KV cache write takes; the model-visible effect is a correctness fix, and decode throughput was measured unchanged. Metal and CUDA builds never copy `patches-rocm/`)

## Executive Summary

Issue #2052 (part of epic #1801) was found while implementing #2050: a GPU `slice_update` on ROCm wrote its result into the source array whenever the source's data buffer had one owner, even if the caller still held the source array. The issue asked whether any mlxcel path reads a pre-update array after a `slice_update`, which would make this a live correctness bug rather than a latent one. It was live. Two model paths produced wrong output on gfx1151 with the old overlay:

- DeepSeek-V4 `PoolingCache` chunked prefill: the first logit moved by about 9e-3 relative against the CPU stream. After the fix the difference is 1e-5.
- Prompt-cache snapshots of a `RotatingKVCache` (Gemma 3/4, AFMoE, Muse Glimmer): the next token's ring write landed inside the saved snapshot. The default in-place KV warmup write hid this; with `MLXCEL_KV_INPLACE_WRITE=0` an existing test failed on the old build.

`SliceUpdate::eval_gpu` and `DynamicSliceUpdate::eval_gpu` now call `copy_gpu`, which donates only when `array::is_donatable()` holds (the array and its buffer each have one reference), as upstream CUDA and Metal do. A source dropped before evaluation, which is how the KV caches write (`self.keys = slice_update(self.keys, ...)`), is still donated. The PR also removes a `graph_active()` override in `DynamicSliceUpdate` that forced donation under HIP graph capture: it could never fire, because HIP graphs are compiled off. Decode throughput on two models did not change beyond run-to-run spread.

## 1. Problem Statement

### Counting the buffer instead of the array

An MLX `array` is a handle to an `array_desc_`, which holds a shared pointer to the data buffer. Two things can keep an array readable after a `slice_update` is built on it: another handle to the same `array_desc_` (a Rust variable, a cache field), or an unevaluated node that takes the array as an input (a lazy `Slice` or `Copy`). Neither of those holds the data buffer directly. Only the array does.

The overlay's `SliceUpdate::eval_gpu` (`mlx/backend/rocm/indexing.hip`) and `DynamicSliceUpdate::eval_gpu` (`mlx/backend/gpu/primitives.cpp`) decided to donate with `in.data_shared_ptr().use_count() == 1` plus contiguity checks. That count is 1 in both situations above, so the output took the source's buffer and the kernel wrote the update into memory the source still pointed at. Any later read of the source, or evaluation of a lazy node over it, saw the updated values.

Upstream MLX at the pin (81ba1c6a) uses `array::is_donatable()`: `array_desc_.use_count() == 1 && array_desc_->data.use_count() == 1`. CUDA's `SliceUpdate::eval_gpu` just calls `copy_gpu`, whose `set_copy_output_data` donates only when `is_donatable(in, out)` holds. Metal takes the same route. The ROCm overlay was the only backend with the buffer-only rule.

### The two reproduced wrong-output paths

**DeepSeek-V4 `PoolingCache` chunked prefill.** In prompt mode, `PoolingCache::accumulate_windows` returns completed compression windows as lazy reads of `buf_kv` and, in the same call, writes the new remainder tail over the start of `buf_kv` with `slice_update`. When a chunk completes a window that spans the previous remainder and also leaves a new remainder, the returned window (`r_kv`) must read the old remainder rows while the tail write replaces them. The model's cache barrier (`eval_state`) evaluates the buffer before anything reads the windows. On the old overlay the tail write donated `buf_kv`'s buffer, so by the time `r_kv` was evaluated it read the new tail. `tiny_model_chunked_prefill_with_pool_remainder_matches_cpu` (a 14 + 4 prompt split at compression ratio 4) differed from the CPU stream by about 9e-3 relative on the first logit. After the fix it agrees within 1e-5.

**`RotatingKVCache` prompt-cache snapshots.** `ModelStateSnapshot::push_tensor` captures a cache tensor through `ModelStateTensor::new`, which is a lazy `ffi::copy` of the live keys array. The copy node holds the keys array, not its buffer. When a one-token suffix wraps the ring, `RotatingKVCache` overwrites one slot with `slice_update`. On the old overlay that write donated the keys buffer, the snapshot's copy was evaluated afterwards, and the snapshot held the new token in place of the one it saved. This affects Gemma 3 and 4, AFMoE and Muse Glimmer, the families that use `RotatingKVCache` with prompt caching. The existing test `rotating_inplace_warmup_and_wrap_keep_shared_snapshots_intact` passed on the old build only because the default in-place warmup write (`MLXCEL_KV_INPLACE_WRITE`, on by default) goes through `inplace_slice_write` and not `SliceUpdate`; with the variable set to 0 the test failed. The PR adds `rotating_steady_state_snapshot_survives_the_next_wrap_write`, which reaches the steady-state wrap directly.

### Paths checked and found unaffected

The PR author traced every mlxcel caller of `slice_update` that could hold the source across the update:

- `KVCache` append and trim, in every mode.
- `RotatingKVCache` speculative and MTP paths.
- `RingSlidingKVCache`.
- Inkling and Gemma 4 MTP rollback.
- Cache detach and adopt.
- Paged pool writes and snapshot restore.
- VLM embedding merges and RoPE merges.

In these the source is either dropped before evaluation (the reassign pattern) or read only after it has been materialized into a separate buffer. The single-token FP16 decode write uses `inplace_slice_write`, which is a separate primitive and is unchanged.

One benign donation is lost. In the paged per-sequence fallback in `llama3.rs` (taken when the batched path declines), the previous sequence's lazy gather still holds the shared slab, so each sequence after the first now copies the slab instead of writing into it. That write was correct only because the gather had already read the rows it needed; the new rule does not know that and copies.

## 2. Change Summary

- **`src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/indexing.hip`** (`SliceUpdate::eval_gpu`): the hand-written `can_donate` branch and `out.copy_shared_buffer(in)` are replaced by a single `copy_gpu(in, out, ctype, stream())`. The copy type (Scalar, Vector or General) is chosen as before. The comment points at `LOCAL_FIXES.md` item 24.
- **`src/lib/mlx-cpp/patches-rocm/mlx/backend/gpu/primitives.cpp`** (`DynamicSliceUpdate::eval_gpu`): the same replacement, plus removal of the `rocm::graph_active()` forward declaration and the `|| graph_active()` term (section 3).
- **`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.{h,cpp}`**, **`src/lib/mlxcel-core/src/lib.rs`**: a test-only `slice_update_dynamic(src, update, start, axes)` bridge that builds the `DynamicSliceUpdate` primitive. No model path builds that primitive, so this is the only way a test can reach its `eval_gpu`. The bridge rejects an axis out of range or a `start` whose length differs from `axes`, and clamps each start offset to `[0, src_dim - update_dim]`, because the GPU kernels write at those offsets without a bounds check and the entry point is safe Rust.
- **`tests/rocm_slice_update_source.rs`** (new, 272 lines, `#![cfg(feature = "rocm")]`): holds an evaluated source, runs one GPU update through `slice_update` (None), `slice_update_reduce` (Sum and Max) and `slice_update_dynamic`, and asserts the output and that the source is unchanged. Two further cases check the dynamic start clamp and that a dropped source still produces the right output. Each test holds `lock_default_device` for its body.
- **`tests/rocm_slice_update_reduce.rs`**: the private source copy that #2050 added as a workaround is removed. The GPU op now runs first on the same `src` the CPU reference then reads, so the test also guards this fix. Peak memory in the largest case drops from about 1.3 GB to 1.1 GB.
- **`src/models/deepseek_v4_tests.rs`**: `pooling_cache_remainder_survives_an_overlapping_tail_write` (the `PoolingCache` sequence in isolation) and `tiny_model_chunked_prefill_with_pool_remainder_matches_cpu` (the model-level form, 3e-3 tolerance to leave room for CUDA's default TF32 matmuls).
- **`src/lib/mlxcel-core/src/cache.rs`**: `rotating_steady_state_snapshot_survives_the_next_wrap_write`.
- **`src/lib/mlxcel-core/src/generate.rs`**: the doc comment on `ModelStateTensor::new` now says the capture is a lazy copy that shares the buffer once evaluated, and that it stays correct because a later `slice_update` copies while the capture still references the array.
- **`src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`**: item 24, with the mechanism, both reproduced bugs, the `graph_active()` reasoning, the tests and the throughput numbers, marked as an upstreaming candidate for #1813.

Commits: `338be1cc` is the fix and tests, `24e22975` adds the dynamic start clamp and hardens the new tests, `198f7a0f` records the throughput in item 24.

## 3. Technical Decisions

### Call `copy_gpu` instead of patching the condition

The issue proposed replacing the buffer count with `in.is_donatable()` in both sites. The PR goes one step further and deletes the hand-written branch, calling `copy_gpu` the way upstream CUDA's `SliceUpdate::eval_gpu` and upstream's `DynamicSliceUpdate::eval_gpu` do. `copy_gpu` already applies `is_donatable(in, out)`, which checks dtype and size compatibility as well as both reference counts. Keeping a local copy of the rule would leave two places to keep in sync with upstream; removing it makes the overlay's slice update identical to upstream in this respect and shrinks the diff the fork carries.

### Remove the `graph_active()` override

`DynamicSliceUpdate` also donated whenever `rocm::graph_active()` returned true. The comment explained why: under HIP graph capture the async pipeline raises the buffer's use count, so the normal rule would copy into a fresh buffer and each replay of the captured graph would rebuild the cache from the frozen capture input, losing accumulation. The PR removes it for two reasons.

First, it could never fire. `graph_active()` is set only in the `CommandEncoder` constructor when `use_hip_graphs()` is true, `use_hip_graphs()` in `mlx/backend/rocm/device.cpp` returns a constant `false`, and `MLX_GRAPH_PREFILL_REPLAY` is read only behind `use_hip_graphs()`. The override was dead code.

Second, when it did fire it was wrong for the same reason as the main bug, only more so: it donated even a buffer other arrays shared. If HIP graphs come back, accumulation under capture has to be handled in the capture (for example by capturing into a persistent buffer), not by writing into a source other code can still read. Item 24 records this so the next person enabling HIP graphs does not reintroduce the override.

### Keep donation for a dropped source

The fix does not disable donation. The KV caches write with the reassign pattern, `self.keys = slice_update(self.keys, ...)`: once the old handle is replaced, the array has one reference and `is_donatable` holds, so the update still reuses the buffer. That is the case donation exists for and the reason throughput did not change. `dropped_source_update_is_correct` pins it.

### Clamp in the test-only bridge

`slice_update_dynamic` takes its start offsets as array data. MLX does not bounds-check them, and the GPU kernel writes at whatever offset it receives. A test-only function is still reachable from safe Rust, so the bridge clamps each offset to where the update fits, and `dynamic_slice_update_clamps_an_out_of_range_start` checks that an out-of-range offset writes the last slot. Clamping instead of throwing keeps the bridge usable with computed offsets while making sure it cannot write outside `src` if a model path ever starts using it.

### Measure throughput before calling the fix free

The issue suspected the shortcut was added for KV append speed. The PR measured decode throughput on gfx1151 with `scripts/bench_decode.sh` (pp512, tg128). Before and after runs were interleaved, each started after 90 s with no other GPU process or compiler running, with GPU state sampled once a second.

| Model | Before (tok/s) | After (tok/s) |
|---|---|---|
| Qwen3-0.6B-4bit | 279.0, 279.8 | 279.5, 279.9 |
| Meta-Llama-3.1-8B-Instruct-4bit | 35.68, 35.67, 35.82 | 35.90, 35.69 |

No change beyond the run-to-run spread, which is under 1%. One further after run on the 8B model (35.40) overlapped a compiler and was discarded. The result is consistent with the analysis: the decode write goes through `inplace_slice_write`, and the prefill writes use the reassign pattern, so neither path lost its donation.

## 4. Validation

PR author (gfx1151):

- `tests/rocm_slice_update_source.rs`: the four held-source cases fail on the old build; all six pass after.
- `pooling_cache_remainder_survives_an_overlapping_tail_write`, `tiny_model_chunked_prefill_with_pool_remainder_matches_cpu` and `rotating_steady_state_snapshot_survives_the_next_wrap_write` fail on the old build and pass after. `rotating_inplace_warmup_and_wrap_keep_shared_snapshots_intact` fails on the old build with `MLXCEL_KV_INPLACE_WRITE=0`.
- `tests/rocm_slice_update_reduce.rs` without the private source copy: 8/8.
- mlxcel-core `cache` tests: 588 pass, 2 fail (`paged_detach` `debug_assert` in the dev profile, identical on the old build and compiled out in the `test-fast` gate). mlxcel `deepseek_v4`, `kv_snapshot`, `prompt_cache` and `gemma4` tests pass.
- `scripts/ci/rocm_smoke.sh` on Qwen3-0.6B-4bit, clippy with `-D warnings` on both crates, fmt and the fast script gates pass.

Orchestrator verification (gfx1151, branch rebased onto origin/main `0d17303d`):

- The rebase conflicted twice in `LOCAL_FIXES.md`. Both were resolved by keeping item 24 (this PR) before item 25 (#2051's, merged as PR #2073).
- `make verify-rocm` ran every step. Versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, workspace clippy with `--features rocm`, and the ROCm smoke run (32 tokens) passed.
- `verify-test-rocm` failed in exactly three targets, all known baseline failures unrelated to this PR: `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`, and #2037's `gelu_approx_matches_mlx_nn_bit_for_bit` and `tests::family_order_is_exhaustive`.
- `tests/rocm_slice_update_source.rs` passed 6/6. `tests/rocm_slice_update_reduce.rs` passed 8/8 without the private source copy. mlxcel-core lib passed 1832 tests.
- The intermittent `rocm_mxfp4_quant` test tracked in #2072 happened to pass in this run.

## 5. Learning Points

- **A buffer's reference count is not an array's.** In MLX a lazy node that reads an array holds the array, not its data buffer. A donation check that counts only the buffer treats "nobody else holds this memory" as "nobody will read this array again", and those differ exactly when a graph still has unevaluated readers. `array::is_donatable()` checks both, and a backend should reach it through `copy_gpu` rather than reimplement it.
- **Lazy snapshots depend on the donation rule.** `ModelStateTensor::new` does not copy eagerly; it relies on MLX never writing into an array that something still references. Any backend that breaks that rule silently corrupts snapshots, prefix caches and any other "save now, read later" structure.
- **A faster default path can hide a bug in the slow one.** The rotating-cache snapshot bug existed on every ROCm build but was invisible with `MLXCEL_KV_INPLACE_WRITE` at its default, because the in-place write is a different primitive. Running the existing tests with the fast path disabled was what showed the old build was wrong. Tests for cache semantics should cover both write paths.
- **A workaround in a test is a bug report.** #2050's test gave the GPU op a private copy of its source to get a stable reference. That copy was the first evidence of this defect, and removing it turns the reduce test into a second guard for the fix.
- **Check whether a guard can fire before keeping it.** The `graph_active()` override came with a plausible comment, but tracing the flag to a constant `false` showed it was dead. Dead code in a correctness-sensitive primitive tends to come back to life with the feature it was written for, carrying the wrong rule with it.

## 6. Caveats and What Is Not Verified

- **Metal and CUDA** were not run; neither is available on this host. The overlay change affects ROCm builds only. The new DeepSeek-V4 and rotating-snapshot tests are not ROCm-gated and will run on every GPU backend, where they are expected to pass because those backends already use `is_donatable`.
- **Only gfx1151** was run, and throughput was measured on two models. Models whose prefill uses a non-reassign `slice_update` pattern could lose a donation that was previously taken; none was found in the audit, but the benchmark does not cover every family.
- **The paged per-sequence fallback now copies.** Each sequence after the first in `llama3.rs`'s fallback path copies the shared slab. The batched path is the normal route, so this was not benchmarked separately.
- **The audit is a code read.** The list of unaffected paths comes from tracing callers, backed by the existing cache and model tests passing, not from a test per path.
- **HIP graphs remain off.** If `use_hip_graphs()` is turned on, KV accumulation under capture needs its own design; the removed override must not be restored as it was.

## 7. Remaining Work

- #1813: propose item 24 to the fork alongside the other upstreaming candidates.
- #2072: the intermittent `rocm_mxfp4_quant` failure, unrelated to this PR, passed in this run by chance.
- The three baseline `verify-test-rocm` failures (`prefill_dense_gemm_matches_qmm_bytes_where_eligible`, and #2037's two) are tracked outside this PR.

Refs: #2052 (closed by this PR), #1801, #2050, #1813, #2037, #2051, #2072, PR #2073.
