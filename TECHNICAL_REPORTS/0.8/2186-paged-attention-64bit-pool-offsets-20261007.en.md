# Technical Report: PR #2186 - Use 64-Bit Pool Offsets in the Fused Paged-Attention Kernels

**Date**: 2026-10-07

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); head `c3fef346` (up to date with origin/main `9c0ae2e9`), PR open, pending merge. Closes #2153.

**Languages**: C++ (embedded Metal, CUDA and HIP kernel sources in `turbo/paged_attention*.cpp` and `paged_attention_hip.h`, the merge launcher), Rust (mlxcel-core `paged_v2/plan.rs`, new `cache/paged_pool_offset_tests.rs`, `plan_tests.rs`), Markdown (`docs/environment-variables.md`)

**Risk Level**: Low. Each kernel body changes one declaration: the pool base becomes 64-bit with the row widened before the multiply. Template args, grid shape, JIT names and dtype keys are unchanged. The two new refusals (merge launcher and plan validation) apply only to sizes far above anything the partial workspace reaches in practice, and both send the batch to the existing gather path.

## Executive Summary

The fused paged-attention kernels computed the per-token K/V pool offset, `(row * page + slot) * stride_kv + kv_head * dim`, in 32-bit unsigned arithmetic in all six bodies: v1 decode and v2 partial, each for Metal, CUDA and HIP. The pool they address is one layer's slab per side, and the server sizes that slab as `ceil(ctx / 32) * batch` blocks. Once a slab holds more than 2^32 elements, the offset wraps and the kernel reads another row of the same buffer. There is no fault and no error, only wrong attention output. For Llama-3.1-8B (8 KV heads, head dim 128) the limit is 131,072 blocks of `[32, 8, 128]`, which a 128K context at `--parallel 32` reaches exactly; anything larger wraps.

PR #2186 widens the base in each body to `ulong` (Metal) or `uint64_t` (CUDA, HIP), casting `row` before the multiply. Sparse decode runs on v2 partial, so it gets the fix with no change of its own. The merge kernel, which also indexes in 32 bits, is not widened; its launcher now refuses inputs or outputs past `UINT32_MAX` elements, and `PagedDecodePlan::validate` rejects such plans before launch so the batched path falls back to gather instead of panicking.

A GPU test builds a real f16 pool of 131,073 blocks (2^32 + 32,768 elements), writes K/V only into the last block, and compares v1 and v2 against host attention. With main's bodies both kernels read row 0 (zeros) and the test fails with max error 0.456; with the PR it passes. A source pin test fails when any one body is reverted. Server decode on gfx1151 is unchanged at a 14.0 tok/s median on both arms. The Metal and CUDA lines were reviewed but not compiled or run. Writing the test surfaced two separate defects, filed as #2184 and #2187.

## 1. Problem Statement

### 1.1 Where the offset wrapped

Every fused body computes the start of one token's K/V vector inside the pool:

```
uint base = (row * block_size + slot) * stride_kv + kv_head * dim;   // v1, Metal
uint32_t base = (row * page_size + entry) * stride_kv + kv_head * dim; // v2, CUDA/HIP
```

`row` is the pool row (block) index, `stride_kv` is `Hkv * D`, and the reads are `k_pool[base + d]` and `v_pool[base + d]`. All terms were 32-bit, so the product wrapped modulo 2^32 before indexing. The six affected lines:

| Backend | v1 decode | v2 partial |
|---|---|---|
| Metal | `paged_attention.cpp` (`PAGED_ATTENTION_DECODE_SOURCE`) | `paged_attention_v2.cpp` (`PAGED_ATTENTION_V2_PARTIAL_SOURCE`) |
| CUDA | `paged_attention.cpp` (`..._CUDA_SOURCE`) | `paged_attention_v2.cpp` (`..._CUDA_SOURCE`) |
| HIP | `paged_attention_hip.h` (`PAGED_ATTENTION_DECODE_HIP_SOURCE`) | `paged_attention_hip.h` (`PAGED_ATTENTION_V2_PARTIAL_HIP_SOURCE`) |

The HIP bodies were introduced by #2103, which copied the CUDA arithmetic unchanged; the defect was found during that review.

### 1.2 Why a production server can reach it

The limit is per layer slab, not per model. `resolve_paged_slab_blocks` sizes a slab as `ceil(per_slot_ctx / block_size) * batch`, capped only by the per-layer share of the block budget when one is set, and `MLXCEL_PAGED_SLAB_BLOCKS` is used verbatim. The wrap point is `slab_blocks * block_size * Hkv * D > 2^32`. For Llama-3.1-8B that is 131,072 blocks of 32 tokens, about 8 GiB per side per layer at f16. A large-memory host serving long contexts at high `--parallel` crosses it, and no launcher check looked at the element count: the host checks validated rank and axes 1 to 3 only.

When it wraps, a token in row `r` is read from row `r - 131072` (for this geometry). The output is a valid-looking attention result over another request's keys and values, or over zeros. Nothing in the logs distinguishes it.

### 1.3 Sparse decode and merge

- **Sparse decode** reshapes the dense `[B, H, Cap, D]` allocation into a `[B*H*Cap, 1, 1, D]` pool and launches v2 partial with page size 1, so it wraps at `B * H * Cap * D > 2^32`. It needs no separate kernel change; fixing v2 partial fixes it.
- **Merge** (`paged_attention_v2_merge.cpp` and the HIP merge body) indexes `v_in[(i * heads + h) * dim + d]` in 32 bits. Its input is the partial workspace `[num_chunks, Hq, D]` in f32, which is far smaller than the pool, but nothing guarded it.

## 2. Change Summary

| Area | Change |
|---|---|
| Six fused bodies (Metal, CUDA, HIP; v1 and v2 partial) | `T base = ((T)row * page + slot) * (T)stride_kv + kv_head * dim;` with `T` = `ulong` or `uint64_t`, plus a one-line comment citing #2153 |
| `paged_attention_v2_merge.cpp` | Launcher throws `std::invalid_argument` naming `paged_attention_merge_states` when `v_in.size()` or `num_outputs * heads * dim` exceeds `UINT32_MAX`; adds `<cstdint>` |
| `paged_v2/plan.rs` | `PagedDecodePlan::validate` returns `Err` when `workspace_partial_v_elems()` exceeds `u32::MAX` |
| `paged_v2/plan_tests.rs` | `validate_rejects_a_merge_input_past_the_u32_index_range`: at 65,535 one-page chunks, `Hq * D = 64 * 1024` passes and `128 * 1024` is declined |
| `cache/paged_pool_offset_tests.rs` (new) | Source pin test, merge refusal test, ignored 2^32 pool test |
| `cache/paged_batch_decode.rs` | Registers the new test module |
| `docs/environment-variables.md` | `MLXCEL_PAGED_SLAB_BLOCKS` row notes that slabs past 2^32 elements per side are supported (#2153) |

Two commits: the fix and tests (`e1184cf2`) and a review follow-up that tightens comments and docs only (`c3fef346`). 9 files, 388 insertions and 7 deletions.

## 3. Design

### 3.1 Widen the address, not every index

The issue chose to widen only the pool base. Loop counters, `block_idx`, `slot`, `row` and the `rows`/`indices` inputs stay 32-bit: row counts are far below 2^31, and only the final element offset exceeds 32 bits. The cast is on `row` before the multiply:

```
ulong base = ((ulong)row * block_size + slot) * (ulong)stride_kv + kv_head * dim;
```

`(ulong)(row * block_size)` would still wrap, because the multiply would happen in 32 bits first. Casting `stride_kv` as well keeps the second multiply in 64 bits regardless of how the compiler orders promotions. The reads `k_pool[base + d]` and `v_pool[base + d]` are unchanged and now use the 64-bit value.

The same arithmetic already used `ulong` and `uint64_t` in `fused_norm.cpp` and `fused_rope_append.cpp`, so this brings the paged bodies in line with kernels that write the same pool.

### 3.2 Template args and dtype keys untouched

The source text changes, but no template argument, grid shape, JIT name or dtype key does. `make verify-kernel-dtype-keys` still reports 9 in scope. Under the source-keyed JIT caches (CUDA upstream, ROCm since #2181) the new source compiles to a new module automatically; on Metal the source is part of the library build. The three backends stay arithmetically identical apart from the type spelling, as the issue required.

### 3.3 Refuse at merge instead of widening it

The merge input is `num_chunks * Hq * D` f32 elements. Reaching 2^32 would need, for example, 65,535 chunks at `Hq * D` above 65,536, which no current model and plan produce. Widening the merge bodies would touch three more kernels for a size that does not occur, so the PR adds a host-side refusal next to the existing checks in `paged_attention_merge_states`. It throws `std::invalid_argument`, which reaches Rust as an `Err` through the existing `Result` bridge. The check reads only shapes, so it runs before any launch.

### 3.4 The plan check, and why it deviates from the issue

The issue assumed callers treat a refused launch as "fall back to gather". That holds for sparse decode, which reports a declined launch as an outcome, but not for the batched path: `paged_batch_decode_attention` in `cache/paged_batch_decode.rs` panics on an `Err` from `PagedBlockPool::paged_decode_batched`, on the reasoning that an error there means inconsistent bookkeeping. A merge refusal surfacing as `Err` would therefore crash the server rather than degrade.

The PR closes that gap one level up. `PagedDecodePlan::validate` now returns `Err` when the partial workspace exceeds `u32::MAX` elements. The pool's decode path in `cache/paged.rs` calls `validate` before launching and records its `Err` as `PlanRejected` with no launch (`Ok(None)`), which the batched caller serves through `gather_fallback`, so an oversize batch is routed to gather before any kernel runs, and the sparse path's own `validate` call declines the same way. The merge launcher check stays as the last line of defense for any caller that bypasses the plan.

### 3.5 Rejected: refusing large pools instead of widening

A host-side refusal for the pool kernels would have been smaller, but it would send exactly the long-context, large-batch shapes the fused path exists for back to gather. One 64-bit multiply per token per lane in a memory-bound kernel costs nothing measurable (section 4.3).

## 4. Verification

### 4.1 The real 2^32 pool test

`paged_pool_past_u32_elements_matches_gather` (`#[ignore]`, about 17 GiB) builds an f16 pool `[131073, 32, 8, 128]`: 2^32 + 32,768 elements, so every element of the last block lies past the wrap. Only the last block holds data; everything else is zero. The test first confirms the pool itself is correct: the last block reads back the values written, and row 0 is all zeros. It then runs v1 decode and v2 partial (one chunk, no merge) over a single sequence whose only block is that last row and compares both with attention computed on the host in f64 from the same f16-rounded values, with tolerance 1e-3.

- **Main's bodies**: both kernels fail with max error 0.456. Row 131,072's base is `131072 * 32 * 1024 = 2^32`, which wraps to 0, so the kernels read row 0 and return zeros; 0.456 is the largest reference value.
- **This PR**: both pass. Runs were inside the `rocm_gpu_guard.sh` window.

The pool is built from two zero halves plus the block rather than one zeros tensor, because a ROCm strided copy of 2^32 or more elements fails to launch (#2184, below). The test avoids that limit so it measures the kernels under test.

### 4.2 Fast tests

- `every_fused_body_computes_the_pool_base_in_64_bits` reads the three source files with `include_str!`, finds every `<type> base = ...;` declaration, expects exactly two per file, and asserts each one's type and full expression, including the cast on `row` before the multiply. It runs on any host, which matters because no host can run all three backends. With only the HIP v2 body reverted it fails and names that file.
- `merge_refuses_inputs_or_outputs_past_u32_elements` passes unevaluated broadcasts of 2^32 + 1,024 elements (input case) and an oversized output count, so nothing large is allocated, and checks both `Err` messages.
- `validate_rejects_a_merge_input_past_the_u32_index_range` checks the plan boundary on either side.
- `cargo test -p mlxcel-core --features rocm --lib -- paged`: 264 passed. Clippy on lib and tests clean; script gates and `verify-rocm-overlay` pass.

### 4.3 Decode performance

Llama-3.1-8B-Instruct-4bit, server with `--parallel 4 --ctx-size 131072`, 4 clients at about 16K prompt tokens, 128 output tokens, fused v2 on both arms (the log shows `fused v2 launch`). A build with main's kernel bodies (this branch with the three kernel files reverted) and this PR were run as a pair inside one `rocm_gpu_guard.sh` window, three times, all clean. Per-request decode tok/s: 14.0, 13.9, 14.0 before and 14.1, 14.0, 14.0 after. Both medians are 14.0, so there is no regression.

The issue's acceptance criterion named `examples/paged_attention_kernel_bench`. It was not used because it keeps the 32-block default slab, so at batch 4 and 16K tokens it measures the gather path rather than the fused kernels. The server run exercises the changed kernels on a real slab.

### 4.4 Orchestrator gate

The orchestrator's `make verify-rocm` on head `c3fef346` (up to date with origin/main `9c0ae2e9`) passed every step except `verify-test-rocm`, which ran 11,984 passed, 5 failed and 383 ignored. The 5 failures are exactly the known #2182 tests tracked in #2187:

- `scheduler_completion_snapshot_tests::model_owned_sequence_primes_the_decode_lookahead_only_with_a_rewind`
- the four `scheduler_model_owned_lookahead_tests`

They fail identically with this PR's files reverted to main, and pass when run alone, so they are unrelated to this change.

### 4.5 Not verified

Metal and CUDA are not available on the host. Their four changed lines (Metal v1 and v2 partial, CUDA v1 and v2 partial) were reviewed only, not compiled or run. The source pin test confirms their text but not that they compile or produce correct results. The 2^32 pool test should be run on a Metal host with at least 20 GiB of unified memory and on a CUDA device before relying on those backends at this scale.

## 5. Technical Decisions

- **Widen one expression per body.** The defect is in the address computation only; widening everything would add cost and churn for indices that never approach 2^31.
- **Cast before the multiply.** The pin test asserts the exact form `((T)row * ...)`, because a cast applied after the multiply looks correct and still wraps.
- **Keep JIT keys and template args stable.** The change is invisible to dispatch, autotuning and `verify-kernel-dtype-keys`.
- **Refuse at merge rather than widen.** The merge workspace cannot realistically reach the limit; a guard costs one comparison and avoids three more kernel edits.
- **Decline at plan validation.** The batched path panics on a launch error, so refusing inside the launcher alone would turn an oversize batch into a crash. Declining in `validate` keeps the outcome a gather fallback.
- **Test with a real oversized pool.** A synthetic or mocked check cannot show that a 64-bit index reaches the right memory. The test pays 17 GiB once, behind `#[ignore]`, to prove the address.
- **Pin the source on every host.** The pin test is the only check that covers all six bodies, including the four this host cannot run.
- **Measure through the server, not the kernel bench.** The bench's default slab would have measured gather.

## 6. Residual Risks and Follow-ups

- **Metal and CUDA unexecuted.** See 4.5. A typo that still matches the pin pattern is unlikely, but compile and runtime behavior at scale are unconfirmed.
- **ROCm strided copy past 2^32 elements (#2184).** While building the test pool, a strided MLX copy of 2^32 or more elements on ROCm requested a 2^32-thread grid, which HIP refuses with `hipErrorInvalidConfiguration`. This is a limit of MLX's ROCm copy kernels, not of the paged kernels, but it means other code paths that copy whole oversize slabs on ROCm can still fail. Filed as #2184.
- **Gemma 3 lookahead test order dependence (#2187).** The 5 `server::batch::scheduler` failures in the full ROCm test run come from #2182, fail the same way on main, and pass in isolation. Filed as #2187.
- **Merge stays 32-bit.** If a future plan or model pushes `num_chunks * Hq * D` past 2^32, those batches go to gather rather than failing. That is a performance cliff, not a correctness one; widening the merge bodies would remove it.
- **Other 32-bit indexing in the paged family.** This PR covers the six fused bodies named in the issue and the merge. Kernels outside that list were not audited here.

## 7. Learning Points

- **Silent wrap is the worst failure mode for an index.** The kernel returned plausible numbers from the wrong row. Only a test that places data past the wrap and checks the result against an independent reference can see it.
- **The limit is per buffer the kernel addresses, not per model.** The slab size depends on `--ctx-size` and `--parallel`, so a model that is small by parameters can still cross 2^32 elements in one layer's pool.
- **Check that the fallback a guard relies on really exists.** The issue assumed every caller treats a launch `Err` as fallback; the batched path panics. Moving the decline into plan validation kept the guard from creating a crash.
- **Confirm the fixture before blaming the kernel.** The pool test asserts that the last block and row 0 hold what it expects before it runs the kernels, which is how it also found #2184 rather than misreading it as a kernel failure.
- **Use a benchmark that actually runs the changed path.** The named kernel bench would have produced a clean but irrelevant number.
