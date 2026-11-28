# Technical Report: PR #2053 - Make the ROCm `slice_update_op_kernel` Grid-Stride

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: HIP C++ (ROCm overlay kernel), C++ and Rust (cxx bridge), Rust (integration test), Markdown

**Risk Level**: Low (one kernel in the ROCm-only overlay, rewritten as a loop around its unchanged body; one test-only bridge function that calls upstream MLX functions present on every backend; no model path reaches the reduce kernel today)

## Executive Summary

Issue #2050 (part of epic #1801) recorded a silent truncation in the ROCm backend's reduce-type `SliceUpdate` (Sum, Prod, Max, Min). `SliceUpdate::eval_gpu` caps its grid at 65535 blocks of 256 threads, and `slice_update_op_kernel` processed exactly one `NWORK`-element chunk per thread without reading `gridDim`. Any update larger than 65535 x 256 x `NWORK` elements (16,776,960, 33,553,920 or 67,107,840 for `NWORK` 1, 2 or 4) left its tail with the input values and raised no error. It is the same failure shape as #1874 and #1823, and it was found, but left unfixed, while PR #2046 deleted the fork's `get_launch_args`: this launch site clamps inline, so it did not go through that helper.

The PR wraps the kernel body in a grid-stride loop over `NWORK`-element chunks and keeps the clamp, matching the other clamped ROCm kernels. Because mlxcel's model code only reaches the None reduce, the PR adds a test-only bridge function, `slice_update_reduce`, and a ROCm integration test that compares each GPU result exactly against the CPU stream on int32 data, just above each of the three limits and through strided and transposed layouts. With the kernel change reverted, every above-limit case fails with a mismatch count equal to the number of elements past the grid's reach; with it, all 8 tests pass.

Writing the test exposed a second, separate defect: ROCm's `SliceUpdate::eval_gpu` donates the source buffer into the output while the source array is still referenced. That defect is filed as #2052 and is not changed here; the test works around it.

## 1. Problem Statement

### The defect

`SliceUpdate::eval_gpu` in `patches-rocm/mlx/backend/rocm/indexing.hip` has two paths. The None reduce (a plain overwrite, which is what mlxcel's KV caches use) goes through `copy_gpu_inplace` and is not affected. The reduce path chooses `nwork` as 4, 2 or 1 by whether the collapsed update's innermost dimension divides by 4 or 2, computes `num_blocks = min(ceil(ceil(update_size / nwork) / 256), 65535)`, and launches `slice_update_op_kernel` on a 1-D grid of that size.

The kernel computed one start index per thread, `(blockIdx.x * blockDim.x + threadIdx.x) * NWORK`, applied at most `NWORK` elements from there, and returned. Nothing in it looked at `gridDim`, so the clamp simply dropped every chunk past the 16,776,960th. The output past that point kept the source values, which is also what a correct Max or Min would produce for some inputs, so the defect is invisible unless the update actually changes the tail.

### Who reaches it

The reduce variants are built by MLX's `slice_update_add`, `slice_update_prod`, `slice_update_max` and `slice_update_min`, and by `SliceUpdate::vjp` for Prod. mlxcel's Rust bridge exposed only the None-reduce `slice_update` before this PR, so no current model or server path is known to reach the kernel. The defect is latent in the vendored backend, and an upstreaming candidate for the ROCm fork (#1813). It still matters because a future op, a training or gradient path, or a fork sync that routes something through `slice_update_*` would lose data silently at sizes that are ordinary for large KV caches or embeddings.

### How upstream avoids it

Upstream CUDA at the MLX pin 81ba1c6a launches the same op through `get_launch_args(upd, large, nwork)`, which does not cap x and spills into a 2-D grid when `large`, and indexes with `cg::this_grid().thread_rank()`. The ROCm port kept the kernel's one-chunk-per-thread shape but added a 65535-block clamp that belongs to the grid-stride convention used elsewhere in the overlay, and the combination is what broke.

## 2. Change Summary

- **`indexing.hip`, `slice_update_op_kernel`.** The body now runs inside `for (base = tid * NWORK; base < update_size; base += gridDim.x * blockDim.x * NWORK)`. At the top of each iteration it recomputes `out_idx` and `update_idx` from `base`: directly when the side is row-contiguous, through `elem_to_loc` when it is not, and 0 for a scalar update. The inner `NWORK` loop with its per-element stride increments is unchanged. The only code that moved is the index prologue, which went from before the loop to inside it.
- **`indexing.hip`, launch site.** The 65535 clamp is kept; a comment now states that it is safe only because the kernel is grid-stride and points to `LOCAL_FIXES.md` item 21.
- **Bridge: `slice_update_reduce`** in `mlx_cxx_bridge.{h,cpp}` and the `ffi` block of `mlxcel-core/src/lib.rs`. `reduce` 0, 1, 2, 3 call `slice_update_add`, `slice_update_prod`, `slice_update_max`, `slice_update_min`; any other value throws `std::invalid_argument`, which cxx turns into a Rust `Err` naming the valid range. It returns `Result<UniquePtr<MlxArray>>`. It is documented as test-only. The four MLX functions exist at the upstream pin as well as in the ROCm overlay, so the bridge compiles on Metal and CUDA too.
- **`tests/rocm_slice_update_reduce.rs` (new, 364 lines, `#![cfg(feature = "rocm")]`).** Skips unless `gpu_backend_kind()` is ROCm. Each case runs the op on the CPU stream first, then on the GPU against a private copy of the source, and compares with `array_equal`; on a mismatch it reports the count and the first differing flat index with both values. Cases:
  - contiguous Sum at 16,777,217 (NWORK=1), 33,554,434 (NWORK=2) and 67,108,868 (NWORK=4) elements, written at offset 3 of a slightly larger source (a full-size slice would take MLX's elementwise shortcut and never reach `SliceUpdate`);
  - Max through a `[4099, 4097]` update into columns 1..4098 of a `[4099, 4099]` output (strided output, NWORK=1, 16,793,603 elements);
  - Sum through a transposed update of the same shape (strided output and strided update, so both indices come from `elem_to_loc` on every chunk);
  - Sum with a scalar update over 16,777,217 elements;
  - every reduce op on a strided 2-D slice below the limit at widths 8, 6 and 5 (NWORK 4, 2, 1), with values chosen so Prod stays small and Max and Min keep a mix of source and update values;
  - an out-of-range `reduce` code, which must be an error naming the range.
  Each test holds `streams::lock_default_device` for its whole body, because building the CPU reference moves the process-global default device.
- **`LOCAL_FIXES.md`.** New item 21 records the defect, the fix, the measurement and the #2052 workaround, marked "Applies to the fork; to be proposed there". Item 19 (the `get_launch_args` removal) no longer describes this site as unfixed and points to item 21.

Commit history: `49c839fd` is the fix, bridge, test and `LOCAL_FIXES.md` entry; `edb1ee47` (from review) widens the device lock from the comparison to the whole test body and adds the transposed-update case.

## 3. Technical Decisions

### Grid-stride loop, clamp kept

The issue weighed removing the clamp instead. That alone only moves the cliff: `num_blocks` is an `int`, AMD caps each grid dimension at 2^32 - 1 threads (`LOCAL_FIXES.md` item 11), and spilling into y would also need the kernel to read `blockIdx.y`. A grid-stride loop is correct for every size with one change, and it is the convention the overlay already uses for clamped launches (`sort.hip`, `hadamard.hip`, `binary.hip`, `unary.hip`), so a reader who sees a 65535 clamp in this overlay can expect a grid-stride kernel behind it.

### Recompute indices per chunk rather than carry them

The non-contiguous branches need `elem_to_loc`, a loop over dimensions with a division per dimension. Carrying the index forward across a stride of `gridDim.x * blockDim.x * NWORK` elements would need a multi-dimensional increment, which is exactly what `elem_to_loc` computes more simply. Below the limit each thread still runs one iteration, so the cost for sizes that already worked is the same prologue it paid before.

### Why a chunk never crosses a row

The inner loop advances `out_idx` and `update_idx` by the innermost stride, which is valid only while it stays inside one innermost row. Each chunk starts at a multiple of `NWORK`, the stride is a multiple of `NWORK`, and the launch site picks `NWORK` only when it divides the innermost dimension. So every chunk starts on an `NWORK` boundary within a row and ends before the row does. The old kernel relied on the same argument for its single chunk; the loop keeps it true for every chunk.

### No new template instantiations

`IdxT` is `int64_t` at every existing instantiation, so `base` and the stride cannot overflow for any size the allocator can hold, and no wider index type is needed. `LOCAL_FIXES.md` item 17 records how expensive this translation unit is to compile, so the fix changes only the kernel body.

### A test-only bridge instead of a production path

The only way to test the reduce kernel from Rust is to reach it. Adding `slice_update_reduce` to the bridge with a numeric `reduce` code is the smallest surface that does, and returning `Result` makes an invalid code an error rather than a silent fall-through to some other op. It is documented as test-only so nobody mistakes it for a supported API; a production caller would want an enum.

### Exact comparison against the CPU stream on int32

int32 makes Sum, Prod, Max and Min exact on both devices, so `array_equal` is the right check and any mismatch is a real defect rather than rounding. The mismatch count doubles as a measurement: with the old kernel it equals the number of elements past `65535 x 256 x NWORK`, which ties each failure to the clamp and not to some other bug.

## 4. Validation

Author's runs (gfx1151, Radeon 8060S):

- With only `indexing.hip` reverted to `main`, `cargo test --features rocm --test rocm_slice_update_reduce -- --test-threads=1`: the six above-limit cases fail and the below-limit and error cases pass (the transposed-update case, added in review, was checked the same way in a separate reverted build). The mismatch counts are exactly the elements past the grid's reach: 257, 514 and 1028 for the three contiguous Sums, 257 for the scalar Sum, and 16,643 for the strided Max and the transposed-update Sum. For the NWORK=1 contiguous Sum the first mismatch is at flat index 16,776,963, which is the limit plus the slice offset of 3.
- With the change: 8 passed, 0 failed (about 155 s for the debug build).
- `cargo clippy --features rocm --test rocm_slice_update_reduce -- -D warnings` and `cargo clippy -p mlxcel-core --features rocm --lib -- -D warnings`: clean.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt` and `cargo test --features rocm --test dead_doc_pointers`: pass.

Orchestrator verification (gfx1151, branch on origin/main `d913fc7d`):

- `make verify-rocm` ran every step. Versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, workspace clippy with `--features rocm`, and the ROCm smoke (32 tokens) passed.
- `verify-test-rocm` failed in three targets with exactly the 37 known baseline failures and nothing else:
  - `-p mlxcel-core --lib`: 35 failures. 34 are fused paged-attention tests without a ROCm port (#1814); the other is the bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible`.
  - `gemma3_backbone_tests::gelu_approx_matches_mlx_nn_bit_for_bit` and `tests::family_order_is_exhaustive`, both from #2037.
- `tests/rocm_slice_update_reduce.rs` ran inside the full suite in the test-fast profile: 8 passed in 63.4 s.

## 5. Learning Points

- **A clamp is a contract with the kernel.** A 65535-block clamp is only correct in front of a grid-stride kernel, and nothing in the type system says so. #2046 removed the helper that hid this contract, but inline clamps can break it just as well. The launch-site comment added here states the contract where the next person will read it; any new clamped launch in the overlay should be checked the same way.
- **Make the mismatch count part of the evidence.** Reporting how many elements differ, not just that they differ, turned "the test fails without the fix" into "the test fails by exactly the number of elements the clamp drops". That rules out coincidental failures and shows which geometry each case exercises.
- **Test layouts, not just sizes.** The contiguous cases only exercise the row-contiguous index path. The strided Max covers `elem_to_loc` on the output side, and the transposed-update Sum, added in review, is the only case that reaches the update-side `elem_to_loc` recomputation. A grid-stride rewrite that got the update index wrong on later chunks would pass every other case.
- **The first test version failed for a different reason: source donation (#2052).** The first draft used the same `src` array for the GPU op and then for the CPU reference. Every case failed, including those below the limit: for example `src[3] = 3` with update 1 gave GPU 4 and CPU 5, and a below-limit Prod applied the product twice. The GPU op had written its result into `src`'s buffer. ROCm's `SliceUpdate::eval_gpu` donates the source (`out.copy_shared_buffer(in)`) when `in.data_shared_ptr().use_count() == 1` and the source is contiguous, which counts only the owners of the data buffer, not whether the `array` itself is still referenced by the caller. Upstream decides with `array::is_donatable()`, which requires both `array_desc_.use_count() == 1` and a single data owner, and upstream CUDA's `SliceUpdate::eval_gpu` goes through `copy_gpu` and never donates a live source. `DynamicSliceUpdate::eval_gpu` in `backend/gpu/primitives.cpp` has the same data-only check, plus a deliberate forced donation during HIP graph capture. The lesson for testing is that a GPU-versus-CPU comparison on the ROCm overlay must not share inputs with an op that may donate: the test now gives the GPU op a private copy and runs the CPU reference first, with a comment pointing at #2052. The lesson for the backend is that "one owner of the buffer" is not "nobody else can see this array".
- **Run the reference on the side that cannot mutate.** Ordering the CPU reference before the GPU op, in addition to the private copy, means that even if a future change made the GPU op mutate its inputs in some other way, the reference would already have been computed from clean data.

## 6. What Is Not Verified

- **Metal and CUDA.** Not available on this host. The only change on those paths is the `slice_update_reduce` bridge function, which calls upstream MLX functions present at the pin; the kernel change is ROCm-only and the test is `#![cfg(feature = "rocm")]`.
- **Other AMD GPUs and ROCm versions.** Only gfx1151 was measured. The fix is geometric and does not depend on the architecture, but no other device ran it.
- **Performance.** No benchmark was run. Below the limit each thread runs one loop iteration with the same prologue as before, so no change is expected; above the limit the old kernel was incorrect, so there is no baseline to compare against.
- **Float dtypes and the `SliceUpdate::vjp` Prod path.** The test uses int32 for exact comparison and calls `slice_update_*` directly. The kernel is templated on the element type and the loop does not depend on it, but float dtypes and the vjp caller were not exercised.
- **A standalone release build.** `cargo build --release --features rocm` was not run separately; `make verify-rocm` covers the release build.

## 7. Remaining Work

- **#2052, source donation in `SliceUpdate` and `DynamicSliceUpdate`.** Replace the data-only use-count check with `in.is_donatable()` at both sites, keeping the `graph_active()` override in `DynamicSliceUpdate` if it is still needed. The None-reduce path is the one mlxcel's KV caches use, so the fix must first establish whether any mlxcel path reads a pre-update array after a `slice_update` (KV cache snapshots, prefix cache, speculative decoding rollback): if one does, #2052 is a live correctness bug, not a latent one. The shortcut was likely added for KV cache append speed, so decode throughput on gfx1151 must be measured before and after. Its acceptance test should hold `src`, run a GPU `slice_update` (None and one reduce type), and assert `src` is unchanged. Once it lands, the private copy in `tests/rocm_slice_update_reduce.rs` can be dropped.
- Propose `LOCAL_FIXES.md` item 21 to the ROCm fork as part of #1813.
- Baseline test failures unrelated to this PR: #1814 (34 fused paged-attention ports), the bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible` failure, and the two #2037 failures.

Refs: #2050 (closed by this PR), #1801, #1813, #1823, #1874, #2046, #2052.
