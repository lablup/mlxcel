# Technical Report: PR #2237 - Strided Copies Past 2^32 Elements on ROCm

**Date**: 2026-10-08

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); code head `e5f4419c` on main `2aa5211f`, PR open, pending merge. Closes #2184.

**Languages**: C++/HIP (ROCm overlay `copy/*.hip`, `copy/copy.hpp`), Rust (new `tests/rocm_strided_copy_grid.rs`), Markdown (`patches-rocm/LOCAL_FIXES.md`)

**Risk Level**: Low. Below 65535 blocks of 256 threads (16.7M elements) each copy kernel instantiation is the old one-thread-per-element body, and a 4M-element transpose measured within noise of main. The behaviour change is for copies past that cap, which loop, and for dynamic copies, whose launch now covers the copied region instead of the whole destination. `copy_contiguous` changes its index width above `INT32_MAX`. The four general copy kernels now have twice the instantiations. Metal and CUDA code paths are untouched.

## Executive Summary

Three ROCm strided copy launchers sized their grid as one thread per element with an `int` block count and no loop. HIP rejects a launch whose total thread count reaches 2^32 (`hipErrorInvalidConfiguration`, "invalid configuration argument"), so any strided copy of 2^32 elements or more failed. `copy_contiguous.hip` already capped its grid at 65535 blocks and looped. #2184 found this while writing the #2153 paged-slab test, which worked around it by concatenating two 2^31-element halves.

A second bug sat in `copy_general_dynamic`: it sized its launch by `out.size()`. `DynamicSliceUpdate` passes the whole destination as `out`, so a one-block update into a 2^32-element slab launched 2^32 threads and failed, and below that size every thread past the update rewrote a wrapped copy of it.

The fix adds `rocm::copy_grid_blocks` (a 64-bit block count clamped to [1, 65535]) and `copy_grid_loops`, and a device helper `for_each_copy_index<kLoop>`. Each of the four general copy kernels takes `kLoop` as a template argument: below the cap the instantiation keeps the old body, past it the threads stride on a 64-bit counter. The dynamic launch is sized by the product of the copied shape. `copy_contiguous` uses the same helper and switches to 64-bit indices above `INT32_MAX`.

On gfx1151 the four `#[ignore]` tests (8 GiB each) fail on main with invalid configuration and pass with the fix. Three fast tests of 42.6M elements pass and fail when the loop is cut to one pass. Microbenchmarks of f16 transposed `contiguous` and `concatenate` are within 0.4% of main. The orchestrator's `make verify-rocm` on `e5f4419c` passes (163 suites, 12,194 passed, 0 failed, 403 ignored, smoke OK). Metal and CUDA were not verified.

## 1. Problem Statement

### 1.1 The uncapped grids

Three launchers computed `int num_blocks = (size + block_size - 1) / block_size` with a block size of 256 and launched one thread per element:

- `copy_general_input.hip` (`copy_g_byval`, strided input to contiguous output),
- `copy_general.hip` (`copy_gg_byval`, strided to strided),
- `copy_general_dynamic.hip` (`copy_gg_dynamic_nd` and `copy_gg_dynamic`, the `DynamicSliceUpdate` path).

HIP refuses a launch whose `gridDim.x * blockDim.x` reaches 2^32, so a copy of 2^32 elements launched a grid of `{16777216, 1, 1}` by `{256, 1, 1}` and failed (reproduced with `AMD_LOG_LEVEL=3` on a `concatenate` of an f16 `[131072, 32, 8, 128]` array with one more block, at main `b7116d1d`). The `int` block count would also overflow past 2^39 elements. `copy_contiguous.hip` instead clamps the grid to 65535 blocks and walks the data in a loop, so it never hit the limit on the grid.

### 1.2 The dynamic copy sized by the destination

`copy_general_dynamic` set `size = out.size()`. For `DynamicSliceUpdate`, `out` is the whole destination and `shape` is the update's shape. The kernel decomposes the thread index with the update's shape, so each thread past the update's element count recomputed an index modulo that shape and rewrote a wrapped copy of the update. That is wasted work at every size. At a destination of 2^32 elements it launched 2^32 threads and failed, even for a one-block update, which is the paged KV cache write pattern. Upstream CUDA at pin `81ba1c6a` launches `copy_general_dynamic` over `out.size()` the same way; whether its launch limits are reached there is not known.

### 1.3 Impact

A paged KV slab past 2^32 elements per side (supported after #2153) could hit this on any op that strided-copies the whole slab. The issue found no production path that does so today, and that was not verified exhaustively.

## 2. Change Summary

| Area | Change |
|---|---|
| `copy/copy.hpp` | New `kMaxCopyBlocks` (65535), `copy_grid_blocks`, `copy_grid_loops` and `for_each_copy_index<kLoop>`. |
| `copy/copy_general_input.hip`, `copy_general.hip` | `copy_g_byval` and `copy_gg_byval` take `bool kLoop` and run their body through `for_each_copy_index`. Launch uses the clamped grid and picks the instantiation from `copy_grid_loops`. |
| `copy/copy_general_dynamic.hip` | Same for `copy_gg_dynamic_nd` and `copy_gg_dynamic`. The launch size is the product of `shape`, and an empty copy returns early. |
| `copy/copy_contiguous.hip` | Block count from `copy_grid_blocks`; 64-bit indices above `INT32_MAX` instead of `UINT32_MAX`. |
| `tests/rocm_strided_copy_grid.rs` | New: three fast tests and four `#[ignore]` tests (section 5). |
| `patches-rocm/LOCAL_FIXES.md` | Item 42 (new). |

One commit, 7 files, 483 insertions, 82 deletions.

## 3. Design

### 3.1 Helpers

```cpp
inline constexpr size_t kMaxCopyBlocks = 65535;

inline uint32_t copy_grid_blocks(size_t size, size_t per_block) {
  size_t blocks = (size + per_block - 1) / per_block;
  return static_cast<uint32_t>(std::min(std::max(blocks, size_t{1}), kMaxCopyBlocks));
}

inline bool copy_grid_loops(size_t size, size_t per_block) {
  return size > kMaxCopyBlocks * per_block;
}
```

The block count is computed in `size_t` and clamped, so it cannot overflow and stays under the 2^32 thread limit (65535 * 256 threads). `copy_grid_loops` is true exactly when one pass of the capped grid does not cover `size`.

### 3.2 The kLoop split

```cpp
template <bool kLoop, typename F>
__device__ __forceinline__ void for_each_copy_index(int64_t size, F&& body) {
  if constexpr (!kLoop) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) body(int64_t(index));
  } else {
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += stride)
      body(i);
  }
}
```

With `kLoop` false the thread index fits in 32 bits (at most 65535 * 256 threads) and the instantiation has no loop, which is the old body. With `kLoop` true the counter is `int64_t`, so it cannot wrap a 32-bit `IdxT`. The index arithmetic inside each kernel stays in the kernel's own `IdxT`. The host picks the instantiation with `loop ? <..., true> : <..., false>` at the launch. The split doubles the instantiations of the four kernels.

### 3.3 Rejected designs

Both were measured on a 4M-element transpose on gfx1151 against the old kernel:

- **A loop for every size.** 1.3% slower.
- **One kernel with both paths behind a runtime branch.** 1.2 to 1.7% slower.

Typical copies are below the cap, so the template split keeps them unchanged at the cost of compile time and code size.

### 3.4 Sizing the dynamic launch

`copy_general_dynamic` now multiplies out `shape` (the copied region) and launches over that. A zero-size copy returns before any launch, which the old code did not special-case. This removes the wrapped rewrites at every size, not only past 2^32.

### 3.5 copy_contiguous

`copy_contiguous` already capped and looped. Reading it showed a separate hazard: its loop counter is `uint32_t` and adds `stride * N_READS` (up to 2^26) per pass, so for a size just under 2^32 the counter could wrap before it reached `size`. It now takes the 64-bit index instantiation above `INT32_MAX` instead of `UINT32_MAX`, and uses `copy_grid_blocks` for the block count. This was found by reading the code and was not reproduced.

## 4. Risks Specific to the Change

- Past the cap the kernels now loop, so copies of 16.7M elements or more run a grid-stride loop where a launch would otherwise be a single pass. The 33.5M-element case measured the same within noise (section 6.2).
- Doubled instantiations of four kernels increase compile time and binary size.
- `copy_contiguous` of sizes between `INT32_MAX` and `UINT32_MAX` now uses 64-bit index math, which is marginally more expensive. It was not benchmarked.
- The dynamic launch is smaller than before for `DynamicSliceUpdate`. A kernel that relied on the extra threads would break; none does, since the extra threads only rewrote wrapped copies.

## 5. Verification

### 5.1 Tests

`tests/rocm_strided_copy_grid.rs` builds inputs as `as_strided` views of a small buffer with every stride 1, so element `[r, j, k, l]` is `buf[r + j + k + l]`. The values differ by position, the host can recompute any row without materializing the input, and `collapse_contiguous_dims` cannot merge the dimensions. Each op picks its kernel through `copy_gpu_inplace`: `contiguous` of a view is `copy_g_byval`, `concatenate` is `copy_gg_byval`, and `slice_update_dynamic` is `copy_gg_dynamic` (4-D update) or `copy_gg_dynamic_nd` (an update that collapses to one dimension).

| Group | Tests | Size | Result |
|---|---|---|---|
| Fast | `strided_input_copy_past_the_grid_cap_matches_host`, `concatenate_past_the_grid_cap_matches_host`, `dynamic_update_past_the_grid_cap_matches_host` | 1300 rows of `[32, 8, 128]` f16, 42.6M elements, about 2.5 passes of the capped grid; every element compared with a host reference | 3 pass; with the loop cut to one pass, all 3 fail |
| `#[ignore]` | `strided_input_copy_past_u32_elements_keeps_the_last_rows`, `concatenate_past_u32_elements_keeps_the_last_rows`, `dynamic_update_past_u32_elements_keeps_the_last_rows`, `dynamic_one_block_update_into_a_slab_past_u32_elements` | 131,073 rows, 2^32 + 32,768 f16 elements, about 8 GiB each, run one at a time; rows checked on both sides of 2^31 and 2^32 and at the end | all 4 fail on main with "invalid configuration argument", all 4 pass with the fix |

The one-block test is the paged-cache pattern from section 1.2: a single block written into a slab past 2^32 elements. Each `#[ignore]` run went through `scripts/rocm_gpu_guard.sh`. `AMD_LOG_LEVEL=3` confirmed each op reaches the kernel it targets. Every test holds `streams::lock_default_device` for its whole body and skips on other backends.

### 5.2 Gates

- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay`: pass. Clippy on the new test target: pass.
- Orchestrator gate (`make verify-rocm` on `e5f4419c`, this branch rebased on `2aa5211f`): OK, 163 suites, 12,194 passed, 0 failed, 403 ignored, smoke included.

Not verified: Metal and CUDA are not available on this host. The change touches only `patches-rocm/` and a test gated on `feature = "rocm"`, so those paths are not built from it.

## 6. Results

### 6.1 Environment

Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`). Every GPU run went through `scripts/rocm_gpu_guard.sh`.

### 6.2 Microbenchmark

f16 `contiguous` of a transposed view per eval, before (main) against after, six interleaved rounds, median of per-round medians:

| Case | Before | After |
|---|---|---|
| `[1, 8, 4096, 128]` | 147.82 us | 147.99 us |
| `[1, 2048, 32, 128]` | 388.85 us | 387.40 us |
| `concatenate` of two `[1, 32, 2048, 128]` views | 836.75 us | 833.55 us |

The differences are +0.1%, -0.4% and -0.4%. `[1, 1, 32, 128]` (12 to 14 us, launch-bound) and `[1, 8192, 32, 128]` (33.5M elements, past the cap; best run 5455 us before, 5423 us after) vary more between rounds than between builds, so no claim is made for them.

## 7. LOCAL_FIXES Item 42

Item 42 records the failing launch and its limit, the dynamic sizing bug, the helpers and the `kLoop` split, the rejected designs with their cost, the `copy_contiguous` index change (marked found by reading), the tests and the microbenchmark numbers, and the upstream CUDA observation. It ends with the fork-policy wording: "Applies to the fork; kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there (lablup/mlxcel#2184)."

## 8. Technical Decisions

- **Template split over a runtime branch.** Typical copies are below the cap and are the hot path, so they keep the old instantiation even though it doubles instantiations.
- **64-bit counter only in the loop.** The no-loop body keeps a 32-bit thread index, which is exact under the cap.
- **Index math stays in `IdxT`.** The kernels already chose `IdxT` from the array's data size; only the loop counter widens.
- **Size the dynamic launch by the copied shape.** It is the number of elements the kernel decomposes, so it is the correct bound.
- **Reuse `copy_contiguous`'s cap.** 65535 blocks was already the convention, now shared through one helper.

## 9. Residual Risks

- **Other launchers have the same bug** (section 11).
- **The `copy_contiguous` wrap was not reproduced.** The change follows from reading the counter arithmetic.
- **Past-cap performance is only roughly measured.** One 33.5M-element case is inside run-to-run variation.
- **Metal and CUDA unverified.** CUDA's `copy_general_dynamic` at `81ba1c6a` has the same `out.size()` launch.

## 10. Learning Points

- **A cap in one launcher does not protect its siblings.** `copy_contiguous` looped while three neighbours did not, and the gap only showed at 2^32 elements.
- **A launch size must come from the work, not the buffer.** `out` is the destination for in-place ops, so `out.size()` is not the number of elements copied.
- **Keep the fast path as its own instantiation.** Both single-kernel alternatives cost 1.2 to 1.7% on a case the fix is not meant to change.
- **Test indexing at both sides of every width boundary.** The large tests check rows around 2^31 and 2^32 as well as the last rows.

## 11. Follow-up

#2234 tracks the other ROCm launchers with an uncapped grid and no grid-stride loop (paths under `patches-rocm/mlx/backend/rocm/`):

- `binary.hip` (`binary_g`)
- `unary.hip` and `ternary.hip` (`unary_g`, `ternary_g`: the y grid is uncapped)
- `reduce/init_reduce.hip` (`init_reduce_kernel`) and `reduce/col_reduce.hip` (`col_reduce_small`)
- `arange.hip` (`arange_kernel`)
- `indexing.hip` (`gather_general_kernel`, `scatter_general_kernel`, `gather_axis_kernel`, `scatter_axis_kernel`)
- `quantized/convert_fp8.hip`, `quantized/affine_quantize.hip`, `quantized/fp_quantize.hip`
- `sort.hip` (`iota_kernel`, with `int` indices)
- `copy/copy_general_input.hip` `copy_col_row` (uncapped 2-D tile grid, `int` row and column math)

`unary.hip:164`, `binary.hip:237` and `ternary.hip:132` are already capped and out of scope. The issue proposes one launcher family per PR, each with a small-size test that needs more than 65535 blocks and an `#[ignore]` test past 2^32 elements.
