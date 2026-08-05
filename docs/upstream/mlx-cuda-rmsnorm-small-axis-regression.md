# MLX CUDA RMSNorm small-axis: the wrong normalizer is in mlxcel's overlay, not in upstream MLX

Started as an upstream bug report draft for ml-explore/mlx, tracked in lablup/mlxcel#830 (follow-up to lablup/mlxcel#824 and lablup/mlxcel#829). Verifying the claim against real source inverted it, so there is nothing to file upstream. The kernel at mlxcel's MLX pin computes the correct normalizer on the axis in question; the kernel mlxcel overlays on top of it does not.

Every upstream file and line reference below was read at MLX commit `b7c3dd6d27f45b5365b08a840310187dc503f1db`, which is mlxcel's current pin (`src/lib/mlx-cpp/CMakeLists.txt:95`) and is upstream PR #3850 itself. The two bisect-boundary commits were read at `1700b39a1dc05611dc6792f4453458d379997037` (parent, PR #3804) and `a5a684db596c117f13f7bacaea9902d0ad6d28a6` (PR #3792). All measurements were run on GB10 (DGX Spark, sm_121), CUDA 13.0, Linux 6.17.

## Summary

lablup/mlxcel#830 states that upstream #3792 ("Fix CUDA RMSNorm small-row dispatch") regressed the small-axis CUDA RMSNorm by changing `groups_per_block` from 2 to 1 at `block_dim == 64`, and that #3850 kept the same axis broken. The dispatch band it names is right and the affected DeepSeek-V2 axis is right. The direction is backwards.

Measured against a float64 CPU reference, with the three kernel bodies copied verbatim from the three upstream commits:

| Kernel | Config at `axis_size == 512`, half precision | Result |
|---|---|---|
| `1700b39` (pre-#3792), which mlxcel ships as `patches/mlx/backend/cuda/rms_norm.cu` | block `{64, 2}`, `BLOCK_DIM=64`, `REDUCE_DIM=32` | wrong normalizer, plus an out-of-bounds `__shared__` read |
| `a5a684d` (#3792) | block `{64, 1}`, `BLOCK_DIM=64`, `REDUCE_DIM=32` | correct |
| `b7c3dd6d` (#3850, the pin) | block `64`, `BLOCK_SIZE=64`, `N_CHUNKS=1` | correct |

`groups_per_block == 2` with `block_dim == 64` puts four warps in a block whose two-level reduction is sized for two, so the second stage indexes two floats past the end of its shared-memory scratch and sums partials belonging to a different row. #3792 changed that single value to 1 and added `static_assert(block_dim <= 32 || groups_per_block() == 1)` (`rms_norm.cu:372` and `:483` at `a5a684d`) to pin the invariant. It is a fix, and its commit title says so.

Two consequences for this repo:

1. Nothing should be filed against ml-explore/mlx for this. The pinned kernel is correct across the whole swept space.
2. mlxcel's overlay reintroduces a defect that upstream had already fixed, on an axis that nine model files reach.

The overlay was measured in lablup/mlxcel#829 to take deepseek-v2-lite from 0/4 to 4/4 coherent. That measurement and this one cannot both be explained by the RMSNorm kernel alone, and the discrepancy is unresolved. See "What #830's bisect got right" below.

## Evidence

The harness copies `BlockBroadcastReduce`, `rms_norm_small`, `dispatch_group_dim`, `dispatch_chunks`, and `dispatch_num_chunks` verbatim from the three upstream files, reproduces `RMSNorm::eval_gpu`'s launch configuration exactly (`add_kernel_node(F*, dim3 grid_dim, dim3 block_dim, ...)` at `device.h:54`, so `{block_dim, groups_per_block()}` is `dim3(block_dim, groups_per_block, 1)`), and compares against a per-row float64 reference. `load_vector` / `store_vector` / `unsafe_load_vector` are semantically equivalent stand-ins; the defect is in the reduction and does not depend on them.

### Sweep: every axis from 8 to 8192, row counts 1, 2, 3, 4, 8, 17, bfloat16

Failures reported at relative error above 1e-2 against the float64 reference.

| Kernel | Failing axis sizes |
|---|---|
| `1700b39` overlay | all 34 swept sizes in `[257, 512]`, at every row count above 1 |
| `a5a684d` (#3792) | none |
| `b7c3dd6d` (#3850) pin | none |

Passing configurations land at 1.6e-3 relative error for bfloat16 and 2.0e-4 for float16, which is the dtype's own rounding. The overlay's failures land at 0.55 to 0.94.

The band matches `(n_per_thread * 32, n_per_thread * 32 * 2]` exactly. With `n_per_thread == N_READS == 16 / sizeof(DataType) == 8` for 16-bit dtypes that is `(256, 512]`, which is what lablup/mlxcel#830 states. For float32, `N_READS == 4` puts the same config at `(128, 256]`; that follows from the dispatch arithmetic and was not measured.

### The `n_rows == 1` case is an uninitialized read, not a stable error

At one row per launch the overlay's answer depends on what the previous kernel left in shared memory. Same input, same kernel, same launch geometry, only the preceding kernel differs:

| Value left in the shared-memory window by a preceding kernel | Normalizer the kernel applied | Fraction of the float64 reference |
|---|---|---|
| 0.0 | 3.431100 | 99.99% |
| 1.0 | 3.184114 | 92.79% |
| 100.0 | 0.829909 | 24.19% |
| 10000.0 | 0.085499 | 2.49% |

Reference normalizer 3.431372, `axis_size == 512`, bfloat16, block `{64, 2}`.

This is also why the overlay passed `n_rows == 1` in the sweep pass but failed the same case in a spot check later in the same process: by then thousands of prior launches had dirtied the window. Decode runs at `n_rows == 1`.

### compute-sanitizer memcheck

```
========= Invalid __shared__ read of size 4 bytes
=========     at BlockBroadcastReduce<float, (int)64, (int)32>::Reduce<...>+0x740
=========     by thread (2,0,0) in block (0,0,0)
=========     Access to 0x408 is out of bounds
=========         Device Frame: BlockBroadcastReduce<float, (int)64, (int)32>::Sum(const float &)
=========         Device Frame: rms_norm_small_old<__half, (int)64, (int)32, (int)8>
```

Four such errors, all from the block `{64, 2}` launch. The `{64, 1}` launch of the same instantiation and both #3850 launches are clean. `0x408` and `0x40c` are the two floats past the end of a `float[2]`.

lablup/mlxcel#830 cites clean `compute-sanitizer` initcheck and memcheck runs as evidence that the fault is numerical rather than memory-safety. Those runs come from the lablup/mlxcel#824 investigation and therefore predate the overlay, so they exercised upstream's kernel, which is clean. The overlay is not.

## Root cause

File: `mlx/backend/cuda/rms_norm.cu`. The reduction helper is byte-identical at all three commits.

```cpp
template <typename T, int BLOCK_DIM, int GROUP_DIM = WARP_SIZE>
struct BlockBroadcastReduce {
  using TempStorage = T[std::max(BLOCK_DIM / WARP_SIZE, 1)];   // :27

  template <typename Op>
  __device__ T Reduce(const T& input, const Op& op, const T& init_value) {
    auto warp = cg::tiled_partition<GROUP_DIM>(block);          // :34
    T x = cg::reduce(warp, input, op);
    if constexpr (BLOCK_DIM > GROUP_DIM) {
      if (warp.thread_rank() == 0) {
        temp[warp.meta_group_rank()] = x;                       // :38
      }
      block.sync();
      x = warp.thread_rank() < warp.meta_group_size() ? temp[warp.thread_rank()]
                                                      : init_value;   // :41
      return cg::reduce(warp, x, op);
    } else {
      return x;
    }
  }
```

`TempStorage` is sized from the template parameter `BLOCK_DIM`. `meta_group_rank()` and `meta_group_size()` are properties of the *launched* block. The helper is only correct when those two agree, that is when the launched block holds exactly `BLOCK_DIM` threads.

The kernel is launched with a 2-D block (`rms_norm.cu:379` at `1700b39`):

```cpp
constexpr int block_dim = n_groups() * group_dim();     // :371
...
encoder.add_kernel_node(
    kernel,
    n_blocks,
    {block_dim, groups_per_block()},                    // :379
```

so the launched block has `block_dim * groups_per_block` threads while the template sees only `block_dim`. Each row is handled by the `block_dim` threads of one `threadIdx.y` slice (`auto row = grid.block_rank() * block.dim_threads().y + block.thread_index().y;`, `:69`), and `index = block.thread_index().x` addresses within the row.

Every band except one keeps those two in step, because `groups_per_block > 1` only appears where `block_dim <= 32` and the `if constexpr (BLOCK_DIM > GROUP_DIM)` branch is never taken. Then `cg::tiled_partition<GROUP_DIM>` slices the block into tiles that each cover exactly one row, and the warp-level reduce is already the whole answer.

The fourth band is the exception (`rms_norm.cu:298-301` at `1700b39`):

```cpp
  } else if (axis_size <= n_per_thread * 32 * 2) {
    f(std::integral_constant<int, 32>{},      // group_dim
      std::integral_constant<int, 2>(),       // n_groups
      std::integral_constant<int, 2>());      // groups_per_block
```

`block_dim = n_groups * group_dim = 64`, so the block-level path is taken, and `groups_per_block == 2` makes the launched block 128 threads:

- `TempStorage` is `float[max(64 / 32, 1)]`, that is `float[2]`.
- `cg::tiled_partition<32>` of a 128-thread block yields four tiles, so `meta_group_rank()` runs 0 to 3 and `meta_group_size()` is 4.
- Line 38 writes `temp[0..3]` into a two-element array. Line 41 reads `temp[0..3]` back.
- A row occupies two of those four warps. Thread ranks are `threadIdx.y * 64 + threadIdx.x`, so row 0 is warps 0 and 1 and row 1 is warps 2 and 3.

Two failure modes follow.

With both rows live, the second-stage `cg::reduce` sums all four partials, so both rows are normalized by the sum of squares over *both* rows. When the rows carry similar energy the normalizer comes out near `1/sqrt(2)` of correct, which is the 0.55 relative error in the sweep. When they do not, the error tracks the energy ratio, which is the 0.94.

With the last row of an odd `n_rows`, or with `n_rows == 1`, the `threadIdx.y == 1` threads take `if (row >= n_rows) return;` (`:71`) and exit before writing their partials. `meta_group_size()` is still 4, so line 41 still reads `temp[2]` and `temp[3]`, which this block never wrote. That is the uninitialized read the sanitizer flags and the residual-shared-memory table measures.

`a5a684d` changed the third value to 1, which makes the launched block 64 threads, `meta_group_size()` 2, and the two indices in range. It added `static_assert(block_dim <= 32 || groups_per_block() == 1)` at `:372` (forward) and `:483` (backward) so the pairing cannot drift again.

`b7c3dd6d` (#3850) replaced the forward path entirely. `dispatch_num_chunks` (`:378`) sends `axis_size <= N_READS * 64` to `BLOCK_SIZE == 64`, `N_CHUNKS == 1`, launched as `n_rows` blocks of a 1-D `BLOCK_SIZE` (`:449`), one row per block, with `BlockBroadcastReduce<float, BLOCK_SIZE>` at the default `GROUP_DIM == WARP_SIZE`. Block size and template parameter are the same value by construction. `dispatch_group_dim` survives at `:320-349` for `RMSNormVJP`, carrying #3792's `groups_per_block == 1` and its `static_assert` at `:562`.

The overlay's backward path has the same defect at `patches/mlx/backend/cuda/rms_norm.cu:500-512`, which mlxcel does not exercise.

## What #830's bisect got right, and what is unexplained

Right: the dispatch band `(256, 512]` for half precision, the `block_dim == 64` config, `groups_per_block` as the value #3792 changed, DeepSeek-V2's `kv_a_layernorm` over `kv_lora_rank == 512` landing on it, and larger axes taking a different path. All of that is confirmed above.

Wrong: which side of the change is broken.

Unexplained: lablup/mlxcel#829 measured deepseek-v2-lite going from 0/4 to 4/4 coherent under `MLX_USE_CUDA_GRAPHS=0` with the overlay as the only change, and bisected `1700b39` coherent against `a5a684d` garbage. Against the measurements here, the overlay should have made that axis worse.

Two observations that may bear on it, neither verified:

- lablup/mlxcel#817 characterized the original DeepSeek-V2 failure as nondeterministic across fresh processes with roughly 50% coherence, present at the 2026-05-19 benchmark commit, long before the 0.32.1 bump. The pre-#3792 kernel was in the tree for that entire window and its `n_rows == 1` behavior is exactly that shape: same input, different answer depending on residual shared memory. The uninitialized read is a candidate root cause for the *original* bug rather than for the regression.
- lablup/mlxcel#829's second commit added a `MLX_USE_CUDA_GRAPHS=0` startup lever for DeepSeek-V2, attributed to a graph-capture hazard. Graph capture changes which kernels precede the norm, which is precisely what determines the overlay's answer at `n_rows == 1`.

Resolving this needs a build with the overlay removed, measured on deepseek-v2-lite the way lablup/mlxcel#829 measured it. That is a separate change with real risk and is not made here.

## Minimal fix

Nothing upstream.

For mlxcel: delete `src/lib/mlx-cpp/patches/mlx/backend/cuda/rms_norm.cu` and re-verify deepseek-v2-lite. The pinned kernel is correct on the affected axis and carries #3850's speedup.

If the overlay is kept for any reason, it needs the one-value change #3792 already made:

```diff
--- a/src/lib/mlx-cpp/patches/mlx/backend/cuda/rms_norm.cu
+++ b/src/lib/mlx-cpp/patches/mlx/backend/cuda/rms_norm.cu
@@
   } else if (axis_size <= n_per_thread * 32 * 2) {
     f(std::integral_constant<int, 32>{},
       std::integral_constant<int, 2>(),
-      std::integral_constant<int, 2>());
+      std::integral_constant<int, 1>());
```

and the same at the backward-path band. Keeping the overlay without this leaves the wrong normalizer in place on the axis the overlay was added to protect.

## Repro

Standalone CUDA, run on GB10 and the basis for every number above. It instantiates the three kernel bodies verbatim, launches each with its own commit's configuration, and diffs against a per-row float64 reference. Two supporting programs run the same overlay config after a kernel that seeds a chosen value into shared memory, and under `compute-sanitizer --tool memcheck`. Build with `nvcc -std=c++17 -arch=sm_121`.

`mlx.core` reproducer, **not run** (no CUDA-enabled MLX Python build was available here). It compares fp16 RMSNorm over 512 against a float64 CPU reference, and is expected to pass at the pin:

```python
import mlx.core as mx
import numpy as np

mx.random.seed(0)
axis, rows = 512, 8                     # 512 lands in (256, 512] for 16-bit
x = mx.random.normal((rows, axis)).astype(mx.float16)
w = mx.ones((axis,)).astype(mx.float16)
eps = 1e-6

y = mx.fast.rms_norm(x, w, eps)
mx.eval(y)

xd = np.array(x, copy=False).astype(np.float64)
ref = xd / np.sqrt((xd * xd).mean(axis=-1, keepdims=True) + eps)
got = np.array(y, copy=False).astype(np.float64)

err = np.sqrt(((got - ref) ** 2).sum(-1) / (ref * ref).sum(-1))
print("worst row relative error:", err.max())   # ~2e-4 correct, >0.5 broken
```

Vary `rows` between 1 and 8. At `rows == 1` a broken build is intermittent, because the answer depends on residual shared memory, so run it after other GPU work rather than in isolation.

Model-level: deepseek-v2-lite on CUDA at temperature 0, greedy, which lablup/mlxcel#829 reports as 0/4 coherent before the overlay and 4/4 after. **Not run here**, and it is the measurement this document cannot reconcile.

## Affected surface

Only mlxcel builds, and only through the overlay. Any `fast::rms_norm` whose last axis falls in `(256, 512]` for float16 or bfloat16, or `(128, 256]` for float32. Above one row the error is deterministic; at one row it is intermittent.

`kv_a_layernorm` normalizes over `kv_lora_rank`, which defaults to 512 in `src/models/deepseek_v2.rs:150` and `src/models/deepseek_v32.rs:175`. Nine model files construct one: `deepseek_v2`, `deepseek_v3`, `deepseek_v32`, `glm4_moe_lite`, `kimi_linear`, `longcat_flash_ngram`, `minicpm3`, `mistral4`, `youtu_vl_lm`, plus `src/distributed/pipeline/stage_executor/deepseek_v3.rs`. Checkpoints that set a different `kv_lora_rank` move off the band; `src/models/glm_moe_dsa.rs` carries the same 512 default but builds no `kv_a_layernorm`.

Prefill is affected whenever the prompt is longer than one token. Decode runs at one row and is affected intermittently, depending on what the preceding kernel left in shared memory.

Upstream MLX is not affected at the pin. `RMSNormVJP` is not affected upstream either; the overlay's copy of it is, but mlxcel is inference-only.
