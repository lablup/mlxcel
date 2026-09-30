# fix(rocm): use the shared get_2d_grid_dims and remove

| | |
|---|---|
| Target | `NripeshN/mlx`, branch `rocm-support`, head `75915908` (the head on 2026-09-30) |
| Patch | [`0001-fix-rocm-use-the-shared-get_2d_grid_dims-and-remove-.patch`](0001-fix-rocm-use-the-shared-get_2d_grid_dims-and-remove-.patch), apply with `git am` |
| Applies cleanly | yes, to `75915908` |
| Depends on | none |
| mlxcel records | `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md` items 19, 22 |
| Reproduction | `repro.py` |
| Status | ready for manual submission |

The PR title is the heading above. The PR body is everything below the rule. Read the submission notes in [the index](../README.md) first: the body is a draft for the submitter to check and put in their own words.

---

## Proposed changes

Two launch-geometry helpers in `mlx/backend/rocm/kernel_utils.hpp`.

1. **`get_2d_grid_dims(shape, strides, divisor)` sized strided scans too large.** This backend carried its own body for the divisor overload, an older form of upstream's that removed a dimension from the divisor only when the whole dimension divided it and never divided the grid by what was left. Upstream's `get_2d_grid_dims_common(shape, strides, divisor)` in `mlx/backend/common/utils.cpp` takes the gcd of each dimension and divides the grid by the remainder, and the CUDA backend delegates to it. `Scan::eval_gpu` sizes the `strided_scan` grid (a scan along any axis but the last) with this overload, so for `[1, 48, 24, 24]` scanned along axis 2 it returned 576 blocks where 48 cover the array, and each extra block read and wrote `axis_size * stride` elements past the end. That shape is the segsum of a 48-head Mamba2 layer over a 24-token chunk, and it is an `HSA_STATUS_ERROR_MEMORY_FAULT` in `strided_scan<float, float, Sum, 4, 32, 32, true, false>`; an overshoot that stays inside mapped memory corrupts silently instead. Whether a shape is hit depends on how the dimensions share factors with the divisor. The overload now delegates to `get_2d_grid_dims_common`, as the non-divisor overload beside it already did.
2. **`get_launch_args` removed.** It fixed the block at 256 threads, ignored `shape`, `strides` and `large`, and capped the grid at 65535 blocks, with nothing in its name or comment saying the kernel had to be grid-stride. A kernel written one index per thread would leave every element past 65535 x 256 unwritten. Upstream CUDA's helper of the same name does not cap x, so a kernel ported from `mlx/backend/cuda` with its call intact would compile here and inherit the truncation. Nothing calls it; launch sites compute their own geometry. It is replaced by `template <typename... Args> void get_launch_args(Args&&...) = delete;` with a comment giving the reason, so a ported call fails to compile at the explanation rather than as an undeclared name someone might fix by restoring the helper.

Measured on gfx1151 (Radeon 8060S) with HIP 7.15, with this change applied to the same backend retargeted onto a newer MLX (ml-explore/mlx `81ba1c6a`); the patch in this PR is the same change on `rocm-support`. With the fix, `cumsum` on the GPU matches the CPU stream exactly (small integers stored as float32) for `[1, 48, 24, 24]` along axis 2, for `[64, 96, 32, 32]` (6144 blocks needed, 98304 launched by the old body), and for shapes the old body already sized correctly.

## Reproduction

`repro.py` runs those `cumsum` cases against the CPU stream, each in a child process with a timeout because the unfixed `[1, 48, 24, 24]` case faults the GPU queue.

## Checklist

Put an `x` in the boxes that apply.

- [ ] I have read the [CONTRIBUTING](https://github.com/ml-explore/mlx/blob/main/CONTRIBUTING.md) document
- [ ] I have run `pre-commit run --all-files` to format my code / installed pre-commit prior to committing changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the necessary documentation (if needed)
