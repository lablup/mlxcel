# fix(rocm): keep ArgReduce's grid inside the per-dimension

| | |
|---|---|
| Target | `NripeshN/mlx`, branch `rocm-support`, head `75915908` (the head on 2026-09-30) |
| Patch | [`0001-fix-rocm-keep-ArgReduce-s-grid-inside-the-per-dimens.patch`](0001-fix-rocm-keep-ArgReduce-s-grid-inside-the-per-dimens.patch), apply with `git am` |
| Applies cleanly | yes, to `75915908` |
| Depends on | none |
| mlxcel records | `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md` item 11 |
| Reproduction | `repro.py` |
| Status | ready for manual submission |

The PR title is the heading above. The PR body is everything below the rule. Read the submission notes in [the index](../README.md) first: the body is a draft for the submitter to check and put in their own words.

---

## Proposed changes

For short reduction axes `ArgReduce::eval_gpu` (`mlx/backend/rocm/arg_reduce.hip`) launched one 1024-thread block per output over a 1-D grid. AMD limits each grid dimension to 2^32 - 1 threads (gridDim times blockDim), so the launch failed with "invalid configuration argument" once there were more than about 4.2M outputs. mxfp4 `quantize` of a 4096x4096 matrix, an argmin over 16.7M groups, hits it.

Axes of at most 64 elements now use a 64-thread block (whole wavefronts on both wave32 and wave64), and the grid spills into y so each dimension stays under the limit. `arg_reduce_general` already indexes `blockIdx.x + blockIdx.y * gridDim.x` with a bounds check, so the kernel is unchanged. Longer axes keep the 1024-thread block and get the same spill.

Measured on gfx1151 (Radeon 8060S) with HIP 7.15, with this change applied to the same backend retargeted onto a newer MLX (ml-explore/mlx `81ba1c6a`); the patch in this PR is the same change on `rocm-support`. With the change, GPU mxfp4 `quantize` is bit-identical to the CPU at 4096x4096.

## Reproduction

`repro.py` runs an argmin over 5M rows of 16 elements and an mxfp4 `quantize` of a 4096x4096 matrix on the GPU and compares both with the CPU stream exactly. Without the fix both raise.

## Checklist

Put an `x` in the boxes that apply.

- [ ] I have read the [CONTRIBUTING](https://github.com/ml-explore/mlx/blob/main/CONTRIBUTING.md) document
- [ ] I have run `pre-commit run --all-files` to format my code / installed pre-commit prior to committing changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the necessary documentation (if needed)
