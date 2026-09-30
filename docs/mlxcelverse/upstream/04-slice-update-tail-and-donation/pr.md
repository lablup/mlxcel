# fix(rocm): fix SliceUpdate grid-stride tail and source

| | |
|---|---|
| Target | `NripeshN/mlx`, branch `rocm-support`, head `75915908` (the head on 2026-09-30) |
| Patch | [`0001-fix-rocm-fix-SliceUpdate-grid-stride-tail-and-source.patch`](0001-fix-rocm-fix-SliceUpdate-grid-stride-tail-and-source.patch), apply with `git am` |
| Applies cleanly | yes, to `75915908` |
| Depends on | none |
| mlxcel records | `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md` items 21, 24 |
| Reproduction | `repro.py` |
| Status | ready for manual submission |

The PR title is the heading above. The PR body is everything below the rule. Read the submission notes in [the index](../README.md) first: the body is a draft for the submitter to check and put in their own words.

---

## Proposed changes

Two `SliceUpdate` fixes, in `mlx/backend/rocm/indexing.hip` and `mlx/backend/gpu/primitives.cpp`.

1. **Reductions left the tail of large updates unapplied.** The reduce path of `SliceUpdate::eval_gpu` (Sum, Prod, Max, Min, as built by `x.at[a:b].add(...)` and the other `.at` slice reductions, and by `SliceUpdate::vjp` for Prod) clamps its grid at 65535 blocks of 256 threads, but `slice_update_op_kernel` handled one `NWORK`-element chunk per thread and never read `gridDim`. `NWORK` is 4, 2 or 1 by what divides the innermost dimension, so every element past 65535 x 256 x `NWORK` (16,776,960, 33,553,920 or 67,107,840) kept its input value, with no error. The kernel now loops over chunks with a stride of `gridDim.x * blockDim.x * NWORK`, recomputing the output and update indices for each chunk (through `elem_to_loc` on the non-contiguous branches, as the old prologue did). A chunk starts at a multiple of `NWORK`, which divides the innermost dimension, so it never crosses a row. The clamp stays, as in the other grid-stride kernels. Measured before the fix: contiguous Sum updates of 16,777,217, 33,554,434 and 67,108,868 elements left exactly the last 257, 514 and 1028 elements unapplied, and a Max through a [4099, 4097] update into columns 1..4098 of a [4099, 4099] output left 16,643; with the fix every case matches the CPU stream.
2. **A still-referenced source was donated.** `SliceUpdate::eval_gpu` and `DynamicSliceUpdate::eval_gpu` gave the source's buffer to the output whenever the buffer had one owner (`in.data_shared_ptr().use_count() == 1`), without checking whether the source array itself was still referenced. A second handle to the same array, or an unevaluated node that reads it, holds the array but not its buffer, so the update was written into an array that could still be read. Upstream's `array::is_donatable()` requires both counts to be one, and Metal and CUDA reach it through `copy_gpu`. Both primitives now call `copy_gpu` exactly as the CUDA `SliceUpdate::eval_gpu` and the upstream `DynamicSliceUpdate::eval_gpu` do. `DynamicSliceUpdate` also forced donation whenever `rocm::graph_active()` was set, for KV cache accumulation under HIP graph capture; that override is removed because it donated even a buffer other arrays shared, and it could not fire: `graph_active()` is set only when `use_hip_graphs()` is true, and `use_hip_graphs()` returns a constant `false`; the decode capture path (`decode_capture_begin`) sets `g_decode_capturing`, not `graph_active()`. If HIP graphs are turned back on, accumulation under capture has to be solved in the capture. A source dropped before evaluation is still donated, which is the usual `cache = slice_update(cache, ...)` pattern; decode throughput on gfx1151 was unchanged within run-to-run spread (under 1%) on Qwen3-0.6B-4bit and Llama-3.1-8B-Instruct-4bit.

Measured on gfx1151 (Radeon 8060S) with HIP 7.15, with this change applied to the same backend retargeted onto a newer MLX (ml-explore/mlx `81ba1c6a`); the patch in this PR is the same change on `rocm-support`.

## Reproduction

`repro.py` runs a contiguous Sum over 16,777,217 int32 elements and a strided Max above the clamp against the CPU stream, then holds a source array through GPU `.at[].add`, `.at[].maximum` and `mx.slice_update` with an array start and checks that the source is unchanged.

## Checklist

Put an `x` in the boxes that apply.

- [ ] I have read the [CONTRIBUTING](https://github.com/ml-explore/mlx/blob/main/CONTRIBUTING.md) document
- [ ] I have run `pre-commit run --all-files` to format my code / installed pre-commit prior to committing changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the necessary documentation (if needed)
