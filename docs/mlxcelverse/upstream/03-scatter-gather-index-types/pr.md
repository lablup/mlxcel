# fix(rocm): fix Scatter's size argument and widen narrow index

| | |
|---|---|
| Target | `NripeshN/mlx`, branch `rocm-support`, head `75915908` (the head on 2026-09-30) |
| Patch | [`0001-fix-rocm-fix-Scatter-s-size-argument-and-widen-narro.patch`](0001-fix-rocm-fix-Scatter-s-size-argument-and-widen-narro.patch), apply with `git am` |
| Applies cleanly | yes, to `75915908` |
| Depends on | none |
| mlxcel records | `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md` items 14, 17 |
| Reproduction | `repro.py` |
| Status | ready for manual submission |

The PR title is the heading above. The PR body is everything below the rule. Read the submission notes in [the index](../README.md) first: the body is a draft for the submitter to check and put in their own words.

---

## Proposed changes

Two indexing fixes in `mlx/backend/rocm/indexing.hip`, plus a compile-time guard in `mlx/backend/rocm/device.h`.

1. **Scatter applied every update to the first index.** `Scatter::eval_gpu` passed `upd_post_idx_size` as `int32_t` while `scatter_general_kernel` declares it `int64_t`. `CommandEncoder::add_kernel_node` hands HIP the address of each argument, so the kernel read eight bytes from a four-byte value, got a huge divisor, and computed index element 0 for every update. `x[idx] = v`, `x.at[idx].add(v)` and everything built on scatter (`eye`, `identity`, `tri`, `diag`) applied a single update. The argument is now `int64_t`, and `add_kernel_node` statically asserts that every argument has the size of its kernel parameter, so this class of mismatch fails to compile (checked by reintroducing the `int32_t`).
2. **Narrow index dtypes in Gather and Scatter.** Five dispatch sites picked the kernel's index type with `int32_t` for int32/uint32 and `int64_t` for everything else, so int8, uint8, int16, uint16 and bool index arrays reached a kernel that reads eight bytes per element from a one or two byte buffer. On ROCm that is an `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`: the queue aborts and the process spins instead of failing. A `take` with uint8 indices, as in a 4-bit codebook lookup, is enough to hit it. Narrow index arrays are now widened to int32 once, before the dispatch.

Why widening rather than one instantiation per index dtype: that alternative was implemented and measured first. It is also correct, but it takes `indexing.hip` from 26 s to 348 s to compile, because the index type multiplies the existing product of twelve value dtypes and five to six index-count branches. The Metal and CUDA backends do instantiate per index dtype, but they compile these kernels at runtime and cache them, so the combinations cost nothing at build time; this backend compiles `indexing.hip` ahead of time. Upstream already uses the same move for the offset type (int32 or int64 by size). The cost here is one copy pass over the index array, only for narrow dtypes.

Measured on gfx1151 (Radeon 8060S) with HIP 7.15, with this change applied to the same backend retargeted onto a newer MLX (ml-explore/mlx `81ba1c6a`); the patch in this PR is the same change on `rocm-support`. Verified against the CPU stream for int8, uint8, int16, uint16, int32, uint32, int64 and uint64 indices at four index shapes.

## Reproduction

`repro.py` compares scatter (`.at[].add`, item assignment, `eye`, `tri`) and gather/scatter with every integer index dtype against the CPU stream, each case in a child process with a timeout because the unfixed narrow-dtype cases fault the GPU queue.

## Checklist

Put an `x` in the boxes that apply.

- [ ] I have read the [CONTRIBUTING](https://github.com/ml-explore/mlx/blob/main/CONTRIBUTING.md) document
- [ ] I have run `pre-commit run --all-files` to format my code / installed pre-commit prior to committing changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the necessary documentation (if needed)
