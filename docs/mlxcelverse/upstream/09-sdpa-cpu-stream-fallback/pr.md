# fix(rocm): fall back from fused SDPA on CPU streams

| | |
|---|---|
| Target | `NripeshN/mlx`, branch `rocm-support`, head `75915908` (the head on 2026-09-30) |
| Patch | [`0001-fix-rocm-fall-back-from-fused-SDPA-on-CPU-streams.patch`](0001-fix-rocm-fall-back-from-fused-SDPA-on-CPU-streams.patch), apply with `git am` |
| Applies cleanly | yes, to `75915908` |
| Depends on | none |
| mlxcel records | `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md` item 23 |
| Reproduction | `repro.py` |
| Status | ready for manual submission |

The PR title is the heading above. The PR body is everything below the rule. Read the submission notes in [the index](../README.md) first: the body is a draft for the submitter to check and put in their own words.

---

## Proposed changes

`ScaledDotProductAttention::use_fallback` in `mlx/backend/rocm/scaled_dot_product_attention.cpp` decided from the shapes alone whether a fused kernel applies. A call on a CPU stream with a decode-shaped query (the vector kernel's shape) therefore built the fused `ScaledDotProductAttention` primitive, whose `eval_cpu` throws "NYI", so attention on the CPU device failed on a ROCm build. `use_fallback` now returns `true` for a CPU stream before the shape checks, which is what the CUDA backend's `use_fallback` does.

Measured on gfx1151 (Radeon 8060S) with HIP 7.15, with this change applied to the same backend retargeted onto a newer MLX (ml-explore/mlx `81ba1c6a`); the patch in this PR is the same change on `rocm-support`. Attention on the CPU device then matches a host softmax reference.

## Reproduction

`repro.py` runs `mx.fast.scaled_dot_product_attention` on the CPU stream with a decode-shaped query and compares it with a softmax reference. Without the fix it raises.

## Checklist

Put an `x` in the boxes that apply.

- [ ] I have read the [CONTRIBUTING](https://github.com/ml-explore/mlx/blob/main/CONTRIBUTING.md) document
- [ ] I have run `pre-commit run --all-files` to format my code / installed pre-commit prior to committing changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the necessary documentation (if needed)
