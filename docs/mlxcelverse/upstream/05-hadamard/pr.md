# feat(rocm): implement Hadamard for power-of-two sizes up to

| | |
|---|---|
| Target | `NripeshN/mlx`, branch `rocm-support`, head `75915908` (the head on 2026-09-30) |
| Patch | [`0001-feat-rocm-implement-Hadamard-for-power-of-two-sizes-.patch`](0001-feat-rocm-implement-Hadamard-for-power-of-two-sizes-.patch), apply with `git am` |
| Applies cleanly | yes, to `75915908` |
| Depends on | none (conflicts only textually with the FFT PR in `CMakeLists.txt` and `primitives.cpp`, adjacent lines) |
| mlxcel records | `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md` item 16 |
| Reproduction | `repro.py` |
| Status | ready for manual submission |

The PR title is the heading above. The PR body is everything below the rule. Read the submission notes in [the index](../README.md) first: the body is a draft for the submitter to check and put in their own words.

---

## Proposed changes

`Hadamard` is a `NO_GPU` stub on this backend, so `mx.hadamard_transform` raises on the GPU. This adds `mlx/backend/rocm/hadamard.hip`: a single-stage radix-2 butterfly in shared memory for `m == 1` and a power-of-two last axis up to 8192, which covers the head dimensions (64, 128, 256) that Hadamard-rotated KV caches use.

- Larger `n` and the `m` in {12, 20, 28} families throw an error naming the limit rather than returning a wrong answer. The CUDA backend covers them by generating a radix-m codelet through NVRTC, which has no ahead-of-time equivalent here.
- The scale is the primitive's `scale_`, which MLX defaults to `1/sqrt(n)`, so the default transform stays an involution. Accumulation is in float regardless of the element type.
- Like the CUDA kernel, the grid is clamped at 65535 blocks and the kernel is grid-stride.

Measured on gfx1151 (Radeon 8060S) with HIP 7.15, with this change applied to the same backend retargeted onto a newer MLX (ml-explore/mlx `81ba1c6a`); the patch in this PR is the same change on `rocm-support`. Against the Metal backend on an M1 Ultra: norm ratio 1.000000 within 1e-5 at n = 64, 128 and 256; float16 relative RMS about 2.0e-4 (Metal 3.8e-4 to 5.0e-4), float32 about 1e-7 (matching Metal). The float16 figure is lower than Metal's because this kernel keeps the butterfly in float in shared memory and rounds once on the way out, where Metal rounds between stages.

## Reproduction

`repro.py` compares the GPU with the CPU stream for n = 64 to 8192 in float32, float16 and bfloat16, checks the round trip, and checks that n = 16384 and n = 768 (m = 12) raise.

## Checklist

Put an `x` in the boxes that apply.

- [ ] I have read the [CONTRIBUTING](https://github.com/ml-explore/mlx/blob/main/CONTRIBUTING.md) document
- [ ] I have run `pre-commit run --all-files` to format my code / installed pre-commit prior to committing changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the necessary documentation (if needed)
