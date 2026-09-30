# feat(rocm): implement FFT through hipFFT

| | |
|---|---|
| Target | `NripeshN/mlx`, branch `rocm-support`, head `75915908` (the head on 2026-09-30) |
| Patch | [`0001-feat-rocm-implement-FFT-through-hipFFT.patch`](0001-feat-rocm-implement-FFT-through-hipFFT.patch), apply with `git am` |
| Applies cleanly | no: apply after the blocking-sync PR (06) and the Hadamard PR (05); on `75915908` alone it conflicts in `CMakeLists.txt` and `primitives.cpp` (adjacent lines) |
| Depends on | 06 at runtime (plan eviction hangs without it); 05 for a clean apply |
| mlxcel records | `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md` items 18, 25 |
| Reproduction | `repro.py` |
| Status | ready for manual submission |

The PR title is the heading above. The PR body is everything below the rule. Read the submission notes in [the index](../README.md) first: the body is a draft for the submitter to check and put in their own words.

---

## Proposed changes

`FFT` is a `NO_GPU` stub on this backend. This adds `mlx/backend/rocm/fft.hip`, a port of `mlx/backend/cuda/fft.cu` that keeps its axis walk, its packing of the transform axis to the back, and its inverse scaling, with two differences:

- This hipFFT has no `hipfftXtMakePlanMany` or `hipfftXtExec`, the generic entry points the CUDA file uses to pass input, output and execution types separately. The typed API takes their place: one `hipfftType` (C2C, R2C or C2R) per plan and one exec function per transform, which covers the same ground because MLX only asks for float32 and complex64 here.
- hipFFT work cannot be a HIP graph kernel node, so execution goes through `CommandEncoder::launch_kernel`, the path the rocBLAS and hipBLASLt GEMMs already take.

`hipfft` joins the link list, with a configure-time error naming the package when it is missing.

The plan cache holds 128 entries, as in the CUDA file. This depends on the blocking-sync fix (the "set blocking-sync before the first HIP queue is created" PR), which has to land first: without it the first plan eviction past about 16 live plans hangs inside `hipfftDestroy`, whose `hipFree` never returns.

`MLX_ROCM_FFT_CACHE_SIZE` overrides the capacity, and its parsing in `lru_cache.h` is fixed. `LRUBytesKeyCache` read it with an unchecked `std::stoul`: `0` was accepted and the first `put()` then called `back()` and `pop_back()` on an empty list (undefined behavior; `std::bad_alloc` on gfx1151); `abc`, an empty value or an overflow threw a bare `stoul` out of the static's initializer in `FFT::eval_gpu`, so every later FFT threw again; `-1` became `SIZE_MAX` and `8abc` became 8. `capacity_from_env()` now takes a whole decimal integer from 1 to `INT_MAX`; anything else prints one line to stderr and uses the default. `LRUCache` and `LRUBytesKeyCache` throw for a capacity of 0, as the CUDA `LRUCache` does.

Measured on gfx1151 (Radeon 8060S) with HIP 7.15, with this change applied to the same backend retargeted onto a newer MLX (ml-explore/mlx `81ba1c6a`); the patch in this PR is the same change on `rocm-support`. Verified against the CPU stream for rfft, irfft, round trips and complex fft at lengths 2, 8, 400, 512, 1024, 1200 and 2048 and batches up to 512, all within 1e-5 relative, and at plan-cache capacities 16, 32 and 128 with 40 distinct plans.

## Reproduction

`repro.py` checks FFT correctness against the CPU stream, runs 48 distinct transform lengths with `MLX_ROCM_FFT_CACHE_SIZE=16` in a child process with a timeout (plan eviction), and checks the variable's validation for valid and invalid values.

## Checklist

Put an `x` in the boxes that apply.

- [ ] I have read the [CONTRIBUTING](https://github.com/ml-explore/mlx/blob/main/CONTRIBUTING.md) document
- [ ] I have run `pre-commit run --all-files` to format my code / installed pre-commit prior to committing changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the necessary documentation (if needed)
