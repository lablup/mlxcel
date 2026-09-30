# fix(rocm): run CPU-stream BLAS single-threaded over fine-grained memory

| | |
|---|---|
| Target | `NripeshN/mlx`, branch `rocm-support`, head `75915908` (the head on 2026-09-30) |
| Patch | [`0001-fix-rocm-run-CPU-stream-BLAS-single-threaded-over-fi.patch`](0001-fix-rocm-run-CPU-stream-BLAS-single-threaded-over-fi.patch), apply with `git am` |
| Applies cleanly | yes, to `75915908` (and on top of packages 01 to 10) |
| Depends on | none |
| mlxcel records | `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md` item 27 |
| Reproduction | `repro.c` (standalone C program, no MLX) |
| Status | ready for manual submission |

The PR title is the heading above. The PR body is everything below the rule. Read the submission notes in [the index](../README.md) first: the body is a draft for the submitter to check and put in their own words.

---

## Proposed changes

On an APU, `unified_malloc` in `mlx/backend/rocm/allocator.cpp` gives every array fine-grained device memory, CPU-stream arrays included, and `Buffer::raw_ptr()` hands the CPU that same pointer. MLX's CPU backend therefore calls BLAS with its output in fine-grained memory, and multithreaded OpenBLAS returns wrong output columns there. Each wrong column is the last column of one OpenBLAS thread's share of the output, the column whose 64-byte line the next thread also writes.

Measured on gfx1151 (Radeon 8060S) with Debian OpenBLAS 0.3.29 at 32 threads:

- A CPU-stream `matmul(x, w.T)` of [1, 2880] x [2880, 2880]^T was wrong in 35 of 60 calls with the host idle, and correct in 300 of 300 with `OPENBLAS_NUM_THREADS=1`.
- A standalone `cblas_sgemm` of the same shape (`repro.c`) was wrong in 37 of 50 calls with inputs and output from `hipExtMallocWithFlags(hipDeviceMallocFinegrained)`, in 10 of 300 with only the output there, and in none with `malloc` or `hipHostMalloc` buffers or with 1, 2 or 4 threads.

The allocator now calls `openblas_set_num_threads(1)` once, on its first fine-grained allocation, which comes before any CPU-stream BLAS call that writes such a buffer. The symbol is declared weak, so a build linked against another BLAS still links and nothing changes there.

Cost: float32 BLAS on the CPU stream gets slower (this gemv went from about 0.2 s to about 1.4 s per call, because the CPU reads fine-grained memory slowly and now does so on one thread). GPU work and MLX's own CPU kernels (quantized matmuls, bf16/f16 matmuls) do not use BLAS. The alternative, giving BLAS a cacheable scratch output, would need a change at every BLAS and LAPACK call site in the CPU backend. Whether other OpenBLAS versions or discrete GPUs (which do not take the fine-grained path) are affected was not checked.

## Reproduction

`repro.c` has the build line and the `MODE` settings in its header: `MODE=11` puts inputs and output in fine-grained memory, `MODE=00` uses `malloc`, and `OPENBLAS_NUM_THREADS=1` removes the errors. It shows the OpenBLAS behavior without MLX; with the patch, the same product through MLX on the CPU stream matches a float64 reference on every call.

## Checklist

Put an `x` in the boxes that apply.

- [ ] I have read the [CONTRIBUTING](https://github.com/ml-explore/mlx/blob/main/CONTRIBUTING.md) document
- [ ] I have run `pre-commit run --all-files` to format my code / installed pre-commit prior to committing changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the necessary documentation (if needed)
