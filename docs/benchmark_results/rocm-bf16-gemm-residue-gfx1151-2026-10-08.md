# bf16 GEMM residue at exact zeros on ROCm: Radeon 8060S (gfx1151), 2026-10-08

Why a bf16 `matmul` on gfx1151 returns values like `-2^-22` where the exact answer is zero, which kernel and which instruction produce it, and whether it is a defect in how mlxcel calls the libraries (issue #2206, found while writing the exact bf16 check of `tests/rocm_hipblaslt_concurrency.rs` for #2200).

Conclusion: it is how the gfx11 matrix and dot instructions accumulate. `v_wmma_f32_16x16x16_bf16` (and its f16 twin, and `v_dot2_f32_bf16`) do not add opposite-sign terms the IEEE way. A single instruction computing `1 * 1 + (-1) * 1` returns `-2^-24`. Every hipBLASLt solution for these shapes and rocBLAS's bf16-output kernel are built on that instruction, so no solution choice, compute type or flag in our call avoids it. The error is at most 0.8% of the standard f32 dot-product bound in every case measured, so nothing in the overlay changes. Exact bf16 checks on ROCm must use a tolerance derived from that bound.

## Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5, wave32), HIP 7.15.26333, hipBLASLt `libhipblaslt.so.1.4`, rocBLAS `librocblas.so.5.6`, Debian 13. mlxcel at `237faf79` plus this change, `--features rocm`. The probes are in [`scripts/rocm_bf16_gemm_residue/`](../../scripts/rocm_bf16_gemm_residue/) (`run.sh` builds and runs both). Results are deterministic: every repeated run printed identical values.

## The instruction, without any library

`wmma_probe.hip` runs one `v_wmma_f32_16x16x16_bf16` on one wave32 with `C = 0`, row `m` of A a short vector padded with zeros and B all ones, so each output is the sum of the vector. The emitted ISA was checked (`--save-temps`: one `v_wmma_f32_16x16x16_bf16`, one `v_wmma_f32_16x16x16_f16`, eight `v_dot2_f32_bf16`). The same rows go through the f16 WMMA, a `v_dot2_f32_bf16` chain, and a plain `fmaf` loop on the widened values.

| row of A | exact | WMMA bf16 | WMMA f16 | dot2 bf16 | fmaf |
|---|---:|---:|---:|---:|---:|
| `[1, -1]` | 0 | -2^-24 | -2^-24 | -2^-24 | 0 |
| `[-1, 1]` | 0 | -2^-24 | -2^-24 | -2^-24 | 0 |
| `[2, -2]` | 0 | -2^-23 | -2^-23 | -2^-23 | 0 |
| `[8, -8]` | 0 | -2^-21 | -2^-21 | -2^-21 | 0 |
| `[0.5, -0.5]` | 0 | -2^-25 | -2^-25 | -2^-25 | 0 |
| `[2, -1, -1]` | 0 | -3 x 2^-24 | same | same | 0 |
| `[1, 2, -3]` | 0 | -2^-23 | same | same | 0 |
| `[1, -1, 1, -1]` | 0 | 0 | 0 | 0 | 0 |
| `[2]`, `[-1]` | 2, -1 | exact | exact | exact | exact |

Every product and partial sum here is exact in any IEEE format and any association order, so no rounding mode or summation order produces these values; the FMA loop on the same data is exact. The residue appears only when terms of opposite sign meet, scales with the magnitude of the cancelling terms, and is about one unit in the 24th significant bit of the largest of them. This matches the silicon model in the public report [ROCm/rocm-systems#12056](https://github.com/ROCm/rocm-systems/issues/12056) (gfx1151, the same four instructions): sign-magnitude addition in which a term opposite in sign to the accumulator is negated by one's complement, without the increment.

## The libraries, without MLX

`gemm_repro.cpp` calls hipBLASLt and rocBLAS `gemm_ex` with the configuration `gemms/hipblaslt_gemm.cpp` uses: bf16 A and B, `HIPBLAS_COMPUTE_32F`, f32 alpha = 1 and beta = 0, TN, `[64, 64] x [64, 64]^T`, against an exact host reference. "at zero" counts wrong outputs whose exact value is 0; "ratio" is the largest `|error| / (K u sum_k |a_k b_k|)` with `u = 2^-24`.

| inputs | hipBLASLt bf16 out | hipBLASLt f32 out | rocBLAS bf16 out | rocBLAS f32 out |
|---|---|---|---|---|
| test generator, `-1..=1` | exact | exact | exact | exact |
| test generator, `-2..=2` | 638 wrong, all at zero, ratio 0.0006 | 1,301 wrong (638 at zero), ratio 0.0075 | 638 wrong, all at zero | exact |
| `-2..=2` with zeros replaced (`{-2,-1,1,2}`) | exact | exact | exact | exact |
| rows `[1, -1, 0, ...]` x ones | 4,096 of 4,096 wrong, `-2^-24`, ratio 0.0078 | same | same | exact |
| rows `[2, -2, 0, ...]` x ones | 4,096 wrong, `-2^-23`, ratio 0.0078 | same | same | exact |
| rows `[2, 0, ...]` x ones | exact | exact | exact | exact |

- Kernels (`hipblaslt_ext::getKernelNameFromAlgo` and `rocprofv3 --kernel-trace`): hipBLASLt bf16 output `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x64_MI16x16x1_..._ISA1151_...`, f32 output `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_UserArgs_MT16x16x64_MI16x16x1_...`; rocBLAS bf16 output `Cijk_Alik_Bljk_BBS_BH_MT128x128x32_MI16x16x16x1_..._ISA1151_...`. `MI16x16x1` is a WMMA tile. rocBLAS's f32-output call ran `Cijk_Alik_Bljk_BSS_BH_MT64x32x8_SE_..._ISA000_...`, the generic fallback kernel without matrix instructions, and is the only exact arm.
- With `--all`, all 64 heuristic solutions for bf16 output and all 9 for f32 output are WMMA kernels, and every one gives the same 638 (bf16) or 1,301 (f32) wrong outputs. No hipBLASLt solution for this problem avoids the instruction.
- MLX takes the same kernel: `MLX_ROCM_GEMM_DEBUG=1` plus `rocprofv3 --kernel-trace` on the test's child with `-2..=2` inputs showed only `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x64_MI16x16x1_...` for the bf16 TN GEMM, and the exact check failed with `-2^-22` and `+2^-23` at zeros.
- The test generator at `-1..=1` is exact only because of its pattern; rows `[1, -1, 0, ...]` show values in `{-1, 0, 1}` are not exact in general.
- The overlay's own `qmm_wmma_dense_kernel` (`quantized/qmm.hip`) and `flash_attention_wmma.hip` use rocWMMA fragments with f32 accumulators, which rocWMMA maps to the gfx11 WMMA instructions, so they are expected to show the same residue; not measured here.

## Against the error bound

Neither hipBLASLt nor rocBLAS documents an accuracy bound. The standard bound for a dot product accumulated in f32 is `|error| <= gamma_K sum_k |a_k b_k|` with `gamma_K ~ K u` (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., section 3.1). Random normal bf16 inputs, f32 output, first heuristic solution, against a double reference:

| size | hipBLASLt (WMMA) max error | ratio | rocBLAS f32 out (FMA fallback) max error | ratio |
|---|---:|---:|---:|---:|
| 64^3 | 3.81e-6 | 0.021 | 2.62e-6 | 0.021 |
| 256^3 | 5.14e-5 | 0.020 | 1.94e-5 | 0.0075 |
| 1024^3 | 3.37e-4 | 0.0079 | 1.40e-4 | 0.0035 |
| 4096^3 | 4.0e-3 | 0.0063 | 6.92e-4 | 0.0011 |

On general data the WMMA kernels stay within 2.1% of the bound; their error is up to 5.8 times that of the FMA fallback at 4096^3, the same order as ordinary f32 rounding, which the residue only adds to. At exact zeros the residue is at most 0.8% of the bound for these shapes. It is visible at zero only because a bf16 output can hold `2^-22` there, while at a non-zero output bf16 rounding (relative step `2^-8`) absorbs it.

## Metal and CUDA

Not run: this host has neither. By code reading of the MLX source at the pin, Metal's steel GEMM loads bf16 tiles into `float` fragments and multiplies with `simdgroup_multiply_accumulate` in `float` (`steel/gemm/mma.h`, `AccumType = float`); Apple does not document that instruction's rounding. CUDA's `cublas_gemm.cpp` asks cuBLASLt for `CUBLAS_COMPUTE_32F` on bf16, which runs on tensor cores; [Fasi, Higham, Mikaitis and Pranesh (2021)](https://eprints.maths.manchester.ac.uk/2761/1/fhms20.pdf) measured that tensor cores compute products exactly and accumulate with truncation and no guard digits. For these small-integer cases no bits are shifted out, so both are expected to be exact, but neither is guaranteed by its vendor.

## What changed

- No overlay change. The residue is in the instruction every candidate kernel uses; the only exact route found, rocBLAS's non-WMMA fallback, takes f32 output, and routing bf16 GEMMs there to remove an error well inside the bound would give up the matrix units on every prefill GEMM.
- `tests/rocm_hipblaslt_concurrency.rs` is back to `-2..=2` bf16 inputs, checked within twice the bound (`bf16_gemm_tolerance`, at most about 0.002 for these shapes, so any wrong operand or tile, an error of at least 1, still fails); the f32 rocBLAS check stays exact. `bf16_gemm_error_is_within_the_accumulation_bound` pins the bound on rows `[h, -h, 0, ...]`: on gfx1151 4,096 of 4,096 outputs are inexact and all are within it.
- Writing exact bf16 or f16 GEMM tests on ROCm: integer inputs do not make the result exact on gfx11. Compare within `K u sum_k |a_k b_k|` (doubled for a bf16 output), or use inputs whose terms all share a sign.
