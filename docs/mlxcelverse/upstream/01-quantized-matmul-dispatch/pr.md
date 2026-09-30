# fix(rocm): fix quantized matmul dispatch for mxfp4/mxfp8, f32

| | |
|---|---|
| Target | `NripeshN/mlx`, branch `rocm-support`, head `75915908` (the head on 2026-09-30) |
| Patch | [`0001-fix-rocm-fix-quantized-matmul-dispatch-for-mxfp4-mxf.patch`](0001-fix-rocm-fix-quantized-matmul-dispatch-for-mxfp4-mxf.patch), apply with `git am` |
| Applies cleanly | yes, to `75915908` |
| Depends on | none |
| mlxcel records | `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md` items 8, 10, 12, 13 |
| Reproduction | `repro.py` (correctness; `--bench` adds the f16/bf16 gather_qmm timing) |
| Status | ready for manual submission |

The PR title is the heading above. The PR body is everything below the rule. Read the submission notes in [the index](../README.md) first: the body is a draft for the submitter to check and put in their own words.

---

## Proposed changes

Four fixes to the quantized matmul dispatch in `mlx/backend/rocm/quantized/qmm.hip`. Three are wrong results or GPU faults, one is a performance cliff.

1. **mxfp4/mxfp8 scales in `QuantizedMatmul` (qmv).** Every mode was dispatched with the activation dtype as the scale type, but mxfp4 and mxfp8 store one E8M0 byte per group. The kernels read the scales at 2 to 4 times their real stride, which gives NaN and out-of-bounds reads; on gfx1151 it is a GPU memory fault in `qmv_warp_shared_kernel`. Non-affine modes now dispatch with `uint8_t` scales.
2. **mxfp4/mxfp8 in `GatherQMM`.** When a non-affine mode missed the affine-only fast paths, the generic dispatch instantiated `gather_qmv_kernel` with `AFFINE=true` and a `T`-typed scale pointer, so fp weights were dequantized with the affine formula (relative error 1.0 to 1e34 for mxfp8). Non-affine modes now launch `gather_qmv_kernel<T, uint8_t, BITS, 32, false>`; other fp group sizes throw instead of returning wrong values. After the fix mxfp8 and mxfp4 `gather_qmm` match a dequantized float32 reference (relative error about 3e-3 for bfloat16) with sorted and unsorted indices.
3. **float32 activations in the tiled qmv path.** The branch was entered for any activation dtype, launched `qmv_tiled_kernel` only for bfloat16 and float16, and then returned. float32 activations therefore produced no kernel launch, and the output was whatever the fresh allocation held (relative error from 1.0 up to 1e35 at every shape tried). The branch is now entered only for bfloat16/float16 with group size 32, 64 or 128; everything else takes the generic qmv dispatch, which matches a dequantized float32 reference (relative error about 1e-7).
4. **float16 in the fast `GatherQMM` path.** Only bfloat16 was sent to `gather_qmv_warp_shared_kernel`, which is already templated on the element type. float16 fell through to the generic `gather_qmv_kernel` at about 205 ms per call for a decode-shaped call at the Mixtral-8x7B expert shape (99.7% of GPU time in rocprofv3 while decoding `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` at about 0.1 tok/s). The same call takes 1.44 ms on the warp-shared kernel and matches a dequantized float32 reference.

Measured on gfx1151 (Radeon 8060S) with HIP 7.15, with this change applied to the same backend retargeted onto a newer MLX (ml-explore/mlx `81ba1c6a`); the patch in this PR is the same change on `rocm-support`.

## Reproduction

`repro.py` compares GPU `quantized_matmul` and `gather_qmm` for affine 4-bit, mxfp4 and mxfp8 weights with float32, float16 and bfloat16 activations against a float32 reference from the CPU dequantization, each case in a child process with a timeout (the unfixed mxfp4/mxfp8 qmv can fault the GPU queue). `python repro.py --bench` also times float16 against bfloat16 `gather_qmm` at the Mixtral expert shape.

## Checklist

Put an `x` in the boxes that apply.

- [ ] I have read the [CONTRIBUTING](https://github.com/ml-explore/mlx/blob/main/CONTRIBUTING.md) document
- [ ] I have run `pre-commit run --all-files` to format my code / installed pre-commit prior to committing changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the necessary documentation (if needed)
