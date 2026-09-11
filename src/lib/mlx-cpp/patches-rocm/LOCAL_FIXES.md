# Local changes relative to NripeshN/mlx@75915908

Every change this overlay carries on top of the fork commit in `UPSTREAM`, with the reason. Fixes that apply to the fork itself should also be proposed there; retarget-only changes follow the fork when it merges newer MLX.

## Retarget onto the MLX pin (81ba1c6a)

The fork branched from upstream on 2026-06-26 (`39886de4`). Moving its ROCm code onto the pin needed:

1. **Core file merges.** `mlx/backend/common/compiled.cpp` keeps upstream's `negative_strides` tracking and the fork's constant-input skipping. `mlx/fast_primitives.h` keeps both upstream's `compile_options` and the fork's `output_input_aliases` on `fast::CustomKernel`. `mlx/io/safetensors.cpp` keeps upstream's empty-tensor guard and the fork's ROCm `staged_write`.
2. **`isnan` lookup.** Upstream `a124ac09` (ml-explore/mlx#4291) added a host-only `mlx::core::isnan` template in `mlx/types/complex.h`, which hides the device `isnan` overloads inside the ROCm namespace. The 23 unqualified device-side calls use `::isnan`.
3. **Compiled kernels.** `compiled_collapse_contiguous_dims` returns a 4-tuple with `negative_strides`; `mlx/backend/rocm/compiled.cpp` unpacks it and forces the large-index kernel for negative strides, as the CUDA backend does.
4. **Custom kernels.** `mlx/backend/rocm/custom_kernel.cpp` passes an empty `CompileOptions::Data` before the aliases argument. `output_input_aliases_` stays out of `CustomKernel::state()` because `mlx/export.cpp` serializes that tuple and has no encoding for it (the fork's `state()` did not include it either).
5. **SDPA.** `ScaledDotProductAttention::use_fallback` takes upstream's new `force_fused` argument, with the CUDA semantics: throw if fused is forced and no fused kernel applies.
6. **New upstream primitives.** `GatherQQMM`, `SearchSorted` and `fast::CrossEntropy` (with its VJP, using the graph fallback) get `NO_GPU` stubs in `mlx/backend/rocm/primitives.cpp`.
7. **Event errors.** `Event::error()` (ml-explore/mlx#3742) has storage in the ROCm `EventImpl`. ROCm GPU failures do not populate it yet; tracked in lablup/mlxcel#1804.

## Fixes to the fork's kernels

8. **mxfp4/mxfp8 scale type in qmv dispatch.** `mlx/backend/rocm/quantized/qmm.hip` dispatched every mode with the activation dtype as the scale type, but mxfp4 and mxfp8 store one E8M0 byte per group. The kernels read scales at 2 to 4 times the real stride, which gave NaN and out-of-bounds reads (a GPU memory fault in `qmv_warp_shared_kernel` on the unfixed fork). Non-affine modes now dispatch with `uint8_t` scales. Applies to the fork; to be proposed there.
9. **Expert-batched gather qmv made opt-in.** On `gfx1151` the expert-batched `gather_qmm` path returns wrong results for bf16 activations with sorted rhs indices (relative error above 1.0 against the unsorted path; f32 and f16 are correct). It is only selected for models with at most 64 experts, so it affected their prefill. It now runs only with `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=1`. Applies to the fork; to be proposed there after a root cause.

## Dropped from the vendored set

- `mlx/backend/metal/custom_kernel.cpp` and `mlx/backend/cuda/custom_kernel.cpp` from the fork are not carried: a ROCm build does not compile them, and Metal and CUDA builds never copy this directory.
- The fork's Python bindings, tests, docs and benchmark scripts are not needed by mlxcel.
