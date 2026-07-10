// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device/qmm_sm80.cuh"

#include "mlx/backend/cuda/cutlass_utils.cuh"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/backend/cuda/quantized/qmm/qmm_utils.h"

#include "cuda_jit_sources.h"

#include <cstdlib>

namespace mlx::core {

namespace {

inline auto make_cta_tiler(int m, int group_size, cu::Device& device) {
  // #637 (mlxcel overlay): on consumer Blackwell (sm_120/121, cc major >= 12)
  // a 128-row CTA tile fills the SMs far better for large-M (prefill / batched)
  // shapes; the upstream Ampere cap of 64 leaves them underutilized there (ncu:
  // ~35-47% SM throughput). Measured on GB10: +38% prefill @8192 on
  // llama-3.1-8b-4bit and +31% on qwen2.5-7b-4bit, greedy-parity identical, no
  // decode regression (decode m=1 takes the qmv path and never reaches here).
  // sm_90/sm_80 keep the stock cap of 64 (untuned here). Wider tile_n / deeper
  // tile_k were also swept but break the fixed-MMA smem layout (JIT failure),
  // so tile_m is the one safely-tunable axis.
  int tile_m_cap = device.compute_capability_major() >= 12 ? 128 : 64;
  int tile_m = std::max(16, std::min(tile_m_cap, next_power_of_2(m)));
  int tile_n = 128;
  int tile_k = std::max(64, group_size);
  // Optional env overrides retained as a tuning / escape hatch (#637 spike).
  if (const char* e = std::getenv("MLXCEL_QMM_TILE_M")) {
    int v = std::atoi(e);
    if (v > 0) {
      tile_m = v;
    }
  }
  if (const char* e = std::getenv("MLXCEL_QMM_TILE_N")) {
    int v = std::atoi(e);
    if (v > 0) {
      tile_n = v;
    }
  }
  if (const char* e = std::getenv("MLXCEL_QMM_TILE_K")) {
    int v = std::atoi(e);
    if (v > 0) {
      tile_k = v;
    }
  }
  return cute::make_shape(tile_m, tile_n, tile_k);
}

} // namespace

void qmm_sm80(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  auto [m, n, k, l, broadcast_b] = make_problem_shape(x, w, out);
  auto cta_tiler = make_cta_tiler(m, group_size, encoder.device());

  std::string module_name = fmt::format(
      "qmm_sm80_tn_{}_m{}_b{}_g{}_{}",
      dtype_to_string(x.dtype()),
      cute::size<0>(cta_tiler),
      bits,
      group_size,
      quantization_mode_to_string(mode));

  auto [ctype_x, ctype_q, ctype_s] = get_qmm_cutlass_types(x, bits, mode);
  std::string kernel_name = fmt::format(
      "mlx::core::cu::qmm_sm80_kernel<{}, {}, {}, {}, {}>",
      group_size,
      ctype_x,
      ctype_q,
      ctype_s,
      cta_tiler_to_string(cta_tiler));

  cu::JitModule& mod = cu::get_jit_module(encoder.device(), module_name, [&]() {
    return std::make_tuple(
        false, jit_source_qmm_sm80, std::vector{kernel_name});
  });

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  if (biases) {
    encoder.set_input_array(*biases);
  }
  if (lhs_indices) {
    encoder.set_input_array(*lhs_indices);
  }
  if (rhs_indices) {
    encoder.set_input_array(*rhs_indices);
  }
  encoder.set_output_array(out);

  dim3 num_blocks{
      uint32_t(cute::ceil_div(m, cute::size<0>(cta_tiler))),
      uint32_t(cute::ceil_div(n, cute::size<1>(cta_tiler))),
      uint32_t(l)};
  dim3 block_dims{uint32_t(cute::size(cu::make_tiled_mma()))};

  auto [sA_layout, sB_layout, sC_layout] = cu::make_smem_layouts(cta_tiler);
  size_t smem_bytes = std::max(
      cute::cosize(sA_layout) * x.itemsize() +
          cute::cosize(sB_layout) * bits / 8,
      cute::cosize(sC_layout) * x.itemsize());

  auto kernel = mod.get_kernel(kernel_name, [&](CUfunction kernel) {
    if (smem_bytes > 48000) {
      cuFuncSetAttribute(
          kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes);
    }
  });

  encoder.add_kernel_node_ex(
      kernel,
      num_blocks,
      block_dims,
      {},
      smem_bytes,
      gpu_ptr<void>(x),
      gpu_ptr<void>(w),
      gpu_ptr<void>(scales),
      biases ? gpu_ptr<void>(*biases) : nullptr,
      lhs_indices ? gpu_ptr<void>(*lhs_indices) : nullptr,
      rhs_indices ? gpu_ptr<void>(*rhs_indices) : nullptr,
      gpu_ptr<void>(out),
      m,
      n,
      k,
      l,
      broadcast_b);
}

} // namespace mlx::core
