// Copyright © 2026 Apple Inc.
// Patched by mlxcel (lablup/mlxcel#1541): size the CTA tile against the
// device's real per-block shared-memory budget instead of a shape rule
// standing in for one, opt into a larger dynamic shared-memory maximum when
// the selected tile needs one, and refuse a tile that does not fit instead of
// handing the driver a launch it cannot honour.
//
// Upstream sized the tile with `bool enough_smem = sm80 && itemsize <= 2 &&
// group_size <= 64`, where `sm80` is `compute_capability_major() >= 8`. Two
// separate things were folded into that one name. `itemsize <= 2 &&
// group_size <= 64` is a shared-memory rule and is now written as a real
// budget comparison in `qmm_naive_tile.h`. `sm80` is not: a Tesla V100 offers
// 96 KB per block through `cudaFuncAttributeMaxDynamicSharedMemorySize` and
// 48 KB without it, against the 24 KB the widest eligible tile needs, so
// shared memory was never what kept the wide tile off pre-Ampere parts. It
// selects which MMA atom the kernel runs on, and it now appears under that
// name.
//
// #1541 measured the wide tile on the pre-Ampere FMA path rather than assuming
// it, and upstream's exclusion holds for a reason its name did not give: on a
// V100 the 128-wide instantiation needs 255 registers and spills 128 bytes per
// thread against the 64-wide one's 224 and none, both reach the same 2 blocks
// per SM, and halving the CTA count on a grid already smaller than an 80-SM
// part makes `qmm_naive` 1.50x slower at a 106-token prompt and 1.5% slower at
// 1906, at identical launch counts. The selected tile is therefore identical
// to upstream's on every architecture, which
// `src/lib/mlxcel-core/src/qmm_naive_tile_tests.rs` asserts by enumeration
// over every `(itemsize, group_size, m)` combination and every real per-block
// budget. Record: `docs/benchmark_results/qmm-naive-tile-v100-2026-08-31.md`.
//
// What does change at this launch site:
// (1) `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)` is
//     issued when the selected tile exceeds the opt-in-free ceiling, matching
//     what `qmm_sm80.cu` and `qmm_sm90.cu` already do upstream. Upstream
//     `qmm_naive.cu` never opted in, and its one tile that crosses the
//     ceiling, f32 activations at group size 128, asks for 64 KB and fails at
//     launch on every architecture.
// (2) A tile that does not fit the device budget throws here, naming the tile,
//     the requirement and the budget, rather than failing inside the driver.
//     A CuTe layout change under an MLX pin bump that moves the real
//     shared-memory figure away from the selector's model throws too.
// (3) The N tile joins the JIT module name. The module name is the sole key
//     for both the in-process module cache and the persistent PTX disk cache,
//     while the tile shape reaches the compiler only through the kernel name,
//     so a module name that does not pin tile_n hands back a cached module
//     with no matching kernel the moment the selection moves. That also
//     invalidates PTX cached by earlier builds, the same hazard #910 fixed in
//     `qmm_sm80.cu` by bumping its module name.
// (4) `MLXCEL_QMM_NAIVE_TILE_N` pins the N tile to 64 or 128 and
//     `MLXCEL_TRACE_QMM_TILE` prints each distinct kernel's tile, register
//     count and occupancy once. Both exist so the #1541 sweep can be re-run in
//     one command, which is what #1543 will need when it gives Volta a
//     tensor-core MMA to re-measure the wide tile against. Unset, neither has
//     any effect.
//
// Everything else is byte-identical upstream at the current pin (9a795735).

#include "mlx/backend/cuda/device/qmm_naive.cuh"

#include "mlx/backend/cuda/cuda_utils.h"
#include "mlx/backend/cuda/cutlass_utils.cuh"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/backend/cuda/quantized/qmm/qmm_naive_tile.h"
#include "mlx/backend/cuda/quantized/qmm/qmm_utils.h"

#include "cuda_jit_sources.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace mlx::core {

namespace {

// `cudaDevAttrMaxSharedMemoryPerBlockOptin` for one device, queried once.
//
// This is the ceiling a kernel can reach after
// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`, not the 48 KB available
// without it: 98304 on sm_70, 65536 on sm_75, 101376 on sm_86/sm_89/sm_120,
// 166912 on sm_80, 232448 on sm_90. If the query fails, fall back to the
// opt-in-free floor every supported architecture guarantees, which selects
// exactly what upstream selected.
long long max_smem_per_block_optin(int cuda_device) {
  static std::mutex mtx;
  static std::unordered_map<int, long long> cache;

  std::lock_guard<std::mutex> lock(mtx);
  auto it = cache.find(cuda_device);
  if (it != cache.end()) {
    return it->second;
  }

  int value = 0;
  cudaError_t err = cudaDeviceGetAttribute(
      &value, cudaDevAttrMaxSharedMemoryPerBlockOptin, cuda_device);
  if (err != cudaSuccess || value <= 0) {
    // Do not leave the failed query latched on the device error state.
    cudaGetLastError();
    value = 48 * 1024;
  }
  cache.emplace(cuda_device, value);
  return value;
}

// `MLXCEL_QMM_NAIVE_TILE_N`. 64 or 128 pins the N tile; anything else,
// including unset, leaves the selection to the shared-memory budget.
//
// Read on every dispatch rather than cached, so one process can run both
// widths and compare their outputs directly. That is what
// `qmm_naive_output_is_identical_across_cta_tile_widths` does, and an in-process
// comparison is the only way to make the parity claim bitwise rather than
// "two runs agreed". It is safe because tile_n is part of the module name, so
// each width gets its own module and kernel. `getenv` is a pointer lookup next
// to two `fmt::format` calls and a map lookup already on this path.
int forced_tile_n() {
  const char* raw = std::getenv("MLXCEL_QMM_NAIVE_TILE_N");
  if (raw == nullptr) {
    return 0;
  }
  int parsed = std::atoi(raw);
  return (parsed == 64 || parsed == 128) ? parsed : 0;
}

// `MLXCEL_TRACE_QMM_TILE`: print the selected tile together with the register
// count and achieved occupancy of the compiled kernel. Runs inside the
// `get_kernel` configure callback, so it fires once per distinct kernel rather
// than once per launch. This is the instrument the #1541 occupancy sweep reads,
// standing in for `-Xptxas -v`, which a runtime-JIT kernel never passes through.
void trace_kernel_occupancy(
    CUfunction kernel,
    const std::string& kernel_name,
    const mlxcel::QmmNaiveTile& tile,
    uint32_t threads,
    size_t smem_bytes) {
  if (std::getenv("MLXCEL_TRACE_QMM_TILE") == nullptr) {
    return;
  }
  int num_regs = 0;
  int static_smem = 0;
  int local_bytes = 0;
  int blocks_per_sm = 0;
  cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);
  cuFuncGetAttribute(&static_smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel);
  // Per-thread local memory, which for this kernel is register spill.
  cuFuncGetAttribute(&local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);
  cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, static_cast<int>(threads), smem_bytes);

  auto line = fmt::format(
      "[mlxcel qmm_naive] tile {}x{}x{} threads {} dyn_smem {} B static_smem {} B "
      "regs {} spill {} B blocks/SM {} warps/SM {} :: {}\n",
      tile.tile_m,
      tile.tile_n,
      tile.tile_k,
      threads,
      smem_bytes,
      static_smem,
      num_regs,
      local_bytes,
      blocks_per_sm,
      blocks_per_sm * static_cast<int>(threads) / 32,
      kernel_name);
  std::fputs(line.c_str(), stderr);
}

} // namespace

void qmm_naive(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  auto [m, n, k, l, broadcast_b] = make_problem_shape(x, w, out);
  // Unchanged: this is the kernel's `SM80` template parameter, which
  // `make_tiled_mma` reads to choose between an `SM80_16x8x16` tensor-core atom
  // and `UniversalFMA`. The tile selector takes it under that meaning rather
  // than as a shared-memory proxy.
  bool sm80 = encoder.device().compute_capability_major() >= 8;
  auto tile = mlxcel::qmm_naive_choose_tile(
      x.itemsize(),
      m,
      group_size,
      max_smem_per_block_optin(encoder.device().cuda_device()),
      /* tensor_core_mma = */ sm80,
      forced_tile_n());
  if (!tile.fits) {
    throw std::runtime_error(fmt::format(
        "[qmm_naive] no CTA tile fits this device: a {}x{}x{} tile at "
        "itemsize {} needs {} bytes of shared memory per block and the device "
        "budget is {} bytes. Refusing to launch rather than failing inside the "
        "driver.",
        tile.tile_m,
        tile.tile_n,
        tile.tile_k,
        x.itemsize(),
        tile.smem_bytes,
        max_smem_per_block_optin(encoder.device().cuda_device())));
  }
  auto cta_tiler = cute::make_shape(tile.tile_m, tile.tile_n, tile.tile_k);
  bool has_k_residue = (k % cute::size<2>(cta_tiler)) != 0;

  // The N tile is part of the module name because the module name is the only
  // key for the in-process module cache and for the persistent PTX disk cache,
  // while the tile shape reaches the compiler only through the kernel name. A
  // module cached under a name that does not pin tile_n would be handed back
  // with no kernel matching the requested instantiation.
  std::string module_name = fmt::format(
      "qmm_naive_t{}_{}_{}_m{}_n{}_b{}_g{}_{}",
      transpose ? "n" : "t",
      has_k_residue ? "residue" : "aligned",
      dtype_to_string(x.dtype()),
      cute::size<0>(cta_tiler),
      cute::size<1>(cta_tiler),
      bits,
      group_size,
      quantization_mode_to_string(mode));

  auto [ctype_x, ctype_q, ctype_s] = get_qmm_cutlass_types(x, bits, mode);
  std::string kernel_name = fmt::format(
      "mlx::core::cu::qmm_naive_kernel<{}, {}, {}, {}, {}, {}, {}, {}>",
      group_size,
      transpose,
      has_k_residue,
      sm80,
      ctype_x,
      ctype_q,
      ctype_s,
      cta_tiler_to_string(cta_tiler));

  cu::JitModule& mod = cu::get_jit_module(encoder.device(), module_name, [&]() {
    return std::make_tuple(
        false, jit_source_qmm_naive, std::vector{kernel_name});
  });

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  if (biases) {
    encoder.set_input_array(*biases);
  }
  if (global_scale) {
    encoder.set_input_array(*global_scale);
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
  dim3 block_dims{uint32_t(cute::size(cu::make_tiled_mma(cta_tiler)))};

  auto [sA_layout, sB_layout] = cu::make_smem_layouts(cta_tiler);
  size_t smem_bytes =
      x.itemsize() * (cute::cosize(sA_layout) + cute::cosize(sB_layout));

  // The selector predicted this number from the tile extents alone. If CuTe
  // now disagrees, the layouts changed under an MLX pin bump and every fit and
  // opt-in decision above was made against the wrong figure, so say so instead
  // of launching on it.
  if (static_cast<long long>(smem_bytes) != tile.smem_bytes) {
    throw std::runtime_error(fmt::format(
        "[qmm_naive] shared-memory model drift: qmm_naive_tile.h predicts {} "
        "bytes for a {}x{}x{} tile at itemsize {}, the CuTe layouts need {}. "
        "Update qmm_naive_smem_bytes() to match the current MLX pin.",
        tile.smem_bytes,
        tile.tile_m,
        tile.tile_n,
        tile.tile_k,
        x.itemsize(),
        smem_bytes));
  }

  auto kernel = mod.get_kernel(kernel_name, [&](CUfunction kernel) {
    // Anything above the opt-in-free ceiling has to raise the kernel's dynamic
    // shared-memory maximum before it is ever launched. Omitting this is a
    // launch failure, not a load or compile error.
    if (tile.needs_smem_opt_in) {
      CHECK_CUDA_ERROR(cuFuncSetAttribute(
          kernel,
          CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          static_cast<int>(smem_bytes)));
    }
    trace_kernel_occupancy(kernel, kernel_name, tile, block_dims.x, smem_bytes);
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
      global_scale ? gpu_ptr<void>(*global_scale) : nullptr,
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
