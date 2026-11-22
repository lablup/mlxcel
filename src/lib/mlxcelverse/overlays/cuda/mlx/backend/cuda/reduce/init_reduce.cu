// Copyright © 2025 Apple Inc.
//
// CUDA patch: bf16 reduce accumulation optimization
//
// Modified from upstream MLX 9a795735 mlx/backend/cuda/reduce/init_reduce.cu
//
// Changes:
//   - Uses ReduceOutput<OP, T> for the output pointer instead of U
//     (accumulation type), and casts the init value to it before writing
//   - This ensures the output buffer matches out.dtype(): bf16 for a bf16
//     Sum/Prod (whose U is fp32), and the accumulation type otherwise (bool
//     for And/Or, a 32-bit integer for Sum/Prod over bool)

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/reduce/reduce.cuh"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

// `T` is the input type and `V` the type of the output array, which is not
// `T` for sum(bool), any(float32) and the like (see `ReduceOutput`).
template <typename T, typename V, typename Op>
__global__ void init_reduce(V* out, size_t size) {
  auto index = cg::this_grid().thread_rank();
  if (index < size) {
    out[index] = cast_to<V>(ReduceInit<Op, T>::value());
  }
}

} // namespace cu

void init_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type) {
  // Allocate if needed
  if (out.data_shared_ptr() == nullptr) {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
  }

  if (out.size() == 0) {
    return;
  }

  encoder.set_output_array(out);
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      using OP = MLX_GET_TYPE(reduce_type_tag);
      using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
      using V = typename cu::ReduceOutput<OP, T>::type;
      auto kernel = cu::init_reduce<T, V, OP>;
      dim3 grid = get_2d_grid_dims(out.shape(), out.strides());
      dim3 block(grid.x < 1024 ? grid.x : 1024, 1, 1);
      grid.x = (grid.x + 1023) / 1024;
      encoder.add_kernel_node(kernel, grid, block, gpu_ptr<V>(out), out.size());
    });
  });
}

} // namespace mlx::core
