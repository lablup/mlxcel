// Copyright © 2025 Apple Inc.
//
// CUDA patch: bf16 reduce accumulation optimization
//
// Modified from upstream MLX 59d600b5e mlx/backend/cuda/reduce/all_reduce.cu
//
// Changes:
//   - Added output type template parameter V (separate from accumulation type U)
//   - Kernel writes static_cast<V>(accumulated) to output, enabling fp32
//     accumulation with bf16 output for Sum/Prod reductions
//   - Dispatch code passes T (input type) as output type for final output,
//     U (accumulation type) for multi-block intermediates

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/reduce/reduce.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_load.cuh>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename T, typename U, typename V, typename ReduceOp, int N = 4>
__global__ void all_reduce(T* in, V* out, size_t block_step, size_t size) {
  // TODO: Process multiple "rows" in each thread
  constexpr int M = 1;

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  const U init = cu::ReduceInit<ReduceOp, T>::value();
  ReduceOp op;

  T vals[N];
  U accs[M];
  accs[0] = init;

  size_t start = grid.block_rank() * block_step;
  size_t end = start + block_step;
  size_t check = min(end, size);

  size_t i = start;
  for (; i + block.size() * N <= check; i += block.size() * N) {
    cub::LoadDirectBlockedVectorized<T, N>(block.thread_rank(), in + i, vals);
    for (int j = 0; j < N; j++) {
      accs[0] = op(accs[0], cast_to<U>(vals[j]));
    }
  }

  if (i < check) {
    cub::LoadDirectBlocked(
        block.thread_rank(), in + i, vals, check - i, cast_to<T>(init));
    for (int i = 0; i < N; i++) {
      accs[0] = op(accs[0], cast_to<U>(vals[i]));
    }
  }

  __shared__ U shared_accumulators[32];
  block_reduce(block, warp, accs, shared_accumulators, op, init);

  if (block.thread_rank() == 0) {
    out[grid.block_rank()] = cast_to<V>(accs[0]);
  }
}

} // namespace cu

void all_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type) {
  constexpr int N_READS = 8;

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  auto get_args = [](size_t size, size_t N) {
    int threads =
        static_cast<int>(std::min(size_t{512}, cuda::ceil_div(size, N)));
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    size_t reductions_per_step = threads * N;
    size_t steps_needed = cuda::ceil_div(size, reductions_per_step);

    int blocks;
    if (steps_needed < 32) {
      blocks = 1;
    } else if (steps_needed < 128) {
      blocks = 32;
    } else if (steps_needed < 512) {
      blocks = 128;
    } else if (steps_needed < 1024) {
      blocks = 512;
    } else {
      blocks = 1024;
    }

    size_t steps_per_block = cuda::ceil_div(steps_needed, blocks);
    size_t block_step = steps_per_block * reductions_per_step;

    return std::make_tuple(blocks, threads, block_step);
  };

  int blocks, threads;
  size_t block_step;
  size_t insize = in.size();
  void* indata = gpu_ptr<void>(in);

  // Large array so allocate an intermediate and accumulate there
  std::tie(blocks, threads, block_step) = get_args(insize, N_READS);
  encoder.set_input_array(in);

  // Single dispatch on input type handles both single-pass and multi-pass
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      using OP = MLX_GET_TYPE(reduce_type_tag);
      using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
      using U = typename cu::ReduceResult<OP, T>::type;

      if (blocks > 1) {
        // Multi-block: first pass accumulates into fp32 intermediate
        // Determine intermediate dtype from accumulation type
        Dtype intermediate_dtype = out.dtype();
        if constexpr (std::is_same_v<U, float> && !std::is_same_v<T, float>) {
          intermediate_dtype = float32;
        }

        array intermediate({blocks}, intermediate_dtype, nullptr, {});
        intermediate.set_data(
            cu::malloc_async(intermediate.nbytes(), encoder));
        encoder.add_temporary(intermediate);
        encoder.set_output_array(intermediate);

        // First pass: T input -> U intermediate (V=U for intermediate)
        auto kernel_pass1 = cu::all_reduce<T, U, U, OP, N_READS>;
        encoder.add_kernel_node(
            kernel_pass1,
            blocks,
            threads,
            indata,
            gpu_ptr<U>(intermediate),
            block_step,
            insize);

        // Recalculate for second pass over the intermediate
        auto indata2 = gpu_ptr<U>(intermediate);
        size_t insize2 = intermediate.size();
        int blocks2, threads2;
        size_t block_step2;
        std::tie(blocks2, threads2, block_step2) =
            get_args(insize2, N_READS);

        encoder.set_input_array(intermediate);
        encoder.set_output_array(out);

        // Second pass: U intermediate -> T output (V=T for final output)
        auto kernel_pass2 = cu::all_reduce<U, U, T, OP, N_READS>;
        encoder.add_kernel_node(
            kernel_pass2,
            blocks2,
            threads2,
            indata2,
            gpu_ptr<T>(out),
            block_step2,
            insize2);
      } else {
        // Single block: T input -> T output with U accumulation
        encoder.set_output_array(out);
        auto kernel = cu::all_reduce<T, U, T, OP, N_READS>;
        encoder.add_kernel_node(
            kernel,
            blocks,
            threads,
            indata,
            gpu_ptr<T>(out),
            block_step,
            insize);
      }
    });
  });
}

} // namespace mlx::core
