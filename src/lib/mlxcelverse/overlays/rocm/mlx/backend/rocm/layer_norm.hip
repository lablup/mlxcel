// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>

#include <climits>
#include <string>

namespace mlx::core {

namespace rocm {

// The 64-bit row of this block under row_grid().
static __device__ inline int64_t row_grid_index() {
  return static_cast<int64_t>(blockIdx.x) +
      static_cast<int64_t>(blockIdx.y) * gridDim.x;
}

// Warp reduce for sum
__device__ float warp_reduce_sum_f(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

// Warp reduce for float3 (sum, sum*t, t*t)
struct float3_sum {
  float x, y, z;
};

__device__ float3_sum warp_reduce_sum_f3(float3_sum val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val.x += __shfl_xor(val.x, offset);
    val.y += __shfl_xor(val.y, offset);
    val.z += __shfl_xor(val.z, offset);
  }
  return val;
}

template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void layer_norm_kernel(
    const T* x,
    const T* w,
    const T* b,
    T* out,
    float eps,
    int32_t axis_size,
    int64_t w_stride,
    int64_t b_stride,
    int64_t n_rows) {
  int64_t row = row_grid_index();
  if (row >= n_rows) {
    return;
  }
  
  x += row * axis_size;
  out += row * axis_size;

  // Sum for mean
  float sum = 0;
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      sum += static_cast<float>(x[i + j]);
    }
  }

  // Block reduce for sum
  __shared__ float shared_sum[BLOCK_DIM / WARP_SIZE + 1];
  
  float warp_sum = warp_reduce_sum_f(sum);
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    sum = (lane < (BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE) ? shared_sum[lane] : 0;
    sum = warp_reduce_sum_f(sum);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_sum[0] = sum;
  }
  __syncthreads();
  float mean = shared_sum[0] / axis_size;
  // All waves must read the mean before shared_sum is reused for variance.
  __syncthreads();

  // Compute variance
  float var_sum = 0;
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      float t = static_cast<float>(x[i + j]) - mean;
      var_sum += t * t;
    }
  }

  // Block reduce for variance
  warp_sum = warp_reduce_sum_f(var_sum);
  
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    var_sum = (lane < (BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE) ? shared_sum[lane] : 0;
    var_sum = warp_reduce_sum_f(var_sum);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_sum[0] = var_sum;
  }
  __syncthreads();
  // Use 1/sqrt instead of rsqrtf for IEEE-compliant precision
  // (matches Metal's metal::precise::rsqrt behavior)
  float normalizer = 1.0f / sqrtf(shared_sum[0] / axis_size + eps);

  // Write output
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      int idx = i + j;
      float norm = (static_cast<float>(x[idx]) - mean) * normalizer;
      float wi = (w_stride == 0) ? static_cast<float>(w[0]) : static_cast<float>(w[idx * w_stride]);
      float bi = (b_stride == 0) ? static_cast<float>(b[0]) : static_cast<float>(b[idx * b_stride]);
      out[idx] = static_cast<T>(wi * norm + bi);
    }
  }
}

template <typename T, bool HAS_W, int BLOCK_DIM, int N_READS = 4>
__global__ void layer_norm_vjp_kernel(
    const T* x,
    const T* w,
    const T* g,
    T* gx,
    T* gw,
    float eps,
    int32_t axis_size,
    int64_t w_stride,
    int64_t n_rows) {
  int64_t row = row_grid_index();
  if (row >= n_rows) {
    return;
  }
  
  x += row * axis_size;
  g += row * axis_size;
  gx += row * axis_size;
  gw += row * axis_size;

  // Sum for mean
  float sum = 0;
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      sum += static_cast<float>(x[i + j]);
    }
  }

  // Block reduce for sum
  __shared__ float shared_sum[BLOCK_DIM / WARP_SIZE + 1];
  __shared__ float3_sum shared_f3[BLOCK_DIM / WARP_SIZE + 1];
  
  float warp_sum = warp_reduce_sum_f(sum);
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    sum = (lane < (BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE) ? shared_sum[lane] : 0;
    sum = warp_reduce_sum_f(sum);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_sum[0] = sum;
  }
  __syncthreads();
  float mean = shared_sum[0] / axis_size;

  // Compute factors: (wg_sum, wg*xc_sum, xc^2_sum)
  float3_sum factors = {0, 0, 0};
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      int idx = i + j;
      float t = static_cast<float>(x[idx]) - mean;
      float wi = (w_stride == 0) ? static_cast<float>(w[0]) : static_cast<float>(w[idx * w_stride]);
      float gi = static_cast<float>(g[idx]);
      float wg = wi * gi;
      factors.x += wg;
      factors.y += wg * t;
      factors.z += t * t;
    }
  }

  // Block reduce for factors
  float3_sum warp_f3 = warp_reduce_sum_f3(factors);
  
  if (lane == 0) {
    shared_f3[warp_id] = warp_f3;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    factors = (lane < (BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE) ? shared_f3[lane] : float3_sum{0, 0, 0};
    factors = warp_reduce_sum_f3(factors);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_f3[0] = factors;
  }
  __syncthreads();
  factors = shared_f3[0];
  
  float meanwg = factors.x / axis_size;
  float meanwgxc = factors.y / axis_size;
  float normalizer2 = 1.0f / (factors.z / axis_size + eps);
  float normalizer = sqrtf(normalizer2);

  // Write outputs
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      int idx = i + j;
      float xi_centered = static_cast<float>(x[idx]) - mean;
      float xi_norm = xi_centered * normalizer;
      float wi = (w_stride == 0) ? static_cast<float>(w[0]) : static_cast<float>(w[idx * w_stride]);
      float gi = static_cast<float>(g[idx]);
      
      // Gradient for x
      gx[idx] = static_cast<T>(normalizer * (wi * gi - meanwg) - xi_norm * meanwgxc * normalizer2);
      
      // Gradient for w (per-element, will be reduced later)
      if constexpr (HAS_W) {
        gw[idx] = static_cast<T>(gi * xi_norm);
      }
    }
  }
}

// One block per row, split over grid x and y (lablup/mlxcel#2619, the ROCm side
// of ml-explore/mlx#4717). HIP refuses gridDim.x * blockDim.x >= 2^32, so grid x
// stops at UINT32_MAX / block_dim and the remaining rows spill into y; below
// that cap the grid is dim3(n_rows) as before. Kernels recover the row with
// row_grid_index() and return when it is >= n_rows.
static inline dim3 row_grid(int64_t n_rows, int block_dim) {
  const int64_t max_blocks_x = static_cast<int64_t>(UINT32_MAX) / block_dim;
  int64_t grid_x = std::max<int64_t>(1, std::min(n_rows, max_blocks_x));
  int64_t grid_y = (n_rows + grid_x - 1) / grid_x;
  return dim3(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y), 1);
}

// lablup/mlxcel#2633: the VJP shapes gw_temp and the column-reduce plans with
// 32-bit dimensions (Shape holds int32), so it cannot reduce more than INT_MAX
// rows. Say so instead of wrapping the row count.
static inline void check_vjp_rows(const char* name, int64_t n_rows) {
  if (n_rows > static_cast<int64_t>(INT_MAX)) {
    throw std::invalid_argument(
        std::string("[") + name + "] " + std::to_string(n_rows) +
        " rows exceed INT_MAX, the most rows the weight and bias gradient "
        "reductions take");
  }
}

// lablup/mlxcel#2633: the gradient of a weight or bias over zero rows is zero.
static inline void fill_zeros(
    array& out, rocm::CommandEncoder& encoder, const Stream& s) {
  if (out.size() == 0) {
    out.set_data(mlx::core::rocm::malloc_async(0, encoder));
    return;
  }
  array zero(0, out.dtype());
  encoder.add_temporary(zero);
  fill_gpu(zero, out, s);
}

} // namespace rocm

namespace fast {

bool LayerNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& out = outputs[0];
  auto& encoder = rocm::get_command_encoder(s);

  // Make sure that the last dimension is contiguous.
  auto set_output = [&s, &out, &encoder](const array& x) {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            mlx::core::rocm::malloc_async(x.data_size() * x.itemsize(), encoder),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  const array x = set_output(inputs[0]);
  const array& w = inputs[1];
  const array& b = inputs[2];

  // Zero rows (lablup/mlxcel#2633): row_grid(0) is not a valid launch.
  if (out.size() == 0) {
    return;
  }

  int32_t axis_size = x.shape().back();
  int64_t n_rows = x.data_size() / axis_size;
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
  int64_t b_stride = (b.ndim() == 1) ? b.strides()[0] : 0;

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  
  constexpr int BLOCK_DIM = 256;
  constexpr int N_READS = 4;
  
  switch (out.dtype()) {
    case float32:
      encoder.add_kernel_node(
          &rocm::layer_norm_kernel<float, BLOCK_DIM, N_READS>,
          rocm::row_grid(n_rows, BLOCK_DIM), dim3(BLOCK_DIM), 0,
          gpu_ptr<float>(x), gpu_ptr<float>(w), gpu_ptr<float>(b), gpu_ptr<float>(out),
          eps_, axis_size, w_stride, b_stride, n_rows);
      break;
    case float16:
      encoder.add_kernel_node(
          &rocm::layer_norm_kernel<__half, BLOCK_DIM, N_READS>,
          rocm::row_grid(n_rows, BLOCK_DIM), dim3(BLOCK_DIM), 0,
          gpu_ptr<__half>(x), gpu_ptr<__half>(w), gpu_ptr<__half>(b), gpu_ptr<__half>(out),
          eps_, axis_size, w_stride, b_stride, n_rows);
      break;
    case bfloat16:
      encoder.add_kernel_node(
          &rocm::layer_norm_kernel<hip_bfloat16, BLOCK_DIM, N_READS>,
          rocm::row_grid(n_rows, BLOCK_DIM), dim3(BLOCK_DIM), 0,
          gpu_ptr<hip_bfloat16>(x), gpu_ptr<hip_bfloat16>(w), gpu_ptr<hip_bfloat16>(b), gpu_ptr<hip_bfloat16>(out),
          eps_, axis_size, w_stride, b_stride, n_rows);
      break;
    default:
      throw std::runtime_error("Unsupported type for layer_norm");
  }
}

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  // Before any copy: a gradient reduction past INT_MAX rows cannot be planned.
  if (inputs[0].size() > 0) {
    int64_t rows = inputs[0].size() / inputs[0].shape().back();
    bool reduces = inputs[1].ndim() != 0 ||
        (outputs[2].ndim() == 1 && outputs[2].size() == inputs[0].shape().back());
    if (reduces) {
      rocm::check_vjp_rows("LayerNormVJP", rows);
    }
  }

  // Ensure row contiguity
  auto check_input = [&s](const array& x, bool& copied) {
    if (x.flags().row_contiguous) {
      copied = false;
      return x;
    }
    copied = true;
    return contiguous_copy_gpu(x, s);
  };
  
  bool donate_x = inputs[0].is_donatable();
  bool donate_g = inputs[3].is_donatable();
  bool copied;
  auto x = check_input(inputs[0], copied);
  donate_x |= copied;
  const array& w = inputs[1];
  const array& b = inputs[2];
  bool g_copied;
  auto g = check_input(inputs[3], g_copied);
  donate_g |= g_copied;
  array& gx = outputs[0];
  array& gw = outputs[1];
  array& gb = outputs[2];

  // Check whether we had a weight
  bool has_w = w.ndim() != 0;

  // Allocate space for the outputs
  bool g_in_gx = false;
  if (donate_x) {
    gx.copy_shared_buffer(x);
  } else if (donate_g) {
    gx.copy_shared_buffer(g);
    g_in_gx = true;
  } else {
    gx.set_data(mlx::core::rocm::malloc_async(gx.nbytes(), encoder));
  }
  if (g_copied && !g_in_gx) {
    encoder.add_temporary(g);
  }

  int32_t axis_size = x.shape().back();

  // The gradient for b in case we had a b
  bool has_gb = (gb.ndim() == 1 && gb.size() == axis_size);

  // Zero rows (lablup/mlxcel#2633): gx is empty and the weight and bias
  // gradients are sums over no rows.
  if (x.size() == 0) {
    if (has_w) {
      rocm::fill_zeros(gw, encoder, s);
    }
    if (has_gb) {
      rocm::fill_zeros(gb, encoder, s);
    }
    return;
  }

  int64_t n_rows = x.data_size() / axis_size;
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;

  // Allocate a temporary to store the gradients for w
  array gw_temp =
      (has_w) ? array({static_cast<int>(n_rows), x.shape().back()}, gw.dtype(), nullptr, {}) : w;
  bool g_in_gw = false;
  if (has_w) {
    if (!g_in_gx && donate_g) {
      g_in_gw = true;
      gw_temp.copy_shared_buffer(g);
    } else {
      gw_temp.set_data(mlx::core::rocm::malloc_async(gw_temp.nbytes(), encoder));
      encoder.add_temporary(gw_temp);
    }
  }

  // The gradient for b: the sum of g over every row (lablup/mlxcel#2632, as
  // in the CUDA backend). It is reduced before the VJP kernel is enqueued,
  // because that kernel overwrites a g donated into gx or gw_temp. g is viewed
  // as [n_rows, axis_size] so that the column reduce runs over every leading
  // axis of an N-D cotangent.
  if (has_gb) {
    array g_rows({static_cast<int>(n_rows), axis_size}, g.dtype(), nullptr, {});
    g_rows.copy_shared_buffer(
        g, {static_cast<int64_t>(axis_size), 1}, g.flags(), g.data_size());
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {static_cast<int>(n_rows)}, {axis_size});
    col_reduce(encoder, g_rows, gb, Reduce::ReduceType::Sum, {0}, plan);
  }
  if ((g_in_gx || g_in_gw) && has_gb) {
    encoder.set_input_array(gb);
  }

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(g);
  encoder.set_output_array(gx);
  encoder.set_output_array(gw_temp);
  
  constexpr int BLOCK_DIM = 256;
  constexpr int N_READS = 4;
  
  if (has_w) {
    switch (gx.dtype()) {
      case float32:
        encoder.add_kernel_node(
            &rocm::layer_norm_vjp_kernel<float, true, BLOCK_DIM, N_READS>,
            rocm::row_grid(n_rows, BLOCK_DIM), dim3(BLOCK_DIM), 0,
            gpu_ptr<float>(x), gpu_ptr<float>(w), gpu_ptr<float>(g),
            gpu_ptr<float>(gx), gpu_ptr<float>(gw_temp),
            eps_, axis_size, w_stride, n_rows);
        break;
      case float16:
        encoder.add_kernel_node(
            &rocm::layer_norm_vjp_kernel<__half, true, BLOCK_DIM, N_READS>,
            rocm::row_grid(n_rows, BLOCK_DIM), dim3(BLOCK_DIM), 0,
            gpu_ptr<__half>(x), gpu_ptr<__half>(w), gpu_ptr<__half>(g),
            gpu_ptr<__half>(gx), gpu_ptr<__half>(gw_temp),
            eps_, axis_size, w_stride, n_rows);
        break;
      case bfloat16:
        encoder.add_kernel_node(
            &rocm::layer_norm_vjp_kernel<hip_bfloat16, true, BLOCK_DIM, N_READS>,
            rocm::row_grid(n_rows, BLOCK_DIM), dim3(BLOCK_DIM), 0,
            gpu_ptr<hip_bfloat16>(x), gpu_ptr<hip_bfloat16>(w), gpu_ptr<hip_bfloat16>(g),
            gpu_ptr<hip_bfloat16>(gx), gpu_ptr<hip_bfloat16>(gw_temp),
            eps_, axis_size, w_stride, n_rows);
        break;
      default:
        throw std::runtime_error("Unsupported type for layer_norm_vjp");
    }
  } else {
    switch (gx.dtype()) {
      case float32: {
        float* gw_null = nullptr;
        encoder.add_kernel_node(
            &rocm::layer_norm_vjp_kernel<float, false, BLOCK_DIM, N_READS>,
            rocm::row_grid(n_rows, BLOCK_DIM), dim3(BLOCK_DIM), 0,
            gpu_ptr<float>(x), gpu_ptr<float>(w), gpu_ptr<float>(g),
            gpu_ptr<float>(gx), gw_null,
            eps_, axis_size, w_stride, n_rows);
        break;
      }
      case float16: {
        __half* gw_null = nullptr;
        encoder.add_kernel_node(
            &rocm::layer_norm_vjp_kernel<__half, false, BLOCK_DIM, N_READS>,
            rocm::row_grid(n_rows, BLOCK_DIM), dim3(BLOCK_DIM), 0,
            gpu_ptr<__half>(x), gpu_ptr<__half>(w), gpu_ptr<__half>(g),
            gpu_ptr<__half>(gx), gw_null,
            eps_, axis_size, w_stride, n_rows);
        break;
      }
      case bfloat16: {
        hip_bfloat16* gw_null = nullptr;
        encoder.add_kernel_node(
            &rocm::layer_norm_vjp_kernel<hip_bfloat16, false, BLOCK_DIM, N_READS>,
            rocm::row_grid(n_rows, BLOCK_DIM), dim3(BLOCK_DIM), 0,
            gpu_ptr<hip_bfloat16>(x), gpu_ptr<hip_bfloat16>(w), gpu_ptr<hip_bfloat16>(g),
            gpu_ptr<hip_bfloat16>(gx), gw_null,
            eps_, axis_size, w_stride, n_rows);
        break;
      }
      default:
        throw std::runtime_error("Unsupported type for layer_norm_vjp");
    }
  }

  // Reduce gw_temp to gw if we have weights
  if (has_w) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {static_cast<int>(n_rows)}, {axis_size});
    col_reduce(encoder, gw_temp, gw, Reduce::ReduceType::Sum, {0}, plan);
  }
}

} // namespace fast

} // namespace mlx::core
