// Copyright © 2025 Apple Inc.

// mlxcelverse: ROCm Hadamard transform (lablup/mlxcel#1825).
//
// The CUDA backend builds this kernel through NVRTC with a generated codelet
// for the radix-m matrix, which lets it cover every size MLX accepts, that is
// n = m * 2^k for m in {1, 12, 20, 28}. This port is ahead-of-time compiled and
// covers m == 1 with n a power of two up to 8192, which is the whole of what
// mlxcel reaches: `wht()` in mlxcel-core asserts a power-of-two last axis, and
// the KV cache advisor demotes Turbo modes for models whose head_dim is not one,
// so the sizes that arrive here are head dims, chiefly 64, 128 and 256. The
// n > 8192 case is where the CUDA path splits the transform into two stages;
// this file throws for it rather than returning a wrong answer.
//
// The scale comes from the primitive, not from this file. MLX defaults it to
// 1/sqrt(n), the orthonormal scaling that makes the transform an involution,
// and mlxcel relies on that: `wht(wht(x))` must return `x`. A kernel that
// hardcoded a different scale would pass a round-trip test that only checks
// shape or finiteness while silently corrupting KV values.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/common/hadamard.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

namespace rocm {

// Largest transform this single-stage kernel accepts. It is bounded by shared
// memory: the buffer holds `n` floats, so 8192 needs 32 KiB of the 64 KiB a
// gfx11 workgroup has. It is also the size at which the CUDA path switches to
// its two-stage decomposition, so the two backends agree on where the simple
// path ends.
constexpr int MAX_HADAMARD_N = 8192;
constexpr int HADAMARD_BLOCK_THREADS = 256;

// One block per transform, grid-stride over transforms so a capped grid still
// covers every row. Accumulation is in float regardless of the element type,
// which matches what the CUDA kernel does and keeps the fp16 round-trip inside
// a single rounding of the input.
template <typename T>
__global__ void hadamard_pow2_kernel(
    const T* in,
    T* out,
    float scale,
    int n,
    int64_t num_transforms) {
  extern __shared__ float buf[];

  for (int64_t transform = blockIdx.x; transform < num_transforms;
       transform += gridDim.x) {
    const int64_t base = transform * static_cast<int64_t>(n);

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
      buf[idx] = static_cast<float>(in[base + idx]);
    }
    __syncthreads();

    // Iterative radix-2 butterfly. At stage `h` the pair (j, j + h) is
    // combined, where j skips the elements already paired at this stage.
    for (int h = 1; h < n; h <<= 1) {
      for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
        const int k = i & (h - 1);
        const int j = ((i - k) << 1) + k;
        const float a = buf[j];
        const float b = buf[j + h];
        buf[j] = a + b;
        buf[j + h] = a - b;
      }
      __syncthreads();
    }

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
      out[base + idx] = static_cast<T>(buf[idx] * scale);
    }
    __syncthreads();
  }
}

} // namespace rocm

namespace {

void hadamard_pow2_contiguous(
    const array& x,
    array& y,
    int n,
    float scale,
    const Stream& s) {
  auto& encoder = rocm::get_command_encoder(s);

  const int64_t num_transforms = x.size() / n;
  // Clamp like the CUDA backend does; the kernel is grid-stride, so the two
  // together cover the whole input.
  const int num_blocks =
      static_cast<int>(std::min<int64_t>(num_transforms, 65535));
  const int block_threads = std::min(rocm::HADAMARD_BLOCK_THREADS, n / 2);
  const size_t shared_bytes = static_cast<size_t>(n) * sizeof(float);

  encoder.set_input_array(x);
  encoder.set_output_array(y);

  dispatch_float_types(x.dtype(), "[hadamard]", [&](auto type_tag) {
    using T = hip_type_t<MLX_GET_TYPE(type_tag)>;
    encoder.add_kernel_node(
        &rocm::hadamard_pow2_kernel<T>,
        dim3(num_blocks),
        dim3(block_threads),
        shared_bytes,
        gpu_ptr<T>(x),
        gpu_ptr<T>(y),
        scale,
        n,
        num_transforms);
  });
}

} // namespace

void Hadamard::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  auto& in = inputs[0];
  if (in.dtype() != float16 && in.dtype() != bfloat16 &&
      in.dtype() != float32) {
    throw std::invalid_argument("[hadamard] Unsupported type.");
  }

  // n = m * 2^k where m is in {1, 12, 20, 28}.
  auto [n, m] = decompose_hadamard(in.shape().back());
  if (m != 1 || n > rocm::MAX_HADAMARD_N) {
    std::ostringstream msg;
    msg << "[hadamard] ROCm supports a power-of-two last axis up to "
        << rocm::MAX_HADAMARD_N << "; got " << in.shape().back()
        << " (m=" << m << ", n=" << n << "). See lablup/mlxcel#1825.";
    throw std::invalid_argument(msg.str());
  }

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  if (in.flags().row_contiguous) {
    if (in.is_donatable()) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
    }
    hadamard_pow2_contiguous(in, out, n, scale_, s);
  } else {
    copy_gpu(in, out, CopyType::General, s);
    hadamard_pow2_contiguous(out, out, n, scale_, s);
  }
}

} // namespace mlx::core
