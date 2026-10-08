// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/utils.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace mlx::core {

namespace rocm {

// Cast operation for copy - general case
template <typename SrcT, typename DstT, typename = void>
struct CastOp {
  static constexpr bool is_castable = std::is_convertible_v<SrcT, DstT>;

  __device__ DstT operator()(SrcT x) {
    return static_cast<DstT>(x);
  }
};

// Castings between complex and boolean
template <>
struct CastOp<hipFloatComplex, bool> {
  static constexpr bool is_castable = true;

  __device__ bool operator()(hipFloatComplex x) {
    return x.x != 0 && x.y != 0;
  }
};

template <>
struct CastOp<bool, hipFloatComplex> {
  static constexpr bool is_castable = true;

  __device__ hipFloatComplex operator()(bool x) {
    return x ? make_hipFloatComplex(1.0f, 1.0f)
             : make_hipFloatComplex(0.0f, 0.0f);
  }
};

// Converting a complex number to real number discards the imaginary part
template <typename DstT>
struct CastOp<
    hipFloatComplex,
    DstT,
    std::enable_if_t<!is_complex_v<DstT> && !std::is_same_v<DstT, bool>>> {
  static constexpr bool is_castable = true;

  __device__ DstT operator()(hipFloatComplex x) {
    return static_cast<DstT>(x.x); // x.x is the real part
  }
};

// Allow converting a real number to complex number
template <typename SrcT>
struct CastOp<
    SrcT,
    hipFloatComplex,
    std::enable_if_t<!is_complex_v<SrcT> && !std::is_same_v<SrcT, bool>>> {
  static constexpr bool is_castable = true;

  __device__ hipFloatComplex operator()(SrcT x) {
    return make_hipFloatComplex(static_cast<float>(x), 0.0f);
  }
};

// Do nothing when no casting is needed
template <typename T>
struct CastOp<T, T, void> {
  static constexpr bool is_castable = true;

  __device__ T operator()(T x) {
    return x;
  }
};

// Specializations for half types
template <>
struct CastOp<__half, float> {
  static constexpr bool is_castable = true;
  __device__ float operator()(__half x) {
    return __half2float(x);
  }
};

template <>
struct CastOp<float, __half> {
  static constexpr bool is_castable = true;
  __device__ __half operator()(float x) {
    return __float2half(x);
  }
};

template <>
struct CastOp<hip_bfloat16, float> {
  static constexpr bool is_castable = true;
  __device__ float operator()(hip_bfloat16 x) {
    return static_cast<float>(x);
  }
};

template <>
struct CastOp<float, hip_bfloat16> {
  static constexpr bool is_castable = true;
  __device__ hip_bfloat16 operator()(float x) {
    return hip_bfloat16(x);
  }
};

// Conversions through float for half types
template <typename DstT>
struct CastOp<
    __half,
    DstT,
    std::enable_if_t<
        !std::is_same_v<DstT, __half> && !std::is_same_v<DstT, float> &&
        !is_complex_v<DstT>>> {
  static constexpr bool is_castable = true;
  __device__ DstT operator()(__half x) {
    return static_cast<DstT>(__half2float(x));
  }
};

template <typename SrcT>
struct CastOp<
    SrcT,
    __half,
    std::enable_if_t<
        !std::is_same_v<SrcT, __half> && !std::is_same_v<SrcT, float> &&
        !is_complex_v<SrcT>>> {
  static constexpr bool is_castable = true;
  __device__ __half operator()(SrcT x) {
    return __float2half(static_cast<float>(x));
  }
};

template <typename DstT>
struct CastOp<
    hip_bfloat16,
    DstT,
    std::enable_if_t<
        !std::is_same_v<DstT, hip_bfloat16> && !std::is_same_v<DstT, float> &&
        !is_complex_v<DstT>>> {
  static constexpr bool is_castable = true;
  __device__ DstT operator()(hip_bfloat16 x) {
    return static_cast<DstT>(static_cast<float>(x));
  }
};

template <typename SrcT>
struct CastOp<
    SrcT,
    hip_bfloat16,
    std::enable_if_t<
        !std::is_same_v<SrcT, hip_bfloat16> && !std::is_same_v<SrcT, float> &&
        !is_complex_v<SrcT>>> {
  static constexpr bool is_castable = true;
  __device__ hip_bfloat16 operator()(SrcT x) {
    return hip_bfloat16(static_cast<float>(x));
  }
};

// Conversion between __half and hip_bfloat16
template <>
struct CastOp<__half, hip_bfloat16> {
  static constexpr bool is_castable = true;
  __device__ hip_bfloat16 operator()(__half x) {
    return hip_bfloat16(__half2float(x));
  }
};

template <>
struct CastOp<hip_bfloat16, __half> {
  static constexpr bool is_castable = true;
  __device__ __half operator()(hip_bfloat16 x) {
    return __float2half(static_cast<float>(x));
  }
};

// Helper to deduce the SrcT
template <typename DstT, typename SrcT>
inline __device__ auto cast_to(SrcT x) {
  return CastOp<SrcT, DstT>{}(x);
}

// Grid cap shared by the copy kernels. Past it they walk their elements with a
// grid-stride loop (for_each_copy_index), so a grid of at most this many
// blocks covers any size: HIP refuses a launch of 2^32 threads or more
// (hipErrorInvalidConfiguration), which one thread per element reached at
// 2^32 elements (LOCAL_FIXES.md 42).
inline constexpr size_t kMaxCopyBlocks = 65535;

// Blocks for a grid-stride launch over `size` elements, `per_block` per block
// per pass, computed in 64 bits and clamped to [1, kMaxCopyBlocks].
inline uint32_t copy_grid_blocks(size_t size, size_t per_block) {
  size_t blocks = (size + per_block - 1) / per_block;
  return static_cast<uint32_t>(
      std::min(std::max(blocks, size_t{1}), kMaxCopyBlocks));
}

// Whether a launch of copy_grid_blocks(size, per_block) blocks needs more
// than one pass, that is whether size is past the grid cap.
inline bool copy_grid_loops(size_t size, size_t per_block) {
  return size > kMaxCopyBlocks * per_block;
}

// Calls body(i) for every i in [0, size) from a kernel launched with
// copy_grid_blocks. The kernels take kLoop = copy_grid_loops(...) as a
// template argument. Below the cap (kLoop false) each thread runs the body at
// most once on an index formed exactly as the one-thread-per-element kernels
// formed it, and the instantiation has no loop in it: one kernel holding both
// paths behind a runtime branch was 1.2 to 1.7% slower on a 4M-element
// transpose on gfx1151. Past the cap (kLoop true) the threads stride over the
// elements on a 64-bit counter, which cannot overflow a 32-bit IdxT.
template <bool kLoop, typename F>
__device__ __forceinline__ void for_each_copy_index(int64_t size, F&& body) {
  if constexpr (!kLoop) {
    // At most kMaxCopyBlocks * blockDim.x threads, so this fits in 32 bits.
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
      body(int64_t(index));
    }
  } else {
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
         i += stride) {
      body(i);
    }
  }
}

} // namespace rocm

// Forward declarations
void copy_contiguous(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset);

void copy_general_input(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset,
    const Shape& shape,
    const Strides& strides_in);

void copy_general(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out);

void copy_general_dynamic(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    const array& dynamic_offset_in,
    const array& dynamic_offset_out);

} // namespace mlx::core
