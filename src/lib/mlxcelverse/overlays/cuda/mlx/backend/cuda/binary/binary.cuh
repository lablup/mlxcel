// Copyright © 2025 Apple Inc.
//
// CUDA patch: Mixed-precision binary operation kernels for bf16
//
// Modified from upstream MLX e9463bb mlx/backend/cuda/binary/binary.cuh
//
// Changes:
//   - Added mixed-type kernel templates (binary_ss_mixed, binary_sv_mixed, etc.)
//     that accept two different input types (InA, InB) with a single output type.
//   - Added supports_mixed_binary_op<Op, InA, InB, Out> trait that returns true
//     when one input is bf16 and the other is fp32 (Out must be bf16), for
//     arithmetic ops: Add, Subtract, Multiply, Divide, Maximum, Minimum.
//   - Added mixed-type dispatch path in binary_op_gpu_inplace: when inputs have
//     different dtypes and a mixed kernel is available, dispatch directly without
//     requiring astype() conversion. The kernel casts both inputs to fp32 for
//     computation, then writes bf16 output.
//
// This eliminates copy_v<float, bf16> kernels that previously consumed ~58% of
// GPU time during bf16 model inference.
//
// The same-type kernels and dispatch below are byte-identical to upstream
// e9463bb (including the large-grid index_rest fix and
// get_launch_args_general); the mixed kernels mirror those conventions.
//
// This patch is CUDA-only -- applied via the overlay system in CMakeLists.txt
// only when MLX_BUILD_CUDA is set. Metal builds use the unmodified upstream file.

#include "mlx/backend/common/binary.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

constexpr int BINARY_MAX_BLOCK_DIM = 1024;

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ __launch_bounds__(BINARY_MAX_BLOCK_DIM) void binary_ss(
    const In* a,
    const In* b,
    Out* out,
    IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (int i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[0], b[0]);
    }
  } else {
    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a[0], b[0]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ __launch_bounds__(BINARY_MAX_BLOCK_DIM) void binary_sv(
    const In* a,
    const In* b,
    Out* out,
    IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[0], b[i]);
    }
  } else {
    auto b_vec = load_vector<N_READS>(b, index);

    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a[0], b_vec[i]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ __launch_bounds__(BINARY_MAX_BLOCK_DIM) void binary_vs(
    const In* a,
    const In* b,
    Out* out,
    IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[i], b[0]);
    }
  } else {
    auto a_vec = load_vector<N_READS>(a, index);

    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a_vec[i], b[0]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ __launch_bounds__(BINARY_MAX_BLOCK_DIM) void binary_vv(
    const In* a,
    const In* b,
    Out* out,
    IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[i], b[i]);
    }
  } else {
    auto a_vec = load_vector<N_READS>(a, index);
    auto b_vec = load_vector<N_READS>(b, index);

    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a_vec[i], b_vec[i]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <
    typename Op,
    typename In,
    typename Out,
    typename IdxT,
    int NDIM,
    int N_READS>
__global__ void binary_g_nd(
    const In* a,
    const In* b,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> a_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> b_strides) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      (grid.block_index().z * grid.dim_blocks().y + grid.block_index().y) *
          block.dim_threads().y +
      block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[NDIM - 1];
  auto a_stride_x = a_strides[NDIM - 1];
  auto b_stride_x = b_strides[NDIM - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [a_idx, b_idx] = elem_to_loc_nd<NDIM>(
      index_rest * shape_x, shape.data(), a_strides.data(), b_strides.data());
  auto a_vec =
      load_vector<N_READS>(a + a_idx, index_x, shape_x, a_stride_x, In(0));
  auto b_vec =
      load_vector<N_READS>(b + b_idx, index_x, shape_x, b_stride_x, In(0));

  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = Op{}(a_vec[i], b_vec[i]);
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_g(
    const In* a,
    const In* b,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides a_strides,
    const __grid_constant__ Strides b_strides,
    int ndim) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      (grid.block_index().z * grid.dim_blocks().y + grid.block_index().y) *
          block.dim_threads().y +
      block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[ndim - 1];
  auto a_stride_x = a_strides[ndim - 1];
  auto b_stride_x = b_strides[ndim - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [a_idx, b_idx] = elem_to_loc(
      index_rest * shape_x,
      shape.data(),
      a_strides.data(),
      b_strides.data(),
      ndim);
  auto a_vec =
      load_vector<N_READS>(a + a_idx, index_x, shape_x, a_stride_x, In(0));
  auto b_vec =
      load_vector<N_READS>(b + b_idx, index_x, shape_x, b_stride_x, In(0));

  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = Op{}(a_vec[i], b_vec[i]);
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

// --------------------------------------------------------------------------
// Mixed-precision kernels: inputs InA and InB may differ
//
// These kernels cast both operands to float for computation, then cast the
// result to the output type. This fuses what would otherwise be separate
// copy_v + binary_op kernel launches into a single kernel.
// --------------------------------------------------------------------------

template <
    typename Op,
    typename InA,
    typename InB,
    typename Out,
    typename IdxT,
    int N_READS>
__global__ __launch_bounds__(BINARY_MAX_BLOCK_DIM) void binary_ss_mixed(
    const InA* a,
    const InB* b,
    Out* out,
    IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (int i = index * N_READS; i < size; ++i) {
      out[i] = static_cast<Out>(
          Op{}(static_cast<float>(a[0]), static_cast<float>(b[0])));
    }
  } else {
    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = static_cast<Out>(
          Op{}(static_cast<float>(a[0]), static_cast<float>(b[0])));
    }
    store_vector<N_READS>(out, index, out_vec);
  }
}

template <
    typename Op,
    typename InA,
    typename InB,
    typename Out,
    typename IdxT,
    int N_READS>
__global__ __launch_bounds__(BINARY_MAX_BLOCK_DIM) void binary_sv_mixed(
    const InA* a,
    const InB* b,
    Out* out,
    IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = static_cast<Out>(
          Op{}(static_cast<float>(a[0]), static_cast<float>(b[i])));
    }
  } else {
    auto b_vec = load_vector<N_READS>(b, index);
    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = static_cast<Out>(
          Op{}(static_cast<float>(a[0]), static_cast<float>(b_vec[i])));
    }
    store_vector<N_READS>(out, index, out_vec);
  }
}

template <
    typename Op,
    typename InA,
    typename InB,
    typename Out,
    typename IdxT,
    int N_READS>
__global__ __launch_bounds__(BINARY_MAX_BLOCK_DIM) void binary_vs_mixed(
    const InA* a,
    const InB* b,
    Out* out,
    IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = static_cast<Out>(
          Op{}(static_cast<float>(a[i]), static_cast<float>(b[0])));
    }
  } else {
    auto a_vec = load_vector<N_READS>(a, index);
    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = static_cast<Out>(
          Op{}(static_cast<float>(a_vec[i]), static_cast<float>(b[0])));
    }
    store_vector<N_READS>(out, index, out_vec);
  }
}

template <
    typename Op,
    typename InA,
    typename InB,
    typename Out,
    typename IdxT,
    int N_READS>
__global__ __launch_bounds__(BINARY_MAX_BLOCK_DIM) void binary_vv_mixed(
    const InA* a,
    const InB* b,
    Out* out,
    IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = static_cast<Out>(
          Op{}(static_cast<float>(a[i]), static_cast<float>(b[i])));
    }
  } else {
    auto a_vec = load_vector<N_READS>(a, index);
    auto b_vec = load_vector<N_READS>(b, index);
    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = static_cast<Out>(
          Op{}(static_cast<float>(a_vec[i]), static_cast<float>(b_vec[i])));
    }
    store_vector<N_READS>(out, index, out_vec);
  }
}

template <
    typename Op,
    typename InA,
    typename InB,
    typename Out,
    typename IdxT,
    int NDIM,
    int N_READS>
__global__ void binary_g_nd_mixed(
    const InA* a,
    const InB* b,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> a_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> b_strides) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      (grid.block_index().z * grid.dim_blocks().y + grid.block_index().y) *
          block.dim_threads().y +
      block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[NDIM - 1];
  auto a_stride_x = a_strides[NDIM - 1];
  auto b_stride_x = b_strides[NDIM - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [a_idx, b_idx] = elem_to_loc_nd<NDIM>(
      index_rest * shape_x, shape.data(), a_strides.data(), b_strides.data());
  auto a_vec =
      load_vector<N_READS>(a + a_idx, index_x, shape_x, a_stride_x, InA(0));
  auto b_vec =
      load_vector<N_READS>(b + b_idx, index_x, shape_x, b_stride_x, InB(0));

  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = static_cast<Out>(
        Op{}(static_cast<float>(a_vec[i]), static_cast<float>(b_vec[i])));
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

template <
    typename Op,
    typename InA,
    typename InB,
    typename Out,
    typename IdxT,
    int N_READS>
__global__ void binary_g_mixed(
    const InA* a,
    const InB* b,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides a_strides,
    const __grid_constant__ Strides b_strides,
    int ndim) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      (grid.block_index().z * grid.dim_blocks().y + grid.block_index().y) *
          block.dim_threads().y +
      block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[ndim - 1];
  auto a_stride_x = a_strides[ndim - 1];
  auto b_stride_x = b_strides[ndim - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [a_idx, b_idx] = elem_to_loc(
      index_rest * shape_x,
      shape.data(),
      a_strides.data(),
      b_strides.data(),
      ndim);
  auto a_vec =
      load_vector<N_READS>(a + a_idx, index_x, shape_x, a_stride_x, InA(0));
  auto b_vec =
      load_vector<N_READS>(b + b_idx, index_x, shape_x, b_stride_x, InB(0));

  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = static_cast<Out>(
        Op{}(static_cast<float>(a_vec[i]), static_cast<float>(b_vec[i])));
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

template <typename Op, typename In, typename Out>
constexpr bool supports_binary_op() {
  if (std::is_same_v<Op, Add> || std::is_same_v<Op, Divide> ||
      std::is_same_v<Op, Maximum> || std::is_same_v<Op, Minimum> ||
      std::is_same_v<Op, Multiply> || std::is_same_v<Op, Subtract> ||
      std::is_same_v<Op, Power> || std::is_same_v<Op, Remainder>) {
    return std::is_same_v<In, Out>;
  }
  if (std::is_same_v<Op, Equal> || std::is_same_v<Op, Greater> ||
      std::is_same_v<Op, GreaterEqual> || std::is_same_v<Op, Less> ||
      std::is_same_v<Op, LessEqual> || std::is_same_v<Op, NotEqual>) {
    return std::is_same_v<Out, bool>;
  }
  if (std::is_same_v<Op, LogicalAnd> || std::is_same_v<Op, LogicalOr>) {
    return std::is_same_v<Out, bool> && std::is_same_v<In, bool>;
  }
  if (std::is_same_v<Op, NaNEqual>) {
    return std::is_same_v<Out, bool> && is_inexact_v<In>;
  }
  if (std::is_same_v<Op, LogAddExp>) {
    return std::is_same_v<In, Out> && is_inexact_v<In>;
  }
  if (std::is_same_v<Op, ArcTan2>) {
    return std::is_same_v<In, Out> && is_floating_v<In>;
  }
  if (std::is_same_v<Op, BitwiseAnd> || std::is_same_v<Op, BitwiseOr> ||
      std::is_same_v<Op, BitwiseXor>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In>;
  }
  if (std::is_same_v<Op, LeftShift> || std::is_same_v<Op, RightShift>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In> &&
        !std::is_same_v<In, bool>;
  }
  return false;
}

// Mixed-precision support: returns true when InA and InB differ but the
// operation can be performed by casting both to fp32, computing, and writing
// the result as Out. Currently enabled for bf16/fp32 pairs with bf16 output.
template <typename Op, typename InA, typename InB, typename Out>
constexpr bool supports_mixed_binary_op() {
  // Only enable for arithmetic ops where fp32 accumulation is valid
  constexpr bool is_arithmetic_op =
      std::is_same_v<Op, Add> || std::is_same_v<Op, Subtract> ||
      std::is_same_v<Op, Multiply> || std::is_same_v<Op, Divide> ||
      std::is_same_v<Op, Maximum> || std::is_same_v<Op, Minimum>;
  if (!is_arithmetic_op) {
    return false;
  }

  // Check for bf16/fp32 mixed pair with bf16 output
  constexpr bool a_bf16_b_fp32 =
      std::is_same_v<InA, __nv_bfloat16> && std::is_same_v<InB, float>;
  constexpr bool a_fp32_b_bf16 =
      std::is_same_v<InA, float> && std::is_same_v<InB, __nv_bfloat16>;
  constexpr bool out_bf16 = std::is_same_v<Out, __nv_bfloat16>;

  return (a_bf16_b_fp32 || a_fp32_b_bf16) && out_bf16;
}

} // namespace cu

template <typename Op>
void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  assert(inputs.size() > 1);
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  if (out.size() == 0) {
    return;
  }

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);

  // -- Mixed-precision path: inputs have different dtypes --
  if (a.dtype() != b.dtype()) {
    bool dispatched = false;
    dispatch_all_types(a.dtype(), [&](auto a_type_tag) {
      dispatch_all_types(b.dtype(), [&](auto b_type_tag) {
        dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
          using CTYPE_A = MLX_GET_TYPE(a_type_tag);
          using CTYPE_B = MLX_GET_TYPE(b_type_tag);
          using CTYPE_OUT = MLX_GET_TYPE(out_type_tag);
          using InTypeA = cuda_type_t<CTYPE_A>;
          using InTypeB = cuda_type_t<CTYPE_B>;
          using OutType = cuda_type_t<CTYPE_OUT>;
          if constexpr (cu::supports_mixed_binary_op<
                            Op,
                            InTypeA,
                            InTypeB,
                            OutType>()) {
            dispatched = true;
            auto bopt = get_binary_op_type(a, b);
            if (bopt == BinaryOpType::General) {
              dispatch_bool(
                  a.data_size() > INT32_MAX || b.data_size() > INT32_MAX ||
                      out.data_size() > INT32_MAX,
                  [&](auto large) {
                    using IdxT =
                        std::conditional_t<large(), int64_t, int32_t>;
                    Shape shape;
                    std::vector<Strides> strides;
                    std::tie(shape, strides) =
                        collapse_contiguous_dims(a, b, out);
                    auto& a_strides = strides[0];
                    auto& b_strides = strides[1];
                    int ndim = shape.size();
                    int work_per_thread = 1;
                    auto dim0 = ndim > 0 ? shape.back() : 1;
                    auto rest = out.size() / dim0;
                    if (dim0 >= 4) {
                      work_per_thread = 4;
                    }
                    auto [grid_dims, block_dims] =
                        get_launch_args_general(dim0, rest, work_per_thread);
                    if (ndim <= 3) {
                      dispatch_1_2_3(ndim, [&](auto dims_constant) {
                        auto kernel = cu::binary_g_nd_mixed<
                            Op,
                            InTypeA,
                            InTypeB,
                            OutType,
                            IdxT,
                            dims_constant(),
                            1>;
                        if (work_per_thread == 4) {
                          kernel = cu::binary_g_nd_mixed<
                              Op,
                              InTypeA,
                              InTypeB,
                              OutType,
                              IdxT,
                              dims_constant(),
                              4>;
                        }
                        encoder.add_kernel_node(
                            kernel,
                            grid_dims,
                            block_dims,
                            gpu_ptr<InTypeA>(a),
                            gpu_ptr<InTypeB>(b),
                            gpu_ptr<OutType>(out),
                            rest,
                            const_param<dims_constant()>(shape),
                            const_param<dims_constant()>(a_strides),
                            const_param<dims_constant()>(b_strides));
                      });
                    } else {
                      auto kernel = cu::
                          binary_g_mixed<Op, InTypeA, InTypeB, OutType, IdxT, 1>;
                      if (work_per_thread == 4) {
                        kernel = cu::binary_g_mixed<
                            Op,
                            InTypeA,
                            InTypeB,
                            OutType,
                            IdxT,
                            4>;
                      }
                      encoder.add_kernel_node(
                          kernel,
                          grid_dims,
                          block_dims,
                          gpu_ptr<InTypeA>(a),
                          gpu_ptr<InTypeB>(b),
                          gpu_ptr<OutType>(out),
                          rest,
                          const_param(shape),
                          const_param(a_strides),
                          const_param(b_strides),
                          ndim);
                    }
                  });
            } else {
              dispatch_bool(out.data_size() > UINT32_MAX, [&](auto large) {
                using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
                // Use the smaller type's size for N_READS to stay safe
                constexpr int N_READS_A = 16 / sizeof(InTypeA);
                constexpr int N_READS_B = 16 / sizeof(InTypeB);
                constexpr int N_READS =
                    N_READS_A < N_READS_B ? N_READS_A : N_READS_B;
                auto kernel = cu::binary_ss_mixed<
                    Op,
                    InTypeA,
                    InTypeB,
                    OutType,
                    IdxT,
                    N_READS>;
                if (bopt == BinaryOpType::ScalarVector) {
                  kernel = cu::binary_sv_mixed<
                      Op,
                      InTypeA,
                      InTypeB,
                      OutType,
                      IdxT,
                      N_READS>;
                } else if (bopt == BinaryOpType::VectorScalar) {
                  kernel = cu::binary_vs_mixed<
                      Op,
                      InTypeA,
                      InTypeB,
                      OutType,
                      IdxT,
                      N_READS>;
                } else if (bopt == BinaryOpType::VectorVector) {
                  kernel = cu::binary_vv_mixed<
                      Op,
                      InTypeA,
                      InTypeB,
                      OutType,
                      IdxT,
                      N_READS>;
                }
                auto [num_blocks, block_dims] = get_launch_args(
                    out.data_size(),
                    out.shape(),
                    out.strides(),
                    large(),
                    N_READS,
                    cu::BINARY_MAX_BLOCK_DIM);
                encoder.add_kernel_node(
                    kernel,
                    num_blocks,
                    block_dims,
                    gpu_ptr<InTypeA>(a),
                    gpu_ptr<InTypeB>(b),
                    gpu_ptr<OutType>(out),
                    out.data_size());
              });
            }
          }
        });
      });
    });
    if (!dispatched) {
      throw std::runtime_error(
          fmt::format(
              "Can not do mixed binary op {} on inputs of {} and {} with result of {}.",
              op,
              dtype_to_string(a.dtype()),
              dtype_to_string(b.dtype()),
              dtype_to_string(out.dtype())));
    }
    return;
  }

  // -- Same-type path (original upstream logic) --
  dispatch_all_types(a.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      using CTYPE_IN = MLX_GET_TYPE(in_type_tag);
      using CTYPE_OUT = MLX_GET_TYPE(out_type_tag);
      if constexpr (cu::supports_binary_op<Op, CTYPE_IN, CTYPE_OUT>()) {
        using InType = cuda_type_t<CTYPE_IN>;
        using OutType = cuda_type_t<CTYPE_OUT>;
        auto bopt = get_binary_op_type(a, b);
        if (bopt == BinaryOpType::General) {
          dispatch_bool(
              a.data_size() > INT32_MAX || b.data_size() > INT32_MAX ||
                  out.data_size() > INT32_MAX,
              [&](auto large) {
                using IdxT = std::conditional_t<large(), int64_t, int32_t>;
                Shape shape;
                std::vector<Strides> strides;
                std::tie(shape, strides) = collapse_contiguous_dims(a, b, out);
                auto& a_strides = strides[0];
                auto& b_strides = strides[1];
                int ndim = shape.size();
                int work_per_thread = 1;
                auto dim0 = ndim > 0 ? shape.back() : 1;
                auto rest = out.size() / dim0;
                if (dim0 >= 4) {
                  work_per_thread = 4;
                }
                auto [grid_dims, block_dims] =
                    get_launch_args_general(dim0, rest, work_per_thread);
                if (ndim <= 3) {
                  dispatch_1_2_3(ndim, [&](auto dims_constant) {
                    auto kernel = cu::binary_g_nd<
                        Op,
                        InType,
                        OutType,
                        IdxT,
                        dims_constant(),
                        1>;
                    if (work_per_thread == 4) {
                      kernel = cu::binary_g_nd<
                          Op,
                          InType,
                          OutType,
                          IdxT,
                          dims_constant(),
                          4>;
                    }
                    encoder.add_kernel_node(
                        kernel,
                        grid_dims,
                        block_dims,
                        gpu_ptr<InType>(a),
                        gpu_ptr<InType>(b),
                        gpu_ptr<OutType>(out),
                        rest,
                        const_param<dims_constant()>(shape),
                        const_param<dims_constant()>(a_strides),
                        const_param<dims_constant()>(b_strides));
                  });
                } else {
                  auto kernel = cu::binary_g<Op, InType, OutType, IdxT, 1>;
                  if (work_per_thread == 4) {
                    kernel = cu::binary_g<Op, InType, OutType, IdxT, 4>;
                  }
                  encoder.add_kernel_node(
                      kernel,
                      grid_dims,
                      block_dims,
                      gpu_ptr<InType>(a),
                      gpu_ptr<InType>(b),
                      gpu_ptr<OutType>(out),
                      rest,
                      const_param(shape),
                      const_param(a_strides),
                      const_param(b_strides),
                      ndim);
                }
              });
        } else {
          dispatch_bool(out.data_size() > UINT32_MAX, [&](auto large) {
            using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
            constexpr int N_READS = 16 / sizeof(InType);
            auto kernel = cu::binary_ss<Op, InType, OutType, IdxT, N_READS>;
            if (bopt == BinaryOpType::ScalarVector) {
              kernel = cu::binary_sv<Op, InType, OutType, IdxT, N_READS>;
            } else if (bopt == BinaryOpType::VectorScalar) {
              kernel = cu::binary_vs<Op, InType, OutType, IdxT, N_READS>;
            } else if (bopt == BinaryOpType::VectorVector) {
              kernel = cu::binary_vv<Op, InType, OutType, IdxT, N_READS>;
            }
            auto [num_blocks, block_dims] = get_launch_args(
                out.data_size(),
                out.shape(),
                out.strides(),
                large(),
                N_READS,
                cu::BINARY_MAX_BLOCK_DIM);
            encoder.add_kernel_node(
                kernel,
                num_blocks,
                block_dims,
                gpu_ptr<InType>(a),
                gpu_ptr<InType>(b),
                gpu_ptr<OutType>(out),
                out.data_size());
          });
        }
      } else {
        throw std::runtime_error(
            fmt::format(
                "Can not do binary op {} on inputs of {} with result of {}.",
                op,
                dtype_to_string(a.dtype()),
                dtype_to_string(out.dtype())));
      }
    });
  });
}

template <typename Op>
void binary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  auto& encoder = cu::get_command_encoder(s);

  set_binary_op_output_data(
      a, b, out, bopt, [&](auto n) { return cu::malloc_async(n, encoder); });
  binary_op_gpu_inplace<Op>(inputs, out, op, s);
}

#define BINARY_GPU(func)                                              \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    nvtx3::scoped_range r(#func "::eval_gpu");                        \
    auto& s = out.primitive().stream();                               \
    binary_op_gpu<cu::func>(inputs, out, name(), s);                  \
  }

} // namespace mlx::core
