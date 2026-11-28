// Copyright © 2025 Apple Inc.
//
// CUDA patch: bf16 reduce accumulation optimization
//
// Modified from upstream MLX ce45c525 mlx/backend/cuda/reduce/reduce_ops.cuh
//
// Changes:
//   - Added ReduceResult<Sum, __nv_bfloat16> specialization: accumulate in fp32
//   - Added ReduceResult<Prod, __nv_bfloat16> specialization: accumulate in fp32
//   - This ensures numerical precision for bf16 sum/prod reductions while
//     allowing the output to remain bf16 (cast back at the write-back site).
//   - Added ReduceOutput<Op, T>: the type the dispatch code writes to the
//     output array (the accumulation type, except bf16 Sum/Prod, which write
//     bf16). Writing the input type instead corrupts every reduction whose
//     output dtype differs from its input: sum/prod over bool and the narrow
//     integers, and any/all over a non-bool input.

#pragma once

#include "mlx/backend/cuda/device/atomic_ops.cuh"
#include "mlx/backend/cuda/device/cast_op.cuh"
#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/reduce/reduce_utils.cuh"

namespace mlx::core::cu {

// Reduce ops.
struct And {
  __device__ __forceinline__ bool operator()(bool a, bool b) {
    return a && b;
  }

  __device__ void atomic_update(bool* x, bool y) {
    atomic_reduce<bool, And>(x, y);
  }
};

struct Or {
  __device__ __forceinline__ bool operator()(bool a, bool b) {
    return a || b;
  }

  __device__ void atomic_update(bool* x, bool y) {
    atomic_reduce<bool, Or>(x, y);
  }
};

struct Sum {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) {
    return a + b;
  }

  template <typename T>
  __device__ void atomic_update(T* x, T y) {
    atomic_reduce<T, Sum>(x, y);
  }

  __device__ void atomic_update(__nv_bfloat16* x, __nv_bfloat16 y) {
    atomic_add(x, y);
  }

  __device__ void atomic_update(int* x, int y) {
    atomic_add(x, y);
  }

  __device__ void atomic_update(float* x, float y) {
    atomic_add(x, y);
  }
};

struct Prod {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) {
    return a * b;
  }

  template <typename T>
  __device__ void atomic_update(T* x, T y) {
    atomic_reduce<T, Prod>(x, y);
  }
};

struct Min {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) {
    if constexpr (is_complex_v<T>) {
      if (cuda::std::isnan(a.real()) || cuda::std::isnan(a.imag())) {
        return a;
      }
      if (cuda::std::isnan(b.real()) || cuda::std::isnan(b.imag())) {
        return b;
      }
    } else if constexpr (!cuda::std::is_integral_v<T>) {
      if (cuda::std::isnan(a) || cuda::std::isnan(b)) {
        return cuda::std::numeric_limits<float>::quiet_NaN();
      }
    }
    return a < b ? a : b;
  }

  template <typename T>
  __device__ void atomic_update(T* x, T y) {
    atomic_reduce<T, Min>(x, y);
  }
};

struct Max {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) {
    if constexpr (is_complex_v<T>) {
      if (cuda::std::isnan(a.real()) || cuda::std::isnan(a.imag())) {
        return a;
      }
      if (cuda::std::isnan(b.real()) || cuda::std::isnan(b.imag())) {
        return b;
      }
    } else if constexpr (!cuda::std::is_integral_v<T>) {
      if (cuda::std::isnan(a) || cuda::std::isnan(b)) {
        return cuda::std::numeric_limits<float>::quiet_NaN();
      }
    }
    return a > b ? a : b;
  }

  template <typename T>
  __device__ void atomic_update(T* x, T y) {
    atomic_reduce<T, Max>(x, y);
  }
};

// Traits to get the accumulation type of reduce op.
//
// It is not always the input type: And / Or accumulate in bool, Sum / Prod
// over bool and the integers up to 32 bits accumulate in int32, and bf16
// Sum / Prod accumulate in fp32 for numerical precision and are cast back to
// bf16 when the final output is written (see ReduceOutput below).
template <typename Op, typename T>
struct ReduceResult;

template <typename T>
struct ReduceResult<And, T> {
  using type = bool;
};

template <typename T>
struct ReduceResult<Or, T> {
  using type = bool;
};

template <typename T>
struct ReduceResult<Sum, T> {
  using type = cuda::std::conditional_t<
      (cuda::std::is_integral_v<T> && sizeof(T) <= 4),
      int32_t,
      T>;
};

// bf16 Sum: accumulate in fp32 for precision
template <>
struct ReduceResult<Sum, __nv_bfloat16> {
  using type = float;
};

template <typename T>
struct ReduceResult<Prod, T> {
  using type = cuda::std::conditional_t<
      (cuda::std::is_integral_v<T> && sizeof(T) <= 4),
      int32_t,
      T>;
};

// bf16 Prod: accumulate in fp32 for precision
template <>
struct ReduceResult<Prod, __nv_bfloat16> {
  using type = float;
};

template <typename T>
struct ReduceResult<Min, T> {
  using type = T;
};

template <typename T>
struct ReduceResult<Max, T> {
  using type = T;
};

// The type a reduction writes to its output array.
//
// It is the accumulation type, as in upstream MLX, because that is the type
// MLX gives the output, up to signedness: `bool` for And / Or whatever the
// input is, a 32-bit integer for Sum / Prod over `bool` and the narrow
// integers (MLX makes it uint32 for an unsigned input; the kernel writes the
// same 32 bits as int32_t, as upstream does), and the input type otherwise.
// The one exception is the bf16 Sum / Prod above, which only accumulates in
// fp32 and still writes bf16.
//
// The dispatch code must take the output pointer from this trait and not from
// the input type: for `sum(bool)` or `any(float32)` the two differ in size,
// and a kernel that writes the input type fills the output with garbage.
template <typename Op, typename T>
struct ReduceOutput {
  using type = typename ReduceResult<Op, T>::type;
};

template <>
struct ReduceOutput<Sum, __nv_bfloat16> {
  using type = __nv_bfloat16;
};

template <>
struct ReduceOutput<Prod, __nv_bfloat16> {
  using type = __nv_bfloat16;
};

// Traits to get the init value of reduce op.
template <typename Op, typename T>
struct ReduceInit;

template <typename T>
struct ReduceInit<And, T> {
  static constexpr __host__ __device__ bool value() {
    return true;
  }
};

template <typename T>
struct ReduceInit<Or, T> {
  static constexpr __host__ __device__ bool value() {
    return false;
  }
};

template <typename T>
struct ReduceInit<Sum, T> {
  static constexpr __host__ __device__ auto value() {
    if constexpr (is_complex_v<T>) {
      return T{0, 0};
    } else {
      return cast_to<typename ReduceResult<Sum, T>::type>(0);
    }
  }
};

template <typename T>
struct ReduceInit<Prod, T> {
  static constexpr __host__ __device__ auto value() {
    if constexpr (is_complex_v<T>) {
      return T{1, 0};
    } else {
      return cast_to<typename ReduceResult<Prod, T>::type>(1);
    }
  }
};

template <typename T>
struct ReduceInit<Min, T> {
  static constexpr __host__ __device__ T value() {
    return Limits<T>::max();
  }
};

template <typename T>
struct ReduceInit<Max, T> {
  static constexpr __host__ __device__ T value() {
    return Limits<T>::min();
  }
};

} // namespace mlx::core::cu
