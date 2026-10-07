// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/env_int.h"

#include <rocblas/rocblas.h>

#include <limits>

namespace mlx::core::rocm {

// rocBLAS GEMM wrapper functions

void rocblas_gemm(
    CommandEncoder& encoder,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const array& a,
    int lda,
    const array& b,
    int ldb,
    float beta,
    array& c,
    int ldc,
    Dtype dtype);

void rocblas_gemm_batched(
    CommandEncoder& encoder,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const array& a,
    int lda,
    int64_t stride_a,
    const array& b,
    int ldb,
    int64_t stride_b,
    float beta,
    array& c,
    int ldc,
    int64_t stride_c,
    int batch_count,
    Dtype dtype);

// Raw-pointer row-major GEMM (same convention as hipblaslt_gemm_ptrs).
// Used by MoE SwiGLU VJP for TN/NT cases where hipBLASLt fails under train.
void rocblas_gemm_ptrs(
    CommandEncoder& encoder,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const void* a,
    int lda,
    const void* b,
    int ldb,
    float beta,
    void* c,
    int ldc,
    Dtype dtype);

// rocBLAS solution index for f32 and bf16 GEMMs
// (MLX_ROCM_GEMM_{F32,BF16}_SOLUTION_INDEX, 0 by default) and for their
// batched calls (the _BATCHED_ variables; unset uses the non-batched index).
// matmul.cpp, gemms/rocblas_gemm.cpp and quantized/qmm.hip all read them.
// They are inline so each static is one object per process and a bad value
// warns once, not once per file (lablup/mlxcel#2152).
inline int gemm_solution_index_f32(bool batched) {
  static const int single_index = env_int_or_default(
      "MLX_ROCM_GEMM_F32_SOLUTION_INDEX",
      0,
      0,
      std::numeric_limits<int>::max(),
      "a non-negative integer");
  // -1: use the non-batched index.
  static const int batched_index = env_int_or_default(
      "MLX_ROCM_GEMM_F32_BATCHED_SOLUTION_INDEX",
      -1,
      0,
      std::numeric_limits<int>::max(),
      "a non-negative integer",
      "the MLX_ROCM_GEMM_F32_SOLUTION_INDEX value");
  if (!batched) {
    return single_index;
  }
  return batched_index >= 0 ? batched_index : single_index;
}

inline int gemm_solution_index_bf16(bool batched) {
  static const int single_index = env_int_or_default(
      "MLX_ROCM_GEMM_BF16_SOLUTION_INDEX",
      0,
      0,
      std::numeric_limits<int>::max(),
      "a non-negative integer");
  // -1: use the non-batched index.
  static const int batched_index = env_int_or_default(
      "MLX_ROCM_GEMM_BF16_BATCHED_SOLUTION_INDEX",
      -1,
      0,
      std::numeric_limits<int>::max(),
      "a non-negative integer",
      "the MLX_ROCM_GEMM_BF16_SOLUTION_INDEX value");
  if (!batched) {
    return single_index;
  }
  return batched_index >= 0 ? batched_index : single_index;
}

} // namespace mlx::core::rocm
