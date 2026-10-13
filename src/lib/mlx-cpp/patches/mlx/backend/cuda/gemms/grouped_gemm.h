// Copyright © 2025 Apple Inc.
// Modified by mlxcel: Added cutlass_gather_mm for general GatherMM support.
// Modified from upstream MLX b98831ad mlx/backend/cuda/gemms/grouped_gemm.h

#pragma once

namespace mlx::core {

namespace cu {
class CommandEncoder;
}

class array;

void cutlass_grouped_gemm_unaligned(
    bool a_transposed,
    int lda,
    bool b_transposed,
    int ldb,
    int group_count,
    const array& a,
    const array& b,
    const array& indices,
    array& out,
    cu::CommandEncoder& encoder);

void cutlass_segmented_mm(
    bool a_transposed,
    int lda,
    bool b_transposed,
    int ldb,
    int num_segments,
    int M,
    int N,
    const array& a,
    const array& b,
    const array& segments,
    array& out,
    cu::CommandEncoder& encoder);

// General gather matmul: each output batch element selects A via lhs_indices
// and B via rhs_indices, then computes A[lhs_idx] @ B[rhs_idx].
// Uses CUTLASS grouped GEMM with GPU-side pointer preparation.
void cutlass_gather_mm(
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    int M,
    int N,
    int K,
    const array& a,
    const array& b,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    cu::CommandEncoder& encoder);

} // namespace mlx::core
