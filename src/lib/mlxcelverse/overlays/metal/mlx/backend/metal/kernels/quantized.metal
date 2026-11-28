// Copyright © 2023-2024 Apple Inc.
// Patched by mlxcel: adds `mlxcel_affine_qmm_t_rows`, a transposed affine
// quantized matmul for up to 8 activation rows.
// Modified from upstream MLX 9f806b3606eaa9f7c807238f6bd465185cd94d6d mlx/backend/metal/kernels/quantized.metal
//
// Why it exists. On GPU generations 13 and 14 (M1 and M2 class) an affine
// quantized matmul with 2 <= M < get_qmv_batch_limit runs `qmv_fast` once per
// activation row, and its per-weight unpack, convert and multiply is paid
// again for every row. Batched decode passes exactly that shape (B rows of
// one token each): on M1 Ultra an 8-row Llama 3.1 8B 4-bit decode step cost
// 6x the 1-row step, nearly all of it in these products. `qmv_wide` and the
// upstream qmm tiles (BM = 32) are slower still there.
//
// This kernel is `qmm_t_impl`'s steel MMA path re-tiled to a single 8-row warp
// tile (BM = 8, WM = 1), so each weight tile is dequantized once for all rows
// and no MMA rows are wasted. It reduces along K in a different order than
// `qmv`, so the dispatch keeps it behind the same `MLXCEL_QMV_WIDE` switch the
// speculative exactness gate turns off (lablup/mlxcel#1187).

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

///////////////////////////////////////////////////////////////////////////////
// affine_gather_qmv, affine_gather_qmv_fast, affine_gather_qvm,
// affine_quantize, affine_dequantize
///////////////////////////////////////////////////////////////////////////////

#define instantiate_simple(name, type, group_size, bits) \
  instantiate_kernel(   \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0", \
      name,             \
      type,             \
      group_size,       \
      bits,             \
      false)

#define instantiate_simple_all(type, group_size, bits)               \
  instantiate_simple(affine_gather_qmv, type, group_size, bits)      \
  instantiate_simple(affine_gather_qmv_fast, type, group_size, bits) \
  instantiate_simple(affine_gather_qvm, type, group_size, bits)      \
  instantiate_simple(affine_quantize, type, group_size, bits)        \
  instantiate_simple(affine_dequantize, type, group_size, bits)

///////////////////////////////////////////////////////////////////////////////
// affine_qmm_t, affine_qmm_t_splitk, affine_gather_qmm_t, affine_qmm_n,
// affine_gather_qmm_n
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmm(name, type, group_size, bits, aligned_n, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_alN_" #aligned_n "_batch_" #batched "_bm_32_bn_32", \
      name,       \
      type,       \
      group_size, \
      bits,       \
      false,      \
      aligned_n,  \
      batched,    \
      32,         \
      32,         \
      32)

#define instantiate_qmm_aligned(name, type, group_size, bits) \
  instantiate_qmm(name, type, group_size, bits, 0, 0)         \
  instantiate_qmm(name, type, group_size, bits, 1, 0)

#define instantiate_qmm_batched(name, type, group_size, bits) \
  instantiate_qmm(name, type, group_size, bits, 0, 0)         \
  instantiate_qmm(name, type, group_size, bits, 0, 1)

#define instantiate_qmm_aligned_batched(name, type, group_size, bits) \
  instantiate_qmm(name, type, group_size, bits, 0, 0)                 \
  instantiate_qmm(name, type, group_size, bits, 0, 1)                 \
  instantiate_qmm(name, type, group_size, bits, 1, 0)                 \
  instantiate_qmm(name, type, group_size, bits, 1, 1)

#define instantiate_qmm_all(type, group_size, bits) \
  instantiate_qmm_aligned_batched(affine_qmm_t, type, group_size, bits) \
  instantiate_qmm_aligned(affine_qmm_t_splitk, type, group_size, bits)  \
  instantiate_qmm_aligned(affine_gather_qmm_t, type, group_size, bits)  \
  instantiate_qmm_batched(affine_qmm_n, type, group_size, bits)         \
  instantiate_qmm(affine_gather_qmm_n, type, group_size, bits, 0, 0)

///////////////////////////////////////////////////////////////////////////////
// affine_qmv, affine_qmv_fast
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv(name, type, group_size, bits, batched, partial_rows) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_batch_" #batched "_pr_" #partial_rows "_r_4", \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      false,          \
      batched,        \
      partial_rows,   \
      4)

#define instantiate_qmv_batched(name, type, group_size, bits, partial_rows) \
  instantiate_qmv(name, type, group_size, bits, 0, partial_rows)            \
  instantiate_qmv(name, type, group_size, bits, 1, partial_rows)

#define instantiate_qmv_all(type, group_size, bits)                   \
  instantiate_qmv_batched(affine_qmv, type, group_size, bits, 0)      \
  instantiate_qmv_batched(affine_qmv_fast, type, group_size, bits, 0) \
  instantiate_qmv_batched(affine_qmv_fast, type, group_size, bits, 1)

///////////////////////////////////////////////////////////////////////////////
// affine_qvm
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qvm(name, type, group_size, bits, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_batch_" #batched, \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      false,          \
      batched)

#define instantiate_qvm_all(type, group_size, bits)       \
  instantiate_qvm(affine_qvm, type, group_size, bits, 0)  \
  instantiate_qvm(affine_qvm, type, group_size, bits, 1)

///////////////////////////////////////////////////////////////////////////////
// affine_qvm_splitk
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qvm_splitk(name, type, group_size, bits, split_k) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k, \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      split_k)

#define instantiate_qvm_splitk_all(type, group_size, bits)              \
  instantiate_qvm_splitk(affine_qvm_splitk, type, group_size, bits, 8)  \
  instantiate_qvm_splitk(affine_qvm_splitk, type, group_size, bits, 32)

///////////////////////////////////////////////////////////////////////////////
// affine_gather_qmm_rhs
///////////////////////////////////////////////////////////////////////////////

#define instantiate_grouped_qmm(name, type, group_size, bits, transpose) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_t_" #transpose, \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      false,          \
      transpose)

#define instantiate_grouped_qmm_all(type, group_size, bits)                 \
  instantiate_grouped_qmm(affine_gather_qmm_rhs, type, group_size, bits, 0) \
  instantiate_grouped_qmm(affine_gather_qmm_rhs, type, group_size, bits, 1)

///////////////////////////////////////////////////////////////////////////////
// affine_qmv_quad
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv_quad(name, type, group_size, bits, d, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #d "_batch_" #batched, \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      d,              \
      batched)

#define instantiate_qmv_quad_batched(name, type, group_size, bits, d) \
  instantiate_qmv_quad(name, type, group_size, bits, d, 0)            \
  instantiate_qmv_quad(name, type, group_size, bits, d, 1)

#define instantiate_qmv_quad_all(type, group_size, bits)                    \
  instantiate_qmv_quad_batched(affine_qmv_quad, type, group_size, bits, 64) \
  instantiate_qmv_quad_batched(affine_qmv_quad, type, group_size, bits, 128)

///////////////////////////////////////////////////////////////////////////////
// affine_qmv_wide
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched, \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      vecs_per_tg,    \
      k_lanes,        \
      batched)

#define instantiate_qmv_wide_batched(name, type, group_size, bits, vecs_per_tg, k_lanes) \
  instantiate_qmv_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, 0)            \
  instantiate_qmv_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, 1)

#define instantiate_qmv_wide_all(type, group_size, bits)                      \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 2, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 3, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 4, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 5, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 6, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 7, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 8, 8)

///////////////////////////////////////////////////////////////////////////////
// affine_qmv_mma
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv_mma(name, type, group_size, bits, wk, bn, tm) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_wk_" #wk "_bn_" #bn "_tm_" #tm, \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      wk,             \
      bn,             \
      tm)

#define instantiate_qmv_mma_all(type, group_size, bits)               \
  instantiate_qmv_mma(affine_qmv_mma, type, group_size, bits, 4, 16, 1) \
  instantiate_qmv_mma(affine_qmv_mma, type, group_size, bits, 4, 16, 2) \
  instantiate_qmv_mma(affine_qmv_mma, type, group_size, bits, 4, 32, 1)

///////////////////////////////////////////////////////////////////////////////
// Instantiation
///////////////////////////////////////////////////////////////////////////////

#define instantiate_quantized_all(type, group_size, bits) \
  instantiate_qmm_all(type, group_size, bits)             \
  instantiate_qmv_all(type, group_size, bits)             \
  instantiate_qvm_all(type, group_size, bits)             \
  instantiate_qvm_splitk_all(type, group_size, bits)      \
  instantiate_simple_all(type, group_size, bits)          \
  instantiate_grouped_qmm_all(type, group_size, bits)     \
  instantiate_qmv_quad_all(type, group_size, bits)        \
  instantiate_qmv_wide_all(type, group_size, bits)

#define instantiate_quantized_all_bits(type, group_size) \
  instantiate_quantized_all(type, group_size, 1)         \
  instantiate_quantized_all(type, group_size, 2)         \
  instantiate_quantized_all(type, group_size, 3)         \
  instantiate_quantized_all(type, group_size, 4)         \
  instantiate_quantized_all(type, group_size, 5)         \
  instantiate_quantized_all(type, group_size, 6)         \
  instantiate_quantized_all(type, group_size, 8)         \
  instantiate_qmv_mma_all(type, group_size, 4)           \
  instantiate_qmv_mma_all(type, group_size, 8)

#define instantiate_quantized_all_groups(type) \
  instantiate_quantized_all_bits(type, 32)     \
  instantiate_quantized_all_bits(type, 64)     \
  instantiate_quantized_all_bits(type, 128)

instantiate_quantized_all_groups(float)
instantiate_quantized_all_groups(float16_t)
instantiate_quantized_all_groups(bfloat16_t)
// clang-format on

// mlxcel: few-row transposed affine qmm (see the header). One 8-row warp
// tile (BM = 8, WM = 1) and WN warps across a BN-column tile. Requires
// N % BN == 0, K % BK == 0, group_size >= BK, and a non-batched,
// row-contiguous x with at most 8 rows.
template <typename T, int group_size, int bits, int BK, int BN, int WN>
[[kernel]] void mlxcel_affine_qmm_t_rows(    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BM = 8;
  constexpr int WM = 1;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_col;

  const short num_els = min(BM, M);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_safe(short2(BK, num_els));
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  mma_op.store_result_safe(y, N, short2(BN, num_els));
}

#define instantiate_mlxcel_qmm_t_rows(type, group_size, bits, bk) \
  instantiate_kernel(                                              \
      "mlxcel_affine_qmm_t_rows_" #type "_gs_" #group_size "_b_" #bits \
      "_bk_" #bk,                                                  \
      mlxcel_affine_qmm_t_rows,                                    \
      type,                                                        \
      group_size,                                                  \
      bits,                                                        \
      bk,                                                          \
      32,                                                          \
      2)

// BK = 64 measured fastest on M1 Ultra; group size 32 caps it at 32.
#define instantiate_mlxcel_qmm_t_rows_types(group_size, bits, bk)  \
  instantiate_mlxcel_qmm_t_rows(float, group_size, bits, bk)       \
  instantiate_mlxcel_qmm_t_rows(float16_t, group_size, bits, bk)   \
  instantiate_mlxcel_qmm_t_rows(bfloat16_t, group_size, bits, bk)

#define instantiate_mlxcel_qmm_t_rows_groups(bits)    \
  instantiate_mlxcel_qmm_t_rows_types(128, bits, 64)  \
  instantiate_mlxcel_qmm_t_rows_types(64, bits, 64)   \
  instantiate_mlxcel_qmm_t_rows_types(32, bits, 32)

// clang-format off
instantiate_mlxcel_qmm_t_rows_groups(4)
instantiate_mlxcel_qmm_t_rows_groups(8) // clang-format on
