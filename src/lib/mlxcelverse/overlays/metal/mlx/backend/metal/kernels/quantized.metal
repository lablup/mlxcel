// Copyright © 2023-2024 Apple Inc.
// Patched by mlxcel: adds `mlxcel_affine_qmm_t_rows`, a transposed affine
// quantized matmul for up to 8 activation rows.
// Modified from upstream MLX 7ca40033132dae444d45f7fc3daae57e24c13f6c mlx/backend/metal/kernels/quantized.metal
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

#define instantiate_quantized(name, type, group_size, bits)     \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits,                    \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits)

#define instantiate_quantized_batched(name, type, group_size, bits, batched)     \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched, \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits,                                                              \
      batched)

#define instantiate_quantized_qmv(name, type, group_size, bits, batched, partial_rows) \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched "_pr_" #partial_rows "_hgs_0_r_4", \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits,                                                              \
      batched,                                                           \
      partial_rows)

#define instantiate_quantized_aligned(name, type, group_size, bits, aligned)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned)

#define instantiate_quantized_aligned_batched(name, type, group_size, bits, aligned, batched)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned,                                                               \
      batched)

#define instantiate_quantized_quad(name, type, group_size, bits, D, batched)     \
  instantiate_kernel(                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #D "_batch_" #batched, \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      D,                                                            \
      batched)

#define instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, batched)               \
  instantiate_kernel(                                                                                          \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched,   \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      vecs_per_tg,                                                  \
      k_lanes,                                                      \
      batched)

#define instantiate_quantized_split_k(name, type, group_size, bits, split_k)     \
  instantiate_kernel(                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k, \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      split_k)

#define instantiate_gather_qmm_rhs(func, name, type, group_size, bits, bm, bn, bk, wm, wn, transpose)        \
  instantiate_kernel(                                                                                        \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      func,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      bm,                                                           \
      bn,                                                           \
      bk,                                                           \
      wm,                                                           \
      wn,                                                           \
      transpose)

#define instantiate_quantized_batched_wrap(name, type, group_size, bits) \
  instantiate_quantized_batched(name, type, group_size, bits, 1)      \
  instantiate_quantized_batched(name, type, group_size, bits, 0)

#define instantiate_quantized_qmv_wrap(name, type, group_size, bits, partial_rows)  \
  instantiate_quantized_qmv(name, type, group_size, bits, 1, partial_rows) \
  instantiate_quantized_qmv(name, type, group_size, bits, 0, partial_rows)

#define instantiate_quantized_all_batched(type, group_size, bits)            \
  instantiate_quantized_qmv_wrap(affine_qmv_fast, type, group_size, bits, 1) \
  instantiate_quantized_qmv_wrap(affine_qmv_fast, type, group_size, bits, 0) \
  instantiate_quantized_qmv_wrap(affine_qmv, type, group_size, bits, 0)      \
  instantiate_quantized_batched_wrap(affine_qvm, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(affine_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_single(type, group_size, bits) \
  instantiate_quantized(affine_quantize, type, group_size, bits) \
  instantiate_quantized(affine_dequantize, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmv_fast, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmv, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qvm, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_aligned(type, group_size, bits)   \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, true) \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, false) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 0) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 0)

#define instantiate_quantized_all_quad(type, group_size, bits)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 64, 1)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 64, 0)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 128, 1)  \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 128, 0)

// vecs_per_tg (input-vector tile) 2..8; affine uses k_lanes=8 (more rows per
// simdgroup) where the fp path uses 16. The affine kernel holds every streamed
// vector in registers and spills above 8.
#define instantiate_quantized_wide_wrap(name, type, group_size, bits, vecs_per_tg, k_lanes) \
  instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, 0)         \
  instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, 1)

#define instantiate_quantized_all_wide(type, group_size, bits) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 2, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 3, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 4, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 5, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 6, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 7, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 8, 8)

#define instantiate_quantized_all_splitk(type, group_size, bits)   \
  instantiate_quantized_split_k(affine_qvm_split_k, type, group_size, bits, 8)   \
  instantiate_quantized_split_k(affine_qvm_split_k, type, group_size, bits, 32)  \

#define instantiate_quantized_splitk_qmm(name, type, group_size, bits, aligned) \
  instantiate_kernel(                                                           \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,         \
      name,                                                                     \
      type,                                                                     \
      group_size,                                                               \
      bits,                                                                     \
      aligned)

#define instantiate_quantized_all_splitk_qmm(type, group_size, bits)                    \
  instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, true)  \
  instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, false)

#define instantiate_quantized_all_rhs(type, group_size, bits) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs, affine_gather_qmm_rhs_nt, type, group_size, bits, 16, 32, 32, 1, 2, true) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs, affine_gather_qmm_rhs_nn, type, group_size, bits, 16, 32, 32, 1, 2, false)

#define instantiate_quantized_funcs(type, group_size, bits) \
  instantiate_quantized_all_single(type, group_size, bits)  \
  instantiate_quantized_all_batched(type, group_size, bits) \
  instantiate_quantized_all_aligned(type, group_size, bits) \
  instantiate_quantized_all_quad(type, group_size, bits)    \
  instantiate_quantized_all_wide(type, group_size, bits)    \
  instantiate_quantized_all_splitk(type, group_size, bits)  \
  instantiate_quantized_all_splitk_qmm(type, group_size, bits) \
  instantiate_quantized_all_rhs(type, group_size, bits)

#define instantiate_quantized_types(group_size, bits)       \
  instantiate_quantized_funcs(float, group_size, bits)      \
  instantiate_quantized_funcs(float16_t, group_size, bits)  \
  instantiate_quantized_funcs(bfloat16_t, group_size, bits)

#define instantiate_quantized_groups(bits) \
  instantiate_quantized_types(128, bits)   \
  instantiate_quantized_types(64, bits)    \
  instantiate_quantized_types(32, bits)

#define instantiate_quantized_all() \
  instantiate_quantized_groups(1) \
  instantiate_quantized_groups(2) \
  instantiate_quantized_groups(3) \
  instantiate_quantized_groups(4) \
  instantiate_quantized_groups(5) \
  instantiate_quantized_groups(6) \
  instantiate_quantized_groups(8)

instantiate_quantized_all() // clang-format on

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
