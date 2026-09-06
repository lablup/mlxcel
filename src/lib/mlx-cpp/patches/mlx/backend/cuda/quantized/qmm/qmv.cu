// Copyright © 2026 Apple Inc.
// Patched by mlxcel (lablup/mlxcel#725): add a weight-amortizing multirow qmv
// path for small-M/batched decode. The stock kernel maps one input row per
// grid.y/grid.z block, so M*B < 8 shapes (batched decode with B in [2,8),
// speculative-verify [1, K] forwards) re-read the full quantized weight
// matrix once per row: O(R) DRAM weight traffic and flat aggregate decode
// throughput on Blackwell sm_120/121, which has no CUTLASS sm90 qmm path and
// whose qmm_sm80 tile wastes ~94% of its 128-wide tile at M=8. The multirow
// kernel below assigns all R = m*l rows to the same warp: each weight tile is
// loaded and dequantized once, then FMA'd against every row's activations, so
// weight traffic is O(1) in R. Per-row arithmetic (dequant -> fma order,
// accumulator types, final float reduction) matches the stock kernel exactly,
// so per-row outputs are bit-identical to the per-row launches it replaces.
// Selection: broadcast weights and 2 <= m*l <= W, kill switch
// MLXCEL_QMV_MULTIROW=0. W is the row-window ceiling, 8 by default and
// narrowable to [1, 8] via MLXCEL_QMV_MULTIROW_MAX_ROWS (lablup/mlxcel#906) so
// the autotuner can tune the crossover instead of hardcoding it.
//
// Second mlxcel change (lablup/mlxcel#1539): the accumulator type below Ampere.
// Upstream reads it off the weight bit width, which hands a bf16 checkpoint a
// bf16 accumulator at bits < 8, and pre-Ampere parts have no bf16 ALU to run it
// on. `qmv_accumulator` below promotes that case to float behind a
// `__CUDA_ARCH__ < 800` guard; sm_80 and later are untouched. Everything else is
// byte-identical upstream (pin 57c66cac, v0.32.0-1).

#include "mlx/backend/cuda/device/cute_dequant.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstdlib>
#include <string>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

// Fused vectorized dequantize and multiply-add:
// w_dq = w * scale + bias
// out = fma(x, w_dq, out)
template <int N, bool has_bias, typename T, typename Q, typename S>
__device__ __forceinline__ void
dequant_fma(const T* x, const Q* w, S scale, T bias, T* out) {
  // Read x/w into registers.
  auto x_vec = *(reinterpret_cast<const cutlass::Array<T, N>*>(x));
  auto w_vec = *(reinterpret_cast<const cutlass::Array<Q, N>*>(w));
  // Output is assumed to be registers.
  auto* out_vec = reinterpret_cast<cutlass::Array<T, N>*>(out);

  // Dequantize w.
  cutlass::NumericArrayConverter<T, Q, N> converter_tq;
  cutlass::Array<T, N> w_dq = converter_tq(w_vec);
  if constexpr (has_bias) {
    if constexpr (cuda::std::is_same_v<T, float>) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        w_dq[i] = w_dq[i] * T(scale) + bias;
      }
    } else {
      w_dq = w_dq * T(scale) + bias;
    }
  } else {
    w_dq = w_dq * T(scale);
  }

  // Multiply and add.
  *out_vec = cutlass::fma(x_vec, w_dq, *out_vec);
}

// Specialization for doing float32 accumulations on narrow types.
template <
    int N,
    bool has_bias,
    typename T,
    typename Q,
    typename S,
    typename = cuda::std::enable_if_t<!cuda::std::is_same_v<T, float>>>
__device__ __forceinline__ void
dequant_fma(const T* x, const Q* w, S scale, T bias, float* out) {
  // Read x/w into registers.
  auto x_vec = *(reinterpret_cast<const cutlass::Array<T, N>*>(x));
  auto w_vec = *(reinterpret_cast<const cutlass::Array<Q, N>*>(w));
  // Output is assumed to be registers.
  auto* out_vec = reinterpret_cast<cutlass::Array<float, N>*>(out);

  // Dequantize w.
  cutlass::NumericArrayConverter<T, Q, N> converter_tq;
  cutlass::Array<T, N> w_dq = converter_tq(w_vec);
  if constexpr (has_bias) {
    w_dq = w_dq * T(scale) + bias;
  } else {
    w_dq = w_dq * T(scale);
  }

  // Promote x/w to float.
  static_assert(!cuda::std::is_same_v<T, float>);
  cutlass::NumericArrayConverter<float, T, N> converter_ft;
  cutlass::Array<float, N> x_f = converter_ft(x_vec);
  cutlass::Array<float, N> w_f = converter_ft(w_dq);

  // Multiply and add.
  *out_vec = cutlass::fma(x_f, w_f, *out_vec);
}

// [mlxcel #725] Split of dequant_fma into a dequantize-once tile helper plus
// per-row fma helpers, so the multirow kernel can share one dequantized weight
// tile across all of its input rows. The arithmetic per row is kept exactly
// equal to the fused dequant_fma above (same converter, same scale/bias order,
// same fma), so a multirow launch produces bit-identical per-row results.
template <int N, bool has_bias, typename T, typename Q, typename S>
__device__ __forceinline__ cutlass::Array<T, N>
dequant_tile(const Q* w, S scale, T bias) {
  auto w_vec = *(reinterpret_cast<const cutlass::Array<Q, N>*>(w));
  cutlass::NumericArrayConverter<T, Q, N> converter_tq;
  cutlass::Array<T, N> w_dq = converter_tq(w_vec);
  if constexpr (has_bias) {
    if constexpr (cuda::std::is_same_v<T, float>) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        w_dq[i] = w_dq[i] * T(scale) + bias;
      }
    } else {
      w_dq = w_dq * T(scale) + bias;
    }
  } else {
    w_dq = w_dq * T(scale);
  }
  return w_dq;
}

// fma against a pre-dequantized tile, same-type accumulators (bits < 8).
template <int N, typename T>
__device__ __forceinline__ void
fma_tile(const T* x, const cutlass::Array<T, N>& w_dq, T* out) {
  auto x_vec = *(reinterpret_cast<const cutlass::Array<T, N>*>(x));
  auto* out_vec = reinterpret_cast<cutlass::Array<T, N>*>(out);
  *out_vec = cutlass::fma(x_vec, w_dq, *out_vec);
}

// fma against a pre-dequantized tile, float accumulators on narrow types
// (bits >= 8), matching the float-accumulation dequant_fma specialization.
template <
    int N,
    typename T,
    typename = cuda::std::enable_if_t<!cuda::std::is_same_v<T, float>>>
__device__ __forceinline__ void
fma_tile(const T* x, const cutlass::Array<T, N>& w_dq, float* out) {
  auto x_vec = *(reinterpret_cast<const cutlass::Array<T, N>*>(x));
  auto* out_vec = reinterpret_cast<cutlass::Array<float, N>*>(out);
  cutlass::NumericArrayConverter<float, T, N> converter_ft;
  cutlass::Array<float, N> x_f = converter_ft(x_vec);
  cutlass::Array<float, N> w_f = converter_ft(w_dq);
  *out_vec = cutlass::fma(x_f, w_f, *out_vec);
}

// [mlxcel] fma against a pre-loaded activation tile.
//
// `fma_tile` above takes `const T* x` and loads it, which is right when one
// warp owns one output row and reads the activation vector once per row. The
// row-blocked kernel loads the activation tile once and applies it to several
// weight rows, so it needs the same arithmetic with the tile already in
// registers. Same converter, same operand order, same `cutlass::fma`, so a
// row-blocked launch produces bit-identical per-row results.
template <int N, typename T>
__device__ __forceinline__ void fma_tile_loaded(
    const cutlass::Array<T, N>& x_vec,
    const cutlass::Array<T, N>& w_dq,
    T* out) {
  auto* out_vec = reinterpret_cast<cutlass::Array<T, N>*>(out);
  *out_vec = cutlass::fma(x_vec, w_dq, *out_vec);
}

template <
    int N,
    typename T,
    typename = cuda::std::enable_if_t<!cuda::std::is_same_v<T, float>>>
__device__ __forceinline__ void fma_tile_loaded(
    const cutlass::Array<T, N>& x_vec,
    const cutlass::Array<T, N>& w_dq,
    float* out) {
  auto* out_vec = reinterpret_cast<cutlass::Array<float, N>*>(out);
  cutlass::NumericArrayConverter<float, T, N> converter_ft;
  cutlass::Array<float, N> x_f = converter_ft(x_vec);
  cutlass::Array<float, N> w_f = converter_ft(w_dq);
  *out_vec = cutlass::fma(x_f, w_f, *out_vec);
}

// [mlxcel #1539] Accumulator width for the qmv family.
//
// Upstream selects it from the weight bit width alone: float at bits >= 8,
// otherwise the element type T. That rule assumes T's own ALU is fast, which
// holds from Ampere on and fails for bfloat16 before it. sm_70 and sm_75 have
// no bf16 arithmetic unit at all, so a cutlass::bfloat16_t accumulator turns
// every FMA in the k-loop into convert-to-float, fma, convert-back. Measured
// on a V100 over identical launch counts, qmv spends 12.84 s on a 4-bit bf16
// checkpoint against 5.99 s on the 8-bit sibling of that same checkpoint, a
// 2.14x gap that accounts for 97.6% of the end-to-end decode difference, while
// qmm_naive (float accumulators at every bit width) moves the other way in the
// same profile. See docs/benchmark_results/volta-sm70-baseline-2026-08-31.md.
//
// So below Ampere bf16 accumulates in float regardless of bit width. The float
// specializations of dequant_fma and fma_tile above already exist and are
// exercised by the bits >= 8 path, so this instantiates nothing new.
//
// Deliberately not widened to half_t: sm_70 and sm_75 do have a native fp16
// FMA at twice the fp32 rate, so promoting f16 here would spend throughput to
// buy precision, which is the opposite trade to the one this makes for bf16.
// f16 policy below Ampere is #1542's subject, not this one's.
//
// __CUDA_ARCH__ rather than a host-side dispatch: qmv is AOT-compiled, the host
// only takes &qmv_kernel<...> as a function pointer and launches it through
// add_kernel_node_raw, and the accumulator appears in neither the template
// signature nor the mangled name. nvcc emits one device pass per entry in
// MLX_CUDA_ARCHITECTURES, so the guard gives exactly per-architecture behavior
// inside a fat binary, with nothing for the single host pass to disagree with.
// Ampere and later keep upstream's rule unchanged.
template <int bits, typename T>
struct qmv_accumulator {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  using type = cuda::std::conditional_t<
      (bits >= 8) || cuda::std::is_same_v<T, cutlass::bfloat16_t>,
      float,
      T>;
#else
  using type = cuda::std::conditional_t<(bits >= 8), float, T>;
#endif
};

template <int bits, typename T>
using qmv_accumulator_t = typename qmv_accumulator<bits, T>::type;

template <
    int elems_per_thread,
    int group_size,
    bool has_bias,
    bool has_residue_k,
    typename T,
    typename Q,
    typename S>
__device__ __forceinline__ void qmv_kernel_impl(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    const float* global_scale,
    T* out,
    int row,
    int w_batch,
    int n,
    int k) {
  auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

  // For sub-byte Q, pointer moves by 8bits for each advance, e.g. w += 1 would
  // move past 2 elements for 4-bit Q.
  constexpr int bits = cute::sizeof_bits_v<Q>;
  auto w_step = [&](int idx) { return idx * cuda::std::min(8, bits) / 8; };

  // How many groups (and scales/biases) in a row.
  int groups_per_row = k / group_size;

  // Advance w/scales/biases to current row.
  w += (static_cast<int64_t>(row) + n * w_batch) * w_step(k);
  scales += (static_cast<int64_t>(row) + n * w_batch) * groups_per_row;
  if constexpr (has_bias) {
    biases += (static_cast<int64_t>(row) + n * w_batch) * groups_per_row;
  }

  // Accumulations of current row.
  qmv_accumulator_t<bits, T> sums[elems_per_thread] = {};

  auto dequant_fma_tile = [&](int idx) {
    S scale = scales[idx / group_size];
    T bias{0};
    if constexpr (has_bias) {
      bias = biases[idx / group_size];
    }
    dequant_fma<elems_per_thread, has_bias>(
        x + idx, w + w_step(idx), scale, bias, sums);
  };

  // Loop over k dimension.
  constexpr int elems_per_warp = WARP_SIZE * elems_per_thread;
  for (int r = 0; r < k / elems_per_warp; ++r) {
    int idx = warp.thread_rank() * elems_per_thread + r * elems_per_warp;
    dequant_fma_tile(idx);
  }

  // Handle remaining elements in k dimension.
  if constexpr (has_residue_k) {
    int rest = k % elems_per_warp;
    int idx = warp.thread_rank() * elems_per_thread + k - rest;
    if (idx < k) {
      dequant_fma_tile(idx);
    }
  }

  // Result for current row.
  float sum{0};
#pragma unroll
  for (int i = 0; i < elems_per_thread; ++i) {
    sum += sums[i];
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});

  // Write result for current warp, which maps to rows 1-to-1.
  if (warp.thread_rank() == 0) {
    if constexpr (
        cuda::std::is_same_v<Q, cutlass::float_e2m1_t> &&
        cuda::std::is_same_v<S, cutlass::float_e4m3_t>) {
      // Only nvfp4 supports global scale.
      if (global_scale) {
        sum *= (*global_scale / (F8E4M3_MAX * F4E2M1_MAX));
      }
    }
    out[row] = static_cast<T>(sum);
  }
}

// [mlxcel #725] Multirow qmv: one warp still owns one weight row, but instead
// of one input row per grid.y/grid.z block, the warp iterates all R = m*l
// input rows against each dequantized weight tile. Weights are broadcast
// (w_batch == 0 for every row), and x/out rows are flat row-major at strides
// k/n, which QuantizedMatmul::eval_gpu guarantees via ensure_row_contiguous.
// The j-loop is unrolled to max_x_rows with a warp-uniform bound check, so
// accumulators stay in registers for every x_rows in [2, max_x_rows].
// Invariant: x_rows is a kernel argument (block-uniform), which is what makes
// the cg::reduce collective inside the `j < x_rows` branch safe; do not turn
// x_rows into anything lane-dependent.
template <
    int elems_per_thread,
    int group_size,
    bool has_bias,
    bool has_residue_k,
    int max_x_rows,
    typename T,
    typename Q,
    typename S>
__device__ __forceinline__ void qmv_multirow_kernel_impl(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    const float* global_scale,
    T* out,
    int row,
    int x_rows,
    int n,
    int k) {
  auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

  constexpr int bits = cute::sizeof_bits_v<Q>;
  auto w_step = [&](int idx) { return idx * cuda::std::min(8, bits) / 8; };

  int groups_per_row = k / group_size;

  // Advance w/scales/biases to the current weight row (broadcast weights, so
  // there is no per-row w_batch term).
  w += static_cast<int64_t>(row) * w_step(k);
  scales += static_cast<int64_t>(row) * groups_per_row;
  if constexpr (has_bias) {
    biases += static_cast<int64_t>(row) * groups_per_row;
  }

  // Per-input-row accumulators, same accumulator type the single-row kernel
  // picks for this (bits, T, architecture), which is what keeps a multirow
  // launch bit-identical to the per-row launches it replaces (#725).
  qmv_accumulator_t<bits, T> sums[max_x_rows][elems_per_thread] = {};

  auto multirow_tile = [&](int idx) {
    S scale = scales[idx / group_size];
    T bias{0};
    if constexpr (has_bias) {
      bias = biases[idx / group_size];
    }
    // Dequantize the weight tile once, then apply it to every input row.
    auto w_dq =
        dequant_tile<elems_per_thread, has_bias, T>(w + w_step(idx), scale, bias);
#pragma unroll
    for (int j = 0; j < max_x_rows; ++j) {
      if (j < x_rows) {
        fma_tile<elems_per_thread>(
            x + static_cast<int64_t>(j) * k + idx, w_dq, sums[j]);
      }
    }
  };

  constexpr int elems_per_warp = WARP_SIZE * elems_per_thread;
  for (int r = 0; r < k / elems_per_warp; ++r) {
    int idx = warp.thread_rank() * elems_per_thread + r * elems_per_warp;
    multirow_tile(idx);
  }

  if constexpr (has_residue_k) {
    int rest = k % elems_per_warp;
    int idx = warp.thread_rank() * elems_per_thread + k - rest;
    if (idx < k) {
      multirow_tile(idx);
    }
  }

#pragma unroll
  for (int j = 0; j < max_x_rows; ++j) {
    if (j < x_rows) {
      float sum{0};
#pragma unroll
      for (int i = 0; i < elems_per_thread; ++i) {
        sum += sums[j][i];
      }
      sum = cg::reduce(warp, sum, cg::plus<float>{});
      if (warp.thread_rank() == 0) {
        if constexpr (
            cuda::std::is_same_v<Q, cutlass::float_e2m1_t> &&
            cuda::std::is_same_v<S, cutlass::float_e4m3_t>) {
          // Only nvfp4 supports global scale.
          if (global_scale) {
            sum *= (*global_scale / (F8E4M3_MAX * F4E2M1_MAX));
          }
        }
        out[static_cast<int64_t>(j) * n + row] = static_cast<T>(sum);
      }
    }
  }
}

// [mlxcel] Row-blocked qmv: one warp owns `rows_per_warp` output rows and
// loads the activation tile once for all of them.
//
// The pointers are `__restrict__` here and not in the stock kernel. Every buffer
// this kernel touches is distinct and the outputs never alias the inputs, so
// telling the compiler that lets it keep the activation tile live across the R
// weight rows instead of reloading it after each store, and opens the read-only
// path for the weight and scale streams. The stock kernel is left alone on
// purpose: R = 1 has to stay a true rollback to what was measured before.
//
// Why. `qmv_kernel` gives each warp one output row and has that warp read the
// whole activation vector, so a launch moves `n * k * 2` bytes of activations
// against `n * k / 2` bytes of 4-bit weights: four bytes of activation per byte
// of weight. Measured on a V100 over a controlled pair of gemma-4-12B-it
// checkpoints with identical launch counts (49021), 4-bit qmv takes 52.1 us per
// launch and 8-bit takes 60.4 us. The weight bytes double and the time rises
// 1.159x, which pins the activation term at 84% of a 4-bit launch. Blocking R
// output rows into one warp divides that term by R while leaving the weight
// traffic alone.
//
// What it costs. Warp-level parallelism drops by R, since the same output rows
// are covered by R times fewer warps, and the accumulator array grows to
// `rows_per_warp * elems_per_thread`. For 4-bit the accumulator element is T,
// so 16 halves is 8 registers per row. Whether the reuse or the lost occupancy
// wins is not predictable, so `MLXCEL_QMV_ROWS_PER_WARP` selects R at runtime
// and 1 (the stock kernel) stays the default.
template <
    int rows_per_warp,
    int elems_per_thread,
    int group_size,
    bool has_bias,
    bool has_residue_k,
    typename T,
    typename Q,
    typename S>
__device__ __forceinline__ void qmv_rowblock_kernel_impl(
    const T* __restrict__ x,
    const Q* __restrict__ w,
    const S* __restrict__ scales,
    const T* __restrict__ biases,
    const float* __restrict__ global_scale,
    T* __restrict__ out,
    int row0,
    int w_batch,
    int n,
    int k) {
  auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

  constexpr int bits = cute::sizeof_bits_v<Q>;
  auto w_step = [&](int idx) { return idx * cuda::std::min(8, bits) / 8; };
  int groups_per_row = k / group_size;

  // Per-row bases. An out-of-range row is clamped to row 0 so its loads stay
  // in bounds; its result is simply not stored. `row0 + r < n` depends only on
  // the block and warp index, so every lane agrees and the warp collective
  // below stays uniform.
  const Q* __restrict__ w_row[rows_per_warp];
  const S* __restrict__ scales_row[rows_per_warp];
  const T* __restrict__ biases_row[rows_per_warp];
  bool active[rows_per_warp];
#pragma unroll
  for (int r = 0; r < rows_per_warp; ++r) {
    active[r] = (row0 + r) < n;
    int64_t off =
        static_cast<int64_t>(active[r] ? row0 + r : 0) + int64_t(n) * w_batch;
    w_row[r] = w + off * w_step(k);
    scales_row[r] = scales + off * groups_per_row;
    if constexpr (has_bias) {
      biases_row[r] = biases + off * groups_per_row;
    }
  }

  qmv_accumulator_t<bits, T> sums[rows_per_warp][elems_per_thread] = {};

  auto step = [&](int idx) {
    auto x_vec =
        *(reinterpret_cast<const cutlass::Array<T, elems_per_thread>*>(x + idx));
#pragma unroll
    for (int r = 0; r < rows_per_warp; ++r) {
      S scale = scales_row[r][idx / group_size];
      T bias{0};
      if constexpr (has_bias) {
        bias = biases_row[r][idx / group_size];
      }
      auto w_dq = dequant_tile<elems_per_thread, has_bias, T, Q, S>(
          w_row[r] + w_step(idx), scale, bias);
      fma_tile_loaded<elems_per_thread>(x_vec, w_dq, sums[r]);
    }
  };

  // Scale and bias cost as many memory transactions per step as the weights do,
  // four bytes against 256, so batching them looked like a quarter of the launch
  // waiting to be reclaimed. It is not: a warp step spans exactly
  // `group_size / elems_per_thread` lanes per group, so four consecutive steps
  // span one group per lane and a single coalesced load plus `__shfl_sync` can
  // serve all four. Measured on a V100 that arm lost everywhere, 60.1 to 51.5
  // tok/s at R = 2 and 58.8 to 46.7 at R = 3, at identical register counts and
  // identical residency. The shuffles and the pack/unpack cost more than the six
  // transactions they save, so the per-step loads stay.
  constexpr int elems_per_warp = WARP_SIZE * elems_per_thread;
  for (int t = 0; t < k / elems_per_warp; ++t) {
    step(warp.thread_rank() * elems_per_thread + t * elems_per_warp);
  }
  if constexpr (has_residue_k) {
    int rest = k % elems_per_warp;
    int idx = warp.thread_rank() * elems_per_thread + k - rest;
    if (idx < k) {
      step(idx);
    }
  }

#pragma unroll
  for (int r = 0; r < rows_per_warp; ++r) {
    float sum{0};
#pragma unroll
    for (int i = 0; i < elems_per_thread; ++i) {
      sum += sums[r][i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    if (warp.thread_rank() == 0 && active[r]) {
      if constexpr (
          cuda::std::is_same_v<Q, cutlass::float_e2m1_t> &&
          cuda::std::is_same_v<S, cutlass::float_e4m3_t>) {
        if (global_scale) {
          sum *= (*global_scale / (F8E4M3_MAX * F4E2M1_MAX));
        }
      }
      out[row0 + r] = static_cast<T>(sum);
    }
  }
}

// `min_blocks` reaches `__launch_bounds__` as the resident-block floor, which is
// how ptxas is told to cap the register count. The R sweep on a V100 is a
// straight line in occupancy, not in loads: R = 2 takes 64 to 72 registers and 3
// to 4 blocks per SM and wins at 56.0 tok/s, R = 3 takes 80 to 110 and 2 to 3
// blocks and falls to 53.4, R = 4 takes 96 to 142 and 1 to 2 blocks and falls to
// 45.2, all while each step up cuts transactions per output row. So the register
// budget is the thing to set, and the row count is what fits inside it. A value
// of 1 is the unconstrained case, since 65536 / (1 * 256) exceeds the 255-register
// hardware ceiling anyway.
template <
    int rows_per_block,
    int rows_per_warp,
    int min_blocks,
    int elems_per_thread,
    int group_size,
    bool has_bias,
    bool has_residue_k,
    typename T,
    typename Q,
    typename S>
__global__ __launch_bounds__(WARP_SIZE* rows_per_block, min_blocks) void qmv_rowblock_kernel(
    const T* __restrict__ x,
    const Q* __restrict__ w,
    const S* __restrict__ scales,
    const T* __restrict__ biases,
    const float* __restrict__ global_scale,
    T* __restrict__ out,
    int n,
    int k,
    bool broadcast_w) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  int warp_index = block.group_index().x * rows_per_block + warp.meta_group_rank();
  int row0 = warp_index * rows_per_warp;
  if (row0 >= n) {
    return;
  }

  int m = grid.dim_blocks().y;
  int l = block.group_index().z;
  x += block.group_index().y * k + m * k * l;
  out += block.group_index().y * n + m * n * l;
  int w_batch = broadcast_w ? 0 : l;

  qmv_rowblock_kernel_impl<
      rows_per_warp,
      elems_per_thread,
      group_size,
      has_bias,
      has_residue_k>(
      x, w, scales, biases, global_scale, out, row0, w_batch, n, k);
}

template <
    int rows_per_block,
    int elems_per_thread,
    int group_size,
    bool has_bias,
    bool has_residue_k,
    typename T,
    typename Q,
    typename S>
__global__ void qmv_kernel(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    const float* global_scale,
    T* out,
    int n,
    int k,
    bool broadcast_w) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  // The row that this warp handles.
  int row = block.group_index().x * rows_per_block + warp.meta_group_rank();
  if (row >= n) {
    return;
  }

  // Advance pointers of x/out for M and batch dimensions.
  int m = grid.dim_blocks().y;
  int l = block.group_index().z;
  x += block.group_index().y * k + m * k * l;
  out += block.group_index().y * n + m * n * l;
  int w_batch = broadcast_w ? 0 : l;

  qmv_kernel_impl<elems_per_thread, group_size, has_bias, has_residue_k>(
      x, w, scales, biases, global_scale, out, row, w_batch, n, k);
}

// [mlxcel #725] Multirow launch: 1D grid over weight rows only; every warp
// covers all x_rows input rows.
template <
    int rows_per_block,
    int elems_per_thread,
    int group_size,
    bool has_bias,
    bool has_residue_k,
    int max_x_rows,
    typename T,
    typename Q,
    typename S>
__global__ void qmv_multirow_kernel(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    const float* global_scale,
    T* out,
    int n,
    int k,
    int x_rows) {
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  int row = block.group_index().x * rows_per_block + warp.meta_group_rank();
  if (row >= n) {
    return;
  }

  qmv_multirow_kernel_impl<
      elems_per_thread,
      group_size,
      has_bias,
      has_residue_k,
      max_x_rows>(x, w, scales, biases, global_scale, out, row, x_rows, n, k);
}

template <
    int rows_per_block,
    int elems_per_thread,
    int group_size,
    bool has_bias,
    bool has_residue_k,
    typename T,
    typename Q,
    typename S>
__global__ void gather_qmv_kernel(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    T* out,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    int n,
    int k) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  int row = block.group_index().x * rows_per_block + warp.meta_group_rank();
  if (row >= n) {
    return;
  }

  int m = grid.dim_blocks().y;
  int l = block.group_index().z;
  uint32_t x_idx = lhs_indices[l];
  uint32_t w_idx = rhs_indices[l];

  x += block.group_index().y * k + m * k * x_idx;
  out += block.group_index().y * n + m * n * l;

  qmv_kernel_impl<elems_per_thread, group_size, has_bias, has_residue_k>(
      x, w, scales, biases, nullptr, out, row, w_idx, n, k);
}

// [mlxcel #725] Kill switch: MLXCEL_QMV_MULTIROW=0 restores the stock per-row
// launches. Read once; the decision is process-wide.
inline bool qmv_multirow_enabled() {
  static const bool enabled = []() {
    const char* e = std::getenv("MLXCEL_QMV_MULTIROW");
    return !(e && e[0] == '0' && e[1] == '\0');
  }();
  return enabled;
}

// [mlxcel #906] Row-window ceiling for the multirow path. The #725 window was
// hardcoded at the compile-time `max_x_rows` (8), but the crossover past which
// the small-M qmm shape takes over is a per-hardware property, not a constant
// (docs/CONTINUOUS_BATCHING.md documents a regression past 7 rows on GB10). The
// autotuner tunes it and publishes the winner here; an operator-set value
// always wins. Read once, so the decision is process-wide, matching the kill
// switch above and keeping `getenv` off the per-launch path.
//
// Clamped to [1, hard_max]: the multirow kernel keeps its accumulators in
// registers sized by the compile-time `max_x_rows`, so the window can only be
// narrowed, never widened. A ceiling of 1 disables the path (the gate below
// requires at least 2 rows), which is exactly the kill switch, so the whole
// window including its off state is representable as one integer.
inline int qmv_multirow_max_rows(int hard_max) {
  static const int configured = []() {
    const char* e = std::getenv("MLXCEL_QMV_MULTIROW_MAX_ROWS");
    if (e == nullptr) {
      return 0; // unset: use the compile-time ceiling
    }
    int v = std::atoi(e);
    return v > 0 ? v : 0;
  }();
  if (configured <= 0 || configured > hard_max) {
    return hard_max;
  }
  return configured;
}

// [mlxcel] Output rows per warp for the row-blocked qmv.
//
// `MLXCEL_QMV_ROWS_PER_WARP` overrides `fallback`, which the caller derives
// from the compute capability. Accepted values are 1, 2, 3, 4 and 6; anything
// else is ignored and the fallback stands, because the value picks a compiled
// instantiation and a silent substitution would make an A/B read the wrong arm.
// 1 routes to the stock `qmv_kernel` with the stock grid. Read once per
// process, since the kernel selection is not per-call state.
//
// Measured on a V100 at 4 bits, decode as the slope between -n 60 and -n 200 on
// gemma-4-12B-it-4bit: R = 1 gives 44.7 tok/s, R = 2 gives 56.0, R = 3 gives
// 53.4, R = 4 gives 45.2, R = 6 gives 42.7. Every step up cuts transactions per
// output row and every step up past 2 costs a resident block per SM, and the
// occupancy wins. So the peak sits where the register budget puts it, which is
// what `MLXCEL_QMV_MIN_BLOCKS` exists to move.
inline int qmv_rows_per_warp(int fallback) {
  static const int configured = []() {
    const char* e = std::getenv("MLXCEL_QMV_ROWS_PER_WARP");
    if (e == nullptr) {
      return 0;
    }
    int v = std::atoi(e);
    return (v == 1 || v == 2 || v == 3 || v == 4) ? v : 0;
  }();
  return configured != 0 ? configured : fallback;
}

// [mlxcel] Resident-block floor handed to `__launch_bounds__`, from
// `MLXCEL_QMV_MIN_BLOCKS`.
//
// The R sweep is a curve in occupancy, so the register budget is the knob that
// actually moves it. 1 is the unconstrained case and is what the R numbers in
// `dispatch_rows_per_warp` were measured under; 3 and 4 cap ptxas at 85 and 64
// registers per thread for a 256-thread block, which is what lets a wider R keep
// the residency that made R = 2 win. Anything else is ignored.
inline int qmv_min_blocks(int fallback) {
  static const int configured = []() {
    const char* e = std::getenv("MLXCEL_QMV_MIN_BLOCKS");
    if (e == nullptr) {
      return 0;
    }
    int v = std::atoi(e);
    return (v == 1 || v == 3 || v == 4) ? v : 0;
  }();
  return configured != 0 ? configured : fallback;
}

template <typename F>
inline void dispatch_min_blocks(int blocks, F&& f) {
  switch (blocks) {
    case 4:
      f(std::integral_constant<int, 4>{});
      return;
    case 3:
      f(std::integral_constant<int, 3>{});
      return;
    default:
      f(std::integral_constant<int, 1>{});
      return;
  }
}

// [mlxcel] Compile-time row count for `qmv_rowblock_kernel`, mirroring
// `dispatch_multirow_width`. Only the values `qmv_rows_per_warp` can return are
// instantiated.
template <typename F>
inline void dispatch_rows_per_warp(int rows, F&& f) {
  switch (rows) {
    case 4:
      f(std::integral_constant<int, 4>{});
      return;
    case 3:
      f(std::integral_constant<int, 3>{});
      return;
    default:
      f(std::integral_constant<int, 2>{});
      return;
  }
}

// [mlxcel] Compile-time accumulator width for the multirow kernel.
//
// `max_x_rows` sizes `sums[max_x_rows][elems_per_thread]`, which lives in
// registers, so pinning it to 8 made every launch pay eight rows of
// accumulators no matter how many rows it actually had. Measured on a V100 with
// `cuobjdump -res-usage` at `elems_per_thread` 16: `qmv_kernel` takes 61
// registers and `qmv_multirow_kernel<..., 8>` takes 168, nothing spilled. With
// 65536 registers per SM and a 256-thread block that is 4 resident blocks
// against 1, and dropping from 32 resident warps to 8 leaves too few in flight
// to hide the k-loop's load latency: a 4-row speculative verify block measured
// only 4.5% faster than the per-row path it replaced, where saving three
// quarters of the weight traffic should have been worth far more.
//
// So round the row count up to the next instantiated width instead. The kernel
// already bounds its unrolled j-loop by the runtime `x_rows`, so a width above
// the actual count stays correct; it only wastes registers. Three widths keep
// the instantiation count down and bound the waste at one power of two.
//
// `Cap` is the largest width the caller instantiates, so a narrowed
// `MLXCEL_QMV_MULTIROW_MAX_ROWS` never selects a width the caller did not
// intend to compile.
template <int Cap, typename F>
inline void dispatch_multirow_width(int x_rows, F&& f) {
  if constexpr (Cap >= 4) {
    if (x_rows <= 2) {
      f(std::integral_constant<int, 2>{});
      return;
    }
  }
  if constexpr (Cap >= 8) {
    if (x_rows <= 4) {
      f(std::integral_constant<int, 4>{});
      return;
    }
  }
  f(std::integral_constant<int, Cap>{});
}

template <
    int group_size,
    bool has_bias,
    typename T,
    typename Q,
    typename S,
    typename F>
void qmv(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    const float* global_scale,
    T* out,
    int m,
    int n,
    int k,
    int l,
    bool broadcast_w,
    int default_rows_per_warp,
    int default_min_blocks,
    F&& launch_kernel) {
  constexpr int rows_per_block = 8;
  constexpr int elems_per_thread =
      (cute::sizeof_bits_v<T> <= 16 && cute::sizeof_bits_v<Q> <= 4) ? 16 : 8;

  // [mlxcel #725] Weight-amortizing multirow path for small-M/batched decode:
  // with broadcast weights, all m*l input rows share the weight matrix, so one
  // warp can apply each dequantized weight tile to every row instead of
  // launching one weight-rereading block column per row. Bounded at 8 rows to
  // match the M*B < 8 qmv dispatch window (and keep accumulators in registers).
  // `max_rows` is the compile-time accumulator width, picked per launch by
  // `dispatch_multirow_width`; `window` is the runtime dispatch ceiling, which
  // #906 lets the autotuner narrow.
  constexpr int max_x_rows = 8;
  const int window = qmv_multirow_max_rows(max_x_rows);
  int x_rows = m * l;
  if (broadcast_w && x_rows >= 2 && x_rows <= window &&
      qmv_multirow_enabled()) {
    dim3 num_blocks{uint32_t(cuda::ceil_div(n, rows_per_block)), 1, 1};
    dim3 block_dims{WARP_SIZE, rows_per_block};
    void* args[] = {
        &x, &w, &scales, &biases, &global_scale, &out, &n, &k, &x_rows};
    dispatch_multirow_width<max_x_rows>(x_rows, [&](auto max_rows) {
      dispatch_bool(
          k % (WARP_SIZE * elems_per_thread), [&](auto has_residue_k) {
            auto* kernel = &qmv_multirow_kernel<
                rows_per_block,
                elems_per_thread,
                group_size,
                has_bias,
                has_residue_k.value,
                decltype(max_rows)::value,
                T,
                Q,
                S>;
            launch_kernel(
                reinterpret_cast<void*>(kernel),
                num_blocks,
                block_dims,
                args);
          });
    });
    return;
  }

  // [mlxcel] Row blocking, off by default. R = 1 keeps the stock kernel and
  // the stock grid, so an unset `MLXCEL_QMV_ROWS_PER_WARP` changes nothing.
  const int rows_per_warp = qmv_rows_per_warp(default_rows_per_warp);
  if (rows_per_warp > 1) {
    dim3 num_blocks{
        uint32_t(cuda::ceil_div(n, rows_per_block * rows_per_warp)),
        uint32_t(m),
        uint32_t(l)};
    dim3 block_dims{WARP_SIZE, rows_per_block};
    void* args[] = {
        &x, &w, &scales, &biases, &global_scale, &out, &n, &k, &broadcast_w};
    dispatch_rows_per_warp(rows_per_warp, [&](auto rows) {
      dispatch_min_blocks(qmv_min_blocks(default_min_blocks), [&](auto blocks) {
        dispatch_bool(
            k % (WARP_SIZE * elems_per_thread), [&](auto has_residue_k) {
              auto* kernel = &qmv_rowblock_kernel<
                  rows_per_block,
                  decltype(rows)::value,
                  decltype(blocks)::value,
                  elems_per_thread,
                  group_size,
                  has_bias,
                  has_residue_k.value,
                  T,
                  Q,
                  S>;
              launch_kernel(
                  reinterpret_cast<void*>(kernel),
                  num_blocks,
                  block_dims,
                  args);
            });
      });
    });
    return;
  }

  dim3 num_blocks{
      uint32_t(cuda::ceil_div(n, rows_per_block)), uint32_t(m), uint32_t(l)};
  dim3 block_dims{WARP_SIZE, rows_per_block};
  void* args[] = {
      &x, &w, &scales, &biases, &global_scale, &out, &n, &k, &broadcast_w};

  dispatch_bool(k % (WARP_SIZE * elems_per_thread), [&](auto has_residue_k) {
    auto* kernel = &qmv_kernel<
        rows_per_block,
        elems_per_thread,
        group_size,
        has_bias,
        has_residue_k.value,
        T,
        Q,
        S>;
    launch_kernel(
        reinterpret_cast<void*>(kernel), num_blocks, block_dims, args);
  });
}

template <
    int group_size,
    bool has_bias,
    typename T,
    typename Q,
    typename S,
    typename F>
void gather_qmv(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    T* out,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    int m,
    int n,
    int k,
    int l,
    F&& launch_kernel) {
  constexpr int rows_per_block = 8;
  constexpr int elems_per_thread =
      (cute::sizeof_bits_v<T> <= 16 && cute::sizeof_bits_v<Q> <= 4) ? 16 : 8;

  dim3 num_blocks{
      uint32_t(cuda::ceil_div(n, rows_per_block)), uint32_t(m), uint32_t(l)};
  dim3 block_dims{WARP_SIZE, rows_per_block};
  void* args[] = {
      &x, &w, &scales, &biases, &out, &lhs_indices, &rhs_indices, &n, &k};

  dispatch_bool(k % (WARP_SIZE * elems_per_thread), [&](auto has_residue_k) {
    auto* kernel = &gather_qmv_kernel<
        rows_per_block,
        elems_per_thread,
        group_size,
        has_bias,
        has_residue_k.value,
        T,
        Q,
        S>;
    launch_kernel(
        reinterpret_cast<void*>(kernel), num_blocks, block_dims, args);
  });
}

} // namespace cu

template <typename F>
inline void dispatch_element_types(Dtype dtype, const char* tag, F&& f) {
  if (dtype == float32) {
    f.template operator()<float>();
  } else if (dtype == float16) {
    f.template operator()<cutlass::half_t>();
  } else if (dtype == bfloat16) {
    f.template operator()<cutlass::bfloat16_t>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} Unsupported dtype: {}.", tag, dtype_to_string(dtype)));
  }
}

template <typename F>
inline void dispatch_groups(int group_size, const char* tag, F&& f) {
  if (group_size == 32) {
    f.template operator()<32>();
  } else if (group_size == 64) {
    f.template operator()<64>();
  } else if (group_size == 128) {
    f.template operator()<128>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} Group size {} is not supported.", tag, group_size));
  }
}

template <typename T, typename F>
inline void dispatch_quant_types(
    int bits,
    int group_size,
    QuantizationMode mode,
    const char* tag,
    F&& f) {
  if (mode == QuantizationMode::Mxfp4) {
    f.template operator()<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32>();
  } else if (mode == QuantizationMode::Mxfp8) {
    f.template operator()<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32>();
  } else if (mode == QuantizationMode::Nvfp4) {
    f.template operator()<cutlass::float_e2m1_t, cutlass::float_e4m3_t, 16>();
  } else {
    dispatch_groups(group_size, tag, [&]<int group_size>() {
      if (bits == 2) {
        f.template operator()<cutlass::uint2b_t, T, group_size>();
      } else if (bits == 3) {
        f.template operator()<cutlass::uint3b_t, T, group_size>();
      } else if (bits == 4) {
        f.template operator()<cutlass::uint4b_t, T, group_size>();
      } else if (bits == 5) {
        f.template operator()<cutlass::uint5b_t, T, group_size>();
      } else if (bits == 6) {
        f.template operator()<cutlass::uint6b_t, T, group_size>();
      } else if (bits == 8) {
        f.template operator()<uint8_t, T, group_size>();
      } else {
        throw std::invalid_argument(
            fmt::format("{} {}-bit quantization is not supported.", tag, bits));
      }
    });
  }
}

void qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  const char* tag = "[quantized_matmul]";
  int m = out.shape(-2);
  int n = out.shape(-1);
  int k = x.shape(-1);
  int l = out.size() / (m * n);
  bool broadcast_w = (w.ndim() <= 2) || (w.size() != w.data_size());

  // [mlxcel] Row blocking is on by default only where it was measured. On a
  // V100, R = 2 moves decode from 44.7 to 57.4 tok/s on gemma-4-12B-it-4bit and
  // from 20.6 to 24.9 on qwen3.8-27B-4bit, with byte-identical output at
  // temperature 0. R = 4 gives it all back on the 12B, because at 96 to 142
  // registers it fits one or two blocks per SM against the stock kernel's four
  // or five. Turing is left at 1: sm_75 has a different register file per SM
  // and no Turing device was available to measure, and guessing here would be
  // guessing about the exact tradeoff that already reversed once at R = 4.
  int cc_major = encoder.device().compute_capability_major();
  int cc_minor = encoder.device().compute_capability_minor();
  bool volta = cc_major == 7 && cc_minor == 0;
  int default_rows_per_warp = volta ? 2 : 1;
  // Pinning the resident-block floor is worth more than the row count past 2.
  // Measured on a V100 at 4 bits on gemma-4-12B-it, decode as the slope between
  // -n 60 and -n 200: R = 2 unconstrained gives 56.0 tok/s at 92 to 95 registers
  // and 2 blocks per SM, and the same R = 2 under a floor of 3 gives 60.1 at 80
  // registers and 3 blocks. A floor of 4 caps at 64 registers and falls to 54.9,
  // so the budget can be set too tight as well as too loose. R = 3 under the same
  // floor of 3 lands at 58.8 despite moving 17% fewer transactions per output
  // row, which is why the floor is the knob here and the row count is not.
  int default_min_blocks = volta ? 3 : 1;

  dispatch_element_types(out.dtype(), tag, [&]<typename T>() {
    dispatch_quant_types<T>(
        bits,
        group_size,
        mode,
        tag,
        [&]<typename Q, typename S, int group_size>() {
          encoder.set_input_array(x);
          encoder.set_input_array(w);
          encoder.set_input_array(scales);
          if (biases) {
            encoder.set_input_array(*biases);
          }
          if (global_scale) {
            encoder.set_input_array(*global_scale);
          }
          encoder.set_output_array(out);
          constexpr bool has_bias = !cutlass::has_negative_zero_v<Q>;
          cu::qmv<group_size, has_bias>(
              gpu_ptr<T>(x),
              gpu_ptr<Q>(w),
              gpu_ptr<S>(scales),
              biases ? gpu_ptr<T>(*biases) : nullptr,
              global_scale ? gpu_ptr<float>(*global_scale) : nullptr,
              gpu_ptr<T>(out),
              m,
              n,
              k,
              l,
              broadcast_w,
              default_rows_per_warp,
              default_min_blocks,
              [&](auto* kernel, dim3 num_blocks, dim3 block_dims, void** args) {
                encoder.add_kernel_node_raw(
                    kernel, num_blocks, block_dims, {}, 0, args);
              });
        });
  });
}

void gather_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  const char* tag = "[gather_qmm]";
  int m = out.shape(-2);
  int n = out.shape(-1);
  int k = x.shape(-1);
  int l = out.size() / (m * n);

  dispatch_element_types(out.dtype(), tag, [&]<typename T>() {
    dispatch_quant_types<T>(
        bits,
        group_size,
        mode,
        tag,
        [&]<typename Q, typename S, int group_size>() {
          encoder.set_input_array(x);
          encoder.set_input_array(w);
          encoder.set_input_array(scales);
          if (biases) {
            encoder.set_input_array(*biases);
          }
          encoder.set_input_array(lhs_indices);
          encoder.set_input_array(rhs_indices);
          encoder.set_output_array(out);
          constexpr bool has_bias = !cutlass::has_negative_zero_v<Q>;
          cu::gather_qmv<group_size, has_bias>(
              gpu_ptr<T>(x),
              gpu_ptr<Q>(w),
              gpu_ptr<S>(scales),
              biases ? gpu_ptr<T>(*biases) : nullptr,
              gpu_ptr<T>(out),
              gpu_ptr<uint32_t>(lhs_indices),
              gpu_ptr<uint32_t>(rhs_indices),
              m,
              n,
              k,
              l,
              [&](auto* kernel, dim3 num_blocks, dim3 block_dims, void** args) {
                encoder.add_kernel_node_raw(
                    kernel, num_blocks, block_dims, {}, 0, args);
              });
        });
  });
}

} // namespace mlx::core
