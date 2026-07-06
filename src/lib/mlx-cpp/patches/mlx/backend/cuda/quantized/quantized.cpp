// Copyright © 2025 Apple Inc.
// Patched by mlxcel: (1) ensure input contiguity in QuantizedMatmul for
// non-contiguous 3D batched weights (e.g. GLM-4 MLA embed_q with
// transpose=false); (2) split long-prompt quantized matmuls so no CUDA launch
// exceeds the gridDim.y/z limit of 65535 and no `l = out.size()/(m*n)` int32
// multiply overflows (see lablup/mlxcel#648). Synced to upstream e9463bb
// (post-#3706/#3576 JIT qmm rework and #3723 qmv global scale; the dispatch
// consumed here kept its public signatures, with qmv gaining an optional
// global_scale that QuantizedMatmul passes as std::nullopt).

#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cstdint>

namespace mlx::core {

namespace {

// CUDA gridDim.y and gridDim.z are capped at 65535. The quantized-matmul
// kernels place the row count (m) in grid.y and the batch/gather count (l) in
// grid.z, and qmv/make_problem_shape derive `l = out.size() / (m * n)` with m,n
// as int. Two long-prompt failure modes result (lablup/mlxcel#648): (a) MoE
// GatherQMM makes l = tokens * num_experts_per_tok, which exceeds 65535 once
// tokens*top_k >= 65536; (b) a dense LM-head qmv has m*n = tokens*vocab, which
// overflows int32 once tokens*vocab >= 2^31, so l wraps to 0 and grid.z = 0 is
// rejected as an invalid launch. Both are avoided by splitting the leading
// (row/batch) dimension into slices small enough that every launch keeps its
// grid dims <= 65535 and its m*n < 2^31.
constexpr int64_t kMaxGridDim = 65535;

// [count, inner] row-contiguous view of `src` at leading element offset
// start*inner, sharing src's device buffer (no copy).
array row_view(const array& src, int64_t start, int64_t count, int64_t inner) {
  array v(Shape{static_cast<int>(count), static_cast<int>(inner)}, src.dtype(),
          nullptr, {});
  v.copy_shared_buffer(
      src, Strides{inner, 1}, src.flags(), count * inner, start * inner);
  return v;
}

// [bc, M, N] row-contiguous view selecting flat batch [b0, b0+bc) of `src`.
array batch_view(
    const array& src, int64_t b0, int64_t bc, int64_t M, int64_t N) {
  array v(Shape{static_cast<int>(bc), static_cast<int>(M), static_cast<int>(N)},
          src.dtype(), nullptr, {});
  v.copy_shared_buffer(
      src, Strides{M * N, N, 1}, src.flags(), bc * M * N, b0 * M * N);
  return v;
}

// [bc] flat view selecting elements [b0, b0+bc) of `src`.
array flat_view(const array& src, int64_t b0, int64_t bc) {
  array v(Shape{static_cast<int>(bc)}, src.dtype(), nullptr, {});
  v.copy_shared_buffer(src, Strides{1}, src.flags(), bc, b0);
  return v;
}

// Invoke call_one(x, out) split into row chunks so grid.y stays <= 65535 and
// each chunk's m*n stays < 2^31. Only the unbatched 2D path (the case that
// overflows) is chunked; batched weights keep M small and pass through.
template <typename F>
void run_row_chunked(
    const array& x, array& out, int M, int N, int K, F&& call_one) {
  int64_t cap = std::min<int64_t>(
      kMaxGridDim, static_cast<int64_t>(INT32_MAX) / std::max(N, 1));
  // Only the single-batch (B == 1) case is split along its M (row) axis: the
  // buffer is then exactly M*N contiguous regardless of any leading size-1 dims
  // (e.g. an LM head with out shape [1, tokens, vocab]). Batched weights (B > 1)
  // keep M small, cannot overflow, and pass through unchanged.
  bool single_batch = static_cast<size_t>(out.size()) ==
      static_cast<size_t>(M) * static_cast<size_t>(N);
  if (!single_batch || static_cast<int64_t>(M) <= cap) {
    call_one(x, out);
    return;
  }
  for (int64_t r0 = 0; r0 < M; r0 += cap) {
    int64_t rc = std::min<int64_t>(cap, static_cast<int64_t>(M) - r0);
    array xc = row_view(x, r0, rc, K);
    array oc = row_view(out, r0, rc, N);
    call_one(xc, oc);
  }
}

// Invoke call_one(lhs, rhs, out) split into batch chunks so grid.z (== l) stays
// <= 65535. lhs/rhs indices are treated as flat [B] arrays.
template <typename F>
void run_batch_chunked(
    array& out, const array& lhs, const array& rhs, int M, int N, F&& call_one) {
  int64_t B = static_cast<int64_t>(out.size()) / M / N;
  if (B <= kMaxGridDim) {
    call_one(lhs, rhs, out);
    return;
  }
  for (int64_t b0 = 0; b0 < B; b0 += kMaxGridDim) {
    int64_t bc = std::min<int64_t>(kMaxGridDim, B - b0);
    array oc = batch_view(out, b0, bc, M, N);
    array lc = flat_view(lhs, b0, bc);
    array rc = flat_view(rhs, b0, bc);
    call_one(lc, rc, oc);
  }
}

} // namespace

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("QuantizedMatmul::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  // Ensure row-contiguous inputs so that all dispatch paths can accept them.
  // Without this, 3D batched weights (e.g. MLA embed_q [heads, latent, packed])
  // may be non-contiguous after reshape/transpose and get rejected by every
  // supports_* check, causing a "No implementation" error.
  array x = ensure_row_contiguous(inputs[0], encoder, s);
  array w = ensure_row_contiguous(inputs[1], encoder, s);
  array scales = ensure_row_contiguous(inputs[2], encoder, s);
  std::optional<array> biases;
  if (inputs.size() > 3) {
    biases = ensure_row_contiguous(inputs[3], encoder, s);
  }

  auto supports = [&](auto&& f) {
    return f(
        x,
        w,
        scales,
        biases,
        out,
        transpose_,
        bits_,
        group_size_,
        mode_,
        encoder.device());
  };
  bool can_use_qmm_sm90 = supports(supports_qmm_sm90);
  bool can_use_qmm_sm80 = supports(supports_qmm_sm80);
  bool can_use_qmm_naive = supports(supports_qmm_naive);
  bool can_use_fp_qmv = supports(supports_fp_qmv);
  bool can_use_qmv = supports(supports_qmv) || can_use_fp_qmv;

  int M = out.ndim() > 1 ? out.shape(-2) : 1;
  int N = out.shape(-1);
  int K = x.shape(-1);
  int B = out.size() / M / N;

  auto call_qmm_sm90 = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    run_row_chunked(x, out, M, N, K, [&](const array& xc, array& oc) {
      qmm_sm90(xc, w, scales, *biases, oc, bits_, group_size_, encoder, s);
    });
  };
  auto call_qmm_sm80 = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    run_row_chunked(x, out, M, N, K, [&](const array& xc, array& oc) {
      qmm_sm80(
          xc,
          w,
          scales,
          biases,
          std::nullopt,
          std::nullopt,
          oc,
          bits_,
          group_size_,
          mode_,
          encoder);
    });
  };
  auto call_qmm_naive = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    run_row_chunked(x, out, M, N, K, [&](const array& xc, array& oc) {
      qmm_naive(
          xc,
          w,
          scales,
          biases,
          std::nullopt,
          std::nullopt,
          oc,
          transpose_,
          bits_,
          group_size_,
          mode_,
          encoder);
    });
  };
  auto call_qmv = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    run_row_chunked(x, out, M, N, K, [&](const array& xc, array& oc) {
      if (can_use_fp_qmv) {
        fp_qmv(xc, w, scales, oc, bits_, group_size_, encoder, s);
      } else {
        qmv(xc,
            w,
            scales,
            biases,
            std::nullopt,
            oc,
            bits_,
            group_size_,
            mode_,
            encoder);
      }
    });
  };

  if (can_use_qmm_sm90) {
    if (can_use_qmv && (M == 1 && B == 1 && N <= 16384 && K <= 16384)) {
      call_qmv();
    } else {
      call_qmm_sm90();
    }
    return;
  }

  if (can_use_qmm_sm80) {
    if (can_use_qmv && (M * B < 8)) {
      call_qmv();
    } else {
      call_qmm_sm80();
    }
    return;
  }

  if (can_use_qmm_naive) {
    if (can_use_qmv && (M * B < 8)) {
      call_qmv();
    } else {
      call_qmm_naive();
    }
    return;
  }

  if (can_use_qmv) {
    call_qmv();
    return;
  }

  throw std::runtime_error(
      fmt::format(
          "[quantized_matmul] No implementation for "
          "problem shape: {}x{}x{}x{}, transpose: {}, "
          "activation: {}, bits: {}, group size: {}, mode: \"{}\".",
          M,
          N,
          K,
          B,
          transpose_,
          dtype_to_string(x.dtype()),
          bits_,
          group_size_,
          quantization_mode_to_string(mode_)));
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("GatherQMM::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  array x = ensure_row_contiguous(inputs[0], encoder, s);
  const array& w = inputs[1];
  const array& scales = inputs[2];
  std::optional<array> biases;
  if (inputs.size() == 6) {
    biases = inputs[3];
  }
  array lhs_indices =
      ensure_row_contiguous(inputs[inputs.size() - 2], encoder, s);
  array rhs_indices =
      ensure_row_contiguous(inputs[inputs.size() - 1], encoder, s);

  int M = out.ndim() > 1 ? out.shape(-2) : 1;
  int N = out.shape(-1);
  int K = x.shape(-1);
  int B = out.size() / M / N;

  auto supports = [&](auto&& f) {
    return f(
        x,
        w,
        scales,
        biases,
        out,
        transpose_,
        bits_,
        group_size_,
        mode_,
        encoder.device());
  };
  bool can_use_qmm_sm80 = supports(supports_qmm_sm80);
  bool can_use_qmm_naive = supports(supports_qmm_naive);
  bool can_use_qmv = supports(supports_qmv);

  auto call_qmm_sm80 = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    run_batch_chunked(
        out, lhs_indices, rhs_indices, M, N,
        [&](const array& lc, const array& rc, array& oc) {
          qmm_sm80(
              x,
              w,
              scales,
              biases,
              lc,
              rc,
              oc,
              bits_,
              group_size_,
              mode_,
              encoder);
        });
  };
  auto call_qmm_naive = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    run_batch_chunked(
        out, lhs_indices, rhs_indices, M, N,
        [&](const array& lc, const array& rc, array& oc) {
          qmm_naive(
              x,
              w,
              scales,
              biases,
              lc,
              rc,
              oc,
              transpose_,
              bits_,
              group_size_,
              mode_,
              encoder);
        });
  };
  auto call_qmv = [&]() {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    run_batch_chunked(
        out, lhs_indices, rhs_indices, M, N,
        [&](const array& lc, const array& rc, array& oc) {
          gather_qmv(
              x,
              w,
              scales,
              biases,
              lc,
              rc,
              oc,
              bits_,
              group_size_,
              mode_,
              encoder);
        });
  };

  if (can_use_qmm_sm80) {
    if (can_use_qmv && (M * B < 8)) {
      call_qmv();
    } else {
      call_qmm_sm80();
    }
    return;
  }

  if (can_use_qmm_naive) {
    if (can_use_qmv && (M * B < 8)) {
      call_qmv();
    } else {
      call_qmm_naive();
    }
    return;
  }

  if (can_use_qmv) {
    call_qmv();
    return;
  }

  throw std::runtime_error(
      fmt::format(
          "[gather_qmm] No implementation for "
          "problem shape: {}x{}x{}x{}, transpose: {}, "
          "activation: {}, bits: {}, group size: {}, mode: \"{}\".",
          M,
          N,
          K,
          B,
          transpose_,
          dtype_to_string(x.dtype()),
          bits_,
          group_size_,
          quantization_mode_to_string(mode_)));
}

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("Quantize::eval_gpu");
  auto& s = stream();
  auto& enc = cu::get_command_encoder(s);
  if (dequantize_) {
    auto wq = ensure_row_contiguous(inputs[0], enc, s);
    auto scales = ensure_row_contiguous(inputs[1], enc, s);
    auto& w = outputs[0];

    w.set_data(cu::malloc_async(w.nbytes(), enc));

    if (mode_ == QuantizationMode::Affine) {
      auto biases = ensure_row_contiguous(inputs[2], enc, s);
      affine_dequantize(wq, scales, biases, w, group_size_, bits_, enc, s);
    } else {
      // 0 -- xq, 1 -- scales, 2 -- could be global scale for nvfp4
      bool use_global_scale =
          mode_ == QuantizationMode::Nvfp4 && inputs.size() > 2;
      std::optional<array> global_scale =
          use_global_scale ? std::make_optional(inputs[2]) : std::nullopt;
      fp_dequantize(wq, scales, w, group_size_, bits_, global_scale, enc, s);
    }
  } else {
    auto w = ensure_contiguous(inputs[0], enc, s);
    auto& wq = outputs[0];
    auto& scales = outputs[1];

    wq.set_data(cu::malloc_async(wq.nbytes(), enc));
    scales.set_data(cu::malloc_async(scales.nbytes(), enc));

    if (mode_ == QuantizationMode::Affine) {
      auto& biases = outputs[2];
      biases.set_data(cu::malloc_async(biases.nbytes(), enc));
      affine_quantize(w, wq, scales, biases, group_size_, bits_, enc, s);
    } else {
      bool use_global_scale =
          mode_ == QuantizationMode::Nvfp4 && inputs.size() > 1;
      std::optional<array> global_scale =
          use_global_scale ? std::make_optional(inputs[1]) : std::nullopt;
      fp_quantize(w, wq, scales, group_size_, bits_, global_scale, enc, s);
    }
  }
}

} // namespace mlx::core
