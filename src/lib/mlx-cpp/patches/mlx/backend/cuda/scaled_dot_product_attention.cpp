// Copyright © 2025 Apple Inc.
//
// Patched by mlxcel. The cuDNN execution plan is cached by an LRU keyed on the
// exact q/k/v/mask shapes and strides (build_sdpa_cache_key), and a speculative
// verify round appends new keys every round, so every round of every attention
// layer class missed the cache and rebuilt a plan on the host (about 22 ms per
// build on GB10, 67 to 76 ms per round on the Laguna DFlash pairing, the whole
// fixed floor of that round), and the LRU's lifetime miss counter then aborted
// the process after 2 * MLX_CUDA_SDPA_CACHE_SIZE misses (lablup/mlxcel#1799).
//
// Two mlxcel changes sit on top of upstream 81ba1c6a:
//
// 1. #1820, the general fix: a small multi-row call carrying an array mask over
//    a fixed-size KV cache is canonicalized the way upstream already
//    canonicalizes the one-row decode step. k/v are unsliced to the whole cache
//    buffer, whose extent T_kv is a multiple of 256, the mask is widened to the
//    same T_kv columns with the new columns blocked, and the true lengths reach
//    cuDNN through set_padding_mask / set_seq_len_{q,kv}. One plan then serves
//    every round inside a 256-position bucket.
//    MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0 disables it without a rebuild, and
//    MLXCEL_SDPA_PLAN_DEBUG=1 traces the key fields and plan builds per call.
//
// 2. #1799, the contained fix, now scoped to what bucketing cannot canonicalize
//    (a causal-mode block with no array mask): at most
//    MLXCEL_SDPA_FALLBACK_MAX_QUERIES query rows over a longer key sequence
//    bypasses cuDNN and takes MLX's own ops fallback, which has no per-shape
//    build cost. MLXCEL_SDPA_FALLBACK_MAX_QUERIES=0 restores upstream dispatch.
//
// The one-row decode step takes the vector kernel and never enters cuDNN, and
// prefill keeps cuDNN unchanged (k_len == q_len, or more rows than the bound).

#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <nvtx3/nvtx3.hpp>

#include <limits>

namespace mlx::core {

namespace {

array prepare_sdpa_input(const array& x, Stream s) {
  // SDPA kernel's requirements on inputs:
  // 1. last dim's stride be 1;
  // 2. pointer be aligned.
  if (x.strides(-1) != 1 || get_alignment(x) < 16) {
    array x_copy = contiguous_copy_gpu(x, s);
    auto& encoder = cu::get_command_encoder(s);
    encoder.add_temporary(x_copy);
    return x_copy;
  }
  return x;
}

array prepare_sdpa_sinks(const array& sinks, Stream s) {
  // cuDNN requires sinks to be float32.
  if (sinks.dtype() == float32) {
    return sinks;
  }
  array sinks_f32(sinks.shape(), float32, nullptr, {});
  copy_gpu(sinks, sinks_f32, CopyType::Vector, s);
  auto& encoder = cu::get_command_encoder(s);
  encoder.add_temporary(sinks_f32);
  return sinks_f32;
}

void malloc_with_same_layout(
    cu::CommandEncoder& encoder,
    array& o,
    const array& q) {
  if (q.flags().row_contiguous) {
    o.set_data(cu::malloc_async(o.nbytes(), encoder));
    return;
  }
  // fill_order = argsort(q.strides())
  Shape fill_order(q.ndim());
  std::iota(fill_order.begin(), fill_order.end(), 0);
  std::stable_sort(
      fill_order.begin(), fill_order.end(), [&q](int idx1, int idx2) {
        auto s1 = q.strides(idx1) > 0 ? q.strides(idx1) : 1;
        auto s2 = q.strides(idx2) > 0 ? q.strides(idx2) : 1;
        return s1 < s2;
      });
  // Generate o_strides with fill_order
  Strides o_strides(q.ndim());
  int64_t stride = 1;
  for (int i : fill_order) {
    o_strides[i] = stride;
    stride *= o.shape(i);
  }
  // o is a transposed contiguous array
  o.set_data(
      cu::malloc_async(o.nbytes(), encoder),
      o.size(),
      o_strides,
      {true, false, false});
}

// The cuDNN SDPA is faster than the vector kernel but for a short sequence the
// overhead would kill the advantage. Both MLX's decode canonicalization and
// mlxcel's own KV cache (src/lib/mlxcel-core/src/cache.rs) step by this.
constexpr int kv_cache_step = 256; // number is from mlx-lm

// Allocated sequence extent of the buffer |kv| is a prefix slice of.
inline int64_t kv_buffer_extent(const array& kv) {
  return kv.strides(1) / kv.strides(2);
}

// Allocated extent when |kv| is a leading slice of one contiguous
// [B, H, T_kv, D] buffer, else 0. That identity is the whole safety property
// unslice_kv needs. Upstream's extra `T_kv % kv_cache_step == 0` test is a
// heuristic for recognizing an mlx-lm cache specifically and is kept only on
// the decode path below: mlxcel grows its own buffers by the same step but
// from bases it sets itself (a trimmed window, a prompt-sized first store),
// so the modulus is an accident of the checkpoint rather than a property to
// gate on.
inline int64_t kv_cache_slice_extent(const array& kv) {
  if (kv.ndim() != 4 || kv.shape(2) <= 0 || kv.strides(3) != 1 ||
      kv.strides(2) != kv.shape(3)) {
    return 0;
  }
  int64_t T_kv = kv_buffer_extent(kv);
  if (T_kv < kv.shape(2)) {
    return 0;
  }
  if (kv.size() / kv.shape(2) * T_kv != kv.buffer_size() / kv.itemsize()) {
    return 0;
  }
  return T_kv;
}

// True when |kv| is a leading slice of a contiguous fixed-size KV cache whose
// allocated extent is a multiple of |kv_cache_step|. Upstream's own test,
// unchanged, so the decode path keeps the dispatch it had.
inline bool is_kv_cache_slice(const array& kv) {
  int64_t T_kv = kv_buffer_extent(kv);
  if (kv.size() / kv.shape(2) * T_kv != kv.buffer_size() / kv.itemsize()) {
    return false;
  }
  // It is possible to use heuristic to check slices, but for now just make
  // mlx-lm work.
  return T_kv % kv_cache_step == 0;
}

bool use_cudnn_for_decoding(
    const array& q,
    const array& k,
    const array& v,
    bool has_arr_mask) {
  if (q.shape(2) != 1) {
    return false;
  }
  if (has_arr_mask) {
    return false;
  }
  if (k.shape(2) < kv_cache_step) {
    return false;
  }
  // When called during graph building the strides is not available, and we
  // rely on |supports_sdpa_vector| to decide whether to use fast sdpa since
  // we can fallback to |sdpa_vector|.
  if ((k.status() != array::evaluated) || (v.status() != array::evaluated)) {
    return false;
  }
  return is_kv_cache_slice(k) && is_kv_cache_slice(v);
}

// Get original kv from slices, i.e. undo keys[..., :offset, :]
array unslice_kv(const array& kv) {
  Shape shape = kv.shape();
  shape[2] = /* T_kv */ kv.strides(1) / kv.strides(2);
  array copy(shape, kv.dtype(), nullptr, {});
  copy.copy_shared_buffer(
      kv,
      make_contiguous_strides(shape),
      {true, true, false},
      /* data_size */ kv.buffer_size() / kv.itemsize(),
      /* offset */ -kv.offset());
  return copy;
}

constexpr int QKV_NDIM = 4;

// ---------------------------------------------------------------------------
// mlxcel: plan-cache key bucketing for small masked multi-row calls (#1820).
//
// A speculative verify round appends a new block of keys every round, so the
// exact k/v sequence length and the additive mask's column count are new every
// round and the plan cache below (keyed on both) never hits. MLX already solves
// this for the one-row decode step: it unslices k/v to the whole fixed-size KV
// cache buffer, whose extent T_kv is a multiple of 256, and tells cuDNN the
// true lengths through set_padding_mask / set_seq_len_{q,kv}. This extends the
// same canonicalization to a small multi-row call carrying an array mask, which
// is the shape the verify round uses, by additionally padding the mask out to
// T_kv with blocked (-inf) columns so its shape and strides stop moving too.
// The bucket is T_kv itself, so no buffer is reallocated and no key position
// outside the already-allocated cache is ever addressed.
// ---------------------------------------------------------------------------

// Upper bound on the query rows eligible for bucketing; 0 disables it.
inline int sdpa_plan_bucket_max_queries() {
  static int max_queries =
      env::get_var("MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES", 32);
  return max_queries;
}

// Per-call key/dispatch tracing for the #1820 record; 0 disables it.
inline bool sdpa_plan_debug() {
  static bool on = env::get_var("MLXCEL_SDPA_PLAN_DEBUG", 0) != 0;
  return on;
}

// Shape-only half of the bucketing gate. Must stay answerable during graph
// building, where strides and buffer sizes are not available yet, so that
// |supports_sdpa_cudnn| gives the same answer then as it does at eval time.
inline bool sdpa_plan_bucket_shape_eligible(
    const array& q,
    const array& k,
    bool has_arr_mask,
    bool do_causal) {
  int max_queries = sdpa_plan_bucket_max_queries();
  if (max_queries <= 0 || !has_arr_mask || do_causal) {
    return false;
  }
  int q_len = q.shape(2);
  return q_len > 1 && q_len <= max_queries && k.shape(2) > q_len;
}

// Per-tensor ceiling on a materialized padded k or v, from
// MLXCEL_SDPA_PLAN_BUCKET_MAX_MB. Above it the call keeps its exact shape and
// pays the plan build, rather than moving hundreds of MB per round per layer.
inline int64_t sdpa_plan_bucket_max_bytes() {
  static int64_t mb = env::get_var("MLXCEL_SDPA_PLAN_BUCKET_MAX_MB", 64);
  return mb * 1024 * 1024;
}

// How a call reaches its bucketed key width.
struct SdpaBucketPlan {
  bool apply = false;
  // k/v are a leading slice of one cache buffer, so unslicing them to the whole
  // buffer costs nothing. Otherwise they are copied into a padded buffer.
  bool by_copy = false;
  int64_t bucket = 0;
};

// Layout half of the bucketing gate, decided once the arrays are evaluated.
//
// The mask must be the [B, H, L, k_len] broadcast of a contiguous [L, k_len]
// plane that fast::scaled_dot_product_attention builds, so widening it costs
// L * bucket elements rather than B * H * L * bucket. Given that, k and v reach
// the bucket one of two ways: unsliced to their own cache buffer when they are
// a leading slice of one with room to spare (the dense and speculative-buffered
// caches), or copied into a zero-padded buffer when they are not (a drafter
// that concatenates its proposal keys onto the cache window, and anything else
// that hands SDPA a freshly built array).
SdpaBucketPlan sdpa_plan_bucket(
    const array& q,
    const array& k,
    const array& v,
    const std::optional<array>& mask_arr,
    bool do_causal,
    bool output_logsumexp) {
  SdpaBucketPlan plan;
  if (output_logsumexp || !mask_arr) {
    return plan;
  }
  if (!sdpa_plan_bucket_shape_eligible(q, k, /* has_arr_mask */ true, do_causal)) {
    return plan;
  }
  if ((k.status() != array::evaluated) || (v.status() != array::evaluated) ||
      (mask_arr->status() != array::evaluated)) {
    return plan;
  }
  int k_len = k.shape(2);
  if (k.ndim() != QKV_NDIM || v.ndim() != QKV_NDIM || v.shape(2) != k_len) {
    return plan;
  }
  const array& m = *mask_arr;
  if (m.ndim() != QKV_NDIM || m.shape(2) != q.shape(2) || m.shape(3) != k_len ||
      m.strides(0) != 0 || m.strides(1) != 0 || m.strides(3) != 1 ||
      m.strides(2) != m.shape(3)) {
    return plan;
  }

  int64_t k_extent = kv_cache_slice_extent(k);
  if (k_extent > k_len && k_extent == kv_cache_slice_extent(v)) {
    plan.apply = true;
    plan.bucket = k_extent;
    return plan;
  }

  // Round up to the step both MLX and mlxcel's KV cache grow by, so one plan
  // covers a whole step of appends.
  int64_t bucket =
      (static_cast<int64_t>(k_len) + kv_cache_step - 1) / kv_cache_step * kv_cache_step;
  int64_t rows = static_cast<int64_t>(k.shape(0)) * k.shape(1) * bucket;
  int64_t max_bytes = sdpa_plan_bucket_max_bytes();
  if (rows * k.shape(3) * k.itemsize() > max_bytes ||
      rows * v.shape(3) * v.itemsize() > max_bytes) {
    return plan;
  }
  plan.apply = true;
  plan.by_copy = true;
  plan.bucket = bucket;
  return plan;
}

// Widen the additive mask from [B, H, L, k_len] to [B, H, L, bucket], the new
// columns blocked with -inf. Only the [L, bucket] plane is materialized; the
// leading two axes stay stride-0 broadcasts, exactly as the input was, so the
// key's mask strides land on {0, 0, bucket, 1} every round.
array pad_mask_to_bucket(
    cu::CommandEncoder& encoder,
    const array& mask,
    int bucket,
    Stream s) {
  int q_len = mask.shape(2);
  int k_len = mask.shape(3);
  array core(Shape{q_len, bucket}, mask.dtype(), nullptr, {});
  array blocked(-std::numeric_limits<float>::infinity(), mask.dtype());
  encoder.add_temporary(blocked);
  fill_gpu(blocked, core, s);
  copy_gpu_inplace(
      mask,
      core,
      Shape{q_len, k_len},
      Strides{mask.strides(2), mask.strides(3)},
      Strides{bucket, 1},
      /* i_offset */ 0,
      /* o_offset */ 0,
      CopyType::GeneralGeneral,
      s);
  Shape padded_shape = mask.shape();
  padded_shape[3] = bucket;
  array padded(padded_shape, mask.dtype(), nullptr, {});
  padded.copy_shared_buffer(
      core,
      Strides{0, 0, bucket, 1},
      /* flags */ {false, false, false},
      /* data_size */ core.size(),
      /* offset */ 0);
  encoder.add_temporary(core);
  encoder.add_temporary(padded);
  return padded;
}

// Copy |kv| into a zero-padded [B, H, bucket, D] buffer. Used when k/v are not
// a leading slice of a cache buffer, so there is nothing to unslice. The tail
// is zeroed rather than left at whatever the allocator last held, because zero
// is the one value that cannot turn into a NaN if a future cuDNN release ever
// applied the bias before the padding mask.
array pad_kv_to_bucket(
    cu::CommandEncoder& encoder,
    const array& kv,
    int bucket,
    Stream s) {
  Shape padded_shape = kv.shape();
  padded_shape[2] = bucket;
  array padded(padded_shape, kv.dtype(), nullptr, {});
  array zero(0, kv.dtype());
  encoder.add_temporary(zero);
  fill_gpu(zero, padded, s);
  copy_gpu_inplace(
      kv,
      padded,
      kv.shape(),
      kv.strides(),
      make_contiguous_strides(padded_shape),
      /* i_offset */ 0,
      /* o_offset */ 0,
      CopyType::GeneralGeneral,
      s);
  encoder.add_temporary(padded);
  return padded;
}

// Thrown when cuDNN refuses to build a graph for the bucketed shape, so the
// call can be retried unbucketed instead of aborting the process.
struct SdpaBucketUnsupported : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct SDPACacheKey {
  int device_id;
  fe::DataType_t cudnn_dtype;
  std::array<int, QKV_NDIM> q_shape;
  std::array<int, QKV_NDIM> k_shape;
  std::array<int, QKV_NDIM> v_shape;
  std::array<int64_t, QKV_NDIM> q_strides;
  std::array<int64_t, QKV_NDIM> k_strides;
  std::array<int64_t, QKV_NDIM> v_strides;
  bool do_causal;
  std::array<int, QKV_NDIM> mask_shape;
  std::array<int64_t, QKV_NDIM> mask_strides;
  bool has_sinks;
  bool output_logsumexp;
};

inline BytesKey<SDPACacheKey> build_sdpa_cache_key(
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const std::optional<array>& sinks,
    bool decoding = false,
    bool output_logsumexp = false) {
  BytesKey<SDPACacheKey> cache_key;
  cache_key.pod.device_id = encoder.device().cuda_device();
  cache_key.pod.cudnn_dtype = dtype_to_cudnn_type(q.dtype());
  cache_key.pod.q_shape = vector_key<QKV_NDIM>(q.shape());
  cache_key.pod.k_shape = vector_key<QKV_NDIM>(k.shape());
  cache_key.pod.v_shape = vector_key<QKV_NDIM>(v.shape());
  cache_key.pod.q_strides = vector_key<QKV_NDIM>(q.strides());
  cache_key.pod.k_strides = vector_key<QKV_NDIM>(k.strides());
  cache_key.pod.v_strides = vector_key<QKV_NDIM>(v.strides());
  cache_key.pod.do_causal = do_causal;
  cache_key.pod.has_sinks = sinks.has_value();
  cache_key.pod.output_logsumexp = output_logsumexp;
  if (mask_arr) {
    cache_key.pod.mask_shape = vector_key<QKV_NDIM>(mask_arr->shape());
    cache_key.pod.mask_strides = vector_key<QKV_NDIM>(mask_arr->strides());
  }
  if (decoding) {
    int64_t T_kv = k.strides(1) / k.strides(2);
    cache_key.pod.k_shape[2] = T_kv;
    cache_key.pod.v_shape[2] = T_kv;
    cache_key.pod.k_strides.fill(0);
    cache_key.pod.v_strides.fill(0);
  }
  return cache_key;
}

auto& sdpa_cache() {
  static thread_local LRUBytesKeyCache<SDPACacheKey, DnnGraph> cache(
      "MLX_CUDA_SDPA_CACHE_SIZE", /* default_capacity */ 256);
  return cache;
}

auto& sdpa_backward_cache() {
  static thread_local LRUBytesKeyCache<SDPACacheKey, DnnGraph> cache(
      "MLX_CUDA_SDPA_BACKWARD_CACHE_SIZE", /* default_capacity */ 64);
  return cache;
}

enum UIDS {
  Q,
  K,
  V,
  SCALE,
  BIAS,
  SINKS,
  SEQ_LEN_Q,
  SEQ_LEN_KV,
  O,
  STATS,
  // Backward graph:
  D_Q,
  D_K,
  D_V,
  D_O,
};

DnnGraph build_sdpa_graph(
    cudnnHandle_t handle,
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const std::optional<array>& sinks,
    const std::optional<array>& seq_len_q,
    const std::optional<array>& seq_len_kv,
    bool output_logsumexp,
    const array& o,
    const std::optional<array>& stats) {
  DnnGraph graph(handle, q.dtype());

  auto q_ = graph.tensor("Q", Q, q);
  auto k_ = graph.tensor("K", K, k);
  auto v_ = graph.tensor("V", V, v);

  auto options = fe::graph::SDPA_attributes()
                     .set_name("sdpa_cudnn")
                     .set_attn_scale(graph.scalar("Scale", SCALE, float32))
                     .set_generate_stats(output_logsumexp);
  if (do_causal) {
    options.set_causal_mask_bottom_right(do_causal);
  }
  if (mask_arr) {
    options.set_bias(graph.tensor("BIAS", BIAS, *mask_arr));
  }
  if (sinks) {
    options.set_sink_token(graph.tensor_4d("SINKS", SINKS, *sinks, 1));
  }
  if (seq_len_q && seq_len_kv) {
    options.set_padding_mask(true);
    options.set_seq_len_q(graph.tensor("SEQ_LEN_Q", SEQ_LEN_Q, *seq_len_q));
    options.set_seq_len_kv(graph.tensor("SEQ_LEN_KV", SEQ_LEN_KV, *seq_len_kv));
  }

  auto [o_, stats_] = graph.sdpa(q_, k_, v_, options);
  graph.tensor(o_, O, o)->set_output(true);
  if (output_logsumexp) {
    graph.tensor(stats_, STATS, *stats)->set_output(true);
  }

  CHECK_CUDNN_ERROR(graph.prepare());
  graph.select_behavior_notes(
      {fe::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
  CHECK_CUDNN_ERROR(graph.build());
  return graph;
}

DnnGraph build_sdpa_backward_graph(
    cudnnHandle_t handle,
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const std::optional<array>& sinks,
    const array& o,
    const array& d_o,
    const array& stats,
    array& d_q,
    array& d_k,
    array& d_v) {
  DnnGraph graph(handle, q.dtype());

  auto q_ = graph.tensor("Q", Q, q);
  auto k_ = graph.tensor("K", K, k);
  auto v_ = graph.tensor("V", V, v);
  auto o_ = graph.tensor("O", O, o);
  auto d_o_ = graph.tensor("D_O", D_O, d_o);
  auto stats_ = graph.tensor("STATS", STATS, stats);

  auto options = fe::graph::SDPA_backward_attributes()
                     .set_name("sdpa_backward_cudnn")
                     .set_attn_scale(graph.scalar("Scale", SCALE, float32));
  if (do_causal) {
    options.set_causal_mask_bottom_right(do_causal);
  }
  if (mask_arr) {
    options.set_bias(graph.tensor("BIAS", BIAS, *mask_arr));
  }
  if (sinks) {
    options.set_sink_token(graph.tensor_4d("SINKS", SINKS, *sinks, 1));
  }

  auto [d_q_, d_k_, d_v_] =
      graph.sdpa_backward(q_, k_, v_, o_, d_o_, stats_, options);
  graph.tensor(d_q_, D_Q, d_q)->set_output(true);
  graph.tensor(d_k_, D_K, d_k)->set_output(true);
  graph.tensor(d_v_, D_V, d_v)->set_output(true);

  CHECK_CUDNN_ERROR(graph.prepare());
  graph.select_behavior_notes(
      {fe::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
  CHECK_CUDNN_ERROR(graph.build());
  return graph;
}

} // namespace

void init_cudnn_sdpa_cache() {
  sdpa_cache();
  sdpa_backward_cache();
}

bool supports_sdpa_cudnn(
    const array& q,
    const array& k,
    const array& v,
    bool has_arr_mask,
    bool do_causal,
    Stream s) {
  static bool enabled = env::get_var("MLX_CUDA_USE_CUDNN_SDPA", 1);
  if (!enabled) {
    return false;
  }

  // cuDNN SDPA requires Ampere and later.
  if (cu::device(s.device).compute_capability_major() < 8) {
    return false;
  }

  // mlxcel: small masked query blocks over a longer key sequence are the
  // speculative verify shape; their k_len changes every round, so a cuDNN plan
  // built for them is used once and rebuilt next round (see the file header).
  // When the bucketing added for #1820 can canonicalize the key for such a
  // call (an array mask, which is what a multi-row append carries) it stays on
  // cuDNN and reuses one plan; otherwise #1799's routing applies and it takes
  // MLX's own ops fallback, which has no per-shape build cost. Both halves
  // read only shapes, so this answers the same during graph building, where
  // strides are not available, as it does at eval time.
  static int fallback_max_queries =
      env::get_var("MLXCEL_SDPA_FALLBACK_MAX_QUERIES", 32);
  if ((has_arr_mask || do_causal) && q.shape(2) > 1 &&
      q.shape(2) <= fallback_max_queries && k.shape(2) > q.shape(2) &&
      !sdpa_plan_bucket_shape_eligible(q, k, has_arr_mask, do_causal)) {
    return false;
  }

  // Only use cuDNN for decoding when k/v are slices from fixed-size kv cache.
  if ((q.shape(2) == 1) && !use_cudnn_for_decoding(q, k, v, has_arr_mask)) {
    return false;
  }

  // cuDNN does not support bottom right mask when T_q > T_kv.
  if (do_causal && (q.shape(2) > k.shape(2))) {
    return false;
  }

  // D_qk and D_v must be a multiple of 8 with maximum value 128.
  if ((q.shape(-1) % 8 != 0) || (q.shape(-1) > 128) || (v.shape(-1) % 8 != 0) ||
      (v.shape(-1) > 128)) {
    return false;
  }

  Dtype dtype = q.dtype();
  return dtype == float16 || dtype == bfloat16;
}

static void sdpa_cudnn_impl(
    const array& q,
    array k,
    array v,
    float scale,
    array& o,
    std::optional<array>& stats,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const std::optional<array>& sinks,
    bool output_logsumexp,
    bool allow_bucket,
    Stream s) {
  auto& encoder = cu::get_command_encoder(s);
  auto handle = get_cudnn_handle(encoder.device());

  // For decoding, and for a bucketed small multi-row block (#1820), unslice
  // k/v to the whole KV cache buffer and describe the true lengths with a
  // padding mask, so consecutive calls share one plan-cache key.
  std::optional<array> seq_len_q;
  std::optional<array> seq_len_kv;
  std::optional<array> mask_eff = mask_arr;
  bool decoding = use_cudnn_for_decoding(q, k, v, mask_arr.has_value());
  SdpaBucketPlan plan;
  if (allow_bucket && !decoding) {
    plan = sdpa_plan_bucket(q, k, v, mask_arr, do_causal, output_logsumexp);
  }
  bool bucketed = plan.apply;
  int k_len = k.shape(2);
  int64_t k_extent_dbg = kv_cache_slice_extent(k);
  int64_t k_rowstride_dbg = k.ndim() == 4 ? k.strides(2) : -1;
  int64_t bucket = decoding ? kv_buffer_extent(k) : plan.bucket;
  if (decoding || bucketed) {
    int B = q.shape(0);
    std::vector<int> seq_len_q_vec(B, q.shape(2));
    std::vector<int> seq_len_kv_vec(B, k_len);
    seq_len_q = array(seq_len_q_vec.begin(), {B, 1, 1, 1});
    seq_len_kv = array(seq_len_kv_vec.begin(), {B, 1, 1, 1});
    encoder.add_temporary(*seq_len_q);
    encoder.add_temporary(*seq_len_kv);
    if (bucketed) {
      mask_eff =
          pad_mask_to_bucket(encoder, *mask_arr, static_cast<int>(bucket), s);
    }
    if (bucketed && plan.by_copy) {
      k = pad_kv_to_bucket(encoder, k, static_cast<int>(bucket), s);
      v = pad_kv_to_bucket(encoder, v, static_cast<int>(bucket), s);
    } else {
      k = unslice_kv(k);
      v = unslice_kv(v);
      encoder.add_temporary(k);
      encoder.add_temporary(v);
    }
  }

  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  encoder.set_output_array(o);
  if (mask_eff) {
    encoder.set_input_array(*mask_eff);
  }
  if (sinks) {
    encoder.set_input_array(*sinks);
  }
  if (seq_len_q && seq_len_kv) {
    encoder.set_input_array(*seq_len_q);
    encoder.set_input_array(*seq_len_kv);
  }
  if (output_logsumexp) {
    stats->set_data(cu::malloc_async(stats->nbytes(), encoder));
    encoder.set_output_array(*stats);
  }

  // Search cache.
  auto cache_key = build_sdpa_cache_key(
      encoder,
      q,
      k,
      v,
      do_causal,
      mask_eff,
      sinks,
      decoding || bucketed,
      output_logsumexp);
  auto it = sdpa_cache().find(cache_key);
  bool built = it == sdpa_cache().end();
  if (built) {
    auto build = [&] {
      return build_sdpa_graph(
          handle,
          q,
          k,
          v,
          do_causal,
          mask_eff,
          sinks,
          seq_len_q,
          seq_len_kv,
          output_logsumexp,
          o,
          stats);
    };
    if (bucketed) {
      // A cuDNN install that cannot plan bias together with a padding mask
      // must degrade to the exact-shape plan, not abort the process.
      try {
        it = sdpa_cache().emplace(cache_key, build()).first;
      } catch (const std::exception& e) {
        throw SdpaBucketUnsupported(e.what());
      }
    } else {
      it = sdpa_cache().emplace(cache_key, build()).first;
    }
  }
  auto& graph = it->second;

  if (sdpa_plan_debug()) {
    fprintf(
        stderr,
        "[mlxcel-sdpa] q=%dx%dx%dx%d k_len=%d k_extent=%lld k_rowstride=%lld "
        "mask_cols=%d mask_rowstride=%lld mask_lead=%lld,%lld causal=%d "
        "sinks=%d decode=%d bucket=%d copy=%d bucket_to=%lld built=%d plans=%zu\n",
        q.shape(0),
        q.shape(1),
        q.shape(2),
        q.shape(3),
        k_len,
        static_cast<long long>(k_extent_dbg),
        static_cast<long long>(k_rowstride_dbg),
        mask_eff ? mask_eff->shape(3) : -1,
        mask_eff ? static_cast<long long>(mask_eff->strides(2)) : -1LL,
        mask_eff ? static_cast<long long>(mask_eff->strides(0)) : -1LL,
        mask_eff ? static_cast<long long>(mask_eff->strides(1)) : -1LL,
        static_cast<int>(do_causal),
        static_cast<int>(sinks.has_value()),
        static_cast<int>(decoding),
        static_cast<int>(bucketed),
        static_cast<int>(plan.by_copy),
        static_cast<long long>(bucket),
        static_cast<int>(built),
        sdpa_cache().size());
  }

  std::unordered_map<int64_t, void*> variant_pack{
      {Q, gpu_ptr<void>(q)},
      {K, gpu_ptr<void>(k)},
      {V, gpu_ptr<void>(v)},
      {SCALE, &scale},
      {O, gpu_ptr<void>(o)}};
  if (mask_eff) {
    variant_pack[BIAS] = gpu_ptr<void>(*mask_eff);
  }
  if (sinks) {
    variant_pack[SINKS] = gpu_ptr<void>(*sinks);
  }
  if (seq_len_q && seq_len_kv) {
    variant_pack[SEQ_LEN_Q] = gpu_ptr<void>(*seq_len_q);
    variant_pack[SEQ_LEN_KV] = gpu_ptr<void>(*seq_len_kv);
  }
  if (output_logsumexp) {
    variant_pack[STATS] = gpu_ptr<void>(*stats);
  }

  CHECK_CUDNN_ERROR(graph.encode_graph(encoder, std::move(variant_pack)));
}

void sdpa_cudnn(
    const array& q,
    array k,
    array v,
    float scale,
    array& o,
    std::optional<array>& stats,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const std::optional<array>& sinks,
    bool output_logsumexp,
    Stream s) {
  auto& encoder = cu::get_command_encoder(s);
  malloc_with_same_layout(encoder, o, q);

  // Cleared for the life of the process the first time cuDNN refuses to plan a
  // bucketed shape, so the retry cost is paid at most once. Kept per sinks
  // setting: a cuDNN that cannot plan a sink token beside a padding mask must
  // not take bucketing away from the layers that have no sinks.
  static bool bucket_supported[2] = {true, true};
  bool& bucket_supported_here = bucket_supported[sinks.has_value() ? 1 : 0];
  if (bucket_supported_here) {
    try {
      sdpa_cudnn_impl(
          q,
          k,
          v,
          scale,
          o,
          stats,
          do_causal,
          mask_arr,
          sinks,
          output_logsumexp,
          /* allow_bucket */ true,
          s);
      return;
    } catch (const SdpaBucketUnsupported& e) {
      bucket_supported_here = false;
      fprintf(
          stderr,
          "[mlxcel-sdpa] cuDNN cannot plan the bucketed attention shape "
          "(sinks=%d), falling back to per-length plans: %s\n",
          static_cast<int>(sinks.has_value()),
          e.what());
    }
  }
  sdpa_cudnn_impl(
      q,
      k,
      v,
      scale,
      o,
      stats,
      do_causal,
      mask_arr,
      sinks,
      output_logsumexp,
      /* allow_bucket */ false,
      s);
}

void sdpa_backward_cudnn(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    const array& o,
    const array& stats,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const std::optional<array>& sinks,
    const array& d_o,
    array& d_q,
    array& d_k,
    array& d_v,
    Stream s) {
  auto& encoder = cu::get_command_encoder(s);
  auto handle = get_cudnn_handle(encoder.device());

  malloc_with_same_layout(encoder, d_q, q);
  malloc_with_same_layout(encoder, d_k, k);
  malloc_with_same_layout(encoder, d_v, v);

  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  encoder.set_input_array(o);
  encoder.set_input_array(stats);
  encoder.set_input_array(d_o);
  encoder.set_output_array(d_q);
  encoder.set_output_array(d_k);
  encoder.set_output_array(d_v);
  if (mask_arr) {
    encoder.set_input_array(*mask_arr);
  }
  if (sinks) {
    encoder.set_input_array(*sinks);
  }

  // Search cache.
  auto cache_key =
      build_sdpa_cache_key(encoder, q, k, v, do_causal, mask_arr, sinks);
  auto it = sdpa_backward_cache().find(cache_key);
  if (it == sdpa_backward_cache().end()) {
    auto graph = build_sdpa_backward_graph(
        handle,
        q,
        k,
        v,
        do_causal,
        mask_arr,
        sinks,
        o,
        d_o,
        stats,
        d_q,
        d_k,
        d_v);
    it = sdpa_backward_cache().emplace(cache_key, std::move(graph)).first;
  }
  auto& graph = it->second;

  std::unordered_map<int64_t, void*> variant_pack{
      {Q, gpu_ptr<void>(q)},
      {K, gpu_ptr<void>(k)},
      {V, gpu_ptr<void>(v)},
      {SCALE, &scale},
      {O, gpu_ptr<void>(o)},
      {STATS, gpu_ptr<void>(stats)},
      {D_O, gpu_ptr<void>(d_o)},
      {D_Q, gpu_ptr<void>(d_q)},
      {D_K, gpu_ptr<void>(d_k)},
      {D_V, gpu_ptr<void>(d_v)}};
  if (mask_arr) {
    variant_pack[BIAS] = gpu_ptr<void>(*mask_arr);
  }
  if (sinks) {
    variant_pack[SINKS] = gpu_ptr<void>(*sinks);
  }

  CHECK_CUDNN_ERROR(graph.encode_graph(encoder, std::move(variant_pack)));
}

// Defined in scaled_dot_product_attention.cu file.
bool supports_sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    bool has_arr_mask,
    bool output_logsumexp);
void sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks,
    Stream s);

namespace fast {

namespace {

std::tuple<bool, std::string> has_fused_kernel(
    const array& q,
    const array& k,
    const array& v,
    bool has_arr_mask,
    bool do_causal,
    bool output_logsumexp,
    Stream s) {
  if (s.device != Device::gpu) {
    return {false, "the fused kernels require a GPU stream."};
  }
  if (!supports_sdpa_cudnn(q, k, v, has_arr_mask, do_causal, s) &&
      !supports_sdpa_vector(q, k, v, has_arr_mask, output_logsumexp)) {
    std::ostringstream msg;
    msg << "neither the cuDNN attention nor the vector attention kernel "
        << "supports this configuration; got query shape " << q.shape()
        << ", key shape " << k.shape() << ", value shape " << v.shape()
        << " with dtype " << q.dtype() << ".";
    return {false, msg.str()};
  }
  return {true, ""};
}

} // namespace

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    bool force_fused,
    Stream s) {
  auto [has_fused, reason] =
      has_fused_kernel(q, k, v, has_arr_mask, do_causal, output_logsumexp, s);
  if (force_fused) {
    if (!has_fused) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] force_fused=True but no fused "
             "kernel is available: "
          << reason;
      throw std::invalid_argument(msg.str());
    }
    return false;
  }
  return !has_fused;
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("ScaledDotProductAttention::eval_gpu");

  auto& s = stream();

  array q = prepare_sdpa_input(inputs[0], s);
  array k = prepare_sdpa_input(inputs[1], s);
  array v = prepare_sdpa_input(inputs[2], s);
  array& out = outputs[0];
  bool has_mask = inputs.size() - has_sinks_ > 3;
  bool has_arr_mask = has_mask && !do_causal_;

  std::optional<array> mask_arr;
  if (has_arr_mask) {
    mask_arr = prepare_sdpa_input(inputs[3], s);
  }
  std::optional<array> sinks;
  if (has_sinks_) {
    sinks = inputs.back();
  }
  std::optional<array> stats;
  if (output_logsumexp_) {
    stats = outputs[1];
  }

  if (supports_sdpa_cudnn(q, k, v, has_arr_mask, do_causal_, s)) {
    if (sinks) {
      sinks = prepare_sdpa_sinks(*sinks, s);
    }
    sdpa_cudnn(
        q,
        k,
        v,
        scale_,
        out,
        stats,
        do_causal_,
        mask_arr,
        sinks,
        output_logsumexp_,
        s);
  } else {
    sdpa_vector(q, k, v, scale_, out, do_causal_, sinks, s);
  }
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  // The frontend adds a padding mask when sequence length is not a multiple of
  // tile size.
  if (q.shape(2) % 128 != 0) {
    return true;
  }
  return s.device == Device::cpu;
}

void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("ScaledDotProductAttentionVJP::eval_gpu");

  auto& s = stream();

  assert(inputs.size() >= 6);
  int primals_size = inputs.size() - 3;
  bool has_arr_mask = primals_size > 3 + has_sinks_;

  array q = prepare_sdpa_input(inputs[0], s);
  array k = prepare_sdpa_input(inputs[1], s);
  array v = prepare_sdpa_input(inputs[2], s);
  array o = prepare_sdpa_input(inputs[primals_size], s);
  array stats = prepare_sdpa_input(inputs[primals_size + 1], s);
  array d_o = prepare_sdpa_input(inputs[primals_size + 2], s);

  std::optional<array> mask_arr;
  if (has_arr_mask) {
    mask_arr = prepare_sdpa_input(inputs[3], s);
  }
  std::optional<array> sinks;
  if (has_sinks_) {
    sinks = prepare_sdpa_sinks(inputs.back(), s);
  }

  assert(outputs.size() == 3);
  auto& d_q = outputs[0];
  auto& d_k = outputs[1];
  auto& d_v = outputs[2];

  sdpa_backward_cudnn(
      q,
      k,
      v,
      scale_,
      o,
      stats,
      do_causal_,
      mask_arr,
      sinks,
      d_o,
      d_q,
      d_k,
      d_v,
      s);
}

} // namespace fast

} // namespace mlx::core
