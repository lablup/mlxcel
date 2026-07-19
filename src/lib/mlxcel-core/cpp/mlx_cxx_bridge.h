// Copyright 2025 mlx-lm-rs authors
// Direct C++ bridge for MLX via cxx

#pragma once

#include <memory>
#include <cstdint>
#include <optional>
#include <vector>
#include "rust/cxx.h"
#include "mlx/mlx.h"

namespace mlx_cxx {

// Opaque wrapper struct to hold mlx::core::array
// This allows cxx to manage the lifetime without exposing the complex internals
struct MlxArray {
    mlx::core::array inner;

    explicit MlxArray(mlx::core::array&& arr) : inner(std::move(arr)) {}
    explicit MlxArray(const mlx::core::array& arr) : inner(arr) {}
};

struct MlxQuantizedWeights {
    mlx::core::array weight;
    mlx::core::array scales;
    std::optional<mlx::core::array> biases;

    explicit MlxQuantizedWeights(std::vector<mlx::core::array>&& arrays)
        : weight(std::move(arrays.at(0))),
          scales(std::move(arrays.at(1))) {
        if (arrays.size() > 2) {
            biases.emplace(std::move(arrays.at(2)));
        }
    }
};

// Opaque wrapper for mlx::core::Stream
struct MlxStream {
    mlx::core::Stream inner;

    explicit MlxStream(mlx::core::Stream s) : inner(s) {}
};

// Opaque wrapper for mlx::core::ThreadLocalStream.
//
// A `ThreadLocalStream` is a stream-like handle whose physical
// `mlx::core::Stream` is resolved per-thread on demand via
// `mlx::core::stream_from_thread_local_stream`. Holding the same handle
// across threads gives every thread its own dedicated MLX stream
// without any explicit coordination between them. Used by
// `mlxcel-core` to back the generation stream of `BatchScheduler`,
// `CxxGenerator`, and `SpeculativeGenerator` (upstream MLX commit `728fab1` in mlx-vlm PR #1050).
struct MlxThreadLocalStream {
    mlx::core::ThreadLocalStream inner;

    explicit MlxThreadLocalStream(mlx::core::ThreadLocalStream s) : inner(s) {}
};


// Stream functions.
std::unique_ptr<MlxStream> default_stream();
std::unique_ptr<MlxStream> new_stream_on_device(bool gpu);
void synchronize_stream(const MlxStream& stream);

// Thread-local stream factory bound to the GPU device.
//
// The returned handle is safe to share across threads: each calling
// thread sees its own per-thread MLX stream when it calls
// `stream_from_thread_local_stream`. Used by the generation stream
// owners so that decoding and synchronization always happen on the
// same per-thread stream, even if the owner is later moved between
// threads.
std::unique_ptr<MlxThreadLocalStream> new_thread_local_stream_gpu();

// Resolve the calling thread's `MlxStream` from a thread-local handle.
//
// Each calling thread receives its own `mlx::core::Stream` for the
// device the handle was created on. The same handle returns the same
// per-thread stream across calls on that thread.
std::unique_ptr<MlxStream> stream_from_thread_local_stream(const MlxThreadLocalStream& tls);

// Synchronize the calling thread's stream associated with this handle.
//
// Equivalent to resolving the handle and calling `synchronize_stream`,
// but goes through MLX's `synchronize(ThreadLocalStream)` overload so
// that synchronization is bound to the same per-thread stream that
// dispatched the work.
void synchronize_thread_local_stream(const MlxThreadLocalStream& tls);

// --- Multi-GPU device-index surface (epic #486, sub-issue #487) ---
//
// The boolean device API (`new_stream_on_device(bool)`,
// `set_default_device(bool)`) can only target GPU index 0. These
// index-aware entry points are the foundation for single-node tensor
// parallelism, where each rank's weights and compute live on their own
// physical GPU. They use MLX's portable device API
// (`mlx::core::device_count`, the `Device(DeviceType, index)`
// constructor) so the same code compiles on Metal (one GPU), a CUDA
// multi-GPU host, and a CPU-only build; no CUDA headers are required.

// Number of usable GPUs for the active backend, via
// `mlx::core::device_count(Device::gpu)`. Metal reports 1 (single
// unified-memory GPU), a CUDA build reports the real adapter count, and a
// CPU-only build clamps to 1. Always returns >= 1.
int32_t gpu_device_count();

// New stream pinned to GPU `index` (0-based). `index` must be in
// `[0, gpu_device_count())`; the Rust wrapper validates this before
// calling, so an out-of-range index here is undefined per MLX.
std::unique_ptr<MlxStream> new_stream_on_gpu_index(int32_t index);

// Make GPU `index` the default device for subsequent ops. Mirrors
// `set_default_device(bool)` but targets a specific GPU.
void set_default_device_index(int32_t index);

// Thread-local stream pinned to GPU `index`. Index-aware sibling of
// `new_thread_local_stream_gpu`; each calling thread resolves its own
// per-thread stream on that GPU.
std::unique_ptr<MlxThreadLocalStream> new_thread_local_stream_gpu_index(int32_t index);

// Materialize `a` on the device backing `stream` via `mlx::core::copy`.
// On a multi-GPU CUDA host this is a genuine cross-device copy (used by
// sub-issue 2 to move shards between GPUs); on a single-GPU backend it is
// a same-device copy.
std::unique_ptr<MlxArray> copy_array_to_stream(const MlxArray& a, const MlxStream& stream);

// Array factory functions.
// Create array filled with zeros
std::unique_ptr<MlxArray> zeros(rust::Slice<const int32_t> shape, int32_t dtype);
std::unique_ptr<MlxArray> zeros_stream(rust::Slice<const int32_t> shape, int32_t dtype, const MlxStream& stream);

// Create array filled with ones
std::unique_ptr<MlxArray> ones(rust::Slice<const int32_t> shape, int32_t dtype);
std::unique_ptr<MlxArray> ones_stream(rust::Slice<const int32_t> shape, int32_t dtype, const MlxStream& stream);

// Create array with specific value
std::unique_ptr<MlxArray> full_f32(rust::Slice<const int32_t> shape, float value, int32_t dtype);

// Create identity/eye matrix
std::unique_ptr<MlxArray> eye(int32_t n, int32_t m, int32_t k, int32_t dtype);

// Create linearly spaced values
std::unique_ptr<MlxArray> linspace(float start, float stop, int32_t num, int32_t dtype);

// Create arrays with same shape as input
std::unique_ptr<MlxArray> zeros_like(const MlxArray& a);
std::unique_ptr<MlxArray> ones_like(const MlxArray& a);
std::unique_ptr<MlxArray> full_like(const MlxArray& a, float value);

// Create array from data
std::unique_ptr<MlxArray> from_slice_f32(rust::Slice<const float> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> from_slice_i32(rust::Slice<const int32_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> from_slice_u32(rust::Slice<const uint32_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> from_slice_i64(rust::Slice<const int64_t> data, rust::Slice<const int32_t> shape);

// Create array from raw bytes with specified dtype
std::unique_ptr<MlxArray> from_bytes(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape, int32_t dtype);
std::unique_ptr<MlxArray> from_bytes_nocopy(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape, int32_t dtype);

// Create half-precision array from raw bytes
std::unique_ptr<MlxArray> from_bytes_f16(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape, bool bfloat16);

// Array property accessors.
rust::Vec<int32_t> array_shape(const MlxArray& arr);
int32_t array_dtype(const MlxArray& arr);
size_t array_size(const MlxArray& arr);
size_t array_ndim(const MlxArray& arr);
size_t array_itemsize(const MlxArray& arr);
size_t array_nbytes(const MlxArray& arr);

// Array data access (scalar extraction).
float item_f32(const MlxArray& arr);
int32_t item_i32(const MlxArray& arr);
int64_t item_i64(const MlxArray& arr);
bool item_bool(const MlxArray& arr);

// Copy evaluated array data to a byte buffer.
// Used by: KV cache serialization for disaggregated inference
rust::Vec<uint8_t> array_to_raw_bytes(const MlxArray& arr);
// Fallible counterpart of `array_to_raw_bytes`; declared `-> Result` in the
// bridge so cxx catches a throw from the contiguous copy or eval and surfaces it
// as a Rust `Err` instead of aborting.
rust::Vec<uint8_t> try_array_to_raw_bytes(const MlxArray& arr);

// Copy an ALREADY-CONTIGUOUS array's data to host, evaluating it with the
// surgical per-array `array::eval()` (waits on the array's own completion
// event) rather than `array_to_raw_bytes`' `contiguous()` + `eval()`. The
// `contiguous()` call enqueues a fresh op onto the stream; when the caller has
// just scheduled a later forward on the same stream (the #632 lookahead
// pipeline), evaluating that fresh op would block on the later forward and
// destroy the overlap. This reader adds no op, so it waits only for `arr`.
// Intended for an already-row-contiguous `arr` (e.g. a `fused_sample` output);
// if the array is not row-contiguous it falls back to the safe
// `array_to_raw_bytes` contiguous copy rather than reading past the allocation.
rust::Vec<uint8_t> array_evaluated_bytes(const MlxArray& arr);

// Evaluation.
void eval(const MlxArray& arr);
// Fallible counterpart of `eval`; declared `-> Result<()>` in the bridge so cxx
// catches any thrown MLX exception at the FFI boundary instead of aborting.
void try_eval(const MlxArray& arr);
void eval_all(rust::Slice<const MlxArray* const> arrays);

// Element-wise binary operations.
std::unique_ptr<MlxArray> add(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> subtract(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> remainder(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> multiply(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> divide(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> maximum(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> minimum(const MlxArray& a, const MlxArray& b);

// Element-wise unary operations.
std::unique_ptr<MlxArray> negative(const MlxArray& a);
std::unique_ptr<MlxArray> abs(const MlxArray& a);
std::unique_ptr<MlxArray> exp(const MlxArray& a);
std::unique_ptr<MlxArray> log(const MlxArray& a);
std::unique_ptr<MlxArray> sqrt(const MlxArray& a);
std::unique_ptr<MlxArray> rsqrt(const MlxArray& a);
std::unique_ptr<MlxArray> square(const MlxArray& a);
std::unique_ptr<MlxArray> sin(const MlxArray& a);
std::unique_ptr<MlxArray> cos(const MlxArray& a);
std::unique_ptr<MlxArray> tanh(const MlxArray& a);
std::unique_ptr<MlxArray> sigmoid(const MlxArray& a);
std::unique_ptr<MlxArray> floor(const MlxArray& a);
std::unique_ptr<MlxArray> ceil(const MlxArray& a);
std::unique_ptr<MlxArray> round(const MlxArray& a);
std::unique_ptr<MlxArray> sign(const MlxArray& a);
std::unique_ptr<MlxArray> reciprocal(const MlxArray& a);

// Trigonometric functions
std::unique_ptr<MlxArray> tan(const MlxArray& a);
std::unique_ptr<MlxArray> sinh(const MlxArray& a);
std::unique_ptr<MlxArray> cosh(const MlxArray& a);
std::unique_ptr<MlxArray> arcsin(const MlxArray& a);
std::unique_ptr<MlxArray> arccos(const MlxArray& a);
std::unique_ptr<MlxArray> arctan(const MlxArray& a);
std::unique_ptr<MlxArray> arctan2(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> arcsinh(const MlxArray& a);
std::unique_ptr<MlxArray> arccosh(const MlxArray& a);
std::unique_ptr<MlxArray> arctanh(const MlxArray& a);
std::unique_ptr<MlxArray> degrees(const MlxArray& a);
std::unique_ptr<MlxArray> radians(const MlxArray& a);

// Mathematical/Special functions
std::unique_ptr<MlxArray> erf(const MlxArray& a);
std::unique_ptr<MlxArray> erfinv(const MlxArray& a);
std::unique_ptr<MlxArray> expm1(const MlxArray& a);
std::unique_ptr<MlxArray> log2(const MlxArray& a);
std::unique_ptr<MlxArray> log10(const MlxArray& a);
std::unique_ptr<MlxArray> logaddexp(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> power(const MlxArray& a, const MlxArray& b);

// Checks
std::unique_ptr<MlxArray> isnan(const MlxArray& a);
std::unique_ptr<MlxArray> isinf(const MlxArray& a);
std::unique_ptr<MlxArray> isfinite(const MlxArray& a);
std::unique_ptr<MlxArray> isneginf(const MlxArray& a);
std::unique_ptr<MlxArray> isposinf(const MlxArray& a);

// Reduction operations.
std::unique_ptr<MlxArray> sum_all(const MlxArray& a);
std::unique_ptr<MlxArray> sum_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> mean_all(const MlxArray& a);
std::unique_ptr<MlxArray> mean_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> max_all(const MlxArray& a);
std::unique_ptr<MlxArray> max_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> min_all(const MlxArray& a);
std::unique_ptr<MlxArray> min_axis(const MlxArray& a, int32_t axis, bool keepdims);

// Product reduction
std::unique_ptr<MlxArray> prod_all(const MlxArray& a);
std::unique_ptr<MlxArray> prod_axis(const MlxArray& a, int32_t axis, bool keepdims);

// Variance and standard deviation
std::unique_ptr<MlxArray> var_all(const MlxArray& a);
std::unique_ptr<MlxArray> var_axis(const MlxArray& a, int32_t axis, bool keepdims, int32_t ddof);
std::unique_ptr<MlxArray> std_all(const MlxArray& a);
std::unique_ptr<MlxArray> std_axis(const MlxArray& a, int32_t axis, bool keepdims, int32_t ddof);

// Logsumexp
std::unique_ptr<MlxArray> logsumexp_all(const MlxArray& a);
std::unique_ptr<MlxArray> logsumexp_axis(const MlxArray& a, int32_t axis, bool keepdims);

// All/any reductions
std::unique_ptr<MlxArray> all_all(const MlxArray& a);
std::unique_ptr<MlxArray> any_all(const MlxArray& a);

// Matrix operations.
std::unique_ptr<MlxArray> matmul(const MlxArray& a, const MlxArray& b);
// Fallible counterpart of `matmul`; declared `-> Result` in the bridge so cxx
// catches MLX's eager shape-mismatch exception (and any other throw) and
// surfaces it as a Rust `Err` instead of aborting.
std::unique_ptr<MlxArray> try_matmul(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> transpose(const MlxArray& a);
std::unique_ptr<MlxArray> transpose_axes(const MlxArray& a, rust::Slice<const int32_t> axes);
std::unique_ptr<MlxArray> reshape(const MlxArray& a, rust::Slice<const int32_t> shape);

// Shape operations.
std::unique_ptr<MlxArray> expand_dims(const MlxArray& a, int32_t axis);
std::unique_ptr<MlxArray> expand_dims_multi(
    const MlxArray& a,
    rust::Slice<const int32_t> axes);
std::unique_ptr<MlxArray> squeeze(const MlxArray& a);
std::unique_ptr<MlxArray> squeeze_axis(const MlxArray& a, int32_t axis);
std::unique_ptr<MlxArray> broadcast_to(const MlxArray& a, rust::Slice<const int32_t> shape);

// Flatten array
std::unique_ptr<MlxArray> flatten(const MlxArray& a);
std::unique_ptr<MlxArray> flatten_range(const MlxArray& a, int32_t start_axis, int32_t end_axis);

// Move axis
std::unique_ptr<MlxArray> moveaxis(const MlxArray& a, int32_t source, int32_t destination);

// Pad array
std::unique_ptr<MlxArray> pad(const MlxArray& a, rust::Slice<const int32_t> pad_width, float pad_value);

// Split array at indices
std::unique_ptr<MlxArray> split_at_indices(const MlxArray& a, rust::Slice<const int32_t> indices, int32_t axis);

// Diagonal operations
std::unique_ptr<MlxArray> diag(const MlxArray& a, int32_t k);
std::unique_ptr<MlxArray> diagonal(const MlxArray& a, int32_t offset, int32_t axis1, int32_t axis2);

// Type conversion.
std::unique_ptr<MlxArray> astype(const MlxArray& a, int32_t dtype);

// Copy.
std::unique_ptr<MlxArray> copy(const MlxArray& a);

// High-level operations for LLM inference.
// Softmax along axis
std::unique_ptr<MlxArray> softmax(const MlxArray& a, int32_t axis);

// Softmax along axis with precise=true (f32 accumulation for f16 inputs)
std::unique_ptr<MlxArray> softmax_precise(const MlxArray& a, int32_t axis);

// Log-softmax along axis (numerically stable)
std::unique_ptr<MlxArray> log_softmax(const MlxArray& a, int32_t axis);

// RMS normalization
std::unique_ptr<MlxArray> rms_norm(const MlxArray& x, const MlxArray& weight, float eps);

// Layer normalization
std::unique_ptr<MlxArray> layer_norm(const MlxArray& x, const MlxArray& weight,
                                     const MlxArray& bias, float eps);

// Concatenate arrays along axis
std::unique_ptr<MlxArray> concatenate(rust::Slice<const MlxArray* const> arrays, int32_t axis);

// Split array into multiple parts
rust::Vec<std::unique_ptr<MlxArray>> split(const MlxArray& a, int32_t num_splits, int32_t axis);

// Slice array with start, stop, step
std::unique_ptr<MlxArray> slice(const MlxArray& a,
                                rust::Slice<const int32_t> starts,
                                rust::Slice<const int32_t> stops);

// Slice update: src[starts:stops] = update (for in-place KV cache updates)
// Returns a new array with the update applied
std::unique_ptr<MlxArray> slice_update(const MlxArray& src,
                                        const MlxArray& update,
                                        rust::Slice<const int32_t> starts,
                                        rust::Slice<const int32_t> stops);

// Argmax along axis
std::unique_ptr<MlxArray> argmax(const MlxArray& a, int32_t axis, bool keepdims);

// Where (conditional select)
std::unique_ptr<MlxArray> where_cond(const MlxArray& condition, const MlxArray& x, const MlxArray& y);

// Comparison operations
std::unique_ptr<MlxArray> greater(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> less(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> equal(const MlxArray& a, const MlxArray& b);

// Seed the global MLX random number generator
void random_seed(uint64_t seed);

// Random categorical sampling
std::unique_ptr<MlxArray> random_categorical(const MlxArray& logits, int32_t axis);

// Transformer-specific high-level operations (reduces FFI calls).
// Rotary position embedding (RoPE)
// Returns (cos, sin) for position embedding
std::unique_ptr<MlxArray> rope_forward(
    const MlxArray& x,
    int32_t head_dim,
    float theta,
    int32_t offset,
    bool traditional
);

// Apply rotary embedding to query/key
std::unique_ptr<MlxArray> apply_rope(
    const MlxArray& x,
    const MlxArray& cos,
    const MlxArray& sin
);

// Scaled dot-product attention (entire attention computation in one call)
// q: [batch, n_heads, seq_len, head_dim]
// k: [batch, n_kv_heads, seq_len, head_dim]
// v: [batch, n_kv_heads, seq_len, head_dim]
// mask: optional attention mask
// scale: attention scale factor
std::unique_ptr<MlxArray> scaled_dot_product_attention(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    const MlxArray* mask  // nullable
);

// Linear layer forward (with optional bias)
std::unique_ptr<MlxArray> linear_forward(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray* bias  // nullable
);

// Quantized linear layer forward
// biases: nullable for mxfp4/nvfp4/mxfp8 modes (no per-group bias)
std::unique_ptr<MlxArray> quantized_linear_forward(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,       // nullable for mxfp4/nvfp4/mxfp8
    const MlxArray* linear_bias,  // nullable
    int32_t group_size,
    int32_t bits,
    rust::Str mode
);

// Quantized linear layer forward with an optional post-qmm global scale.
// Applies `global_scale` before the optional dense linear bias, matching the
// ModelOpt NVFP4 sidecar semantics in QuantizedWeight::apply_global_scale.
std::unique_ptr<MlxArray> quantized_linear_forward_global_scale(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,        // nullable for mxfp4/nvfp4/mxfp8
    const MlxArray* global_scale,  // nullable
    const MlxArray* linear_bias,   // nullable
    int32_t group_size,
    int32_t bits,
    rust::Str mode
);

// SwiGLU MLP forward (common in LLMs like Llama)
// output = down_proj(silu(gate_proj(x)) * up_proj(x))
std::unique_ptr<MlxArray> swiglu_mlp_forward(
    const MlxArray& x,
    const MlxArray& gate_proj,
    const MlxArray& up_proj,
    const MlxArray& down_proj
);

// Compiled relu_squared: square(maximum(x, 0)) — single fused kernel
std::unique_ptr<MlxArray> compiled_relu_squared(const MlxArray& x);

// Compiled silu: x * sigmoid(x) — single fused kernel
std::unique_ptr<MlxArray> compiled_silu(const MlxArray& x);

// Compiled gelu: x * 0.5 * (1 + erf(x / sqrt(2))) — single fused kernel
// Used by: StarCoder2 and other precise GELU-based models
std::unique_ptr<MlxArray> compiled_gelu(const MlxArray& x);

// Compiled gelu_approx: erf-based GELU (x * 0.5 * (1 + erf(x / sqrt(2)))) — fused kernel
// Uses erf instead of tanh for numerical stability with bf16 inputs.
// Used by: legacy/tests
std::unique_ptr<MlxArray> compiled_gelu_approx(const MlxArray& x);

// Compiled gelu_topk: sparse GELU with dynamic threshold — single fused kernel
// gelu_approx(max(0, x - (mean + std * multiplier)))
// Used by: Gemma3n MLP layers with activation_sparsity > 0
std::unique_ptr<MlxArray> compiled_gelu_topk(
    const MlxArray& x,
    float std_multiplier
);

// SwiGLU activation only - compiled with kernel fusion (shapeless=true)
// output = silu(gate) * x
// Uses mlx::core::compile for kernel fusion (like Python's @mx.compile)
std::unique_ptr<MlxArray> compiled_swiglu_activation(
    const MlxArray& gate,
    const MlxArray& x
);

// GptOss SwiGLU activation only - compiled with kernel fusion (shapeless=true)
// output = clipped_gate * sigmoid(1.702 * clipped_gate) * (clipped_up + 1)
// Used by: GptOss
std::unique_ptr<MlxArray> compiled_gpt_oss_swiglu_activation(
    const MlxArray& x_linear,
    const MlxArray& x_glu
);

// GeGLU activation - compiled with kernel fusion (shapeless=true)
// output = gelu(gate) * x
// Used by: legacy/tests for precise GeGLU
std::unique_ptr<MlxArray> compiled_geglu_activation(
    const MlxArray& gate,
    const MlxArray& x
);

// GeGLU activation with Python MLX tanh-approx GELU.
// output = gelu_approx(gate) * x
// Used by: Gemma4 MLP and SwitchGeGLU layers
std::unique_ptr<MlxArray> compiled_geglu_approx_activation(
    const MlxArray& gate,
    const MlxArray& x
);

// Compiled softcap attention scores: tanh(scores * inv_cap) * cap
// Fuses divide + tanh + multiply into single compiled kernel
// Used by: Gemma2 attention with logit softcapping
std::unique_ptr<MlxArray> compiled_softcap(
    const MlxArray& scores,
    float cap
);

// Compiled clip_residual for float16 overflow prevention
// When float16: cast to f32, add, clip to f16 range, cast back
// When other dtype: simple addition
// Used by: Gemma3 residual connections
std::unique_ptr<MlxArray> compiled_clip_residual(
    const MlxArray& x,
    const MlxArray& y
);

// Softcap SDPA: Q@K^T * scale -> softcap -> mask -> softmax -> @V
// Combines the entire manual attention path into one compiled call
// Used by: Gemma2 attention with logit softcapping
std::unique_ptr<MlxArray> compiled_softcap_sdpa(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    float softcap,
    const MlxArray* mask
);

// Softcap SDPA with GQA: handles repeat_kv + attention in compiled graph
// Avoids separate repeat_kv FFI calls by incorporating GQA internally
// Used by: Gemma2 attention (GQA + softcap)
std::unique_ptr<MlxArray> compiled_softcap_sdpa_gqa(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    float softcap,
    int32_t n_rep,
    const MlxArray* mask
);

// Compiled GELU MLP forward: down_proj(gelu(gate_proj(x)) * up_proj(x))
// Fuses gate_proj + gelu + up_proj + multiply + down_proj into compiled graph
// Used by: legacy/tests for precise GELU-gated MLP models
std::unique_ptr<MlxArray> compiled_gelu_mlp_forward(
    const MlxArray& x,
    const MlxArray& gate_proj,
    const MlxArray& gate_scales,
    const MlxArray* gate_biases,
    const MlxArray& up_proj,
    const MlxArray& up_scales,
    const MlxArray* up_biases,
    const MlxArray& down_proj,
    const MlxArray& down_scales,
    const MlxArray* down_biases,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
);

// Compiled GELU-approx MLP forward: down_proj(gelu_approx(gate_proj(x)) * up_proj(x))
// Fuses the quantized projections and Python MLX tanh-approx GeGLU.
// Used by: Gemma2, Gemma3, Gemma4 dense MLP
std::unique_ptr<MlxArray> compiled_gelu_approx_mlp_forward(
    const MlxArray& x,
    const MlxArray& gate_proj,
    const MlxArray& gate_scales,
    const MlxArray* gate_biases,
    const MlxArray& up_proj,
    const MlxArray& up_scales,
    const MlxArray* up_biases,
    const MlxArray& down_proj,
    const MlxArray& down_scales,
    const MlxArray* down_biases,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
);

// Compiled GELU-approx MLP forward with per-projection NVFP4 global-scale
// sidecars folded in (issues #698/#705). The gate scale is applied before the
// GeGLU activation, the up scale on the up product, and the down scale on the
// fused output, each reproducing `apply_global_scale` byte-for-byte. Null scale
// pointers mean no multiply for that projection. Native NVFP4 prefill uses a
// shape-specific compiled graph so MLX can select the prefill qmm kernel while
// still fusing the sidecar folds. NVFP4 carries no quant biases, so this
// signature omits the bias operands. Used by: Gemma 4 dense MLP.
std::unique_ptr<MlxArray> compiled_gelu_approx_mlp_forward_global_scale(
    const MlxArray& x,
    const MlxArray& gate_proj,
    const MlxArray& gate_scales,
    const MlxArray& up_proj,
    const MlxArray& up_scales,
    const MlxArray& down_proj,
    const MlxArray& down_scales,
    const MlxArray* gate_global_scale,
    const MlxArray* up_global_scale,
    const MlxArray* down_global_scale,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
);

// Compiled GeGLU SwitchGLU MLP forward for quantized MoE experts.
// Wraps three `gather_qmm` calls (gate/up/down) plus a tanh-approx
// GeGLU activation into a single `mx::core::compile` window so MLX
// can schedule gate/up in parallel and fuse the intermediate
// element-wise ops. Only the no-sort path is fused; callers should
// fall back to separate `gather_qmm` calls when `sorted_indices` is
// true (prefill). Used by: Gemma 4 26B-a4b SwitchGeGLU experts.
std::unique_ptr<MlxArray> compiled_switch_qgeglu_forward(
    const MlxArray& x,
    const MlxArray& gate_w,
    const MlxArray& gate_s,
    const MlxArray* gate_b,
    const MlxArray& up_w,
    const MlxArray& up_s,
    const MlxArray* up_b,
    const MlxArray& down_w,
    const MlxArray& down_s,
    const MlxArray* down_b,
    const MlxArray& rhs_indices,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
);

// Compiled SwiGLU MLP forward for non-quantized (FP16/BF16) weights:
//   down_proj(silu(gate_proj(x)) * up_proj(x))
// Fuses gate_proj + silu + up_proj + multiply + down_proj into compiled graph.
// Used by: Llama, Qwen2, Qwen3, Mistral and other SwiGLU FP models
std::unique_ptr<MlxArray> compiled_swiglu_mlp_forward_fp16(
    const MlxArray& x,
    const MlxArray& gate_weight,
    const MlxArray& up_weight,
    const MlxArray& down_weight,
    const MlxArray* gate_bias,
    const MlxArray* up_bias,
    const MlxArray* down_bias
);

// Compiled GELU MLP forward for non-quantized (FP16/BF16) weights:
//   down_proj(gelu(gate_proj(x)) * up_proj(x))
// Fuses gate_proj + gelu + up_proj + multiply + down_proj into compiled graph.
// Used by: Gemma, Gemma4 and other GELU-gated FP models
std::unique_ptr<MlxArray> compiled_gelu_mlp_forward_fp16(
    const MlxArray& x,
    const MlxArray& gate_weight,
    const MlxArray& up_weight,
    const MlxArray& down_weight,
    const MlxArray* gate_bias,
    const MlxArray* up_bias,
    const MlxArray* down_bias
);

// Gemma3n dense MLP forward for non-quantized bf16 language MLP weights:
//   cast input to bf16 -> gate/up -> gelu_approx or gelu_topk -> down -> bf16.
// Keeps the same bf16 semantics as the Rust op-at-a-time path while collapsing
// the decode-hot MLP graph construction into one C++ bridge call.
std::unique_ptr<MlxArray> gemma3n_mlp_forward(
    const MlxArray& x,
    const MlxArray& gate_weight,
    const MlxArray& up_weight,
    const MlxArray& down_weight,
    const MlxArray* gate_bias,
    const MlxArray* up_bias,
    const MlxArray* down_bias,
    float activation_sparsity,
    float std_multiplier
);

// Full transformer layer forward (maximum FFI reduction)
// Combines: attention + MLP + residuals + norms
std::unique_ptr<MlxArray> transformer_layer_forward(
    const MlxArray& x,
    const MlxArray& attn_norm_weight,
    const MlxArray& q_proj,
    const MlxArray& k_proj,
    const MlxArray& v_proj,
    const MlxArray& o_proj,
    const MlxArray& ffn_norm_weight,
    const MlxArray& gate_proj,
    const MlxArray& up_proj,
    const MlxArray& down_proj,
    const MlxArray* kv_cache_k,  // nullable for first token
    const MlxArray* kv_cache_v,  // nullable for first token
    int32_t n_heads,
    int32_t n_kv_heads,
    int32_t head_dim,
    float rope_theta,
    int32_t rope_offset,
    float norm_eps
);

// Advanced indexing operations.
// Take elements along an axis using indices
std::unique_ptr<MlxArray> take(const MlxArray& a, const MlxArray& indices, int32_t axis);

// Gather elements using indices (multi-dimensional indexing)
// indices can be a vector of index arrays for each dimension
std::unique_ptr<MlxArray> gather(
    const MlxArray& a,
    rust::Slice<const MlxArray* const> indices,
    rust::Slice<const int32_t> axes,
    rust::Slice<const int32_t> slice_sizes
);

// Take along axis (like numpy.take_along_axis)
std::unique_ptr<MlxArray> take_along_axis(const MlxArray& a, const MlxArray& indices, int32_t axis);

// Put along axis (scatter update)
std::unique_ptr<MlxArray> put_along_axis(const MlxArray& a, const MlxArray& indices,
                                          const MlxArray& values, int32_t axis);

// Stack arrays along new axis
std::unique_ptr<MlxArray> stack(rust::Slice<const MlxArray* const> arrays, int32_t axis);

// Tile/repeat array
std::unique_ptr<MlxArray> tile(const MlxArray& a, rust::Slice<const int32_t> reps);
std::unique_ptr<MlxArray> repeat(const MlxArray& a, int32_t repeats, int32_t axis);

// Arange
std::unique_ptr<MlxArray> arange_f32(float start, float stop, float step);
std::unique_ptr<MlxArray> arange_i32(int32_t start, int32_t stop, int32_t step);

// Logical operations.
std::unique_ptr<MlxArray> logical_not(const MlxArray& a);
std::unique_ptr<MlxArray> logical_and(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> logical_or(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> all_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> any_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> greater_equal(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> less_equal(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> not_equal(const MlxArray& a, const MlxArray& b);

// Activation functions.
std::unique_ptr<MlxArray> silu(const MlxArray& a);
std::unique_ptr<MlxArray> gelu(const MlxArray& a);
std::unique_ptr<MlxArray> gelu_approx(const MlxArray& a);
std::unique_ptr<MlxArray> relu(const MlxArray& a);
std::unique_ptr<MlxArray> leaky_relu(const MlxArray& a, float negative_slope);

// Sorting and searching.
std::unique_ptr<MlxArray> argsort(const MlxArray& a, int32_t axis);
std::unique_ptr<MlxArray> argpartition(const MlxArray& a, int32_t kth, int32_t axis);
std::unique_ptr<MlxArray> argmin(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> topk(const MlxArray& a, int32_t k, int32_t axis);

// Sort and partition
std::unique_ptr<MlxArray> sort(const MlxArray& a, int32_t axis);
std::unique_ptr<MlxArray> partition(const MlxArray& a, int32_t kth, int32_t axis);

// Cumulative operations
std::unique_ptr<MlxArray> cummax(const MlxArray& a, int32_t axis, bool reverse, bool inclusive);
std::unique_ptr<MlxArray> cummin(const MlxArray& a, int32_t axis, bool reverse, bool inclusive);
std::unique_ptr<MlxArray> cumprod(const MlxArray& a, int32_t axis, bool reverse, bool inclusive);

// Scatter operations
std::unique_ptr<MlxArray> scatter(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis);
std::unique_ptr<MlxArray> scatter_add(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis);
std::unique_ptr<MlxArray> scatter_max(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis);
std::unique_ptr<MlxArray> scatter_min(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis);
std::unique_ptr<MlxArray> scatter_prod(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis);

// Bitwise operations
std::unique_ptr<MlxArray> bitwise_and(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> bitwise_or(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> bitwise_xor(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> left_shift(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> right_shift(const MlxArray& a, const MlxArray& b);

// Linear algebra
std::unique_ptr<MlxArray> tensordot(const MlxArray& a, const MlxArray& b, int32_t axes);
std::unique_ptr<MlxArray> inner(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> outer(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> trace(const MlxArray& a, int32_t offset, int32_t axis1, int32_t axis2);

// Roll (circular shift)
std::unique_ptr<MlxArray> roll(const MlxArray& a, int32_t shift, int32_t axis);

// Nan handling
std::unique_ptr<MlxArray> nan_to_num(const MlxArray& a, float nan_val, float posinf_val, float neginf_val);

// Stop gradient
std::unique_ptr<MlxArray> stop_gradient(const MlxArray& a);

// 2D convolution
std::unique_ptr<MlxArray> conv2d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t groups
);
// Fallible counterpart of `conv2d`; declared `-> Result` in the bridge so cxx
// catches MLX's eager shape-mismatch exception (and any other throw) and
// surfaces it as a Rust `Err` instead of aborting.
std::unique_ptr<MlxArray> try_conv2d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t groups
);

// 2D average pooling
// Used by: VisionModule (Gemma3 AvgPool projector)
std::unique_ptr<MlxArray> avg_pool2d(
    const MlxArray& input,
    int32_t kernel_h, int32_t kernel_w,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w
);

// MoE (Mixture of Experts) operations.
// Gather matrix multiply for MoE
// sorted_indices: if true, lhs_indices are pre-sorted for better memory access
std::unique_ptr<MlxArray> gather_mm(
    const MlxArray& a,
    const MlxArray& b,
    const MlxArray* lhs_indices,    // nullable
    const MlxArray* rhs_indices,    // nullable
    bool sorted_indices
);

// Gather quantized matrix multiply for MoE
// sorted_indices: if true, lhs_indices are pre-sorted for better memory access
std::unique_ptr<MlxArray> gather_qmm(
    const MlxArray& x,
    const MlxArray& w,
    const MlxArray& scales,
    const MlxArray* biases,         // nullable for no-bias quantization
    const MlxArray* lhs_indices,    // nullable
    const MlxArray* rhs_indices,    // nullable
    bool transpose,
    int32_t group_size,
    int32_t bits,
    bool sorted_indices,
    rust::Str mode
);

// Direct quantized matrix multiplication
// y = x @ dequantize(w, scales, biases).T if transpose else x @ dequantize(w, scales, biases)
std::unique_ptr<MlxArray> quantized_matmul(
    const MlxArray& x,
    const MlxArray& w,
    const MlxArray& scales,
    const MlxArray* biases,         // nullable for no-bias quantization
    bool transpose,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
);

// Dequantize quantized weights
// Returns full-precision weights from quantized representation
std::unique_ptr<MlxArray> dequantize(
    const MlxArray& w,
    const MlxArray& scales,
    const MlxArray* biases,     // nullable for mxfp4/nvfp4/mxfp8
    int32_t group_size,
    int32_t bits,
    rust::Str mode
);

// Embedding.
std::unique_ptr<MlxArray> embedding(const MlxArray& weight, const MlxArray& indices);

// Quantized embedding lookup with dequantization
std::unique_ptr<MlxArray> quantized_embedding(
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,     // nullable for mxfp4/nvfp4/mxfp8
    const MlxArray& indices,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
);

// Fast operations (using MLX fast kernels).
// Fast RoPE using MLX fast kernel
std::unique_ptr<MlxArray> fast_rope(
    const MlxArray& x,
    int32_t dims,
    bool traditional,
    float base,
    float scale,
    int32_t offset
);

// Fast RoPE with custom frequencies (for Yarn RoPE)
std::unique_ptr<MlxArray> fast_rope_with_freqs(
    const MlxArray& x,
    int32_t dims,
    bool traditional,
    float scale,
    int32_t offset,
    const MlxArray& freqs
);

// Compiled ProportionalRoPE (Gemma 4 full-attention layers). Wraps the
// mlx-lm full-head `fast::rope` call with an `inf` frequency tail in one
// `mx::core::compile` window. Requires `rotated_dims > 0` and
// `last_dim == head_dim`; the rare `last_dim > head_dim` tail case must
// stay on the op-at-a-time path. `offset` flows through as a scalar array
// input so the same compiled graph serves every decode step.
std::unique_ptr<MlxArray> compiled_proportional_rope(
    const MlxArray& x,
    const MlxArray& freqs,
    int32_t head_dim,
    int32_t rotated_dims,
    int32_t offset
);

// Compiled Gemma 4 Q-path with proportional RoPE. Folds
// `reshape → fast::rms_norm → transpose → full-head ProportionalRoPE`
// into one compile window so MLX sees a single fused subgraph instead of
// four cxx-bridge calls. Used on Gemma 4 full-attention layers only.
std::unique_ptr<MlxArray> compiled_q_path_proportional(
    const MlxArray& q_proj_out,
    const MlxArray& q_norm_weight,
    const MlxArray& freqs,
    float rms_eps,
    int32_t n_heads,
    int32_t head_dim,
    int32_t rotated_dims,
    int32_t offset
);

// Compiled Gemma 4 per-layer-input-gate chain (e2b / e4b variants).
// Fuses `gate_proj → gelu_approx → multiply(per_layer) → proj →
// post_norm → add(after_ffn)` into one compile window. Requires
// affine / gs=64 / bits=4 with biases present; other modes fall
// through to an op-at-a-time fallback.
std::unique_ptr<MlxArray> compiled_per_layer_input_gate(
    const MlxArray& after_ffn,
    const MlxArray& per_layer_input,
    const MlxArray& gate_w,
    const MlxArray& gate_s,
    const MlxArray* gate_b,
    const MlxArray& proj_w,
    const MlxArray& proj_s,
    const MlxArray* proj_b,
    const MlxArray& post_norm_w,
    const MlxArray* gate_global_scale,
    const MlxArray* proj_global_scale,
    float post_norm_eps,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
);

// Fast RMS norm using MLX fast kernel
std::unique_ptr<MlxArray> fast_rms_norm(
    const MlxArray& x,
    const MlxArray& weight,
    float eps
);

// Fast RMS norm without a learnable scale
std::unique_ptr<MlxArray> fast_rms_norm_no_weight(
    const MlxArray& x,
    float eps
);

// Fast layer norm using MLX fast kernel
std::unique_ptr<MlxArray> fast_layer_norm(
    const MlxArray& x,
    const MlxArray* weight,  // nullable
    const MlxArray* bias,    // nullable
    float eps
);

// Fast scaled dot product attention using MLX fast kernel
std::unique_ptr<MlxArray> fast_scaled_dot_product_attention(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    const MlxArray* mask  // nullable
);

// Fast SDPA with optional sinks (per-head attention bias for first position)
// Used by: GptOss
std::unique_ptr<MlxArray> fast_scaled_dot_product_attention_with_sinks(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    const MlxArray* mask,
    const MlxArray* sinks
);

// SDPA with explicit causal masking for prefill (no mask array needed)
std::unique_ptr<MlxArray> fast_scaled_dot_product_attention_causal(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale
);

// Decode-only paged attention over dense compatibility KV caches.
std::unique_ptr<MlxArray> paged_decode_attention_dense_compat(
    const MlxArray& q,
    rust::Slice<const MlxArray* const> cache_keys,
    rust::Slice<const MlxArray* const> cache_values,
    rust::Slice<const int32_t> kv_lens,
    rust::Slice<const int32_t> block_tables,
    rust::Slice<const int32_t> block_table_offsets,
    int32_t block_size,
    float scale
);

// Decode-only paged attention over rotating ring-buffer KV caches.
std::unique_ptr<MlxArray> paged_decode_attention_rotating_compat(
    const MlxArray& q,
    rust::Slice<const MlxArray* const> cache_keys,
    rust::Slice<const MlxArray* const> cache_values,
    rust::Slice<const int32_t> kv_lens,
    rust::Slice<const int32_t> logical_starts,
    int32_t block_size,
    float scale
);

// Upstream MLX SDPA capability helpers for Metal/NAX instrumentation.
bool sdpa_supports_fast_path(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal
);

bool sdpa_supports_nax(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal
);

// Fused QKV projection + reshape + transpose + RoPE
// Reduces FFI overhead for the projection chain
std::unique_ptr<MlxArray> fused_qkv_project_and_rope(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,     // nullable for mxfp4/nvfp4/mxfp8
    int32_t num_heads,
    int32_t head_dim,
    int32_t rope_dims,
    float rope_base,
    int32_t cache_offset,
    int32_t group_size,
    int32_t bits,
    bool apply_rope,
    rust::Str mode
);

// Fused concatenated QKV projection + split + reshape + transpose + RoPE.
// Used by: Llama3-family and Gemma2 fused attention preparation paths.
void fused_qkv_project_split_rope(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,     // nullable for mxfp4/nvfp4/mxfp8
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t rope_dims,
    float rope_base,
    int32_t cache_offset,
    int32_t group_size,
    int32_t bits,
    rust::Str mode,
    std::unique_ptr<MlxArray>& q_out,
    std::unique_ptr<MlxArray>& k_out,
    std::unique_ptr<MlxArray>& v_out
);

// Fused concatenated QKV projection + split + reshape + transpose +
// GemmaRMSNorm(Q/K) + RoPE.
// Used by: Gemma3 dense attention preparation path.
void fused_qkv_project_split_norm_rope(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,     // nullable for mxfp4/nvfp4/mxfp8
    const MlxArray& q_norm_weight,
    const MlxArray& k_norm_weight,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t rope_dims,
    float rope_base,
    float rms_eps,
    int32_t cache_offset,
    int32_t group_size,
    int32_t bits,
    rust::Str mode,
    std::unique_ptr<MlxArray>& q_out,
    std::unique_ptr<MlxArray>& k_out,
    std::unique_ptr<MlxArray>& v_out
);

// Fused concatenated QKV projection + split + reshape + transpose +
// SuScaledRoPE. Mirrors mlx-lm/mlx-vlm SuScaledRoPE by scaling only the
// rotary prefix before applying custom frequency RoPE.
// Used by: Phi3/Phi3V longrope-su attention path.
void fused_qkv_project_split_su_scaled_rope(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,     // nullable for mxfp4/nvfp4/mxfp8
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t rope_dims,
    const MlxArray& rope_freqs,
    float rope_input_scale,
    int32_t cache_offset,
    int32_t group_size,
    int32_t bits,
    rust::Str mode,
    std::unique_ptr<MlxArray>& q_out,
    std::unique_ptr<MlxArray>& k_out,
    std::unique_ptr<MlxArray>& v_out
);

// Experimental dense causal prefill attention path:
// qkv projection + split + rope + native causal SDPA + output projection.
// Returns output plus K/V tensors so Rust can populate the KV cache.
void fused_causal_prefill_attention(
    const MlxArray& x,
    const MlxArray& qkv_weight,
    const MlxArray& qkv_scales,
    const MlxArray* qkv_biases,
    const MlxArray& o_weight,
    const MlxArray& o_scales,
    const MlxArray* o_biases,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t rope_dims,
    float rope_base,
    float scale,
    int32_t group_size,
    int32_t bits,
    rust::Str mode,
    std::unique_ptr<MlxArray>& output_out,
    std::unique_ptr<MlxArray>& k_out,
    std::unique_ptr<MlxArray>& v_out
);

// Compiled operations (with kernel fusion).
// Compiled full MoE expert forward
// Compiles: silu(gate_proj(x)) * up_proj(x), then down_proj
// Note: compiled path only supports affine mode; non-affine modes fall back to non-compiled
std::unique_ptr<MlxArray> compiled_moe_expert_forward(
    const MlxArray& x,
    const MlxArray& gate_proj,
    const MlxArray& gate_scales,
    const MlxArray* gate_biases,    // nullable for mxfp4/nvfp4/mxfp8
    const MlxArray& up_proj,
    const MlxArray& up_scales,
    const MlxArray* up_biases,      // nullable for mxfp4/nvfp4/mxfp8
    const MlxArray& down_proj,
    const MlxArray& down_scales,
    const MlxArray* down_biases,    // nullable for mxfp4/nvfp4/mxfp8
    int32_t group_size,
    int32_t bits,
    rust::Str mode
);

// Memory and stream management.
void clear_memory_cache();

// Async evaluation
void async_eval(const MlxArray& arr);
// Fallible counterpart of `async_eval`; declared `-> Result<()>` in the bridge
// so cxx catches any thrown MLX exception at the FFI boundary instead of
// aborting. Body is identical to `async_eval`.
void try_async_eval(const MlxArray& arr);
void async_eval_all(rust::Slice<const MlxArray* const> arrays);
void detach_all(rust::Slice<const MlxArray* const> arrays);

// Synchronize stream
void synchronize_default();

// Set default device for subsequent operations
void set_default_device(bool gpu);

// Memory limits
size_t set_wired_limit(size_t limit);
size_t get_wired_limit();

// MLX runtime memory accounting (issue #55).
//
// These wrap `mlx::core::get_active_memory()` / `get_peak_memory()` /
// `get_cache_memory()` / `set_memory_limit()` / `get_memory_limit()` /
// `set_cache_limit()` / `reset_peak_memory()` from `mlx/memory.h`. The
// numbers are populated by whichever allocator is active (Metal, CUDA, or
// the no-gpu common allocator) — see the per-backend implementations in
// `mlx/backend/<metal|cuda|no_gpu>/allocator.cpp`. On the no-gpu CPU
// allocator `get_cache_memory()` / `set_cache_limit()` are inert no-ops
// and return 0 by design; this matches MLX upstream semantics and lets
// the same Rust wrapper compile and run on Linux without panicking.
size_t get_active_memory();
size_t get_peak_memory();
size_t get_cache_memory();
size_t set_memory_limit(size_t limit);
size_t get_memory_limit();
size_t set_cache_limit(size_t limit);
void reset_peak_memory();

// GPU memory info (works across Metal and CUDA backends)
size_t gpu_max_memory_size();

// Create new stream on GPU
std::unique_ptr<MlxStream> new_gpu_stream();

// Optimized generation functions.
// Extract last token logits: logits[:, -1, :] -> [batch, vocab]
// Optimized for sampling during generation
std::unique_ptr<MlxArray> slice_last_logits(const MlxArray& logits);

// Slice on the last dimension only: a[..., start:end]
// Useful for fused QKV/gate_up projections
std::unique_ptr<MlxArray> slice_last_dim(const MlxArray& a, int32_t start, int32_t end);

// Argmax on last axis for greedy sampling
std::unique_ptr<MlxArray> argmax_last_axis(const MlxArray& a);

// Reshape token for next forward pass: [] or [batch] -> [batch, 1]
std::unique_ptr<MlxArray> reshape_token_for_forward(const MlxArray& token);

// Async eval two arrays at once (for lookahead pipelining)
void async_eval_pair(const MlxArray& a, const MlxArray& b);

// Export a pair of unevaluated arrays as a DOT graph for profiling.
void export_to_dot_pair(rust::Str path, const MlxArray& a, const MlxArray& b);

// Count AsType (dtype-conversion) nodes in the unevaluated graph for a pair.
uint64_t count_astype_nodes_pair(const MlxArray& a, const MlxArray& b);

// Human-readable AsType breakdown (counts + per src->dst dtype pair) for a pair.
rust::String astype_breakdown_pair(const MlxArray& a, const MlxArray& b);

// Set default stream for subsequent operations
void set_default_stream(const MlxStream& stream);

// Check if GPU is available
bool is_gpu_available();

// Fused sampling: temperature scaling + top-k + top-p + min-p + categorical
// in a single function call to minimize FFI round-trips.
// Input: 2D logits [batch, vocab] (already sliced, penalties already applied)
// Returns sampled token
std::unique_ptr<MlxArray> fused_sample(
    const MlxArray& logits,
    float temperature,
    int32_t top_k,
    float top_p,
    float min_p
);

// SSM (State Space Model) primitives for Mamba/Jamba/Nemotron-H.
// Cumulative sum along axis
std::unique_ptr<MlxArray> cumsum(const MlxArray& a, int32_t axis, bool reverse, bool inclusive);

// Lower triangular matrix (keeps elements on and below k-th diagonal)
std::unique_ptr<MlxArray> tril(const MlxArray& a, int32_t k);

// Upper triangular matrix (keeps elements on and above k-th diagonal)
std::unique_ptr<MlxArray> triu(const MlxArray& a, int32_t k);

// Clip values to range [a_min, a_max]
std::unique_ptr<MlxArray> clip(const MlxArray& a, const MlxArray& a_min, const MlxArray& a_max);

// log(1 + x) - numerically stable for small x, used for softplus
std::unique_ptr<MlxArray> log1p(const MlxArray& a);

// Softplus activation: log(1 + exp(x))
std::unique_ptr<MlxArray> softplus(const MlxArray& a);

// 1D convolution with groups support (for depthwise conv when groups=channels)
std::unique_ptr<MlxArray> conv1d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups
);
// Fallible counterpart of `conv1d`; declared `-> Result` in the bridge so cxx
// catches MLX's eager shape-mismatch exception (and any other throw) and
// surfaces it as a Rust `Err` instead of aborting.
std::unique_ptr<MlxArray> try_conv1d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups
);

// Swap axes (convenient for SSM attention)
std::unique_ptr<MlxArray> swap_axes(const MlxArray& a, int32_t axis1, int32_t axis2);

// Core ops additions.
// Array creation
std::unique_ptr<MlxArray> identity(int32_t n, int32_t dtype);
std::unique_ptr<MlxArray> tri(int32_t n, int32_t m, int32_t k, int32_t dtype);

// Shape manipulation
std::unique_ptr<MlxArray> unflatten(const MlxArray& a, int32_t axis, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> as_strided(const MlxArray& a, rust::Slice<const int32_t> shape, rust::Slice<const int64_t> strides, size_t offset);
std::unique_ptr<MlxArray> contiguous(const MlxArray& a, bool allow_col_major);
std::unique_ptr<MlxArray> broadcast_arrays_get(rust::Slice<const MlxArray* const> arrays, size_t index);
size_t broadcast_arrays_count(rust::Slice<const MlxArray* const> arrays);

// Arithmetic
std::unique_ptr<MlxArray> floor_divide(const MlxArray& a, const MlxArray& b);

// Comparison & Boolean
std::unique_ptr<MlxArray> array_equal(const MlxArray& a, const MlxArray& b, bool equal_nan);
std::unique_ptr<MlxArray> allclose(const MlxArray& a, const MlxArray& b, double rtol, double atol);
std::unique_ptr<MlxArray> isclose(const MlxArray& a, const MlxArray& b, double rtol, double atol);

// Reductions
std::unique_ptr<MlxArray> median_all(const MlxArray& a);
std::unique_ptr<MlxArray> median_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> logcumsumexp(const MlxArray& a, int32_t axis, bool reverse, bool inclusive);

// Bitwise
std::unique_ptr<MlxArray> bitwise_invert(const MlxArray& a);

// Complex number ops
std::unique_ptr<MlxArray> real_part(const MlxArray& a);
std::unique_ptr<MlxArray> imag_part(const MlxArray& a);
std::unique_ptr<MlxArray> conjugate(const MlxArray& a);

// View/reinterpret
std::unique_ptr<MlxArray> view(const MlxArray& a, int32_t dtype);

// Kronecker product
std::unique_ptr<MlxArray> kron(const MlxArray& a, const MlxArray& b);

// Matrix operations
std::unique_ptr<MlxArray> addmm(const MlxArray& c, const MlxArray& a, const MlxArray& b, float alpha, float beta);
std::unique_ptr<MlxArray> block_masked_mm(
    const MlxArray& a,
    const MlxArray& b,
    int32_t block_size,
    const MlxArray* mask_out,   // nullable
    const MlxArray* mask_lhs,   // nullable
    const MlxArray* mask_rhs    // nullable
);
std::unique_ptr<MlxArray> segmented_mm(const MlxArray& a, const MlxArray& b, const MlxArray& segments);

// Hadamard
std::unique_ptr<MlxArray> hadamard_transform(const MlxArray& a);

// Number of elements
std::unique_ptr<MlxArray> number_of_elements(const MlxArray& a, rust::Slice<const int32_t> axes, bool inverted, int32_t dtype);

// Convolution additions.
std::unique_ptr<MlxArray> conv3d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_d, int32_t stride_h, int32_t stride_w,
    int32_t padding_d, int32_t padding_h, int32_t padding_w,
    int32_t dilation_d, int32_t dilation_h, int32_t dilation_w,
    int32_t groups
);

std::unique_ptr<MlxArray> conv_transpose1d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t output_padding,
    int32_t groups
);

std::unique_ptr<MlxArray> conv_transpose2d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t output_padding_h, int32_t output_padding_w,
    int32_t groups
);

std::unique_ptr<MlxArray> conv_transpose3d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_d, int32_t stride_h, int32_t stride_w,
    int32_t padding_d, int32_t padding_h, int32_t padding_w,
    int32_t dilation_d, int32_t dilation_h, int32_t dilation_w,
    int32_t output_padding_d, int32_t output_padding_h, int32_t output_padding_w,
    int32_t groups
);

// Einsum.
std::unique_ptr<MlxArray> einsum(rust::Str subscripts, rust::Slice<const MlxArray* const> operands);

// Linear algebra (mlx/linalg.h).
std::unique_ptr<MlxArray> linalg_norm(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> linalg_norm_ord(const MlxArray& a, double ord, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> linalg_norm_str(const MlxArray& a, rust::Str ord, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> linalg_qr_q(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_qr_r(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_svd_u(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_svd_s(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_svd_vt(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_inv(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_pinv(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_cholesky(const MlxArray& a, bool upper);
std::unique_ptr<MlxArray> linalg_solve(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> linalg_solve_triangular(const MlxArray& a, const MlxArray& b, bool upper);
std::unique_ptr<MlxArray> linalg_lu_p(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_lu_l(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_lu_u(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_lu_factor_lu(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_lu_factor_pivots(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_eig_values(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_eig_vectors(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_eigvals(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_eigh_values(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_eigh_vectors(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_eigvalsh(const MlxArray& a);
std::unique_ptr<MlxArray> linalg_cross(const MlxArray& a, const MlxArray& b, int32_t axis);
std::unique_ptr<MlxArray> linalg_tri_inv(const MlxArray& a, bool upper);
std::unique_ptr<MlxArray> linalg_cholesky_inv(const MlxArray& a, bool upper);

// FFT (mlx/fft.h).
std::unique_ptr<MlxArray> fft(const MlxArray& a, int32_t n, int32_t axis);
std::unique_ptr<MlxArray> ifft(const MlxArray& a, int32_t n, int32_t axis);
std::unique_ptr<MlxArray> rfft(const MlxArray& a, int32_t n, int32_t axis);
std::unique_ptr<MlxArray> irfft(const MlxArray& a, int32_t n, int32_t axis);
std::unique_ptr<MlxArray> fft2(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes);
std::unique_ptr<MlxArray> ifft2(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes);
std::unique_ptr<MlxArray> rfft2(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes);
std::unique_ptr<MlxArray> irfft2(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes);
std::unique_ptr<MlxArray> fftn_axes(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes);
std::unique_ptr<MlxArray> ifftn_axes(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes);
std::unique_ptr<MlxArray> rfftn_axes(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes);
std::unique_ptr<MlxArray> irfftn_axes(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes);
std::unique_ptr<MlxArray> fftshift(const MlxArray& a, rust::Slice<const int32_t> axes);
std::unique_ptr<MlxArray> ifftshift(const MlxArray& a, rust::Slice<const int32_t> axes);

// Random (mlx/random.h).
std::unique_ptr<MlxArray> random_key(uint64_t seed);
std::unique_ptr<MlxArray> random_split_key(const MlxArray& key, int32_t num);
std::unique_ptr<MlxArray> random_uniform(float low, float high, rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key);
std::unique_ptr<MlxArray> random_normal(rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key);
std::unique_ptr<MlxArray> random_bernoulli_p(float p, rust::Slice<const int32_t> shape, const MlxArray* key);
std::unique_ptr<MlxArray> random_randint(int32_t low, int32_t high, rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key);
std::unique_ptr<MlxArray> random_truncated_normal(float lower, float upper, rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key);
std::unique_ptr<MlxArray> random_gumbel(rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key);
std::unique_ptr<MlxArray> random_laplace(rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key);
std::unique_ptr<MlxArray> random_permutation(int32_t x, const MlxArray* key);
std::unique_ptr<MlxArray> random_permutation_array(const MlxArray& a, int32_t axis, const MlxArray* key);
std::unique_ptr<MlxArray> random_multivariate_normal(
    const MlxArray& mean,
    const MlxArray& cov,
    rust::Slice<const int32_t> shape,
    int32_t dtype,
    const MlxArray* key
);

// Quantization additions.
std::unique_ptr<MlxQuantizedWeights> quantize_weights(const MlxArray& w, int32_t group_size, int32_t bits);
std::unique_ptr<MlxQuantizedWeights> quantize_weights_with_mode(const MlxArray& w, int32_t group_size, int32_t bits, rust::Str mode);
std::unique_ptr<MlxArray> quantized_weights_w(const MlxQuantizedWeights& weights);
std::unique_ptr<MlxArray> quantized_weights_scales(const MlxQuantizedWeights& weights);
bool quantized_weights_has_biases(const MlxQuantizedWeights& weights);
std::unique_ptr<MlxArray> quantized_weights_biases(const MlxQuantizedWeights& weights);
std::unique_ptr<MlxArray> quantize_weights_w(const MlxArray& w, int32_t group_size, int32_t bits);
std::unique_ptr<MlxArray> quantize_weights_scales(const MlxArray& w, int32_t group_size, int32_t bits);
std::unique_ptr<MlxArray> quantize_weights_biases(const MlxArray& w, int32_t group_size, int32_t bits);

// SSM (Mamba2) fused Metal kernel for single-token decode.
// Replaces ~55 individual ops with a single Metal kernel call.
// Used by: NemotronH, NemotronNAS, Mamba2
// Returns (output, next_state) packed as a pair via output pointers.
// output: [batch, 1, num_heads, head_dim]
// next_state: same shape as state_in
void ssm_update_kernel(
    const MlxArray& hidden_states,   // [batch, 1, num_heads, head_dim]
    const MlxArray& A_log,           // [num_heads]
    const MlxArray& B,               // [batch, 1, n_groups, state_dim]
    const MlxArray& C,               // [batch, 1, n_groups, state_dim]
    const MlxArray& D,               // [num_heads]
    const MlxArray& dt,              // [batch, 1, num_heads]
    const MlxArray& dt_bias,         // [num_heads]
    const MlxArray& state_in,        // [batch, n_groups, n_heads/n_groups, head_dim, state_dim]
    float time_step_min,
    float time_step_max,
    std::unique_ptr<MlxArray>& output,
    std::unique_ptr<MlxArray>& next_state
);

// Fused MoE expert kernel for single-token decode (power-of-2 bits, affine).
std::unique_ptr<MlxArray> fused_moe_expert_kernel(
    const MlxArray& x,            // [Din] activation
    const MlxArray& indices,      // [K] selected expert ids
    const MlxArray& gate_w, const MlxArray& gate_s, const MlxArray& gate_b,
    const MlxArray& up_w,   const MlxArray& up_s,   const MlxArray& up_b,
    const MlxArray& down_w, const MlxArray& down_s, const MlxArray& down_b,
    const MlxArray& scores,       // [K] combine weights
    int32_t din, int32_t dff, int32_t k,
    int32_t gu_bits, int32_t d_bits, int32_t group_size
);

// Same as fused_moe_expert_kernel but with GeGLU (gelu tanh approx) instead of
// SwiGLU for the gate/up activation (gemma4 experts).
std::unique_ptr<MlxArray> fused_moe_geglu_kernel(
    const MlxArray& x,
    const MlxArray& indices,
    const MlxArray& gate_w, const MlxArray& gate_s, const MlxArray& gate_b,
    const MlxArray& up_w,   const MlxArray& up_s,   const MlxArray& up_b,
    const MlxArray& down_w, const MlxArray& down_s, const MlxArray& down_b,
    const MlxArray& scores,
    int32_t din, int32_t dff, int32_t k,
    int32_t gu_bits, int32_t d_bits, int32_t group_size
);

// Fused xIELU activation (Apertus). Collapses the ~11 elementwise ops in
// apertus_xielu into one launch over the MLP intermediate buffer. Falls back to
// an equivalent elementwise graph on non-Metal back-ends.
std::unique_ptr<MlxArray> fused_xielu(
    const MlxArray& x,
    float alpha_p,
    float alpha_n,
    float beta,
    float eps
);

// BitLinear ternary matmul (BitNet b1.58): multiply on 2-bit-packed ternary
// weights [out_features/4, in_features] uint8 scaled by weight_scale[0].
std::unique_ptr<MlxArray> bitlinear_matmul(
    const MlxArray& x,
    const MlxArray& packed_weights,
    const MlxArray& weight_scale,
    int32_t in_features,
    int32_t out_features,
    bool invert_weight_scales
);

// Fused MoE forward: gate + switch_mlp + score weighting + optional shared expert
// Combines ~25 FFI calls into a single C++ function
// Used by: NemotronH, NemotronNAS
std::unique_ptr<MlxArray> fused_moe_forward(
    const MlxArray& x,                  // [tokens, hidden]
    const MlxArray& gate_weight,         // [num_experts, hidden]
    const MlxArray& correction_bias,     // [num_experts]
    const MlxArray& fc1_weight,          // [num_experts, intermediate, packed_hidden]
    const MlxArray& fc1_scales,
    const MlxArray& fc1_biases,
    const MlxArray& fc2_weight,          // [num_experts, hidden, packed_intermediate]
    const MlxArray& fc2_scales,
    const MlxArray& fc2_biases,
    const MlxArray* shared_up_weight,    // nullable: [intermediate, hidden]
    const MlxArray* shared_up_scales,
    const MlxArray* shared_up_biases,
    const MlxArray* shared_down_weight,  // nullable: [hidden, intermediate]
    const MlxArray* shared_down_scales,
    const MlxArray* shared_down_biases,
    int32_t top_k,
    float scaling_factor,
    bool norm_topk_prob,
    int32_t group_size,
    int32_t bits
);

// Check if SSM Metal kernel is available (Metal GPU only)
bool ssm_kernel_available();

// Compiled MoE gate: sigmoid scoring + bias + topk + normalize + scale
// Matches Python @mx.compile group_expert_select()
// Returns (topk_indices, topk_scores) via output pointers
void compiled_moe_gate(
    const MlxArray& gates,           // [tokens, num_experts] - already matmul'd
    const MlxArray& correction_bias, // [num_experts]
    int32_t top_k,
    float scaling_factor,
    bool norm_topk_prob,
    std::unique_ptr<MlxArray>& indices_out,
    std::unique_ptr<MlxArray>& scores_out
);

// Fused Mamba2 mixer forward for single-token decode.
// Replaces ~23 FFI calls with a single C++ function call.
// Used by: NemotronH (and any other model with quantized Mamba2Mixer).
// Assumes seq_len == 1 (decode step with existing conv/SSM state).
//
// hidden_states:   [batch, 1, hidden_size]
// conv_state_in:   [batch, conv_kernel_size-1, conv_dim]
// ssm_state_in:    [batch, n_groups, num_heads/n_groups, head_dim, ssm_state_size]
//
// output:          [batch, 1, hidden_size]
// conv_state_out:  [batch, conv_kernel_size-1, conv_dim]
// ssm_state_out:   same shape as ssm_state_in
//
// Fused gated-delta single-token decode step.
// Combines: decay → kv_mem → delta → state_update → output into one call.
// Used by: Qwen3.5, Qwen3Next, KimiLinear (GatedDeltaNet T=1 decode)
void fused_gated_delta_decode_step(
    const MlxArray& q,       // [B, H, D]
    const MlxArray& k,       // [B, H, D]
    const MlxArray& v,       // [B, H, Dv]
    const MlxArray& g,       // [B, H] or [B, H, Dk]
    const MlxArray& beta,    // [B, H]
    const MlxArray& state,   // [B, H, Dv, Dk]
    int32_t q_dtype,
    std::unique_ptr<MlxArray>& output,
    std::unique_ptr<MlxArray>& new_state_out
);

// Check if GatedDeltaNet Metal kernel is available (Metal GPU only)
bool gated_delta_kernel_available();

// True when the MLX Metal backend is available at runtime (macOS Apple
// Silicon); false on CUDA-only and CPU-only builds. Mirrors the
// metal::is_available() gate used to pick the Metal vs CUDA kernel port. #626.
bool metal_is_available();

// True when the MLX CUDA backend is available at runtime; false on Metal-only
// and CPU-only builds. Backend-agnostic (`no_cuda` stub off CUDA). Used by the
// paged-attention decode gate to allow the fused native kernel on CUDA. #634.
bool cuda_is_available();

// Start/stop Metal GPU capture. Requires the process to run under
// `MTL_CAPTURE_ENABLED=1`; otherwise Metal drops the capture silently.
void metal_start_capture(rust::Str path);
void metal_stop_capture();

// GatedDeltaNet custom Metal kernel forward.
// Handles both T=1 (decode) and T>1 (prefill) in a single GPU dispatch.
// Replaces ops-based gated_delta_step with a fused Metal shader using SIMD reductions.
// Used by: Qwen3.5, Qwen3Next, KimiLinear
void metal_gated_delta_forward(
    const MlxArray& q,       // [B, T, Hk, Dk]
    const MlxArray& k,       // [B, T, Hk, Dk]
    const MlxArray& v,       // [B, T, Hv, Dv]
    const MlxArray& g,       // [B, T, Hv] or [B, T, Hv, Dk]
    const MlxArray& beta,    // [B, T, Hv]
    const MlxArray& state,   // [B, Hv, Dv, Dk]
    const MlxArray* mask,    // nullable: [B, T]
    std::unique_ptr<MlxArray>& output,      // [B, T, Hv, Dv]
    std::unique_ptr<MlxArray>& new_state    // [B, Hv, Dv, Dk]
);

// Quantization mode is "affine" (standard mlx-community models).
void fused_mamba2_forward(
    // Input
    const MlxArray& hidden_states,
    // in_proj weights (quantized, affine mode)
    const MlxArray& in_proj_weight,
    const MlxArray& in_proj_scales,
    const MlxArray* in_proj_biases,      // nullable for no-bias quantization
    // conv1d weights
    const MlxArray& conv_weight,
    const MlxArray* conv_bias,           // nullable
    // SSM parameters
    const MlxArray& A_log,               // [num_heads]
    const MlxArray& D,                   // [num_heads]
    const MlxArray& dt_bias,             // [num_heads]
    // Norm weight for MambaRMSNormGated
    const MlxArray& norm_weight,         // [intermediate_size]
    // out_proj weights (quantized, affine mode)
    const MlxArray& out_proj_weight,
    const MlxArray& out_proj_scales,
    const MlxArray* out_proj_biases,     // nullable for no-bias quantization
    // Cache state inputs
    const MlxArray& conv_state_in,       // [batch, conv_kernel_size-1, conv_dim]
    const MlxArray& ssm_state_in,        // [batch, n_groups, heads/groups, head_dim, state_dim]
    // Mamba2 config
    int32_t intermediate_size,
    int32_t conv_dim,
    int32_t conv_kernel_size,
    int32_t num_heads,
    int32_t head_dim,
    int32_t n_groups,
    int32_t ssm_state_size,
    float time_step_min,
    float time_step_max,
    float norm_eps,
    // Quantization config
    int32_t group_size,
    int32_t bits,
    // Outputs
    std::unique_ptr<MlxArray>& output,
    std::unique_ptr<MlxArray>& conv_state_out,
    std::unique_ptr<MlxArray>& ssm_state_out
);

// NemotronH opaque model handle for C++-side full forward decode.
// Weights are registered once, then nemotron_decode_step builds the full
// computation graph within C++ (zero Rust↔C++ round-trips during layer loop).
//
// Register: pass all weight pointers as flat arrays per block type.
// Returns an opaque handle (uint64_t) used for subsequent decode calls.
uint64_t nemotron_register_model(
    // Embedding (quantized)
    const MlxArray& embed_w, const MlxArray& embed_s, const MlxArray& embed_b,
    // Final norm + lm_head
    const MlxArray& final_norm_w,
    const MlxArray& lm_head_w, const MlxArray& lm_head_s, const MlxArray* lm_head_b,
    // Per-layer norm weights
    rust::Slice<const MlxArray* const> norm_weights,
    // Block types per layer: 0=Mamba, 1=Attention, 3=MoE
    rust::Slice<const int32_t> block_types,
    // Mamba layer weights (indexed by mamba_layer_idx, not global)
    rust::Slice<const MlxArray* const> mamba_weights,   // 13 ptrs per layer: in(w,s,b), conv(w,b), a_log, d, dt_bias, norm, out(w,s,b)
    // MoE layer weights (indexed by moe_layer_idx)
    rust::Slice<const MlxArray* const> moe_weights,     // 14 ptrs per layer: gate_w, bias, fc1(w,s,b), fc2(w,s,b), shared_up(w,s,b), shared_down(w,s,b)
    // Attention layer weights (indexed by attn_layer_idx)
    rust::Slice<const MlxArray* const> attn_weights,    // 12 ptrs per layer: q(w,s,b), k(w,s,b), v(w,s,b), o(w,s,b)
    // Config
    float norm_eps, int32_t group_size, int32_t bits,
    // Mamba config
    int32_t m_inter, int32_t m_conv_dim, int32_t m_conv_k,
    int32_t m_heads, int32_t m_head_dim, int32_t m_groups, int32_t m_state_size,
    float m_ts_min, float m_ts_max, float m_norm_eps,
    // MoE config
    int32_t moe_top_k, float moe_scale, bool moe_norm,
    // Attention config
    int32_t a_heads, int32_t a_kv_heads, int32_t a_head_dim,
    float a_rope_theta, float a_scale
);

// Single decode step using registered model handle.
// Builds full graph in C++ — one FFI call per token.
void nemotron_decode_step(
    uint64_t handle,
    const MlxArray& input_ids,
    // Mamba cache states (in): conv[num_mamba], ssm[num_mamba]
    rust::Slice<const MlxArray* const> mamba_conv_in,
    rust::Slice<const MlxArray* const> mamba_ssm_in,
    // Attention KV cache pointers (managed by Rust KVCache, just pass current state)
    rust::Slice<const MlxArray* const> attn_kv_keys,
    rust::Slice<const MlxArray* const> attn_kv_values,
    rust::Slice<const int32_t> attn_kv_offsets,
    // Outputs
    std::unique_ptr<MlxArray>& logits,
    // Updated Mamba cache states
    rust::Slice<std::unique_ptr<MlxArray>> mamba_conv_out,
    rust::Slice<std::unique_ptr<MlxArray>> mamba_ssm_out
);

// Free registered model
void nemotron_free_model(uint64_t handle);

#if 0
void nemotron_full_forward(
    // Input
    const MlxArray& input_ids,           // [batch, 1]
    // Embedding (quantized)
    const MlxArray& embed_weight, const MlxArray& embed_scales, const MlxArray& embed_biases,
    // Per-layer weights: arrays of pointers indexed by layer
    // RMSNorm weight per layer
    rust::Slice<const MlxArray* const> norm_weights,    // [num_layers]
    float norm_eps,
    // Block type per layer
    rust::Slice<const int32_t> block_type_codes,        // [num_layers]
    // Final norm + lm_head
    const MlxArray& final_norm_weight,
    const MlxArray& lm_head_weight, const MlxArray& lm_head_scales, const MlxArray* lm_head_biases,
    // Per-layer Mamba weights (indexed by mamba_layer_index, not global index)
    rust::Slice<const MlxArray* const> mamba_in_proj_w,
    rust::Slice<const MlxArray* const> mamba_in_proj_s,
    rust::Slice<const MlxArray* const> mamba_in_proj_b,
    rust::Slice<const MlxArray* const> mamba_conv_w,
    rust::Slice<const MlxArray* const> mamba_conv_b,
    rust::Slice<const MlxArray* const> mamba_a_log,
    rust::Slice<const MlxArray* const> mamba_d,
    rust::Slice<const MlxArray* const> mamba_dt_bias,
    rust::Slice<const MlxArray* const> mamba_norm_w,
    rust::Slice<const MlxArray* const> mamba_out_proj_w,
    rust::Slice<const MlxArray* const> mamba_out_proj_s,
    rust::Slice<const MlxArray* const> mamba_out_proj_b,
    // Mamba cache states (in/out)
    rust::Slice<const MlxArray* const> mamba_conv_state_in,
    rust::Slice<const MlxArray* const> mamba_ssm_state_in,
    // Mamba config
    int32_t mamba_intermediate_size, int32_t mamba_conv_dim, int32_t mamba_conv_kernel_size,
    int32_t mamba_num_heads, int32_t mamba_head_dim, int32_t mamba_n_groups, int32_t mamba_ssm_state_size,
    float mamba_ts_min, float mamba_ts_max, float mamba_norm_eps,
    // Per-layer MoE weights (indexed by moe_layer_index)
    rust::Slice<const MlxArray* const> moe_gate_w,
    rust::Slice<const MlxArray* const> moe_correction_bias,
    rust::Slice<const MlxArray* const> moe_fc1_w,
    rust::Slice<const MlxArray* const> moe_fc1_s,
    rust::Slice<const MlxArray* const> moe_fc1_b,
    rust::Slice<const MlxArray* const> moe_fc2_w,
    rust::Slice<const MlxArray* const> moe_fc2_s,
    rust::Slice<const MlxArray* const> moe_fc2_b,
    rust::Slice<const MlxArray* const> moe_shared_up_w,
    rust::Slice<const MlxArray* const> moe_shared_up_s,
    rust::Slice<const MlxArray* const> moe_shared_up_b,
    rust::Slice<const MlxArray* const> moe_shared_down_w,
    rust::Slice<const MlxArray* const> moe_shared_down_s,
    rust::Slice<const MlxArray* const> moe_shared_down_b,
    int32_t moe_top_k, float moe_scaling_factor, bool moe_norm_topk_prob,
    // Per-layer Attention weights (indexed by attn_layer_index)
    rust::Slice<const MlxArray* const> attn_q_w,
    rust::Slice<const MlxArray* const> attn_q_s,
    rust::Slice<const MlxArray* const> attn_q_b,
    rust::Slice<const MlxArray* const> attn_k_w,
    rust::Slice<const MlxArray* const> attn_k_s,
    rust::Slice<const MlxArray* const> attn_k_b,
    rust::Slice<const MlxArray* const> attn_v_w,
    rust::Slice<const MlxArray* const> attn_v_s,
    rust::Slice<const MlxArray* const> attn_v_b,
    rust::Slice<const MlxArray* const> attn_o_w,
    rust::Slice<const MlxArray* const> attn_o_s,
    rust::Slice<const MlxArray* const> attn_o_b,
    int32_t attn_num_heads, int32_t attn_num_kv_heads, int32_t attn_head_dim,
    float attn_rope_theta, float attn_scale,
    // Quantization config
    int32_t group_size, int32_t bits,
    // KV cache (updated in-place via slice_update, managed externally)
    // We pass cache offsets and let the function update internally
    // ... (KV cache management is complex, simplified for now)
    // Output
    std::unique_ptr<MlxArray>& logits,
    // Updated cache states
    rust::Slice<std::unique_ptr<MlxArray>> mamba_conv_state_out,
    rust::Slice<std::unique_ptr<MlxArray>> mamba_ssm_state_out
);
#endif

// Metal 4 attention dispatch.
//
// With upstream MLX main, `fast::scaled_dot_product_attention()` already
// selects the M5 NAX-backed SDPA implementation when the hardware and shape
// constraints match. This bridge preserves the Rust-side `softcap` and
// `window_size` plumbing while delegating the actual kernel body to MLX.
//
// Supported attention patterns (via MLX SDPA fallback, and future kernel):
//   - Standard MHA: n_heads == n_kv_heads
//   - GQA: n_heads > n_kv_heads (e.g., Llama 3.1: 32 Q / 8 KV)
//   - MQA: n_kv_heads == 1
//
// Mask handling: boolean/integer masks are passed through unchanged. Float
// masks are cast to Q's dtype. See fast_scaled_dot_product_attention() for
// the reference implementation.
//
// Use `mlxcel_core::layers::metal4_attention()` from Rust, which queries
// `hardware::get_hardware()` to set `use_metal4` automatically.
//
// See docs/metal4-fused-attention-research.md for design notes.
std::unique_ptr<MlxArray> fused_metal4_attention(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    const MlxArray* mask,  // nullable; supports boolean, integer, and float masks
    float softcap,
    int32_t window_size,
    bool use_metal4
);

// Fused Sparse-V SDPA Metal kernel launcher.
// Wraps `mlxcel::turbo::sparse_v_weighted_sum` so the cxx bridge can expose it
// via the `turbo_sparse_v_weighted_sum` FFI symbol. Implementation lives in
// `src/lib/mlx-cpp/turbo/sparse_v_sdpa.cpp`.
//
// `v_rescale` (4th argument) carries the precomputed per-token rescale
// `norm[t] / |y_hat[t]|` introduced. The previous kernel
// computed this scalar per token via a threadgroup tree reduction; the
// precompute moves that work to quantize time and removes the per-token
// threadgroup barrier chain that dominated decode latency on M5 Max.
std::unique_ptr<MlxArray> turbo_sparse_v_weighted_sum(
    const MlxArray& attn_weights,
    const MlxArray& v_packed,
    const MlxArray& v_rescale,
    const MlxArray& codebook,
    int32_t dim,
    int32_t n_rep,
    float threshold);

// Fused Turbo4Delegated cold-V weighted-sum kernel launcher.
// Wraps `mlxcel::turbo::turbo4_delegated_cold_weighted_sum` so the cxx bridge
// can expose it via the `turbo4_delegated_cold_weighted_sum` FFI symbol.
// Implementation lives in `src/lib/mlx-cpp/turbo/turbo4_delegated_sdpa.cpp`.
//
// The kernel returns the unrotated weighted sum of the cold V tokens. The
// host caller applies the inverse Turbo4 rotation to that result and adds
// the hot-V matmul contribution to produce the final FP16 SDPA output. The
// dequantised cold V never materialises in global memory — that property is
// the reason this kernel exists (vs. the earlier path which built an FP16 `cold_v_dequant` memo plus a per-step concat with hot V).
std::unique_ptr<MlxArray> turbo4_delegated_cold_weighted_sum(
    const MlxArray& attn_weights_cold,
    const MlxArray& v_packed_cold,
    const MlxArray& v_rescale_cold,
    const MlxArray& codebook,
    int32_t dim,
    int32_t n_rep,
    float threshold);

// Bulk rotated dequant launcher for the Swift-LM-style dequant-first SDPA
// route. One Metal dispatch converts `[B,H,T,D/2]` packed Turbo4 V plus
// `[B,H,T,1]` rescale into `[B,H,T,D]` FP16 in rotated codec space.
std::unique_ptr<MlxArray> turbo4_delegated_bulk_dequant_rotated(
    const MlxArray& v_packed,
    const MlxArray& v_rescale,
    const MlxArray& codebook,
    int32_t dim);

// Steel-attention-envelope fused Turbo4Delegated SDPA launcher.
// Wraps `mlxcel::turbo::turbo4_delegated_steel_sdpa`. Returns a pair of MLX
// arrays (`out_cold_pre`, `out_hot`) that the host sums after applying the
// linear inverse Turbo4 rotation to the cold output. The pair is exposed
// through cxx as a struct so the FFI surface stays inside the cxx-supported
// type set (cxx does not directly model multiple return values).
//
// `cold_packed` and `cold_rescale` may be 1-token zero placeholders when
// `cold_offset == 0` (MLX's `metal_kernel` rejects buffers with any zero-shape
// axis). Same for `hot_v` when `hot_offset == 0`. The host launcher in
// `mlxcel-core/src/cache/turbo/sparse_v.rs::attention_turbo4_delegated_steel`
// substitutes these placeholders and the kernel takes the empty-range
// early-out via the explicit `cold_offset` / `hot_offset` scalar inputs.
struct Turbo4DelegatedSteelOutputs {
    std::unique_ptr<MlxArray> out_cold_pre;  // [B*Hq, Tq, D] f32, unrotated
    std::unique_ptr<MlxArray> out_hot;       // [B*Hq, Tq, D] f32, hot weighted sum
};

std::unique_ptr<Turbo4DelegatedSteelOutputs> turbo4_delegated_steel_sdpa(
    const MlxArray& scores,         // [B*Hq, Tq, T_total]   f32
    const MlxArray& cold_packed,    // [B*Hkv, T_cold, D/2]  u8
    const MlxArray& cold_rescale,   // [B*Hkv, T_cold]       f16
    const MlxArray& hot_v,          // [B*Hkv, T_hot, D]     f16
    const MlxArray& codebook,       // [16]                  f32
    int32_t dim,
    int32_t n_rep,
    int32_t cold_offset,
    int32_t hot_offset,
    float threshold);

// Move out the cold/hot tensors from a steel-SDPA outputs struct. The cxx
// bridge does not directly model "destructure a struct returned by FFI", so we
// expose two takers that the Rust side calls before dropping the struct.
// After both are taken, the struct is empty.
std::unique_ptr<MlxArray> steel_outputs_take_cold(Turbo4DelegatedSteelOutputs& o);
std::unique_ptr<MlxArray> steel_outputs_take_hot(Turbo4DelegatedSteelOutputs& o);

// Fused paged-attention decode kernel launcher (epic #116 Phase 6, #123).
// Wraps `mlxcel::turbo::paged_attention_decode`. Reads scattered KV blocks out
// of the global pool via the block table with no separate gather copy. `q` is
// `[B, Hq, 1, D]` f32; `k_pool` / `v_pool` are `[num_blocks, block_size, Hkv,
// D]` f16; `rows` / `row_offsets` / `logical_starts` / `visible_lens` are i32
// block-table metadata. Returns `[B, Hq, 1, D]` f32.
std::unique_ptr<MlxArray> paged_attention_decode(
    const MlxArray& q,
    const MlxArray& k_pool,
    const MlxArray& v_pool,
    const MlxArray& rows,
    const MlxArray& row_offsets,
    const MlxArray& logical_starts,
    const MlxArray& visible_lens,
    float scale);

// Opaque holder for weights loaded via MLX's native load_safetensors().
// Arrays are lazy — MLX manages the mmap internally, no eager copy needed.
struct MlxLoadedWeights {
    std::vector<std::string> names;
    std::vector<std::unique_ptr<MlxArray>> arrays;
};

// Load safetensors file using MLX's native loader (lazy arrays, MLX-managed mmap)
std::unique_ptr<MlxLoadedWeights> mlx_load_safetensors(rust::Str path);

// Access loaded weights
size_t loaded_weights_len(const MlxLoadedWeights& w);
rust::String loaded_weights_name(const MlxLoadedWeights& w, size_t index);
std::unique_ptr<MlxArray> loaded_weights_take(MlxLoadedWeights& w, size_t index);

} // namespace mlx_cxx
