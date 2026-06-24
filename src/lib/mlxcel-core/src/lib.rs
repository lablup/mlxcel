// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Direct C++ bindings to MLX via cxx
//!
//! This crate provides direct bindings to MLX C++ API, bypassing the mlx-c wrapper
//! for improved performance.

#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "mlx_cxx")]
mod ffi {
    // Opaque types - these are defined in C++ and we just hold pointers to them
    unsafe extern "C++" {
        include!("mlx_cxx_bridge.h");

        /// Opaque wrapper for mlx::core::array
        type MlxArray;

        /// Opaque wrapper for mlx::core::Stream
        type MlxStream;

        /// Opaque wrapper for mlx::core::ThreadLocalStream.
        ///
        /// Holds a TLS handle that resolves to a per-thread `MlxStream`
        /// on demand. Used by the generation stream owners so dispatch and synchronization stay on the same
        /// per-thread stream.
        type MlxThreadLocalStream;

        // Stream functions.
        /// Get the default stream
        fn default_stream() -> UniquePtr<MlxStream>;

        /// Create a new stream on the specified device
        fn new_stream_on_device(gpu: bool) -> UniquePtr<MlxStream>;

        /// Synchronize a stream
        fn synchronize_stream(stream: &MlxStream);

        /// Create a new thread-local stream bound to the GPU device.
        ///
        /// The returned handle is safe to share across threads — each
        /// thread gets its own `MlxStream` on first resolution.
        fn new_thread_local_stream_gpu() -> UniquePtr<MlxThreadLocalStream>;

        /// Resolve the calling thread's `MlxStream` from a TLS handle.
        fn stream_from_thread_local_stream(tls: &MlxThreadLocalStream) -> UniquePtr<MlxStream>;

        /// Synchronize the calling thread's stream for this TLS handle.
        fn synchronize_thread_local_stream(tls: &MlxThreadLocalStream);

        // Array factory functions.
        /// Create array filled with zeros
        fn zeros(shape: &[i32], dtype: i32) -> UniquePtr<MlxArray>;

        /// Create array filled with zeros on specific stream
        fn zeros_stream(shape: &[i32], dtype: i32, stream: &MlxStream) -> UniquePtr<MlxArray>;

        /// Create array filled with ones
        fn ones(shape: &[i32], dtype: i32) -> UniquePtr<MlxArray>;

        /// Create array filled with ones on specific stream
        fn ones_stream(shape: &[i32], dtype: i32, stream: &MlxStream) -> UniquePtr<MlxArray>;

        /// Create array with specific f32 value
        fn full_f32(shape: &[i32], value: f32, dtype: i32) -> UniquePtr<MlxArray>;

        /// Create identity/eye matrix (n rows, m cols, k diagonal offset)
        fn eye(n: i32, m: i32, k: i32, dtype: i32) -> UniquePtr<MlxArray>;

        /// Create linearly spaced values
        fn linspace(start: f32, stop: f32, num: i32, dtype: i32) -> UniquePtr<MlxArray>;

        /// Create zeros with same shape as input
        fn zeros_like(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Create ones with same shape as input
        fn ones_like(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Create array filled with value, same shape as input
        fn full_like(a: &MlxArray, value: f32) -> UniquePtr<MlxArray>;

        /// Create array from f32 slice
        fn from_slice_f32(data: &[f32], shape: &[i32]) -> UniquePtr<MlxArray>;

        /// Create array from i32 slice
        fn from_slice_i32(data: &[i32], shape: &[i32]) -> UniquePtr<MlxArray>;

        /// Create array from u32 slice (for quantized weights)
        fn from_slice_u32(data: &[u32], shape: &[i32]) -> UniquePtr<MlxArray>;

        /// Create array from i64 slice
        fn from_slice_i64(data: &[i64], shape: &[i32]) -> UniquePtr<MlxArray>;

        /// Create array from raw bytes with specified dtype
        fn from_bytes(data: &[u8], shape: &[i32], dtype: i32) -> UniquePtr<MlxArray>;

        /// Create array from raw bytes without copying.
        ///
        /// The caller must keep the backing buffer alive for the full lifetime
        /// of the returned array.
        fn from_bytes_nocopy(data: &[u8], shape: &[i32], dtype: i32) -> UniquePtr<MlxArray>;

        /// Create half-precision array from raw bytes
        fn from_bytes_f16(data: &[u8], shape: &[i32], bfloat16: bool) -> UniquePtr<MlxArray>;

        // Array property accessors.
        /// Get the shape of an array
        fn array_shape(arr: &MlxArray) -> Vec<i32>;

        /// Get the dtype of an array (as integer code)
        fn array_dtype(arr: &MlxArray) -> i32;

        /// Get the number of elements in an array
        fn array_size(arr: &MlxArray) -> usize;

        /// Get the number of dimensions of an array
        fn array_ndim(arr: &MlxArray) -> usize;

        /// Get the size of each element in bytes
        fn array_itemsize(arr: &MlxArray) -> usize;

        /// Get the total size in bytes
        fn array_nbytes(arr: &MlxArray) -> usize;

        // Scalar extraction.
        /// Extract f32 scalar value
        fn item_f32(arr: &MlxArray) -> f32;

        /// Extract i32 scalar value
        fn item_i32(arr: &MlxArray) -> i32;

        /// Extract i64 scalar value
        fn item_i64(arr: &MlxArray) -> i64;

        /// Extract bool scalar value
        fn item_bool(arr: &MlxArray) -> bool;

        /// Copy evaluated array data to a raw byte buffer.
        /// Used by: KV cache serialization for disaggregated inference
        fn array_to_raw_bytes(arr: &MlxArray) -> Vec<u8>;

        /// Fallible counterpart of [`array_to_raw_bytes`].
        ///
        /// Makes the array contiguous, evaluates it, and copies the bytes out,
        /// all inside the cxx try/catch boundary (declared `-> Result`), so an
        /// MLX C++ exception during the contiguous copy or the eval (for example
        /// an allocation failure on a large data-dependent tensor) surfaces as a
        /// Rust `Err` instead of an uncaught exception that aborts the process.
        /// Used by the audio synthesis readback so a fault on a pool/worker
        /// thread becomes a structured per-request error.
        fn try_array_to_raw_bytes(arr: &MlxArray) -> Result<Vec<u8>>;

        // Evaluation.
        /// Evaluate an array
        fn eval(arr: &MlxArray);

        /// Evaluate an array, returning an error instead of throwing.
        ///
        /// Same effect as [`eval`], but any MLX C++ exception (missing stream,
        /// shape mismatch, allocation failure) is caught at the FFI boundary and
        /// surfaced as a Rust `Err` rather than an uncaught exception that
        /// aborts the process. Off-worker callers that evaluate a graph on a
        /// pool thread (the audio providers) use this so a failure becomes a
        /// structured error.
        fn try_eval(arr: &MlxArray) -> Result<()>;

        /// Evaluate multiple arrays at once
        unsafe fn eval_all(arrays: &[*const MlxArray]);

        // Element-wise binary operations.
        /// Element-wise addition
        fn add(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise subtraction
        fn subtract(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise multiplication
        fn multiply(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise division
        fn divide(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise remainder (modulo)
        fn remainder(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise maximum
        fn maximum(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise minimum
        fn minimum(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        // Element-wise unary operations.
        /// Element-wise negation
        fn negative(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise absolute value
        fn abs(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise exponential
        fn exp(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise natural logarithm
        fn log(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise square root
        fn sqrt(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise reciprocal square root
        fn rsqrt(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise square
        fn square(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise sine
        fn sin(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise cosine
        fn cos(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise hyperbolic tangent
        fn tanh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise sigmoid
        fn sigmoid(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise floor
        fn floor(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise ceiling
        fn ceil(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise round
        fn round(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise sign
        fn sign(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise reciprocal (1/x)
        fn reciprocal(a: &MlxArray) -> UniquePtr<MlxArray>;

        // Trigonometric functions
        /// Element-wise tangent
        fn tan(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise hyperbolic sine
        fn sinh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise hyperbolic cosine
        fn cosh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise arc sine
        fn arcsin(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise arc cosine
        fn arccos(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise arc tangent
        fn arctan(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise arc tangent of a/b
        fn arctan2(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise hyperbolic arc sine
        fn arcsinh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise hyperbolic arc cosine
        fn arccosh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise hyperbolic arc tangent
        fn arctanh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Convert radians to degrees
        fn degrees(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Convert degrees to radians
        fn radians(a: &MlxArray) -> UniquePtr<MlxArray>;

        // Mathematical/Special functions
        /// Error function
        fn erf(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Inverse error function
        fn erfinv(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// exp(x) - 1, numerically stable for small x
        fn expm1(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Base-2 logarithm
        fn log2(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Base-10 logarithm
        fn log10(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// log(exp(a) + exp(b)), numerically stable
        fn logaddexp(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise power a^b
        fn power(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        // Checks
        /// Check if elements are NaN
        fn isnan(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Check if elements are infinite
        fn isinf(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Check if elements are finite
        fn isfinite(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Check if elements are negative infinity
        fn isneginf(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Check if elements are positive infinity
        fn isposinf(a: &MlxArray) -> UniquePtr<MlxArray>;

        // Reduction operations.
        /// Sum all elements
        fn sum_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Sum along axis
        fn sum_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Mean of all elements
        fn mean_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Mean along axis
        fn mean_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Max of all elements
        fn max_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Max along axis
        fn max_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Min of all elements
        fn min_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Min along axis
        fn min_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Product of all elements
        fn prod_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Product along axis
        fn prod_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Variance of all elements
        fn var_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Variance along axis (ddof = degrees of freedom correction)
        fn var_axis(a: &MlxArray, axis: i32, keepdims: bool, ddof: i32) -> UniquePtr<MlxArray>;

        /// Standard deviation of all elements
        fn std_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Standard deviation along axis
        fn std_axis(a: &MlxArray, axis: i32, keepdims: bool, ddof: i32) -> UniquePtr<MlxArray>;

        /// Logsumexp of all elements
        fn logsumexp_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Logsumexp along axis
        fn logsumexp_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// All elements are true
        fn all_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Any element is true
        fn any_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        // Matrix operations.
        /// Matrix multiplication
        fn matmul(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Fallible matrix multiplication.
        ///
        /// Same effect as [`matmul`], but declared `-> Result` so cxx catches
        /// any MLX C++ exception at the FFI boundary and returns it as a Rust
        /// `Err` instead of letting it cross uncaught and abort the process.
        /// MLX validates matmul shapes eagerly at graph-build time, so a shape
        /// mismatch throws at construction (not only at eval); this variant
        /// catches that. Used on the audio synthesis forward path for the
        /// data-dependent alignment-expansion matmuls, whose inner dimension is
        /// derived from the runtime duration prediction.
        fn try_matmul(a: &MlxArray, b: &MlxArray) -> Result<UniquePtr<MlxArray>>;

        /// Transpose (swap last two dimensions)
        fn transpose(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Transpose with specified axes
        fn transpose_axes(a: &MlxArray, axes: &[i32]) -> UniquePtr<MlxArray>;

        /// Reshape array
        fn reshape(a: &MlxArray, shape: &[i32]) -> UniquePtr<MlxArray>;

        // Shape operations.
        /// Expand dimensions at axis
        fn expand_dims(a: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Expand dimensions at multiple axes in a single call. Equivalent to
        /// `mx.expand_dims(a, tuple(axes))`. Cheaper than calling
        /// `expand_dims(a, axis)` back-to-back because there is only one FFI
        /// boundary crossing and one MLX graph node added.
        fn expand_dims_multi(a: &MlxArray, axes: &[i32]) -> UniquePtr<MlxArray>;

        /// Remove all singleton dimensions
        fn squeeze(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Remove singleton dimension at axis
        fn squeeze_axis(a: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Broadcast to shape
        fn broadcast_to(a: &MlxArray, shape: &[i32]) -> UniquePtr<MlxArray>;

        /// Flatten array to 1D
        fn flatten(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Flatten array between start_axis and end_axis
        fn flatten_range(a: &MlxArray, start_axis: i32, end_axis: i32) -> UniquePtr<MlxArray>;

        /// Move axis from source to destination
        fn moveaxis(a: &MlxArray, source: i32, destination: i32) -> UniquePtr<MlxArray>;

        /// Pad array (pad_width is pairs of [before, after] for each dim)
        fn pad(a: &MlxArray, pad_width: &[i32], pad_value: f32) -> UniquePtr<MlxArray>;

        /// Extract or construct diagonal
        fn diag(a: &MlxArray, k: i32) -> UniquePtr<MlxArray>;

        /// Extract diagonal from matrix
        fn diagonal(a: &MlxArray, offset: i32, axis1: i32, axis2: i32) -> UniquePtr<MlxArray>;

        // Type conversion.
        /// Convert array to specified dtype
        fn astype(a: &MlxArray, dtype: i32) -> UniquePtr<MlxArray>;

        // Copy.
        /// Copy array
        fn copy(a: &MlxArray) -> UniquePtr<MlxArray>;

        // High-level operations for LLM inference.
        /// Softmax along axis
        fn softmax(a: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Softmax along axis with precise=true (f32 accumulation for f16 inputs)
        fn softmax_precise(a: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Log-softmax along axis (numerically stable)
        fn log_softmax(a: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// RMS normalization
        fn rms_norm(x: &MlxArray, weight: &MlxArray, eps: f32) -> UniquePtr<MlxArray>;

        /// Layer normalization
        fn layer_norm(
            x: &MlxArray,
            weight: &MlxArray,
            bias: &MlxArray,
            eps: f32,
        ) -> UniquePtr<MlxArray>;

        /// Slice array with start, stop
        fn slice(a: &MlxArray, starts: &[i32], stops: &[i32]) -> UniquePtr<MlxArray>;

        /// Slice update: src[starts:stops] = update
        fn slice_update(
            src: &MlxArray,
            update: &MlxArray,
            starts: &[i32],
            stops: &[i32],
        ) -> UniquePtr<MlxArray>;

        /// Argmax along axis
        fn argmax(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Where (conditional select)
        fn where_cond(condition: &MlxArray, x: &MlxArray, y: &MlxArray) -> UniquePtr<MlxArray>;

        /// Greater than comparison
        fn greater(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Less than comparison
        fn less(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Equal comparison
        fn equal(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Seed the global MLX random number generator
        fn random_seed(seed: u64);

        /// Random categorical sampling
        fn random_categorical(logits: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        // Transformer-specific high-level operations.
        /// Apply rotary embedding to input
        fn apply_rope(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray>;

        /// RoPE forward (compute cos, sin for position embedding)
        fn rope_forward(
            x: &MlxArray,
            head_dim: i32,
            theta: f32,
            offset: i32,
            traditional: bool,
        ) -> UniquePtr<MlxArray>;

        /// Scaled dot-product attention
        unsafe fn scaled_dot_product_attention(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            scale: f32,
            mask: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Linear layer forward (with optional bias)
        unsafe fn linear_forward(
            x: &MlxArray,
            weight: &MlxArray,
            bias: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Quantized linear layer forward
        /// biases: nullable for mxfp4/nvfp4/mxfp8 modes
        unsafe fn quantized_linear_forward(
            x: &MlxArray,
            weight: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            linear_bias: *const MlxArray,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        /// SwiGLU MLP forward
        fn swiglu_mlp_forward(
            x: &MlxArray,
            gate_proj: &MlxArray,
            up_proj: &MlxArray,
            down_proj: &MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Compiled SwiGLU activation with kernel fusion
        /// Uses mlx::core::compile(shapeless=true) like Python's @mx.compile
        /// output = silu(gate) * x
        fn compiled_swiglu_activation(gate: &MlxArray, x: &MlxArray) -> UniquePtr<MlxArray>;

        /// Compiled GptOss SwiGLU activation with kernel fusion
        /// Matches mlx-lm gpt_oss.swiglu: clipped gate/up + sigmoid(1.702*gate).
        /// Used by: GptOss
        fn compiled_gpt_oss_swiglu_activation(
            x_linear: &MlxArray,
            x_glu: &MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Compiled relu_squared: square(maximum(x, 0)) — single fused kernel
        fn compiled_relu_squared(x: &MlxArray) -> UniquePtr<MlxArray>;

        /// Compiled silu: x * sigmoid(x) — single fused kernel
        fn compiled_silu(x: &MlxArray) -> UniquePtr<MlxArray>;

        /// Compiled gelu: x * 0.5 * (1 + erf(x / sqrt(2))) — single fused kernel
        /// Used by: StarCoder2 and other precise GELU-based models
        fn compiled_gelu(x: &MlxArray) -> UniquePtr<MlxArray>;

        /// Compiled gelu_approx: erf-based GELU (x * 0.5 * (1 + erf(x / sqrt(2))))
        /// Uses erf instead of tanh for numerical stability with bf16 inputs.
        /// Used by: legacy/tests
        fn compiled_gelu_approx(x: &MlxArray) -> UniquePtr<MlxArray>;

        /// Compiled GeGLU activation: gelu(gate) * x — single fused kernel
        /// Used by: legacy/tests for precise GeGLU
        fn compiled_geglu_activation(gate: &MlxArray, x: &MlxArray) -> UniquePtr<MlxArray>;

        /// Compiled GeGLU activation using Python MLX tanh-approx GELU.
        /// Used by: Gemma2, Gemma3, Gemma4 MLP and SwitchGeGLU layers
        fn compiled_geglu_approx_activation(gate: &MlxArray, x: &MlxArray) -> UniquePtr<MlxArray>;

        /// Compiled gelu_topk: sparse GELU with dynamic threshold — single fused kernel
        /// gelu_approx(max(0, x - (mean + std * multiplier)))
        /// Used by: Gemma3n MLP layers with activation_sparsity > 0
        fn compiled_gelu_topk(x: &MlxArray, std_multiplier: f32) -> UniquePtr<MlxArray>;

        /// Compiled softcap: tanh(scores / cap) * cap — single fused kernel
        /// Used by: Gemma2 attention with logit softcapping
        fn compiled_softcap(scores: &MlxArray, cap: f32) -> UniquePtr<MlxArray>;

        /// Compiled clip_residual for float16 overflow prevention
        /// Used by: Gemma3 residual connections
        fn compiled_clip_residual(x: &MlxArray, y: &MlxArray) -> UniquePtr<MlxArray>;

        /// Compiled softcap SDPA: Q@K^T * scale -> softcap -> mask -> softmax -> @V
        /// Fuses the entire manual attention path into one compiled call
        /// Used by: Gemma2 attention with logit softcapping
        unsafe fn compiled_softcap_sdpa(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            scale: f32,
            softcap: f32,
            mask: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Compiled softcap SDPA with GQA: fuses repeat_kv + attention
        /// Avoids separate repeat_kv FFI calls by incorporating GQA internally
        /// Used by: Gemma2 attention (GQA + softcap)
        unsafe fn compiled_softcap_sdpa_gqa(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            scale: f32,
            softcap: f32,
            n_rep: i32,
            mask: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Compiled GELU MLP forward: down_proj(gelu(gate_proj(x)) * up_proj(x))
        /// Fuses entire MLP into a single compiled graph
        /// Used by: legacy/tests for precise GELU-gated MLP models
        unsafe fn compiled_gelu_mlp_forward(
            x: &MlxArray,
            gate_proj: &MlxArray,
            gate_scales: &MlxArray,
            gate_biases: *const MlxArray,
            up_proj: &MlxArray,
            up_scales: &MlxArray,
            up_biases: *const MlxArray,
            down_proj: &MlxArray,
            down_scales: &MlxArray,
            down_biases: *const MlxArray,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        /// Compiled GELU-approx MLP forward:
        /// down_proj(gelu_approx(gate_proj(x)) * up_proj(x)).
        /// Used by: Gemma2, Gemma3, Gemma4 dense MLP
        unsafe fn compiled_gelu_approx_mlp_forward(
            x: &MlxArray,
            gate_proj: &MlxArray,
            gate_scales: &MlxArray,
            gate_biases: *const MlxArray,
            up_proj: &MlxArray,
            up_scales: &MlxArray,
            up_biases: *const MlxArray,
            down_proj: &MlxArray,
            down_scales: &MlxArray,
            down_biases: *const MlxArray,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        /// Compiled GeGLU SwitchGLU MLP forward for quantized MoE experts.
        /// Wraps three `gather_qmm` calls (gate/up/down) plus a tanh-approx
        /// GeGLU activation into a single `mx::core::compile` window so
        /// MLX can schedule gate/up in parallel and fuse the intermediate
        /// element-wise ops. Only the no-sort path is fused; callers
        /// should fall back to separate `gather_qmm` calls when
        /// `sorted_indices == true` (prefill).
        /// Used by: Gemma 4 26B-a4b SwitchGeGLU experts (decode)
        unsafe fn compiled_switch_qgeglu_forward(
            x: &MlxArray,
            gate_w: &MlxArray,
            gate_s: &MlxArray,
            gate_b: *const MlxArray,
            up_w: &MlxArray,
            up_s: &MlxArray,
            up_b: *const MlxArray,
            down_w: &MlxArray,
            down_s: &MlxArray,
            down_b: *const MlxArray,
            rhs_indices: &MlxArray,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        /// Compiled SwiGLU MLP forward for non-quantized (FP16/BF16) weights
        /// Fuses gate_proj + silu + up_proj + multiply + down_proj into compiled graph
        /// Used by: Llama, Qwen2, Qwen3, Mistral and other SwiGLU FP models
        unsafe fn compiled_swiglu_mlp_forward_fp16(
            x: &MlxArray,
            gate_weight: &MlxArray,
            up_weight: &MlxArray,
            down_weight: &MlxArray,
            gate_bias: *const MlxArray,
            up_bias: *const MlxArray,
            down_bias: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Compiled GELU MLP forward for non-quantized (FP16/BF16) weights
        /// Fuses gate_proj + gelu + up_proj + multiply + down_proj into compiled graph
        /// Used by: Gemma, Gemma4 and other GELU-gated FP models
        unsafe fn compiled_gelu_mlp_forward_fp16(
            x: &MlxArray,
            gate_weight: &MlxArray,
            up_weight: &MlxArray,
            down_weight: &MlxArray,
            gate_bias: *const MlxArray,
            up_bias: *const MlxArray,
            down_bias: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Gemma3n dense MLP forward for non-quantized bf16 language MLP weights:
        /// cast input to bf16, run gate/up + gelu_approx or gelu_topk + down,
        /// then cast back to bf16. Preserves the Gemma3n precision policy
        /// while avoiding several Rust↔C++ bridge round-trips per layer.
        /// Used by: Gemma3n bf16 language MLP decode path
        unsafe fn gemma3n_mlp_forward(
            x: &MlxArray,
            gate_weight: &MlxArray,
            up_weight: &MlxArray,
            down_weight: &MlxArray,
            gate_bias: *const MlxArray,
            up_bias: *const MlxArray,
            down_bias: *const MlxArray,
            activation_sparsity: f32,
            std_multiplier: f32,
        ) -> UniquePtr<MlxArray>;

        /// Full transformer layer forward (maximum FFI reduction)
        unsafe fn transformer_layer_forward(
            x: &MlxArray,
            attn_norm_weight: &MlxArray,
            q_proj: &MlxArray,
            k_proj: &MlxArray,
            v_proj: &MlxArray,
            o_proj: &MlxArray,
            ffn_norm_weight: &MlxArray,
            gate_proj: &MlxArray,
            up_proj: &MlxArray,
            down_proj: &MlxArray,
            kv_cache_k: *const MlxArray,
            kv_cache_v: *const MlxArray,
            n_heads: i32,
            n_kv_heads: i32,
            head_dim: i32,
            rope_theta: f32,
            rope_offset: i32,
            norm_eps: f32,
        ) -> UniquePtr<MlxArray>;

        // Advanced indexing operations.
        /// Take elements along axis using indices
        fn take(a: &MlxArray, indices: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Gather elements using multi-dimensional indexing
        unsafe fn gather(
            a: &MlxArray,
            indices: &[*const MlxArray],
            axes: &[i32],
            slice_sizes: &[i32],
        ) -> UniquePtr<MlxArray>;

        /// Take along axis (like numpy.take_along_axis)
        fn take_along_axis(a: &MlxArray, indices: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Put along axis (scatter update)
        fn put_along_axis(
            a: &MlxArray,
            indices: &MlxArray,
            values: &MlxArray,
            axis: i32,
        ) -> UniquePtr<MlxArray>;

        /// Tile/repeat array
        fn tile(a: &MlxArray, reps: &[i32]) -> UniquePtr<MlxArray>;

        /// Repeat array along axis
        fn repeat(a: &MlxArray, repeats: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// Stack arrays along new axis
        unsafe fn stack(arrays: &[*const MlxArray], axis: i32) -> UniquePtr<MlxArray>;

        /// Concatenate arrays along existing axis
        unsafe fn concatenate(arrays: &[*const MlxArray], axis: i32) -> UniquePtr<MlxArray>;

        /// Create range of f32 values
        fn arange_f32(start: f32, stop: f32, step: f32) -> UniquePtr<MlxArray>;

        /// Create range of i32 values
        fn arange_i32(start: i32, stop: i32, step: i32) -> UniquePtr<MlxArray>;

        // Logical operations.
        /// Element-wise logical NOT
        fn logical_not(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise logical AND
        fn logical_and(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise logical OR
        fn logical_or(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// All along axis
        fn all_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Any along axis
        fn any_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Greater than or equal comparison
        fn greater_equal(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Less than or equal comparison
        fn less_equal(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Not equal comparison
        fn not_equal(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        // Activation functions.
        /// SiLU (Swish) activation: x * sigmoid(x)
        fn silu(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// GELU activation
        fn gelu(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Approximate GELU (erf-based for numerical stability)
        fn gelu_approx(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// ReLU activation
        fn relu(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Leaky ReLU activation
        fn leaky_relu(a: &MlxArray, negative_slope: f32) -> UniquePtr<MlxArray>;

        // Sorting and searching.
        /// Argsort along axis
        fn argsort(a: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Argpartition along axis
        fn argpartition(a: &MlxArray, kth: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// Argmin along axis
        fn argmin(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Top-k elements along axis
        fn topk(a: &MlxArray, k: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// Sort along axis
        fn sort(a: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Partition along axis
        fn partition(a: &MlxArray, kth: i32, axis: i32) -> UniquePtr<MlxArray>;

        // Cumulative operations
        /// Cumulative max along axis
        fn cummax(a: &MlxArray, axis: i32, reverse: bool, inclusive: bool) -> UniquePtr<MlxArray>;

        /// Cumulative min along axis
        fn cummin(a: &MlxArray, axis: i32, reverse: bool, inclusive: bool) -> UniquePtr<MlxArray>;

        /// Cumulative product along axis
        fn cumprod(a: &MlxArray, axis: i32, reverse: bool, inclusive: bool) -> UniquePtr<MlxArray>;

        // Scatter operations
        /// Scatter updates into array at indices
        fn scatter(
            a: &MlxArray,
            indices: &MlxArray,
            updates: &MlxArray,
            axis: i32,
        ) -> UniquePtr<MlxArray>;

        /// Scatter add updates into array at indices
        fn scatter_add(
            a: &MlxArray,
            indices: &MlxArray,
            updates: &MlxArray,
            axis: i32,
        ) -> UniquePtr<MlxArray>;

        /// Scatter max updates into array at indices
        fn scatter_max(
            a: &MlxArray,
            indices: &MlxArray,
            updates: &MlxArray,
            axis: i32,
        ) -> UniquePtr<MlxArray>;

        /// Scatter min updates into array at indices
        fn scatter_min(
            a: &MlxArray,
            indices: &MlxArray,
            updates: &MlxArray,
            axis: i32,
        ) -> UniquePtr<MlxArray>;

        /// Scatter multiply updates into array at indices
        fn scatter_prod(
            a: &MlxArray,
            indices: &MlxArray,
            updates: &MlxArray,
            axis: i32,
        ) -> UniquePtr<MlxArray>;

        // Bitwise operations
        /// Bitwise AND
        fn bitwise_and(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Bitwise OR
        fn bitwise_or(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Bitwise XOR
        fn bitwise_xor(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Left shift
        fn left_shift(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Right shift
        fn right_shift(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        // Linear algebra
        /// Tensor dot product
        fn tensordot(a: &MlxArray, b: &MlxArray, axes: i32) -> UniquePtr<MlxArray>;

        /// Inner product
        fn inner(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Outer product
        fn outer(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Matrix trace
        fn trace(a: &MlxArray, offset: i32, axis1: i32, axis2: i32) -> UniquePtr<MlxArray>;

        /// Roll (circular shift) along axis
        fn roll(a: &MlxArray, shift: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// Replace NaN/Inf with specified values
        fn nan_to_num(
            a: &MlxArray,
            nan_val: f32,
            posinf_val: f32,
            neginf_val: f32,
        ) -> UniquePtr<MlxArray>;

        /// Stop gradient (for autograd)
        fn stop_gradient(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// 2D convolution
        fn conv2d(
            input: &MlxArray,
            weight: &MlxArray,
            stride_h: i32,
            stride_w: i32,
            padding_h: i32,
            padding_w: i32,
            dilation_h: i32,
            dilation_w: i32,
            groups: i32,
        ) -> UniquePtr<MlxArray>;

        /// 2D average pooling
        /// Used by: VisionModule (Gemma3 AvgPool projector)
        fn avg_pool2d(
            input: &MlxArray,
            kernel_h: i32,
            kernel_w: i32,
            stride_h: i32,
            stride_w: i32,
            padding_h: i32,
            padding_w: i32,
        ) -> UniquePtr<MlxArray>;

        // MoE (Mixture of Experts) operations.
        /// Gather matrix multiply for MoE
        /// sorted_indices: if true, lhs_indices are pre-sorted for better memory access
        unsafe fn gather_mm(
            a: &MlxArray,
            b: &MlxArray,
            lhs_indices: *const MlxArray,
            rhs_indices: *const MlxArray,
            sorted_indices: bool,
        ) -> UniquePtr<MlxArray>;

        /// Gather quantized matrix multiply for MoE
        /// sorted_indices: if true, lhs_indices are pre-sorted for better memory access
        unsafe fn gather_qmm(
            x: &MlxArray,
            w: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            lhs_indices: *const MlxArray,
            rhs_indices: *const MlxArray,
            transpose: bool,
            group_size: i32,
            bits: i32,
            sorted_indices: bool,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        /// Direct quantized matrix multiplication
        /// y = x @ dequantize(w, scales, biases).T if transpose else x @ dequantize(w, scales, biases)
        unsafe fn quantized_matmul(
            x: &MlxArray,
            w: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            transpose: bool,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        /// Dequantize quantized weights to full precision
        /// biases: nullable for mxfp4/nvfp4/mxfp8 modes
        unsafe fn dequantize(
            w: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        // Embedding.
        /// Embedding lookup (same as take along axis 0)
        fn embedding(weight: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray>;

        /// Quantized embedding lookup with dequantization
        /// biases: nullable for mxfp4/nvfp4/mxfp8 modes
        unsafe fn quantized_embedding(
            weight: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            indices: &MlxArray,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        // Fast operations (using MLX fast kernels).
        /// Fast RoPE using MLX fast kernel
        fn fast_rope(
            x: &MlxArray,
            dims: i32,
            traditional: bool,
            base: f32,
            scale: f32,
            offset: i32,
        ) -> UniquePtr<MlxArray>;

        /// Fast RoPE with custom frequencies (for Yarn RoPE)
        fn fast_rope_with_freqs(
            x: &MlxArray,
            dims: i32,
            traditional: bool,
            scale: f32,
            offset: i32,
            freqs: &MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Compiled ProportionalRoPE: mirrors mlx-lm's full-head RoPE call
        /// with an `inf` frequency tail inside one `mx::core::compile`
        /// graph. Only valid when `rotated_dims > 0` and the input's last
        /// dim equals `head_dim`; callers must short-circuit the trivial /
        /// tail cases. Offset flows through as a scalar array inside the
        /// compile so per-step recompilation is avoided.
        fn compiled_proportional_rope(
            x: &MlxArray,
            freqs: &MlxArray,
            head_dim: i32,
            rotated_dims: i32,
            offset: i32,
        ) -> UniquePtr<MlxArray>;

        /// Compiled Gemma 4 Q-path with proportional RoPE:
        /// `reshape → fast::rms_norm → transpose → full-head ProportionalRoPE`
        /// folded into one compile window. `q_proj_out` is shaped
        /// `[B, L, n_heads * head_dim]` (output of `q_proj`). Applies only to
        /// Gemma 4 full-attention layers.
        fn compiled_q_path_proportional(
            q_proj_out: &MlxArray,
            q_norm_weight: &MlxArray,
            freqs: &MlxArray,
            rms_eps: f32,
            n_heads: i32,
            head_dim: i32,
            rotated_dims: i32,
            offset: i32,
        ) -> UniquePtr<MlxArray>;

        /// Compiled Gemma 4 per-layer-input-gate chain (e2b / e4b
        /// variants): `gate_proj → gelu_approx → mul(per_layer) →
        /// proj → post_norm → add(after_ffn)` in one compile window.
        /// Falls back to op-at-a-time inside C++ when the quantized
        /// config is not affine / gs=64 / bits=4 with biases.
        #[allow(clippy::too_many_arguments)]
        unsafe fn compiled_per_layer_input_gate(
            after_ffn: &MlxArray,
            per_layer_input: &MlxArray,
            gate_w: &MlxArray,
            gate_s: &MlxArray,
            gate_b: *const MlxArray,
            proj_w: &MlxArray,
            proj_s: &MlxArray,
            proj_b: *const MlxArray,
            post_norm_w: &MlxArray,
            post_norm_eps: f32,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        /// Fast RMS norm using MLX fast kernel
        fn fast_rms_norm(x: &MlxArray, weight: &MlxArray, eps: f32) -> UniquePtr<MlxArray>;

        /// Fast RMS norm without a learnable scale
        fn fast_rms_norm_no_weight(x: &MlxArray, eps: f32) -> UniquePtr<MlxArray>;

        /// Fast layer norm using MLX fast kernel
        unsafe fn fast_layer_norm(
            x: &MlxArray,
            weight: *const MlxArray,
            bias: *const MlxArray,
            eps: f32,
        ) -> UniquePtr<MlxArray>;

        /// Fast scaled dot product attention using MLX fast kernel
        unsafe fn fast_scaled_dot_product_attention(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            scale: f32,
            mask: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Fast SDPA with optional sinks (per-head attention bias for first position)
        /// Used by: GptOss
        #[allow(clippy::too_many_arguments)]
        unsafe fn fast_scaled_dot_product_attention_with_sinks(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            scale: f32,
            mask: *const MlxArray,
            sinks: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// SDPA with explicit causal masking for prefill
        #[rust_name = "ffi_fast_scaled_dot_product_attention_causal"]
        fn fast_scaled_dot_product_attention_causal(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            scale: f32,
        ) -> UniquePtr<MlxArray>;

        /// Decode-only paged attention over dense compatibility KV caches.
        ///
        /// `q` is `[B, Hq, 1, D]`. `cache_keys[i]` / `cache_values[i]` are
        /// per-sequence dense KV buffers, while `kv_lens`, `block_tables`, and
        /// `block_table_offsets` describe the logical paged layout.
        unsafe fn paged_decode_attention_dense_compat(
            q: &MlxArray,
            cache_keys: &[*const MlxArray],
            cache_values: &[*const MlxArray],
            kv_lens: &[i32],
            block_tables: &[i32],
            block_table_offsets: &[i32],
            block_size: i32,
            scale: f32,
        ) -> UniquePtr<MlxArray>;

        /// Decode-only paged attention over rotating ring-buffer KV caches.
        ///
        /// `logical_starts[i]` identifies the physical start of the logical
        /// visible window within `cache_keys[i]` / `cache_values[i]`.
        unsafe fn paged_decode_attention_rotating_compat(
            q: &MlxArray,
            cache_keys: &[*const MlxArray],
            cache_values: &[*const MlxArray],
            kv_lens: &[i32],
            logical_starts: &[i32],
            block_size: i32,
            scale: f32,
        ) -> UniquePtr<MlxArray>;

        /// Fused paged-attention decode Metal kernel (epic #116 Phase 6, #123).
        ///
        /// Reads scattered KV blocks directly out of the global pool via the
        /// block table, with no separate gather copy. `q` is `[B, Hq, 1, D]`
        /// f32; `k_pool` / `v_pool` are the layer's pool tensors
        /// `[num_blocks, block_size, Hkv, D]` f16. The metadata are i32 arrays:
        /// `rows` is every sequence's physical pool rows concatenated in
        /// block-table order, `row_offsets` (`[B + 1]`) the start of each
        /// sequence's rows, and `logical_starts` / `visible_lens` (`[B]`) bound
        /// each sequence's visible window. Returns `[B, Hq, 1, D]` f32.
        fn paged_attention_decode(
            q: &MlxArray,
            k_pool: &MlxArray,
            v_pool: &MlxArray,
            rows: &MlxArray,
            row_offsets: &MlxArray,
            logical_starts: &MlxArray,
            visible_lens: &MlxArray,
            scale: f32,
        ) -> UniquePtr<MlxArray>;

        fn sdpa_supports_fast_path(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            has_mask: bool,
            has_arr_mask: bool,
            do_causal: bool,
        ) -> bool;

        fn sdpa_supports_nax(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            has_mask: bool,
            has_arr_mask: bool,
            do_causal: bool,
        ) -> bool;

        /// Metal 4 attention dispatch via upstream MLX main SDPA.
        ///
        /// When `use_metal4` is true on supported M5 hardware, upstream MLX
        /// main may route `fast::scaled_dot_product_attention()` to its NAX
        /// kernel internally. This bridge preserves Rust-side softcap/window
        /// plumbing while delegating the kernel body to MLX.
        ///
        /// Prefer calling `layers::metal4_attention()` from model code — it
        /// queries `hardware::get_hardware()` and sets `use_metal4` automatically
        /// based on the chip generation and macOS version at runtime.
        unsafe fn fused_metal4_attention(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            scale: f32,
            mask: *const MlxArray,
            softcap: f32,
            window_size: i32,
            use_metal4: bool,
        ) -> UniquePtr<MlxArray>;

        /// Fused QKV projection + reshape + transpose + RoPE
        /// Reduces ~5 FFI calls (projection, reshape, transpose, rope) to 1
        #[allow(clippy::too_many_arguments)]
        unsafe fn fused_qkv_project_and_rope(
            x: &MlxArray,
            weight: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            num_heads: i32,
            head_dim: i32,
            rope_dims: i32,
            rope_base: f32,
            cache_offset: i32,
            group_size: i32,
            bits: i32,
            apply_rope: bool,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        /// Fused concatenated QKV projection + split + reshape + transpose + RoPE.
        /// Used by: Llama3-family and Gemma2 fused attention preparation paths.
        unsafe fn fused_qkv_project_split_rope(
            x: &MlxArray,
            weight: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            num_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            rope_dims: i32,
            rope_base: f32,
            cache_offset: i32,
            group_size: i32,
            bits: i32,
            mode: &str,
            q_out: &mut UniquePtr<MlxArray>,
            k_out: &mut UniquePtr<MlxArray>,
            v_out: &mut UniquePtr<MlxArray>,
        );

        /// Fused concatenated QKV projection + split + reshape + transpose +
        /// RMSNorm(Q/K) + RoPE. Applies the plain
        /// `x * weight * rsqrt(mean(x^2) + eps)` form; the caller passes the raw
        /// weight for a standard RMSNorm or `(1 + weight)` for Gemma.
        /// Used by: Gemma3, Qwen3, Qwen3-MoE attention preparation paths.
        unsafe fn fused_qkv_project_split_norm_rope(
            x: &MlxArray,
            weight: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            q_norm_weight: &MlxArray,
            k_norm_weight: &MlxArray,
            num_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            rope_dims: i32,
            rope_base: f32,
            rms_eps: f32,
            cache_offset: i32,
            group_size: i32,
            bits: i32,
            mode: &str,
            q_out: &mut UniquePtr<MlxArray>,
            k_out: &mut UniquePtr<MlxArray>,
            v_out: &mut UniquePtr<MlxArray>,
        );

        /// Fused concatenated QKV projection + split + reshape + transpose +
        /// SuScaledRoPE.
        /// Used by: Phi3/Phi3V longrope-su attention path.
        unsafe fn fused_qkv_project_split_su_scaled_rope(
            x: &MlxArray,
            weight: &MlxArray,
            scales: &MlxArray,
            biases: *const MlxArray,
            num_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            rope_dims: i32,
            rope_freqs: &MlxArray,
            rope_input_scale: f32,
            cache_offset: i32,
            group_size: i32,
            bits: i32,
            mode: &str,
            q_out: &mut UniquePtr<MlxArray>,
            k_out: &mut UniquePtr<MlxArray>,
            v_out: &mut UniquePtr<MlxArray>,
        );

        /// Experimental dense causal prefill attention path:
        /// qkv projection + split + rope + native causal SDPA + output projection.
        unsafe fn fused_causal_prefill_attention(
            x: &MlxArray,
            qkv_weight: &MlxArray,
            qkv_scales: &MlxArray,
            qkv_biases: *const MlxArray,
            o_weight: &MlxArray,
            o_scales: &MlxArray,
            o_biases: *const MlxArray,
            num_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            rope_dims: i32,
            rope_base: f32,
            scale: f32,
            group_size: i32,
            bits: i32,
            mode: &str,
            output_out: &mut UniquePtr<MlxArray>,
            k_out: &mut UniquePtr<MlxArray>,
            v_out: &mut UniquePtr<MlxArray>,
        );

        // Compiled operations (with kernel fusion).
        /// Compiled MoE expert forward with quantized weights
        /// Falls back to non-compiled for non-affine modes (mxfp4/nvfp4/mxfp8)
        unsafe fn compiled_moe_expert_forward(
            x: &MlxArray,
            gate_proj: &MlxArray,
            gate_scales: &MlxArray,
            gate_biases: *const MlxArray,
            up_proj: &MlxArray,
            up_scales: &MlxArray,
            up_biases: *const MlxArray,
            down_proj: &MlxArray,
            down_scales: &MlxArray,
            down_biases: *const MlxArray,
            group_size: i32,
            bits: i32,
            mode: &str,
        ) -> UniquePtr<MlxArray>;

        // Compiled MoE gate (sigmoid + topk + normalize + scale in one graph)
        /// Matches Python @mx.compile group_expert_select()
        fn compiled_moe_gate(
            gates: &MlxArray,
            correction_bias: &MlxArray,
            top_k: i32,
            scaling_factor: f32,
            norm_topk_prob: bool,
            indices_out: &mut UniquePtr<MlxArray>,
            scores_out: &mut UniquePtr<MlxArray>,
        );

        // Fused MoE forward: gate + switch_mlp + score weighting + shared expert
        /// Combines ~25 FFI calls into a single C++ function
        #[allow(clippy::too_many_arguments)]
        unsafe fn fused_moe_forward(
            x: &MlxArray,
            gate_weight: &MlxArray,
            correction_bias: &MlxArray,
            fc1_weight: &MlxArray,
            fc1_scales: &MlxArray,
            fc1_biases: &MlxArray,
            fc2_weight: &MlxArray,
            fc2_scales: &MlxArray,
            fc2_biases: &MlxArray,
            shared_up_weight: *const MlxArray,
            shared_up_scales: *const MlxArray,
            shared_up_biases: *const MlxArray,
            shared_down_weight: *const MlxArray,
            shared_down_scales: *const MlxArray,
            shared_down_biases: *const MlxArray,
            top_k: i32,
            scaling_factor: f32,
            norm_topk_prob: bool,
            group_size: i32,
            bits: i32,
        ) -> UniquePtr<MlxArray>;

        // SSM (Mamba2) fused Metal kernel.
        /// Check if SSM Metal kernel is available
        fn ssm_kernel_available() -> bool;

        /// Fused SSM update kernel for single-token decode
        /// Replaces ~55 individual ops with a single Metal kernel call
        /// Used by: NemotronH, NemotronNAS, Mamba2
        fn ssm_update_kernel(
            hidden_states: &MlxArray,
            a_log: &MlxArray,
            b: &MlxArray,
            c: &MlxArray,
            d: &MlxArray,
            dt: &MlxArray,
            dt_bias: &MlxArray,
            state_in: &MlxArray,
            time_step_min: f32,
            time_step_max: f32,
            output: &mut UniquePtr<MlxArray>,
            next_state: &mut UniquePtr<MlxArray>,
        );

        /// Fused MoE expert kernel for single-token decode. gate/up use
        /// `gu_bits` (power-of-2: 4/8), down uses `d_bits` (4/8/6); group_size
        /// is shared. Mixed widths support e.g. dots.llm1 (gate/up 4, down 6).
        #[allow(clippy::too_many_arguments)]
        fn fused_moe_expert_kernel(
            x: &MlxArray,
            indices: &MlxArray,
            gate_w: &MlxArray,
            gate_s: &MlxArray,
            gate_b: &MlxArray,
            up_w: &MlxArray,
            up_s: &MlxArray,
            up_b: &MlxArray,
            down_w: &MlxArray,
            down_s: &MlxArray,
            down_b: &MlxArray,
            scores: &MlxArray,
            din: i32,
            dff: i32,
            k: i32,
            gu_bits: i32,
            d_bits: i32,
            group_size: i32,
        ) -> UniquePtr<MlxArray>;

        /// Like `fused_moe_expert_kernel` but GeGLU (gelu tanh approx) instead
        /// of SwiGLU for the gate/up activation (gemma4 experts).
        #[allow(clippy::too_many_arguments)]
        fn fused_moe_geglu_kernel(
            x: &MlxArray,
            indices: &MlxArray,
            gate_w: &MlxArray,
            gate_s: &MlxArray,
            gate_b: &MlxArray,
            up_w: &MlxArray,
            up_s: &MlxArray,
            up_b: &MlxArray,
            down_w: &MlxArray,
            down_s: &MlxArray,
            down_b: &MlxArray,
            scores: &MlxArray,
            din: i32,
            dff: i32,
            k: i32,
            gu_bits: i32,
            d_bits: i32,
            group_size: i32,
        ) -> UniquePtr<MlxArray>;

        /// Fused xIELU activation (Apertus): one Metal launch covering the
        /// `apertus_xielu` elementwise graph (square/min/expm1/where/...). The
        /// per-layer scalars `alpha_p` / `alpha_n` (post-softplus) and `beta` /
        /// `eps` are passed by value. Greedy temp-0 byte-identical to the
        /// elementwise path on Apple Silicon; falls back to an equivalent
        /// elementwise graph on non-Metal back-ends. Gated by `MLXCEL_FUSED_XIELU`.
        fn fused_xielu(
            x: &MlxArray,
            alpha_p: f32,
            alpha_n: f32,
            beta: f32,
            eps: f32,
        ) -> UniquePtr<MlxArray>;

        /// BitLinear ternary matmul (BitNet b1.58). `packed_weights` is
        /// [out_features/4, in_features] uint8 (2-bit ternary, 4 rows/byte),
        /// scaled by `weight_scale[0]` (inverted unless linear_class is
        /// autobitlinear).
        fn bitlinear_matmul(
            x: &MlxArray,
            packed_weights: &MlxArray,
            weight_scale: &MlxArray,
            in_features: i32,
            out_features: i32,
            invert_weight_scales: bool,
        ) -> UniquePtr<MlxArray>;

        /// Fused gated-delta single-token decode step.
        /// Combines: decay → kv_mem → delta → state_update → output into one C++ call.
        /// Replaces ~26 FFI round-trips with 1.
        /// Used by: Qwen3.5, Qwen3Next, KimiLinear (GatedDeltaNet T=1 decode)
        #[allow(clippy::too_many_arguments)]
        unsafe fn fused_gated_delta_decode_step(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            g: &MlxArray,
            beta: &MlxArray,
            state: &MlxArray,
            q_dtype: i32,
            output: &mut UniquePtr<MlxArray>,
            new_state_out: &mut UniquePtr<MlxArray>,
        );

        // GatedDeltaNet custom Metal kernel.
        /// Check if GatedDeltaNet Metal kernel is available
        fn gated_delta_kernel_available() -> bool;

        /// Start a Metal GPU trace capture. `path` must be an absolute
        /// path ending in `.gputrace` and must not already exist. The
        /// process must have been launched with `MTL_CAPTURE_ENABLED=1`;
        /// otherwise Metal drops the capture silently. Mirrors Python's
        /// `mx.metal.start_capture` so mlxcel traces can be compared
        /// side-by-side with mlx-lm traces in Xcode's Metal Debugger.
        fn metal_start_capture(path: &str);

        /// Stop an active Metal GPU trace capture.
        fn metal_stop_capture();

        /// GatedDeltaNet custom Metal kernel forward.
        /// Handles both T=1 (decode) and T>1 (prefill) in a single GPU dispatch.
        /// Used by: Qwen3.5, Qwen3Next, KimiLinear
        #[allow(clippy::too_many_arguments)]
        unsafe fn metal_gated_delta_forward(
            q: &MlxArray,
            k: &MlxArray,
            v: &MlxArray,
            g: &MlxArray,
            beta: &MlxArray,
            state: &MlxArray,
            mask: *const MlxArray, // nullable
            output: &mut UniquePtr<MlxArray>,
            new_state: &mut UniquePtr<MlxArray>,
        );

        // Fused Mamba2 mixer forward for single-token decode.
        /// Combines in_proj + conv1d + SSM kernel + MambaRMSNormGated + out_proj into one C++ call.
        /// Replaces ~23 FFI round-trips for the hot decode path.
        /// Used by: NemotronH
        #[allow(clippy::too_many_arguments)]
        unsafe fn fused_mamba2_forward(
            hidden_states: &MlxArray,
            // in_proj (quantized)
            in_proj_weight: &MlxArray,
            in_proj_scales: &MlxArray,
            in_proj_biases: *const MlxArray, // nullable
            // conv1d
            conv_weight: &MlxArray,
            conv_bias: *const MlxArray, // nullable
            // SSM parameters
            a_log: &MlxArray,
            d: &MlxArray,
            dt_bias: &MlxArray,
            // norm weight
            norm_weight: &MlxArray,
            // out_proj (quantized)
            out_proj_weight: &MlxArray,
            out_proj_scales: &MlxArray,
            out_proj_biases: *const MlxArray, // nullable
            // cache state inputs
            conv_state_in: &MlxArray,
            ssm_state_in: &MlxArray,
            // mamba2 config
            intermediate_size: i32,
            conv_dim: i32,
            conv_kernel_size: i32,
            num_heads: i32,
            head_dim: i32,
            n_groups: i32,
            ssm_state_size: i32,
            time_step_min: f32,
            time_step_max: f32,
            norm_eps: f32,
            // quantization config
            group_size: i32,
            bits: i32,
            // outputs
            output: &mut UniquePtr<MlxArray>,
            conv_state_out: &mut UniquePtr<MlxArray>,
            ssm_state_out: &mut UniquePtr<MlxArray>,
        );

        // NemotronH full-forward decode (opaque handle pattern).
        #[allow(clippy::too_many_arguments)]
        unsafe fn nemotron_register_model(
            embed_w: &MlxArray,
            embed_s: &MlxArray,
            embed_b: &MlxArray,
            final_norm_w: &MlxArray,
            lm_head_w: &MlxArray,
            lm_head_s: &MlxArray,
            lm_head_b: *const MlxArray,
            norm_weights: &[*const MlxArray],
            block_types: &[i32],
            mamba_weights: &[*const MlxArray],
            moe_weights: &[*const MlxArray],
            attn_weights: &[*const MlxArray],
            norm_eps: f32,
            group_size: i32,
            bits: i32,
            m_inter: i32,
            m_cdim: i32,
            m_ck: i32,
            m_heads: i32,
            m_hdim: i32,
            m_groups: i32,
            m_state: i32,
            m_ts_min: f32,
            m_ts_max: f32,
            m_neps: f32,
            moe_tk: i32,
            moe_sc: f32,
            moe_norm: bool,
            a_heads: i32,
            a_kvh: i32,
            a_hdim: i32,
            a_rope: f32,
            a_scale: f32,
        ) -> u64;

        fn nemotron_free_model(handle: u64);

        #[allow(clippy::too_many_arguments)]
        unsafe fn nemotron_decode_step(
            handle: u64,
            input_ids: &MlxArray,
            mamba_conv_in: &[*const MlxArray],
            mamba_ssm_in: &[*const MlxArray],
            attn_kv_keys: &[*const MlxArray],
            attn_kv_values: &[*const MlxArray],
            attn_kv_offsets: &[i32],
            logits: &mut UniquePtr<MlxArray>,
            mamba_conv_out: &mut [UniquePtr<MlxArray>],
            mamba_ssm_out: &mut [UniquePtr<MlxArray>],
        );

        // Memory management.
        /// Clear memory cache
        fn clear_memory_cache();

        /// Async eval single array
        fn async_eval(arr: &MlxArray);

        /// Async eval multiple arrays at once
        unsafe fn async_eval_all(arrays: &[*const MlxArray]);

        /// Detach evaluated arrays from their construction graphs.
        unsafe fn detach_all(arrays: &[*const MlxArray]);

        /// Synchronize default stream
        fn synchronize_default();

        /// Set default device for subsequent operations
        fn set_default_device(gpu: bool);

        /// Set wired memory limit, returns previous limit
        fn set_wired_limit(limit: usize) -> usize;

        /// Get current wired memory limit
        fn get_wired_limit() -> usize;

        // MLX runtime memory accounting (issue #55).
        //
        // These are raw bridge entry points wired to `mlx::core::get_*_memory`
        // and friends in `mlx/memory.h`. The active allocator (Metal /
        // CUDA / no-gpu CommonAllocator) decides what each value means.
        // Prefer the typed wrappers in `crate::memory` over calling these
        // directly — they return `u64` for cross-platform clarity and
        // bundle the four most useful counters into a single snapshot.

        /// Bytes actively allocated by the MLX allocator (excludes cache).
        fn get_active_memory() -> usize;

        /// Peak active bytes seen since process start or last reset.
        fn get_peak_memory() -> usize;

        /// Bytes held in the allocator's free-buffer cache (0 on CPU-only backend).
        fn get_cache_memory() -> usize;

        /// Set the soft allocator memory limit in bytes. Returns previous limit.
        fn set_memory_limit(limit: usize) -> usize;

        /// Get the current soft allocator memory limit in bytes.
        fn get_memory_limit() -> usize;

        /// Set the allocator cache limit in bytes. Returns previous limit.
        /// On the no-gpu CPU backend this is a no-op that returns 0.
        fn set_cache_limit(limit: usize) -> usize;

        /// Reset the recorded peak memory counter to 0.
        fn reset_peak_memory();

        /// Get max GPU memory size (works across Metal and CUDA backends)
        fn gpu_max_memory_size() -> usize;

        /// Create new GPU stream
        fn new_gpu_stream() -> UniquePtr<MlxStream>;

        // Optimized generation functions.
        /// Extract last token logits: logits[:, -1, :] -> [batch, vocab]
        /// Optimized for sampling during generation
        fn slice_last_logits(logits: &MlxArray) -> UniquePtr<MlxArray>;

        /// Slice on the last dimension only: a[..., start:end]
        /// Useful for fused QKV/gate_up projections
        fn slice_last_dim(a: &MlxArray, start: i32, end: i32) -> UniquePtr<MlxArray>;

        /// Argmax on last axis for greedy sampling
        fn argmax_last_axis(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Reshape token for next forward pass: [] or [batch] -> [batch, 1]
        /// Allows passing token array directly without extracting scalar
        fn reshape_token_for_forward(token: &MlxArray) -> UniquePtr<MlxArray>;

        /// Async eval two arrays at once (for lookahead pipelining)
        fn async_eval_pair(a: &MlxArray, b: &MlxArray);

        /// Export two unevaluated arrays as a DOT graph for profiling.
        fn export_to_dot_pair(path: &str, a: &MlxArray, b: &MlxArray);

        /// Set default stream for subsequent operations
        fn set_default_stream(stream: &MlxStream);

        /// Check whether the current default device is GPU
        fn is_gpu_available() -> bool;

        /// Fused sampling: temperature + top-k + top-p + min-p + categorical
        /// in a single C++ call to minimize FFI round-trips.
        /// Input: 2D logits [batch, vocab] (already sliced, penalties applied)
        /// Returns sampled token
        fn fused_sample(
            logits: &MlxArray,
            temperature: f32,
            top_k: i32,
            top_p: f32,
            min_p: f32,
        ) -> UniquePtr<MlxArray>;

        // SSM (State Space Model) primitives for Mamba/Jamba/Nemotron-H.
        /// Cumulative sum along axis
        fn cumsum(a: &MlxArray, axis: i32, reverse: bool, inclusive: bool) -> UniquePtr<MlxArray>;

        /// Lower triangular matrix (keeps elements on and below k-th diagonal)
        fn tril(a: &MlxArray, k: i32) -> UniquePtr<MlxArray>;

        /// Upper triangular matrix (keeps elements on and above k-th diagonal)
        fn triu(a: &MlxArray, k: i32) -> UniquePtr<MlxArray>;

        /// Clip values to range [a_min, a_max]
        fn clip(a: &MlxArray, a_min: &MlxArray, a_max: &MlxArray) -> UniquePtr<MlxArray>;

        /// log(1 + x) - numerically stable for small x
        fn log1p(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Softplus activation: log(1 + exp(x))
        fn softplus(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// 1D convolution with groups support (for depthwise conv when groups=channels)
        fn conv1d(
            input: &MlxArray,
            weight: &MlxArray,
            stride: i32,
            padding: i32,
            dilation: i32,
            groups: i32,
        ) -> UniquePtr<MlxArray>;

        /// Swap axes (convenient for SSM attention)
        fn swap_axes(a: &MlxArray, axis1: i32, axis2: i32) -> UniquePtr<MlxArray>;

        // Core ops additions.
        /// Create identity matrix of size n x n
        fn identity(n: i32, dtype: i32) -> UniquePtr<MlxArray>;

        /// Create lower triangular matrix of ones
        fn tri(n: i32, m: i32, k: i32, dtype: i32) -> UniquePtr<MlxArray>;

        /// Unflatten axis into given shape
        fn unflatten(a: &MlxArray, axis: i32, shape: &[i32]) -> UniquePtr<MlxArray>;

        /// View array with given shape and strides
        fn as_strided(
            a: &MlxArray,
            shape: &[i32],
            strides: &[i64],
            offset: usize,
        ) -> UniquePtr<MlxArray>;

        /// Ensure the array memory is contiguous
        fn contiguous(a: &MlxArray, allow_col_major: bool) -> UniquePtr<MlxArray>;

        /// Broadcast a list of arrays against one another (returns i-th result)
        unsafe fn broadcast_arrays_get(
            arrays: &[*const MlxArray],
            index: usize,
        ) -> UniquePtr<MlxArray>;

        /// Number of results from broadcast_arrays
        unsafe fn broadcast_arrays_count(arrays: &[*const MlxArray]) -> usize;

        /// Floor division element-wise
        fn floor_divide(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Element-wise equality check (with nan handling option)
        fn array_equal(a: &MlxArray, b: &MlxArray, equal_nan: bool) -> UniquePtr<MlxArray>;

        /// Check if arrays are close within tolerances (returns scalar bool array)
        fn allclose(a: &MlxArray, b: &MlxArray, rtol: f64, atol: f64) -> UniquePtr<MlxArray>;

        /// Element-wise close check within tolerances
        fn isclose(a: &MlxArray, b: &MlxArray, rtol: f64, atol: f64) -> UniquePtr<MlxArray>;

        /// Compute median over all elements
        fn median_all(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Compute median along axis
        fn median_axis(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Cumulative log-sum-exp along axis
        fn logcumsumexp(
            a: &MlxArray,
            axis: i32,
            reverse: bool,
            inclusive: bool,
        ) -> UniquePtr<MlxArray>;

        /// Bitwise invert
        fn bitwise_invert(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Real part of complex array
        fn real_part(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Imaginary part of complex array
        fn imag_part(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Complex conjugate
        fn conjugate(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Reinterpret array as given dtype (bitwise view)
        fn view(a: &MlxArray, dtype: i32) -> UniquePtr<MlxArray>;

        /// Kronecker product of two arrays
        fn kron(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// addmm: beta * c + alpha * (a @ b)
        fn addmm(
            c: &MlxArray,
            a: &MlxArray,
            b: &MlxArray,
            alpha: f32,
            beta: f32,
        ) -> UniquePtr<MlxArray>;

        /// Block-sparse masked matrix multiply
        unsafe fn block_masked_mm(
            a: &MlxArray,
            b: &MlxArray,
            block_size: i32,
            mask_out: *const MlxArray,
            mask_lhs: *const MlxArray,
            mask_rhs: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Segmented matrix multiply
        fn segmented_mm(a: &MlxArray, b: &MlxArray, segments: &MlxArray) -> UniquePtr<MlxArray>;

        /// Hadamard transform
        fn hadamard_transform(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Number of elements along axes as scalar array
        fn number_of_elements(
            a: &MlxArray,
            axes: &[i32],
            inverted: bool,
            dtype: i32,
        ) -> UniquePtr<MlxArray>;

        // Convolution additions.
        /// 3D convolution
        fn conv3d(
            input: &MlxArray,
            weight: &MlxArray,
            stride_d: i32,
            stride_h: i32,
            stride_w: i32,
            padding_d: i32,
            padding_h: i32,
            padding_w: i32,
            dilation_d: i32,
            dilation_h: i32,
            dilation_w: i32,
            groups: i32,
        ) -> UniquePtr<MlxArray>;

        /// 1D transposed convolution
        fn conv_transpose1d(
            input: &MlxArray,
            weight: &MlxArray,
            stride: i32,
            padding: i32,
            dilation: i32,
            output_padding: i32,
            groups: i32,
        ) -> UniquePtr<MlxArray>;

        /// 2D transposed convolution
        fn conv_transpose2d(
            input: &MlxArray,
            weight: &MlxArray,
            stride_h: i32,
            stride_w: i32,
            padding_h: i32,
            padding_w: i32,
            dilation_h: i32,
            dilation_w: i32,
            output_padding_h: i32,
            output_padding_w: i32,
            groups: i32,
        ) -> UniquePtr<MlxArray>;

        /// 3D transposed convolution
        fn conv_transpose3d(
            input: &MlxArray,
            weight: &MlxArray,
            stride_d: i32,
            stride_h: i32,
            stride_w: i32,
            padding_d: i32,
            padding_h: i32,
            padding_w: i32,
            dilation_d: i32,
            dilation_h: i32,
            dilation_w: i32,
            output_padding_d: i32,
            output_padding_h: i32,
            output_padding_w: i32,
            groups: i32,
        ) -> UniquePtr<MlxArray>;

        // Einsum.
        /// Einsum contraction
        unsafe fn einsum(subscripts: &str, operands: &[*const MlxArray]) -> UniquePtr<MlxArray>;

        // Linear algebra (linalg).
        /// Vector/matrix norm along axis
        fn linalg_norm(a: &MlxArray, axis: i32, keepdims: bool) -> UniquePtr<MlxArray>;

        /// Vector/matrix norm with numeric ord along axis
        fn linalg_norm_ord(
            a: &MlxArray,
            ord: f64,
            axis: i32,
            keepdims: bool,
        ) -> UniquePtr<MlxArray>;

        /// Vector/matrix norm with string ord (e.g. "fro", "nuc") along axis
        fn linalg_norm_str(
            a: &MlxArray,
            ord: &str,
            axis: i32,
            keepdims: bool,
        ) -> UniquePtr<MlxArray>;

        /// QR decomposition — Q matrix
        fn linalg_qr_q(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// QR decomposition — R matrix
        fn linalg_qr_r(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// SVD — U matrix
        fn linalg_svd_u(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// SVD — singular values S
        fn linalg_svd_s(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// SVD — Vt matrix
        fn linalg_svd_vt(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Matrix inverse
        fn linalg_inv(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Moore-Penrose pseudoinverse
        fn linalg_pinv(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Cholesky decomposition
        fn linalg_cholesky(a: &MlxArray, upper: bool) -> UniquePtr<MlxArray>;

        /// Solve linear system a @ x = b
        fn linalg_solve(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray>;

        /// Solve triangular linear system
        fn linalg_solve_triangular(a: &MlxArray, b: &MlxArray, upper: bool) -> UniquePtr<MlxArray>;

        /// LU decomposition — P permutation
        fn linalg_lu_p(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// LU decomposition — L lower triangular
        fn linalg_lu_l(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// LU decomposition — U upper triangular
        fn linalg_lu_u(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// LU factorization — LU combined matrix
        fn linalg_lu_factor_lu(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// LU factorization — pivots
        fn linalg_lu_factor_pivots(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Eigen decomposition — eigenvalues
        fn linalg_eig_values(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Eigen decomposition — eigenvectors
        fn linalg_eig_vectors(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Eigenvalues only
        fn linalg_eigvals(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Symmetric eigen decomposition — eigenvalues (default UPLO="L")
        fn linalg_eigh_values(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Symmetric eigen decomposition — eigenvectors (default UPLO="L")
        fn linalg_eigh_vectors(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Symmetric eigenvalues only
        fn linalg_eigvalsh(a: &MlxArray) -> UniquePtr<MlxArray>;

        /// Cross product along axis
        fn linalg_cross(a: &MlxArray, b: &MlxArray, axis: i32) -> UniquePtr<MlxArray>;

        /// Triangular matrix inverse
        fn linalg_tri_inv(a: &MlxArray, upper: bool) -> UniquePtr<MlxArray>;

        /// Cholesky factor inverse
        fn linalg_cholesky_inv(a: &MlxArray, upper: bool) -> UniquePtr<MlxArray>;

        // FFT.
        /// 1D FFT with explicit n and axis
        fn fft(a: &MlxArray, n: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// 1D inverse FFT
        fn ifft(a: &MlxArray, n: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// 1D real FFT
        fn rfft(a: &MlxArray, n: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// 1D inverse real FFT
        fn irfft(a: &MlxArray, n: i32, axis: i32) -> UniquePtr<MlxArray>;

        /// 2D FFT
        fn fft2(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// 2D inverse FFT
        fn ifft2(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// 2D real FFT
        fn rfft2(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// 2D inverse real FFT
        fn irfft2(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// N-dimensional FFT with explicit axes
        fn fftn_axes(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// N-dimensional inverse FFT with explicit axes
        fn ifftn_axes(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// N-dimensional real FFT with explicit axes
        fn rfftn_axes(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// N-dimensional inverse real FFT with explicit axes
        fn irfftn_axes(a: &MlxArray, n: &[i32], axes: &[i32]) -> UniquePtr<MlxArray>;

        /// FFT shift along axes
        fn fftshift(a: &MlxArray, axes: &[i32]) -> UniquePtr<MlxArray>;

        /// Inverse FFT shift along axes
        fn ifftshift(a: &MlxArray, axes: &[i32]) -> UniquePtr<MlxArray>;

        // Random.
        /// Create a random key from seed
        fn random_key(seed: u64) -> UniquePtr<MlxArray>;

        /// Split key into num subkeys (returns array of shape [num, 2])
        fn random_split_key(key: &MlxArray, num: i32) -> UniquePtr<MlxArray>;

        /// Uniform random in [low, high)
        unsafe fn random_uniform(
            low: f32,
            high: f32,
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Standard normal random samples
        unsafe fn random_normal(
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Bernoulli random samples with probability p
        unsafe fn random_bernoulli_p(
            p: f32,
            shape: &[i32],
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Random integers in [low, high)
        unsafe fn random_randint(
            low: i32,
            high: i32,
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Truncated normal random samples
        unsafe fn random_truncated_normal(
            lower: f32,
            upper: f32,
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Gumbel random samples
        unsafe fn random_gumbel(
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Laplace random samples
        unsafe fn random_laplace(
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Random permutation of integers 0..x
        unsafe fn random_permutation(x: i32, key: *const MlxArray) -> UniquePtr<MlxArray>;

        /// Random permutation of array along axis
        unsafe fn random_permutation_array(
            a: &MlxArray,
            axis: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        /// Multivariate normal random samples
        unsafe fn random_multivariate_normal(
            mean: &MlxArray,
            cov: &MlxArray,
            shape: &[i32],
            dtype: i32,
            key: *const MlxArray,
        ) -> UniquePtr<MlxArray>;

        // Quantization additions.
        /// Quantize weights — quantized weights
        fn quantize_weights_w(w: &MlxArray, group_size: i32, bits: i32) -> UniquePtr<MlxArray>;

        /// Quantize weights — scales
        fn quantize_weights_scales(w: &MlxArray, group_size: i32, bits: i32)
            -> UniquePtr<MlxArray>;

        /// Quantize weights — biases
        fn quantize_weights_biases(w: &MlxArray, group_size: i32, bits: i32)
            -> UniquePtr<MlxArray>;

        // -------------------------------------------------------------------
        // Fused Sparse-V SDPA Metal kernel.
        // -------------------------------------------------------------------
        /// Fused-skip Sparse-V weighted sum (Turbo4Asym KV cache, optimized).
        ///
        /// Runs a Metal kernel that computes
        ///   `out[b, h, q, d] = Σ_t attn_weights[b, h, q, t] * V_dq[b, h, t, d]`
        /// while *skipping per-thread* the inner-loop work for tokens whose
        /// attention weight ≤ `threshold`. The output is *unrotated* — the
        /// caller must apply the inverse `signs1 · WHT · signs2` Turbo4
        /// rotation to produce the final FP16 attention output.
        ///
        /// Inputs (caller must pre-flatten):
        /// - `attn_weights`: `[B*Hq, Tq, Tk]` FP32 — post-softmax weights.
        /// - `v_packed`:     `[B*Hkv, Tk, D/2]` UINT8 — nibble-packed indices.
        /// - `v_rescale`:    `[B*Hkv, Tk]` FP16 — precomputed per-token
        ///   rescale `norm[t] / max(|y_hat[t]|, 1e-10)`. The
        ///   previous kernel implementation re-derived this on-GPU per token
        ///   via a `log2(Dim) + 2`-barrier threadgroup tree reduction; that
        ///   reduction dominated decode latency on M5 Max at 4 K context
        ///   for `turbo4-asym` (A/B). Precomputing eliminates the
        ///   threadgroup barriers from the kernel hot path.
        /// - `codebook`:     `[16]` FP32 — Lloyd-Max centroids.
        /// - `dim`: head dimension `D`.
        /// - `n_rep`: `Hq / Hkv` (1 for non-grouped attention).
        /// - `threshold`: alive cutoff. `0.0` disables skipping.
        ///
        /// Output: `[B*Hq, Tq, D]` FP32.
        ///
        /// Metal-only — fails to link on non-macOS targets. Callers are
        /// expected to gate via `KVCache::sparse_v_attention`, which only
        /// dispatches when on macOS + Turbo4Asym + threshold > 0.
        fn turbo_sparse_v_weighted_sum(
            attn_weights: &MlxArray,
            v_packed: &MlxArray,
            v_rescale: &MlxArray,
            codebook: &MlxArray,
            dim: i32,
            n_rep: i32,
            threshold: f32,
        ) -> UniquePtr<MlxArray>;

        // -------------------------------------------------------------------
        // Fused Turbo4Delegated cold-V weighted-sum kernel.
        // -------------------------------------------------------------------
        /// Fused Turbo4Delegated cold-V weighted sum.
        ///
        /// Runs a Metal kernel that computes
        ///   `out_cold[b, h, q, d] = Σ_t attn_cold[b, h, q, t] * V_dq[b, h, t, d]`
        /// over the cold token range, reading the packed Turbo4 V indices
        /// directly. The dequantised cold V never materialises in global
        /// memory; that is the point (replaces the earlier
        /// `cold_v_dequant_cache` memo + per-step `concat(cold_v, hot_v)`).
        ///
        /// The output is unrotated — the caller must apply the inverse
        /// `signs1 · WHT · signs2` Turbo4 rotation to produce the rotated
        /// cold contribution, then add the hot-V matmul `attn_hot @ hot_v`
        /// to get the final FP16 SDPA output.
        ///
        /// Inputs (caller must pre-flatten):
        /// - `attn_weights_cold`: `[B*Hq, Tq, T_cold]` FP32 — post-softmax
        ///   weights restricted to the cold range. The caller slices the
        ///   full softmax output to the first `T_cold` columns.
        /// - `v_packed_cold`:     `[B*Hkv, T_cold, D/2]` UINT8 — nibble-
        ///   packed Turbo4 indices for the cold body.
        /// - `v_rescale_cold`:    `[B*Hkv, T_cold]` FP16 — precomputed
        ///   per-token rescale `norm[t] / max(|y_hat[t]|, 1e-10)` (same semantic content as the Sparse-V kernel rescale).
        /// - `codebook`:          `[16]` FP32 — Lloyd-Max centroids.
        /// - `dim`: head dimension `D`.
        /// - `n_rep`: `Hq / Hkv` (1 for non-grouped attention).
        /// - `threshold`: alive cutoff for sparse-V skipping. `0.0` runs the
        ///   full cold sweep without skipping (default for the
        ///   Turbo4Delegated decode path; gated on
        ///   `--turbo-sparse-v-threshold`).
        ///
        /// Output: `[B*Hq, Tq, D]` FP32 — unrotated cold weighted sum.
        ///
        /// Metal-only — fails to link on non-macOS targets. Callers are
        /// expected to gate via `KVCache::update_and_turbo4_delegated_attention`,
        /// which only dispatches when on macOS + Turbo4Delegated.
        fn turbo4_delegated_cold_weighted_sum(
            attn_weights_cold: &MlxArray,
            v_packed_cold: &MlxArray,
            v_rescale_cold: &MlxArray,
            codebook: &MlxArray,
            dim: i32,
            n_rep: i32,
            threshold: f32,
        ) -> UniquePtr<MlxArray>;

        /// Fused bulk rotated dequant for the Swift-LM-style
        /// Turbo4Delegated dequant-first SDPA path.
        ///
        /// Inputs:
        /// - `v_packed`:  `[B, H, T, D/2]` UINT8 packed Turbo4 V indices.
        /// - `v_rescale`: `[B, H, T, 1]` FP16 precomputed `norm/|y_hat|`.
        /// - `codebook`:  `[16]` FP32 Lloyd-Max centroids.
        /// - `dim`:       head dimension `D`.
        ///
        /// Output: `[B, H, T, D]` FP16 in rotated value basis.
        fn turbo4_delegated_bulk_dequant_rotated(
            v_packed: &MlxArray,
            v_rescale: &MlxArray,
            codebook: &MlxArray,
            dim: i32,
        ) -> UniquePtr<MlxArray>;

        // Steel-attention-envelope fused Turbo4Delegated SDPA
        // kernel launcher. One Metal dispatch performs the entire post-Q·K
        // SDPA inline (numerically stable softmax over T_total + cold-V
        // dequant + hot-V accumulation), returning two FP32 outputs that
        // share the same softmax denominator:
        //
        // - `out_cold_pre`: `[B*Hq, Tq, D]` unrotated cold weighted sum. The
        //   host applies `signs1·WHT(signs2·)` to produce the rotated cold
        //   contribution.
        // - `out_hot`:      `[B*Hq, Tq, D]` hot weighted sum (unrotated; hot V
        //   is plain FP16 without a quantize-time rotation).
        //
        // The host sums the rotated cold contribution and the hot
        // contribution and casts to FP16. Bit-equivalent (within FP16
        // round-off) to the reference path
        // `softmax(Q·K^T·scale + mask) @ V_full` because the rotation is
        // linear and the kernel uses one shared softmax denominator.
        //
        // Inputs:
        // - `scores`:        `[B*Hq, Tq, T_total]` FP32 — pre-computed
        //   `Q·K^T * scale` with the additive mask already added (causal
        //   positions carry `-inf` so they contribute zero post-softmax).
        // - `cold_packed`:   `[B*Hkv, T_cold, D/2]` UINT8 — nibble-packed
        //   cold V indices, or a 1-token zero placeholder when `T_cold == 0`
        //   (MLX `metal_kernel` rejects zero-shape buffers).
        // - `cold_rescale`:  `[B*Hkv, T_cold]` FP16 — precomputed per-token
        //   cold-V rescale `norm[t] / max(|y_hat[t]|, 1e-10)`.
        // - `hot_v`:         `[B*Hkv, T_hot, D]` FP16 — plain FP16 hot V, or
        //   a 1-token zero placeholder when `T_hot == 0`.
        // - `codebook`:      `[16]` FP32 — Lloyd-Max centroids.
        // - `dim`:           head dimension `D` (must be a power of 2).
        // - `n_rep`:         `Hq / Hkv` (1 for non-grouped attention).
        // - `cold_offset`:   `T_cold` — number of cold tokens to sweep.
        //   Pass `0` to skip the cold loop entirely; the kernel reads no
        //   data from `cold_packed` / `cold_rescale` in that case (the
        //   placeholder buffers above are never dereferenced).
        // - `hot_offset`:    `T_hot` — number of hot tokens to sweep. Pass
        //   `0` to skip the hot loop.
        // - `threshold`:     alive cutoff for cold-V skipping. The kernel
        //   compares `exp(score - max) > threshold * sum_exp` (equivalent to
        //   `attn > threshold`). Hot tokens are never skipped.
        //
        // Metal-only — fails to link on non-macOS targets. Callers gate via
        // [`KVCache::update_and_turbo4_delegated_attention`] which only
        // dispatches the kernel on macOS + Turbo4Delegated + power-of-2 D.
        type Turbo4DelegatedSteelOutputs;

        fn turbo4_delegated_steel_sdpa(
            scores: &MlxArray,
            cold_packed: &MlxArray,
            cold_rescale: &MlxArray,
            hot_v: &MlxArray,
            codebook: &MlxArray,
            dim: i32,
            n_rep: i32,
            cold_offset: i32,
            hot_offset: i32,
            threshold: f32,
        ) -> UniquePtr<Turbo4DelegatedSteelOutputs>;

        /// Take (move out) the unrotated cold weighted sum from a steel SDPA
        /// outputs struct. After this call the struct's `out_cold_pre` slot
        /// is empty; calling again returns a null UniquePtr.
        fn steel_outputs_take_cold(o: Pin<&mut Turbo4DelegatedSteelOutputs>)
            -> UniquePtr<MlxArray>;

        /// Take (move out) the hot weighted sum from a steel SDPA outputs
        /// struct. After this call the struct's `out_hot` slot is empty.
        fn steel_outputs_take_hot(o: Pin<&mut Turbo4DelegatedSteelOutputs>) -> UniquePtr<MlxArray>;

        // Native safetensors loading (MLX-managed mmap, lazy arrays).
        /// Opaque holder for weights loaded via MLX's native load_safetensors()
        type MlxLoadedWeights;

        /// Load safetensors file using MLX's native loader (lazy arrays, MLX-managed mmap).
        /// Returns Result so C++ exceptions (bad path, corrupt file) become recoverable errors.
        fn mlx_load_safetensors(path: &str) -> Result<UniquePtr<MlxLoadedWeights>>;

        /// Number of weight entries in the loaded weights
        fn loaded_weights_len(w: &MlxLoadedWeights) -> usize;

        /// Get the name of the i-th weight entry
        fn loaded_weights_name(w: &MlxLoadedWeights, index: usize) -> String;

        /// Take (move out) the i-th weight array, leaving the slot empty
        fn loaded_weights_take(w: Pin<&mut MlxLoadedWeights>, index: usize) -> UniquePtr<MlxArray>;
    }
}

// Re-export the FFI types and functions
pub use ffi::*;

// Re-export cxx::UniquePtr for consumers of this crate
pub use cxx::UniquePtr;
pub use ops::{concatenate, divide_scalar, multiply_scalar, stack, stack_owned, wht};

// Re-export sampling primitives needed by generation-loop wiring (B8) and server layers.
pub use sampling::TokenBiasMap;
// Re-export B9 observability counter accessors so the server `/metrics` handler
// can read process-wide lang-bias counters without a struct dependency.
// Includes the byte-fragment suppression counter added.
pub use sampling::{
    lang_bias_applied_total, lang_bias_byte_fragment_suppressions_total,
    lang_bias_tokens_suppressed_total,
};

// Re-export Axis B language-steering types so downstream consumers (CLI, server, B6–B8)
// can use them without referencing the internal module path.
pub use lang_analyzer::{
    ExceptionConfig, InclusionPolicy, LangAnalyzerError, LangBiasConfig, LangBiasSet, LanguageCode,
};

// Re-export the shared mlxcel cache-root resolver (`${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}`)
// so downstream crates (e.g. the `mlxcel` CLI downloader's global model store, issue #93)
// derive their on-disk locations from the exact same root and env-var semantics the
// tokenizer language-analysis disk cache already uses.
pub use lang_analyzer::cache_root;

/// Default the CUDA NVRTC PTX cache to a persistent, MLX-pin-scoped directory
/// under the mlxcel cache root, unless `MLX_PTX_CACHE_DIR` is already set.
///
/// MLX's own default places the JIT cache in the system temp dir
/// (`$TMPDIR/mlx/<version>/ptx`), which is cleared on reboot, so the first-run
/// kernel compilation is paid again every boot. A persistent location pays it
/// once per machine. The directory is scoped by the pinned MLX commit because
/// the cache is keyed only by kernel name and is not validated against the
/// kernel source, so entries must not survive an MLX upgrade. No-op on non-CUDA
/// builds, when `MLX_PTX_CACHE_DIR` is already set, and when the cache root
/// cannot be resolved. Call once at startup before the first inference.
pub fn ensure_persistent_ptx_cache() {
    if !cfg!(feature = "cuda") {
        return;
    }
    if std::env::var_os("MLX_PTX_CACHE_DIR").is_some() {
        return;
    }
    let Some(root) = cache_root() else {
        return;
    };
    let commit = env!("MLXCEL_MLX_COMMIT");
    let scope = &commit[..commit.len().min(12)];
    let dir = root.join("cuda-ptx").join(scope);
    if std::fs::create_dir_all(&dir).is_ok() {
        std::env::set_var("MLX_PTX_CACHE_DIR", &dir);
    }
}

fn use_single_query_maskless_path() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("MLXCEL_DISABLE_SINGLE_QUERY_MASKLESS")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
        )
    })
}

fn use_bool_causal_mask_path() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Explicit opt-in only: kept experimental until repeated A/B shows stable wins.
        matches!(
            std::env::var("MLXCEL_EXPERIMENTAL_BOOL_CAUSAL_MASK")
                .ok()
                .as_deref(),
            Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
        )
    })
}

/// Apply RoPE to a batched tensor using independent per-sequence offsets.
///
/// Slices the batch dimension and applies MLX's scalar-offset `fast_rope`
/// kernel to each `[1, n_heads, T, D]` slice independently, then
/// concatenates the results back together.  For `batch == 1` the slice /
/// concat overhead is skipped via an early return.
///
/// **Why there is no uniform-batch fast path**: when every offset in
/// `offsets` is the same value one might expect to call `fast_rope(x, ...,
/// offsets[0])` directly on the full `[B, n_heads, T, D]` tensor.  In
/// practice the attention key/value tensors arrive here after a
/// `transpose_axes([0,2,1,3])` from `[B, T, n_heads, D]`, which produces a
/// non-contiguous tensor whose batch stride equals the sequence stride when
/// `T == 1` (decode step).  MLX's `fast::rope` Metal kernel confuses these
/// equal strides and assigns position `offset + b` to batch-element `b`
/// instead of `offset` for all `b`, producing asymmetric K/V rotations and
/// causing divergence between batch rows even for identical input tokens.
/// The per-row path calls `fast_rope` with `B == 1`, where a single batch
/// element cannot encounter inter-batch confusion regardless of strides.
/// See `ffi_tests::test_fast_rope_batched_non_contiguous_t1_symmetric` and
/// the paged batched B=2 parity regression.
///
/// Used by: Llama3 batched decode, Qwen3 batched decode, Gemma3 batched decode, Llama4 batched decode
pub fn fast_rope_batched(
    x: &MlxArray,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offsets: &[i32],
) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(x);
    let batch = shape.first().copied().unwrap_or(0).max(0) as usize;
    assert_eq!(
        batch,
        offsets.len(),
        "fast_rope_batched expected {} offsets, got {}",
        batch,
        offsets.len()
    );

    if batch == 0 {
        return ffi::copy(x);
    }
    if batch == 1 {
        return ffi::fast_rope(x, dims, traditional, base, scale, offsets[0]);
    }

    // NOTE: The uniform-batch fast path (calling fast_rope on the full
    // [B, n_heads, T, D] tensor in one shot) is intentionally DISABLED.
    //
    // Root cause: in the batched decode path the Q/K/V tensors arrive here
    // already transposed from [B, T, n_heads, D] → [B, n_heads, T, D].
    // When T == 1 (single decode token), the transpose produces a
    // non-contiguous tensor whose strides satisfy
    //   stride[batch_dim] == stride[sequence_dim]   (= n_heads * D)
    // because swapping a size-1 dimension does not change the physical
    // memory layout.  MLX's fast::rope Metal kernel distinguishes
    // sequence position from batch position using the stride ratio; when
    // these two strides are equal the kernel confuses the batch offset for
    // an intra-sequence offset, assigning position `offset + b` to
    // batch-element `b` instead of `offset` for all `b`.  The result is
    // asymmetric K/V rotations across the batch even when every sequence
    // receives the same input token at the same cache position.
    //
    // The existing unit test (`test_fast_rope_batched_uniform_offsets_match_per_row_path`)
    // uses a directly-constructed contiguous [3, 1, 2, 4] tensor where all
    // strides are distinct, so the bug does not manifest there.
    //
    // The per-row path below slices to [1, n_heads, T, D] for each batch
    // element and calls fast_rope with B == 1.  With only one batch row the
    // kernel can never confuse a batch offset for a sequence offset, so the
    // non-contiguous strides are harmless.  See the paged batched B=2 parity
    // regression: row 1 diverged from the dense reference when the uniform
    // path was in effect.

    let rank = shape.len();
    let mut begin = vec![0; rank];
    let mut end = vec![i32::MAX; rank];
    end[0] = 1;

    let first = ffi::slice(x, &begin, &end);
    let mut result = ffi::fast_rope(&first, dims, traditional, base, scale, offsets[0]);

    for (batch_idx, &offset) in offsets.iter().enumerate().skip(1) {
        begin[0] = batch_idx as i32;
        end[0] = batch_idx as i32 + 1;
        let chunk = ffi::slice(x, &begin, &end);
        let chunk = ffi::fast_rope(&chunk, dims, traditional, base, scale, offset);
        result = crate::concatenate(&result, &chunk, 0);
    }

    result
}

/// Causal SDPA wrapper with transparent M5 Neural Accelerator routing.
///
/// On M5-class hardware (`has_neural_accelerator && macos_supports_na`), the
/// plain causal case preserves MLX's native `"causal"` mask mode so upstream
/// Metal/NAX SDPA can select the dedicated causal kernel. Sliding-window and
/// softcap variants still route through the shared attention dispatcher with an
/// explicit mask array. Other hardware keeps using MLX's native causal SDPA
/// entry point.
///
/// Sliding-window handling splits by query length: a single-query decode
/// (`q_len == 1`) slices K/V to the trailing `window_size` keys and uses the
/// clamped `(1, window_size)` mask, while a multi-token prefill that exceeds
/// the window (`q_len > 1 && k_len > window_size`) keeps all keys and builds a
/// full-width windowed-causal mask over them. Slicing K/V for a multi-token
/// prefill would strand the earliest query rows with an all-`-inf` row -> NaN
/// (issue #401/#408); the windowed correctness instead comes from the mask.
///
/// Used by: Llama, Qwen, Mixtral, Gemma, Cohere, Phi, OLMo, Exaone, GLM4,
/// MiniCPM, DeepSeek, Hunyuan, StarCoder2 and other causal prefill call sites
pub fn causal_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    softcap: f32,
    window_size: i32,
) -> UniquePtr<MlxArray> {
    let q_len = ffi::array_shape(q)[2];
    let k_len = ffi::array_shape(k)[2];

    // Decode single-query fast path:
    // - Causal masking is unnecessary when q_len == 1 (no future positions).
    // - Sliding-window masking is only needed if cached KV exceeds window.
    // This avoids per-token mask materialization for common decode cases.
    let needs_window_mask = window_size > 0 && k_len > window_size;
    if q_len == 1 && !needs_window_mask && use_single_query_maskless_path() {
        return layers::attention(q, k, v, scale, None, softcap, window_size);
    }

    if softcap > 0.0 || window_size > 0 {
        // A multi-token prefill that exceeds the window must NOT slice K/V to
        // the trailing `window_size` keys: a fresh `RotatingKVCache` (and a
        // dense `KVCache`) keep every prefill key, and a trailing slice strands
        // the earliest query rows (logical position `< k_len - window_size`)
        // with no visible key, producing an all-`-inf` softmax row -> NaN /
        // `<pad>` (issue #401/#408). For that case build a full-width
        // windowed-causal mask over ALL keys instead; windowed correctness
        // comes from the mask, mirroring mlx-lm's `RotatingKVCache`. The
        // single-query decode path (`q_len == 1`) keeps the trailing-window K/V
        // slice fast path, byte-for-byte unchanged.
        let over_window_prefill = needs_window_mask && q_len > 1;

        let (k_used, v_used, effective_k_len) = if needs_window_mask && !over_window_prefill {
            // Decode (`q_len == 1`) beyond the window: slice K/V to the last
            // `window_size` keys so they line up with the (1, window_size) mask
            // produced by `create_causal_mask_with_window`. Production callers
            // backed by `RotatingKVCache` already deliver K/V truncated to
            // `max_size = window_size`; this keeps the wrapper self-consistent
            // for any other single-query caller.
            let k_shape = ffi::array_shape(k);
            let v_shape = ffi::array_shape(v);
            let start = k_len - window_size;
            let k_sliced = ffi::slice(
                k,
                &[0, 0, start, 0],
                &[k_shape[0], k_shape[1], k_len, k_shape[3]],
            );
            let v_sliced = ffi::slice(
                v,
                &[0, 0, start, 0],
                &[v_shape[0], v_shape[1], k_len, v_shape[3]],
            );
            (Some(k_sliced), Some(v_sliced), window_size)
        } else {
            (None, None, k_len)
        };
        let k_ref: &MlxArray = k_used.as_deref().unwrap_or(k);
        let v_ref: &MlxArray = v_used.as_deref().unwrap_or(v);

        let offset = (effective_k_len - q_len).max(0);
        let mask = if over_window_prefill {
            // Full-width windowed-causal mask over all `k_len` keys (same
            // builder as the gemma3/gemma4 `sliding_prefill_mask`, issue #401).
            // Only reached for the previously-degenerate `q_len > 1`,
            // `k_len > window` case, so there is no decode bit-exactness
            // constraint here; the experimental bool-mask path is intentionally
            // bypassed (this case has no all-`-inf` rows left to reproduce).
            utils::create_causal_mask_with_window_full(q_len, offset, Some(window_size))
        } else if softcap == 0.0 && use_bool_causal_mask_path() {
            if window_size > 0 {
                utils::create_causal_bool_mask_with_window(q_len, offset, Some(window_size))
            } else {
                utils::create_causal_bool_mask(q_len, offset)
            }
        } else if window_size > 0 {
            utils::create_causal_mask_with_window(q_len, offset, Some(window_size))
        } else {
            utils::create_causal_mask(q_len, offset)
        };
        return layers::attention(
            q,
            k_ref,
            v_ref,
            scale,
            Some(mask.as_ref().unwrap()),
            softcap,
            window_size,
        );
    }

    let hw = hardware::get_hardware();
    if hw.has_neural_accelerator && hw.macos_supports_na {
        return layers::metal4_causal_attention(q, k, v, scale);
    }

    ffi::ffi_fast_scaled_dot_product_attention_causal(q, k, v, scale)
}

/// Causal SDPA wrapper with transparent M5 Neural Accelerator routing.
///
/// This is the zero-softcap, full-window shorthand used by existing model code.
pub fn fast_scaled_dot_product_attention_causal(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
) -> UniquePtr<MlxArray> {
    causal_attention(q, k, v, scale, 0.0, 0)
}

// High-level layer abstractions
pub mod layers;

// Cache state machines shared by attention families.
// Public so that `server` and `CachePool` consumers can import directly.
pub mod cache;

// Pure-Rust wrappers around frequently used FFI entry points.
mod ops;

// Common utility functions
pub mod utils;

// Weight loading utilities
pub mod weights;

// Token generation
pub mod generate;

// Decode-loop setup helpers shared by standard and speculative generation.
// Public so that the server batch scheduler can reuse seed/EOS/history helpers.
pub mod generation_policy;

// Shared sampling and token-penalty policy helpers.
// Public so that the server batch scheduler can perform step-level sampling.
pub mod sampling;

// Speculative decoding
pub mod speculative;

// Drafter trait + DrafterKind enum + model_type auto-detection. Foundational scaffolding for the Gemma 4 MTP and Qwen 3.5
// DFlash drafter ports. Concrete drafter impls land.
// The existing classic `SpeculativeGenerator` above is unchanged — MTP and
// DFlash are peer code paths, not replacements.
// TODO: wrap the existing SpeculativeGenerator in a
// Drafter-trait adapter so the round-loop drivers can dispatch uniformly.
pub mod drafter;

// Generation-time stream selection and installation wrappers.
// Public so that the server batch scheduler can install its own generation stream.
pub mod streams;

pub mod dtype;

// Runtime Apple Silicon generation detection.
// Public so that mlxcel (the main crate) can log hardware info at startup.
pub mod hardware;

// Typed wrappers around MLX's runtime memory accounting APIs (issue #55).
// Public so that the CLI generate path can surface post-load resident
// memory and the preflight (#56) can call `set_memory_limit` to fail fast.
pub mod memory;

// RoPE variants that are not exposed directly by `mlx::core::fast::rope`.
// Currently: proportional RoPE used by Gemma 4 full-attention layers.
pub mod rope_proportional;

// Unicode-script classifier and language-steering index for Axis B.
// Public so that mlxcel (the main crate) and downstream sub-issues (B3–B8)
// can consume it without further structural changes.
pub mod lang_analyzer;

// Crate-wide helpers for `#[cfg(test)]` paths. Provides the single shared
// `ENV_LOCK` that every env-mutating test in this crate must acquire; see `test_support::env_lock` for the rationale. `pub(crate)` so
// that test modules at any depth (e.g. `crate::lang_analyzer::cache::tests`)
// can name it as `crate::test_support::env_lock`.
#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
#[path = "ffi_tests.rs"]
mod ffi_tests;
