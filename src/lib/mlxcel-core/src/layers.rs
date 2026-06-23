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

//! High-level model layer implementations using mlx-cxx
//!
//! This module provides Rust wrappers for common neural network layers
//! using the mlx-cxx bindings for optimal performance.
//!
//! Cache state machines now live in `crate::cache`; this module re-exports
//! them so existing model code can keep importing cache types through
//! `mlxcel_core::layers`.

use crate::ffi;
use crate::ffi::MlxArray;
use cxx::UniquePtr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

pub use crate::cache::{ChunkedKVCache, KVCache, KVCacheMode, RotatingKVCache};

/// Quantized weight structure for 4-bit/8-bit quantized layers
/// Supports affine, mxfp4, nvfp4, and mxfp8 quantization modes.
/// For mxfp4/nvfp4/mxfp8 modes, biases is None.
pub struct QuantizedWeight {
    pub weight: UniquePtr<MlxArray>,
    pub scales: UniquePtr<MlxArray>,
    pub biases: Option<UniquePtr<MlxArray>>,
    pub group_size: i32,
    pub bits: i32,
    pub mode: String,
}

impl QuantizedWeight {
    /// Create a new quantized weight from raw components (affine mode)
    pub fn new(
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: UniquePtr<MlxArray>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self {
            weight,
            scales,
            biases: Some(biases),
            group_size,
            bits,
            mode: "affine".to_string(),
        }
    }

    /// Create a new quantized weight with explicit mode
    pub fn new_with_mode(
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: Option<UniquePtr<MlxArray>>,
        group_size: i32,
        bits: i32,
        mode: String,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
            mode,
        }
    }

    /// Get raw pointer to biases (null if not present, e.g. mxfp4/nvfp4/mxfp8)
    pub fn biases_ptr(&self) -> *const MlxArray {
        match &self.biases {
            Some(b) => b.as_ref().unwrap() as *const MlxArray,
            None => std::ptr::null(),
        }
    }

    /// Produce an independent handle that shares the same underlying MLX
    /// quantized-weight buffers.
    ///
    /// Used by speculative drafters that lazy-bind an untied target
    /// `lm_head` without copying the actual tensor payloads.
    pub fn clone_shared(&self) -> Self {
        Self {
            weight: ffi::copy(&self.weight),
            scales: ffi::copy(&self.scales),
            biases: self.biases.as_ref().map(|b| ffi::copy(b)),
            group_size: self.group_size,
            bits: self.bits,
            mode: self.mode.clone(),
        }
    }
}

// Embedding Layers.
/// Non-quantized embedding layer
pub struct Embedding {
    pub weight: UniquePtr<MlxArray>,
}

impl Embedding {
    /// Create a new embedding layer
    pub fn new(weight: UniquePtr<MlxArray>) -> Self {
        Self { weight }
    }

    /// Load from weight map
    pub fn from_weights(weights: &crate::weights::WeightMap, prefix: &str) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let weight = weights
            .get(&weight_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
        Ok(Self { weight })
    }

    /// Embedding lookup: indices -> embeddings
    pub fn forward(&self, indices: &MlxArray) -> UniquePtr<MlxArray> {
        ffi::embedding(&self.weight, indices)
    }

    /// Use embedding as linear projection (for tied embeddings/lm_head)
    /// y = x @ W (no transpose needed since embedding weight is [vocab, dim])
    pub fn as_linear(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let wt = ffi::transpose(&self.weight);
        ffi::matmul(x, &wt)
    }

    /// Produce an independent handle that shares the same underlying MLX
    /// weight buffer.
    ///
    /// `ffi::copy` creates a new lazy-array node pointing at the same data
    /// (no element copy), so this is cheap. Used by speculative drafters
    /// (DFlash) that lazy-bind the target's embedding table at `bind()`
    /// time instead of loading their own `embed_tokens.weight`.
    ///
    /// Used by: DFlash drafter lazy-bind path (`Drafter::bind`)
    pub fn clone_shared(&self) -> Self {
        Self {
            weight: ffi::copy(&self.weight),
        }
    }
}

/// Quantized embedding layer (4-bit/8-bit)
/// Supports affine, mxfp4, nvfp4, and mxfp8 quantization modes.
pub struct QuantizedEmbedding {
    pub weight: UniquePtr<MlxArray>,
    pub scales: UniquePtr<MlxArray>,
    pub biases: Option<UniquePtr<MlxArray>>,
    pub group_size: i32,
    pub bits: i32,
    pub mode: String,
}

impl QuantizedEmbedding {
    /// Create a new quantized embedding layer (affine mode)
    pub fn new(
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: UniquePtr<MlxArray>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self {
            weight,
            scales,
            biases: Some(biases),
            group_size,
            bits,
            mode: "affine".to_string(),
        }
    }

    /// Load from weight map
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        // Auto-detect the quantization mode: affine stores zero-point biases,
        // so their absence means a block-float scheme (mxfp4 / nvfp4 / mxfp8),
        // distinguished by bits and group_size. This lets callers that do not
        // thread an explicit mode (e.g. vision encoders, which always called
        // this affine default) load non-affine weights correctly instead of
        // aborting in quantized_matmul ("Biases must be provided for affine").
        let mode = if weights.get(&format!("{}.biases", prefix)).is_some() {
            "affine"
        } else if bits == 8 {
            "mxfp8"
        } else if group_size == 16 {
            "nvfp4"
        } else {
            "mxfp4"
        };
        Self::from_weights_with_mode(weights, prefix, group_size, bits, mode)
    }

    /// Load from weight map with explicit quantization mode
    pub fn from_weights_with_mode(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let scales_name = format!("{}.scales", prefix);
        let biases_name = format!("{}.biases", prefix);

        let weight = weights
            .get(&weight_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
        let scales = weights
            .get(&scales_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Scales not found: {}", scales_name))?;
        // biases may not exist for mxfp4/nvfp4/mxfp8 modes
        let biases = weights.get(&biases_name).map(|w| ffi::copy(w));

        // Reconcile caller-supplied bits with the actual tensor shapes, the same
        // way UnifiedLinear does. Mixed-precision exports quantize the embedding
        // at a different bit width than the model's top-level `config.quantization`
        // (e.g. diffusiongemma / gemma4 store embed_tokens, attention, and MLP at
        // 8-bit while the top-level default is 4-bit). Without this, the embedding
        // lookup, the tied-embedding lm_head, and dequantized_weight() all
        // dequantize with the wrong bits and abort. Uniform-quant checkpoints are
        // unaffected (inference returns the caller bits unchanged).
        let effective_bits = if mode == "affine" {
            let w_shape = ffi::array_shape(&weight);
            let s_shape = ffi::array_shape(&scales);
            infer_quantization_bits(&w_shape, &s_shape, group_size, bits)
                .map_err(|e| format!("{} (prefix: {})", e, prefix))?
        } else {
            bits
        };

        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits: effective_bits,
            mode: mode.to_string(),
        })
    }

    fn biases_ptr(&self) -> *const MlxArray {
        match &self.biases {
            Some(b) => b.as_ref().unwrap() as *const MlxArray,
            None => std::ptr::null(),
        }
    }

    /// Quantized embedding lookup with dequantization
    pub fn forward(&self, indices: &MlxArray) -> UniquePtr<MlxArray> {
        unsafe {
            ffi::quantized_embedding(
                &self.weight,
                &self.scales,
                self.biases_ptr(),
                indices,
                self.group_size,
                self.bits,
                &self.mode,
            )
        }
    }

    /// Use embedding as linear projection (for tied embeddings/lm_head)
    pub fn as_linear(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        unsafe {
            ffi::quantized_linear_forward(
                x,
                &self.weight,
                &self.scales,
                self.biases_ptr(),
                std::ptr::null(),
                self.group_size,
                self.bits,
                &self.mode,
            )
        }
    }

    /// Produce an independent handle that shares the same underlying MLX
    /// weight / scales / biases buffers (lazy-array share via `ffi::copy`,
    /// no element copy).
    ///
    /// Used by: DFlash drafter lazy-bind path (`Drafter::bind`)
    pub fn clone_shared(&self) -> Self {
        Self {
            weight: ffi::copy(&self.weight),
            scales: ffi::copy(&self.scales),
            biases: self.biases.as_ref().map(|b| ffi::copy(b)),
            group_size: self.group_size,
            bits: self.bits,
            mode: self.mode.clone(),
        }
    }
}

/// Unified embedding that auto-detects quantization
pub enum UnifiedEmbedding {
    Quantized(QuantizedEmbedding),
    Regular(Embedding),
}

impl UnifiedEmbedding {
    /// Load from weight map, auto-detecting quantization
    ///
    /// Detects quantization by checking for `.scales` key in weights.
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let scales_name = format!("{}.scales", prefix);

        if weights.contains_key(&scales_name) {
            // Quantized embedding
            Ok(Self::Quantized(QuantizedEmbedding::from_weights(
                weights, prefix, group_size, bits,
            )?))
        } else {
            // Regular embedding
            Ok(Self::Regular(Embedding::from_weights(weights, prefix)?))
        }
    }

    /// Embedding lookup
    pub fn forward(&self, indices: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized(e) => e.forward(indices),
            Self::Regular(e) => e.forward(indices),
        }
    }

    /// Use embedding as linear projection
    pub fn as_linear(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized(e) => e.as_linear(x),
            Self::Regular(e) => e.as_linear(x),
        }
    }

    /// Check if this is a quantized embedding
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }

    /// Produce an independent [`UnifiedEmbedding`] that shares the same
    /// underlying MLX buffers (lazy-array share, no element copy).
    ///
    /// This is the lazy-bind primitive used by speculative drafters whose
    /// checkpoint omits `embed_tokens.weight` and instead resolve it from
    /// the target at `bind()` time — most notably the upstream
    /// `z-lab/Qwen3.5-4B-DFlash` checkpoint (see
    /// `crate::drafter::dflash::DFlashDraftModel`). The drafter holds the
    /// returned handle for the lifetime of the speculative session; it
    /// stays valid independently of the target because MLX arrays are
    /// reference-counted.
    ///
    /// Used by: DFlash drafter lazy-bind path (`Drafter::bind`)
    pub fn clone_shared(&self) -> Self {
        match self {
            Self::Quantized(e) => Self::Quantized(e.clone_shared()),
            Self::Regular(e) => Self::Regular(e.clone_shared()),
        }
    }

    /// Return the raw weight tensor (the `[vocab_size, hidden_size]` matrix).
    ///
    /// For non-quantized embeddings this is the plain f32/bf16/f16 tensor.
    /// For quantized embeddings this is the packed quantized weight (not yet
    /// dequantized). Callers that need the dequantized weight for a matmul
    /// should use [`Self::as_linear`] instead; this accessor is provided for
    /// use-cases that need to pass the weight reference to an external op
    /// (e.g. [`crate::drafter::masked_embedder::MaskedEmbedder::forward`]).
    ///
    /// Used by: Gemma4AssistantDraftModel (centroid/MaskedEmbedder LM head path)
    pub fn weight(&self) -> &MlxArray {
        match self {
            Self::Quantized(e) => e.weight.as_ref().expect("non-null quantized weight"),
            Self::Regular(e) => e.weight.as_ref().expect("non-null embedding weight"),
        }
    }

    /// Return a float `[vocab_size, hidden_size]` weight usable as
    /// `probs @ weight`.
    ///
    /// For quantized embeddings this dequantizes the packed table (callers
    /// should do this once per generation call and reuse the result; for a
    /// 262144 x 2816 fp16 table the transient is roughly 1.4 GiB). For
    /// non-quantized embeddings this is a cheap lazy-array share of the
    /// existing tensor.
    ///
    /// Used by: DiffusionGemma self-conditioning soft embeddings (issue #217).
    /// The reference implementation measured `quantized_matmul(...,
    /// transpose=false)` several times slower at this shape, hence the
    /// dequantize-once approach.
    pub fn dequantized_weight(&self) -> UniquePtr<MlxArray> {
        match self {
            Self::Regular(e) => ffi::copy(&e.weight),
            Self::Quantized(e) => unsafe {
                ffi::dequantize(
                    &e.weight,
                    &e.scales,
                    e.biases_ptr(),
                    e.group_size,
                    e.bits,
                    &e.mode,
                )
            },
        }
    }
}

/// RMS Normalization layer
pub struct RMSNorm {
    pub weight: UniquePtr<MlxArray>,
    pub eps: f32,
}

impl RMSNorm {
    /// Create a new RMS norm layer
    pub fn new(weight: UniquePtr<MlxArray>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Forward pass using fast RMS norm kernel
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        ffi::fast_rms_norm(x, &self.weight, self.eps)
    }
}

/// Gemma-style RMS Normalization layer with (1 + weight) pattern
/// Gemma-style RMS normalization with (1 + weight) adjustment.
/// Used by: Gemma, Gemma2, Gemma3, Gemma3n
pub struct GemmaRMSNorm {
    pub weight: UniquePtr<MlxArray>,
    /// Pre-computed (1 + weight) to avoid per-forward allocation
    adjusted_weight: UniquePtr<MlxArray>,
    pub eps: f32,
}

impl GemmaRMSNorm {
    /// Create a new Gemma RMS norm layer.
    /// Pre-computes (1 + weight) once at construction time.
    pub fn new(weight: UniquePtr<MlxArray>, eps: f32) -> Self {
        let ones = ffi::ones(&[ffi::array_shape(&weight)[0]], ffi::array_dtype(&weight));
        let adjusted_weight = ffi::add(&ones, &weight);
        Self {
            weight,
            adjusted_weight,
            eps,
        }
    }

    /// Forward pass using fast RMS norm kernel with pre-computed (1 + weight)
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        ffi::fast_rms_norm(x, &self.adjusted_weight, self.eps)
    }

    /// Pre-adjusted `(1 + weight)` tensor used by fused Gemma attention paths.
    pub fn adjusted_weight(&self) -> &MlxArray {
        &self.adjusted_weight
    }
}

/// Per-element RMS scale + epsilon that the fused QKV kernel feeds straight to
/// `mlx::core::fast::rms_norm`, which computes `x * weight * rsqrt(mean(x^2) + eps)`.
///
/// The kernel only ever applies that plain formula. The `(1 + weight)` offset
/// that distinguishes Gemma from a standard RMSNorm lives in the weight each
/// impl hands back, never in the kernel:
///
/// - [`RMSNorm`] returns its raw `weight`, realizing `x * weight * rsqrt(...)`.
/// - [`GemmaRMSNorm`] returns its pre-computed `(1 + weight)`, realizing
///   `x * (1 + weight) * rsqrt(...)`.
///
/// This lets one fused primitive serve both norm variants while keeping the
/// `(1 + weight)` semantics selected by type, not by a branch inside the kernel.
///
/// Used by: FusedQKVLinear::forward_split_norm_rope_quantized
/// (Gemma3, Qwen3, Qwen3-MoE).
pub trait FusedQkNorm {
    /// Effective per-element scale fed to `fast_rms_norm`: the raw weight for a
    /// standard RMSNorm, `(1 + weight)` for Gemma.
    fn fused_norm_weight(&self) -> &MlxArray;
    /// RMS normalization epsilon.
    fn rms_eps(&self) -> f32;
}

impl FusedQkNorm for RMSNorm {
    fn fused_norm_weight(&self) -> &MlxArray {
        &self.weight
    }
    fn rms_eps(&self) -> f32 {
        self.eps
    }
}

impl FusedQkNorm for GemmaRMSNorm {
    fn fused_norm_weight(&self) -> &MlxArray {
        self.adjusted_weight()
    }
    fn rms_eps(&self) -> f32 {
        self.eps
    }
}

/// Whether the fused single-token decode QK-norm+RoPE kernel (#326) is enabled.
///
/// Default-OFF (opt-in). The primitive matches the graph path within RMS < 5e-3
/// (the reduction is over the transpose-invariant head_dim axis), but it cuts
/// Rust<->C++ FFI crossings rather than MLX op count, so it does not speed up the
/// GPU/bandwidth-bound decode loop. On M1 Ultra (fast FFI) it measured ~1-3.4%
/// SLOWER than the graph path (qwen3-0.6b 275 vs 284, qwen3-8b 82.3 vs 83.2
/// tok/s). CUDA was the open per-backend question; on GB10 (SM 12.1) it is also
/// slower (qwen3-0.6b 0.96x, qwen3-8b ~1.0x, qwen3-30b-a3b 0.92x fused/graph;
/// see docs/benchmark_results/fused-qk-norm-decode-gb10.md). It ships as a
/// reusable shared primitive for the deferred QK-norm families and stays gated
/// off, mirroring the opt-in treatment of the neutral fused-relu2 MoE path
/// (`MLXCEL_FUSED_MOE_RELU2`).
///
/// Greedy temp-0 is not byte-identical to the graph path over long generation:
/// the two numerical paths can diverge at a near-tie argmax. This is not a
/// regression. On CUDA the graph path is itself non-deterministic run-to-run
/// (GPU FP-reduction order), while the fused path is deterministic, so its
/// output stays inside the graph baseline's own run-to-run envelope.
///
/// Set `MLXCEL_FUSED_QK_NORM=1` (also `true`/`on`/`yes`, case-insensitive,
/// trimmed) to opt into the fused path in Qwen3 and Qwen3-MoE decode.
///
/// The variable is read once at first call and cached for the process lifetime.
/// Reading it per decode step would add per-token overhead to the hot path.
pub fn fused_qk_norm_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        fused_qk_norm_enabled_from(std::env::var("MLXCEL_FUSED_QK_NORM").ok().as_deref())
    })
}

/// Pure decision behind [`fused_qk_norm_enabled`], split out for unit testing
/// without touching process-global env state. `None` (unset) is off; an explicit
/// `1`/`true`/`on`/`yes` (case-insensitive, trimmed) is on; any other value is off.
fn fused_qk_norm_enabled_from(value: Option<&str>) -> bool {
    match value {
        Some(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        ),
        None => false,
    }
}

/// Layer Normalization layer (standard LayerNorm with weight and optional bias)
pub struct LayerNorm {
    pub weight: UniquePtr<MlxArray>,
    pub bias: Option<UniquePtr<MlxArray>>,
    pub eps: f32,
}

impl LayerNorm {
    /// Create a new layer norm
    pub fn new(weight: UniquePtr<MlxArray>, bias: Option<UniquePtr<MlxArray>>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Forward pass using fast layer norm kernel
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let weight_ptr = self.weight.as_ref().unwrap() as *const MlxArray;
        let bias_ptr = self
            .bias
            .as_ref()
            .map(|b| b.as_ref().unwrap() as *const MlxArray)
            .unwrap_or(std::ptr::null());

        unsafe { ffi::fast_layer_norm(x, weight_ptr, bias_ptr, self.eps) }
    }
}

/// Optional LoRA weights for runtime on-the-fly application.
/// Used by: Phi4MM VLM (vision LoRA active during prefill only)
pub struct LoRAWeights {
    /// LoRA A weight: [rank, in_features]
    pub a: UniquePtr<MlxArray>,
    /// LoRA B weight: [out_features, rank]
    pub b: UniquePtr<MlxArray>,
    pub scale: f32,
    /// Whether LoRA is currently active. Uses Cell for interior mutability
    /// so that after_prefill(&self) can toggle it without &mut self.
    active: std::cell::Cell<bool>,
}

/// Regular (non-quantized) Linear layer
pub struct Linear {
    pub weight: UniquePtr<MlxArray>,
    pub bias: Option<UniquePtr<MlxArray>>,
    /// Optional runtime LoRA (applied on-the-fly, not fused into weight)
    lora: Option<LoRAWeights>,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(weight: UniquePtr<MlxArray>, bias: Option<UniquePtr<MlxArray>>) -> Self {
        Self {
            weight,
            bias,
            lora: None,
        }
    }

    /// Load from weight map
    pub fn from_weights(weights: &crate::weights::WeightMap, prefix: &str) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);

        let weight = weights
            .get(&weight_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;

        // Check for optional bias
        let bias_name = format!("{}.bias", prefix);
        let bias = weights.get(&bias_name).map(|w| ffi::copy(w));

        Ok(Self {
            weight,
            bias,
            lora: None,
        })
    }

    /// Set runtime LoRA weights. Starts active.
    pub fn set_lora(&mut self, a: UniquePtr<MlxArray>, b: UniquePtr<MlxArray>, scale: f32) {
        self.lora = Some(LoRAWeights {
            a,
            b,
            scale,
            active: std::cell::Cell::new(true),
        });
    }

    /// Toggle LoRA active state (no-op if no LoRA weights set)
    pub fn set_lora_active(&self, active: bool) {
        if let Some(ref lora) = self.lora {
            lora.active.set(active);
        }
    }

    /// Forward pass: y = x @ W.T + bias [+ LoRA if active]
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Transpose weight from [out, in] to [in, out]
        let wt = ffi::transpose(&self.weight);
        let mut result = ffi::matmul(x, &wt);

        // Apply runtime LoRA: result += (x @ A.T) @ B.T * scale
        if let Some(ref lora) = self.lora {
            if lora.active.get() {
                let at = ffi::transpose(&lora.a);
                let bt = ffi::transpose(&lora.b);
                let lora_out = ffi::matmul(&ffi::matmul(x, &at), &bt);
                let scaled = crate::multiply_scalar(&lora_out, lora.scale);
                result = ffi::add(&result, &scaled);
            }
        }

        match &self.bias {
            Some(b) => ffi::add(&result, b),
            None => result,
        }
    }

    /// Produce an independent handle that shares the same underlying MLX
    /// weight/bias buffers.
    ///
    /// Runtime LoRA state is intentionally not carried over: this helper is
    /// for binding target projection heads into speculative drafters after
    /// load-time adapter fusion/sanitization has already happened.
    pub fn clone_shared(&self) -> Self {
        Self {
            weight: ffi::copy(&self.weight),
            bias: self.bias.as_ref().map(|b| ffi::copy(b)),
            lora: None,
        }
    }
}

/// Infer actual per-tensor quantization bits from weight and scales shapes.
///
/// MLX stores affine-quantized linears as:
/// - `weight`: u32 packed, shape `(..., out_features, packed_in_features)`
/// - `scales`: float, shape `(..., out_features, num_groups)`
///
/// with the invariant `packed_in_features * 32 == bits * num_groups * group_size`.
///
/// When `caller_bits` is consistent with the observed shapes, it is returned
/// unchanged. When it is not (e.g. per-layer override where a gate is stored
/// at 8-bit while the rest of the model is 4-bit), this infers the actual bits
/// by trusting `group_size` and solving the invariant.
///
/// This mirrors the upstream `nn.quantize(class_predicate=...)` pattern used
/// by mlx-lm/mlx-vlm, which emits per-path bit overrides in `config.quantization`
/// (e.g. Qwen3.5/3.6 MoE router gates).
///
/// Returns an error only when the inferred bits are not a valid MLX bit width
/// `{2, 3, 4, 5, 6, 8}` — in that case `group_size` itself is likely wrong.
fn infer_quantization_bits(
    weight_shape: &[i32],
    scales_shape: &[i32],
    group_size: i32,
    caller_bits: i32,
) -> Result<i32, String> {
    if weight_shape.is_empty() || scales_shape.is_empty() || group_size <= 0 {
        return Ok(caller_bits);
    }
    let packed_in = *weight_shape.last().unwrap();
    let num_groups = *scales_shape.last().unwrap();
    if packed_in <= 0 || num_groups <= 0 {
        return Ok(caller_bits);
    }

    let numerator = packed_in.checked_mul(32).ok_or_else(|| {
        format!(
            "Quantized weight shape overflow: packed_in={}, weight.shape={:?}",
            packed_in, weight_shape
        )
    })?;
    let denominator = num_groups.checked_mul(group_size).ok_or_else(|| {
        format!(
            "Quantized scales shape overflow: num_groups={}, group_size={}",
            num_groups, group_size
        )
    })?;
    if denominator == 0 || numerator % denominator != 0 {
        return Err(format!(
            "Quantized weight shape inconsistency: weight.shape={:?}, scales.shape={:?}, \
             group_size={}: cannot derive integer bits",
            weight_shape, scales_shape, group_size
        ));
    }
    let inferred_bits = numerator / denominator;
    if inferred_bits == caller_bits {
        return Ok(caller_bits);
    }
    if ![2, 3, 4, 5, 6, 8].contains(&inferred_bits) {
        return Err(format!(
            "Quantized weight shape inconsistency: inferred bits={} not in {{2,3,4,5,6,8}}; \
             weight.shape={:?}, scales.shape={:?}, group_size={}, caller_bits={}",
            inferred_bits, weight_shape, scales_shape, group_size, caller_bits
        ));
    }
    Ok(inferred_bits)
}

/// Unified Linear layer that auto-detects quantization
///
/// Checks for `.scales` key in weight map to determine whether to use
/// quantized or regular linear operations. Replaces both the old
/// `QuantizedLinear` and `UnifiedLinear` types.
///
/// Used by: all text/VLM model implementations
pub enum UnifiedLinear {
    Quantized {
        weight: QuantizedWeight,
        bias: Option<UniquePtr<MlxArray>>,
    },
    Regular(Linear),
}

/// Backward-compatible alias
pub type QuantizedLinear = UnifiedLinear;

impl UnifiedLinear {
    /// Create a new quantized linear layer (explicit construction)
    pub fn new(weight: QuantizedWeight, bias: Option<UniquePtr<MlxArray>>) -> Self {
        Self::Quantized { weight, bias }
    }

    /// Load from weight map, auto-detecting quantization (affine mode)
    ///
    /// Detects quantization by checking for `.scales` key in weights.
    /// Falls back to regular Linear if scales are absent.
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        // Auto-detect the quantization mode: affine stores zero-point biases,
        // so their absence means a block-float scheme (mxfp4 / nvfp4 / mxfp8),
        // distinguished by bits and group_size. This lets callers that do not
        // thread an explicit mode (e.g. vision encoders, which always called
        // this affine default) load non-affine weights correctly instead of
        // aborting in quantized_matmul ("Biases must be provided for affine").
        let mode = if weights.get(&format!("{}.biases", prefix)).is_some() {
            "affine"
        } else if bits == 8 {
            "mxfp8"
        } else if group_size == 16 {
            "nvfp4"
        } else {
            "mxfp4"
        };
        Self::from_weights_with_mode(weights, prefix, group_size, bits, mode)
    }

    /// Load from weight map with explicit quantization mode
    ///
    /// Detects quantization by checking for `.scales` key in weights.
    /// Falls back to regular Linear if scales are absent.
    /// For mxfp4/nvfp4/mxfp8 modes, biases are optional.
    pub fn from_weights_with_mode(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Result<Self, String> {
        let scales_name = format!("{}.scales", prefix);

        if weights.contains_key(&scales_name) {
            // Quantized path
            let weight_name = format!("{}.weight", prefix);
            let biases_name = format!("{}.biases", prefix);

            let weight = weights
                .get(&weight_name)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
            let scales = weights
                .get(&scales_name)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Scales not found: {}", scales_name))?;
            // biases may not exist for mxfp4/nvfp4/mxfp8 modes
            let biases = weights.get(&biases_name).map(|w| ffi::copy(w));

            // Reconcile caller-supplied quant params with the actual tensor
            // shapes; mixed exports vary them per layer. For affine, bits are
            // inferred with the trusted group_size (Qwen3.5/3.6 MoE gates). For
            // the block-float modes bits are fixed by the mode, so reconcile
            // group_size from the shapes instead: in_features = packed_in *
            // (32/bits) = num_groups * group_size (e.g. minicpm-v-4.6 mxfp4
            // stores some weights at group_size 32 while config says 64).
            let (effective_bits, effective_group_size) = if mode == "affine" {
                let w_shape = ffi::array_shape(&weight);
                let s_shape = ffi::array_shape(&scales);
                let eb = infer_quantization_bits(&w_shape, &s_shape, group_size, bits)
                    .map_err(|e| format!("{} (prefix: {})", e, prefix))?;
                (eb, group_size)
            } else {
                let w_shape = ffi::array_shape(&weight);
                let s_shape = ffi::array_shape(&scales);
                let egs = if w_shape.len() >= 2 && s_shape.len() >= 2 && bits > 0 {
                    let packed_in = *w_shape.last().unwrap();
                    let num_groups = *s_shape.last().unwrap();
                    let in_features = packed_in * (32 / bits);
                    if num_groups > 0 && in_features % num_groups == 0 {
                        in_features / num_groups
                    } else {
                        group_size
                    }
                } else {
                    group_size
                };
                (bits, egs)
            };

            let qweight = QuantizedWeight {
                weight,
                scales,
                biases,
                group_size: effective_group_size,
                bits: effective_bits,
                mode: mode.to_string(),
            };

            let bias_name = format!("{}.bias", prefix);
            let bias = weights.get(&bias_name).map(|w| ffi::copy(w));

            Ok(Self::Quantized {
                weight: qweight,
                bias,
            })
        } else {
            // Fallback to regular linear (non-quantized model)
            Ok(Self::Regular(Linear::from_weights(weights, prefix)?))
        }
    }

    pub fn as_quantized_weight(&self) -> Option<&QuantizedWeight> {
        match self {
            UnifiedLinear::Quantized { weight, .. } => Some(weight),
            UnifiedLinear::Regular(_) => None,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized { weight, bias } => {
                let bias_ptr = bias
                    .as_ref()
                    .map(|b| b.as_ref().unwrap() as *const MlxArray)
                    .unwrap_or(std::ptr::null());

                unsafe {
                    ffi::quantized_linear_forward(
                        x,
                        &weight.weight,
                        &weight.scales,
                        weight.biases_ptr(),
                        bias_ptr,
                        weight.group_size,
                        weight.bits,
                        &weight.mode,
                    )
                }
            }
            Self::Regular(linear) => linear.forward(x),
        }
    }

    /// Set runtime LoRA weights (only for Regular variant)
    pub fn set_lora(&mut self, a: UniquePtr<MlxArray>, b: UniquePtr<MlxArray>, scale: f32) {
        if let Self::Regular(linear) = self {
            linear.set_lora(a, b, scale);
        }
    }

    /// Toggle LoRA active state
    pub fn set_lora_active(&self, active: bool) {
        if let Self::Regular(linear) = self {
            linear.set_lora_active(active);
        }
    }

    /// Check if this is a quantized linear layer
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }

    /// Get a reference to the inner QuantizedWeight (if quantized)
    /// Used by compiled MLP operations that need direct access to weight/scales/biases
    pub fn quantized_weight(&self) -> Option<&QuantizedWeight> {
        match self {
            Self::Quantized { weight, .. } => Some(weight),
            Self::Regular(_) => None,
        }
    }

    /// Fused concatenated QKV projection + split + reshape + transpose + SuScaledRoPE.
    ///
    /// Returns Q/K/V already shaped `[B, H, T, D]`. Q/K match mlx-lm/mlx-vlm
    /// SuScaledRoPE semantics by scaling only the rotary prefix before custom
    /// frequency RoPE. Only available for quantized fused-QKV weights.
    ///
    /// Used by: Phi3/Phi3V longrope-su attention path.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_fused_qkv_split_su_scaled_rope(
        &self,
        x: &MlxArray,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_dims: i32,
        rope_freqs: &MlxArray,
        rope_input_scale: f32,
        cache_offset: i32,
    ) -> Option<(
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    )> {
        match self {
            Self::Quantized { weight, .. } => {
                let mut q = cxx::UniquePtr::null();
                let mut k = cxx::UniquePtr::null();
                let mut v = cxx::UniquePtr::null();
                unsafe {
                    ffi::fused_qkv_project_split_su_scaled_rope(
                        x,
                        &weight.weight,
                        &weight.scales,
                        weight.biases_ptr(),
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        rope_dims,
                        rope_freqs,
                        rope_input_scale,
                        cache_offset,
                        weight.group_size,
                        weight.bits,
                        &weight.mode,
                        &mut q,
                        &mut k,
                        &mut v,
                    );
                }
                Some((q, k, v))
            }
            Self::Regular(_) => None,
        }
    }

    /// Get a reference to the inner Linear (if non-quantized)
    /// Used by compiled FP MLP operations that need direct weight/bias access
    pub fn regular_weight(&self) -> Option<&Linear> {
        match self {
            Self::Regular(linear) => Some(linear),
            Self::Quantized { .. } => None,
        }
    }

    /// Produce an independent handle that shares the same underlying MLX
    /// buffers.
    ///
    /// Used by: DFlash lazy binding for untied target `lm_head` projections.
    pub fn clone_shared(&self) -> Self {
        match self {
            Self::Quantized { weight, bias } => Self::Quantized {
                weight: weight.clone_shared(),
                bias: bias.as_ref().map(|b| ffi::copy(b)),
            },
            Self::Regular(linear) => Self::Regular(linear.clone_shared()),
        }
    }
}

/// Fused QKV linear layer for GQA models.
///
/// Stores Q, K, V weights concatenated along the output dimension into a single
/// `UnifiedLinear`. A single matmul replaces 3 separate projections, improving
/// Neural Engine tile utilisation (especially on M5).
///
/// Weight layout (output axis 0):
///   `[q_dim | k_dim | v_dim, hidden_dim]`  →  `q_dim = n_heads * head_dim`
///                                           →  `k_dim = v_dim = n_kv_heads * head_dim`
///
/// Used by: Llama3, Qwen2/3, Qwen3-VL, Qwen3-VL-MoE, Gemma v1/2/3, Mistral,
/// Cohere2, StarCoder2, InternLM3, Jamba
pub struct FusedQKVLinear {
    /// Single concatenated QKV projection weight.
    pub qkv_proj: UnifiedLinear,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
}

impl FusedQKVLinear {
    /// Load and concatenate separate q/k/v weights from the weight map.
    ///
    /// Concatenates `{prefix}.q_proj`, `{prefix}.k_proj`, `{prefix}.v_proj`
    /// along axis 0 into a single `UnifiedLinear`.  Both quantized and
    /// non-quantized weight layouts are supported.
    pub fn from_weights_separate(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
    ) -> Result<Self, String> {
        Self::from_weights_separate_with_mode(
            weights, prefix, group_size, bits, n_heads, n_kv_heads, head_dim, "affine",
        )
    }

    /// Load and concatenate separate q/k/v weights with explicit quantization mode.
    ///
    /// Used by: Llama3, Qwen2/Qwen2.5, Phi3-style fused attention wrappers.
    /// Preserves both quantization `biases` and true linear `bias` tensors
    /// when present; Qwen2-family checkpoints require q/k/v linear bias for
    /// sane logits.
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights_separate_with_mode(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        mode: &str,
    ) -> Result<Self, String> {
        let q_prefix = format!("{}.q_proj", prefix);
        let k_prefix = format!("{}.k_proj", prefix);
        let v_prefix = format!("{}.v_proj", prefix);

        let q_scales_key = format!("{}.scales", q_prefix);
        let is_quantized = weights.contains_key(&q_scales_key);

        let qkv_proj = if is_quantized {
            // Quantized path: concatenate weight, scales, and biases tensors
            // along axis 0 (output dimension).
            let q_w = weights
                .get(&format!("{}.weight", q_prefix))
                .ok_or_else(|| format!("Weight not found: {}.weight", q_prefix))?;
            let k_w = weights
                .get(&format!("{}.weight", k_prefix))
                .ok_or_else(|| format!("Weight not found: {}.weight", k_prefix))?;
            let v_w = weights
                .get(&format!("{}.weight", v_prefix))
                .ok_or_else(|| format!("Weight not found: {}.weight", v_prefix))?;

            let q_s = weights
                .get(&format!("{}.scales", q_prefix))
                .ok_or_else(|| format!("Scales not found: {}.scales", q_prefix))?;
            let k_s = weights
                .get(&format!("{}.scales", k_prefix))
                .ok_or_else(|| format!("Scales not found: {}.scales", k_prefix))?;
            let v_s = weights
                .get(&format!("{}.scales", v_prefix))
                .ok_or_else(|| format!("Scales not found: {}.scales", v_prefix))?;

            // Reconcile caller-supplied bits with actual tensor shapes (affine only).
            // Fused QKV concatenates along axis 0, so q/k/v must share bits; infer from q.
            let effective_bits = if mode == "affine" {
                let w_shape = ffi::array_shape(q_w);
                let s_shape = ffi::array_shape(q_s);
                infer_quantization_bits(&w_shape, &s_shape, group_size, bits)
                    .map_err(|e| format!("{} (prefix: {})", e, q_prefix))?
            } else {
                bits
            };

            // Concatenate along axis 0 (output dimension)
            let qkv_weight = {
                let ptrs: &[*const MlxArray] = &[
                    q_w.as_ref().unwrap() as *const MlxArray,
                    k_w.as_ref().unwrap() as *const MlxArray,
                    v_w.as_ref().unwrap() as *const MlxArray,
                ];
                unsafe { ffi::concatenate(ptrs, 0) }
            };
            let qkv_scales = {
                let ptrs: &[*const MlxArray] = &[
                    q_s.as_ref().unwrap() as *const MlxArray,
                    k_s.as_ref().unwrap() as *const MlxArray,
                    v_s.as_ref().unwrap() as *const MlxArray,
                ];
                unsafe { ffi::concatenate(ptrs, 0) }
            };

            // Biases are optional (absent for mxfp4/nvfp4/mxfp8)
            let qkv_biases = {
                let q_b = weights.get(&format!("{}.biases", q_prefix));
                let k_b = weights.get(&format!("{}.biases", k_prefix));
                let v_b = weights.get(&format!("{}.biases", v_prefix));
                match (q_b, k_b, v_b) {
                    (Some(qb), Some(kb), Some(vb)) => {
                        let ptrs: &[*const MlxArray] = &[
                            qb.as_ref().unwrap() as *const MlxArray,
                            kb.as_ref().unwrap() as *const MlxArray,
                            vb.as_ref().unwrap() as *const MlxArray,
                        ];
                        Some(unsafe { ffi::concatenate(ptrs, 0) })
                    }
                    _ => None,
                }
            };

            let qkv_bias = {
                let q_b = weights.get(&format!("{}.bias", q_prefix));
                let k_b = weights.get(&format!("{}.bias", k_prefix));
                let v_b = weights.get(&format!("{}.bias", v_prefix));
                match (q_b, k_b, v_b) {
                    (Some(qb), Some(kb), Some(vb)) => {
                        let ptrs: &[*const MlxArray] = &[
                            qb.as_ref().unwrap() as *const MlxArray,
                            kb.as_ref().unwrap() as *const MlxArray,
                            vb.as_ref().unwrap() as *const MlxArray,
                        ];
                        Some(unsafe { ffi::concatenate(ptrs, 0) })
                    }
                    _ => None,
                }
            };

            let qweight = QuantizedWeight {
                weight: qkv_weight,
                scales: qkv_scales,
                biases: qkv_biases,
                group_size,
                bits: effective_bits,
                mode: mode.to_string(),
            };
            UnifiedLinear::Quantized {
                weight: qweight,
                bias: qkv_bias,
            }
        } else {
            // Non-quantized path: concatenate weight tensors along axis 0.
            let q_w = weights
                .get(&format!("{}.weight", q_prefix))
                .ok_or_else(|| format!("Weight not found: {}.weight", q_prefix))?;
            let k_w = weights
                .get(&format!("{}.weight", k_prefix))
                .ok_or_else(|| format!("Weight not found: {}.weight", k_prefix))?;
            let v_w = weights
                .get(&format!("{}.weight", v_prefix))
                .ok_or_else(|| format!("Weight not found: {}.weight", v_prefix))?;

            let qkv_weight = {
                let ptrs: &[*const MlxArray] = &[
                    q_w.as_ref().unwrap() as *const MlxArray,
                    k_w.as_ref().unwrap() as *const MlxArray,
                    v_w.as_ref().unwrap() as *const MlxArray,
                ];
                unsafe { ffi::concatenate(ptrs, 0) }
            };

            // Optional bias per projection (rare, but handle it)
            let q_bias = weights
                .get(&format!("{}.bias", q_prefix))
                .map(|b| ffi::copy(b));
            let k_bias = weights
                .get(&format!("{}.bias", k_prefix))
                .map(|b| ffi::copy(b));
            let v_bias = weights
                .get(&format!("{}.bias", v_prefix))
                .map(|b| ffi::copy(b));
            let bias = match (q_bias, k_bias, v_bias) {
                (Some(qb), Some(kb), Some(vb)) => {
                    let ptrs: &[*const MlxArray] = &[
                        qb.as_ref().unwrap() as *const MlxArray,
                        kb.as_ref().unwrap() as *const MlxArray,
                        vb.as_ref().unwrap() as *const MlxArray,
                    ];
                    Some(unsafe { ffi::concatenate(ptrs, 0) })
                }
                _ => None,
            };

            UnifiedLinear::Regular(Linear::new(qkv_weight, bias))
        };

        Ok(Self {
            qkv_proj,
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }

    /// Load from a pre-concatenated `qkv_proj` weight (e.g., Phi3 layout).
    pub fn from_weights_fused(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
    ) -> Result<Self, String> {
        let qkv_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.qkv_proj", prefix),
            group_size,
            bits,
        )?;
        Ok(Self {
            qkv_proj,
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }

    /// Fused QKV projection + split.
    ///
    /// Returns `(q, k, v)` each shaped `[batch, seq_len, proj_dim]` (pre-reshape).
    /// The caller is responsible for reshape/transpose/RoPE.
    pub fn forward(
        &self,
        x: &MlxArray,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let qkv = self.qkv_proj.forward(x);

        let q_size = self.n_heads * self.head_dim;
        let kv_size = self.n_kv_heads * self.head_dim;

        let q = ffi::slice_last_dim(&qkv, 0, q_size);
        let k = ffi::slice_last_dim(&qkv, q_size, q_size + kv_size);
        let v = ffi::slice_last_dim(&qkv, q_size + kv_size, q_size + 2 * kv_size);

        (q, k, v)
    }

    /// Fused concatenated QKV projection + split + reshape + transpose + RoPE.
    ///
    /// Returns Q/K/V already shaped `[B, H, T, D]`. Q/K have RoPE applied.
    /// Only available for quantized fused-QKV weights.
    ///
    /// Used by: Llama3-family fused attention path.
    pub fn forward_split_rope(
        &self,
        x: &MlxArray,
        rope_dims: i32,
        rope_base: f32,
        cache_offset: i32,
    ) -> Option<(
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    )> {
        if std::env::var("MLXCEL_ENABLE_FUSED_QKV_SPLIT_ROPE").is_err() {
            return None;
        }
        match &self.qkv_proj {
            UnifiedLinear::Quantized { weight, .. } => {
                let mut q = cxx::UniquePtr::null();
                let mut k = cxx::UniquePtr::null();
                let mut v = cxx::UniquePtr::null();
                unsafe {
                    ffi::fused_qkv_project_split_rope(
                        x,
                        &weight.weight,
                        &weight.scales,
                        weight.biases_ptr(),
                        self.n_heads,
                        self.n_kv_heads,
                        self.head_dim,
                        rope_dims,
                        rope_base,
                        cache_offset,
                        weight.group_size,
                        weight.bits,
                        &weight.mode,
                        &mut q,
                        &mut k,
                        &mut v,
                    );
                }
                Some((q, k, v))
            }
            UnifiedLinear::Regular(_) => None,
        }
    }

    /// Fused concatenated QKV projection + split + reshape + transpose + RoPE.
    ///
    /// Returns Q/K/V already shaped `[B, H, T, D]`. Q/K have RoPE applied.
    /// Only available for quantized fused-QKV weights.
    ///
    /// Used by: Gemma2 dense attention path.
    pub fn forward_split_rope_quantized(
        &self,
        x: &MlxArray,
        rope_dims: i32,
        rope_base: f32,
        cache_offset: i32,
    ) -> Option<(
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    )> {
        match &self.qkv_proj {
            UnifiedLinear::Quantized { weight, .. } => {
                let mut q = cxx::UniquePtr::null();
                let mut k = cxx::UniquePtr::null();
                let mut v = cxx::UniquePtr::null();
                unsafe {
                    ffi::fused_qkv_project_split_rope(
                        x,
                        &weight.weight,
                        &weight.scales,
                        weight.biases_ptr(),
                        self.n_heads,
                        self.n_kv_heads,
                        self.head_dim,
                        rope_dims,
                        rope_base,
                        cache_offset,
                        weight.group_size,
                        weight.bits,
                        &weight.mode,
                        &mut q,
                        &mut k,
                        &mut v,
                    );
                }
                Some((q, k, v))
            }
            UnifiedLinear::Regular(_) => None,
        }
    }

    /// Fused concatenated QKV projection + split + reshape + transpose +
    /// RMSNorm(Q/K) + RoPE.
    ///
    /// Returns Q/K/V already shaped `[B, H, T, D]`. Q/K are normalized and
    /// have RoPE applied. Only available for quantized fused-QKV weights;
    /// returns `None` for regular (non-quantized) weights so the caller can
    /// fall back to the graph path.
    ///
    /// The Q/K norm variant is selected through the [`FusedQkNorm`] trait: a
    /// standard [`RMSNorm`] feeds its raw weight, [`GemmaRMSNorm`] feeds its
    /// pre-computed `(1 + weight)`. The underlying kernel always applies the
    /// plain `x * weight * rsqrt(mean(x^2) + eps)` formula, so the
    /// Gemma-vs-standard distinction stays in the weight, not in a kernel
    /// branch. Q/K must use the same norm type (`N`); their eps is taken from
    /// `q_norm`.
    ///
    /// Norm is applied here after the head transpose, whereas the Qwen3 graph
    /// fallback norms before the transpose. RMSNorm reduces over the last axis
    /// (head_dim), which the `[B, T, H, D]` -> `[B, H, T, D]` permutation leaves
    /// untouched, so both orders are numerically equivalent per element.
    ///
    /// Used by: Gemma3 dense attention path, Qwen3 / Qwen3-MoE decode attention
    /// path.
    pub fn forward_split_norm_rope_quantized<N: FusedQkNorm>(
        &self,
        x: &MlxArray,
        q_norm: &N,
        k_norm: &N,
        rope_dims: i32,
        rope_base: f32,
        cache_offset: i32,
    ) -> Option<(
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    )> {
        match &self.qkv_proj {
            UnifiedLinear::Quantized { weight, .. } => {
                // At the C++ boundary the primitive passes a single eps value for
                // both the Q and K RMS norm. Assert (debug builds only) that the
                // caller supplies identical eps on both sides; a mismatch would
                // silently apply the wrong eps to K.
                debug_assert_eq!(
                    q_norm.rms_eps(),
                    k_norm.rms_eps(),
                    "fused QK-norm primitive uses a single eps; q_norm eps {:.2e} != k_norm eps {:.2e}",
                    q_norm.rms_eps(),
                    k_norm.rms_eps(),
                );
                let mut q = cxx::UniquePtr::null();
                let mut k = cxx::UniquePtr::null();
                let mut v = cxx::UniquePtr::null();
                unsafe {
                    ffi::fused_qkv_project_split_norm_rope(
                        x,
                        &weight.weight,
                        &weight.scales,
                        weight.biases_ptr(),
                        q_norm.fused_norm_weight(),
                        k_norm.fused_norm_weight(),
                        self.n_heads,
                        self.n_kv_heads,
                        self.head_dim,
                        rope_dims,
                        rope_base,
                        q_norm.rms_eps(),
                        cache_offset,
                        weight.group_size,
                        weight.bits,
                        &weight.mode,
                        &mut q,
                        &mut k,
                        &mut v,
                    );
                }
                Some((q, k, v))
            }
            UnifiedLinear::Regular(_) => None,
        }
    }
}

/// Quantized per-head linear layer for MLA (Multi-head Latent Attention)
/// Weight shape: [num_heads, output_dim, input_dim_packed]
/// Used in GLM4 MoE Lite, DeepSeek-V2, etc.
pub struct QuantizedMultiLinear {
    pub weight: UniquePtr<MlxArray>,
    pub scales: UniquePtr<MlxArray>,
    pub biases: Option<UniquePtr<MlxArray>>,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantizedMultiLinear {
    /// Create a new quantized multi-linear layer
    pub fn new(
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: Option<UniquePtr<MlxArray>>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        }
    }

    /// Load from weight map
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let scales_name = format!("{}.scales", prefix);
        let biases_name = format!("{}.biases", prefix);

        let weight = weights
            .get(&weight_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
        let scales = weights
            .get(&scales_name)
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Scales not found: {}", scales_name))?;
        let biases = weights.get(&biases_name).map(|w| ffi::copy(w));

        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    /// Forward pass: per-head linear projection
    /// x: [batch, heads, seq, input_dim]
    /// Returns: [batch, heads, seq, output_dim]
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let biases_ptr: *const MlxArray = match &self.biases {
            Some(b) => b.as_ref().unwrap() as *const MlxArray,
            None => std::ptr::null(),
        };

        unsafe {
            ffi::quantized_matmul(
                x,
                &self.weight,
                &self.scales,
                biases_ptr,
                true, // transpose
                self.group_size,
                self.bits,
                "affine",
            )
        }
    }

    /// Forward pass without transpose: x @ weight
    /// Used by MLA embed_q(kv_latent, transpose=False) for projecting latent to K
    pub fn forward_no_transpose(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let biases_ptr: *const MlxArray = match &self.biases {
            Some(b) => b.as_ref().unwrap() as *const MlxArray,
            None => std::ptr::null(),
        };

        unsafe {
            ffi::quantized_matmul(
                x,
                &self.weight,
                &self.scales,
                biases_ptr,
                false, // no transpose
                self.group_size,
                self.bits,
                "affine",
            )
        }
    }

    /// Dequantize weights to full precision
    /// Returns: [num_heads, output_dim, input_dim]
    pub fn dequantize(&self) -> UniquePtr<MlxArray> {
        let biases_ptr: *const MlxArray = match &self.biases {
            Some(b) => b.as_ref().unwrap() as *const MlxArray,
            None => std::ptr::null(),
        };

        unsafe {
            ffi::dequantize(
                &self.weight,
                &self.scales,
                biases_ptr,
                self.group_size,
                self.bits,
                "affine",
            )
        }
    }
}

/// SwiGLU MLP layer with optional compilation for kernel fusion
pub struct SwiGLUMLP {
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
    pub use_compiled: bool,
}

impl SwiGLUMLP {
    /// Create a new SwiGLU MLP layer
    pub fn new(
        gate_proj: QuantizedWeight,
        up_proj: QuantizedWeight,
        down_proj: QuantizedWeight,
        use_compiled: bool,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
            use_compiled,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if self.use_compiled {
            // Use compiled version with kernel fusion
            // Falls back to non-compiled for non-affine modes inside C++
            unsafe {
                ffi::compiled_moe_expert_forward(
                    x,
                    &self.gate_proj.weight,
                    &self.gate_proj.scales,
                    self.gate_proj.biases_ptr(),
                    &self.up_proj.weight,
                    &self.up_proj.scales,
                    self.up_proj.biases_ptr(),
                    &self.down_proj.weight,
                    &self.down_proj.scales,
                    self.down_proj.biases_ptr(),
                    self.gate_proj.group_size,
                    self.gate_proj.bits,
                    &self.gate_proj.mode,
                )
            }
        } else {
            // Non-compiled version
            let gate = unsafe {
                ffi::quantized_linear_forward(
                    x,
                    &self.gate_proj.weight,
                    &self.gate_proj.scales,
                    self.gate_proj.biases_ptr(),
                    std::ptr::null(),
                    self.gate_proj.group_size,
                    self.gate_proj.bits,
                    &self.gate_proj.mode,
                )
            };

            let up = unsafe {
                ffi::quantized_linear_forward(
                    x,
                    &self.up_proj.weight,
                    &self.up_proj.scales,
                    self.up_proj.biases_ptr(),
                    std::ptr::null(),
                    self.up_proj.group_size,
                    self.up_proj.bits,
                    &self.up_proj.mode,
                )
            };

            let silu_gate = ffi::silu(&gate);
            let activated = ffi::multiply(&silu_gate, &up);

            unsafe {
                ffi::quantized_linear_forward(
                    &activated,
                    &self.down_proj.weight,
                    &self.down_proj.scales,
                    self.down_proj.biases_ptr(),
                    std::ptr::null(),
                    self.down_proj.group_size,
                    self.down_proj.bits,
                    &self.down_proj.mode,
                )
            }
        }
    }
}

/// MoE Switch layer using gather_qmm for efficient expert routing
pub struct MoESwitch {
    /// Expert weights: [num_experts, ...]
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
    pub num_experts: i32,
}

impl MoESwitch {
    /// Create a new MoE switch layer
    pub fn new(
        gate_proj: QuantizedWeight,
        up_proj: QuantizedWeight,
        down_proj: QuantizedWeight,
        num_experts: i32,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
            num_experts,
        }
    }

    /// Forward pass with expert indices
    /// x: [batch, seq_len, hidden_dim]
    /// indices: [batch, seq_len] - expert index for each token
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = unsafe {
            ffi::gather_qmm(
                x,
                &self.gate_proj.weight,
                &self.gate_proj.scales,
                self.gate_proj.biases_ptr(),
                std::ptr::null(),    // lhs_indices
                indices as *const _, // rhs_indices
                true,                // transpose
                self.gate_proj.group_size,
                self.gate_proj.bits,
                false, // sorted_indices
                &self.gate_proj.mode,
            )
        };

        let up = unsafe {
            ffi::gather_qmm(
                x,
                &self.up_proj.weight,
                &self.up_proj.scales,
                self.up_proj.biases_ptr(),
                std::ptr::null(),
                indices as *const _,
                true,
                self.up_proj.group_size,
                self.up_proj.bits,
                false,
                &self.up_proj.mode,
            )
        };

        let activated = ffi::compiled_swiglu_activation(&gate, &up);

        unsafe {
            ffi::gather_qmm(
                &activated,
                &self.down_proj.weight,
                &self.down_proj.scales,
                self.down_proj.biases_ptr(),
                std::ptr::null(),
                indices as *const _,
                true,
                self.down_proj.group_size,
                self.down_proj.bits,
                false,
                &self.down_proj.mode,
            )
        }
    }
}

/// Attention layer with RoPE and KV cache
pub struct Attention {
    pub q_proj: QuantizedWeight,
    pub k_proj: QuantizedWeight,
    pub v_proj: QuantizedWeight,
    pub o_proj: QuantizedWeight,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_dims: i32,
    pub rope_base: f32,
    pub rope_scale: f32,
}

impl Attention {
    /// Create a new attention layer
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_proj: QuantizedWeight,
        k_proj: QuantizedWeight,
        v_proj: QuantizedWeight,
        o_proj: QuantizedWeight,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rope_dims: i32,
        rope_base: f32,
        rope_scale: f32,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dims,
            rope_base,
            rope_scale,
        }
    }

    /// Forward pass
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(x);
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V
        let q = unsafe {
            ffi::quantized_linear_forward(
                x,
                &self.q_proj.weight,
                &self.q_proj.scales,
                self.q_proj.biases_ptr(),
                std::ptr::null(),
                self.q_proj.group_size,
                self.q_proj.bits,
                &self.q_proj.mode,
            )
        };

        let k = unsafe {
            ffi::quantized_linear_forward(
                x,
                &self.k_proj.weight,
                &self.k_proj.scales,
                self.k_proj.biases_ptr(),
                std::ptr::null(),
                self.k_proj.group_size,
                self.k_proj.bits,
                &self.k_proj.mode,
            )
        };

        let v = unsafe {
            ffi::quantized_linear_forward(
                x,
                &self.v_proj.weight,
                &self.v_proj.scales,
                self.v_proj.biases_ptr(),
                std::ptr::null(),
                self.v_proj.group_size,
                self.v_proj.bits,
                &self.v_proj.mode,
            )
        };

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = ffi::reshape(&q, &[batch_size, seq_len, self.n_heads, self.head_dim]);
        let k = ffi::reshape(&k, &[batch_size, seq_len, self.n_kv_heads, self.head_dim]);
        let v = ffi::reshape(&v, &[batch_size, seq_len, self.n_kv_heads, self.head_dim]);

        // Apply RoPE
        let offset = cache.offset;
        let q = ffi::fast_rope(
            &q,
            self.rope_dims,
            false,
            self.rope_base,
            self.rope_scale,
            offset,
        );
        let k = ffi::fast_rope(
            &k,
            self.rope_dims,
            false,
            self.rope_base,
            self.rope_scale,
            offset,
        );

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = ffi::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = ffi::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = ffi::transpose_axes(&v, &[0, 2, 1, 3]);

        // Update KV cache and get sliced views
        let (k, v) = cache.update_and_fetch(k, v);

        // Compute attention scale
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Scaled dot-product attention
        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        let attn_out =
            unsafe { ffi::fast_scaled_dot_product_attention(&q, &k, &v, scale, mask_ptr) };

        // Transpose back and reshape
        let attn_out = ffi::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = ffi::reshape(
            &attn_out,
            &[batch_size, seq_len, self.n_heads * self.head_dim],
        );

        // Output projection
        unsafe {
            ffi::quantized_linear_forward(
                &attn_out,
                &self.o_proj.weight,
                &self.o_proj.scales,
                self.o_proj.biases_ptr(),
                std::ptr::null(),
                self.o_proj.group_size,
                self.o_proj.bits,
                &self.o_proj.mode,
            )
        }
    }
}

// MultiLinear Layer (for MLA attention: embed_q, unembed_out).
/// MultiLinear layer for per-head linear projections.
///
/// Weight shape: `[num_heads, output_dims, input_dims]`
///
/// Used by MLA attention (DeepSeek V3/V3.2, GLM4 MoE Lite) for:
/// - `embed_q`: projects Q_nope into KV latent space
/// - `unembed_out`: projects attention output from latent space to V dimensions
///
/// Supports both quantized and non-quantized weights.
/// Used by: DeepSeek V3, DeepSeek V3.2, GLM4 MoE Lite
pub enum MultiLinear {
    Quantized(QuantizedMultiLinear),
    Regular(RegularMultiLinear),
}

/// Non-quantized multi-head linear layer.
/// Weight shape: `[num_heads, output_dims, input_dims]`
pub struct RegularMultiLinear {
    pub weight: UniquePtr<MlxArray>,
}

impl MultiLinear {
    /// Load from weight map, auto-detecting quantization.
    ///
    /// Checks for `.scales` key to determine if weights are quantized.
    pub fn from_weights(
        weights: &crate::weights::WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let scales_name = format!("{}.scales", prefix);

        if weights.contains_key(&scales_name) {
            // Quantized: use existing QuantizedMultiLinear
            Ok(MultiLinear::Quantized(QuantizedMultiLinear::from_weights(
                weights, prefix, group_size, bits,
            )?))
        } else {
            // Non-quantized: regular weight
            let weight = weights
                .get(&weight_name)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
            Ok(MultiLinear::Regular(RegularMultiLinear { weight }))
        }
    }

    /// Forward pass with transpose (default behavior).
    ///
    /// Computes `x @ weight.swapaxes(-1, -2)`.
    /// Input x: `[..., num_heads, seq_len, input_dims]`
    /// Output: `[..., num_heads, seq_len, output_dims]`
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MultiLinear::Quantized(q) => q.forward(x),
            MultiLinear::Regular(r) => {
                // weight is [num_heads, output_dims, input_dims]
                // swapaxes(-1, -2) → [num_heads, input_dims, output_dims]
                let wt = ffi::transpose_axes(&r.weight, &[0, 2, 1]);
                ffi::matmul(x, &wt)
            }
        }
    }

    /// Forward pass without transpose.
    ///
    /// Computes `x @ weight`.
    /// Input x: `[..., 1_or_heads, seq_len, output_dims]`
    /// Output: `[..., num_heads, seq_len, input_dims]`
    pub fn forward_no_transpose(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MultiLinear::Quantized(q) => q.forward_no_transpose(x),
            MultiLinear::Regular(r) => ffi::matmul(x, &r.weight),
        }
    }
}

/// SwiGLU MLP forward for non-quantized (FP16/BF16) UnifiedLinear layers.
///
/// When all three projections are non-quantized, calls the C++ path which
/// runs matmul operations directly and fuses only the SwiGLU activation via
/// compiled element-wise kernels.
///
/// Returns `None` if any projection is quantized (caller should use the quantized path).
///
/// Used by: Llama, Qwen2, Qwen3, Mistral and other SwiGLU FP models
pub fn compiled_swiglu_mlp_fp16(
    x: &MlxArray,
    gate_proj: &UnifiedLinear,
    up_proj: &UnifiedLinear,
    down_proj: &UnifiedLinear,
) -> Option<crate::UniquePtr<MlxArray>> {
    let gate_lin = gate_proj.regular_weight()?;
    let up_lin = up_proj.regular_weight()?;
    let down_lin = down_proj.regular_weight()?;

    let gate_bias_ptr = gate_lin
        .bias
        .as_ref()
        .map(|b| b.as_ref().unwrap() as *const MlxArray)
        .unwrap_or(std::ptr::null());
    let up_bias_ptr = up_lin
        .bias
        .as_ref()
        .map(|b| b.as_ref().unwrap() as *const MlxArray)
        .unwrap_or(std::ptr::null());
    let down_bias_ptr = down_lin
        .bias
        .as_ref()
        .map(|b| b.as_ref().unwrap() as *const MlxArray)
        .unwrap_or(std::ptr::null());

    Some(unsafe {
        ffi::compiled_swiglu_mlp_forward_fp16(
            x,
            &gate_lin.weight,
            &up_lin.weight,
            &down_lin.weight,
            gate_bias_ptr,
            up_bias_ptr,
            down_bias_ptr,
        )
    })
}

/// SwiGLU MLP forward for either quantized or non-quantized UnifiedLinear layers.
///
/// Returns a fused compiled path when all three projections are either:
/// - non-quantized regular weights, via `compiled_swiglu_mlp_fp16()`
/// - quantized weights, via `compiled_moe_expert_forward()`
///
/// Returns `None` for mixed projection kinds so callers can preserve their
/// existing fallback path.
///
/// Used by: Llama3, Qwen3, SmolLM3, StableLM, Exaone4 and other SwiGLU models
pub fn compiled_swiglu_mlp(
    x: &MlxArray,
    gate_proj: &UnifiedLinear,
    up_proj: &UnifiedLinear,
    down_proj: &UnifiedLinear,
) -> Option<crate::UniquePtr<MlxArray>> {
    if let (Some(gate_qw), Some(up_qw), Some(down_qw)) = (
        gate_proj.quantized_weight(),
        up_proj.quantized_weight(),
        down_proj.quantized_weight(),
    ) {
        return Some(unsafe {
            ffi::compiled_moe_expert_forward(
                x,
                &gate_qw.weight,
                &gate_qw.scales,
                gate_qw.biases_ptr(),
                &up_qw.weight,
                &up_qw.scales,
                up_qw.biases_ptr(),
                &down_qw.weight,
                &down_qw.scales,
                down_qw.biases_ptr(),
                gate_qw.group_size,
                gate_qw.bits,
                &gate_qw.mode,
            )
        });
    }

    compiled_swiglu_mlp_fp16(x, gate_proj, up_proj, down_proj)
}

// ── Metal 4 fused attention dispatch ─────────────────────────────────────────

fn should_use_metal4_attention() -> bool {
    let hw = crate::hardware::get_hardware();
    hw.has_neural_accelerator && hw.macos_supports_na
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NaAttentionLogMode {
    Off,
    Sampled,
    All,
}

fn na_attention_log_mode() -> NaAttentionLogMode {
    static MODE: OnceLock<NaAttentionLogMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("MLXCEL_LOG_NA_ATTENTION")
            .map(|v| v.trim().to_ascii_lowercase())
            .ok()
            .as_deref()
        {
            Some("1" | "true" | "yes" | "on" | "sample" | "sampled") => NaAttentionLogMode::Sampled,
            Some("all" | "full") => NaAttentionLogMode::All,
            _ => NaAttentionLogMode::Off,
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn classify_na_attention_dispatch(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    softcap: f32,
    window_size: i32,
    has_mask: bool,
    has_array_mask: bool,
    do_causal: bool,
) -> (&'static str, bool, bool) {
    let q_shape = ffi::array_shape(q);
    let k_shape = ffi::array_shape(k);
    let query_sequence_length = q_shape[2];
    let key_sequence_length = k_shape[2];

    let route = if softcap > 0.0 {
        "softcap"
    } else if has_array_mask
        && window_size == 0
        && query_sequence_length > 1
        && query_sequence_length == key_sequence_length
    {
        "padded_prefill"
    } else if do_causal && !has_mask {
        "native_causal"
    } else if has_array_mask {
        "array_mask"
    } else if has_mask {
        "mask"
    } else {
        "unmasked"
    };

    let fast_path_eligible =
        ffi::sdpa_supports_fast_path(q, k, v, has_mask, has_array_mask, do_causal);
    let nax_eligible = ffi::sdpa_supports_nax(q, k, v, has_mask, has_array_mask, do_causal);

    (route, fast_path_eligible, nax_eligible)
}

#[allow(clippy::too_many_arguments)]
fn na_attention_eligibility_reasons(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    has_mask: bool,
    has_array_mask: bool,
    do_causal: bool,
    fast_path_eligible: bool,
    nax_eligible: bool,
) -> (&'static str, &'static str) {
    let q_shape = ffi::array_shape(q);
    let k_shape = ffi::array_shape(k);
    let v_shape = ffi::array_shape(v);

    let q_len = q_shape[2];
    let k_len = k_shape[2];
    let q_heads = q_shape[1];
    let kv_heads = k_shape[1].max(1);
    let head_dim = q_shape[3];
    let value_head_dim = v_shape[3];
    let gqa_factor = q_heads / kv_heads;
    let tf32_enabled = std::env::var("MLX_ENABLE_TF32")
        .map(|v| v.trim() != "0")
        .unwrap_or(true);

    let fast_reason = if fast_path_eligible {
        "eligible"
    } else if q_len <= 8 {
        if q_len > k_len {
            "vector_q_len_gt_k_len"
        } else if head_dim != value_head_dim {
            "vector_head_dim_mismatch"
        } else if !matches!(head_dim, 64 | 96 | 128 | 256) {
            "vector_head_dim_unsupported"
        } else if q_len * gqa_factor > 32 {
            "vector_gqa_over_limit"
        } else {
            "vector_other"
        }
    } else if head_dim != value_head_dim {
        "full_head_dim_mismatch"
    } else if !matches!(head_dim, 64 | 80 | 128 | 256) {
        "full_head_dim_unsupported"
    } else if has_mask && !has_array_mask && !(q_len <= k_len && do_causal) {
        "full_mask_mode_unsupported"
    } else {
        "full_other"
    };

    let nax_reason = if nax_eligible {
        "eligible"
    } else if !fast_path_eligible {
        "fast_path_blocked"
    } else if q_len <= 8 {
        "vector_path_only"
    } else if head_dim == 80 {
        "head_dim_80_excluded"
    } else if ffi::array_dtype(q) == crate::dtype::FLOAT32 && !tf32_enabled {
        "float32_tf32_disabled"
    } else {
        "other"
    };

    (fast_reason, nax_reason)
}

#[allow(clippy::too_many_arguments)]
fn record_na_attention_dispatch(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    softcap: f32,
    window_size: i32,
    has_mask: bool,
    has_array_mask: bool,
    do_causal: bool,
) {
    static DISPATCH_COUNT: AtomicUsize = AtomicUsize::new(0);
    let count = DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let should_log = match na_attention_log_mode() {
        NaAttentionLogMode::Off => false,
        NaAttentionLogMode::Sampled => count <= 8 || count.is_multiple_of(100),
        NaAttentionLogMode::All => true,
    };
    if should_log {
        let q_shape = ffi::array_shape(q);
        let k_shape = ffi::array_shape(k);
        let phase = if q_shape[2] > 1 { "prefill" } else { "decode" };
        let (route, fast_path_eligible, nax_eligible) = classify_na_attention_dispatch(
            q,
            k,
            v,
            softcap,
            window_size,
            has_mask,
            has_array_mask,
            do_causal,
        );
        let (fast_reason, nax_reason) = na_attention_eligibility_reasons(
            q,
            k,
            v,
            has_mask,
            has_array_mask,
            do_causal,
            fast_path_eligible,
            nax_eligible,
        );
        eprintln!(
            "[mlxcel][na-attention] dispatch={} phase={} route={} fast_path_eligible={} fast_path_reason={} nax_eligible={} nax_reason={} q={:?} k={:?} scale={:.6} softcap={:.3} window_size={}",
            count,
            phase,
            route,
            fast_path_eligible,
            fast_reason,
            nax_eligible,
            nax_reason,
            q_shape,
            k_shape,
            scale,
            softcap,
            window_size
        );
    }
}

/// Causal SDPA dispatch that preserves MLX's native `"causal"` mask mode.
///
/// Used by: `crate::causal_attention()` on M5-class hardware for full-window
/// causal prefill, so upstream MLX can select the NAX causal kernel directly.
pub fn metal4_causal_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
) -> UniquePtr<MlxArray> {
    if should_use_metal4_attention() {
        record_na_attention_dispatch(q, k, v, scale, 0.0, 0, false, false, true);
    }
    ffi::ffi_fast_scaled_dot_product_attention_causal(q, k, v, scale)
}

/// Unified attention dispatch with transparent Metal 4 and softcap fallback.
///
/// Used by: Llama3, Qwen3, Gemma2, Gemma3 and other standard SDPA call sites
pub fn attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: Option<&MlxArray>,
    softcap: f32,
    window_size: i32,
) -> UniquePtr<MlxArray> {
    let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
    if should_use_metal4_attention() {
        return metal4_attention(q, k, v, scale, mask, softcap, window_size);
    }

    if softcap > 0.0 {
        let q_heads = ffi::array_shape(q)[1];
        let kv_heads = ffi::array_shape(k)[1];
        if q_heads > kv_heads && q_heads % kv_heads == 0 {
            let n_rep = q_heads / kv_heads;
            // SAFETY: q/k/v/mask_ptr are valid for the duration of this call.
            return unsafe {
                ffi::compiled_softcap_sdpa_gqa(q, k, v, scale, softcap, n_rep, mask_ptr)
            };
        }
        // SAFETY: q/k/v/mask_ptr are valid for the duration of this call.
        return unsafe { ffi::compiled_softcap_sdpa(q, k, v, scale, softcap, mask_ptr) };
    }

    // SAFETY: q/k/v/mask_ptr are valid for the duration of this call.
    unsafe { ffi::fast_scaled_dot_product_attention(q, k, v, scale, mask_ptr) }
}

/// Pointer-friendly attention wrapper for existing model call sites.
///
/// Used by: Model attention call sites that still store masks as raw pointers
///
/// # Safety
///
/// `mask` must be either null or point to a valid, live `MlxArray`. The
/// referent must remain valid for the duration of this call.
pub unsafe fn attention_from_ptr(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: *const MlxArray,
    softcap: f32,
    window_size: i32,
) -> UniquePtr<MlxArray> {
    let mask = if mask.is_null() {
        None
    } else {
        // SAFETY: callers must pass either null or a valid `MlxArray` pointer.
        Some(unsafe { &*mask })
    };
    attention(q, k, v, scale, mask, softcap, window_size)
}

fn validate_paged_decode_inputs(
    q: &MlxArray,
    cache_keys: &[*const MlxArray],
    cache_values: &[*const MlxArray],
    metadata: &crate::cache::PagedDecodeMetadata,
) -> Result<(), String> {
    let q_shape = ffi::array_shape(q);
    if q_shape.len() != 4 {
        return Err(format!(
            "paged decode attention expected q rank 4, got shape {:?}",
            q_shape
        ));
    }
    if q_shape[2] != 1 {
        return Err(format!(
            "paged decode attention only supports decode-only q_len == 1, got {}",
            q_shape[2]
        ));
    }

    let batch = q_shape[0].max(0) as usize;
    if cache_keys.len() != batch || cache_values.len() != batch {
        return Err(format!(
            "paged decode attention expected {} cache pointers, got {} keys and {} values",
            batch,
            cache_keys.len(),
            cache_values.len()
        ));
    }
    if cache_keys.iter().any(|ptr| ptr.is_null()) || cache_values.iter().any(|ptr| ptr.is_null()) {
        return Err(
            "paged decode attention received a null dense compatibility cache pointer".to_string(),
        );
    }
    if metadata.len() != batch {
        return Err(format!(
            "paged decode attention expected {} metadata entries, got {}",
            batch,
            metadata.len()
        ));
    }
    if metadata.block_table_offsets.len() != batch + 1 {
        return Err(format!(
            "paged decode attention expected {} block offsets, got {}",
            batch + 1,
            metadata.block_table_offsets.len()
        ));
    }
    if metadata
        .block_table_offsets
        .last()
        .copied()
        .unwrap_or_default() as usize
        != metadata.block_tables.len()
    {
        return Err(
            "paged decode attention block table offsets do not cover the flattened block table"
                .to_string(),
        );
    }

    Ok(())
}

/// Reference paged decode path over dense compatibility KV caches.
///
/// This stays entirely in Rust/FFI wrappers and is primarily used as a
/// correctness baseline and benchmark fallback for the native paged decode
/// kernel.
pub fn paged_decode_attention_dense_fallback(
    q: &MlxArray,
    cache_keys: &[*const MlxArray],
    cache_values: &[*const MlxArray],
    metadata: &crate::cache::PagedDecodeMetadata,
    scale: f32,
) -> Result<UniquePtr<MlxArray>, String> {
    validate_paged_decode_inputs(q, cache_keys, cache_values, metadata)?;

    let q_shape = ffi::array_shape(q);
    let batch = q_shape[0].max(0) as usize;
    let mut outputs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(batch);

    for batch_idx in 0..batch {
        let kv_len = metadata.kv_lens[batch_idx];
        if kv_len <= 0 {
            return Err(format!(
                "paged decode fallback requires kv_len > 0 for batch index {batch_idx}, got {kv_len}"
            ));
        }

        let q_i = ffi::slice(
            q,
            &[batch_idx as i32, 0, 0, 0],
            &[batch_idx as i32 + 1, i32::MAX, 1, i32::MAX],
        );

        let table_begin = metadata.block_table_offsets[batch_idx] as usize;
        let table_end = metadata.block_table_offsets[batch_idx + 1] as usize;
        let key_cache = unsafe { &*cache_keys[batch_idx] };
        let value_cache = unsafe { &*cache_values[batch_idx] };

        let mut key_visible: Option<UniquePtr<MlxArray>> = None;
        let mut value_visible: Option<UniquePtr<MlxArray>> = None;

        for &logical_block in &metadata.block_tables[table_begin..table_end] {
            let block_start = logical_block * metadata.block_size;
            if block_start >= kv_len {
                continue;
            }
            let block_end = (block_start + metadata.block_size).min(kv_len);

            let key_block = ffi::slice(
                key_cache,
                &[0, 0, block_start, 0],
                &[1, i32::MAX, block_end, i32::MAX],
            );
            let value_block = ffi::slice(
                value_cache,
                &[0, 0, block_start, 0],
                &[1, i32::MAX, block_end, i32::MAX],
            );

            key_visible = Some(match key_visible {
                Some(prev) => crate::concatenate(&prev, &key_block, 2),
                None => key_block,
            });
            value_visible = Some(match value_visible {
                Some(prev) => crate::concatenate(&prev, &value_block, 2),
                None => value_block,
            });
        }

        let key_visible = key_visible.ok_or_else(|| {
            format!("paged decode fallback built no visible key blocks for batch index {batch_idx}")
        })?;
        let value_visible = value_visible.ok_or_else(|| {
            format!(
                "paged decode fallback built no visible value blocks for batch index {batch_idx}"
            )
        })?;

        outputs.push(unsafe {
            attention_from_ptr(
                &q_i,
                &key_visible,
                &value_visible,
                scale,
                std::ptr::null(),
                0.0,
                0,
            )
        });
    }

    // `drain(..1)` panics on an empty vec, so reject `batch == 0` explicitly
    // with a clean error instead (issue #195).
    if outputs.is_empty() {
        return Err("paged decode fallback received an empty batch (batch == 0)".to_string());
    }
    let mut outputs = outputs.into_iter();
    let mut result = outputs.next().expect("outputs is non-empty: checked above");
    for output in outputs {
        result = crate::concatenate(&result, &output, 0);
    }
    Ok(result)
}

fn validate_pooled_paged_decode_inputs(
    q: &MlxArray,
    states: &[&crate::cache::PagedSequenceState],
) -> Result<(), String> {
    let q_shape = ffi::array_shape(q);
    if q_shape.len() != 4 {
        return Err(format!(
            "pooled paged decode attention expected q rank 4, got shape {:?}",
            q_shape
        ));
    }
    if q_shape[2] != 1 {
        return Err(format!(
            "pooled paged decode attention only supports decode-only q_len == 1, got {}",
            q_shape[2]
        ));
    }

    let batch = q_shape[0].max(0) as usize;
    if states.len() != batch {
        return Err(format!(
            "pooled paged decode attention expected {batch} sequence states, got {}",
            states.len()
        ));
    }

    Ok(())
}

/// Reference paged decode path that gathers K/V from the [`PagedBlockPool`].
///
/// This is the Phase 2 (#119) pooled read path of the unified paged KV cache
/// (epic #116). It is the pooled analogue of [`paged_decode_attention_dense_fallback`],
/// which stays the live decode path and the parity baseline this function is
/// validated against. The two differ only in where the per-sequence visible
/// K/V comes from: the dense fallback slices contiguous dense compatibility
/// buffers, while this function calls [`crate::cache::PagedBlockPool::gather_visible`]
/// to pull each sequence's physical blocks out of the pool by block-table order.
/// Everything downstream — the per-batch `q` slice, the fused SDPA dispatch via
/// [`attention_from_ptr`], and the axis-0 batch concat — is identical.
///
/// Per ADR 0001 (`docs/adr/0001-paged-attention-gather-vs-fused-kernel.md`) this
/// is strategy option A, *gather-then-SDPA*: `gather_visible` builds a
/// `take`/`reshape`/`transpose` MLX graph that MLX fuses into the fused-SDPA
/// read, so the scattered-block gather adds no separate full-copy of the
/// sequence KV at the common context lengths. The fused Metal paged-attention
/// kernel (option B) is deferred to #123; this Rust path remains its correctness
/// reference and fallback. Live model routing onto this path is #121 (it needs
/// the pool to be populated by the #120 prefill writer and routed by the #121
/// scheduler), so this function is additive machinery and the live decode path
/// is unchanged until then.
///
/// The path handles full-attention and sliding-window sequences uniformly. In
/// the paged model there is no ring-buffer wrap: the visible window is encoded
/// entirely in the per-sequence block table plus `logical_start` (which
/// `trim_tokens` advances as the window slides), so `gather_visible` returns
/// exactly the `[logical_start, len)` window already shaped
/// `[1, n_kv_heads, visible_len, head_dim]` (SDPA-ready). That is why there is
/// no separate "rotating" pooled variant mirroring
/// [`paged_decode_attention_rotating_fallback`].
///
/// `gather_visible` returning `Ok(None)` (no visible tokens, or the layer has no
/// pool storage yet) is treated as an error here, because decode requires a
/// non-empty visible window — the pooled analogue of the dense fallback's
/// `kv_len > 0` precondition.
pub fn paged_decode_attention_pooled_fallback(
    q: &MlxArray,
    pool: &crate::cache::PagedBlockPool,
    states: &[&crate::cache::PagedSequenceState],
    layer_idx: usize,
    scale: f32,
) -> Result<UniquePtr<MlxArray>, String> {
    validate_pooled_paged_decode_inputs(q, states)?;

    let q_shape = ffi::array_shape(q);
    let batch = q_shape[0].max(0) as usize;
    let mut outputs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(batch);

    for (batch_idx, state) in states.iter().enumerate() {
        let q_i = ffi::slice(
            q,
            &[batch_idx as i32, 0, 0, 0],
            &[batch_idx as i32 + 1, i32::MAX, 1, i32::MAX],
        );

        let (key_visible, value_visible) =
            pool.gather_visible(state, layer_idx)?.ok_or_else(|| {
                format!(
                    "pooled paged decode requires visible tokens for batch index {batch_idx} on layer {layer_idx}, but the pool gathered none"
                )
            })?;

        outputs.push(unsafe {
            attention_from_ptr(
                &q_i,
                &key_visible,
                &value_visible,
                scale,
                std::ptr::null(),
                0.0,
                0,
            )
        });
    }

    // `drain(..1)` panics on an empty vec, so reject `batch == 0` explicitly
    // with a clean error instead (issue #195).
    if outputs.is_empty() {
        return Err(
            "pooled paged decode fallback received an empty batch (batch == 0)".to_string(),
        );
    }
    let mut outputs = outputs.into_iter();
    let mut result = outputs.next().expect("outputs is non-empty: checked above");
    for output in outputs {
        result = crate::concatenate(&result, &output, 0);
    }
    Ok(result)
}

/// Process-wide env override for the fused paged-attention kernel (#123).
///
/// `MLXCEL_PAGED_ATTENTION_NATIVE=1` (or `true` / `on` / `yes`) force-enables
/// the native kernel regardless of the per-config `use_native_paged_kernel`
/// flag, so operators can A/B the kernel without rebuilding. Read once and
/// cached so the decode hot path never touches the environment.
fn native_paged_kernel_env() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MLXCEL_PAGED_ATTENTION_NATIVE")
            .map(|v| {
                matches!(
                    v.trim(),
                    "1" | "true" | "on" | "yes" | "TRUE" | "ON" | "YES"
                )
            })
            .unwrap_or(false)
    })
}

/// Gated pooled paged decode attention (epic #116 Phase 6, #123).
///
/// When the native kernel is enabled (the per-config `use_native_paged_kernel`
/// flag or the `MLXCEL_PAGED_ATTENTION_NATIVE` env override), dispatches to the
/// fused Metal kernel ([`crate::cache::PagedBlockPool::paged_decode_fused`]),
/// which reads scattered pool blocks directly with no gather copy (ADR 0001
/// strategy B). Otherwise, and whenever the kernel declines (the layer's pool
/// tensors are not yet allocated, or no sequence has visible tokens), it falls
/// back to [`paged_decode_attention_pooled_fallback`], the gather-then-SDPA
/// reference. The two paths agree within RMS < 5e-3, so the gate is a pure
/// performance switch with no behavioural change.
pub fn paged_decode_attention_pooled(
    q: &MlxArray,
    pool: &crate::cache::PagedBlockPool,
    states: &[&crate::cache::PagedSequenceState],
    layer_idx: usize,
    scale: f32,
    use_native_paged_kernel: bool,
) -> Result<UniquePtr<MlxArray>, String> {
    if use_native_paged_kernel || native_paged_kernel_env() {
        if let Some(out) = pool.paged_decode_fused(q, states, layer_idx, scale)? {
            return Ok(out);
        }
    }
    paged_decode_attention_pooled_fallback(q, pool, states, layer_idx, scale)
}

/// Native paged decode path over dense compatibility KV caches.
///
/// The C++ bridge consumes the logical block table metadata and performs the
/// per-sequence block gathering plus SDPA dispatch natively, reducing Rust-side
/// FFI churn on the decode hot path.
pub fn paged_decode_attention_dense_compat(
    q: &MlxArray,
    cache_keys: &[*const MlxArray],
    cache_values: &[*const MlxArray],
    metadata: &crate::cache::PagedDecodeMetadata,
    scale: f32,
) -> Result<UniquePtr<MlxArray>, String> {
    validate_paged_decode_inputs(q, cache_keys, cache_values, metadata)?;
    Ok(unsafe {
        ffi::paged_decode_attention_dense_compat(
            q,
            cache_keys,
            cache_values,
            &metadata.kv_lens,
            &metadata.block_tables,
            &metadata.block_table_offsets,
            metadata.block_size,
            scale,
        )
    })
}

fn validate_rotating_paged_decode_inputs(
    q: &MlxArray,
    cache_keys: &[*const MlxArray],
    cache_values: &[*const MlxArray],
    metadata: &crate::cache::RotatingPagedDecodeMetadata,
) -> Result<(), String> {
    let q_shape = ffi::array_shape(q);
    if q_shape.len() != 4 {
        return Err(format!(
            "rotating paged decode attention expected q rank 4, got shape {:?}",
            q_shape
        ));
    }
    if q_shape[2] != 1 {
        return Err(format!(
            "rotating paged decode attention only supports decode-only q_len == 1, got {}",
            q_shape[2]
        ));
    }

    let batch = q_shape[0].max(0) as usize;
    if cache_keys.len() != batch || cache_values.len() != batch {
        return Err(format!(
            "rotating paged decode attention expected {} cache pointers, got {} keys and {} values",
            batch,
            cache_keys.len(),
            cache_values.len()
        ));
    }
    if cache_keys.iter().any(|ptr| ptr.is_null()) || cache_values.iter().any(|ptr| ptr.is_null()) {
        return Err(
            "rotating paged decode attention received a null ring-buffer cache pointer".to_string(),
        );
    }
    if metadata.len() != batch {
        return Err(format!(
            "rotating paged decode attention expected {} metadata entries, got {}",
            batch,
            metadata.len()
        ));
    }
    Ok(())
}

/// Reference paged decode path over rotating ring-buffer KV caches.
pub fn paged_decode_attention_rotating_fallback(
    q: &MlxArray,
    cache_keys: &[*const MlxArray],
    cache_values: &[*const MlxArray],
    metadata: &crate::cache::RotatingPagedDecodeMetadata,
    scale: f32,
) -> Result<UniquePtr<MlxArray>, String> {
    validate_rotating_paged_decode_inputs(q, cache_keys, cache_values, metadata)?;

    let q_shape = ffi::array_shape(q);
    let batch = q_shape[0].max(0) as usize;
    let mut outputs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(batch);

    for batch_idx in 0..batch {
        let kv_len = metadata.kv_lens[batch_idx];
        let logical_start = metadata.logical_starts[batch_idx];
        if kv_len <= 0 {
            return Err(format!(
                "rotating paged decode fallback requires kv_len > 0 for batch index {batch_idx}, got {kv_len}"
            ));
        }

        let q_i = ffi::slice(
            q,
            &[batch_idx as i32, 0, 0, 0],
            &[batch_idx as i32 + 1, i32::MAX, 1, i32::MAX],
        );

        let key_cache = unsafe { &*cache_keys[batch_idx] };
        let value_cache = unsafe { &*cache_values[batch_idx] };
        let buffer_len = ffi::array_shape(key_cache)
            .get(2)
            .copied()
            .unwrap_or_default();
        if buffer_len <= 0 {
            return Err(format!(
                "rotating paged decode fallback requires non-empty buffer for batch index {batch_idx}"
            ));
        }
        if logical_start < 0 || logical_start >= buffer_len {
            return Err(format!(
                "rotating paged decode fallback received invalid logical_start={logical_start} for buffer_len={buffer_len}"
            ));
        }

        let block_count = (kv_len + metadata.block_size - 1) / metadata.block_size;
        let mut key_visible: Option<UniquePtr<MlxArray>> = None;
        let mut value_visible: Option<UniquePtr<MlxArray>> = None;
        for logical_block in 0..block_count {
            let logical_pos = logical_block * metadata.block_size;
            let logical_end = (logical_pos + metadata.block_size).min(kv_len);
            let token_count = logical_end - logical_pos;
            if token_count <= 0 {
                continue;
            }

            let physical_start = (logical_start + logical_pos).rem_euclid(buffer_len);
            let physical_end = physical_start + token_count;
            let key_block = if physical_end <= buffer_len {
                ffi::slice(
                    key_cache,
                    &[0, 0, physical_start, 0],
                    &[1, i32::MAX, physical_end, i32::MAX],
                )
            } else {
                let key_tail = ffi::slice(
                    key_cache,
                    &[0, 0, physical_start, 0],
                    &[1, i32::MAX, buffer_len, i32::MAX],
                );
                let key_head = ffi::slice(
                    key_cache,
                    &[0, 0, 0, 0],
                    &[1, i32::MAX, physical_end - buffer_len, i32::MAX],
                );
                crate::concatenate(&key_tail, &key_head, 2)
            };
            let value_block = if physical_end <= buffer_len {
                ffi::slice(
                    value_cache,
                    &[0, 0, physical_start, 0],
                    &[1, i32::MAX, physical_end, i32::MAX],
                )
            } else {
                let value_tail = ffi::slice(
                    value_cache,
                    &[0, 0, physical_start, 0],
                    &[1, i32::MAX, buffer_len, i32::MAX],
                );
                let value_head = ffi::slice(
                    value_cache,
                    &[0, 0, 0, 0],
                    &[1, i32::MAX, physical_end - buffer_len, i32::MAX],
                );
                crate::concatenate(&value_tail, &value_head, 2)
            };

            key_visible = Some(match key_visible {
                Some(prev) => crate::concatenate(&prev, &key_block, 2),
                None => key_block,
            });
            value_visible = Some(match value_visible {
                Some(prev) => crate::concatenate(&prev, &value_block, 2),
                None => value_block,
            });
        }

        let key_visible = key_visible.ok_or_else(|| {
            format!(
                "rotating paged decode fallback built no visible key blocks for batch index {batch_idx}"
            )
        })?;
        let value_visible = value_visible.ok_or_else(|| {
            format!(
                "rotating paged decode fallback built no visible value blocks for batch index {batch_idx}"
            )
        })?;

        outputs.push(unsafe {
            attention_from_ptr(
                &q_i,
                &key_visible,
                &value_visible,
                scale,
                std::ptr::null(),
                0.0,
                0,
            )
        });
    }

    // `drain(..1)` panics on an empty vec, so reject `batch == 0` explicitly
    // with a clean error instead (issue #195).
    if outputs.is_empty() {
        return Err(
            "rotating paged decode fallback received an empty batch (batch == 0)".to_string(),
        );
    }
    let mut outputs = outputs.into_iter();
    let mut result = outputs.next().expect("outputs is non-empty: checked above");
    for output in outputs {
        result = crate::concatenate(&result, &output, 0);
    }
    Ok(result)
}

/// Native paged decode path over rotating ring-buffer KV caches.
pub fn paged_decode_attention_rotating_compat(
    q: &MlxArray,
    cache_keys: &[*const MlxArray],
    cache_values: &[*const MlxArray],
    metadata: &crate::cache::RotatingPagedDecodeMetadata,
    scale: f32,
) -> Result<UniquePtr<MlxArray>, String> {
    validate_rotating_paged_decode_inputs(q, cache_keys, cache_values, metadata)?;
    Ok(unsafe {
        ffi::paged_decode_attention_rotating_compat(
            q,
            cache_keys,
            cache_values,
            &metadata.kv_lens,
            &metadata.logical_starts,
            metadata.block_size,
            scale,
        )
    })
}

/// Metal 4 fused attention dispatch with automatic hardware detection.
///
/// Dispatches SDPA to the Metal 4 fused kernel when the hardware supports it,
/// or falls back to the standard MLX `fast_scaled_dot_product_attention` path.
///
/// Queries `hardware::get_hardware()` to determine whether the current
/// hardware supports Metal 4 TensorOps (M5+ with macOS 26.2+) and passes
/// the `use_metal4` flag to the C++ bridge.
///
/// # Metal 4 path
///
/// When `hw.has_neural_accelerator && hw.macos_supports_na` is true, this
/// function sets `use_metal4 = true`. The C++ bridge delegates standard SDPA
/// to upstream MLX `fast::scaled_dot_product_attention()`, allowing backend
/// Metal/NAX kernel selection on M5-class hardware.
///
/// The Metal 4 path is designed to handle:
///   - Standard MHA and GQA (n_heads != n_kv_heads): upstream MLX broadcasts
///     KV heads internally, so no caller-side repeat_kv is needed.
///   - Softcap (Gemma 2/3): still handled by `compiled_softcap_sdpa*` until
///     upstream MLX grows a fused softcap variant.
///   - Sliding window (Gemma 3, Ministral): still handled by passing an
///     explicit pre-built mask from Rust model code.
///
/// # Current behaviour
///
/// This helper now routes M5-capable requests through upstream MLX main's
/// NAX-backed SDPA implementation via the C++ bridge. Boolean/integer masks
/// are passed through unchanged; float masks are cast to Q's dtype.
///
/// Used by: Llama, Qwen, Gemma2, Gemma3, Mistral, Ministral, DeepSeek, Phi,
/// Exaone, Cohere, InternLM, OLMo, StableLM, StarCoder2, GLM4, Ernie4.5,
/// Hunyuan, Gemma VLMs, Qwen VLMs, SigLIP/CLIP-style vision encoders
pub fn metal4_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: Option<&MlxArray>,
    softcap: f32,
    window_size: i32,
) -> UniquePtr<MlxArray> {
    // Enable Metal 4 dispatch only on M5+ hardware running macOS 26+.
    // Uses the canonical NA detection condition from apple-silicon-precision.md.
    let use_metal4 = should_use_metal4_attention();
    if use_metal4 {
        record_na_attention_dispatch(
            q,
            k,
            v,
            scale,
            softcap,
            window_size,
            mask.is_some(),
            mask.is_some(),
            false,
        );
    }
    let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
    // SAFETY: mask_ptr is either null or a valid reference with lifetime tied
    // to the `mask` argument, which outlives this function call.
    unsafe {
        ffi::fused_metal4_attention(q, k, v, scale, mask_ptr, softcap, window_size, use_metal4)
    }
}

/// GELU MLP forward for non-quantized (FP16/BF16) UnifiedLinear layers.
///
/// When all three projections are non-quantized, calls the C++ path which
/// runs matmul operations directly and fuses only the GELU activation via
/// compiled element-wise kernels.
///
/// Returns `None` if any projection is quantized (caller should use the quantized path).
///
/// Used by: Gemma, Gemma4 and other GELU-gated FP models
pub fn compiled_gelu_mlp_fp16(
    x: &MlxArray,
    gate_proj: &UnifiedLinear,
    up_proj: &UnifiedLinear,
    down_proj: &UnifiedLinear,
) -> Option<crate::UniquePtr<MlxArray>> {
    let gate_lin = gate_proj.regular_weight()?;
    let up_lin = up_proj.regular_weight()?;
    let down_lin = down_proj.regular_weight()?;

    let gate_bias_ptr = gate_lin
        .bias
        .as_ref()
        .map(|b| b.as_ref().unwrap() as *const MlxArray)
        .unwrap_or(std::ptr::null());
    let up_bias_ptr = up_lin
        .bias
        .as_ref()
        .map(|b| b.as_ref().unwrap() as *const MlxArray)
        .unwrap_or(std::ptr::null());
    let down_bias_ptr = down_lin
        .bias
        .as_ref()
        .map(|b| b.as_ref().unwrap() as *const MlxArray)
        .unwrap_or(std::ptr::null());

    Some(unsafe {
        ffi::compiled_gelu_mlp_forward_fp16(
            x,
            &gate_lin.weight,
            &up_lin.weight,
            &down_lin.weight,
            gate_bias_ptr,
            up_bias_ptr,
            down_bias_ptr,
        )
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn insert_quantized_qkv_projection(
        weights: &mut crate::weights::WeightMap,
        prefix: &str,
        out_dim: i32,
    ) {
        use crate::dtype;

        weights.insert(
            format!("{prefix}.weight"),
            ffi::zeros(&[out_dim, 8], dtype::UINT32),
        );
        weights.insert(
            format!("{prefix}.scales"),
            ffi::zeros(&[out_dim, 1], dtype::FLOAT16),
        );
        weights.insert(
            format!("{prefix}.biases"),
            ffi::zeros(&[out_dim, 1], dtype::FLOAT16),
        );
        weights.insert(
            format!("{prefix}.bias"),
            ffi::zeros(&[out_dim], dtype::FLOAT16),
        );
    }

    #[test]
    fn fused_qkv_quantized_preserves_linear_bias() {
        let mut weights = crate::weights::WeightMap::new();
        insert_quantized_qkv_projection(&mut weights, "model.layers.0.self_attn.q_proj", 8);
        insert_quantized_qkv_projection(&mut weights, "model.layers.0.self_attn.k_proj", 4);
        insert_quantized_qkv_projection(&mut weights, "model.layers.0.self_attn.v_proj", 4);

        let fused = FusedQKVLinear::from_weights_separate(
            &weights,
            "model.layers.0.self_attn",
            64,
            4,
            2,
            1,
            4,
        )
        .expect("valid fused QKV weights");

        match fused.qkv_proj {
            UnifiedLinear::Quantized { weight, bias } => {
                let bias = bias.expect("Qwen2-style q/k/v linear bias must be preserved");
                assert_eq!(ffi::array_shape(&bias).as_slice(), &[16]);

                let quant_biases = weight
                    .biases
                    .expect("quantization biases should still be preserved");
                assert_eq!(ffi::array_shape(&quant_biases).as_slice(), &[16, 1]);
            }
            UnifiedLinear::Regular(_) => panic!("expected quantized fused QKV"),
        }
    }

    /// `Embedding::clone_shared` produces an independent handle whose
    /// `forward` result matches the original — the lazy-array share
    /// (`ffi::copy`) points at the same data, so an embedding lookup
    /// through the clone is byte-identical to the original. Used by the
    /// DFlash drafter lazy-bind path.
    #[test]
    fn embedding_clone_shared_matches_original_forward() {
        // [vocab = 4, hidden = 3] embedding table with distinct rows so a
        // lookup actually exercises the shared buffer.
        let weight = ffi::from_slice_f32(
            &[
                0.0, 1.0, 2.0, // row 0
                3.0, 4.0, 5.0, // row 1
                6.0, 7.0, 8.0, // row 2
                9.0, 10.0, 11.0, // row 3
            ],
            &[4, 3],
        );
        let embed = Embedding::new(weight);
        let cloned = embed.clone_shared();

        let ids = ffi::from_slice_i32(&[2, 0], &[1, 2]);
        let orig_out = embed.forward(&ids);
        let clone_out = cloned.forward(&ids);
        ffi::eval(&orig_out);
        ffi::eval(&clone_out);

        assert_eq!(
            ffi::array_shape(&orig_out),
            ffi::array_shape(&clone_out),
            "clone_shared forward shape must match the original",
        );
        // Output is [batch = 1, seq = 2, hidden = 3]: row 2 of the table
        // at seq position 0, row 0 at seq position 1.
        let expected: [[f32; 3]; 2] = [[6.0, 7.0, 8.0], [0.0, 1.0, 2.0]];
        for (s, want_row) in expected.iter().enumerate() {
            for (h, &want) in want_row.iter().enumerate() {
                let elem = ffi::slice(
                    &clone_out,
                    &[0_i32, s as i32, h as i32],
                    &[1_i32, s as i32 + 1, h as i32 + 1],
                );
                ffi::eval(&elem);
                let got = ffi::item_f32(&elem);
                assert!(
                    (got - want).abs() < 1e-6,
                    "clone_shared forward element [seq={s}, hidden={h}]: got {got}, want {want}",
                );
            }
        }
    }

    /// `UnifiedEmbedding::clone_shared` on the `Regular` arm yields a
    /// `Regular` clone (the variant is preserved) usable as a standalone
    /// embedding. This is the exact shape the Qwen 3.5 target hands to a
    /// lazy-bind DFlash drafter.
    #[test]
    fn unified_embedding_clone_shared_preserves_regular_variant() {
        use crate::dtype;

        let weight = ffi::zeros(&[4, 3], dtype::FLOAT32);
        let unified = UnifiedEmbedding::Regular(Embedding::new(weight));
        let cloned = unified.clone_shared();

        assert!(
            !cloned.is_quantized(),
            "clone_shared of a Regular UnifiedEmbedding must stay Regular",
        );
        // The clone is a usable embedding: a lookup returns the right shape.
        let ids = ffi::from_slice_i32(&[0], &[1, 1]);
        let out = cloned.forward(&ids);
        ffi::eval(&out);
        assert_eq!(ffi::array_shape(&out).as_slice(), &[1, 1, 3]);
    }

    /// Verify that `metal4_attention` falls back correctly on all hardware.
    ///
    /// The test creates small Q / K / V arrays and checks that the function
    /// returns an array with the expected shape without panicking.  On current
    /// hardware (M1–M4) `use_metal4` will be false; on future M5 hardware with
    /// macOS 26.2+ it will be true — both paths currently delegate to the same
    /// MLX fast SDPA implementation, so the result must be identical.
    #[test]
    fn metal4_attention_does_not_panic() {
        use crate::dtype;
        // [batch=1, heads=2, seq=4, head_dim=8]
        let q = ffi::zeros(&[1, 2, 4, 8], dtype::FLOAT16);
        let k = ffi::zeros(&[1, 2, 4, 8], dtype::FLOAT16);
        let v = ffi::zeros(&[1, 2, 4, 8], dtype::FLOAT16);
        let scale = 1.0 / 8.0_f32.sqrt();

        let out = metal4_attention(&q, &k, &v, scale, None, 0.0, 0);

        let shape = ffi::array_shape(&out);
        assert_eq!(shape.as_slice(), &[1, 2, 4, 8]);
    }

    /// Confirm that hardware detection does not interfere with the fallback path.
    #[test]
    fn metal4_attention_output_matches_fast_sdpa() {
        use crate::dtype;
        let q = ffi::zeros(&[1, 1, 2, 4], dtype::FLOAT16);
        let k = ffi::zeros(&[1, 1, 2, 4], dtype::FLOAT16);
        let v = ffi::zeros(&[1, 1, 2, 4], dtype::FLOAT16);
        let scale = 0.5_f32;

        // Both calls should produce the same result because the Metal 4 kernel
        // body is not yet implemented — both paths fall back to fast SDPA.
        let out_m4 = metal4_attention(&q, &k, &v, scale, None, 0.0, 0);
        let out_fast =
            unsafe { ffi::fast_scaled_dot_product_attention(&q, &k, &v, scale, std::ptr::null()) };

        // Compare shapes (we cannot easily compare values without eval + copy,
        // but shape equality is sufficient to confirm the dispatch works).
        assert_eq!(ffi::array_shape(&out_m4), ffi::array_shape(&out_fast));
    }

    /// Verify GQA pattern: n_heads=4 Q heads with n_kv_heads=2 KV heads.
    /// MLX SDPA handles the head broadcasting internally.
    #[test]
    fn metal4_attention_gqa_shape() {
        use crate::dtype;
        // GQA: 4 Q heads, 2 KV heads, seq=3, head_dim=8
        let q = ffi::zeros(&[1, 4, 3, 8], dtype::FLOAT16);
        let k = ffi::zeros(&[1, 2, 3, 8], dtype::FLOAT16);
        let v = ffi::zeros(&[1, 2, 3, 8], dtype::FLOAT16);
        let scale = 1.0 / 8.0_f32.sqrt();

        let out = metal4_attention(&q, &k, &v, scale, None, 0.0, 0);
        let shape = ffi::array_shape(&out);
        // Output should have Q's head count, not KV's
        assert_eq!(shape.as_slice(), &[1, 4, 3, 8]);
    }

    /// Verify the unified attention helper preserves Gemma-style softcap GQA shapes.
    #[test]
    fn attention_softcap_gqa_shape() {
        use crate::dtype;
        let q = ffi::zeros(&[1, 4, 3, 8], dtype::FLOAT16);
        let k = ffi::zeros(&[1, 2, 3, 8], dtype::FLOAT16);
        let v = ffi::zeros(&[1, 2, 3, 8], dtype::FLOAT16);

        let out = attention(&q, &k, &v, 0.5, None, 30.0, 0);
        assert_eq!(ffi::array_shape(&out).as_slice(), &[1, 4, 3, 8]);
    }

    /// Verify the public causal SDPA wrapper preserves the original shape
    /// contract while routing through the centralized attention dispatcher.
    #[test]
    fn causal_attention_wrapper_matches_fast_causal_shape() {
        use crate::dtype;
        let q = ffi::zeros(&[1, 2, 3, 8], dtype::FLOAT16);
        let k = ffi::zeros(&[1, 2, 5, 8], dtype::FLOAT16);
        let v = ffi::zeros(&[1, 2, 5, 8], dtype::FLOAT16);
        let scale = 1.0 / 8.0_f32.sqrt();

        let out = crate::fast_scaled_dot_product_attention_causal(&q, &k, &v, scale);
        let out_fast = ffi::ffi_fast_scaled_dot_product_attention_causal(&q, &k, &v, scale);

        assert_eq!(ffi::array_shape(&out), ffi::array_shape(&out_fast));
    }

    #[test]
    fn causal_attention_single_query_matches_explicit_mask() {
        let q = crate::from_slice_f32(&[0.1, -0.2, 0.3, -0.1], &[1, 1, 1, 4]);
        let k = crate::from_slice_f32(
            &[
                0.2, 0.0, -0.1, 0.3, 0.1, -0.2, 0.2, 0.0, -0.3, 0.2, 0.1, -0.2, 0.4, -0.1, 0.0, 0.2,
            ],
            &[1, 1, 4, 4],
        );
        let v = crate::from_slice_f32(
            &[
                0.5, 0.1, -0.3, 0.2, -0.2, 0.4, 0.3, -0.1, 0.1, -0.5, 0.2, 0.6, -0.4, 0.2, 0.1,
                -0.2,
            ],
            &[1, 1, 4, 4],
        );
        let scale = 0.5_f32;

        let out_fast = crate::causal_attention(&q, &k, &v, scale, 0.0, 0);
        let mask = crate::utils::create_causal_mask(1, 3);
        let out_masked = attention(&q, &k, &v, scale, Some(mask.as_ref().unwrap()), 0.0, 0);

        let diff = crate::subtract(out_fast.as_ref().unwrap(), out_masked.as_ref().unwrap());
        let diff_abs = crate::abs(&diff);
        let diff_sum = crate::sum_all(&diff_abs);
        crate::eval(&diff_sum);
        assert!(
            crate::item_f32(&diff_sum) < 1e-5,
            "single-query causal fast path should match explicit mask path"
        );
    }

    #[test]
    fn causal_attention_single_query_softcap_matches_explicit_mask() {
        let q = crate::from_slice_f32(&[0.1, -0.2, 0.3, -0.1], &[1, 1, 1, 4]);
        let k = crate::from_slice_f32(
            &[
                0.2, 0.0, -0.1, 0.3, 0.1, -0.2, 0.2, 0.0, -0.3, 0.2, 0.1, -0.2, 0.4, -0.1, 0.0, 0.2,
            ],
            &[1, 1, 4, 4],
        );
        let v = crate::from_slice_f32(
            &[
                0.5, 0.1, -0.3, 0.2, -0.2, 0.4, 0.3, -0.1, 0.1, -0.5, 0.2, 0.6, -0.4, 0.2, 0.1,
                -0.2,
            ],
            &[1, 1, 4, 4],
        );
        let scale = 0.5_f32;
        let softcap = 30.0_f32;

        let out_fast = crate::causal_attention(&q, &k, &v, scale, softcap, 0);
        let mask = crate::utils::create_causal_mask(1, 3);
        let out_masked = attention(&q, &k, &v, scale, Some(mask.as_ref().unwrap()), softcap, 0);

        let diff = crate::subtract(out_fast.as_ref().unwrap(), out_masked.as_ref().unwrap());
        let diff_abs = crate::abs(&diff);
        let diff_sum = crate::sum_all(&diff_abs);
        crate::eval(&diff_sum);
        assert!(
            crate::item_f32(&diff_sum) < 1e-5,
            "single-query softcap fast path should match explicit mask path"
        );
    }

    #[test]
    fn causal_attention_single_query_window_matches_explicit_mask() {
        let q = crate::from_slice_f32(&[0.2, 0.1, -0.4, 0.3], &[1, 1, 1, 4]);
        let k = crate::from_slice_f32(
            &[
                0.1, 0.0, -0.1, 0.2, -0.2, 0.3, 0.0, 0.1, 0.3, -0.1, 0.2, 0.0, -0.1, 0.4, -0.2, 0.2,
            ],
            &[1, 1, 4, 4],
        );
        let v = crate::from_slice_f32(
            &[
                0.3, -0.1, 0.2, 0.4, -0.1, 0.2, -0.3, 0.1, 0.5, 0.0, -0.2, 0.2, -0.4, 0.3, 0.1,
                -0.1,
            ],
            &[1, 1, 4, 4],
        );
        let scale = 0.5_f32;
        let window_size = 8_i32; // k_len <= window, so fast path should skip mask creation.

        let out_fast = crate::causal_attention(&q, &k, &v, scale, 0.0, window_size);
        let mask = crate::utils::create_causal_mask_with_window(1, 3, Some(window_size));
        let out_masked = attention(
            &q,
            &k,
            &v,
            scale,
            Some(mask.as_ref().unwrap()),
            0.0,
            window_size,
        );

        let diff = crate::subtract(out_fast.as_ref().unwrap(), out_masked.as_ref().unwrap());
        let diff_abs = crate::abs(&diff);
        let diff_sum = crate::sum_all(&diff_abs);
        crate::eval(&diff_sum);
        assert!(
            crate::item_f32(&diff_sum) < 1e-5,
            "single-query window fast path should match explicit mask path"
        );
    }

    #[test]
    fn causal_attention_single_query_window_respects_mask_when_needed() {
        let q = crate::from_slice_f32(&[0.2, -0.1, 0.4, 0.0], &[1, 1, 1, 4]);
        let k = crate::from_slice_f32(
            &[
                0.1, 0.1, 0.0, -0.1, -0.2, 0.3, 0.1, 0.0, 0.4, -0.2, 0.2, 0.1, -0.3, 0.2, 0.3, -0.1,
            ],
            &[1, 1, 4, 4],
        );
        let v = crate::from_slice_f32(
            &[
                0.2, 0.0, -0.2, 0.4, 0.1, -0.1, 0.3, -0.2, 0.5, 0.2, -0.1, 0.1, -0.3, 0.4, 0.0,
                -0.1,
            ],
            &[1, 1, 4, 4],
        );
        let scale = 0.5_f32;
        let window_size = 2_i32; // k_len > window, mask is required.

        let out = crate::causal_attention(&q, &k, &v, scale, 0.0, window_size);
        // After `create_causal_mask_with_window` caps T_k to
        // `window_size` when `q_len + offset > window`. The explicit-mask
        // reference must therefore receive K/V sliced to the last
        // `window_size` slots so its score tensor lines up with the mask —
        // this mirrors what `causal_attention` now does internally.
        let mask = crate::utils::create_causal_mask_with_window(1, 3, Some(window_size));
        let k_sliced = crate::slice(&k, &[0, 0, 4 - window_size, 0], &[1, 1, 4, 4]);
        let v_sliced = crate::slice(&v, &[0, 0, 4 - window_size, 0], &[1, 1, 4, 4]);
        let out_masked = attention(
            &q,
            &k_sliced,
            &v_sliced,
            scale,
            Some(mask.as_ref().unwrap()),
            0.0,
            window_size,
        );

        let diff = crate::subtract(out.as_ref().unwrap(), out_masked.as_ref().unwrap());
        let diff_abs = crate::abs(&diff);
        let diff_sum = crate::sum_all(&diff_abs);
        crate::eval(&diff_sum);
        assert!(
            crate::item_f32(&diff_sum) < 1e-5,
            "single-query window path should keep explicit mask semantics when k_len > window"
        );
    }

    #[test]
    fn causal_attention_multi_token_over_window_uses_full_windowed_mask() {
        // Multi-token prefill (`q_len = 3`) whose key length (`k_len = 5`)
        // exceeds the window. Before issue #408 `causal_attention` sliced K/V
        // to the trailing `window_size` keys and used the clamped
        // `(q_len, window_size)` mask, which strands the earliest query rows.
        // The corrected behavior keeps all keys and builds the full
        // `(q_len, k_len)` windowed-causal mask, so the reference must use the
        // full K/V and `create_causal_mask_with_window_full`.
        let q = crate::from_slice_f32(
            &[
                0.1, -0.2, 0.3, 0.4, 0.2, 0.0, -0.1, 0.5, -0.3, 0.1, 0.2, -0.4,
            ],
            &[1, 1, 3, 4],
        );
        let k = crate::from_slice_f32(
            &[
                0.2, 0.1, -0.1, 0.0, -0.2, 0.3, 0.1, -0.1, 0.0, -0.3, 0.2, 0.4, 0.1, 0.2, -0.2,
                0.3, -0.1, 0.0, 0.4, -0.2,
            ],
            &[1, 1, 5, 4],
        );
        let v = crate::from_slice_f32(
            &[
                0.5, -0.2, 0.1, 0.0, -0.1, 0.3, 0.2, -0.3, 0.2, 0.1, -0.4, 0.5, 0.3, -0.1, 0.0,
                0.2, -0.2, 0.4, 0.1, -0.1,
            ],
            &[1, 1, 5, 4],
        );
        let scale = 0.5_f32;
        // window_size=2 is the previously-degenerate case: under the old
        // clamp+slice path query row 0 (logical position 0) had an all-`-inf`
        // mask row and softmaxed to NaN. The full mask gives it its own
        // window, so no row is fully masked.
        let window_size = 2_i32;

        let out = crate::causal_attention(&q, &k, &v, scale, 0.0, window_size);
        // Reference: full key set + uncapped windowed mask over offset =
        // k_len - q_len = 2.
        let mask_full = crate::utils::create_causal_mask_with_window_full(3, 2, Some(window_size));
        let out_ref = attention(
            &q,
            &k,
            &v,
            scale,
            Some(mask_full.as_ref().unwrap()),
            0.0,
            window_size,
        );

        let diff = crate::subtract(out.as_ref().unwrap(), out_ref.as_ref().unwrap());
        let diff_abs = crate::abs(&diff);
        let diff_sum = crate::sum_all(&diff_abs);
        crate::eval(&diff_sum);
        let total = crate::item_f32(&diff_sum);
        assert!(
            total.is_finite() && total < 1e-5,
            "multi-token over-window causal attention should match the full \
             windowed-mask reference (no NaN, no stranded rows); diff_sum = {total}"
        );

        // The output itself must be finite (the pre-#408 path produced NaN for
        // the earliest query row under window_size=2).
        let out_abs = crate::abs(out.as_ref().unwrap());
        let out_sum = crate::sum_all(&out_abs);
        crate::eval(&out_sum);
        assert!(
            crate::item_f32(&out_sum).is_finite(),
            "multi-token over-window causal attention output must be finite"
        );
    }

    #[test]
    fn softcap_gqa_decode_matches_explicit_repeat_kv_reference() {
        let q = crate::from_slice_f32(
            &[
                0.2, -0.1, 0.3, 0.0, 0.1, 0.4, -0.2, 0.2, -0.3, 0.2, 0.1, -0.1, 0.0, -0.2, 0.5, 0.3,
            ],
            &[1, 4, 1, 4],
        );
        let k = crate::from_slice_f32(
            &[
                0.1, 0.0, -0.2, 0.3, -0.1, 0.2, 0.1, 0.0, 0.2, -0.3, 0.0, 0.4, 0.3, 0.1, -0.1, -0.2,
            ],
            &[1, 2, 2, 4],
        );
        let v = crate::from_slice_f32(
            &[
                0.5, -0.1, 0.2, 0.0, -0.2, 0.4, 0.1, -0.3, 0.1, 0.3, -0.4, 0.2, 0.2, -0.2, 0.0, 0.5,
            ],
            &[1, 2, 2, 4],
        );

        let scale = 0.5_f32;
        let softcap = 30.0_f32;

        let out_new = attention(&q, &k, &v, scale, None, softcap, 0);
        let rk = crate::utils::repeat_kv(&k, 2);
        let rv = crate::utils::repeat_kv(&v, 2);
        let out_ref = unsafe {
            ffi::compiled_softcap_sdpa(
                &q,
                rk.as_ref().unwrap(),
                rv.as_ref().unwrap(),
                scale,
                softcap,
                std::ptr::null(),
            )
        };

        let diff = crate::subtract(out_new.as_ref().unwrap(), out_ref.as_ref().unwrap());
        let diff_abs = crate::abs(&diff);
        let diff_sum = crate::sum_all(&diff_abs);
        crate::eval(&diff_sum);
        let diff_val = crate::item_f32(&diff_sum);
        assert!(
            diff_val < 1e-5,
            "decode GQA softcap path should match explicit repeat_kv reference (diff_sum={diff_val})"
        );
    }

    /// Verify that the detection condition uses the canonical NA check
    /// (has_neural_accelerator && macos_supports_na) rather than just
    /// metal_version >= 4.
    #[test]
    fn metal4_detection_uses_canonical_na_check() {
        let hw = crate::hardware::get_hardware();
        // On test hardware (M1-M4), use_metal4 should be false because
        // has_neural_accelerator is false.  On M5 with macOS 26.2+, both
        // conditions are true.  Either way, the function should not panic.
        let _use_metal4 = hw.has_neural_accelerator && hw.macos_supports_na;

        // Verify the function works regardless of the detection result
        use crate::dtype;
        let q = ffi::zeros(&[1, 1, 1, 4], dtype::FLOAT16);
        let k = ffi::zeros(&[1, 1, 1, 4], dtype::FLOAT16);
        let v = ffi::zeros(&[1, 1, 1, 4], dtype::FLOAT16);
        let out = metal4_attention(&q, &k, &v, 0.5, None, 0.0, 0);
        assert_eq!(ffi::array_shape(&out).as_slice(), &[1, 1, 1, 4]);
    }

    /// Caller-supplied bits consistent with shapes → pass through unchanged.
    /// Shapes taken from qwen3.5 out_proj: u32 weight (2048, 512), scales (2048, 64).
    /// Invariant: `packed_in * 32 == bits * num_groups * group_size`
    ///            `512 * 32 == 4 * 64 * 64`  (16384 == 16384)
    #[test]
    fn infer_bits_pass_through_when_consistent() {
        let bits = infer_quantization_bits(&[2048, 512], &[2048, 64], 64, 4).unwrap();
        assert_eq!(bits, 4);
    }

    /// Qwen3.5/3.6 MoE gates store the router at 8-bit while the rest of the
    /// model is 4-bit. The loader's caller_bits reflects the top-level config
    /// (4), so we must detect the 8-bit override from the tensor shapes.
    /// Shapes taken from real qwen3.6 tensors.
    #[test]
    fn infer_bits_detects_per_layer_8bit_override() {
        // mlp.gate: u32 weight (256, 512), scales (256, 32), gs=64 → 8-bit
        // 512 * 32 == 8 * 32 * 64  (16384 == 16384)
        let bits = infer_quantization_bits(&[256, 512], &[256, 32], 64, 4).unwrap();
        assert_eq!(bits, 8, "router gate is 8-bit under top-level 4-bit config");

        // mlp.shared_expert_gate: u32 weight (1, 512), scales (1, 32) → 8-bit
        let bits = infer_quantization_bits(&[1, 512], &[1, 32], 64, 4).unwrap();
        assert_eq!(
            bits, 8,
            "shared_expert_gate is 8-bit under top-level 4-bit config"
        );
    }

    /// Mixed-precision exports (e.g. diffusiongemma / gemma4) store the
    /// embedding at 8-bit while the top-level config default is 4-bit. The
    /// quantized-embedding loader must detect the override from the shapes, or
    /// the embedding lookup and tied lm_head dequantize at the wrong bits and
    /// abort (issue #291). Shapes from real diffusiongemma-26B-A4B-it-4bit.
    #[test]
    fn infer_bits_detects_8bit_embedding_under_4bit_config() {
        // embed_tokens: u32 weight (262144, 704), scales (262144, 44), gs=64
        // 704 * 32 == 8 * 44 * 64  (22528 == 22528)
        let bits = infer_quantization_bits(&[262144, 704], &[262144, 44], 64, 4).unwrap();
        assert_eq!(bits, 8, "embedding is 8-bit under top-level 4-bit config");
    }

    /// 3D MoE expert weights (switch_mlp.*_proj) must use the last two axes.
    #[test]
    fn infer_bits_handles_3d_moe_experts() {
        // down_proj: weight (256, 2048, 64) u32 / scales (256, 2048, 8)
        // bits = 32 * 64 / (8 * 64) = 4
        let bits = infer_quantization_bits(&[256, 2048, 64], &[256, 2048, 8], 64, 4).unwrap();
        assert_eq!(bits, 4);

        // gate_proj: weight (256, 512, 256) / scales (256, 512, 32)
        // bits = 32 * 256 / (32 * 64) = 4
        let bits = infer_quantization_bits(&[256, 512, 256], &[256, 512, 32], 64, 4).unwrap();
        assert_eq!(bits, 4);
    }

    /// Invalid inferred bits → error (prevents silently accepting a bogus
    /// `group_size` that happens to satisfy the arithmetic).
    #[test]
    fn infer_bits_rejects_non_canonical_widths() {
        // packed_in * 32 / (num_groups * group_size) = 32*4 / (32*2) = 2 is valid,
        // so use a combination that yields 1 (invalid).
        // packed_in=32, num_groups=32, group_size=32 → 32*32/(32*32) = 1.
        let err = infer_quantization_bits(&[16, 32], &[16, 32], 32, 4);
        assert!(err.is_err(), "bits=1 should be rejected");
    }

    /// Empty/zero shapes are treated as pass-through (defensive — real arrays
    /// never have empty shape but we should not panic if called eagerly).
    #[test]
    fn infer_bits_pass_through_on_empty() {
        assert_eq!(infer_quantization_bits(&[], &[], 64, 4).unwrap(), 4);
        assert_eq!(infer_quantization_bits(&[0, 0], &[0, 0], 64, 4).unwrap(), 4);
    }

    // ---- #326: generic fused QKV + RMSNorm + RoPE ----

    /// f16 array from f32 values (the activation/scale dtype the fused kernel
    /// runs in).
    fn f16_from(values: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
        use crate::dtype;
        ffi::astype(&ffi::from_slice_f32(values, shape), dtype::FLOAT16)
    }

    /// Quantized fused QKV built by quantizing a non-trivial float weight with
    /// MLX's own `quantize`, so the projection is a valid, non-zero matmul that
    /// varies per output channel. A meaningful projection is what lets the Q/K
    /// RMSNorm comparison exercise the `(1 + weight)` vs `weight` choice (a zero
    /// projection would normalize to zero regardless). The 4-bit quantization
    /// error cancels in the fused-vs-graph comparison because both paths share
    /// this exact dequantized weight.
    fn synthetic_quantized_fused_qkv(
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
    ) -> FusedQKVLinear {
        let in_dim = 64;
        let group_size = in_dim; // single group
        let bits = 4;
        let out_dim = (n_heads + 2 * n_kv_heads) * head_dim;

        // Vary per (out, in) so the projection (and each per-head head_dim slice)
        // is non-degenerate.
        let mut w_vals = Vec::with_capacity((out_dim * in_dim) as usize);
        for o in 0..out_dim {
            for i in 0..in_dim {
                w_vals.push(0.01 * (((o * 13 + i * 7) % 23) as f32 - 11.0));
            }
        }
        let w = f16_from(&w_vals, &[out_dim, in_dim]);
        let weight = ffi::quantize_weights_w(&w, group_size, bits);
        let scales = ffi::quantize_weights_scales(&w, group_size, bits);
        let biases = ffi::quantize_weights_biases(&w, group_size, bits);
        let qweight = QuantizedWeight::new(weight, scales, biases, group_size, bits);
        FusedQKVLinear {
            qkv_proj: UnifiedLinear::new(qweight, None),
            n_heads,
            n_kv_heads,
            head_dim,
        }
    }

    /// Max absolute element, used as a non-degeneracy guard so a comparison
    /// test cannot pass by silently normalizing an all-zero projection.
    fn max_abs(a: &MlxArray) -> f32 {
        scalar_f32(&ffi::max_all(&ffi::abs(a)))
    }

    /// Read a scalar array as f32. The decode arrays are f16, and
    /// `item_f32` (MLX `item<float>()`) reinterprets the buffer rather than
    /// casting, so the scalar must be cast to f32 first.
    fn scalar_f32(a: &MlxArray) -> f32 {
        use crate::dtype;
        let f = ffi::astype(a, dtype::FLOAT32);
        ffi::eval(&f);
        ffi::item_f32(&f)
    }

    fn rms_diff(a: &MlxArray, b: &MlxArray) -> f32 {
        scalar_f32(&ffi::sqrt(&ffi::mean_all(&ffi::square(&ffi::subtract(
            a, b,
        )))))
    }

    fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
        scalar_f32(&ffi::max_all(&ffi::abs(&ffi::subtract(a, b))))
    }

    /// Projection + reshape to `[b, l, heads, head_dim]`, mirroring the model
    /// graph fallback before the head transpose.
    fn project_reshape_heads(
        qkv: &FusedQKVLinear,
        x: &MlxArray,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let shape = ffi::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let (q, k, v) = qkv.forward(x);
        let q = ffi::reshape(&q, &[b, l, qkv.n_heads, qkv.head_dim]);
        let k = ffi::reshape(&k, &[b, l, qkv.n_kv_heads, qkv.head_dim]);
        let v = ffi::reshape(&v, &[b, l, qkv.n_kv_heads, qkv.head_dim]);
        (q, k, v)
    }

    fn transpose_then_rope(
        arr: &MlxArray,
        rope_dims: i32,
        rope_base: f32,
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        let t = ffi::transpose_axes(arr, &[0, 2, 1, 3]);
        ffi::fast_rope(&t, rope_dims, false, rope_base, 1.0, offset)
    }

    /// The generalized fused primitive with a standard `RMSNorm` Q/K norm must
    /// match the explicit graph sequence (projection -> split -> q/k RMSNorm ->
    /// transpose -> RoPE) within RMS < 5e-3. This is the Qwen3 / Qwen3-MoE
    /// decode path (#326). RoPE runs at a non-zero offset so the rotation is
    /// actually exercised.
    #[test]
    fn fused_qkv_split_norm_rope_standard_rmsnorm_matches_graph() {
        let (n_heads, n_kv_heads, head_dim) = (2, 1, 8);
        let (rope_dims, rope_base, offset, eps) = (head_dim, 10000.0_f32, 7, 1e-6_f32);

        let qkv = synthetic_quantized_fused_qkv(n_heads, n_kv_heads, head_dim);
        let q_norm_w: Vec<f32> = (0..head_dim).map(|i| 0.5 + 0.1 * i as f32).collect();
        let k_norm_w: Vec<f32> = (0..head_dim).map(|i| 0.7 + 0.05 * i as f32).collect();
        let q_norm = RMSNorm::new(f16_from(&q_norm_w, &[head_dim]), eps);
        let k_norm = RMSNorm::new(f16_from(&k_norm_w, &[head_dim]), eps);

        let x_vals: Vec<f32> = (0..64).map(|c| 0.2 + 0.01 * c as f32).collect();
        let x = f16_from(&x_vals, &[1, 1, 64]);

        // Graph reference: norm via the real RMSNorm::forward (raw weight),
        // BEFORE the head transpose, exactly like the model fallback.
        let (rq, rk, rv) = project_reshape_heads(&qkv, &x);
        let rq = q_norm.forward(&rq);
        let rk = k_norm.forward(&rk);
        let ref_q = transpose_then_rope(&rq, rope_dims, rope_base, offset);
        let ref_k = transpose_then_rope(&rk, rope_dims, rope_base, offset);
        let ref_v = ffi::transpose_axes(&rv, &[0, 2, 1, 3]);

        // Guard: a trivially-zero projection would make the RMS comparison vacuous.
        assert!(
            max_abs(&ref_q) > 0.1,
            "reference Q must be non-degenerate (max-abs {})",
            max_abs(&ref_q)
        );

        let (fq, fk, fv) = qkv
            .forward_split_norm_rope_quantized(&x, &q_norm, &k_norm, rope_dims, rope_base, offset)
            .expect("quantized fused QKV must take the fused path");

        let (dq, dk, dv) = (
            rms_diff(&fq, &ref_q),
            rms_diff(&fk, &ref_k),
            rms_diff(&fv, &ref_v),
        );
        assert!(dq < 5e-3, "fused Q vs graph Q RMS too large: {dq}");
        assert!(dk < 5e-3, "fused K vs graph K RMS too large: {dk}");
        assert!(dv < 5e-3, "fused V vs graph V RMS too large: {dv}");
    }

    /// The existing `GemmaRMSNorm` path must still match its graph reference
    /// after the primitive was generalized: no Gemma regression (#326). The
    /// graph reference uses `GemmaRMSNorm::forward`, which applies `(1 + weight)`.
    #[test]
    fn fused_qkv_split_norm_rope_gemma_unchanged() {
        let (n_heads, n_kv_heads, head_dim) = (2, 1, 8);
        let (rope_dims, rope_base, offset, eps) = (head_dim, 10000.0_f32, 7, 1e-6_f32);

        let qkv = synthetic_quantized_fused_qkv(n_heads, n_kv_heads, head_dim);
        let q_norm_w: Vec<f32> = (0..head_dim).map(|i| -0.4 + 0.1 * i as f32).collect();
        let k_norm_w: Vec<f32> = (0..head_dim).map(|i| -0.3 + 0.08 * i as f32).collect();
        let q_norm = GemmaRMSNorm::new(f16_from(&q_norm_w, &[head_dim]), eps);
        let k_norm = GemmaRMSNorm::new(f16_from(&k_norm_w, &[head_dim]), eps);

        let x_vals: Vec<f32> = (0..64).map(|c| 0.2 + 0.01 * c as f32).collect();
        let x = f16_from(&x_vals, &[1, 1, 64]);

        let (rq, rk, rv) = project_reshape_heads(&qkv, &x);
        let rq = q_norm.forward(&rq);
        let rk = k_norm.forward(&rk);
        let ref_q = transpose_then_rope(&rq, rope_dims, rope_base, offset);
        let ref_k = transpose_then_rope(&rk, rope_dims, rope_base, offset);
        let ref_v = ffi::transpose_axes(&rv, &[0, 2, 1, 3]);

        assert!(
            max_abs(&ref_q) > 0.1,
            "reference Q must be non-degenerate (max-abs {})",
            max_abs(&ref_q)
        );

        let (fq, fk, fv) = qkv
            .forward_split_norm_rope_quantized(&x, &q_norm, &k_norm, rope_dims, rope_base, offset)
            .expect("quantized fused QKV must take the fused path");

        let (dq, dk, dv) = (
            rms_diff(&fq, &ref_q),
            rms_diff(&fk, &ref_k),
            rms_diff(&fv, &ref_v),
        );
        assert!(dq < 5e-3, "Gemma fused Q vs graph Q RMS too large: {dq}");
        assert!(dk < 5e-3, "Gemma fused K vs graph K RMS too large: {dk}");
        assert!(dv < 5e-3, "Gemma fused V vs graph V RMS too large: {dv}");
    }

    /// The `(1 + weight)` Gemma offset must actually change the fused output:
    /// feeding the SAME raw weights as a standard `RMSNorm` vs a `GemmaRMSNorm`
    /// produces materially different Q/K. Guards against the primitive silently
    /// hardcoding or dropping the Gemma offset, the core correctness risk in
    /// #326.
    #[test]
    fn fused_qkv_norm_variant_selects_one_plus_weight() {
        let (n_heads, n_kv_heads, head_dim) = (2, 1, 8);
        let (rope_dims, rope_base, offset, eps) = (head_dim, 10000.0_f32, 7, 1e-6_f32);

        let qkv = synthetic_quantized_fused_qkv(n_heads, n_kv_heads, head_dim);
        let norm_w: Vec<f32> = (0..head_dim).map(|i| 0.5 + 0.1 * i as f32).collect();
        let std_q = RMSNorm::new(f16_from(&norm_w, &[head_dim]), eps);
        let std_k = RMSNorm::new(f16_from(&norm_w, &[head_dim]), eps);
        let gemma_q = GemmaRMSNorm::new(f16_from(&norm_w, &[head_dim]), eps);
        let gemma_k = GemmaRMSNorm::new(f16_from(&norm_w, &[head_dim]), eps);

        let x_vals: Vec<f32> = (0..64).map(|c| 0.2 + 0.01 * c as f32).collect();
        let x = f16_from(&x_vals, &[1, 1, 64]);

        let (sq, sk, _sv) = qkv
            .forward_split_norm_rope_quantized(&x, &std_q, &std_k, rope_dims, rope_base, offset)
            .expect("standard fused path");
        let (gq, gk, _gv) = qkv
            .forward_split_norm_rope_quantized(&x, &gemma_q, &gemma_k, rope_dims, rope_base, offset)
            .expect("gemma fused path");

        // Guard: both outputs must be non-degenerate, else the difference below
        // would be a meaningless 0 vs 0.
        assert!(
            max_abs(&sq) > 0.1,
            "standard fused Q must be non-degenerate (max-abs {})",
            max_abs(&sq)
        );

        // Same projection, same raw norm weights: the only difference is the
        // `(1 + weight)` Gemma offset, which must move the output well beyond
        // f16 noise.
        let (dq, dk) = (max_abs_diff(&sq, &gq), max_abs_diff(&sk, &gk));
        assert!(dq > 0.05, "standard vs Gemma Q must differ (max-abs {dq})");
        assert!(dk > 0.05, "standard vs Gemma K must differ (max-abs {dk})");
    }

    /// The `fused_qk_norm_enabled_from` helper must be default-on when the env
    /// var is absent and must respect the recognised disable strings.
    #[test]
    fn fused_qk_norm_enabled_defaults_off_and_respects_enable_values() {
        // Unset -> off (default-off, opt-in).
        assert!(!fused_qk_norm_enabled_from(None));
        // Recognised enable values (case-insensitive, trimmed) -> on.
        for v in ["1", "true", "on", "yes", "ON", "True", " 1 ", "Yes"] {
            assert!(
                fused_qk_norm_enabled_from(Some(v)),
                "{v:?} should enable the fused QK-norm kernel"
            );
        }
        // Any other value, including empty string -> off.
        for v in ["0", "false", "off", "no", "", "anything"] {
            assert!(
                !fused_qk_norm_enabled_from(Some(v)),
                "{v:?} should keep the fused QK-norm kernel off"
            );
        }
    }
}
