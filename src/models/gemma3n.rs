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

//! Gemma 3n (Nano) model implementation using mlxcel-core
//!
//! Key features:
//! - AltUp (Alternating Updates): Multiple parallel inputs with predict/correct mechanism
//! - LAUREL (Learned Augmented Residual Layer): Low-rank augmentation
//! - Per-layer inputs: Layer-specific input embeddings
//! - KV sharing: Some layers share KV cache
//! - Sliding window + Full attention mix
//! - gelu_topk activation with sparsity pattern
//! - Logit softcapping

use crate::models::gemma3n_helpers::{
    apply_softcap, compute_magnitude, mean_arrays, normalize_magnitudes,
    normalize_magnitudes_from_idx, slice_altup_plane, slice_layer_input,
    split_altup_after_per_layer_update, split_altup_planes, stack_arrays,
};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, Linear, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_sliding_window_prefill_mask_dense};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: serde_json::Value, // Can be int or list
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,
    pub num_kv_shared_layers: usize,
    pub vocab_size_per_layer_input: usize,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub rope_local_base_freq: f32,
    pub rope_theta: f32,
    pub final_logit_softcapping: Option<f32>,
    pub layer_types: Vec<String>,
    pub activation_sparsity_pattern: Option<Vec<f32>>,
    pub hidden_size_per_layer_input: usize,
    pub altup_num_inputs: usize,
    pub altup_coef_clip: Option<f32>,
    pub altup_correct_scale: bool,
    pub altup_active_idx: usize,
    pub laurel_rank: usize,

    #[serde(default)]
    pub quantization: Option<QuantizationArgs>,
}

impl TextConfig {
    pub fn get_intermediate_size(&self, layer_idx: usize) -> usize {
        match &self.intermediate_size {
            serde_json::Value::Number(n) => n.as_u64().unwrap() as usize,
            serde_json::Value::Array(arr) => arr[layer_idx].as_u64().unwrap() as usize,
            _ => panic!("Invalid intermediate_size"),
        }
    }

    pub fn get_activation_sparsity(&self, layer_idx: usize) -> f32 {
        self.activation_sparsity_pattern
            .as_ref()
            .map(|p| p[layer_idx])
            .unwrap_or(0.0)
    }

    pub fn first_kv_shared_layer_idx(&self) -> usize {
        self.num_hidden_layers - self.num_kv_shared_layers
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationArgs {
    pub group_size: usize,
    pub bits: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub text_config: serde_json::Value,
    #[serde(default)]
    pub quantization: Option<RootQuantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RootQuantization {
    pub group_size: usize,
    pub bits: usize,
}

impl ModelArgs {
    pub fn text_args(&self) -> TextConfig {
        let mut config: TextConfig =
            serde_json::from_value(self.text_config.clone()).expect("Failed to parse text_config");
        // Apply root quantization if text_config doesn't have it
        if config.quantization.is_none()
            && let Some(ref q) = self.quantization
        {
            config.quantization = Some(QuantizationArgs {
                group_size: q.group_size,
                bits: q.bits,
            });
        }
        config
    }
}

// Helper functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

/// Inverse error function approximation (Winitzki approximation)
fn erfinv(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return f32::INFINITY;
    }
    if x <= -1.0 {
        return f32::NEG_INFINITY;
    }

    let a = 0.147;
    let ln_one_minus_x2 = (1.0 - x * x).ln();
    let term_a = 2.0 / (std::f32::consts::PI * a) + ln_one_minus_x2 / 2.0;
    let term_b = ln_one_minus_x2 / a;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * ((term_a * term_a - term_b).sqrt() - term_a).sqrt()
}

// RMSNoScale - RMSNorm without learnable scale (uses unit weight).
// Used by: Gemma3n text attention v_norm, Gemma3n VLM projection norm.
pub struct RMSNoScale {
    pub eps: f32,
}

impl RMSNoScale {
    pub fn new(_dim: i32, eps: f32) -> Self {
        Self { eps }
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::fast_rms_norm_no_weight(x, self.eps)
    }
}

// LAUREL (Learned Augmented Residual Layer).
pub struct LaurelBlock {
    pub linear_left: UnifiedLinear,
    pub linear_right: UnifiedLinear,
    pub post_laurel_norm: RMSNorm,
}

impl LaurelBlock {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let laurel_x = self.linear_left.forward(x);
        let laurel_x = self.linear_right.forward(&laurel_x);
        let normed = self.post_laurel_norm.forward(&laurel_x);
        mlxcel_core::add(x, &normed)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);

        let linear_left = UnifiedLinear::from_weights(
            weights,
            &format!("{}.linear_left", prefix),
            group_size,
            bits,
        )?;
        let linear_right = UnifiedLinear::from_weights(
            weights,
            &format!("{}.linear_right", prefix),
            group_size,
            bits,
        )?;

        let norm_weight = get_weight_copy(weights, &format!("{}.post_laurel_norm.weight", prefix))?;
        let post_laurel_norm = RMSNorm::new(norm_weight, config.rms_norm_eps);

        Ok(Self {
            linear_left,
            linear_right,
            post_laurel_norm,
        })
    }
}

// AltUp (Alternating Updates).
pub struct AltUp {
    pub correct_output_scale: UniquePtr<MlxArray>,
    // correction_coefs and prediction_coefs are NOT quantized (small 4x4 and 16x4 layers)
    pub correction_coefs: Linear,
    pub prediction_coefs: Linear,
    // modality_router IS quantized
    pub modality_router: UnifiedLinear,
    pub router_norm: RMSNorm,
    pub altup_num_inputs: usize,
    pub altup_active_idx: usize,
    pub hidden_size: usize,
}

impl AltUp {
    fn compute_router_modalities(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.router_norm.forward(x);
        let scale_val = (self.hidden_size as f32).powf(-1.0);
        let scaled = mlxcel_core::multiply_scalar(&normed, scale_val);
        let routed = self.modality_router.forward(&scaled);
        let routed = mlxcel_core::astype(&routed, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::tanh(&routed)
    }

    /// Predict: expand inputs through altup_num_inputs parallel paths.
    ///
    /// The decode-hot layer path consumes the stacked tensor directly and
    /// avoids immediately slicing the four AltUp planes only to stack them
    /// again during correction. Keeping this as one `[altup, B, L, hidden]`
    /// graph island mirrors mlx-lm's tensor scheduling more closely while
    /// preserving the public Vec-returning wrapper below for parity tests.
    pub fn predict_stacked(&self, x: &[UniquePtr<MlxArray>]) -> UniquePtr<MlxArray> {
        // x is [altup_num_inputs] arrays, each [B, L, hidden_size]
        // Get active input for routing
        let active = &x[self.altup_active_idx];
        let modalities = self.compute_router_modalities(active);

        // Get prediction coefficients
        let coefs = self.prediction_coefs.forward(&modalities);
        // coefs shape: [B, L, altup_num_inputs * altup_num_inputs]

        let n = self.altup_num_inputs as i32;
        let shape = mlxcel_core::array_shape(&modalities);
        let b = shape[0];
        let l = shape[1];

        // Reshape to [B, L, n, n] then transpose to [B, L, n, n] with axes swapped
        let all_coefs = mlxcel_core::reshape(&coefs, &[b, l, n, n]);
        let all_coefs = mlxcel_core::transpose_axes(&all_coefs, &[0, 1, 3, 2]);

        // Stack first, then cast once to match mlx-lm's `x.astype(mx.float32)`
        // scheduling. This keeps the four AltUp planes in one graph island
        // instead of creating one bf16→f32 cast node per plane.
        let x_stacked_native = stack_arrays(x, 0);
        let x_stacked = mlxcel_core::astype(&x_stacked_native, mlxcel_core::dtype::FLOAT32);
        // x_stacked shape: [altup, B, L, hidden]
        let x_permuted = mlxcel_core::transpose_axes(&x_stacked, &[1, 2, 3, 0]);
        // x_permuted shape: [B, L, hidden, altup]

        // Matrix multiply: [B, L, hidden, altup] @ [B, L, altup, altup] = [B, L, hidden, altup]
        let predictions = mlxcel_core::matmul(&x_permuted, &all_coefs);
        // Transpose back to [altup, B, L, hidden]
        let predictions = mlxcel_core::transpose_axes(&predictions, &[3, 0, 1, 2]);

        // Add residual
        let predictions = mlxcel_core::add(&predictions, &x_stacked);
        mlxcel_core::astype(&predictions, mlxcel_core::array_dtype(&x[0]))
    }

    /// Predict: expand inputs through altup_num_inputs parallel paths.
    pub fn predict(&self, x: &[UniquePtr<MlxArray>]) -> Vec<UniquePtr<MlxArray>> {
        let predictions = self.predict_stacked(x);
        split_altup_planes(&predictions, self.altup_num_inputs)
    }

    /// Correct: apply correction to stacked predictions based on activated output.
    pub fn correct_stacked(
        &self,
        predictions: &MlxArray,
        active_prediction: &MlxArray,
        activated: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let modalities = self.compute_router_modalities(activated);

        // correction_coefs output shape: [B, L, altup_num_inputs]
        let all_coefs = self.correction_coefs.forward(&modalities);
        let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(&all_coefs));
        let all_coefs = mlxcel_core::add(&all_coefs, &one);

        // innovation = activated - active_prediction
        let innovation = mlxcel_core::subtract(activated, active_prediction);

        let shape = mlxcel_core::array_shape(&all_coefs);
        let b = shape[0];
        let l = shape[1];

        // Move axis: [B, L, altup] -> [altup, B, L]
        let all_coefs = mlxcel_core::transpose_axes(&all_coefs, &[2, 0, 1]);

        // innovation[None] shape: [1, B, L, hidden]
        let hidden = mlxcel_core::array_shape(&innovation)[2];
        let innovation_expanded = mlxcel_core::reshape(&innovation, &[1, b, l, hidden]);

        // all_coefs[..., None] shape: [altup, B, L, 1]
        let altup = self.altup_num_inputs as i32;
        let coefs_expanded = mlxcel_core::reshape(&all_coefs, &[altup, b, l, 1]);

        // Element-wise multiply: [1, B, L, hidden] * [altup, B, L, 1] = [altup, B, L, hidden]
        let correction = mlxcel_core::multiply(&innovation_expanded, &coefs_expanded);

        // Add correction to the existing stacked prediction tensor.
        let corrected = mlxcel_core::add(predictions, &correction);

        // Cast back to original dtype
        let original_dtype = mlxcel_core::array_dtype(activated);
        mlxcel_core::astype(&corrected, original_dtype)
    }

    /// Correct: apply correction to predictions based on activated output.
    pub fn correct(
        &self,
        predictions: &[UniquePtr<MlxArray>],
        activated: &MlxArray,
    ) -> Vec<UniquePtr<MlxArray>> {
        let predictions_stacked = stack_arrays(predictions, 0);
        let active_prediction = &predictions[self.altup_active_idx];
        let corrected = self.correct_stacked(&predictions_stacked, active_prediction, activated);
        split_altup_planes(&corrected, self.altup_num_inputs)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);

        // correction_coefs and prediction_coefs are NOT quantized (small 4x4 and 16x4 layers)
        let mut correction_coefs =
            Linear::from_weights(weights, &format!("{}.correction_coefs", prefix))?;
        let mut prediction_coefs =
            Linear::from_weights(weights, &format!("{}.prediction_coefs", prefix))?;
        correction_coefs.weight =
            mlxcel_core::astype(&correction_coefs.weight, mlxcel_core::dtype::FLOAT32);
        prediction_coefs.weight =
            mlxcel_core::astype(&prediction_coefs.weight, mlxcel_core::dtype::FLOAT32);
        if let Some(bias) = correction_coefs.bias.take() {
            correction_coefs.bias = Some(mlxcel_core::astype(&bias, mlxcel_core::dtype::FLOAT32));
        }
        if let Some(bias) = prediction_coefs.bias.take() {
            prediction_coefs.bias = Some(mlxcel_core::astype(&bias, mlxcel_core::dtype::FLOAT32));
        }

        if let Some(clip) = config.altup_coef_clip {
            let low = mlxcel_core::full_f32(&[1], -clip, mlxcel_core::dtype::FLOAT32);
            let high = mlxcel_core::full_f32(&[1], clip, mlxcel_core::dtype::FLOAT32);
            correction_coefs.weight = mlxcel_core::clip(&correction_coefs.weight, &low, &high);
            prediction_coefs.weight = mlxcel_core::clip(&prediction_coefs.weight, &low, &high);
        }
        mlxcel_core::eval(&correction_coefs.weight);
        mlxcel_core::eval(&prediction_coefs.weight);
        if let Some(bias) = &correction_coefs.bias {
            mlxcel_core::eval(bias);
        }
        if let Some(bias) = &prediction_coefs.bias {
            mlxcel_core::eval(bias);
        }

        // modality_router IS quantized
        let modality_router = UnifiedLinear::from_weights(
            weights,
            &format!("{}.modality_router", prefix),
            group_size,
            bits,
        )?;

        let norm_weight = get_weight_copy(weights, &format!("{}.router_norm.weight", prefix))?;
        let router_norm = RMSNorm::new(norm_weight, config.rms_norm_eps);

        let correct_output_scale =
            get_weight_copy(weights, &format!("{}.correct_output_scale", prefix))?;

        Ok(Self {
            correct_output_scale,
            correction_coefs,
            prediction_coefs,
            modality_router,
            router_norm,
            altup_num_inputs: config.altup_num_inputs,
            altup_active_idx: config.altup_active_idx,
            hidden_size: config.hidden_size,
        })
    }
}

// Attention.
pub struct Gemma3nAttention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub v_norm: RMSNoScale,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub is_sliding: bool,
    pub window_size: i32,
    pub is_kv_shared_layer: bool,
    pub rope_theta: f32,
    pub scale: f32,
}

impl Gemma3nAttention {
    pub fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Query projection and reshape
        let queries = self.q_proj.forward(x);
        let queries = mlxcel_core::reshape(&queries, &[b, l, self.num_heads, self.head_dim]);

        // Apply Q norm
        let queries = self.q_norm.forward(&queries);
        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);

        let cache_offset = cache.offset;

        // Compute KV (or get from cache for KV-shared layers)
        let (keys, values) = if self.is_kv_shared_layer {
            // For KV-shared layers, return slice view of filled portion only.
            // cache.keys is a pre-allocated buffer; we must slice to the live
            // window length (`live_len() = offset - live_start`) — NOT the
            // monotonic `offset` — so this stays correct when's
            // `--max-kv-size` trim_front advances `live_start`. With no trim
            // (`live_start == 0`) this is bit-identical to slicing at
            // `cache.offset`.
            let live_len = cache.live_len();
            let k = cache.keys.as_ref().unwrap();
            let v = cache.values.as_ref().unwrap();
            let ks = mlxcel_core::array_shape(k);
            let vs = mlxcel_core::array_shape(v);
            (
                mlxcel_core::slice(k, &[0, 0, 0, 0], &[ks[0], ks[1], live_len, ks[3]]),
                mlxcel_core::slice(v, &[0, 0, 0, 0], &[vs[0], vs[1], live_len, vs[3]]),
            )
        } else {
            self.compute_kv(x, b, l, cache_offset, cache)
        };

        // Apply RoPE to queries
        let queries = mlxcel_core::fast_rope(
            &queries,
            self.head_dim,
            false,
            self.rope_theta,
            1.0,
            cache_offset,
        );

        // Scaled dot-product attention
        let attn_out = if l > 1 && mask.is_none() {
            // Prefill: use causal masking
            mlxcel_core::causal_attention(
                &queries,
                &keys,
                &values,
                self.scale,
                0.0,
                self.window_size,
            )
        } else {
            // Single token or explicit mask path. A sliding-window layer's
            // mask is either the clamped `(q_len, window_size)` mask or the
            // full `(q_len, k_len)` mask for a fresh single-pass prefill that
            // exceeds the window (issue #408). Plain `KVCache` returns
            // full-length K/V, so slice K/V to the mask's key axis (not blindly
            // to `window_size`): a full mask keeps every key, a clamped mask
            // drops the oldest. Mirrors `causal_attention`'s internal handling.
            let k_shape = mlxcel_core::array_shape(&keys);
            let k_len = k_shape[2];
            let mask_klen = mask
                .map(|m| *mlxcel_core::array_shape(m).last().unwrap_or(&k_len))
                .unwrap_or(k_len);
            let (k_used, v_used) = if self.window_size > 0 && k_len > mask_klen {
                let v_shape = mlxcel_core::array_shape(&values);
                let start = k_len - mask_klen;
                (
                    Some(mlxcel_core::slice(
                        &keys,
                        &[0, 0, start, 0],
                        &[k_shape[0], k_shape[1], k_len, k_shape[3]],
                    )),
                    Some(mlxcel_core::slice(
                        &values,
                        &[0, 0, start, 0],
                        &[v_shape[0], v_shape[1], k_len, v_shape[3]],
                    )),
                )
            } else {
                (None, None)
            };
            let k_ref: &MlxArray = k_used
                .as_ref()
                .map(|p| p.as_ref().unwrap())
                .unwrap_or_else(|| keys.as_ref().unwrap());
            let v_ref: &MlxArray = v_used
                .as_ref()
                .map(|p| p.as_ref().unwrap())
                .unwrap_or_else(|| values.as_ref().unwrap());

            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &queries,
                    k_ref,
                    v_ref,
                    self.scale,
                    mask_ptr,
                    0.0,
                    self.window_size,
                )
            }
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        self.o_proj.forward(&attn_out)
    }

    fn compute_kv(
        &self,
        x: &MlxArray,
        b: i32,
        l: i32,
        offset: i32,
        cache: &mut KVCache,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let keys = self.k_proj.forward(x);
        let keys = mlxcel_core::reshape(&keys, &[b, l, self.num_kv_heads, self.head_dim]);
        let keys = self.k_norm.forward(&keys);
        let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);

        let values = self.v_proj.forward(x);
        let values = mlxcel_core::reshape(&values, &[b, l, self.num_kv_heads, self.head_dim]);
        let values = self.v_norm.forward(&values);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        // Apply RoPE to keys
        let keys =
            mlxcel_core::fast_rope(&keys, self.head_dim, false, self.rope_theta, 1.0, offset);

        // Update KV cache
        cache.update_and_fetch(keys, values)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        layer_idx: usize,
        is_kv_shared_layer: bool,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);

        let is_sliding = config.layer_types[layer_idx] == "sliding_attention";
        let rope_theta = if is_sliding {
            config.rope_local_base_freq
        } else {
            config.rope_theta
        };

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;

        let q_norm = RMSNorm::new(q_norm_weight, config.rms_norm_eps);
        let k_norm = RMSNorm::new(k_norm_weight, config.rms_norm_eps);
        let v_norm = RMSNoScale::new(config.head_dim as i32, config.rms_norm_eps);

        let head_dim = config.head_dim as i32;
        let scale = 1.0; // Gemma3n uses scale=1.0, softcapping handles scaling

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            num_heads: config.num_attention_heads as i32,
            num_kv_heads: config.num_key_value_heads as i32,
            head_dim,
            is_sliding,
            window_size: if is_sliding {
                config.sliding_window as i32
            } else {
                0
            },
            is_kv_shared_layer,
            rope_theta,
            scale,
        })
    }
}

// MLP with gelu_topk activation.
pub struct MLP {
    pub gate_proj: MlpInputProjection,
    pub up_proj: MlpInputProjection,
    pub down_proj: UnifiedLinear,
    pub activation_sparsity: f32,
    pub std_multiplier: f32,
}

// M5 non-quantized Gemma3n decode GEMVs stream gate/up weights faster when MLX
// sees materialized transposed weights. Quantized layers keep UnifiedLinear so
// their specialized 4bit path is unchanged.
pub enum MlpInputProjection {
    Standard(UnifiedLinear),
    Pretransposed {
        weight_t: UniquePtr<MlxArray>,
        bias: Option<UniquePtr<MlxArray>>,
    },
}

impl MlpInputProjection {
    fn from_weights_maybe_pretransposed(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let hw = mlxcel_core::hardware::get_hardware();
        let is_m5_na = hw.has_neural_accelerator && hw.macos_supports_na;
        let scales_name = format!("{}.scales", prefix);
        if !is_m5_na || weights.contains_key(&scales_name) {
            return Ok(Self::Standard(UnifiedLinear::from_weights(
                weights, prefix, group_size, bits,
            )?));
        }

        let weight_name = format!("{}.weight", prefix);
        let weight = weights
            .get(&weight_name)
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;
        let weight_t = mlxcel_core::transpose(weight);
        let weight_t = mlxcel_core::contiguous(&weight_t, false);
        mlxcel_core::eval(&weight_t);

        let bias_name = format!("{}.bias", prefix);
        let bias = weights.get(&bias_name).map(|b| mlxcel_core::copy(b));
        Ok(Self::Pretransposed { weight_t, bias })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Standard(linear) => linear.forward(x),
            Self::Pretransposed { weight_t, bias } => {
                let out = mlxcel_core::matmul(x, weight_t);
                match bias {
                    Some(bias) => mlxcel_core::add(&out, bias),
                    None => out,
                }
            }
        }
    }

    fn regular_weight(&self) -> Option<&Linear> {
        match self {
            Self::Standard(linear) => linear.regular_weight(),
            Self::Pretransposed { .. } => None,
        }
    }
}

/// #60 introduced a fused Gemma3n decode path (stacked AltUp predict/
/// correct plus the `gemma3n_mlp_forward` bridge call) that cuts Rust↔C++
/// graph-construction overhead. It improves decode on Apple Silicon without a
/// Neural Accelerator (M1 Ultra: +3.6%). On M5-class (NA) hardware it is
/// neutral (re-measured ~0% on 2026-06-18 with MLX 0.31.2; it previously
/// regressed M5 by ~-6.3%, since closed by MLX/code evolution), so the gate
/// stays off NA hardware: no upside there, and the per-op split path is the
/// validated default. Full numbers:
/// docs/benchmark_results/gemma3n-decode-profile-m5max.md.
#[inline]
fn use_fused_decode_path() -> bool {
    let hw = mlxcel_core::hardware::get_hardware();
    !(hw.has_neural_accelerator && hw.macos_supports_na)
}

impl MLP {
    /// Apply gelu_topk activation with sparsity.
    /// Uses a compiled fused kernel matching Python's @mx.compile gelu_topk.
    fn gelu_topk(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if self.activation_sparsity <= 0.0 {
            return mlxcel_core::gelu_approx(x);
        }

        // Single compiled kernel: mean/std/cutoff/max/gelu fused together.
        // Matches Python mlx-lm's @partial(mx.compile, shapeless=True) gelu_topk.
        mlxcel_core::compiled_gelu_topk(x, self.std_multiplier)
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // The fused MLP bridge call (`gemma3n_mlp_forward`, added in #60) cuts
        // decode graph overhead on non-NA Apple Silicon. On M5-class (NA)
        // hardware it is neutral (see use_fused_decode_path), so it is gated off
        // and M5 uses the per-op bf16 path below.
        if use_fused_decode_path()
            && let (Some(gate), Some(up), Some(down)) = (
                self.gate_proj.regular_weight(),
                self.up_proj.regular_weight(),
                self.down_proj.regular_weight(),
            )
        {
            let gate_bias_ptr = gate
                .bias
                .as_ref()
                .map(|b| b.as_ref().unwrap() as *const MlxArray)
                .unwrap_or(std::ptr::null());
            let up_bias_ptr = up
                .bias
                .as_ref()
                .map(|b| b.as_ref().unwrap() as *const MlxArray)
                .unwrap_or(std::ptr::null());
            let down_bias_ptr = down
                .bias
                .as_ref()
                .map(|b| b.as_ref().unwrap() as *const MlxArray)
                .unwrap_or(std::ptr::null());
            return unsafe {
                mlxcel_core::gemma3n_mlp_forward(
                    x,
                    &gate.weight,
                    &up.weight,
                    &down.weight,
                    gate_bias_ptr,
                    up_bias_ptr,
                    down_bias_ptr,
                    self.activation_sparsity,
                    self.std_multiplier,
                )
            };
        }

        let x_cast;
        let x_mlp = if mlxcel_core::array_dtype(x) == mlxcel_core::dtype::BFLOAT16 {
            x
        } else {
            x_cast = mlxcel_core::astype(x, mlxcel_core::dtype::BFLOAT16);
            &x_cast
        };
        let gate = self.gate_proj.forward(x_mlp);
        let up = self.up_proj.forward(x_mlp);
        let prod = if self.activation_sparsity <= 0.0 {
            mlxcel_core::compiled_geglu_approx_activation(&gate, &up)
        } else {
            let activated = self.gelu_topk(&gate);
            mlxcel_core::multiply(&activated, &up)
        };
        let out = self.down_proj.forward(&prod);
        mlxcel_core::astype(&out, mlxcel_core::dtype::BFLOAT16)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);

        let gate_proj = MlpInputProjection::from_weights_maybe_pretransposed(
            weights,
            &format!("{}.gate_proj", prefix),
            group_size,
            bits,
        )?;
        let up_proj = MlpInputProjection::from_weights_maybe_pretransposed(
            weights,
            &format!("{}.up_proj", prefix),
            group_size,
            bits,
        )?;
        let down_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", prefix),
            group_size,
            bits,
        )?;

        let activation_sparsity = config.get_activation_sparsity(layer_idx);
        let std_multiplier = if activation_sparsity > 0.0 {
            std::f32::consts::SQRT_2 * erfinv(2.0 * activation_sparsity - 1.0)
        } else {
            0.0
        };

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            activation_sparsity,
            std_multiplier,
        })
    }
}

// Decoder Layer.
pub struct DecoderLayer {
    pub self_attn: Gemma3nAttention,
    pub mlp: MLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub pre_feedforward_layernorm: RMSNorm,
    pub post_feedforward_layernorm: RMSNorm,
    pub altup: AltUp,
    pub laurel: LaurelBlock,
    pub per_layer_input_gate: UnifiedLinear,
    pub per_layer_projection: UnifiedLinear,
    pub post_per_layer_input_norm: RMSNorm,
    pub altup_active_idx: usize,
    pub altup_correct_scale: bool,
}

impl DecoderLayer {
    pub fn forward(
        &self,
        x: &[UniquePtr<MlxArray>],
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
        per_layer_input: &MlxArray,
    ) -> Vec<UniquePtr<MlxArray>> {
        // The stacked AltUp path (added in #60) speeds up decode on non-NA
        // Apple Silicon but regresses M5-class hardware. Dispatch to the split
        // path on NA hardware; keep the stacked path elsewhere.
        if use_fused_decode_path() {
            self.forward_stacked(x, mask, cache, per_layer_input)
        } else {
            self.forward_split(x, mask, cache, per_layer_input)
        }
    }

    /// Fused-path layer forward (added in #60): keeps AltUp predictions stacked
    /// from predict through correct, slicing only the active plane. Faster on
    /// Apple Silicon without a Neural Accelerator; selected by `forward`.
    fn forward_stacked(
        &self,
        x: &[UniquePtr<MlxArray>],
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
        per_layer_input: &MlxArray,
    ) -> Vec<UniquePtr<MlxArray>> {
        // AltUp predict. Keep the prediction tensor stacked until correction
        // so decode avoids slicing four planes and stacking them again within
        // the same layer.
        let predictions = self.altup.predict_stacked(x);
        let active_prediction = slice_altup_plane(&predictions, self.altup_active_idx);

        // Input layernorm
        let active_normed = self.input_layernorm.forward(&active_prediction);

        // LAUREL
        let laurel_output = self.laurel.forward(&active_normed);

        // Self attention
        let attn = self.self_attn.forward(&active_normed, mask, cache);
        let attn = self.post_attention_layernorm.forward(&attn);

        // Residual + LAUREL
        let attn_gated = mlxcel_core::add(&active_prediction, &attn);

        let sum = mlxcel_core::add(&attn_gated, &laurel_output);
        let attn_laurel = mlxcel_core::multiply_scalar(&sum, std::f32::consts::FRAC_1_SQRT_2);

        // FFN
        let attn_norm = self.pre_feedforward_layernorm.forward(&attn_laurel);
        let ffw = self.mlp.forward(&attn_norm);
        let ffw_norm = self.post_feedforward_layernorm.forward(&ffw);
        let ffw_gated = mlxcel_core::add(&attn_laurel, &ffw_norm);
        let ffw_gated = mlxcel_core::astype(&ffw_gated, mlxcel_core::dtype::BFLOAT16);

        // AltUp correct
        let corrected = self
            .altup
            .correct_stacked(&predictions, &active_prediction, &ffw_gated);

        // Per-layer input processing
        let first = slice_altup_plane(&corrected, self.altup_active_idx);
        let first = if self.altup_correct_scale {
            mlxcel_core::multiply(&first, &self.altup.correct_output_scale)
        } else {
            first
        };

        let first = self.per_layer_input_gate.forward(&first);
        let first = mlxcel_core::compiled_geglu_approx_activation(&first, per_layer_input);
        let first = self.per_layer_projection.forward(&first);
        let first_prediction = self.post_per_layer_input_norm.forward(&first);

        // Add first_prediction to corrected[1:] and split for the next layer.
        split_altup_after_per_layer_update(
            &corrected,
            &first_prediction,
            self.altup.altup_num_inputs,
        )
    }

    /// Split-path layer forward (the path before #60's fused path):
    /// `AltUp::predict`/`correct` return per-plane Vecs. Used on M5-class
    /// (Neural Accelerator) hardware where the stacked path regresses decode.
    fn forward_split(
        &self,
        x: &[UniquePtr<MlxArray>],
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
        per_layer_input: &MlxArray,
    ) -> Vec<UniquePtr<MlxArray>> {
        // AltUp predict
        let predictions = self.altup.predict(x);
        let active_prediction = &predictions[self.altup_active_idx];

        // Input layernorm
        let active_normed = self.input_layernorm.forward(active_prediction);

        // LAUREL
        let laurel_output = self.laurel.forward(&active_normed);

        // Self attention
        let attn = self.self_attn.forward(&active_normed, mask, cache);
        let attn = self.post_attention_layernorm.forward(&attn);

        // Residual + LAUREL
        let attn_gated = mlxcel_core::add(active_prediction, &attn);

        let sum = mlxcel_core::add(&attn_gated, &laurel_output);
        let attn_laurel = mlxcel_core::multiply_scalar(&sum, std::f32::consts::FRAC_1_SQRT_2);

        // FFN
        let attn_norm = self.pre_feedforward_layernorm.forward(&attn_laurel);
        let ffw = self.mlp.forward(&attn_norm);
        let ffw_norm = self.post_feedforward_layernorm.forward(&ffw);
        let ffw_gated = mlxcel_core::add(&attn_laurel, &ffw_norm);
        let ffw_gated = mlxcel_core::astype(&ffw_gated, mlxcel_core::dtype::BFLOAT16);

        // AltUp correct
        let corrected = self.altup.correct(&predictions, &ffw_gated);

        // Per-layer input processing
        let first = &corrected[self.altup_active_idx];
        let first = if self.altup_correct_scale {
            mlxcel_core::multiply(first, &self.altup.correct_output_scale)
        } else {
            mlxcel_core::copy(first)
        };

        let first = self.per_layer_input_gate.forward(&first);
        let first = mlxcel_core::compiled_geglu_approx_activation(&first, per_layer_input);
        let first = self.per_layer_projection.forward(&first);
        let first_prediction = self.post_per_layer_input_norm.forward(&first);

        // Add first_prediction to corrected[1:].
        let mut result = Vec::with_capacity(corrected.len());
        let mut corrected = corrected.into_iter();
        if let Some(first) = corrected.next() {
            result.push(first);
        }
        for item in corrected {
            result.push(mlxcel_core::add(&item, &first_prediction));
        }

        result
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        layer_idx: usize,
        is_kv_shared_layer: bool,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);

        let self_attn = Gemma3nAttention::from_weights(
            weights,
            config,
            layer_idx,
            is_kv_shared_layer,
            &format!("{}.self_attn", prefix),
        )?;
        let mlp = MLP::from_weights(weights, config, layer_idx, &format!("{}.mlp", prefix))?;
        let altup = AltUp::from_weights(weights, config, &format!("{}.altup", prefix))?;
        let laurel = LaurelBlock::from_weights(weights, config, &format!("{}.laurel", prefix))?;

        // Norms
        let input_layernorm = RMSNorm::new(
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?,
            config.rms_norm_eps,
        );
        let post_attention_layernorm = RMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{}.post_attention_layernorm.weight", prefix),
            )?,
            config.rms_norm_eps,
        );
        let pre_feedforward_layernorm = RMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{}.pre_feedforward_layernorm.weight", prefix),
            )?,
            config.rms_norm_eps,
        );
        let post_feedforward_layernorm = RMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{}.post_feedforward_layernorm.weight", prefix),
            )?,
            config.rms_norm_eps,
        );
        let post_per_layer_input_norm = RMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{}.post_per_layer_input_norm.weight", prefix),
            )?,
            config.rms_norm_eps,
        );

        // Per-layer projections
        let per_layer_input_gate = UnifiedLinear::from_weights(
            weights,
            &format!("{}.per_layer_input_gate", prefix),
            group_size,
            bits,
        )?;
        let per_layer_projection = UnifiedLinear::from_weights(
            weights,
            &format!("{}.per_layer_projection", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            altup,
            laurel,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            altup_active_idx: config.altup_active_idx,
            altup_correct_scale: config.altup_correct_scale,
        })
    }
}

// Language Model.
pub struct Gemma3nLanguageModel {
    pub embed_tokens: UnifiedEmbedding,
    pub embed_tokens_weight_t: Option<UniquePtr<MlxArray>>,
    pub embed_tokens_per_layer: UnifiedEmbedding,
    pub per_layer_model_projection: UnifiedLinear,
    pub per_layer_projection_norm: RMSNorm,
    pub layers: Vec<DecoderLayer>,
    pub altup_projections: Vec<UnifiedLinear>,
    pub altup_unembed_projections: Vec<UnifiedLinear>,
    pub norm: RMSNorm,
    pub config: TextConfig,
    pub layer_idx_to_cache_idx: Vec<usize>,
    pub first_sliding_idx: usize,
    pub first_full_idx: usize,
}

impl Gemma3nLanguageModel {
    fn pretranspose_large_m5_embedding(
        embedding: &UnifiedEmbedding,
    ) -> Option<UniquePtr<MlxArray>> {
        let hw = mlxcel_core::hardware::get_hardware();
        if embedding.is_quantized() || !(hw.has_neural_accelerator && hw.macos_supports_na) {
            return None;
        }

        // The tied LM head is a very wide decode GEMV; materializing the
        // transpose improves M5 bandwidth on non-quantized Gemma3n.
        let weight_t = mlxcel_core::transpose(embedding.weight());
        let weight_t = mlxcel_core::contiguous(&weight_t, false);
        mlxcel_core::eval(&weight_t);
        Some(weight_t)
    }

    fn lm_head(&self, out: &MlxArray) -> UniquePtr<MlxArray> {
        match &self.embed_tokens_weight_t {
            Some(weight_t) => mlxcel_core::matmul(out, weight_t),
            None => self.embed_tokens.as_linear(out),
        }
    }

    pub fn forward(&self, inputs: &MlxArray, caches: &mut [KVCache]) -> UniquePtr<MlxArray> {
        // Embed tokens
        let h = self.embed_tokens.forward(inputs);
        let h = mlxcel_core::multiply_scalar(&h, (self.config.hidden_size as f32).sqrt());

        let shape = mlxcel_core::array_shape(&h);
        let b = shape[0];
        let l = shape[1];

        // Get per-layer inputs
        let per_layer_inputs = self.get_per_layer_inputs(inputs);
        let per_layer_inputs = self.project_per_layer_inputs(&h, &per_layer_inputs);

        // Create masks. Size them from the cache's live window
        // (`live_len() = offset - live_start`), not the monotonic `offset`.
        // Under `--max-kv-size`, `trim_front` slices the buffer to the live
        // window and advances `live_start` while `offset` keeps growing to
        // preserve the RoPE relative positions, so `update_and_fetch` returns
        // only `live_len` keys. A mask sized from `offset` would be wider than
        // the returned K/V and break the attention broadcast. The KV-shared
        // attention layer already slices its K/V to `cache.live_len()` (see
        // `Gemma3nAttention::forward`), so the model-level mask must use the
        // same bound. With no trim (`live_start == 0`), `live_len == offset`,
        // so this is byte-identical to the untrimmed path. See issue #419.
        let global_live_len = caches[self.first_full_idx].live_len();
        let sliding_live_len = caches[self.first_sliding_idx].live_len();

        let global_mask = if l > 1 {
            Some(create_causal_mask(l, global_live_len))
        } else {
            None
        };
        let sliding_mask = if l > 1 {
            // Dense `KVCache` keeps every key in the live window, so the
            // prefill mask is the full windowed-causal mask over the retained
            // (live) keys; the attention layer slices K/V to the mask's key
            // axis. The window is enforced by the mask, not by dropping keys.
            // See issues #408, #413, #419.
            Some(create_sliding_window_prefill_mask_dense(
                l,
                sliding_live_len,
                self.config.sliding_window as i32,
            ))
        } else {
            None
        };

        // Expand for AltUp
        let h0 = mlxcel_core::copy(&h);
        let target_magnitude = compute_magnitude(&h);

        // Create h_list: [h0, proj1(h0), proj2(h0), ...]
        let mut h_list = vec![mlxcel_core::copy(&h0)];
        for proj in &self.altup_projections {
            h_list.push(proj.forward(&h0));
        }

        // Normalize magnitudes of projected inputs
        normalize_magnitudes(&mut h_list, &target_magnitude);

        // Process layers
        for (i, layer) in self.layers.iter().enumerate() {
            let cache_idx = self.layer_idx_to_cache_idx[i];
            let mask = if self.config.layer_types[i] == "full_attention" {
                global_mask.as_ref()
            } else {
                sliding_mask.as_ref()
            };

            // Get per-layer input for this layer
            let per_layer_input = slice_layer_input(
                &per_layer_inputs,
                i as i32,
                b,
                l,
                self.config.hidden_size_per_layer_input as i32,
            );

            h_list = layer.forward(
                &h_list,
                mask.map(|m| m.as_ref().unwrap()),
                &mut caches[cache_idx],
                &per_layer_input,
            );
        }

        // Collapse AltUp dimension
        let h0_out = &h_list[0];
        let target_magnitude = compute_magnitude(h0_out);

        // Apply unembed projections
        for (i, proj) in self.altup_unembed_projections.iter().enumerate() {
            h_list[i + 1] = proj.forward(&h_list[i + 1]);
        }

        // Normalize magnitudes
        normalize_magnitudes_from_idx(&mut h_list, 1, &target_magnitude);

        // Mean across altup dimension
        let h = mean_arrays(&h_list);

        // Final norm
        let out = self.norm.forward(&h);
        let out = if self.embed_tokens.is_quantized() {
            out
        } else {
            mlxcel_core::astype(&out, mlxcel_core::array_dtype(self.embed_tokens.weight()))
        };

        // LM head (tied embeddings)
        let mut logits = self.lm_head(&out);

        // Apply logit softcapping if configured
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = apply_softcap(&logits, cap);
        }

        logits
    }

    pub fn get_per_layer_inputs(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let vocab_limit = mlxcel_core::full_f32(
            &[1],
            self.config.vocab_size_per_layer_input as f32,
            mlxcel_core::dtype::INT32,
        );
        let vocab_limit = mlxcel_core::astype(&vocab_limit, mlxcel_core::dtype::INT32);
        let mask = mlxcel_core::less(input_ids, &vocab_limit);
        let zeros = mlxcel_core::zeros(
            &mlxcel_core::array_shape(input_ids),
            mlxcel_core::dtype::INT32,
        );
        let tokens = mlxcel_core::where_cond(&mask, input_ids, &zeros);

        let embedded = self.embed_tokens_per_layer.forward(&tokens);
        let result = mlxcel_core::multiply_scalar(
            &embedded,
            (self.config.hidden_size_per_layer_input as f32).sqrt(),
        );

        // Reshape to [B, L, num_hidden_layers, hidden_size_per_layer_input]
        let shape = mlxcel_core::array_shape(input_ids);
        let b = shape[0];
        let l = shape[1];
        mlxcel_core::reshape(
            &result,
            &[
                b,
                l,
                self.config.num_hidden_layers as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        )
    }

    pub fn project_per_layer_inputs(
        &self,
        inputs_embeds: &MlxArray,
        per_layer_inputs: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let proj = self.per_layer_model_projection.forward(inputs_embeds);
        let proj = mlxcel_core::multiply_scalar(&proj, (self.config.hidden_size as f32).powf(-0.5));

        let shape = mlxcel_core::array_shape(inputs_embeds);
        let b = shape[0];
        let l = shape[1];
        let proj = mlxcel_core::reshape(
            &proj,
            &[
                b,
                l,
                self.config.num_hidden_layers as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        );

        let proj_normed = self.per_layer_projection_norm.forward(&proj);

        let sum = mlxcel_core::add(&proj_normed, per_layer_inputs);
        mlxcel_core::multiply_scalar(&sum, std::f32::consts::FRAC_1_SQRT_2)
    }

    /// Get embedded token representations (for VLM use)
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.embed_tokens.forward(input_ids);
        mlxcel_core::multiply_scalar(&h, (self.config.hidden_size as f32).sqrt())
    }

    /// Forward pass starting from pre-computed embeddings + per_layer_inputs (for VLM)
    pub fn forward_with_inputs(
        &self,
        inputs_embeds: &MlxArray,
        per_layer_inputs: &MlxArray,
        caches: &mut [KVCache],
    ) -> UniquePtr<MlxArray> {
        let h = inputs_embeds;

        let shape = mlxcel_core::array_shape(h);
        let _b = shape[0];
        let l = shape[1];

        // Create masks. Size them from the cache's live window
        // (`live_len() = offset - live_start`), not the monotonic `offset`.
        // Under `--max-kv-size`, `trim_front` slices the buffer to the live
        // window and advances `live_start` while `offset` keeps growing to
        // preserve the RoPE relative positions, so `update_and_fetch` returns
        // only `live_len` keys. A mask sized from `offset` would be wider than
        // the returned K/V and break the attention broadcast. The KV-shared
        // attention layer already slices its K/V to `cache.live_len()` (see
        // `Gemma3nAttention::forward`), so the model-level mask must use the
        // same bound. With no trim (`live_start == 0`), `live_len == offset`,
        // so this is byte-identical to the untrimmed path. See issue #419.
        let global_live_len = caches[self.first_full_idx].live_len();
        let sliding_live_len = caches[self.first_sliding_idx].live_len();

        let global_mask = if l > 1 {
            Some(create_causal_mask(l, global_live_len))
        } else {
            None
        };
        let sliding_mask = if l > 1 {
            // Dense `KVCache` keeps every key in the live window, so the
            // prefill mask is the full windowed-causal mask over the retained
            // (live) keys; the attention layer slices K/V to the mask's key
            // axis. The window is enforced by the mask, not by dropping keys.
            // See issues #408, #413, #419.
            Some(create_sliding_window_prefill_mask_dense(
                l,
                sliding_live_len,
                self.config.sliding_window as i32,
            ))
        } else {
            None
        };

        // Expand for AltUp
        let h0 = mlxcel_core::copy(h);
        let target_magnitude = compute_magnitude(h);

        let mut h_list = vec![mlxcel_core::copy(&h0)];
        for proj in &self.altup_projections {
            h_list.push(proj.forward(&h0));
        }
        normalize_magnitudes(&mut h_list, &target_magnitude);

        // Process layers
        for (i, layer) in self.layers.iter().enumerate() {
            let cache_idx = self.layer_idx_to_cache_idx[i];
            let mask = if self.config.layer_types[i] == "full_attention" {
                global_mask.as_ref()
            } else {
                sliding_mask.as_ref()
            };

            let per_layer_input = slice_layer_input(
                per_layer_inputs,
                i as i32,
                shape[0],
                l,
                self.config.hidden_size_per_layer_input as i32,
            );

            h_list = layer.forward(
                &h_list,
                mask.map(|m| m.as_ref().unwrap()),
                &mut caches[cache_idx],
                &per_layer_input,
            );
        }

        // Collapse AltUp dimension
        let h0_out = &h_list[0];
        let target_magnitude = compute_magnitude(h0_out);

        for (i, proj) in self.altup_unembed_projections.iter().enumerate() {
            h_list[i + 1] = proj.forward(&h_list[i + 1]);
        }
        normalize_magnitudes_from_idx(&mut h_list, 1, &target_magnitude);

        let h = mean_arrays(&h_list);
        let out = self.norm.forward(&h);
        let out = if self.embed_tokens.is_quantized() {
            out
        } else {
            mlxcel_core::astype(&out, mlxcel_core::array_dtype(self.embed_tokens.weight()))
        };
        let mut logits = self.lm_head(&out);

        if let Some(cap) = self.config.final_logit_softcapping {
            logits = apply_softcap(&logits, cap);
        }

        logits
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        let first_kv_shared = self.config.first_kv_shared_layer_idx();
        (0..first_kv_shared).map(|_| KVCache::new()).collect()
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config
            .quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64);
        let bits = config
            .quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4);

        let first_kv_shared = config.first_kv_shared_layer_idx();

        // Embeddings
        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            &format!("{}.embed_tokens", prefix),
            group_size,
            bits,
        )?;
        let embed_tokens_weight_t = Self::pretranspose_large_m5_embedding(&embed_tokens);
        let embed_tokens_per_layer = UnifiedEmbedding::from_weights(
            weights,
            &format!("{}.embed_tokens_per_layer", prefix),
            group_size,
            bits,
        )?;

        // Per-layer projection
        let per_layer_model_projection = UnifiedLinear::from_weights(
            weights,
            &format!("{}.per_layer_model_projection", prefix),
            group_size,
            bits,
        )?;
        let per_layer_projection_norm = RMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{}.per_layer_projection_norm.weight", prefix),
            )?,
            config.rms_norm_eps,
        );

        // Layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let is_kv_shared = i >= first_kv_shared;
            let layer = DecoderLayer::from_weights(
                weights,
                config,
                i,
                is_kv_shared,
                &format!("{}.layers.{}", prefix, i),
            )?;
            layers.push(layer);
        }

        // AltUp projections
        let mut altup_projections = Vec::new();
        let mut altup_unembed_projections = Vec::new();
        for i in 0..(config.altup_num_inputs - 1) {
            altup_projections.push(UnifiedLinear::from_weights(
                weights,
                &format!("{}.altup_projections.{}", prefix, i),
                group_size,
                bits,
            )?);
            altup_unembed_projections.push(UnifiedLinear::from_weights(
                weights,
                &format!("{}.altup_unembed_projections.{}", prefix, i),
                group_size,
                bits,
            )?);
        }

        // Final norm
        let norm = RMSNorm::new(
            get_weight_copy(weights, &format!("{}.norm.weight", prefix))?,
            config.rms_norm_eps,
        );

        // Build layer to cache mapping
        let mut layer_idx_to_cache_idx = Vec::with_capacity(config.num_hidden_layers);
        let concrete_layers: Vec<_> = config.layer_types[..first_kv_shared].to_vec();
        let shared_full_idx = concrete_layers
            .iter()
            .rposition(|t| t == "full_attention")
            .unwrap_or(0);
        let shared_sliding_idx = concrete_layers
            .iter()
            .rposition(|t| t == "sliding_attention")
            .unwrap_or(0);

        for (i, layer_type) in config.layer_types.iter().enumerate() {
            if i < first_kv_shared {
                layer_idx_to_cache_idx.push(i);
            } else if layer_type == "full_attention" {
                layer_idx_to_cache_idx.push(shared_full_idx);
            } else {
                layer_idx_to_cache_idx.push(shared_sliding_idx);
            }
        }

        // Find first indices
        let first_sliding_idx = config
            .layer_types
            .iter()
            .position(|t| t == "sliding_attention")
            .unwrap_or(0);
        let first_full_idx = config
            .layer_types
            .iter()
            .position(|t| t == "full_attention")
            .unwrap_or(0);

        Ok(Self {
            embed_tokens,
            embed_tokens_weight_t,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            altup_projections,
            altup_unembed_projections,
            norm,
            config: config.clone(),
            layer_idx_to_cache_idx,
            first_sliding_idx,
            first_full_idx,
        })
    }
}

// Model.
pub struct Gemma3nModel {
    pub language_model: Gemma3nLanguageModel,
    pub config: TextConfig,
}

impl Gemma3nModel {
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;
        let config = args.text_args();

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let language_model =
            Gemma3nLanguageModel::from_weights(&weights, &config, "language_model.model")?;

        Ok(Self {
            language_model,
            config,
        })
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        self.language_model.make_caches()
    }

    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
}

impl LanguageModel for Gemma3nModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.language_model.forward(input_ids, caches)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.language_model.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Gemma3nModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.language_model.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Gemma 3n EOS tokens: <eos> (1), <end_of_turn> (106)
        vec![1, 106]
    }
}

// Multimodal Embedder (for VLM).
/// Embeds soft tokens (vision features) into language model space.
///
/// soft_embedding_norm → embedding_projection → post_projection_norm
pub struct Gemma3nMultimodalEmbedder {
    pub soft_embedding_norm: RMSNorm,        // with scale
    pub embedding_projection: UnifiedLinear, // vision_hidden → text_hidden
    pub post_projection_norm: RMSNoScale,    // without scale
}

impl Gemma3nMultimodalEmbedder {
    /// Forward pass for soft tokens (vision features)
    pub fn forward_soft(&self, inputs_embeds: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.soft_embedding_norm.forward(inputs_embeds);
        let projected = self.embedding_projection.forward(&normed);
        self.post_projection_norm.forward(&projected)
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str, // "embed_vision"
        vision_hidden_size: usize,
        text_hidden_size: usize,
        eps: f32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let soft_norm_weight =
            get_weight_copy(weights, &format!("{}.soft_embedding_norm.weight", prefix))?;
        let soft_embedding_norm = RMSNorm::new(soft_norm_weight, eps);

        let embedding_projection = UnifiedLinear::from_weights(
            weights,
            &format!("{}.embedding_projection", prefix),
            group_size,
            bits,
        )?;

        let post_projection_norm = RMSNoScale::new(text_hidden_size as i32, eps);

        let _ = vision_hidden_size; // used for documentation

        Ok(Self {
            soft_embedding_norm,
            embedding_projection,
            post_projection_norm,
        })
    }
}
