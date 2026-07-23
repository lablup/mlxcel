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

struct DecoderLayerForward {
    planes: Vec<UniquePtr<MlxArray>>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    laurel_output: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    corrected_active: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    ple_scaled_active: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    ple_gate: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    ple_activated: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    ple_projected: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    ple_injected: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    ple_residual: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    ple_residual_updated: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    ple_residuals: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    ple_residuals_updated: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    altup_predicted: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    altup_predicted_active: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    altup_activated: UniquePtr<MlxArray>,
    #[cfg(all(test, feature = "xla-diagnostics"))]
    altup_corrected: UniquePtr<MlxArray>,
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
        let traced = if use_fused_decode_path() {
            self.forward_stacked(x, mask, cache, per_layer_input)
        } else {
            self.forward_split(x, mask, cache, per_layer_input)
        };
        traced.planes
    }

    #[cfg(all(test, feature = "xla-diagnostics"))]
    fn forward_diagnostics(
        &self,
        x: &[UniquePtr<MlxArray>],
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
        per_layer_input: &MlxArray,
    ) -> DecoderLayerForward {
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
    ) -> DecoderLayerForward {
        // AltUp predict. Keep the prediction tensor stacked until correction
        // so decode avoids slicing four planes and stacking them again within
        // the same layer.
        let predictions = self.altup.predict_stacked(x);
        let active_prediction = slice_altup_plane(&predictions, self.altup_active_idx);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let altup_predicted = mlxcel_core::copy(&predictions);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let altup_predicted_active = mlxcel_core::copy(&active_prediction);

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
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let altup_activated = mlxcel_core::copy(&ffw_gated);

        // AltUp correct
        let corrected = self
            .altup
            .correct_stacked(&predictions, &active_prediction, &ffw_gated);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let altup_corrected = mlxcel_core::copy(&corrected);

        // Per-layer input processing
        let first = slice_altup_plane(&corrected, self.altup_active_idx);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let corrected_active = mlxcel_core::copy(&first);
        let first = if self.altup_correct_scale {
            mlxcel_core::multiply(&first, &self.altup.correct_output_scale)
        } else {
            first
        };

        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_scaled_active = mlxcel_core::copy(&first);
        let ple_gate_value = self.per_layer_input_gate.forward(&first);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_gate = mlxcel_core::copy(&ple_gate_value);
        let ple_activated_value =
            mlxcel_core::compiled_geglu_approx_activation(&ple_gate_value, per_layer_input);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_activated = mlxcel_core::copy(&ple_activated_value);
        let ple_projected_value = self.per_layer_projection.forward(&ple_activated_value);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_projected = mlxcel_core::copy(&ple_projected_value);
        let first_prediction = self.post_per_layer_input_norm.forward(&ple_projected_value);

        // Add first_prediction to corrected[1:] and split for the next layer.
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_residual = slice_altup_plane(&corrected, 1);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_residuals = {
            let residuals = (1..self.altup.altup_num_inputs)
                .map(|plane| slice_altup_plane(&corrected, plane))
                .collect::<Vec<_>>();
            stack_arrays(&residuals, 0)
        };
        let planes = split_altup_after_per_layer_update(
            &corrected,
            &first_prediction,
            self.altup.altup_num_inputs,
        );
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_residual_updated = mlxcel_core::copy(&planes[1]);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_residuals_updated = stack_arrays(&planes[1..], 0);
        DecoderLayerForward {
            planes,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            laurel_output,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            corrected_active,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_scaled_active,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_gate,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_activated,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_projected,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_injected: first_prediction,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_residual,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_residual_updated,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_residuals,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_residuals_updated,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            altup_predicted,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            altup_predicted_active,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            altup_activated,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            altup_corrected,
        }
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
    ) -> DecoderLayerForward {
        // AltUp predict
        let predictions = self.altup.predict(x);
        let active_prediction = &predictions[self.altup_active_idx];
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let altup_predicted = stack_arrays(&predictions, 0);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let altup_predicted_active = mlxcel_core::copy(active_prediction);

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
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let altup_activated = mlxcel_core::copy(&ffw_gated);

        // AltUp correct
        let corrected = self.altup.correct(&predictions, &ffw_gated);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let altup_corrected = stack_arrays(&corrected, 0);

        // Per-layer input processing
        let first = &corrected[self.altup_active_idx];
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let corrected_active = mlxcel_core::copy(first);
        let first = if self.altup_correct_scale {
            mlxcel_core::multiply(first, &self.altup.correct_output_scale)
        } else {
            mlxcel_core::copy(first)
        };

        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_scaled_active = mlxcel_core::copy(&first);
        let ple_gate_value = self.per_layer_input_gate.forward(&first);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_gate = mlxcel_core::copy(&ple_gate_value);
        let ple_activated_value =
            mlxcel_core::compiled_geglu_approx_activation(&ple_gate_value, per_layer_input);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_activated = mlxcel_core::copy(&ple_activated_value);
        let ple_projected_value = self.per_layer_projection.forward(&ple_activated_value);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_projected = mlxcel_core::copy(&ple_projected_value);
        let first_prediction = self.post_per_layer_input_norm.forward(&ple_projected_value);

        // Add first_prediction to corrected[1:].
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_residual = mlxcel_core::copy(&corrected[1]);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_residuals = stack_arrays(&corrected[1..], 0);
        let mut result = Vec::with_capacity(corrected.len());
        let mut corrected = corrected.into_iter();
        if let Some(first) = corrected.next() {
            result.push(first);
        }
        for item in corrected {
            result.push(mlxcel_core::add(&item, &first_prediction));
        }
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_residual_updated = mlxcel_core::copy(&result[1]);
        #[cfg(all(test, feature = "xla-diagnostics"))]
        let ple_residuals_updated = stack_arrays(&result[1..], 0);

        DecoderLayerForward {
            planes: result,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            laurel_output,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            corrected_active,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_scaled_active,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_gate,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_activated,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_projected,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_injected: first_prediction,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_residual,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_residual_updated,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_residuals,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            ple_residuals_updated,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            altup_predicted,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            altup_predicted_active,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            altup_activated,
            #[cfg(all(test, feature = "xla-diagnostics"))]
            altup_corrected,
        }
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

#[cfg(all(test, feature = "xla-diagnostics"))]
struct Gemma3nMlxDiagnostics {
    scaled_embeddings: UniquePtr<MlxArray>,
    projected_ple: UniquePtr<MlxArray>,
    layer0_laurel: UniquePtr<MlxArray>,
    layer0_ple_injected: UniquePtr<MlxArray>,
    layer0_all_planes: UniquePtr<MlxArray>,
    layer0_active_plane: UniquePtr<MlxArray>,
    layer_mid_all_planes: UniquePtr<MlxArray>,
    layer_mid_active_plane: UniquePtr<MlxArray>,
    layer_last_all_planes: UniquePtr<MlxArray>,
    layer_last_active_plane: UniquePtr<MlxArray>,
    all_layer_all_planes: Vec<UniquePtr<MlxArray>>,
    all_layer_active_planes: Vec<UniquePtr<MlxArray>>,
    layer0_k: UniquePtr<MlxArray>,
    layer0_v: UniquePtr<MlxArray>,
    final_hidden: UniquePtr<MlxArray>,
    logits: UniquePtr<MlxArray>,
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

    #[cfg(all(test, feature = "xla-diagnostics"))]
    fn forward_diagnostics(
        &self,
        inputs: &MlxArray,
        caches: &mut [KVCache],
    ) -> Gemma3nMlxDiagnostics {
        let scaled_embeddings = self.get_embed_tokens(inputs);
        let token_ple = self.get_per_layer_inputs(inputs);
        let projected_ple = self.project_per_layer_inputs(&scaled_embeddings, &token_ple);
        let shape = mlxcel_core::array_shape(&scaled_embeddings);
        let b = shape[0];
        let l = shape[1];
        let global_live_len = caches[self.first_full_idx].live_len();
        let sliding_live_len = caches[self.first_sliding_idx].live_len();
        let global_mask = (l > 1).then(|| create_causal_mask(l, global_live_len));
        let sliding_mask = (l > 1).then(|| {
            create_sliding_window_prefill_mask_dense(
                l,
                sliding_live_len,
                self.config.sliding_window as i32,
            )
        });

        let target_magnitude = compute_magnitude(&scaled_embeddings);
        let mut planes = vec![mlxcel_core::copy(&scaled_embeddings)];
        for projection in &self.altup_projections {
            planes.push(projection.forward(&scaled_embeddings));
        }
        normalize_magnitudes(&mut planes, &target_magnitude);

        let mut layer0_all_planes = None;
        let mut layer0_active_plane = None;
        let mut layer0_laurel = None;
        let mut layer0_ple_injected = None;
        let mut layer_mid_all_planes = None;
        let mut layer_mid_active_plane = None;
        let mut layer_last_all_planes = None;
        let mut layer_last_active_plane = None;
        let mut all_layer_all_planes = Vec::with_capacity(self.layers.len());
        let mut all_layer_active_planes = Vec::with_capacity(self.layers.len());
        let mut layer0_k = None;
        let mut layer0_v = None;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let cache_index = self.layer_idx_to_cache_idx[layer_index];
            let mask = if self.config.layer_types[layer_index] == "full_attention" {
                global_mask.as_ref()
            } else {
                sliding_mask.as_ref()
            };
            let per_layer_input = slice_layer_input(
                &projected_ple,
                layer_index as i32,
                b,
                l,
                self.config.hidden_size_per_layer_input as i32,
            );
            let traced = layer.forward_diagnostics(
                &planes,
                mask.map(|value| value.as_ref().unwrap()),
                &mut caches[cache_index],
                &per_layer_input,
            );
            planes = traced.planes;
            all_layer_all_planes.push(stack_arrays(&planes, 0));
            all_layer_active_planes.push(mlxcel_core::copy(&planes[self.config.altup_active_idx]));
            if layer_index == 0 {
                assert_eq!(
                    cache_index, 0,
                    "pinned Gemma3n layer0 must map to concrete physical cache0"
                );
                layer0_all_planes = Some(stack_arrays(&planes, 0));
                layer0_active_plane =
                    Some(mlxcel_core::copy(&planes[self.config.altup_active_idx]));
                layer0_laurel = Some(traced.laurel_output);
                layer0_ple_injected = Some(traced.ple_injected);
                layer0_k = Some(mlxcel_core::copy(
                    caches[cache_index]
                        .keys
                        .as_ref()
                        .expect("layer0 concrete K cache"),
                ));
                layer0_v = Some(mlxcel_core::copy(
                    caches[cache_index]
                        .values
                        .as_ref()
                        .expect("layer0 concrete V cache"),
                ));
            }
            if layer_index == self.layers.len() / 2 {
                layer_mid_all_planes = Some(stack_arrays(&planes, 0));
                layer_mid_active_plane =
                    Some(mlxcel_core::copy(&planes[self.config.altup_active_idx]));
            }
            if layer_index + 1 == self.layers.len() {
                layer_last_all_planes = Some(stack_arrays(&planes, 0));
                layer_last_active_plane =
                    Some(mlxcel_core::copy(&planes[self.config.altup_active_idx]));
            }
        }

        let target_magnitude = compute_magnitude(&planes[0]);
        for (index, projection) in self.altup_unembed_projections.iter().enumerate() {
            planes[index + 1] = projection.forward(&planes[index + 1]);
        }
        normalize_magnitudes_from_idx(&mut planes, 1, &target_magnitude);
        let collapsed = mean_arrays(&planes);
        let final_hidden = self.norm.forward(&collapsed);
        let final_hidden = if self.embed_tokens.is_quantized() {
            final_hidden
        } else {
            mlxcel_core::astype(
                &final_hidden,
                mlxcel_core::array_dtype(self.embed_tokens.weight()),
            )
        };
        let mut logits = self.lm_head(&final_hidden);
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = apply_softcap(&logits, cap);
        }
        Gemma3nMlxDiagnostics {
            scaled_embeddings,
            projected_ple,
            layer0_laurel: layer0_laurel.expect("layer0 LAUREL"),
            layer0_ple_injected: layer0_ple_injected.expect("layer0 PLE injection"),
            layer0_all_planes: layer0_all_planes.expect("layer0 planes"),
            layer0_active_plane: layer0_active_plane.expect("layer0 active plane"),
            layer_mid_all_planes: layer_mid_all_planes.expect("middle-layer planes"),
            layer_mid_active_plane: layer_mid_active_plane.expect("middle-layer active plane"),
            layer_last_all_planes: layer_last_all_planes.expect("last-layer planes"),
            layer_last_active_plane: layer_last_active_plane.expect("last-layer active plane"),
            all_layer_all_planes,
            all_layer_active_planes,
            layer0_k: layer0_k.expect("layer0 K"),
            layer0_v: layer0_v.expect("layer0 V"),
            final_hidden,
            logits,
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

    /// Opt out of cache-level chunked prefill (#674): the parity battery
    /// showed gemma3n-e2b degenerating into repeated code fences under a
    /// multi-call prefill while every rotating-cache family passed after the
    /// #678 fix. Gemma 3n's AltUp / per-layer-input stack has no validated
    /// multi-call prefill parity yet, so keep the single-pass path until a
    /// dedicated investigation clears it.
    fn supports_chunked_prefill(&self) -> bool {
        false
    }
}

#[cfg(all(test, feature = "xla-diagnostics"))]
mod xla_canonical_diagnostics {
    use super::*;
    use crate::loaded_model::LoadedModel;
    use mlxcel_xla::{
        dequantize_gemma3n_affine_diagnostic, run_gemma3n_all_layer_diagnostics,
        run_gemma3n_altup_correct_diagnostic_probe, run_gemma3n_altup_predict_diagnostic_probe,
        run_gemma3n_attention_diagnostic_probe, run_gemma3n_canonical_diagnostics,
        run_gemma3n_decode_attention_diagnostic_probe, run_gemma3n_dense_mlp_diagnostic_probe,
        run_gemma3n_initial_altup_diagnostic_probe, run_gemma3n_ple_diagnostic_probe,
        run_gemma3n_ple_injection_all_planes_diagnostic_probe,
        run_gemma3n_ple_injection_diagnostic_probe, run_gemma3n_post_attention_diagnostic_probe,
        run_gemma3n_prefix_decode_diagnostic, run_gemma3n_qmv_diagnostic_probe,
        run_gemma3n_rms_diagnostic_probe, run_gemma3n_sdpa_vector_context_diagnostic_probe,
    };
    use sha2::{Digest, Sha256};

    const MODEL_ID: &str = "mlx-community/gemma-3n-E2B-it-4bit";
    const CONFIG_SHA256: &str = "f865367012e0738cef66e19353f46cb96c67aead717c4e86e72f130450d8e52a";
    const INDEX_SHA256: &str = "3ed4d02b8c245c61ba2f542e48cc40ed98398f7098457d63d3d7d0fda8d09c0c";
    const MODEL_SHA256: &str = "8d5af4ae73617d12496301ebad11226f9056c4b0d51dbb9ef9409234fb87d6fa";
    const CAPACITY: usize = 8;
    const PROMPT: [i32; 3] = [2, 100, 200];

    fn sha256(path: &Path) -> String {
        use std::io::Read;

        let file = std::fs::File::open(path).unwrap();
        let mut reader = std::io::BufReader::with_capacity(1024 * 1024, file);
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut digest = Sha256::new();
        loop {
            let read = reader.read(&mut buffer).unwrap();
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
        format!("{:x}", digest.finalize())
    }

    fn array_to_f32(array: &MlxArray) -> Vec<f32> {
        let array = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&array);
        mlxcel_core::array_to_raw_bytes(&array)
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn argmax(values: &[f32]) -> i32 {
        values
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map_or(0, |(index, _)| index as i32)
    }

    fn top_k(values: &[f32], count: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<_> = values.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        indexed.truncate(count);
        indexed
    }

    fn xla_segment_prefix(
        flat: &[f32],
        layout: &mlxcel_xla::Gemma3nDiagnosticLayout,
        name: &str,
        real_len: usize,
    ) -> Vec<f32> {
        let segment = layout.segment(name).unwrap();
        let values = &flat[segment.offset..segment.offset + segment.len];
        match name {
            "scaled_embeddings" | "final_hidden" => values[..real_len * segment.shape[1]].to_vec(),
            "projected_ple" | "layer0_k" | "layer0_v" => {
                let row: usize = segment.shape[1..].iter().product();
                values[..real_len * row].to_vec()
            }
            "layer0_all_planes" | "layer_mid_all_planes" | "layer_last_all_planes" => {
                let row = segment.shape[2];
                let capacity = segment.shape[1];
                (0..segment.shape[0])
                    .flat_map(|plane| {
                        let start = plane * capacity * row;
                        values[start..start + real_len * row].iter().copied()
                    })
                    .collect()
            }
            "layer0_laurel"
            | "layer0_ple_injected"
            | "layer0_active_plane"
            | "layer_mid_active_plane"
            | "layer_last_active_plane" => values[..real_len * segment.shape[1]].to_vec(),
            "logits" => values.to_vec(),
            _ => panic!("unknown diagnostic segment {name}"),
        }
    }

    fn mlx_layer0_cache_prefix(array: &MlxArray, real_len: usize) -> Vec<f32> {
        let shape = mlxcel_core::array_shape(array);
        assert_eq!(shape.len(), 4);
        let values = array_to_f32(array);
        mlx_layer0_cache_prefix_values(
            &values,
            [
                shape[0] as usize,
                shape[1] as usize,
                shape[2] as usize,
                shape[3] as usize,
            ],
            real_len,
        )
    }

    fn mlx_layer0_cache_prefix_values(
        values: &[f32],
        shape: [usize; 4],
        real_len: usize,
    ) -> Vec<f32> {
        assert_eq!(shape[0], 1);
        let heads = shape[1];
        let cache_capacity = shape[2];
        let dim = shape[3];
        assert!(cache_capacity >= real_len);
        assert_eq!(values.len(), shape.iter().product::<usize>());
        let mut transposed = Vec::with_capacity(real_len * heads * dim);
        for token in 0..real_len {
            for head in 0..heads {
                let start = (head * cache_capacity + token) * dim;
                transposed.extend_from_slice(&values[start..start + dim]);
            }
        }
        transposed
    }

    #[test]
    fn mlx_cache_prefix_uses_physical_capacity_stride() {
        let values = (0..30).map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(
            mlx_layer0_cache_prefix_values(&values, [1, 2, 5, 3], 2),
            vec![
                0.0, 1.0, 2.0, 15.0, 16.0, 17.0, 3.0, 4.0, 5.0, 18.0, 19.0, 20.0,
            ]
        );
    }

    #[derive(Clone, Copy)]
    struct RegressionEnvelope {
        max_abs: f32,
        min_cosine: f64,
        max_normalized_rmse: f64,
    }

    impl RegressionEnvelope {
        const fn pinned_cuda(max_abs: f32) -> Self {
            Self {
                max_abs,
                min_cosine: 0.90,
                max_normalized_rmse: 0.50,
            }
        }
    }

    fn compare(
        name: &str,
        mlx: &[f32],
        xla: &[f32],
        envelope: RegressionEnvelope,
        failures: &mut Vec<String>,
    ) {
        if mlx.len() != xla.len() {
            eprintln!("{name}: length mlx={} xla={}", mlx.len(), xla.len());
            failures.push(format!("{name}: length mismatch"));
            return;
        }
        let nonfinite = mlx
            .iter()
            .zip(xla)
            .enumerate()
            .find(|(_, (left, right))| !left.is_finite() || !right.is_finite())
            .map(|(index, (&left, &right))| (index, left, right));
        if let Some(nonfinite) = nonfinite {
            eprintln!("{name}: non-finite value at {nonfinite:?}");
            failures.push(format!("{name}: non-finite value at {nonfinite:?}"));
            return;
        }
        let mut max_abs = 0.0f32;
        let mut first = None;
        let mut square_error = 0.0f64;
        let mut mlx_square = 0.0f64;
        let mut xla_square = 0.0f64;
        let mut dot = 0.0f64;
        for (index, (&left, &right)) in mlx.iter().zip(xla).enumerate() {
            let delta = (left - right).abs();
            max_abs = max_abs.max(delta);
            if delta != 0.0 && first.is_none() {
                first = Some((index, left, right, delta));
            }
            let left = f64::from(left);
            let right = f64::from(right);
            let error = left - right;
            square_error += error * error;
            mlx_square += left * left;
            xla_square += right * right;
            dot += left * right;
        }
        let count = mlx.len().max(1) as f64;
        let rmse = (square_error / count).sqrt();
        let mlx_rms = (mlx_square / count).sqrt();
        let normalized_rmse = if mlx_rms == 0.0 {
            if rmse == 0.0 { 0.0 } else { f64::INFINITY }
        } else {
            rmse / mlx_rms
        };
        let cosine = if mlx_square == 0.0 || xla_square == 0.0 {
            if mlx_square == 0.0 && xla_square == 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            dot / (mlx_square.sqrt() * xla_square.sqrt())
        };
        eprintln!(
            "{name}: len={} max_abs={max_abs} cosine={cosine:.8} \
             normalized_rmse={normalized_rmse:.8} first={first:?} \
             envelope(max_abs<={}, cosine>={}, normalized_rmse<={})",
            mlx.len(),
            envelope.max_abs,
            envelope.min_cosine,
            envelope.max_normalized_rmse,
        );
        if max_abs > envelope.max_abs
            || cosine < envelope.min_cosine
            || normalized_rmse > envelope.max_normalized_rmse
        {
            failures.push(format!(
                "{name}: max_abs={max_abs} cosine={cosine:.8} \
                 normalized_rmse={normalized_rmse:.8} first={first:?}"
            ));
        }
    }

    fn compare_top1(name: &str, mlx: &[f32], xla: &[f32], failures: &mut Vec<String>) {
        let mlx_top1 = argmax(mlx);
        let xla_top1 = argmax(xla);
        eprintln!("{name} top1: mlx={mlx_top1} xla={xla_top1}");
        if mlx_top1 != xla_top1 {
            failures.push(format!(
                "{name}: top1 mismatch mlx={mlx_top1} xla={xla_top1}"
            ));
        }
    }

    fn compare_exact(name: &str, left: &[f32], right: &[f32], failures: &mut Vec<String>) {
        if left.len() != right.len() {
            eprintln!("{name}: length left={} right={}", left.len(), right.len());
            failures.push(format!("{name}: length mismatch"));
            return;
        }
        if let Some((index, (&left, &right))) = left
            .iter()
            .zip(right)
            .enumerate()
            .find(|(_, (left, right))| left.to_bits() != right.to_bits())
        {
            let delta = (left - right).abs();
            eprintln!(
                "{name}: first mismatch index={index} left={left} right={right} delta={delta}"
            );
            failures.push(format!(
                "{name}: first mismatch index={index} left={left} right={right} delta={delta}"
            ));
        } else {
            eprintln!("{name}: exact match across {} values", left.len());
        }
    }

    fn probe_real_projection(name: &str, linear: &UnifiedLinear, input: &MlxArray) {
        let input_shape = mlxcel_core::array_shape(input);
        let k = *input_shape.last().unwrap() as usize;
        let m = input_shape[..input_shape.len() - 1]
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>();
        let input_2d = mlxcel_core::reshape(input, &[m as i32, k as i32]);
        let full = array_to_f32(&linear.forward(&input_2d));

        let mut per_row = Vec::with_capacity(full.len());
        for row in 0..m {
            let one = mlxcel_core::slice(&input_2d, &[row as i32, 0], &[row as i32 + 1, k as i32]);
            per_row.extend(array_to_f32(&linear.forward(&one)));
        }
        let stock_mismatch = full
            .iter()
            .zip(&per_row)
            .enumerate()
            .find(|(_, (left, right))| left.to_bits() != right.to_bits());

        let weight = linear.dequantized_weight();
        let weight_shape = mlxcel_core::array_shape(&weight);
        assert_eq!(weight_shape.len(), 2);
        assert_eq!(weight_shape[1] as usize, k);
        let n = weight_shape[0] as usize;
        let input_values = array_to_f32(&input_2d);
        let weight_values = array_to_f32(&weight);
        let quantized = linear
            .as_quantized_weight()
            .expect("real projection probe requires an affine quantized weight");
        assert_eq!(quantized.mode, "affine");
        let packed_shape = mlxcel_core::array_shape(&quantized.weight);
        assert_eq!(
            packed_shape,
            [n as i32, (k * quantized.bits as usize / 32) as i32]
        );
        let biases = quantized
            .biases
            .as_ref()
            .expect("affine quantized weight requires biases");
        let carrier_values = dequantize_gemma3n_affine_diagnostic(
            &mlxcel_core::array_to_raw_bytes(&quantized.weight),
            &mlxcel_core::array_to_raw_bytes(&quantized.scales),
            &mlxcel_core::array_to_raw_bytes(biases),
            n,
            packed_shape[1] as usize,
            quantized.bits as usize,
            quantized.group_size as usize,
        )
        .unwrap();
        let mut carrier_max_abs = 0.0f32;
        let mut carrier_first = None;
        let mut carrier_bit_mismatches = 0usize;
        for (index, (&mlx, &rust)) in weight_values.iter().zip(&carrier_values).enumerate() {
            carrier_max_abs = carrier_max_abs.max((mlx - rust).abs());
            if mlx.to_bits() != rust.to_bits() {
                carrier_bit_mismatches += 1;
                carrier_first.get_or_insert((index, mlx, rust, (mlx - rust).abs()));
            }
        }
        let native =
            run_gemma3n_qmv_diagnostic_probe(&input_values, &weight_values, m, n, k).unwrap();
        let mut max_abs = 0.0f32;
        let mut first = None;
        let mut bit_mismatches = 0usize;
        for (index, (&mlx, &xla)) in full.iter().zip(&native).enumerate() {
            max_abs = max_abs.max((mlx - xla).abs());
            if mlx.to_bits() != xla.to_bits() {
                bit_mismatches += 1;
                first.get_or_insert((index, mlx, xla, (mlx - xla).abs()));
            }
        }
        eprintln!(
            "real_qmv {name}: shape M={m} N={n} K={k} \
             multirow_vs_perrow_first={stock_mismatch:?} \
             carrier_bit_mismatches={carrier_bit_mismatches}/{} \
             carrier_max_abs={carrier_max_abs} carrier_first={carrier_first:?} \
             native_bit_mismatches={bit_mismatches}/{} max_abs={max_abs} first={first:?}",
            weight_values.len(),
            full.len()
        );
        assert_eq!(
            carrier_bit_mismatches, 0,
            "{name}: Rust carrier differs from MLX dequantized weight"
        );
    }

    fn assert_real_projection_weight_carrier(name: &str, linear: &UnifiedLinear) -> Vec<f32> {
        let weight = linear.dequantized_weight();
        let shape = mlxcel_core::array_shape(&weight);
        assert_eq!(shape.len(), 2);
        let n = shape[0] as usize;
        let k = shape[1] as usize;
        let mlx = array_to_f32(&weight);
        let quantized = linear
            .as_quantized_weight()
            .expect("real projection carrier probe requires an affine quantized weight");
        assert_eq!(quantized.mode, "affine");
        let packed_shape = mlxcel_core::array_shape(&quantized.weight);
        assert_eq!(
            packed_shape,
            [n as i32, (k * quantized.bits as usize / 32) as i32]
        );
        let biases = quantized
            .biases
            .as_ref()
            .expect("affine quantized weight requires biases");
        let rust = dequantize_gemma3n_affine_diagnostic(
            &mlxcel_core::array_to_raw_bytes(&quantized.weight),
            &mlxcel_core::array_to_raw_bytes(&quantized.scales),
            &mlxcel_core::array_to_raw_bytes(biases),
            n,
            packed_shape[1] as usize,
            quantized.bits as usize,
            quantized.group_size as usize,
        )
        .unwrap();
        let mut mismatches = 0usize;
        let mut max_abs = 0.0f32;
        let mut first = None;
        for (index, (&mlx, &rust)) in mlx.iter().zip(&rust).enumerate() {
            let delta = (mlx - rust).abs();
            max_abs = max_abs.max(delta);
            if mlx.to_bits() != rust.to_bits() {
                mismatches += 1;
                first.get_or_insert((index, mlx, rust, delta));
            }
        }
        eprintln!(
            "real_weight_carrier {name}: shape N={n} K={k} \
             bit_mismatches={mismatches}/{} max_abs={max_abs} first={first:?}",
            mlx.len()
        );
        assert_eq!(
            mismatches, 0,
            "{name}: Rust carrier differs from MLX dequantized weight"
        );
        mlx
    }

    fn assert_checkpoint_bf16_weight_carrier(
        model_dir: &Path,
        canonical_name: &str,
        mlx_weight: &MlxArray,
        clip: Option<f32>,
    ) -> Vec<f32> {
        let file = std::fs::File::open(model_dir.join("model.safetensors")).unwrap();
        // Safety: the checkpoint is mapped read-only and remains open while the
        // safetensors view is consumed in this helper.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&mmap).unwrap();
        let suffix = canonical_name
            .strip_prefix("model.language_model.")
            .expect("canonical Gemma3n language-model tensor name");
        let checkpoint_name = format!("language_model.model.{suffix}");
        let tensor = tensors.tensor(&checkpoint_name).unwrap();
        assert_eq!(tensor.dtype(), safetensors::Dtype::BF16);
        let mut checkpoint = tensor
            .data()
            .chunks_exact(2)
            .map(|bytes| f32::from_bits((u16::from_le_bytes([bytes[0], bytes[1]]) as u32) << 16))
            .collect::<Vec<_>>();
        if let Some(limit) = clip {
            for value in &mut checkpoint {
                *value = value.clamp(-limit, limit);
            }
        }
        let mlx = array_to_f32(mlx_weight);
        assert_eq!(mlx.len(), checkpoint.len());
        let mut mismatches = 0usize;
        let mut max_abs = 0.0f32;
        let mut first = None;
        for (index, (&mlx, &checkpoint)) in mlx.iter().zip(&checkpoint).enumerate() {
            let delta = (mlx - checkpoint).abs();
            max_abs = max_abs.max(delta);
            if mlx.to_bits() != checkpoint.to_bits() {
                mismatches += 1;
                first.get_or_insert((index, mlx, checkpoint, delta));
            }
        }
        eprintln!(
            "checkpoint_bf16_weight_carrier {canonical_name}: shape={:?} \
             bit_mismatches={mismatches}/{} max_abs={max_abs} first={first:?}",
            tensor.shape(),
            mlx.len()
        );
        assert_eq!(
            mismatches, 0,
            "{canonical_name}: checkpoint carrier differs from MLX f32 weight"
        );
        mlx
    }

    fn probe_real_rms_norm(name: &str, norm: &RMSNorm, input: &MlxArray) {
        let shape = mlxcel_core::array_shape(input);
        let width = *shape.last().unwrap() as usize;
        let rows = shape[..shape.len() - 1]
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>();
        let mlx = array_to_f32(&norm.forward(input));
        let input = array_to_f32(input);
        let weight = array_to_f32(&norm.weight);
        let xla = run_gemma3n_rms_diagnostic_probe(&input, &weight, rows, width, norm.eps).unwrap();
        let mut max_abs = 0.0f32;
        let mut first = None;
        let mut bit_mismatches = 0usize;
        for (index, (&mlx, &xla)) in mlx.iter().zip(&xla).enumerate() {
            max_abs = max_abs.max((mlx - xla).abs());
            if mlx.to_bits() != xla.to_bits() {
                bit_mismatches += 1;
                first.get_or_insert((index, mlx, xla, (mlx - xla).abs()));
            }
        }
        eprintln!(
            "real_rms {name}: shape rows={rows} width={width} \
             bit_mismatches={bit_mismatches}/{} max_abs={max_abs} first={first:?}",
            mlx.len()
        );
        assert_eq!(
            bit_mismatches, 0,
            "{name}: narrow XLA RMSNorm differs from MLX"
        );
    }

    fn compare_narrow_stage(name: &str, mlx: &[f32], xla: &[f32], failures: &mut Vec<String>) {
        assert_eq!(mlx.len(), xla.len(), "{name}: stage length mismatch");
        let mut max_abs = 0.0f32;
        let mut first = None;
        let mut bit_mismatches = 0usize;
        let mlx_bf16_carriers = mlx
            .iter()
            .filter(|value| value.to_bits() & 0xffff == 0)
            .count();
        let xla_bf16_carriers = xla
            .iter()
            .filter(|value| value.to_bits() & 0xffff == 0)
            .count();
        for (index, (&mlx, &xla)) in mlx.iter().zip(xla).enumerate() {
            max_abs = max_abs.max((mlx - xla).abs());
            if mlx.to_bits() != xla.to_bits() {
                bit_mismatches += 1;
                first.get_or_insert((index, mlx, xla, (mlx - xla).abs()));
            }
        }
        eprintln!(
            "narrow_stage {name}: bit_mismatches={bit_mismatches}/{} \
             max_abs={max_abs} first={first:?} \
             bf16_carriers_mlx={mlx_bf16_carriers}/{} \
             bf16_carriers_xla={xla_bf16_carriers}/{}",
            mlx.len(),
            xla.len(),
            mlx.len()
        );
        if bit_mismatches != 0 {
            failures.push(format!(
                "{name}: {bit_mismatches}/{} mismatches, max_abs={max_abs}",
                mlx.len()
            ));
        }
    }

    fn take_diagnostic_stage<'a>(values: &'a [f32], cursor: &mut usize, len: usize) -> &'a [f32] {
        let start = *cursor;
        *cursor += len;
        values
            .get(start..*cursor)
            .expect("diagnostic stage output is truncated")
    }

    fn round_tf32_rne(value: f32) -> f32 {
        let bits = value.to_bits();
        if bits & 0x7f80_0000 == 0x7f80_0000 {
            return value;
        }
        let retained_lsb = (bits >> 13) & 1;
        f32::from_bits(bits.wrapping_add(0x0fff + retained_lsb) & 0xffff_e000)
    }

    fn round_bf16_rne(value: f32) -> f32 {
        let bits = value.to_bits();
        if bits & 0x7f80_0000 == 0x7f80_0000 {
            return value;
        }
        f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
    }

    fn permutations4() -> Vec<[usize; 4]> {
        let mut permutations = Vec::with_capacity(24);
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        if a != b && a != c && a != d && b != c && b != d && c != d {
                            permutations.push([a, b, c, d]);
                        }
                    }
                }
            }
        }
        permutations
    }

    fn add_tree4(products: &[f32; 4], order: [usize; 4], tree: usize) -> f32 {
        let [a, b, c, d] = order.map(|index| products[index]);
        match tree {
            0 => ((a + b) + c) + d,
            1 => (a + (b + c)) + d,
            2 => a + ((b + c) + d),
            3 => a + (b + (c + d)),
            4 => (a + b) + (c + d),
            _ => unreachable!("five binary trees for four leaves"),
        }
    }

    fn analyze_altup_predict_schedule(
        name: &str,
        layer: &DecoderLayer,
        stacked_planes: &MlxArray,
        rows: usize,
        hidden: usize,
        plane_count: usize,
    ) -> Vec<f32> {
        assert_eq!(
            plane_count, 4,
            "schedule enumeration is pinned to AltUp K=4"
        );
        let active = slice_altup_plane(stacked_planes, layer.altup_active_idx);
        let router_norm = layer.altup.router_norm.forward(&active);
        let router_scaled = mlxcel_core::multiply_scalar(&router_norm, (hidden as f32).powf(-1.0));
        let modalities = layer.altup.modality_router.forward(&router_scaled);
        let modalities = mlxcel_core::astype(&modalities, mlxcel_core::dtype::FLOAT32);
        let modalities = mlxcel_core::tanh(&modalities);
        let coefficients = layer.altup.prediction_coefs.forward(&modalities);
        let coefficients = mlxcel_core::reshape(
            &coefficients,
            &[1, rows as i32, plane_count as i32, plane_count as i32],
        );
        let coefficients = mlxcel_core::transpose_axes(&coefficients, &[0, 1, 3, 2]);
        let planes_f32 = mlxcel_core::astype(stacked_planes, mlxcel_core::dtype::FLOAT32);
        let by_feature = mlxcel_core::transpose_axes(&planes_f32, &[1, 2, 3, 0]);
        let delta = mlxcel_core::matmul(&by_feature, &coefficients);
        let delta = mlxcel_core::transpose_axes(&delta, &[3, 0, 1, 2]);
        let pre_round = mlxcel_core::add(&planes_f32, &delta);
        let predicted = mlxcel_core::astype(&pre_round, mlxcel_core::dtype::BFLOAT16);
        let planes = array_to_f32(&planes_f32);
        let coefficients = array_to_f32(&coefficients);
        let delta = array_to_f32(&delta);
        let predicted = array_to_f32(&predicted);
        let plane_len = rows * hidden;
        let output_len = plane_count * plane_len;
        assert_eq!(planes.len(), output_len);
        assert_eq!(delta.len(), output_len);
        assert_eq!(predicted.len(), output_len);

        let measure = |candidate: &[f32], expected: &[f32]| {
            let mut mismatches = 0usize;
            let mut max_abs = 0.0f32;
            let mut first = None;
            for (index, (&candidate, &expected)) in candidate.iter().zip(expected).enumerate() {
                let difference = (candidate - expected).abs();
                max_abs = max_abs.max(difference);
                if candidate.to_bits() != expected.to_bits() {
                    mismatches += 1;
                    first.get_or_insert((index, expected, candidate, difference));
                }
            }
            (mismatches, max_abs, first)
        };
        let mut separate_from_mlx_delta = Vec::with_capacity(output_len);
        for index in 0..output_len {
            separate_from_mlx_delta.push(round_bf16_rne(planes[index] + delta[index]));
        }
        let separate_summary = measure(&separate_from_mlx_delta, &predicted);

        let permutations = permutations4();
        let mut best_fma_delta = None;
        let mut best_fma_final = None;
        let mut best_seeded_final = None;
        let mut best_tree_delta = None;
        let mut best_tree_final = None;
        for order in permutations {
            let mut fma_delta = Vec::with_capacity(output_len);
            let mut fma_final = Vec::with_capacity(output_len);
            let mut seeded_final = Vec::with_capacity(output_len);
            let mut tree_deltas = (0..5)
                .map(|_| Vec::with_capacity(output_len))
                .collect::<Vec<_>>();
            let mut tree_finals = (0..5)
                .map(|_| Vec::with_capacity(output_len))
                .collect::<Vec<_>>();
            for target in 0..plane_count {
                for row in 0..rows {
                    for feature in 0..hidden {
                        let within = row * hidden + feature;
                        let output_index = target * plane_len + within;
                        let mut lhs = [0.0f32; 4];
                        let mut rhs = [0.0f32; 4];
                        let mut products = [0.0f32; 4];
                        for source in 0..plane_count {
                            lhs[source] = round_tf32_rne(planes[source * plane_len + within]);
                            rhs[source] = round_tf32_rne(
                                coefficients[(row * plane_count + source) * plane_count + target],
                            );
                            products[source] = lhs[source] * rhs[source];
                        }
                        let mut sum = 0.0f32;
                        let mut seeded = planes[output_index];
                        for source in order {
                            sum = lhs[source].mul_add(rhs[source], sum);
                            seeded = lhs[source].mul_add(rhs[source], seeded);
                        }
                        fma_delta.push(sum);
                        fma_final.push(round_bf16_rne(planes[output_index] + sum));
                        seeded_final.push(round_bf16_rne(seeded));
                        for tree in 0..5 {
                            let tree_sum = add_tree4(&products, order, tree);
                            tree_deltas[tree].push(tree_sum);
                            tree_finals[tree].push(round_bf16_rne(planes[output_index] + tree_sum));
                        }
                    }
                }
            }
            let fma_delta_summary = measure(&fma_delta, &delta);
            let fma_final_summary = measure(&fma_final, &predicted);
            let seeded_final_summary = measure(&seeded_final, &predicted);
            if best_fma_delta.as_ref().is_none_or(
                |(_, (mismatches, _, _)): &([usize; 4], (usize, f32, _))| {
                    fma_delta_summary.0 < *mismatches
                },
            ) {
                best_fma_delta = Some((order, fma_delta_summary));
            }
            if best_fma_final.as_ref().is_none_or(
                |(_, (mismatches, _, _)): &([usize; 4], (usize, f32, _))| {
                    fma_final_summary.0 < *mismatches
                },
            ) {
                best_fma_final = Some((order, fma_final_summary));
            }
            if best_seeded_final.as_ref().is_none_or(
                |(_, (mismatches, _, _)): &([usize; 4], (usize, f32, _))| {
                    seeded_final_summary.0 < *mismatches
                },
            ) {
                best_seeded_final = Some((order, seeded_final_summary));
            }
            for tree in 0..5 {
                let tree_delta_summary = measure(&tree_deltas[tree], &delta);
                let tree_final_summary = measure(&tree_finals[tree], &predicted);
                if best_tree_delta.as_ref().is_none_or(
                    |(_, _, (mismatches, _, _)): &([usize; 4], usize, (usize, f32, _))| {
                        tree_delta_summary.0 < *mismatches
                    },
                ) {
                    best_tree_delta = Some((order, tree, tree_delta_summary));
                }
                if best_tree_final.as_ref().is_none_or(
                    |(_, _, (mismatches, _, _)): &([usize; 4], usize, (usize, f32, _))| {
                        tree_final_summary.0 < *mismatches
                    },
                ) {
                    best_tree_final = Some((order, tree, tree_final_summary));
                }
            }
        }
        eprintln!(
            "altup_predict_schedule {name}: separate_mlx_delta={separate_summary:?} \
             best_fma_delta={best_fma_delta:?} best_fma_final={best_fma_final:?} \
             best_seeded_final={best_seeded_final:?} \
             best_tree_delta={best_tree_delta:?} best_tree_final={best_tree_final:?}"
        );
        if rows > 1 && hidden > 472 {
            let row = 1;
            let feature = 472;
            let target = 1;
            let within = row * hidden + feature;
            let output_index = target * plane_len + within;
            let inputs = (0..plane_count)
                .map(|source| planes[source * plane_len + within])
                .collect::<Vec<_>>();
            let coefficients = (0..plane_count)
                .map(|source| coefficients[(row * plane_count + source) * plane_count + target])
                .collect::<Vec<_>>();
            eprintln!(
                "altup_predict_schedule_carrier {name}: plane=1 row=1 hidden=472 \
                 inputs={inputs:?} coefficients={coefficients:?} delta_mlx={} \
                 residual={} pre_round_mlx={} final_bf16_mlx={}",
                delta[output_index],
                planes[output_index],
                planes[output_index] + delta[output_index],
                predicted[output_index],
            );
        }
        assert_eq!(
            separate_summary.0, 0,
            "{name}: MLX residual add and BF16 cast are not separate"
        );
        predicted
    }

    fn mlp_input_projection_weight(projection: &MlpInputProjection) -> UniquePtr<MlxArray> {
        match projection {
            MlpInputProjection::Standard(linear) => linear.dequantized_weight(),
            MlpInputProjection::Pretransposed { weight_t, .. } => mlxcel_core::transpose(weight_t),
        }
    }

    #[test]
    #[ignore = "requires pinned Gemma3n E2B checkpoint plus MLX and IREE CUDA"]
    fn pinned_e2b_real_projection_qmv_probe() {
        let model_dir_env = std::env::var_os("GEMMA3N_MODEL_DIR").expect("set GEMMA3N_MODEL_DIR");
        let model_dir = Path::new(&model_dir_env);
        assert_eq!(sha256(&model_dir.join("model.safetensors")), MODEL_SHA256);
        let (loaded, _) = crate::load_model(model_dir).expect("load pinned MLX checkpoint");
        let language_model = match &loaded {
            LoadedModel::Gemma3n(model) => &model.language_model,
            LoadedModel::Gemma3nVLM(model) => &model.text_model.language_model,
            _ => panic!("pinned checkpoint did not load as Gemma3n"),
        };
        let input = mlxcel_core::from_slice_i32(&PROMPT, &[1, PROMPT.len() as i32]);
        let scaled = language_model.get_embed_tokens(&input);
        let rows = PROMPT.len();
        let hidden = language_model.config.hidden_size;
        let layer_count = language_model.config.num_hidden_layers;
        let ple_width = language_model.config.hidden_size_per_layer_input;
        let plane_count = language_model.config.altup_num_inputs;
        let mut failures = Vec::new();

        probe_real_projection(
            "per_layer_model_projection",
            &language_model.per_layer_model_projection,
            &scaled,
        );

        let token_ple = language_model.get_per_layer_inputs(&input);
        let raw_ple = language_model.per_layer_model_projection.forward(&scaled);
        let scaled_ple = mlxcel_core::multiply_scalar(&raw_ple, (hidden as f32).powf(-0.5));
        let reshaped_ple = mlxcel_core::reshape(
            &scaled_ple,
            &[1, rows as i32, layer_count as i32, ple_width as i32],
        );
        let normed_ple = language_model
            .per_layer_projection_norm
            .forward(&reshaped_ple);
        let added_ple = mlxcel_core::add(&normed_ple, &token_ple);
        let combined_ple =
            mlxcel_core::multiply_scalar(&added_ple, std::f32::consts::FRAC_1_SQRT_2);
        let raw_ple_values = array_to_f32(&raw_ple);
        let token_ple_values = array_to_f32(&token_ple);
        let ple_stage_len = rows * layer_count * ple_width;
        let xla_ple = run_gemma3n_ple_diagnostic_probe(
            &raw_ple_values,
            &token_ple_values,
            &array_to_f32(&language_model.per_layer_projection_norm.weight),
            rows,
            layer_count,
            ple_width,
            hidden,
            language_model.per_layer_projection_norm.eps,
        );
        let xla_ple = xla_ple.unwrap();
        for (index, (name, expected)) in [
            ("ple_raw_projection", raw_ple_values),
            ("ple_hidden^-0.5_scale", array_to_f32(&scaled_ple)),
            ("ple_rms_norm", array_to_f32(&normed_ple)),
            ("ple_token_add", array_to_f32(&added_ple)),
            ("ple_inv_sqrt2", array_to_f32(&combined_ple)),
        ]
        .into_iter()
        .enumerate()
        {
            compare_narrow_stage(
                name,
                &expected,
                &xla_ple[index * ple_stage_len..(index + 1) * ple_stage_len],
                &mut failures,
            );
        }

        let target_magnitude = compute_magnitude(&scaled);
        let mut planes = vec![mlxcel_core::copy(&scaled)];
        for (index, projection) in language_model.altup_projections.iter().enumerate() {
            probe_real_projection(&format!("initial_projection{index}"), projection, &scaled);
            planes.push(projection.forward(&scaled));
        }
        let raw_projection_values = planes[1..]
            .iter()
            .flat_map(|plane| array_to_f32(plane))
            .collect::<Vec<_>>();
        let xla_initial = run_gemma3n_initial_altup_diagnostic_probe(
            &array_to_f32(&scaled),
            &raw_projection_values,
            plane_count - 1,
            rows,
            hidden,
            1e-6,
        )
        .unwrap();
        compare_narrow_stage(
            "initial_target_magnitude",
            &array_to_f32(&target_magnitude),
            &xla_initial[..rows],
            &mut failures,
        );
        normalize_magnitudes(&mut planes, &target_magnitude);
        let plane_len = rows * hidden;
        for (index, plane) in planes[1..].iter().enumerate() {
            let offset = rows + index * plane_len;
            compare_narrow_stage(
                &format!("initial_normalized_plane{}", index + 1),
                &array_to_f32(plane),
                &xla_initial[offset..offset + plane_len],
                &mut failures,
            );
        }

        let layer0 = &language_model.layers[0];
        let active = &planes[layer0.altup_active_idx];
        let router_norm = layer0.altup.router_norm.forward(active);
        let router_scaled = mlxcel_core::multiply_scalar(&router_norm, (hidden as f32).powf(-1.0));
        let modalities = layer0.altup.modality_router.forward(&router_scaled);
        let modalities = mlxcel_core::astype(&modalities, mlxcel_core::dtype::FLOAT32);
        let modalities = mlxcel_core::tanh(&modalities);
        assert!(
            layer0.altup.prediction_coefs.bias.is_none(),
            "pinned AltUp prediction probe does not model a bias"
        );
        let coefficients = layer0.altup.prediction_coefs.forward(&modalities);
        let coefficients = mlxcel_core::reshape(
            &coefficients,
            &[1, rows as i32, plane_count as i32, plane_count as i32],
        );
        let coefficients = mlxcel_core::transpose_axes(&coefficients, &[0, 1, 3, 2]);
        let predicted_stacked = layer0.altup.predict_stacked(&planes);
        let predicted_active = slice_altup_plane(&predicted_stacked, layer0.altup_active_idx);
        let input_norm = layer0.input_layernorm.forward(&predicted_active);
        let stacked_planes = stack_arrays(&planes, 0);
        let xla_predict = run_gemma3n_altup_predict_diagnostic_probe(
            &array_to_f32(&stacked_planes),
            &array_to_f32(&layer0.altup.router_norm.weight),
            &array_to_f32(&layer0.altup.modality_router.dequantized_weight()),
            &array_to_f32(&layer0.altup.prediction_coefs.weight),
            &array_to_f32(&layer0.input_layernorm.weight),
            plane_count,
            layer0.altup_active_idx,
            rows,
            hidden,
            layer0.input_layernorm.eps,
            language_model.config.altup_coef_clip,
        )
        .unwrap();
        let mut cursor = 0usize;
        for (name, expected) in [
            ("layer0_router_norm", array_to_f32(&router_norm)),
            ("layer0_router_scale", array_to_f32(&router_scaled)),
            ("layer0_modalities", array_to_f32(&modalities)),
            (
                "layer0_prediction_coefficients",
                array_to_f32(&coefficients),
            ),
            ("layer0_predicted_all", array_to_f32(&predicted_stacked)),
            ("layer0_predicted_active", array_to_f32(&predicted_active)),
            ("layer0_input_norm", array_to_f32(&input_norm)),
        ] {
            let len = expected.len();
            compare_narrow_stage(
                name,
                &expected,
                &xla_predict[cursor..cursor + len],
                &mut failures,
            );
            cursor += len;
        }
        assert_eq!(cursor, xla_predict.len());

        let predicted = split_altup_planes(&predicted_stacked, plane_count);
        probe_real_rms_norm(
            "layer0_input_norm",
            &layer0.input_layernorm,
            &predicted[layer0.altup_active_idx],
        );
        let normalized = layer0
            .input_layernorm
            .forward(&predicted[layer0.altup_active_idx]);
        probe_real_projection("layer0_q_proj", &layer0.self_attn.q_proj, &normalized);
        probe_real_projection(
            "layer0_laurel_left",
            &layer0.laurel.linear_left,
            &normalized,
        );

        assert_eq!(
            layer0.altup_active_idx, 0,
            "pinned PLE residual probe expects active AltUp plane zero"
        );
        let per_layer_input = slice_layer_input(&combined_ple, 0, 1, rows as i32, ple_width as i32);
        let mut layer0_caches = language_model.make_caches();
        let cache_index = language_model.layer_idx_to_cache_idx[0];
        let mask = if language_model.config.layer_types[0] == "full_attention" {
            create_causal_mask(rows as i32, layer0_caches[cache_index].live_len())
        } else {
            create_sliding_window_prefill_mask_dense(
                rows as i32,
                layer0_caches[cache_index].live_len(),
                language_model.config.sliding_window as i32,
            )
        };
        let traced = layer0.forward_diagnostics(
            &planes,
            Some(mask.as_ref().unwrap()),
            &mut layer0_caches[cache_index],
            &per_layer_input,
        );

        // Bisect the earliest canonical layer0 mismatch using exact MLX stage
        // carriers as the XLA probe inputs. This deliberately reconstructs the
        // MLX attention stages outside `forward` so diagnostics do not alter
        // the production graph or its fusion choices.
        let q_projection = layer0.self_attn.q_proj.forward(&normalized);
        let q = mlxcel_core::reshape(
            &q_projection,
            &[
                1,
                rows as i32,
                layer0.self_attn.num_heads,
                layer0.self_attn.head_dim,
            ],
        );
        let q = layer0.self_attn.q_norm.forward(&q);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let q = mlxcel_core::fast_rope(
            &q,
            layer0.self_attn.head_dim,
            false,
            layer0.self_attn.rope_theta,
            1.0,
            0,
        );
        let q_rows = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let mut attention_caches = language_model.make_caches();
        let (keys, values) = layer0.self_attn.compute_kv(
            &normalized,
            1,
            rows as i32,
            0,
            &mut attention_caches[cache_index],
        );
        assert_eq!(
            mlxcel_core::array_shape(&keys),
            vec![
                1,
                layer0.self_attn.num_kv_heads,
                rows as i32,
                layer0.self_attn.head_dim,
            ],
            "attention diagnostic must use the live K slice"
        );
        assert_eq!(
            mlxcel_core::array_shape(&values),
            vec![
                1,
                layer0.self_attn.num_kv_heads,
                rows as i32,
                layer0.self_attn.head_dim,
            ],
            "attention diagnostic must use the live V slice"
        );
        let context = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &q,
                &keys,
                &values,
                layer0.self_attn.scale,
                mask.as_ref().unwrap() as *const MlxArray,
                0.0,
                layer0.self_attn.window_size,
            )
        };
        let context = mlxcel_core::transpose_axes(&context, &[0, 2, 1, 3]);
        let context = mlxcel_core::reshape(
            &context,
            &[
                1,
                rows as i32,
                layer0.self_attn.num_heads * layer0.self_attn.head_dim,
            ],
        );
        let query_group = layer0.self_attn.num_heads / layer0.self_attn.num_kv_heads;
        let q_scaled = mlxcel_core::multiply_scalar(&q, layer0.self_attn.scale);
        let q_grouped = mlxcel_core::reshape(
            &q_scaled,
            &[
                1,
                layer0.self_attn.num_kv_heads,
                query_group,
                rows as i32,
                layer0.self_attn.head_dim,
            ],
        );
        let keys_grouped = mlxcel_core::reshape(
            &keys,
            &[
                1,
                layer0.self_attn.num_kv_heads,
                1,
                rows as i32,
                layer0.self_attn.head_dim,
            ],
        );
        let keys_transposed = mlxcel_core::transpose_axes(&keys_grouped, &[0, 1, 2, 4, 3]);
        let raw_scores = mlxcel_core::matmul(&q_grouped, &keys_transposed);
        let mask_bf16 = mlxcel_core::astype(mask.as_ref().unwrap(), mlxcel_core::dtype::BFLOAT16);
        let mask_heads = mlxcel_core::broadcast_to(
            &mask_bf16,
            &[1, layer0.self_attn.num_heads, rows as i32, rows as i32],
        );
        let mask_grouped = mlxcel_core::reshape(
            &mask_heads,
            &[
                1,
                layer0.self_attn.num_kv_heads,
                query_group,
                rows as i32,
                rows as i32,
            ],
        );
        let masked_scores = mlxcel_core::add(&raw_scores, &mask_grouped);
        let probabilities = mlxcel_core::softmax_precise(&masked_scores, -1);
        let values_grouped = mlxcel_core::reshape(
            &values,
            &[
                1,
                layer0.self_attn.num_kv_heads,
                1,
                rows as i32,
                layer0.self_attn.head_dim,
            ],
        );
        let explicit_context = mlxcel_core::matmul(&probabilities, &values_grouped);
        let explicit_context = mlxcel_core::transpose_axes(&explicit_context, &[0, 3, 1, 2, 4]);
        let explicit_context = mlxcel_core::reshape(
            &explicit_context,
            &[
                1,
                rows as i32,
                layer0.self_attn.num_heads * layer0.self_attn.head_dim,
            ],
        );
        for (name, value) in [
            ("q", q.as_ref().unwrap()),
            ("raw_scores", raw_scores.as_ref().unwrap()),
            ("masked_scores", masked_scores.as_ref().unwrap()),
            ("probabilities", probabilities.as_ref().unwrap()),
            ("context", explicit_context.as_ref().unwrap()),
        ] {
            assert_eq!(
                mlxcel_core::array_dtype(value),
                mlxcel_core::dtype::BFLOAT16,
                "MLX materialized SDPA {name} must remain BF16"
            );
        }
        compare_narrow_stage(
            "layer0_attention_explicit_context_vs_dispatch",
            &array_to_f32(&context),
            &array_to_f32(&explicit_context),
            &mut failures,
        );
        let raw_scores = mlxcel_core::transpose_axes(&raw_scores, &[0, 1, 3, 2, 4]);
        let masked_scores = mlxcel_core::transpose_axes(&masked_scores, &[0, 1, 3, 2, 4]);
        let probabilities = mlxcel_core::transpose_axes(&probabilities, &[0, 1, 3, 2, 4]);
        let attention_projected = layer0.self_attn.o_proj.forward(&context);
        let attention_post_norm = layer0
            .post_attention_layernorm
            .forward(&attention_projected);
        let attention_residual =
            mlxcel_core::add(&predicted[layer0.altup_active_idx], &attention_post_norm);
        let attention_with_laurel = mlxcel_core::add(&attention_residual, &traced.laurel_output);
        let attention_laurel =
            mlxcel_core::multiply_scalar(&attention_with_laurel, std::f32::consts::FRAC_1_SQRT_2);
        let keys_rows = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
        let values_rows = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);
        let xla_attention = run_gemma3n_attention_diagnostic_probe(
            &array_to_f32(&q_projection),
            &array_to_f32(&layer0.self_attn.q_norm.weight),
            &array_to_f32(&keys_rows),
            &array_to_f32(&values_rows),
            &array_to_f32(mask.as_ref().unwrap()),
            &array_to_f32(&layer0.self_attn.o_proj.dequantized_weight()),
            &array_to_f32(&layer0.post_attention_layernorm.weight),
            &array_to_f32(&predicted[layer0.altup_active_idx]),
            &array_to_f32(&traced.laurel_output),
            rows,
            layer0.self_attn.num_heads as usize,
            layer0.self_attn.num_kv_heads as usize,
            layer0.self_attn.head_dim as usize,
            layer0.self_attn.rope_theta as f64,
            layer0.post_attention_layernorm.eps,
        )
        .unwrap();
        let mut attention_cursor = 0usize;
        for (name, expected) in [
            ("layer0_q_after_rope", array_to_f32(&q_rows)),
            ("layer0_attention_raw_scores", array_to_f32(&raw_scores)),
            (
                "layer0_attention_masked_scores",
                array_to_f32(&masked_scores),
            ),
            (
                "layer0_attention_probabilities",
                array_to_f32(&probabilities),
            ),
            ("layer0_attention_context", array_to_f32(&context)),
            (
                "layer0_attention_o_projection",
                array_to_f32(&attention_projected),
            ),
            (
                "layer0_attention_post_norm",
                array_to_f32(&attention_post_norm),
            ),
            (
                "layer0_attention_residual",
                array_to_f32(&attention_residual),
            ),
            ("layer0_attention_laurel", array_to_f32(&attention_laurel)),
        ] {
            let len = expected.len();
            compare_narrow_stage(
                name,
                &expected,
                &xla_attention[attention_cursor..attention_cursor + len],
                &mut failures,
            );
            attention_cursor += len;
        }
        assert_eq!(attention_cursor, xla_attention.len());

        // Continue the actual-input bisect from the now-exact scaled-LAUREL
        // carrier through the sparse MLP and AltUp correction. The standalone
        // MLX graph below spells out every gelu_topk intermediate; pin its final
        // BF16 output against the compiled production helper before using the
        // intermediates as a reference.
        let pre_ff_norm = layer0.pre_feedforward_layernorm.forward(&attention_laurel);
        let gate = layer0.mlp.gate_proj.forward(&pre_ff_norm);
        let up = layer0.mlp.up_proj.forward(&pre_ff_norm);
        assert!(
            layer0.mlp.activation_sparsity > 0.0,
            "pinned layer0 post-attention probe expects sparse gelu_topk"
        );
        let sparse_mean = mlxcel_core::mean_axis(&gate, -1, true);
        let sparse_centered = mlxcel_core::subtract(&gate, &sparse_mean);
        let sparse_variance =
            mlxcel_core::mean_axis(&mlxcel_core::square(&sparse_centered), -1, true);
        let sparse_stddev = mlxcel_core::sqrt(&sparse_variance);
        let sparse_multiplier =
            mlxcel_core::full_f32(&[1], layer0.mlp.std_multiplier, mlxcel_core::dtype::FLOAT32);
        let sparse_cutoff = mlxcel_core::add(
            &sparse_mean,
            &mlxcel_core::multiply(&sparse_stddev, &sparse_multiplier),
        );
        let sparse_shifted_raw = mlxcel_core::subtract(&gate, &sparse_cutoff);
        let sparse_zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
        let sparse_shifted = mlxcel_core::maximum(&sparse_shifted_raw, &sparse_zero);
        let sparse_sqrt2 =
            mlxcel_core::full_f32(&[1], std::f32::consts::SQRT_2, mlxcel_core::dtype::FLOAT32);
        let sparse_erf = mlxcel_core::erf(&mlxcel_core::divide(&sparse_shifted, &sparse_sqrt2));
        let sparse_half = mlxcel_core::full_f32(&[1], 0.5, mlxcel_core::dtype::FLOAT32);
        let sparse_one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
        let sparse_scale =
            mlxcel_core::multiply(&sparse_half, &mlxcel_core::add(&sparse_one, &sparse_erf));
        let sparse_activation_manual = mlxcel_core::astype(
            &mlxcel_core::multiply(&sparse_shifted, &sparse_scale),
            mlxcel_core::array_dtype(&gate),
        );
        let sparse_activation = layer0.mlp.gelu_topk(&gate);
        compare_narrow_stage(
            "layer0_sparse_activation_manual_vs_compiled",
            &array_to_f32(&sparse_activation),
            &array_to_f32(&sparse_activation_manual),
            &mut failures,
        );
        let sparse_product = mlxcel_core::multiply(&sparse_activation, &up);
        let mlp_down = layer0.mlp.down_proj.forward(&sparse_product);
        let post_ff_norm = layer0.post_feedforward_layernorm.forward(&mlp_down);
        let ff_residual = mlxcel_core::astype(
            &mlxcel_core::add(&attention_laurel, &post_ff_norm),
            mlxcel_core::dtype::BFLOAT16,
        );

        let correction_router_norm = layer0.altup.router_norm.forward(&ff_residual);
        let correction_router_scaled =
            mlxcel_core::multiply_scalar(&correction_router_norm, (hidden as f32).powf(-1.0));
        let correction_modalities = layer0
            .altup
            .modality_router
            .forward(&correction_router_scaled);
        let correction_modalities =
            mlxcel_core::astype(&correction_modalities, mlxcel_core::dtype::FLOAT32);
        let correction_modalities = mlxcel_core::tanh(&correction_modalities);
        assert!(
            layer0.altup.correction_coefs.bias.is_none(),
            "pinned AltUp correction probe does not model a bias"
        );
        let correction_coefficients = layer0
            .altup
            .correction_coefs
            .forward(&correction_modalities);
        let correction_one = mlxcel_core::full_f32(
            &[1],
            1.0,
            mlxcel_core::array_dtype(&correction_coefficients),
        );
        let correction_coefficients = mlxcel_core::add(&correction_coefficients, &correction_one);
        let correction_innovation = mlxcel_core::subtract(&ff_residual, &predicted_active);
        let correction_coefficients_planes =
            mlxcel_core::transpose_axes(&correction_coefficients, &[2, 0, 1]);
        let correction_coefficients_broadcast = mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(
                &correction_coefficients_planes,
                &[plane_count as i32, 1, rows as i32, 1],
            ),
            &[plane_count as i32, 1, rows as i32, hidden as i32],
        );
        let correction_innovation_expanded =
            mlxcel_core::reshape(&correction_innovation, &[1, 1, rows as i32, hidden as i32]);
        let correction_products = mlxcel_core::multiply(
            &correction_innovation_expanded,
            &correction_coefficients_broadcast,
        );
        let corrected_manual = mlxcel_core::astype(
            &mlxcel_core::add(&predicted_stacked, &correction_products),
            mlxcel_core::dtype::BFLOAT16,
        );
        let corrected_stacked =
            layer0
                .altup
                .correct_stacked(&predicted_stacked, &predicted_active, &ff_residual);
        compare_narrow_stage(
            "layer0_corrected_manual_vs_compiled",
            &array_to_f32(&corrected_stacked),
            &array_to_f32(&corrected_manual),
            &mut failures,
        );
        let corrected_active = slice_altup_plane(&corrected_stacked, layer0.altup_active_idx);
        let scaled_corrected_active =
            mlxcel_core::multiply(&corrected_active, &layer0.altup.correct_output_scale);

        for (name, value, expected_dtype) in [
            (
                "pre_ff_norm",
                pre_ff_norm.as_ref().unwrap(),
                mlxcel_core::dtype::BFLOAT16,
            ),
            ("gate", gate.as_ref().unwrap(), mlxcel_core::dtype::BFLOAT16),
            ("up", up.as_ref().unwrap(), mlxcel_core::dtype::BFLOAT16),
            (
                "sparse_activation",
                sparse_activation.as_ref().unwrap(),
                mlxcel_core::dtype::BFLOAT16,
            ),
            (
                "sparse_product",
                sparse_product.as_ref().unwrap(),
                mlxcel_core::dtype::BFLOAT16,
            ),
            (
                "mlp_down",
                mlp_down.as_ref().unwrap(),
                mlxcel_core::dtype::BFLOAT16,
            ),
            (
                "post_ff_norm",
                post_ff_norm.as_ref().unwrap(),
                mlxcel_core::dtype::BFLOAT16,
            ),
            (
                "ff_residual",
                ff_residual.as_ref().unwrap(),
                mlxcel_core::dtype::BFLOAT16,
            ),
        ] {
            assert_eq!(
                mlxcel_core::array_dtype(value),
                expected_dtype,
                "MLX post-attention {name} dtype drifted"
            );
        }

        let intermediate = mlxcel_core::array_shape(&gate).last().copied().unwrap() as usize;
        let gate_weight = mlp_input_projection_weight(&layer0.mlp.gate_proj);
        let up_weight = mlp_input_projection_weight(&layer0.mlp.up_proj);
        let xla_post_attention = run_gemma3n_post_attention_diagnostic_probe(
            &array_to_f32(&attention_laurel),
            &array_to_f32(&layer0.pre_feedforward_layernorm.weight),
            &array_to_f32(&gate_weight),
            &array_to_f32(&up_weight),
            &array_to_f32(&layer0.mlp.down_proj.dequantized_weight()),
            &array_to_f32(&layer0.post_feedforward_layernorm.weight),
            rows,
            hidden,
            intermediate,
            layer0.pre_feedforward_layernorm.eps,
            layer0.mlp.activation_sparsity,
        )
        .unwrap();
        let mut post_attention_cursor = 0usize;
        for (name, expected) in [
            ("layer0_pre_ff_norm", array_to_f32(&pre_ff_norm)),
            ("layer0_mlp_gate", array_to_f32(&gate)),
            ("layer0_mlp_up", array_to_f32(&up)),
            ("layer0_sparse_mean", array_to_f32(&sparse_mean)),
            ("layer0_sparse_variance", array_to_f32(&sparse_variance)),
            ("layer0_sparse_stddev", array_to_f32(&sparse_stddev)),
            ("layer0_sparse_cutoff", array_to_f32(&sparse_cutoff)),
            (
                "layer0_sparse_shifted_raw",
                array_to_f32(&sparse_shifted_raw),
            ),
            ("layer0_sparse_shifted", array_to_f32(&sparse_shifted)),
            ("layer0_sparse_erf", array_to_f32(&sparse_erf)),
            ("layer0_sparse_activation", array_to_f32(&sparse_activation)),
            ("layer0_sparse_product", array_to_f32(&sparse_product)),
            ("layer0_mlp_down", array_to_f32(&mlp_down)),
            ("layer0_post_ff_norm", array_to_f32(&post_ff_norm)),
            ("layer0_ff_residual", array_to_f32(&ff_residual)),
        ] {
            let len = expected.len();
            compare_narrow_stage(
                name,
                &expected,
                &xla_post_attention[post_attention_cursor..post_attention_cursor + len],
                &mut failures,
            );
            post_attention_cursor += len;
        }
        assert_eq!(post_attention_cursor, xla_post_attention.len());

        let xla_altup_correct = run_gemma3n_altup_correct_diagnostic_probe(
            &array_to_f32(&predicted_stacked),
            &array_to_f32(&predicted_active),
            &array_to_f32(&ff_residual),
            &array_to_f32(&layer0.altup.router_norm.weight),
            &array_to_f32(&layer0.altup.modality_router.dequantized_weight()),
            &array_to_f32(&layer0.altup.correction_coefs.weight),
            &array_to_f32(&layer0.altup.correct_output_scale),
            rows,
            hidden,
            plane_count,
            layer0.altup_active_idx,
            layer0.altup.router_norm.eps,
            language_model.config.altup_coef_clip,
        )
        .unwrap();
        let mut altup_correct_cursor = 0usize;
        for (name, expected) in [
            (
                "layer0_correction_router_norm",
                array_to_f32(&correction_router_norm),
            ),
            (
                "layer0_correction_router_scaled",
                array_to_f32(&correction_router_scaled),
            ),
            (
                "layer0_correction_modalities",
                array_to_f32(&correction_modalities),
            ),
            (
                "layer0_correction_coefficients",
                array_to_f32(&correction_coefficients),
            ),
            (
                "layer0_correction_innovation",
                array_to_f32(&correction_innovation),
            ),
            (
                "layer0_correction_product",
                array_to_f32(&correction_products),
            ),
            ("layer0_corrected_all", array_to_f32(&corrected_stacked)),
            ("layer0_corrected_active", array_to_f32(&corrected_active)),
            (
                "layer0_corrected_scaled_active",
                array_to_f32(&scaled_corrected_active),
            ),
        ] {
            let len = expected.len();
            compare_narrow_stage(
                name,
                &expected,
                &xla_altup_correct[altup_correct_cursor..altup_correct_cursor + len],
                &mut failures,
            );
            altup_correct_cursor += len;
        }
        assert_eq!(altup_correct_cursor, xla_altup_correct.len());

        let xla_ple_injection = run_gemma3n_ple_injection_diagnostic_probe(
            &array_to_f32(&traced.corrected_active),
            &array_to_f32(&layer0.altup.correct_output_scale),
            &array_to_f32(&per_layer_input),
            &array_to_f32(&layer0.per_layer_input_gate.dequantized_weight()),
            &array_to_f32(&layer0.per_layer_projection.dequantized_weight()),
            &array_to_f32(&layer0.post_per_layer_input_norm.weight),
            &array_to_f32(&traced.ple_residual),
            rows,
            hidden,
            ple_width,
            layer0.post_per_layer_input_norm.eps,
        )
        .unwrap();
        let mut ple_cursor = 0usize;
        for (name, expected) in [
            (
                "layer0_ple_scaled_active",
                array_to_f32(&traced.ple_scaled_active),
            ),
            ("layer0_ple_gate", array_to_f32(&traced.ple_gate)),
            ("layer0_ple_geglu", array_to_f32(&traced.ple_activated)),
            ("layer0_ple_projection", array_to_f32(&traced.ple_projected)),
            ("layer0_ple_injected", array_to_f32(&traced.ple_injected)),
            (
                "layer0_ple_residual",
                array_to_f32(&traced.ple_residual_updated),
            ),
        ] {
            let len = expected.len();
            compare_narrow_stage(
                name,
                &expected,
                &xla_ple_injection[ple_cursor..ple_cursor + len],
                &mut failures,
            );
            ple_cursor += len;
        }
        assert_eq!(ple_cursor, xla_ple_injection.len());
        assert!(
            failures.is_empty(),
            "narrow Gemma3n stages diverged:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    #[ignore = "requires pinned Gemma3n E2B checkpoint plus MLX and IREE CUDA"]
    fn pinned_e2b_all_layer_altup_plane_trace() {
        let model_dir_env = std::env::var_os("GEMMA3N_MODEL_DIR").expect("set GEMMA3N_MODEL_DIR");
        let model_dir = Path::new(&model_dir_env);
        assert_eq!(sha256(&model_dir.join("config.json")), CONFIG_SHA256);
        assert_eq!(
            sha256(&model_dir.join("model.safetensors.index.json")),
            INDEX_SHA256
        );
        assert_eq!(sha256(&model_dir.join("model.safetensors")), MODEL_SHA256);

        let (
            mlx_all_layers,
            mlx_active_layers,
            layer_types,
            cache_map,
            sparsities,
            altup_correct_scale,
            altup_coef_clip,
            sliding_window,
            num_kv_shared_layers,
            plane_count,
            active_index,
            hidden,
        ) = {
            let (loaded, _) = crate::load_model(model_dir).expect("load pinned MLX checkpoint");
            let language_model = match &loaded {
                LoadedModel::Gemma3n(model) => &model.language_model,
                LoadedModel::Gemma3nVLM(model) => &model.text_model.language_model,
                _ => panic!("pinned checkpoint did not load as Gemma3n"),
            };
            let input = mlxcel_core::from_slice_i32(&PROMPT, &[1, PROMPT.len() as i32]);
            let mut caches = language_model.make_caches();
            let traced = language_model.forward_diagnostics(&input, &mut caches);
            let all_layers = traced
                .all_layer_all_planes
                .iter()
                .map(|value| {
                    assert_eq!(
                        mlxcel_core::array_dtype(value),
                        mlxcel_core::dtype::BFLOAT16
                    );
                    array_to_f32(value)
                })
                .collect::<Vec<_>>();
            let active_layers = traced
                .all_layer_active_planes
                .iter()
                .map(|value| {
                    assert_eq!(
                        mlxcel_core::array_dtype(value),
                        mlxcel_core::dtype::BFLOAT16
                    );
                    array_to_f32(value)
                })
                .collect::<Vec<_>>();
            (
                all_layers,
                active_layers,
                language_model.config.layer_types.clone(),
                language_model.layer_idx_to_cache_idx.clone(),
                (0..language_model.layers.len())
                    .map(|layer| language_model.config.get_activation_sparsity(layer))
                    .collect::<Vec<_>>(),
                language_model.config.altup_correct_scale,
                language_model.config.altup_coef_clip,
                language_model.config.sliding_window,
                language_model.config.num_kv_shared_layers,
                language_model.config.altup_num_inputs,
                language_model.config.altup_active_idx,
                language_model.config.hidden_size,
            )
        };
        mlxcel_core::memory::clear_cache();

        let device = std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "cuda".to_string());
        let xla = run_gemma3n_all_layer_diagnostics(model_dir, &device, CAPACITY, &PROMPT).unwrap();
        let all_segment = xla.layout.segment("all_layer_all_planes").unwrap();
        let active_segment = xla.layout.segment("all_layer_active_planes").unwrap();
        let rows = PROMPT.len();
        let trace_layer_start = std::env::var("MLXCEL_XLA_GEMMA3N_TRACE_LAYER_START")
            .ok()
            .map(|value| value.parse::<usize>().expect("valid trace layer start"))
            .unwrap_or(0);
        let layer_count = all_segment.shape[0];
        let total_layers = layer_types.len();
        assert_eq!(
            layer_count,
            (total_layers - trace_layer_start).min(10),
            "bounded trace window length"
        );
        assert_eq!(
            all_segment.shape,
            vec![layer_count, plane_count, rows, hidden]
        );
        assert_eq!(active_segment.shape, vec![layer_count, rows, hidden]);
        assert!(mlx_all_layers.len() >= trace_layer_start + layer_count);
        assert!(mlx_active_layers.len() >= trace_layer_start + layer_count);
        assert!(layer_types.len() >= trace_layer_start + layer_count);
        assert!(cache_map.len() >= trace_layer_start + layer_count);

        let summarize = |mlx: &[f32], xla: &[f32]| {
            assert_eq!(mlx.len(), xla.len());
            let mut mismatches = 0usize;
            let mut max_abs = 0.0f32;
            let mut first = None;
            for (index, (&mlx, &xla)) in mlx.iter().zip(xla).enumerate() {
                let delta = (mlx - xla).abs();
                max_abs = max_abs.max(delta);
                if mlx.to_bits() != xla.to_bits() {
                    mismatches += 1;
                    first.get_or_insert((index, mlx, xla, delta));
                }
            }
            (mismatches, max_abs, first)
        };

        let plane_len = rows * hidden;
        let layer_all_len = plane_count * plane_len;
        let mut earliest = None;
        for window_layer in 0..layer_count {
            let layer = trace_layer_start + window_layer;
            let layer_all_start = all_segment.offset + window_layer * layer_all_len;
            let xla_all = &xla.intermediates[layer_all_start..layer_all_start + layer_all_len];
            let active_start = active_segment.offset + window_layer * plane_len;
            let xla_active = &xla.intermediates[active_start..active_start + plane_len];
            let mlx_all = &mlx_all_layers[layer];
            let mlx_active = &mlx_active_layers[layer];
            assert_eq!(mlx_all.len(), plane_count * rows * hidden);
            assert_eq!(mlx_active.len(), rows * hidden);

            let active_plane_start = active_index * plane_len;
            let mlx_all_active = &mlx_all[active_plane_start..active_plane_start + plane_len];
            let xla_all_active = &xla_all[active_plane_start..active_plane_start + plane_len];
            assert_eq!(
                mlx_active
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                mlx_all_active
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                xla_active
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                xla_all_active
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );

            let (all_mismatches, all_max_abs, all_first) = summarize(mlx_all, xla_all);
            let (active_mismatches, active_max_abs, active_first) =
                summarize(mlx_active, xla_active);
            let shared_kv = layer >= total_layers - num_kv_shared_layers;
            eprintln!(
                "all_layer_trace layer={layer} type={} cache_map={layer}->{} shared_kv={shared_kv} \
                 sparsity={} all_bit_mismatches={all_mismatches}/{} all_max_abs={all_max_abs} \
                 all_first={all_first:?} active_bit_mismatches={active_mismatches}/{} \
                 active_max_abs={active_max_abs} active_first={active_first:?} \
                 altup_correct_scale={altup_correct_scale} altup_coef_clip={altup_coef_clip:?} \
                 sliding_window={sliding_window}",
                layer_types[layer],
                cache_map[layer],
                sparsities[layer],
                mlx_all.len(),
                mlx_active.len(),
            );
            if earliest.is_none() && (all_mismatches != 0 || active_mismatches != 0) {
                earliest = Some(layer);
            }
        }
        eprintln!(
            "all_layer_trace_summary earliest_divergent_layer={earliest:?} \
             window={trace_layer_start}..{} layers={layer_count} \
             planes={plane_count} rows={rows} hidden={hidden} \
             num_kv_shared_layers={num_kv_shared_layers}",
            trace_layer_start + layer_count,
        );
    }

    #[test]
    #[ignore = "requires pinned Gemma3n E2B checkpoint plus MLX and IREE CUDA"]
    fn pinned_e2b_layer10_dense_mlp_actual_input_bisect() {
        const TARGET_LAYER: usize = 10;

        let model_dir_env = std::env::var_os("GEMMA3N_MODEL_DIR").expect("set GEMMA3N_MODEL_DIR");
        let model_dir = Path::new(&model_dir_env);
        assert_eq!(sha256(&model_dir.join("config.json")), CONFIG_SHA256);
        assert_eq!(
            sha256(&model_dir.join("model.safetensors.index.json")),
            INDEX_SHA256
        );
        assert_eq!(sha256(&model_dir.join("model.safetensors")), MODEL_SHA256);

        let (loaded, _) = crate::load_model(model_dir).expect("load pinned MLX checkpoint");
        let language_model = match &loaded {
            LoadedModel::Gemma3n(model) => &model.language_model,
            LoadedModel::Gemma3nVLM(model) => &model.text_model.language_model,
            _ => panic!("pinned checkpoint did not load as Gemma3n"),
        };
        let rows = PROMPT.len();
        let hidden = language_model.config.hidden_size;
        let ple_width = language_model.config.hidden_size_per_layer_input;
        let layer = &language_model.layers[TARGET_LAYER];
        assert_eq!(layer.mlp.activation_sparsity, 0.0);
        assert_eq!(
            layer.pre_feedforward_layernorm.eps,
            layer.post_feedforward_layernorm.eps
        );

        let input = mlxcel_core::from_slice_i32(&PROMPT, &[1, rows as i32]);
        let scaled_embeddings = language_model.get_embed_tokens(&input);
        let token_ple = language_model.get_per_layer_inputs(&input);
        let projected_ple = language_model.project_per_layer_inputs(&scaled_embeddings, &token_ple);
        let target_magnitude = compute_magnitude(&scaled_embeddings);
        let mut planes = vec![mlxcel_core::copy(&scaled_embeddings)];
        for projection in &language_model.altup_projections {
            planes.push(projection.forward(&scaled_embeddings));
        }
        normalize_magnitudes(&mut planes, &target_magnitude);

        let mut caches = language_model.make_caches();
        let global_live_len = caches[language_model.first_full_idx].live_len();
        let sliding_live_len = caches[language_model.first_sliding_idx].live_len();
        let global_mask = create_causal_mask(rows as i32, global_live_len);
        let sliding_mask = create_sliding_window_prefill_mask_dense(
            rows as i32,
            sliding_live_len,
            language_model.config.sliding_window as i32,
        );
        let mut traced_target = None;
        for layer_index in 0..=TARGET_LAYER {
            let cache_index = language_model.layer_idx_to_cache_idx[layer_index];
            let mask = if language_model.config.layer_types[layer_index] == "full_attention" {
                global_mask.as_ref().unwrap()
            } else {
                sliding_mask.as_ref().unwrap()
            };
            let per_layer_input = slice_layer_input(
                &projected_ple,
                layer_index as i32,
                1,
                rows as i32,
                ple_width as i32,
            );
            let traced = language_model.layers[layer_index].forward_diagnostics(
                &planes,
                Some(mask),
                &mut caches[cache_index],
                &per_layer_input,
            );
            if layer_index == TARGET_LAYER {
                traced_target = Some(traced);
                break;
            }
            planes = traced.planes;
        }
        let traced = traced_target.expect("capture layer10 MLX trace");
        let attention_laurel = traced.laurel_output;
        let pre_ff_norm = layer.pre_feedforward_layernorm.forward(&attention_laurel);
        let gate = layer.mlp.gate_proj.forward(&pre_ff_norm);
        let up = layer.mlp.up_proj.forward(&pre_ff_norm);
        let product = mlxcel_core::compiled_geglu_approx_activation(&gate, &up);
        let down = layer.mlp.down_proj.forward(&product);
        let down = mlxcel_core::astype(&down, mlxcel_core::dtype::BFLOAT16);
        let post_ff_norm = layer.post_feedforward_layernorm.forward(&down);
        let ff_residual = mlxcel_core::astype(
            &mlxcel_core::add(&attention_laurel, &post_ff_norm),
            mlxcel_core::dtype::BFLOAT16,
        );

        let intermediate = mlxcel_core::array_shape(&gate).last().copied().unwrap() as usize;
        let gate_weight = mlp_input_projection_weight(&layer.mlp.gate_proj);
        let up_weight = mlp_input_projection_weight(&layer.mlp.up_proj);
        let xla = run_gemma3n_dense_mlp_diagnostic_probe(
            &array_to_f32(&attention_laurel),
            &array_to_f32(&layer.pre_feedforward_layernorm.weight),
            &array_to_f32(&gate_weight),
            &array_to_f32(&up_weight),
            &array_to_f32(&layer.mlp.down_proj.dequantized_weight()),
            &array_to_f32(&layer.post_feedforward_layernorm.weight),
            rows,
            hidden,
            intermediate,
            layer.pre_feedforward_layernorm.eps,
        )
        .unwrap();

        let rh = rows * hidden;
        let ri = rows * intermediate;
        let mut cursor = 0usize;
        macro_rules! stage {
            ($len:expr) => {{
                let start = cursor;
                cursor += $len;
                &xla[start..cursor]
            }};
        }
        let xla_pre_ff_norm = stage!(rh);
        let xla_gate = stage!(ri);
        let xla_up = stage!(ri);
        let xla_product = stage!(ri);
        let xla_down = stage!(rh);
        let xla_post = stage!(rh);
        let xla_residual = stage!(rh);
        assert_eq!(cursor, xla.len());

        let mut failures = Vec::new();
        compare_narrow_stage(
            "layer10_dense_pre_ff_norm",
            &array_to_f32(&pre_ff_norm),
            xla_pre_ff_norm,
            &mut failures,
        );
        compare_narrow_stage(
            "layer10_dense_gate",
            &array_to_f32(&gate),
            xla_gate,
            &mut failures,
        );
        compare_narrow_stage(
            "layer10_dense_up",
            &array_to_f32(&up),
            xla_up,
            &mut failures,
        );
        compare_narrow_stage(
            "layer10_dense_product",
            &array_to_f32(&product),
            xla_product,
            &mut failures,
        );
        compare_narrow_stage(
            "layer10_dense_down",
            &array_to_f32(&down),
            xla_down,
            &mut failures,
        );
        compare_narrow_stage(
            "layer10_dense_post_norm",
            &array_to_f32(&post_ff_norm),
            xla_post,
            &mut failures,
        );
        compare_narrow_stage(
            "layer10_dense_residual",
            &array_to_f32(&ff_residual),
            xla_residual,
            &mut failures,
        );
        assert!(
            failures.is_empty(),
            "production layer10 dense MLP path diverged:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    #[ignore = "requires pinned Gemma3n E2B checkpoint plus MLX and IREE CUDA"]
    fn pinned_e2b_layer3_ple_actual_input_bisect() {
        const TARGET_LAYER: usize = 3;

        let model_dir_env = std::env::var_os("GEMMA3N_MODEL_DIR").expect("set GEMMA3N_MODEL_DIR");
        let model_dir = Path::new(&model_dir_env);
        assert_eq!(sha256(&model_dir.join("config.json")), CONFIG_SHA256);
        assert_eq!(
            sha256(&model_dir.join("model.safetensors.index.json")),
            INDEX_SHA256
        );
        assert_eq!(sha256(&model_dir.join("model.safetensors")), MODEL_SHA256);

        let (loaded, _) = crate::load_model(model_dir).expect("load pinned MLX checkpoint");
        let language_model = match &loaded {
            LoadedModel::Gemma3n(model) => &model.language_model,
            LoadedModel::Gemma3nVLM(model) => &model.text_model.language_model,
            _ => panic!("pinned checkpoint did not load as Gemma3n"),
        };
        let rows = PROMPT.len();
        let hidden = language_model.config.hidden_size;
        let plane_count = language_model.config.altup_num_inputs;
        let non_active_planes = plane_count - 1;
        let ple_width = language_model.config.hidden_size_per_layer_input;
        assert_eq!(
            language_model.config.altup_active_idx, 0,
            "layer3 residual mapping assumes active AltUp plane zero"
        );
        assert_eq!(
            language_model.config.layer_types[TARGET_LAYER],
            "sliding_attention"
        );
        assert_eq!(
            language_model.layer_idx_to_cache_idx[TARGET_LAYER],
            TARGET_LAYER
        );

        let input = mlxcel_core::from_slice_i32(&PROMPT, &[1, rows as i32]);
        let scaled_embeddings = language_model.get_embed_tokens(&input);
        let token_ple = language_model.get_per_layer_inputs(&input);
        let projected_ple = language_model.project_per_layer_inputs(&scaled_embeddings, &token_ple);
        let target_magnitude = compute_magnitude(&scaled_embeddings);
        let mut planes = vec![mlxcel_core::copy(&scaled_embeddings)];
        for projection in &language_model.altup_projections {
            planes.push(projection.forward(&scaled_embeddings));
        }
        normalize_magnitudes(&mut planes, &target_magnitude);
        let layer0_input_planes = stack_arrays(&planes, 0);

        let mut caches = language_model.make_caches();
        let global_live_len = caches[language_model.first_full_idx].live_len();
        let sliding_live_len = caches[language_model.first_sliding_idx].live_len();
        let global_mask = create_causal_mask(rows as i32, global_live_len);
        let sliding_mask = create_sliding_window_prefill_mask_dense(
            rows as i32,
            sliding_live_len,
            language_model.config.sliding_window as i32,
        );
        let mut target_ple = None;
        let mut target_input_planes = None;
        let mut traced_target = None;
        for layer_index in 0..=TARGET_LAYER {
            let layer = &language_model.layers[layer_index];
            let cache_index = language_model.layer_idx_to_cache_idx[layer_index];
            let mask = if language_model.config.layer_types[layer_index] == "full_attention" {
                global_mask.as_ref().unwrap()
            } else {
                sliding_mask.as_ref().unwrap()
            };
            let per_layer_input = slice_layer_input(
                &projected_ple,
                layer_index as i32,
                1,
                rows as i32,
                ple_width as i32,
            );
            if layer_index == TARGET_LAYER {
                target_input_planes = Some(stack_arrays(&planes, 0));
            }
            let traced = layer.forward_diagnostics(
                &planes,
                Some(mask),
                &mut caches[cache_index],
                &per_layer_input,
            );
            if layer_index == TARGET_LAYER {
                target_ple = Some(per_layer_input);
                traced_target = Some(traced);
                break;
            }
            planes = traced.planes;
        }
        let traced = traced_target.expect("capture layer3 MLX trace");
        let per_layer_input = target_ple.expect("capture layer3 dense PLE slice");
        let input_planes = target_input_planes.expect("capture exact layer2 output planes");
        let layer = &language_model.layers[TARGET_LAYER];

        let router_weight = assert_real_projection_weight_carrier(
            "layer3_altup_modality_router",
            &layer.altup.modality_router,
        );
        assert_eq!(
            mlxcel_core::array_dtype(&layer.altup.prediction_coefs.weight),
            mlxcel_core::dtype::FLOAT32
        );
        assert_eq!(
            mlxcel_core::array_dtype(&layer.altup.correction_coefs.weight),
            mlxcel_core::dtype::FLOAT32
        );
        assert!(layer.altup.prediction_coefs.bias.is_none());
        assert!(layer.altup.correction_coefs.bias.is_none());
        let prediction_weight = assert_checkpoint_bf16_weight_carrier(
            model_dir,
            "model.language_model.layers.3.altup.prediction_coefs.weight",
            &layer.altup.prediction_coefs.weight,
            language_model.config.altup_coef_clip,
        );
        let correction_weight = assert_checkpoint_bf16_weight_carrier(
            model_dir,
            "model.language_model.layers.3.altup.correction_coefs.weight",
            &layer.altup.correction_coefs.weight,
            language_model.config.altup_coef_clip,
        );
        eprintln!(
            "layer3_altup_f32_weight_carriers prediction_values={} correction_values={} \
             prediction_bf16_aligned={} correction_bf16_aligned={}",
            prediction_weight.len(),
            correction_weight.len(),
            prediction_weight
                .iter()
                .filter(|value| value.to_bits() & 0xffff == 0)
                .count(),
            correction_weight
                .iter()
                .filter(|value| value.to_bits() & 0xffff == 0)
                .count(),
        );

        let active_input = slice_altup_plane(&input_planes, layer.altup_active_idx);
        let predict_router_norm = layer.altup.router_norm.forward(&active_input);
        let predict_router_scaled =
            mlxcel_core::multiply_scalar(&predict_router_norm, (hidden as f32).powf(-1.0));
        let predict_modalities = layer.altup.modality_router.forward(&predict_router_scaled);
        let predict_modalities =
            mlxcel_core::astype(&predict_modalities, mlxcel_core::dtype::FLOAT32);
        let predict_modalities = mlxcel_core::tanh(&predict_modalities);
        let predict_coefficients = layer.altup.prediction_coefs.forward(&predict_modalities);
        let predict_coefficients = mlxcel_core::reshape(
            &predict_coefficients,
            &[1, rows as i32, plane_count as i32, plane_count as i32],
        );
        let predict_coefficients =
            mlxcel_core::transpose_axes(&predict_coefficients, &[0, 1, 3, 2]);
        let predicted_all = array_to_f32(&traced.altup_predicted);
        let predicted_active = array_to_f32(&traced.altup_predicted_active);
        let layer0_schedule = analyze_altup_predict_schedule(
            "layer0",
            &language_model.layers[0],
            &layer0_input_planes,
            rows,
            hidden,
            plane_count,
        );
        let layer0_split = split_altup_planes(&layer0_input_planes, plane_count);
        let layer0_predicted = language_model.layers[0]
            .altup
            .predict_stacked(&layer0_split);
        assert_eq!(
            layer0_schedule
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            array_to_f32(&layer0_predicted)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "layer0 standalone MLX schedule reconstruction drifted"
        );
        let layer3_schedule = analyze_altup_predict_schedule(
            "layer3",
            layer,
            &input_planes,
            rows,
            hidden,
            plane_count,
        );
        assert_eq!(
            layer3_schedule
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            predicted_all
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "layer3 standalone MLX schedule reconstruction drifted"
        );
        let predict_input_norm = layer
            .input_layernorm
            .forward(&traced.altup_predicted_active);
        let xla_predict = run_gemma3n_altup_predict_diagnostic_probe(
            &array_to_f32(&input_planes),
            &array_to_f32(&layer.altup.router_norm.weight),
            &router_weight,
            &prediction_weight,
            &array_to_f32(&layer.input_layernorm.weight),
            plane_count,
            layer.altup_active_idx,
            rows,
            hidden,
            layer.input_layernorm.eps,
            language_model.config.altup_coef_clip,
        )
        .unwrap();
        let mut failures = Vec::new();
        let mut predict_cursor = 0usize;
        for (name, expected) in [
            (
                "layer3_predict_router_norm",
                array_to_f32(&predict_router_norm),
            ),
            (
                "layer3_predict_router_scaled",
                array_to_f32(&predict_router_scaled),
            ),
            (
                "layer3_predict_modalities_native_qmv_tanh",
                array_to_f32(&predict_modalities),
            ),
            (
                "layer3_predict_coefficients_tf32",
                array_to_f32(&predict_coefficients),
            ),
            ("layer3_predicted_all_tf32", predicted_all.clone()),
            ("layer3_predicted_active", predicted_active.clone()),
            (
                "layer3_predict_input_norm",
                array_to_f32(&predict_input_norm),
            ),
        ] {
            let len = expected.len();
            compare_narrow_stage(
                name,
                &expected,
                &xla_predict[predict_cursor..predict_cursor + len],
                &mut failures,
            );
            predict_cursor += len;
        }
        assert_eq!(predict_cursor, xla_predict.len());

        let activated = array_to_f32(&traced.altup_activated);
        let correction_router_norm = layer.altup.router_norm.forward(&traced.altup_activated);
        let correction_router_scaled =
            mlxcel_core::multiply_scalar(&correction_router_norm, (hidden as f32).powf(-1.0));
        let correction_modalities = layer
            .altup
            .modality_router
            .forward(&correction_router_scaled);
        let correction_modalities =
            mlxcel_core::astype(&correction_modalities, mlxcel_core::dtype::FLOAT32);
        let correction_modalities = mlxcel_core::tanh(&correction_modalities);
        let correction_coefficients = layer.altup.correction_coefs.forward(&correction_modalities);
        let correction_one = mlxcel_core::full_f32(
            &[1],
            1.0,
            mlxcel_core::array_dtype(&correction_coefficients),
        );
        let correction_coefficients = mlxcel_core::add(&correction_coefficients, &correction_one);
        let correction_innovation =
            mlxcel_core::subtract(&traced.altup_activated, &traced.altup_predicted_active);
        let correction_coefficients_planes =
            mlxcel_core::transpose_axes(&correction_coefficients, &[2, 0, 1]);
        let correction_coefficients_broadcast = mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(
                &correction_coefficients_planes,
                &[plane_count as i32, 1, rows as i32, 1],
            ),
            &[plane_count as i32, 1, rows as i32, hidden as i32],
        );
        let correction_innovation_expanded =
            mlxcel_core::reshape(&correction_innovation, &[1, 1, rows as i32, hidden as i32]);
        let correction_products = mlxcel_core::multiply(
            &correction_innovation_expanded,
            &correction_coefficients_broadcast,
        );
        let corrected_all = array_to_f32(&traced.altup_corrected);
        let corrected_active = array_to_f32(&traced.corrected_active);
        let xla_correct = run_gemma3n_altup_correct_diagnostic_probe(
            &predicted_all,
            &predicted_active,
            &activated,
            &array_to_f32(&layer.altup.router_norm.weight),
            &router_weight,
            &correction_weight,
            &array_to_f32(&layer.altup.correct_output_scale),
            rows,
            hidden,
            plane_count,
            layer.altup_active_idx,
            layer.altup.router_norm.eps,
            language_model.config.altup_coef_clip,
        )
        .unwrap();
        let mut correct_cursor = 0usize;
        for (name, expected) in [
            (
                "layer3_correction_router_norm",
                array_to_f32(&correction_router_norm),
            ),
            (
                "layer3_correction_router_scaled",
                array_to_f32(&correction_router_scaled),
            ),
            (
                "layer3_correction_modalities_native_qmv_tanh",
                array_to_f32(&correction_modalities),
            ),
            (
                "layer3_correction_coefficients_tf32",
                array_to_f32(&correction_coefficients),
            ),
            (
                "layer3_correction_innovation",
                array_to_f32(&correction_innovation),
            ),
            (
                "layer3_correction_products",
                array_to_f32(&correction_products),
            ),
            ("layer3_corrected_all", corrected_all.clone()),
            ("layer3_corrected_active", corrected_active.clone()),
            (
                "layer3_corrected_scaled_active",
                array_to_f32(&traced.ple_scaled_active),
            ),
        ] {
            let len = expected.len();
            compare_narrow_stage(
                name,
                &expected,
                &xla_correct[correct_cursor..correct_cursor + len],
                &mut failures,
            );
            correct_cursor += len;
        }
        assert_eq!(correct_cursor, xla_correct.len());

        let gate_weight =
            assert_real_projection_weight_carrier("layer3_ple_gate", &layer.per_layer_input_gate);
        let projection_weight = assert_real_projection_weight_carrier(
            "layer3_ple_projection",
            &layer.per_layer_projection,
        );
        let correct_scale = array_to_f32(&layer.altup.correct_output_scale);
        let per_layer_input_values = array_to_f32(&per_layer_input);
        let norm_weight = array_to_f32(&layer.post_per_layer_input_norm.weight);
        let residuals = array_to_f32(&traced.ple_residuals);
        let residuals_updated = array_to_f32(&traced.ple_residuals_updated);
        assert_eq!(residuals.len(), non_active_planes * rows * hidden);
        assert_eq!(residuals_updated.len(), non_active_planes * rows * hidden);
        let injected = array_to_f32(&traced.ple_injected);
        let mut raw_residual_sums = Vec::with_capacity(residuals.len());
        for plane in 0..non_active_planes {
            let plane_start = plane * rows * hidden;
            for index in 0..rows * hidden {
                raw_residual_sums.push(residuals[plane_start + index] + injected[index]);
            }
        }

        let xla = run_gemma3n_ple_injection_all_planes_diagnostic_probe(
            &corrected_active,
            &correct_scale,
            &per_layer_input_values,
            &gate_weight,
            &projection_weight,
            &norm_weight,
            &residuals,
            rows,
            hidden,
            ple_width,
            non_active_planes,
            layer.post_per_layer_input_norm.eps,
        )
        .unwrap();
        let mut cursor = 0usize;
        let scaled_active = array_to_f32(&traced.ple_scaled_active);
        let gate = array_to_f32(&traced.ple_gate);
        let activated = array_to_f32(&traced.ple_activated);
        let projected = array_to_f32(&traced.ple_projected);
        for (name, expected) in [
            ("layer3_ple_scaled_active", scaled_active.as_slice()),
            ("layer3_ple_gate_native_qmv", gate.as_slice()),
            ("layer3_ple_geglu", activated.as_slice()),
            ("layer3_ple_projection_native_qmv", projected.as_slice()),
            ("layer3_ple_injected", injected.as_slice()),
            ("layer3_ple_residuals_pre", residuals.as_slice()),
            ("layer3_ple_residuals_raw_add", raw_residual_sums.as_slice()),
            (
                "layer3_ple_residuals_post_bf16",
                residuals_updated.as_slice(),
            ),
        ] {
            let len = expected.len();
            compare_narrow_stage(name, expected, &xla[cursor..cursor + len], &mut failures);
            cursor += len;
        }
        assert_eq!(cursor, xla.len());

        // The all-layer trace's flat index 8664 maps to logical plane 1,
        // prompt row 1, hidden column 472. In the non-active residual stack,
        // plane 1 is residual plane 0.
        const MISMATCH_ROW: usize = 1;
        const MISMATCH_HIDDEN: usize = 472;
        let mismatch_index = MISMATCH_ROW * hidden + MISMATCH_HIDDEN;
        let plane_len = rows * hidden;
        let plane1_index = plane_len + mismatch_index;
        let predict_all_offset =
            plane_len * 2 + rows * plane_count + rows * plane_count * plane_count;
        let correction_products_offset = plane_len * 3 + rows * plane_count * 2;
        let corrected_all_offset = correction_products_offset + plane_count * plane_len;
        eprintln!(
            "layer3_altup_mismatch_carrier plane=1 row={MISMATCH_ROW} hidden={MISMATCH_HIDDEN} \
             input_mlx={} predicted_mlx={} predicted_xla={} correction_product_mlx={} \
             correction_product_xla={} corrected_mlx={} corrected_xla={}",
            array_to_f32(&input_planes)[plane1_index],
            predicted_all[plane1_index],
            xla_predict[predict_all_offset + plane1_index],
            array_to_f32(&correction_products)[plane1_index],
            xla_correct[correction_products_offset + plane1_index],
            corrected_all[plane1_index],
            xla_correct[corrected_all_offset + plane1_index],
        );
        let stage_prefix = rows * (hidden * 3 + ple_width * 2);
        let residual_stage_len = non_active_planes * rows * hidden;
        eprintln!(
            "layer3_ple_mismatch_carrier plane=1 row={MISMATCH_ROW} hidden={MISMATCH_HIDDEN} \
             residual_mlx={} injected_mlx={} raw_sum_mlx={} post_bf16_mlx={} \
             residual_xla={} raw_sum_xla={} post_bf16_xla={}",
            residuals[mismatch_index],
            injected[mismatch_index],
            raw_residual_sums[mismatch_index],
            residuals_updated[mismatch_index],
            xla[stage_prefix + mismatch_index],
            xla[stage_prefix + residual_stage_len + mismatch_index],
            xla[stage_prefix + residual_stage_len * 2 + mismatch_index],
        );
        assert!(
            failures.is_empty(),
            "layer3 actual-input PLE stages diverged:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    #[ignore = "requires pinned Gemma3n E2B checkpoint plus MLX and IREE CUDA"]
    fn pinned_e2b_layer0_prefix_decode_attention_bisect() {
        let model_dir_env = std::env::var_os("GEMMA3N_MODEL_DIR").expect("set GEMMA3N_MODEL_DIR");
        let model_dir = Path::new(&model_dir_env);
        assert_eq!(sha256(&model_dir.join("config.json")), CONFIG_SHA256);
        assert_eq!(
            sha256(&model_dir.join("model.safetensors.index.json")),
            INDEX_SHA256
        );
        assert_eq!(sha256(&model_dir.join("model.safetensors")), MODEL_SHA256);

        let (loaded, _) = crate::load_model(model_dir).expect("load pinned MLX checkpoint");
        let language_model = match &loaded {
            LoadedModel::Gemma3n(model) => &model.language_model,
            LoadedModel::Gemma3nVLM(model) => &model.text_model.language_model,
            _ => panic!("pinned checkpoint did not load as Gemma3n"),
        };
        let layer0 = &language_model.layers[0];
        let cache_index = language_model.layer_idx_to_cache_idx[0];
        assert_eq!(cache_index, 0);
        let hidden = language_model.config.hidden_size;
        let query_heads = layer0.self_attn.num_heads as usize;
        let kv_heads = layer0.self_attn.num_kv_heads as usize;
        let head_dim = layer0.self_attn.head_dim as usize;
        let kv_width = kv_heads * head_dim;
        assert_eq!(hidden, query_heads * head_dim);

        let mut caches = language_model.make_caches();
        let prefix =
            mlxcel_core::from_slice_i32(&PROMPT[..PROMPT.len() - 1], &[1, PROMPT.len() as i32 - 1]);
        let seeded = language_model.forward(&prefix, &mut caches);
        mlxcel_core::eval(&seeded);
        let position = caches[cache_index].offset as usize;
        assert_eq!(position, PROMPT.len() - 1);
        let prefix_keys =
            mlx_layer0_cache_prefix(caches[cache_index].keys.as_ref().unwrap(), position);
        let prefix_values =
            mlx_layer0_cache_prefix(caches[cache_index].values.as_ref().unwrap(), position);

        let last = mlxcel_core::from_slice_i32(&PROMPT[PROMPT.len() - 1..], &[1, 1]);
        let scaled_embeddings = language_model.get_embed_tokens(&last);
        let token_ple = language_model.get_per_layer_inputs(&last);
        let projected_ple = language_model.project_per_layer_inputs(&scaled_embeddings, &token_ple);
        let target_magnitude = compute_magnitude(&scaled_embeddings);
        let mut planes = vec![mlxcel_core::copy(&scaled_embeddings)];
        for projection in &language_model.altup_projections {
            planes.push(projection.forward(&scaled_embeddings));
        }
        normalize_magnitudes(&mut planes, &target_magnitude);
        let predicted_stacked = layer0.altup.predict_stacked(&planes);
        let predicted_active = slice_altup_plane(&predicted_stacked, layer0.altup_active_idx);
        let normalized = layer0.input_layernorm.forward(&predicted_active);
        let laurel = layer0.laurel.forward(&normalized);
        let per_layer_input = slice_layer_input(
            &projected_ple,
            0,
            1,
            1,
            language_model.config.hidden_size_per_layer_input as i32,
        );
        assert_eq!(
            mlxcel_core::array_shape(&per_layer_input)[1],
            1,
            "decode layer0 PLE must carry one row"
        );

        let q_projection = layer0.self_attn.q_proj.forward(&normalized);
        let q = mlxcel_core::reshape(
            &q_projection,
            &[1, 1, layer0.self_attn.num_heads, layer0.self_attn.head_dim],
        );
        let q_norm = layer0.self_attn.q_norm.forward(&q);
        let q = mlxcel_core::transpose_axes(&q_norm, &[0, 2, 1, 3]);
        let q_rope = mlxcel_core::fast_rope(
            &q,
            layer0.self_attn.head_dim,
            false,
            layer0.self_attn.rope_theta,
            1.0,
            position as i32,
        );

        let k_projection = layer0.self_attn.k_proj.forward(&normalized);
        let k = mlxcel_core::reshape(
            &k_projection,
            &[
                1,
                1,
                layer0.self_attn.num_kv_heads,
                layer0.self_attn.head_dim,
            ],
        );
        let k_norm = layer0.self_attn.k_norm.forward(&k);
        let k = mlxcel_core::transpose_axes(&k_norm, &[0, 2, 1, 3]);
        let k_rope = mlxcel_core::fast_rope(
            &k,
            layer0.self_attn.head_dim,
            false,
            layer0.self_attn.rope_theta,
            1.0,
            position as i32,
        );
        let v_projection = layer0.self_attn.v_proj.forward(&normalized);
        let v = mlxcel_core::reshape(
            &v_projection,
            &[
                1,
                1,
                layer0.self_attn.num_kv_heads,
                layer0.self_attn.head_dim,
            ],
        );
        let v_norm = layer0.self_attn.v_norm.forward(&v);
        let v_norm = mlxcel_core::transpose_axes(&v_norm, &[0, 2, 1, 3]);
        let (keys, values) = caches[cache_index].update_and_fetch(k_rope, v_norm);
        let valid_len = position + 1;
        assert_eq!(
            mlxcel_core::array_shape(&keys),
            vec![1, kv_heads as i32, valid_len as i32, head_dim as i32]
        );
        let keys_rows = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
        let values_rows = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        let native_context = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &q_rope,
                &keys,
                &values,
                layer0.self_attn.scale,
                std::ptr::null(),
                0.0,
                layer0.self_attn.window_size,
            )
        };
        let native_context = mlxcel_core::transpose_axes(&native_context, &[0, 2, 1, 3]);
        let native_context = mlxcel_core::reshape(&native_context, &[1, 1, hidden as i32]);

        let production = run_gemma3n_decode_attention_diagnostic_probe(
            &array_to_f32(&normalized),
            &array_to_f32(&layer0.self_attn.q_proj.dequantized_weight()),
            &array_to_f32(&layer0.self_attn.k_proj.dequantized_weight()),
            &array_to_f32(&layer0.self_attn.v_proj.dequantized_weight()),
            &array_to_f32(&layer0.self_attn.q_norm.weight),
            &array_to_f32(&layer0.self_attn.k_norm.weight),
            &prefix_keys,
            &prefix_values,
            &array_to_f32(&layer0.self_attn.o_proj.dequantized_weight()),
            &array_to_f32(&layer0.post_attention_layernorm.weight),
            &array_to_f32(&predicted_active),
            &array_to_f32(&laurel),
            position,
            CAPACITY,
            query_heads,
            kv_heads,
            head_dim,
            layer0.self_attn.rope_theta as f64,
            layer0.post_attention_layernorm.eps,
            Some(language_model.config.sliding_window),
        )
        .unwrap();

        let lengths = [
            hidden,
            hidden,
            hidden,
            kv_width,
            kv_width,
            kv_width,
            kv_width,
            kv_width,
            valid_len * kv_width,
            valid_len * kv_width,
            hidden,
            hidden,
            hidden,
            hidden,
            hidden,
        ];
        let parse = |values: &[f32]| {
            let mut cursor = 0;
            let stages = lengths
                .iter()
                .map(|&len| take_diagnostic_stage(values, &mut cursor, len).to_vec())
                .collect::<Vec<_>>();
            assert_eq!(cursor, values.len());
            stages
        };
        let production = parse(&production);

        let mut carrier_failures = Vec::new();
        for (name, expected, index) in [
            ("decode_q_projection", array_to_f32(&q_projection), 0),
            ("decode_q_norm", array_to_f32(&q_norm), 1),
            ("decode_q_rope", array_to_f32(&q_rope), 2),
            ("decode_k_projection", array_to_f32(&k_projection), 3),
            ("decode_k_norm", array_to_f32(&k_norm), 4),
            (
                "decode_k_rope",
                array_to_f32(&mlxcel_core::slice(
                    &keys,
                    &[0, 0, position as i32, 0],
                    &[1, kv_heads as i32, valid_len as i32, head_dim as i32],
                )),
                5,
            ),
            ("decode_v_projection", array_to_f32(&v_projection), 6),
            (
                "decode_v_norm",
                array_to_f32(&mlxcel_core::slice(
                    &values,
                    &[0, 0, position as i32, 0],
                    &[1, kv_heads as i32, valid_len as i32, head_dim as i32],
                )),
                7,
            ),
            ("decode_cache_k_valid", array_to_f32(&keys_rows), 8),
            ("decode_cache_v_valid", array_to_f32(&values_rows), 9),
        ] {
            compare_narrow_stage(name, &expected, &production[index], &mut carrier_failures);
        }

        let native_projected = layer0.self_attn.o_proj.forward(&native_context);
        let native_post_norm = layer0.post_attention_layernorm.forward(&native_projected);
        let native_residual = mlxcel_core::astype(
            &mlxcel_core::add(&predicted_active, &native_post_norm),
            mlxcel_core::dtype::BFLOAT16,
        );
        let native_with_laurel = mlxcel_core::astype(
            &mlxcel_core::add(&native_residual, &laurel),
            mlxcel_core::dtype::BFLOAT16,
        );
        let native_attention_laurel = mlxcel_core::astype(
            &mlxcel_core::multiply_scalar(&native_with_laurel, std::f32::consts::FRAC_1_SQRT_2),
            mlxcel_core::dtype::BFLOAT16,
        );
        let mut production_attention_failures = Vec::new();
        for (name, expected, index) in [
            (
                "decode_production_context",
                array_to_f32(&native_context),
                10,
            ),
            (
                "decode_production_o_projection",
                array_to_f32(&native_projected),
                11,
            ),
            (
                "decode_production_post_norm",
                array_to_f32(&native_post_norm),
                12,
            ),
            (
                "decode_production_residual",
                array_to_f32(&native_residual),
                13,
            ),
            (
                "decode_production_attention_laurel",
                array_to_f32(&native_attention_laurel),
                14,
            ),
        ] {
            compare_narrow_stage(
                name,
                &expected,
                &production[index],
                &mut production_attention_failures,
            );
        }

        let pre_ff_norm = layer0
            .pre_feedforward_layernorm
            .forward(&native_attention_laurel);
        let gate = layer0.mlp.gate_proj.forward(&pre_ff_norm);
        let up = layer0.mlp.up_proj.forward(&pre_ff_norm);
        assert!(layer0.mlp.activation_sparsity > 0.0);
        let sparse_mean = mlxcel_core::mean_axis(&gate, -1, true);
        let sparse_centered = mlxcel_core::subtract(&gate, &sparse_mean);
        let sparse_variance =
            mlxcel_core::mean_axis(&mlxcel_core::square(&sparse_centered), -1, true);
        let sparse_stddev = mlxcel_core::sqrt(&sparse_variance);
        let sparse_multiplier =
            mlxcel_core::full_f32(&[1], layer0.mlp.std_multiplier, mlxcel_core::dtype::FLOAT32);
        let sparse_cutoff = mlxcel_core::add(
            &sparse_mean,
            &mlxcel_core::multiply(&sparse_stddev, &sparse_multiplier),
        );
        let sparse_shifted_raw = mlxcel_core::subtract(&gate, &sparse_cutoff);
        let sparse_zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
        let sparse_shifted = mlxcel_core::maximum(&sparse_shifted_raw, &sparse_zero);
        let sparse_sqrt2 =
            mlxcel_core::full_f32(&[1], std::f32::consts::SQRT_2, mlxcel_core::dtype::FLOAT32);
        let sparse_erf = mlxcel_core::erf(&mlxcel_core::divide(&sparse_shifted, &sparse_sqrt2));
        let sparse_activation = layer0.mlp.gelu_topk(&gate);
        let sparse_product = mlxcel_core::multiply(&sparse_activation, &up);
        let mlp_down = layer0.mlp.down_proj.forward(&sparse_product);
        let post_ff_norm = layer0.post_feedforward_layernorm.forward(&mlp_down);
        let ff_residual = mlxcel_core::astype(
            &mlxcel_core::add(&native_attention_laurel, &post_ff_norm),
            mlxcel_core::dtype::BFLOAT16,
        );
        let intermediate = mlxcel_core::array_shape(&gate).last().copied().unwrap() as usize;
        let gate_weight = mlp_input_projection_weight(&layer0.mlp.gate_proj);
        let up_weight = mlp_input_projection_weight(&layer0.mlp.up_proj);
        let xla_post_attention = run_gemma3n_post_attention_diagnostic_probe(
            &array_to_f32(&native_attention_laurel),
            &array_to_f32(&layer0.pre_feedforward_layernorm.weight),
            &array_to_f32(&gate_weight),
            &array_to_f32(&up_weight),
            &array_to_f32(&layer0.mlp.down_proj.dequantized_weight()),
            &array_to_f32(&layer0.post_feedforward_layernorm.weight),
            1,
            hidden,
            intermediate,
            layer0.pre_feedforward_layernorm.eps,
            layer0.mlp.activation_sparsity,
        )
        .unwrap();
        let mut post_attention_cursor = 0;
        let mut post_attention_failures = Vec::new();
        for (name, expected) in [
            ("decode_pre_ff_norm", array_to_f32(&pre_ff_norm)),
            ("decode_mlp_gate", array_to_f32(&gate)),
            ("decode_mlp_up", array_to_f32(&up)),
            ("decode_sparse_mean", array_to_f32(&sparse_mean)),
            ("decode_sparse_variance", array_to_f32(&sparse_variance)),
            ("decode_sparse_stddev", array_to_f32(&sparse_stddev)),
            ("decode_sparse_cutoff", array_to_f32(&sparse_cutoff)),
            (
                "decode_sparse_shifted_raw",
                array_to_f32(&sparse_shifted_raw),
            ),
            ("decode_sparse_shifted", array_to_f32(&sparse_shifted)),
            ("decode_sparse_erf", array_to_f32(&sparse_erf)),
            ("decode_sparse_activation", array_to_f32(&sparse_activation)),
            ("decode_sparse_product", array_to_f32(&sparse_product)),
            ("decode_mlp_down", array_to_f32(&mlp_down)),
            ("decode_post_ff_norm", array_to_f32(&post_ff_norm)),
            ("decode_ff_residual", array_to_f32(&ff_residual)),
        ] {
            let actual = take_diagnostic_stage(
                &xla_post_attention,
                &mut post_attention_cursor,
                expected.len(),
            );
            compare_narrow_stage(name, &expected, actual, &mut post_attention_failures);
        }
        assert_eq!(post_attention_cursor, xla_post_attention.len());

        let correction_router_norm = layer0.altup.router_norm.forward(&ff_residual);
        let correction_router_scaled =
            mlxcel_core::multiply_scalar(&correction_router_norm, (hidden as f32).powf(-1.0));
        let correction_modalities = layer0
            .altup
            .modality_router
            .forward(&correction_router_scaled);
        let correction_modalities =
            mlxcel_core::astype(&correction_modalities, mlxcel_core::dtype::FLOAT32);
        let correction_modalities = mlxcel_core::tanh(&correction_modalities);
        let correction_coefficients = layer0
            .altup
            .correction_coefs
            .forward(&correction_modalities);
        let correction_one = mlxcel_core::full_f32(
            &[1],
            1.0,
            mlxcel_core::array_dtype(&correction_coefficients),
        );
        let correction_coefficients = mlxcel_core::add(&correction_coefficients, &correction_one);
        let correction_innovation = mlxcel_core::subtract(&ff_residual, &predicted_active);
        let plane_count = planes.len();
        let correction_coefficients_planes =
            mlxcel_core::transpose_axes(&correction_coefficients, &[2, 0, 1]);
        let correction_coefficients_broadcast = mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(
                &correction_coefficients_planes,
                &[plane_count as i32, 1, 1, 1],
            ),
            &[plane_count as i32, 1, 1, hidden as i32],
        );
        let correction_innovation_expanded =
            mlxcel_core::reshape(&correction_innovation, &[1, 1, 1, hidden as i32]);
        let correction_products = mlxcel_core::multiply(
            &correction_innovation_expanded,
            &correction_coefficients_broadcast,
        );
        let corrected_stacked =
            layer0
                .altup
                .correct_stacked(&predicted_stacked, &predicted_active, &ff_residual);
        let corrected_active = slice_altup_plane(&corrected_stacked, layer0.altup_active_idx);
        let scaled_corrected_active =
            mlxcel_core::multiply(&corrected_active, &layer0.altup.correct_output_scale);
        let xla_altup_correct = run_gemma3n_altup_correct_diagnostic_probe(
            &array_to_f32(&predicted_stacked),
            &array_to_f32(&predicted_active),
            &array_to_f32(&ff_residual),
            &array_to_f32(&layer0.altup.router_norm.weight),
            &array_to_f32(&layer0.altup.modality_router.dequantized_weight()),
            &array_to_f32(&layer0.altup.correction_coefs.weight),
            &array_to_f32(&layer0.altup.correct_output_scale),
            1,
            hidden,
            plane_count,
            layer0.altup_active_idx,
            layer0.altup.router_norm.eps,
            language_model.config.altup_coef_clip,
        )
        .unwrap();
        let mut altup_cursor = 0;
        let mut altup_failures = Vec::new();
        let mut coefficient_reports = Vec::new();
        for (name, expected) in [
            (
                "decode_correction_router_norm",
                array_to_f32(&correction_router_norm),
            ),
            (
                "decode_correction_router_scaled",
                array_to_f32(&correction_router_scaled),
            ),
            (
                "decode_correction_modalities",
                array_to_f32(&correction_modalities),
            ),
            (
                "decode_correction_coefficients",
                array_to_f32(&correction_coefficients),
            ),
            (
                "decode_correction_innovation",
                array_to_f32(&correction_innovation),
            ),
            (
                "decode_correction_products",
                array_to_f32(&correction_products),
            ),
            ("decode_corrected_all", array_to_f32(&corrected_stacked)),
            ("decode_corrected_active", array_to_f32(&corrected_active)),
            (
                "decode_scaled_corrected_active",
                array_to_f32(&scaled_corrected_active),
            ),
        ] {
            let actual =
                take_diagnostic_stage(&xla_altup_correct, &mut altup_cursor, expected.len());
            if name == "decode_correction_coefficients" {
                compare(
                    name,
                    &expected,
                    actual,
                    RegressionEnvelope {
                        max_abs: 2.0e-6,
                        min_cosine: 0.999_999,
                        max_normalized_rmse: 1.0e-5,
                    },
                    &mut coefficient_reports,
                );
            } else {
                compare_narrow_stage(name, &expected, actual, &mut altup_failures);
            }
        }
        assert_eq!(altup_cursor, xla_altup_correct.len());
        assert!(
            coefficient_reports.is_empty(),
            "decode AltUp F32 coefficient drift exceeded its narrow envelope:\n{}",
            coefficient_reports.join("\n")
        );

        assert!(
            carrier_failures.is_empty(),
            "decode layer0 carrier stages diverged before attention:\n{}",
            carrier_failures.join("\n")
        );
        assert!(
            production_attention_failures.is_empty(),
            "decode layer0 production attention stages diverged:\n{}",
            production_attention_failures.join("\n")
        );
        assert!(
            post_attention_failures.is_empty(),
            "decode layer0 sparse MLP stages diverged:\n{}",
            post_attention_failures.join("\n")
        );
        assert!(
            altup_failures.is_empty(),
            "decode layer0 AltUp correction stages diverged:\n{}",
            altup_failures.join("\n")
        );
    }

    #[test]
    #[ignore = "requires MLX CUDA plus pinned IREE compiler/runtime"]
    fn synthetic_d256_production_sdpa_matches_mlx_vector_boundaries() {
        const HEAD_DIM: usize = 256;
        const QUERY_HEADS: usize = 8;
        const KV_HEADS: usize = 2;
        const CAPACITY: usize = 1024;
        let hidden = QUERY_HEADS * HEAD_DIM;

        let mlx_context = |query: &[f32],
                           keys: &[f32],
                           values: &[f32],
                           live: usize,
                           window: usize| {
            let query = mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(query, &[1, QUERY_HEADS as i32, 1, HEAD_DIM as i32]),
                mlxcel_core::dtype::BFLOAT16,
            );
            let mut head_major_keys = vec![0.0f32; live * KV_HEADS * HEAD_DIM];
            let mut head_major_values = vec![0.0f32; live * KV_HEADS * HEAD_DIM];
            for token in 0..live {
                for head in 0..KV_HEADS {
                    let time_offset = (token * KV_HEADS + head) * HEAD_DIM;
                    let head_offset = (head * live + token) * HEAD_DIM;
                    head_major_keys[head_offset..head_offset + HEAD_DIM]
                        .copy_from_slice(&keys[time_offset..time_offset + HEAD_DIM]);
                    head_major_values[head_offset..head_offset + HEAD_DIM]
                        .copy_from_slice(&values[time_offset..time_offset + HEAD_DIM]);
                }
            }
            let keys = mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(
                    &head_major_keys,
                    &[1, KV_HEADS as i32, live as i32, HEAD_DIM as i32],
                ),
                mlxcel_core::dtype::BFLOAT16,
            );
            let values = mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(
                    &head_major_values,
                    &[1, KV_HEADS as i32, live as i32, HEAD_DIM as i32],
                ),
                mlxcel_core::dtype::BFLOAT16,
            );
            let output = unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &query,
                    &keys,
                    &values,
                    1.0,
                    std::ptr::null(),
                    0.0,
                    window as i32,
                )
            };
            array_to_f32(&output)
        };

        let compare_case = |name: &str,
                            query: &[f32],
                            keys: &[f32],
                            values: &[f32],
                            live: usize,
                            window: Option<usize>| {
            let expected = mlx_context(query, keys, values, live, window.unwrap_or(0));
            let actual = run_gemma3n_sdpa_vector_context_diagnostic_probe(
                query,
                keys,
                values,
                live - 1,
                CAPACITY,
                QUERY_HEADS,
                KV_HEADS,
                1.0,
                window,
            )
            .unwrap();
            assert_eq!(expected.len(), hidden);
            assert_eq!(actual.len(), hidden);
            let mut failures = Vec::new();
            compare_narrow_stage(name, &expected, &actual, &mut failures);
            assert!(
                failures.is_empty(),
                "{name} diverged from MLX vector oracle:\n{}",
                failures.join("\n")
            );
            actual
        };

        let zero_query = vec![0.0f32; hidden];
        for live in [1usize, 2, 31, 32, 33, 1024] {
            let keys = vec![0.0f32; CAPACITY * KV_HEADS * HEAD_DIM];
            let mut values = vec![31.0f32; CAPACITY * KV_HEADS * HEAD_DIM];
            for token in 0..live {
                for head in 0..KV_HEADS {
                    for feature in 0..HEAD_DIM {
                        values[(token * KV_HEADS + head) * HEAD_DIM + feature] =
                            head as f32 * 4.0 + (token % 7) as f32 * 0.25 - 0.75
                                + (feature % 16) as f32 * 0.03125
                                - 0.25;
                    }
                }
            }
            compare_case(
                &format!("synthetic_sdpa_equal_l{live}"),
                &zero_query,
                &keys,
                &values,
                live,
                None,
            );
        }

        // A configured window spanning the fixed cache capacity is accepted
        // and must retain the same fixed-capacity address mapping. A truncated
        // window is routed to materialized attention and rejected by the
        // direct-native builder guard in the structural emitter regressions.
        let live = 33;
        let keys = vec![0.0f32; CAPACITY * KV_HEADS * HEAD_DIM];
        let mut values = vec![29.0f32; CAPACITY * KV_HEADS * HEAD_DIM];
        for token in live - 8..live {
            for head in 0..KV_HEADS {
                for feature in 0..HEAD_DIM {
                    values[(token * KV_HEADS + head) * HEAD_DIM + feature] =
                        head as f32 * 2.0 + (feature % 8) as f32 * 0.125 - 0.5;
                }
            }
        }
        compare_case(
            "synthetic_sdpa_full_window_boundary",
            &zero_query,
            &keys,
            &values,
            live,
            Some(CAPACITY),
        );

        // MLX's underflow-safe exp2 path evaluates half the exponent and
        // squares. A natural-score gap of -100 falls below -126 in log2 space.
        let mut underflow_query = vec![0.0f32; hidden];
        for head in 0..QUERY_HEADS {
            underflow_query[head * HEAD_DIM] = 1.0;
        }
        let mut underflow_keys = vec![0.0f32; CAPACITY * KV_HEADS * HEAD_DIM];
        let mut underflow_values = vec![0.0f32; CAPACITY * KV_HEADS * HEAD_DIM];
        for head in 0..KV_HEADS {
            underflow_keys[(KV_HEADS + head) * HEAD_DIM] = -100.0;
            for feature in 0..HEAD_DIM {
                let carrier = head as f32 + (feature % 8) as f32 * 0.125 + 0.5;
                underflow_values[head * HEAD_DIM + feature] = carrier;
                underflow_values[(KV_HEADS + head) * HEAD_DIM + feature] = -carrier;
            }
        }
        compare_case(
            "synthetic_sdpa_exp2_underflow",
            &underflow_query,
            &underflow_keys,
            &underflow_values,
            2,
            None,
        );

        // Equal scores and adjacent BF16 values produce the exact halfway
        // 1.00390625; round-to-nearest-even must select 1.0.
        let halfway_keys = vec![0.0f32; CAPACITY * KV_HEADS * HEAD_DIM];
        let mut halfway_values = vec![0.0f32; CAPACITY * KV_HEADS * HEAD_DIM];
        for head in 0..KV_HEADS {
            for feature in 0..HEAD_DIM {
                halfway_values[head * HEAD_DIM + feature] = 1.0;
                halfway_values[(KV_HEADS + head) * HEAD_DIM + feature] = 1.0078125;
            }
        }
        let halfway = compare_case(
            "synthetic_sdpa_bf16_halfway",
            &zero_query,
            &halfway_keys,
            &halfway_values,
            2,
            None,
        );
        assert!(
            halfway
                .iter()
                .all(|value| value.to_bits() == 1.0f32.to_bits()),
            "BF16 halfway fixture did not round to the even 1.0 carrier"
        );
    }

    #[test]
    #[ignore = "requires pinned Gemma3n E2B checkpoint plus MLX and IREE CUDA"]
    fn pinned_e2b_one_row_prefix_decode_matches_mlx() {
        let model_dir_env = std::env::var_os("GEMMA3N_MODEL_DIR").expect("set GEMMA3N_MODEL_DIR");
        let model_dir = Path::new(&model_dir_env);
        let readme = std::fs::read_to_string(model_dir.join("README.md")).unwrap();
        assert!(
            readme.contains(&format!("# {MODEL_ID}")),
            "checkpoint README does not identify {MODEL_ID}"
        );
        assert_eq!(sha256(&model_dir.join("config.json")), CONFIG_SHA256);
        assert_eq!(
            sha256(&model_dir.join("model.safetensors.index.json")),
            INDEX_SHA256
        );
        assert_eq!(sha256(&model_dir.join("model.safetensors")), MODEL_SHA256);

        let mlx_prefix_decode = {
            let (loaded, _) = crate::load_model(model_dir).expect("load pinned MLX checkpoint");
            let language_model = match &loaded {
                LoadedModel::Gemma3n(model) => &model.language_model,
                LoadedModel::Gemma3nVLM(model) => &model.text_model.language_model,
                _ => panic!("pinned checkpoint did not load as Gemma3n"),
            };
            let mut caches = language_model.make_caches();
            let prefix = mlxcel_core::from_slice_i32(
                &PROMPT[..PROMPT.len() - 1],
                &[1, PROMPT.len() as i32 - 1],
            );
            let seeded = language_model.forward(&prefix, &mut caches);
            mlxcel_core::eval(&seeded);
            let last = mlxcel_core::from_slice_i32(&PROMPT[PROMPT.len() - 1..], &[1, 1]);
            array_to_f32(&language_model.forward(&last, &mut caches))
        };
        assert_eq!(
            argmax(&mlx_prefix_decode),
            236798,
            "pinned production-CUDA prefix-decode MLX top1 drifted"
        );
        mlxcel_core::memory::clear_cache();

        let device = std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "cuda".to_string());
        let xla = run_gemma3n_prefix_decode_diagnostic(model_dir, &device, CAPACITY, &PROMPT)
            .expect("one-row XLA prefix-decode diagnostic");
        assert_eq!(xla.active_slot, 3);
        assert_eq!(xla.carrier_tokens, [0, 0, 0, PROMPT[2]]);
        assert_eq!(xla.carrier_positions, [0, 0, 0, 2]);
        assert_eq!(xla.carrier_cache_lengths, [0, 0, 0, 2]);
        assert_eq!(xla.top1, argmax(&xla.logits));

        let mut failures = Vec::new();
        compare(
            "one_row_prefix_decode_logits",
            &mlx_prefix_decode,
            &xla.logits,
            RegressionEnvelope::pinned_cuda(1.5),
            &mut failures,
        );
        compare_top1(
            "one_row_prefix_decode_logits",
            &mlx_prefix_decode,
            &xla.logits,
            &mut failures,
        );
        eprintln!(
            "one-row prefix decode carrier token={:?} position={:?} cache_len={:?} \
             top1 mlx={} xla={}",
            xla.carrier_tokens,
            xla.carrier_positions,
            xla.carrier_cache_lengths,
            argmax(&mlx_prefix_decode),
            xla.top1,
        );
        assert!(
            failures.is_empty(),
            "one-row prefix decode mismatches:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    #[ignore = "requires pinned Gemma3n E2B checkpoint plus MLX and IREE CUDA"]
    fn pinned_e2b_intermediates_prefill_decode_and_greedy_match() {
        let model_dir_env = std::env::var_os("GEMMA3N_MODEL_DIR").expect("set GEMMA3N_MODEL_DIR");
        let model_dir = Path::new(&model_dir_env);
        let readme = std::fs::read_to_string(model_dir.join("README.md")).unwrap();
        assert!(
            readme.contains(&format!("# {MODEL_ID}")),
            "checkpoint README does not identify {MODEL_ID}"
        );
        assert_eq!(sha256(&model_dir.join("config.json")), CONFIG_SHA256);
        assert_eq!(
            sha256(&model_dir.join("model.safetensors.index.json")),
            INDEX_SHA256
        );
        assert_eq!(sha256(&model_dir.join("model.safetensors")), MODEL_SHA256);

        let (mlx_prefix_decode, mlx_greedy, mlx_logits, mlx_segments) = {
            let (loaded, _) = crate::load_model(model_dir).expect("load pinned MLX checkpoint");
            let language_model = match &loaded {
                LoadedModel::Gemma3n(model) => &model.language_model,
                LoadedModel::Gemma3nVLM(model) => &model.text_model.language_model,
                _ => panic!("pinned checkpoint did not load as Gemma3n"),
            };
            assert_eq!(language_model.layer_idx_to_cache_idx[0], 0);
            let input = mlxcel_core::from_slice_i32(&PROMPT, &[1, PROMPT.len() as i32]);
            let mut caches = language_model.make_caches();
            let mlx = language_model.forward_diagnostics(&input, &mut caches);

            let mut prefix_caches = language_model.make_caches();
            let prefix = mlxcel_core::from_slice_i32(
                &PROMPT[..PROMPT.len() - 1],
                &[1, PROMPT.len() as i32 - 1],
            );
            let seeded = language_model.forward(&prefix, &mut prefix_caches);
            mlxcel_core::eval(&seeded);
            let last = mlxcel_core::from_slice_i32(&PROMPT[PROMPT.len() - 1..], &[1, 1]);
            let mlx_prefix_decode =
                array_to_f32(&language_model.forward(&last, &mut prefix_caches));

            let mut greedy_caches = language_model.make_caches();
            let full = language_model.forward(&input, &mut greedy_caches);
            let full_values = array_to_f32(&full);
            let vocab = language_model.config.vocab_size;
            let first = argmax(&full_values[full_values.len() - vocab..]);
            let next = mlxcel_core::from_slice_i32(&[first], &[1, 1]);
            let second_logits = array_to_f32(&language_model.forward(&next, &mut greedy_caches));
            let mlx_greedy = vec![first, argmax(&second_logits)];
            let mlx_logits_all = array_to_f32(&mlx.logits);
            let mlx_logits = mlx_logits_all[mlx_logits_all.len() - vocab..].to_vec();
            let mlx_segments = [
                ("scaled_embeddings", array_to_f32(&mlx.scaled_embeddings)),
                ("projected_ple", array_to_f32(&mlx.projected_ple)),
                ("layer0_laurel", array_to_f32(&mlx.layer0_laurel)),
                (
                    "layer0_ple_injected",
                    array_to_f32(&mlx.layer0_ple_injected),
                ),
                ("layer0_all_planes", array_to_f32(&mlx.layer0_all_planes)),
                (
                    "layer0_active_plane",
                    array_to_f32(&mlx.layer0_active_plane),
                ),
                (
                    "layer_mid_all_planes",
                    array_to_f32(&mlx.layer_mid_all_planes),
                ),
                (
                    "layer_mid_active_plane",
                    array_to_f32(&mlx.layer_mid_active_plane),
                ),
                (
                    "layer_last_all_planes",
                    array_to_f32(&mlx.layer_last_all_planes),
                ),
                (
                    "layer_last_active_plane",
                    array_to_f32(&mlx.layer_last_active_plane),
                ),
                (
                    "layer0_k",
                    mlx_layer0_cache_prefix(&mlx.layer0_k, PROMPT.len()),
                ),
                (
                    "layer0_v",
                    mlx_layer0_cache_prefix(&mlx.layer0_v, PROMPT.len()),
                ),
                ("final_hidden", array_to_f32(&mlx.final_hidden)),
                ("logits", mlx_logits.clone()),
            ];
            (mlx_prefix_decode, mlx_greedy, mlx_logits, mlx_segments)
        };
        assert_eq!(
            argmax(&mlx_logits),
            236798,
            "pinned production-CUDA full-prefill MLX top1 drifted"
        );
        assert_eq!(
            argmax(&mlx_prefix_decode),
            236798,
            "pinned production-CUDA prefix-decode MLX top1 drifted"
        );
        assert_eq!(
            mlx_greedy,
            vec![236798, 236789],
            "pinned production-CUDA MLX greedy reference drifted"
        );
        // XLA allocates a second copy of the multi-GiB checkpoint. Release every
        // MLX model/cache/array handle and its allocator cache before loading it.
        mlxcel_core::memory::clear_cache();

        let device = std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "cuda".to_string());
        let xla = run_gemma3n_canonical_diagnostics(
            model_dir,
            &device,
            CAPACITY,
            &PROMPT,
            mlx_greedy.len(),
        )
        .expect("single-load XLA canonical diagnostics");
        xla.layout.validate().unwrap();

        let mut failures = Vec::new();
        // These max-absolute thresholds are no wider than 2x the immediately
        // preceding pinned production-CUDA measurement (most are exactly 2x).
        // Cosine and reference-RMS-normalized RMSE additionally prevent a broad
        // directional or scale regression from hiding inside a local max bound.
        for (name, mlx_values) in mlx_segments {
            let xla_values =
                xla_segment_prefix(&xla.intermediates, &xla.layout, name, PROMPT.len());
            let max_abs = match name {
                "scaled_embeddings" => 0.0,
                "projected_ple" => 0.125,
                "layer0_laurel" => 1.0,
                "layer0_ple_injected" => 0.25,
                "layer0_all_planes" => 3.0,
                "layer0_active_plane" => 1.0,
                "layer_mid_all_planes" => 3.0,
                "layer_mid_active_plane" => 1.0,
                "layer_last_all_planes" => 3.0,
                "layer_last_active_plane" => 1.0,
                "layer0_k" => 0.016,
                "layer0_v" => 0.0625,
                "final_hidden" => 6.0,
                "logits" => 1.5,
                _ => panic!("missing pinned CUDA regression envelope for {name}"),
            };
            compare(
                name,
                &mlx_values,
                &xla_values,
                RegressionEnvelope::pinned_cuda(max_abs),
                &mut failures,
            );
        }
        compare(
            "token_prefill_logits",
            &mlx_logits,
            &xla.token_prefill_logits,
            RegressionEnvelope::pinned_cuda(1.5),
            &mut failures,
        );
        compare(
            "prepared_prefill_logits",
            &mlx_logits,
            &xla.prepared_prefill_logits,
            RegressionEnvelope::pinned_cuda(1.5),
            &mut failures,
        );
        compare_exact(
            "token_vs_prepared_logits",
            &xla.token_prefill_logits,
            &xla.prepared_prefill_logits,
            &mut failures,
        );
        compare(
            "prefix_decode_logits",
            &mlx_prefix_decode,
            &xla.prefix_decode_logits,
            RegressionEnvelope::pinned_cuda(1.5),
            &mut failures,
        );
        let diagnostic_logits =
            xla_segment_prefix(&xla.intermediates, &xla.layout, "logits", PROMPT.len());
        compare_top1(
            "diagnostic_logits",
            &mlx_logits,
            &diagnostic_logits,
            &mut failures,
        );
        compare_top1(
            "token_prefill_logits",
            &mlx_logits,
            &xla.token_prefill_logits,
            &mut failures,
        );
        compare_top1(
            "prepared_prefill_logits",
            &mlx_logits,
            &xla.prepared_prefill_logits,
            &mut failures,
        );
        compare_top1(
            "token_vs_prepared_logits",
            &xla.token_prefill_logits,
            &xla.prepared_prefill_logits,
            &mut failures,
        );
        compare_top1(
            "prefix_decode_logits",
            &mlx_prefix_decode,
            &xla.prefix_decode_logits,
            &mut failures,
        );
        eprintln!(
            "logits top10 mlx={:?} token={:?} prepared={:?} decode_mlx={:?} decode_xla={:?}",
            top_k(&mlx_logits, 10),
            top_k(&xla.token_prefill_logits, 10),
            top_k(&xla.prepared_prefill_logits, 10),
            top_k(&mlx_prefix_decode, 10),
            top_k(&xla.prefix_decode_logits, 10),
        );
        eprintln!("greedy mlx={mlx_greedy:?} xla={:?}", xla.greedy_tokens);
        if mlx_greedy != xla.greedy_tokens {
            failures.push(format!(
                "greedy mismatch mlx={mlx_greedy:?} xla={:?}",
                xla.greedy_tokens
            ));
        }
        assert!(
            failures.is_empty(),
            "pinned Gemma3n canonical mismatches:\n{}",
            failures.join("\n")
        );
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

/// Gemma 3n audio embedder with both the learned hard-token table and the
/// soft encoder projection. Audio padding uses the final hard-vocabulary row,
/// while real encoder frames use the separate soft normalization path.
pub struct Gemma3nAudioEmbedder {
    embedding: UnifiedEmbedding,
    hard_embedding_norm: RMSNorm,
    soft_embedding_norm: RMSNorm,
    embedding_projection: UnifiedLinear,
    post_projection_norm: RMSNoScale,
    vocab_size: usize,
    vocab_offset: i32,
}

impl Gemma3nAudioEmbedder {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &crate::audio::gemma3n::Gemma3nAudioConfig,
        text_hidden_size: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let embedding_prefix = format!("{prefix}.embedding");
        let weight_key = format!("{embedding_prefix}.weight");
        let weight = weights
            .get(&weight_key)
            .ok_or_else(|| format!("Gemma3n audio weight not found: {weight_key}"))?;
        let shape = mlxcel_core::array_shape(weight);
        let quantized = weights.contains_key(&format!("{embedding_prefix}.scales"));
        if shape.len() != 2
            || shape[0] != config.vocab_size as i32
            || (!quantized && shape[1] != config.hidden_size as i32)
        {
            return Err(format!(
                "{weight_key} has shape {shape:?}; expected logical [{}, {}]",
                config.vocab_size, config.hidden_size
            ));
        }
        if let Some(scales) = weights.get(&format!("{embedding_prefix}.scales")) {
            let scales_shape = mlxcel_core::array_shape(scales);
            if scales_shape.len() != 2 || scales_shape[0] != config.vocab_size as i32 {
                return Err(format!(
                    "{embedding_prefix}.scales has invalid shape {scales_shape:?}"
                ));
            }
            if let Some(biases) = weights.get(&format!("{embedding_prefix}.biases"))
                && mlxcel_core::array_shape(biases) != scales_shape
            {
                return Err(format!(
                    "{embedding_prefix}.biases must match scales shape {scales_shape:?}"
                ));
            }
        }
        let embedding =
            UnifiedEmbedding::from_weights(weights, &embedding_prefix, group_size, bits)?;
        let hard_weight =
            get_weight_copy(weights, &format!("{prefix}.hard_embedding_norm.weight"))?;
        let soft_weight =
            get_weight_copy(weights, &format!("{prefix}.soft_embedding_norm.weight"))?;
        if mlxcel_core::array_shape(&hard_weight) != [config.hidden_size as i32]
            || mlxcel_core::array_shape(&soft_weight) != [config.hidden_size as i32]
        {
            return Err(format!(
                "{prefix} audio RMS norm weights must have {} elements",
                config.hidden_size
            ));
        }
        Ok(Self {
            embedding,
            hard_embedding_norm: RMSNorm::new(hard_weight, config.rms_norm_eps),
            soft_embedding_norm: RMSNorm::new(soft_weight, config.rms_norm_eps),
            embedding_projection: crate::audio::gemma3n::checked_unified_linear(
                weights,
                &format!("{prefix}.embedding_projection"),
                config.hidden_size,
                text_hidden_size,
                group_size,
                bits,
            )?,
            post_projection_norm: RMSNoScale::new(text_hidden_size as i32, config.rms_norm_eps),
            vocab_size: config.vocab_size,
            vocab_offset: config.vocab_offset,
        })
    }

    fn project(&self, normalized: &MlxArray) -> UniquePtr<MlxArray> {
        self.post_projection_norm
            .forward(&self.embedding_projection.forward(normalized))
    }

    pub fn forward_soft(&self, inputs: &MlxArray) -> UniquePtr<MlxArray> {
        self.project(&self.soft_embedding_norm.forward(inputs))
    }

    /// Replace the text-table rows for hard audio vocabulary tokens. The
    /// official model performs this before replacing `<audio_soft_token>`
    /// rows with encoder output; notably `<end_of_audio>` remains a hard
    /// audio embedding and must not use the text embedding table.
    pub fn merge_hard_tokens(
        &self,
        input_ids: &MlxArray,
        inputs_embeds: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let offset = mlxcel_core::from_slice_i32(&[self.vocab_offset], &[1]);
        let audio_mask = mlxcel_core::greater_equal(input_ids, &offset);
        let dummy =
            mlxcel_core::from_slice_i32(&[self.vocab_offset + self.vocab_size as i32 - 1], &[1]);
        let global_ids = mlxcel_core::where_cond(&audio_mask, input_ids, &dummy);
        let local_ids = mlxcel_core::subtract(&global_ids, &offset);
        let hard = self.embedding.forward(&local_ids);
        let hard = self.project(&self.hard_embedding_norm.forward(&hard));
        mlxcel_core::where_cond(
            &mlxcel_core::expand_dims(&audio_mask, -1),
            &hard,
            inputs_embeds,
        )
    }

    pub fn padding_embedding(&self) -> UniquePtr<MlxArray> {
        let token = mlxcel_core::from_slice_i32(&[(self.vocab_size - 1) as i32], &[1, 1]);
        let embedded = self.embedding.forward(&token);
        self.project(&self.hard_embedding_norm.forward(&embedded))
    }
}

#[cfg(test)]
mod audio_embedder_tests {
    use super::*;

    fn values(array: &MlxArray) -> Vec<f32> {
        mlxcel_core::eval(array);
        mlxcel_core::array_to_raw_bytes(array)
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn hard_audio_vocabulary_replaces_only_tokens_at_or_above_offset() {
        let config = crate::audio::gemma3n::Gemma3nAudioConfig {
            vocab_size: 2,
            vocab_offset: 100,
            hidden_size: 2,
            ..crate::audio::gemma3n::Gemma3nAudioConfig::default()
        };
        let mut weights = WeightMap::new();
        weights.insert(
            "embed.embedding.weight".into(),
            mlxcel_core::from_slice_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]),
        );
        for name in ["hard_embedding_norm", "soft_embedding_norm"] {
            weights.insert(
                format!("embed.{name}.weight"),
                mlxcel_core::ones(&[2], mlxcel_core::dtype::FLOAT32),
            );
        }
        weights.insert(
            "embed.embedding_projection.weight".into(),
            mlxcel_core::from_slice_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]),
        );
        let embedder =
            Gemma3nAudioEmbedder::from_weights(&weights, "embed", &config, 2, 64, 4).unwrap();
        let ids = mlxcel_core::from_slice_i32(&[99, 100, 101], &[1, 3]);
        let text = mlxcel_core::zeros(&[1, 3, 2], mlxcel_core::dtype::FLOAT32);
        let merged = embedder.merge_hard_tokens(&ids, &text);
        let output = values(&merged);
        assert_eq!(&output[..2], &[0.0, 0.0]);
        assert!(output[2] > 1.4 && output[3].abs() < 1e-6);
        assert!(output[4].abs() < 1e-6 && output[5] > 1.4);
    }
}
