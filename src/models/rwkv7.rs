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

// RWKV7: Recurrent neural network with time mixing and channel mixing
// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rwkv7.py

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Rwkv7Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub a_low_rank_dim: usize,
    pub v_low_rank_dim: usize,
    pub gate_low_rank_dim: usize,
    pub decay_low_rank_dim: usize,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<Quantization>,
}

fn default_norm_eps() -> f32 {
    1e-5
}

fn default_true() -> bool {
    true
}

impl Rwkv7Config {
    pub fn num_heads(&self) -> usize {
        self.hidden_size / self.head_dim
    }

    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }
}

// RWKV7 Cache.
pub struct Rwkv7Cache {
    pub token_shift_cache: Option<UniquePtr<MlxArray>>,
    pub state_cache: Option<UniquePtr<MlxArray>>,
    pub ffn_cache: Option<UniquePtr<MlxArray>>,
}

impl Rwkv7Cache {
    pub fn new() -> Self {
        Self {
            token_shift_cache: None,
            state_cache: None,
            ffn_cache: None,
        }
    }
}

impl Default for Rwkv7Cache {
    fn default() -> Self {
        Self::new()
    }
}

// Helper Functions.
/// Helper function: x + y * z
fn addcmul(x: &MlxArray, y: &MlxArray, z: &MlxArray) -> UniquePtr<MlxArray> {
    let yz = mlxcel_core::multiply(y, z);
    mlxcel_core::add(x, &yz)
}

/// L2 normalization with epsilon for numerical stability
fn l2_norm(x: &MlxArray) -> UniquePtr<MlxArray> {
    let norm = mlxcel_core::linalg_norm(x, -1, true);
    let eps = mlxcel_core::full_f32(&[1], 1e-7, mlxcel_core::array_dtype(&norm));
    let max_norm = mlxcel_core::maximum(&norm, &eps);
    mlxcel_core::divide(x, &max_norm)
}

/// Token shift module - shifts input by one timestep
fn token_shift(x: &MlxArray, state: Option<&MlxArray>) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let batch = shape[0];
    let seq_len = shape[1];
    let dim = shape[2];

    let state_arr = match state {
        Some(s) => mlxcel_core::copy(s),
        None => mlxcel_core::zeros(&[batch, 1, dim], mlxcel_core::array_dtype(x)),
    };

    if seq_len == 1 {
        state_arr
    } else {
        // x_prev = x[:, :-1, :]
        let x_prev = mlxcel_core::slice(x, &[0, 0, 0], &[batch, seq_len - 1, dim]);
        mlxcel_core::concatenate(&state_arr, &x_prev, 1)
    }
}

/// WKV7 step operation (non-Metal fallback)
fn wkv7_step_ops(
    r: &MlxArray,     // [B, H, D]
    w: &MlxArray,     // [B, H, D]
    k: &MlxArray,     // [B, H, D]
    v: &MlxArray,     // [B, H, D]
    a: &MlxArray,     // [B, H, D]
    b: &MlxArray,     // [B, H, D]
    state: &MlxArray, // [B, H, D, D]
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // sab = (state @ a[..., None]) @ b[..., None, :]
    let a_expanded = mlxcel_core::expand_dims(a, -1);
    let b_expanded_t = mlxcel_core::expand_dims(b, -2);
    let state_a = mlxcel_core::matmul(state, &a_expanded);
    let sab = mlxcel_core::matmul(&state_a, &b_expanded_t);

    // state = state * w[:, :, None, :] + v[..., None] @ k[..., None, :] + sab
    let w_expanded = mlxcel_core::expand_dims(w, -2);
    let v_expanded = mlxcel_core::expand_dims(v, -1);
    let k_expanded = mlxcel_core::expand_dims(k, -2);

    let state_w = mlxcel_core::multiply(state, &w_expanded);
    let vk = mlxcel_core::matmul(&v_expanded, &k_expanded);
    let new_state = mlxcel_core::add(&mlxcel_core::add(&state_w, &vk), &sab);

    // y = state @ r[..., None]
    let r_expanded = mlxcel_core::expand_dims(r, -1);
    let y = mlxcel_core::matmul(&new_state, &r_expanded);

    (y, new_state)
}

// Per-head Layer Normalization.
struct LayerNormPerHead {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    eps: f32,
}

impl LayerNormPerHead {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let bias_name = format!("{}.bias", prefix);

        let weight = weights
            .get(&weight_name)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;

        let bias = weights
            .get(&bias_name)
            .map(|b| mlxcel_core::copy(b))
            .ok_or_else(|| format!("Bias not found: {}", bias_name))?;

        // Reshape if needed
        let weight = mlxcel_core::reshape(&weight, &[num_heads, head_dim]);
        let bias = mlxcel_core::reshape(&bias, &[num_heads, head_dim]);

        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Manual layer norm implementation per head
        let mean = mlxcel_core::mean_axis(x, -1, true);
        let x_centered = mlxcel_core::subtract(x, &mean);
        let x_sq = mlxcel_core::square(&x_centered);
        let var = mlxcel_core::mean_axis(&x_sq, -1, true);
        let eps_arr = mlxcel_core::full_f32(&[1], self.eps, mlxcel_core::array_dtype(&var));
        let var_eps = mlxcel_core::add(&var, &eps_arr);
        let std = mlxcel_core::sqrt(&var_eps);
        let normalized = mlxcel_core::divide(&x_centered, &std);

        // weight * normalized + bias
        let weighted = mlxcel_core::multiply(&self.weight, &normalized);
        mlxcel_core::add(&weighted, &self.bias)
    }
}

// LoRA (Low-Rank Adaptation) Module.
struct LoRA {
    linear1: UnifiedLinear,
    linear2: UnifiedLinear,
    activation: String,
}

impl LoRA {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        activation: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let linear1_prefix = format!("{}.lora.0", prefix);
        let linear2_prefix = format!("{}.lora.2", prefix);

        let linear1 = UnifiedLinear::from_weights(weights, &linear1_prefix, group_size, bits)?;
        let linear2 = UnifiedLinear::from_weights(weights, &linear2_prefix, group_size, bits)?;

        Ok(Self {
            linear1,
            linear2,
            activation: activation.to_string(),
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.linear1.forward(x);
        let h_act = match self.activation.as_str() {
            "tanh" => mlxcel_core::tanh(&h),
            "sigmoid" => mlxcel_core::sigmoid(&h),
            "relu" => mlxcel_core::relu(&h),
            _ => h, // Identity
        };
        self.linear2.forward(&h_act)
    }
}

// RWKV7 Channel Mixing (FFN).
struct Rwkv7ChannelMixing {
    key: UnifiedLinear,
    value: UnifiedLinear,
    x_k: UniquePtr<MlxArray>,
}

impl Rwkv7ChannelMixing {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Rwkv7Config,
    ) -> Result<Self, String> {
        let key_prefix = format!("{}.key", prefix);
        let value_prefix = format!("{}.value", prefix);
        let x_k_name = format!("{}.x_k", prefix);

        let key =
            UnifiedLinear::from_weights(weights, &key_prefix, config.group_size(), config.bits())?;
        let value = UnifiedLinear::from_weights(
            weights,
            &value_prefix,
            config.group_size(),
            config.bits(),
        )?;
        let x_k = weights
            .get(&x_k_name)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("x_k not found: {}", x_k_name))?;

        Ok(Self { key, value, x_k })
    }

    fn forward(&self, x: &MlxArray, cache: &mut Option<&mut Rwkv7Cache>) -> UniquePtr<MlxArray> {
        let state = cache.as_ref().and_then(|c| c.ffn_cache.as_ref());
        let x_prev = token_shift(x, state.map(|s| s.as_ref().unwrap()));

        // xx = addcmul(x, x_prev - x, self.x_k)
        let diff = mlxcel_core::subtract(&x_prev, x);
        let xx = addcmul(x, &diff, &self.x_k);

        // Update cache
        if let Some(c) = cache {
            let shape = mlxcel_core::array_shape(x);
            let batch = shape[0];
            let seq_len = shape[1];
            let dim = shape[2];
            let last_x = mlxcel_core::slice(x, &[0, seq_len - 1, 0], &[batch, seq_len, dim]);
            c.ffn_cache = Some(last_x);
        }

        // relu^2 = relu(x)^2
        let key_out = self.key.forward(&xx);
        let relu_out = mlxcel_core::relu(&key_out);
        let relu2 = mlxcel_core::square(&relu_out);

        self.value.forward(&relu2)
    }
}

// RWKV7 Time Mixing (main recurrent component).
#[allow(dead_code)]
struct Rwkv7TimeMixing {
    layer_idx: usize,
    hidden_size: usize,
    head_dim: usize,
    num_heads: usize,

    // Learnable shift parameters
    x_r: UniquePtr<MlxArray>,
    x_w: UniquePtr<MlxArray>,
    x_k: UniquePtr<MlxArray>,
    x_v: UniquePtr<MlxArray>,
    x_a: UniquePtr<MlxArray>,
    x_g: UniquePtr<MlxArray>,

    // Per-head parameters
    k_k: UniquePtr<MlxArray>,
    k_a: UniquePtr<MlxArray>,
    r_k: UniquePtr<MlxArray>,

    // Projections
    r_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,

    // Layer norm
    g_norm: LayerNormPerHead,

    // LoRA modules
    w_lora: LoRA,
    v_lora: Option<LoRA>,
    a_lora: LoRA,
    g_lora: LoRA,
}

impl Rwkv7TimeMixing {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Rwkv7Config,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim;
        let num_heads = config.num_heads();

        // Load shift parameters
        let x_r = weights
            .get(&format!("{}.x_r", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "x_r not found".to_string())?;
        let x_w = weights
            .get(&format!("{}.x_w", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "x_w not found".to_string())?;
        let x_k = weights
            .get(&format!("{}.x_k", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "x_k not found".to_string())?;
        let x_v = weights
            .get(&format!("{}.x_v", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "x_v not found".to_string())?;
        let x_a = weights
            .get(&format!("{}.x_a", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "x_a not found".to_string())?;
        let x_g = weights
            .get(&format!("{}.x_g", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "x_g not found".to_string())?;

        // Load per-head parameters
        let k_k = weights
            .get(&format!("{}.k_k", prefix))
            .map(|w| mlxcel_core::reshape(w, &[num_heads as i32, head_dim as i32]))
            .ok_or_else(|| "k_k not found".to_string())?;
        let k_a = weights
            .get(&format!("{}.k_a", prefix))
            .map(|w| mlxcel_core::reshape(w, &[num_heads as i32, head_dim as i32]))
            .ok_or_else(|| "k_a not found".to_string())?;
        let r_k = weights
            .get(&format!("{}.r_k", prefix))
            .map(|w| mlxcel_core::reshape(w, &[num_heads as i32, head_dim as i32]))
            .ok_or_else(|| "r_k not found".to_string())?;

        // Load projections
        let r_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.r_proj", prefix),
            config.group_size(),
            config.bits(),
        )?;
        let k_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.k_proj", prefix),
            config.group_size(),
            config.bits(),
        )?;
        let v_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.v_proj", prefix),
            config.group_size(),
            config.bits(),
        )?;
        let o_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.o_proj", prefix),
            config.group_size(),
            config.bits(),
        )?;

        // Load layer norm
        let g_norm = LayerNormPerHead::from_weights(
            weights,
            &format!("{}.g_norm", prefix),
            num_heads as i32,
            head_dim as i32,
            64e-5,
        )?;

        // Load LoRA modules
        let w_lora = LoRA::from_weights(
            weights,
            &format!("{}.w_lora", prefix),
            "tanh",
            config.group_size(),
            config.bits(),
        )?;

        let v_lora = if layer_idx > 0 {
            Some(LoRA::from_weights(
                weights,
                &format!("{}.v_lora", prefix),
                "none",
                config.group_size(),
                config.bits(),
            )?)
        } else {
            None
        };

        let a_lora = LoRA::from_weights(
            weights,
            &format!("{}.a_lora", prefix),
            "none",
            config.group_size(),
            config.bits(),
        )?;
        let g_lora = LoRA::from_weights(
            weights,
            &format!("{}.g_lora", prefix),
            "sigmoid",
            config.group_size(),
            config.bits(),
        )?;

        Ok(Self {
            layer_idx,
            hidden_size,
            head_dim,
            num_heads,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            k_k,
            k_a,
            r_k,
            r_proj,
            k_proj,
            v_proj,
            o_proj,
            g_norm,
            w_lora,
            v_lora,
            a_lora,
            g_lora,
        })
    }

    fn wkv7(
        &self,
        r: &MlxArray,
        w: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        a: &MlxArray,
        b: &MlxArray,
        state: Option<&MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = mlxcel_core::array_shape(r);
        let batch = shape[0];
        let seq_len = shape[1] as usize;

        let state_arr = match state {
            Some(s) => mlxcel_core::copy(s),
            None => mlxcel_core::zeros(
                &[
                    batch,
                    self.num_heads as i32,
                    self.head_dim as i32,
                    self.head_dim as i32,
                ],
                mlxcel_core::array_dtype(r),
            ),
        };

        // Use step operations for CPU fallback
        let mut ys = Vec::new();
        let mut current_state = state_arr;

        for t in 0..seq_len {
            // Slice each timestep
            let r_t = mlxcel_core::slice(
                r,
                &[0, t as i32, 0, 0],
                &[
                    batch,
                    t as i32 + 1,
                    self.num_heads as i32,
                    self.head_dim as i32,
                ],
            );
            let r_t = mlxcel_core::squeeze_axis(&r_t, 1);

            let w_t = mlxcel_core::slice(
                w,
                &[0, t as i32, 0, 0],
                &[
                    batch,
                    t as i32 + 1,
                    self.num_heads as i32,
                    self.head_dim as i32,
                ],
            );
            let w_t = mlxcel_core::squeeze_axis(&w_t, 1);

            let k_t = mlxcel_core::slice(
                k,
                &[0, t as i32, 0, 0],
                &[
                    batch,
                    t as i32 + 1,
                    self.num_heads as i32,
                    self.head_dim as i32,
                ],
            );
            let k_t = mlxcel_core::squeeze_axis(&k_t, 1);

            let v_t = mlxcel_core::slice(
                v,
                &[0, t as i32, 0, 0],
                &[
                    batch,
                    t as i32 + 1,
                    self.num_heads as i32,
                    self.head_dim as i32,
                ],
            );
            let v_t = mlxcel_core::squeeze_axis(&v_t, 1);

            let a_t = mlxcel_core::slice(
                a,
                &[0, t as i32, 0, 0],
                &[
                    batch,
                    t as i32 + 1,
                    self.num_heads as i32,
                    self.head_dim as i32,
                ],
            );
            let a_t = mlxcel_core::squeeze_axis(&a_t, 1);

            let b_t = mlxcel_core::slice(
                b,
                &[0, t as i32, 0, 0],
                &[
                    batch,
                    t as i32 + 1,
                    self.num_heads as i32,
                    self.head_dim as i32,
                ],
            );
            let b_t = mlxcel_core::squeeze_axis(&b_t, 1);

            let (y, new_state) = wkv7_step_ops(&r_t, &w_t, &k_t, &v_t, &a_t, &b_t, &current_state);
            ys.push(y);
            current_state = new_state;
        }

        // Stack outputs along sequence dimension
        let y_stacked = mlxcel_core::utils::stack_arrays(&ys, 1);
        let y_out = mlxcel_core::astype(&y_stacked, mlxcel_core::array_dtype(r));

        (y_out, current_state)
    }

    fn forward(
        &self,
        x: &MlxArray,
        v_first: Option<&MlxArray>,
        cache: &mut Option<&mut Rwkv7Cache>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let (token_shift_cache, state_cache) = if let Some(c) = cache.as_ref() {
            (
                c.token_shift_cache.as_ref().map(|s| s.as_ref().unwrap()),
                c.state_cache.as_ref().map(|s| s.as_ref().unwrap()),
            )
        } else {
            (None, None)
        };

        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let seq_len = shape[1];
        let dim = shape[2];

        let x_prev = token_shift(x, token_shift_cache);
        let xx = mlxcel_core::subtract(&x_prev, x);

        // Compute shifted inputs
        let xr = addcmul(x, &xx, &self.x_r);
        let xw = addcmul(x, &xx, &self.x_w);
        let xk = addcmul(x, &xx, &self.x_k);
        let xv = addcmul(x, &xx, &self.x_v);
        let xa = addcmul(x, &xx, &self.x_a);
        let xg = addcmul(x, &xx, &self.x_g);

        // Projections
        let key = self.k_proj.forward(&xk);
        let key = mlxcel_core::reshape(
            &key,
            &[batch, seq_len, self.num_heads as i32, self.head_dim as i32],
        );

        let value = self.v_proj.forward(&xv);
        let value = mlxcel_core::reshape(
            &value,
            &[batch, seq_len, self.num_heads as i32, self.head_dim as i32],
        );

        let receptance = self.r_proj.forward(&xr);
        let receptance = mlxcel_core::reshape(
            &receptance,
            &[batch, seq_len, self.num_heads as i32, self.head_dim as i32],
        );

        let iclr = mlxcel_core::sigmoid(&self.a_lora.forward(&xa));
        let iclr = mlxcel_core::reshape(
            &iclr,
            &[batch, seq_len, self.num_heads as i32, self.head_dim as i32],
        );

        let gate = self.g_lora.forward(&xg);

        // Handle v_first for layer connections
        let (value, new_v_first) = if self.layer_idx == 0 {
            (mlxcel_core::copy(&value), value)
        } else if let Some(v_lora) = &self.v_lora {
            let vv = mlxcel_core::sigmoid(&v_lora.forward(&xv));
            let vv = mlxcel_core::reshape(
                &vv,
                &[batch, seq_len, self.num_heads as i32, self.head_dim as i32],
            );
            let v_first_arr = v_first.unwrap();
            let diff = mlxcel_core::subtract(v_first_arr, &value);
            let adjusted_value = addcmul(&value, &diff, &vv);
            (adjusted_value, mlxcel_core::copy(v_first_arr))
        } else {
            (
                mlxcel_core::copy(&value),
                mlxcel_core::copy(v_first.unwrap()),
            )
        };

        // Compute decay
        let decay_raw = self.w_lora.forward(&xw);
        let decay_raw = mlxcel_core::reshape(
            &decay_raw,
            &[batch, seq_len, self.num_heads as i32, self.head_dim as i32],
        );
        let decay = mlxcel_core::sigmoid(&decay_raw);

        let decay_dtype = mlxcel_core::array_dtype(&decay);
        let neg_coef = mlxcel_core::full_f32(&[1], -0.606531, decay_dtype);
        let decay_scaled = mlxcel_core::multiply(&neg_coef, &decay);
        let decay_exp = mlxcel_core::exp(&decay_scaled);

        // Compute k normalization and adjustments
        let k_kk = mlxcel_core::multiply(&key, &self.k_k);
        let kk = l2_norm(&k_kk);
        let key_dtype = mlxcel_core::array_dtype(&key);
        let one = mlxcel_core::full_f32(&[1], 1.0, key_dtype);
        let iclr_minus_one = mlxcel_core::subtract(&iclr, &one);
        let k_adj = mlxcel_core::multiply(&iclr_minus_one, &self.k_a);
        let one_plus_k_adj = mlxcel_core::add(&one, &k_adj);
        let key_adjusted = mlxcel_core::multiply(&key, &one_plus_k_adj);

        let neg_one = mlxcel_core::full_f32(&[1], -1.0, mlxcel_core::array_dtype(&kk));
        let a = mlxcel_core::multiply(&neg_one, &kk);
        let b_val = mlxcel_core::multiply(&kk, &iclr);

        // Run WKV7
        let (mut out, new_state_cache) = self.wkv7(
            &receptance,
            &decay_exp,
            &key_adjusted,
            &value,
            &a,
            &b_val,
            state_cache,
        );

        // Apply layer norm
        out = self.g_norm.forward(&out);

        // Add receptance * key * r_k contribution
        let rk_product = mlxcel_core::multiply(&receptance, &key);
        let rk_product = mlxcel_core::multiply(&rk_product, &self.r_k);
        let rk_sum = mlxcel_core::sum_axis(&rk_product, -1, true);
        let rk_contribution = mlxcel_core::multiply(&rk_sum, &value);
        out = mlxcel_core::add(&out, &rk_contribution);
        out = mlxcel_core::reshape(&out, &[batch, seq_len, dim]);

        // Update cache
        if let Some(c) = cache {
            let last_x = mlxcel_core::slice(x, &[0, seq_len - 1, 0], &[batch, seq_len, dim]);
            c.token_shift_cache = Some(last_x);
            c.state_cache = Some(new_state_cache);
        }

        // Output projection with gate
        let out_gated = mlxcel_core::multiply(&out, &gate);
        let final_out = self.o_proj.forward(&out_gated);

        (final_out, new_v_first)
    }
}

// RWKV7 Layer.
struct Rwkv7Layer {
    layer_idx: usize,
    pre_norm: Option<RMSNorm>,
    attn: Rwkv7TimeMixing,
    ffn: Rwkv7ChannelMixing,
    attn_norm: RMSNorm,
    ffn_norm: RMSNorm,
}

impl Rwkv7Layer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Rwkv7Config,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let pre_norm = if layer_idx == 0 {
            let pre_norm_prefix = format!("{}.pre_norm", prefix);
            let weight = weights
                .get(&format!("{}.weight", pre_norm_prefix))
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| "Pre-norm weight not found".to_string())?;
            Some(RMSNorm::new(weight, config.norm_eps))
        } else {
            None
        };

        let attn =
            Rwkv7TimeMixing::from_weights(weights, &format!("{}.attn", prefix), config, layer_idx)?;

        let ffn = Rwkv7ChannelMixing::from_weights(weights, &format!("{}.ffn", prefix), config)?;

        let attn_norm_weight = weights
            .get(&format!("{}.attn_norm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "Attn norm weight not found".to_string())?;
        let attn_norm = RMSNorm::new(attn_norm_weight, config.norm_eps);

        let ffn_norm_weight = weights
            .get(&format!("{}.ffn_norm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "FFN norm weight not found".to_string())?;
        let ffn_norm = RMSNorm::new(ffn_norm_weight, config.norm_eps);

        Ok(Self {
            layer_idx,
            pre_norm,
            attn,
            ffn,
            attn_norm,
            ffn_norm,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        v_first: Option<&MlxArray>,
        cache: &mut Option<&mut Rwkv7Cache>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let x = if self.layer_idx == 0 {
            if let Some(ref pre_norm) = self.pre_norm {
                pre_norm.forward(x)
            } else {
                mlxcel_core::copy(x)
            }
        } else {
            mlxcel_core::copy(x)
        };

        let normed = self.attn_norm.forward(&x);
        let (h, v_first_out) = self.attn.forward(&normed, v_first, cache);
        let h = mlxcel_core::add(&x, &h);

        let ffn_normed = self.ffn_norm.forward(&h);
        let ffn_out = self.ffn.forward(&ffn_normed, cache);
        let out = mlxcel_core::add(&h, &ffn_out);

        (out, v_first_out)
    }
}

// RWKV7 Model.
struct Rwkv7Model {
    embeddings: UnifiedEmbedding,
    layers: Vec<Rwkv7Layer>,
    norm: RMSNorm,
}

impl Rwkv7Model {
    fn from_weights(weights: &WeightMap, config: &Rwkv7Config) -> Result<Self, String> {
        let embeddings = UnifiedEmbedding::from_weights(
            weights,
            "model.embeddings",
            config.group_size(),
            config.bits(),
        )?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer =
                Rwkv7Layer::from_weights(weights, &format!("model.layers.{}", i), config, i)?;
            layers.push(layer);
        }

        let norm_weight = weights
            .get("model.norm.weight")
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "Final norm weight not found".to_string())?;
        let norm = RMSNorm::new(norm_weight, config.norm_eps);

        Ok(Self {
            embeddings,
            layers,
            norm,
        })
    }

    fn forward(&self, x: &MlxArray, cache: &mut Option<Vec<Rwkv7Cache>>) -> UniquePtr<MlxArray> {
        let mut h = self.embeddings.forward(x);

        let mut v_first: Option<UniquePtr<MlxArray>> = None;

        if let Some(caches) = cache {
            for (i, layer) in self.layers.iter().enumerate() {
                let mut layer_cache = Some(&mut caches[i]);
                let (new_h, new_v_first) = layer.forward(
                    &h,
                    v_first.as_ref().map(|v| v.as_ref().unwrap()),
                    &mut layer_cache,
                );
                h = new_h;
                v_first = Some(new_v_first);
            }
        } else {
            for layer in &self.layers {
                let (new_h, new_v_first) =
                    layer.forward(&h, v_first.as_ref().map(|v| v.as_ref().unwrap()), &mut None);
                h = new_h;
                v_first = Some(new_v_first);
            }
        }

        self.norm.forward(&h)
    }
}

// Full RWKV7 Language Model.
#[allow(dead_code)]
pub struct Rwkv7 {
    config: Rwkv7Config,
    model: Rwkv7Model,
    lm_head: Option<UnifiedLinear>,
    rwkv7_cache: Option<Vec<Rwkv7Cache>>,
}

impl Rwkv7 {
    pub fn from_weights(weights: &WeightMap, config: Rwkv7Config) -> Result<Self, String> {
        let model = Rwkv7Model::from_weights(weights, &config)?;

        let lm_head = if !config.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                weights,
                "lm_head",
                config.group_size(),
                config.bits(),
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            model,
            lm_head,
            rwkv7_cache: None,
        })
    }

    pub fn load(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Rwkv7Config = serde_json::from_str(&config_str)?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        Ok(Self::from_weights(&weights, config)?)
    }

    #[allow(dead_code)]
    fn make_rwkv7_cache(&self) -> Vec<Rwkv7Cache> {
        (0..self.config.num_hidden_layers)
            .map(|_| Rwkv7Cache::new())
            .collect()
    }
}

impl LanguageModel for Rwkv7 {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [mlxcel_core::layers::KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // RWKV7 doesn't use standard KV cache, it uses its own Rwkv7Cache
        // The standard KVCache parameter is ignored
        // TODO: Properly integrate RWKV7Cache with the generation framework

        // For now, do stateless forward pass
        let mut cache: Option<Vec<Rwkv7Cache>> = None;
        let h = self.model.forward(input_ids, &mut cache);

        if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)
        } else {
            self.model.embeddings.as_linear(&h)
        }
    }

    fn make_caches(&self) -> Vec<mlxcel_core::layers::KVCache> {
        // RWKV7 doesn't use standard KV caches, return empty vector
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn supports_padded_prefill(&self) -> bool {
        false // Padding tokens corrupt RWKV recurrent state
    }

    fn supports_batching(&self) -> bool {
        false // RWKV7 uses internal RNN-like state, not compatible with per-sequence KV isolation
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Default EOS token ID, should be loaded from tokenizer config
        vec![0]
    }
}
