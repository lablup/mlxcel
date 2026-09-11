// Copyright 2025-2026 Lablup Inc.
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

//! Gemma4 vision encoder.
//!
//! Used by: Gemma4 VLM

use mlxcel_core::layers::{RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VisionRopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_rope_theta() -> f32 {
    100.0
}

fn default_position_embedding_size() -> usize {
    10_240
}

fn default_default_output_length() -> usize {
    280
}

fn default_pooling_kernel_size() -> usize {
    3
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4VisionConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    #[serde(default = "default_rope_parameters")]
    pub rope_parameters: VisionRopeParameters,
    #[serde(default)]
    pub rms_norm_eps: f32,
    pub patch_size: usize,
    #[serde(default = "default_position_embedding_size")]
    pub position_embedding_size: usize,
    #[serde(default = "default_default_output_length")]
    pub default_output_length: usize,
    #[serde(default = "default_pooling_kernel_size")]
    pub pooling_kernel_size: usize,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default)]
    pub standardize: bool,
}

fn default_rope_parameters() -> VisionRopeParameters {
    VisionRopeParameters {
        rope_theta: default_rope_theta(),
    }
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

impl Gemma4VisionConfig {
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters.rope_theta
    }

    pub fn rms_norm_eps(&self) -> f32 {
        if self.rms_norm_eps == 0.0 {
            default_rms_norm_eps()
        } else {
            self.rms_norm_eps
        }
    }
}

fn copy_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|weight| mlxcel_core::copy(weight))
        .ok_or_else(|| format!("Weight not found: {}", key))
}

fn copy_weight_opt(weights: &WeightMap, key: &str) -> Option<UniquePtr<MlxArray>> {
    weights.get(key).map(|weight| mlxcel_core::copy(weight))
}

fn take_2d_embedding(
    table: &MlxArray,
    indices: &MlxArray,
    batch: i32,
    seq_len: i32,
    hidden_size: i32,
) -> UniquePtr<MlxArray> {
    let flat = mlxcel_core::reshape(indices, &[batch * seq_len]);
    let gathered = mlxcel_core::take(table, &flat, 0);
    mlxcel_core::reshape(&gathered, &[batch, seq_len, hidden_size])
}

fn build_patch_position_ids(
    patch_h: usize,
    patch_w: usize,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let num_patches = patch_h * patch_w;
    let mut x_ids = Vec::with_capacity(num_patches);
    let mut y_ids = Vec::with_capacity(num_patches);
    for y in 0..patch_h {
        for x in 0..patch_w {
            x_ids.push(x as i32);
            y_ids.push(y as i32);
        }
    }
    (
        mlxcel_core::from_slice_i32(&x_ids, &[1, num_patches as i32]),
        mlxcel_core::from_slice_i32(&y_ids, &[1, num_patches as i32]),
    )
}

fn build_rope_values(
    positions: &[i32],
    channels_per_dim: usize,
    base_frequency: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_per_dim = channels_per_dim / 2;
    let mut cos = Vec::with_capacity(positions.len() * channels_per_dim);
    let mut sin = Vec::with_capacity(positions.len() * channels_per_dim);

    let timescales: Vec<f32> = (0..half_per_dim)
        .map(|idx| {
            let exponent = (2.0 * idx as f32) / channels_per_dim as f32;
            base_frequency.powf(exponent)
        })
        .collect();

    for &position in positions {
        for &timescale in &timescales {
            let theta = position as f32 / timescale;
            cos.push(theta.cos());
            sin.push(theta.sin());
        }
        for &timescale in &timescales {
            let theta = position as f32 / timescale;
            cos.push(theta.cos());
            sin.push(theta.sin());
        }
    }

    (cos, sin)
}

fn slice_last_dim(x: &MlxArray, start: i32, end: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let ndim = shape.len();
    let mut starts = vec![0; ndim];
    let mut ends = shape;
    starts[ndim - 1] = start;
    ends[ndim - 1] = end;
    mlxcel_core::slice(x, &starts, &ends)
}

fn rotate_half(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let half = shape[shape.len() - 1] / 2;
    let x1 = slice_last_dim(x, 0, half);
    let x2 = slice_last_dim(x, half, half * 2);
    let neg_x2 = mlxcel_core::multiply_scalar(&x2, -1.0);
    mlxcel_core::concatenate(&neg_x2, &x1, shape.len() as i32 - 1)
}

struct Gemma4VisionRope {
    cos_x: UniquePtr<MlxArray>, // [1, L, 1, channels_per_dim]
    sin_x: UniquePtr<MlxArray>,
    cos_y: UniquePtr<MlxArray>,
    sin_y: UniquePtr<MlxArray>,
    channels_per_dim: i32,
}

impl Gemma4VisionRope {
    fn new(patch_h: usize, patch_w: usize, head_dim: usize, base_frequency: f32) -> Self {
        let mut x_positions = Vec::with_capacity(patch_h * patch_w);
        let mut y_positions = Vec::with_capacity(patch_h * patch_w);
        for y in 0..patch_h {
            for x in 0..patch_w {
                x_positions.push(x as i32);
                y_positions.push(y as i32);
            }
        }

        let channels_per_dim = 2 * (head_dim / 4);
        let (cos_x, sin_x) = build_rope_values(&x_positions, channels_per_dim, base_frequency);
        let (cos_y, sin_y) = build_rope_values(&y_positions, channels_per_dim, base_frequency);
        let seq_len = (patch_h * patch_w) as i32;
        let channels = channels_per_dim as i32;

        Self {
            cos_x: mlxcel_core::from_slice_f32(&cos_x, &[1, seq_len, 1, channels]),
            sin_x: mlxcel_core::from_slice_f32(&sin_x, &[1, seq_len, 1, channels]),
            cos_y: mlxcel_core::from_slice_f32(&cos_y, &[1, seq_len, 1, channels]),
            sin_y: mlxcel_core::from_slice_f32(&sin_y, &[1, seq_len, 1, channels]),
            channels_per_dim: channels,
        }
    }
}

fn apply_multidimensional_rope(inputs: &MlxArray, rope: &Gemma4VisionRope) -> UniquePtr<MlxArray> {
    let x_part = slice_last_dim(inputs, 0, rope.channels_per_dim);
    let y_part = slice_last_dim(inputs, rope.channels_per_dim, rope.channels_per_dim * 2);

    let x = mlxcel_core::add(
        &mlxcel_core::multiply(&x_part, &rope.cos_x),
        &mlxcel_core::multiply(&rotate_half(&x_part), &rope.sin_x),
    );
    let y = mlxcel_core::add(
        &mlxcel_core::multiply(&y_part, &rope.cos_y),
        &mlxcel_core::multiply(&rotate_half(&y_part), &rope.sin_y),
    );

    mlxcel_core::concatenate(&x, &y, -1)
}

struct ClippableLinear {
    linear: UnifiedLinear,
    input_min: Option<UniquePtr<MlxArray>>,
    input_max: Option<UniquePtr<MlxArray>>,
    output_min: Option<UniquePtr<MlxArray>>,
    output_max: Option<UniquePtr<MlxArray>>,
}

impl ClippableLinear {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        use_clipping: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            linear: UnifiedLinear::from_weights(
                weights,
                &format!("{}.linear", prefix),
                group_size,
                bits,
            )?,
            input_min: if use_clipping {
                copy_weight_opt(weights, &format!("{}.input_min", prefix))
            } else {
                None
            },
            input_max: if use_clipping {
                copy_weight_opt(weights, &format!("{}.input_max", prefix))
            } else {
                None
            },
            output_min: if use_clipping {
                copy_weight_opt(weights, &format!("{}.output_min", prefix))
            } else {
                None
            },
            output_max: if use_clipping {
                copy_weight_opt(weights, &format!("{}.output_max", prefix))
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let clipped_input = match (&self.input_min, &self.input_max) {
            (Some(min), Some(max)) => Some(mlxcel_core::clip(
                x,
                min.as_ref().unwrap(),
                max.as_ref().unwrap(),
            )),
            _ => None,
        };
        let linear_input = clipped_input
            .as_ref()
            .map_or(x, |arr| arr.as_ref().unwrap());
        let output = self.linear.forward(linear_input);
        match (&self.output_min, &self.output_max) {
            (Some(min), Some(max)) => {
                mlxcel_core::clip(&output, min.as_ref().unwrap(), max.as_ref().unwrap())
            }
            _ => output,
        }
    }
}

struct VisionMLP {
    gate_proj: ClippableLinear,
    up_proj: ClippableLinear,
    down_proj: ClippableLinear,
}

impl VisionMLP {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma4VisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: ClippableLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                config.use_clipped_linears,
                group_size,
                bits,
            )?,
            up_proj: ClippableLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                config.use_clipped_linears,
                group_size,
                bits,
            )?,
            down_proj: ClippableLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                config.use_clipped_linears,
                group_size,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::gelu_approx(&gate);
        self.down_proj
            .forward(&mlxcel_core::multiply(&activated, &up))
    }
}

struct VisionAttention {
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    o_proj: ClippableLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    v_norm: crate::models::gemma4::RMSNormNoScale,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
}

impl VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma4VisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            q_proj: ClippableLinear::from_weights(
                weights,
                &format!("{}.q_proj", prefix),
                config.use_clipped_linears,
                group_size,
                bits,
            )?,
            k_proj: ClippableLinear::from_weights(
                weights,
                &format!("{}.k_proj", prefix),
                config.use_clipped_linears,
                group_size,
                bits,
            )?,
            v_proj: ClippableLinear::from_weights(
                weights,
                &format!("{}.v_proj", prefix),
                config.use_clipped_linears,
                group_size,
                bits,
            )?,
            o_proj: ClippableLinear::from_weights(
                weights,
                &format!("{}.o_proj", prefix),
                config.use_clipped_linears,
                group_size,
                bits,
            )?,
            q_norm: RMSNorm::new(
                copy_weight(weights, &format!("{}.q_norm.weight", prefix))?,
                config.rms_norm_eps(),
            ),
            k_norm: RMSNorm::new(
                copy_weight(weights, &format!("{}.k_norm.weight", prefix))?,
                config.rms_norm_eps(),
            ),
            v_norm: crate::models::gemma4::RMSNormNoScale::new(
                config.head_dim as i32,
                config.rms_norm_eps(),
            ),
            num_heads: config.num_attention_heads as i32,
            num_kv_heads: config.num_key_value_heads as i32,
            head_dim: config.head_dim as i32,
        })
    }

    fn forward(&self, x: &MlxArray, rope: &Gemma4VisionRope) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let seq_len = shape[1];

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[batch, seq_len, self.num_kv_heads, self.head_dim]);

        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);
        let v = self.v_norm.forward(&v);

        let q = apply_multidimensional_rope(&q, rope);
        let k = apply_multidimensional_rope(&k, rope);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let attn = unsafe {
            mlxcel_core::layers::attention_from_ptr(&q, &k, &v, 1.0, std::ptr::null(), 0.0, 0)
        };
        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[batch, seq_len, self.num_heads * self.head_dim]);
        self.o_proj.forward(&attn)
    }
}

struct VisionTransformerBlock {
    self_attn: VisionAttention,
    mlp: VisionMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    pre_feedforward_layernorm: RMSNorm,
    post_feedforward_layernorm: RMSNorm,
}

impl VisionTransformerBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma4VisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            self_attn: VisionAttention::from_weights(
                weights,
                &format!("{}.self_attn", prefix),
                config,
                group_size,
                bits,
            )?,
            mlp: VisionMLP::from_weights(
                weights,
                &format!("{}.mlp", prefix),
                config,
                group_size,
                bits,
            )?,
            input_layernorm: RMSNorm::new(
                copy_weight(weights, &format!("{}.input_layernorm.weight", prefix))?,
                config.rms_norm_eps(),
            ),
            post_attention_layernorm: RMSNorm::new(
                copy_weight(
                    weights,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                config.rms_norm_eps(),
            ),
            pre_feedforward_layernorm: RMSNorm::new(
                copy_weight(
                    weights,
                    &format!("{}.pre_feedforward_layernorm.weight", prefix),
                )?,
                config.rms_norm_eps(),
            ),
            post_feedforward_layernorm: RMSNorm::new(
                copy_weight(
                    weights,
                    &format!("{}.post_feedforward_layernorm.weight", prefix),
                )?,
                config.rms_norm_eps(),
            ),
        })
    }

    fn forward(&self, x: &MlxArray, rope: &Gemma4VisionRope) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn = self.self_attn.forward(&normed, rope);
        let attn = self.post_attention_layernorm.forward(&attn);
        let h = mlxcel_core::add(x, &attn);

        let normed_h = self.pre_feedforward_layernorm.forward(&h);
        let ffw = self.mlp.forward(&normed_h);
        let ffw = self.post_feedforward_layernorm.forward(&ffw);
        mlxcel_core::add(&h, &ffw)
    }
}

struct VisionTransformerModel {
    layers: Vec<VisionTransformerBlock>,
}

impl VisionTransformerModel {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma4VisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(VisionTransformerBlock::from_weights(
                weights,
                &format!("{}.layers.{}", prefix, layer_idx),
                config,
                group_size,
                bits,
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, hidden_states: &MlxArray, rope: &Gemma4VisionRope) -> UniquePtr<MlxArray> {
        let mut hidden = mlxcel_core::copy(hidden_states);
        for layer in &self.layers {
            hidden = layer.forward(&hidden, rope);
        }
        hidden
    }
}

struct VisionPatchEmbedder {
    input_proj: UnifiedLinear,
    position_embedding_x: UniquePtr<MlxArray>, // [pos_size, hidden]
    position_embedding_y: UniquePtr<MlxArray>, // [pos_size, hidden]
    patch_size: usize,
}

impl VisionPatchEmbedder {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma4VisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let table = copy_weight(weights, &format!("{}.position_embedding_table", prefix))?;
        let shape = mlxcel_core::array_shape(&table);
        let pos_size = shape[1];
        let hidden_size = shape[2];

        let x_table = mlxcel_core::slice(&table, &[0, 0, 0], &[1, pos_size, hidden_size]);
        let y_table = mlxcel_core::slice(&table, &[1, 0, 0], &[2, pos_size, hidden_size]);
        let x_table = mlxcel_core::squeeze_axis(&x_table, 0);
        let y_table = mlxcel_core::squeeze_axis(&y_table, 0);

        Ok(Self {
            input_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.input_proj", prefix),
                group_size,
                bits,
            )?,
            position_embedding_x: x_table,
            position_embedding_y: y_table,
            patch_size: config.patch_size,
        })
    }

    fn patchify(&self, pixel_values: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(pixel_values);
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let patch = self.patch_size as i32;
        let patch_h = height / patch;
        let patch_w = width / patch;

        let patches = mlxcel_core::reshape(
            pixel_values,
            &[batch, channels, patch_h, patch, patch_w, patch],
        );
        let patches = mlxcel_core::transpose_axes(&patches, &[0, 2, 4, 3, 5, 1]);
        let patches = mlxcel_core::reshape(
            &patches,
            &[batch, patch_h * patch_w, channels * patch * patch],
        );
        let half = mlxcel_core::full_f32(&[1], 0.5, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::multiply_scalar(&mlxcel_core::subtract(&patches, &half), 2.0)
    }

    fn forward(
        &self,
        pixel_values: &MlxArray,
        patch_x: &MlxArray,
        patch_y: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let hidden_states = self.patchify(pixel_values);
        let shape = mlxcel_core::array_shape(&hidden_states);
        let batch = shape[0];
        let seq_len = shape[1];
        // After patchify, last dim is patch_size² * channels (768 for 16×16×3);
        // only after input_proj does it become config.hidden_size, which is what
        // position_embedding_{x,y} are shaped for. Reading hidden_size from the
        // post-projection tensor fixes 26B/31B (vision.hidden_size=1152) where
        // the two values diverge. e2b/e4b happen to have both equal 768.
        let hidden_states = self.input_proj.forward(&hidden_states);
        let hidden_size = mlxcel_core::array_shape(&hidden_states)[2];
        let pos_x = take_2d_embedding(
            &self.position_embedding_x,
            patch_x,
            batch,
            seq_len,
            hidden_size,
        );
        let pos_y = take_2d_embedding(
            &self.position_embedding_y,
            patch_y,
            batch,
            seq_len,
            hidden_size,
        );
        let pos = mlxcel_core::add(&pos_x, &pos_y);
        mlxcel_core::add(&hidden_states, &pos)
    }
}

pub struct Gemma4VisionModel {
    config: Gemma4VisionConfig,
    patch_embedder: VisionPatchEmbedder,
    encoder: VisionTransformerModel,
}

impl Gemma4VisionModel {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma4VisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            config: config.clone(),
            patch_embedder: VisionPatchEmbedder::from_weights(
                weights,
                &format!("{}.patch_embedder", prefix),
                config,
                group_size,
                bits,
            )?,
            encoder: VisionTransformerModel::from_weights(
                weights,
                &format!("{}.encoder", prefix),
                config,
                group_size,
                bits,
            )?,
        })
    }

    pub fn forward(
        &self,
        pixel_values: &MlxArray,
        patch_grid: (usize, usize),
    ) -> UniquePtr<MlxArray> {
        let (patch_h, patch_w) = patch_grid;
        let (patch_x, patch_y) = build_patch_position_ids(patch_h, patch_w);
        let rope = Gemma4VisionRope::new(
            patch_h,
            patch_w,
            self.config.head_dim,
            self.config.rope_theta(),
        );

        let hidden = self.patch_embedder.forward(
            pixel_values,
            patch_x.as_ref().unwrap(),
            patch_y.as_ref().unwrap(),
        );
        let hidden = self.encoder.forward(&hidden, &rope);

        let shape = mlxcel_core::array_shape(&hidden);
        let batch = shape[0];
        let hidden_size = shape[2];
        let pool = self.config.pooling_kernel_size as i32;
        let pooled_h = patch_h as i32 / pool;
        let pooled_w = patch_w as i32 / pool;

        let hidden = mlxcel_core::reshape(
            &hidden,
            &[batch, patch_h as i32, patch_w as i32, hidden_size],
        );
        let hidden = mlxcel_core::reshape(
            &hidden,
            &[batch, pooled_h, pool, pooled_w, pool, hidden_size],
        );
        let hidden = mlxcel_core::transpose_axes(&hidden, &[0, 1, 3, 2, 4, 5]);
        let hidden = mlxcel_core::mean_axis(&hidden, 4, false);
        let hidden = mlxcel_core::mean_axis(&hidden, 3, false);
        let hidden = mlxcel_core::reshape(&hidden, &[batch, pooled_h * pooled_w, hidden_size]);
        let hidden = mlxcel_core::multiply_scalar(&hidden, (hidden_size as f32).sqrt());

        if self.config.standardize {
            let zeros = mlxcel_core::zeros(&[hidden_size], mlxcel_core::array_dtype(&hidden));
            let ones = mlxcel_core::ones(&[hidden_size], mlxcel_core::array_dtype(&hidden));
            mlxcel_core::multiply(&mlxcel_core::subtract(&hidden, &zeros), &ones)
        } else {
            hidden
        }
    }
}
