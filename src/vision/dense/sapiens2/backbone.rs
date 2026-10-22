// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Sapiens2 ViT backbone: 2D RoPE, QK RMSNorm, layer-wise GQA and SwiGLU.

use anyhow::{Result, ensure};
use mlxcel_core::layers::{Linear, RMSNorm};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::Sapiens2Config;
use super::weights;

pub struct BackboneOutput {
    pub tokens: UniquePtr<MlxArray>,
    pub height: usize,
    pub width: usize,
}

struct Attention {
    qkv: Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    projection: Linear,
    gamma: UniquePtr<MlxArray>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn from_weights(
        weights_map: &WeightMap,
        prefix: &str,
        config: &Sapiens2Config,
        layer: usize,
    ) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(Self {
            qkv: weights::linear(weights_map, &format!("{prefix}.wqkv"))?,
            q_norm: weights::rms_norm(
                weights_map,
                &format!("{prefix}.q_norm"),
                config.rms_norm_eps,
            )?,
            k_norm: weights::rms_norm(
                weights_map,
                &format!("{prefix}.k_norm"),
                config.rms_norm_eps,
            )?,
            projection: weights::linear(weights_map, &format!("{prefix}.proj"))?,
            gamma: weights::array(weights_map, &format!("{prefix}.gamma.weight"))?,
            heads: config.num_attention_heads,
            kv_heads: config.kv_heads_per_layer[layer],
            head_dim,
        })
    }

    fn forward(&self, x: &MlxArray, rope: &RopeTables) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (batch, tokens) = (shape[0], shape[1]);
        let hidden = (self.heads * self.head_dim) as i32;
        let kv_size = (self.kv_heads * self.head_dim) as i32;
        let qkv = self.qkv.forward(x);
        let q = mlxcel_core::slice(&qkv, &[0, 0, 0], &[batch, tokens, hidden]);
        let k = mlxcel_core::slice(&qkv, &[0, 0, hidden], &[batch, tokens, hidden + kv_size]);
        let v = mlxcel_core::slice(
            &qkv,
            &[0, 0, hidden + kv_size],
            &[batch, tokens, hidden + 2 * kv_size],
        );
        let q = mlxcel_core::transpose_axes(
            &mlxcel_core::reshape(
                &q,
                &[batch, tokens, self.heads as i32, self.head_dim as i32],
            ),
            &[0, 2, 1, 3],
        );
        let k = mlxcel_core::transpose_axes(
            &mlxcel_core::reshape(
                &k,
                &[batch, tokens, self.kv_heads as i32, self.head_dim as i32],
            ),
            &[0, 2, 1, 3],
        );
        let v = mlxcel_core::transpose_axes(
            &mlxcel_core::reshape(
                &v,
                &[batch, tokens, self.kv_heads as i32, self.head_dim as i32],
            ),
            &[0, 2, 1, 3],
        );
        let q = rope.apply(&self.q_norm.forward(&q));
        let k = rope.apply(&self.k_norm.forward(&k));
        // SAFETY: q/k/v are live rank-4 arrays and a null mask means unmasked
        // bidirectional attention. Native MLX SDPA broadcasts GQA KV heads.
        let attended = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(
                &q,
                &k,
                &v,
                (self.head_dim as f32).powf(-0.5),
                std::ptr::null(),
            )
        };
        let attended = mlxcel_core::transpose_axes(&attended, &[0, 2, 1, 3]);
        let attended = mlxcel_core::reshape(&attended, &[batch, tokens, hidden]);
        mlxcel_core::multiply(&self.projection.forward(&attended), &self.gamma)
    }
}

struct FeedForward {
    w12: Linear,
    w3: Linear,
    intermediate_size: usize,
}

impl FeedForward {
    fn from_weights(
        weights_map: &WeightMap,
        prefix: &str,
        intermediate_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            w12: weights::linear(weights_map, &format!("{prefix}.w12"))?,
            w3: weights::linear(weights_map, &format!("{prefix}.w3"))?,
            intermediate_size,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let projected = self.w12.forward(x);
        let shape = mlxcel_core::array_shape(&projected);
        let starts = vec![0; shape.len()];
        let mut middle = shape.clone();
        middle[shape.len() - 1] = self.intermediate_size as i32;
        let first = mlxcel_core::slice(&projected, &starts, &middle);
        let mut second_start = starts;
        second_start[shape.len() - 1] = self.intermediate_size as i32;
        let second = mlxcel_core::slice(&projected, &second_start, &shape);
        self.w3
            .forward(&mlxcel_core::multiply(&mlxcel_core::silu(&first), &second))
    }
}

struct Block {
    norm1: RMSNorm,
    attention: Attention,
    norm2: RMSNorm,
    feed_forward: FeedForward,
}

impl Block {
    fn from_weights(
        weights_map: &WeightMap,
        index: usize,
        config: &Sapiens2Config,
    ) -> Result<Self> {
        let prefix = format!("backbone.blocks.{index}");
        Ok(Self {
            norm1: weights::rms_norm(weights_map, &format!("{prefix}.ln1"), config.rms_norm_eps)?,
            attention: Attention::from_weights(
                weights_map,
                &format!("{prefix}.attn"),
                config,
                index,
            )?,
            norm2: weights::rms_norm(weights_map, &format!("{prefix}.ln2"), config.rms_norm_eps)?,
            feed_forward: FeedForward::from_weights(
                weights_map,
                &format!("{prefix}.ffn"),
                config.intermediate_size,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray, rope: &RopeTables) -> UniquePtr<MlxArray> {
        let attended = self.attention.forward(&self.norm1.forward(x), rope);
        let hidden = mlxcel_core::add(x, &attended);
        mlxcel_core::add(
            &hidden,
            &self.feed_forward.forward(&self.norm2.forward(&hidden)),
        )
    }
}

struct RopeTables {
    sin: UniquePtr<MlxArray>,
    cos: UniquePtr<MlxArray>,
    head_dim: usize,
}

impl RopeTables {
    fn new(height: usize, width: usize, prefix: usize, head_dim: usize, periods: &[f32]) -> Self {
        let quarter = periods.len();
        let half = head_dim / 2;
        let tokens = prefix + height * width;
        let mut sin = vec![0.0f32; tokens * half];
        let mut cos = vec![1.0f32; tokens * half];
        for row in 0..height {
            let y = 2.0 * ((row as f32 + 0.5) / height as f32) - 1.0;
            for column in 0..width {
                let x = 2.0 * ((column as f32 + 0.5) / width as f32) - 1.0;
                let token = prefix + row * width + column;
                for (axis, coordinate) in [y, x].into_iter().enumerate() {
                    for (frequency, period) in periods.iter().enumerate() {
                        let angle = std::f32::consts::TAU * coordinate / period;
                        let index = token * half + axis * quarter + frequency;
                        sin[index] = angle.sin();
                        cos[index] = angle.cos();
                    }
                }
            }
        }
        Self {
            sin: mlxcel_core::from_slice_f32(&sin, &[1, 1, tokens as i32, half as i32]),
            cos: mlxcel_core::from_slice_f32(&cos, &[1, 1, tokens as i32, half as i32]),
            head_dim,
        }
    }

    fn apply(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let dtype = mlxcel_core::array_dtype(x);
        let shape = mlxcel_core::array_shape(x);
        let half = (self.head_dim / 2) as i32;
        let xf = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
        let first = mlxcel_core::slice(&xf, &[0, 0, 0, 0], &[shape[0], shape[1], shape[2], half]);
        let second = mlxcel_core::slice(
            &xf,
            &[0, 0, 0, half],
            &[shape[0], shape[1], shape[2], self.head_dim as i32],
        );
        let out_first = mlxcel_core::subtract(
            &mlxcel_core::multiply(&first, &self.cos),
            &mlxcel_core::multiply(&second, &self.sin),
        );
        let out_second = mlxcel_core::add(
            &mlxcel_core::multiply(&second, &self.cos),
            &mlxcel_core::multiply(&first, &self.sin),
        );
        mlxcel_core::astype(
            &mlxcel_core::concatenate(&out_first, &out_second, -1),
            dtype,
        )
    }
}

pub struct Sapiens2Backbone {
    patch_weight: UniquePtr<MlxArray>,
    patch_bias: UniquePtr<MlxArray>,
    cls_token: UniquePtr<MlxArray>,
    register_tokens: Option<UniquePtr<MlxArray>>,
    blocks: Vec<Block>,
    final_norm: Option<RMSNorm>,
    hidden_size: usize,
    patch_size: usize,
    num_prefix_tokens: usize,
    num_channels: usize,
    rope_periods: Vec<f32>,
    num_heads: usize,
}

impl Sapiens2Backbone {
    pub fn from_weights(weights_map: &WeightMap, config: &Sapiens2Config) -> Result<Self> {
        let patch_weight = weights::conv_weight(
            weights_map,
            "backbone.patch_embed.projection.weight",
            config.hidden_size,
            config.patch_size,
            config.num_channels,
        )?;
        let patch_bias = weights::array(weights_map, "backbone.patch_embed.projection.bias")?;
        let cls_token = weights::array(weights_map, "backbone.cls_token")?;
        let register_tokens = (config.num_register_tokens > 0)
            .then(|| weights::array(weights_map, "backbone.storage_tokens"))
            .transpose()?;
        let rope_periods = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
            &weights::array(weights_map, "backbone.rope_embed.periods")?,
            mlxcel_core::dtype::FLOAT32,
        ));
        ensure!(
            rope_periods.len() == config.hidden_size / config.num_attention_heads / 4
                && rope_periods
                    .iter()
                    .all(|period| period.is_finite() && *period > 0.0),
            "Sapiens2 checkpoint has invalid 2D RoPE periods"
        );
        let blocks = (0..config.num_hidden_layers)
            .map(|index| Block::from_weights(weights_map, index, config))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = config
            .normalize_backbone_outputs
            .then(|| weights::rms_norm(weights_map, "backbone.ln1", config.rms_norm_eps))
            .transpose()?;
        Ok(Self {
            patch_weight,
            patch_bias,
            cls_token,
            register_tokens,
            blocks,
            final_norm,
            hidden_size: config.hidden_size,
            patch_size: config.patch_size,
            num_prefix_tokens: 1 + config.num_register_tokens,
            num_channels: config.num_channels,
            rope_periods,
            num_heads: config.num_attention_heads,
        })
    }

    pub fn prefix_tokens(&self) -> usize {
        self.num_prefix_tokens
    }

    pub fn forward(&self, pixel_values: &MlxArray) -> Result<BackboneOutput> {
        let input_shape = mlxcel_core::array_shape(pixel_values);
        ensure!(
            input_shape.len() == 4 && input_shape[3] == self.num_channels as i32,
            "Sapiens2 backbone expects NHWC RGB input, got {input_shape:?}"
        );
        ensure!(
            input_shape[1] % self.patch_size as i32 == 0
                && input_shape[2] % self.patch_size as i32 == 0,
            "Sapiens2 input dimensions must be patch aligned"
        );
        let batch = input_shape[0];
        let input = mlxcel_core::astype(pixel_values, mlxcel_core::array_dtype(&self.patch_weight));
        let patches = mlxcel_core::add(
            &mlxcel_core::conv2d(
                &input,
                &self.patch_weight,
                self.patch_size as i32,
                self.patch_size as i32,
                0,
                0,
                1,
                1,
                1,
            ),
            &self.patch_bias,
        );
        let patch_shape = mlxcel_core::array_shape(&patches);
        let (height, width) = (patch_shape[1] as usize, patch_shape[2] as usize);
        let patches = mlxcel_core::reshape(
            &patches,
            &[batch, (height * width) as i32, self.hidden_size as i32],
        );
        let cls = mlxcel_core::broadcast_to(&self.cls_token, &[batch, 1, self.hidden_size as i32]);
        let mut hidden = if let Some(registers) = &self.register_tokens {
            let registers = mlxcel_core::broadcast_to(
                registers,
                &[
                    batch,
                    (self.num_prefix_tokens - 1) as i32,
                    self.hidden_size as i32,
                ],
            );
            let prefix = mlxcel_core::concatenate(&cls, &registers, 1);
            mlxcel_core::concatenate(&prefix, &patches, 1)
        } else {
            mlxcel_core::concatenate(&cls, &patches, 1)
        };
        let rope = RopeTables::new(
            height,
            width,
            self.num_prefix_tokens,
            self.hidden_size / self.num_heads,
            &self.rope_periods,
        );
        for block in &self.blocks {
            hidden = block.forward(&hidden, &rope);
        }
        if let Some(norm) = &self.final_norm {
            hidden = norm.forward(&hidden);
        }
        Ok(BackboneOutput {
            tokens: hidden,
            height,
            width,
        })
    }
}
