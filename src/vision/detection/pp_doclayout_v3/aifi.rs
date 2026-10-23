// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Attention-based intra-scale feature interaction (AIFI).

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

#[cfg(test)]
use crate::vision::detection::rt_detr_v2::hybrid_encoder::sine_position_embedding as host_sine_position_embedding;
use crate::vision::detection::rt_detr_v2::hybrid_encoder::{SelfAttention, load_layer_norm};

use super::config::PpDocLayoutV3Config;
use super::layers::Activation;

struct AifiLayer {
    attention: SelfAttention,
    attention_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_norm: LayerNorm,
    activation: Activation,
}

impl AifiLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &PpDocLayoutV3Config,
    ) -> Result<Self, String> {
        Ok(Self {
            attention: SelfAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config.encoder_hidden_dim,
                config.encoder_attention_heads,
            )?,
            attention_norm: load_layer_norm(
                weights,
                &format!("{prefix}.self_attn_layer_norm"),
                config.layer_norm_eps,
            )?,
            fc1: Linear::from_weights(weights, &format!("{prefix}.fc1"))?,
            fc2: Linear::from_weights(weights, &format!("{prefix}.fc2"))?,
            final_norm: load_layer_norm(
                weights,
                &format!("{prefix}.final_layer_norm"),
                config.layer_norm_eps,
            )?,
            activation: Activation::parse(&config.encoder_activation_function)?,
        })
    }

    fn forward(&self, x: &MlxArray, position: &MlxArray) -> UniquePtr<MlxArray> {
        let attended = self.attention.forward(x, Some(position));
        let attended = self.attention_norm.forward(&mlxcel_core::add(x, &attended));
        let hidden = self
            .fc2
            .forward(&self.activation.apply(&self.fc1.forward(&attended)));
        self.final_norm
            .forward(&mlxcel_core::add(&attended, &hidden))
    }
}

pub struct Aifi {
    layers: Vec<AifiLayer>,
    hidden_dim: usize,
    #[cfg(test)]
    attention_heads: usize,
    temperature: f32,
}

impl Aifi {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &PpDocLayoutV3Config,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for index in 0..config.encoder_layers {
            layers.push(AifiLayer::from_weights(
                weights,
                &format!("{prefix}.layers.{index}"),
                config,
            )?);
        }
        Ok(Self {
            layers,
            hidden_dim: config.encoder_hidden_dim,
            #[cfg(test)]
            attention_heads: config.encoder_attention_heads,
            temperature: config.positional_encoding_temperature,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
        #[cfg(test)]
        super::diagnostics::capture("aifi.input", x);
        let mut sequence = mlxcel_core::reshape(x, &[batch, height * width, channels]);
        #[cfg(test)]
        super::diagnostics::capture("aifi.sequence", &sequence);
        let position =
            upstream_sine_position_embedding(height, width, self.hidden_dim, self.temperature);
        let position = mlxcel_core::astype(&position, mlxcel_core::array_dtype(&sequence));
        #[cfg(test)]
        {
            let host_position = mlxcel_core::astype(
                &host_sine_position_embedding(height, width, self.hidden_dim, self.temperature),
                mlxcel_core::array_dtype(&sequence),
            );
            super::diagnostics::capture("aifi.position.host", &host_position);
            super::diagnostics::capture("aifi.position.upstream", &position);
        }
        for (index, layer) in self.layers.iter().enumerate() {
            #[cfg(test)]
            if index == 0 {
                let [query, key, value] =
                    diagnostic_qkv(&layer.attention, &sequence, &position, self.attention_heads);
                super::diagnostics::capture("aifi.layer0.q", &query);
                super::diagnostics::capture("aifi.layer0.k", &key);
                super::diagnostics::capture("aifi.layer0.v", &value);
            }
            sequence = layer.forward(&sequence, &position);
            #[cfg(test)]
            super::diagnostics::capture(&format!("aifi.layer{index}.output"), &sequence);
        }
        let output = mlxcel_core::reshape(&sequence, &[batch, height, width, channels]);
        #[cfg(test)]
        super::diagnostics::capture("aifi.output", &output);
        output
    }
}

#[cfg(test)]
fn diagnostic_qkv(
    attention: &SelfAttention,
    sequence: &MlxArray,
    position: &MlxArray,
    attention_heads: usize,
) -> [UniquePtr<MlxArray>; 3] {
    let shape = mlxcel_core::array_shape(sequence);
    let (batch, length, hidden) = (shape[0], shape[1], shape[2]);
    let heads = attention_heads as i32;
    let head_dim = hidden / heads;
    let qk = mlxcel_core::add(sequence, position);
    let split = |projection: &Linear, source: &MlxArray| {
        let projected = projection.forward(source);
        let projected = mlxcel_core::reshape(&projected, &[batch, length, heads, head_dim]);
        mlxcel_core::transpose_axes(&projected, &[0, 2, 1, 3])
    };
    [
        split(&attention.q_proj, &qk),
        split(&attention.k_proj, &qk),
        split(&attention.v_proj, sequence),
    ]
}

/// Device-constructed form of the pinned upstream `SinePE` graph.
fn upstream_sine_position_embedding(
    height: i32,
    width: i32,
    embed_dim: usize,
    temperature: f32,
) -> UniquePtr<MlxArray> {
    let position_dim = (embed_dim / 4) as i32;
    let steps = mlxcel_core::arange_f32(0.0, position_dim as f32, 1.0);
    let exponent = mlxcel_core::divide_scalar(&steps, position_dim as f32);
    let base = mlxcel_core::full_f32(&[1], temperature, mlxcel_core::dtype::FLOAT32);
    let omega = mlxcel_core::reciprocal(&mlxcel_core::power(&base, &exponent));
    let omega = mlxcel_core::reshape(&omega, &[1, position_dim]);

    let rows = mlxcel_core::reshape(
        &mlxcel_core::arange_f32(0.0, height as f32, 1.0),
        &[height, 1],
    );
    let rows = mlxcel_core::broadcast_to(&rows, &[height, width]);
    let rows = mlxcel_core::reshape(&rows, &[height * width, 1]);
    let columns = mlxcel_core::reshape(
        &mlxcel_core::arange_f32(0.0, width as f32, 1.0),
        &[1, width],
    );
    let columns = mlxcel_core::broadcast_to(&columns, &[height, width]);
    let columns = mlxcel_core::reshape(&columns, &[height * width, 1]);
    let row_angles = mlxcel_core::multiply(&rows, &omega);
    let column_angles = mlxcel_core::multiply(&columns, &omega);
    let mut position = mlxcel_core::concatenate(
        &mlxcel_core::sin(&row_angles),
        &mlxcel_core::cos(&row_angles),
        1,
    );
    position = mlxcel_core::concatenate(&position, &mlxcel_core::sin(&column_angles), 1);
    position = mlxcel_core::concatenate(&position, &mlxcel_core::cos(&column_angles), 1);
    mlxcel_core::expand_dims(&position, 0)
}
