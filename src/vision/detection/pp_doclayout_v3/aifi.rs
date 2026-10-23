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

    fn forward(&self, x: &MlxArray, position: &MlxArray, _index: usize) -> UniquePtr<MlxArray> {
        let attention = super::attention::forward_fast(&self.attention, x, Some(position));
        #[cfg(test)]
        {
            super::diagnostics::capture(&format!("aifi.layer{_index}.q"), &attention.query);
            super::diagnostics::capture(&format!("aifi.layer{_index}.k"), &attention.key);
            super::diagnostics::capture(&format!("aifi.layer{_index}.v"), &attention.value);
            super::diagnostics::capture(
                &format!("aifi.layer{_index}.attention.sdpa"),
                &attention.sdpa,
            );
            super::diagnostics::capture(
                &format!("aifi.layer{_index}.attention.merged"),
                &attention.merged,
            );
        }
        let attended = attention.output;
        #[cfg(test)]
        super::diagnostics::capture(&format!("aifi.layer{_index}.attention.output"), &attended);
        let attention_residual = mlxcel_core::add(x, &attended);
        #[cfg(test)]
        super::diagnostics::capture(
            &format!("aifi.layer{_index}.attention.residual"),
            &attention_residual,
        );
        let attended = self.attention_norm.forward(&attention_residual);
        #[cfg(test)]
        super::diagnostics::capture(&format!("aifi.layer{_index}.attention.norm"), &attended);
        let fc1 = self.fc1.forward(&attended);
        #[cfg(test)]
        super::diagnostics::capture(&format!("aifi.layer{_index}.ffn.fc1"), &fc1);
        let activated = self.activation.apply(&fc1);
        #[cfg(test)]
        super::diagnostics::capture(&format!("aifi.layer{_index}.ffn.activation"), &activated);
        let hidden = self.fc2.forward(&activated);
        #[cfg(test)]
        super::diagnostics::capture(&format!("aifi.layer{_index}.ffn.fc2"), &hidden);
        let final_residual = mlxcel_core::add(&attended, &hidden);
        #[cfg(test)]
        super::diagnostics::capture(&format!("aifi.layer{_index}.ffn.residual"), &final_residual);
        self.final_norm.forward(&final_residual)
    }
}

pub struct Aifi {
    layers: Vec<AifiLayer>,
    hidden_dim: usize,
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
        for (_index, layer) in self.layers.iter().enumerate() {
            sequence = layer.forward(&sequence, &position, _index);
            #[cfg(test)]
            super::diagnostics::capture(&format!("aifi.layer{_index}.output"), &sequence);
        }
        let output = mlxcel_core::reshape(&sequence, &[batch, height, width, channels]);
        #[cfg(test)]
        super::diagnostics::capture("aifi.output", &output);
        output
    }
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
