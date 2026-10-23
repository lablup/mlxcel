// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! RF-DETR query decoder.

use std::f32::consts::PI;

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::RfDetrConfig;
use super::deformable::DeformableAttention;
use super::layers::{Activation, Mlp, concat, load_norm};

struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i32,
    head_dim: i32,
}

impl SelfAttention {
    fn from_weights(weights: &WeightMap, prefix: &str, heads: usize) -> Result<Self, String> {
        Ok(Self {
            query: Linear::from_weights(weights, &format!("{prefix}.q_proj"))?,
            key: Linear::from_weights(weights, &format!("{prefix}.k_proj"))?,
            value: Linear::from_weights(weights, &format!("{prefix}.v_proj"))?,
            output: Linear::from_weights(weights, &format!("{prefix}.out_proj"))?,
            heads: heads as i32,
            head_dim: 256 / heads as i32,
        })
    }

    fn forward(&self, input: &MlxArray, position: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input);
        let (batch, queries) = (shape[0], shape[1]);
        let query_key = mlxcel_core::add(input, position);
        let split = |linear: &Linear, source: &MlxArray| {
            let projected = mlxcel_core::reshape(
                &linear.forward(source),
                &[batch, queries, self.heads, self.head_dim],
            );
            mlxcel_core::transpose_axes(&projected, &[0, 2, 1, 3])
        };
        let query = split(&self.query, &query_key);
        let key = split(&self.key, &query_key);
        let value = split(&self.value, input);
        let key = mlxcel_core::transpose_axes(&key, &[0, 1, 3, 2]);
        let scores = mlxcel_core::multiply_scalar(
            &mlxcel_core::matmul(&query, &key),
            (self.head_dim as f32).powf(-0.5),
        );
        let attention = mlxcel_core::softmax(&scores, -1);
        let output = mlxcel_core::matmul(&attention, &value);
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        self.output
            .forward(&mlxcel_core::reshape(&output, &[batch, queries, 256]))
    }
}

struct DecoderLayer {
    self_attention: SelfAttention,
    norm1: LayerNorm,
    cross_attention: DeformableAttention,
    norm2: LayerNorm,
    first: Linear,
    second: Linear,
    norm3: LayerNorm,
}

impl DecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        index: usize,
        config: &RfDetrConfig,
    ) -> Result<Self, String> {
        let prefix = format!("transformer.decoder.layers.{index}");
        Ok(Self {
            self_attention: SelfAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config.sa_nheads,
            )?,
            norm1: load_norm(weights, &format!("{prefix}.norm1"), 1.0e-5)?,
            cross_attention: DeformableAttention::from_weights(
                weights,
                &format!("{prefix}.cross_attn"),
                config.ca_nheads,
                config.dec_n_points,
            )?,
            norm2: load_norm(weights, &format!("{prefix}.norm2"), 1.0e-5)?,
            first: Linear::from_weights(weights, &format!("{prefix}.linear1"))?,
            second: Linear::from_weights(weights, &format!("{prefix}.linear2"))?,
            norm3: load_norm(weights, &format!("{prefix}.norm3"), 1.0e-5)?,
        })
    }

    fn forward(
        &self,
        input: &MlxArray,
        memory: &MlxArray,
        reference: &MlxArray,
        spatial: (i32, i32),
        position: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let input = self.norm1.forward(&mlxcel_core::add(
            input,
            &self.self_attention.forward(input, position),
        ));
        let query = mlxcel_core::add(&input, position);
        let input = self.norm2.forward(&mlxcel_core::add(
            &input,
            &self
                .cross_attention
                .forward(&query, reference, memory, spatial),
        ));
        let ffn = self
            .second
            .forward(&mlxcel_core::relu(&self.first.forward(&input)));
        self.norm3.forward(&mlxcel_core::add(&input, &ffn))
    }
}

pub struct Decoder {
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    refpoint_head: Mlp,
}

impl Decoder {
    pub fn from_weights(weights: &WeightMap, config: &RfDetrConfig) -> Result<Self, String> {
        Ok(Self {
            layers: (0..config.dec_layers)
                .map(|index| DecoderLayer::from_weights(weights, index, config))
                .collect::<Result<_, _>>()?,
            norm: load_norm(weights, "transformer.decoder.norm", 1.0e-5)?,
            refpoint_head: Mlp::from_weights(
                weights,
                "transformer.decoder.ref_point_head",
                2,
                Activation::Relu,
            )?,
        })
    }

    pub fn forward(
        &self,
        target: &MlxArray,
        memory: &MlxArray,
        reference: &MlxArray,
        spatial: (i32, i32),
    ) -> UniquePtr<MlxArray> {
        let position = self.refpoint_head.forward(&sine_embedding(reference));
        let reference_level = mlxcel_core::expand_dims(reference, 2);
        let mut output = mlxcel_core::copy(target);
        for layer in &self.layers {
            output = layer.forward(&output, memory, &reference_level, spatial, &position);
        }
        self.norm.forward(&output)
    }
}

pub(crate) fn sine_embedding(position: &MlxArray) -> UniquePtr<MlxArray> {
    const DIM: i32 = 128;
    let dtype = mlxcel_core::dtype::FLOAT32;
    let dim = mlxcel_core::arange_f32(0.0, DIM as f32, 1.0);
    let halves = mlxcel_core::floor_divide(&dim, &mlxcel_core::full_f32(&[1], 2.0, dtype));
    let exponent =
        mlxcel_core::divide_scalar(&mlxcel_core::multiply_scalar(&halves, 2.0), DIM as f32);
    let frequencies = mlxcel_core::power(&mlxcel_core::full_f32(&[1], 10_000.0, dtype), &exponent);
    let shape = mlxcel_core::array_shape(position);
    let (batch, queries, coords) = (shape[0], shape[1], shape[2]);
    let even: Vec<i32> = (0..DIM).step_by(2).collect();
    let odd: Vec<i32> = (1..DIM).step_by(2).collect();
    let even = mlxcel_core::from_slice_i32(&even, &[DIM / 2]);
    let odd = mlxcel_core::from_slice_i32(&odd, &[DIM / 2]);
    let mut embedded = Vec::with_capacity(coords as usize);
    for index in 0..coords {
        let coordinate = mlxcel_core::slice(position, &[0, 0, index], &[batch, queries, index + 1]);
        let coordinate = mlxcel_core::multiply_scalar(&coordinate, 2.0 * PI);
        let values = mlxcel_core::divide(&coordinate, &frequencies);
        let sin = mlxcel_core::sin(&mlxcel_core::take(&values, &even, -1));
        let cos = mlxcel_core::cos(&mlxcel_core::take(&values, &odd, -1));
        let pair = mlxcel_core::stack_owned(&[sin, cos], -1);
        embedded.push(mlxcel_core::reshape(&pair, &[batch, queries, DIM]));
    }
    // Upstream coordinate order is y, x, then w, h.
    if embedded.len() == 4 {
        concat(
            &[
                mlxcel_core::copy(&embedded[1]),
                mlxcel_core::copy(&embedded[0]),
                mlxcel_core::copy(&embedded[2]),
                mlxcel_core::copy(&embedded[3]),
            ],
            -1,
        )
    } else {
        concat(&embedded, -1)
    }
}
