// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! AnimateDiff-style temporal motion blocks used by the DPT head.

use anyhow::{Result, ensure};
use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::weights;

struct TemporalAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    output: Linear,
    position: UniquePtr<MlxArray>,
    heads: usize,
    channels: usize,
}

impl TemporalAttention {
    fn from_weights(map: &WeightMap, prefix: &str, channels: usize, heads: usize) -> Result<Self> {
        ensure!(
            channels.is_multiple_of(heads),
            "VDA temporal channels must be divisible by heads"
        );
        Ok(Self {
            q: weights::linear(map, &format!("{prefix}.to_q"), channels, channels, false)?,
            k: weights::linear(map, &format!("{prefix}.to_k"), channels, channels, false)?,
            v: weights::linear(map, &format!("{prefix}.to_v"), channels, channels, false)?,
            output: weights::linear(map, &format!("{prefix}.to_out.0"), channels, channels, true)?,
            position: weights::array_exact(
                map,
                &format!("{prefix}.pos_encoder.pe"),
                &[1, 32, channels as i32],
            )?,
            heads,
            channels,
        })
    }

    fn forward(&self, hidden: &MlxArray, video_length: usize) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden);
        let bt = shape[0];
        let positions = shape[1];
        let batches = bt / video_length as i32;
        let x = mlxcel_core::reshape(
            hidden,
            &[
                batches,
                video_length as i32,
                positions,
                self.channels as i32,
            ],
        );
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
        let x = mlxcel_core::reshape(
            &x,
            &[
                batches * positions,
                video_length as i32,
                self.channels as i32,
            ],
        );
        let pe = mlxcel_core::slice(
            &self.position,
            &[0, 0, 0],
            &[1, video_length as i32, self.channels as i32],
        );
        let pe = mlxcel_core::astype(&pe, mlxcel_core::array_dtype(&x));
        let x = mlxcel_core::add(&x, &pe);
        let head_dim = (self.channels / self.heads) as i32;
        let split = |value: UniquePtr<MlxArray>| {
            let value = mlxcel_core::reshape(
                &value,
                &[
                    batches * positions,
                    video_length as i32,
                    self.heads as i32,
                    head_dim,
                ],
            );
            mlxcel_core::transpose_axes(&value, &[0, 2, 1, 3])
        };
        let q = split(self.q.forward(&x));
        let k = split(self.k.forward(&x));
        let v = split(self.v.forward(&x));
        let output =
            mlxcel_core::layers::attention(&q, &k, &v, (head_dim as f32).powf(-0.5), None, 0.0, 0);
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(
            &output,
            &[
                batches * positions,
                video_length as i32,
                self.channels as i32,
            ],
        );
        let output = self.output.forward(&output);
        let output = mlxcel_core::reshape(
            &output,
            &[
                batches,
                positions,
                video_length as i32,
                self.channels as i32,
            ],
        );
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        mlxcel_core::reshape(&output, &[bt, positions, self.channels as i32])
    }
}

struct FeedForward {
    projection: Linear,
    output: Linear,
    channels: usize,
}

impl FeedForward {
    fn from_weights(map: &WeightMap, prefix: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            projection: weights::linear(
                map,
                &format!("{prefix}.net.0.proj"),
                channels,
                channels * 8,
                true,
            )?,
            output: weights::linear(
                map,
                &format!("{prefix}.net.2"),
                channels * 4,
                channels,
                true,
            )?,
            channels,
        })
    }
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let projected = self.projection.forward(x);
        let shape = mlxcel_core::array_shape(&projected);
        let half = (shape[shape.len() - 1] / 2) as i32;
        let starts = vec![0; shape.len()];
        let mut middle = shape.clone();
        middle[shape.len() - 1] = half;
        let first = mlxcel_core::slice(&projected, &starts, &middle);
        let mut gate_start = starts;
        gate_start[shape.len() - 1] = half;
        let mut gate_end = shape;
        let last_axis = gate_end.len() - 1;
        gate_end[last_axis] = (self.channels * 8) as i32;
        let gate = mlxcel_core::slice(&projected, &gate_start, &gate_end);
        self.output
            .forward(&mlxcel_core::multiply(&first, &mlxcel_core::gelu(&gate)))
    }
}

struct TransformerBlock {
    attentions: Vec<TemporalAttention>,
    norms: Vec<LayerNorm>,
    feed_forward: FeedForward,
    ff_norm: LayerNorm,
}

impl TransformerBlock {
    fn from_weights(
        map: &WeightMap,
        prefix: &str,
        channels: usize,
        heads: usize,
        attention_blocks: usize,
    ) -> Result<Self> {
        let mut attentions = Vec::with_capacity(attention_blocks);
        let mut norms = Vec::with_capacity(attention_blocks);
        for index in 0..attention_blocks {
            attentions.push(TemporalAttention::from_weights(
                map,
                &format!("{prefix}.attention_blocks.{index}"),
                channels,
                heads,
            )?);
            norms.push(weights::layer_norm(
                map,
                &format!("{prefix}.norms.{index}"),
                channels,
                1e-5,
            )?);
        }
        Ok(Self {
            attentions,
            norms,
            feed_forward: FeedForward::from_weights(map, &format!("{prefix}.ff"), channels)?,
            ff_norm: weights::layer_norm(map, &format!("{prefix}.ff_norm"), channels, 1e-5)?,
        })
    }
    fn forward(&self, hidden: &MlxArray, video_length: usize) -> UniquePtr<MlxArray> {
        let mut x = mlxcel_core::copy(hidden);
        for (attention, norm) in self.attentions.iter().zip(&self.norms) {
            x = mlxcel_core::add(&x, &attention.forward(&norm.forward(&x), video_length));
        }
        mlxcel_core::add(&x, &self.feed_forward.forward(&self.ff_norm.forward(&x)))
    }
}

struct GroupNorm {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    groups: i32,
}

impl GroupNorm {
    fn from_weights(map: &WeightMap, prefix: &str, channels: usize, groups: usize) -> Result<Self> {
        Ok(Self {
            weight: weights::array_exact(map, &format!("{prefix}.weight"), &[channels as i32])?,
            bias: weights::array_exact(map, &format!("{prefix}.bias"), &[channels as i32])?,
            groups: groups as i32,
        })
    }
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let dtype = mlxcel_core::array_dtype(x);
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let channels = shape[3];
        let spatial = shape[1] * shape[2];
        let x = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
        let x = mlxcel_core::reshape(&x, &[batch, spatial, self.groups, channels / self.groups]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
        let x = mlxcel_core::reshape(&x, &[batch, self.groups, spatial * channels / self.groups]);
        let x =
            unsafe { mlxcel_core::fast_layer_norm(&x, std::ptr::null(), std::ptr::null(), 1e-6) };
        let x = mlxcel_core::reshape(&x, &[batch, self.groups, spatial, channels / self.groups]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
        let x = mlxcel_core::reshape(&x, &shape);
        mlxcel_core::astype(
            &mlxcel_core::add(&mlxcel_core::multiply(&x, &self.weight), &self.bias),
            dtype,
        )
    }
}

pub struct TemporalModule {
    norm: GroupNorm,
    input: Linear,
    blocks: Vec<TransformerBlock>,
    output: Linear,
    channels: usize,
}

impl TemporalModule {
    pub fn from_weights(
        map: &WeightMap,
        prefix: &str,
        channels: usize,
        heads: usize,
        groups: usize,
        layers: usize,
        attention_blocks: usize,
    ) -> Result<Self> {
        let root = format!("{prefix}.temporal_transformer");
        let mut blocks = Vec::with_capacity(layers);
        for index in 0..layers {
            blocks.push(TransformerBlock::from_weights(
                map,
                &format!("{root}.transformer_blocks.{index}"),
                channels,
                heads,
                attention_blocks,
            )?);
        }
        Ok(Self {
            norm: GroupNorm::from_weights(map, &format!("{root}.norm"), channels, groups)?,
            input: weights::linear(map, &format!("{root}.proj_in"), channels, channels, true)?,
            blocks,
            output: weights::linear(map, &format!("{root}.proj_out"), channels, channels, true)?,
            channels,
        })
    }

    pub fn forward(&self, x: &MlxArray, video_length: usize) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let height = shape[2];
        let width = shape[3];
        let flat = mlxcel_core::reshape(
            x,
            &[
                batch * video_length as i32,
                height,
                width,
                self.channels as i32,
            ],
        );
        let residual = mlxcel_core::copy(&flat);
        let normalized = self.norm.forward(&flat);
        let mut hidden = self.input.forward(&mlxcel_core::reshape(
            &normalized,
            &[
                batch * video_length as i32,
                height * width,
                self.channels as i32,
            ],
        ));
        for block in &self.blocks {
            hidden = block.forward(&hidden, video_length);
        }
        let hidden = self.output.forward(&hidden);
        let hidden = mlxcel_core::reshape(
            &hidden,
            &[
                batch * video_length as i32,
                height,
                width,
                self.channels as i32,
            ],
        );
        mlxcel_core::reshape(
            &mlxcel_core::add(&hidden, &residual),
            &[
                batch,
                video_length as i32,
                height,
                width,
                self.channels as i32,
            ],
        )
    }
}
