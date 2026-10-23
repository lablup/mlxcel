// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Windowed DINOv2-small backbone used by released RF-DETR checkpoints.

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::dense::interpolate::resize_bicubic_nhwc;

use super::config::RfDetrConfig;
use super::layers::{Conv2d, concat, copy_weight, load_norm};

const HIDDEN: i32 = 384;
const HEADS: i32 = 6;
const HEAD_DIM: i32 = HIDDEN / HEADS;
const LAYERS: usize = 12;

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
}

impl Attention {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            query: Linear::from_weights(weights, &format!("{prefix}.q_proj"))?,
            key: Linear::from_weights(weights, &format!("{prefix}.k_proj"))?,
            value: Linear::from_weights(weights, &format!("{prefix}.v_proj"))?,
            output: Linear::from_weights(weights, &format!("{prefix}.o_proj"))?,
        })
    }

    fn forward(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input);
        let (batch, tokens) = (shape[0], shape[1]);
        let project = |linear: &Linear| {
            let projected =
                mlxcel_core::reshape(&linear.forward(input), &[batch, tokens, HEADS, HEAD_DIM]);
            mlxcel_core::transpose_axes(&projected, &[0, 2, 1, 3])
        };
        let query = project(&self.query);
        let key = project(&self.key);
        let value = project(&self.value);
        let attended = mlxcel_core::layers::attention(
            &query,
            &key,
            &value,
            (HEAD_DIM as f32).powf(-0.5),
            None,
            0.0,
            0,
        );
        let attended = mlxcel_core::transpose_axes(&attended, &[0, 2, 1, 3]);
        self.output
            .forward(&mlxcel_core::reshape(&attended, &[batch, tokens, HIDDEN]))
    }
}

struct Block {
    norm1: LayerNorm,
    attention: Attention,
    scale1: UniquePtr<MlxArray>,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    scale2: UniquePtr<MlxArray>,
}

impl Block {
    fn from_weights(weights: &WeightMap, index: usize) -> Result<Self, String> {
        let prefix = format!("backbone.encoder.layers.{index}");
        Ok(Self {
            norm1: load_norm(weights, &format!("{prefix}.norm1"), 1.0e-6)?,
            attention: Attention::from_weights(weights, &format!("{prefix}.attention"))?,
            scale1: copy_weight(weights, &format!("{prefix}.layer_scale1"))?,
            norm2: load_norm(weights, &format!("{prefix}.norm2"), 1.0e-6)?,
            fc1: Linear::from_weights(weights, &format!("{prefix}.mlp.fc1"))?,
            fc2: Linear::from_weights(weights, &format!("{prefix}.mlp.fc2"))?,
            scale2: copy_weight(weights, &format!("{prefix}.layer_scale2"))?,
        })
    }

    fn forward(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        let attention = self.attention.forward(&self.norm1.forward(input));
        let hidden = mlxcel_core::add(input, &mlxcel_core::multiply(&attention, &self.scale1));
        let mlp = self.fc2.forward(&mlxcel_core::gelu(
            &self.fc1.forward(&self.norm2.forward(&hidden)),
        ));
        mlxcel_core::add(&hidden, &mlxcel_core::multiply(&mlp, &self.scale2))
    }
}

pub struct WindowedDinov2 {
    patch: Conv2d,
    cls_token: UniquePtr<MlxArray>,
    position: UniquePtr<MlxArray>,
    blocks: Vec<Block>,
    final_norm: LayerNorm,
    patch_size: usize,
    windows: usize,
    feature_layers: Vec<usize>,
    global_layers: Vec<usize>,
}

impl WindowedDinov2 {
    pub fn from_weights(weights: &WeightMap, config: &RfDetrConfig) -> Result<Self, String> {
        Ok(Self {
            patch: Conv2d::from_weights(
                weights,
                "backbone.embeddings.patch_embeddings.projection",
                config.patch_size,
                0,
                1,
            )?,
            cls_token: copy_weight(weights, "backbone.embeddings.cls_token")?,
            position: copy_weight(weights, "backbone.embeddings.position_embeddings")?,
            blocks: (0..LAYERS)
                .map(|index| Block::from_weights(weights, index))
                .collect::<Result<_, _>>()?,
            final_norm: load_norm(weights, "backbone.layernorm", 1.0e-6)?,
            patch_size: config.patch_size,
            windows: config.num_windows,
            feature_layers: config.feature_layers(),
            global_layers: config.global_layers(),
        })
    }

    pub fn forward(&self, input: &MlxArray) -> Result<Vec<UniquePtr<MlxArray>>, String> {
        let input_shape = mlxcel_core::array_shape(input);
        if input_shape.len() != 4 || input_shape[3] != 3 {
            return Err(format!(
                "RF-DETR backbone expects NHWC RGB, got {input_shape:?}"
            ));
        }
        let (batch, height, width) = (input_shape[0], input_shape[1], input_shape[2]);
        let patch_h = height / self.patch_size as i32;
        let patch_w = width / self.patch_size as i32;
        let windows = self.windows as i32;
        let window_count = windows * windows;
        let patches = self.patch.forward(input);
        let patches = mlxcel_core::reshape(&patches, &[batch, patch_h * patch_w, HIDDEN]);
        let cls = mlxcel_core::broadcast_to(&self.cls_token, &[batch, 1, HIDDEN]);
        let tokens = concat(&[cls, patches], 1);
        let position = self.position_encoding(patch_h, patch_w)?;
        let tokens = mlxcel_core::add(&tokens, &position);
        let cls = mlxcel_core::slice(&tokens, &[0, 0, 0], &[batch, 1, HIDDEN]);
        let patches =
            mlxcel_core::slice(&tokens, &[0, 1, 0], &[batch, 1 + patch_h * patch_w, HIDDEN]);
        let patches = window_partition(&patches, batch, patch_h, patch_w, windows);
        let cls = mlxcel_core::tile(&cls, &[window_count, 1, 1]);
        let mut hidden = concat(&[cls, patches], 1);
        let mut features = Vec::with_capacity(self.feature_layers.len());
        for (index, block) in self.blocks.iter().enumerate() {
            let is_global = self.global_layers.contains(&index);
            let window_tokens = mlxcel_core::array_shape(&hidden)[1];
            if is_global {
                hidden =
                    mlxcel_core::reshape(&hidden, &[batch, window_count * window_tokens, HIDDEN]);
            }
            hidden = block.forward(&hidden);
            if is_global {
                hidden =
                    mlxcel_core::reshape(&hidden, &[batch * window_count, window_tokens, HIDDEN]);
            }
            if self.feature_layers.contains(&index) {
                let normalized = self.final_norm.forward(&hidden);
                let patches = mlxcel_core::slice(
                    &normalized,
                    &[0, 1, 0],
                    &[batch * window_count, window_tokens, HIDDEN],
                );
                let patches = window_unpartition(&patches, batch, patch_h, patch_w, windows);
                features.push(mlxcel_core::reshape(
                    &patches,
                    &[batch, patch_h, patch_w, HIDDEN],
                ));
            }
        }
        Ok(features)
    }

    fn position_encoding(&self, patch_h: i32, patch_w: i32) -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(&self.position);
        let stored = shape[1] - 1;
        if stored == patch_h * patch_w {
            return Ok(mlxcel_core::copy(&self.position));
        }
        let side = (stored as f64).sqrt() as i32;
        if side * side != stored {
            return Err(format!("RF-DETR position table is not square: {shape:?}"));
        }
        let cls = mlxcel_core::slice(&self.position, &[0, 0, 0], &[1, 1, HIDDEN]);
        let patch = mlxcel_core::slice(&self.position, &[0, 1, 0], &[1, 1 + stored, HIDDEN]);
        let patch = mlxcel_core::reshape(&patch, &[1, side, side, HIDDEN]);
        let patch = resize_bicubic_nhwc(
            &patch,
            Some((patch_h as usize, patch_w as usize)),
            None,
            true,
        )
        .map_err(|error| format!("RF-DETR position interpolation failed: {error}"))?;
        let patch = mlxcel_core::reshape(&patch, &[1, patch_h * patch_w, HIDDEN]);
        Ok(concat(&[cls, patch], 1))
    }
}

pub(crate) fn window_partition(
    input: &MlxArray,
    batch: i32,
    height: i32,
    width: i32,
    windows: i32,
) -> UniquePtr<MlxArray> {
    let window_h = height / windows;
    let window_w = width / windows;
    let input = mlxcel_core::reshape(
        input,
        &[batch, windows, window_h, windows, window_w, HIDDEN],
    );
    let input = mlxcel_core::transpose_axes(&input, &[0, 1, 3, 2, 4, 5]);
    mlxcel_core::reshape(
        &input,
        &[batch * windows * windows, window_h * window_w, HIDDEN],
    )
}

pub(crate) fn window_unpartition(
    input: &MlxArray,
    batch: i32,
    height: i32,
    width: i32,
    windows: i32,
) -> UniquePtr<MlxArray> {
    let window_h = height / windows;
    let window_w = width / windows;
    let input = mlxcel_core::reshape(
        input,
        &[batch, windows, windows, window_h, window_w, HIDDEN],
    );
    let input = mlxcel_core::transpose_axes(&input, &[0, 1, 3, 2, 4, 5]);
    mlxcel_core::reshape(&input, &[batch, height * width, HIDDEN])
}
