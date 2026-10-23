// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint loading and PyTorch-to-MLX convolution layout conversion.

use anyhow::{Context, Result, ensure};
use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::{WeightMap, load_weights_from_dir};
use mlxcel_core::{MlxArray, UniquePtr};
use std::path::Path;

pub fn load(path: &Path) -> Result<WeightMap> {
    let mut weights = load_weights_from_dir(path)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("load Video Depth Anything weights from {}", path.display()))?;
    sanitize_convolutions(&mut weights)?;
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path.join("config.json"))
            .context("read VDA config for dtype policy")?,
    )?;
    if crate::models::bf16_to_f16_at_load(false, Some(&config)) {
        let _ = crate::models::convert_bf16_weights(&mut weights);
    }
    validate_uniform_floating_dtype(&weights)?;
    Ok(weights)
}

fn validate_uniform_floating_dtype(weights: &WeightMap) -> Result<()> {
    let target = weights
        .get("pretrained.blocks.0.attn.qkv.weight")
        .map(|value| mlxcel_core::array_dtype(value))
        .context("VDA checkpoint lacks representative attention weight")?;
    ensure!(
        matches!(
            target,
            mlxcel_core::dtype::FLOAT16
                | mlxcel_core::dtype::BFLOAT16
                | mlxcel_core::dtype::FLOAT32
        ),
        "VDA checkpoint weights must use a floating dtype"
    );
    for (key, value) in weights {
        let dtype = mlxcel_core::array_dtype(value);
        ensure!(
            dtype == target,
            "VDA weight {key} has dtype {dtype}, expected uniform dtype {target}"
        );
        let shape = mlxcel_core::array_shape(value);
        ensure!(
            !shape.is_empty() && shape.iter().all(|dimension| *dimension > 0),
            "VDA weight {key} has invalid shape {shape:?}"
        );
    }
    Ok(())
}

fn sanitize_convolutions(weights: &mut WeightMap) -> Result<()> {
    let patch = weights
        .get("pretrained.patch_embed.proj.weight")
        .context("VDA checkpoint lacks pretrained.patch_embed.proj.weight")?;
    let shape = mlxcel_core::array_shape(patch);
    let torch_layout = shape.len() == 4 && shape[1] == 3;
    let mlx_layout = shape.len() == 4 && shape[3] == 3;
    ensure!(
        torch_layout || mlx_layout,
        "VDA patch projection has unsupported shape {shape:?}"
    );
    if !torch_layout {
        return Ok(());
    }
    let keys: Vec<String> = weights
        .iter()
        .filter(|(_, value)| mlxcel_core::array_shape(value).len() == 4)
        .map(|(key, _)| key.clone())
        .collect();
    for key in keys {
        let value = weights.remove(&key).expect("key came from map");
        let axes: &[i32] =
            if key.contains("head.resize_layers.0") || key.contains("head.resize_layers.1") {
                &[1, 2, 3, 0]
            } else {
                &[0, 2, 3, 1]
            };
        weights.insert(key, mlxcel_core::transpose_axes(&value, axes));
    }
    Ok(())
}

pub fn array(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>> {
    weights
        .get(key)
        .map(|value| mlxcel_core::copy(value))
        .with_context(|| format!("VDA weight not found: {key}"))
}

pub fn array_exact(
    weights: &WeightMap,
    key: &str,
    expected: &[i32],
) -> Result<UniquePtr<MlxArray>> {
    let value = array(weights, key)?;
    let shape = mlxcel_core::array_shape(&value);
    ensure!(
        shape == expected,
        "VDA weight {key} has shape {shape:?}, expected {expected:?}"
    );
    Ok(value)
}

pub fn linear(
    weights: &WeightMap,
    prefix: &str,
    input: usize,
    output: usize,
    biased: bool,
) -> Result<Linear> {
    let weight = array_exact(
        weights,
        &format!("{prefix}.weight"),
        &[output as i32, input as i32],
    )?;
    let bias_key = format!("{prefix}.bias");
    let bias = if biased {
        Some(array_exact(weights, &bias_key, &[output as i32])?)
    } else {
        ensure!(
            !weights.contains_key(&bias_key),
            "VDA bias-free layer {prefix} unexpectedly contains a bias"
        );
        None
    };
    Ok(Linear::new(weight, bias))
}

pub fn layer_norm(
    weights: &WeightMap,
    prefix: &str,
    channels: usize,
    eps: f32,
) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        array_exact(weights, &format!("{prefix}.weight"), &[channels as i32])?,
        Some(array_exact(
            weights,
            &format!("{prefix}.bias"),
            &[channels as i32],
        )?),
        eps,
    ))
}

pub fn conv(
    weights: &WeightMap,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    bias: bool,
) -> Result<Conv2d> {
    Ok(Conv2d {
        weight: array_exact(
            weights,
            &format!("{prefix}.weight"),
            &[output as i32, kernel as i32, kernel as i32, input as i32],
        )?,
        bias: bias
            .then(|| array_exact(weights, &format!("{prefix}.bias"), &[output as i32]))
            .transpose()?,
    })
}

pub struct Conv2d {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
}

impl Conv2d {
    pub fn forward(&self, x: &MlxArray, stride: i32, padding: i32) -> UniquePtr<MlxArray> {
        let y = mlxcel_core::conv2d(x, &self.weight, stride, stride, padding, padding, 1, 1, 1);
        match &self.bias {
            Some(bias) => mlxcel_core::add(&y, bias),
            None => y,
        }
    }
}

pub struct ConvTranspose2d {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
}

impl ConvTranspose2d {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        channels: usize,
        kernel: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: array_exact(
                weights,
                &format!("{prefix}.weight"),
                &[
                    channels as i32,
                    kernel as i32,
                    kernel as i32,
                    channels as i32,
                ],
            )?,
            bias: Some(array_exact(
                weights,
                &format!("{prefix}.bias"),
                &[channels as i32],
            )?),
        })
    }
    pub fn forward(&self, x: &MlxArray, stride: i32) -> UniquePtr<MlxArray> {
        let y = mlxcel_core::conv_transpose2d(x, &self.weight, stride, stride, 0, 0, 1, 1, 0, 0, 1);
        match &self.bias {
            Some(bias) => mlxcel_core::add(&y, bias),
            None => y,
        }
    }
}
