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

//! Hugging Face convolution-layout conversion for SAM 3 checkpoints.
//!
//! The official checkpoint stores PyTorch OIHW/IOHW convolution kernels;
//! mlx-community checkpoints already store MLX channels-last kernels. The
//! patch embedding has an unambiguous RGB input axis and is therefore the
//! layout marker for the entire checkpoint. Inspecting the whole key set
//! before deciding makes the result deterministic for `WeightMap`'s randomized
//! iteration order and prevents a second sanitize pass from transposing an
//! already-converted checkpoint.

use anyhow::{Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::preprocess::conv_weight_to_channels_last;

const PATCH_EMBED_SUFFIX: &str = "patch_embeddings.projection.weight";

/// Return whether `weights` carries raw PyTorch convolution layouts.
#[must_use]
pub fn needs_sanitize(weights: &WeightMap) -> Result<bool> {
    let mut has_hf_marker = false;
    let mut has_mlx_marker = false;

    for (key, value) in weights {
        if !key.ends_with(PATCH_EMBED_SUFFIX) {
            continue;
        }
        let shape = mlxcel_core::array_shape(value);
        if shape.len() != 4 {
            continue;
        }
        has_hf_marker |= shape[1] == 3 && shape[3] != 3;
        has_mlx_marker |= shape[3] == 3 && shape[1] != 3;
    }

    match (has_hf_marker, has_mlx_marker) {
        (true, true) => bail!(
            "SAM 3 checkpoint mixes PyTorch and MLX convolution layouts; use a consistently converted checkpoint"
        ),
        (true, false) => Ok(true),
        (false, true) => Ok(false),
        (false, false) => {
            bail!("SAM 3 checkpoint has no unambiguous patch-embedding convolution layout marker")
        }
    }
}

fn is_non_convolution_parameter(key: &str) -> bool {
    key.contains("memory_temporal_positional_encoding")
}

fn is_transposed_convolution(key: &str) -> bool {
    key.contains("scale_layers.") || key.contains("upscale_conv")
}

fn convert_rank_four(key: &str, value: UniquePtr<MlxArray>) -> Result<UniquePtr<MlxArray>> {
    if mlxcel_core::array_shape(&value).len() != 4 || is_non_convolution_parameter(key) {
        return Ok(value);
    }

    if is_transposed_convolution(key) {
        // PyTorch ConvTranspose2d IOHW -> MLX OHWI.
        Ok(mlxcel_core::transpose_axes(&value, &[1, 2, 3, 0]))
    } else {
        // The checkpoint-level RGB marker establishes that these tensors are
        // raw OIHW. The dense foundation guard then validates each kernel's
        // input-channel axis before converting it to OHWI.
        let input_channels = mlxcel_core::array_shape(&value)[1];
        conv_weight_to_channels_last(&value, input_channels)
    }
}

/// Convert raw-HF convolution kernels to MLX layout exactly once.
#[must_use]
pub fn sanitize(weights: WeightMap) -> Result<WeightMap> {
    if !needs_sanitize(&weights)? {
        return Ok(weights);
    }

    weights
        .into_iter()
        .map(|(key, value)| {
            let value = convert_rank_four(&key, value)?;
            Ok((key, value))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[i32]) -> UniquePtr<MlxArray> {
        mlxcel_core::zeros(shape, mlxcel_core::dtype::FLOAT32)
    }

    #[test]
    fn already_converted_checkpoint_is_idempotent() {
        let marker = "detector_model.vision_encoder.backbone.patch_embeddings.projection.weight";
        let conv = "tracker_model.memory_encoder.feature_projection.weight";
        let mut weights = WeightMap::new();
        weights.insert(marker.into(), tensor(&[8, 14, 14, 3]));
        weights.insert(conv.into(), tensor(&[8, 1, 1, 4]));

        let weights = sanitize(weights).unwrap();
        assert_eq!(
            mlxcel_core::array_shape(&weights[marker]),
            vec![8, 14, 14, 3]
        );
        assert_eq!(mlxcel_core::array_shape(&weights[conv]), vec![8, 1, 1, 4]);
        let weights = sanitize(weights).unwrap();
        assert_eq!(
            mlxcel_core::array_shape(&weights[marker]),
            vec![8, 14, 14, 3]
        );
        assert_eq!(mlxcel_core::array_shape(&weights[conv]), vec![8, 1, 1, 4]);
    }

    #[test]
    fn raw_conv_and_transposed_conv_are_converted() {
        let marker = "detector_model.vision_encoder.backbone.patch_embeddings.projection.weight";
        let up = "tracker_model.mask_decoder.upscale_conv1.weight";
        let temporal = "tracker_model.memory_temporal_positional_encoding";
        let mut weights = WeightMap::new();
        weights.insert(marker.into(), tensor(&[8, 3, 14, 14]));
        weights.insert(up.into(), tensor(&[8, 4, 2, 2]));
        weights.insert(temporal.into(), tensor(&[7, 1, 1, 64]));

        let weights = sanitize(weights).unwrap();
        assert_eq!(
            mlxcel_core::array_shape(&weights[marker]),
            vec![8, 14, 14, 3]
        );
        assert_eq!(mlxcel_core::array_shape(&weights[up]), vec![4, 2, 2, 8]);
        assert_eq!(
            mlxcel_core::array_shape(&weights[temporal]),
            vec![7, 1, 1, 64]
        );
    }

    #[test]
    fn mixed_checkpoint_layout_is_rejected() {
        let mut weights = WeightMap::new();
        weights.insert(
            "detector_model.vision_encoder.backbone.patch_embeddings.projection.weight".into(),
            tensor(&[8, 3, 14, 14]),
        );
        weights.insert(
            "tracker_model.vision_encoder.backbone.patch_embeddings.projection.weight".into(),
            tensor(&[8, 14, 14, 3]),
        );
        assert!(sanitize(weights).is_err());
    }
}
