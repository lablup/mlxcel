// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::*;
// Used by: Dinov2, VideoDepthAnything, MoGe3, Sam3dObjects
pub struct Dinov2Encoder {
    pub config: Dinov2Config,
    pub backbone: Dinov2,
}

/// Upstream-compatible spelling retained for dense-model consumers.
pub type DINOv2Encoder = Dinov2Encoder;

impl Dinov2Encoder {
    pub fn new(backbone: Dinov2) -> Self {
        Self {
            config: backbone.config.clone(),
            backbone,
        }
    }

    pub fn call(
        &self,
        image_01_nhwc: &MlxArray,
        token_rows: usize,
        token_cols: usize,
    ) -> Result<Vec<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>> {
        ensure!(
            token_rows > 0 && token_cols > 0,
            "DINOv2 token grid must be positive"
        );
        let resized_rows = token_rows
            .checked_mul(self.config.patch_size)
            .ok_or_else(|| anyhow::anyhow!("DINOv2 token row size overflowed"))?;
        let resized_cols = token_cols
            .checked_mul(self.config.patch_size)
            .ok_or_else(|| anyhow::anyhow!("DINOv2 token column size overflowed"))?;
        ensure!(
            resized_rows <= i32::MAX as usize && resized_cols <= i32::MAX as usize,
            "DINOv2 token grid exceeds the MLX shape range"
        );
        let patch_count = token_rows
            .checked_mul(token_cols)
            .ok_or_else(|| anyhow::anyhow!("DINOv2 token grid count overflowed"))?;
        ensure!(
            patch_count <= MAX_DINOV2_PATCH_TOKENS,
            "DINOv2 token grid exceeds the safety limit {MAX_DINOV2_PATCH_TOKENS}"
        );
        let shape = mlxcel_core::array_shape(image_01_nhwc);
        ensure!(
            shape.len() == 4 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0 && shape[3] == 3,
            "DINOv2Encoder expects NHWC RGB"
        );
        let source_pixels = (shape[0] as usize)
            .checked_mul(shape[1] as usize)
            .and_then(|pixels| pixels.checked_mul(shape[2] as usize))
            .ok_or_else(|| anyhow::anyhow!("DINOv2 source pixel count overflowed"))?;
        ensure!(
            source_pixels <= MAX_DINOV2_RUNTIME_PIXELS,
            "DINOv2 source pixel count exceeds the safety limit {MAX_DINOV2_RUNTIME_PIXELS}"
        );
        let batch_patch_count = (shape[0] as usize)
            .checked_mul(patch_count)
            .ok_or_else(|| anyhow::anyhow!("DINOv2 batch token grid count overflowed"))?;
        ensure!(
            batch_patch_count <= MAX_DINOV2_BATCH_PATCH_TOKENS,
            "DINOv2 batch token grid exceeds the safety limit {MAX_DINOV2_BATCH_PATCH_TOKENS}"
        );
        let runtime_pixels = batch_patch_count
            .checked_mul(self.config.patch_size)
            .and_then(|pixels| pixels.checked_mul(self.config.patch_size))
            .ok_or_else(|| anyhow::anyhow!("DINOv2 resized pixel count overflowed"))?;
        ensure!(
            runtime_pixels <= MAX_DINOV2_RUNTIME_PIXELS,
            "DINOv2 resized pixel count exceeds the safety limit {MAX_DINOV2_RUNTIME_PIXELS}"
        );
        let resized =
            resize_bilinear_nhwc(image_01_nhwc, (resized_rows, resized_cols), false, true)?;
        let normalized = normalize_imagenet(&resized)?;
        let batch = shape[0];
        self.backbone
            .get_intermediate_layers(&normalized, &self.config.intermediate_layers)?
            .into_iter()
            .map(|(tokens, cls)| {
                Ok((
                    mlxcel_core::reshape(
                        &tokens,
                        &[
                            batch,
                            token_rows as i32,
                            token_cols as i32,
                            self.config.hidden_size as i32,
                        ],
                    ),
                    cls,
                ))
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "dinov2_tests.rs"]
mod tests;
