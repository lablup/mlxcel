// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! DINOv2 backbone plus temporal DPT model.

use anyhow::{Context, Result, ensure};
use mlxcel_core::{MlxArray, UniquePtr};
use std::path::Path;

use super::config::{ResolvedVdaConfig, VdaConfig};
use super::dpt::DptHead;
use super::weights;
use crate::vision::dense::interpolate::resize_bilinear_nhwc;
use crate::vision::encoders::dinov2::Dinov2;

const BACKBONE_CHUNK_FRAMES: usize = 16;
const MAX_BATCH_SEQUENCE_TOKENS: usize = 65_536;

pub struct VideoDepthModel {
    pub config: ResolvedVdaConfig,
    backbone: Dinov2,
    head: DptHead,
    #[cfg(test)]
    representative_weight_dtype: i32,
    #[cfg(test)]
    representative_head_dtype: i32,
    #[cfg(test)]
    representative_norm_dtype: i32,
}

impl VideoDepthModel {
    pub fn load(path: &Path) -> Result<Self> {
        let config = VdaConfig::from_path(path)?;
        let weights = weights::load(path)?;
        #[cfg(test)]
        let representative_weight_dtype = weights
            .get("pretrained.blocks.0.attn.qkv.weight")
            .map(|value| mlxcel_core::array_dtype(value))
            .context("VDA checkpoint lacks representative backbone weight")?;
        #[cfg(test)]
        let representative_head_dtype = weights
            .get("head.motion_modules.0.temporal_transformer.proj_in.weight")
            .map(|value| mlxcel_core::array_dtype(value))
            .context("VDA checkpoint lacks representative temporal-head weight")?;
        #[cfg(test)]
        let representative_norm_dtype = weights
            .get("head.motion_modules.0.temporal_transformer.norm.weight")
            .map(|value| mlxcel_core::array_dtype(value))
            .context("VDA checkpoint lacks representative norm weight")?;
        let backbone =
            Dinov2::from_hf_weights_preserve_dtype(&weights, config.dinov2.clone(), "pretrained")
                .map_err(anyhow::Error::msg)
                .context("load VDA DINOv2 backbone")?;
        let head =
            DptHead::from_weights(&weights, &config).context("load VDA temporal DPT head")?;
        Ok(Self {
            config,
            backbone,
            head,
            #[cfg(test)]
            representative_weight_dtype,
            #[cfg(test)]
            representative_head_dtype,
            #[cfg(test)]
            representative_norm_dtype,
        })
    }

    #[cfg(test)]
    pub fn representative_weight_dtype(&self) -> i32 {
        self.representative_weight_dtype
    }
    #[cfg(test)]
    pub fn representative_head_dtype(&self) -> i32 {
        self.representative_head_dtype
    }
    #[cfg(test)]
    pub fn representative_norm_dtype(&self) -> i32 {
        self.representative_norm_dtype
    }

    fn backbone_features(
        &self,
        frames: &MlxArray,
        patch_h: usize,
        patch_w: usize,
    ) -> Result<Vec<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>> {
        let shape = mlxcel_core::array_shape(frames);
        let frame_count = shape[0] as usize;
        let sequence = patch_h
            .checked_mul(patch_w)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("VDA patch sequence overflowed"))?;
        let cap = (MAX_BATCH_SEQUENCE_TOKENS / sequence)
            .max(1)
            .min(BACKBONE_CHUNK_FRAMES);
        let mut patch_chunks: [Vec<UniquePtr<MlxArray>>; 4] = std::array::from_fn(|_| Vec::new());
        let mut cls_chunks: [Vec<UniquePtr<MlxArray>>; 4] = std::array::from_fn(|_| Vec::new());
        for start in (0..frame_count).step_by(cap) {
            let end = (start + cap).min(frame_count);
            let chunk = mlxcel_core::slice(
                frames,
                &[start as i32, 0, 0, 0],
                &[end as i32, shape[1], shape[2], shape[3]],
            );
            let features = self
                .backbone
                .get_intermediate_layers(&chunk, &self.config.intermediate_layers)?;
            for (index, (patch, class)) in features.into_iter().enumerate() {
                patch_chunks[index].push(patch);
                cls_chunks[index].push(class);
            }
        }
        let mut output = Vec::with_capacity(4);
        for index in 0..4 {
            let patches: Vec<&MlxArray> = patch_chunks[index]
                .iter()
                .map(|value| value.as_ref().expect("MLX patch feature"))
                .collect();
            let classes: Vec<&MlxArray> = cls_chunks[index]
                .iter()
                .map(|value| value.as_ref().expect("MLX class feature"))
                .collect();
            output.push((
                mlxcel_core::concatenate_many(&patches, 0),
                mlxcel_core::concatenate_many(&classes, 0),
            ));
        }
        Ok(output)
    }

    /// Infer one exact 32-frame temporal window of normalized NHWC frames.
    pub fn forward_window(&self, frames: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(frames);
        ensure!(
            shape.len() == 4 && shape[0] as usize == self.config.num_frames && shape[3] == 3,
            "VDA window must have shape [32, height, width, 3], got {shape:?}"
        );
        let height = shape[1] as usize;
        let width = shape[2] as usize;
        ensure!(
            height.is_multiple_of(self.config.dinov2.patch_size)
                && width.is_multiple_of(self.config.dinov2.patch_size),
            "VDA window dimensions must be patch multiples"
        );
        let patch_h = height / self.config.dinov2.patch_size;
        let patch_w = width / self.config.dinov2.patch_size;
        let features = self.backbone_features(frames, patch_h, patch_w)?;
        let depth = self
            .head
            .forward(&features, patch_h, patch_w, self.config.num_frames)?;
        let depth = resize_bilinear_nhwc(&depth, (height, width), true, false)?;
        let depth = mlxcel_core::relu(&depth);
        Ok(mlxcel_core::reshape(
            &depth,
            &[self.config.num_frames as i32, height as i32, width as i32],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn backbone_chunk_respects_shared_token_cap() {
        let sequence = 37 * 66 + 1;
        let cap = (MAX_BATCH_SEQUENCE_TOKENS / sequence)
            .max(1)
            .min(BACKBONE_CHUNK_FRAMES);
        assert_eq!(cap, 16);
        assert!(cap * sequence <= MAX_BATCH_SEQUENCE_TOKENS);
        assert!(32 * sequence > MAX_BATCH_SEQUENCE_TOKENS);
    }
}
