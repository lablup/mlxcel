// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Configuration of the released DINOv3/MHR Body graph.
use anyhow::{Result, ensure};
use serde::Deserialize;
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model_type: String,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub image_size: [usize; 2],
    pub ffn_ratio: f64,
    pub num_storage_tokens: usize,
    pub rope_periods: usize,
    pub decoder_dim: usize,
    pub decoder_depth: usize,
    pub decoder_heads: usize,
    pub decoder_head_dim: usize,
    pub decoder_mlp_dim: usize,
    pub num_joints: usize,
    pub num_vertices: usize,
    pub num_faces: usize,
    pub num_shape_comps: usize,
    pub num_face_comps: usize,
    pub pose_output_dim: usize,
    pub camera_output_dim: usize,
    pub num_point_embeddings: usize,
    pub prompt_embed_dim: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}
impl Default for Config {
    fn default() -> Self {
        Self {
            model_type: "sam3d_body".into(),
            embed_dim: 1280,
            depth: 32,
            num_heads: 20,
            head_dim: 64,
            patch_size: 16,
            image_size: [512, 384],
            ffn_ratio: 4.0,
            num_storage_tokens: 4,
            rope_periods: 16,
            decoder_dim: 1024,
            decoder_depth: 6,
            decoder_heads: 8,
            decoder_head_dim: 64,
            decoder_mlp_dim: 1024,
            num_joints: 127,
            num_vertices: 18439,
            num_faces: 36874,
            num_shape_comps: 45,
            num_face_comps: 72,
            pose_output_dim: 519,
            camera_output_dim: 3,
            num_point_embeddings: 70,
            prompt_embed_dim: 1280,
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225],
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "sam3d_body",
            "expected sam3d_body model_type"
        );
        let d = Self::default();
        ensure!(
            self.embed_dim == d.embed_dim
                && self.depth == d.depth
                && self.num_heads == d.num_heads
                && self.head_dim == d.head_dim
                && self.patch_size == d.patch_size
                && self.ffn_ratio == d.ffn_ratio
                && self.num_storage_tokens == d.num_storage_tokens
                && self.rope_periods == d.rope_periods,
            "unsupported Body DINOv3 architecture"
        );
        ensure!(
            self.decoder_dim == 1024
                && self.decoder_depth == 6
                && self.decoder_heads == 8
                && self.decoder_head_dim == 64
                && self.decoder_mlp_dim == 1024,
            "unsupported Body decoder architecture"
        );
        ensure!(
            self.num_joints == 127
                && self.num_vertices == 18439
                && self.num_faces == 36874
                && self.num_shape_comps == 45
                && self.num_face_comps == 72
                && self.pose_output_dim == 519
                && self.camera_output_dim == 3
                && self.num_point_embeddings == 70
                && self.prompt_embed_dim == 1280,
            "unsupported MHR topology or parameter layout"
        );
        ensure!(
            self.image_size
                .iter()
                .all(|&v| v > 0 && v <= 1024 && v % 16 == 0),
            "Body image_size must contain positive multiples of 16 no larger than 1024"
        );
        ensure!(
            self.image_mean.iter().all(|v| v.is_finite())
                && self.image_std.iter().all(|v| v.is_finite() && *v > 0.0),
            "invalid Body normalization"
        );
        Ok(())
    }
}
