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

//! MoonViT3D `vision_config` (Kimi K3): the exact published field names,
//! their defaults, and the validation that refuses every variant
//! `encoders::moonvit3d` does not implement.
//!
//! Used by: `encoders::moonvit3d`, `loading::vlm_kimi_k3`.

use serde::Deserialize;

/// MoonViT3D vision configuration, over the exact `vision_config` field names
/// of the published `config.json`. Defaults are the published values.
#[derive(Debug, Clone, Deserialize)]
pub struct MoonViT3DConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_init_pos_emb_side")]
    pub init_pos_emb_height: usize,
    #[serde(default = "default_init_pos_emb_side")]
    pub init_pos_emb_width: usize,
    #[serde(default = "default_init_pos_emb_time")]
    pub init_pos_emb_time: usize,
    #[serde(default = "default_num_heads")]
    pub vt_num_attention_heads: usize,
    #[serde(default = "default_num_layers")]
    pub vt_num_hidden_layers: usize,
    #[serde(default = "default_hidden_size")]
    pub vt_hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub vt_intermediate_size: usize,
    #[serde(default = "default_qkv_hidden_size")]
    pub qkv_hidden_size: usize,
    #[serde(default)]
    pub attn_bias: bool,
    #[serde(default)]
    pub linear_bias: bool,
    #[serde(default)]
    pub patch_embed_proj_bias: bool,
    #[serde(default = "default_merge_kernel_size")]
    pub merge_kernel_size: [usize; 2],
    #[serde(default = "default_merge_type")]
    pub merge_type: String,
    #[serde(default = "default_projector_type")]
    pub mm_projector_type: String,
    #[serde(default = "default_hidden_size")]
    pub mm_hidden_size: usize,
    #[serde(default = "default_projector_act")]
    pub projector_hidden_act: String,
    #[serde(default = "default_projector_ln_eps")]
    pub projector_ln_eps: f32,
    #[serde(default = "default_activation_func")]
    pub activation_func: String,
    #[serde(default = "default_norm_type")]
    pub norm_type: String,
    #[serde(default = "default_text_hidden_size")]
    pub text_hidden_size: usize,
    #[serde(default = "default_pos_emb_interpolation_mode")]
    pub pos_emb_interpolation_mode: String,
}

fn default_model_type() -> String {
    "moonvit3d".to_string()
}
fn default_patch_size() -> usize {
    14
}
fn default_init_pos_emb_side() -> usize {
    64
}
fn default_init_pos_emb_time() -> usize {
    4
}
fn default_num_heads() -> usize {
    12
}
fn default_num_layers() -> usize {
    27
}
fn default_hidden_size() -> usize {
    1024
}
fn default_intermediate_size() -> usize {
    4096
}
fn default_qkv_hidden_size() -> usize {
    1536
}
fn default_merge_kernel_size() -> [usize; 2] {
    [2, 2]
}
fn default_merge_type() -> String {
    "sd2_tpool".to_string()
}
fn default_projector_type() -> String {
    "patchmergerv2".to_string()
}
fn default_projector_act() -> String {
    "gelu".to_string()
}
fn default_projector_ln_eps() -> f32 {
    1e-5
}
fn default_activation_func() -> String {
    "gelu_pytorch_tanh".to_string()
}
fn default_norm_type() -> String {
    "rmsnorm".to_string()
}
fn default_text_hidden_size() -> usize {
    7168
}
fn default_pos_emb_interpolation_mode() -> String {
    "bilinear".to_string()
}

impl MoonViT3DConfig {
    /// Refuse every variant this port does not implement, naming the field.
    pub fn validate(&self) -> Result<(), String> {
        let expect = |field: &str, value: &str, want: &str| {
            if value == want {
                Ok(())
            } else {
                Err(format!(
                    "moonvit3d: vision_config.{field} = {value:?} is not supported; this port \
                     implements {want:?}"
                ))
            }
        };
        expect("norm_type", &self.norm_type, "rmsnorm")?;
        expect("merge_type", &self.merge_type, "sd2_tpool")?;
        expect(
            "mm_projector_type",
            &self.mm_projector_type,
            "patchmergerv2",
        )?;
        expect(
            "activation_func",
            &self.activation_func,
            "gelu_pytorch_tanh",
        )?;
        expect("projector_hidden_act", &self.projector_hidden_act, "gelu")?;
        expect(
            "pos_emb_interpolation_mode",
            &self.pos_emb_interpolation_mode,
            "bilinear",
        )?;
        if self.vt_num_attention_heads == 0
            || !self
                .qkv_hidden_size
                .is_multiple_of(self.vt_num_attention_heads)
        {
            return Err(format!(
                "moonvit3d: qkv_hidden_size ({}) must be a positive multiple of \
                 vt_num_attention_heads ({})",
                self.qkv_hidden_size, self.vt_num_attention_heads
            ));
        }
        if !self.head_dim().is_multiple_of(4) {
            return Err(format!(
                "moonvit3d: head_dim ({}) must be divisible by 4 for the 2D rotary embedding",
                self.head_dim()
            ));
        }
        if self.merge_kernel_size.contains(&0) {
            return Err("moonvit3d: merge_kernel_size entries must be positive".to_string());
        }
        if self.mm_hidden_size != self.vt_hidden_size {
            return Err(format!(
                "moonvit3d: mm_hidden_size ({}) must equal vt_hidden_size ({})",
                self.mm_hidden_size, self.vt_hidden_size
            ));
        }
        Ok(())
    }

    /// `qkv_hidden_size / vt_num_attention_heads` (128 on the published config).
    pub fn head_dim(&self) -> usize {
        self.qkv_hidden_size / self.vt_num_attention_heads
    }

    /// `(kh, kw)` of the spatial merge.
    pub fn merge(&self) -> (i32, i32) {
        (
            self.merge_kernel_size[0] as i32,
            self.merge_kernel_size[1] as i32,
        )
    }

    /// Width of one merged token handed to the projector:
    /// `vt_hidden_size * kh * kw` (4096 on the published config).
    pub fn merged_hidden(&self) -> usize {
        self.vt_hidden_size * self.merge_kernel_size[0] * self.merge_kernel_size[1]
    }
}
