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

//! Reusable DINOv2 layers shared by standalone and dense models.

use anyhow::{Result, ensure};
use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::dinov2_config::{
    Dinov2Config, MAX_DINOV2_BATCH_PATCH_TOKENS, MAX_DINOV2_BATCH_SEQUENCE_TOKENS,
    MAX_DINOV2_HIDDEN_SIZE, MAX_DINOV2_INTERMEDIATE_SIZE, MAX_DINOV2_PATCH_TOKENS,
    MAX_DINOV2_RUNTIME_PIXELS,
};

pub(super) fn take_weight(
    weights: &WeightMap,
    key: &str,
) -> std::result::Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|value| mlxcel_core::copy(value))
        .ok_or_else(|| format!("DINOv2 weight not found: {key}"))
}

pub(super) fn check_shape(
    array: &MlxArray,
    expected: &[i32],
    key: &str,
) -> std::result::Result<(), String> {
    let actual = mlxcel_core::array_shape(array);
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "DINOv2 {key}: expected shape {expected:?}, got {actual:?}"
        ))
    }
}

fn load_linear(
    weights: &WeightMap,
    prefix: &str,
    output: usize,
    input: usize,
    require_bias: bool,
) -> std::result::Result<Linear, String> {
    let linear = Linear::from_weights(weights, prefix)?;
    check_shape(
        &linear.weight,
        &[output as i32, input as i32],
        &format!("{prefix}.weight"),
    )?;
    if let Some(bias) = &linear.bias {
        check_shape(bias, &[output as i32], &format!("{prefix}.bias"))?;
    } else if require_bias {
        return Err(format!("DINOv2 weight not found: {prefix}.bias"));
    }
    Ok(linear)
}

pub(super) fn load_norm(
    weights: &WeightMap,
    prefix: &str,
    hidden: usize,
    eps: f32,
) -> std::result::Result<LayerNorm, String> {
    let weight = take_weight(weights, &format!("{prefix}.weight"))?;
    let bias = take_weight(weights, &format!("{prefix}.bias"))?;
    check_shape(&weight, &[hidden as i32], &format!("{prefix}.weight"))?;
    check_shape(&bias, &[hidden as i32], &format!("{prefix}.bias"))?;
    Ok(LayerNorm::new(weight, Some(bias), eps))
}

pub(super) struct PatchEmbed {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    patch_size: usize,
    channels: usize,
    hidden_size: usize,
}

impl PatchEmbed {
    pub(super) fn from_weights(
        weights: &WeightMap,
        config: &Dinov2Config,
    ) -> std::result::Result<Self, String> {
        let weight = take_weight(weights, "patch_embed.proj.weight")?;
        let bias = take_weight(weights, "patch_embed.proj.bias")?;
        check_shape(
            &weight,
            &[
                config.hidden_size as i32,
                config.patch_size as i32,
                config.patch_size as i32,
                config.num_channels as i32,
            ],
            "patch_embed.proj.weight",
        )?;
        check_shape(&bias, &[config.hidden_size as i32], "patch_embed.proj.bias")?;
        Ok(Self {
            weight,
            bias,
            patch_size: config.patch_size,
            channels: config.num_channels,
            hidden_size: config.hidden_size,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(x);
        ensure!(
            shape.len() == 4 && shape[0] > 0 && shape[3] == self.channels as i32,
            "DINOv2 expects NHWC RGB input, got {shape:?}"
        );
        ensure!(
            shape[1] > 0
                && shape[2] > 0
                && (shape[1] as usize).is_multiple_of(self.patch_size)
                && (shape[2] as usize).is_multiple_of(self.patch_size),
            "DINOv2 input height and width must be positive multiples of {}",
            self.patch_size
        );
        let patch_rows = shape[1] as usize / self.patch_size;
        let patch_cols = shape[2] as usize / self.patch_size;
        let runtime_pixels = (shape[0] as usize)
            .checked_mul(shape[1] as usize)
            .and_then(|pixels| pixels.checked_mul(shape[2] as usize))
            .ok_or_else(|| anyhow::anyhow!("DINOv2 runtime pixel count overflowed"))?;
        ensure!(
            runtime_pixels <= MAX_DINOV2_RUNTIME_PIXELS,
            "DINOv2 runtime pixel count exceeds the safety limit {MAX_DINOV2_RUNTIME_PIXELS}"
        );
        let patch_count = patch_rows
            .checked_mul(patch_cols)
            .ok_or_else(|| anyhow::anyhow!("DINOv2 patch token count overflowed"))?;
        ensure!(
            patch_count <= MAX_DINOV2_PATCH_TOKENS,
            "DINOv2 patch count exceeds the safety limit {MAX_DINOV2_PATCH_TOKENS}"
        );
        let batch_patch_count = (shape[0] as usize)
            .checked_mul(patch_count)
            .ok_or_else(|| anyhow::anyhow!("DINOv2 batch patch count overflowed"))?;
        ensure!(
            batch_patch_count <= MAX_DINOV2_BATCH_PATCH_TOKENS,
            "DINOv2 batch patch count exceeds the safety limit {MAX_DINOV2_BATCH_PATCH_TOKENS}"
        );
        let patches = mlxcel_core::conv2d(
            x,
            &self.weight,
            self.patch_size as i32,
            self.patch_size as i32,
            0,
            0,
            1,
            1,
            1,
        );
        let patches = mlxcel_core::add(&patches, &self.bias);
        let shape = mlxcel_core::array_shape(&patches);
        let patch_count = shape[1]
            .checked_mul(shape[2])
            .ok_or_else(|| anyhow::anyhow!("DINOv2 patch token count overflowed"))?;
        Ok(mlxcel_core::reshape(
            &patches,
            &[shape[0], patch_count, self.hidden_size as i32],
        ))
    }
}

// Used by: Dinov2, VideoDepthAnything, MoGe3, Sam3dObjects
pub struct Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Attention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: usize,
        num_heads: usize,
        qkv_bias: bool,
    ) -> std::result::Result<Self, String> {
        if hidden_size == 0
            || hidden_size > MAX_DINOV2_HIDDEN_SIZE
            || num_heads == 0
            || !hidden_size.is_multiple_of(num_heads)
        {
            return Err(format!(
                "DINOv2 attention requires hidden_size 1..={MAX_DINOV2_HIDDEN_SIZE} divisible by positive num_heads"
            ));
        }
        let qkv_size = hidden_size
            .checked_mul(3)
            .filter(|size| *size <= i32::MAX as usize)
            .ok_or_else(|| "DINOv2 fused QKV size exceeds the MLX shape range".to_string())?;
        let qkv = load_linear(
            weights,
            &format!("{prefix}.qkv"),
            qkv_size,
            hidden_size,
            qkv_bias,
        )?;
        let proj = load_linear(
            weights,
            &format!("{prefix}.proj"),
            hidden_size,
            hidden_size,
            true,
        )?;
        let head_dim = hidden_size / num_heads;
        Ok(Self {
            qkv,
            proj,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    pub fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(x);
        ensure!(
            shape.len() == 3
                && shape[0] > 0
                && shape[1] > 0
                && shape[2] == (self.num_heads * self.head_dim) as i32,
            "DINOv2 attention expects [batch, tokens, {}], got {shape:?}",
            self.num_heads * self.head_dim
        );
        let batch_tokens = (shape[0] as usize)
            .checked_mul(shape[1] as usize)
            .ok_or_else(|| anyhow::anyhow!("DINOv2 attention token count overflowed"))?;
        ensure!(
            batch_tokens <= MAX_DINOV2_BATCH_SEQUENCE_TOKENS,
            "DINOv2 attention token count exceeds the safety limit {MAX_DINOV2_BATCH_SEQUENCE_TOKENS}"
        );
        let (batch, tokens) = (shape[0], shape[1]);
        let qkv = self.qkv.forward(x);
        let split = |part: i32| {
            let start = part * (self.num_heads * self.head_dim) as i32;
            let value = mlxcel_core::slice(
                &qkv,
                &[0, 0, start],
                &[
                    batch,
                    tokens,
                    start + (self.num_heads * self.head_dim) as i32,
                ],
            );
            let value = mlxcel_core::reshape(
                &value,
                &[batch, tokens, self.num_heads as i32, self.head_dim as i32],
            );
            mlxcel_core::transpose_axes(&value, &[0, 2, 1, 3])
        };
        let (query, key, value) = (split(0), split(1), split(2));
        let attended =
            mlxcel_core::layers::attention(&query, &key, &value, self.scale, None, 0.0, 0);
        let attended = mlxcel_core::transpose_axes(&attended, &[0, 2, 1, 3]);
        let attended = mlxcel_core::reshape(
            &attended,
            &[batch, tokens, (self.num_heads * self.head_dim) as i32],
        );
        Ok(self.proj.forward(&attended))
    }
}

// Used by: Dinov2, VideoDepthAnything, MoGe3, Sam3dObjects
pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> std::result::Result<Self, String> {
        if hidden_size == 0
            || hidden_size > MAX_DINOV2_HIDDEN_SIZE
            || intermediate_size == 0
            || intermediate_size > MAX_DINOV2_INTERMEDIATE_SIZE
        {
            return Err(format!(
                "DINOv2 MLP dimensions exceed supported limits ({MAX_DINOV2_HIDDEN_SIZE}, {MAX_DINOV2_INTERMEDIATE_SIZE})"
            ));
        }
        Ok(Self {
            fc1: load_linear(
                weights,
                &format!("{prefix}.fc1"),
                intermediate_size,
                hidden_size,
                true,
            )?,
            fc2: load_linear(
                weights,
                &format!("{prefix}.fc2"),
                hidden_size,
                intermediate_size,
                true,
            )?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.fc2.forward(&mlxcel_core::gelu(&self.fc1.forward(x)))
    }

    /// Exact compiled, weak-scalar GELU used by the SAM 3D Objects BF16 graph.
    pub fn forward_objects(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let hidden = mlxcel_core::compiled_objects_gelu(&self.fc1.forward(x))?;
        Ok(self.fc2.forward(&hidden))
    }
}

struct SwiGluFfn {
    w12: Linear,
    w3: Linear,
    intermediate_size: usize,
}

impl SwiGluFfn {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> std::result::Result<Self, String> {
        Ok(Self {
            w12: load_linear(
                weights,
                &format!("{prefix}.w12"),
                intermediate_size * 2,
                hidden_size,
                true,
            )?,
            w3: load_linear(
                weights,
                &format!("{prefix}.w3"),
                hidden_size,
                intermediate_size,
                true,
            )?,
            intermediate_size,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let projected = self.w12.forward(x);
        let shape = mlxcel_core::array_shape(&projected);
        let hidden = self.intermediate_size as i32;
        let gate = mlxcel_core::slice(&projected, &[0, 0, 0], &[shape[0], shape[1], hidden]);
        let value = mlxcel_core::slice(
            &projected,
            &[0, 0, hidden],
            &[shape[0], shape[1], hidden * 2],
        );
        self.w3
            .forward(&mlxcel_core::multiply(&mlxcel_core::silu(&gate), &value))
    }
}

enum FeedForward {
    Mlp(Mlp),
    SwiGlu(SwiGluFfn),
}

impl FeedForward {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Mlp(mlp) => mlp.forward(x),
            Self::SwiGlu(mlp) => mlp.forward(x),
        }
    }
}

pub(super) struct Block {
    norm1: LayerNorm,
    attention: Attention,
    scale1: UniquePtr<MlxArray>,
    norm2: LayerNorm,
    mlp: FeedForward,
    scale2: UniquePtr<MlxArray>,
}

impl Block {
    pub(super) fn from_weights(
        weights: &WeightMap,
        config: &Dinov2Config,
        index: usize,
    ) -> std::result::Result<Self, String> {
        let prefix = format!("blocks.{index}");
        let hidden = config.hidden_size;
        let dense_intermediate = (hidden as f32 * config.mlp_ratio) as usize;
        let mlp = if config.use_swiglu_ffn {
            let reduced = (dense_intermediate * 2 / 3).div_ceil(8) * 8;
            FeedForward::SwiGlu(SwiGluFfn::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                hidden,
                reduced,
            )?)
        } else {
            FeedForward::Mlp(Mlp::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                hidden,
                dense_intermediate,
            )?)
        };
        let scale1 = take_weight(weights, &format!("{prefix}.ls1.gamma"))?;
        let scale2 = take_weight(weights, &format!("{prefix}.ls2.gamma"))?;
        check_shape(&scale1, &[hidden as i32], &format!("{prefix}.ls1.gamma"))?;
        check_shape(&scale2, &[hidden as i32], &format!("{prefix}.ls2.gamma"))?;
        Ok(Self {
            norm1: load_norm(
                weights,
                &format!("{prefix}.norm1"),
                hidden,
                config.layer_norm_eps,
            )?,
            attention: Attention::from_weights(
                weights,
                &format!("{prefix}.attn"),
                hidden,
                config.num_attention_heads,
                config.qkv_bias,
            )?,
            scale1,
            norm2: load_norm(
                weights,
                &format!("{prefix}.norm2"),
                hidden,
                config.layer_norm_eps,
            )?,
            mlp,
            scale2,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        self.forward_with_norm_policy(x, false)
    }

    pub(super) fn forward_with_norm_policy(
        &self,
        x: &MlxArray,
        upcast: bool,
    ) -> Result<UniquePtr<MlxArray>> {
        let attention = self
            .attention
            .forward(&norm_with_policy(&self.norm1, x, upcast))?;
        let x = mlxcel_core::add(x, &mlxcel_core::multiply(&attention, &self.scale1));
        let normalized = norm_with_policy(&self.norm2, &x, upcast);
        let mlp = match &self.mlp {
            FeedForward::Mlp(mlp) if upcast => mlp.forward_objects(&normalized)?,
            _ => self.mlp.forward(&normalized),
        };
        Ok(mlxcel_core::add(
            &x,
            &mlxcel_core::multiply(&mlp, &self.scale2),
        ))
    }
}

#[path = "dinov2_policy.rs"]
mod policy;
pub(super) use policy::norm_with_policy;
