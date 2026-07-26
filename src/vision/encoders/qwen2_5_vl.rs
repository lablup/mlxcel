// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Qwen2.5-VL Vision Encoder
//!
//! Evolution of Qwen2-VL vision encoder with:
//! - RMSNorm instead of LayerNorm
//! - SwiGLU MLP (gate_proj/up_proj/down_proj with SiLU) instead of GELU fc1/fc2
//! - Windowed attention with fullatt_block_indexes for selective full attention
//! - PatchMerger with RMSNorm
//!
//! Used by: Qwen2.5-VL
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen2_5_vl/vision.py

use super::VisionEncoderOutput;
use super::qwen2_vl::{VisionRotaryEmbedding, apply_rotary_pos_emb_vision, concat_many};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

/// Qwen2.5-VL vision encoder configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen25VLVisionConfig {
    #[serde(default = "default_depth")]
    pub depth: usize,
    /// Vision hidden size (replaces embed_dim in Qwen2-VL)
    pub hidden_size: usize,
    /// Explicit MLP intermediate size (replaces mlp_ratio)
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    /// Output hidden size (projection to text space)
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(alias = "in_chans", default = "default_in_channels")]
    pub in_channels: usize,
    /// Window size for windowed attention
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    /// Block indices that use full attention (rest use windowed)
    #[serde(default = "default_fullatt_block_indexes")]
    pub fullatt_block_indexes: Vec<usize>,
    /// Quantization group_size (inherited from top-level config)
    #[serde(default)]
    pub quant_group_size: i32,
    /// Quantization bits (inherited from top-level config)
    #[serde(default)]
    pub quant_bits: i32,
}

fn default_depth() -> usize {
    32
}
fn default_intermediate_size() -> usize {
    3420
}
fn default_out_hidden_size() -> usize {
    1536
}
fn default_num_heads() -> usize {
    16
}
fn default_patch_size() -> usize {
    14
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_in_channels() -> usize {
    3
}
fn default_window_size() -> usize {
    112
}
fn default_fullatt_block_indexes() -> Vec<usize> {
    vec![7, 15, 23, 31]
}

fn restoration_indices(window_index: &[i32]) -> Vec<i32> {
    let mut restore = vec![0i32; window_index.len()];
    let mut seen = vec![false; window_index.len()];
    for (window_position, &original_position) in window_index.iter().enumerate() {
        let original_position = usize::try_from(original_position)
            .expect("Qwen2.5-VL window indices are validated as non-negative");
        assert!(
            original_position < restore.len() && !seen[original_position],
            "Qwen2.5-VL window indices must be an in-range permutation"
        );
        seen[original_position] = true;
        restore[original_position] = window_position as i32;
    }
    restore
}

// RMSNorm for vision encoder.
struct VisionRMSNorm {
    weight: UniquePtr<MlxArray>,
    eps: f32,
}

impl VisionRMSNorm {
    fn from_weights(weights: &WeightMap, prefix: &str, eps: f32) -> Result<Self, String> {
        let weight_key = format!("{}.weight", prefix);
        let weight = weights
            .get(&weight_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_key))?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::fast_rms_norm(x, &self.weight, self.eps)
    }
}

// PatchEmbed - Conv3d degenerated to Linear (same as Qwen2-VL).
struct PatchEmbed {
    proj_weight: UniquePtr<MlxArray>,
    proj_bias: Option<UniquePtr<MlxArray>>,
    in_channels: usize,
    temporal_patch_size: usize,
    patch_size: usize,
}

impl PatchEmbed {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen25VLVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let weight_key = format!("{}.proj.weight", prefix);
        let w = weights
            .get(&weight_key)
            .ok_or_else(|| format!("Missing {}", weight_key))?;

        let shape = mlxcel_core::array_shape(w);
        let out_features = config.hidden_size as i32;
        let in_features = (config.in_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size) as i32;

        // Handle Conv3d weight shape -> 2D Linear weight
        let w_reshaped = if shape.len() == 5 {
            // MLX Conv3d weight: [out, kT, kH, kW, in_channels]
            // Input data is in TCHW order (temporal, channel, height, width)
            // Reorder weight to [out, T, C, H, W] to match input layout
            let w_reordered = mlxcel_core::transpose_axes(w, &[0, 1, 4, 2, 3]);
            mlxcel_core::reshape(&w_reordered, &[out_features, in_features])
        } else if shape.len() == 2 {
            mlxcel_core::copy(w)
        } else {
            return Err(format!("Unexpected patch_embed weight shape: {:?}", shape));
        };

        let bias_key = format!("{}.proj.bias", prefix);
        let proj_bias = weights.get(&bias_key).map(|b| mlxcel_core::copy(b));

        Ok(Self {
            proj_weight: w_reshaped,
            proj_bias,
            in_channels: config.in_channels,
            temporal_patch_size: config.temporal_patch_size,
            patch_size: config.patch_size,
        })
    }

    fn forward(&self, hidden_states: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden_states);
        let total_elements = shape[0];
        let n = total_elements / self.temporal_patch_size as i32;
        let in_features =
            (self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size)
                as i32;

        let h = mlxcel_core::reshape(
            hidden_states,
            &[n, self.temporal_patch_size as i32, shape[1]],
        );
        let h = mlxcel_core::reshape(&h, &[n, in_features]);

        let wt = mlxcel_core::transpose(&self.proj_weight);
        let result = mlxcel_core::matmul(&h, &wt);
        match &self.proj_bias {
            Some(b) => mlxcel_core::add(&result, b),
            None => result,
        }
    }
}

// Vision Attention - same as Qwen2-VL.
struct VisionAttention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

#[derive(Clone, Copy)]
enum VisionAttentionDiagnosticStage {
    Query,
    Key,
    Value,
    Context,
}

impl VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen25VLVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let qkv = UnifiedLinear::from_weights(weights, &format!("{}.attn.qkv", prefix), gs, bits)?;
        let proj =
            UnifiedLinear::from_weights(weights, &format!("{}.attn.proj", prefix), gs, bits)?;
        let head_dim = (config.hidden_size / config.num_heads) as i32;

        Ok(Self {
            qkv,
            proj,
            num_heads: config.num_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_observer(x, cu_seqlens, rotary_pos_emb, |_, _| {})
    }

    fn forward_with_observer<F>(
        &self,
        x: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
        mut observe: F,
    ) -> UniquePtr<MlxArray>
    where
        F: FnMut(VisionAttentionDiagnosticStage, &MlxArray),
    {
        let shape = mlxcel_core::array_shape(x);
        let seq_length = shape[0];

        let qkv = self.qkv.forward(x);
        let qkv = mlxcel_core::reshape(&qkv, &[seq_length, 3, self.num_heads, self.head_dim]);
        let qkv = mlxcel_core::transpose_axes(&qkv, &[1, 0, 2, 3]);

        let q = mlxcel_core::slice(
            &qkv,
            &[0, 0, 0, 0],
            &[1, seq_length, self.num_heads, self.head_dim],
        );
        let k = mlxcel_core::slice(
            &qkv,
            &[1, 0, 0, 0],
            &[2, seq_length, self.num_heads, self.head_dim],
        );
        let v = mlxcel_core::slice(
            &qkv,
            &[2, 0, 0, 0],
            &[3, seq_length, self.num_heads, self.head_dim],
        );
        let q = mlxcel_core::squeeze_axis(&q, 0);
        let k = mlxcel_core::squeeze_axis(&k, 0);
        let v = mlxcel_core::squeeze_axis(&v, 0);

        let q = apply_rotary_pos_emb_vision(&q, rotary_pos_emb);
        let k = apply_rotary_pos_emb_vision(&k, rotary_pos_emb);
        observe(VisionAttentionDiagnosticStage::Query, &q);
        observe(VisionAttentionDiagnosticStage::Key, &k);
        observe(VisionAttentionDiagnosticStage::Value, &v);

        // [seq, heads, head_dim] -> [1, heads, seq, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[1, 0, 2]);
        let k = mlxcel_core::transpose_axes(&k, &[1, 0, 2]);
        let v = mlxcel_core::transpose_axes(&v, &[1, 0, 2]);
        let q = mlxcel_core::expand_dims(&q, 0);
        let k = mlxcel_core::expand_dims(&k, 0);
        let v = mlxcel_core::expand_dims(&v, 0);

        // Per-segment attention
        let num_segments = cu_seqlens.len() - 1;
        let mut attn_outputs = Vec::with_capacity(num_segments);

        for seg in 0..num_segments {
            let start = cu_seqlens[seg];
            let end = cu_seqlens[seg + 1];

            let q_seg = mlxcel_core::slice(
                &q,
                &[0, 0, start, 0],
                &[1, self.num_heads, end, self.head_dim],
            );
            let k_seg = mlxcel_core::slice(
                &k,
                &[0, 0, start, 0],
                &[1, self.num_heads, end, self.head_dim],
            );
            let v_seg = mlxcel_core::slice(
                &v,
                &[0, 0, start, 0],
                &[1, self.num_heads, end, self.head_dim],
            );

            let attn = unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q_seg,
                    &k_seg,
                    &v_seg,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            };
            attn_outputs.push(attn);
        }

        let output = if attn_outputs.len() == 1 {
            attn_outputs.into_iter().next().unwrap()
        } else {
            concat_many(&attn_outputs, 2)
        };

        let output = mlxcel_core::squeeze_axis(&output, 0);
        let output = mlxcel_core::transpose_axes(&output, &[1, 0, 2]);
        let output = mlxcel_core::reshape(&output, &[seq_length, -1]);
        observe(VisionAttentionDiagnosticStage::Context, &output);

        self.proj.forward(&output)
    }
}

// Vision MLP - SwiGLU (gate_proj/up_proj/down_proj with SiLU).
struct VisionMLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

#[cfg(feature = "xla-diagnostics")]
#[derive(Clone, Copy)]
enum VisionMLPDiagnosticStage {
    GateProjection,
    GateActivation,
    UpProjection,
    GatedProduct,
}

impl VisionMLP {
    fn from_weights(weights: &WeightMap, prefix: &str, gs: i32, bits: i32) -> Result<Self, String> {
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.gate_proj", prefix),
                gs,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.up_proj", prefix),
                gs,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.down_proj", prefix),
                gs,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let gate = mlxcel_core::silu(&gate);
        let up = self.up_proj.forward(x);
        let h = mlxcel_core::multiply(&gate, &up);
        self.down_proj.forward(&h)
    }

    #[cfg(feature = "xla-diagnostics")]
    fn dense_f32_projection(linear: &UnifiedLinear, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
        let weight = linear.dequantized_weight();
        let weight = mlxcel_core::astype(&weight, mlxcel_core::dtype::FLOAT32);
        let weight = mlxcel_core::transpose(&weight);
        let mut output = mlxcel_core::matmul(&x, &weight);
        let (global_scale, bias) = match linear {
            UnifiedLinear::Quantized { weight, bias } => {
                (weight.global_scale.as_ref(), bias.as_ref())
            }
            UnifiedLinear::Regular(linear) => (None, linear.bias.as_ref()),
        };
        if let Some(global_scale) = global_scale {
            let global_scale = mlxcel_core::astype(global_scale, mlxcel_core::dtype::FLOAT32);
            output = mlxcel_core::multiply(&output, &global_scale);
        }
        if let Some(bias) = bias {
            let bias = mlxcel_core::astype(bias, mlxcel_core::dtype::FLOAT32);
            output = mlxcel_core::add(&output, &bias);
        }
        output
    }

    #[cfg(feature = "xla-diagnostics")]
    fn dense_f32_gate_up_controls(
        &self,
        x: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        (
            Self::dense_f32_projection(&self.gate_proj, x),
            Self::dense_f32_projection(&self.up_proj, x),
        )
    }

    #[cfg(feature = "xla-diagnostics")]
    fn forward_with_observer<F>(&self, x: &MlxArray, mut observe: F) -> UniquePtr<MlxArray>
    where
        F: FnMut(VisionMLPDiagnosticStage, &MlxArray),
    {
        let gate = self.gate_proj.forward(x);
        observe(VisionMLPDiagnosticStage::GateProjection, &gate);
        let gate = mlxcel_core::silu(&gate);
        observe(VisionMLPDiagnosticStage::GateActivation, &gate);
        let up = self.up_proj.forward(x);
        observe(VisionMLPDiagnosticStage::UpProjection, &up);
        let gated_product = mlxcel_core::multiply(&gate, &up);
        observe(VisionMLPDiagnosticStage::GatedProduct, &gated_product);
        self.down_proj.forward(&gated_product)
    }
}

// VisionBlock - RMSNorm + SwiGLU MLP.
struct VisionBlock {
    norm1: VisionRMSNorm,
    norm2: VisionRMSNorm,
    attn: VisionAttention,
    mlp: VisionMLP,
}

#[cfg(feature = "xla-diagnostics")]
struct Qwen25VLBlockDiagnostics {
    input: Vec<f32>,
    norm1: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    attention_context: Vec<f32>,
    attention: Vec<f32>,
    post_attention_residual: Vec<f32>,
    norm2: Vec<f32>,
    mlp_gate_projection: Vec<f32>,
    mlp_gate_projection_dense_f32_control: Vec<f32>,
    mlp_gate_activation: Vec<f32>,
    mlp_up_projection: Vec<f32>,
    mlp_up_projection_dense_f32_control: Vec<f32>,
    mlp_gated_product: Vec<f32>,
    mlp_down_projection: Vec<f32>,
}

#[cfg(any(test, feature = "xla-diagnostics"))]
fn host_f32_diagnostic_snapshot(array: &MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&array);
    let bytes = mlxcel_core::array_to_raw_bytes(&array);
    assert!(
        bytes.len().is_multiple_of(std::mem::size_of::<f32>()),
        "Qwen2.5-VL diagnostic F32 snapshot has a partial element"
    );
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

impl VisionBlock {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen25VLVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            norm1: VisionRMSNorm::from_weights(weights, &format!("{}.norm1", prefix), 1e-6)?,
            norm2: VisionRMSNorm::from_weights(weights, &format!("{}.norm2", prefix), 1e-6)?,
            attn: VisionAttention::from_weights(weights, config, prefix, gs, bits)?,
            mlp: VisionMLP::from_weights(weights, prefix, gs, bits)?,
        })
    }

    fn forward(
        &self,
        hidden_states: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let normed = self.norm1.forward(hidden_states);
        let attn_out = self.attn.forward(&normed, cu_seqlens, rotary_pos_emb);
        let h = mlxcel_core::add(hidden_states, &attn_out);
        let normed = self.norm2.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    #[cfg(feature = "xla-diagnostics")]
    fn forward_with_diagnostics(
        &self,
        hidden_states: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
    ) -> (UniquePtr<MlxArray>, Qwen25VLBlockDiagnostics) {
        // Copy each diagnostic value to host-owned F32 memory before its source
        // feeds the next substage. Retaining even an evaluated MlxArray handle
        // is insufficient because later graph evaluation may donate or reuse
        // its backing storage.
        let input = host_f32_diagnostic_snapshot(hidden_states);
        let norm1 = self.norm1.forward(hidden_states);
        let norm1_snapshot = host_f32_diagnostic_snapshot(&norm1);
        let mut query = None;
        let mut key = None;
        let mut value = None;
        let mut attention_context = None;
        let attention =
            self.attn
                .forward_with_observer(&norm1, cu_seqlens, rotary_pos_emb, |stage, array| {
                    let snapshot = host_f32_diagnostic_snapshot(array);
                    match stage {
                        VisionAttentionDiagnosticStage::Query => query = Some(snapshot),
                        VisionAttentionDiagnosticStage::Key => key = Some(snapshot),
                        VisionAttentionDiagnosticStage::Value => value = Some(snapshot),
                        VisionAttentionDiagnosticStage::Context => {
                            attention_context = Some(snapshot)
                        }
                    }
                });
        let attention_snapshot = host_f32_diagnostic_snapshot(&attention);
        let post_attention_residual = mlxcel_core::add(hidden_states, &attention);
        let post_attention_residual_snapshot =
            host_f32_diagnostic_snapshot(&post_attention_residual);
        let norm2 = self.norm2.forward(&post_attention_residual);
        let norm2_snapshot = host_f32_diagnostic_snapshot(&norm2);
        let (mlp_gate_projection_dense_f32_control, mlp_up_projection_dense_f32_control) =
            self.mlp.dense_f32_gate_up_controls(&norm2);
        let mlp_gate_projection_dense_f32_control =
            host_f32_diagnostic_snapshot(&mlp_gate_projection_dense_f32_control);
        let mlp_up_projection_dense_f32_control =
            host_f32_diagnostic_snapshot(&mlp_up_projection_dense_f32_control);
        let mut mlp_gate_projection = None;
        let mut mlp_gate_activation = None;
        let mut mlp_up_projection = None;
        let mut mlp_gated_product = None;
        let mlp_down_projection = self.mlp.forward_with_observer(&norm2, |stage, array| {
            let snapshot = host_f32_diagnostic_snapshot(array);
            match stage {
                VisionMLPDiagnosticStage::GateProjection => mlp_gate_projection = Some(snapshot),
                VisionMLPDiagnosticStage::GateActivation => mlp_gate_activation = Some(snapshot),
                VisionMLPDiagnosticStage::UpProjection => mlp_up_projection = Some(snapshot),
                VisionMLPDiagnosticStage::GatedProduct => mlp_gated_product = Some(snapshot),
            }
        });
        let mlp_down_projection_snapshot = host_f32_diagnostic_snapshot(&mlp_down_projection);
        let output = mlxcel_core::add(&post_attention_residual, &mlp_down_projection);
        (
            output,
            Qwen25VLBlockDiagnostics {
                input,
                norm1: norm1_snapshot,
                query: query.expect("diagnostics capture attention query"),
                key: key.expect("diagnostics capture attention key"),
                value: value.expect("diagnostics capture attention value"),
                attention_context: attention_context
                    .expect("diagnostics capture pre-projection attention context"),
                attention: attention_snapshot,
                post_attention_residual: post_attention_residual_snapshot,
                norm2: norm2_snapshot,
                mlp_gate_projection: mlp_gate_projection
                    .expect("diagnostics capture MLP gate projection"),
                mlp_gate_projection_dense_f32_control,
                mlp_gate_activation: mlp_gate_activation
                    .expect("diagnostics capture MLP gate activation"),
                mlp_up_projection: mlp_up_projection
                    .expect("diagnostics capture MLP up projection"),
                mlp_up_projection_dense_f32_control,
                mlp_gated_product: mlp_gated_product
                    .expect("diagnostics capture MLP gated product"),
                mlp_down_projection: mlp_down_projection_snapshot,
            },
        )
    }
}

// PatchMerger - RMSNorm + GELU MLP (projection to text hidden size).
struct PatchMerger {
    ln_q: VisionRMSNorm,
    mlp_0: UnifiedLinear,
    mlp_2: UnifiedLinear,
    hidden_size: usize,
}

impl PatchMerger {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        context_dim: usize,
        spatial_merge_size: usize,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let hidden_size = context_dim * spatial_merge_size * spatial_merge_size;
        Ok(Self {
            ln_q: VisionRMSNorm::from_weights(weights, &format!("{}.ln_q", prefix), 1e-6)?,
            mlp_0: UnifiedLinear::from_weights(weights, &format!("{}.mlp.0", prefix), gs, bits)?,
            mlp_2: UnifiedLinear::from_weights(weights, &format!("{}.mlp.2", prefix), gs, bits)?,
            hidden_size,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.ln_q.forward(x);
        let h = mlxcel_core::reshape(&h, &[-1, self.hidden_size as i32]);
        let h = self.mlp_0.forward(&h);
        let h = mlxcel_core::gelu(&h);
        self.mlp_2.forward(&h)
    }
}

// Qwen2.5-VL Vision Encoder.
/// Qwen2.5-VL Vision Model with windowed attention
///
/// Used by: Qwen2.5-VL
pub struct Qwen25VLVisionEncoder {
    patch_embed: PatchEmbed,
    rotary_pos_emb: VisionRotaryEmbedding,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    spatial_merge_size: usize,
    window_size: usize,
    patch_size: usize,
    fullatt_block_indexes: Vec<usize>,
}

#[cfg(feature = "xla-diagnostics")]
pub struct Qwen25VLVisionDiagnostics {
    pub window_index: Vec<i32>,
    pub restore_indices: Vec<i32>,
    pub packed_cu_seqlens: Vec<i32>,
    pub window_cu_seqlens: Vec<i32>,
    pub reordered_patch_embedding: UniquePtr<MlxArray>,
    pub window_layer_index: usize,
    pub post_window_layer: UniquePtr<MlxArray>,
    pub full_layer_indices: Vec<usize>,
    pub post_full_layers: Vec<UniquePtr<MlxArray>>,
    pub final_interval_layer_indices: Vec<usize>,
    pub post_final_interval_layers: Vec<UniquePtr<MlxArray>>,
    pub diagnostic_layer_indices: Vec<usize>,
    pub post_diagnostic_layers: Vec<UniquePtr<MlxArray>>,
    pub substage_probe_layer_index: usize,
    pub substage_probe_layer_input: Vec<f32>,
    pub substage_probe_layer_norm1: Vec<f32>,
    pub substage_probe_layer_query: Vec<f32>,
    pub substage_probe_layer_key: Vec<f32>,
    pub substage_probe_layer_value: Vec<f32>,
    pub substage_probe_layer_attention_context: Vec<f32>,
    pub substage_probe_layer_attention: Vec<f32>,
    pub substage_probe_layer_post_attention_residual: Vec<f32>,
    pub substage_probe_layer_norm2: Vec<f32>,
    pub substage_probe_layer_mlp_gate_projection: Vec<f32>,
    pub substage_probe_layer_mlp_gate_projection_dense_f32_control: Vec<f32>,
    pub substage_probe_layer_mlp_gate_activation: Vec<f32>,
    pub substage_probe_layer_mlp_up_projection: Vec<f32>,
    pub substage_probe_layer_mlp_up_projection_dense_f32_control: Vec<f32>,
    pub substage_probe_layer_mlp_gated_product: Vec<f32>,
    pub substage_probe_layer_mlp_down_projection: Vec<f32>,
    pub merger_window_ordered: UniquePtr<MlxArray>,
    pub restored_projection: UniquePtr<MlxArray>,
}

#[cfg(feature = "xla-diagnostics")]
type Qwen25VLCapture = Option<Qwen25VLVisionDiagnostics>;
#[cfg(not(feature = "xla-diagnostics"))]
type Qwen25VLCapture = ();

impl Qwen25VLVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Qwen25VLVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.quant_group_size;
        let bits = config.quant_bits;

        let patch_embed =
            PatchEmbed::from_weights(weights, config, &format!("{}.patch_embed", prefix))?;
        let head_dim = config.hidden_size / config.num_heads;
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2);

        let mut blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            blocks.push(VisionBlock::from_weights(
                weights,
                config,
                &format!("{}.blocks.{}", prefix, i),
                gs,
                bits,
            )?);
        }

        let merger = PatchMerger::from_weights(
            weights,
            &format!("{}.merger", prefix),
            config.hidden_size,
            config.spatial_merge_size,
            gs,
            bits,
        )?;

        Ok(Self {
            patch_embed,
            rotary_pos_emb,
            blocks,
            merger,
            spatial_merge_size: config.spatial_merge_size,
            window_size: config.window_size,
            patch_size: config.patch_size,
            fullatt_block_indexes: config.fullatt_block_indexes.clone(),
        })
    }

    /// Compute 2D rotary position embeddings from grid_thw (same as Qwen2-VL)
    fn rot_pos_emb(&self, grid_thw: &[(i32, i32, i32)]) -> UniquePtr<MlxArray> {
        let mut all_pos_ids: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut max_grid_dim: i32 = 0;

        for &(t, h, w) in grid_thw {
            if h > max_grid_dim {
                max_grid_dim = h;
            }
            if w > max_grid_dim {
                max_grid_dim = w;
            }
            let merge = self.spatial_merge_size as i32;

            let h_arange = mlxcel_core::arange_i32(0, h, 1);
            let h_col = mlxcel_core::reshape(&h_arange, &[h, 1]);
            let hpos = mlxcel_core::repeat(&h_col, w, 1);
            let hpos = mlxcel_core::reshape(&hpos, &[h / merge, merge, w / merge, merge]);
            let hpos = mlxcel_core::transpose_axes(&hpos, &[0, 2, 1, 3]);
            let hpos = mlxcel_core::flatten(&hpos);

            let w_arange = mlxcel_core::arange_i32(0, w, 1);
            let w_row = mlxcel_core::reshape(&w_arange, &[1, w]);
            let wpos = mlxcel_core::repeat(&w_row, h, 0);
            let wpos = mlxcel_core::reshape(&wpos, &[h / merge, merge, w / merge, merge]);
            let wpos = mlxcel_core::transpose_axes(&wpos, &[0, 2, 1, 3]);
            let wpos = mlxcel_core::flatten(&wpos);

            let stacked = mlxcel_core::stack_owned(&[hpos, wpos], -1);
            let tiled = mlxcel_core::tile(&stacked, &[t, 1]);
            all_pos_ids.push(tiled);
        }

        let pos_ids = if all_pos_ids.len() == 1 {
            all_pos_ids.into_iter().next().unwrap()
        } else {
            concat_many(&all_pos_ids, 0)
        };

        let rotary_table = self.rotary_pos_emb.forward(max_grid_dim);
        let pos_ids_flat = mlxcel_core::flatten(&pos_ids);
        let all_freqs = mlxcel_core::take(&rotary_table, &pos_ids_flat, 0);
        let total_shape = mlxcel_core::array_shape(&pos_ids);
        let total_tokens = total_shape[0];
        let freq_shape = mlxcel_core::array_shape(&all_freqs);
        let half_dim = freq_shape[1];
        let all_freqs = mlxcel_core::reshape(&all_freqs, &[total_tokens, 2, half_dim]);
        mlxcel_core::reshape(&all_freqs, &[total_tokens, 2 * half_dim])
    }

    /// Compute windowed attention indices and cu_seqlens
    ///
    /// Returns (window_index, cu_window_seqlens) where:
    /// - window_index: reordering of merged patches into window groups
    /// - cu_window_seqlens: cumulative sequence lengths for windowed attention
    fn get_window_index(&self, grid_thw: &[(i32, i32, i32)]) -> (Vec<i32>, Vec<i32>) {
        let spatial_merge_unit = (self.spatial_merge_size * self.spatial_merge_size) as i32;
        let vit_merger_window_size =
            (self.window_size / self.spatial_merge_size / self.patch_size) as i32;

        let mut window_index: Vec<i32> = Vec::new();
        let mut cu_window_seqlens: Vec<i32> = vec![0];
        let mut window_index_id: i32 = 0;

        for &(grid_t, grid_h, grid_w) in grid_thw {
            let llm_grid_h = grid_h / self.spatial_merge_size as i32;
            let llm_grid_w = grid_w / self.spatial_merge_size as i32;

            let total = grid_t * llm_grid_h * llm_grid_w;

            // Create index array [0..total)
            let index_3d: Vec<i32> = (0..total).collect();

            // Compute padding
            let pad_h = if llm_grid_h % vit_merger_window_size == 0 {
                0
            } else {
                vit_merger_window_size - llm_grid_h % vit_merger_window_size
            };
            let pad_w = if llm_grid_w % vit_merger_window_size == 0 {
                0
            } else {
                vit_merger_window_size - llm_grid_w % vit_merger_window_size
            };
            let num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
            let num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;
            let padded_h = llm_grid_h + pad_h;
            let padded_w = llm_grid_w + pad_w;

            // Pad index to [grid_t, padded_h, padded_w] with -100
            let mut index_padded = vec![-100i32; (grid_t * padded_h * padded_w) as usize];
            for ti in 0..grid_t {
                for hi in 0..llm_grid_h {
                    for wi in 0..llm_grid_w {
                        let src_idx =
                            (ti * llm_grid_h * llm_grid_w + hi * llm_grid_w + wi) as usize;
                        let dst_idx = (ti * padded_h * padded_w + hi * padded_w + wi) as usize;
                        index_padded[dst_idx] = index_3d[src_idx];
                    }
                }
            }

            // Reshape to [grid_t, num_windows_h, ws, num_windows_w, ws]
            // Then transpose to [grid_t, num_windows_h, num_windows_w, ws, ws]
            // Then reshape to [grid_t, num_windows_h*num_windows_w, ws, ws]
            let ws = vit_merger_window_size;
            let mut reordered =
                vec![-100i32; (grid_t * num_windows_h * num_windows_w * ws * ws) as usize];

            for ti in 0..grid_t {
                for wh in 0..num_windows_h {
                    for ww in 0..num_windows_w {
                        for sh in 0..ws {
                            for sw in 0..ws {
                                let src_h = wh * ws + sh;
                                let src_w = ww * ws + sw;
                                let src =
                                    (ti * padded_h * padded_w + src_h * padded_w + src_w) as usize;
                                let win_idx = wh * num_windows_w + ww;
                                let dst = (ti * num_windows_h * num_windows_w * ws * ws
                                    + win_idx * ws * ws
                                    + sh * ws
                                    + sw) as usize;
                                reordered[dst] = index_padded[src];
                            }
                        }
                    }
                }
            }

            // Compute seqlens per window (count non-padding entries)
            let num_windows = (grid_t * num_windows_h * num_windows_w) as usize;
            let ws2 = (ws * ws) as usize;
            let mut seqlens: Vec<i32> = Vec::with_capacity(num_windows);
            for win in 0..num_windows {
                let mut count = 0i32;
                for j in 0..ws2 {
                    if reordered[win * ws2 + j] != -100 {
                        count += 1;
                    }
                }
                seqlens.push(count);
            }

            // Extract non-padding indices in order
            let mut valid_indices: Vec<i32> = Vec::new();
            for &val in &reordered {
                if val != -100 {
                    valid_indices.push(val + window_index_id);
                }
            }
            window_index.extend_from_slice(&valid_indices);

            // Compute cu_window_seqlens
            let last_cum = *cu_window_seqlens.last().unwrap();
            let mut cum = last_cum;
            for &sl in &seqlens {
                cum += sl * spatial_merge_unit;
                cu_window_seqlens.push(cum);
            }

            window_index_id += total;
        }

        // Deduplicate cu_window_seqlens (remove consecutive duplicates)
        let mut deduped: Vec<i32> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &val in &cu_window_seqlens {
            if seen.insert(val) {
                deduped.push(val);
            }
        }

        (window_index, deduped)
    }

    /// Compute cu_seqlens for full attention from grid_thw
    /// Returns cumulative counts in pre-merge token space (multiplied by spatial_merge_unit)
    fn compute_cu_seqlens(grid_thw: &[(i32, i32, i32)], spatial_merge_size: i32) -> Vec<i32> {
        let spatial_merge_unit = spatial_merge_size * spatial_merge_size;
        let mut cu_seqlens = vec![0i32];
        let mut cumulative = 0i32;
        for &(t, h, w) in grid_thw {
            let merged_h = h / spatial_merge_size;
            let merged_w = w / spatial_merge_size;
            // Each merged patch corresponds to spatial_merge_unit pre-merge tokens
            let tokens_per_frame = merged_h * merged_w * spatial_merge_unit;
            for _ in 0..t {
                cumulative += tokens_per_frame;
                cu_seqlens.push(cumulative);
            }
        }
        cu_seqlens
    }

    /// Forward pass with windowed attention
    pub fn forward_with_grid(
        &self,
        hidden_states: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> VisionEncoderOutput {
        self.forward_with_grid_inner::<false>(hidden_states, grid_thw)
            .0
    }

    #[cfg(feature = "xla-diagnostics")]
    pub fn forward_with_grid_diagnostics(
        &self,
        hidden_states: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> Qwen25VLVisionDiagnostics {
        self.forward_with_grid_inner::<true>(hidden_states, grid_thw)
            .1
            .expect("Qwen2.5-VL diagnostics requested a capture")
    }

    fn forward_with_grid_inner<const CAPTURE: bool>(
        &self,
        hidden_states: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> (VisionEncoderOutput, Qwen25VLCapture) {
        let mut h = self.patch_embed.forward(hidden_states);
        let rotary_pos_emb = self.rot_pos_emb(grid_thw);

        let (window_index, cu_window_seqlens) = self.get_window_index(grid_thw);

        let spatial_merge_unit = (self.spatial_merge_size * self.spatial_merge_size) as i32;

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[0];
        let dim = shape[1];
        // Reorder hidden states by window index
        // [seq, dim] -> [seq/merge_unit, merge_unit, dim]
        let h_grouped =
            mlxcel_core::reshape(&h, &[seq_len / spatial_merge_unit, spatial_merge_unit, dim]);
        let window_idx_arr =
            mlxcel_core::from_slice_i32(&window_index, &[window_index.len() as i32]);
        let h_reordered = mlxcel_core::take(&h_grouped, &window_idx_arr, 0);
        h = mlxcel_core::reshape(&h_reordered, &[-1, dim]);
        #[cfg(feature = "xla-diagnostics")]
        let reordered_patch_embedding = CAPTURE
            .then(|| mlxcel_core::copy(h.as_ref().expect("Qwen2.5-VL reordered patch embedding")));

        // Reorder rotary_pos_emb similarly
        let rope_shape = mlxcel_core::array_shape(&rotary_pos_emb);
        let rope_dim = rope_shape[1];
        let rope_grouped = mlxcel_core::reshape(
            &rotary_pos_emb,
            &[seq_len / spatial_merge_unit, spatial_merge_unit, rope_dim],
        );
        let rope_reordered = mlxcel_core::take(&rope_grouped, &window_idx_arr, 0);
        let rotary_pos_emb = mlxcel_core::reshape(&rope_reordered, &[-1, rope_dim]);

        // Compute full-attention cu_seqlens
        let cu_seqlens = Self::compute_cu_seqlens(grid_thw, self.spatial_merge_size as i32);

        // Run blocks with windowed or full attention
        #[cfg(feature = "xla-diagnostics")]
        let window_layer_index =
            (0..self.blocks.len()).find(|layer| !self.fullatt_block_indexes.contains(layer));
        #[cfg(feature = "xla-diagnostics")]
        let mut post_window_layer = None;
        #[cfg(feature = "xla-diagnostics")]
        let mut post_full_layers = Vec::new();
        #[cfg(feature = "xla-diagnostics")]
        let full_layer_indices = (0..self.blocks.len())
            .filter(|layer| self.fullatt_block_indexes.contains(layer))
            .collect::<Vec<_>>();
        #[cfg(feature = "xla-diagnostics")]
        let target_full_layer_index = full_layer_indices.last().copied();
        #[cfg(feature = "xla-diagnostics")]
        let final_interval_start = full_layer_indices
            .iter()
            .rev()
            .nth(1)
            .map_or(0, |layer| layer + 1);
        #[cfg(feature = "xla-diagnostics")]
        let final_interval_layer_indices = target_full_layer_index
            .map(|target| (final_interval_start..=target).collect::<Vec<_>>())
            .unwrap_or_default();
        // The two-media CUDA oracle passes layer 16 and first diverges at layer
        // 17. Probe that first failing layer so the next bounded run
        // distinguishes Q/K/V, masked attention context, projection, residual,
        // norm, and MLP without changing production execution.
        #[cfg(feature = "xla-diagnostics")]
        let target_probe_boundary = full_layer_indices
            .iter()
            .rev()
            .nth(1)
            .copied()
            .or(target_full_layer_index);
        #[cfg(feature = "xla-diagnostics")]
        let substage_probe_layer_index = target_probe_boundary.map(|target| {
            full_layer_indices
                .iter()
                .copied()
                .take_while(|&layer| layer < target)
                .last()
                .and_then(|previous_full| previous_full.checked_add(2))
                .filter(|&layer| layer <= target)
                .unwrap_or(target)
        });
        #[cfg(feature = "xla-diagnostics")]
        let diagnostic_layer_indices = substage_probe_layer_index
            .and_then(|probe| {
                full_layer_indices
                    .iter()
                    .copied()
                    .take_while(|&layer| layer < probe)
                    .last()
                    .map(|previous_full| ((previous_full + 1)..=probe).collect::<Vec<_>>())
            })
            .unwrap_or_default();
        #[cfg(feature = "xla-diagnostics")]
        let mut post_final_interval_layers = Vec::new();
        #[cfg(feature = "xla-diagnostics")]
        let mut post_diagnostic_layers = Vec::new();
        #[cfg(feature = "xla-diagnostics")]
        let mut substage_probe_layer_state = None;
        for (layer_num, block) in self.blocks.iter().enumerate() {
            let cu_seqlens_now = if self.fullatt_block_indexes.contains(&layer_num) {
                &cu_seqlens
            } else {
                &cu_window_seqlens
            };
            #[cfg(feature = "xla-diagnostics")]
            if CAPTURE && Some(layer_num) == substage_probe_layer_index {
                let (output, capture) =
                    block.forward_with_diagnostics(&h, cu_seqlens_now, &rotary_pos_emb);
                h = output;
                substage_probe_layer_state = Some(capture);
            } else {
                h = block.forward(&h, cu_seqlens_now, &rotary_pos_emb);
            }
            #[cfg(not(feature = "xla-diagnostics"))]
            {
                h = block.forward(&h, cu_seqlens_now, &rotary_pos_emb);
            }
            #[cfg(feature = "xla-diagnostics")]
            if CAPTURE {
                if Some(layer_num) == window_layer_index {
                    post_window_layer = Some(mlxcel_core::copy(
                        h.as_ref().expect("Qwen2.5-VL window layer state"),
                    ));
                }
                if self.fullatt_block_indexes.contains(&layer_num) {
                    post_full_layers.push(mlxcel_core::copy(
                        h.as_ref().expect("Qwen2.5-VL full layer state"),
                    ));
                }
                if final_interval_layer_indices.contains(&layer_num) {
                    post_final_interval_layers.push(mlxcel_core::copy(
                        h.as_ref().expect("Qwen2.5-VL final interval layer state"),
                    ));
                }
                if diagnostic_layer_indices.contains(&layer_num) {
                    post_diagnostic_layers.push(mlxcel_core::copy(
                        h.as_ref().expect("Qwen2.5-VL diagnostic layer state"),
                    ));
                }
            }
        }

        // Merge patches
        h = self.merger.forward(&h);
        #[cfg(feature = "xla-diagnostics")]
        let merger_window_ordered = CAPTURE.then(|| {
            mlxcel_core::copy(h.as_ref().expect("Qwen2.5-VL window-ordered merger output"))
        });

        // Un-reorder: destination original_position reads its corresponding
        // window_position. This is the inverse permutation, not window_index
        // itself (the two differ for non-self-inverse window layouts).
        let reverse_indices = restoration_indices(&window_index);

        // After merger: h has shape [total_merged, out_hidden_size]
        // reverse_indices maps from window-ordered to original order
        let reverse_arr =
            mlxcel_core::from_slice_i32(&reverse_indices, &[reverse_indices.len() as i32]);
        h = mlxcel_core::take(&h, &reverse_arr, 0);

        #[cfg(feature = "xla-diagnostics")]
        if CAPTURE {
            let substage_probe_layer_index = substage_probe_layer_index
                .expect("Qwen2.5-VL diagnostics require a substage probe layer");
            let substage_probe_layer_state =
                substage_probe_layer_state.expect("Qwen2.5-VL substage probe layer capture");
            let restored_projection =
                mlxcel_core::copy(h.as_ref().expect("Qwen2.5-VL restored projection output"));
            let output = VisionEncoderOutput { hidden_states: h };
            return (
                output,
                Some(Qwen25VLVisionDiagnostics {
                    window_index,
                    restore_indices: reverse_indices,
                    packed_cu_seqlens: cu_seqlens,
                    window_cu_seqlens: cu_window_seqlens,
                    reordered_patch_embedding: reordered_patch_embedding
                        .expect("Qwen2.5-VL patch capture"),
                    window_layer_index: window_layer_index
                        .expect("Qwen2.5-VL diagnostics require a window layer"),
                    post_window_layer: post_window_layer.expect("Qwen2.5-VL window layer capture"),
                    full_layer_indices,
                    post_full_layers,
                    final_interval_layer_indices,
                    post_final_interval_layers,
                    diagnostic_layer_indices,
                    post_diagnostic_layers,
                    substage_probe_layer_index,
                    substage_probe_layer_input: substage_probe_layer_state.input,
                    substage_probe_layer_norm1: substage_probe_layer_state.norm1,
                    substage_probe_layer_query: substage_probe_layer_state.query,
                    substage_probe_layer_key: substage_probe_layer_state.key,
                    substage_probe_layer_value: substage_probe_layer_state.value,
                    substage_probe_layer_attention_context: substage_probe_layer_state
                        .attention_context,
                    substage_probe_layer_attention: substage_probe_layer_state.attention,
                    substage_probe_layer_post_attention_residual: substage_probe_layer_state
                        .post_attention_residual,
                    substage_probe_layer_norm2: substage_probe_layer_state.norm2,
                    substage_probe_layer_mlp_gate_projection: substage_probe_layer_state
                        .mlp_gate_projection,
                    substage_probe_layer_mlp_gate_projection_dense_f32_control:
                        substage_probe_layer_state.mlp_gate_projection_dense_f32_control,
                    substage_probe_layer_mlp_gate_activation: substage_probe_layer_state
                        .mlp_gate_activation,
                    substage_probe_layer_mlp_up_projection: substage_probe_layer_state
                        .mlp_up_projection,
                    substage_probe_layer_mlp_up_projection_dense_f32_control:
                        substage_probe_layer_state.mlp_up_projection_dense_f32_control,
                    substage_probe_layer_mlp_gated_product: substage_probe_layer_state
                        .mlp_gated_product,
                    substage_probe_layer_mlp_down_projection: substage_probe_layer_state
                        .mlp_down_projection,
                    merger_window_ordered: merger_window_ordered
                        .expect("Qwen2.5-VL merger capture"),
                    restored_projection,
                }),
            );
        }
        #[cfg(not(feature = "xla-diagnostics"))]
        {
            (VisionEncoderOutput { hidden_states: h }, ())
        }
        #[cfg(feature = "xla-diagnostics")]
        {
            (VisionEncoderOutput { hidden_states: h }, None)
        }
    }
}

/// VisionEncoder trait - panics since grid_thw is required
impl super::VisionEncoder for Qwen25VLVisionEncoder {
    fn forward(&self, _pixel_values: &MlxArray) -> VisionEncoderOutput {
        panic!("Qwen2.5-VL vision encoder requires grid_thw; use forward_with_grid() instead");
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "xla-diagnostics")]
    use super::VisionMLP;
    use super::{host_f32_diagnostic_snapshot, restoration_indices};

    #[test]
    fn restoration_is_the_true_inverse_for_non_self_inverse_permutation() {
        let window_index = [2, 0, 3, 1];
        let restore = restoration_indices(&window_index);
        assert_eq!(restore, vec![1, 3, 0, 2]);
        let window_ordered = ["two", "zero", "three", "one"];
        let original = restore
            .into_iter()
            .map(|index| window_ordered[index as usize])
            .collect::<Vec<_>>();
        assert_eq!(original, vec!["zero", "one", "two", "three"]);
    }

    #[cfg(feature = "xla-diagnostics")]
    #[test]
    fn dense_f32_projection_control_preserves_linear_order_and_dtype() {
        use mlxcel_core::layers::{Linear, UnifiedLinear};

        let input = mlxcel_core::from_slice_f32(&[1.0, 2.0, -1.0, 0.5], &[2, 2]);
        let input = mlxcel_core::astype(&input, mlxcel_core::dtype::FLOAT16);
        let weight = mlxcel_core::from_slice_f32(&[3.0, 4.0, -2.0, 5.0], &[2, 2]);
        let weight = mlxcel_core::astype(&weight, mlxcel_core::dtype::FLOAT16);
        let bias = mlxcel_core::from_slice_f32(&[0.5, -1.0], &[2]);
        let bias = mlxcel_core::astype(&bias, mlxcel_core::dtype::FLOAT16);
        let linear = UnifiedLinear::Regular(Linear::new(weight, Some(bias)));

        let output = VisionMLP::dense_f32_projection(&linear, &input);
        assert_eq!(
            mlxcel_core::array_dtype(&output),
            mlxcel_core::dtype::FLOAT32
        );
        let output = host_f32_diagnostic_snapshot(&output);
        assert_eq!(output, vec![11.5, 7.0, -0.5, 3.5]);
    }

    #[test]
    fn host_snapshot_preserves_nonzero_rms_norm_after_later_graph_evaluation() {
        let input = mlxcel_core::from_slice_f32(&[1.0, -2.0, 3.0, -4.0], &[1, 4]);
        let weight = mlxcel_core::ones(&[4], mlxcel_core::dtype::FLOAT32);
        let normalized = mlxcel_core::rms_norm(&input, &weight, 1e-6);
        let snapshot = host_f32_diagnostic_snapshot(&normalized);

        assert!(
            snapshot
                .iter()
                .any(|value| value.is_finite() && *value != 0.0),
            "RMSNorm snapshot must be non-vacuous"
        );

        let residual = mlxcel_core::add(&input, &normalized);
        let donated_candidate = mlxcel_core::multiply(&residual, &residual);
        mlxcel_core::eval(&donated_candidate);

        let expected = [0.365_148_37, -0.730_296_73, 1.095_445_2, -1.460_593_5];
        assert_eq!(snapshot.len(), expected.len());
        for (index, (actual, expected)) in snapshot.iter().zip(expected).enumerate() {
            assert!(
                (*actual - expected).abs() <= 2.0e-6,
                "host snapshot changed at index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn f16_manual_rms_norm_characterizes_square_overflow() {
        let input = mlxcel_core::from_slice_f32(&[300.0, 1.0, -2.0, 3.0], &[1, 4]);
        let input = mlxcel_core::astype(&input, mlxcel_core::dtype::FLOAT16);
        let weight = mlxcel_core::ones(&[4], mlxcel_core::dtype::FLOAT16);

        let manual = mlxcel_core::rms_norm(&input, &weight, 1e-6);
        let manual = host_f32_diagnostic_snapshot(&manual);
        assert!(
            manual.iter().all(|value| *value == 0.0),
            "the current F16 square/mean path should expose its overflow signature"
        );

        let fast = mlxcel_core::fast_rms_norm(&input, &weight, 1e-6);
        let fast = host_f32_diagnostic_snapshot(&fast);
        assert!(
            fast.iter().any(|value| value.is_finite() && *value != 0.0),
            "the production fast RMSNorm must accumulate F16 squares in F32"
        );

        let promoted = mlxcel_core::astype(&input, mlxcel_core::dtype::FLOAT32);
        let promoted_weight = mlxcel_core::astype(&weight, mlxcel_core::dtype::FLOAT32);
        let promoted = mlxcel_core::rms_norm(&promoted, &promoted_weight, 1e-6);
        let promoted = host_f32_diagnostic_snapshot(&promoted);
        assert!(
            promoted
                .iter()
                .any(|value| value.is_finite() && *value != 0.0),
            "F32 square/mean must preserve a non-zero normalized row"
        );
        for (index, (fast, promoted)) in fast.iter().zip(promoted).enumerate() {
            assert!(
                (*fast - promoted).abs() <= 2.0e-3,
                "production fast RMSNorm diverged from F32 accumulation at {index}: fast={fast}, promoted={promoted}"
            );
        }
    }
}
