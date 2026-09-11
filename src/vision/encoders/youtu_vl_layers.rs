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

//! Per-block primitives for the Youtu-VL vision encoder.
//!
//! `VisionEmbeddings`, `VisionAttention`, `VisionMLP`, and `EncoderLayer` are
//! defined here so the encoder file can stay small and focused on the
//! windowed-forward orchestration. Exposed at `pub(super)` so the encoder
//! body can compose them without any external crate having to know they
//! exist.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::YoutuVisionConfig;
use super::rope::apply_rotary_pos_emb_vision;

/// Patch embeddings — Linear over flattened patches (no positional embedding
/// stored on the module; vision RoPE is applied inside the encoder).
pub(super) struct VisionEmbeddings {
    patch_embedding: UnifiedLinear,
    embed_dim: i32,
}

impl VisionEmbeddings {
    pub(super) fn from_weights(
        weights: &WeightMap,
        config: &YoutuVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let patch_embedding =
            UnifiedLinear::from_weights(weights, &format!("{}.patch_embedding", prefix), gs, bits)?;
        Ok(Self {
            patch_embedding,
            embed_dim: config.hidden_size as i32,
        })
    }

    /// Input: `pixel_values` of shape `[batch, num_patches, patch_dim]` (or
    /// any rank where the last dim is `patch_dim`). Output: `[total_patches,
    /// embed_dim]`.
    pub(super) fn forward(&self, pixel_values: &MlxArray) -> UniquePtr<MlxArray> {
        let projected = self.patch_embedding.forward(pixel_values);
        mlxcel_core::reshape(&projected, &[-1, self.embed_dim])
    }
}

/// Vision attention with windowed segments via cu_seqlens.
pub(super) struct VisionAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    pub(super) fn from_weights(
        weights: &WeightMap,
        config: &YoutuVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let q_proj = UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), gs, bits)?;
        let k_proj = UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), gs, bits)?;
        let v_proj = UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), gs, bits)?;
        let out_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.out_proj", prefix), gs, bits)?;

        let head_dim = (config.hidden_size / config.num_attention_heads) as i32;
        let scale = (head_dim as f32).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: config.num_attention_heads as i32,
            head_dim,
            scale,
        })
    }

    pub(super) fn forward(
        &self,
        hidden_states: &MlxArray,
        cu_seqlens: &[i32],
        cos: &MlxArray,
        sin: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden_states);
        let seq_length = shape[0];

        let q = self.q_proj.forward(hidden_states);
        let k = self.k_proj.forward(hidden_states);
        let v = self.v_proj.forward(hidden_states);

        // Reshape to [seq_len, num_heads, head_dim] before applying RoPE.
        let q = mlxcel_core::reshape(&q, &[seq_length, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[seq_length, self.num_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[seq_length, self.num_heads, self.head_dim]);

        let (q, k) = apply_rotary_pos_emb_vision(&q, &k, cos, sin);

        // Transpose to [num_heads, seq_len, head_dim] for windowed SDPA.
        let q = mlxcel_core::transpose_axes(&q, &[1, 0, 2]);
        let k = mlxcel_core::transpose_axes(&k, &[1, 0, 2]);
        let v = mlxcel_core::transpose_axes(&v, &[1, 0, 2]);

        // Window-by-window SDPA. For each consecutive (start, end) pair we
        // gather the window slice, expand to a batched 4-D shape for SDPA,
        // and concat the per-window outputs along the seq dim.
        // M4: guard against an empty cu_seqlens slice — a caller that passes
        // fewer than two entries would produce zero segments and an empty
        // attn_outputs vec, causing the fold below to panic.
        assert!(
            cu_seqlens.len() >= 2,
            "VisionAttention::forward: cu_seqlens must have at least 2 entries (start + end); \
             got {} — did the window index computation produce an empty seqlen table?",
            cu_seqlens.len()
        );
        let num_segments = cu_seqlens.len() - 1;
        let mut attn_outputs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(num_segments);

        for seg in 0..num_segments {
            let start = cu_seqlens[seg];
            let end = cu_seqlens[seg + 1];

            // [num_heads, win_len, head_dim]
            let q_win =
                mlxcel_core::slice(&q, &[0, start, 0], &[self.num_heads, end, self.head_dim]);
            let k_win =
                mlxcel_core::slice(&k, &[0, start, 0], &[self.num_heads, end, self.head_dim]);
            let v_win =
                mlxcel_core::slice(&v, &[0, start, 0], &[self.num_heads, end, self.head_dim]);

            // Add a leading batch dim for `attention_from_ptr` (which expects 4D).
            let q_win = mlxcel_core::expand_dims(&q_win, 0);
            let k_win = mlxcel_core::expand_dims(&k_win, 0);
            let v_win = mlxcel_core::expand_dims(&v_win, 0);

            let attn = unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q_win,
                    &k_win,
                    &v_win,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            };
            // attn: [1, num_heads, win_len, head_dim] → [num_heads, win_len, head_dim]
            let attn = mlxcel_core::squeeze_axis(&attn, 0);
            attn_outputs.push(attn);
        }

        // Concat along the seq dim (axis=1 in [num_heads, seq, head_dim]).
        // attn_outputs is non-empty because num_segments >= 1 (guarded above).
        let concatenated = if attn_outputs.len() == 1 {
            attn_outputs.into_iter().next().unwrap()
        } else {
            let mut iter = attn_outputs.into_iter();
            let first = iter.next().unwrap();
            iter.fold(first, |acc, next| mlxcel_core::concatenate(&acc, &next, 1))
        };

        // [num_heads, seq, head_dim] → [seq, num_heads, head_dim] → [seq, hidden]
        let output = mlxcel_core::transpose_axes(&concatenated, &[1, 0, 2]);
        let output = mlxcel_core::reshape(&output, &[seq_length, -1]);
        self.out_proj.forward(&output)
    }
}

/// Vision MLP — `fc1 → GELU(tanh approx) → fc2`.
pub(super) struct VisionMLP {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl VisionMLP {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let fc1 = UnifiedLinear::from_weights(weights, &format!("{}.fc1", prefix), gs, bits)?;
        let fc2 = UnifiedLinear::from_weights(weights, &format!("{}.fc2", prefix), gs, bits)?;
        Ok(Self { fc1, fc2 })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.fc1.forward(x);
        // gelu_pytorch_tanh — matches `nn.gelu_approx` in upstream MLX usage.
        let x = mlxcel_core::utils::gelu_approx(&x);
        self.fc2.forward(&x)
    }
}

/// Encoder block (LayerNorm → MHA → residual → LayerNorm → MLP → residual).
pub(super) struct EncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: VisionAttention,
    layer_norm2: LayerNorm,
    mlp: VisionMLP,
}

impl EncoderLayer {
    pub(super) fn from_weights(
        weights: &WeightMap,
        config: &YoutuVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let layer_norm1 = load_layer_norm(
            weights,
            &format!("{}.layer_norm1", prefix),
            config.layer_norm_eps,
        )?;
        let layer_norm2 = load_layer_norm(
            weights,
            &format!("{}.layer_norm2", prefix),
            config.layer_norm_eps,
        )?;
        let self_attn = VisionAttention::from_weights(
            weights,
            config,
            &format!("{}.self_attn", prefix),
            gs,
            bits,
        )?;
        let mlp = VisionMLP::from_weights(weights, &format!("{}.mlp", prefix), gs, bits)?;
        Ok(Self {
            layer_norm1,
            self_attn,
            layer_norm2,
            mlp,
        })
    }

    pub(super) fn forward(
        &self,
        hidden_states: &MlxArray,
        cu_seqlens: &[i32],
        cos: &MlxArray,
        sin: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let normed = self.layer_norm1.forward(hidden_states);
        let attn_out = self.self_attn.forward(&normed, cu_seqlens, cos, sin);
        let h = mlxcel_core::add(hidden_states, &attn_out);

        let normed = self.layer_norm2.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }
}

pub(super) fn load_layer_norm(
    weights: &WeightMap,
    prefix: &str,
    eps: f32,
) -> Result<LayerNorm, String> {
    let weight_key = format!("{}.weight", prefix);
    let bias_key = format!("{}.bias", prefix);

    let weight = weights
        .get(&weight_key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", weight_key))?;
    let bias = weights.get(&bias_key).map(|w| mlxcel_core::copy(w));
    Ok(LayerNorm::new(weight, bias, eps))
}
