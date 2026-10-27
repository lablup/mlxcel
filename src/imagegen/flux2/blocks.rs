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

//! FLUX.2 double-stream and single-stream transformer blocks.
//!
//! Ports of `transformer_block.py` and `single_transformer_block.py` in
//! <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/flux2/transformer>.
//! The blocks own no modulation weights: the transformer computes the shared
//! `(shift, scale, gate)` sets once per call and hands them to every block.
//!
//! Used by: Flux2, Bonsai.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::attention::{JointAttention, ParallelSelfAttention};
use super::embeddings::{FeedForward, LinearSource, ModParams, layer_norm, modulate};
use super::kv_cache::{KvCacheMode, KvLayerCache};

/// `x + gate * y`.
fn gated(x: &MlxArray, gate: &MlxArray, y: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::add(x, &mlxcel_core::multiply(gate, y))
}

/// `Flux2TransformerBlock`: joint attention over text and image streams, then
/// a SwiGLU feed-forward per stream.
pub struct DoubleStreamBlock {
    attn: JointAttention,
    ff: FeedForward,
    ff_context: FeedForward,
}

impl DoubleStreamBlock {
    pub fn load(
        weights: &WeightMap,
        source: &dyn LinearSource,
        prefix: &str,
        heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        Ok(Self {
            attn: JointAttention::load(
                weights,
                source,
                &format!("{prefix}.attn"),
                heads,
                head_dim,
            )?,
            ff: FeedForward::load(weights, source, &format!("{prefix}.ff"))?,
            ff_context: FeedForward::load(weights, source, &format!("{prefix}.ff_context"))?,
        })
    }

    /// Returns `(encoder_hidden_states, hidden_states)`, as upstream.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden: &MlxArray,
        encoder: &MlxArray,
        mod_img: &[ModParams],
        mod_txt: &[ModParams],
        rope: (&MlxArray, &MlxArray),
        mode: Option<KvCacheMode>,
        cache: Option<&mut KvLayerCache>,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        let [shift_msa, scale_msa, gate_msa] = &mod_img[0];
        let [shift_mlp, scale_mlp, gate_mlp] = &mod_img[1];
        let [c_shift_msa, c_scale_msa, c_gate_msa] = &mod_txt[0];
        let [c_shift_mlp, c_scale_mlp, c_gate_mlp] = &mod_txt[1];

        let norm_hidden = modulate(&layer_norm(hidden), shift_msa, scale_msa);
        let norm_encoder = modulate(&layer_norm(encoder), c_shift_msa, c_scale_msa);
        let (attn_out, encoder_attn_out) =
            self.attn
                .forward(&norm_hidden, &norm_encoder, rope, mode, cache)?;

        let hidden = gated(hidden, gate_msa, &attn_out);
        let encoder = gated(encoder, c_gate_msa, &encoder_attn_out);

        let norm_hidden = modulate(&layer_norm(&hidden), shift_mlp, scale_mlp);
        let hidden = gated(&hidden, gate_mlp, &self.ff.forward(&norm_hidden));

        let norm_encoder = modulate(&layer_norm(&encoder), c_shift_mlp, c_scale_mlp);
        let encoder = gated(
            &encoder,
            c_gate_mlp,
            &self.ff_context.forward(&norm_encoder),
        );
        Ok((encoder, hidden))
    }
}

/// `Flux2SingleTransformerBlock`: parallel attention and MLP over the joined
/// `[text, image]` sequence.
pub struct SingleStreamBlock {
    attn: ParallelSelfAttention,
}

impl SingleStreamBlock {
    pub fn load(
        weights: &WeightMap,
        source: &dyn LinearSource,
        prefix: &str,
        heads: i32,
        head_dim: i32,
        mlp_hidden_dim: i32,
    ) -> Result<Self> {
        Ok(Self {
            attn: ParallelSelfAttention::load(
                weights,
                source,
                &format!("{prefix}.attn"),
                heads,
                head_dim,
                mlp_hidden_dim,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden: &MlxArray,
        mod_params: &ModParams,
        rope: (&MlxArray, &MlxArray),
        num_txt: i32,
        compute_dtype: i32,
        mode: Option<KvCacheMode>,
        cache: Option<&mut KvLayerCache>,
    ) -> Result<UniquePtr<MlxArray>> {
        let [shift, scale, gate] = mod_params;
        let norm = modulate(&layer_norm(hidden), shift, scale);
        let attn = self
            .attn
            .forward(&norm, rope, num_txt, compute_dtype, mode, cache)?;
        Ok(gated(hidden, gate, &attn))
    }
}
