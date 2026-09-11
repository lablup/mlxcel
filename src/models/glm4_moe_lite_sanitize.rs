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

//! Weight-map sanitization for GLM4 MoE Lite.
//!
//! Split out of `glm4_moe_lite.rs` rather than appended to it, the same way
//! `youtu_vl_lm_sanitize.rs` was split out of `youtu_vl_lm.rs`: the runtime
//! module is already well past the 500-line target and this keeps the new code
//! from pushing it further. [`sanitize_weights`] is re-exported from
//! `glm4_moe_lite` so call sites name it the way the other MLA families do.

use mlxcel_core::mla::{KvBProjGeometry, decompose_kv_b_proj};
use mlxcel_core::weights::WeightMap;

use super::ModelArgs;

/// The `kv_b_proj` geometry of one `glm4_moe_lite` config, shared by the
/// decoder-layer sanitizer and the MTP drafter's sanitizer so the two cannot
/// disagree on which head dimension is which.
///
/// Used by: [`sanitize_weights`], `glm4_moe_lite_mtp_drafter`.
pub fn kv_b_proj_geometry(args: &ModelArgs) -> KvBProjGeometry {
    KvBProjGeometry {
        num_heads: args.num_attention_heads,
        qk_nope_head_dim: args.qk_nope_head_dim,
        v_head_dim: args.v_head_dim,
        kv_lora_rank: args.kv_lora_rank,
    }
}

/// Decompose `kv_b_proj` into the per-head `embed_q` / `unembed_out` pair that
/// `MlaAttention::from_weights` loads.
///
/// Every public `glm4_moe_lite` checkpoint stores the MLA up-projection as a
/// single `self_attn.kv_b_proj.weight` and ships no `embed_q` tensor at all, so
/// without this step the family cannot load the canonical layout of its own
/// architecture: `MultiLinear::from_weights` fails with `Weight not found:
/// model.layers.0.self_attn.embed_q.weight` (issue #1029). The other five MLA
/// families in the tree (DeepSeek V3, DeepSeek V3.2, Kimi Linear, LongCat Flash
/// NGram, Youtu-VL) all carry the same decomposition; this one is adapted from
/// `youtu_vl_lm_sanitize.rs`, the only one that cross-checks the tensor shape
/// against the config before reshaping.
///
/// The per-prefix decomposition itself is
/// [`mlxcel_core::mla::decompose_kv_b_proj`], shared with the `mlxcel
/// split-mtp` surgery op so the drafter's next-token block is split exactly
/// the way the decoder layers are (issue #1326). Only the decoder layers
/// `0..num_hidden_layers` are visited: a raw checkpoint's
/// `model.layers.{num_hidden_layers}.*` next-token-prediction tensors are left
/// untouched and unused by the target loader.
///
/// Returns `Err` rather than panicking or aborting when a `kv_b_proj` cannot be
/// decomposed: scales with no biases, a solved quantization pair no packing can
/// describe, or a tensor whose shape disagrees with `config.json`.
///
/// Used by: [`super::Glm4MoeLiteModel::load`].
pub fn sanitize_weights(mut weights: WeightMap, args: &ModelArgs) -> Result<WeightMap, String> {
    let geometry = kv_b_proj_geometry(args);
    for layer_idx in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{layer_idx}.self_attn");
        decompose_kv_b_proj(
            &mut weights,
            &prefix,
            geometry,
            &format!("layer {layer_idx}"),
        )?;
    }
    Ok(weights)
}

#[cfg(test)]
#[path = "glm4_moe_lite_sanitize_tests.rs"]
mod tests;
