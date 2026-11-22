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

//! Tencent Hy4 (`model_type: hy_v4`, `HYV4ForCausalLM`).
//!
//! A DeepSeek-V3.2 derivative: MLA with absorbed `embed_q` / `unembed_out`,
//! the DSA lightning indexer and a noaux_tc MoE, plus
//!
//! * an identity hyper-connection residual of width `hc_mult`
//!   ([`hyper`], no Sinkhorn mixing);
//! * a per-head sigmoid output gate and learnable attention sinks
//!   ([`attention`]);
//! * indexer sharing: `indexer_types[i]` is `full` (the layer owns an indexer
//!   and caches its key beside the latent) or `shared` (no indexer, no
//!   indexer key in the cache; the layer reuses the selection of the nearest
//!   earlier `full` layer).
//!
//! This module is declared as a child of `deepseek_v32` so it can reuse that
//! module's private MLA, indexer and MoE pieces and their loaders without
//! widening their visibility or changing the V3.2 forward.
//!
//! Scope: the mlx-community affine conversion, which ships `embed_q` /
//! `unembed_out` and stacked `switch_mlp.{gate,up,down}_proj`. The original
//! mxfp8 `tencent/Hy4-preview` layout (`kv_b_proj`, fused `gate_up_proj`) and
//! the `hy_v4_mtp` drafter are refused or ignored, not converted.
//!
//! Decisions recorded against the two references, upstream mlx-vlm
//! (`mlx_vlm/models/hy_v4`, the parity target) and the transformers
//! implementation (`modeling_hy_v4.py`):
//!
//! * `swiglu_limit` (10.0 in the checkpoint config): transformers clamps the
//!   routed experts with it; mlx-vlm applies nothing. Applied here, to the
//!   routed experts only, because it is part of the model as trained. The
//!   `no_swiglu_limit` arm of `MLXCEL_HY4_ARM` reproduces mlx-vlm.
//! * `enable_lm_head_fp32`: transformers keeps `lm_head` float32. The MLX
//!   checkpoint ships `lm_head` 4-bit with bf16 scales and mlx-vlm runs it in
//!   the activation dtype; so does this port. Not applied.
//! * RoPE pairing: mlx-vlm rotates interleaved pairs (`traditional=True`,
//!   inherited from its DeepSeek-V3.2 attention), for the rope half and for
//!   the first `qk_rope_head_dim` dims of the indexer head. transformers
//!   rotates half-split pairs, and on the last `qk_rope_head_dim` dims of the
//!   indexer head. The default follows mlx-vlm. The `rope_half_split` arm
//!   switches the pairing (not the indexer placement), so it matches
//!   transformers only while `kv_len <= index_topk`. Which one the checkpoint
//!   was trained with is to be settled on the real weights.
//! * Router: scored in float32 as both references do ([`moe`]).
//!
//! Batched decode uses the per-row fallback: every attention kernel here is
//! per sequence (the sinks rule out the shared batched MLA decode rows), so a
//! native batch would only share the projections and the MoE.

#[path = "hy_v4_attention.rs"]
mod attention;
#[path = "hy_v4_config.rs"]
mod config;
#[path = "hy_v4_hyper.rs"]
mod hyper;
#[path = "hy_v4_moe.rs"]
mod moe;

#[cfg(test)]
#[path = "hy_v4_tests.rs"]
pub(crate) mod hy_v4_tests;

#[cfg(test)]
#[path = "hy_v4_storage_tests.rs"]
mod storage_tests;

use std::path::Path;

use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) use self::config::ParityArms;
pub use self::config::{HyV4Config, IndexerType};

use self::attention::HyV4Attention;
use self::hyper::{IdentityHyperConnection, IdentityHyperHead, hc_expand};
use super::{
    DenseMLP, MoEBlock, ModelArgs, get_weight_copy, load_dense_mlp, load_mla_attention,
    load_moe_block,
};

enum Mlp {
    Dense(DenseMLP),
    MoE(MoEBlock),
}

struct DecoderLayer {
    self_attn: HyV4Attention,
    mlp: Mlp,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    hc_attn: IdentityHyperConnection,
    hc_mlp: IdentityHyperConnection,
    swiglu_limit: Option<f32>,
}

impl DecoderLayer {
    /// `h` is the widened residual `[B, L, hc, D]`.
    fn forward(
        &self,
        h: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
        topk: &mut Option<UniquePtr<MlxArray>>,
    ) -> UniquePtr<MlxArray> {
        let (x, post) = self.hc_attn.forward(h);
        let x = self.input_layernorm.forward(&x);
        let x = self.self_attn.forward(&x, mask, cache, topk);
        let h = hc_expand(&x, h, &post);

        let (x, post) = self.hc_mlp.forward(&h);
        let x = self.post_attention_layernorm.forward(&x);
        let x = match &self.mlp {
            Mlp::Dense(mlp) => mlp.forward(&x),
            Mlp::MoE(block) => moe::moe_forward(block, &x, self.swiglu_limit),
        };
        hc_expand(&x, &h, &post)
    }
}

/// Hy4 text model.
pub struct HyV4Model {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    hc_head: IdentityHyperHead,
    norm: RMSNorm,
    lm_head: Option<UnifiedLinear>,
    hc_mult: i32,
    eos_token_ids: Vec<i32>,
}

impl HyV4Model {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_rows(input_ids, caches, mask, None)
    }

    /// [`Self::forward`] projecting every position (`row == None`) or only
    /// position `row` (`[B, 1, vocab]`, the prefill's sampled row).
    fn forward_rows(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let h = self.embed_tokens.forward(input_ids);
        let shape = mlxcel_core::array_shape(&h);
        let (b, l, d) = (shape[0], shape[1], shape[2]);
        let h = mlxcel_core::expand_dims(&h, 2);
        let mut h = mlxcel_core::broadcast_to(&h, &[b, l, self.hc_mult, d]);

        // The DSA selection a `full` layer hands to the `shared` layers after
        // it; it never outlives one forward.
        let mut topk: Option<UniquePtr<MlxArray>> = None;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, mask, cache, &mut topk);
        }

        // The head, the norm and the LM head act per position, so slicing the
        // sampled row first yields what the full projection would have.
        let h = match row {
            Some(row) => slice_axis(&h, 1, row as i32, row as i32 + 1),
            None => h,
        };
        let h = self.norm.forward(&self.hc_head.forward(&h));
        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, HyV4Config), String> {
        let model_dir = model_dir.as_ref();
        let config = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config = HyV4Config::from_json_str(&config)?;
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &config)?;
        Ok((model, config))
    }

    /// Build from a checkpoint-named weight map. Keys outside the model
    /// (`model.mtp_layers.*`, layers past `num_hidden_layers`) are never read.
    pub fn from_weights(weights: &WeightMap, config: &HyV4Config) -> Result<Self, String> {
        Self::from_weights_with_arms(weights, config, ParityArms::from_env()?)
    }

    pub(crate) fn from_weights_with_arms(
        weights: &WeightMap,
        config: &HyV4Config,
        arms: ParityArms,
    ) -> Result<Self, String> {
        let mut base = config.base.clone();
        if arms.rope_half_split {
            base.indexer_rope_interleave = false;
        }
        let group_size = base.group_size();
        let bits = base.bits();
        let hidden = base.hidden_size as i32;
        let hc_mult = config.hc_mult as i32;

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;
        let layers = (0..base.num_hidden_layers)
            .map(|layer| load_decoder_layer(weights, config, &base, layer, arms))
            .collect::<Result<Vec<_>, String>>()?;
        let hc_head = IdentityHyperHead::from_weights(
            weights,
            "model.hc_head",
            hidden,
            hc_mult,
            config.hc_eps,
            base.rms_norm_eps,
        )?;
        let norm = RMSNorm::new(
            get_weight_copy(weights, "model.norm.weight")?,
            base.rms_norm_eps,
        );
        let lm_head = if base.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };

        Ok(Self {
            embed_tokens,
            layers,
            hc_head,
            norm,
            lm_head,
            hc_mult,
            eos_token_ids: config.eos_token_ids.clone(),
        })
    }
}

fn load_decoder_layer(
    weights: &WeightMap,
    config: &HyV4Config,
    base: &ModelArgs,
    layer: usize,
    arms: ParityArms,
) -> Result<DecoderLayer, String> {
    let prefix = format!("model.layers.{layer}");
    let attn_prefix = format!("{prefix}.self_attn");
    let mlp_prefix = format!("{prefix}.mlp");
    let group_size = base.group_size();
    let bits = base.bits();
    let hidden = base.hidden_size as i32;
    let hc_mult = config.hc_mult as i32;

    if weights.contains_key(&format!("{attn_prefix}.kv_b_proj.weight")) {
        return Err(format!(
            "{attn_prefix}: the checkpoint ships `kv_b_proj` (the original tencent/Hy4-preview \
             layout), which is not supported; use a conversion that ships `embed_q` and \
             `unembed_out`, such as mlx-community/Hy4-preview-4bit"
        ));
    }

    // Indexer ownership must agree between `indexer_types` and the checkpoint:
    // a mismatch would otherwise change which positions every later shared
    // layer attends to with no error.
    let has_indexer = weights.contains_key(&format!("{attn_prefix}.indexer.wk.weight"));
    let shares_indexer = config.indexer_types[layer] == IndexerType::Shared;
    match (shares_indexer, has_indexer) {
        (true, true) => {
            return Err(format!(
                "{attn_prefix}: indexer_types[{layer}] is `shared`, but the checkpoint ships \
                 indexer weights for this layer"
            ));
        }
        (false, false) => {
            return Err(format!(
                "{attn_prefix}: indexer_types[{layer}] is `full`, but the checkpoint ships no \
                 indexer weights (`{attn_prefix}.indexer.wk.weight`) for this layer"
            ));
        }
        _ => {}
    }

    let mla = load_mla_attention(weights, base, &attn_prefix)?;
    let linear_gate = if arms.no_gate {
        None
    } else {
        Some(UnifiedLinear::from_weights(
            weights,
            &format!("{attn_prefix}.linear_gate"),
            group_size,
            bits,
        )?)
    };
    let sinks = if arms.no_sinks {
        None
    } else {
        let name = format!("{attn_prefix}.learnable_sink_param");
        let sinks = get_weight_copy(weights, &name)?;
        let shape = mlxcel_core::array_shape(&sinks);
        if shape != [base.num_attention_heads as i32] {
            return Err(format!(
                "{name}: expected shape [{}] (num_attention_heads), checkpoint ships {shape:?}",
                base.num_attention_heads
            ));
        }
        Some(mlxcel_core::astype(&sinks, mlxcel_core::dtype::FLOAT32))
    };
    let self_attn = HyV4Attention {
        mla,
        linear_gate,
        sinks,
        shares_indexer,
        rope_traditional: !arms.rope_half_split,
    };

    let mlp = if config.mlp_is_moe[layer] {
        for fused in ["switch_mlp.gate_up_proj.weight", "experts.gate_up_proj"] {
            if weights.contains_key(&format!("{mlp_prefix}.{fused}")) {
                return Err(format!(
                    "{mlp_prefix}.{fused}: the fused `gate_up_proj` expert layout is not \
                     supported; this port reads separate `switch_mlp.{{gate,up,down}}_proj` planes"
                ));
            }
        }
        Mlp::MoE(load_moe_block(weights, base, &mlp_prefix)?)
    } else {
        Mlp::Dense(load_dense_mlp(weights, &mlp_prefix, group_size, bits)?)
    };

    let norm = |name: &str| -> Result<RMSNorm, String> {
        Ok(RMSNorm::new(
            get_weight_copy(weights, &format!("{prefix}.{name}.weight"))?,
            base.rms_norm_eps,
        ))
    };
    let hyper = |name: &str| {
        IdentityHyperConnection::from_weights(
            weights,
            &format!("{prefix}.{name}.hc_pre"),
            hidden,
            hc_mult,
            config.hc_eps,
            config.hc_magnitude,
            base.rms_norm_eps,
        )
    };

    Ok(DecoderLayer {
        self_attn,
        mlp,
        input_layernorm: norm("input_layernorm")?,
        post_attention_layernorm: norm("post_attention_layernorm")?,
        hc_attn: hyper("hc_attn_layer")?,
        hc_mlp: hyper("hc_mlp_layer")?,
        swiglu_limit: config.swiglu_limit.filter(|_| !arms.no_swiglu_limit),
    })
}

impl LanguageModel for HyV4Model {
    // Every attention step is per sequence (see the module docs), so the
    // default per-row `forward_batched` is the batched decode.
    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        BatchedDecodeKind::PerRowFallback
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        HyV4Model::forward(self, input_ids, caches, mask)
    }

    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _seq_id: Option<mlxcel_core::cache::SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        // No embeddings path: decline so the default keeps its behavior.
        if input_embeddings.is_some() {
            return None;
        }
        Some(self.forward_rows(input_ids, caches, mask, Some(last_pos)))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        HyV4Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}
