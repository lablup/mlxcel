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

//! GLM-4V MoE text backbone with sectioned half-split MRoPE.
//!
//! GLM-4.5V-class text model: a GLM-4 MoE backbone (grouped `noaux_tc` routing,
//! shared experts, `first_k_dense_replace` dense layers) driven by 3D MRoPE.
//! Unlike GLM-4V (`sectioned_even_odd`), GLM-4V MoE uses the
//! `sectioned_half_split` style (GPT-NeoX rotation), applies no q/k norm, and
//! uses two RMSNorm layers per decoder block. The MoE / dense MLP machinery is
//! reused from [`crate::models::glm4_moe`]; the sectioned MRoPE is reused from
//! [`crate::models::glm4v`].
//!
//! Used by: GLM-4V MoE
//! Reference: references/mlx-vlm/mlx_vlm/models/glm4v_moe/language.py

use crate::models::glm4_moe::{DenseMLP, FFN, Glm4Moe, ModelArgs};
use crate::models::glm4v::{Glm4vMRoPE, Glm4vRopePairing};
use crate::models::qwen_mrope_state::{
    MRopeState, concatenate_scattered, decode_offsets, decode_positions_from_offsets,
    scatter_group_logits, select_cache_rows, take_rows,
};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, route_batched_decode_with_ids};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

fn load_rms_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<RMSNorm, String> {
    let key = format!("{}.weight", prefix);
    let weight = weights
        .get(&key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", key))?;
    Ok(RMSNorm::new(weight, eps))
}

// Attention with sectioned half-split MRoPE (no q/k norm).
struct Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    mrope: Glm4vMRoPE,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl mlxcel_core::batched_decode::SplitAttention for Attention {
    fn split_shape(&self) -> mlxcel_core::batched_decode::SplitAttentionShape {
        mlxcel_core::batched_decode::SplitAttentionShape {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            scale: self.scale,
        }
    }

    fn rope_batched(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        offsets: &[i32],
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let positions = decode_positions_from_offsets(offsets, 3);
        let (cos, sin) = self.mrope.cos_sin(&positions);
        (
            self.mrope.apply(q, &cos, &sin),
            self.mrope.apply(k, &cos, &sin),
        )
    }
}

struct Glm4vMoeMropeAttention<'a> {
    attention: &'a Attention,
    cos: &'a MlxArray,
    sin: &'a MlxArray,
}

impl mlxcel_core::batched_decode::SplitAttention for Glm4vMoeMropeAttention<'_> {
    fn split_shape(&self) -> mlxcel_core::batched_decode::SplitAttentionShape {
        mlxcel_core::batched_decode::SplitAttention::split_shape(self.attention)
    }

    fn rope_batched(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        _offsets: &[i32],
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        (
            self.attention.mrope.apply(q, self.cos, self.sin),
            self.attention.mrope.apply(k, self.cos, self.sin),
        )
    }
}

impl Attention {
    fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        mrope_section: &[i32],
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();
        let q_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.self_attn.q_proj", prefix),
            gs,
            bits,
        )?;
        let k_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.self_attn.k_proj", prefix),
            gs,
            bits,
        )?;
        let v_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.self_attn.v_proj", prefix),
            gs,
            bits,
        )?;
        let o_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.self_attn.o_proj", prefix),
            gs,
            bits,
        )?;

        let head_dim = args.head_dim();
        let mrope = Glm4vMRoPE::new(
            args.rope_theta,
            args.rope_dims(),
            mrope_section,
            Glm4vRopePairing::HalfSplit,
        );

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            mrope,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim: head_dim as i32,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        cos_f: &MlxArray,
        sin_f: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let q = self.mrope.apply(&q, cos_f, sin_f);
        let k = self.mrope.apply(&k, cos_f, sin_f);

        let (k, v) = cache.update_and_fetch(k, v);

        // K and V stay GQA-shaped: the fused SDPA below broadcasts KV heads
        // internally, so expanding them here only writes an n_rep-sized copy of
        // the whole live cache on every decode step (#1686).

        let output = if let Some(m) = mask {
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    m as *const MlxArray,
                    0.0,
                    0,
                )
            }
        } else {
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            }
        };

        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);
        self.o_proj.forward(&output)
    }
}

// Decoder layer: two RMSNorm layers, MoE-or-dense MLP reused from glm4_moe.
struct DecoderLayer {
    attn: Attention,
    mlp: FFN,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        mrope_section: &[i32],
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);
        let attn = Attention::from_weights(weights, args, mrope_section, &prefix)?;
        let mlp = if args.is_moe_layer(layer_idx) {
            FFN::Moe(Glm4Moe::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            FFN::Dense(DenseMLP::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        };
        Ok(Self {
            attn,
            mlp,
            input_layernorm: load_rms_norm(
                weights,
                &format!("{}.input_layernorm", prefix),
                args.rms_norm_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                weights,
                &format!("{}.post_attention_layernorm", prefix),
                args.rms_norm_eps,
            )?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        cos_f: &MlxArray,
        sin_f: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let r = self
            .attn
            .forward(&self.input_layernorm.forward(x), cache, mask, cos_f, sin_f);
        let h = mlxcel_core::add(x, &r);
        let r = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &r)
    }

    fn forward_batched<A: mlxcel_core::batched_decode::SplitAttention + ?Sized>(
        &self,
        x: &MlxArray,
        caches: &mut [&mut KVCache],
        layer_idx: usize,
        mask: Option<&MlxArray>,
        attention: &A,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let q = self.attn.q_proj.forward(&normed);
        let k = self.attn.k_proj.forward(&normed);
        let v = self.attn.v_proj.forward(&normed);
        let seq_len = mlxcel_core::array_shape(&q)[1];
        let metadata = crate::models::batched_decode_cache::uniform_layer_metadata(
            caches,
            seq_len,
            0,
            "GLM-4V-MoE",
            layer_idx,
        );
        let attn_out = mlxcel_core::batched_decode::forward_split_attention(
            attention, &q, &k, &v, caches, &metadata, mask,
        );
        let attn_out = self.attn.o_proj.forward(&attn_out);
        let h = mlxcel_core::add(x, &attn_out);
        let r = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &r)
    }
}

/// GLM-4V MoE text model (language backbone with MRoPE).
pub struct Glm4vMoeTextModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: UnifiedLinear,
    mrope: Glm4vMRoPE,
    eos_token_ids: Vec<i32>,
    mrope_state: MRopeState,
}

impl Glm4vMoeTextModel {
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        mrope_section: &[i32],
        eos_token_ids: Vec<i32>,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();

        let embed_tokens = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", gs, bits)?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            layers.push(DecoderLayer::from_weights(weights, args, mrope_section, i)?);
        }

        let norm = load_rms_norm(weights, "model.norm", args.rms_norm_eps)?;

        let lm_head = if args.tie_word_embeddings {
            UnifiedLinear::from_weights(weights, "model.embed_tokens", gs, bits)?
        } else {
            UnifiedLinear::from_weights(weights, "lm_head", gs, bits)?
        };

        let mrope = Glm4vMRoPE::new(
            args.rope_theta,
            args.rope_dims(),
            mrope_section,
            Glm4vRopePairing::HalfSplit,
        );

        let eos_token_ids = if eos_token_ids.is_empty() {
            vec![151329, 151336, 151338]
        } else {
            eos_token_ids
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            mrope,
            eos_token_ids,
            mrope_state: MRopeState::new(),
        })
    }

    pub fn set_mrope_state(&self, position_ids: UniquePtr<MlxArray>, rope_deltas: i32) {
        self.mrope_state.set_fallback(position_ids, rope_deltas);
    }

    pub fn release_mrope_sequence(&self, seq_id: SequenceId) {
        self.mrope_state.release_sequence(seq_id);
    }

    pub fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.mrope_state.bind_fallback_to_sequence(seq_id);
    }

    /// Clear the fallback MRoPE state without binding it (admission abort
    /// after media preparation, issue #2625).
    pub fn clear_mrope_state(&self) {
        self.mrope_state.clear_fallback();
    }

    pub(crate) fn take_mrope_entry(
        &self,
        seq_id: SequenceId,
    ) -> Option<crate::models::qwen_mrope_state::MRopeEntry> {
        self.mrope_state.take_for_sequence(seq_id)
    }

    pub(crate) fn install_mrope_entry(
        &self,
        seq_id: SequenceId,
        entry: crate::models::qwen_mrope_state::MRopeEntry,
    ) {
        self.mrope_state.bind_for_sequence(seq_id, entry);
    }

    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, input_embeddings, caches, mask, None)
    }

    pub(crate) fn forward_for_sequence(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.embed_tokens.forward(input_ids)
        };

        let ids_shape = mlxcel_core::array_shape(input_ids);
        let batch = ids_shape[0];
        let seq_len = ids_shape[1];
        let cache_offset = caches[0].offset;

        let position_ids = self.mrope_state.with_entry(seq_id, |entry| {
            if let Some(ref stored_pos) = entry.position_ids {
                let pos_shape = mlxcel_core::array_shape(stored_pos);
                if pos_shape.len() == 3
                    && pos_shape[1] == batch
                    && pos_shape[2] >= cache_offset + seq_len
                {
                    return mlxcel_core::slice(
                        stored_pos,
                        &[0, 0, cache_offset],
                        &[pos_shape[0], pos_shape[1], cache_offset + seq_len],
                    );
                }
                Self::compute_position_ids_with_delta(
                    entry.rope_deltas.unwrap_or(0),
                    batch,
                    seq_len,
                    cache_offset,
                )
            } else if cache_offset > 0 {
                Self::compute_position_ids_with_delta(
                    entry.rope_deltas.unwrap_or(0),
                    batch,
                    seq_len,
                    cache_offset,
                )
            } else {
                let pos = mlxcel_core::arange_i32(0, seq_len, 1);
                let pos = mlxcel_core::reshape(&pos, &[1, seq_len]);
                let pos = mlxcel_core::broadcast_to(&pos, &[batch, seq_len]);
                let pos = mlxcel_core::expand_dims(&pos, 0);
                mlxcel_core::broadcast_to(&pos, &[3, batch, seq_len])
            }
        });

        let (cos_f, sin_f) = self.mrope.cos_sin(&position_ids);

        let auto_mask;
        let mask = if mask.is_some() {
            mask
        } else if seq_len > 1 {
            auto_mask = mlxcel_core::utils::create_causal_mask(seq_len, caches[0].live_len());
            Some(auto_mask.as_ref().unwrap() as &MlxArray)
        } else {
            // Decode width. The one query row sits at logical position
            // `live_len`, so every key column is permitted: the mask would be
            // uniformly zero, constrain nothing, and only force the masked arm
            // of the fused SDPA. mlx-lm returns None here as well
            // (`create_attention_mask`), and `qwen3_5.rs` carries the same
            // `seq_len > 1` guard (#1686).
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask, &cos_f, &sin_f);
        }

        h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    fn compute_position_ids_with_delta(
        delta: i32,
        batch: i32,
        seq_len: i32,
        cache_offset: i32,
    ) -> UniquePtr<MlxArray> {
        let offset = cache_offset + delta;
        let pos = mlxcel_core::arange_i32(offset, offset + seq_len, 1);
        let pos = mlxcel_core::reshape(&pos, &[1, seq_len]);
        let pos = mlxcel_core::broadcast_to(&pos, &[batch, seq_len]);
        let pos = mlxcel_core::expand_dims(&pos, 0);
        mlxcel_core::broadcast_to(&pos, &[3, batch, seq_len])
    }

    fn can_use_text_only_fast_path(&self, seq_id: SequenceId) -> bool {
        !self.mrope_state.with_entry(Some(seq_id), |entry| {
            entry.position_ids.is_some() || entry.rope_deltas.unwrap_or(0) != 0
        })
    }

    pub(crate) fn forward_batched_decode_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        route_batched_decode_with_ids(self, input_ids, seq_ids, batch_caches, |batch_caches| {
            let shape = mlxcel_core::array_shape(input_ids);
            let Some(seq_ids) = seq_ids.filter(|ids| ids.len() == batch_caches.len()) else {
                return crate::multimodal::qwen_vl::forward_batched_with_seq_ids_dispatch(
                    self,
                    input_ids,
                    seq_ids,
                    batch_caches,
                    mask,
                );
            };
            if shape.len() != 2 || shape[1] != 1 {
                return crate::multimodal::qwen_vl::forward_batched_with_seq_ids_dispatch(
                    self,
                    input_ids,
                    Some(seq_ids),
                    batch_caches,
                    mask,
                );
            }
            let offsets = decode_offsets(batch_caches, "GLM-4V-MoE");
            let Some(positions) = self.mrope_state.batched_decode_positions(seq_ids, &offsets)
            else {
                return crate::multimodal::qwen_vl::forward_batched_with_seq_ids_dispatch(
                    self,
                    input_ids,
                    Some(seq_ids),
                    batch_caches,
                    mask,
                );
            };
            let mut plain_rows = Vec::new();
            let mut mrope_rows = Vec::new();
            for (row, &seq_id) in seq_ids.iter().enumerate() {
                if self.can_use_text_only_fast_path(seq_id) {
                    plain_rows.push(row);
                } else {
                    mrope_rows.push(row);
                }
            }
            self.forward_decode_groups(
                input_ids,
                seq_ids,
                batch_caches,
                mask,
                &positions,
                &plain_rows,
                &mrope_rows,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_decode_groups(
        &self,
        input_ids: &MlxArray,
        seq_ids: &[SequenceId],
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        positions: &MlxArray,
        plain_rows: &[usize],
        mrope_rows: &[usize],
    ) -> UniquePtr<MlxArray> {
        let mut outputs: Vec<Option<UniquePtr<MlxArray>>> = std::iter::repeat_with(|| None)
            .take(seq_ids.len())
            .collect();
        for (rows, use_mrope) in [(plain_rows, false), (mrope_rows, true)] {
            if rows.is_empty() {
                continue;
            }
            let ids = take_rows(input_ids, rows, 0);
            let row_mask = mask.map(|value| take_rows(value, rows, 0));
            let logits = if rows.len() == 1 {
                let row = rows[0];
                self.forward_with_sequence_id(
                    &ids,
                    Some(seq_ids[row]),
                    batch_caches[row],
                    row_mask.as_deref(),
                )
            } else {
                let mut caches = select_cache_rows(batch_caches, rows);
                let row_positions = use_mrope.then(|| take_rows(positions, rows, 1));
                self.forward_decode_group(
                    &ids,
                    &mut caches,
                    row_mask.as_deref(),
                    row_positions.as_deref(),
                )
            };
            scatter_group_logits(&mut outputs, rows, &logits);
        }
        concatenate_scattered(outputs, "GLM-4V-MoE")
    }

    fn forward_decode_group(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        positions: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        let tables = positions.map(|positions| self.mrope.cos_sin(positions));
        for layer_idx in 0..self.layers.len() {
            let mut layer_caches = crate::models::batched_decode_cache::collect_layer_caches(
                batch_caches,
                "GLM-4V-MoE",
                layer_idx,
            );
            h = if let Some((cos, sin)) = tables.as_ref() {
                let attention = Glm4vMoeMropeAttention {
                    attention: &self.layers[layer_idx].attn,
                    cos,
                    sin,
                };
                self.layers[layer_idx].forward_batched(
                    &h,
                    &mut layer_caches,
                    layer_idx,
                    mask,
                    &attention,
                )
            } else {
                self.layers[layer_idx].forward_batched(
                    &h,
                    &mut layer_caches,
                    layer_idx,
                    mask,
                    &self.layers[layer_idx].attn,
                )
            };
        }
        self.lm_head.forward(&self.norm.forward(&h))
    }
}

impl mlxcel_core::generate::LanguageModel for Glm4vMoeTextModel {
    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        BatchedDecodeKind::Native
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, None, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, input_embeddings, caches, mask)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, None, caches, mask, seq_id)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, input_embeddings, caches, mask, seq_id)
    }

    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_decode_with_ids(input_ids, seq_ids, batch_caches, mask)
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.release_mrope_sequence(seq_id);
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Glm4vMoeTextModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}
