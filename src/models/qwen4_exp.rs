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

//! Qwen4-Exp (`qwen4_exp`, Qwen3.8-Flash-Next) text backbone.
//!
//! Ported from mlx-vlm's `mlx_vlm/models/qwen4_exp/language.py`. The layer
//! stack is Qwen3.5's (GatedDeltaNet layers with every
//! `full_attention_interval`-th layer attention, interleaved MRoPE, a
//! shared-expert MoE) with four additions, each in its own file:
//!
//! * gated hyper-connection residual streams ([`hyper`]),
//! * Qwen Sparse Attention on the attention layers ([`qsa`]),
//! * a hashed n-gram embedding injected at `ple_layer_ids` ([`ple`]),
//! * zero-centered RMSNorm weights throughout ([`norm`]).
//!
//! Each attention layer carries an index cache besides its KV cache, and each
//! PLE layer a convolution window and token context besides its
//! GatedDeltaNet state, so the model owns its per-sequence state.
//!
//! Out of scope: MTP drafting (`mtp.*` tensors are dropped), external
//! `ple_storage`, FP8 checkpoints, the fused QSA Metal kernel, prompt-cache
//! snapshots and batched prefill.

#[path = "qwen4_exp_config.rs"]
pub mod config;
#[path = "qwen4_exp_gated_delta.rs"]
mod gated_delta;
#[path = "qwen4_exp_hyper.rs"]
mod hyper;
#[path = "qwen4_exp_linear.rs"]
mod linear;
#[path = "qwen4_exp_moe.rs"]
mod moe;
#[path = "qwen4_exp_norm.rs"]
mod norm;
#[path = "qwen4_exp_ple.rs"]
mod ple;
#[path = "qwen4_exp_qsa.rs"]
mod qsa;
#[path = "qwen4_exp_sanitize.rs"]
pub mod sanitize;
#[path = "qwen4_exp_shapes.rs"]
mod shapes;

pub use config::{
    Qwen4ExpConfig, Qwen4ExpConfigError, ResolvedQwen4ExpConfig, text_config_from_full,
};
pub use sanitize::{SanitizedWeights, sanitize_weights};

use mlxcel_core::cache::{KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::gated_delta::GatedDeltaCache;
use crate::models::model_owned::{KvCacheLayerModes, ModelOwnedSequenceState};
use crate::models::qwen_mrope_state::{MRopeEntry, MRopeState};
use crate::models::qwen3_5::{MRopePositionSource, mrope_position_source};
use crate::models::qwen3_vl::InterleavedMRoPE;

use gated_delta::Qwen4ExpGatedDeltaNet;
use hyper::GatedResidual;
use moe::Qwen4ExpMoeBlock;
use ple::{PleCache, PleLayer};
use qsa::{QsaAttention, QsaIndexCache, RopeContext};

pub(crate) enum Qwen4ExpCache {
    Linear {
        delta: GatedDeltaCache,
        ple: Option<PleCache>,
    },
    Attention {
        kv: KVCache,
        index: QsaIndexCache,
    },
}

enum Mixer {
    Linear(Qwen4ExpGatedDeltaNet),
    Attention(QsaAttention),
}

struct DecoderLayer {
    mixer: Mixer,
    ple: Option<PleLayer>,
    mlp: Qwen4ExpMoeBlock,
    attn_hyper_connection: GatedResidual,
    mlp_hyper_connection: GatedResidual,
}

impl DecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &ResolvedQwen4ExpConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let mixer = if config.is_linear_layer(layer_idx) {
            Mixer::Linear(Qwen4ExpGatedDeltaNet::from_weights(
                weights,
                config,
                &format!("{prefix}.linear_attn"),
            )?)
        } else {
            Mixer::Attention(QsaAttention::from_weights(
                weights,
                config,
                &format!("{prefix}.self_attn"),
            )?)
        };
        let ple = match config.ple_index(layer_idx) {
            Some(index) => Some(PleLayer::from_weights(
                weights,
                config,
                &format!("{prefix}.ple"),
                index,
            )?),
            None => None,
        };
        Ok(Self {
            mixer,
            ple,
            mlp: Qwen4ExpMoeBlock::from_weights(weights, config, &format!("{prefix}.mlp"))?,
            attn_hyper_connection: GatedResidual::from_weights(
                weights,
                config,
                &format!("{prefix}.attn_hyper_connection"),
                true,
            )?,
            mlp_hyper_connection: GatedResidual::from_weights(
                weights,
                config,
                &format!("{prefix}.mlp_hyper_connection"),
                true,
            )?,
        })
    }

    fn forward(
        &self,
        hidden: UniquePtr<MlxArray>,
        tokens: Option<&[Vec<i64>]>,
        cache: &mut Qwen4ExpCache,
        rope: &RopeContext<'_>,
    ) -> UniquePtr<MlxArray> {
        let hidden = match (&self.ple, &mut *cache) {
            (
                Some(ple),
                Qwen4ExpCache::Linear {
                    ple: Some(ple_cache),
                    ..
                },
            ) => {
                let tokens = tokens.expect("PLE layers need the input token ids");
                mlxcel_core::add(&hidden, &ple.forward(&hidden, tokens, ple_cache))
            }
            (None, _) => hidden,
            (Some(_), _) => unreachable!("a PLE layer always gets a PLE cache"),
        };

        let (mixed, inject) = self.attn_hyper_connection.split(&hidden);
        let branch = match (&self.mixer, cache) {
            (Mixer::Linear(delta_net), Qwen4ExpCache::Linear { delta, .. }) => {
                delta_net.forward(&mixed, Some(delta))
            }
            (Mixer::Attention(attention), Qwen4ExpCache::Attention { kv, index }) => {
                attention.forward(&mixed, rope, kv, index)
            }
            _ => unreachable!("layer caches are built from the same layer list"),
        };
        let hidden = self.attn_hyper_connection.inject(&branch, &hidden, &inject);

        let (mixed, inject) = self.mlp_hyper_connection.split(&hidden);
        let branch = self.mlp.forward(&mixed);
        self.mlp_hyper_connection.inject(&branch, &hidden, &inject)
    }
}

pub struct Qwen4ExpModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    hyper_connection_mixer: GatedResidual,
    lm_head: Option<UnifiedLinear>,
    config: ResolvedQwen4ExpConfig,
    mrope: InterleavedMRoPE,
    sequence_state: ModelOwnedSequenceState<Qwen4ExpCache>,
    kv_cache_layer_modes: KvCacheLayerModes,
    mrope_state: MRopeState,
}

impl Qwen4ExpModel {
    /// Build from sanitized text weights (see [`sanitize_weights`]).
    pub fn from_weights(
        weights: &WeightMap,
        config: &ResolvedQwen4ExpConfig,
    ) -> Result<Self, String> {
        let raw = &config.raw;
        let vocabulary = [raw.vocab_size, raw.hidden_size];
        let (group_size, bits) = config.quantization.for_module("model.embed_tokens");
        shapes::expect_matrix(
            weights,
            "model.embed_tokens",
            (group_size, bits),
            &vocabulary,
        )?;
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;
        let layers = (0..raw.num_hidden_layers)
            .map(|layer_idx| DecoderLayer::from_weights(weights, config, layer_idx))
            .collect::<Result<Vec<_>, _>>()?;
        let hyper_connection_mixer =
            GatedResidual::from_weights(weights, config, "model.hyper_connection_mixer", false)?;
        let lm_head = if raw.tie_word_embeddings {
            None
        } else {
            let (group_size, bits) = config.quantization.for_module("lm_head");
            shapes::expect_matrix(weights, "lm_head", (group_size, bits), &vocabulary)?;
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };
        let mrope = InterleavedMRoPE::new_with_backend_frequencies(
            config.rope_dims(),
            config.rope.rope_theta,
            config.rope.mrope_section.clone(),
        );
        let mut model = Self {
            embed_tokens,
            layers,
            hyper_connection_mixer,
            lm_head,
            config: config.clone(),
            mrope,
            sequence_state: ModelOwnedSequenceState::new(Vec::new()),
            kv_cache_layer_modes: KvCacheLayerModes::new(),
            mrope_state: MRopeState::new(),
        };
        model.sequence_state = ModelOwnedSequenceState::new(model.make_internal_caches());
        Ok(model)
    }

    pub fn config(&self) -> &ResolvedQwen4ExpConfig {
        &self.config
    }

    pub(crate) fn make_internal_caches(&self) -> Vec<Qwen4ExpCache> {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| match layer.mixer {
                Mixer::Linear(_) => Qwen4ExpCache::Linear {
                    delta: GatedDeltaCache::new(),
                    ple: layer.ple.as_ref().map(|_| PleCache::default()),
                },
                Mixer::Attention(_) => Qwen4ExpCache::Attention {
                    kv: KVCache::new_with_mode(self.kv_cache_layer_modes.mode_for_layer(layer_idx)),
                    index: QsaIndexCache::default(),
                },
            })
            .collect()
    }

    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    fn cache_offset(caches: &[Qwen4ExpCache]) -> i32 {
        caches
            .iter()
            .find_map(|cache| match cache {
                Qwen4ExpCache::Attention { kv, .. } => Some(kv.offset),
                Qwen4ExpCache::Linear { .. } => None,
            })
            .unwrap_or(0)
    }

    /// `[3, B, L]` text positions continuing from `offset`.
    fn text_positions(batch: i32, seq_len: i32, offset: i32) -> UniquePtr<MlxArray> {
        let positions = mlxcel_core::arange_i32(offset, offset + seq_len, 1);
        let positions = mlxcel_core::reshape(&positions, &[1, 1, seq_len]);
        mlxcel_core::broadcast_to(&positions, &[3, batch, seq_len])
    }

    fn host_tokens(input_ids: &MlxArray) -> Vec<Vec<i64>> {
        let shape = mlxcel_core::array_shape(input_ids);
        let ids = mlxcel_core::astype(input_ids, mlxcel_core::dtype::INT32);
        mlxcel_core::eval(&ids);
        let flat: Vec<i64> = mlxcel_core::array_to_raw_bytes(&ids)
            .chunks_exact(4)
            .map(|chunk| i64::from(i32::from_le_bytes(chunk.try_into().expect("4-byte chunk"))))
            .collect();
        flat.chunks(shape[1].max(1) as usize)
            .map(<[i64]>::to_vec)
            .collect()
    }

    /// Run the layer stack and collapse the hyper-connection streams:
    /// `[B, L, hidden]` before the LM head.
    pub(crate) fn forward_hidden(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Qwen4ExpCache],
        position_ids: Option<UniquePtr<MlxArray>>,
    ) -> UniquePtr<MlxArray> {
        let embeddings = match input_embeddings {
            Some(embeddings) => mlxcel_core::copy(embeddings),
            None => self.embed_tokens.forward(input_ids),
        };
        let shape = mlxcel_core::array_shape(&embeddings);
        let (batch, seq_len) = (shape[0], shape[1]);
        let mut hidden = mlxcel_core::tile(&embeddings, &[1, 1, self.config.raw.hc_count as i32]);

        let positions = match position_ids {
            Some(positions) => mlxcel_core::astype(&positions, mlxcel_core::dtype::INT32),
            None => Self::text_positions(batch, seq_len, Self::cache_offset(caches)),
        };
        let rope = RopeContext::new(&self.mrope, positions);
        let tokens = self
            .layers
            .iter()
            .any(|layer| layer.ple.is_some())
            .then(|| Self::host_tokens(input_ids));

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward(hidden, tokens.as_deref(), cache, &rope);
            if seq_len == 1 {
                // Match mlx-vlm's decode scheduling: submit each layer while
                // the CPU builds the next one instead of queuing the whole graph.
                mlxcel_core::async_eval(&hidden);
            }
        }
        self.hyper_connection_mixer.collapse(&hidden)
    }

    fn logits(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        match &self.lm_head {
            Some(lm_head) => lm_head.forward(hidden),
            None => self.embed_tokens.as_linear(hidden),
        }
    }

    /// MRoPE positions for this call: the window of a stored multimodal
    /// prefill, the stored delta continued past it, or `None` for plain text
    /// positions.
    fn resolve_positions(
        &self,
        seq_id: Option<SequenceId>,
        batch: i32,
        seq_len: i32,
        cache_offset: i32,
    ) -> Option<UniquePtr<MlxArray>> {
        self.mrope_state.with_entry(seq_id, |entry| {
            let stored_shape = entry
                .position_ids
                .as_ref()
                .map(|stored| mlxcel_core::array_shape(stored));
            match mrope_position_source(stored_shape.as_deref(), batch, seq_len, cache_offset) {
                MRopePositionSource::SliceStored { start, end } => {
                    let stored = entry.position_ids.as_ref()?;
                    let shape = stored_shape.as_ref()?;
                    Some(mlxcel_core::slice(
                        stored,
                        &[0, 0, start],
                        &[shape[0], shape[1], end],
                    ))
                }
                MRopePositionSource::Recompute => entry
                    .rope_deltas
                    .map(|delta| Self::text_positions(batch, seq_len, cache_offset + delta)),
            }
        })
    }

    fn forward_sequence_hidden(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        let ids_shape = mlxcel_core::array_shape(input_ids);
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| {
                let positions = self.resolve_positions(
                    seq_id,
                    ids_shape[0],
                    ids_shape[1],
                    Self::cache_offset(caches),
                );
                self.forward_hidden(input_ids, input_embeddings, caches, positions)
            },
        )
    }

    fn forward_sequence(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        self.logits(&self.forward_sequence_hidden(input_ids, input_embeddings, seq_id))
    }

    fn forward_sequence_last(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        let hidden = self.forward_sequence_hidden(input_ids, input_embeddings, seq_id);
        let shape = mlxcel_core::array_shape(&hidden);
        let last = last_pos as i32;
        let hidden = mlxcel_core::slice(&hidden, &[0, last, 0], &[shape[0], last + 1, shape[2]]);
        self.logits(&hidden)
    }

    fn forward_rows_sequential(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
    ) -> UniquePtr<MlxArray> {
        // Anonymous batches keep their row dimension in the fallback state.
        // Splitting them into singleton calls would append every row to the
        // same history, mixing KV, recurrent and PLE state across requests.
        if seq_ids.is_none() {
            return self.forward_sequence(input_ids, None, None);
        }
        let shape = mlxcel_core::array_shape(input_ids);
        let mut result: Option<UniquePtr<MlxArray>> = None;
        for row in 0..shape[0] {
            let ids = mlxcel_core::slice(input_ids, &[row, 0], &[row + 1, shape[1]]);
            let seq_id = seq_ids.and_then(|ids| ids.get(row as usize).copied());
            let logits = self.forward_sequence(&ids, None, seq_id);
            result = Some(match result {
                None => logits,
                Some(previous) => mlxcel_core::concatenate(&previous, &logits, 0),
            });
        }
        result.expect("at least one batch row")
    }

    /// Store the MRoPE positions of a multimodal prefill for the next call
    /// without a sequence id (the CLI path and the VLM wrapper).
    pub fn set_mrope_state(&self, position_ids: UniquePtr<MlxArray>, rope_deltas: i32) {
        self.mrope_state.set_fallback(position_ids, rope_deltas);
    }

    pub fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.mrope_state.bind_fallback_to_sequence(seq_id);
    }

    /// Clear the fallback MRoPE state without binding it (admission abort
    /// after media preparation, issue #2625).
    pub fn clear_mrope_state(&self) {
        self.mrope_state.clear_fallback();
    }

    pub(crate) fn take_mrope_entry(&self, seq_id: SequenceId) -> Option<MRopeEntry> {
        self.mrope_state.take_for_sequence(seq_id)
    }

    pub(crate) fn install_mrope_entry(&self, seq_id: SequenceId, entry: MRopeEntry) {
        self.mrope_state.bind_for_sequence(seq_id, entry);
    }

    fn eos_ids(&self) -> Vec<i32> {
        let mut ids = vec![248046];
        if let Some(eos) = self
            .config
            .eos_token_id
            .and_then(|id| i32::try_from(id).ok())
            && !ids.contains(&eos)
        {
            ids.push(eos);
        }
        ids
    }

    /// Run `input_ids` through a fresh cache set, as one call or as
    /// consecutive chunks, returning the logits of every position.
    #[cfg(test)]
    pub(crate) fn forward_fresh(
        &self,
        input_ids: &MlxArray,
        chunks: &[i32],
    ) -> UniquePtr<MlxArray> {
        let mut caches = self.make_internal_caches();
        let shape = mlxcel_core::array_shape(input_ids);
        let mut start = 0;
        let mut out: Option<UniquePtr<MlxArray>> = None;
        for &len in chunks
            .iter()
            .chain(std::iter::once(&(shape[1] - chunks.iter().sum::<i32>())))
        {
            if len == 0 {
                continue;
            }
            let ids = mlxcel_core::slice(input_ids, &[0, start], &[shape[0], start + len]);
            let logits = self.logits(&self.forward_hidden(&ids, None, &mut caches, None));
            out = Some(match out {
                None => logits,
                Some(previous) => mlxcel_core::concatenate(&previous, &logits, 1),
            });
            start += len;
        }
        out.expect("non-empty input")
    }
}

impl LanguageModel for Qwen4ExpModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_sequence(input_ids, None, None)
    }

    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.forward_sequence_last(input_ids, None, None, last_pos)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_sequence(input_ids, input_embeddings, None)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embed_tokens.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_ids()
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.layers.len())
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn supports_batched_prefill(&self) -> bool {
        false
    }

    /// Pad positions would be folded into the GatedDeltaNet state, the PLE
    /// n-gram context and the QSA index, none of which can be trimmed.
    fn supports_padded_prefill(&self) -> bool {
        false
    }

    fn supports_paged_decode_backend(&self) -> bool {
        false
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        self.kv_cache_layer_modes.set(modes);
        self.sequence_state
            .replace_internal(self.make_internal_caches());
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        self.kv_cache_layer_modes.clone_modes()
    }

    fn reset_runtime_state(&self) {
        // Keep the MRoPE fallback: the VLM wrapper fills it while preparing
        // embeddings, right before the generation it belongs to.
        self.sequence_state
            .replace_internal(self.make_internal_caches());
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.make_internal_caches());
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
        self.mrope_state.release_sequence(seq_id);
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_sequence(input_ids, None, seq_id)
    }

    fn forward_last_logits_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.forward_sequence_last(input_ids, None, seq_id, last_pos)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_sequence(input_ids, input_embeddings, seq_id)
    }

    fn forward_last_logits_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.forward_sequence_last(input_ids, input_embeddings, seq_id, last_pos)
    }

    /// Identified rows run one at a time against their own model-owned state.
    /// Anonymous rows stay together in the batched fallback state.
    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        _batch_caches: &mut [&mut [KVCache]],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_rows_sequential(input_ids, seq_ids)
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_with_ids(input_ids, None, batch_caches, mask)
    }
}

#[cfg(test)]
#[path = "qwen4_exp_parity_tests.rs"]
mod parity_tests;

#[cfg(test)]
#[path = "qwen4_exp_load_tests.rs"]
mod load_tests;
