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
//! ## Speculative decoding (issue #2880)
//!
//! A speculative verify block appends `K` positions that may be partly
//! rejected. [`Qwen4ExpModel::forward_speculative`] captures what a rollback
//! needs ([`rollback`]), and [`Qwen4ExpModel::rollback_speculative`] rewinds
//! every state to the accepted prefix: KV and QSA index by trimming,
//! GatedDeltaNet and the PLE convolution window by replay, the PLE n-gram
//! context by recomputation. The same hooks serve the native MTP head
//! (`qwen4_exp_mtp_target`, drafter `qwen4_exp_mtp_drafter`) and the
//! engine's prompt-lookup verify rounds (`rewind_decode_appends`). The MTP
//! drafter consumes the pre-mixer `[B, L, hc_count * hidden]` state, which
//! [`Qwen4ExpModel::forward_hidden_streams`] exposes; the `mtp.*` tensors of
//! a full checkpoint are still dropped by the target's sanitize, because the
//! drafter is loaded from its own directory.
//!
//! Rollback restores per-sequence state and never touches the n-gram table,
//! so it holds for resident and external (`ple_storage`) row stores alike.
//!
//! Out of scope: FP8 checkpoints, the fused QSA Metal kernel, prompt-cache
//! snapshots (#2885), batched prefill, MTP windows with `B > 1` or with
//! image and video requests, DFlash.

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
pub(crate) mod norm;
#[path = "qwen4_exp_ple.rs"]
mod ple;
#[path = "qwen4_exp_ple_cache.rs"]
pub mod ple_cache;
#[path = "qwen4_exp_ple_direct.rs"]
mod ple_direct;
#[path = "qwen4_exp_ple_footprint.rs"]
pub mod ple_footprint;
#[path = "qwen4_exp_ple_io.rs"]
pub mod ple_io;
#[path = "qwen4_exp_ple_manifest.rs"]
pub mod ple_manifest;
#[path = "qwen4_exp_ple_prepare.rs"]
pub mod ple_prepare;
#[path = "qwen4_exp_ple_stats.rs"]
pub mod ple_stats;
#[path = "qwen4_exp_ple_storage.rs"]
pub mod ple_storage;
#[path = "qwen4_exp_qsa.rs"]
mod qsa;
#[path = "qwen4_exp_rollback.rs"]
pub(crate) mod rollback;
#[path = "qwen4_exp_sanitize.rs"]
pub mod sanitize;
#[path = "qwen4_exp_shapes.rs"]
pub(crate) mod shapes;

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
use crate::models::speculative_exactness::{
    BlockChainExactness, ProbeKey, compare_block_against_chain, mtp_exactness_gate,
};

use gated_delta::Qwen4ExpGatedDeltaNet;
pub(crate) use hyper::GatedResidual;
use moe::Qwen4ExpMoeBlock;
use ple::{PleCache, PleLayer};
pub(crate) use qsa::RopeContext;
use qsa::{QsaAttention, QsaIndexCache};
use rollback::{LayerCapture, RollbackLog, RollbackParts, SpeculativeCapture};

/// Test-only recorder of intermediate activations, so the rollback tests can
/// bisect a block-versus-chain difference to the stage that introduced it.
/// Compiles to nothing outside tests.
#[cfg(test)]
pub(crate) mod trace {
    use std::cell::RefCell;

    use mlxcel_core::{MlxArray, UniquePtr};

    thread_local! {
        static TRACE: RefCell<Option<Vec<(String, UniquePtr<MlxArray>)>>> =
            const { RefCell::new(None) };
    }

    pub(crate) fn start() {
        TRACE.with(|trace| *trace.borrow_mut() = Some(Vec::new()));
    }

    pub(crate) fn take() -> Vec<(String, UniquePtr<MlxArray>)> {
        TRACE.with(|trace| trace.borrow_mut().take().unwrap_or_default())
    }

    pub(crate) fn record(label: &str, array: &MlxArray) {
        TRACE.with(|trace| {
            if let Some(records) = trace.borrow_mut().as_mut() {
                records.push((label.to_string(), mlxcel_core::copy(array)));
            }
        });
    }
}

#[cfg(not(test))]
pub(crate) mod trace {
    #[inline(always)]
    pub(crate) fn record(_label: &str, _array: &mlxcel_core::MlxArray) {}
}

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

impl Qwen4ExpCache {
    /// Positions the attention layers hold, the sequence position every
    /// caller reads as the cache offset.
    pub(crate) fn offset(caches: &[Self]) -> i32 {
        caches
            .iter()
            .find_map(|cache| match cache {
                Qwen4ExpCache::Attention { kv, .. } => Some(kv.offset),
                Qwen4ExpCache::Linear { .. } => None,
            })
            .unwrap_or(0)
    }
}

enum Mixer {
    Linear(Qwen4ExpGatedDeltaNet),
    Attention(QsaAttention),
}

/// One decoder layer: a GatedDeltaNet or QSA mixer and the shared-expert MoE,
/// each behind a gated hyper-connection, with an optional PLE embedding.
///
/// Used by: [`Qwen4ExpModel`], the `qwen4_exp_mtp` drafter (one QSA layer
/// loaded under the `layers.0` prefix).
pub(crate) struct DecoderLayer {
    mixer: Mixer,
    ple: Option<PleLayer>,
    mlp: Qwen4ExpMoeBlock,
    attn_hyper_connection: GatedResidual,
    mlp_hyper_connection: GatedResidual,
}

impl DecoderLayer {
    /// Load decoder layer `layer_idx` of `config` from the tensors under
    /// `prefix` (`model.layers.{i}` in a target checkpoint, `layers.0` in a
    /// split-out MTP head).
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &ResolvedQwen4ExpConfig,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, String> {
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

    pub(crate) fn is_linear(&self) -> bool {
        matches!(self.mixer, Mixer::Linear(_))
    }

    /// An empty per-sequence cache for this layer.
    pub(crate) fn make_cache(&self, kv_mode: KVCacheMode) -> Qwen4ExpCache {
        match self.mixer {
            Mixer::Linear(_) => Qwen4ExpCache::Linear {
                delta: GatedDeltaCache::new(),
                ple: self.ple.as_ref().map(|_| PleCache::default()),
            },
            Mixer::Attention(_) => Qwen4ExpCache::Attention {
                kv: KVCache::new_with_mode(kv_mode),
                index: QsaIndexCache::default(),
            },
        }
    }

    pub(crate) fn forward(
        &self,
        hidden: UniquePtr<MlxArray>,
        tokens: Option<&[Vec<i64>]>,
        cache: &mut Qwen4ExpCache,
        rope: &RopeContext<'_>,
    ) -> UniquePtr<MlxArray> {
        self.forward_captured(hidden, tokens, cache, rope, None)
    }

    /// [`Self::forward`] that also records what [`Self::rollback`] needs to
    /// unwind part of this call, when `capture` is given.
    ///
    /// A captured call is a speculative verify block, and its rows have to
    /// be the bytes single-row decode produces at each position or a
    /// temperature-0 verify would silently differ from classic decode. Every
    /// component therefore runs in its decode-equivalent form here: the QSA
    /// layer attends one query at a time, the convolutions are evaluated one
    /// output row at a time, the GatedDeltaNet projections take the fused
    /// decode matrix, the gate projections take the plain kernel rather than
    /// the prefill-only gathered form, and the MoE gathers row by row.
    pub(crate) fn forward_captured(
        &self,
        hidden: UniquePtr<MlxArray>,
        tokens: Option<&[Vec<i64>]>,
        cache: &mut Qwen4ExpCache,
        rope: &RopeContext<'_>,
        capture: Option<&mut Vec<LayerCapture>>,
    ) -> UniquePtr<MlxArray> {
        let exact = capture.is_some();
        let mut ple_capture = None;
        let hidden = match (&self.ple, &mut *cache) {
            (
                Some(ple),
                Qwen4ExpCache::Linear {
                    ple: Some(ple_cache),
                    ..
                },
            ) => {
                let tokens = tokens.expect("PLE layers need the input token ids");
                let update = ple.forward_captured(
                    &hidden,
                    tokens,
                    ple_cache,
                    exact.then_some(&mut ple_capture),
                    exact,
                );
                trace::record("ple update", &update);
                mlxcel_core::add(&hidden, &update)
            }
            (None, _) => hidden,
            (Some(_), _) => unreachable!("a PLE layer always gets a PLE cache"),
        };

        let (mixed, inject) = self.attn_hyper_connection.split(&hidden, exact);
        trace::record("attn mixed", &mixed);
        trace::record("attn inject weights", &inject);
        let mut delta_capture = None;
        let branch = match (&self.mixer, cache) {
            (Mixer::Linear(delta_net), Qwen4ExpCache::Linear { delta, .. }) => delta_net
                .forward_captured(
                    &mixed,
                    Some(delta),
                    exact.then_some(&mut delta_capture),
                    exact,
                ),
            (Mixer::Attention(attention), Qwen4ExpCache::Attention { kv, index }) => {
                if exact {
                    attention.forward_per_position(&mixed, rope, kv, index)
                } else {
                    attention.forward(&mixed, rope, kv, index)
                }
            }
            _ => unreachable!("layer caches are built from the same layer list"),
        };
        if let Some(capture) = capture {
            capture.push(match delta_capture {
                Some(delta) => LayerCapture::Linear {
                    delta,
                    ple: ple_capture,
                },
                None => LayerCapture::Attention,
            });
        }
        trace::record("mixer branch", &branch);
        let hidden = self.attn_hyper_connection.inject(&branch, &hidden, &inject);
        trace::record("after mixer", &hidden);

        let (mixed, inject) = self.mlp_hyper_connection.split(&hidden, exact);
        trace::record("mlp mixed", &mixed);
        let branch = self.mlp.forward(&mixed, exact);
        trace::record("mlp branch", &branch);
        let out = self.mlp_hyper_connection.inject(&branch, &hidden, &inject);
        trace::record("after mlp", &out);
        out
    }

    /// Drop the last `n` positions of an attention layer's cache without a
    /// capture (the `qwen4_exp_mtp` drafter's in-round trim); a linear
    /// layer cannot be trimmed.
    pub(crate) fn trim_cache(&self, cache: &mut Qwen4ExpCache, n: i32) -> Result<(), String> {
        match (&self.mixer, cache) {
            (Mixer::Attention(attention), Qwen4ExpCache::Attention { kv, index }) => {
                attention.rollback(kv, index, n, true);
                Ok(())
            }
            _ => Err(
                "Qwen4-Exp: a GatedDeltaNet layer cannot be trimmed without a capture".to_string(),
            ),
        }
    }

    /// Rewind `cache` to the first `keep` positions of the block `capture`
    /// describes, dropping `trim` positions.
    pub(crate) fn rollback(
        &self,
        cache: &mut Qwen4ExpCache,
        capture: &LayerCapture,
        keep: i32,
        trim: i32,
        parts: RollbackParts,
    ) -> Result<(), String> {
        match (&self.mixer, cache, capture) {
            (
                Mixer::Linear(delta_net),
                Qwen4ExpCache::Linear { delta, ple },
                LayerCapture::Linear {
                    delta: delta_capture,
                    ple: ple_capture,
                },
            ) => {
                if parts.gdn {
                    delta_net.rollback(delta, delta_capture, keep, trim);
                } else {
                    delta.offset -= trim;
                }
                match (&self.ple, ple, ple_capture) {
                    (Some(layer), Some(cache), Some(capture)) => {
                        layer.rollback(cache, capture, keep as usize, parts);
                    }
                    (None, None, None) => {}
                    _ => {
                        return Err(
                            "Qwen4-Exp rollback: the PLE capture does not match the layer"
                                .to_string(),
                        );
                    }
                }
                Ok(())
            }
            (
                Mixer::Attention(attention),
                Qwen4ExpCache::Attention { kv, index },
                LayerCapture::Attention,
            ) => {
                attention.rollback(kv, index, trim, parts.qsa_blocks);
                Ok(())
            }
            _ => Err("Qwen4-Exp rollback: the capture does not match the layer kind".to_string()),
        }
    }
}

/// Logits and the pre-mixer residual streams of one speculative forward,
/// after its rollback state has been recorded.
pub(crate) struct SpeculativeForward {
    pub(crate) logits: UniquePtr<MlxArray>,
    /// `[B, L, hc_count * hidden]`, the state the native MTP head consumes.
    pub(crate) streams: UniquePtr<MlxArray>,
    pub(crate) capture: SpeculativeCapture,
}

/// Positions the exactness probe prefills before comparing a verify block
/// against the single-token chain. Short on purpose: the probe runs once per
/// process on the real checkpoint and compares the block against the chain,
/// not the prefill against anything.
const PROBE_PROMPT_LEN: usize = 48;
/// Prompt and block draws per probe, varied so a last-ulp difference that
/// happens to land on equal bytes once does not read as exact.
const PROBE_DRAWS: usize = 2;

pub struct Qwen4ExpModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    hyper_connection_mixer: GatedResidual,
    lm_head: Option<UnifiedLinear>,
    config: ResolvedQwen4ExpConfig,
    mrope: InterleavedMRoPE,
    sequence_state: ModelOwnedSequenceState<Qwen4ExpCache>,
    /// The most recent speculative forwards of each sequence, for
    /// `rewind_decode_appends` and the MTP adapter's finalize. One entry per
    /// sequence slot; the `Vec` is the container `ModelOwnedSequenceState`
    /// keys by sequence.
    rollback_log: ModelOwnedSequenceState<RollbackLog>,
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
            .map(|layer_idx| {
                DecoderLayer::from_weights(
                    weights,
                    config,
                    &format!("model.layers.{layer_idx}"),
                    layer_idx,
                )
            })
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
        let mrope = Self::mrope_for(config);
        let mut model = Self {
            embed_tokens,
            layers,
            hyper_connection_mixer,
            lm_head,
            config: config.clone(),
            mrope,
            sequence_state: ModelOwnedSequenceState::new(Vec::new()),
            rollback_log: ModelOwnedSequenceState::new(vec![RollbackLog::default()]),
            kv_cache_layer_modes: KvCacheLayerModes::new(),
            mrope_state: MRopeState::new(),
        };
        model.sequence_state = ModelOwnedSequenceState::new(model.make_internal_caches());
        Ok(model)
    }

    /// The interleaved MRoPE tables `config` describes.
    ///
    /// Used by: [`Self::from_weights`], the `qwen4_exp_mtp` drafter.
    pub(crate) fn mrope_for(config: &ResolvedQwen4ExpConfig) -> InterleavedMRoPE {
        InterleavedMRoPE::new_with_backend_frequencies(
            config.rope_dims(),
            config.rope.rope_theta,
            config.rope.mrope_section.clone(),
        )
    }

    pub fn config(&self) -> &ResolvedQwen4ExpConfig {
        &self.config
    }

    pub(crate) fn make_internal_caches(&self) -> Vec<Qwen4ExpCache> {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| {
                layer.make_cache(self.kv_cache_layer_modes.mode_for_layer(layer_idx))
            })
            .collect()
    }

    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// `[3, B, L]` text positions continuing from `offset`.
    ///
    /// Used by: [`Self::forward_hidden_streams`], the `qwen4_exp_mtp` drafter.
    pub(crate) fn text_positions(batch: i32, seq_len: i32, offset: i32) -> UniquePtr<MlxArray> {
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

    /// Run the layer stack: the `[B, L, hc_count * hidden]` residual streams
    /// before the final mixer, which is what the native MTP head consumes
    /// (upstream captures it as the `capture_layer_ids == []` hidden).
    ///
    /// With `capture`, every layer records what a rollback of this call needs.
    pub(crate) fn forward_hidden_streams(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Qwen4ExpCache],
        position_ids: Option<UniquePtr<MlxArray>>,
        mut capture: Option<&mut Vec<LayerCapture>>,
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
            None => Self::text_positions(batch, seq_len, Qwen4ExpCache::offset(caches)),
        };
        let rope = RopeContext::new(&self.mrope, positions);
        let tokens = self
            .layers
            .iter()
            .any(|layer| layer.ple.is_some())
            .then(|| Self::host_tokens(input_ids));

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward_captured(
                hidden,
                tokens.as_deref(),
                cache,
                &rope,
                capture.as_deref_mut(),
            );
            if seq_len == 1 {
                // Match mlx-vlm's decode scheduling: submit each layer while
                // the CPU builds the next one instead of queuing the whole graph.
                mlxcel_core::async_eval(&hidden);
            }
        }
        hidden
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
        let streams =
            self.forward_hidden_streams(input_ids, input_embeddings, caches, position_ids, None);
        self.collapse_streams(&streams, false)
    }

    /// The final mixer: `[.., hc_count * hidden]` streams to the `[.., hidden]`
    /// state the LM head reads (upstream `_mtp_logits_hidden`).
    /// `decode_equivalent` is the speculative verify block's contract
    /// (`DecoderLayer::forward_captured`).
    pub(crate) fn collapse_streams(
        &self,
        streams: &MlxArray,
        decode_equivalent: bool,
    ) -> UniquePtr<MlxArray> {
        self.hyper_connection_mixer
            .collapse(streams, decode_equivalent)
    }

    pub(crate) fn logits(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        match &self.lm_head {
            Some(lm_head) => lm_head.forward(hidden),
            None => self.embed_tokens.as_linear(hidden),
        }
    }

    /// [`Self::logits`] for a speculative verify block: an untied LM head
    /// is applied row by row where its kernel would otherwise depend on the
    /// row count (see `linear::decode_equivalent_projection`).
    fn logits_decode_equivalent(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        match &self.lm_head {
            Some(lm_head) => linear::decode_equivalent_projection(lm_head, hidden),
            None => self.embed_tokens.as_linear(hidden),
        }
    }

    /// A speculative verify forward over `input_ids`: the logits, the
    /// pre-mixer streams and the rollback capture of the appended block.
    pub(crate) fn forward_speculative(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Qwen4ExpCache],
        position_ids: Option<UniquePtr<MlxArray>>,
    ) -> SpeculativeForward {
        let appended = mlxcel_core::array_shape(input_ids)[1];
        let mut layers = Vec::with_capacity(self.layers.len());
        let streams =
            self.forward_hidden_streams(input_ids, None, caches, position_ids, Some(&mut layers));
        let collapsed = self.collapse_streams(&streams, true);
        trace::record("collapse", &collapsed);
        let logits = self.logits_decode_equivalent(&collapsed);
        trace::record("logits", &logits);
        SpeculativeForward {
            logits,
            streams,
            capture: SpeculativeCapture { appended, layers },
        }
    }

    /// Rewind `caches` to the first `keep` positions of the block `capture`
    /// describes; `keep == capture.appended` is a no-op.
    pub(crate) fn rollback_speculative(
        &self,
        caches: &mut [Qwen4ExpCache],
        capture: &SpeculativeCapture,
        keep: i32,
    ) -> Result<(), String> {
        self.rollback_speculative_parts(caches, capture, keep, RollbackParts::ALL)
    }

    /// [`Self::rollback_speculative`] restoring only `parts`; the rollback
    /// tests use it to show what each part is for.
    pub(crate) fn rollback_speculative_parts(
        &self,
        caches: &mut [Qwen4ExpCache],
        capture: &SpeculativeCapture,
        keep: i32,
        parts: RollbackParts,
    ) -> Result<(), String> {
        if keep < 0 || keep > capture.appended {
            return Err(format!(
                "Qwen4-Exp rollback: cannot keep {keep} of a {}-position block",
                capture.appended
            ));
        }
        if capture.layers.len() != self.layers.len() || caches.len() != self.layers.len() {
            return Err(format!(
                "Qwen4-Exp rollback: {} captured layers and {} caches for {} layers",
                capture.layers.len(),
                caches.len(),
                self.layers.len()
            ));
        }
        let trim = capture.appended - keep;
        if trim == 0 {
            return Ok(());
        }
        for ((layer, cache), layer_capture) in self
            .layers
            .iter()
            .zip(caches.iter_mut())
            .zip(&capture.layers)
        {
            layer.rollback(cache, layer_capture, keep, trim, parts)?;
        }
        Ok(())
    }

    /// Unwind the last `n` positions appended by the speculative forwards in
    /// `log`, newest first. A block that is only partly unwound stays in the
    /// log, truncated, so a later rewind can reach into it again.
    fn rewind_appends(
        &self,
        caches: &mut [Qwen4ExpCache],
        log: &mut RollbackLog,
        mut n: i32,
    ) -> Result<(), String> {
        if n > log.appended() {
            return Err(format!(
                "Qwen4-Exp rewind: {n} positions requested but the {} logged speculative \
                 forward(s) appended {}",
                log.len(),
                log.appended()
            ));
        }
        while n > 0 {
            let capture = log
                .pop()
                .ok_or_else(|| "Qwen4-Exp rewind: no speculative forward left".to_string())?;
            if n >= capture.appended {
                self.rollback_speculative(caches, &capture, 0)?;
                n -= capture.appended;
            } else {
                let keep = capture.appended - n;
                self.rollback_speculative(caches, &capture, keep)?;
                log.push(capture.truncated(keep));
                n = 0;
            }
        }
        Ok(())
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

    fn with_rollback_log<R>(
        &self,
        seq_id: Option<SequenceId>,
        f: impl FnOnce(&mut RollbackLog) -> R,
    ) -> R {
        self.rollback_log.with_or_create_sequence_state(
            seq_id,
            || vec![RollbackLog::default()],
            |logs| f(&mut logs[0]),
        )
    }

    /// The classic and engine forward: inside an engine speculative scope the
    /// appended block is captured for `rewind_decode_appends`; any other
    /// forward commits what came before it and drops the captures.
    fn forward_sequence_hidden(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        let ids_shape = mlxcel_core::array_shape(input_ids);
        let speculative = mlxcel_core::cache::speculative_append_active();
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| {
                let positions = self.resolve_positions(
                    seq_id,
                    ids_shape[0],
                    ids_shape[1],
                    Qwen4ExpCache::offset(caches),
                );
                if speculative {
                    let mut layers = Vec::with_capacity(self.layers.len());
                    let streams = self.forward_hidden_streams(
                        input_ids,
                        input_embeddings,
                        caches,
                        positions,
                        Some(&mut layers),
                    );
                    self.with_rollback_log(seq_id, |log| {
                        log.push(SpeculativeCapture {
                            appended: ids_shape[1],
                            layers,
                        });
                    });
                    self.collapse_streams(&streams, true)
                } else {
                    self.with_rollback_log(seq_id, RollbackLog::clear);
                    self.forward_hidden(input_ids, input_embeddings, caches, positions)
                }
            },
        )
    }

    fn forward_sequence(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        let hidden = self.forward_sequence_hidden(input_ids, input_embeddings, seq_id);
        if mlxcel_core::cache::speculative_append_active() {
            self.logits_decode_equivalent(&hidden)
        } else {
            self.logits(&hidden)
        }
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

    // === Sequence-routed speculative hooks (MTP target adapter) ============
    //
    // `MtpTarget` is `&self`-only with no cache parameter, so the adapter
    // (`qwen4_exp_mtp_target`) stays a stateless view over the model and
    // every call resolves its sequence slot here, as the Qwen 3.5 adapter
    // does. The verify capture goes through the same per-sequence log the
    // engine's prompt-lookup rewind uses, so the two paths share one
    // rollback.

    /// Prefill `input_ids` on the sequence's slot: the last position's
    /// logits (`[B, 1, vocab]`, the LM head applied to that row alone, as
    /// the classic engine prefill applies it, so the first bonus token is the
    /// token classic decode samples) and the pre-mixer streams of every
    /// position. Not speculative; clears the log.
    ///
    /// Used by: `Qwen4ExpMtpTargetAdapter::prefill_and_seed`.
    pub(crate) fn forward_prefill_with_streams_for_sequence(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.prefill_with_streams_in_chunks(
            input_ids,
            seq_id,
            mlxcel_core::generate::prefill_chunk_len(),
        )
    }

    /// [`Self::forward_prefill_with_streams_for_sequence`] under an explicit
    /// chunk size. A prompt is forwarded in the pieces of the plan classic
    /// prefill builds, because a one-call prefill of a prompt longer than the
    /// chunk reaches different KV and recurrent bytes than the chunked one and
    /// its temperature-0 output drifts from classic decode.
    fn prefill_with_streams_in_chunks(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        chunk: usize,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let ids_shape = mlxcel_core::array_shape(input_ids);
        self.with_rollback_log(seq_id, RollbackLog::clear);
        let caps = mlxcel_core::prefill_plan::PrefillCaps::for_model(self, false);
        let plan = mlxcel_core::prefill_plan::PrefillPlan::new(ids_shape[1] as usize, chunk, caps);
        let pieces = plan.pieces();
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| {
                let mut logits = None;
                let mut all_streams: Option<UniquePtr<MlxArray>> = None;
                for (index, piece) in pieces.iter().enumerate() {
                    let (start, stop) = (piece.range.start as i32, piece.range.end as i32);
                    let ids = mlxcel_core::slice(input_ids, &[0, start], &[ids_shape[0], stop]);
                    let positions = self.resolve_positions(
                        seq_id,
                        ids_shape[0],
                        stop - start,
                        Qwen4ExpCache::offset(caches),
                    );
                    let streams = self.forward_hidden_streams(&ids, None, caches, positions, None);
                    if index + 1 == pieces.len() {
                        let hidden = self.collapse_streams(&streams, false);
                        let shape = mlxcel_core::array_shape(&hidden);
                        let last = shape[1] - 1;
                        let last_hidden = mlxcel_core::slice(
                            &hidden,
                            &[0, last, 0],
                            &[shape[0], last + 1, shape[2]],
                        );
                        logits = Some(self.logits(&last_hidden));
                    } else {
                        // As classic prefill: one piece's transients are
                        // released before the next piece's graph is built.
                        mlxcel_core::eval(&streams);
                        mlxcel_core::clear_memory_cache();
                    }
                    all_streams = Some(match all_streams {
                        None => streams,
                        Some(previous) => mlxcel_core::concatenate(&previous, &streams, 1),
                    });
                }
                (
                    logits.expect("a prefill has at least one piece"),
                    all_streams.expect("a prefill has at least one piece"),
                )
            },
        )
    }

    /// Verify forward on the sequence's slot: `(logits, pre-mixer streams)`;
    /// the block's rollback capture is pushed onto the sequence's log for
    /// [`Self::rollback_last_speculative_for_sequence`].
    ///
    /// Used by: `Qwen4ExpMtpTargetAdapter::verify_forward`.
    pub(crate) fn forward_speculative_for_sequence(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let ids_shape = mlxcel_core::array_shape(input_ids);
        let out = self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| {
                let positions = self.resolve_positions(
                    seq_id,
                    ids_shape[0],
                    ids_shape[1],
                    Qwen4ExpCache::offset(caches),
                );
                self.forward_speculative(input_ids, caches, positions)
            },
        );
        self.with_rollback_log(seq_id, |log| log.push(out.capture));
        (out.logits, out.streams)
    }

    /// Rewind the sequence's slot to the first `keep` positions of its most
    /// recent speculative forward, which is then committed (removed from the
    /// log).
    ///
    /// Used by: `Qwen4ExpMtpTargetAdapter::verify_finalize`.
    pub(crate) fn rollback_last_speculative_for_sequence(
        &self,
        seq_id: Option<SequenceId>,
        keep: i32,
    ) -> Result<(), String> {
        let capture = self
            .with_rollback_log(seq_id, RollbackLog::pop)
            .ok_or_else(|| "Qwen4-Exp rollback: no speculative forward to roll back".to_string())?;
        self.sequence_state.with_sequence_state(seq_id, |caches| {
            self.rollback_speculative(caches, &capture, keep)
        })
    }

    /// Keep every position of the sequence's most recent speculative forward
    /// (a fully accepted block), dropping its capture.
    ///
    /// Used by: `Qwen4ExpMtpTargetAdapter::verify_finalize`.
    pub(crate) fn commit_last_speculative_for_sequence(&self, seq_id: Option<SequenceId>) {
        let _ = self.with_rollback_log(seq_id, RollbackLog::pop);
    }

    /// Absolute offset of the sequence's attention caches (the value the
    /// adapter rebinds the drafter with after a rollback).
    ///
    /// Used by: `Qwen4ExpMtpTargetAdapter`.
    pub(crate) fn speculative_cache_offset_for_sequence(&self, seq_id: Option<SequenceId>) -> i32 {
        self.sequence_state
            .with_sequence_state(seq_id, |caches| Qwen4ExpCache::offset(caches))
    }

    /// Whether MTP may engage on this checkpoint at `block_size`.
    ///
    /// Metal first: the GatedDeltaNet state is float32 end to end, so the
    /// precise Metal kernel gives a `T = K` block the same per-step rounding
    /// as `K` single steps, while the ops fallback runs a block through the
    /// chunked scan and a step through the fused single step, which do not
    /// agree bit for bit. Then [`Self::probe_block_chain_exactness`] measures
    /// everything else in the forward (the `M = 1` versus `M = K` kernel
    /// dispatch of every quantized projection), memoized per (model, width)
    /// through [`mtp_exactness_gate`], which also owns the decline log and
    /// the `MLXCEL_MTP_ALLOW_INEXACT` override.
    ///
    /// Used by: `speculative_burst::mtp_capable_target`, the offline
    /// `run_offline_mtp` gate.
    pub fn mtp_exactness_allows(&self, block_size: usize) -> bool {
        if !mlxcel_core::metal_is_available() {
            return false;
        }
        let key = ProbeKey {
            block_size: block_size as u32,
            hidden_size: self.config.raw.hidden_size as u32,
            num_hidden_layers: self.config.raw.num_hidden_layers as u32,
        };
        mtp_exactness_gate(key, || self.probe_block_chain_exactness(block_size))
    }

    /// Measure whether a `block_size`-wide verify forward is byte-identical
    /// to `block_size` single-token forwards after the same prefill, on
    /// synthetic tokens and fresh caches.
    pub fn probe_block_chain_exactness(&self, block_size: usize) -> BlockChainExactness {
        if block_size < 2 {
            return BlockChainExactness::NotRun("block width below 2 drafts nothing");
        }
        let vocab = self.config.raw.vocab_size;
        if vocab < 3 {
            return BlockChainExactness::NotRun("degenerate vocabulary");
        }
        for draw in 0..PROBE_DRAWS {
            let verdict = self.probe_one_draw(block_size, vocab, draw);
            if !verdict.is_equal() {
                return verdict;
            }
        }
        BlockChainExactness::Equal
    }

    fn probe_one_draw(&self, block_size: usize, vocab: usize, draw: usize) -> BlockChainExactness {
        // Synthetic ids, varied per draw and kept off the PLE EOS so the
        // n-gram context does not reset mid-probe.
        let eos = self.config.eos_token_id.unwrap_or(-1);
        let salt = draw * 977 + 1;
        let wrap = |i: usize, stride: usize, offset: usize| {
            let mut id = ((i * stride + offset + salt) % vocab) as i64;
            if id == eos {
                id = (id + 1) % vocab as i64;
            }
            id as i32
        };
        let prompt: Vec<i32> = (0..PROBE_PROMPT_LEN).map(|i| wrap(i, 7, 1)).collect();
        let block: Vec<i32> = (0..block_size).map(|i| wrap(i, 13, 3)).collect();
        let as_input =
            |tokens: &[i32]| mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
        let position_bytes = |logits: &MlxArray, index: i32| -> Vec<u8> {
            let shape = mlxcel_core::array_shape(logits);
            let row = mlxcel_core::slice(logits, &[0, index, 0], &[shape[0], index + 1, shape[2]]);
            mlxcel_core::array_to_raw_bytes(&row)
        };

        let mut chain_caches = self.make_internal_caches();
        let _ = self.forward_hidden(&as_input(&prompt), None, &mut chain_caches, None);
        let chain_positions: Vec<Vec<u8>> = block
            .iter()
            .map(|token| {
                let logits = self.logits(&self.forward_hidden(
                    &as_input(&[*token]),
                    None,
                    &mut chain_caches,
                    None,
                ));
                position_bytes(&logits, 0)
            })
            .collect();

        let mut block_caches = self.make_internal_caches();
        let _ = self.forward_hidden(&as_input(&prompt), None, &mut block_caches, None);
        let out = self.forward_speculative(&as_input(&block), &mut block_caches, None);
        let block_positions: Vec<Vec<u8>> = (0..block_size)
            .map(|i| position_bytes(&out.logits, i as i32))
            .collect();
        compare_block_against_chain(&block_positions, &chain_positions)
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

    fn embed_tokens_module(&self) -> Option<UnifiedEmbedding> {
        Some(self.embed_tokens.clone_shared())
    }

    fn lm_head_module(&self) -> Option<UnifiedLinear> {
        self.lm_head.as_ref().map(UnifiedLinear::clone_shared)
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

    /// A verify block's rejected tail is unwound by [`Self::rewind_decode_appends`]
    /// from the capture the speculative forward recorded (issue #2880).
    /// Decode lookahead stays off: a pipelined step's misprediction would pay
    /// the same GatedDeltaNet and PLE replay for one token.
    fn supports_speculative_verify_rewind(&self) -> bool {
        true
    }

    fn rewind_decode_appends(&self, seq_id: SequenceId, n: i32) -> Result<(), String> {
        self.rollback_log
            .with_existing_sequence_state(seq_id, |logs| {
                self.sequence_state
                    .with_existing_sequence_state(seq_id, |caches| {
                        self.rewind_appends(caches, &mut logs[0], n)
                    })
            })
            .map_err(|err| format!("Qwen4-Exp rewind, sequence {seq_id}: {err}"))??
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        self.kv_cache_layer_modes.set(modes);
        self.sequence_state
            .replace_internal(self.make_internal_caches());
        self.rollback_log
            .replace_internal(vec![RollbackLog::default()]);
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        self.kv_cache_layer_modes.clone_modes()
    }

    /// Gated-delta linear layers hold no KV (issue #2790).
    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<KVCacheMode>>> {
        self.kv_cache_layer_modes
            .storage_modes(|layer_idx| match self.layers.get(layer_idx) {
                Some(layer) if !layer.is_linear() => crate::models::model_owned::LayerKv::Full,
                _ => crate::models::model_owned::LayerKv::None,
            })
    }

    fn reset_runtime_state(&self) {
        // Keep the MRoPE fallback: the VLM wrapper fills it while preparing
        // embeddings, right before the generation it belongs to.
        self.sequence_state
            .replace_internal(self.make_internal_caches());
        self.rollback_log
            .replace_internal(vec![RollbackLog::default()]);
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.make_internal_caches());
        self.rollback_log
            .prepare_sequence_state(seq_id, vec![RollbackLog::default()]);
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
        self.rollback_log.release_sequence_state(seq_id);
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
pub(crate) mod parity_tests;

#[cfg(test)]
#[path = "qwen4_exp_load_tests.rs"]
mod load_tests;

#[cfg(test)]
#[path = "qwen4_exp_rollback_tests.rs"]
mod rollback_tests;
