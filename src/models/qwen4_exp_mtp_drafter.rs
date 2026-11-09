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

//! The Qwen3.8-Flash-Next (`qwen4_exp`) native MTP drafter
//! (`model_type: "qwen4_exp_mtp"`, issue #2880).
//!
//! Rust port of mlx-vlm's `Qwen4ExpMTPDraftModel`
//! (`mlx_vlm/speculative/drafters/qwen4_exp_mtp/qwen4_exp_mtp.py` over
//! `mtp_base.py`). It lives in this crate rather than in `mlxcel-core`
//! because its decoder layer is the target's own QSA layer, MoE block and
//! hyper-connections ([`crate::models::qwen4_exp`]), the arrangement the
//! GLM-4.7-Flash drafter established (issue #1326): core resolves the kind
//! and refuses the build by name, [`crate::models::drafter_loader`] builds it.
//!
//! ## Contract with the target
//!
//! The head consumes the target's **pre-mixer** residual streams,
//! `[B, L, hc_count * hidden_size]`, captured before `hyper_connection_mixer`
//! (upstream `speculative_draft_hidden`). The Qwen 3.5 adapter hands its
//! drafter the post-final-norm `[B, L, hidden]` state; feeding that here is
//! a shape error, which [`Qwen4ExpMtpDraftModel::fuse_inputs`] refuses by
//! name rather than letting a broadcast hide it.
//!
//! Input fusion: `pre_fc_norm_embedding` (zero-centered RMSNorm over
//! `hidden_size`) on the token embedding, `pre_fc_norm_hidden` as one global
//! RMSNorm over all `hc_count * hidden_size` values, `fc_embedding` and
//! `fc_hidden` (`hidden_size -> hidden_size`, no bias) applied to the
//! embedding and to each stream, summed with the embedding broadcast over the
//! stream axis. The fused streams run through one QSA decoder layer with its
//! own KV and index cache (no PLE), and the drafter's own
//! `hyper_connection_mixer` collapses them for the target's LM head. The
//! layer output *before* that mixer is the recurrent input of the next draft
//! step, as upstream's `pre_hc_hidden` is.
//!
//! ## Lifecycle
//!
//! Stateful like the Qwen 3.5 drafter: one cache entry per emitted target
//! position, position `i` pairing `(token_{i+1}, target streams at i)`.
//! `prefill_from_target_hidden` runs the shifted prompt, `draft_block`
//! extends autoregressively from the seed, `accept_verified_tokens` trims the
//! rejected in-round tail and appends the accepted tokens paired with the
//! target's true verify streams. A cleared cache re-anchors on the next
//! `set_shared_kv`, costing draft context and never correctness.
//!
//! ## Block size
//!
//! Upstream runs this head with `default_runtime_block_size = 4` (three
//! drafts) and `prefer_requested_block_size = False`: the configured
//! `block_size` (`mtp_num_hidden_layers + 1 = 2`) is the depth the adaptive
//! controller starts at, and it expands toward the requested
//! `--draft-block-size` (the flat MTP default, 4) as acceptance allows. The
//! port keeps both knobs where they are: nothing here hard-codes 4, and
//! `--draft-block-size 2|3|4` sets the ceiling the policy measures against.
//!
//! The drafter keeps the checkpoint's own precision (no load-time
//! quantization): the target runs in its checkpoint dtype too, and the
//! acceptance-against-cost comparison of the bf16, 8-bit and 4-bit splits
//! needs a real drafter to measure.

use std::path::Path;

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::drafter::{Drafter, DrafterError, DrafterKind, SharedKv};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::layers::{UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::qwen3_vl::InterleavedMRoPE;
use crate::models::qwen4_exp::norm::ZeroCenteredRmsNorm;
use crate::models::qwen4_exp::sanitize::{canonicalize_expert_block, is_mtp_key};
use crate::models::qwen4_exp::{
    DecoderLayer, GatedResidual, Qwen4ExpCache, Qwen4ExpModel, RopeContext, shapes,
};
use crate::models::qwen4_exp_mtp_config::Qwen4ExpMtpConfig;

/// LM head borrowed from the target at `bind()` time.
enum MtpLmHead {
    Linear(UnifiedLinear),
    /// Tied checkpoints project through the target embedding table.
    TiedEmbed(UnifiedEmbedding),
}

/// The head's own layer prefix after the `mtp.` prefix is stripped.
const LAYER_PREFIX: &str = "layers.0";

pub struct Qwen4ExpMtpDraftModel {
    config: Qwen4ExpMtpConfig,
    pre_fc_norm_embedding: ZeroCenteredRmsNorm,
    pre_fc_norm_hidden: ZeroCenteredRmsNorm,
    fc_embedding: UnifiedLinear,
    fc_hidden: UnifiedLinear,
    layer: DecoderLayer,
    hyper_connection_mixer: GatedResidual,
    mrope: InterleavedMRoPE,
    /// The one QSA layer's KV and index cache.
    cache: Qwen4ExpCache,

    target_embed: Option<UnifiedEmbedding>,
    lm_head: Option<MtpLmHead>,

    /// Next round's first proposal and the streams it was predicted from,
    /// computed by the prefill and accept hooks from target streams.
    seed_token: Option<i32>,
    seed_hidden: Option<UniquePtr<MlxArray>>,
    /// Absolute target position of the next cache append (the RoPE offset).
    next_position: i32,
    /// In-round appends since the last accept.
    round_appended: i32,
}

impl std::fmt::Debug for Qwen4ExpMtpDraftModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen4ExpMtpDraftModel")
            .field("block_size", &self.config.block_size)
            .field("hc_count", &self.config.hc_count())
            .field("bound", &self.lm_head.is_some())
            .field("cache_offset", &self.cache_offset())
            .field("next_position", &self.next_position)
            .field("round_appended", &self.round_appended)
            .field("has_seed", &self.seed_token.is_some())
            .finish()
    }
}

impl Qwen4ExpMtpDraftModel {
    /// Load from a drafter directory (`config.json` plus safetensors).
    pub fn from_path(path: &Path) -> Result<Self, DrafterError> {
        let config = Qwen4ExpMtpConfig::from_dir(path)?;
        let mut weights = mlxcel_core::weights::load_weights_from_dir(path)
            .map_err(|reason| DrafterError::WeightLoad { reason })?;
        Self::sanitize_weights(&mut weights, config.layer.raw.num_experts)
            .map_err(|reason| DrafterError::WeightLoad { reason })?;
        Self::from_weights(&weights, config)
    }

    /// Upstream `Qwen4ExpMTPDraftModel.sanitize`: strip the
    /// `language_model.mtp.` / `model.mtp.` / `mtp.` prefix, split a fused
    /// `layers.0.mlp.experts.gate_up_proj` (with its `.scales` / `.biases`)
    /// along the row axis into `switch_mlp.gate_proj` / `switch_mlp.up_proj`
    /// and move `down_proj` alongside. Per-expert `experts.{e}.*` tensors are
    /// stacked the same way the target sanitize stacks them. The router
    /// `layers.0.mlp.gate.weight` is left exactly as stored (upstream keeps
    /// it unquantized).
    pub fn sanitize_weights(weights: &mut WeightMap, num_experts: usize) -> Result<(), String> {
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in keys {
            if !is_mtp_key(&key) {
                continue;
            }
            let stripped = ["language_model.mtp.", "model.mtp.", "mtp."]
                .iter()
                .find_map(|prefix| key.strip_prefix(prefix))
                .expect("is_mtp_key matched one of the prefixes")
                .to_string();
            let value = weights.remove(&key).expect("key just enumerated");
            weights.insert(stripped, value);
        }
        canonicalize_expert_block(weights, &format!("{LAYER_PREFIX}.mlp"), num_experts)
    }

    /// Build from sanitized weights.
    pub fn from_weights(
        weights: &WeightMap,
        config: Qwen4ExpMtpConfig,
    ) -> Result<Self, DrafterError> {
        let layer_config = &config.layer;
        let raw = &layer_config.raw;
        let (hidden, streams) = (raw.hidden_size, config.stream_width());
        let load = |reason: String| DrafterError::WeightLoad { reason };
        let linear =
            |name: &str, rows: usize, columns: usize| -> Result<UnifiedLinear, DrafterError> {
                let (group_size, bits) = layer_config.quantization.for_module(name);
                shapes::expect_matrix(weights, name, (group_size, bits), &[rows, columns])
                    .map_err(load)?;
                UnifiedLinear::from_weights(weights, name, group_size, bits).map_err(load)
            };
        let pre_fc_norm_embedding = ZeroCenteredRmsNorm::from_weights(
            weights,
            "pre_fc_norm_embedding",
            hidden,
            None,
            raw.rms_norm_eps,
        )
        .map_err(load)?;
        // One global normalization over every stream, not one per stream.
        let pre_fc_norm_hidden = ZeroCenteredRmsNorm::from_weights(
            weights,
            "pre_fc_norm_hidden",
            streams,
            None,
            raw.rms_norm_eps,
        )
        .map_err(load)?;
        let fc_embedding = linear("fc_embedding", hidden, hidden)?;
        let fc_hidden = linear("fc_hidden", hidden, hidden)?;
        let layer =
            DecoderLayer::from_weights(weights, layer_config, LAYER_PREFIX, 0).map_err(load)?;
        let hyper_connection_mixer =
            GatedResidual::from_weights(weights, layer_config, "hyper_connection_mixer", false)
                .map_err(load)?;
        Self::warn_unknown_keys(weights);
        let cache = layer.make_cache(KVCacheMode::default());
        Ok(Self {
            mrope: Qwen4ExpModel::mrope_for(layer_config),
            config,
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc_embedding,
            fc_hidden,
            layer,
            hyper_connection_mixer,
            cache,
            target_embed: None,
            lm_head: None,
            seed_token: None,
            seed_hidden: None,
            next_position: 0,
            round_appended: 0,
        })
    }

    /// Tensors no module read. Logged rather than refused: the real split has
    /// not been loaded by this port yet, and a stray auxiliary tensor must not
    /// turn a working drafter into a load error.
    fn warn_unknown_keys(weights: &WeightMap) {
        let known_roots = [
            "pre_fc_norm_embedding.",
            "pre_fc_norm_hidden.",
            "fc_embedding.",
            "fc_hidden.",
            "hyper_connection_mixer.",
            &format!("{LAYER_PREFIX}."),
        ];
        let mut unknown: Vec<&String> = weights
            .keys()
            .filter(|key| !known_roots.iter().any(|root| key.starts_with(root)))
            .collect();
        if unknown.is_empty() {
            return;
        }
        unknown.sort();
        let shown: Vec<&str> = unknown.iter().take(5).map(|s| s.as_str()).collect();
        tracing::warn!(
            count = unknown.len(),
            first = ?shown,
            "qwen4_exp_mtp drafter: tensors no module reads were left in the checkpoint"
        );
    }

    fn cache_offset(&self) -> i32 {
        match &self.cache {
            Qwen4ExpCache::Attention { kv, .. } => kv.offset,
            Qwen4ExpCache::Linear { .. } => 0,
        }
    }

    fn clear_runtime_state(&mut self) {
        self.cache = self.layer.make_cache(KVCacheMode::default());
        self.seed_token = None;
        self.seed_hidden = None;
        self.next_position = 0;
        self.round_appended = 0;
    }

    fn require_bound(&self) -> Result<(), DrafterError> {
        if self.target_embed.is_none() || self.lm_head.is_none() {
            return Err(DrafterError::BindNotCalled);
        }
        Ok(())
    }

    /// Check that `hidden` is the target's pre-mixer streams and return it as
    /// `[B, L, hc_count * hidden_size]` (a `[B, L, hc_count, hidden_size]`
    /// view is accepted, as upstream `_target_hidden` does).
    fn target_streams(&self, hidden: &MlxArray) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let shape = mlxcel_core::array_shape(hidden);
        let expected = self.config.stream_width() as i32;
        let (hc, width) = (
            self.config.hc_count() as i32,
            self.config.hidden_size() as i32,
        );
        match shape.as_slice() {
            [b, l, w] if *w == expected => Ok(mlxcel_core::reshape(hidden, &[*b, *l, expected])),
            [b, l, c, w] if *c == hc && *w == width => {
                Ok(mlxcel_core::reshape(hidden, &[*b, *l, expected]))
            }
            _ => Err(DrafterError::DraftFailed {
                reason: format!(
                    "qwen4_exp_mtp drafter expects the target's pre-mixer hidden of shape \
                     [batch, tokens, hc_count * hidden_size] = [.., .., {expected}] (hc_count \
                     {hc} x hidden_size {width}), got {shape:?}; the post-mixer [.., .., \
                     {width}] hidden the Qwen 3.5 adapter hands out is not this head's input"
                ),
            }),
        }
    }

    /// Upstream `fuse_inputs`: `[B, L, hidden]` token embeddings and
    /// `[B, L, hc_count * hidden]` target streams to the layer input.
    pub(crate) fn fuse_inputs(
        &self,
        token_embed: &MlxArray,
        hidden: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let hidden = self.target_streams(hidden)?;
        let shape = mlxcel_core::array_shape(&hidden);
        let (b, l) = (shape[0], shape[1]);
        let (hc, width) = (
            self.config.hc_count() as i32,
            self.config.hidden_size() as i32,
        );
        let embedding = self
            .fc_embedding
            .forward(&self.pre_fc_norm_embedding.forward(token_embed));
        let normed = self.pre_fc_norm_hidden.forward(&hidden);
        let per_stream = mlxcel_core::reshape(&normed, &[b * l * hc, width]);
        let projected =
            mlxcel_core::reshape(&self.fc_hidden.forward(&per_stream), &[b, l, hc, width]);
        let fused = mlxcel_core::add(&mlxcel_core::expand_dims(&embedding, -2), &projected);
        Ok(mlxcel_core::reshape(&fused, &[b, l, hc * width]))
    }

    /// One head forward over `token_ids` paired with the target streams
    /// `hidden` `[1, S, hc * H]`: appends `S` entries to the cache, advances
    /// the position, and returns `(mixed [1, S, H], streams [1, S, hc * H])`,
    /// upstream's `(logits_hidden, pre_hc_hidden)`.
    fn forward_hidden_stack(
        &mut self,
        token_ids: &[i32],
        hidden: &MlxArray,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), DrafterError> {
        let target_embed = self
            .target_embed
            .as_ref()
            .ok_or(DrafterError::BindNotCalled)?;
        let s = token_ids.len() as i32;
        debug_assert!(s >= 1, "forward_hidden_stack requires at least one token");
        if mlxcel_core::array_shape(hidden)[1] != s {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "qwen4_exp_mtp drafter: {s} tokens but the paired hidden covers {} positions",
                    mlxcel_core::array_shape(hidden)[1]
                ),
            });
        }
        let ids = mlxcel_core::from_slice_i32(token_ids, &[1, s]);
        let token_embed = target_embed.forward(&ids);
        let fused = self.fuse_inputs(&token_embed, hidden)?;
        // Text positions continuing from the drafter's logical position; with
        // an intact cache this is the target's cache offset.
        let positions = Qwen4ExpModel::text_positions(1, s, self.next_position);
        let rope = RopeContext::new(&self.mrope, positions);
        let streams = self.layer.forward(fused, None, &mut self.cache, &rope);
        let mixed = self.hyper_connection_mixer.collapse(&streams, false);
        self.next_position += s;
        Ok((mixed, streams))
    }

    fn project_logits(&self, mixed: &MlxArray) -> Result<UniquePtr<MlxArray>, DrafterError> {
        match self.lm_head.as_ref().ok_or(DrafterError::BindNotCalled)? {
            MtpLmHead::Linear(lm) => Ok(lm.forward(mixed)),
            MtpLmHead::TiedEmbed(embed) => Ok(embed.as_linear(mixed)),
        }
    }

    /// Temperature 0 is argmax; other configs sample through the fused kernel.
    fn sample_one(logits: &MlxArray, sampler: &SamplingConfig) -> i32 {
        let last = mlxcel_core::slice_last_logits(logits);
        let tok = mlxcel_core::fused_sample(
            &last,
            sampler.temperature,
            sampler.top_k,
            sampler.top_p,
            sampler.min_p,
        );
        mlxcel_core::eval(&tok);
        mlxcel_core::item_i32(&tok)
    }

    fn last_position(x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let last = shape[1] - 1;
        mlxcel_core::slice(x, &[0, last, 0], &[shape[0], last + 1, shape[2]])
    }

    /// Next round's seed: the head's prediction at the last position, and the
    /// pre-mixer streams there as its recurrent input.
    fn set_seed(
        &mut self,
        mixed: &MlxArray,
        streams: &MlxArray,
        sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        let logits = self.project_logits(&Self::last_position(mixed))?;
        self.seed_token = Some(Self::sample_one(&logits, sampler));
        self.seed_hidden = Some(Self::last_position(streams));
        Ok(())
    }

    fn trim_cache(&mut self, n: i32) -> Result<(), DrafterError> {
        self.layer
            .trim_cache(&mut self.cache, n)
            .map_err(|reason| DrafterError::DraftFailed { reason })
    }

    /// Test and diagnostic accessor: `(cache_offset, next_position,
    /// round_appended, has_seed)`.
    pub fn state_probe(&self) -> (i32, i32, i32, bool) {
        (
            self.cache_offset(),
            self.next_position,
            self.round_appended,
            self.seed_token.is_some(),
        )
    }

    pub fn config(&self) -> &Qwen4ExpMtpConfig {
        &self.config
    }
}

impl Drafter for Qwen4ExpMtpDraftModel {
    fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        let embed = target
            .embed_tokens_module()
            .ok_or(DrafterError::TargetMissingFeature {
                feature: "embed_tokens_module",
            })?;
        self.lm_head = Some(match target.lm_head_module() {
            Some(lm) => MtpLmHead::Linear(lm),
            None => MtpLmHead::TiedEmbed(embed.clone_shared()),
        });
        self.target_embed = Some(embed);
        Ok(())
    }

    /// The target's stream count, text hidden size and vocabulary must match
    /// the head's, as upstream `validate_target_compatibility` requires. A
    /// wrong `hc_count` has to fail here: past the bind it is a draft error,
    /// which the MTP loop treats as the end of the generation and which
    /// would cut every reply to the seed token.
    fn validate_target_compat(&self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        if let Some(target_hc) = target.residual_stream_count()
            && target_hc != self.config.hc_count()
        {
            return Err(DrafterError::BindFailed {
                reason: format!(
                    "qwen4_exp_mtp drafter is incompatible with this target: drafter \
                     text_config.hc_count = {} but the target carries {target_hc} residual \
                     streams; pair the head split from the same checkpoint",
                    self.config.hc_count()
                ),
            });
        }
        let expected_hidden = self.config.hidden_size() as i32;
        let sentinel = mlxcel_core::from_slice_i32(&[0_i32], &[1, 1]);
        let embedded =
            target
                .embed_tokens(&sentinel)
                .ok_or(DrafterError::TargetMissingFeature {
                    feature: "embed_tokens",
                })?;
        let target_hidden = mlxcel_core::array_shape(&embedded)
            .last()
            .copied()
            .unwrap_or(0);
        if target_hidden != expected_hidden {
            return Err(DrafterError::BindFailed {
                reason: format!(
                    "qwen4_exp_mtp drafter is incompatible with this target: drafter \
                     text_config.hidden_size = {expected_hidden} but the target's text hidden \
                     size = {target_hidden}; pair the head split from the same checkpoint"
                ),
            });
        }
        let drafter_vocab = self.config.vocab_size() as i32;
        let zero_hidden = mlxcel_core::zeros(&[1, 1, target_hidden], mlxcel_core::dtype::FLOAT32);
        let logits = match target.lm_head_module() {
            Some(lm) => lm.forward(&zero_hidden),
            None => match target.embed_tokens_module() {
                Some(embed) => embed.as_linear(&zero_hidden),
                None => return Ok(()),
            },
        };
        let target_vocab = mlxcel_core::array_shape(&logits)
            .last()
            .copied()
            .unwrap_or(0);
        if target_vocab != drafter_vocab {
            return Err(DrafterError::BindFailed {
                reason: format!(
                    "qwen4_exp_mtp drafter vocabulary is incompatible with this target: drafter \
                     text_config.vocab_size = {drafter_vocab} but the target vocabulary = \
                     {target_vocab}"
                ),
            });
        }
        Ok(())
    }

    /// No shared K/V (the head owns its cache); only the position is
    /// consumed. An empty cache re-anchors at the target's offset; a cache
    /// whose position disagrees is stale and is cleared.
    fn set_shared_kv(
        &mut self,
        _shared_kv: SharedKv<'_>,
        kv_offset: usize,
        _position: usize,
        _left_padding: usize,
    ) -> Result<(), DrafterError> {
        if self.cache_offset() == 0 {
            self.next_position = kv_offset as i32;
        } else if self.next_position != kv_offset as i32 {
            tracing::debug!(
                next_position = self.next_position,
                kv_offset,
                "qwen4_exp_mtp drafter position drifted from the target cache; clearing drafter \
                 state and re-anchoring (draft context lost, correctness unaffected)"
            );
            self.clear_runtime_state();
            self.next_position = kv_offset as i32;
        }
        Ok(())
    }

    fn reset(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.clear_runtime_state();
        self.bind(target)
    }

    fn configured_block_size(&self) -> Option<usize> {
        Some(self.config.block_size)
    }

    /// Upstream `prefer_requested_block_size = False`: start at the
    /// configured depth and let the adaptive controller expand toward the
    /// requested block size.
    fn prefer_requested_block_size(&self) -> bool {
        false
    }

    fn prefill_from_target_hidden(
        &mut self,
        prompt_tokens: &[i32],
        hidden: &MlxArray,
        first_bonus: i32,
        sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        self.require_bound()?;
        let p = prompt_tokens.len();
        if p == 0 {
            return Ok(());
        }
        let shape = mlxcel_core::array_shape(hidden);
        if shape.len() != 3 || shape[1] != p as i32 {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "qwen4_exp_mtp drafter prefill: hidden shape {shape:?} does not cover the \
                     {p}-token prompt"
                ),
            });
        }
        self.clear_runtime_state();
        let mut shifted: Vec<i32> = prompt_tokens[1..].to_vec();
        shifted.push(first_bonus);
        let (mixed, streams) = self.forward_hidden_stack(&shifted, hidden)?;
        self.set_seed(&mixed, &streams, sampler)
    }

    fn accept_verified_tokens(
        &mut self,
        verify_hidden: &MlxArray,
        draft_tokens: &[i32],
        accepted: usize,
        new_tokens: &[i32],
        sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        self.require_bound()?;
        let shape = mlxcel_core::array_shape(verify_hidden);
        let block = draft_tokens.len() + 1;
        if shape.len() != 3 || (shape[1] as usize) < block || accepted > draft_tokens.len() {
            self.clear_runtime_state();
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "qwen4_exp_mtp drafter accept: verify_hidden shape {shape:?} does not cover \
                     block={block} / accepted={accepted}"
                ),
            });
        }
        let keep = accepted.min(self.round_appended.max(0) as usize);
        let trim = self.round_appended - keep as i32;
        if trim > 0 {
            self.trim_cache(trim)?;
            self.next_position -= trim;
        }
        let mut tokens: Vec<i32> = draft_tokens[keep..accepted].to_vec();
        if let Some(&last) = new_tokens.last() {
            tokens.push(last);
        }
        if !tokens.is_empty() {
            let start = keep as i32;
            let end = start + tokens.len() as i32;
            let hiddens =
                mlxcel_core::slice(verify_hidden, &[0, start, 0], &[shape[0], end, shape[2]]);
            let (mixed, streams) = self.forward_hidden_stack(&tokens, &hiddens)?;
            self.set_seed(&mixed, &streams, sampler)?;
        }
        self.round_appended = 0;
        Ok(())
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        self.require_bound()?;
        if block_size <= 1 {
            return Ok(Vec::new());
        }
        let mut tokens: Vec<i32> = Vec::with_capacity(block_size - 1);
        self.round_appended = 0;
        let (mut tok, mut h_prev): (i32, UniquePtr<MlxArray>) =
            match (self.seed_token.take(), self.seed_hidden.take()) {
                (Some(seed_tok), Some(seed_hidden)) => {
                    tokens.push(seed_tok);
                    (seed_tok, seed_hidden)
                }
                _ => {
                    let hidden = hidden.ok_or(DrafterError::DraftBlockMissingHidden)?;
                    (last_bonus, self.target_streams(hidden)?)
                }
            };
        while tokens.len() < block_size - 1 {
            let (mixed, streams) = self.forward_hidden_stack(&[tok], &h_prev)?;
            self.round_appended += 1;
            let logits = self.project_logits(&mixed)?;
            tok = Self::sample_one(&logits, sampler);
            tokens.push(tok);
            h_prev = streams;
        }
        tokens.truncate(block_size - 1);
        Ok(tokens)
    }

    fn sanitize(&mut self, weights: &mut WeightMap) -> Result<(), DrafterError> {
        Self::sanitize_weights(weights, self.config.layer.raw.num_experts)
            .map_err(|reason| DrafterError::WeightLoad { reason })
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Mtp
    }
}

#[cfg(test)]
#[path = "qwen4_exp_mtp_drafter_tests.rs"]
mod tests;
