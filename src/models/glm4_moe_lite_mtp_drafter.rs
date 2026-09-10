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

//! `Glm4MoeLiteMtpDraftModel`: the GLM-4.7-Flash next-token-prediction
//! block as a stateful MTP drafter (issue #1326).
//!
//! The drafter directory is produced by `mlxcel split-mtp` from the raw
//! `zai-org/GLM-4.7-Flash` checkpoint and holds the DeepSeek-V3-style nextn
//! block: its own token table, `enorm`, `hnorm`, `eh_proj`, one
//! `glm4_moe_lite` decoder block (MLA plus MoE), `shared_head_norm` and its
//! own untied `lm_head`. Nothing is borrowed from the target. The block is
//! loaded with [`TransformerBlock::from_weights_with_prefix`], so its
//! attention, routing and expert arithmetic are the decoder's own code.
//!
//! ## One step
//!
//! With `e = embed_tokens(token_{t+1})` and `h_t` the target's hidden state
//! at position `t` (post-final-norm by default, see the target adapter):
//!
//! ```text
//! x           = eh_proj(concat(enorm(e), hnorm(h_t), axis=-1))
//! x           = block(x, cache)          # RoPE at cache.offset, own KVCache
//! logits      = lm_head(shared_head_norm(x))
//! hidden_next = x                        # fed back for the next draft step
//! ```
//!
//! ## Lifecycle
//!
//! The same as the Qwen 3.5 MTP drafter: the cache holds one entry per
//! target position consumed, `prefill_from_target_hidden` runs the shifted
//! prompt paired with the target's prompt hidden, `accept_verified_tokens`
//! trims the rejected in-round tail and extends with the accepted tokens
//! paired with the target's true verify hidden, and `draft_block` runs
//! `block_size - 1` autoregressive steps. A drafter whose cache was cleared
//! mid-session re-anchors in `set_shared_kv` and continues with an empty
//! history: that costs acceptance, never correctness, because the target's
//! verify pass alone enforces greedy parity.
//!
//! ## Hidden-state tap
//!
//! The target adapter hands over the post-final-norm hidden by default (the
//! DeepSeek-V3 MTP convention and the Qwen 3.5 adapter's choice);
//! `MLXCEL_GLM_MTP_HIDDEN_TAP=pre` switches the adapter to the pre-norm
//! residual. Measured on `models/glm-4.7-flash-4bit` paired with the 4-bit
//! drafter `mlxcel split-mtp` produced from the raw checkpoint (Apple M5
//! Max, prompt "Explain in two sentences why the sky is blue.", 128 greedy
//! tokens, `block_size = 2`): the post-final-norm tap accepted 55 of 73
//! proposals (mean accepted length 1.753, rate 0.753) and the pre-norm tap
//! accepted 53 of 74 (mean accepted length 1.716, rate 0.716). Post-final-norm
//! wins and stays the default. The emitted 128-token id streams were identical
//! under both taps, as the target's verify pass guarantees.

use std::path::Path;
use std::time::Instant;

use mlxcel_core::drafter::{
    DraftStepProfile, Drafter, DrafterError, DrafterKind, SharedKv, draft_step_profiling_enabled,
};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::create_causal_mask;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::glm4_moe_lite::{ModelArgs, TransformerBlock, kv_b_proj_geometry};
use super::glm4_moe_lite_mtp_config::{GLM4_MOE_LITE_MTP_MODEL_TYPE, Glm4MoeLiteMtpConfig};

/// Key prefix of the decoder block inside the drafter directory.
pub const BLOCK_PREFIX: &str = "model.mtp_block";

/// Every key the drafter consumes starts with one of these.
const ROOT_PREFIXES: [&str; 7] = [
    "model.embed_tokens.",
    "model.enorm.",
    "model.hnorm.",
    "model.eh_proj.",
    "model.shared_head_norm.",
    "lm_head.",
    "model.mtp_block.",
];

/// The GLM-4.7-Flash MTP drafter. Implements [`Drafter`] and is built by
/// `crate::models::drafter_loader::load_drafter` for
/// `model_type == "glm4_moe_lite_mtp"`.
pub struct Glm4MoeLiteMtpDraftModel {
    config: Glm4MoeLiteMtpConfig,
    embed_tokens: UnifiedEmbedding,
    enorm: RMSNorm,
    hnorm: RMSNorm,
    eh_proj: UnifiedLinear,
    block: TransformerBlock,
    shared_head_norm: RMSNorm,
    lm_head: UnifiedLinear,

    /// The block's own KV cache: one entry per target position consumed.
    cache: KVCache,
    /// Set by `bind`; nothing is borrowed, the flag only enforces the
    /// validate-then-bind call order every caller follows.
    bound: bool,

    /// Precomputed seed for the next `draft_block`, from TARGET hidden.
    seed_token: Option<i32>,
    seed_hidden: Option<UniquePtr<MlxArray>>,
    /// Absolute target-sequence position of the next cache append; also the
    /// RoPE offset of the next forward.
    next_position: i32,
    /// In-round cache appends since the last accept.
    round_appended: i32,
    /// Diagnostics only: the trait's `kv_offset` / `position` arguments.
    kv_valid_len: i32,
    position: i32,
    profile: DraftStepProfile,
}

impl std::fmt::Debug for Glm4MoeLiteMtpDraftModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Glm4MoeLiteMtpDraftModel")
            .field("block_size", &self.config.block_size())
            .field("bound", &self.bound)
            .field("cache_offset", &self.cache.offset)
            .field("next_position", &self.next_position)
            .field("round_appended", &self.round_appended)
            .field("has_seed", &self.seed_token.is_some())
            .finish()
    }
}

impl Glm4MoeLiteMtpDraftModel {
    /// Construct from a `split-mtp` output directory.
    pub fn from_path(path: &Path) -> Result<Self, DrafterError> {
        let cfg_path = path.join("config.json");
        let bytes = std::fs::read(&cfg_path).map_err(|e| DrafterError::ConfigIo {
            path: cfg_path.display().to_string(),
            source: e,
        })?;
        let raw: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| DrafterError::ConfigParse {
                path: cfg_path.display().to_string(),
                source: e,
            })?;
        let config: Glm4MoeLiteMtpConfig =
            serde_json::from_value(raw.clone()).map_err(|e| DrafterError::ConfigParse {
                path: cfg_path.display().to_string(),
                source: e,
            })?;
        let config = config.normalize().map_err(DrafterError::Config)?;

        let mut weights = mlxcel_core::weights::load_weights_from_dir(path)
            .map_err(|reason| DrafterError::WeightLoad { reason })?;
        Self::sanitize_weights(&mut weights, config.text_config())?;
        // The same load-time dtype policy the target loader applies: a
        // quantized directory keeps its bf16 side data, a dense one is cast
        // to f16 on Apple Silicon, so the drafter's activations match the
        // target's for the same quantization state.
        if crate::models::sanitize::bf16_to_f16_at_load(config.is_quantized(), Some(&raw)) {
            let _ = crate::models::sanitize::convert_bf16_weights(&mut weights);
        }
        Self::from_weights(&weights, config)
    }

    /// Decompose a `kv_b_proj` shipped under the block prefix into the
    /// `embed_q` / `unembed_out` pair. `split-mtp` already does this; a
    /// hand-assembled directory that kept `kv_b_proj` loads through the same
    /// helper the target sanitizer uses.
    pub fn sanitize_weights(weights: &mut WeightMap, args: &ModelArgs) -> Result<(), DrafterError> {
        mlxcel_core::mla::decompose_kv_b_proj(
            weights,
            &format!("{BLOCK_PREFIX}.self_attn"),
            kv_b_proj_geometry(args),
            "MTP block",
        )
        .map(|_| ())
        .map_err(|reason| DrafterError::WeightLoad { reason })
    }

    /// Construct from an in-memory weight map in the `split-mtp` layout.
    pub fn from_weights(
        weights: &WeightMap,
        config: Glm4MoeLiteMtpConfig,
    ) -> Result<Self, DrafterError> {
        Self::check_weight_inventory(weights)?;
        let args = config.text_config().clone();
        let (group_size, bits) = (args.group_size(), args.bits());
        let load = |reason: String| DrafterError::WeightLoad { reason };
        let norm_w = |key: &str| -> Result<UniquePtr<MlxArray>, DrafterError> {
            weights
                .get(key)
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| DrafterError::WeightLoad {
                    reason: format!("Weight not found: {key}"),
                })
        };

        let eh_rows = weights
            .get("model.eh_proj.weight")
            .map(|w| mlxcel_core::array_shape(w))
            .and_then(|s| s.first().copied())
            .unwrap_or(0) as usize;
        if eh_rows != args.hidden_size {
            return Err(load(format!(
                "glm4_moe_lite_mtp drafter: model.eh_proj.weight has {eh_rows} output rows but \
                 text_config.hidden_size is {}; the block projects the 2 * hidden fusion back \
                 to hidden",
                args.hidden_size
            )));
        }

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)
                .map_err(load)?;
        let enorm = RMSNorm::new(norm_w("model.enorm.weight")?, args.rms_norm_eps);
        let hnorm = RMSNorm::new(norm_w("model.hnorm.weight")?, args.rms_norm_eps);
        let eh_proj = UnifiedLinear::from_weights(weights, "model.eh_proj", group_size, bits)
            .map_err(load)?;
        let is_moe =
            weights.contains_key(&format!("{BLOCK_PREFIX}.mlp.switch_mlp.gate_proj.weight"));
        let block =
            TransformerBlock::from_weights_with_prefix(weights, &args, BLOCK_PREFIX, is_moe)
                .map_err(load)?;
        let shared_head_norm =
            RMSNorm::new(norm_w("model.shared_head_norm.weight")?, args.rms_norm_eps);
        let lm_head =
            UnifiedLinear::from_weights(weights, "lm_head", group_size, bits).map_err(load)?;

        Ok(Self {
            config,
            embed_tokens,
            enorm,
            hnorm,
            eh_proj,
            block,
            shared_head_norm,
            lm_head,
            cache: KVCache::new(),
            bound: false,
            seed_token: None,
            seed_hidden: None,
            next_position: 0,
            round_appended: 0,
            kv_valid_len: 0,
            position: 0,
            profile: DraftStepProfile {
                synchronized: draft_step_profiling_enabled(),
                ..DraftStepProfile::default()
            },
        })
    }

    /// Fail closed on a directory that is not a standalone MTP block (a full
    /// target checkpoint, or a layout this port does not understand).
    fn check_weight_inventory(weights: &WeightMap) -> Result<(), DrafterError> {
        let mut unknown: Vec<&String> = weights
            .keys()
            .filter(|k| !ROOT_PREFIXES.iter().any(|p| k.starts_with(p)))
            .collect();
        if !unknown.is_empty() {
            unknown.sort();
            let shown: Vec<&str> = unknown.iter().take(5).map(|s| s.as_str()).collect();
            return Err(DrafterError::WeightLoad {
                reason: format!(
                    "glm4_moe_lite_mtp drafter: {} unexpected tensor(s) in checkpoint (first: \
                     {}); this directory does not look like a `mlxcel split-mtp` output",
                    unknown.len(),
                    shown.join(", ")
                ),
            });
        }
        Ok(())
    }

    fn clear_runtime_state(&mut self) {
        self.cache = KVCache::new();
        self.seed_token = None;
        self.seed_hidden = None;
        self.next_position = 0;
        self.round_appended = 0;
    }

    fn require_bound(&self) -> Result<(), DrafterError> {
        if self.bound {
            Ok(())
        } else {
            Err(DrafterError::BindNotCalled)
        }
    }

    /// One block forward over `token_ids` (length `S`) paired with the
    /// matching target hidden `[1, S, H]`. Appends `S` cache entries and
    /// advances `next_position` by `S`. Returns the block output `[1, S, H]`
    /// (pre `shared_head_norm`), which is also the hidden fed back on the
    /// next draft step.
    fn forward_hidden_stack(
        &mut self,
        token_ids: &[i32],
        hidden: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let s = token_ids.len() as i32;
        let hshape = mlxcel_core::array_shape(hidden);
        if s < 1 || hshape.len() != 3 || hshape[1] != s {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "glm4_moe_lite_mtp drafter: hidden shape {hshape:?} does not pair with \
                     {s} token(s)"
                ),
            });
        }
        let ids = mlxcel_core::from_slice_i32(token_ids, &[1, s]);
        let e = self.embed_tokens.forward(&ids);
        let a = self.enorm.forward(&e);
        let b = self.hnorm.forward(hidden);
        let fused = mlxcel_core::concatenate(&a, &b, -1);
        let x = self.eh_proj.forward(&fused);
        // Multi-token forwards attend causally over the cache; a single draft
        // step needs no mask (the block builds none for `l == 1`).
        let mask = (s > 1).then(|| create_causal_mask(s, self.cache.seq_len()));
        let x = self.block.forward(&x, &mut self.cache, mask.as_deref());
        self.next_position += s;
        Ok(x)
    }

    fn project_logits(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.shared_head_norm.forward(x);
        self.lm_head.forward(&normed)
    }

    /// Single-token sample from the last position of `logits`. Temperature 0
    /// is argmax; the target verifies every proposal, so a non-greedy sampler
    /// only changes acceptance, never output.
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

    /// Compute and stash the next-round seed from the last position of a
    /// block output.
    fn set_seed_from_hidden(&mut self, x: &MlxArray, sampler: &SamplingConfig) {
        let shape = mlxcel_core::array_shape(x);
        let last = shape[1] - 1;
        let x_last = mlxcel_core::slice(x, &[0, last, 0], &[shape[0], last + 1, shape[2]]);
        let logits = self.project_logits(&x_last);
        self.seed_token = Some(Self::sample_one(&logits, sampler));
        self.seed_hidden = Some(x_last);
    }

    /// Test/diagnostic accessor: `(cache_offset, next_position,
    /// round_appended, has_seed)`.
    pub fn state_probe(&self) -> (i32, i32, i32, bool) {
        (
            self.cache.offset,
            self.next_position,
            self.round_appended,
            self.seed_token.is_some(),
        )
    }

    /// Hidden width the block consumes.
    pub fn hidden_size(&self) -> usize {
        self.config.text_config().hidden_size
    }
}

impl Drafter for Glm4MoeLiteMtpDraftModel {
    /// Nothing is borrowed from the target; `bind` re-runs the geometry
    /// check so a caller that skipped `validate_target_compat` still cannot
    /// pair the drafter with a foreign target.
    fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.validate_target_compat(target)?;
        self.bound = true;
        Ok(())
    }

    /// Reject a target whose hidden width or vocabulary does not match the
    /// block: the block consumes the target's hidden at `hidden_size` width
    /// and proposes ids the target verifies against its own vocabulary.
    fn validate_target_compat(&self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        let expected_hidden = self.config.text_config().hidden_size as i32;
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
                    "{GLM4_MOE_LITE_MTP_MODEL_TYPE} drafter is incompatible with this target: \
                     drafter text_config.hidden_size = {expected_hidden} but the target's hidden \
                     size = {target_hidden}. The MTP block consumes the target's hidden state, so \
                     these must be equal; pair the drafter split from the same GLM-4.7-Flash \
                     checkpoint."
                ),
            });
        }
        let drafter_vocab = self.config.text_config().vocab_size as i32;
        let zero_hidden = mlxcel_core::zeros(&[1, 1, target_hidden], mlxcel_core::dtype::FLOAT32);
        let logits = match target.lm_head_module() {
            Some(lm) => lm.forward(&zero_hidden),
            None => match target.embed_tokens_module() {
                Some(embed) => embed.as_linear(&zero_hidden),
                // Mock targets that only expose `embed_tokens` pass on the
                // hidden-size gate alone, as with the other MTP drafters.
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
                    "{GLM4_MOE_LITE_MTP_MODEL_TYPE} drafter vocabulary is incompatible with this \
                     target: drafter text_config.vocab_size = {drafter_vocab} but the target \
                     vocabulary = {target_vocab}."
                ),
            });
        }
        Ok(())
    }

    /// No shared-K/V concept: only the position metadata is consumed. An
    /// empty cache re-anchors to the target's offset; a non-empty cache that
    /// disagrees with it is stale and is cleared.
    fn set_shared_kv(
        &mut self,
        _shared_kv: SharedKv<'_>,
        kv_offset: usize,
        position: usize,
        _left_padding: usize,
    ) -> Result<(), DrafterError> {
        self.kv_valid_len = kv_offset as i32;
        self.position = position as i32;
        if self.cache.offset == 0 {
            self.next_position = kv_offset as i32;
        } else if self.next_position != kv_offset as i32 {
            tracing::debug!(
                next_position = self.next_position,
                kv_offset,
                "glm4_moe_lite_mtp drafter position drifted from target cache; clearing \
                 drafter state and re-anchoring (draft context lost, correctness unaffected)"
            );
            self.clear_runtime_state();
            self.next_position = kv_offset as i32;
        }
        Ok(())
    }

    fn make_cache(&self) -> Vec<KVCache> {
        Vec::new()
    }

    /// Clears the accumulated history (the slice-grant rotation resets a
    /// parked drafter; it resumes in the empty-cache mode with reduced draft
    /// context and identical output).
    fn reset(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.clear_runtime_state();
        self.bind(target)
    }

    fn configured_block_size(&self) -> Option<usize> {
        Some(self.config.runtime_block_size())
    }

    fn prefer_requested_block_size(&self) -> bool {
        true
    }

    fn prefill_from_target_hidden(
        &mut self,
        prompt_tokens: &[i32],
        hidden: &MlxArray,
        first_bonus: i32,
        sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        self.require_bound()?;
        // Before the early return, not after: a session that reached here
        // with an empty prompt would otherwise keep the previous session's
        // seed token and hidden. The server rejects empty prompts twice
        // upstream, so this only removes the dependence on those guards.
        self.clear_runtime_state();
        let p = prompt_tokens.len();
        if p == 0 {
            return Ok(());
        }
        let hshape = mlxcel_core::array_shape(hidden);
        if hshape.len() != 3 || hshape[1] != p as i32 {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "glm4_moe_lite_mtp drafter prefill: hidden shape {hshape:?} does not cover \
                     the {p}-token prompt"
                ),
            });
        }
        // Position i pairs token_{i+1} with hidden_i: shift the prompt left
        // by one and append the just-sampled first bonus.
        let mut shifted: Vec<i32> = prompt_tokens[1..].to_vec();
        shifted.push(first_bonus);
        let x = self.forward_hidden_stack(&shifted, hidden)?;
        self.set_seed_from_hidden(&x, sampler);
        Ok(())
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
        let hshape = mlxcel_core::array_shape(verify_hidden);
        let block = draft_tokens.len() + 1;
        if hshape.len() != 3 || (hshape[1] as usize) < block || accepted > draft_tokens.len() {
            // Validate before mutating; poisoned state is cleared so the next
            // `set_shared_kv` re-anchors cleanly.
            self.clear_runtime_state();
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "glm4_moe_lite_mtp drafter accept: verify_hidden shape {hshape:?} does not \
                     cover block={block} / accepted={accepted}"
                ),
            });
        }

        // Trim the in-round entries beyond the accepted prefix.
        let keep = accepted.min(self.round_appended.max(0) as usize);
        let trim = self.round_appended - keep as i32;
        if trim > 0 {
            self.cache.trim(trim);
            self.next_position -= trim;
        }

        // Extend with the accepted tokens not yet in the cache, paired with
        // the target's true verify hidden, plus the new bonus.
        let mut tokens: Vec<i32> = draft_tokens[keep..accepted].to_vec();
        if let Some(&last) = new_tokens.last() {
            tokens.push(last);
        }
        if !tokens.is_empty() {
            let start = keep as i32;
            let end = start + tokens.len() as i32;
            let hiddens =
                mlxcel_core::slice(verify_hidden, &[0, start, 0], &[hshape[0], end, hshape[2]]);
            let x = self.forward_hidden_stack(&tokens, &hiddens)?;
            self.set_seed_from_hidden(&x, sampler);
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

        // Seed fast path: the prefill / accept hook already computed this
        // round's first proposal from TARGET hidden and appended its entry.
        let (mut tok, mut h_prev): (i32, UniquePtr<MlxArray>) =
            match (self.seed_token.take(), self.seed_hidden.take()) {
                (Some(seed_tok), Some(seed_hidden)) => {
                    tokens.push(seed_tok);
                    (seed_tok, seed_hidden)
                }
                _ => {
                    let hidden = hidden.ok_or(DrafterError::DraftBlockMissingHidden)?;
                    (last_bonus, mlxcel_core::copy(hidden))
                }
            };

        while tokens.len() < block_size - 1 {
            let step_started = Instant::now();
            let x = self.forward_hidden_stack(&[tok], &h_prev)?;
            self.round_appended += 1;
            let logits = self.project_logits(&x);
            tok = Self::sample_one(&logits, sampler);
            self.profile.total_ms += step_started.elapsed().as_secs_f64() * 1e3;
            self.profile.steps += 1;
            tokens.push(tok);
            h_prev = x;
        }
        tokens.truncate(block_size - 1);
        Ok(tokens)
    }

    fn draft_profile(&self) -> Option<DraftStepProfile> {
        Some(self.profile)
    }

    fn sanitize(&mut self, weights: &mut WeightMap) -> Result<(), DrafterError> {
        let args = self.config.text_config().clone();
        Self::sanitize_weights(weights, &args)
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Mtp
    }
}

#[cfg(test)]
#[path = "glm4_moe_lite_mtp_drafter_tests.rs"]
mod tests;
