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

//! `Qwen35MtpDraftModel` — Rust port of the Qwen 3.5 / 3.6 / 3.8 MTP drafter.
//!
//! Mirrors
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py
//! (B = 1 surface; the batched drafter path is out of scope and declines via
//! the trait's erroring `draft_block_batched` default).
//!
//! ## Lifecycle
//!
//! Unlike the Gemma 4 assistant (stateless per round, cross-attends into the
//! target's shared K/V), the Qwen MTP drafter is **stateful**: it owns one KV
//! cache per MTP layer and accumulates one entry per emitted target position,
//! exactly like the trained MTP head does. Position `i` of the drafter's
//! sequence is the pair `(token_{i+1} embedding, target hidden at i)` fused
//! through `fc`.
//!
//! 1. **Load.** [`Qwen35MtpDraftModel::from_path`] parses `config.json`,
//!    loads and sanitizes the 15-tensor head, converts bf16 → f16 (Apple
//!    Silicon precision rules), and constructs the model.
//! 2. **Bind.** [`Drafter::bind`] borrows the target's `embed_tokens` and
//!    `lm_head` (falling back to the target embedding's `as_linear` for tied
//!    checkpoints). The drafter owns neither.
//! 3. **Prompt prefill.** [`Drafter::prefill_from_target_hidden`] runs the
//!    shifted prompt (`prompt[1..] ++ bonus`) paired with the target's
//!    post-final-norm prompt hidden through the MTP stack, populating the
//!    drafter cache and computing the first seed token.
//! 4. **Rounds.** [`Drafter::set_shared_kv`] discards the tensors (Qwen has
//!    no shared-K/V concept) and only aligns positions;
//!    [`Drafter::draft_block`] consumes the seed and runs small
//!    autoregressive forwards; after the target verify,
//!    [`Drafter::accept_verified_tokens`] trims the rejected in-round cache
//!    tail and extends the cache with the accepted tokens paired with the
//!    target's verify hidden, computing the next seed.
//!
//! A drafter whose cache was cleared mid-session (slice-grant rotation resets
//! the shared worker handle, or a hook failure poisoned the state) degrades
//! gracefully: `set_shared_kv` re-anchors `next_position` to the target's
//! cache offset and drafts continue with an empty history. That costs
//! acceptance, never correctness — greedy parity is enforced by the target's
//! verify pass alone.

use crate::cache::KVCache;
use crate::drafter::{
    DraftForwardCost, DraftStepProfile, Drafter, DrafterError, DrafterKind, SharedKv,
};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};
use crate::weights::WeightMap;
use cxx::UniquePtr;
use std::path::Path;
use std::time::Instant;

use super::config::Qwen35MtpConfig;
use super::layer::Qwen35MtpDecoderLayer;

/// LM head borrowed from the target at `bind()` time.
enum MtpLmHead {
    /// Target has an untied `lm_head` (Qwen3.8-27B).
    Linear(UnifiedLinear),
    /// Tied checkpoints project through the target embedding table.
    TiedEmbed(UnifiedEmbedding),
}

/// Force `array` to evaluate when `MLXCEL_MTP_DRAFT_PROFILE` is on, so the
/// next timer measures its own GPU work rather than the previous stage's
/// deferred graph.
///
/// A no-op by default. The syncs are what make the component split real
/// and also what make the profiled total run above the honest step cost,
/// which is why the mode is opt-in and the log line reports which one
/// produced the numbers (issue #1185, Phase 0).
#[inline]
fn maybe_sync(array: &MlxArray) {
    if crate::drafter::draft_step_profiling_enabled() {
        ffi::eval(array);
    }
}

#[inline]
fn ms_since(start: Instant, end: Instant) -> f64 {
    end.saturating_duration_since(start).as_secs_f64() * 1e3
}

/// Qwen 3.5 MTP drafter — `fc`-fused single-decoder-layer head with its own
/// per-layer KV cache, borrowing embeddings and the LM head from the target.
///
/// Implements [`Drafter`] and is wired into the `Mtp` arm of
/// [`crate::drafter::load_drafter`] for `model_type == "qwen3_5_mtp"`.
pub struct Qwen35MtpDraftModel {
    config: Qwen35MtpConfig,
    fc: UnifiedLinear,
    pre_fc_norm_embedding: RMSNorm,
    pre_fc_norm_hidden: RMSNorm,
    norm: RMSNorm,
    layers: Vec<Qwen35MtpDecoderLayer>,

    /// Per-layer drafter-owned KV cache. One entry per MTP layer.
    cache: Vec<KVCache>,

    /// Target's embedding table, captured by `bind()`.
    target_embed: Option<UnifiedEmbedding>,
    /// Upstream reads `inner.embed_scale` defaulting to 1.0; Qwen 3.5 has no
    /// embedding scale, so this stays 1.0 and the multiply is skipped.
    target_embed_scale: f32,
    /// Target's LM head, captured by `bind()`.
    lm_head: Option<MtpLmHead>,

    /// Precomputed seed for the next `draft_block`: the MTP head's prediction
    /// one past the last accepted token, computed by the prompt prefill /
    /// accept hooks from TARGET hidden (so its cache entry uses the true
    /// pair, unlike in-round entries that substitute the drafter's own
    /// hidden).
    seed_token: Option<i32>,
    seed_hidden: Option<UniquePtr<MlxArray>>,

    /// Absolute target-sequence position of the next drafter cache append.
    /// Tracks the target cache offset exactly while state is intact; also the
    /// RoPE offset for the next forward.
    next_position: i32,
    /// Number of in-round cache appends since the last accept, i.e. how many
    /// entries the current draft block added. The accept hook keeps
    /// `min(accepted, round_appended)` of them and trims the rest.
    round_appended: i32,
    /// Diagnostics only — the trait's `kv_offset` / `position` arguments.
    kv_valid_len: i32,
    position: i32,
    /// Per-component attribution for this drafter's steps (issue #1185,
    /// Phase 0). Accumulates across the session; the round loop reads it
    /// through [`Drafter::draft_profile`] at the end of a run.
    profile: DraftStepProfile,
    /// Whether the forward currently running belongs to a `draft_block`
    /// step. `forward_hidden_stack` is also reached from the accept hook
    /// and the prefill seed, and charging those to the step bucket makes
    /// the step components sum above the step total.
    in_draft_step: bool,
}

impl std::fmt::Debug for Qwen35MtpDraftModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35MtpDraftModel")
            .field("model_type", &self.config.model_type)
            .field("block_size", &self.config.block_size())
            .field("num_layers", &self.layers.len())
            .field("bound", &self.lm_head.is_some())
            .field("cache_offset", &self.cache.first().map(|c| c.offset))
            .field("next_position", &self.next_position)
            .field("round_appended", &self.round_appended)
            .field("has_seed", &self.seed_token.is_some())
            .finish()
    }
}

/// `MLXCEL_MTP_QUANTIZE_DRAFTER=0` keeps a dense drafter dense.
///
/// On by default because the measurement says the cost is throughput-only:
/// the target verifies every proposal, so the switch trades acceptance, not
/// correctness, and acceptance did not move measurably (#1185 Phase 3). Read
/// once per process, matching the other `MLXCEL_*` switches.
fn quantize_drafter_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("MLXCEL_MTP_QUANTIZE_DRAFTER")
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !matches!(v.as_str(), "0" | "false" | "no" | "off")
            })
            .unwrap_or(true)
    })
}

impl Qwen35MtpDraftModel {
    /// Construct from a checkpoint directory containing `config.json` and
    /// safetensors shards. Used by [`crate::drafter::load_drafter`]'s `Mtp`
    /// arm when the drafter's `model_type` is `qwen3_5_mtp`.
    pub fn from_path(path: &Path) -> Result<Self, DrafterError> {
        let cfg_path = path.join("config.json");
        let bytes = std::fs::read(&cfg_path).map_err(|e| DrafterError::ConfigIo {
            path: cfg_path.display().to_string(),
            source: e,
        })?;
        let config: Qwen35MtpConfig =
            serde_json::from_slice(&bytes).map_err(|e| DrafterError::ConfigParse {
                path: cfg_path.display().to_string(),
                source: e,
            })?;
        let config = config.normalize().map_err(DrafterError::Config)?;

        let mut weights = crate::weights::load_weights_from_dir(path)
            .map_err(|reason| DrafterError::WeightLoad { reason })?;
        Self::sanitize_weights(&mut weights);
        // Quantize the drafter's own projections before the dtype pass, so a
        // dense checkpoint costs what a 4-bit one costs (issue #1185 Phase 3).
        // Must run first: the bf16 -> f16 conversion below deliberately skips
        // quantization auxiliaries, and the packed payload is uint32 either
        // way, so ordering it after would only leave scales as f16 where every
        // shipped checkpoint keeps them bf16.
        Self::quantize_dense_projections(&mut weights, &config);
        // Apple Silicon precision: bf16 → f16 on non-quantized tensors at
        // the weight-loading boundary, matching the target model loaders and
        // the DFlash drafter loader. The published drafter is bf16; the
        // paired 4-bit target's activations are f16, and a dtype-mixed
        // concat/matmul would silently promote to f32.
        crate::drafter::dflash::drafter::convert_bf16_to_f16_non_quantized(&mut weights);
        Self::from_weights(&weights, config)
    }

    /// Quantize the drafter's dense 2-D projections in place, to the scheme
    /// its own config declares (group 64, 4-bit affine by default).
    ///
    /// The drafter is memory-bound and ships bf16: 810 MiB of weights read
    /// once per drafted token. Quantizing to 4-bit takes that to 228 MiB, and
    /// measured on an M5 Max the drafter step went from 10.5 to 2.7 ms per
    /// round while the target verify forward was unchanged (#1185).
    ///
    /// Safe by construction rather than by tolerance: the drafter only
    /// proposes and the target verifies every proposal, so drafter numerics
    /// cannot reach the output. The exposure is acceptance rate, and that was
    /// measured across two generation lengths: 0.683 to 0.660 at 120 tokens
    /// and 0.650 to 0.659 at 300, which is noise in both directions. Output
    /// stayed byte-identical to classic decode.
    ///
    /// Skips a tensor whose `.scales` sibling already exists (a pre-converted
    /// checkpoint) and one whose contraction axis is not a multiple of the
    /// group size, leaving those dense rather than failing the load.
    ///
    /// `MLXCEL_MTP_QUANTIZE_DRAFTER=0` keeps the checkpoint's own precision,
    /// for an acceptance A/B on a pairing this has not been measured on.
    fn quantize_dense_projections(weights: &mut WeightMap, config: &Qwen35MtpConfig) {
        if !quantize_drafter_enabled() {
            return;
        }
        let text_cfg = config.text_config();
        let (group_size, bits) = (text_cfg.group_size(), text_cfg.bits());

        let candidates: Vec<String> = weights
            .iter()
            .filter(|(key, value)| {
                key.ends_with(".weight")
                    && ffi::array_shape(value).len() == 2
                    && !weights.contains_key(&format!("{}.scales", key.trim_end_matches(".weight")))
            })
            .map(|(key, _)| key.clone())
            .collect();

        let mut converted = 0usize;
        for key in candidates {
            let prefix = key.trim_end_matches(".weight").to_string();
            let shape = weights.get(&key).map(|w| ffi::array_shape(w));
            let Some(shape) = shape else { continue };
            if shape[shape.len() - 1] % group_size != 0 {
                continue;
            }
            let quantized = {
                let Some(w) = weights.get(&key) else { continue };
                ffi::quantize_weights_with_mode(w, group_size, bits, "affine")
            };
            let packed = ffi::quantized_weights_w(&quantized);
            let scales = ffi::quantized_weights_scales(&quantized);
            let has_biases = ffi::quantized_weights_has_biases(&quantized);
            weights.insert(key.clone(), packed);
            weights.insert(format!("{prefix}.scales"), scales);
            if has_biases {
                weights.insert(
                    format!("{prefix}.biases"),
                    ffi::quantized_weights_biases(&quantized),
                );
            }
            converted += 1;
        }

        if converted > 0 {
            tracing::debug!(
                converted,
                group_size,
                bits,
                "quantized the MTP drafter's dense projections at load"
            );
        }
    }

    /// Construct from an in-memory weight map (already sanitized). Used by
    /// `from_path` and unit tests with synthetic fixtures.
    pub fn from_weights(
        weights: &WeightMap,
        config: Qwen35MtpConfig,
    ) -> Result<Self, DrafterError> {
        let text_cfg = config.text_config().clone();

        Self::check_weight_inventory(weights, text_cfg.mtp_num_hidden_layers)?;

        let fc = UnifiedLinear::from_weights(weights, "fc", text_cfg.group_size(), text_cfg.bits())
            .map_err(|reason| DrafterError::WeightLoad { reason })?;
        let norm_w = |key: &str| -> Result<UniquePtr<MlxArray>, DrafterError> {
            weights
                .get(key)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| DrafterError::WeightLoad {
                    reason: format!("Weight not found: {key}"),
                })
        };
        let pre_fc_norm_embedding = RMSNorm::new(
            norm_w("pre_fc_norm_embedding.weight")?,
            text_cfg.rms_norm_eps,
        );
        let pre_fc_norm_hidden =
            RMSNorm::new(norm_w("pre_fc_norm_hidden.weight")?, text_cfg.rms_norm_eps);
        let norm = RMSNorm::new(norm_w("norm.weight")?, text_cfg.rms_norm_eps);

        let mut layers = Vec::with_capacity(text_cfg.mtp_num_hidden_layers);
        for i in 0..text_cfg.mtp_num_hidden_layers {
            layers.push(
                Qwen35MtpDecoderLayer::from_weights(weights, &format!("layers.{i}"), &text_cfg)
                    .map_err(|reason| DrafterError::WeightLoad { reason })?,
            );
        }

        let cache = (0..layers.len()).map(|_| KVCache::new()).collect();

        Ok(Self {
            config,
            fc,
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            norm,
            layers,
            cache,
            target_embed: None,
            target_embed_scale: 1.0,
            lm_head: None,
            seed_token: None,
            seed_hidden: None,
            next_position: 0,
            round_appended: 0,
            kv_valid_len: 0,
            position: 0,
            profile: DraftStepProfile {
                synchronized: crate::drafter::draft_step_profiling_enabled(),
                ..DraftStepProfile::default()
            },
            in_draft_step: false,
        })
    }

    /// Apply the upstream `sanitize` rules to a freshly loaded weight map:
    ///
    /// - Strip the `mtp.` prefix from raw-HF-layout keys.
    /// - Apply the `+1.0` RMSNorm offset ONLY to `mtp.`-prefixed norm keys
    ///   (the seven suffixes upstream lists). The published
    ///   `mlx-community/Qwen3.8-27B-MTP-bf16` layout is already stripped —
    ///   i.e. the conversion already applied the shift — so already-stripped
    ///   keys MUST NOT be shifted again; a double shift corrupts every norm.
    ///
    /// The MoE `experts.gate_up_proj` split from upstream `sanitize` is
    /// intentionally not ported: MoE text configs are rejected at
    /// [`Qwen35MtpConfig::normalize`].
    pub fn sanitize_weights(weights: &mut WeightMap) {
        const NORM_SUFFIXES: [&str; 7] = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            "norm.weight",
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
        ];
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in keys {
            let Some(stripped) = key.strip_prefix("mtp.") else {
                continue;
            };
            let stripped = stripped.to_string();
            let mut value = weights.remove(&key).expect("key just enumerated");
            if NORM_SUFFIXES.iter().any(|s| stripped.ends_with(s)) {
                let shape = ffi::array_shape(&value);
                if shape.len() == 1 {
                    let one = ffi::ones(&shape, ffi::array_dtype(&value));
                    value = ffi::add(&value, &one);
                }
            }
            weights.insert(stripped, value);
        }
    }

    /// Strict inventory gate: every key in the (sanitized) weight map must be
    /// one this drafter consumes. A leftover unknown key means the directory
    /// is not a standalone `qwen3_5_mtp` head (e.g. a full target checkpoint,
    /// or a layout this port does not understand) — fail closed with the
    /// offending names instead of silently ignoring tensors.
    fn check_weight_inventory(weights: &WeightMap, num_layers: usize) -> Result<(), DrafterError> {
        let mut expected: Vec<String> = vec![
            "fc.weight".into(),
            "pre_fc_norm_embedding.weight".into(),
            "pre_fc_norm_hidden.weight".into(),
            "norm.weight".into(),
        ];
        for i in 0..num_layers {
            for name in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                expected.push(format!("layers.{i}.{name}"));
            }
        }
        let is_expected = |key: &str| -> bool {
            expected.iter().any(|e| {
                key == e
                    || key.strip_suffix(".scales") == Some(strip_weight(e))
                    || key.strip_suffix(".biases") == Some(strip_weight(e))
            })
        };
        let mut unknown: Vec<&String> = weights.keys().filter(|k| !is_expected(k)).collect();
        if !unknown.is_empty() {
            unknown.sort();
            let shown: Vec<&str> = unknown.iter().take(5).map(|s| s.as_str()).collect();
            return Err(DrafterError::WeightLoad {
                reason: format!(
                    "qwen3_5_mtp drafter: {} unexpected tensor(s) in checkpoint (first: {}); \
                     this directory does not look like a standalone qwen3_5_mtp head",
                    unknown.len(),
                    shown.join(", ")
                ),
            });
        }
        Ok(())
    }

    /// Clear all per-run state (cache, seed, counters) while keeping the
    /// bind. Leaves the drafter in the "empty cache" mode `set_shared_kv`
    /// re-anchors from.
    fn clear_runtime_state(&mut self) {
        self.cache = (0..self.layers.len()).map(|_| KVCache::new()).collect();
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

    /// One MTP-stack forward over `token_ids` (length `S`) paired with the
    /// matching target hidden `[1, S, H]`:
    /// `norm(layers(fc(concat(pre_fc_norm_embedding(embed(tokens)),
    /// pre_fc_norm_hidden(hidden)))))`. Appends `S` entries to every layer
    /// cache and advances `next_position` by `S`. Returns the post-`norm`
    /// hidden `[1, S, H]`.
    /// The component bucket the current forward belongs to.
    ///
    /// `forward_hidden_stack` serves three callers with different meanings
    /// to the round loop: a `draft_block` step, the accept hook's append
    /// forward, and the prefill seed. Only the first is what
    /// `per_step_ms` describes.
    fn active_cost(&mut self) -> &mut DraftForwardCost {
        if self.in_draft_step {
            &mut self.profile.step
        } else {
            self.profile.other_forwards += 1;
            &mut self.profile.other
        }
    }

    fn forward_hidden_stack(
        &mut self,
        token_ids: &[i32],
        hidden: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let target_embed = self
            .target_embed
            .as_ref()
            .ok_or(DrafterError::BindNotCalled)?;
        let s = token_ids.len() as i32;
        debug_assert!(s >= 1, "forward_hidden_stack requires at least one token");
        debug_assert_eq!(
            ffi::array_shape(hidden)[1],
            s,
            "hidden seq len must match token count"
        );

        let ids_started = Instant::now();
        let ids = ffi::from_slice_i32(token_ids, &[1, s]);
        let mut tok_embed = target_embed.forward(&ids);
        if self.target_embed_scale != 1.0 {
            tok_embed = crate::ops::multiply_scalar(&tok_embed, self.target_embed_scale);
        }
        maybe_sync(&tok_embed);
        let layers_started = Instant::now();
        self.active_cost().ids_ms += ms_since(ids_started, layers_started);

        let a = self.pre_fc_norm_embedding.forward(&tok_embed);
        let b = self.pre_fc_norm_hidden.forward(hidden);
        let fused = crate::ops::concatenate(&a, &b, -1);
        let mut h = self.fc.forward(&fused);

        // Multi-token forwards use a causal mask offset by the drafter
        // cache's key length (mirrors upstream
        // `create_attention_mask(h, layer_cache)`); single-token draft steps
        // need none. The RoPE offset is the drafter's logical position, which
        // equals the cache offset while state is intact and runs ahead of it
        // in the empty-cache degraded mode.
        let mask = if s > 1 {
            let cache_offset = self.cache.first().map(|c| c.offset).unwrap_or(0);
            Some(crate::utils::create_causal_mask(s, cache_offset))
        } else {
            None
        };
        let rope_offset = self.next_position;
        for (layer, cache) in self.layers.iter().zip(self.cache.iter_mut()) {
            h = layer.forward(&h, mask.as_deref(), cache, rope_offset);
        }
        let h = self.norm.forward(&h);
        maybe_sync(&h);
        self.active_cost().layers_ms += layers_started.elapsed().as_secs_f64() * 1e3;
        self.next_position += s;
        Ok(h)
    }

    /// Project drafter hidden through the borrowed target LM head.
    fn project_logits(&self, h: &MlxArray) -> Result<UniquePtr<MlxArray>, DrafterError> {
        match self.lm_head.as_ref().ok_or(DrafterError::BindNotCalled)? {
            MtpLmHead::Linear(lm) => Ok(lm.forward(h)),
            MtpLmHead::TiedEmbed(embed) => Ok(embed.as_linear(h)),
        }
    }

    /// Single-token sample from the last position of `logits`. Temperature 0
    /// is argmax; non-greedy configs sample through the fused kernel (the
    /// round loop owns the quality consequences, exactly as with the Gemma 4
    /// assistant drafter).
    fn sample_one(logits: &MlxArray, sampler: &SamplingConfig) -> i32 {
        let last = ffi::slice_last_logits(logits);
        let tok = ffi::fused_sample(
            &last,
            sampler.temperature,
            sampler.top_k,
            sampler.top_p,
            sampler.min_p,
        );
        ffi::eval(&tok);
        ffi::item_i32(&tok)
    }

    /// [`Self::sample_one`] with the fused-sample build and the readback
    /// charged to separate buckets.
    ///
    /// The `eval` here is unconditional and is not a profiling artifact:
    /// the drafter must have the sampled id on the host to feed the next
    /// step. It is the one place a drafter step synchronizes by
    /// construction, which is why an unprofiled run attributes all GPU
    /// work to `readback_ms`.
    fn sample_one_profiled(
        &mut self,
        logits: &MlxArray,
        sampler: &SamplingConfig,
        started: Instant,
    ) -> i32 {
        let last = ffi::slice_last_logits(logits);
        let tok = ffi::fused_sample(
            &last,
            sampler.temperature,
            sampler.top_k,
            sampler.top_p,
            sampler.min_p,
        );
        let readback_started = Instant::now();
        self.active_cost().sample_ms += ms_since(started, readback_started);
        ffi::eval(&tok);
        let id = ffi::item_i32(&tok);
        self.active_cost().readback_ms += readback_started.elapsed().as_secs_f64() * 1e3;
        id
    }

    /// Compute and stash the next-round seed from the last position of a
    /// post-`norm` drafter hidden block.
    fn set_seed_from_hidden(
        &mut self,
        h: &MlxArray,
        sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        let shape = ffi::array_shape(h);
        let last = shape[1] - 1;
        let h_last = ffi::slice(h, &[0, last, 0], &[shape[0], last + 1, shape[2]]);
        let logits = self.project_logits(&h_last)?;
        let tok = Self::sample_one(&logits, sampler);
        self.seed_token = Some(tok);
        self.seed_hidden = Some(h_last);
        Ok(())
    }

    /// Test/diagnostic accessor: `(cache_offset, next_position,
    /// round_appended, has_seed)`.
    pub fn state_probe(&self) -> (i32, i32, i32, bool) {
        (
            self.cache.first().map(|c| c.offset).unwrap_or(0),
            self.next_position,
            self.round_appended,
            self.seed_token.is_some(),
        )
    }
}

fn strip_weight(key: &str) -> &str {
    key.strip_suffix(".weight").unwrap_or(key)
}

impl Drafter for Qwen35MtpDraftModel {
    fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        let embed = target
            .embed_tokens_module()
            .ok_or(DrafterError::TargetMissingFeature {
                feature: "embed_tokens_module",
            })?;
        // Upstream: `lm_head = target.lm_head or embed_tokens.as_linear`.
        self.lm_head = Some(match target.lm_head_module() {
            Some(lm) => MtpLmHead::Linear(lm),
            None => MtpLmHead::TiedEmbed(embed.clone_shared()),
        });
        // Upstream reads `inner.embed_scale` with a 1.0 default; the Qwen 3.5
        // family carries no embedding scale.
        self.target_embed_scale = 1.0;
        self.target_embed = Some(embed);
        Ok(())
    }

    /// Reject a target whose text hidden size or vocabulary does not match
    /// this MTP head. The head consumes the target's post-final-norm hidden
    /// at `hidden_size` width and emits through the target's own LM head, so
    /// both must line up.
    fn validate_target_compat(&self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        let expected_hidden = self.config.text_config().hidden_size as i32;
        let sentinel = ffi::from_slice_i32(&[0_i32], &[1, 1]);
        let embedded =
            target
                .embed_tokens(&sentinel)
                .ok_or(DrafterError::TargetMissingFeature {
                    feature: "embed_tokens",
                })?;
        let target_hidden = ffi::array_shape(&embedded).last().copied().unwrap_or(0);
        if target_hidden != expected_hidden {
            return Err(DrafterError::BindFailed {
                reason: format!(
                    "qwen3_5_mtp drafter is incompatible with this target: drafter \
                     text_config.hidden_size = {expected_hidden} but the target's text hidden \
                     size = {target_hidden}. The MTP head consumes the target's last hidden \
                     state, so these must be equal; pair the drafter split from the same \
                     checkpoint family (e.g. mlx-community/Qwen3.8-27B-MTP-bf16 with a \
                     Qwen3.8-27B target)."
                ),
            });
        }

        // Vocabulary probe via the head the drafter would actually borrow.
        let drafter_vocab = self.config.text_config().vocab_size as i32;
        let zero_hidden = ffi::zeros(&[1, 1, target_hidden], crate::dtype::FLOAT32);
        let logits = match target.lm_head_module() {
            Some(lm) => lm.forward(&zero_hidden),
            None => match target.embed_tokens_module() {
                Some(embed) => embed.as_linear(&zero_hidden),
                // Mock targets that only expose `embed_tokens` pass on the
                // hidden-size gate alone, mirroring the Gemma 4 adapter.
                None => return Ok(()),
            },
        };
        let target_vocab = ffi::array_shape(&logits).last().copied().unwrap_or(0);
        if target_vocab != drafter_vocab {
            return Err(DrafterError::BindFailed {
                reason: format!(
                    "qwen3_5_mtp drafter vocabulary is incompatible with this target: drafter \
                     text_config.vocab_size = {drafter_vocab} but the target vocabulary = \
                     {target_vocab}. Draft token ids must index the same vocabulary the \
                     target verifies against."
                ),
            });
        }
        Ok(())
    }

    /// Qwen MTP has no shared-K/V concept: the tensors are ignored whatever
    /// their count (the target adapter passes an empty slice) and only the
    /// position metadata is consumed, mirroring upstream `set_shared_kv`'s
    /// `del shared_kv_states`.
    ///
    /// With an empty cache the drafter re-anchors `next_position` to the
    /// target's post-rollback cache offset; with an intact cache the two are
    /// already equal by construction. A non-empty cache whose position
    /// disagrees is stale (a hook failure or an out-of-band reset) and is
    /// cleared so the round continues in the empty-cache degraded mode
    /// instead of drafting from corrupt state.
    fn set_shared_kv(
        &mut self,
        _shared_kv: SharedKv<'_>,
        kv_offset: usize,
        position: usize,
        _left_padding: usize,
    ) -> Result<(), DrafterError> {
        self.kv_valid_len = kv_offset as i32;
        self.position = position as i32;
        let cache_empty = self.cache.iter().all(|c| c.offset == 0);
        if cache_empty {
            self.next_position = kv_offset as i32;
        } else if self.next_position != kv_offset as i32 {
            tracing::debug!(
                next_position = self.next_position,
                kv_offset,
                "qwen3_5_mtp drafter position drifted from target cache; clearing drafter \
                 state and re-anchoring (draft context lost, correctness unaffected)"
            );
            self.clear_runtime_state();
            self.next_position = kv_offset as i32;
        }
        Ok(())
    }

    /// The drafter owns its cache internally (`self.cache`), so the trait's
    /// external-cache factory stays empty, like the Gemma 4 assistant.
    fn make_cache(&self) -> Vec<KVCache> {
        Vec::new()
    }

    /// Clear all per-run state and re-bind against `target`.
    ///
    /// NOTE for the tick-slice grant rotation: unlike the Gemma 4 assistant
    /// (whose reset is the trait default no-op), this reset DESTROYS the
    /// drafter's accumulated history. A parked slice session whose drafter
    /// was reset resumes correctly — `set_shared_kv` re-anchors into the
    /// empty-cache mode — but with reduced draft context until the history
    /// rebuilds from accepted tokens.
    fn reset(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.clear_runtime_state();
        self.bind(target)
    }

    fn configured_block_size(&self) -> Option<usize> {
        Some(self.config.block_size())
    }

    /// Upstream sets `prefer_requested_block_size = True`: the user-requested
    /// block size is honored directly instead of the Gemma-style adaptive
    /// configured-depth warm-up.
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
        let p = prompt_tokens.len();
        if p == 0 {
            return Ok(());
        }
        let hshape = ffi::array_shape(hidden);
        if hshape.len() != 3 || hshape[1] != p as i32 {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "qwen3_5_mtp drafter prefill: hidden shape {hshape:?} does not cover the \
                     {p}-token prompt"
                ),
            });
        }
        // Start clean: the prompt prefill defines position 0 of the drafter
        // sequence, so any prior state is stale by construction.
        self.clear_runtime_state();

        // Position i pairs token_{i+1} with hidden_i: shift the prompt left
        // by one and append the just-sampled first bonus.
        let mut shifted: Vec<i32> = prompt_tokens[1..].to_vec();
        shifted.push(first_bonus);
        let h = self.forward_hidden_stack(&shifted, hidden)?;
        self.set_seed_from_hidden(&h, sampler)
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
        let hshape = ffi::array_shape(verify_hidden);
        let block = draft_tokens.len() + 1;
        if hshape.len() != 3 || (hshape[1] as usize) < block || accepted > draft_tokens.len() {
            // Validate BEFORE mutating: a malformed accept would otherwise
            // leave a half-trimmed cache. Poisoned state is cleared so the
            // next `set_shared_kv` re-anchors cleanly.
            self.clear_runtime_state();
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "qwen3_5_mtp drafter accept: verify_hidden shape {hshape:?} does not cover \
                     block={block} / accepted={accepted}"
                ),
            });
        }

        // Trim the in-round cache entries beyond the accepted prefix. Entries
        // appended during `draft_block` used the drafter's own hidden as the
        // pair input; accepted ones are kept as-is (upstream does the same),
        // rejected ones are removed.
        let keep = accepted.min(self.round_appended.max(0) as usize);
        let trim = self.round_appended - keep as i32;
        if trim > 0 {
            for cache in &mut self.cache {
                cache.trim(trim);
            }
            self.next_position -= trim;
        }

        // Extend with the accepted tokens not yet in the cache, paired with
        // the target's true verify hidden, plus the newly emitted bonus.
        let h_dim = hshape[2];
        let mut tokens: Vec<i32> = Vec::new();
        let mut hidden_cat: Option<UniquePtr<MlxArray>> = None;
        let push_slice = |pos: usize, hidden_cat: &mut Option<UniquePtr<MlxArray>>| {
            let pos = pos as i32;
            let s = ffi::slice(verify_hidden, &[0, pos, 0], &[hshape[0], pos + 1, h_dim]);
            *hidden_cat = Some(match hidden_cat.take() {
                None => s,
                Some(prev) => crate::ops::concatenate(&prev, &s, 1),
            });
        };
        for (draft_idx, &draft_tok) in draft_tokens.iter().enumerate().take(accepted).skip(keep) {
            tokens.push(draft_tok);
            push_slice(draft_idx, &mut hidden_cat);
        }
        if let Some(&last) = new_tokens.last() {
            tokens.push(last);
            push_slice(accepted, &mut hidden_cat);
        }

        if let (false, Some(hiddens)) = (tokens.is_empty(), hidden_cat) {
            let h = self.forward_hidden_stack(&tokens, &hiddens)?;
            self.set_seed_from_hidden(&h, sampler)?;
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
            // Degenerate budget: drop any pending seed (its position
            // assumption still holds, but the round loop only reaches here
            // on terminal budgets).
            return Ok(Vec::new());
        }

        let mut tokens: Vec<i32> = Vec::with_capacity(block_size - 1);
        self.round_appended = 0;

        // Seed fast path: the prompt prefill / accept hook already computed
        // this round's first proposal from TARGET hidden (and its cache entry
        // is already appended), so consume it instead of re-deriving from the
        // caller's (bonus, hidden) pair — the two computations are identical
        // by construction.
        let (mut tok, mut h_prev): (i32, UniquePtr<MlxArray>) =
            match (self.seed_token.take(), self.seed_hidden.take()) {
                (Some(seed_tok), Some(seed_hidden)) => {
                    tokens.push(seed_tok);
                    (seed_tok, seed_hidden)
                }
                _ => {
                    let hidden = hidden.ok_or(DrafterError::DraftBlockMissingHidden)?;
                    (last_bonus, ffi::copy(hidden))
                }
            };

        self.in_draft_step = true;
        while tokens.len() < block_size - 1 {
            let step_started = Instant::now();
            let h = self.forward_hidden_stack(&[tok], &h_prev)?;
            self.round_appended += 1;
            let lm_head_started = Instant::now();
            let logits = self.project_logits(&h)?;
            maybe_sync(&logits);
            let sampled_started = Instant::now();
            tok = self.sample_one_profiled(&logits, sampler, sampled_started);
            self.active_cost().lm_head_ms += ms_since(lm_head_started, sampled_started);
            self.profile.total_ms += step_started.elapsed().as_secs_f64() * 1e3;
            self.profile.steps += 1;
            tokens.push(tok);
            h_prev = h;
        }
        self.in_draft_step = false;
        tokens.truncate(block_size - 1);
        Ok(tokens)
    }

    fn draft_profile(&self) -> Option<DraftStepProfile> {
        Some(self.profile)
    }

    fn sanitize(&mut self, weights: &mut WeightMap) -> Result<(), DrafterError> {
        Self::sanitize_weights(weights);
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Mtp
    }
}
