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

//! [`DFlashDrafter`] — adapter that wraps a [`DFlashDraftModel`] and a
//! per-layer K/V cache slice behind the
//! [`Drafter`](crate::drafter::Drafter) trait surface.
//!
//! The wrapper holds the owned model + its own caches. It exposes the
//! object-safe trait so [`load_drafter`](crate::drafter::load_drafter)
//! can return a `Box<dyn Drafter>` from the `Dflash` arm.
//!
//! The fully end-to-end DFlash round loop (target prefill → capture
//! hiddens → drafter `draft_block` → target verify → rollback) lands
//! in epic- sub-12. This file ships only what the trait
//! surface needs today.
//!
//! The same wrapper serves the LFM2 / LFM2.5 DSpark drafters (issue
//! #1339): a DSpark checkpoint is a DFlash backbone plus a Markov head, so
//! it loads through the same path and differs only in the draft step
//! (`DFlashDraftModel::draft_block_dspark_array`), in being greedy-only,
//! and in the block-size policy (`configured_block_size` /
//! `prefer_requested_block_size`).

use crate::cache::KVCache;
use crate::drafter::{Drafter, DrafterError, DrafterKind};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::weights::WeightMap;
use cxx::UniquePtr;
use std::path::Path;

use super::config::{DFlashConfig, DSPARK_MAX_VERIFY_WIDTH};
use super::model::DFlashDraftModel;

/// Boxed [`Drafter`] implementation for the Qwen 3.5 DFlash drafter.
///
/// Wraps a [`DFlashDraftModel`] plus the per-layer K/V cache list that
/// the round-loop driver passes through `draft_block`. The wrapper owns
/// the caches because the trait surface does not let the caller hand
/// per-layer caches in alongside the `last_bonus` token (this would
/// have required `&mut [KVCache]` plumbing through the trait). Owning
/// them inside the wrapper keeps the trait surface uniform across the
/// MTP / DFlash / InternalMtp shapes.
///
/// ## Lifecycle
///
/// 1. `DFlashDrafter::load(path)` — load + sanitize weights, build
///    the model, allocate the per-layer cache slice. The published
///    `z-lab/Qwen3.5-4B-DFlash` checkpoint omits `embed_tokens.weight`,
///    so the model is built with an `embed_tokens = None` tombstone.
/// 2. `bind(target)` — resolve the binding contract. For the lazy-bind
///    checkpoint this installs a shared-buffer handle to the *target's*
///    `embed_tokens` module into the tombstone (matching upstream
///    Python's `self.embed_tokens = target.embed_tokens`); for a
///    self-contained checkpoint that shipped its own table this is just
///    a capability smoke-test that the target can embed at all.
/// 3. `set_target_hidden(hidden)` (optional pre-flight) — store the
///    target-hidden buffer for the next `draft_block` call. The
///    trait-level `draft_block` signature takes `hidden: Option<&MlxArray>`
///    so callers can pass it directly.
/// 4. `draft_block(last_bonus, hidden, block_size, sampler)` — run one
///    masked-forward draft round. Returns `block_size - 1` proposal
///    tokens.
/// 5. `reset(target)` — between full generation calls; clears caches.
pub struct DFlashDrafter {
    /// Owned drafter model.
    pub model: DFlashDraftModel,

    /// Per-layer K/V cache slice. `caches.len() == model.layers.len()`.
    caches: Vec<KVCache>,

    /// Records whether `bind` has been called at least once. The DFlash
    /// round-loop driver in sub-12 reads this flag to confirm
    /// the drafter is wired before invoking `draft_block`.
    bound: bool,
}

impl DFlashDrafter {
    /// Load the drafter checkpoint at `path` (a directory containing
    /// `config.json`, `model.safetensors` or sharded equivalents, plus
    /// whatever auxiliary files the published `z-lab/Qwen3.5-4B-DFlash`
    /// checkpoint ships).
    ///
    /// Steps:
    ///
    /// 1. Read `path/config.json` and parse a [`DFlashConfig`].
    /// 2. Load all `*.safetensors` shards via
    ///    [`crate::weights::load_weights_from_dir`].
    /// 3. Sanitize the weight keys (strip `model.` prefix) via
    ///    [`DFlashDraftModel::sanitize`].
    /// 4. Apply the host's load-time dtype policy to the non-quantized
    ///    tensors ([`apply_drafter_load_dtype_policy`]): bf16 → f16 on
    ///    Apple Silicon and pre-Ampere CUDA, bf16 kept elsewhere.
    /// 5. Build the model and allocate its per-layer K/V cache slice.
    pub fn load(path: &Path) -> Result<Self, DrafterError> {
        let config_path = path.join("config.json");
        let config_bytes = std::fs::read(&config_path).map_err(|e| DrafterError::ConfigIo {
            path: config_path.display().to_string(),
            source: e,
        })?;
        let config_json: serde_json::Value =
            serde_json::from_slice(&config_bytes).map_err(|e| DrafterError::ConfigParse {
                path: config_path.display().to_string(),
                source: e,
            })?;
        let config =
            DFlashConfig::from_json(&config_json).map_err(|e| DrafterError::ConfigParse {
                path: config_path.display().to_string(),
                source: serde::de::Error::custom(e),
            })?;

        let mut weights = crate::weights::load_weights_from_dir(path)
            .map_err(|msg| DrafterError::LoadFailed { reason: msg })?;

        // Strip `model.` prefix from any key carrying it. Mirrors upstream
        // `DFlashDraftModel.sanitize`.
        DFlashDraftModel::sanitize(&mut weights);

        // Load-time dtype: f16 where the target loaders convert to f16 (Apple
        // Silicon, pre-Ampere CUDA), bf16 where they keep bf16 (Ampere and
        // later CUDA). Quantized tensors keep their bf16 scales/biases as-is
        // because `quantized_matmul` handles bf16 natively.
        apply_drafter_load_dtype_policy(&mut weights);

        let model = DFlashDraftModel::from_weights(&weights, config)
            .map_err(|msg| DrafterError::LoadFailed { reason: msg })?;
        let caches = model.make_cache();

        Ok(Self {
            model,
            caches,
            bound: false,
        })
    }

    /// Whether `bind` has been called at least once on this drafter.
    pub fn is_bound(&self) -> bool {
        self.bound
    }

    /// Borrowed access to the drafter's per-layer K/V caches.
    pub fn caches(&self) -> &[KVCache] {
        &self.caches
    }

    /// Mutably borrowed access to the drafter's per-layer K/V caches.
    /// Used by tests pinning the "context K/V only" invariant.
    pub fn caches_mut(&mut self) -> &mut [KVCache] {
        &mut self.caches
    }
}

/// Rewrite a drafter's bf16 tensors to f16 when, and only when, this host's
/// target loaders do the same.
///
/// The target loaders in the binary crate (`bf16_to_f16_at_load` in
/// `src/models/sanitize.rs`) convert an unquantized bf16 checkpoint to f16 on
/// Apple Silicon and on pre-Ampere CUDA, and keep it bf16 on Ampere and later
/// CUDA unless `MLXCEL_CUDA_F16_NORMALIZE` opts in. A drafter has to land in
/// the same dtype as the residual stream it reads, because MLX promotes a
/// bf16 activation times an f16 weight to float32: the whole drafter forward
/// then runs in f32, and every weight is re-upcast on every round.
///
/// Measured on GB10 (sm_121) with `qwen3.5-4b-4bit` and its DFlash drafter
/// (issue #1782): before this gate the drafter's f16 weights cost about 240
/// launches and 30 to 38 ms of device time per verify round at every block
/// width (72 `copy_v<__half, float>` upcasts of the 540M drafter weights plus
/// f32 cutlass GEMMs), which was the largest fixed term in the round.
///
/// `MLXCEL_KEEP_BF16` and `MLXCEL_CUDA_F16_NORMALIZE` are honored the way the
/// target loaders honor them, so the two sides cannot drift apart under an
/// operator override.
///
/// Used by: DFlash drafter load, Muse Glimmer drafter load, Inkling MTP
/// drafter load, Qwen 3.5 MTP drafter load.
pub(crate) fn apply_drafter_load_dtype_policy(weights: &mut WeightMap) {
    if drafter_bf16_to_f16_at_load() {
        convert_bf16_to_f16_non_quantized(weights);
    }
}

/// Whether this host converts an unquantized bf16 drafter to f16 at load.
///
/// Used by: [`apply_drafter_load_dtype_policy`].
pub fn drafter_bf16_to_f16_at_load() -> bool {
    let apple_silicon =
        crate::hardware::get_hardware().silicon_gen != crate::hardware::AppleSiliconGen::Unknown;
    drafter_bf16_to_f16_policy(
        std::env::var_os("MLXCEL_KEEP_BF16").is_some(),
        apple_silicon,
        crate::cuda_arch::cuda_compute_capability(),
        EnvFlag::read("MLXCEL_CUDA_F16_NORMALIZE"),
    )
}

/// Tri-state reading of a boolean environment flag, with the same spelling
/// rules as the binary crate's `env_flag_enabled` / `env_flag_disabled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EnvFlag {
    Unset,
    Enabled,
    Disabled,
}

impl EnvFlag {
    fn read(name: &str) -> Self {
        match std::env::var(name) {
            Ok(raw) => Self::parse(&raw),
            Err(_) => Self::Unset,
        }
    }

    fn parse(raw: &str) -> Self {
        let v = raw.trim();
        if v.is_empty() {
            return Self::Unset;
        }
        if v == "0"
            || v.eq_ignore_ascii_case("false")
            || v.eq_ignore_ascii_case("off")
            || v.eq_ignore_ascii_case("no")
        {
            return Self::Disabled;
        }
        Self::Enabled
    }
}

/// Pure form of [`drafter_bf16_to_f16_at_load`], mirroring the arms of the
/// binary crate's `bf16_to_f16_at_load` for an unquantized checkpoint:
///
/// - `MLXCEL_KEEP_BF16` set: keep bf16 everywhere (the dtype-policy A/B
///   instrument).
/// - pre-Ampere CUDA (compute capability below 8): convert, unless
///   `MLXCEL_CUDA_F16_NORMALIZE` is explicitly disabled. Those parts have no
///   native bf16, so the target converts too.
/// - Apple Silicon: convert (Metal's bfloat is storage-only).
/// - Ampere and later CUDA: keep bf16 unless `MLXCEL_CUDA_F16_NORMALIZE` is
///   explicitly enabled.
/// - anything else (CPU-only builds): keep bf16.
pub(crate) fn drafter_bf16_to_f16_policy(
    keep_bf16: bool,
    apple_silicon: bool,
    cuda_compute_capability: Option<(u32, u32)>,
    cuda_f16_normalize: EnvFlag,
) -> bool {
    if keep_bf16 {
        return false;
    }
    if let Some((major, _)) = cuda_compute_capability
        && major < 8
    {
        return cuda_f16_normalize != EnvFlag::Disabled;
    }
    if apple_silicon {
        return true;
    }
    cuda_compute_capability.is_some() && cuda_f16_normalize == EnvFlag::Enabled
}

/// Convert every bf16 tensor in `weights` to f16, unconditionally. Quantized
/// scales and biases (recognised by living next to a `.scales` or `.biases`
/// key in the map) are kept as-is. Callers that load a checkpoint go through
/// [`apply_drafter_load_dtype_policy`] instead, which decides per host.
///
/// This mirrors the binary crate's `convert_bf16_weights` (in
/// `src/models/sanitize.rs`) but is duplicated here because the factory
/// in `mlxcel-core` cannot reach the binary's helpers. The Apple
/// Silicon precision rules in `docs/apple-silicon-precision.md` require
/// every weight loader to apply the same bf16 → f16 rewrite before
/// handing weights to the model constructor.
///
/// `weights` is mutated in place; non-bf16 tensors and quantization
/// auxiliaries (scales, biases) are untouched.
///
/// Used by: [`apply_drafter_load_dtype_policy`], and the loader tests below.
pub(crate) fn convert_bf16_to_f16_non_quantized(weights: &mut WeightMap) {
    let bf16_keys: Vec<String> = weights
        .iter()
        .filter(|(k, v)| {
            // Skip quantization auxiliaries — even though they are often
            // bf16, quantized_matmul handles bf16 natively for these.
            !k.ends_with(".scales")
                && !k.ends_with(".biases")
                && ffi::array_dtype(v) == crate::dtype::BFLOAT16
        })
        .map(|(k, _)| k.clone())
        .collect();

    for key in bf16_keys {
        if let Some(tensor) = weights.get(&key) {
            let converted = ffi::astype(tensor, crate::dtype::FLOAT16);
            weights.insert(key, converted);
        }
    }
}

/// Whether `sampler` selects tokens by argmax, the only mode a DSpark
/// drafter can draft under. Mirrors the greedy test the per-position DFlash
/// sampling helpers below use.
///
/// Used by: `DFlashDrafter::draft_block*`, the server DFlash dispatch gate.
pub fn sampler_is_greedy(sampler: &SamplingConfig) -> bool {
    sampler.temperature == 0.0 || sampler.top_k == 1
}

/// Structural half of the DSpark pairing gate: whether the drafter's own
/// config and `fc` projection agree with each other, with no target in
/// hand. `fc_in_features` is the `fc` weight's input width.
///
/// Returns the operator-facing reason on failure, `None` when the config is
/// self-consistent. A free function so it can be tested without a drafter
/// checkpoint on disk.
///
/// Used by: [`DFlashDrafter::validate_target_compat`].
pub(crate) fn dspark_config_pairing_error(
    config: &DFlashConfig,
    fc_in_features: usize,
) -> Option<String> {
    let ids = &config.target_layer_ids;
    let strictly_increasing = ids.windows(2).all(|w| w[0] < w[1]);
    if ids.is_empty() || !strictly_increasing {
        return Some(format!(
            "DSpark drafter target_layer_ids {ids:?} must be non-empty and strictly increasing"
        ));
    }
    if ids
        .last()
        .is_some_and(|&last| last >= config.num_target_layers)
    {
        return Some(format!(
            "DSpark drafter target_layer_ids {ids:?} reach past num_target_layers = {}",
            config.num_target_layers
        ));
    }
    let expected = ids.len() * config.hidden_size;
    if fc_in_features != expected {
        return Some(format!(
            "DSpark drafter fc projection reads {fc_in_features} features but \
             len(target_layer_ids) * hidden_size = {} * {} = {expected}",
            ids.len(),
            config.hidden_size
        ));
    }
    // The mask id indexes the TARGET's embedding table, because a DSpark
    // drafter ships none of its own, and MLX range-checks no positive gather
    // index: an id past the last row reads whatever follows the table in the
    // buffer and feeds it to the logits. The target half of this gate pins the
    // target's vocabulary to `vocab_size`, so bounding the id against
    // `vocab_size` here bounds the gather.
    if config.mask_token_id < 0
        || usize::try_from(config.mask_token_id).is_ok_and(|id| id >= config.vocab_size)
    {
        return Some(format!(
            "DSpark drafter mask_token_id {} is outside the drafter vocabulary [0, {})",
            config.mask_token_id, config.vocab_size
        ));
    }
    if config.runtime_verify_width() < 2 {
        return Some(format!(
            "DSpark drafter runs at {} verify row(s) (block_size = {}, runtime_block_size = \
             {:?}); a verify width below 2 proposes nothing and the round loop would emit one \
             token per burst",
            config.runtime_verify_width(),
            config.block_size,
            config.runtime_block_size
        ));
    }
    // High side of the same bound. `runtime_verify_width()` already clamps at
    // the ceiling, which is what protects the exactness probe (it runs on the
    // scheduler thread, one target forward per row, before this gate is ever
    // reached). Refusing the config here as well is so an operator whose
    // drafter asks for an impossible width is told, rather than quietly
    // served at 32.
    if config.requested_verify_width() > DSPARK_MAX_VERIFY_WIDTH {
        return Some(format!(
            "DSpark drafter asks for {} verify rows (block_size = {}, runtime_block_size = \
             {:?}), above the {} ceiling; the published checkpoints run at 8 to 10 rows, and \
             each row costs one target forward in the block-versus-chain exactness probe",
            config.requested_verify_width(),
            config.block_size,
            config.runtime_block_size,
            DSPARK_MAX_VERIFY_WIDTH
        ));
    }
    None
}

/// Target half of the DSpark pairing gate. `target_vocab` is `None` when the
/// target does not hand out a tied embedding to measure, in which case the
/// vocabulary check is skipped rather than guessed.
///
/// Used by: [`DFlashDrafter::validate_target_compat`].
pub(crate) fn dspark_target_pairing_error(
    config: &DFlashConfig,
    target_hidden: usize,
    target_layers: usize,
    target_vocab: Option<usize>,
) -> Option<String> {
    if target_hidden != config.hidden_size {
        return Some(format!(
            "DSpark drafter is incompatible with this target: drafter hidden_size = {} but the \
             target's hidden size = {target_hidden}. The drafter's fc projection reads the \
             target's residual streams, so these must be equal (pair the LFM2.5-2.6B target with \
             LFM2.5-2.6B-DSpark, the 8B-A1B target with 8B-A1B-DSpark, and so on).",
            config.hidden_size
        ));
    }
    if target_layers != config.num_target_layers {
        return Some(format!(
            "DSpark drafter is incompatible with this target: drafter num_target_layers = {} but \
             the target has {target_layers} layers. The captured target_layer_ids {:?} index \
             into the target's layer stack, so the drafter must be the one published for this \
             exact target.",
            config.num_target_layers, config.target_layer_ids
        ));
    }
    if let Some(vocab) = target_vocab
        && vocab != config.vocab_size
    {
        return Some(format!(
            "DSpark drafter vocabulary is incompatible with this target: drafter vocab_size = {} \
             but the target's tied LM head emits {vocab} logits. The Markov head is a transition \
             table over the target vocabulary, so the two must match.",
            config.vocab_size
        ));
    }
    None
}

impl DFlashDrafter {
    /// Whether this wrapper drives a DSpark drafter (issue #1339).
    pub fn is_dspark(&self) -> bool {
        self.model.is_dspark()
    }

    /// DSpark cannot draft under a stochastic sampler; every other DFlash
    /// checkpoint samples each masked position under the caller's config.
    fn ensure_sampler_supported(&self, sampler: &SamplingConfig) -> Result<(), DrafterError> {
        if self.is_dspark() && !sampler_is_greedy(sampler) {
            return Err(DrafterError::GreedyOnly {
                kind: self.kind(),
                temperature: sampler.temperature,
                top_k: sampler.top_k,
            });
        }
        Ok(())
    }

    /// The structural half of the DSpark pairing check: the drafter's own
    /// config must be self-consistent before the target is consulted.
    fn validate_dspark_config(&self) -> Result<(), DrafterError> {
        let fc_in = ffi::array_shape(&self.model.fc.weight)
            .last()
            .copied()
            .unwrap_or(0) as usize;
        match dspark_config_pairing_error(&self.model.config, fc_in) {
            Some(reason) => Err(DrafterError::BindFailed { reason }),
            None => Ok(()),
        }
    }
}

impl Drafter for DFlashDrafter {
    /// DSpark pairing gate (issue #1339). A DSpark drafter consumes the
    /// target's residual streams at `target_layer_ids` through an `fc`
    /// projection sized `len(target_layer_ids) * hidden_size`, borrows the
    /// target's tied embedding table as both its input embedding and its
    /// LM head, and chains proposals over the target's vocabulary, so the
    /// target's hidden size, layer count and vocabulary must all match the
    /// drafter config. A Qwen 3.5 DFlash drafter keeps the no-op default:
    /// the Qwen family has no such gate today and adding one would change
    /// which pairings it accepts.
    ///
    /// The three measured quantities stand in for the `model_type in {lfm2,
    /// lfm2_moe}` check the issue describes, because [`LanguageModel`]
    /// exposes no architecture string and the server's own dispatch gate
    /// already restricts the burst to the LFM2 `LoadedModel` variants. They
    /// are also the stricter test: two different LFM2 checkpoints are both
    /// `lfm2` and still cannot be paired with each other's drafter.
    fn validate_target_compat(&self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        if !self.is_dspark() {
            return Ok(());
        }
        self.validate_dspark_config()?;
        let config = &self.model.config;

        let sentinel = ffi::from_slice_i32(&[0_i32], &[1, 1]);
        let embedded =
            target
                .embed_tokens(&sentinel)
                .ok_or(DrafterError::TargetMissingFeature {
                    feature: "embed_tokens",
                })?;
        let target_hidden = ffi::array_shape(&embedded).last().copied().unwrap_or(0) as usize;
        // The tied head is what the drafter borrows as its LM head, so its
        // width is the vocabulary the Markov chain must index. A target that
        // hands out no embedding module leaves the check unmeasured rather
        // than guessed.
        let target_vocab = target.embed_tokens_module().map(|embed_module| {
            let zero_hidden = ffi::zeros(&[1, 1, target_hidden as i32], crate::dtype::FLOAT32);
            let logits = embed_module.as_linear(&zero_hidden);
            ffi::array_shape(&logits).last().copied().unwrap_or(0) as usize
        });

        match dspark_target_pairing_error(config, target_hidden, target.num_layers(), target_vocab)
        {
            Some(reason) => Err(DrafterError::BindFailed { reason }),
            None => Ok(()),
        }
    }

    fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        // Two embedding cases, mirroring upstream Python's lazy-bind shape
        // (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/qwen3_dflash/dflash.py
        // lines 88, 92-108):
        //
        // 1. Lazy-bind checkpoint (the published `z-lab/Qwen3.5-4B-DFlash`):
        //    the drafter shipped NO `embed_tokens.weight`, so
        //    `DFlashDraftModel::from_weights` left `embed_tokens = None`.
        //    Resolve it now from the target's embedding *module* via
        //    `LanguageModel::embed_tokens_module` and install it into the
        //    tombstone. This is the load-bearing fix for the published
        //    checkpoint — without it `forward()` panics on the unbound
        //    embedding.
        //
        // 2. Self-contained checkpoint: the drafter shipped its own
        //    `embed_tokens.weight`. We only need the legacy capability
        //    smoke-test — confirm the target can embed at all — so the
        //    binding contract still fails fast on a target that does not
        //    expose `embed_tokens` (e.g. a non-Qwen-3.5 model fed in by
        //    mistake).
        if self.model.needs_embed_binding() {
            let embed = target
                .embed_tokens_module()
                .ok_or_else(|| DrafterError::BindFailed {
                    reason: format!(
                        "DFlash drafter checkpoint omits embed_tokens.weight \
                         and the target does not expose embed_tokens_module(); \
                         a lazy-bind DFlash drafter requires a target that \
                         hands out its embedding table (the Qwen 3.5 and LFM2 \
                         families do; check the target family matches the \
                         drafter) (kind = {})",
                        self.kind()
                    ),
                })?;
            self.model.bind_target_embedding(embed);
        } else {
            // Legacy capability smoke-test for self-contained checkpoints:
            // a 1-element dummy id array proves the target can embed.
            let dummy = ffi::from_slice_i32(&[0_i32], &[1, 1]);
            if target.embed_tokens(&dummy).is_none() {
                return Err(DrafterError::BindFailed {
                    reason: format!(
                        "target model does not expose embed_tokens; \
                         DFlash drafter requires a target with a working \
                         embed_tokens method (kind = {})",
                        self.kind()
                    ),
                });
            }
        }
        if self.model.needs_lm_head_binding() {
            // Some official larger DFlash checkpoints (for example
            // `z-lab/Qwen3.5-27B-DFlash`) also omit `lm_head.weight` while
            // setting `tie_word_embeddings = false`. Upstream Python binds
            // the target's untied output head in the same `bind()` step as
            // `embed_tokens`, falling back to the tied embedding projection
            // if the target exposes no explicit LM head. The `Option` keeps
            // that fallback behavior in the model forward path.
            self.model.bind_target_lm_head(target.lm_head_module());
        }
        self.bound = true;
        Ok(())
    }

    fn make_cache(&self) -> Vec<KVCache> {
        // The trait contract returns a freshly-allocated cache slice
        // for the *caller* to manage. DFlashDrafter holds its own
        // caches in `self.caches` for in-loop use; `make_cache` is
        // exposed in case the caller wants to spin up an alternate
        // drafter session.
        self.model.make_cache()
    }

    fn reset(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        // Re-bind (a no-op outside of the bound-flag check) and clear
        // every cache to its initial state.
        self.bind(target)?;
        self.caches = self.model.make_cache();
        Ok(())
    }

    fn dflash_target_layer_ids(&self) -> Option<&[usize]> {
        Some(&self.model.config.target_layer_ids)
    }

    /// DSpark (issue #1339): the verify width the checkpoint runs at by
    /// default, `min(block_size + 1, runtime_block_size)`. A plain DFlash
    /// drafter keeps the trait default (`None`).
    ///
    /// Declarative on this drafter. The DFlash round loop reads this hook
    /// and [`Self::prefer_requested_block_size`] since issue #1343, but only
    /// widens between a configured depth and the requested ceiling when the
    /// drafter does NOT prefer the requested width; DSpark does, so the
    /// no-backoff property below holds whatever this returns. The width a
    /// DSpark run actually gets is decided one layer up, before the drafter
    /// is loaded: `resolve_draft_block_size` peeks the drafter config
    /// through `peek_dspark_configured_block_size` and passes the same
    /// `runtime_verify_width()` in as `block_size`, unless
    /// `--draft-block-size` overrides it.
    fn configured_block_size(&self) -> Option<usize> {
        self.is_dspark()
            .then(|| self.model.config.runtime_verify_width())
    }

    /// DSpark never backs its block size off on low acceptance; the
    /// requested width (the config default or `--draft-block-size`) is the
    /// width it runs at. See [`Self::configured_block_size`] for why this is
    /// currently a declaration rather than a control input.
    fn prefer_requested_block_size(&self) -> bool {
        self.is_dspark()
    }

    fn greedy_only(&self) -> bool {
        self.is_dspark()
    }

    /// Delegates to the inherent [`DFlashDrafter::is_dspark`] by its
    /// qualified name rather than repeating its body. An inherent method and
    /// a trait method of the same name both resolve here (the inherent one
    /// wins on a concrete `DFlashDrafter`, the trait one through
    /// `dyn Drafter`), and two copies of the same expression would be free to
    /// drift apart without any call site noticing.
    fn is_dspark(&self) -> bool {
        DFlashDrafter::is_dspark(self)
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        let target_hidden = hidden.ok_or_else(|| DrafterError::DraftFailed {
            reason: "DFlash drafter requires a target hidden state \
                     (target_layer_ids concatenation); got hidden = None"
                .to_string(),
        })?;

        if block_size < 2 {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "DFlash drafter requires block_size >= 2 (got {block_size}); \
                     block_size 1 has no masked positions to sample"
                ),
            });
        }
        self.ensure_sampler_supported(sampler)?;
        if self.is_dspark() {
            return Ok(self.model.draft_block_dspark(
                last_bonus,
                target_hidden,
                &mut self.caches,
                block_size,
            ));
        }

        let mask_id = self.model.config.mask_token_id;
        let mut block: Vec<i32> = Vec::with_capacity(block_size);
        block.push(last_bonus);
        for _ in 1..block_size {
            block.push(mask_id);
        }
        let inputs = ffi::from_slice_i32(&block, &[1, block_size as i32]);

        let logits = self.model.forward(&inputs, target_hidden, &mut self.caches);

        // Sample one token per masked position. The block layout is
        // [last_bonus, mask, mask, ..., mask], so positions [1, ..., L-1]
        // are the proposal slots; we sample those.
        sample_block_per_position(&logits, block_size, sampler)
    }

    fn draft_block_array(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let target_hidden = hidden.ok_or_else(|| DrafterError::DraftFailed {
            reason: "DFlash drafter requires a target hidden state \
                     (target_layer_ids concatenation); got hidden = None"
                .to_string(),
        })?;

        if block_size < 2 {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "DFlash drafter requires block_size >= 2 (got {block_size}); \
                     block_size 1 has no masked positions to sample"
                ),
            });
        }
        self.ensure_sampler_supported(sampler)?;
        if self.is_dspark() {
            return Ok(self.model.draft_block_dspark_array(
                last_bonus,
                target_hidden,
                &mut self.caches,
                block_size,
            ));
        }

        let mask_id = self.model.config.mask_token_id;
        let mut block: Vec<i32> = Vec::with_capacity(block_size);
        block.push(last_bonus);
        for _ in 1..block_size {
            block.push(mask_id);
        }
        let inputs = ffi::from_slice_i32(&block, &[1, block_size as i32]);

        let logits = self.model.forward(&inputs, target_hidden, &mut self.caches);
        sample_block_per_position_array(&logits, block_size, sampler)
    }

    fn draft_block_batched(
        &mut self,
        last_bonus: &[i32],
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<Vec<Vec<i32>>, DrafterError> {
        let target_hidden = hidden.ok_or_else(|| DrafterError::DraftFailed {
            reason: "DFlash drafter (batched) requires a target hidden state \
                     (target_layer_ids concatenation); got hidden = None"
                .to_string(),
        })?;

        if block_size < 2 {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "DFlash drafter requires block_size >= 2 (got {block_size}); \
                     block_size 1 has no masked positions to sample"
                ),
            });
        }
        if last_bonus.is_empty() {
            return Err(DrafterError::DraftFailed {
                reason: "DFlash drafter (batched) requires B >= 1 bonus tokens".to_string(),
            });
        }
        if self.is_dspark() {
            return Err(DrafterError::DraftFailed {
                reason: "DSpark drafter is B = 1 only; the batched DFlash path must decline \
                         to classic decode for this pairing"
                    .to_string(),
            });
        }

        let batch_size = last_bonus.len();
        let mask_id = self.model.config.mask_token_id;

        // Build the per-row block layout: row r = [bonus[r], mask, mask, ..., mask].
        // Final tensor shape is [B, block_size]. We materialize the entire
        // [B * block_size] buffer in i32 then hand it to from_slice_i32.
        let mut block: Vec<i32> = Vec::with_capacity(batch_size * block_size);
        for &bonus in last_bonus {
            block.push(bonus);
            for _ in 1..block_size {
                block.push(mask_id);
            }
        }
        let inputs = ffi::from_slice_i32(&block, &[batch_size as i32, block_size as i32]);

        // The model's forward already handles [B, L] inputs; the
        // returned logits are [B, L, vocab].
        let logits = self.model.forward(&inputs, target_hidden, &mut self.caches);

        // Sample one token per (row, masked-position) pair.
        sample_block_per_position_batched(&logits, batch_size, block_size, sampler)
    }

    fn sanitize(&mut self, weights: &mut WeightMap) -> Result<(), DrafterError> {
        // The trait contract is "drop weight keys this drafter must not
        // carry into runtime". For DFlash, that's the upstream
        // `model.` prefix strip — applied at load time too, but exposed
        // here for callers that re-feed weights through the trait.
        DFlashDraftModel::sanitize(weights);
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Dflash
    }
}

/// Per-row, per-position sampling helper for the batched DFlash draft.
///
/// Given `logits` of shape `[B, block_size, vocab]` and a sampler config,
/// sample one token from each (row, masked-position) cell. Returns
/// `Vec<Vec<i32>>` with shape `[B][block_size - 1]`.
///
/// Greedy (temperature == 0.0 OR `top_k == 1`) uses per-position argmax.
/// Stochastic uses `fused_sample` per position over the `[1, vocab]`
/// slice for that position.
///
/// Used by: `DFlashDrafter::draft_block_batched`.
fn sample_block_per_position_batched(
    logits: &MlxArray,
    batch_size: usize,
    block_size: usize,
    sampler: &SamplingConfig,
) -> Result<Vec<Vec<i32>>, DrafterError> {
    let shape = ffi::array_shape(logits);
    if shape.len() != 3 || shape[0] != batch_size as i32 || shape[1] != block_size as i32 {
        return Err(DrafterError::DraftFailed {
            reason: format!(
                "DFlash drafter (batched) expected logits shape \
                 [{batch_size}, {block_size}, vocab]; got {shape:?}"
            ),
        });
    }
    let vocab = shape[2];
    let n = block_size - 1;
    let mut out: Vec<Vec<i32>> = (0..batch_size).map(|_| Vec::with_capacity(n)).collect();

    let greedy = sampler.temperature == 0.0 || sampler.top_k == 1;
    if greedy {
        // Greedy DFlash is the hot server path. Argmax the whole
        // `[B, block_size - 1, vocab]` proposal slab in one MLX op and
        // materialize all token ids with one contiguous host copy. The old
        // row-by-row implementation performed `(B * (K - 1))` slice/eval/item
        // synchronizations per round, which showed up as a dominant short-run
        // Qwen3.5-4B overhead.
        let masked_logits = ffi::slice(
            logits,
            &[0_i32, 1_i32, 0_i32],
            &[batch_size as i32, block_size as i32, vocab],
        );
        let argmax = ffi::argmax_last_axis(&masked_logits);
        let flat = super::materialize_argmax_i32_vec(&argmax, batch_size * n);
        for (b, row) in out.iter_mut().enumerate() {
            let start = b * n;
            row.extend_from_slice(&flat[start..start + n]);
        }
        return Ok(out);
    }

    for b in 0..batch_size as i32 {
        for i in 0..n {
            // Row `(b, i+1)` of the [B, L, V] logits.
            let pos = (i + 1) as i32;
            let row = ffi::slice(logits, &[b, pos, 0_i32], &[b + 1, pos + 1, vocab]);
            // Drop the seq axis so we get a `[1, vocab]` 2D slice (fused_sample
            // / argmax expect `[batch, vocab]`).
            let row = ffi::reshape(&row, &[1_i32, vocab]);
            let token = ffi::fused_sample(
                &row,
                sampler.temperature,
                sampler.top_k,
                sampler.top_p,
                sampler.min_p,
            );
            ffi::eval(&token);
            out[b as usize].push(ffi::item_i32(&token));
        }
    }
    Ok(out)
}

/// Per-position sampling helper.
///
/// Given `logits` of shape `[1, block_size, vocab]` and a sampler config,
/// sample one token from each masked position (rows `[1, ..., block_size - 1]`).
/// Returns `Vec<i32>` of length `block_size - 1`.
///
/// Greedy (temperature == 0.0 OR `top_k == 1`) uses per-position argmax.
/// Stochastic uses `fused_sample` per position over the `[1, vocab]`
/// slice for that position.
fn sample_block_per_position(
    logits: &MlxArray,
    block_size: usize,
    sampler: &SamplingConfig,
) -> Result<Vec<i32>, DrafterError> {
    let shape = ffi::array_shape(logits);
    if shape.len() != 3 || shape[0] != 1 || shape[1] != block_size as i32 {
        return Err(DrafterError::DraftFailed {
            reason: format!(
                "DFlash drafter expected logits shape [1, {block_size}, vocab]; got {shape:?}"
            ),
        });
    }
    let vocab = shape[2];
    let n = block_size - 1;

    let greedy = sampler.temperature == 0.0 || sampler.top_k == 1;
    if greedy {
        // Greedy DFlash is the hot server path. Argmax all masked proposal
        // positions in one op and copy all token ids at once. This removes
        // the per-position slice/eval/item synchronization loop that made the
        // Qwen3.5-4B DFlash path slower than baseline for short generations
        let masked_logits = ffi::slice(
            logits,
            &[0_i32, 1_i32, 0_i32],
            &[1_i32, block_size as i32, vocab],
        );
        let argmax = ffi::argmax_last_axis(&masked_logits);
        return Ok(super::materialize_argmax_i32_vec(&argmax, n));
    }

    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        // Row `i + 1` of the [1, L, V] logits.
        let row_idx = (i + 1) as i32;
        let row = ffi::slice(
            logits,
            &[0_i32, row_idx, 0_i32],
            &[1_i32, row_idx + 1, vocab],
        );
        // Drop the seq axis so we get a `[1, vocab]` 2D slice (fused_sample
        // / argmax expect `[batch, vocab]`).
        let row = ffi::reshape(&row, &[1_i32, vocab]);
        let token = ffi::fused_sample(
            &row,
            sampler.temperature,
            sampler.top_k,
            sampler.top_p,
            sampler.min_p,
        );
        ffi::eval(&token);
        out.push(ffi::item_i32(&token));
    }
    Ok(out)
}

/// Device-side variant of [`sample_block_per_position`].
///
/// Greedy DFlash keeps the whole proposal vector as an MLX array so the
/// round loop can concatenate it into the target verify input without first
/// copying token ids back to the host. This mirrors upstream mlx-vlm's
/// `draft_tokens = draft_model.draft_block(...); verify_input =
/// mx.concatenate([bonus, draft_tokens], axis=1)` pipeline. Stochastic
/// sampling falls back to the scalar helper because stochastic DFlash parity
/// is outside the hot path optimized.
///
/// Used by: `DFlashDrafter::draft_block_array`,
/// `MuseAssistantDrafter::draft_block` / `draft_block_array` (issue #1343).
pub(crate) fn sample_block_per_position_array(
    logits: &MlxArray,
    block_size: usize,
    sampler: &SamplingConfig,
) -> Result<UniquePtr<MlxArray>, DrafterError> {
    let shape = ffi::array_shape(logits);
    if shape.len() != 3 || shape[0] != 1 || shape[1] != block_size as i32 {
        return Err(DrafterError::DraftFailed {
            reason: format!(
                "DFlash drafter expected logits shape [1, {block_size}, vocab]; got {shape:?}"
            ),
        });
    }
    let vocab = shape[2];

    let greedy = sampler.temperature == 0.0 || sampler.top_k == 1;
    if greedy {
        let masked_logits = ffi::slice(
            logits,
            &[0_i32, 1_i32, 0_i32],
            &[1_i32, block_size as i32, vocab],
        );
        return Ok(ffi::argmax_last_axis(&masked_logits));
    }

    let tokens = sample_block_per_position(logits, block_size, sampler)?;
    Ok(ffi::from_slice_i32(&tokens, &[1, (block_size - 1) as i32]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;
    use crate::ffi;

    #[test]
    fn convert_bf16_skips_quantization_auxiliaries() {
        let mut weights: WeightMap = std::collections::HashMap::new();
        // A regular bf16 weight that SHOULD be converted.
        weights.insert(
            "embed_tokens.weight".to_string(),
            ffi::zeros(&[4, 4], dtype::BFLOAT16),
        );
        // A `.scales` aux that SHOULD NOT be converted (quantized_matmul
        // handles bf16 scales natively).
        weights.insert(
            "layers.0.self_attn.q_proj.scales".to_string(),
            ffi::zeros(&[4, 4], dtype::BFLOAT16),
        );
        // A `.biases` aux: also skip.
        weights.insert(
            "layers.0.self_attn.q_proj.biases".to_string(),
            ffi::zeros(&[4, 4], dtype::BFLOAT16),
        );
        // A non-bf16 tensor: should pass through.
        weights.insert("fc.weight".to_string(), ffi::zeros(&[4, 4], dtype::FLOAT16));

        convert_bf16_to_f16_non_quantized(&mut weights);

        assert_eq!(
            ffi::array_dtype(weights.get("embed_tokens.weight").unwrap()),
            dtype::FLOAT16,
            "embed_tokens.weight must be converted to f16"
        );
        assert_eq!(
            ffi::array_dtype(weights.get("layers.0.self_attn.q_proj.scales").unwrap()),
            dtype::BFLOAT16,
            "scales aux must NOT be converted"
        );
        assert_eq!(
            ffi::array_dtype(weights.get("layers.0.self_attn.q_proj.biases").unwrap()),
            dtype::BFLOAT16,
            "biases aux must NOT be converted"
        );
        assert_eq!(
            ffi::array_dtype(weights.get("fc.weight").unwrap()),
            dtype::FLOAT16,
            "non-bf16 tensor must pass through unchanged"
        );
    }

    #[test]
    fn drafter_dtype_policy_mirrors_target_loaders() {
        use super::{EnvFlag, drafter_bf16_to_f16_policy as policy};
        // Ampere and later CUDA keeps bf16 (the target does too), unless the
        // operator opts into f16 normalization.
        assert!(!policy(false, false, Some((12, 1)), EnvFlag::Unset));
        assert!(!policy(false, false, Some((8, 0)), EnvFlag::Unset));
        assert!(policy(false, false, Some((12, 1)), EnvFlag::Enabled));
        assert!(!policy(false, false, Some((12, 1)), EnvFlag::Disabled));
        // Pre-Ampere CUDA converts unless normalization is disabled.
        assert!(policy(false, false, Some((7, 0)), EnvFlag::Unset));
        assert!(policy(false, false, Some((7, 5)), EnvFlag::Enabled));
        assert!(!policy(false, false, Some((7, 0)), EnvFlag::Disabled));
        // Apple Silicon converts.
        assert!(policy(false, true, None, EnvFlag::Unset));
        // CPU-only builds keep bf16.
        assert!(!policy(false, false, None, EnvFlag::Unset));
        // MLXCEL_KEEP_BF16 wins everywhere.
        assert!(!policy(true, true, None, EnvFlag::Unset));
        assert!(!policy(true, false, Some((7, 0)), EnvFlag::Enabled));
    }

    #[test]
    fn env_flag_parsing_matches_binary_spelling() {
        use super::EnvFlag;
        assert_eq!(EnvFlag::parse(""), EnvFlag::Unset);
        assert_eq!(EnvFlag::parse("  "), EnvFlag::Unset);
        assert_eq!(EnvFlag::parse("1"), EnvFlag::Enabled);
        assert_eq!(EnvFlag::parse("true"), EnvFlag::Enabled);
        assert_eq!(EnvFlag::parse("0"), EnvFlag::Disabled);
        assert_eq!(EnvFlag::parse("OFF"), EnvFlag::Disabled);
        assert_eq!(EnvFlag::parse("no"), EnvFlag::Disabled);
        assert_eq!(EnvFlag::parse("False"), EnvFlag::Disabled);
    }

    #[test]
    fn convert_bf16_no_op_on_already_f16_weights() {
        let mut weights: WeightMap = std::collections::HashMap::new();
        weights.insert("a".to_string(), ffi::zeros(&[2, 2], dtype::FLOAT16));
        weights.insert("b".to_string(), ffi::zeros(&[2, 2], dtype::FLOAT32));

        convert_bf16_to_f16_non_quantized(&mut weights);

        assert_eq!(
            ffi::array_dtype(weights.get("a").unwrap()),
            dtype::FLOAT16,
            "f16 must remain f16"
        );
        assert_eq!(
            ffi::array_dtype(weights.get("b").unwrap()),
            dtype::FLOAT32,
            "f32 must remain f32"
        );
    }

    /// The published `LFM2.5-2.6B-DSpark` pairing shape (issue #1339).
    fn dspark_config() -> DFlashConfig {
        DFlashConfig {
            hidden_size: 2048,
            vocab_size: 128_000,
            block_size: 9,
            mask_token_id: 125_017,
            markov_rank: 256,
            target_layer_ids: vec![2, 9, 17, 21, 27],
            num_target_layers: 30,
            ..DFlashConfig::default()
        }
    }

    /// The structural half of the DSpark pairing gate accepts the published
    /// config against its own `fc` width and names each way it can be wrong.
    #[test]
    fn dspark_config_pairing_gate_checks_layer_ids_and_fc_width() {
        let config = dspark_config();
        // 5 captured layers x 2048 hidden = the fc input width.
        assert!(dspark_config_pairing_error(&config, 5 * 2048).is_none());

        let err = dspark_config_pairing_error(&config, 4 * 2048).expect("fc width mismatch");
        assert!(err.contains("fc projection reads 8192"), "{err}");

        let unsorted = DFlashConfig {
            target_layer_ids: vec![2, 17, 9, 21, 27],
            ..dspark_config()
        };
        let err = dspark_config_pairing_error(&unsorted, 5 * 2048).expect("unsorted ids");
        assert!(err.contains("strictly increasing"), "{err}");

        let past_end = DFlashConfig {
            target_layer_ids: vec![2, 9, 17, 21, 30],
            ..dspark_config()
        };
        let err = dspark_config_pairing_error(&past_end, 5 * 2048).expect("id past the stack");
        assert!(err.contains("reach past num_target_layers = 30"), "{err}");

        let empty = DFlashConfig {
            target_layer_ids: vec![],
            ..dspark_config()
        };
        assert!(dspark_config_pairing_error(&empty, 0).is_some());
    }

    /// The same gate bounds the two checkpoint-supplied numbers that index a
    /// buffer or size the round: a mask id outside the vocabulary would be
    /// gathered from the target's embedding table (MLX range-checks no
    /// positive gather index), and a verify width below two rows proposes
    /// nothing.
    #[test]
    fn dspark_config_pairing_gate_bounds_the_mask_id_and_the_verify_width() {
        let past_vocab = DFlashConfig {
            mask_token_id: 128_000,
            ..dspark_config()
        };
        let err = dspark_config_pairing_error(&past_vocab, 5 * 2048).expect("mask id past vocab");
        assert!(err.contains("mask_token_id 128000"), "{err}");

        let negative = DFlashConfig {
            mask_token_id: -1,
            ..dspark_config()
        };
        assert!(dspark_config_pairing_error(&negative, 5 * 2048).is_some());

        let degenerate = DFlashConfig {
            block_size: 0,
            ..dspark_config()
        };
        let err = dspark_config_pairing_error(&degenerate, 5 * 2048).expect("degenerate width");
        assert!(err.contains("verify row(s)"), "{err}");

        let clamped_low = DFlashConfig {
            runtime_block_size: Some(1),
            ..dspark_config()
        };
        assert!(dspark_config_pairing_error(&clamped_low, 5 * 2048).is_some());
    }

    /// The verify width is the one checkpoint-supplied number that becomes a
    /// server-wide control input: `resolve_draft_block_size` peeks the drafter
    /// config and hands `runtime_verify_width()` to the scheduler, which then
    /// reaches the block-versus-chain exactness probe. That probe runs one
    /// target forward per row on the scheduler thread and memoizes its
    /// verdict, so an absurd width there is a hang and not a slow request, and
    /// it runs BEFORE this gate. Both halves of the answer are pinned: the
    /// clamp, which is what actually holds ahead of the gate, and the gate's
    /// refusal, which is what tells the operator.
    #[test]
    fn a_dspark_config_cannot_put_an_absurd_verify_width_into_effect() {
        let absurd = DFlashConfig {
            block_size: 1_000_000,
            runtime_block_size: Some(1_000_000),
            ..dspark_config()
        };
        assert_eq!(absurd.requested_verify_width(), 1_000_000);
        assert_eq!(absurd.runtime_verify_width(), DSPARK_MAX_VERIFY_WIDTH);
        let err = dspark_config_pairing_error(&absurd, 5 * 2048).expect("absurd width");
        assert!(err.contains("above the"), "{err}");

        // `usize::MAX` is the case `verify_width`'s saturating add exists for:
        // it must not wrap to a zero-row width, and it must not escape the
        // ceiling either.
        let saturating = DFlashConfig {
            block_size: usize::MAX,
            runtime_block_size: Some(usize::MAX),
            ..dspark_config()
        };
        assert_eq!(saturating.runtime_verify_width(), DSPARK_MAX_VERIFY_WIDTH);
        assert!(dspark_config_pairing_error(&saturating, 5 * 2048).is_some());

        // The published width is untouched by the ceiling and still admitted.
        let published = dspark_config();
        assert_eq!(published.requested_verify_width(), 8);
        assert_eq!(published.runtime_verify_width(), 8);
        assert!(dspark_config_pairing_error(&published, 5 * 2048).is_none());
    }

    /// The target half rejects a target whose hidden size, layer count or
    /// tied-head vocabulary does not match the drafter, and skips the
    /// vocabulary check when the target hands out no embedding module.
    #[test]
    fn dspark_target_pairing_gate_rejects_a_mismatched_target() {
        let config = dspark_config();
        assert!(dspark_target_pairing_error(&config, 2048, 30, Some(128_000)).is_none());

        let err = dspark_target_pairing_error(&config, 4096, 30, Some(128_000))
            .expect("hidden size mismatch");
        assert!(err.contains("target's hidden size = 4096"), "{err}");

        let err = dspark_target_pairing_error(&config, 2048, 24, Some(128_000))
            .expect("layer count mismatch");
        assert!(err.contains("the target has 24 layers"), "{err}");

        // The local `lfm2-8b-a1b-4bit` checkpoint pairs this way: right
        // hidden size, right layer count for its own drafter, wrong
        // vocabulary for the Markov transition table.
        let err = dspark_target_pairing_error(&config, 2048, 30, Some(65_536))
            .expect("vocabulary mismatch");
        assert!(err.contains("emits 65536 logits"), "{err}");

        assert!(
            dspark_target_pairing_error(&config, 2048, 30, None).is_none(),
            "an unmeasurable vocabulary must not be guessed at"
        );
    }

    /// The trait conformance check: a `DFlashDrafter` must be
    /// usable as `Box<dyn Drafter>` (object-safe behind the trait).
    #[test]
    fn dflash_drafter_is_object_safe() {
        // We cannot construct a real DFlashDrafter without a model on disk,
        // but we *can* assert the trait dispatch works by way of a
        // compile-time cast on a stub. The cast itself is the check.
        fn _assert_object_safe(d: Box<dyn Drafter>) -> DrafterKind {
            d.kind()
        }
    }
}
