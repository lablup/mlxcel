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

use crate::cache::KVCache;
use crate::drafter::{Drafter, DrafterError, DrafterKind};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::weights::WeightMap;
use cxx::UniquePtr;
use std::path::Path;

use super::config::DFlashConfig;
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
    /// 4. Convert bf16 → f16 on all non-quantized tensors (Apple
    ///    Silicon precision rules; see `docs/apple-silicon-precision.md`).
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

        // Apple Silicon precision: convert bf16 → f16 on non-quantized
        // tensors. Quantized tensors keep their bf16 scales/biases as-is
        // because `quantized_matmul` handles bf16 natively.
        convert_bf16_to_f16_non_quantized(&mut weights);

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

/// Convert every bf16 tensor in `weights` to f16. Quantized scales and
/// biases (recognised by living next to a `.scales` or `.biases` key in
/// the map) are kept as-is.
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
fn convert_bf16_to_f16_non_quantized(weights: &mut WeightMap) {
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

impl Drafter for DFlashDrafter {
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
                         hands out its embedding table (the Qwen 3.5 family \
                         does — check the target is a Qwen 3.5 checkpoint) \
                         (kind = {})",
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
fn sample_block_per_position_array(
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
