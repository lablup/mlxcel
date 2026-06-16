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

//! `DFlashDraftModel` — the assembled 5-layer Qwen 3.5 DFlash drafter.
//!
//! Builds on the [`super::attention::DFlashAttention`],
//! [`super::layer::DFlashDecoderLayer`], and [`super::mlp::DFlashMlp`]
//! blocks ported in earlier commits.
//!
//! Apple Silicon precision rules (see `docs/apple-silicon-precision.md`):
//! - Hidden, K, V, and projected proposal/context tensors stay in bf16/f16.
//! - No `f32` promotion in the forward path.
//! - Weights remain bf16 or quantized as-loaded; the non-quantized bf16 →
//!   f16 conversion happens in the binary's `load_text_weights`
//!   path that ultimately hands the [`WeightMap`] to
//!   [`DFlashDraftModel::from_weights`].

use crate::cache::KVCache;
use crate::ffi::{self, MlxArray};
use crate::layers::{Linear, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::config::DFlashConfig;
use super::layer::DFlashDecoderLayer;

/// LM head selector: own checkpoint weights, embedding-tied projection, or
/// target-bound untied projection.
///
/// - [`LmHead::Own`] — drafter checkpoint shipped a dedicated
///   `lm_head.weight`. Used as a regular Linear.
/// - [`LmHead::Tied`] — drafter uses `embed_tokens.as_linear` for the
///   final projection. This is the published `z-lab/Qwen3.5-4B-DFlash`
///   default (`tie_word_embeddings = true`).
/// - [`LmHead::TargetBound`] — drafter config says the head is untied but
///   the checkpoint omits `lm_head.weight`; upstream Python binds the
///   target model's `lm_head` at runtime for this shape (for example
///   `z-lab/Qwen3.5-27B-DFlash`).
enum LmHead {
    Own(Linear),
    Tied,
    TargetBound(Option<UnifiedLinear>),
}

/// Qwen 3.5 DFlash drafter (`DFlashDraftModel`).
///
/// The end-to-end forward consumes a `[B, L]` int32 token sequence (the
/// proposal block) and a `[B, T, num_target_layers * hidden_size]`
/// target-hidden buffer, producing `[B, L, vocab_size]` logits.
///
/// One [`KVCache`] per drafter layer is owned by the caller; the
/// drafter writes only the context-side K/V into each cache via
/// [`super::attention::DFlashAttention::forward`].
pub struct DFlashDraftModel {
    pub config: DFlashConfig,

    /// Token embedding table.
    ///
    /// **Lazy-bind tombstone.** The upstream `z-lab/Qwen3.5-4B-DFlash`
    /// checkpoint does NOT ship `embed_tokens.weight` — upstream Python
    /// sets `self.embed_tokens = None` at construction and binds it to
    /// the target's `embed_tokens` later via `bind()`
    /// (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/qwen3_dflash/dflash.py,
    /// lines 88, 92-108). This field mirrors that shape:
    ///
    /// - `Some(_)` — a drafter checkpoint that *did* ship its own
    ///   `embed_tokens.weight` (used both as the input embedder and, via
    ///   [`LmHead::Tied`], as the LM head when `tie_word_embeddings = true`).
    /// - `None` — the published lazy-bind checkpoint. [`Self::bind_target_embedding`]
    ///   installs a shared-buffer handle to the *target's* embedding
    ///   during the drafter's `bind()` call, before the first
    ///   [`Self::forward`].
    ///
    /// [`Self::forward`] panics if this is still `None` at call time —
    /// the round-loop driver always `bind()`s before `draft_block`, so a
    /// `None` here at forward time is a wiring bug, not a recoverable
    /// runtime state.
    pub embed_tokens: Option<UnifiedEmbedding>,

    /// Per-target-layer projection: maps the
    /// `5 * hidden_size`-dim concatenation to `hidden_size`.
    /// Bias-free per upstream.
    pub fc: Linear,

    /// Norm over the projected target hidden (before consumption by
    /// every decoder layer).
    pub hidden_norm: RMSNorm,

    /// Five drafter transformer layers.
    pub layers: Vec<DFlashDecoderLayer>,

    /// Final RMSNorm before the LM head.
    pub norm: RMSNorm,

    /// LM head dispatch: own, tied, or target-bound untied.
    lm_head: LmHead,
}

impl DFlashDraftModel {
    /// Construct the drafter from a sanitized weight map.
    ///
    /// `weights` must already have the `model.` prefix stripped (see
    /// [`Self::sanitize`]). The drafter is dtype-agnostic — caller
    /// converts bf16 → f16 in the load pipeline per Apple Silicon
    /// precision rules.
    pub fn from_weights(weights: &WeightMap, config: DFlashConfig) -> Result<Self, String> {
        // DFlash uses non-quantized weights in the published checkpoint;
        // but `UnifiedLinear::from_weights` auto-detects quantization so
        // a future quantized DFlash will Just Work without further changes.
        // The group_size/bits fallback only matters if the checkpoint
        // *is* quantized; for the published bf16 checkpoint these
        // values are dead code.
        let group_size = 64;
        let bits = 4;

        // Lazy-bind tombstone: the published `z-lab/Qwen3.5-4B-DFlash`
        // checkpoint omits `embed_tokens.weight` (and its `.scales` for a
        // hypothetical quantized variant). When the embedding is absent
        // from the index we construct the model with `embed_tokens = None`
        // and resolve it from the target during `DFlashDrafter::bind` via
        // [`Self::bind_target_embedding`], mirroring upstream Python's
        // `self.embed_tokens = None` construction shape. A drafter
        // checkpoint that *does* ship its own table loads it eagerly here
        // (no regression for self-contained DFlash checkpoints).
        let has_embed_tokens = weights.contains_key("embed_tokens.weight")
            || weights.contains_key("embed_tokens.scales");
        let embed_tokens = if has_embed_tokens {
            Some(UnifiedEmbedding::from_weights(
                weights,
                "embed_tokens",
                group_size,
                bits,
            )?)
        } else {
            None
        };
        let fc = Linear::from_weights(weights, "fc")?;

        let hidden_norm_w = weights
            .get("hidden_norm.weight")
            .map(|w| ffi::copy(w))
            .ok_or_else(|| "Weight not found: hidden_norm.weight".to_string())?;
        let hidden_norm = RMSNorm::new(hidden_norm_w, config.rms_norm_eps);

        let norm_w = weights
            .get("norm.weight")
            .map(|w| ffi::copy(w))
            .ok_or_else(|| "Weight not found: norm.weight".to_string())?;
        let norm = RMSNorm::new(norm_w, config.rms_norm_eps);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("layers.{i}");
            layers.push(DFlashDecoderLayer::from_weights(
                weights, &prefix, &config, group_size, bits,
            )?);
        }

        // LM head: present iff the checkpoint ships `lm_head.weight`.
        // `tie_word_embeddings = true` (the 4B default) means no
        // `lm_head.weight` and we route through `embed_tokens.as_linear`
        // instead. Some official DFlash checkpoints (notably 27B) set
        // `tie_word_embeddings = false` while still omitting `lm_head.weight`;
        // upstream Python resolves that by binding the target model's
        // explicit `lm_head` during `bind()`, with an embedding-tied fallback
        // if the target has no separate head.
        let lm_head = if weights.contains_key("lm_head.weight") {
            LmHead::Own(Linear::from_weights(weights, "lm_head")?)
        } else if config.tie_word_embeddings {
            LmHead::Tied
        } else {
            LmHead::TargetBound(None)
        };

        Ok(Self {
            config,
            embed_tokens,
            fc,
            hidden_norm,
            layers,
            norm,
            lm_head,
        })
    }

    /// Install the target's embedding table into the lazy-bind tombstone.
    ///
    /// Called from [`super::drafter::DFlashDrafter::bind`] when the drafter
    /// checkpoint omitted its own `embed_tokens.weight` (the published
    /// `z-lab/Qwen3.5-4B-DFlash` shape). `embed` is a shared-buffer handle
    /// to the target's `UnifiedEmbedding` (via
    /// [`crate::layers::UnifiedEmbedding::clone_shared`] — lazy-array
    /// share, no element copy), obtained through
    /// [`crate::generate::LanguageModel::embed_tokens_module`].
    ///
    /// Mirrors upstream Python's `bind()` lazy assignment
    /// (`self.embed_tokens = target.embed_tokens`). Idempotent: re-binding
    /// (e.g. on `reset()` between generation calls) simply overwrites the
    /// handle with a fresh shared view, which is harmless because the
    /// underlying buffer is identical.
    ///
    /// A drafter that *did* ship its own `embed_tokens.weight` keeps its
    /// own table — `DFlashDrafter::bind` only calls this when
    /// [`Self::needs_embed_binding`] is `true`.
    pub fn bind_target_embedding(&mut self, embed: UnifiedEmbedding) {
        self.embed_tokens = Some(embed);
    }

    /// Install the target's untied LM head into an `lm_head.weight` tombstone.
    ///
    /// This mirrors the same upstream `bind()` contract as the embedding
    /// table, but applies only to DFlash checkpoints whose config says
    /// `tie_word_embeddings = false` while the checkpoint itself omits
    /// `lm_head.weight` (for example `z-lab/Qwen3.5-27B-DFlash`). Passing
    /// `None` intentionally resolves to the upstream fallback path: use the
    /// bound embedding table as a tied projection if the target does not
    /// expose an explicit output head.
    pub fn bind_target_lm_head(&mut self, lm_head: Option<UnifiedLinear>) {
        if matches!(self.lm_head, LmHead::TargetBound(_)) {
            self.lm_head = match lm_head {
                Some(linear) => LmHead::TargetBound(Some(linear)),
                None => LmHead::Tied,
            };
        }
    }

    /// Whether this drafter still needs its `embed_tokens` table resolved
    /// from the target.
    ///
    /// `true` for the published lazy-bind checkpoint until
    /// [`Self::bind_target_embedding`] runs; `false` for a self-contained
    /// drafter checkpoint that shipped its own `embed_tokens.weight`.
    pub fn needs_embed_binding(&self) -> bool {
        self.embed_tokens.is_none()
    }

    /// Whether this drafter still needs an untied target `lm_head` binding.
    ///
    /// `true` for checkpoints such as `z-lab/Qwen3.5-27B-DFlash` until
    /// [`Self::bind_target_lm_head`] runs; `false` for checkpoints that ship
    /// their own head or intentionally tie the head to the embedding table.
    pub fn needs_lm_head_binding(&self) -> bool {
        matches!(self.lm_head, LmHead::TargetBound(None))
    }

    /// Borrow the resolved embedding table.
    ///
    /// # Panics
    ///
    /// Panics if the embedding has not been bound yet (lazy-bind
    /// checkpoint whose `bind()` has not run). The round-loop driver
    /// always `bind()`s before `draft_block`, so a panic here is a wiring
    /// bug rather than a recoverable runtime condition — the message
    /// names the fix.
    fn embed(&self) -> &UnifiedEmbedding {
        self.embed_tokens.as_ref().expect(
            "DFlash drafter embed_tokens is not bound — the published \
             z-lab/Qwen3.5-4B-DFlash checkpoint omits embed_tokens.weight \
             and the drafter must resolve it from the target via \
             DFlashDrafter::bind() before the first forward()",
        )
    }

    /// Construct fresh per-layer K/V caches, one per drafter layer.
    ///
    /// Each cache is an FP16 [`KVCache`] (the default mode). Mirrors
    /// upstream `make_cache(self) -> List[KVCache]`.
    pub fn make_cache(&self) -> Vec<KVCache> {
        (0..self.config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect()
    }

    /// End-to-end forward.
    ///
    /// Shapes (matching upstream `__call__`):
    ///
    /// - `inputs`        — `[B, L]`, dtype int32 (token ids in the
    ///   proposal block, with `mask_token_id` at positions `1..L`).
    /// - `target_hidden` — `[B, T, num_target_layers * hidden_size]`,
    ///   the multi-layer concatenation produced by the Qwen 3.5 target
    ///   hooks (`return_hidden=true`, `capture_layer_ids=[1,8,15,22,29]`).
    /// - `cache` — per-layer drafter K/V caches; length must equal
    ///   `config.num_hidden_layers`.
    ///
    /// Returns logits of shape `[B, L, vocab_size]`.
    pub fn forward(
        &self,
        inputs: &MlxArray,
        target_hidden: &MlxArray,
        cache: &mut [KVCache],
    ) -> UniquePtr<MlxArray> {
        debug_assert_eq!(
            cache.len(),
            self.layers.len(),
            "DFlashDraftModel::forward: cache length {} does not match num_hidden_layers {}",
            cache.len(),
            self.layers.len()
        );

        // h = embed_tokens(inputs)  →  [B, L, hidden_size]
        // `embed()` panics with a fix-naming message if the lazy-bind
        // tombstone was never resolved (see `Self::embed`).
        let embed = self.embed();
        let mut h = embed.forward(inputs);

        // h_ctx = hidden_norm(fc(target_hidden))  →  [B, T, hidden_size]
        let fc_out = self.fc.forward(target_hidden);
        let h_ctx = self.hidden_norm.forward(&fc_out);

        // Five decoder layers, threaded through their own cache slot.
        for (layer, c) in self.layers.iter().zip(cache.iter_mut()) {
            h = layer.forward(&h, &h_ctx, c);
        }

        // Final norm + LM head. The `Tied` arm routes through the same
        // (possibly target-bound) embedding table.
        let h = self.norm.forward(&h);
        match &self.lm_head {
            LmHead::Own(linear) => linear.forward(&h),
            LmHead::Tied => embed.as_linear(&h),
            LmHead::TargetBound(Some(linear)) => linear.forward(&h),
            LmHead::TargetBound(None) => embed.as_linear(&h),
        }
    }

    /// Run a single masked-forward draft block and argmax-sample one
    /// token per proposal position.
    ///
    /// Builds `block = [last_bonus, mask_id, mask_id, ..., mask_id]` of
    /// shape `[1, block_size]`, runs the masked forward, slices
    /// `logits[:, 1 - block_size:]` (the last `block_size - 1` rows
    /// corresponding to the masked positions), and argmax-samples each
    /// row.
    ///
    /// Returns `Vec<i32>` of length `block_size - 1`.
    ///
    /// This is the B = 1 variant; the trait surface uses `B = 1` for the
    /// classic single-stream draft loop. A batched variant lives behind a
    /// future-facing helper if/when batched DFlash drafting lands.
    pub fn draft_block(
        &self,
        last_bonus: i32,
        target_hidden: &MlxArray,
        cache: &mut [KVCache],
        block_size: usize,
    ) -> Vec<i32> {
        assert!(
            block_size >= 2,
            "DFlash draft_block requires block_size >= 2 (got {block_size})",
        );

        let mask_id = self.config.mask_token_id;
        let mut block: Vec<i32> = Vec::with_capacity(block_size);
        block.push(last_bonus);
        for _ in 1..block_size {
            block.push(mask_id);
        }
        let inputs = ffi::from_slice_i32(&block, &[1, block_size as i32]);

        let logits = self.forward(&inputs, target_hidden, cache);

        // Slice [B=1, L=block_size, vocab] → [1, block_size-1, vocab]
        // along axis 1 (rows `[1..block_size]`).
        let shape = ffi::array_shape(&logits);
        debug_assert_eq!(shape.len(), 3, "logits must be [B, L, V], got {shape:?}");
        let vocab = shape[2];
        let starts = [0_i32, 1_i32, 0_i32];
        let stops = [1_i32, block_size as i32, vocab];
        let slice = ffi::slice(&logits, &starts, &stops);

        // Argmax over the vocab axis (axis=2 → keepdims=false → [1, L-1]).
        let argmax = ffi::argmax(&slice, 2, false);
        super::materialize_argmax_i32_vec(&argmax, block_size - 1)
    }

    /// Strip the `model.` prefix from any weight key carrying it, matching
    /// upstream `sanitize` exactly:
    ///
    /// ```python
    /// def sanitize(self, weights: dict) -> dict:
    ///     out = {}
    ///     for k, v in weights.items():
    ///         if k.startswith("model."):
    ///             k = k[len("model."):]
    ///         out[k] = v
    ///     return out
    /// ```
    ///
    /// This is a borrow-friendly variant: it mutates the existing weight
    /// map in place rather than allocating a new HashMap, because the
    /// values are `UniquePtr<MlxArray>` and cloning them defeats the
    /// MLX lazy-array sharing.
    pub fn sanitize(weights: &mut WeightMap) {
        const PREFIX: &str = "model.";

        let renames: Vec<(String, String)> = weights
            .keys()
            .filter(|k| k.starts_with(PREFIX))
            .map(|k| (k.clone(), k[PREFIX.len()..].to_string()))
            .collect();

        for (old, new) in renames {
            if let Some(v) = weights.remove(&old) {
                weights.insert(new, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;
    use crate::ffi;

    /// Pure compile-time check: enum variants used in `forward`.
    /// If a future patch adds a third variant without updating the match
    /// in `forward`, the corresponding match arm will fail to compile.
    #[test]
    fn lm_head_match_arms_are_exhaustive_compile_time() {
        fn _exhaustive(h: &LmHead) -> &'static str {
            match h {
                LmHead::Own(_) => "own",
                LmHead::Tied => "tied",
                LmHead::TargetBound(_) => "target-bound",
            }
        }
    }

    /// Build a minimal `DFlashConfig` with a single decoder layer and tiny
    /// dimensions so `from_weights` is cheap to drive in a unit test.
    fn tiny_config() -> DFlashConfig {
        DFlashConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            vocab_size: 8,
            num_target_layers: 4,
            target_layer_ids: vec![0, 1, 2, 3],
            ..DFlashConfig::default()
        }
    }

    /// Populate every weight `DFlashDraftModel::from_weights` requires for
    /// [`tiny_config`] *except* `embed_tokens.weight` and `lm_head.weight`
    /// (the caller decides whether to add those). Shapes only have to be
    /// 2D/1D parseable tensors — `from_weights` does construction, not a
    /// forward pass, so the exact values are irrelevant here.
    fn tiny_weights_without_embed() -> WeightMap {
        let mut w: WeightMap = std::collections::HashMap::new();
        // fc: [hidden, num_target_layers * hidden] = [4, 16]
        w.insert(
            "fc.weight".to_string(),
            ffi::zeros(&[4, 16], dtype::FLOAT32),
        );
        w.insert(
            "hidden_norm.weight".to_string(),
            ffi::zeros(&[4], dtype::FLOAT32),
        );
        w.insert("norm.weight".to_string(), ffi::zeros(&[4], dtype::FLOAT32));
        // Layer 0 projections. q out = n_heads*head_dim = 4; k/v out =
        // n_kv_heads*head_dim = 2; o in = 4.
        w.insert(
            "layers.0.self_attn.q_proj.weight".to_string(),
            ffi::zeros(&[4, 4], dtype::FLOAT32),
        );
        w.insert(
            "layers.0.self_attn.k_proj.weight".to_string(),
            ffi::zeros(&[2, 4], dtype::FLOAT32),
        );
        w.insert(
            "layers.0.self_attn.v_proj.weight".to_string(),
            ffi::zeros(&[2, 4], dtype::FLOAT32),
        );
        w.insert(
            "layers.0.self_attn.o_proj.weight".to_string(),
            ffi::zeros(&[4, 4], dtype::FLOAT32),
        );
        w.insert(
            "layers.0.self_attn.q_norm.weight".to_string(),
            ffi::zeros(&[2], dtype::FLOAT32),
        );
        w.insert(
            "layers.0.self_attn.k_norm.weight".to_string(),
            ffi::zeros(&[2], dtype::FLOAT32),
        );
        w.insert(
            "layers.0.mlp.gate_proj.weight".to_string(),
            ffi::zeros(&[8, 4], dtype::FLOAT32),
        );
        w.insert(
            "layers.0.mlp.up_proj.weight".to_string(),
            ffi::zeros(&[8, 4], dtype::FLOAT32),
        );
        w.insert(
            "layers.0.mlp.down_proj.weight".to_string(),
            ffi::zeros(&[4, 8], dtype::FLOAT32),
        );
        w.insert(
            "layers.0.input_layernorm.weight".to_string(),
            ffi::zeros(&[4], dtype::FLOAT32),
        );
        w.insert(
            "layers.0.post_attention_layernorm.weight".to_string(),
            ffi::zeros(&[4], dtype::FLOAT32),
        );
        w
    }

    /// The published `z-lab/Qwen3.5-4B-DFlash` shape: no `embed_tokens.weight`
    /// in the index, `tie_word_embeddings = true`, no `lm_head.weight`.
    /// `from_weights` must succeed (no `LoadFailed`) and leave the model
    /// with an `embed_tokens = None` lazy-bind tombstone.
    #[test]
    fn from_weights_builds_tombstone_when_embed_tokens_absent() {
        let weights = tiny_weights_without_embed();
        let model = DFlashDraftModel::from_weights(&weights, tiny_config())
            .expect("lazy-bind checkpoint (no embed_tokens.weight) must still construct");
        assert!(
            model.embed_tokens.is_none(),
            "embed_tokens must be a None tombstone when the index omits the table",
        );
        assert!(
            model.needs_embed_binding(),
            "needs_embed_binding() must report true before bind()",
        );
    }

    /// The published `z-lab/Qwen3.5-27B-DFlash` shape: no
    /// `embed_tokens.weight`, no `lm_head.weight`, and
    /// `tie_word_embeddings = false`. Construction must succeed and leave
    /// both target-owned modules as lazy-bind tombstones.
    #[test]
    fn from_weights_allows_target_bound_lm_head_when_untied_head_missing() {
        let weights = tiny_weights_without_embed();
        let mut config = tiny_config();
        config.tie_word_embeddings = false;

        let model = DFlashDraftModel::from_weights(&weights, config)
            .expect("untied lazy-bind checkpoint without lm_head.weight must construct");

        assert!(
            model.needs_embed_binding(),
            "embed_tokens must still be resolved from the target",
        );
        assert!(
            model.needs_lm_head_binding(),
            "untied missing lm_head.weight must request target lm_head binding",
        );
    }

    /// A self-contained DFlash checkpoint that *does* ship its own
    /// `embed_tokens.weight` must eager-load it — no tombstone, no
    /// regression to the earlier behavior.
    #[test]
    fn from_weights_eager_loads_when_embed_tokens_present() {
        let mut weights = tiny_weights_without_embed();
        // embed_tokens: [vocab, hidden] = [8, 4]
        weights.insert(
            "embed_tokens.weight".to_string(),
            ffi::zeros(&[8, 4], dtype::FLOAT32),
        );
        let model = DFlashDraftModel::from_weights(&weights, tiny_config())
            .expect("self-contained checkpoint must construct");
        assert!(
            model.embed_tokens.is_some(),
            "embed_tokens must be eager-loaded when the index ships the table",
        );
        assert!(
            !model.needs_embed_binding(),
            "needs_embed_binding() must report false for a self-contained checkpoint",
        );
    }

    /// `bind_target_embedding` installs the resolved embedding into the
    /// tombstone, flipping `needs_embed_binding()` to `false`. This is the
    /// lazy-bind assignment the DFlash drafter performs in `bind()`.
    #[test]
    fn bind_target_embedding_resolves_the_tombstone() {
        let weights = tiny_weights_without_embed();
        let mut model = DFlashDraftModel::from_weights(&weights, tiny_config())
            .expect("lazy-bind checkpoint must construct");
        assert!(model.needs_embed_binding());

        // Stand-in for the target's embedding table — a regular
        // [vocab, hidden] tensor, the same shape Qwen 3.5 hands out.
        let target_embed = crate::layers::UnifiedEmbedding::Regular(crate::layers::Embedding::new(
            ffi::zeros(&[8, 4], dtype::FLOAT32),
        ));
        model.bind_target_embedding(target_embed);

        assert!(
            model.embed_tokens.is_some(),
            "bind_target_embedding must install the resolved embedding",
        );
        assert!(
            !model.needs_embed_binding(),
            "needs_embed_binding() must report false after bind_target_embedding",
        );
    }

    /// `bind_target_lm_head` installs the target's untied output projection
    /// into a 27B-style lazy-bind tombstone.
    #[test]
    fn bind_target_lm_head_resolves_the_tombstone() {
        let weights = tiny_weights_without_embed();
        let mut config = tiny_config();
        config.tie_word_embeddings = false;
        let mut model = DFlashDraftModel::from_weights(&weights, config)
            .expect("untied lazy-bind checkpoint must construct");
        assert!(model.needs_lm_head_binding());

        let target_head = crate::layers::UnifiedLinear::Regular(crate::layers::Linear::new(
            ffi::zeros(&[8, 4], dtype::FLOAT32),
            None,
        ));
        model.bind_target_lm_head(Some(target_head));

        assert!(
            !model.needs_lm_head_binding(),
            "needs_lm_head_binding() must report false after bind_target_lm_head",
        );
    }

    /// Sanitize strips the `model.` prefix from keys without losing data.
    #[test]
    fn sanitize_strips_model_prefix() {
        let mut weights: WeightMap = std::collections::HashMap::new();
        // Use a stand-in MLX array (an empty f32 array). We don't care
        // about contents; only the rename semantics matter.
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            ffi::zeros(&[2, 2], dtype::FLOAT32),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            ffi::zeros(&[2, 2], dtype::FLOAT32),
        );
        weights.insert(
            "lm_head.weight".to_string(),
            ffi::zeros(&[2, 2], dtype::FLOAT32),
        );

        DFlashDraftModel::sanitize(&mut weights);

        assert!(weights.contains_key("embed_tokens.weight"));
        assert!(weights.contains_key("layers.0.self_attn.q_proj.weight"));
        assert!(weights.contains_key("lm_head.weight"));
        // The prefixed keys must be gone.
        assert!(!weights.contains_key("model.embed_tokens.weight"));
        assert!(!weights.contains_key("model.layers.0.self_attn.q_proj.weight"));
    }

    /// Sanitize is idempotent: a second call leaves the map unchanged.
    #[test]
    fn sanitize_is_idempotent() {
        let mut weights: WeightMap = std::collections::HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            ffi::zeros(&[2, 2], dtype::FLOAT32),
        );
        weights.insert("norm.weight".to_string(), ffi::zeros(&[2], dtype::FLOAT32));

        DFlashDraftModel::sanitize(&mut weights);
        let after_first: std::collections::HashSet<String> = weights.keys().cloned().collect();

        DFlashDraftModel::sanitize(&mut weights);
        let after_second: std::collections::HashSet<String> = weights.keys().cloned().collect();

        assert_eq!(
            after_first, after_second,
            "sanitize must be idempotent: second pass changed keys"
        );
    }

    /// Sanitize preserves non-prefixed keys untouched.
    #[test]
    fn sanitize_preserves_non_prefixed_keys() {
        let mut weights: WeightMap = std::collections::HashMap::new();
        weights.insert("fc.weight".to_string(), ffi::zeros(&[2, 2], dtype::FLOAT32));
        weights.insert(
            "hidden_norm.weight".to_string(),
            ffi::zeros(&[2], dtype::FLOAT32),
        );

        let initial_len = weights.len();
        DFlashDraftModel::sanitize(&mut weights);
        assert_eq!(weights.len(), initial_len, "no keys should be lost");
        assert!(weights.contains_key("fc.weight"));
        assert!(weights.contains_key("hidden_norm.weight"));
    }

    /// Sanitize handles the `model.lm_head.weight` rename — important
    /// because the LM head may live at either `lm_head.weight` (un-tied)
    /// or `model.lm_head.weight` (some checkpoint conventions).
    #[test]
    fn sanitize_renames_model_lm_head_weight() {
        let mut weights: WeightMap = std::collections::HashMap::new();
        weights.insert(
            "model.lm_head.weight".to_string(),
            ffi::zeros(&[2, 2], dtype::FLOAT32),
        );

        DFlashDraftModel::sanitize(&mut weights);

        assert!(
            weights.contains_key("lm_head.weight"),
            "lm_head.weight must be reachable after sanitize"
        );
        assert!(
            !weights.contains_key("model.lm_head.weight"),
            "the prefixed key must be gone"
        );
    }
}
