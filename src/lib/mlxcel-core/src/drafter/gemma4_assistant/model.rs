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

//! `Gemma4AssistantDraftModel` — Rust port of the Gemma 4 MTP assistant
//! drafter.
//!
//! Mirrors
//! `references/mlx-vlm/mlx_vlm/speculative/drafters/gemma4_assistant/gemma4_assistant.py`.
//!
//! ## Lifecycle
//!
//! 1. **Load.** [`Gemma4AssistantDraftModel::from_path`] parses
//!    `config.json`, loads safetensors weights, sanitises them
//!    (`tie_word_embeddings` handling, `token_ordering` int32 cast), and
//!    constructs the model.
//! 2. **Bind.** [`Gemma4AssistantDraftModel::bind`] picks up the target's
//!    `embed_tokens` and `embed_scale`. Target wrapper depth is resolved
//!    against three known shapes (text-only / mid-wrapper / VLM-wrapped),
//!    matching upstream Python.
//! 3. **Per-block setup.** The round-loop calls
//!    [`Gemma4AssistantDraftModel::set_shared_kv`] with the target's last
//!    full / SWA K/V slabs, the bonus-token absolute position, and an
//!    optional `left_padding` for batched MTP.
//! 4. **Draft block.** [`Gemma4AssistantDraftModel::draft_block`] runs `K`
//!    autoregressive steps, returning `block_size - 1` proposal tokens.

use crate::drafter::gemma4_assistant::config::{DrafterTextConfig, Gemma4AssistantConfig};
use crate::drafter::gemma4_assistant::layer::{DraftDecoderLayer, RopeOffset};
use crate::drafter::masked_embedder::MaskedEmbedder;
use crate::drafter::masks::{make_drafter_masks_with_valid_len, BatchScalar, LayerType};
use crate::drafter::{Drafter, DrafterError, DrafterKind, SharedKv};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::{KVCache, Linear, RMSNorm, UnifiedEmbedding};
use crate::weights::WeightMap;
use cxx::UniquePtr;
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// LM head dispatch
// ---------------------------------------------------------------------------

/// LM head variant resolved at `bind()`-time.
///
/// - `Tied` — use the drafter's `embed_tokens` as a linear projection,
///   matching the upstream `model.embed_tokens.as_linear` path. This is the
///   26B-A4B / 31B drafter case.
/// - `Linear` — explicit `lm_head` with its own `[vocab_size, hidden_size]`
///   weight matrix (when `tie_word_embeddings=False`).
/// - `Centroid` — sparse softmax via `MaskedEmbedder`. Active on
///   `use_ordered_embeddings=True` (E2B / E4B drafters).
enum LmHead {
    Tied,
    Linear(Linear),
    /// Centroid-routed sparse softmax LM head. Used by E2B / E4B drafters
    /// (`use_ordered_embeddings=True`). The `MaskedEmbedder::forward` call
    /// requires the tied embed-tokens weight as its `lm_head_weight` input,
    /// which the drafter's `forward` supplies from `inner.embed_tokens`.
    Centroid(MaskedEmbedder),
}

// ---------------------------------------------------------------------------
// _DraftInner equivalent (mirrors upstream `_DraftInner`)
// ---------------------------------------------------------------------------

/// Drafter inner module — mirrors the upstream `_DraftInner`. Owns the
/// drafter's own `embed_tokens` (used for the tied-dense LM head path) and
/// the `K`-layer transformer stack.
pub(crate) struct DraftInner {
    pub(crate) embed_tokens: UnifiedEmbedding,
    pub(crate) layers: Vec<DraftDecoderLayer>,
    pub(crate) norm: RMSNorm,
}

impl DraftInner {
    fn from_weights(
        weights: &WeightMap,
        config: &DrafterTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            &format!("{prefix}.embed_tokens"),
            config.group_size(),
            config.bits(),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DraftDecoderLayer::from_weights(
                weights,
                config,
                i,
                &format!("{prefix}.layers.{i}"),
            )?);
        }

        let norm = RMSNorm::new(
            weights
                .get(&format!("{prefix}.norm.weight"))
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Weight not found: {prefix}.norm.weight"))?,
            config.rms_norm_eps,
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }
}

// ---------------------------------------------------------------------------
// Shared K/V capture (issue-internal projection of `SharedKv`)
// ---------------------------------------------------------------------------

/// Owned drafter view of the target's shared K/V slabs.
///
/// `SharedKv<'a>` is borrow-typed at the trait boundary to forbid the drafter
/// from mutating target tensors in place. Internally the drafter copies a
/// fresh handle (cheap MLX-array clone — no device memory allocation) and
/// associates each K/V pair with its layer-type key (`"full_attention"` /
/// `"sliding_attention"`). This matches the upstream Python dict layout
/// `shared_kv_states[layer_type] = (K, V)`.
///
/// Until finalises the shape of `SharedKv::tensors`, this struct
/// expects the tensor order `[k_full, v_full, k_swa, v_swa]` documented on
/// [`SharedKv`].
struct OwnedSharedKv {
    full_attention: Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    sliding_attention: Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
}

impl OwnedSharedKv {
    fn from_shared_kv(shared: &SharedKv<'_>) -> Result<Self, DrafterError> {
        // Documented `SharedKv` layout: tensors are ordered
        // `[k_full, v_full, k_swa, v_swa]`. Allow either 2 (full-attention
        // only — Gemma 4 minimal) or 4 (full + SWA — Gemma 4 production)
        // tensors so future Gemma 4 variants without sliding layers don't
        // trip the early validation.
        let owned = match shared.tensors.len() {
            2 => Self {
                full_attention: Some((ffi::copy(shared.tensors[0]), ffi::copy(shared.tensors[1]))),
                sliding_attention: None,
            },
            4 => Self {
                full_attention: Some((ffi::copy(shared.tensors[0]), ffi::copy(shared.tensors[1]))),
                sliding_attention: Some((
                    ffi::copy(shared.tensors[2]),
                    ffi::copy(shared.tensors[3]),
                )),
            },
            n => {
                return Err(DrafterError::SharedKvShape {
                    got: n,
                    expected: &[2, 4],
                });
            }
        };
        Ok(owned)
    }

    /// Batched-MTP-only constructor that runs the per-row left-padding
    /// normalization documented in [`crate::drafter::masks::normalize_batched_shared_kv_states`]
    /// before storing.
    ///
    /// Mirrors upstream Python's `_batch_cache_left_padding`-then-store
    /// shape: the drafter receives the target's shared K/V slabs as if
    /// they were prefix-valid against the drafter's "single row each
    /// occupies `[0, kv_valid_len[b]), tail zeroed" invariant.
    ///
    /// The current scalar `left_padding` arg from the trait is broadcast
    /// across rows. A follow-up will accept per-row `left_padding` vectors
    /// directly (tracked alongside the batched MTP wiring); until then,
    /// the round-loop driver collapses per-row `left_padding` to its max
    /// and the masks helper handles the (defensive) broadcast.
    fn from_shared_kv_normalized(
        shared: &SharedKv<'_>,
        left_padding: usize,
    ) -> Result<Self, DrafterError> {
        let kv_len = shared_kv_len(shared)?;
        let kv_valid_len = kv_len.saturating_sub(left_padding as i32);
        let valid_scalar = BatchScalar::Scalar(kv_valid_len);
        let left_scalar = BatchScalar::Scalar(left_padding as i32);
        Self::from_shared_kv_normalized_with_metadata(shared, &valid_scalar, Some(&left_scalar))
    }

    /// Batched-MTP constructor that accepts explicit per-row valid lengths
    /// and left-padding extents. This is the reference-parity path used
    /// after rows diverge: each row's prefix is normalized independently
    /// before the drafter builds masks and cross-attends into the shared
    /// K/V slabs.
    fn from_shared_kv_normalized_with_metadata(
        shared: &SharedKv<'_>,
        kv_valid_len: &BatchScalar<'_>,
        left_padding: Option<&BatchScalar<'_>>,
    ) -> Result<Self, DrafterError> {
        use crate::drafter::masks::normalize_batched_shared_kv_states;

        // Build the `LayerType -> (K, V)` map the masks helper expects.
        let (k_full, v_full, k_swa, v_swa) = match shared.tensors.len() {
            2 => (shared.tensors[0], shared.tensors[1], None, None),
            4 => (
                shared.tensors[0],
                shared.tensors[1],
                Some(shared.tensors[2]),
                Some(shared.tensors[3]),
            ),
            n => {
                return Err(DrafterError::SharedKvShape {
                    got: n,
                    expected: &[2, 4],
                });
            }
        };

        let mut map: HashMap<LayerType, (&MlxArray, &MlxArray)> = HashMap::new();
        map.insert(LayerType::FullAttention, (k_full, v_full));
        if let (Some(ks), Some(vs)) = (k_swa, v_swa) {
            map.insert(LayerType::SlidingWindowAttention, (ks, vs));
        }

        let normalized = normalize_batched_shared_kv_states(&map, kv_valid_len, left_padding);

        let take_pair = |layer: LayerType| -> Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
            normalized
                .get(&layer)
                .map(|(k, v)| (ffi::copy(k), ffi::copy(v)))
        };

        Ok(Self {
            full_attention: take_pair(LayerType::FullAttention),
            sliding_attention: take_pair(LayerType::SlidingWindowAttention),
        })
    }

    fn for_layer_type(&self, layer_type: &str) -> Result<(&MlxArray, &MlxArray), DrafterError> {
        let pair = match layer_type {
            "full_attention" => self.full_attention.as_ref(),
            "sliding_attention" => self.sliding_attention.as_ref(),
            other => {
                return Err(DrafterError::UnknownLayerType {
                    got: other.to_string(),
                });
            }
        };
        let (k, v) = pair.ok_or_else(|| DrafterError::MissingSharedKvForLayerType {
            layer_type: layer_type.to_string(),
        })?;
        Ok((
            k.as_ref().expect("non-null K"),
            v.as_ref().expect("non-null V"),
        ))
    }

    /// Build a `HashMap<LayerType, (&MlxArray, &MlxArray)>` from the owned
    /// shared K/V, as required by [`make_drafter_masks`]. Only layer types
    /// with data present are included, matching the number of tensors
    /// provided at `set_shared_kv()` time (2 or 4).
    fn as_layer_type_map(&self) -> HashMap<LayerType, (&MlxArray, &MlxArray)> {
        let mut map = HashMap::new();
        if let Some((k, v)) = &self.full_attention {
            map.insert(
                LayerType::FullAttention,
                (
                    k.as_ref().expect("non-null K"),
                    v.as_ref().expect("non-null V"),
                ),
            );
        }
        if let Some((k, v)) = &self.sliding_attention {
            map.insert(
                LayerType::SlidingWindowAttention,
                (
                    k.as_ref().expect("non-null K"),
                    v.as_ref().expect("non-null V"),
                ),
            );
        }
        map
    }
}

fn shared_kv_len(shared: &SharedKv<'_>) -> Result<i32, DrafterError> {
    let first = match shared.tensors.len() {
        2 | 4 => shared.tensors[0],
        n => {
            return Err(DrafterError::SharedKvShape {
                got: n,
                expected: &[2, 4],
            });
        }
    };
    let shape = ffi::array_shape(first);
    Ok(if shape.len() == 4 { shape[2] } else { 0 })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a string layer-type key to the `LayerType` enum.
///
/// The drafter layers carry their type as `&str` (from `DraftDecoderLayer::layer_type()`),
/// while `make_drafter_masks` and `OwnedSharedKv::as_layer_type_map` use the
/// `LayerType` enum. This bridge keeps the string ↔ enum conversion in one place.
fn str_to_layer_type(s: &str) -> Result<LayerType, DrafterError> {
    match s {
        "full_attention" => Ok(LayerType::FullAttention),
        "sliding_attention" => Ok(LayerType::SlidingWindowAttention),
        other => Err(DrafterError::UnknownLayerType {
            got: other.to_string(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Gemma4AssistantDraftModel
// ---------------------------------------------------------------------------

/// Gemma 4 MTP "assistant" drafter — 4-layer transformer with pre/post
/// projections and frozen RoPE cross-attention into the target's last-layer
/// K/V slabs.
///
/// Implements [`Drafter`] and is wired into the `Mtp` arm of
/// [`crate::drafter::load_drafter`].
pub struct Gemma4AssistantDraftModel {
    config: Gemma4AssistantConfig,
    inner: DraftInner,
    pre_projection: Linear,
    post_projection: Linear,
    /// Explicit `lm_head` weight when `tie_word_embeddings == false`. `None`
    /// means the LM head is one of the tied / centroid variants resolved by
    /// `bind()`.
    lm_head_weight: Option<Linear>,
    /// Pre-built centroid LM head for E-series drafters (`use_ordered_embeddings=true`).
    /// Constructed in `from_weights` when the checkpoint carries
    /// `masked_embedding.*` weights, then consumed by `resolve_lm_head` at
    /// `bind()` time. `None` for the 26B-A4B / 31B tied-dense paths.
    centroid_lm_head: Option<MaskedEmbedder>,
    /// LM head dispatch — finalised by `bind()`. Until then, callers that
    /// invoke `draft_block()` get an explicit "must call bind() first" error.
    lm_head: Option<LmHead>,

    /// Target's embedding table — captured by `bind()`. `None` means
    /// `bind()` has not run yet; `draft_block()` rejects that state.
    /// Stored as `MlxArray` from
    /// `LanguageModel::embed_tokens(&target_input_ids)`-returned tensors
    /// rather than holding a target reference, so the drafter doesn't need
    /// to keep a `&dyn LanguageModel` alive across calls.
    target_embed: Option<TargetEmbedAdapter>,
    target_embed_scale: f32,

    /// State set by `set_shared_kv()`. `None` means the round-loop has not
    /// armed the drafter yet.
    shared_kv: Option<OwnedSharedKv>,
    /// `kv_offset` from `set_shared_kv()` — kept for diagnostics.
    kv_offset: i32,
    /// Valid target-cache length associated with the current shared K/V.
    ///
    /// Latest upstream Gemma 4 MTP distinguishes this from `position`:
    /// `position` is the frozen query/RoPE anchor (`kv_valid_len - 1`),
    /// while masks must still treat all keys before `kv_valid_len` as
    /// valid. Keeping both values prevents the drafter from accidentally
    /// masking the final verified key after each rebind.
    kv_valid_len: i32,
    /// Bonus-token absolute position. Used as the RoPE offset for every
    /// step inside a draft block (the "frozen anchor" semantics).
    position: i32,
    /// Batched MTP per-row bonus-token positions. When present, this is
    /// used for RoPE anchors; [`position`] remains the max for diagnostics
    /// and scalar fallback paths.
    position_per_row: Option<Vec<i32>>,
    /// Device-side copy of [`position_per_row`] used by the mask helpers.
    position_per_row_array: Option<UniquePtr<MlxArray>>,
    /// Device-side per-row valid target-cache lengths used by the mask
    /// helpers. When absent, [`kv_valid_len`] is broadcast to every row.
    kv_valid_len_per_row_array: Option<UniquePtr<MlxArray>>,
}

// Manual `Debug` impl: `Linear`, `Embedding`, and `MlxArray` are FFI-opaque
// and do not derive `Debug`. The values themselves are not safe to materialise
// off the dispatch thread, so this surface intentionally renders only the
// scalar metadata diagnostic consumers actually want.
impl std::fmt::Debug for Gemma4AssistantDraftModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemma4AssistantDraftModel")
            .field("model_type", &self.config.model_type)
            .field("backbone_hidden_size", &self.config.backbone_hidden_size)
            .field("block_size", &self.config.block_size)
            .field("tie_word_embeddings", &self.config.tie_word_embeddings)
            .field(
                "use_ordered_embeddings",
                &self.config.use_ordered_embeddings,
            )
            .field("num_layers", &self.inner.layers.len())
            .field("centroid_lm_head_ready", &self.centroid_lm_head.is_some())
            .field("bound", &self.lm_head.is_some())
            .field("shared_kv_set", &self.shared_kv.is_some())
            .field("kv_offset", &self.kv_offset)
            .field("kv_valid_len", &self.kv_valid_len)
            .field("position", &self.position)
            .field(
                "position_per_row_len",
                &self.position_per_row.as_ref().map(Vec::len),
            )
            .finish()
    }
}

/// Captured target embedding plumbing.
///
/// Holding a `&dyn LanguageModel` for the lifetime of the drafter would force
/// the drafter to outlive the target wrapper, which would in turn force the
/// caller to wrap the target in `Arc<dyn LanguageModel>`. Instead, `bind()`
/// asks the target for a shared-buffer [`UnifiedEmbedding`] handle and stashes
/// it here so subsequent `embed(token_id)` lookups inside `draft_block()` use
/// the target backbone's embedding width (for example Gemma 4 31B's 5376-wide
/// table) instead of the assistant's smaller internal embedding table.
struct TargetEmbedAdapter {
    /// Shared-buffer target embedding module. Cloning this handle does not
    /// copy device memory; MLX arrays are reference-counted.
    embed_tokens: UnifiedEmbedding,
}

impl Gemma4AssistantDraftModel {
    /// Construct from a checkpoint directory containing `config.json` and
    /// safetensors shards. Used by [`crate::drafter::load_drafter`]'s `Mtp`
    /// arm.
    pub fn from_path(path: &Path) -> Result<Self, DrafterError> {
        let config = load_config(path)?
            .normalize()
            .map_err(DrafterError::Config)?;
        let mut weights = crate::weights::load_weights_from_dir(path)
            .map_err(|e| DrafterError::WeightLoad { reason: e })?;
        Self::sanitize_weights(&mut weights, &config);
        Self::from_weights(weights, config)
    }

    /// Construct from an in-memory weight map. Used by both `from_path` and
    /// unit tests that build small fixture weight maps.
    pub fn from_weights(
        weights: WeightMap,
        config: Gemma4AssistantConfig,
    ) -> Result<Self, DrafterError> {
        let text_cfg = config.text_config().clone();

        let inner = DraftInner::from_weights(&weights, &text_cfg, "model")
            .map_err(|e| DrafterError::WeightLoad { reason: e })?;

        let pre_projection = Linear::from_weights(&weights, "pre_projection")
            .map_err(|e| DrafterError::WeightLoad { reason: e })?;
        let post_projection = Linear::from_weights(&weights, "post_projection")
            .map_err(|e| DrafterError::WeightLoad { reason: e })?;

        let lm_head_weight = if config.tie_word_embeddings {
            None
        } else {
            Some(
                Linear::from_weights(&weights, "lm_head")
                    .map_err(|e| DrafterError::WeightLoad { reason: e })?,
            )
        };

        // Pre-build the centroid LM head for E-series drafters. Must happen
        // here while the WeightMap is still available (before it is consumed
        // or dropped). For 26B-A4B / 31B drafters (`use_ordered_embeddings ==
        // false`) this branch is a no-op.
        let centroid_lm_head = if config.use_ordered_embeddings {
            let embedder = MaskedEmbedder::from_weights(
                &weights,
                "masked_embedding",
                text_cfg.hidden_size,
                text_cfg.vocab_size,
                config.num_centroids,
                config.centroid_intermediate_top_k,
            )
            .map_err(|e| DrafterError::WeightLoad {
                reason: format!("MaskedEmbedder load failed: {e}"),
            })?;
            Some(embedder)
        } else {
            None
        };

        Ok(Self {
            config,
            inner,
            pre_projection,
            post_projection,
            lm_head_weight,
            centroid_lm_head,
            lm_head: None,
            target_embed: None,
            target_embed_scale: 1.0,
            shared_kv: None,
            kv_offset: 0,
            kv_valid_len: 0,
            position: 0,
            position_per_row: None,
            position_per_row_array: None,
            kv_valid_len_per_row_array: None,
        })
    }

    /// Apply the upstream Python `sanitize` rules to a freshly-loaded weight
    /// map:
    ///
    /// - When `tie_word_embeddings == true`, drop `lm_head.weight` (and any
    ///   sister tensors) — it must not be loaded as a standalone Linear.
    /// - Cast `masked_embedding.token_ordering` from int64 to int32 (used
    ///   only on E-series drafters with the centroid LM head).
    ///
    /// Mirrors upstream `Gemma4AssistantDraftModel.sanitize` in
    /// `references/mlx-vlm/mlx_vlm/speculative/drafters/gemma4_assistant/gemma4_assistant.py`.
    pub fn sanitize_weights(weights: &mut WeightMap, config: &Gemma4AssistantConfig) {
        if config.tie_word_embeddings {
            weights.remove("lm_head.weight");
            weights.remove("lm_head.scales");
            weights.remove("lm_head.biases");
        }
        // Cast `masked_embedding.token_ordering` to int32. HuggingFace
        // checkpoints ship this as int64; mlxcel uses int32 throughout for
        // indexing efficiency and to match the dtype that `take` /
        // `put_along_axis` expect. No-op on already-int32 buffers and on
        // 26B-A4B / 31B drafters that carry no centroid table.
        // Mirrors upstream `if k == "masked_embedding.token_ordering": v = v.astype(mx.int32)`.
        crate::drafter::masked_embedder::sanitize_token_ordering(weights, "masked_embedding");
    }

    /// Configure the LM head dispatch based on the drafter's config and
    /// captured weights. Called from [`Self::bind`].
    fn resolve_lm_head(&mut self) -> Result<(), DrafterError> {
        let head = if self.config.use_ordered_embeddings {
            // Centroid LM head — construct the real `MaskedEmbedder` from
            // the already-loaded and sanitized weight map. The weights were
            // loaded in `from_weights` and sanitized in `sanitize_weights`
            // (token_ordering cast to int32). We cannot go back to the weight
            // map at this point, so we reconstruct from the config metadata.
            // The actual weight tensors were placed in `lm_head_weight` slot
            // only when `tie_word_embeddings == false`. For the E-series
            // drafters `tie_word_embeddings == true`, so we re-derive from
            // the config parameters that were loaded at construction time.
            //
            // MaskedEmbedder::from_weights needs the WeightMap, but we don't
            // hold it at bind()-time. Instead, we stash the constructed
            // `MaskedEmbedder` in `lm_head_weight`-adjacent storage via
            // a dedicated `centroid_lm_head` field on the model. Since we
            // can't reach the weights here, the centroid head is pre-built
            // during `from_weights` (via `centroid_lm_head: Option<MaskedEmbedder>`)
            // and this method just takes it out.
            let centroid =
                self.centroid_lm_head
                    .take()
                    .ok_or_else(|| DrafterError::WeightLoad {
                        reason: "use_ordered_embeddings=true but MaskedEmbedder was not pre-built \
                             during from_weights; ensure the checkpoint contains \
                             masked_embedding.centroids.weight and \
                             masked_embedding.token_ordering"
                            .into(),
                    })?;
            LmHead::Centroid(centroid)
        } else if self.config.tie_word_embeddings {
            LmHead::Tied
        } else {
            // Explicit lm_head — already loaded into `lm_head_weight` by
            // `from_weights`.
            let head_weight =
                self.lm_head_weight
                    .take()
                    .ok_or_else(|| DrafterError::WeightLoad {
                        reason: "tie_word_embeddings=false but lm_head.weight was not loaded"
                            .into(),
                    })?;
            LmHead::Linear(head_weight)
        };
        self.lm_head = Some(head);
        Ok(())
    }

    /// Resolve the target's inner module via the three known wrapper depths
    /// and capture its embedding scale.
    ///
    /// Mirrors upstream:
    /// ```python
    /// if hasattr(target_model, "embed_tokens"):
    ///     inner = target_model
    /// elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
    ///     inner = target_model.model
    /// elif (hasattr(target_model, "language_model")
    ///       and hasattr(target_model.language_model, "model")
    ///       and hasattr(target_model.language_model.model, "embed_tokens")):
    ///     inner = target_model.language_model.model
    /// ```
    ///
    /// In Rust, all three depths converge on the [`LanguageModel`] trait's
    /// [`LanguageModel::embed_tokens`] method, which the gemma4 wrappers
    /// implement at every depth and forward to the text model's embedding
    /// table. The drafter only needs to know that the method returns
    /// `Some(_)` — if it returns `None`, the target does not expose its
    /// embedding plumbing (e.g. some text-only models that do not implement
    /// embed_tokens) and `bind()` fails.
    fn capture_target_embedding(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        // Build a single-element sentinel input so we can call
        // `target.embed_tokens(input_ids)` and confirm the target exposes its
        // embedding plumbing. The returned tensor's first row is the
        // embedding of token id 0 — we only need it to fail-fast when the
        // target lacks the override.
        let sentinel_ids = ffi::from_slice_i32(&[0], &[1, 1]);
        let embedded =
            target
                .embed_tokens(&sentinel_ids)
                .ok_or(DrafterError::TargetMissingFeature {
                    feature: "embed_tokens",
                })?;

        let embed_shape = ffi::array_shape(&embedded);
        let target_hidden = embed_shape.last().copied().unwrap_or(1).max(1);

        // Gemma 4 target forward scales token embeddings by
        // `sqrt(hidden_size)` before the transformer. The MTP assistant
        // receives the same embedding stream concatenated with the target
        // hidden state, so compute the scale from the target embedding
        // width observed above. This keeps the drafter input identical to
        // upstream without adding a new target-specific trait method.
        self.target_embed_scale = (target_hidden as f32).sqrt();

        // Store the target embedding module when the target can hand out a
        // shared-buffer handle. Older unit-test mocks only implement
        // `embed_tokens`; those still bind successfully and fall back to
        // the assistant's own embedding table in `draft_block()`.
        self.target_embed = target
            .embed_tokens_module()
            .map(|embed_tokens| TargetEmbedAdapter { embed_tokens });

        // Capture target's layer_types via a best-effort interface query.
        // The current `LanguageModel` trait does not expose `layer_types`
        // so the drafter falls back to the drafter's own text_config
        // layer_types (which mirror the target by construction on all four
        // supported pairings). When wires a richer round-loop API,
        // pass the target's actual layer_types in here.
        self.config.target_layer_types = self.config.text_config().layer_types.clone();

        Ok(())
    }

    /// Embed a single token id via the target's embedding table, applying
    /// `embed_scale` per Gemma convention.
    ///
    /// `token_id` is `[B=1, 1]`-shape; returns `[1, 1, backbone_hidden_size]`.
    /// The result feeds into `pre_projection` after concatenation with the
    /// drafter's recurrent hidden state.
    ///
    /// Kept as a direct target-trait helper for tests / future call sites.
    /// The hot `draft_block` path uses the shared module captured at
    /// `bind()` time instead of re-entering the target trait object.
    #[allow(dead_code)]
    fn embed_with_scale(
        &self,
        target: &dyn LanguageModel,
        token_id: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let embedded = target
            .embed_tokens(token_id)
            .ok_or(DrafterError::TargetMissingFeature {
                feature: "embed_tokens",
            })?;
        // multiply_scalar accepts f32; embed_scale = 1.0 is a no-op fast
        // path inside MLX so this is free in the default case.
        Ok(crate::multiply_scalar(&embedded, self.target_embed_scale))
    }

    /// Embed token ids for the drafter input stream.
    ///
    /// Real Gemma 4 targets expose [`LanguageModel::embed_tokens_module`]
    /// and this method uses that target table (scaled by
    /// `sqrt(target_hidden_size)`) so concatenation with the target's
    /// hidden state has width `2 * backbone_hidden_size`. Synthetic unit
    /// tests may bind against mocks that only expose `embed_tokens`; those
    /// fall back to the assistant's own embedding table.
    fn draft_input_embed(&self, token_ids: &MlxArray) -> UniquePtr<MlxArray> {
        if let Some(target_embed) = &self.target_embed {
            let embedded = target_embed.embed_tokens.forward(token_ids);
            crate::multiply_scalar(&embedded, self.target_embed_scale)
        } else {
            self.inner.embed_tokens.forward(token_ids)
        }
    }

    /// Forward through the drafter's transformer stack with the current
    /// `shared_kv` slabs. Mirrors `Gemma4AssistantDraftModel.__call__` from
    /// upstream Python.
    ///
    /// `inputs_embeds`: `[1, 1, 2 * backbone_hidden_size]` (concat of target
    /// embed + last hidden). Returns `(last_hidden, logits)` where
    /// `last_hidden` has shape `[1, 1, backbone_hidden_size]` (output of
    /// `post_projection`) and `logits` has shape `[1, 1, vocab_size]`.
    fn forward(
        &self,
        inputs_embeds: &MlxArray,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), DrafterError> {
        let shared = self
            .shared_kv
            .as_ref()
            .ok_or(DrafterError::SetSharedKvNotCalled)?;
        let lm_head = self.lm_head.as_ref().ok_or(DrafterError::BindNotCalled)?;

        // pre_projection: [1, 1, 2 * backbone] → [1, 1, drafter_hidden]
        let mut h = self.pre_projection.forward(inputs_embeds);

        // Build the per-layer-type mask map via the real `make_drafter_masks`
        // helper. This replaces the old `make_drafter_masks_stub` — the real
        // helper uses `HashMap<LayerType, ...>` keys and handles both the B=1
        // fast path (returns `None` masks when query_offset >= kv_len and
        // kv_len <= sliding_window) and the batched path.
        //
        // Upstream Gemma 4 MTP now separates the drafter's query/RoPE
        // anchor from the K/V valid length:
        //
        //   position     = max(kv_valid_len - 1, 0)
        //   kv_valid_len = target cache length
        //
        // The drafter queries from `position` (the last verified token) but
        // masks must still expose every key before `kv_valid_len`.
        let shared_kv_map = shared.as_layer_type_map();
        let query_offset = self
            .position_per_row_array
            .as_ref()
            .and_then(|arr| arr.as_ref())
            .map(BatchScalar::PerRow)
            .unwrap_or(BatchScalar::Scalar(self.position));
        let kv_valid_len = self
            .kv_valid_len_per_row_array
            .as_ref()
            .and_then(|arr| arr.as_ref())
            .map(BatchScalar::PerRow)
            .unwrap_or(BatchScalar::Scalar(self.kv_valid_len));
        let rope_offset = self
            .position_per_row
            .as_deref()
            .map(RopeOffset::PerRow)
            .unwrap_or(RopeOffset::Scalar(self.position));
        let sliding_window = self.config.text_config().sliding_window as i32;
        let dtype = ffi::array_dtype(inputs_embeds);
        let masks = make_drafter_masks_with_valid_len(
            &shared_kv_map,
            /*query_len=*/ 1,
            &query_offset,
            sliding_window,
            dtype,
            Some(&kv_valid_len),
        );

        // Run each drafter layer with shared K/V and the frozen RoPE offset.
        for layer in &self.inner.layers {
            let (k, v) = shared.for_layer_type(layer.layer_type())?;
            // Convert the string layer_type to the `LayerType` enum to look
            // up in the `HashMap<LayerType, ...>` returned by `make_drafter_masks`.
            let layer_type_enum = str_to_layer_type(layer.layer_type())?;
            let mask_opt = masks.get(&layer_type_enum).and_then(|m| m.as_deref());
            h = layer.forward(&h, mask_opt, k, v, rope_offset);
        }

        // Final RMSNorm + post_projection.
        let h = self.inner.norm.forward(&h);
        let last_hidden = self.post_projection.forward(&h);

        // LM head: tied dense uses drafter's `embed_tokens.as_linear`,
        // explicit linear uses its own weight, centroid uses the
        // `MaskedEmbedder` with the drafter's tied embed-tokens weight as
        // the `lm_head_weight` input (upstream ties the embed table).
        let logits = match lm_head {
            LmHead::Tied => self.inner.embed_tokens.as_linear(&h),
            LmHead::Linear(linear) => linear.forward(&h),
            LmHead::Centroid(embedder) => {
                // Centroid path: the MaskedEmbedder needs the tied
                // `embed_tokens.weight` as its `lm_head_weight` argument.
                // The drafter's embed table is shared with the LM head by
                // construction on E-series drafters (`tie_word_embeddings=true`).
                let embed_weight = self.inner.embed_tokens.weight();
                embedder.forward(&h, embed_weight)
            }
        };

        Ok((last_hidden, logits))
    }

    /// Single-row argmax sample from a `[1, 1, vocab]` logits tensor.
    ///
    /// The full `SamplingConfig`-aware sampler currently operates on
    /// `[batch, seq, vocab]` and re-enters per-sequence state tracking —
    /// the drafter only needs a degenerate path. Temperature 0 (greedy) is
    /// the only path verified byte-identical by upstream (see README:
    /// "Quality matches the target at temperature 0"), so non-greedy
    /// configs are a quality-loss path the round-loop owns. We still
    /// honour `temperature` via the existing `fused_sample` kernel so that
    /// future temperature-aware MTP variants get the correct draws.
    fn sample_one(logits: &MlxArray, sampler: &SamplingConfig) -> UniquePtr<MlxArray> {
        let last_logits = ffi::slice_last_logits(logits);
        ffi::fused_sample(
            &last_logits,
            sampler.temperature,
            sampler.top_k,
            sampler.top_p,
            sampler.min_p,
        )
    }
}

impl Drafter for Gemma4AssistantDraftModel {
    fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.capture_target_embedding(target)?;
        self.resolve_lm_head()?;
        Ok(())
    }

    fn set_shared_kv(
        &mut self,
        shared_kv: SharedKv<'_>,
        kv_offset: usize,
        position: usize,
        left_padding: usize,
    ) -> Result<(), DrafterError> {
        // (batched MTP): when the round-loop is running B > 1
        // with left-padded shared K/V, the drafter normalizes each row so
        // the cross-attention forward sees the simpler invariant: each
        // row's real keys occupy `[0, kv_valid_len)` and the tail is
        // zeroed. Routes through
        // [`crate::drafter::masks::normalize_batched_shared_kv_states`].
        //
        // For B = 1 (or `left_padding == 0`), we skip the normalization
        // path entirely — it is a no-op on the unbatched MVP shape and
        // would only add an unnecessary tensor copy. The bit-identity test
        // in `tests.rs` (`round_loop_full_accept_emits_all_proposals_plus_bonus_per_round`)
        // pins the no-normalize path's behaviour.
        if left_padding > 0 {
            self.shared_kv = Some(OwnedSharedKv::from_shared_kv_normalized(
                &shared_kv,
                left_padding,
            )?);
        } else {
            self.shared_kv = Some(OwnedSharedKv::from_shared_kv(&shared_kv)?);
        }
        self.kv_offset = kv_offset as i32;
        self.kv_valid_len = kv_offset as i32;
        self.position = position as i32;
        self.position_per_row = None;
        self.position_per_row_array = None;
        self.kv_valid_len_per_row_array = None;
        Ok(())
    }

    fn set_shared_kv_batched(
        &mut self,
        shared_kv: SharedKv<'_>,
        kv_offset_per_row: &[usize],
        position_per_row: &[usize],
        kv_valid_len_per_row: &[usize],
        left_padding_per_row: &[usize],
    ) -> Result<(), DrafterError> {
        let batch = kv_offset_per_row.len();
        if position_per_row.len() != batch
            || kv_valid_len_per_row.len() != batch
            || left_padding_per_row.len() != batch
        {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Gemma 4 assistant batched set_shared_kv metadata length mismatch: \
                     kv_offset={}, position={}, kv_valid_len={}, left_padding={}",
                    kv_offset_per_row.len(),
                    position_per_row.len(),
                    kv_valid_len_per_row.len(),
                    left_padding_per_row.len()
                ),
            });
        }
        if batch == 0 {
            return Err(DrafterError::DraftFailed {
                reason: "Gemma 4 assistant batched set_shared_kv requires B >= 1".to_string(),
            });
        }

        let to_i32 = |values: &[usize], label: &str| -> Result<Vec<i32>, DrafterError> {
            values
                .iter()
                .copied()
                .map(|v| {
                    i32::try_from(v).map_err(|_| DrafterError::DraftFailed {
                        reason: format!(
                            "Gemma 4 assistant batched set_shared_kv {label} value {v} \
                             exceeds i32::MAX"
                        ),
                    })
                })
                .collect()
        };

        let kv_offsets_i32 = to_i32(kv_offset_per_row, "kv_offset")?;
        let positions_i32 = to_i32(position_per_row, "position")?;
        let valid_i32 = to_i32(kv_valid_len_per_row, "kv_valid_len")?;
        let left_i32 = to_i32(left_padding_per_row, "left_padding")?;

        let position_arr = ffi::from_slice_i32(&positions_i32, &[batch as i32]);
        let valid_arr = ffi::from_slice_i32(&valid_i32, &[batch as i32]);
        let left_arr = ffi::from_slice_i32(&left_i32, &[batch as i32]);
        let valid = BatchScalar::PerRow(valid_arr.as_ref().expect("non-null valid metadata"));
        let left = BatchScalar::PerRow(left_arr.as_ref().expect("non-null left metadata"));
        self.shared_kv = Some(OwnedSharedKv::from_shared_kv_normalized_with_metadata(
            &shared_kv,
            &valid,
            Some(&left),
        )?);

        self.kv_offset = kv_offsets_i32.iter().copied().max().unwrap_or(0);
        self.kv_valid_len = valid_i32.iter().copied().max().unwrap_or(0);
        self.position = positions_i32.iter().copied().max().unwrap_or(0);
        self.position_per_row = Some(positions_i32);
        self.position_per_row_array = Some(position_arr);
        self.kv_valid_len_per_row_array = Some(valid_arr);
        Ok(())
    }

    fn make_cache(&self) -> Vec<KVCache> {
        // The MTP drafter has no own KV cache — its only recurrent state is
        // the target's last hidden, projected through `post_projection`. The
        // default trait impl already returns an empty Vec, so the override
        // here is only to be explicit about intent.
        Vec::new()
    }

    fn configured_block_size(&self) -> Option<usize> {
        Some(self.config.block_size)
    }

    fn draft_block_batched(
        &mut self,
        last_bonus: &[i32],
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<Vec<Vec<i32>>, DrafterError> {
        // Batched autoregressive draft. Performs `K-1` small
        // forwards with `[B, 1, ...]` shapes, sampling one token per row
        // each step. Mirrors the B = 1 path in [`Self::draft_block`] but
        // keeps the batch dim throughout.
        //
        // The drafter MUST have been `bind`()-ed and `set_shared_kv`()-ed
        // before reaching this point (the round-loop driver enforces
        // both). The shared K/V's batch dim has to match `last_bonus.len()`
        // for the cross-attention forward to produce the right per-row
        // outputs.
        if self.shared_kv.is_none() {
            return Err(DrafterError::SetSharedKvNotCalled);
        }
        if self.lm_head.is_none() {
            return Err(DrafterError::BindNotCalled);
        }
        if block_size == 0 || last_bonus.is_empty() {
            return Ok(last_bonus.iter().map(|_| Vec::new()).collect());
        }
        let hidden = hidden.ok_or(DrafterError::DraftBlockMissingHidden)?;

        let batch_size = last_bonus.len();
        let proposals = (block_size as i32).saturating_sub(1).max(0);
        if proposals == 0 {
            return Ok((0..batch_size).map(|_| Vec::new()).collect());
        }

        // Per-row token-stream accumulators.
        let mut tokens_per_row: Vec<Vec<i32>> = (0..batch_size)
            .map(|_| Vec::with_capacity(proposals as usize))
            .collect();

        // Per-step recurrent state: `h_prev` starts at the caller's
        // [B, 1, backbone] target hidden; `last_tokens` starts at the
        // per-row bonus slice.
        let mut h_prev = ffi::copy(hidden);
        let mut last_tokens: Vec<i32> = last_bonus.to_vec();

        for _ in 0..proposals {
            // Per-row embed: build a [B, 1] token-id tensor, embed, scale.
            let tok_ids = ffi::from_slice_i32(&last_tokens, &[batch_size as i32, 1]);
            let tok_embed = self.draft_input_embed(&tok_ids);

            // [B, 1, hidden] + [B, 1, backbone] → [B, 1, 2 * backbone]
            let inputs_embeds = crate::concatenate(&tok_embed, &h_prev, -1);

            let (next_hidden, logits) = self.forward(&inputs_embeds)?;

            // Per-row argmax (or sampled) tokens. Greedy at temp=0 is the
            // load-bearing correctness path; non-greedy is the
            // quality-loss path the round-loop owns.
            let last_logits = ffi::slice_last_logits(&logits);
            let sampled = ffi::fused_sample(
                &last_logits,
                sampler.temperature,
                sampler.top_k,
                sampler.top_p,
                sampler.min_p,
            );
            ffi::eval(&sampled);

            // Materialize each row's sampled token. Shape of `sampled`
            // is `[B]` (one int per batch row).
            for r in 0..batch_size {
                let cell = ffi::slice(&sampled, &[r as i32], &[(r as i32) + 1]);
                let scalar = ffi::reshape(&cell, &[]);
                let tok = ffi::item_i32(&scalar);
                tokens_per_row[r].push(tok);
                last_tokens[r] = tok;
            }

            h_prev = next_hidden;
        }

        Ok(tokens_per_row)
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        // `bind()` captures the target embedding module when the target
        // exposes one, so real Gemma 4 pairings embed `last_bonus` with
        // the backbone-width target table every step. Unit-test mocks
        // that only implement `embed_tokens` still bind successfully and
        // fall back to the assistant's own embedding table.

        if self.shared_kv.is_none() {
            return Err(DrafterError::SetSharedKvNotCalled);
        }
        if self.lm_head.is_none() {
            return Err(DrafterError::BindNotCalled);
        }
        if block_size == 0 {
            return Ok(Vec::new());
        }
        let hidden = hidden.ok_or(DrafterError::DraftBlockMissingHidden)?;

        let mut tokens: Vec<i32> = Vec::with_capacity(block_size.saturating_sub(1));

        // Per upstream Python `draft_block`:
        //   for _ in range(block_size - 1):
        //     tok_embed = self._input_embed(tok) * self._input_embed_scale
        //     inputs_embeds = mx.concatenate([tok_embed, h_prev], axis=-1)
        //     h_prev, logits = self(inputs_embeds, shared_kv, position_ids)
        //     tok = sampler(logits)
        //     tokens.append(tok)
        //
        // `h_prev` starts at the target's last hidden (`hidden`).
        let mut h_prev = ffi::copy(hidden);
        let mut last_token = last_bonus;

        for _ in 0..block_size.saturating_sub(1) {
            // Embed last_token using the drafter's own embed_tokens as a
            // fallback for mocks that do not expose an embedding module.
            // Real Gemma 4 targets bind a shared target embedding table so
            // this tensor has backbone width and matches the target hidden.
            let tok_ids = ffi::from_slice_i32(&[last_token], &[1, 1]);
            let tok_embed = self.draft_input_embed(&tok_ids);

            // Concatenate along the last axis: [1, 1, hidden_size] + [1, 1,
            // backbone_hidden_size] → [1, 1, 2 * backbone_hidden_size]. The
            // upstream code keeps `h_prev` at `backbone_hidden_size` (it
            // came from `post_projection`), so the drafter's `pre_projection`
            // expects `2 * backbone_hidden_size`.
            let inputs_embeds = crate::concatenate(&tok_embed, &h_prev, -1);

            let (next_hidden, logits) = self.forward(&inputs_embeds)?;
            let token = Self::sample_one(&logits, sampler);
            ffi::eval(&token);
            let token_i32 = ffi::item_i32(&token);
            tokens.push(token_i32);

            last_token = token_i32;
            h_prev = next_hidden;
        }

        Ok(tokens)
    }

    fn sanitize(&mut self, weights: &mut WeightMap) -> Result<(), DrafterError> {
        Self::sanitize_weights(weights, &self.config);
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Mtp
    }
}

/// Best-effort `config.json` loader. Routes through `serde_json::from_slice`
/// rather than the project's heavier config-loading utilities so this stays
/// free of `mlxcel`-crate dependencies.
fn load_config(path: &Path) -> Result<Gemma4AssistantConfig, DrafterError> {
    let cfg_path = path.join("config.json");
    let bytes = std::fs::read(&cfg_path).map_err(|e| DrafterError::ConfigIo {
        path: cfg_path.display().to_string(),
        source: e,
    })?;
    serde_json::from_slice::<Gemma4AssistantConfig>(&bytes).map_err(|e| DrafterError::ConfigParse {
        path: cfg_path.display().to_string(),
        source: e,
    })
}
