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

//! Speculative drafter abstraction shared by both MTP and DFlash drafter
//! families being ported from `mlx-vlm`.
//!
//! ## Why a trait?
//!
//! mlxcel originally shipped one speculative path — the classic
//! [`crate::speculative::SpeculativeGenerator`] — where the drafter owns its
//! own KV cache and emits tokens one at a time. The Gemma 4 Multi-Token
//! Prediction (MTP) "assistant" drafter and the Qwen 3.5 DFlash drafter have
//! fundamentally different lifecycles than that classic path:
//!
//! - **MTP-style** (Gemma 4 assistant) — shares K/V from the target (no own
//!   KV cache), runs `K` small autoregressive forwards per draft block,
//!   keeps cross-attention queries RoPE-rotated at the bonus token's
//!   absolute position constant across the block, single verify per block,
//!   per-row tail-zero rollback.
//! - **DFlash-style** (Qwen 3.5 DFlash) — owns its KV cache, takes a
//!   multi-layer hidden-state concatenation from the target's captured
//!   layers as input, produces `block_size - 1` proposal tokens in a
//!   single masked forward, and on a hybrid Mamba+Transformer target
//!   (Qwen 3.5) requires GDN-aware rollback alongside the standard KV
//!   trim.
//!
//! The [`Drafter`] trait defined here unifies these shapes behind a single
//! interface so the round-loop drivers (sub-6 for MTP, sub-12 for DFlash) can drive any drafter uniformly. Each concrete drafter
//! overrides only the methods it actually needs and lets the trait's
//! default no-ops cover the rest.
//!
//! ## Upstream reference
//!
//! This module ports the public surface of
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/__init__.py:
//!
//! ```python
//! KNOWN_DRAFTER_KINDS = {"dflash", "mtp"}
//! DRAFTER_KIND_BY_MODEL_TYPE = {
//!     "gemma4_assistant": "mtp",
//!     "gemma4_unified_assistant": "mtp",
//! }
//! DEFAULT_DRAFTER_KIND = "dflash"
//! ```
//!
//! The Rust port mirrors these constants and the [`resolve_drafter_kind`]
//! reconciliation semantics exactly. A third variant
//! [`DrafterKind::InternalMtp`] is added for the peer
//! (Qwen 3.5 / 3.6 built-in MTP head) per the in-issue amendment on;
//! see [`DrafterKind`] for details.
//!
//! ## Scope of this sub-issue
//!
//! This module ships **only** the trait, the kind enum, the auto-detector,
//! and the [`load_drafter`] factory shell. The concrete drafter
//! implementations land in later sub-issues:
//!
//! | Variant | Concrete impl | Wired by |
//! |---------|---------------|----------|
//! | [`DrafterKind::Mtp`] | `Gemma4AssistantDraftModel` | |
//! | [`DrafterKind::Dflash`] | `DFlashDraftModel` | |
//! | [`DrafterKind::InternalMtp`] | `InternalMtpDrafter` | |
//!
//! Until those land, [`load_drafter`] returns a typed
//! [`DrafterError::NotYetImplemented`] error pointing at the responsible
//! sub-issue, so calling code gets a clear actionable message instead of
//! an opaque `unimplemented!` panic.

pub mod masks;

use crate::ffi::MlxArray;
use crate::generate::LanguageModel;
use crate::layers::KVCache;
use crate::weights::WeightMap;
use cxx::UniquePtr;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

pub mod dflash;
/// Concrete Gemma 4 MTP "assistant" drafter implementation. Wired into
/// [`load_drafter`]'s `Mtp` arm.
pub mod gemma4_assistant;
/// Centroid-routed sparse softmax LM head used by Gemma 4 E2B / E4B
/// assistant drafters. Wired into `Gemma4AssistantDraftModel` in sub-3
/// — landed here independently so the layer can
/// be unit-tested in isolation before integration.
pub mod masked_embedder;

/// Drafter shapes recognised by mlxcel.
///
/// Each variant selects a fundamentally different round-loop driver and a
/// different concrete drafter implementation. The corresponding string
/// names (used on the CLI and in `config.json`) are exposed through
/// [`DrafterKind::as_str`] / [`DrafterKind::from_str`]:
///
/// - `"dflash"` — external Qwen 3.5 DFlash drafter (5-layer, own KV cache,
///   multi-layer hidden input, single masked forward, GDN-aware rollback).
/// - `"mtp"` — external Gemma 4 MTP "assistant" drafter (4-layer, shares
///   K/V from target, autoregressive draft block, per-row tail-zero
///   rollback).
/// - `"internal-mtp"` — built-in MTP head carried by Qwen 3.5 / 3.6
///   checkpoints as `mtp.layers.0.*` weights; no separate drafter
///   checkpoint required. Added for the peer.
///
/// The enum is marked `#[non_exhaustive]` so adding new drafter shapes in
/// follow-up epics does not break downstream `match` exhaustiveness
/// assertions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DrafterKind {
    /// External DFlash drafter (e.g. `z-lab/Qwen3.5-4B-DFlash`). Default
    /// fallback when no other rule applies — matches upstream
    /// `DEFAULT_DRAFTER_KIND = "dflash"`.
    Dflash,
    /// External MTP "assistant" drafter (e.g.
    /// `mlx-community/gemma-4-31B-it-assistant-bf16`). Auto-detected
    /// from `model_type == "gemma4_assistant"`.
    Mtp,
    /// Built-in MTP head living inside the target checkpoint
    /// (`mtp.layers.0.*` weights on Qwen 3.5 / 3.6). Auto-detected by
    /// checkpoint inspection sub-H, not by drafter
    /// `model_type`.
    InternalMtp,
}

impl DrafterKind {
    /// Canonical string name used on the CLI and in `config.json`.
    pub const fn as_str(self) -> &'static str {
        match self {
            DrafterKind::Dflash => "dflash",
            DrafterKind::Mtp => "mtp",
            DrafterKind::InternalMtp => "internal-mtp",
        }
    }
}

impl std::fmt::Display for DrafterKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parse a canonical drafter-kind name produced by [`DrafterKind::as_str`].
///
/// Implemented via the standard [`std::str::FromStr`] trait so CLI flag
/// parsing (sub-7) can use `"dflash".parse::<DrafterKind>()`
/// directly. Returns [`DrafterError::UnknownKind`] when the string does
/// not match any known variant.
impl std::str::FromStr for DrafterKind {
    type Err = DrafterError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dflash" => Ok(DrafterKind::Dflash),
            "mtp" => Ok(DrafterKind::Mtp),
            "internal-mtp" => Ok(DrafterKind::InternalMtp),
            other => Err(DrafterError::UnknownKind {
                got: other.to_string(),
                known: KNOWN_DRAFTER_KINDS.iter().map(|s| s.to_string()).collect(),
            }),
        }
    }
}

/// Set of drafter kinds known to mlxcel. Used by CLI help text and to
/// build the "known kinds" hint in [`DrafterError::UnknownKind`].
///
/// Mirrors upstream `KNOWN_DRAFTER_KINDS = {"dflash", "mtp"}` plus
/// `"internal-mtp"` from the amendment for peer.
pub const KNOWN_DRAFTER_KINDS: &[&str] = &["dflash", "mtp", "internal-mtp"];

/// Default drafter kind selected when the drafter's `config.json` does not
/// declare a recognised `model_type` and the caller did not pass an
/// explicit override. Matches upstream `DEFAULT_DRAFTER_KIND = "dflash"`.
///
/// This is the right default because the Qwen 3.5 DFlash drafter's
/// `DFlashConfig` does not declare a dedicated `model_type` field —
/// auto-detect must fall back to DFlash for it to work without an
/// explicit `--draft-kind dflash` flag.
pub const DEFAULT_DRAFTER_KIND: DrafterKind = DrafterKind::Dflash;

/// Static map from `config.json::model_type` to the required
/// [`DrafterKind`]. Mirrors upstream
/// `DRAFTER_KIND_BY_MODEL_TYPE = {"gemma4_assistant": "mtp",
/// "gemma4_unified_assistant": "mtp"}`.
///
/// Returned as `&'static HashMap` so call sites can perform `.get()`
/// without rebuilding the map on every call. Built lazily on first
/// access via [`OnceLock`].
pub fn drafter_kind_by_model_type() -> &'static HashMap<&'static str, DrafterKind> {
    static MAP: OnceLock<HashMap<&'static str, DrafterKind>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("gemma4_assistant", DrafterKind::Mtp);
        // The Gemma 4 Unified target ships its own MTP "assistant" drafter under
        // the `gemma4_unified_assistant` model_type. It loads with the same
        // `Gemma4AssistantDraftModel` class (identical centroid head + 4-layer
        // shared-K/V stack) and therefore resolves to the same MTP round loop.
        m.insert("gemma4_unified_assistant", DrafterKind::Mtp);
        m
    })
}

/// Errors that can occur during drafter resolution / loading.
///
/// Marked `#[non_exhaustive]` so adding new failure modes for future
/// drafter shapes (e.g. quantization mismatches for an MoE-flavored
/// drafter) does not break downstream `match` exhaustiveness.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DrafterError {
    /// User passed an unknown drafter kind string on the CLI.
    #[error("unknown drafter kind {got:?}; known: {}", known.join(", "))]
    UnknownKind { got: String, known: Vec<String> },

    /// I/O failure while peeking at the drafter's `config.json`. Treated as
    /// non-fatal by [`resolve_drafter_kind`]: an unreadable config falls
    /// back to [`DEFAULT_DRAFTER_KIND`] so the auto-detect path matches
    /// upstream's exception-swallowing behaviour exactly.
    #[error("failed to read drafter config at {path}: {source}")]
    ConfigIo {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// JSON parse failure on the drafter's `config.json`. Also non-fatal in
    /// [`resolve_drafter_kind`] — see [`DrafterError::ConfigIo`].
    #[error("failed to parse drafter config at {path}: {source}")]
    ConfigParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    /// A concrete drafter arm has not yet been wired into [`load_drafter`].
    /// The error message names the responsible sub-issue so callers can
    /// follow the trail.
    #[error("drafter kind {kind} is not yet implemented; tracked by issue #{issue}")]
    NotYetImplemented { kind: DrafterKind, issue: u32 },

    /// Weight loading or model construction failed (missing key,
    /// quantization mismatch, etc.). Carries the underlying reason for
    /// operator triage. Used by the DFlash drafter load path.
    #[error("drafter load failed: {reason}")]
    LoadFailed { reason: String },

    /// The drafter could not be bound to the supplied target (e.g. the
    /// target lacks an `embed_tokens` capability that the drafter
    /// requires).
    #[error("drafter bind failed: {reason}")]
    BindFailed { reason: String },

    /// `draft_block` could not complete (e.g. missing target hidden,
    /// out-of-range `block_size`, sampling failure).
    #[error("drafter draft_block failed: {reason}")]
    DraftFailed { reason: String },

    /// Config-layer error surfaced from
    /// [`Gemma4AssistantConfig::normalize`](crate::drafter::gemma4_assistant::Gemma4AssistantConfig::normalize)
    /// and friends. Wraps a free-form reason string to keep the
    /// concrete-drafter modules free of new `thiserror` variants.
    #[error("drafter config error: {0}")]
    Config(String),

    /// Failure while loading drafter weights (missing tensor, malformed
    /// safetensors, etc.). Used by the Gemma 4 assistant drafter
    /// [`Gemma4AssistantDraftModel::from_weights`](crate::drafter::gemma4_assistant::Gemma4AssistantDraftModel::from_weights)
    /// path. Sibling of [`Self::LoadFailed`], kept distinct so the
    /// two drafter families surface different operator hints.
    #[error("drafter weight load failed: {reason}")]
    WeightLoad { reason: String },

    /// Caller invoked [`Drafter::set_shared_kv`] with a tensor count outside
    /// the documented set. The drafter currently accepts 2 (full-only) or 4
    /// (full + sliding) tensors per [`SharedKv`]'s documented layout.
    #[error(
        "shared_kv has {got} tensors but the Gemma 4 drafter expects one of {expected:?}; \
         see the `SharedKv` doc for the canonical layout"
    )]
    SharedKvShape {
        got: usize,
        expected: &'static [usize],
    },

    /// Drafter encountered a layer with a `layer_type` string that does not
    /// map to any shared-K/V bucket the round-loop set up. The two known
    /// buckets are `"full_attention"` and `"sliding_attention"`.
    #[error(
        "drafter saw unknown layer_type {got:?}; expected full_attention or sliding_attention"
    )]
    UnknownLayerType { got: String },

    /// Drafter has a layer of the named layer-type but the round-loop did
    /// not supply matching shared K/V tensors. Typically means the target's
    /// shared K/V capture and the drafter's `layer_types` field
    /// are out of sync.
    #[error(
        "drafter layer needs shared K/V for layer_type {layer_type:?} but the round-loop \
         did not provide it; expected the target to capture both full_attention and \
         sliding_attention slabs"
    )]
    MissingSharedKvForLayerType { layer_type: String },

    /// The target language model does not expose a feature the drafter
    /// needs. Most commonly: `embed_tokens` was never overridden (the
    /// default trait impl returns `None`).
    #[error("target language model is missing required feature: {feature}")]
    TargetMissingFeature { feature: &'static str },

    /// [`Drafter::draft_block`] was called before [`Drafter::bind`]. The
    /// upstream Python code asserts the same precondition with a runtime
    /// error.
    #[error(
        "Gemma 4 assistant drafter requires bind(target_model) to be called before \
         draft_block() so the drafter can use the target's input embeddings"
    )]
    BindNotCalled,

    /// [`Drafter::draft_block`] was called before [`Drafter::set_shared_kv`].
    /// The MTP round-loop must arm the drafter with the target's shared K/V
    /// at the start of each draft block.
    #[error(
        "Gemma 4 assistant drafter requires the MTP round-loop, but no shared K/V was set \
         before draft_block() — this typically means the DFlash round-loop ran instead. \
         Pass --draft-kind mtp on the CLI (or MLX_VLM_DRAFT_KIND=mtp on the server)"
    )]
    SetSharedKvNotCalled,

    /// [`Drafter::draft_block`] requires a `hidden` input for the MTP path
    /// (the target's last hidden, projected through `post_projection` on
    /// subsequent steps). The trait signature uses `Option<&MlxArray>` so
    /// DFlash callers that never need this argument can pass `None`; the
    /// MTP path rejects `None` here.
    #[error("Gemma 4 assistant drafter requires `hidden` to be Some(_) for the MTP path")]
    DraftBlockMissingHidden,
}

/// Subset of the drafter's `config.json` that [`resolve_drafter_kind`]
/// needs. Unknown / extra fields are ignored.
#[derive(Debug, Deserialize)]
struct DrafterConfigPeek {
    #[serde(default)]
    model_type: Option<String>,
}

/// Read `model_path/config.json` and return the `model_type` field if it
/// exists. Returns `Ok(None)` when the file is missing OR unparseable —
/// this mirrors upstream's blanket `(FileNotFoundError, json.JSONDecodeError,
/// OSError) -> None` behaviour, which is load-bearing for the DFlash
/// fallback path (DFlash configs intentionally omit `model_type`).
fn peek_drafter_model_type(model_path: &Path) -> Result<Option<String>, DrafterError> {
    let cfg_path = model_path.join("config.json");
    let bytes = match fs::read(&cfg_path) {
        Ok(b) => b,
        // Treat any I/O error as "model_type unknown" — this matches the
        // upstream exception-swallowing semantics in `_peek_drafter_model_type`.
        Err(_) => return Ok(None),
    };
    match serde_json::from_slice::<DrafterConfigPeek>(&bytes) {
        Ok(peek) => Ok(peek.model_type),
        // Malformed JSON: also treated as unknown, same as upstream.
        Err(_) => Ok(None),
    }
}

/// Reconcile the caller's `kind` choice with the drafter's actual
/// `config.json::model_type`.
///
/// Semantics:
///
/// - `kind == None`: auto-detect. Read `model_type`. If it maps to a known
///   kind via [`drafter_kind_by_model_type`], return that. Otherwise
///   return [`DEFAULT_DRAFTER_KIND`].
/// - `kind == Some(k)` and the drafter's `model_type` maps to a different
///   kind `expected`: emit `tracing::warn!` and **honor the explicit
///   choice** (`k`). The warning surfaces the mismatch so an operator
///   debugging weird draft behaviour can immediately see they may have
///   the wrong drafter; the choice itself is left to the caller because
///   they presumably know what they're doing.
/// - `kind == Some(k)` and the drafter's `model_type` either matches `k`
///   or is unknown: return `k` unchanged. This is the path Qwen 3.5
///   DFlash takes (`DFlashConfig` has no dedicated `model_type` field).
///
/// ## Deviation from upstream
///
/// Upstream `mlx-vlm` *overrides* the user's choice when it disagrees
/// with `model_type`. mlxcel intentionally honors the explicit choice
/// instead, because:
///
/// 1. The explicit CLI flag (`--draft-kind`) is a clear user signal that
///    should not be silently rewritten — silent overrides make failures
///    harder to attribute.
/// 2. A warning gives the operator enough information to course-correct
///    on the next run if the explicit choice was a mistake.
/// 3. The override-on-mismatch path would mask checkpoint corruption or
///    a `model_type` field that has drifted from convention; honoring
///    the explicit choice makes such drift fail closer to its source.
pub fn resolve_drafter_kind(
    model_path: &Path,
    kind: Option<DrafterKind>,
) -> Result<DrafterKind, DrafterError> {
    let model_type = peek_drafter_model_type(model_path)?;
    let expected = model_type
        .as_deref()
        .and_then(|mt| drafter_kind_by_model_type().get(mt).copied());

    match (kind, expected) {
        (None, Some(exp)) => {
            tracing::info!(
                drafter = %model_path.display(),
                model_type = ?model_type,
                resolved = %exp,
                "Auto-detected --draft-kind from drafter model_type"
            );
            Ok(exp)
        }
        (None, None) => {
            tracing::info!(
                drafter = %model_path.display(),
                model_type = ?model_type,
                resolved = %DEFAULT_DRAFTER_KIND,
                "Auto-detected --draft-kind using default fallback (no \
                 dedicated model_type in drafter config)"
            );
            Ok(DEFAULT_DRAFTER_KIND)
        }
        (Some(user), Some(exp)) if exp != user => {
            tracing::warn!(
                drafter = %model_path.display(),
                model_type = ?model_type,
                expected = %exp,
                got = %user,
                "Explicit --draft-kind disagrees with drafter model_type; \
                 honoring the explicit choice (see resolve_drafter_kind \
                 docs for rationale)"
            );
            Ok(user)
        }
        (Some(user), _) => Ok(user),
    }
}

/// Shared K/V tensors handed to an MTP-style drafter by the target.
///
/// Borrowed (`&'a MlxArray`) rather than owned so the target retains
/// ownership of its KV cache contents — the drafter is forbidden from
/// mutating them in place. This is foundational scaffolding; the exact
/// shape of the shared K/V transfer is finalised by sub-2 (Gemma 4
/// target-side speculative hooks). Until then, the slice carries the
/// upstream `[k_full, v_full, k_swa, v_swa]` four-tensor convention
/// (Gemma 4's last full-attention + last sliding-window-attention layer
/// K/V pair) so a placeholder drafter can wire in without further changes
/// to the trait surface.
///
/// No-op for DFlash and InternalMtp (both have their own KV cache).
pub struct SharedKv<'a> {
    /// Borrowed shared K/V tensors from the target. Layout finalised.
    pub tensors: &'a [&'a MlxArray],
}

impl<'a> SharedKv<'a> {
    /// Convenience constructor for tests and future drafter impls.
    pub fn new(tensors: &'a [&'a MlxArray]) -> Self {
        Self { tensors }
    }
}

// Manual Debug impl because `MlxArray` is an opaque FFI type that does not
// derive `Debug`. We render only the tensor count, which is the only
// scalar metadata callers reliably want in log lines (the array bodies
// themselves are GPU-resident and not safe to read on the dispatch
// thread).
impl<'a> std::fmt::Debug for SharedKv<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedKv")
            .field("num_tensors", &self.tensors.len())
            .finish()
    }
}

/// Speculative drafter abstraction shared by MTP, DFlash, and InternalMtp
/// shapes.
///
/// The trait surface is **object-safe**: all returned references and
/// borrowed parameters use erased mlxcel-core types (`MlxArray`,
/// `KVCache`, `WeightMap`, `SamplingConfig`) rather than generic
/// associated types, so [`load_drafter`] can return `Box<dyn Drafter>`.
///
/// ### Method matrix
///
/// | Method                | MTP                              | DFlash                          | InternalMtp                     |
/// |-----------------------|----------------------------------|---------------------------------|---------------------------------|
/// | [`bind`]              | required                         | required                        | required                        |
/// | [`set_shared_kv`]     | required                         | no-op (default)                 | no-op (default)                 |
/// | [`set_shared_kv_batched`] | required for B>1 MTP         | no-op (default)                 | no-op (default)                 |
/// | [`make_cache`]        | empty vec (default)              | required (own KV cache)         | required (own KV cache)         |
/// | [`reset`]             | no-op (default)                  | required                        | required                        |
/// | [`draft_block`]       | K autoregressive forwards        | single masked forward           | K autoregressive forwards       |
/// | [`sanitize`]          | drop assistant-specific keys     | drop `mtp.*` from target ckpts  | pass-through (handled upstream) |
///
/// Default no-op methods let each concrete impl focus on only the
/// methods it actually overrides, while the round-loop drivers can call
/// the full surface uniformly without `match`-on-kind dispatch.
///
/// [`bind`]: Drafter::bind
/// [`set_shared_kv`]: Drafter::set_shared_kv
/// [`make_cache`]: Drafter::make_cache
/// [`reset`]: Drafter::reset
/// [`draft_block`]: Drafter::draft_block
/// [`sanitize`]: Drafter::sanitize
pub trait Drafter {
    /// Bind the drafter to its target for embed and LM-head resolution.
    ///
    /// Both shapes use this: MTP captures the target's `embed_tokens` and
    /// the bonus-token RoPE position so cross-attention queries can be
    /// rotated at the right absolute position; DFlash uses the target
    /// only to size the per-layer hidden concatenation. The trait takes
    /// `&dyn LanguageModel` so the caller can pass any concrete model
    /// (text or VLM) without monomorphisation.
    ///
    /// Returns `Err` if the target lacks a feature the drafter needs
    /// (e.g. no `embed_tokens` override for MTP).
    fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError>;

    /// Validate that this drafter is compatible with the supplied target
    /// **before** [`Self::bind`] is called.
    ///
    /// This is the dimension/vocabulary compatibility gate. The MTP
    /// assistant drafter consumes the target's last backbone hidden state
    /// concatenated with the target's token embeddings as a
    /// `2 × backbone_hidden_size` input, so its `backbone_hidden_size` MUST
    /// equal the target's text hidden size and its LM-head vocabulary MUST
    /// match the target vocabulary. A mismatched pairing (e.g. a 12B Unified
    /// target fed an assistant built for a different backbone) would either
    /// crash deep inside the first `draft_block` matmul or — worse — produce
    /// silently wrong drafts. Catching it here yields a clear, actionable
    /// operator error at dispatch time instead.
    ///
    /// The default implementation is a no-op (`Ok(())`); shapes that do not
    /// require a strict dimension match (DFlash, InternalMtp) keep the
    /// default. Concrete MTP drafters override this to compare
    /// `backbone_hidden_size` / vocab against the bound target.
    #[allow(unused_variables)]
    fn validate_target_compat(&self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        Ok(())
    }

    /// Inform the drafter of the target's freshly-captured shared K/V
    /// tensors at the start of a new draft block.
    ///
    /// **MTP-only.** Default implementation is a no-op so DFlash and
    /// InternalMtp do not need to override.
    ///
    /// - `shared_kv`: borrowed shared K/V tensors from the target's
    ///   last full-attention and last sliding-window layers.
    /// - `kv_offset`: absolute position offset of the shared K/V slice
    ///   in the target's KV cache (used to RoPE-rotate the drafter's
    ///   cross-attention queries at the bonus token's absolute position).
    /// - `position`: absolute position of the bonus token whose
    ///   prediction the drafter is extending.
    /// - `left_padding`: per-row left-padding extent in the shared K/V
    ///   (used by the batched MTP path; B=1 callers pass 0).
    #[allow(unused_variables)]
    fn set_shared_kv(
        &mut self,
        shared_kv: SharedKv<'_>,
        kv_offset: usize,
        position: usize,
        left_padding: usize,
    ) -> Result<(), DrafterError> {
        Ok(())
    }

    /// Batched variant of [`Self::set_shared_kv`] that preserves per-row
    /// cache positions and padding metadata.
    ///
    /// **MTP-only.** The default collapses per-row metadata to scalar
    /// maxima and calls [`Self::set_shared_kv`], preserving compatibility
    /// for non-batched drafters and tests. Concrete B>1 MTP drafters should
    /// override this so masks and RoPE anchors use each row's logical
    /// position instead of the longest row's value.
    ///
    /// - `kv_offset_per_row`: logical per-row post-rollback cache offsets.
    /// - `position_per_row`: per-row frozen bonus-token RoPE anchors.
    /// - `kv_valid_len_per_row`: per-row valid prefix lengths in the shared
    ///   K/V slabs.
    /// - `left_padding_per_row`: per-row left-padding extents in the shared
    ///   K/V slabs.
    #[allow(unused_variables)]
    fn set_shared_kv_batched(
        &mut self,
        shared_kv: SharedKv<'_>,
        kv_offset_per_row: &[usize],
        position_per_row: &[usize],
        kv_valid_len_per_row: &[usize],
        left_padding_per_row: &[usize],
    ) -> Result<(), DrafterError> {
        let kv_offset = kv_offset_per_row.iter().copied().max().unwrap_or(0);
        let position = position_per_row.iter().copied().max().unwrap_or(0);
        let left_padding = left_padding_per_row.iter().copied().max().unwrap_or(0);
        self.set_shared_kv(shared_kv, kv_offset, position, left_padding)
    }

    /// Create the drafter's own KV cache (one slot per drafter layer).
    ///
    /// **DFlash- and InternalMtp-only.** Default returns an empty `Vec`
    /// because MTP has no own KV cache — its only recurrent state is the
    /// target's last hidden, projected through `post_projection`.
    fn make_cache(&self) -> Vec<KVCache> {
        Vec::new()
    }

    /// Reset the drafter's own KV cache between full generation calls.
    ///
    /// **DFlash- and InternalMtp-only.** Default no-op for MTP.
    #[allow(unused_variables)]
    fn reset(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        Ok(())
    }

    /// Target-layer hidden states this drafter expects for DFlash.
    ///
    /// DFlash checkpoints carry their own `target_layer_ids` in
    /// `config.json`; larger Qwen 3.5 drafts use a different capture list
    /// than the 4B default. The round-loop and server prefill paths consult
    /// this optional slice to pass the checkpoint-specific list through the
    /// target verify hooks. Non-DFlash drafters return `None`.
    fn dflash_target_layer_ids(&self) -> Option<&[usize]> {
        None
    }

    /// Drafter checkpoint's configured speculative block size, when known.
    ///
    /// MTP uses this to mirror upstream adaptive block sizing: a user may
    /// request a larger `--draft-block-size`, but Gemma-style assistants
    /// start at their configured depth and only expand after recent
    /// acceptance proves the configured prefix is usually fully accepted.
    fn configured_block_size(&self) -> Option<usize> {
        None
    }

    /// Whether the drafter should always honor a user-requested block size
    /// instead of using the adaptive configured-depth warm-up.
    ///
    /// Upstream Qwen 3.5 MTP sets `prefer_requested_block_size = True`.
    /// Gemma 4 assistant leaves this false.
    fn prefer_requested_block_size(&self) -> bool {
        false
    }

    /// Produce a draft block of proposal tokens.
    ///
    /// Semantics are kind-specific:
    ///
    /// - **MTP**: `K = block_size` autoregressive small forwards. `hidden`
    ///   carries the target's last hidden, used as the single recurrent
    ///   state alongside the bonus token's embedding. Returns
    ///   `block_size` proposal tokens.
    /// - **DFlash**: a single masked forward with `block_size`
    ///   placeholder positions. `hidden` carries the multi-layer
    ///   concatenation of the target's captured layer hiddens (e.g.,
    ///   layers `[1, 8, 15, 22, 29]` for Qwen 3.5). Returns
    ///   `block_size - 1` proposal tokens (the first masked position is
    ///   used as scaffolding for the rest of the block).
    /// - **InternalMtp**: K autoregressive forwards driven by the
    ///   target's built-in `mtp.layers.0.*` head. `hidden` is the
    ///   target's last hidden.
    ///
    /// - `last_bonus`: the verified bonus token whose prediction the
    ///   drafter is extending (the right-most token from the previous
    ///   round).
    /// - `hidden`: optional target-hidden input. `None` is permitted at
    ///   bring-up time for tests; concrete drafters reject `None` if
    ///   their kind requires the hidden state.
    /// - `block_size`: caller's target draft block length.
    /// - `sampler`: sampling configuration applied to each proposal step.
    ///
    /// Note: token type is `i32` to match
    /// [`LanguageModel::eos_token_ids`] / `SpeculativeGenerator` /
    /// `generated_tokens: Vec<i32>` throughout the crate.
    fn draft_block(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &crate::generate::SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError>;

    /// Produce a draft block as a device-side token array.
    ///
    /// DFlash can feed proposal tokens directly into the target verify graph,
    /// matching the upstream `mx.concatenate([bonus, draft_tokens])` path and
    /// avoiding an early device→host synchronization before verify. The
    /// default preserves compatibility for other drafters by falling back to
    /// [`Self::draft_block`] and rebuilding a `[1, K]` int32 array.
    ///
    /// Used by: DFlash round loop; non-DFlash drafters use the default bridge.
    fn draft_block_array(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &crate::generate::SamplingConfig,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let tokens = self.draft_block(last_bonus, hidden, block_size, sampler)?;
        Ok(crate::ffi::from_slice_i32(
            &tokens,
            &[1, tokens.len() as i32],
        ))
    }

    /// Produce a batched draft block of proposal tokens (B > 1).
    ///
    /// Mirrors [`Self::draft_block`] but accepts a per-row bonus slice and
    /// a `[B, T, dim]` hidden tensor. Returns a per-row proposal matrix:
    /// `out[r]` is the proposal sequence for row `r`, with length
    /// `block_size - 1` (DFlash) or `block_size` (MTP / InternalMtp).
    ///
    /// ** (DFlash B>1 path)**: implemented by `DFlashDrafter`.
    /// MTP / InternalMtp drafters return [`DrafterError::DraftFailed`] by
    /// default — their batched paths land in their own respective sub-issues.
    ///
    /// - `last_bonus`: per-row bonus tokens, length `B`.
    /// - `hidden`: optional target-hidden input with leading batch dim
    ///   `B`. `None` is permitted at bring-up time for tests; concrete
    ///   batched drafters reject `None` if their kind requires the
    ///   hidden state.
    /// - `block_size`: caller's target draft block length.
    /// - `sampler`: sampling configuration applied to each proposal step.
    ///
    /// Default implementation returns
    /// [`DrafterError::DraftFailed`] with a "batched-not-implemented"
    /// message so drafters opt in to batched support explicitly.
    #[allow(unused_variables)]
    fn draft_block_batched(
        &mut self,
        last_bonus: &[i32],
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &crate::generate::SamplingConfig,
    ) -> Result<Vec<Vec<i32>>, DrafterError> {
        Err(DrafterError::DraftFailed {
            reason: format!(
                "draft_block_batched not implemented for kind = {:?}; \
                 B > 1 path requires per-drafter override",
                self.kind()
            ),
        })
    }

    /// Drop weight keys that this drafter kind must not carry into
    /// runtime (mutates `weights` in place).
    ///
    /// Examples by kind:
    ///
    /// - **MTP**: drop assistant-specific keys that the upstream
    ///   `Gemma4AssistantDraftModel.sanitize` removes.
    /// - **DFlash**: drop the target's `mtp.*` keys when reusing a
    ///   Qwen 3.5 / 3.6 checkpoint that carries an internal MTP head
    ///   the runtime path is not going to use. Matches
    ///   https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py#L308-L313
    ///   (`weights.pop("lm_head.weight", None)` and friends).
    /// - **InternalMtp**: pass-through. The actual `mtp.*` extraction
    ///   happens sub-B *before* the drafter sees
    ///   the weight map.
    fn sanitize(&mut self, weights: &mut WeightMap) -> Result<(), DrafterError>;

    /// Returns the kind of this drafter. Useful for diagnostic logging
    /// and for round-loop dispatch when the driver does not already know
    /// which kind it received from [`load_drafter`].
    fn kind(&self) -> DrafterKind;
}

/// Returned by [`load_drafter`].
///
/// Yields a boxed trait object plus the resolved [`DrafterKind`]. Callers
/// should use the returned kind for downstream dispatch rather than
/// trusting any original kind argument, because [`resolve_drafter_kind`]
/// may have overridden the user's choice.
pub type LoadedDrafter = (Box<dyn Drafter>, DrafterKind);

/// Factory entrypoint: load a drafter from a model directory, reconciling
/// the caller's optional `kind` with the drafter's `config.json`.
///
/// The signature is final and downstream sub-issues
/// fill in their concrete arms in this function. Until those arms land,
/// each variant returns [`DrafterError::NotYetImplemented`] referencing
/// the responsible sub-issue, so the round-loop driver and the CLI
/// flag plumbing (sub-7) can wire against this signature today.
///
/// Auto-detection (`kind == None`) is delegated to
/// [`resolve_drafter_kind`].
pub fn load_drafter(path: &Path, kind: Option<DrafterKind>) -> Result<LoadedDrafter, DrafterError> {
    let resolved = resolve_drafter_kind(path, kind)?;
    match resolved {
        DrafterKind::Dflash => {
            // Wired in by load weights, sanitize, build the model,
            // hand back the boxed trait object.
            let drafter = dflash::drafter::DFlashDrafter::load(path)?;
            Ok((Box::new(drafter), resolved))
        }
        DrafterKind::Mtp => {
            // Wired in — Gemma 4 MTP assistant drafter.
            let model = gemma4_assistant::Gemma4AssistantDraftModel::from_path(path)?;
            Ok((Box::new(model), resolved))
        }
        // Remaining variant lands in peer — returning a typed
        // error rather than `unimplemented!()` here gives users an
        // actionable hint instead of a panic.
        DrafterKind::InternalMtp => Err(DrafterError::NotYetImplemented {
            kind: resolved,
            issue: 640,
        }),
    }
}

impl std::fmt::Debug for dyn Drafter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Drafter")
            .field("kind", &self.kind())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr as _;
    use tempfile::tempdir;
    use tracing_test::traced_test;

    /// Helper: write a `config.json` with the given `model_type` into a
    /// fresh temp dir and return its path. Mirrors the smallest-possible
    /// fixture the upstream `_peek_drafter_model_type` consumes.
    fn write_drafter_config(dir: &tempfile::TempDir, model_type: Option<&str>) {
        let content = match model_type {
            Some(mt) => format!(r#"{{"model_type": "{mt}"}}"#),
            None => "{}".to_string(),
        };
        fs::write(dir.path().join("config.json"), content).expect("write config.json");
    }

    // ----- DrafterKind round-tripping --------------------------------------

    #[test]
    fn drafter_kind_string_roundtrip_for_all_variants() {
        for &kind in &[
            DrafterKind::Dflash,
            DrafterKind::Mtp,
            DrafterKind::InternalMtp,
        ] {
            let s = kind.as_str();
            assert_eq!(
                DrafterKind::from_str(s).expect("from_str"),
                kind,
                "round-trip via {s:?}"
            );
        }
    }

    #[test]
    fn drafter_kind_from_str_rejects_unknown() {
        let err = DrafterKind::from_str("bogus").expect_err("must reject");
        match err {
            DrafterError::UnknownKind { got, known } => {
                assert_eq!(got, "bogus");
                assert!(known.iter().any(|s| s == "dflash"));
                assert!(known.iter().any(|s| s == "mtp"));
                assert!(known.iter().any(|s| s == "internal-mtp"));
            }
            other => panic!("expected UnknownKind, got {other:?}"),
        }
    }

    #[test]
    fn known_drafter_kinds_match_canonical_strings() {
        // Every variant's canonical name must appear in KNOWN_DRAFTER_KINDS
        // so CLI help text stays in sync with the enum.
        for &kind in &[
            DrafterKind::Dflash,
            DrafterKind::Mtp,
            DrafterKind::InternalMtp,
        ] {
            assert!(
                KNOWN_DRAFTER_KINDS.contains(&kind.as_str()),
                "{} missing from KNOWN_DRAFTER_KINDS",
                kind.as_str()
            );
        }
    }

    #[test]
    fn default_drafter_kind_matches_upstream_dflash() {
        assert_eq!(DEFAULT_DRAFTER_KIND, DrafterKind::Dflash);
    }

    #[test]
    fn drafter_kind_by_model_type_maps_gemma4_assistant_to_mtp() {
        let map = drafter_kind_by_model_type();
        assert_eq!(map.get("gemma4_assistant"), Some(&DrafterKind::Mtp));
        // The Gemma 4 Unified assistant resolves to the same MTP round loop.
        assert_eq!(map.get("gemma4_unified_assistant"), Some(&DrafterKind::Mtp));
        // Both assistant spellings, nothing else: parity with upstream
        // `DRAFTER_KIND_BY_MODEL_TYPE = {"gemma4_assistant": "mtp",
        // "gemma4_unified_assistant": "mtp"}`.
        assert_eq!(map.len(), 2);
    }

    // ----- resolve_drafter_kind -------------------------------------------

    #[test]
    fn auto_detect_gemma4_assistant_resolves_to_mtp() {
        let dir = tempdir().unwrap();
        write_drafter_config(&dir, Some("gemma4_assistant"));
        let resolved = resolve_drafter_kind(dir.path(), None).unwrap();
        assert_eq!(resolved, DrafterKind::Mtp);
    }

    #[test]
    fn auto_detect_gemma4_unified_assistant_resolves_to_mtp() {
        // The Gemma 4 Unified 12B drafter ships model_type
        // `gemma4_unified_assistant`; it must auto-detect to the MTP round
        // loop exactly like the non-unified `gemma4_assistant`.
        let dir = tempdir().unwrap();
        write_drafter_config(&dir, Some("gemma4_unified_assistant"));
        let resolved = resolve_drafter_kind(dir.path(), None).unwrap();
        assert_eq!(resolved, DrafterKind::Mtp);
    }

    #[test]
    fn auto_detect_unknown_model_type_falls_back_to_dflash_default() {
        // DFlash config.json intentionally omits `model_type`, so the
        // resolver MUST fall back to DEFAULT_DRAFTER_KIND (Dflash).
        let dir = tempdir().unwrap();
        write_drafter_config(&dir, None);
        let resolved = resolve_drafter_kind(dir.path(), None).unwrap();
        assert_eq!(resolved, DrafterKind::Dflash);
    }

    #[test]
    fn auto_detect_unrecognised_model_type_falls_back_to_dflash_default() {
        // Some random model_type the map doesn't know about -> still DFlash.
        let dir = tempdir().unwrap();
        write_drafter_config(&dir, Some("some_unknown_model_type_v2"));
        let resolved = resolve_drafter_kind(dir.path(), None).unwrap();
        assert_eq!(resolved, DrafterKind::Dflash);
    }

    #[test]
    fn auto_detect_missing_config_falls_back_to_dflash_default() {
        // No config.json at all: upstream swallows the FileNotFoundError
        // and falls back to DEFAULT_DRAFTER_KIND; we must do the same.
        let dir = tempdir().unwrap();
        let resolved = resolve_drafter_kind(dir.path(), None).unwrap();
        assert_eq!(resolved, DrafterKind::Dflash);
    }

    #[test]
    fn auto_detect_malformed_config_falls_back_to_dflash_default() {
        // Garbage that is not valid JSON. Upstream swallows
        // json.JSONDecodeError; we must do the same so a corrupt
        // drafter dir does not break auto-detect.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.json"), "not valid json").unwrap();
        let resolved = resolve_drafter_kind(dir.path(), None).unwrap();
        assert_eq!(resolved, DrafterKind::Dflash);
    }

    #[test]
    fn explicit_kind_passes_through_when_model_type_agrees() {
        // model_type == "gemma4_assistant", caller passes Mtp -> Mtp.
        let dir = tempdir().unwrap();
        write_drafter_config(&dir, Some("gemma4_assistant"));
        let resolved = resolve_drafter_kind(dir.path(), Some(DrafterKind::Mtp)).unwrap();
        assert_eq!(resolved, DrafterKind::Mtp);
    }

    #[test]
    fn explicit_kind_passes_through_when_model_type_is_unmapped() {
        // model_type not in the map -> trust the explicit kind. This is
        // the DFlash path (DFlash configs have no dedicated model_type).
        let dir = tempdir().unwrap();
        write_drafter_config(&dir, None);
        let resolved = resolve_drafter_kind(dir.path(), Some(DrafterKind::Dflash)).unwrap();
        assert_eq!(resolved, DrafterKind::Dflash);

        // Also: an explicit Mtp against an unmapped config must pass
        // through unchanged (no "expected" to override against).
        let resolved = resolve_drafter_kind(dir.path(), Some(DrafterKind::Mtp)).unwrap();
        assert_eq!(resolved, DrafterKind::Mtp);
    }

    #[test]
    #[traced_test]
    fn warn_and_honor_explicit_kind_when_disagreeing_with_model_type() {
        // model_type == "gemma4_assistant" maps to Mtp, but the caller
        // explicitly asked for DFlash. Resolver MUST honor the explicit
        // choice (DFlash) and emit a `tracing::warn!` so the operator
        // sees the mismatch. This pins the acceptance
        // criterion verbatim: "when the caller passes
        // Some(DrafterKind::Dflash) but the model_type says
        // gemma4_assistant, the resolver returns Dflash and emits a
        // tracing::warn!".
        let dir = tempdir().unwrap();
        write_drafter_config(&dir, Some("gemma4_assistant"));
        let resolved = resolve_drafter_kind(dir.path(), Some(DrafterKind::Dflash)).unwrap();
        assert_eq!(
            resolved,
            DrafterKind::Dflash,
            "explicit choice must be honored"
        );
        assert!(logs_contain(
            "Explicit --draft-kind disagrees with drafter model_type"
        ));
    }

    #[test]
    #[traced_test]
    fn warn_also_fires_when_explicit_mtp_disagrees_against_unmapped_inverse() {
        // This is the symmetric mismatch: caller passes Mtp but the
        // drafter's model_type maps to something else. Per the design
        // note in resolve_drafter_kind, the warn fires only when the
        // model_type maps to a *known* kind that differs from the
        // caller's. If model_type is unknown / absent, no warn fires
        // because there is nothing to disagree with. This test pins
        // that boundary.
        let dir = tempdir().unwrap();
        write_drafter_config(&dir, None);
        let resolved = resolve_drafter_kind(dir.path(), Some(DrafterKind::Mtp)).unwrap();
        assert_eq!(resolved, DrafterKind::Mtp);
        assert!(
            !logs_contain("Explicit --draft-kind disagrees"),
            "no warn should fire when model_type is unknown / absent"
        );
    }

    #[test]
    #[traced_test]
    fn auto_detect_emits_info_log_for_default_fallback() {
        let dir = tempdir().unwrap();
        write_drafter_config(&dir, None);
        let _ = resolve_drafter_kind(dir.path(), None).unwrap();
        assert!(logs_contain(
            "Auto-detected --draft-kind using default fallback"
        ));
    }

    // ----- load_drafter ---------------------------------------------------

    #[test]
    fn load_drafter_dflash_fails_without_weights_with_typed_load_error() {
        // The stub `config.json` is present but no safetensors files
        // accompany it. `DFlashDrafter::load` must surface a typed
        // `LoadFailed` (not `NotYetImplemented` — DFlash is wired in). Pin this to make sure a future re-stub of the
        // `Dflash` arm cannot silently regress to `NotYetImplemented`.
        let dir = tempdir().unwrap();
        write_drafter_config(&dir, None);
        let err = load_drafter(dir.path(), Some(DrafterKind::Dflash))
            .expect_err("load_drafter must fail on a config-only fixture with no safetensors");
        match err {
            DrafterError::LoadFailed { reason } => {
                // Reason is implementation-defined; the typed variant
                // is what matters for the contract.
                assert!(!reason.is_empty(), "LoadFailed reason must not be empty");
            }
            DrafterError::NotYetImplemented { .. } => {
                panic!("DFlash must NOT be NotYetImplemented after the drafter lands");
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
    }

    #[test]
    fn load_drafter_routes_mtp_to_gemma4_assistant() {
        // The Mtp arm is wired. With a fixture that has the
        // gemma4_assistant `model_type` but no full text_config, the
        // factory should reach into Gemma4AssistantDraftModel::from_path
        // and fail at config parsing (no text_config) rather than the
        // earlier NotYetImplemented short-circuit. That proves the arm
        // now dispatches into the concrete impl.
        let dir = tempdir().unwrap();
        write_drafter_config(&dir, Some("gemma4_assistant"));
        let err = load_drafter(dir.path(), None).expect_err("stub config has no text_config");
        match err {
            DrafterError::NotYetImplemented { .. } => panic!(
                "Mtp arm should no longer return NotYetImplemented; \
 wired the concrete loader"
            ),
            DrafterError::ConfigParse { .. } | DrafterError::Config(_) => {
                // Expected: factory got past the stub gate and into the
                // concrete loader, where the bare fixture fails parse /
                // normalize.
            }
            other => panic!("expected ConfigParse / Config after Mtp dispatch, got {other:?}"),
        }
    }

    #[test]
    fn load_drafter_returns_typed_not_yet_implemented_for_internal_mtp() {
        let dir = tempdir().unwrap();
        write_drafter_config(&dir, None);
        let err = load_drafter(dir.path(), Some(DrafterKind::InternalMtp)).expect_err("stub");
        match err {
            DrafterError::NotYetImplemented { kind, issue } => {
                assert_eq!(kind, DrafterKind::InternalMtp);
                assert_eq!(issue, 640);
            }
            other => panic!("expected NotYetImplemented, got {other:?}"),
        }
    }

    // ----- Trait object-safety --------------------------------------------

    /// Compile-time assertion that [`Drafter`] is object-safe — i.e. we
    /// can hold a `Box<dyn Drafter>`. If a future trait edit accidentally
    /// adds a generic method or a Self-by-value method, this will fail
    /// to compile and fence the regression at the trait boundary
    /// instead of at every call site. This is the foundational
    /// invariant the rest depends on.
    #[test]
    fn drafter_trait_is_object_safe() {
        struct StubDrafter;
        impl Drafter for StubDrafter {
            fn bind(&mut self, _t: &dyn LanguageModel) -> Result<(), DrafterError> {
                Ok(())
            }
            fn draft_block(
                &mut self,
                _last_bonus: i32,
                _hidden: Option<&MlxArray>,
                _block_size: usize,
                _sampler: &crate::generate::SamplingConfig,
            ) -> Result<Vec<i32>, DrafterError> {
                Ok(Vec::new())
            }
            fn sanitize(&mut self, _w: &mut WeightMap) -> Result<(), DrafterError> {
                Ok(())
            }
            fn kind(&self) -> DrafterKind {
                DrafterKind::Dflash
            }
        }

        // The cast itself is the object-safety check. If `Drafter` is not
        // object-safe, `Box::new(StubDrafter) as Box<dyn Drafter>` does
        // not compile.
        let boxed: Box<dyn Drafter> = Box::new(StubDrafter);
        assert_eq!(boxed.kind(), DrafterKind::Dflash);
    }

    /// Verify that default no-op methods on the trait actually work
    /// without an override. This is the contract that lets each concrete
    /// drafter only implement the methods it cares about.
    #[test]
    fn default_no_op_methods_are_safe_to_call() {
        struct MinimalDrafter;
        impl Drafter for MinimalDrafter {
            fn bind(&mut self, _t: &dyn LanguageModel) -> Result<(), DrafterError> {
                Ok(())
            }
            fn draft_block(
                &mut self,
                _last_bonus: i32,
                _hidden: Option<&MlxArray>,
                _block_size: usize,
                _sampler: &crate::generate::SamplingConfig,
            ) -> Result<Vec<i32>, DrafterError> {
                Ok(Vec::new())
            }
            fn sanitize(&mut self, _w: &mut WeightMap) -> Result<(), DrafterError> {
                Ok(())
            }
            fn kind(&self) -> DrafterKind {
                DrafterKind::Mtp
            }
        }

        let mut d = MinimalDrafter;
        // make_cache default = empty Vec
        assert!(d.make_cache().is_empty());
        // Default no-ops do not need a real LanguageModel; we can't
        // construct one in this unit test without pulling the FFI, so
        // the contract is exercised in the trait object-safety check
        // above plus the make_cache assertion here.
        let _ = &mut d;
    }
}
