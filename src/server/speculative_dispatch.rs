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

//! Server-side speculative-decoding dispatch resolution.
//!
//! When `mlxcel-server` is started with the `--draft-model` flag (plus the
//! optional `--draft-kind` / `--draft-block-size` overrides from the
//! [`speculative_args`](crate::cli::speculative_args) flag group), this
//! module resolves the operator-facing inputs into a typed
//! [`SpeculativeDispatch`] value that the continuous-batching scheduler
//! consumes at request time.
//!
//! ## Where this fits in the dispatch matrix
//!
//! The CLI flags land in [`crate::server::config::ServerConfig`] verbatim:
//!
//! - [`ServerConfig::draft_model_path`](crate::server::config::ServerConfig::draft_model_path)
//!   — `--model-draft` / `--draft-model` value.
//! - [`ServerConfig::draft_kind`](crate::server::config::ServerConfig::draft_kind)
//!   — raw `--draft-kind` string (`Some("dflash")` / `Some("mtp")` / `None`
//!   for auto-detect).
//! - [`ServerConfig::draft_block_size`](crate::server::config::ServerConfig::draft_block_size)
//!   — explicit block-size override; `None` resolves to the per-kind default
//!   via [`crate::cli::speculative_args::default_block_size_for_kind`].
//!
//! At worker-startup, [`SpeculativeDispatch::resolve`] turns those raw
//! fields into one of:
//!
//! - [`SpeculativeDispatch::Disabled`] — no `--draft-model`; classic decode
//!   path runs ('s [`crate::SpeculativeGenerator`] is itself NOT used by the server today, only by the offline `mlxcel generate` path).
//! - [`SpeculativeDispatch::Classic { .. }`] — `--draft-model` set,
//!   `--draft-kind` unset, drafter's `config.json::model_type` resolves to
//!   a kind that the server has not yet wired to its kind-specific round
//!   loop. The server logs the auto-detected kind and falls back to the
//!   classic [`crate::SpeculativeGenerator`] dispatch path for backward
//!   compatibility with the historical `--draft-model <path>` workflow.
//! - [`SpeculativeDispatch::Mtp { .. }`] — `--draft-kind mtp` (or auto-
//!   detect resolved to MTP). The scheduler constructs the kind-specific
//!   [`mlxcel_core::speculative::mtp::MtpGenerator`] (B=1) or
//!   [`mlxcel_core::speculative::mtp::MtpBatchedGenerator`] (B>1) per
//!   request.
//! - [`SpeculativeDispatch::DFlash { .. }`] — `--draft-kind dflash` (or
//!   auto-detect resolved to DFlash and operator explicitly opted in via
//!   `--draft-kind`). The scheduler constructs the kind-specific
//!   [`mlxcel_core::drafter::dflash::DFlashGenerator`] (B=1) or
//!   [`mlxcel_core::drafter::dflash::DFlashBatchedGenerator`] (B>1) per
//!   request.
//!
//! ## Why resolution happens once, at startup
//!
//! The drafter's `config.json` parse and the kind reconciliation are the
//! expensive parts; doing them per-request would re-read the file on every
//! `/v1/chat/completions` call. Resolution at worker startup amortizes the
//! cost and surfaces invalid configurations *before* any HTTP traffic
//! reaches the scheduler.
//!
//! ## What this module does NOT do
//!
//! - It does not load the drafter weights. The scheduler is responsible for
//!   that, because the drafter must be loaded on the same thread as the
//!   target (MLX is not thread-safe across stream boundaries) and the
//!   scheduler owns the worker thread.
//! - It does not construct the round-loop driver. That happens at request
//!   time when the scheduler has the per-sequence cache slot for the target.
//!
//! ## Reference
//!
//! Mirrors the offline-CLI dispatch in `src/commands/generate.rs` lines
//! 666–770. The same `resolve_drafter_kind` + `resolve_draft_block_size`
//! helpers are reused so the offline and server paths cannot drift.

use std::path::PathBuf;

use mlxcel_core::drafter::{DrafterKind, resolve_drafter_kind};

use crate::cli::speculative_args::resolve_draft_block_size;
use crate::server::config::ServerConfig;

/// Resolved speculative-decoding dispatch shape.
///
/// Constructed once at worker startup by [`Self::resolve`]. The scheduler
/// holds this enum for the lifetime of the worker thread; per-request
/// dispatch is a single match-arm on this value with no further IO.
#[derive(Debug, Clone)]
pub enum SpeculativeDispatch {
    /// No drafter configured. Scheduler runs the classic per-step decode
    /// loop. This is the default and the bit-exact preserved path.
    Disabled,

    /// Drafter is configured but its resolved kind is one the server has
    /// not yet wired to a kind-specific round loop. Scheduler falls back
    /// to the classic [`crate::SpeculativeGenerator`] dispatch (the same
    /// path the offline `mlxcel generate` command takes when
    /// `--draft-kind` is unset and the auto-detect resolves to a known
    /// kind). The auto-detected kind is logged for operator visibility.
    Classic {
        /// Drafter checkpoint directory.
        draft_model_path: PathBuf,
        /// Number of draft tokens per speculation step (from
        /// `--num-draft-tokens` / `ServerConfig::num_draft_tokens`).
        num_draft_tokens: usize,
        /// The auto-detected drafter kind. Stored for diagnostic logging
        /// only — the classic generator does not branch on it.
        auto_detected_kind: DrafterKind,
    },

    /// MTP speculative decoding (Gemma 4 assistant drafter family).
    /// Scheduler constructs a kind-specific
    /// [`mlxcel_core::speculative::mtp::MtpGenerator`] (B=1) or
    /// [`mlxcel_core::speculative::mtp::MtpBatchedGenerator`] (B>1) per
    /// request, after the target's prefill completes.
    Mtp {
        /// Drafter checkpoint directory.
        draft_model_path: PathBuf,
        /// Resolved draft block size (the verify input has `block_size`
        /// positions: `[bonus, draft_0, …, draft_{K-2}]`).
        block_size: u32,
        /// Whether the kind was explicitly requested via `--draft-kind`
        /// (true) or auto-detected from the drafter's `config.json`
        /// (false). Affects the operator-facing error message when the
        /// target model does not implement
        /// [`mlxcel_core::speculative::mtp::target::MtpTarget`].
        user_requested_explicit_kind: bool,
    },

    /// DFlash speculative decoding (Qwen 3.5 DFlash drafter family).
    /// Scheduler constructs a kind-specific
    /// [`mlxcel_core::drafter::dflash::DFlashGenerator`] (B=1) or
    /// [`mlxcel_core::drafter::dflash::DFlashBatchedGenerator`] (B>1)
    /// per request, after the target's prefill completes.
    DFlash {
        /// Drafter checkpoint directory.
        draft_model_path: PathBuf,
        /// Resolved draft block size — the drafter's masked forward
        /// produces `block_size - 1` proposal tokens per round.
        block_size: u32,
        /// Whether the kind was explicitly requested via `--draft-kind`
        /// (true) or auto-detected from the drafter's `config.json`
        /// (false). Same operator-facing semantics as
        /// [`Self::Mtp::user_requested_explicit_kind`].
        user_requested_explicit_kind: bool,
    },
}

/// Error variants for [`SpeculativeDispatch::resolve`].
///
/// All variants carry the **operator-facing** error message verbatim so
/// the worker's startup log can surface it without re-formatting. The
/// inner messages are stable and unit-tested.
#[derive(Debug, thiserror::Error)]
pub enum SpeculativeDispatchError {
    /// `--draft-kind` was passed an unrecognized value (not one of
    /// `KNOWN_DRAFTER_KINDS - {internal-mtp}`).
    #[error("invalid --draft-kind: {message}")]
    InvalidKind { message: String },

    /// `--draft-model` was supplied but the drafter's `config.json` could
    /// not be read or parsed (or its `model_type` could not be mapped to
    /// a known [`DrafterKind`]).
    #[error("drafter config error at {path}: {message}")]
    DrafterConfig { path: PathBuf, message: String },
}

impl SpeculativeDispatch {
    /// Resolve the speculative dispatch from a [`ServerConfig`].
    ///
    /// Returns [`SpeculativeDispatch::Disabled`] when no drafter is
    /// configured. Otherwise:
    ///
    /// 1. Parses `config.draft_kind` (the raw `--draft-kind` string) into
    ///    an `Option<DrafterKind>`. An unparseable string produces
    ///    [`SpeculativeDispatchError::InvalidKind`].
    /// 2. Auto-detects the drafter kind from the drafter's `config.json`
    ///    via [`resolve_drafter_kind`], reconciling against the explicit
    ///    kind (if any). A config-file failure produces
    ///    [`SpeculativeDispatchError::DrafterConfig`].
    /// 3. Resolves the effective block size via
    ///    [`resolve_draft_block_size`] (using the per-kind default when
    ///    `--draft-block-size` is unset).
    /// 4. Returns the kind-specific variant.
    ///
    /// The classic [`Self::Classic`] arm is selected only when the
    /// operator did **not** pass `--draft-kind` and the auto-detect
    /// resolved to [`DrafterKind::InternalMtp`] (which is not currently
    /// wired to a server-side round loop). This preserves the historical
    /// `--draft-model <path>` workflow for older operator scripts that
    /// pre-date the `--draft-kind` flag.
    ///
    /// **Note**: when `--draft-kind` is explicitly passed AND the kind is
    /// supported by a server-side round loop, this method returns the
    /// kind-specific variant. The scheduler is then responsible for
    /// surfacing a clear error if the target model does not implement the
    /// matching trait (`MtpTarget` for MTP, `SpeculativeTarget` for
    /// DFlash) — see [`SpeculativeDispatch::Mtp::user_requested_explicit_kind`].
    pub fn resolve(config: &ServerConfig) -> Result<Self, SpeculativeDispatchError> {
        let Some(draft_model_path) = config.draft_model_path.clone() else {
            return Ok(Self::Disabled);
        };

        // Parse the raw `--draft-kind` string into a typed `DrafterKind`,
        // mirroring `SpeculativeArgs::parse_kind` (which lives in the CLI
        // crate and is not reachable from `crate::server::*` without a
        // dependency inversion).
        let explicit_kind: Option<DrafterKind> = match config.draft_kind.as_deref() {
            None => None,
            Some("dflash") => Some(DrafterKind::Dflash),
            Some("mtp") => Some(DrafterKind::Mtp),
            Some("internal-mtp") => {
                return Err(SpeculativeDispatchError::InvalidKind {
                    message: "--draft-kind=internal-mtp is not user-selectable; the \
                              InternalMtp drafter is auto-detected from the target \
                              checkpoint. Pass --draft-kind dflash or --draft-kind mtp."
                        .to_string(),
                });
            }
            Some(other) => {
                return Err(SpeculativeDispatchError::InvalidKind {
                    message: format!(
                        "--draft-kind={other:?} is not recognised; accepted values: dflash, mtp"
                    ),
                });
            }
        };

        // Auto-detect (or reconcile against the explicit kind) from the
        // drafter's `config.json`. This is the same call the offline CLI
        // path makes in `src/commands/generate.rs`.
        let resolved_kind =
            resolve_drafter_kind(&draft_model_path, explicit_kind).map_err(|e| {
                SpeculativeDispatchError::DrafterConfig {
                    path: draft_model_path.clone(),
                    message: e.to_string(),
                }
            })?;

        let user_requested_explicit_kind = explicit_kind.is_some();
        let block_size = resolve_draft_block_size(config.draft_block_size, resolved_kind);

        // Dispatch matrix:
        // - explicit MTP / DFlash → kind-specific generator
        // - auto-detected MTP / DFlash → kind-specific generator
        //   (the scheduler logs the auto-detect for visibility)
        // - InternalMtp (peer) is auto-detected only and falls
        //   back to the Classic path until it has a server-side round
        //   loop.
        match resolved_kind {
            DrafterKind::Mtp => Ok(Self::Mtp {
                draft_model_path,
                block_size,
                user_requested_explicit_kind,
            }),
            DrafterKind::Dflash => Ok(Self::DFlash {
                draft_model_path,
                block_size,
                user_requested_explicit_kind,
            }),
            DrafterKind::InternalMtp => Ok(Self::Classic {
                draft_model_path,
                num_draft_tokens: config.num_draft_tokens,
                auto_detected_kind: resolved_kind,
            }),
            // `DrafterKind` is `#[non_exhaustive]`; route unknown future
            // variants through Classic so a stale binary does not panic
            // when an operator points at a newer drafter checkpoint.
            _ => Ok(Self::Classic {
                draft_model_path,
                num_draft_tokens: config.num_draft_tokens,
                auto_detected_kind: resolved_kind,
            }),
        }
    }

    /// Human-readable summary of the resolved dispatch, used by the
    /// worker's startup log so an operator can confirm at a glance which
    /// path is active.
    pub fn summary(&self) -> String {
        match self {
            Self::Disabled => "speculative=off".to_string(),
            Self::Classic {
                draft_model_path,
                num_draft_tokens,
                auto_detected_kind,
            } => format!(
                "speculative=classic (drafter={}, num_draft_tokens={num_draft_tokens}, \
                 auto_detected_kind={auto_detected_kind})",
                draft_model_path.display()
            ),
            Self::Mtp {
                draft_model_path,
                block_size,
                user_requested_explicit_kind,
            } => format!(
                "speculative=mtp (drafter={}, block_size={block_size}, \
                 explicit_kind={user_requested_explicit_kind})",
                draft_model_path.display()
            ),
            Self::DFlash {
                draft_model_path,
                block_size,
                user_requested_explicit_kind,
            } => format!(
                "speculative=dflash (drafter={}, block_size={block_size}, \
                 explicit_kind={user_requested_explicit_kind})",
                draft_model_path.display()
            ),
        }
    }

    /// Returns the drafter checkpoint path, if any. `None` for
    /// [`Self::Disabled`].
    pub fn draft_model_path(&self) -> Option<&std::path::Path> {
        match self {
            Self::Disabled => None,
            Self::Classic {
                draft_model_path, ..
            }
            | Self::Mtp {
                draft_model_path, ..
            }
            | Self::DFlash {
                draft_model_path, ..
            } => Some(draft_model_path.as_path()),
        }
    }

    /// Returns the resolved [`DrafterKind`] when the dispatch is a
    /// kind-specific variant or carries an auto-detected kind. Returns
    /// `None` for [`Self::Disabled`].
    pub fn drafter_kind(&self) -> Option<DrafterKind> {
        match self {
            Self::Disabled => None,
            Self::Classic {
                auto_detected_kind, ..
            } => Some(*auto_detected_kind),
            Self::Mtp { .. } => Some(DrafterKind::Mtp),
            Self::DFlash { .. } => Some(DrafterKind::Dflash),
        }
    }

    /// Returns the resolved draft block size for [`Self::Mtp`] and
    /// [`Self::DFlash`]. `None` for [`Self::Disabled`] and
    /// [`Self::Classic`] (the classic path uses `num_draft_tokens`, not
    /// `block_size`).
    pub fn block_size(&self) -> Option<u32> {
        match self {
            Self::Mtp { block_size, .. } | Self::DFlash { block_size, .. } => Some(*block_size),
            _ => None,
        }
    }

    /// Whether the dispatch is one of the kind-specific server-side
    /// round-loop variants (MTP or DFlash).
    pub fn is_kind_specific(&self) -> bool {
        matches!(self, Self::Mtp { .. } | Self::DFlash { .. })
    }
}

#[cfg(test)]
#[path = "speculative_dispatch_tests.rs"]
mod tests;
