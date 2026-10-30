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
//!   — explicit block-size override; `None` resolves through
//!   [`crate::cli::speculative_args::resolve_draft_block_size_for_target`],
//!   which prefers a width the drafter checkpoint declares, then a measured
//!   default for the running device and the target's quantization
//!   ([`crate::cli::draft_block_policy`], issue #1797), then the flat
//!   [`crate::cli::speculative_args::default_block_size_for_kind`].
//!
//! At worker-startup, [`SpeculativeDispatch::resolve`] turns those raw
//! fields into one of:
//!
//! - [`SpeculativeDispatch::Disabled`] — no `--draft-model`; plain decode
//!   runs.
//! - [`SpeculativeDispatch::Mtp { .. }`] — `--draft-kind mtp` (or auto-
//!   detect resolved to MTP). The scheduler constructs the kind-specific
//!   [`mlxcel_core::speculative::mtp::MtpGenerator`] (B=1) or
//!   [`mlxcel_core::speculative::mtp::MtpBatchedGenerator`] (B>1) per
//!   request.
//! - [`SpeculativeDispatch::PromptLookup { .. }`]: set by `--draft-kind
//!   prompt-lookup` with no `--draft-model` (#2255). Prompt lookup needs no
//!   checkpoint, so it is resolved before the draft-model early return. It
//!   is not a kind-specific burst ([`SpeculativeDispatch::is_kind_specific`]
//!   is `false`): the scheduler runs at most one verify round per tick per
//!   eligible sequence inside the regular decode tick and keeps the
//!   lookahead pipeline for ticks where no row proposes (see
//!   `server::batch::scheduler::prompt_lookup`).
//! - [`SpeculativeDispatch::FullModel { .. }`]: `--draft-kind full-model`, or
//!   a `--draft-model` that auto-detects as an ordinary full model (#2757,
//!   [`mlxcel_core::drafter::auto_detected_full_model_dir`]). The worker loads
//!   the checkpoint through the model loader next to the target and each
//!   eligible row drafts from it through
//!   [`mlxcel_core::speculative::full_model_drafter::FullModelDrafter`] on the
//!   same token-only tick as prompt lookup, so it is not a kind-specific
//!   burst either.
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
//! Mirrors the offline-CLI dispatch in `src/commands/generate.rs`
//! (`run_generation_mode`, its `--draft-model` arm). The same
//! `resolve_drafter_kind` + `resolve_draft_block_size_for_target` helpers
//! are reused so the offline and server paths cannot drift.

use std::path::PathBuf;

use mlxcel_core::drafter::{
    DrafterKind, auto_detected_full_model_dir, drafter_config_problem, full_model_vocab_problem,
    resolve_drafter_kind,
};
use mlxcel_core::speculative::prompt_lookup::{DraftPolicy, PromptLookupConfig};

use crate::cli::draft_block_policy::BlockSizeSource;
use crate::cli::speculative_args::resolve_draft_block_size_for_target;
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
        /// What produced `block_size`: an operator override, the drafter
        /// checkpoint's own declaration, a measured hardware default
        /// (issue #1797), or the flat per-kind constant. Printed by
        /// [`Self::summary`] so the startup log distinguishes a measured
        /// narrowing from a flag the operator forgot they exported.
        block_size_source: BlockSizeSource,
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
        /// Same provenance record as [`Self::Mtp::block_size_source`].
        block_size_source: BlockSizeSource,
        /// Whether the kind was explicitly requested via `--draft-kind`
        /// (true) or auto-detected from the drafter's `config.json`
        /// (false). Same operator-facing semantics as
        /// [`Self::Mtp::user_requested_explicit_kind`].
        user_requested_explicit_kind: bool,
    },

    /// Prompt-lookup (n-gram) speculative decoding (#2255): `--draft-kind
    /// prompt-lookup` with no drafter checkpoint. Each eligible sequence gets
    /// a [`mlxcel_core::speculative::prompt_lookup_drafter::PromptLookupDrafter`]
    /// at prefill completion and runs at most one verify round per decode
    /// tick; everything else decodes in the regular batched step.
    PromptLookup {
        /// The drafter's tunables: `--draft-block-size` sets `max_draft`,
        /// CUDA gated dispatch additionally sets the server emission floor,
        /// shadow-confirmation length, and proposal-free pipeline cadence.
        /// Other settings retain [`PromptLookupConfig::default`].
        config: PromptLookupConfig,
        /// `--prompt-lookup-max-batch`: verify rounds run only on ticks whose
        /// active decode batch has at most this many rows.
        max_batch: usize,
    },

    /// An ordinary full model drafting from its own KV caches (#2757):
    /// `--draft-kind full-model`, or `--draft-model` naming a directory with
    /// no DFlash-family marker and no MTP `model_type`. The worker loads the
    /// checkpoint at startup and the scheduler gives each eligible sequence a
    /// per-row drafter state at prefill completion; rows then run at most one
    /// verify round per decode tick, exactly as prompt-lookup rows do, so
    /// every batch width the scheduler serves is supported and the rounds run
    /// while the active batch is at most `max_batch` rows.
    FullModel {
        /// Draft checkpoint directory (a standalone model).
        draft_model_path: PathBuf,
        /// Proposals per round (the verify input has `block_size + 1`
        /// positions).
        block_size: u32,
        /// Same provenance record as [`Self::Mtp::block_size_source`].
        block_size_source: BlockSizeSource,
        /// Whether `--draft-kind full-model` was passed (true) or the
        /// directory auto-detected as a full model (false).
        user_requested_explicit_kind: bool,
        /// `--prompt-lookup-max-batch`, the one gate the token-only tick
        /// applies: rows verify only while the active decode batch has at
        /// most this many rows.
        max_batch: usize,
    },
}

/// The settings the scheduler's token-only tick reads for a dispatch whose
/// rows run their own verify rounds inside the decode tick ([`SpeculativeDispatch::PromptLookup`]
/// and [`SpeculativeDispatch::FullModel`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenOnlyRounds {
    /// Largest proposal a round verifies.
    pub max_draft: usize,
    /// Largest active decode batch at which rows are asked for proposals.
    pub max_batch: usize,
}

/// Default `--prompt-lookup-max-batch`: the largest active decode batch at
/// which a prompt-lookup row may run its own verify forward.
///
/// Each proposing row costs one verify forward on top of the batched step,
/// so the gain shrinks as the batch grows. Measured on GB10 (Qwen3-1.7B
/// 4-bit, `--parallel 8`;
/// `docs/benchmark_results/server-prompt-lookup-gb10-2026-10-11.md`): with
/// this default, concurrency 4 and 8 match the flag-off server in every
/// workload, while a gate of 8 gains 15 to 63 % on raw-completion workloads
/// but loses 13 % on the chat-template copy workload at concurrency 4.
pub const DEFAULT_PROMPT_LOOKUP_MAX_BATCH: usize = 2;

/// Minimum recent tokens emitted per verify round on the CUDA server.
/// GB10 Qwen3-1.7B width 3 costs 1.76 pipelined steps and width 8 costs
/// 3.91 (the cost table in `DraftPolicy`). Four emitted tokens cover the
/// steady wide forward; narrow rounds emitting only two or three cannot
/// amortize the switch tick and synchronous re-prime gap. Full copy blocks
/// emit eight and amortize those transition costs across consecutive rounds.
/// Once the emitted-token EMA is below the floor, a partial low-emission round
/// pauses immediately. A complete miss gets one more probe only when the
/// accepted-token EMA was already at least four, so an abrupt field boundary
/// does not discard an established profitable copy run. Consecutive misses
/// still pause. A round meeting the floor repays startup EMA debt.
/// See `docs/benchmark_results/server-prompt-lookup-break-even-gb10-2026-10-11.md`.
pub(crate) const PROMPT_LOOKUP_MIN_TOKENS_PER_ROUND: usize = 4;
/// Require a whole default proposal's worth of shadow evidence to avoid
/// resuming on the incidental two-token matches common in chat prose.
pub(crate) const PROMPT_LOOKUP_SHADOW_CONFIRM_TOKENS: usize = 7;

/// Draft-policy name for [`SpeculativeDispatch::summary`].
fn draft_policy_name(policy: DraftPolicy) -> &'static str {
    match policy {
        DraftPolicy::Graded => "graded",
        DraftPolicy::Gated => "gated",
        _ => "other",
    }
}

/// Error variants for [`SpeculativeDispatch::resolve`].
///
/// All variants carry the **operator-facing** error message verbatim so
/// the worker's startup log can surface it without re-formatting. The
/// inner messages are stable and unit-tested.
#[derive(Debug, thiserror::Error)]
pub enum SpeculativeDispatchError {
    /// The speculative settings cannot be served: `--draft-kind` was passed an
    /// unrecognized or non-user-selectable value (not one of
    /// `KNOWN_DRAFTER_KINDS - {internal-mtp}`), `--draft-model` resolved to a
    /// kind with no server round loop, or the prompt-lookup settings are invalid.
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
    ///    [`resolve_draft_block_size_for_target`], which needs
    ///    `target_model_path` (the `-m` checkpoint) because the measured
    ///    default is keyed on the target's quantization mode, not the
    ///    drafter's. The resolved width carries its own provenance so
    ///    [`Self::summary`] can name it.
    /// 4. Returns the kind-specific variant. A kind with no server round loop
    ///    returns [`SpeculativeDispatchError::InvalidKind`].
    ///
    /// **Note**: when `--draft-kind` is explicitly passed AND the kind is
    /// supported by a server-side round loop, this method returns the
    /// kind-specific variant. The scheduler is then responsible for
    /// surfacing a clear error if the target model does not implement the
    /// matching trait (`MtpTarget` for MTP, `SpeculativeTarget` for
    /// DFlash) — see [`SpeculativeDispatch::Mtp::user_requested_explicit_kind`].
    pub fn resolve(
        config: &ServerConfig,
        target_model_path: &std::path::Path,
    ) -> Result<Self, SpeculativeDispatchError> {
        // Prompt lookup takes no checkpoint, so it resolves before the
        // draft-model early return below (#2255).
        if config.draft_kind.as_deref() == Some("prompt-lookup") {
            return Self::resolve_prompt_lookup(config);
        }
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
            Some("full-model") => Some(DrafterKind::FullModel),
            Some("internal-mtp") => {
                return Err(SpeculativeDispatchError::InvalidKind {
                    message: "--draft-kind=internal-mtp is not user-selectable; the \
                              InternalMtp drafter is auto-detected from the target \
                              checkpoint. Pass --draft-kind dflash, --draft-kind mtp or \
                              --draft-kind full-model."
                        .to_string(),
                });
            }
            Some(other) => {
                return Err(SpeculativeDispatchError::InvalidKind {
                    message: format!(
                        "--draft-kind={other:?} is not recognised; accepted values: dflash, mtp, \
                         full-model, prompt-lookup"
                    ),
                });
            }
        };

        // A full-model drafter loads through the model loader, which refuses
        // a DFlash-family directory as "not a standalone model"; say so here,
        // before the worker thread starts (#2757).
        if explicit_kind == Some(DrafterKind::FullModel)
            && mlxcel_core::drafter::dflash::is_dflash_drafter_dir(&draft_model_path)
        {
            return Err(SpeculativeDispatchError::InvalidKind {
                message: format!(
                    "--draft-kind full-model: {} is a DFlash-family drafter checkpoint, not an \
                     ordinary full model; pass --draft-kind dflash or omit --draft-kind.",
                    draft_model_path.display()
                ),
            });
        }
        // With no explicit kind the resolver swallows an unreadable config and
        // falls back to DFlash, where the real cause would only surface as a
        // drafter load error on the first request. Report it at startup.
        if explicit_kind.is_none()
            && let Some(problem) = drafter_config_problem(&draft_model_path)
        {
            return Err(SpeculativeDispatchError::DrafterConfig {
                path: draft_model_path,
                message: problem,
            });
        }

        // Auto-detect (or reconcile against the explicit kind) from the
        // drafter's `config.json`. This is the same call the offline CLI
        // path makes in `src/commands/generate.rs`.
        let mut resolved_kind =
            resolve_drafter_kind(&draft_model_path, explicit_kind).map_err(|e| {
                SpeculativeDispatchError::DrafterConfig {
                    path: draft_model_path.clone(),
                    message: e.to_string(),
                }
            })?;
        // The resolver's DFlash default covers an ordinary full model too
        // (#1168); the structural rule tells the two apart (#2757).
        if auto_detected_full_model_dir(&draft_model_path, explicit_kind, resolved_kind) {
            tracing::info!(
                drafter = %draft_model_path.display(),
                "Auto-detected --draft-kind full-model: no DFlash-family marker and no MTP model_type"
            );
            resolved_kind = DrafterKind::FullModel;
        }

        let user_requested_explicit_kind = explicit_kind.is_some();
        // The block-width policy is keyed on the TARGET's quantization, not
        // the drafter's: the verify block runs through the target's quantized
        // projections, and that is where the CUDA row boundaries are
        // (issue #1797).
        let block_size = resolve_draft_block_size_for_target(
            config.draft_block_size,
            resolved_kind,
            &draft_model_path,
            target_model_path,
        );

        // Dispatch matrix:
        // - explicit MTP / DFlash → kind-specific generator
        // - auto-detected MTP / DFlash → kind-specific generator
        //   (the scheduler logs the auto-detect for visibility)
        // - any other kind (`InternalMtp`, which no drafter `model_type` maps
        //   to, or a future `#[non_exhaustive]` variant) has no server-side
        //   round loop and is refused at startup.
        match resolved_kind {
            DrafterKind::Mtp => Ok(Self::Mtp {
                draft_model_path,
                block_size: block_size.width,
                block_size_source: block_size.source,
                user_requested_explicit_kind,
            }),
            DrafterKind::Dflash => Ok(Self::DFlash {
                draft_model_path,
                block_size: block_size.width,
                block_size_source: block_size.source,
                user_requested_explicit_kind,
            }),
            DrafterKind::FullModel => {
                if config.prompt_lookup_max_batch == 0 {
                    return Err(SpeculativeDispatchError::InvalidKind {
                        message: "--prompt-lookup-max-batch must be at least 1".to_string(),
                    });
                }
                if block_size.width == 0 {
                    return Err(SpeculativeDispatchError::InvalidKind {
                        message: "--draft-kind full-model needs --draft-block-size of at least 1"
                            .to_string(),
                    });
                }
                if let Some(problem) =
                    full_model_vocab_problem(&draft_model_path, target_model_path)
                {
                    return Err(SpeculativeDispatchError::InvalidKind {
                        message: format!("--draft-kind full-model: {problem}"),
                    });
                }
                Ok(Self::FullModel {
                    draft_model_path,
                    block_size: block_size.width,
                    block_size_source: block_size.source,
                    user_requested_explicit_kind,
                    max_batch: config.prompt_lookup_max_batch,
                })
            }
            other => Err(SpeculativeDispatchError::InvalidKind {
                message: format!(
                    "--draft-model {} resolved to drafter kind {other}, which has no server-side \
                     round loop; pass --draft-kind dflash, --draft-kind mtp or --draft-kind \
                     full-model with a matching drafter, or --draft-kind prompt-lookup with no \
                     drafter checkpoint.",
                    draft_model_path.display()
                ),
            }),
        }
    }

    /// [`Self::PromptLookup`] from `--draft-kind prompt-lookup`: a drafter
    /// checkpoint is refused, `--draft-block-size` sets `max_draft` (checked
    /// by [`PromptLookupConfig::validate`]), and `--prompt-lookup-max-batch`
    /// must admit at least one row.
    fn resolve_prompt_lookup(config: &ServerConfig) -> Result<Self, SpeculativeDispatchError> {
        if config.draft_model_path.is_some() {
            return Err(SpeculativeDispatchError::InvalidKind {
                message: "prompt-lookup takes no draft model; remove --draft-model/--model-draft"
                    .to_string(),
            });
        }
        let mut lookup = PromptLookupConfig::default();
        if lookup.policy == DraftPolicy::Gated {
            lookup.min_tokens_per_round = Some(PROMPT_LOOKUP_MIN_TOKENS_PER_ROUND);
            lookup.shadow_confirm_tokens = PROMPT_LOOKUP_SHADOW_CONFIRM_TOKENS;
            lookup.plain_rounds_before_pipeline = 0;
        }
        if let Some(width) = config.draft_block_size {
            lookup.max_draft = width as usize;
        }
        lookup.min_tokens_per_round = lookup
            .min_tokens_per_round
            .map(|floor| floor.min(lookup.max_draft.saturating_add(1)));
        lookup
            .validate()
            .map_err(|message| SpeculativeDispatchError::InvalidKind {
                message: format!("--draft-kind prompt-lookup: {message} (--draft-block-size)"),
            })?;
        if config.prompt_lookup_max_batch == 0 {
            return Err(SpeculativeDispatchError::InvalidKind {
                message: "--prompt-lookup-max-batch must be at least 1".to_string(),
            });
        }
        Ok(Self::PromptLookup {
            config: lookup,
            max_batch: config.prompt_lookup_max_batch,
        })
    }

    /// Human-readable summary of the resolved dispatch, used by the
    /// worker's startup log so an operator can confirm at a glance which
    /// path is active.
    pub fn summary(&self) -> String {
        match self {
            Self::Disabled => "speculative=off".to_string(),
            Self::Mtp {
                draft_model_path,
                block_size,
                block_size_source,
                user_requested_explicit_kind,
            } => format!(
                "speculative=mtp (drafter={}, block_size={block_size}, \
                 block_size_from={}, explicit_kind={user_requested_explicit_kind})",
                draft_model_path.display(),
                block_size_source.reason()
            ),
            Self::DFlash {
                draft_model_path,
                block_size,
                block_size_source,
                user_requested_explicit_kind,
            } => format!(
                "speculative=dflash (drafter={}, block_size={block_size}, \
                 block_size_from={}, explicit_kind={user_requested_explicit_kind})",
                draft_model_path.display(),
                block_size_source.reason()
            ),
            Self::PromptLookup { config, max_batch } => format!(
                "speculative=prompt-lookup (max_draft={}, policy={}, max_batch={max_batch}, min_tokens_per_round={:?} (ema_with_profitable_boundary_miss_grace), shadow_confirm_tokens={}, plain_rounds_before_pipeline={})",
                config.max_draft,
                draft_policy_name(config.policy),
                config.min_tokens_per_round,
                config.shadow_confirm_tokens,
                config.plain_rounds_before_pipeline
            ),
            Self::FullModel {
                draft_model_path,
                block_size,
                block_size_source,
                user_requested_explicit_kind,
                max_batch,
            } => format!(
                "speculative=full-model (drafter={}, block_size={block_size}, \
                 block_size_from={}, explicit_kind={user_requested_explicit_kind}, \
                 max_batch={max_batch})",
                draft_model_path.display(),
                block_size_source.reason()
            ),
        }
    }

    /// Returns the drafter checkpoint path, if any. `None` for
    /// [`Self::Disabled`].
    pub fn draft_model_path(&self) -> Option<&std::path::Path> {
        match self {
            Self::Disabled | Self::PromptLookup { .. } => None,
            Self::Mtp {
                draft_model_path, ..
            }
            | Self::DFlash {
                draft_model_path, ..
            }
            | Self::FullModel {
                draft_model_path, ..
            } => Some(draft_model_path.as_path()),
        }
    }

    /// Returns the resolved [`DrafterKind`] of an active dispatch. Returns
    /// `None` for [`Self::Disabled`].
    pub fn drafter_kind(&self) -> Option<DrafterKind> {
        match self {
            Self::Disabled => None,
            Self::Mtp { .. } => Some(DrafterKind::Mtp),
            Self::DFlash { .. } => Some(DrafterKind::Dflash),
            Self::PromptLookup { .. } => Some(DrafterKind::PromptLookup),
            Self::FullModel { .. } => Some(DrafterKind::FullModel),
        }
    }

    /// Returns the resolved draft block size for [`Self::Mtp`],
    /// [`Self::DFlash`] and [`Self::FullModel`]. `None` otherwise.
    pub fn block_size(&self) -> Option<u32> {
        match self {
            Self::Mtp { block_size, .. }
            | Self::DFlash { block_size, .. }
            | Self::FullModel { block_size, .. } => Some(*block_size),
            _ => None,
        }
    }

    /// Whether the dispatch is one of the kind-specific server-side
    /// round-loop variants (MTP or DFlash). `false` for
    /// [`Self::PromptLookup`] and [`Self::FullModel`], whose verify rounds
    /// run inside the regular decode tick rather than a burst; see
    /// [`Self::token_only_rounds`].
    pub fn is_kind_specific(&self) -> bool {
        matches!(self, Self::Mtp { .. } | Self::DFlash { .. })
    }

    /// The token-only tick's settings for a [`Self::PromptLookup`] or
    /// [`Self::FullModel`] dispatch, `None` otherwise. The scheduler checks
    /// this, not [`Self::is_kind_specific`], to run per-row verify rounds
    /// inside the decode tick.
    pub fn token_only_rounds(&self) -> Option<TokenOnlyRounds> {
        match self {
            Self::PromptLookup { config, max_batch } => Some(TokenOnlyRounds {
                max_draft: config.max_draft,
                max_batch: *max_batch,
            }),
            Self::FullModel {
                block_size,
                max_batch,
                ..
            } => Some(TokenOnlyRounds {
                max_draft: *block_size as usize,
                max_batch: *max_batch,
            }),
            _ => None,
        }
    }

    /// The draft checkpoint of a [`Self::FullModel`] dispatch, `None`
    /// otherwise; the worker loads it next to the target.
    pub fn full_model_draft_path(&self) -> Option<&std::path::Path> {
        match self {
            Self::FullModel {
                draft_model_path, ..
            } => Some(draft_model_path.as_path()),
            _ => None,
        }
    }

    /// The drafter config and `--prompt-lookup-max-batch` of a
    /// [`Self::PromptLookup`] dispatch, `None` otherwise; the scheduler
    /// primes a row's prompt-lookup drafter from it.
    pub fn prompt_lookup_config(&self) -> Option<(&PromptLookupConfig, usize)> {
        match self {
            Self::PromptLookup { config, max_batch } => Some((config, *max_batch)),
            _ => None,
        }
    }
}

#[cfg(test)]
#[path = "speculative_dispatch_tests.rs"]
mod tests;
