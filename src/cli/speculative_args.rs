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

//! Speculative decoding CLI flag group.
//!
//! This module owns the single canonical clap definition for the
//! `--draft-kind` and `--draft-block-size` flags that select between the
//! classic [`SpeculativeGenerator`](mlxcel_core::speculative::SpeculativeGenerator),
//! the MTP round-loop driver (`MtpGenerator`, sub-6), and the
//! DFlash round-loop driver (`DFlashGenerator`, sub-12).
//!
//! Every mlxcel binary that exposes a generation surface flattens
//! [`SpeculativeArgs`] via `#[command(flatten)]`, which means:
//!
//! - The user-visible `--help` text is identical across `mlxcel generate`,
//!   `mlxcel serve`, and `mlxcel-server`.
//! - Adding, renaming, or extending a flag only requires editing this file.
//! - The shared resolution helpers ([`resolve_draft_block_size`] and the
//!   `env_fallback_draft_*` family) live next to the flag definitions so the
//!   binary entry points stay slim.
//!
//! The existing `--draft-model` (`--model-draft` on `mlxcel-server` for
//! llama.cpp compatibility) and `--draft-max` / `--num-draft-tokens` flags
//! intentionally remain on the per-binary `Args` structs because their
//! names diverge across binaries (the `mlxcel-server` binary needs the
//! llama.cpp `--model-draft` spelling) and the `--num-draft-tokens` /
//! `--draft-max` knob has separate semantics on the offline `generate` vs.
//! the continuous-batched `serve` paths.
//!
//! Used by: mlxcel generate, mlxcel serve, mlxcel-server.

use clap::Args;
use mlxcel_core::drafter::{DrafterKind, KNOWN_DRAFTER_KINDS};

/// Per-kind default `--draft-block-size` when the operator does not pass
/// the flag explicitly.
///
/// Values match upstream:
///
/// - **MTP** → `4` — the Gemma 4 MTP "assistant" draft block length used
///   by https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/gemma4_assistant/config.py.
/// - **DFlash** → `16` — the Qwen 3.5 DFlash drafter's `block_size`
///   declared in https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/qwen3_dflash/config.py#L31.
///   Mirrors [`mlxcel_core::drafter::dflash::DEFAULT_BLOCK_SIZE`].
pub const DEFAULT_MTP_BLOCK_SIZE: u32 = 4;
pub const DEFAULT_DFLASH_BLOCK_SIZE: u32 = 16;

/// Shared speculative-decoding flag group.
///
/// Flattened into the clap `Args` struct of every binary that exposes a
/// generation surface. The flags below MUST stay in sync across all
/// callers; see the integration test `tests/cli_help_consistency.rs`.
///
/// The group only owns the **dispatch-selecting** flags (`--draft-kind`,
/// `--draft-block-size`). The `--draft-model` path, the `--draft-max` /
/// `--num-draft-tokens` knob, and the llama.cpp-compat `--model-draft`
/// alias all remain on the per-binary `Args` structs because their
/// spellings and semantics diverge.
#[derive(Args, Debug, Clone, Default)]
#[command(next_help_heading = "Speculative Decoding Options")]
pub struct SpeculativeArgs {
    /// Speculative drafter kind. Optional.
    ///
    /// Accepted values: `dflash`, `mtp`. When unset AND a drafter path is
    /// supplied (`--draft-model` on `mlxcel`, `--model-draft` on
    /// `mlxcel-server`), the kind is auto-detected from the drafter's
    /// `config.json::model_type` via
    /// `mlxcel_core::drafter::resolve_drafter_kind`. Auto-detect maps
    /// `gemma4_assistant -> mtp`; everything else falls back to
    /// `dflash` (matching the upstream
    /// `DEFAULT_DRAFTER_KIND = "dflash"` convention).
    ///
    /// When unset AND no drafter path is supplied, mlxcel runs without a
    /// speculative drafter. The offline `mlxcel generate` command keeps the
    /// classic non-MTP / non-DFlash `SpeculativeGenerator` path when a
    /// drafter is supplied without an explicit kind.
    ///
    /// Also read from `LLAMA_ARG_DRAFT_KIND` (and the mlxcel-native
    /// alias `MLXCEL_DRAFT_KIND`).
    #[arg(long = "draft-kind", env = "LLAMA_ARG_DRAFT_KIND", value_name = "KIND")]
    pub draft_kind: Option<String>,

    /// Draft block size in tokens. Optional.
    ///
    /// When unset, the default depends on the resolved drafter kind:
    /// `4` for `mtp` and `16` for `dflash`. Mirrors the upstream
    /// per-drafter `block_size` config field.
    ///
    /// Also read from `LLAMA_ARG_DRAFT_BLOCK_SIZE` (and the mlxcel-native
    /// alias `MLXCEL_DRAFT_BLOCK_SIZE`).
    #[arg(
        long = "draft-block-size",
        env = "LLAMA_ARG_DRAFT_BLOCK_SIZE",
        value_name = "N"
    )]
    pub draft_block_size: Option<u32>,
}

impl SpeculativeArgs {
    /// Parse the raw `--draft-kind` string into a typed [`DrafterKind`].
    ///
    /// Returns `Ok(None)` when no kind was supplied. Returns an
    /// `anyhow::Error` whose message lists the accepted values from
    /// [`KNOWN_DRAFTER_KINDS`] when the value does not parse.
    ///
    /// Note that we intentionally **do not** accept the third
    /// `internal-mtp` variant of [`DrafterKind`] on the CLI — that
    /// variant is auto-detected from the target checkpoint
    /// and is not user-selectable today. The accepted set on the CLI is
    /// the upstream `KNOWN_DRAFTER_KINDS = {"dflash", "mtp"}` only;
    /// passing `internal-mtp` returns a parse error with a hint.
    pub fn parse_kind(&self) -> anyhow::Result<Option<DrafterKind>> {
        let Some(raw) = self.draft_kind.as_deref() else {
            return Ok(None);
        };
        match raw {
            "dflash" => Ok(Some(DrafterKind::Dflash)),
            "mtp" => Ok(Some(DrafterKind::Mtp)),
            "internal-mtp" => Err(anyhow::anyhow!(
                "--draft-kind=internal-mtp is not user-selectable; the \
                 InternalMtp drafter is auto-detected from the target \
                 checkpoint. Pass --draft-kind dflash or --draft-kind mtp."
            )),
            other => Err(anyhow::anyhow!(
                "--draft-kind={other:?} is not recognised; accepted values: {}",
                user_selectable_kinds().join(", ")
            )),
        }
    }
}

/// Set of drafter kinds the CLI accepts. This is a subset of
/// [`KNOWN_DRAFTER_KINDS`] that excludes `internal-mtp` because that
/// variant is auto-detected, not user-selectable.
pub fn user_selectable_kinds() -> Vec<&'static str> {
    KNOWN_DRAFTER_KINDS
        .iter()
        .copied()
        .filter(|k| *k != "internal-mtp")
        .collect()
}

/// Per-kind default `--draft-block-size` lookup.
///
/// When `--draft-block-size` is not supplied on the CLI, every consumer
/// (offline `generate`, server scheduler, llama-server compat binary)
/// must agree on the same per-kind default. This helper centralises that
/// rule so the agreement is enforced at one source point.
pub fn default_block_size_for_kind(kind: DrafterKind) -> u32 {
    match kind {
        DrafterKind::Mtp => DEFAULT_MTP_BLOCK_SIZE,
        // DFlash and the future InternalMtp variant both share the
        // upstream-published 16-token default. InternalMtp's user-facing
        // override surface lands in for now its CLI knob
        // shares the DFlash default.
        DrafterKind::Dflash | DrafterKind::InternalMtp => DEFAULT_DFLASH_BLOCK_SIZE,
        // `DrafterKind` is `#[non_exhaustive]` so future variants force a
        // CI failure. Until a new variant lands the wildcard is
        // unreachable; we route it through the upstream default so the
        // crate compiles without a `todo!()` panic risk.
        _ => DEFAULT_DFLASH_BLOCK_SIZE,
    }
}

/// Resolve the effective draft block size given an explicit CLI override
/// and the resolved drafter kind.
///
/// When `override_value` is `Some(n)`, returns `n` (with no further
/// validation here — concrete generators enforce their own minimums).
/// When `override_value` is `None`, returns
/// [`default_block_size_for_kind`] for `kind`.
pub fn resolve_draft_block_size(override_value: Option<u32>, kind: DrafterKind) -> u32 {
    override_value.unwrap_or_else(|| default_block_size_for_kind(kind))
}

/// Apply the `MLXCEL_DRAFT_KIND` env-var fallback to the raw
/// `--draft-kind` CLI value.
///
/// `clap` already reads `LLAMA_ARG_DRAFT_KIND` via the `env = "..."` attr
/// on the flag. This helper layers the mlxcel-native `MLXCEL_DRAFT_KIND`
/// alias on top, with the same warn-on-conflict pattern used by the
/// other `MLXCEL_*` / `LLAMA_ARG_*` pairs in the crate.
///
/// Precedence (highest first):
///   1. `--draft-kind` CLI flag (after clap's `env = "..."` injection)
///   2. `MLXCEL_DRAFT_KIND` env var (this helper)
pub fn env_fallback_draft_kind(value: &mut Option<String>) {
    apply_optional_string_env_fallback(value, "MLXCEL_DRAFT_KIND", "draft-kind");
}

/// Apply the `MLXCEL_DRAFT_BLOCK_SIZE` env-var fallback to the raw
/// `--draft-block-size` CLI value.
///
/// Same precedence rules as [`env_fallback_draft_kind`]. Unparseable env
/// values are logged and ignored so a malformed env var does not break
/// server startup.
pub fn env_fallback_draft_block_size(value: &mut Option<u32>) {
    if value.is_some() {
        // CLI flag already set (possibly via LLAMA_ARG_DRAFT_BLOCK_SIZE).
        // Log a collision if MLXCEL_DRAFT_BLOCK_SIZE is also set so the
        // operator can see the precedence outcome.
        if let Ok(env_val) = std::env::var("MLXCEL_DRAFT_BLOCK_SIZE") {
            tracing::info!(
                "MLXCEL_DRAFT_BLOCK_SIZE={env_val} is set but the CLI \
                 --draft-block-size flag (or LLAMA_ARG_DRAFT_BLOCK_SIZE) \
                 already provides a value; keeping CLI"
            );
        }
        return;
    }
    if let Ok(raw) = std::env::var("MLXCEL_DRAFT_BLOCK_SIZE") {
        match raw.parse::<u32>() {
            Ok(n) => *value = Some(n),
            Err(e) => {
                tracing::warn!("MLXCEL_DRAFT_BLOCK_SIZE={raw:?} is not a valid u32 ({e}); ignoring")
            }
        }
    }
}

/// Shared helper: if `value` is `None` and the named env var is set, fill
/// `value` from the env var. If `value` is `Some` (CLI was set) and the env
/// var is also present and differs, log an INFO and keep the CLI value.
/// When `value` already equals the env var string (because clap's
/// `env = "..."` injected it), no conflict log is emitted since there is no
/// real conflict.
fn apply_optional_string_env_fallback(
    value: &mut Option<String>,
    env_name: &'static str,
    flag_name: &'static str,
) {
    let env_value = std::env::var(env_name).ok();
    match (&value, env_value) {
        (Some(cli), Some(env)) if *cli != env => {
            tracing::info!(
                "{env_name}={env:?} differs from --{flag_name}={cli:?}; keeping CLI value"
            );
        }
        (None, Some(env)) => {
            *value = Some(env);
        }
        _ => {}
    }
}

#[cfg(test)]
#[path = "speculative_args_tests.rs"]
mod tests;
