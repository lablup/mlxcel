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
//! The existing `--draft-model` / `--model-draft` and `--draft-max` /
//! `--draft` flags intentionally remain on the per-binary `Args` structs
//! rather than joining this shared group, because `mlxcel generate` uses
//! the unrelated `--num-draft-tokens` spelling with offline-only semantics.
//! On the two server binaries (`mlxcel serve`, `mlxcel-server`) both
//! spellings are cross-aliased (`visible_alias`) so a command line written
//! for one parses unchanged on the other: `--draft-model` /
//! `--model-draft` both resolve to the drafter checkpoint path, and
//! `--draft-max` / `--draft` both resolve to the per-step draft-token
//! budget. See `tests/cli_help_consistency.rs` for the cross-binary
//! parity test.
//!
//! Used by: mlxcel generate, mlxcel serve, mlxcel-server.

use clap::Args;
use mlxcel_core::drafter::{DrafterKind, KNOWN_DRAFTER_KINDS};

use crate::cli::draft_block_policy::{
    BlockSizeSource, ResolvedBlockSize, TargetQuantization, peek_target_quantization,
    resolve_measured_block_size,
};

/// Per-kind default `--draft-block-size` when the operator does not pass
/// the flag explicitly.
///
/// Values match upstream:
///
/// - **MTP** → `4`: the Gemma 4 MTP "assistant" draft block length used
///   by https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/gemma4_assistant/config.py.
///   `DrafterKind::Mtp` also covers the unrelated Qwen 3.5 MTP family; for
///   that family [`resolve_draft_block_size`] prefers the drafter
///   checkpoint's own configured `block_size`
///   (`mlxcel_core::drafter::peek_qwen35_mtp_configured_block_size`) over
///   this constant, so this Gemma-4-derived value is only the fallback
///   when no Qwen 3.5 MTP hint is found.
/// - **DFlash** → `16`: the Qwen 3.5 DFlash drafter's `block_size`
///   declared in https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/qwen3_dflash/config.py#L31.
///   Mirrors [`mlxcel_core::drafter::dflash::DEFAULT_BLOCK_SIZE`]. An LFM2
///   DSpark drafter (also `DrafterKind::Dflash`) instead resolves to its own
///   runtime verify width through
///   `mlxcel_core::drafter::peek_dspark_configured_block_size` (8 on the
///   published checkpoints, `block_size + 1 = 10` with `--draft-block-size 10`).
///
/// Since issue #1797 these are the **last** fallback rather than the only
/// default: a drafter checkpoint's own declared width wins, and so does a
/// measured entry in [`crate::cli::draft_block_policy`] for the running
/// device and the target's quantization. On GB10 the flat DFlash 16 lost 17%
/// against classic decode where width 4 won 32%, so a host with a
/// measurement narrows it. Hosts with no measurement keep these constants
/// unchanged.
pub const DEFAULT_MTP_BLOCK_SIZE: u32 = 4;
pub const DEFAULT_DFLASH_BLOCK_SIZE: u32 = 16;

/// Shared speculative-decoding flag group.
///
/// Flattened into the clap `Args` struct of every binary that exposes a
/// generation surface. The flags below MUST stay in sync across all
/// callers; see the integration test `tests/cli_help_consistency.rs`.
///
/// The group only owns the **dispatch-selecting** flags (`--draft-kind`,
/// `--draft-block-size`). The drafter-path flag (`--draft-model` /
/// `--model-draft`) and the draft-token-count knob (`--draft-max` /
/// `--draft` on the servers, `--num-draft-tokens` on offline `generate`)
/// remain on the per-binary `Args` structs; on `mlxcel serve` and
/// `mlxcel-server` both spellings of each are cross-aliased for parity,
/// see the module-level docs above.
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
    /// `gemma4_assistant`, `gemma4_unified_assistant`, and `qwen3_5_mtp`
    /// to `mtp`; everything else falls back to `dflash` (matching the
    /// upstream `DEFAULT_DRAFTER_KIND = "dflash"` convention).
    ///
    /// When unset AND no drafter path is supplied, mlxcel runs without a
    /// speculative drafter. The offline `mlxcel generate` command keeps the
    /// classic `SpeculativeGenerator` path when a drafter without an MTP
    /// `model_type` is supplied without an explicit kind; an auto-detected
    /// MTP drafter routes to the MTP round loop (none of the MTP drafter
    /// checkpoints can load as a standalone classic draft model).
    ///
    /// Parity note (offline `mlxcel generate` with `--draft-kind mtp`):
    /// at temperature 0 with no sampling penalties the output matches the
    /// non-speculative path (within the f16 / #203 jitter class). When any
    /// repetition, frequency, presence, or DRY penalty is active, only the
    /// first bonus token is penalized; subsequent tokens in each verify
    /// window are greedy, so penalized requests are not byte-identical to
    /// the non-speculative path.
    ///
    /// Also read from `LLAMA_ARG_DRAFT_KIND` (and the mlxcel-native
    /// alias `MLXCEL_DRAFT_KIND`).
    #[arg(long = "draft-kind", env = "LLAMA_ARG_DRAFT_KIND", value_name = "KIND")]
    pub draft_kind: Option<String>,

    /// Draft block size in tokens. Optional.
    ///
    /// When unset, the default is resolved in this order: a width the
    /// drafter checkpoint declares for itself, then a measured default for
    /// the running device and the target's quantization, then the flat
    /// per-kind constant (`4` for `mtp`, `16` for `dflash`, mirroring the
    /// upstream per-drafter `block_size` config field). The resolved value
    /// and which of those produced it are logged at startup.
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
    /// `internal-mtp` variant of [`DrafterKind`] on the CLI. That
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
        // DFlash and InternalMtp both share the upstream-published
        // 16-token default. InternalMtp is auto-detected rather than
        // selectable via `--draft-kind` and has no user-facing override
        // surface of its own yet, so for now its `--draft-block-size`
        // default shares the DFlash value.
        DrafterKind::Dflash | DrafterKind::InternalMtp => DEFAULT_DFLASH_BLOCK_SIZE,
        // `DrafterKind` is `#[non_exhaustive]` so future variants force a
        // CI failure. Until a new variant lands the wildcard is
        // unreachable; we route it through the upstream default so the
        // crate compiles without a `todo!()` panic risk.
        _ => DEFAULT_DFLASH_BLOCK_SIZE,
    }
}

/// Resolve the effective draft block size, reporting what produced it.
///
/// Precedence, highest first:
///
/// 1. `override_value`: `--draft-block-size` and its two environment
///    spellings, returned verbatim with no further validation here (concrete
///    generators enforce their own minimums).
/// 2. A width the drafter checkpoint declares for itself (Qwen 3.5 MTP,
///    Inkling, GLM 4 MoE Lite, LFM2 DSpark, Muse Glimmer).
/// 3. A measured per-device, per-quantization default
///    ([`crate::cli::draft_block_policy::measured_default_block_size`],
///    issue #1797).
/// 4. The flat [`default_block_size_for_kind`] constant, which is what every
///    host and quantization path with no measurement takes.
///
/// Step 3 is a default only and can never displace an override or a
/// checkpoint's own declaration; it sits between them and the flat constant
/// precisely so an unmeasured platform is bit-for-bit unchanged.
///
/// Why step 2 exists at all, since it is not obvious that a checkpoint knows
/// better than a constant: [`DrafterKind::Mtp`] covers two unrelated drafter
/// families (Gemma 4 assistant and Qwen 3.5 MTP) and
/// [`DEFAULT_MTP_BLOCK_SIZE`] is the Gemma-4-derived one, so applying it to a
/// Qwen 3.5 MTP drafter ships whatever that constant happens to be rather
/// than the checkpoint's own declared block size. Issue #1165 measured the
/// published checkpoint's configured block size (3) faster than the flat
/// constant (4): 16.48 against 13.81 tok/s, 0.591 against 0.465 acceptance
/// (`docs/benchmark_results/qwen38-mtp-m1ultra-2026-08-16.md`).
pub fn resolve_draft_block_size_detailed(
    override_value: Option<u32>,
    kind: DrafterKind,
    model_path: &std::path::Path,
    cuda_compute_capability: Option<(u32, u32)>,
    target_quantization: TargetQuantization,
) -> ResolvedBlockSize {
    let from_checkpoint = |width: u32, which: &'static str| ResolvedBlockSize {
        width,
        source: BlockSizeSource::DrafterCheckpoint(which),
    };
    if let Some(n) = override_value {
        return ResolvedBlockSize {
            width: n,
            source: BlockSizeSource::Override,
        };
    }
    if kind == DrafterKind::Mtp
        && let Some(configured) =
            mlxcel_core::drafter::peek_qwen35_mtp_configured_block_size(model_path)
        && let Ok(n) = u32::try_from(configured)
    {
        return from_checkpoint(n, "the Qwen 3.5 MTP drafter's configured block size");
    }
    if kind == DrafterKind::Mtp
        && let Some(configured) =
            mlxcel_core::drafter::peek_inkling_mtp_configured_block_size(model_path)
        && let Ok(n) = u32::try_from(configured)
    {
        return from_checkpoint(n, "the Inkling MTP drafter's layer count");
    }
    if kind == DrafterKind::Mtp
        && let Some(configured) =
            mlxcel_core::drafter::peek_glm4_moe_lite_mtp_configured_block_size(model_path)
        && let Ok(n) = u32::try_from(configured)
    {
        return from_checkpoint(n, "the GLM 4 MoE Lite MTP drafter's configured block size");
    }
    // An LFM2 DSpark drafter (issue #1339) counts proposals in `block_size`
    // and runs at `min(block_size + 1, runtime_block_size)` rows by default
    // (8 on the published checkpoints); the flat DFlash default of 16 would
    // ask it for 15 proposals from a 9-proposal head.
    if kind == DrafterKind::Dflash
        && let Some(configured) =
            mlxcel_core::drafter::peek_dspark_configured_block_size(model_path)
        && let Ok(n) = u32::try_from(configured)
    {
        return from_checkpoint(n, "the DSpark drafter's runtime verify width");
    }
    // The Muse Glimmer assistant (issue #1343) publishes `block_size 16`,
    // which is the flat DFlash default; the peek exists so a checkpoint that
    // narrows it through `runtime_block_size` is honoured.
    if kind == DrafterKind::Dflash
        && let Some(configured) =
            mlxcel_core::drafter::dflash::peek_muse_assistant_configured_block_size(model_path)
        && let Ok(n) = u32::try_from(configured)
    {
        return from_checkpoint(n, "the Muse Glimmer assistant's configured block size");
    }
    resolve_measured_block_size(
        kind,
        cuda_compute_capability,
        target_quantization,
        default_block_size_for_kind(kind),
    )
}

/// [`resolve_draft_block_size_detailed`] reduced to the width alone, for
/// call sites that do not log the reason.
pub fn resolve_draft_block_size(
    override_value: Option<u32>,
    kind: DrafterKind,
    model_path: &std::path::Path,
    cuda_compute_capability: Option<(u32, u32)>,
    target_quantization: TargetQuantization,
) -> u32 {
    resolve_draft_block_size_detailed(
        override_value,
        kind,
        model_path,
        cuda_compute_capability,
        target_quantization,
    )
    .width
}

/// [`resolve_draft_block_size_detailed`] with the two hardware inputs read
/// from the running process instead of passed in.
///
/// This is the thin wrapper every binary entry point calls; the pure
/// function above stays free of the device and the filesystem so the policy
/// table is unit-testable on a host with no GPU at all. It mirrors the
/// `drafter_bf16_to_f16_policy` / `drafter_bf16_to_f16_at_load` split.
///
/// `target_model_path` is the **target** checkpoint (`-m`), not the drafter:
/// the quantization that decides the kernel path is the one the verify block
/// runs through.
///
/// Used by: the server dispatch resolution
/// (`crate::server::SpeculativeDispatch::resolve`) and the offline
/// `mlxcel generate` speculative path.
pub fn resolve_draft_block_size_for_target(
    override_value: Option<u32>,
    kind: DrafterKind,
    drafter_path: &std::path::Path,
    target_model_path: &std::path::Path,
) -> ResolvedBlockSize {
    resolve_draft_block_size_detailed(
        override_value,
        kind,
        drafter_path,
        mlxcel_core::cuda_arch::cuda_compute_capability(),
        peek_target_quantization(target_model_path),
    )
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

/// Apply the legacy `LLAMA_ARG_DRAFT_MAX` env-var fallback to the resolved
/// draft-token cap (#1433).
///
/// b10621 removed `--draft` / `--draft-max` and their `LLAMA_ARG_DRAFT_MAX`
/// binding in favor of `--spec-draft-n-max` / `LLAMA_ARG_SPEC_DRAFT_N_MAX`,
/// which is what the clap `env = ...` attribute now binds. Deployments that
/// still export the legacy variable keep working through this fallback,
/// which applies only when neither the CLI flag (any spelling) nor the
/// canonical variable provided a value.
pub fn env_fallback_draft_max(value: &mut usize, flag_was_set: bool) {
    *value = resolve_draft_max_fallback(
        *value,
        flag_was_set,
        std::env::var("LLAMA_ARG_SPEC_DRAFT_N_MAX").ok().as_deref(),
        std::env::var("LLAMA_ARG_DRAFT_MAX").ok().as_deref(),
    );
}

/// Pure core of [`env_fallback_draft_max`], separated so the precedence
/// table is unit-testable without process-global environment mutation.
///
/// Precedence (highest first): any CLI spelling of the flag, the canonical
/// `LLAMA_ARG_SPEC_DRAFT_N_MAX` (already injected by clap when set), then
/// the legacy `LLAMA_ARG_DRAFT_MAX`. An unparseable legacy value is logged
/// and ignored.
fn resolve_draft_max_fallback(
    current: usize,
    flag_was_set: bool,
    canonical_env: Option<&str>,
    legacy_env: Option<&str>,
) -> usize {
    if flag_was_set || canonical_env.is_some() {
        return current;
    }
    match legacy_env {
        Some(raw) => match raw.parse::<usize>() {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!("LLAMA_ARG_DRAFT_MAX={raw:?} is not a valid count ({e}); ignoring");
                current
            }
        },
        None => current,
    }
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
