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

//! The command-line half of the in-process client settings (issue #2173):
//! reading the shared sampling group into
//! [`mlxcel::cli::in_process_client::CliServerSettings`], which owns the
//! mapping onto the server's options.

use anyhow::Result;
use mlxcel::cli::in_process_client::{CliServerSettings, ServerSamplingOptions};
use mlxcel::cli::max_tokens::UNLIMITED_MAX_TOKENS;
use mlxcel_core::cache::KVCacheMode;

use crate::SamplingOptions;

/// Whether a flag appeared on this process's command line, in its long
/// spelling (with or without `=value`) or its short one (`-x`, `-xVALUE`).
pub(crate) fn cli_flag_was_set(long: &str, short: Option<char>) -> bool {
    if mlxcel::server::long_cli_flag_was_set(long) {
        return true;
    }
    short.is_some_and(|short| {
        short_numeric_flag_in(
            std::env::args_os().map(|arg| arg.to_string_lossy().into_owned()),
            short,
        )
    })
}

/// Whether `args` holds the short numeric flag `-x`, as `-x` alone or with
/// its value attached (`-x0.7`, `-x=0.7`). An argument that only starts with
/// `-x`, such as the prompt in `-p -xylophone`, does not count: the attached
/// part must be a number, which every short sampling flag takes.
pub(crate) fn short_numeric_flag_in(args: impl IntoIterator<Item = String>, short: char) -> bool {
    let standalone = format!("-{short}");
    args.into_iter().any(|arg| {
        let Some(rest) = arg.strip_prefix(&standalone) else {
            return false;
        };
        let value = rest.strip_prefix('=').unwrap_or(rest);
        rest.is_empty() || value.parse::<f64>().is_ok()
    })
}

/// Read the shared sampling group. `was_set` reports whether a flag was given
/// explicitly, which is what separates "use the server default" from a value
/// equal to the clap default.
pub(crate) fn settings_from_flags(
    sampling: &SamplingOptions,
    extra: ServerSamplingOptions,
    max_tokens: usize,
    kv_cache_mode: KVCacheMode,
    was_set: &dyn Fn(&str, Option<char>) -> bool,
) -> CliServerSettings {
    CliServerSettings {
        temperature: was_set("temp", Some('t')).then_some(sampling.temp),
        top_k: was_set("top-k", None).then_some(sampling.top_k),
        top_p: was_set("top-p", None).then_some(sampling.top_p),
        min_p: sampling.min_p,
        typical_p: sampling.typical_p,
        top_n_sigma: sampling.top_n_sigma,
        p_less: sampling.p_less,
        repeat_penalty: sampling.repetition_penalty,
        repeat_last_n: was_set("repeat-last-n", None).then_some(sampling.repeat_last_n),
        dry_multiplier: sampling.dry_multiplier,
        dry_base: sampling.dry_base,
        dry_allowed_length: sampling.dry_allowed_length,
        dry_penalty_last_n: was_set("dry-penalty-last-n", None)
            .then_some(sampling.dry_penalty_last_n),
        seed: sampling.seed,
        extra,
        max_tokens: (max_tokens != UNLIMITED_MAX_TOKENS).then_some(max_tokens),
        kv_cache_mode,
        ..CliServerSettings::default()
    }
}

/// Resolve the `--cache-type-k` / `--cache-type-v` / `--kv-cache-mode` group.
pub(crate) fn resolve_cli_kv_cache_mode(
    turbo: &mlxcel::cli::turbo_args::TurboKvCacheArgs,
) -> Result<KVCacheMode> {
    mlxcel::cli::turbo_args::resolve_kv_cache_mode(
        turbo.cache_type_k.as_deref(),
        turbo.cache_type_v.as_deref(),
        turbo.kv_cache_mode.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(test)]
#[path = "cli_server_tests.rs"]
mod tests;
