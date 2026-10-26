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

//! Shared CLI argument groups exposed across the mlxcel binaries.
//!
//! Modules under this namespace own the canonical clap definitions for flag
//! sets that more than one binary (`mlxcel`, `mlxcel-server`, future tooling)
//! must accept identically. Any binary that wants to expose a shared flag
//! group flattens the corresponding struct via `#[command(flatten)]` so the
//! help text, env-var bridges, and resolution helpers stay defined in exactly
//! one place.

pub mod batch_quant_args;
pub mod cache_args;
pub mod cache_list;
pub mod chat_compat_args;
pub mod completion;
pub mod context_args;
pub mod draft_block_policy;
pub mod embedding_compat_args;
pub mod flag_surface;
pub mod ggml_compat_args;
pub mod grammar_args;
pub mod in_process_client;
pub mod infill_args;
pub mod llama_short_flags;
pub mod logging_compat_args;
pub mod max_tokens;
pub mod multimodal_compat_args;
pub mod rope_args;
pub mod slot_args;
pub mod spec_compat_args;
pub mod speculative_args;
pub mod turbo_args;
pub mod ui_compat_args;

/// Fill one half of a b10621 `--x` / `--no-x` pair from its `LLAMA_ARG_*`
/// variable, when neither half was given on the command line.
///
/// This is what clap's `env` attribute did for these pairs, minus the bug:
/// clap counts an env-sourced value as present and registers every
/// `overrides_with` partner as a direct conflict, while only a command-line
/// occurrence removes the overridden half, so `LLAMA_ARG_X=1 --no-x` failed to
/// parse (#2609). Reading the variable after parsing keeps the precedence
/// b10621 has: an explicit flag, either half, beats the environment.
///
/// The value domain is clap's `BoolishValueParser`, which the flag itself
/// still uses (`y`/`yes`/`t`/`true`/`on`/`1` and `n`/`no`/`f`/`false`/`off`/`0`,
/// case-insensitive), so an environment value means exactly what it meant
/// under the clap binding.
///
/// # Errors
///
/// Returns the variable name and its raw value when the value is not one of
/// those spellings.
pub(crate) fn apply_boolish_pair_env(
    var: &'static str,
    value: &mut Option<bool>,
    negated: bool,
) -> Result<(), (&'static str, String)> {
    if value.is_some() || negated {
        return Ok(());
    }
    let Some(raw) = std::env::var_os(var) else {
        return Ok(());
    };
    let raw = raw.to_string_lossy().into_owned();
    match parse_boolish(&raw) {
        Some(parsed) => {
            *value = Some(parsed);
            Ok(())
        }
        None => Err((var, raw)),
    }
}

/// clap's `BoolishValueParser` vocabulary, case-insensitive.
fn parse_boolish(raw: &str) -> Option<bool> {
    let lowered = raw.to_ascii_lowercase();
    match lowered.as_str() {
        "y" | "yes" | "t" | "true" | "on" | "1" => Some(true),
        "n" | "no" | "f" | "false" | "off" | "0" => Some(false),
        _ => None,
    }
}
