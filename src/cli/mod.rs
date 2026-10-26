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

/// List every argument of `command` and its subcommands that binds an
/// environment variable through clap's `env` attribute while it overrides,
/// or is overridden by, another argument.
///
/// clap counts an env-sourced value as present and registers each
/// `overrides_with` partner as a direct conflict, while only a command-line
/// occurrence removes the overridden half (`clap_builder`
/// `parser/validator.rs`). An `env` binding on either half of an
/// `overrides_with` pair therefore turns `VAR=... --other-half` into a parse
/// error (#2609). Pairs read their variable after parsing instead (see
/// [`apply_boolish_pair_env`]); the binaries' tests assert that this list is
/// empty so a new pair cannot reintroduce the binding (#2618).
///
/// Each entry reads `<command path> <arg id> (env <VAR>) <-> <partners>`.
///
/// # Panics
///
/// Panics when clap's `Debug` output for an argument no longer carries an
/// `overrides: [...]` field, because clap has no public getter for it and a
/// silent parse failure would turn the guard into a no-op.
#[doc(hidden)]
pub fn env_bound_override_args(command: &clap::Command) -> Vec<String> {
    let mut found = Vec::new();
    collect_env_bound_override_args(command, command.get_name(), &mut found);
    found
}

fn collect_env_bound_override_args(command: &clap::Command, path: &str, found: &mut Vec<String>) {
    let overrides: Vec<(String, Vec<String>)> = command
        .get_arguments()
        .map(|arg| (arg.get_id().to_string(), arg_overrides(arg)))
        .collect();
    for arg in command.get_arguments() {
        let Some(env) = arg.get_env() else {
            continue;
        };
        let id = arg.get_id().as_str();
        let mut partners: Vec<String> = overrides
            .iter()
            .filter(|(other, _)| other != id)
            .filter(|(_, list)| list.iter().any(|target| target == id))
            .map(|(other, _)| other.clone())
            .collect();
        for target in overrides
            .iter()
            .find(|(other, _)| other == id)
            .map(|(_, list)| list.as_slice())
            .unwrap_or_default()
        {
            if target != id && !partners.contains(target) {
                partners.push(target.clone());
            }
        }
        if !partners.is_empty() {
            partners.sort();
            found.push(format!(
                "{path} {id} (env {}) <-> {}",
                env.to_string_lossy(),
                partners.join(", ")
            ));
        }
    }
    for sub in command.get_subcommands() {
        collect_env_bound_override_args(sub, &format!("{path} {}", sub.get_name()), found);
    }
}

/// The ids in an argument's private `overrides` list, read from its `Debug`
/// output (`overrides: ["a", "b"]`) because clap exposes no getter.
///
/// clap prints `help` and `long_help` before `settings` and `overrides`, so
/// the search starts after the `settings: ArgFlags(` field; help text that
/// happens to contain `overrides: [` cannot be mistaken for the list.
fn arg_overrides(arg: &clap::Arg) -> Vec<String> {
    const SETTINGS: &str = ", settings: ArgFlags(";
    const OVERRIDES: &str = ", overrides: [";
    let debug = format!("{arg:?}");
    let start = debug
        .find(SETTINGS)
        .and_then(|settings| {
            debug[settings..]
                .find(OVERRIDES)
                .map(|offset| settings + offset + OVERRIDES.len())
        })
        .unwrap_or_else(|| {
            panic!(
                "clap's Debug output for arg {:?} has no `settings` field followed by \
                 `overrides: [`; update cli::arg_overrides for the new format",
                arg.get_id()
            )
        });
    let mut ids = Vec::new();
    let mut current: Option<String> = None;
    let mut chars = debug[start..].chars();
    while let Some(ch) = chars.next() {
        match (&mut current, ch) {
            (None, ']') => return ids,
            (None, '"') => current = Some(String::new()),
            (None, _) => {}
            (Some(_), '"') => ids.extend(current.take()),
            (Some(id), '\\') => id.extend(chars.next()),
            (Some(id), _) => id.push(ch),
        }
    }
    panic!(
        "clap's Debug output for arg {:?} has an unterminated `overrides` list; \
         update cli::arg_overrides for the new format",
        arg.get_id()
    );
}

#[cfg(test)]
mod env_override_guard_tests {
    use super::env_bound_override_args;
    use clap::{Arg, ArgAction, Command};

    fn pair(env_on_positive: bool, env_on_negative: bool) -> Command {
        let mut positive = Arg::new("feature")
            .long("feature")
            .action(ArgAction::SetTrue)
            .overrides_with("no_feature");
        if env_on_positive {
            positive = positive.env("GUARD_SELF_TEST_FEATURE");
        }
        let mut negative = Arg::new("no_feature")
            .long("no-feature")
            .action(ArgAction::SetTrue);
        if env_on_negative {
            negative = negative.env("GUARD_SELF_TEST_NO_FEATURE");
        }
        Command::new("probe").arg(positive).arg(negative)
    }

    /// The guard reads clap's private `overrides` field from `Debug` output,
    /// so this pins that it still finds a synthetic `env` + `overrides_with`
    /// argument; a clap format change fails here rather than silently.
    #[test]
    fn flags_env_on_the_overriding_half() {
        assert_eq!(
            env_bound_override_args(&pair(true, false)),
            vec!["probe feature (env GUARD_SELF_TEST_FEATURE) <-> no_feature".to_string()]
        );
    }

    #[test]
    fn flags_env_on_the_overridden_half() {
        assert_eq!(
            env_bound_override_args(&pair(false, true)),
            vec!["probe no_feature (env GUARD_SELF_TEST_NO_FEATURE) <-> feature".to_string()]
        );
    }

    #[test]
    fn ignores_pairs_without_env_and_env_without_pairs() {
        assert!(env_bound_override_args(&pair(false, false)).is_empty());
        let unpaired = Command::new("probe").arg(Arg::new("port").long("port").env("GUARD_PORT"));
        assert!(env_bound_override_args(&unpaired).is_empty());
    }

    #[test]
    fn walks_subcommands() {
        let root = Command::new("root").subcommand(pair(true, false).name("serve"));
        assert_eq!(
            env_bound_override_args(&root),
            vec!["root serve feature (env GUARD_SELF_TEST_FEATURE) <-> no_feature".to_string()]
        );
    }

    /// Help text is printed before the `overrides` field, so a help string
    /// that looks like the list must not be read as it.
    #[test]
    fn help_text_that_looks_like_the_overrides_field_is_ignored() {
        let command = Command::new("probe")
            .arg(
                Arg::new("feature")
                    .long("feature")
                    .action(ArgAction::SetTrue)
                    .env("GUARD_SELF_TEST_FEATURE")
                    .help("decoy, overrides: [\"bogus\"]"),
            )
            .arg(
                Arg::new("other")
                    .long("other")
                    .action(ArgAction::SetTrue)
                    .help("decoy, overrides: [\"feature\"]"),
            );
        assert!(env_bound_override_args(&command).is_empty());
    }
}
