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
/// is overridden by, conflicts with, or is conflicted by another argument,
/// or is `exclusive`.
///
/// clap counts an env-sourced value as present and registers each
/// `overrides_with` partner as a direct conflict, while only a command-line
/// occurrence removes the overridden half (`clap_builder`
/// `parser/validator.rs`). An `env` binding on either half of an
/// `overrides_with` pair therefore turns `VAR=... --other-half` into a parse
/// error (#2609), and a `conflicts_with` pair fails the same way with no
/// override to rescue it (#2664). The same holds for the conflicts clap
/// derives from groups and from `exclusive` (#2697): members of an
/// `ArgGroup` that is not `multiple` conflict with each other, a group's own
/// `conflicts_with` applies to every member, an `exclusive` argument
/// conflicts with every other argument, and any argument conflicts with an
/// `exclusive` one. Such arguments read their variable after
/// parsing instead (see [`apply_boolish_pair_env`] and
/// `crate::server::env_fallback_cors_origins`); the binaries' tests assert
/// that this list is empty so a new pair cannot reintroduce the binding
/// (#2618).
///
/// Each entry reads `<command path> <arg id> (env <VAR>)` followed by
/// `<-> <override partners>`, `conflicts with <conflict partners>` and/or
/// `exclusive`, joined by `; ` when more than one applies.
///
/// # Panics
///
/// Panics when clap's `Debug` output for an argument no longer carries an
/// `overrides: [...]` field, or for a group no longer carries a
/// `conflicts: [...]` field, because clap has no public getter for either
/// and a silent parse failure would turn the guard into a no-op.
#[doc(hidden)]
pub fn env_bound_override_args(command: &clap::Command) -> Vec<String> {
    // clap copies `Arg::group("x")` (derive `#[arg(group = "x")]`) into the
    // group's member list only in `Command::build`, so an unbuilt command
    // would hide that membership from the group checks below.
    let mut command = command.clone();
    command.build();
    let mut found = Vec::new();
    collect_env_bound_override_args(&command, command.get_name(), &mut found);
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
        let mut conflicts = arg_conflict_partners(command, arg);
        let exclusive = arg.is_exclusive_set();
        if partners.is_empty() && conflicts.is_empty() && !exclusive {
            continue;
        }
        partners.sort();
        conflicts.sort();
        let mut relations = Vec::new();
        if !partners.is_empty() {
            relations.push(format!("<-> {}", partners.join(", ")));
        }
        if !conflicts.is_empty() {
            relations.push(format!("conflicts with {}", conflicts.join(", ")));
        }
        if exclusive {
            relations.push("exclusive".to_string());
        }
        found.push(format!(
            "{path} {id} (env {}) {}",
            env.to_string_lossy(),
            relations.join("; ")
        ));
    }
    for sub in command.get_subcommands() {
        collect_env_bound_override_args(sub, &format!("{path} {}", sub.get_name()), found);
    }
}

/// The ids of the arguments that `arg` conflicts with or that declare a
/// conflict with `arg`. clap enforces a conflict in both directions whichever
/// side declared it, so both count. Besides an argument's own
/// `conflicts_with` (through clap's public
/// [`clap::Command::get_arg_conflicts_with`]), this includes what clap's
/// validator derives (#2697): the other members of a non-`multiple` group,
/// the targets of a containing group's `conflicts_with`, and every
/// `exclusive` argument, which conflicts with anything else present.
fn arg_conflict_partners(command: &clap::Command, arg: &clap::Arg) -> Vec<String> {
    let id = arg.get_id().as_str();
    let mut partners = direct_conflicts(command, arg);
    for other in command.get_arguments() {
        let other_id = other.get_id().as_str();
        if other_id == id || partners.iter().any(|partner| partner == other_id) {
            continue;
        }
        if other.is_exclusive_set() || direct_conflicts(command, other).iter().any(|t| t == id) {
            partners.push(other_id.to_string());
        }
    }
    partners.retain(|partner| partner.as_str() != id);
    partners
}

/// The argument ids clap's validator treats as conflicting with `arg` when
/// `arg` is present (`gather_arg_direct_conflicts` in `clap_builder`
/// `parser/validator.rs`), minus overrides, which the caller reports
/// separately: the argument's own `conflicts_with`, the conflicts of every
/// group that contains it, and the other members of each such group that is
/// not `multiple`. Group ids are unrolled into their member arguments.
fn direct_conflicts(command: &clap::Command, arg: &clap::Arg) -> Vec<String> {
    let id = arg.get_id().as_str();
    let mut ids: Vec<String> = command
        .get_arg_conflicts_with(arg)
        .into_iter()
        .map(|other| other.get_id().to_string())
        .collect();
    for group in command.get_groups() {
        if !group.get_args().any(|member| member.as_str() == id) {
            continue;
        }
        if !group.clone().is_multiple() {
            ids.extend(group.get_args().map(|member| member.to_string()));
        }
        for target in group_conflicts(group) {
            ids.extend(unroll_group_or_arg(command, &target));
        }
    }
    ids.retain(|other| other.as_str() != id);
    ids.sort();
    ids.dedup();
    ids
}

/// The argument ids an id names: the argument itself, or every argument a
/// group (and any nested group) contains.
fn unroll_group_or_arg(command: &clap::Command, id: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut pending = vec![id.to_string()];
    let mut seen = Vec::new();
    while let Some(next) = pending.pop() {
        if seen.contains(&next) {
            continue;
        }
        seen.push(next.clone());
        match command
            .get_groups()
            .find(|group| group.get_id().as_str() == next)
        {
            Some(group) => pending.extend(group.get_args().map(|member| member.to_string())),
            None => out.push(next),
        }
    }
    out
}

/// The ids in a group's private `conflicts` list, read from its `Debug`
/// output (`conflicts: ["a", "b"]`) because clap exposes no getter. The
/// search starts after the `args: [` field so that it cannot match inside
/// the group's own id.
fn group_conflicts(group: &clap::ArgGroup) -> Vec<String> {
    let debug = format!("{group:?}");
    let start = debug
        .find(", args: [")
        .and_then(|args| {
            debug[args..]
                .find(", conflicts: [")
                .map(|offset| args + offset + ", conflicts: [".len())
        })
        .unwrap_or_else(|| {
            panic!(
                "clap's Debug output for group {:?} has no `args` field followed by \
                 `conflicts: [`; update cli::group_conflicts for the new format",
                group.get_id()
            )
        });
    parse_debug_id_list(&debug[start..]).unwrap_or_else(|| {
        panic!(
            "clap's Debug output for group {:?} has an unterminated `conflicts` list; \
             update cli::group_conflicts for the new format",
            group.get_id()
        )
    })
}

/// Parse the quoted, `Debug`-escaped ids of a list whose opening `[` has
/// already been consumed, up to its closing `]`. `None` when the list is
/// unterminated.
fn parse_debug_id_list(list: &str) -> Option<Vec<String>> {
    let mut ids = Vec::new();
    let mut current: Option<String> = None;
    let mut chars = list.chars();
    while let Some(ch) = chars.next() {
        match (&mut current, ch) {
            (None, ']') => return Some(ids),
            (None, '"') => current = Some(String::new()),
            (None, _) => {}
            (Some(_), '"') => ids.extend(current.take()),
            (Some(id), '\\') => id.extend(chars.next()),
            (Some(id), _) => id.push(ch),
        }
    }
    None
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
    parse_debug_id_list(&debug[start..]).unwrap_or_else(|| {
        panic!(
            "clap's Debug output for arg {:?} has an unterminated `overrides` list; \
             update cli::arg_overrides for the new format",
            arg.get_id()
        )
    })
}

#[cfg(test)]
mod env_override_guard_tests {
    use super::env_bound_override_args;
    use clap::{Arg, ArgAction, ArgGroup, Command};

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

    fn conflicting(env_on_declaring: bool, env_on_target: bool) -> Command {
        let mut declaring = Arg::new("cors_origins")
            .long("cors-origins")
            .conflicts_with("allowed_origins");
        if env_on_declaring {
            declaring = declaring.env("GUARD_SELF_TEST_CORS_ORIGINS");
        }
        let mut target = Arg::new("allowed_origins").long("allowed-origins");
        if env_on_target {
            target = target.env("GUARD_SELF_TEST_ALLOWED_ORIGINS");
        }
        Command::new("probe").arg(declaring).arg(target)
    }

    /// clap counts an env value as present for `conflicts_with` too, so
    /// `VAR=... --other` fails to parse (#2664); the guard reports an env
    /// binding on the side that declares the conflict.
    #[test]
    fn flags_env_on_the_side_declaring_a_conflict() {
        assert_eq!(
            env_bound_override_args(&conflicting(true, false)),
            vec![
                "probe cors_origins (env GUARD_SELF_TEST_CORS_ORIGINS) conflicts with \
                 allowed_origins"
                    .to_string()
            ]
        );
    }

    #[test]
    fn flags_env_on_the_conflicted_side() {
        assert_eq!(
            env_bound_override_args(&conflicting(false, true)),
            vec![
                "probe allowed_origins (env GUARD_SELF_TEST_ALLOWED_ORIGINS) conflicts with \
                 cors_origins"
                    .to_string()
            ]
        );
    }

    #[test]
    fn ignores_conflicts_without_env() {
        assert!(env_bound_override_args(&conflicting(false, false)).is_empty());
    }

    #[test]
    fn walks_subcommands() {
        let root = Command::new("root").subcommand(pair(true, false).name("serve"));
        assert_eq!(
            env_bound_override_args(&root),
            vec!["root serve feature (env GUARD_SELF_TEST_FEATURE) <-> no_feature".to_string()]
        );
    }

    fn grouped(multiple: bool) -> Command {
        Command::new("probe")
            .arg(Arg::new("first").long("first").env("GUARD_SELF_TEST_FIRST"))
            .arg(Arg::new("second").long("second"))
            .group(
                ArgGroup::new("pair")
                    .args(["first", "second"])
                    .multiple(multiple),
            )
    }

    /// #2697: members of a group that is not `multiple` conflict with each
    /// other in clap's validator, so an env-bound member turns
    /// `VAR=... --other-member` into a parse error.
    #[test]
    fn flags_env_in_a_non_multiple_group() {
        assert_eq!(
            env_bound_override_args(&grouped(false)),
            vec!["probe first (env GUARD_SELF_TEST_FIRST) conflicts with second".to_string()]
        );
        assert!(
            env_bound_override_args(&grouped(true)).is_empty(),
            "a multiple group without conflicts adds none"
        );
    }

    /// #2697: group membership declared on the argument (`Arg::group`, the
    /// derive `#[arg(group = ...)]` form) only reaches the group when clap
    /// builds the command, and a group created that way is not `multiple`.
    #[test]
    fn flags_env_in_a_group_declared_on_the_argument() {
        let command = Command::new("probe")
            .arg(
                Arg::new("first")
                    .long("first")
                    .env("GUARD_SELF_TEST_FIRST")
                    .group("pair"),
            )
            .arg(Arg::new("second").long("second").group("pair"));
        assert_eq!(
            env_bound_override_args(&command),
            vec!["probe first (env GUARD_SELF_TEST_FIRST) conflicts with second".to_string()]
        );
    }

    /// #2697: a group's own `conflicts_with` applies to every member, in
    /// both directions.
    #[test]
    fn flags_env_in_a_group_that_conflicts_with_another_arg() {
        let command = |env_on_member: bool, env_on_target: bool| {
            let mut member = Arg::new("member").long("member");
            if env_on_member {
                member = member.env("GUARD_SELF_TEST_MEMBER");
            }
            let mut target = Arg::new("target").long("target");
            if env_on_target {
                target = target.env("GUARD_SELF_TEST_TARGET");
            }
            Command::new("probe").arg(member).arg(target).group(
                ArgGroup::new("members")
                    .arg("member")
                    .multiple(true)
                    .conflicts_with("target"),
            )
        };
        assert_eq!(
            env_bound_override_args(&command(true, false)),
            vec!["probe member (env GUARD_SELF_TEST_MEMBER) conflicts with target".to_string()]
        );
        assert_eq!(
            env_bound_override_args(&command(false, true)),
            vec!["probe target (env GUARD_SELF_TEST_TARGET) conflicts with member".to_string()]
        );
        assert!(env_bound_override_args(&command(false, false)).is_empty());
    }

    /// #2697: a group's `conflicts_with` naming another group conflicts with
    /// every member of that group.
    #[test]
    fn flags_env_in_a_group_conflicting_with_another_group() {
        let command = Command::new("probe")
            .arg(Arg::new("left").long("left").env("GUARD_SELF_TEST_LEFT"))
            .arg(Arg::new("right").long("right"))
            .group(
                ArgGroup::new("lefts")
                    .arg("left")
                    .multiple(true)
                    .conflicts_with("rights"),
            )
            .group(ArgGroup::new("rights").arg("right").multiple(true));
        assert_eq!(
            env_bound_override_args(&command),
            vec!["probe left (env GUARD_SELF_TEST_LEFT) conflicts with right".to_string()]
        );
    }

    /// #2697: an `exclusive` argument conflicts with every other argument
    /// present, an env-sourced value included.
    #[test]
    fn flags_an_exclusive_env_arg() {
        let command = Command::new("probe")
            .arg(
                Arg::new("alone")
                    .long("alone")
                    .exclusive(true)
                    .env("GUARD_SELF_TEST_ALONE"),
            )
            .arg(Arg::new("other").long("other"));
        assert_eq!(
            env_bound_override_args(&command),
            vec!["probe alone (env GUARD_SELF_TEST_ALONE) exclusive".to_string()]
        );
    }

    /// #2697: an env-bound argument beside an `exclusive` one turns
    /// `VAR=... --exclusive-flag` into a parse error.
    #[test]
    fn flags_an_env_arg_beside_an_exclusive_arg() {
        let command = Command::new("probe")
            .arg(Arg::new("alone").long("alone").exclusive(true))
            .arg(Arg::new("other").long("other").env("GUARD_SELF_TEST_OTHER"));
        assert_eq!(
            env_bound_override_args(&command),
            vec!["probe other (env GUARD_SELF_TEST_OTHER) conflicts with alone".to_string()]
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
