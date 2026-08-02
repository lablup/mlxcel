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

//! Help-text consistency invariants for the shared CLI flag groups.
//!
//! Two flag groups are covered:
//!
//! 1. **TurboQuant KV-cache** (`KV Cache (TurboQuant) Options`), every
//!    binary flattens the same `TurboKvCacheArgs`, so the `--help` text
//!    for `--cache-type-k`, `--cache-type-v`, `--kv-cache-mode`, and
//!    `--turbo-boundary-v` MUST be identical across binaries.
//! 2. **Speculative decoding** (`Speculative Decoding Options`), every binary flattens the same `SpeculativeArgs`, so the
//!    `--help` text for `--draft-kind` and `--draft-block-size` MUST be
//!    identical across binaries.
//!
//! All three CLI surfaces, `mlxcel generate`, `mlxcel serve`, and
//! `mlxcel-server`, flatten these groups via `#[command(flatten)]`. The
//! tests run each binary's `--help`, extract the relevant section block,
//! and assert that every required flag, value-name, and accepted token
//! appears in all three. Inside the same block they also forbid
//! closed-repo issue/epic numbers from leaking back into operator-facing
//! help text. The forbidden-substring check is intentionally scoped to
//! each block, pre-existing references on unrelated flags elsewhere in
//! the help output are out of scope for this invariant.
//!
//! When a future flag or mode is added to either shared group, all three
//! binaries fail the test together, no drift is possible.

use std::path::Path;
use std::process::Command;

mod common;
use common::resolve_repo_binary;

const HEADING: &str = "KV Cache (TurboQuant) Options";

/// Long-form flag names that MUST appear under the heading on every binary.
const EXPECTED_FLAGS: &[&str] = &[
    "--cache-type-k",
    "--cache-type-v",
    "--kv-cache-mode",
    "--turbo-boundary-v",
];

/// Mode tokens that MUST appear in the help block on every binary. The aliases
/// are part of the contract so a user reading the help on any binary can
/// discover the alternate spellings without reading the source.
const EXPECTED_MODES: &[&str] = &[
    "fp16",
    "int8",
    "fp16+turbo4",
    "fp16+turbo3",
    "turbo4",
    "turbo4-delegated",
    "turbo4-asym",
    "turbo3-asym",
    "turbo4-sym",
];

/// Substrings that MUST NOT appear within the TurboQuant KV-cache help block.
/// Closed-repo issue/epic numbers leak internal tracking IDs into the public
/// help text and have no value for end users. The check is scoped to the
/// block introduced by this PR (the four shared TurboQuant flags); pre-existing
/// references on unrelated server flags are out of scope for this invariant.
const FORBIDDEN_SUBSTRINGS: &[&str] = &["issue #", "epic #", "B-step #", "Issue #", "Epic #"];

/// Env vars that any of the binaries' clap definitions consult via
/// `#[arg(env = "...")]` AND that could materialize as `[env: NAME=value]`
/// in the rendered help block. Cleared before spawn so the test is
/// deterministic regardless of the host shell's environment (CI runners or
/// developer shells with `LLAMA_ARG_*` set otherwise leak values into help
/// output and could in theory trip the forbidden-substring check).
///
/// Only the env vars consumed by `TurboKvCacheArgs` itself need to be
/// cleared, the broader llama-server compatibility env vars are scoped to
/// other flag groups whose help isn't asserted here.
const ENV_TO_CLEAR_FOR_HELP: &[&str] = &["LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V"];

/// Run `--help` on a binary and return the resulting stdout. Panics with a
/// descriptive message when the binary fails to execute.
fn help_output(bin_name: &str, args: &[&str]) -> String {
    let (path, resolution) = resolve_repo_binary(bin_name);
    let mut cmd = Command::new(&path);
    cmd.args(args);
    for key in ENV_TO_CLEAR_FOR_HELP {
        cmd.env_remove(key);
    }
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {bin_name} from {path:?}: {e}\n{resolution}"));
    assert!(
        output.status.success(),
        "{} {:?} exited with status {:?}: stderr=\n{}",
        bin_name,
        args,
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Slice the help output to the `KV Cache (TurboQuant) Options` block:
/// from the heading line to the next help heading or end of input.
///
/// clap renders section headings on their own line ending in `:` followed
/// by an indented list of flag entries. The next heading is the next
/// non-indented line that ends with `:` (e.g. `Options:`, `Generation
/// Options:`). For the invariant tests we only need to ensure the block
/// contains the relevant flags, so an over-inclusive cut is fine, better
/// to fail loud than to silently miss content.
fn extract_kv_cache_block(help: &str) -> &str {
    let start = help
        .find(HEADING)
        .unwrap_or_else(|| panic!("help heading {HEADING:?} not found in:\n{help}"));

    // Walk lines after the heading and find the byte offset of the next
    // section heading. clap headings are non-indented (do not start with a
    // space) and end with `:`.
    let after_heading_offset = help[start..]
        .find('\n')
        .map(|i| start + i + 1)
        .unwrap_or(help.len());

    let mut cursor = after_heading_offset;
    let mut end = help.len();
    for line in help[after_heading_offset..].split_inclusive('\n') {
        let trimmed = line.trim_end_matches('\n');
        let is_section_heading = trimmed.ends_with(':')
            && !trimmed.starts_with(' ')
            && !trimmed.contains("--")
            && trimmed.chars().next().is_some_and(|c| c.is_uppercase())
            && trimmed.len() < 80;
        if is_section_heading {
            end = cursor;
            break;
        }
        cursor += line.len();
    }
    &help[start..end]
}

/// Assert every required flag, mode, and absence of forbidden substrings
/// for one binary's help block.
fn assert_invariants(label: &str, help: &str) {
    let block = extract_kv_cache_block(help);

    for flag in EXPECTED_FLAGS {
        assert!(
            block.contains(flag),
            "{label}: KV Cache help block is missing flag {flag:?}.\n\
             Block was:\n{block}"
        );
    }

    for mode in EXPECTED_MODES {
        assert!(
            block.contains(mode),
            "{label}: KV Cache help block is missing mode token {mode:?}.\n\
             Block was:\n{block}"
        );
    }

    for forbidden in FORBIDDEN_SUBSTRINGS {
        if let Some(idx) = block.find(forbidden) {
            // Multi-byte chars (em-dashes etc.) would panic on raw byte
            // slicing, so use char-boundary-safe windowing for the error
            // context.
            let window_start = idx.saturating_sub(40);
            let window_end = (idx + forbidden.len() + 40).min(block.len());
            let safe_start = (0..=window_start)
                .rev()
                .find(|&i| block.is_char_boundary(i))
                .unwrap_or(0);
            let safe_end = (window_end..=block.len())
                .find(|&i| block.is_char_boundary(i))
                .unwrap_or(block.len());
            panic!(
                "{label}: KV Cache help block contains forbidden closed-repo \
                 reference {forbidden:?}. Move the reference into a non-doc \
                 `//` comment in src/cli/turbo_args.rs or remove it.\n\
                 Match context: {:?}",
                &block[safe_start..safe_end]
            );
        }
    }
}

#[test]
fn mlxcel_generate_help_lists_all_turbo_flags_and_modes() {
    let help = help_output("mlxcel", &["generate", "--help"]);
    assert_invariants("mlxcel generate", &help);
}

#[test]
fn mlxcel_serve_help_lists_all_turbo_flags_and_modes() {
    let help = help_output("mlxcel", &["serve", "--help"]);
    assert_invariants("mlxcel serve", &help);
}

#[test]
fn mlxcel_server_help_lists_all_turbo_flags_and_modes() {
    let help = help_output("mlxcel-server", &["--help"]);
    assert_invariants("mlxcel-server", &help);
}

/// Issue #95: `mlxcel run` flattens the same `GenerationOptions` group as
/// `mlxcel generate` (which carries the shared `TurboKvCacheArgs`), so its
/// `--help` MUST expose the identical TurboQuant KV-cache flag block. This
/// locks the new `run` verb into the same cross-binary invariant the other
/// surfaces already satisfy.
#[test]
fn mlxcel_run_help_lists_all_turbo_flags_and_modes() {
    let help = help_output("mlxcel", &["run", "--help"]);
    assert_invariants("mlxcel run", &help);
}

/// Issue #95: `mlxcel run` shares `generate`'s sampling/generation flag groups
/// and documents the mlx-lm-style default-model fallback. Assert the shared
/// flags and the documented default repo-id are present so the `run` surface
/// cannot silently drop them or change the default without updating this test.
#[test]
fn mlxcel_run_help_lists_shared_flags_and_default_model() {
    let help = help_output("mlxcel", &["run", "--help"]);

    // Shared generation/sampling flags (the same clap groups `generate` uses).
    for sig in [
        "--prompt <TEXT>",
        "--max-tokens <N>",
        "--temp <FLOAT>",
        "--top-p <FLOAT>",
        "--top-k <K>",
        "--no-chat-template",
        "--adapter <PATH>",
    ] {
        assert!(
            help.contains(sig),
            "mlxcel run help is missing shared flag {sig:?}.\nHelp was:\n{help}"
        );
    }

    // The documented default model. If the default repo-id changes, this test
    // forces the help text + README to be updated too.
    assert!(
        help.contains("mlx-community/gemma-4-e2b-it-4bit"),
        "mlxcel run help must document the default model repo-id.\nHelp was:\n{help}"
    );
}

/// Cross-binary equivalence: the four shared flags should appear with the
/// same names and same value-name in every binary's help block. We do NOT
/// require byte-identical blocks because clap interleaves binary-specific
/// flags around the heading boundary on `mlxcel-server` (no-subcommand
/// invocation). Instead, we assert each flag's "long-form + value-name"
/// pair appears identically.
#[test]
fn turbo_flag_signatures_match_across_binaries() {
    let generate_help = help_output("mlxcel", &["generate", "--help"]);
    let serve_help = help_output("mlxcel", &["serve", "--help"]);
    let server_help = help_output("mlxcel-server", &["--help"]);
    // Issue #95: `run` flattens the same `GenerationOptions` (TurboKvCacheArgs)
    // group, so it must carry the identical flag signatures.
    let run_help = help_output("mlxcel", &["run", "--help"]);

    let signatures = [
        "--cache-type-k <TYPE>",
        "--cache-type-v <TYPE>",
        "--kv-cache-mode <MODE>",
        "--turbo-boundary-v <COUNT>",
    ];
    for sig in signatures {
        assert!(
            generate_help.contains(sig),
            "mlxcel generate is missing flag signature {sig:?}"
        );
        assert!(
            serve_help.contains(sig),
            "mlxcel serve is missing flag signature {sig:?}"
        );
        assert!(
            server_help.contains(sig),
            "mlxcel-server is missing flag signature {sig:?}"
        );
        assert!(
            run_help.contains(sig),
            "mlxcel run is missing flag signature {sig:?}"
        );
    }
}

// ── Speculative decoding flag group ─────────────────────────────

/// Heading set by `SpeculativeArgs` (`src/cli/speculative_args.rs`).
const SPECULATIVE_HEADING: &str = "Speculative Decoding Options";

/// Long-form flag names that MUST appear under the
/// [`SPECULATIVE_HEADING`] block on every binary.
const SPECULATIVE_EXPECTED_FLAGS: &[&str] = &["--draft-kind", "--draft-block-size"];

/// Drafter-kind tokens that MUST appear in the help block on every
/// binary. Mirrors the user-selectable subset of
/// `mlxcel_core::drafter::KNOWN_DRAFTER_KINDS` (the third
/// `internal-mtp` variant is intentionally excluded from CLI parsing,
/// see `SpeculativeArgs::parse_kind`).
const SPECULATIVE_EXPECTED_KINDS: &[&str] = &["dflash", "mtp"];

/// Env vars consulted by the speculative-decoding flag group via
/// `#[arg(env = "...")]`. Cleared before spawn so help output is
/// deterministic regardless of the host shell's environment.
const SPECULATIVE_ENV_TO_CLEAR_FOR_HELP: &[&str] =
    &["LLAMA_ARG_DRAFT_KIND", "LLAMA_ARG_DRAFT_BLOCK_SIZE"];

/// Slice the help output to the `Speculative Decoding Options` block,
/// using the same logic as [`extract_kv_cache_block`].
fn extract_speculative_block(help: &str) -> &str {
    let start = help
        .find(SPECULATIVE_HEADING)
        .unwrap_or_else(|| panic!("help heading {SPECULATIVE_HEADING:?} not found in:\n{help}"));

    let after_heading_offset = help[start..]
        .find('\n')
        .map(|i| start + i + 1)
        .unwrap_or(help.len());

    let mut cursor = after_heading_offset;
    let mut end = help.len();
    for line in help[after_heading_offset..].split_inclusive('\n') {
        let trimmed = line.trim_end_matches('\n');
        let is_section_heading = trimmed.ends_with(':')
            && !trimmed.starts_with(' ')
            && !trimmed.contains("--")
            && trimmed.chars().next().is_some_and(|c| c.is_uppercase())
            && trimmed.len() < 80;
        if is_section_heading {
            end = cursor;
            break;
        }
        cursor += line.len();
    }
    &help[start..end]
}

/// Run `--help` on a binary with the speculative env vars cleared and
/// return stdout.
fn help_output_for_speculative(bin_name: &str, args: &[&str]) -> String {
    let (path, resolution) = resolve_repo_binary(bin_name);
    let mut cmd = Command::new(&path);
    cmd.args(args);
    for key in SPECULATIVE_ENV_TO_CLEAR_FOR_HELP {
        cmd.env_remove(key);
    }
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {bin_name} from {path:?}: {e}\n{resolution}"));
    assert!(
        output.status.success(),
        "{} {:?} exited with status {:?}: stderr=\n{}",
        bin_name,
        args,
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Per-binary assertion for the speculative-decoding help block.
fn assert_speculative_invariants(label: &str, help: &str) {
    let block = extract_speculative_block(help);

    for flag in SPECULATIVE_EXPECTED_FLAGS {
        assert!(
            block.contains(flag),
            "{label}: Speculative Decoding help block is missing flag {flag:?}.\n\
             Block was:\n{block}"
        );
    }

    for kind in SPECULATIVE_EXPECTED_KINDS {
        assert!(
            block.contains(kind),
            "{label}: Speculative Decoding help block is missing kind token {kind:?}.\n\
             Block was:\n{block}"
        );
    }

    for forbidden in FORBIDDEN_SUBSTRINGS {
        if let Some(idx) = block.find(forbidden) {
            let window_start = idx.saturating_sub(40);
            let window_end = (idx + forbidden.len() + 40).min(block.len());
            let safe_start = (0..=window_start)
                .rev()
                .find(|&i| block.is_char_boundary(i))
                .unwrap_or(0);
            let safe_end = (window_end..=block.len())
                .find(|&i| block.is_char_boundary(i))
                .unwrap_or(block.len());
            panic!(
                "{label}: Speculative Decoding help block contains forbidden \
                 closed-repo reference {forbidden:?}. Move the reference into a \
                 non-doc `//` comment in src/cli/speculative_args.rs or remove it.\n\
                 Match context: {:?}",
                &block[safe_start..safe_end]
            );
        }
    }
}

#[test]
fn mlxcel_generate_help_lists_all_speculative_flags_and_kinds() {
    let help = help_output_for_speculative("mlxcel", &["generate", "--help"]);
    assert_speculative_invariants("mlxcel generate", &help);
}

#[test]
fn mlxcel_serve_help_lists_all_speculative_flags_and_kinds() {
    let help = help_output_for_speculative("mlxcel", &["serve", "--help"]);
    assert_speculative_invariants("mlxcel serve", &help);
}

#[test]
fn mlxcel_server_help_lists_all_speculative_flags_and_kinds() {
    let help = help_output_for_speculative("mlxcel-server", &["--help"]);
    assert_speculative_invariants("mlxcel-server", &help);
}

/// Cross-binary equivalence: each speculative flag should appear with
/// the same long-form + value-name pair in every binary's help block.
#[test]
fn speculative_flag_signatures_match_across_binaries() {
    let generate_help = help_output_for_speculative("mlxcel", &["generate", "--help"]);
    let serve_help = help_output_for_speculative("mlxcel", &["serve", "--help"]);
    let server_help = help_output_for_speculative("mlxcel-server", &["--help"]);

    let signatures = ["--draft-kind <KIND>", "--draft-block-size <N>"];
    for sig in signatures {
        assert!(
            generate_help.contains(sig),
            "mlxcel generate is missing flag signature {sig:?}"
        );
        assert!(
            serve_help.contains(sig),
            "mlxcel serve is missing flag signature {sig:?}"
        );
        assert!(
            server_help.contains(sig),
            "mlxcel-server is missing flag signature {sig:?}"
        );
    }
}

// ── Drafter flag aliases (issue #464) ───────────────────────────
//
// `mlxcel serve` and `mlxcel-server` intentionally keep opposite primary
// spellings for the drafter-path and draft-token-count flags (mlx-lm vs.
// llama-server style), but both must accept both spellings so a command
// line copied between the two binaries parses unchanged. This pins the
// `--help` output on each binary to document the alias so an operator
// discovers the alternate spelling without reading the source. Unit tests
// in `src/main_tests.rs` and `src/bin/mlx_server.rs` cover that the
// aliases resolve to the identical parsed value.

/// Leading-space count of a help line.
fn indent_width(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

/// Slice the `--help` entry clap rendered for `flag_signature`: the signature
/// line plus every following line indented deeper than it, which is the
/// flag's description and its bracketed annotations. Stops before the next
/// flag entry (same or shallower indent) and before the next section heading.
///
/// `flag_signature` must be the complete rendered signature including the
/// value name, for example `--draft-model <PATH>`, because the anchor line is
/// matched in full.
///
/// Scoping to one entry matters: the prose on `--draft-model` names the
/// alternate spelling in its own description, so a whole-help substring search
/// for the alias name would pass even with the `visible_alias` removed.
fn flag_help_entry<'a>(help: &'a str, flag_signature: &str) -> Option<&'a str> {
    let mut offset = 0usize;
    let mut entry_start = None;
    for line in help.split_inclusive('\n') {
        let trimmed = line.trim();
        // clap's long help puts the signature alone on its line, as
        // `      --flag <VALUE>` for a long-only flag or `  -s, --flag <VALUE>`
        // for one carrying a short form. Match the whole line rather than a
        // prefix: `--metrics` is a prefix of the separate `--metrics-port
        // <PORT>` entry, and a description line that merely mentions the flag
        // would otherwise anchor the slice on prose. A short form is always
        // exactly two characters, which rules out a hyphen-bulleted line.
        let starts_entry = trimmed == flag_signature
            || trimmed.split_once(", ").is_some_and(|(short, rest)| {
                short.len() == 2 && short.starts_with('-') && rest == flag_signature
            });
        if starts_entry {
            entry_start = Some(offset);
            break;
        }
        offset += line.len();
    }
    let start = entry_start?;
    let rest = &help[start..];

    let mut lines = rest.split_inclusive('\n');
    let signature_line = lines.next()?;
    let signature_indent = indent_width(signature_line);

    let mut consumed = signature_line.len();
    let mut body_indent: Option<usize> = None;
    for line in lines {
        if line.trim().is_empty() {
            consumed += line.len();
            continue;
        }
        let indent = indent_width(line);
        match body_indent {
            // The first non-blank line fixes the body indent. A line no
            // deeper than the signature means this flag has no body at all.
            None if indent <= signature_indent => break,
            None => body_indent = Some(indent),
            Some(body) if indent < body => break,
            Some(_) => {}
        }
        consumed += line.len();
    }
    Some(&rest[..consumed])
}

/// Alias names clap documented for one flag entry, parsed out of its
/// `[alias: ...]` / `[aliases: ...]` annotation.
///
/// The annotation must START a line, which is how clap renders it (on its own
/// line at the description indent). A bare substring search would also accept
/// a bracketed span someone wrote by hand inside a doc comment, and the help
/// output already contains such spans on other flags, so the anchor is what
/// keeps a prose `[alias: --foo]` from silently satisfying the contract.
///
/// Both labels are accepted because the label is not part of the contract:
/// `clap_builder` renders `[alias: ...]` for exactly one visible alias and
/// `[aliases: ...]` for two or more (the `pluralize` call in its help
/// template, added in 4.6.2). Pinning the singular literal instead would
/// break again the moment a second alias is added to any of these flags.
/// What the contract needs is the alias NAME, and that it is carried by a
/// real clap alias annotation rather than by prose.
fn documented_aliases(entry: &str) -> Vec<String> {
    let mut offset = 0usize;
    for line in entry.split_inclusive('\n') {
        let trimmed = line.trim_start();
        let trimmed_offset = offset + (line.len() - trimmed.len());
        offset += line.len();
        for label in ["[aliases: ", "[alias: "] {
            if !trimmed.starts_with(label) {
                continue;
            }
            let inner_start = trimmed_offset + label.len();
            let Some(len) = entry[inner_start..].find(']') else {
                continue;
            };
            return entry[inner_start..inner_start + len]
                .split(',')
                // The annotation wraps across lines when it is long, so
                // collapse whitespace rather than trimming spaces only.
                .map(|alias| alias.split_whitespace().collect::<Vec<_>>().join(" "))
                .filter(|alias| !alias.is_empty())
                .collect();
        }
    }
    Vec::new()
}

/// Assert one binary's `--help` documents `alias` as a clap alias of the flag
/// rendered as `flag_signature`.
fn assert_flag_documents_alias(label: &str, help: &str, flag_signature: &str, alias: &str) {
    let entry = flag_help_entry(help, flag_signature).unwrap_or_else(|| {
        panic!("{label} --help has no flag entry for {flag_signature:?}.\nHelp was:\n{help}")
    });
    let aliases = documented_aliases(entry);
    assert!(
        aliases.iter().any(|documented| documented == alias),
        "{label} --help must document {flag_signature:?} with a {alias:?} alias (issue #464). \
         The alias annotation clap rendered for that flag listed {aliases:?}.\n\
         Entry was:\n{entry}"
    );
}

/// Regression test for the defect that actually took the nightly down: the
/// CLI binaries must be resolved from the path cargo built them at, never from
/// a `<manifest>/target/<profile>/` reconstruction. `CARGO_BIN_EXE_*` accounts
/// for `CARGO_TARGET_DIR`, the profile name, and the target triple, so this
/// holds for a plain run, for the nightly's external target directory, for
/// `--profile test-fast`, and for a cross build (issue #962).
#[test]
fn cli_binaries_resolve_to_the_path_cargo_built_them_at() {
    for (name, built_by_cargo) in [
        ("mlxcel", env!("CARGO_BIN_EXE_mlxcel")),
        ("mlxcel-server", env!("CARGO_BIN_EXE_mlxcel-server")),
    ] {
        let (resolved, report) = resolve_repo_binary(name);
        assert_eq!(
            resolved,
            Path::new(built_by_cargo),
            "{name} must resolve to the binary cargo built, not a reconstructed path.\n{report}"
        );
        assert!(resolved.exists(), "{name} was not built.\n{report}");
    }
}

#[test]
fn drafter_flag_aliases_are_documented_on_both_binaries() {
    let serve_help = help_output("mlxcel", &["serve", "--help"]);
    let server_help = help_output("mlxcel-server", &["--help"]);

    assert_flag_documents_alias(
        "mlxcel serve",
        &serve_help,
        "--draft-model <PATH>",
        "--model-draft",
    );
    assert_flag_documents_alias(
        "mlxcel serve",
        &serve_help,
        "--draft-max <DRAFT_MAX>",
        "--draft",
    );

    assert_flag_documents_alias(
        "mlxcel-server",
        &server_help,
        "--model-draft <PATH>",
        "--draft-model",
    );
    assert_flag_documents_alias(
        "mlxcel-server",
        &server_help,
        "--draft <DRAFT>",
        "--draft-max",
    );
}

/// Guard for the guard: the alias assertion above must reject an entry whose
/// clap alias annotation is missing, even when the alias name still appears
/// in that flag's prose. `--draft-model`'s real help does exactly that, so a
/// whole-help substring search would not catch a dropped `visible_alias`.
#[test]
fn documented_aliases_ignores_alias_names_that_only_appear_in_prose() {
    let without_annotation = "      --draft-model <PATH>\n          \
        Accepts the mlx-lm-style `--draft-model` spelling (primary) and the \
        llama-server-style `--model-draft` spelling.\n\n          [default: none]\n";
    assert!(
        documented_aliases(without_annotation).is_empty(),
        "prose mentioning an alias must not count as a documented alias"
    );

    let singular =
        "      --draft-max <DRAFT_MAX>\n          Max draft tokens\n\n          [alias: --draft]\n";
    assert_eq!(documented_aliases(singular), vec!["--draft".to_string()]);

    let plural = "      --draft-max <DRAFT_MAX>\n          Max draft tokens\n\n          [aliases: --draft, --n-draft]\n";
    assert_eq!(
        documented_aliases(plural),
        vec!["--draft".to_string(), "--n-draft".to_string()]
    );

    // A bracketed span written by hand mid-sentence must not count either.
    // The real help already carries prose brackets on other flags (for
    // example `[llama-server alias for --prefill-chunk-size]`), so an
    // unanchored substring search would let a doc comment satisfy the
    // contract with no `visible_alias` attribute behind it.
    let inline_prose =
        "      --draft-max <DRAFT_MAX>\n          Max draft tokens [alias: --draft] per step\n";
    assert!(
        documented_aliases(inline_prose).is_empty(),
        "a bracketed span inside prose must not count as a clap alias annotation"
    );

    // The annotation still parses when clap wraps it across lines.
    let wrapped = "      --draft-max <DRAFT_MAX>\n          Max draft tokens\n\n          [aliases: --draft,\n           --n-draft]\n";
    assert_eq!(
        documented_aliases(wrapped),
        vec!["--draft".to_string(), "--n-draft".to_string()]
    );
}

/// Mutation check on the real help text: dropping a `visible_alias` from the
/// CLI removes exactly one rendered `[alias: ...]` / `[aliases: ...]`
/// annotation, so cut that annotation out of the live `mlxcel serve --help`
/// and confirm the alias is then undocumented. Answers "would a build that
/// dropped one of the four aliases still fail?" against real output, without
/// rebuilding the CLI with the attribute removed.
///
/// `--draft-model` is the interesting case: its own description names
/// `--model-draft` in prose, so this also shows the prose alone does not
/// satisfy the contract.
#[test]
fn removing_the_rendered_alias_annotation_leaves_the_alias_undocumented() {
    let serve_help = help_output("mlxcel", &["serve", "--help"]);
    let entry = flag_help_entry(&serve_help, "--draft-model <PATH>")
        .expect("mlxcel serve --help has no --draft-model entry");
    assert_eq!(
        documented_aliases(entry),
        vec!["--model-draft".to_string()],
        "precondition: the live help must document exactly the one alias"
    );

    let annotation_start = entry
        .find("[alias")
        .expect("precondition: the entry carries an alias annotation");
    let annotation_end = annotation_start
        + entry[annotation_start..]
            .find(']')
            .expect("alias annotation is unterminated")
        + 1;
    let without_annotation = format!("{}{}", &entry[..annotation_start], &entry[annotation_end..]);

    // Splice the mutated entry back into the full help so the whole
    // resolution path runs, not just the annotation parser.
    let mutated_help = serve_help.replace(entry, &without_annotation);
    let mutated_entry = flag_help_entry(&mutated_help, "--draft-model <PATH>")
        .expect("the flag entry itself must survive the mutation");

    assert!(
        mutated_entry.contains("--model-draft"),
        "precondition: the alias name still appears in the flag's prose"
    );
    assert!(
        documented_aliases(mutated_entry).is_empty(),
        "a dropped visible_alias must leave the alias undocumented, but the \
         assertion still found: {:?}\nEntry was:\n{mutated_entry}",
        documented_aliases(mutated_entry)
    );
}

/// The entry slicer must not spill into the neighbouring flag: an alias
/// annotation on the *next* flag must never satisfy the assertion for this
/// one.
#[test]
fn flag_help_entry_stops_at_the_next_flag() {
    let help = "Options:\n      \
        --draft-model <PATH>\n          Path to drafter checkpoint\n\n      \
        --draft-max <DRAFT_MAX>\n          Max draft tokens\n\n          [alias: --draft]\n\n\
        Other Options:\n      --unrelated\n";

    let entry = flag_help_entry(help, "--draft-model <PATH>").expect("entry not found");
    assert!(entry.contains("Path to drafter checkpoint"));
    assert!(
        !entry.contains("--draft-max"),
        "entry leaked into the next flag:\n{entry}"
    );
    assert!(documented_aliases(entry).is_empty());

    let entry = flag_help_entry(help, "--draft-max <DRAFT_MAX>").expect("entry not found");
    assert_eq!(documented_aliases(entry), vec!["--draft".to_string()]);
    assert!(
        !entry.contains("Other Options:"),
        "entry leaked into the next section:\n{entry}"
    );

    // A signature that is only a prefix of a real entry must not resolve to
    // it. `--draft` is a prefix of `--draft-max <DRAFT_MAX>`, and in the live
    // help `--metrics` is a prefix of the separate `--metrics-port <PORT>`.
    assert!(
        flag_help_entry(help, "--draft").is_none(),
        "a prefix of another flag's signature must not match its entry"
    );
}

/// A hyphen-bulleted description line must not be mistaken for a flag entry.
/// The short-form branch of the matcher accepts `-s, --flag <VALUE>`, and
/// without the two-character constraint a bullet such as `- foo, --draft-max
/// <DRAFT_MAX> ...` would anchor the entry on prose and pick up whatever
/// followed it.
#[test]
fn flag_help_entry_ignores_hyphen_bulleted_prose() {
    let help = "Options:\n      \
        --other <VALUE>\n          Describes something:\n          \
        - see also, --draft-max <DRAFT_MAX> for the token budget\n\n      \
        --draft-max <DRAFT_MAX>\n          Max draft tokens\n\n          [alias: --draft]\n";

    let entry = flag_help_entry(help, "--draft-max <DRAFT_MAX>").expect("entry not found");
    assert!(
        entry.starts_with("      --draft-max <DRAFT_MAX>"),
        "matcher anchored on a prose bullet instead of the flag entry:\n{entry}"
    );
    assert_eq!(documented_aliases(entry), vec!["--draft".to_string()]);
}
