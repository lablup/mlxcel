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

use std::process::Command;

mod common;
use common::repo_binary_path;

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
    let path = repo_binary_path(bin_name);
    let mut cmd = Command::new(&path);
    cmd.args(args);
    for key in ENV_TO_CLEAR_FOR_HELP {
        cmd.env_remove(key);
    }
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {} from {:?}: {e}", bin_name, path));
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
    let path = repo_binary_path(bin_name);
    let mut cmd = Command::new(&path);
    cmd.args(args);
    for key in SPECULATIVE_ENV_TO_CLEAR_FOR_HELP {
        cmd.env_remove(key);
    }
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {} from {:?}: {e}", bin_name, path));
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
#[test]
fn drafter_flag_aliases_are_documented_on_both_binaries() {
    let serve_help = help_output("mlxcel", &["serve", "--help"]);
    let server_help = help_output("mlxcel-server", &["--help"]);

    assert!(
        serve_help.contains("--draft-model <PATH>")
            && serve_help.contains("[aliases: --model-draft]"),
        "mlxcel serve --help must document --draft-model with a --model-draft alias.\nHelp was:\n{serve_help}"
    );
    assert!(
        serve_help.contains("--draft-max <DRAFT_MAX>") && serve_help.contains("[aliases: --draft]"),
        "mlxcel serve --help must document --draft-max with a --draft alias.\nHelp was:\n{serve_help}"
    );

    assert!(
        server_help.contains("--model-draft <PATH>")
            && server_help.contains("[aliases: --draft-model]"),
        "mlxcel-server --help must document --model-draft with a --draft-model alias.\nHelp was:\n{server_help}"
    );
    assert!(
        server_help.contains("--draft <DRAFT>") && server_help.contains("[aliases: --draft-max]"),
        "mlxcel-server --help must document --draft with a --draft-max alias.\nHelp was:\n{server_help}"
    );
}
