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

//! Unit tests for the llama.cpp short-option argv pre-pass (issue #1434).

use super::*;

/// A stand-in command carrying the shapes the pass has to reason about: a
/// value-taking long with a short, a value-taking long without one, and a
/// value-less flag. Using a purpose-built command rather than the real server
/// surface keeps these tests independent of unrelated flag churn; the real
/// surfaces are covered end to end by `tests/llama_model_source_cli.rs`.
fn sample_command() -> clap::Command {
    clap::Command::new("sample")
        .arg(clap::Arg::new("model").short('m').long("model").num_args(1))
        .arg(
            clap::Arg::new("chat-template")
                .long("chat-template")
                .num_args(1),
        )
        .arg(
            clap::Arg::new("offline")
                .long("offline")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(clap::Arg::new("hf-repo").long("hf-repo").num_args(1))
        .arg(clap::Arg::new("hf-token").long("hf-token").num_args(1))
}

fn expand(argv: &[&str]) -> Vec<String> {
    let mut cmd = sample_command();
    let args: Vec<OsString> = argv.iter().map(OsString::from).collect();
    expand_llama_short_options(&mut cmd, args, 1)
        .into_iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect()
}

#[test]
fn every_table_entry_is_rewritten_to_its_long_spelling() {
    for (short, long) in SHORT_ALIASES {
        let out = expand(&["sample", short, "value"]);
        assert_eq!(
            out,
            vec!["sample".to_owned(), (*long).to_owned(), "value".to_owned()],
            "{short} must become {long}"
        );
    }
}

#[test]
fn the_table_is_sorted_and_free_of_duplicate_tokens() {
    let tokens: Vec<&str> = SHORT_ALIASES.iter().map(|(short, _)| *short).collect();
    let mut sorted = tokens.clone();
    sorted.sort_unstable();
    assert_eq!(tokens, sorted, "SHORT_ALIASES must stay sorted by token");
    let mut deduped = sorted.clone();
    deduped.dedup();
    assert_eq!(sorted, deduped, "a token may map to only one long spelling");
}

#[test]
fn the_inline_equals_form_is_rewritten_too() {
    assert_eq!(
        expand(&["sample", "-hf=owner/name"]),
        vec!["sample".to_owned(), "--hf-repo=owner/name".to_owned()]
    );
}

#[test]
fn the_program_name_is_never_rewritten() {
    // A binary genuinely named `-hf` is absurd, but the pass must not depend
    // on that: index 0 is out of range by construction.
    assert_eq!(expand(&["-hf"]), vec!["-hf".to_owned()]);
}

#[test]
fn a_value_that_looks_like_a_short_option_is_left_alone() {
    // `--chat-template` consumes the next entry, so that entry is a value and
    // must survive verbatim even when it spells one of the table's tokens.
    assert_eq!(
        expand(&["sample", "--chat-template", "-hf", "--offline"]),
        vec![
            "sample".to_owned(),
            "--chat-template".to_owned(),
            "-hf".to_owned(),
            "--offline".to_owned()
        ]
    );
    // Same through a short spelling of a value-taking option.
    assert_eq!(
        expand(&["sample", "-m", "-mu"]),
        vec!["sample".to_owned(), "-m".to_owned(), "-mu".to_owned()]
    );
}

#[test]
fn the_value_of_a_rewritten_option_is_never_itself_rewritten() {
    assert_eq!(
        expand(&["sample", "-hf", "-hft"]),
        vec![
            "sample".to_owned(),
            "--hf-repo".to_owned(),
            "-hft".to_owned()
        ]
    );
}

#[test]
fn nothing_after_a_double_dash_terminator_is_rewritten() {
    assert_eq!(
        expand(&["sample", "--", "-hf", "owner/name"]),
        vec![
            "sample".to_owned(),
            "--".to_owned(),
            "-hf".to_owned(),
            "owner/name".to_owned()
        ]
    );
}

#[test]
fn a_value_less_flag_does_not_swallow_the_next_token() {
    assert_eq!(
        expand(&["sample", "--offline", "-hf", "owner/name"]),
        vec![
            "sample".to_owned(),
            "--offline".to_owned(),
            "--hf-repo".to_owned(),
            "owner/name".to_owned()
        ]
    );
}

#[test]
fn unknown_short_clusters_are_left_for_clap_to_reject() {
    assert_eq!(
        expand(&["sample", "-zz", "x"]),
        vec!["sample".to_owned(), "-zz".to_owned(), "x".to_owned()]
    );
}

#[test]
fn a_start_offset_protects_a_subcommand_token() {
    let mut cmd = sample_command();
    let args: Vec<OsString> = ["mlxcel", "serve", "-hf", "owner/name"]
        .iter()
        .map(OsString::from)
        .collect();
    let out: Vec<String> = expand_llama_short_options(&mut cmd, args, 2)
        .into_iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        out,
        vec![
            "mlxcel".to_owned(),
            "serve".to_owned(),
            "--hf-repo".to_owned(),
            "owner/name".to_owned()
        ]
    );
}

#[test]
fn a_non_utf8_argument_passes_through_unchanged() {
    use std::os::unix::ffi::OsStringExt;
    let weird = OsString::from_vec(vec![0x2d, 0x6d, 0xff]);
    let args = vec![
        OsString::from("sample"),
        weird.clone(),
        OsString::from("-hf"),
        OsString::from("owner/name"),
    ];
    let mut cmd = sample_command();
    let out = expand_llama_short_options(&mut cmd, args, 1);
    assert_eq!(out[1], weird, "a non-UTF-8 entry must survive verbatim");
    // It is not a known value-taking token, so the following `-hf` is still
    // in an option position and is rewritten.
    assert_eq!(out[2], OsString::from("--hf-repo"));
}
