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

//! Unit tests for the `--completion-bash` generator (issue #1448).
//!
//! The generator is exercised against a hand-built `clap::Command` rather
//! than a server binary so the hidden / visible distinction is stated by the
//! test itself. `tests/llama_logging_presets.rs` runs the same generator
//! against the real binaries and pipes the result through `bash -n`.

use super::*;

/// A miniature command carrying one of each interesting argument shape.
fn sample_command() -> clap::Command {
    clap::Command::new("sample")
        .arg(
            clap::Arg::new("model")
                .short('m')
                .long("model")
                .visible_alias("model-path")
                .value_name("PATH_OR_REPO_ID"),
        )
        .arg(
            clap::Arg::new("log_file")
                .long("log-file")
                .value_name("FNAME"),
        )
        .arg(
            clap::Arg::new("port")
                .long("port")
                .value_name("N")
                .default_value("8080"),
        )
        .arg(
            clap::Arg::new("verbose")
                .short('v')
                .long("verbose")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            clap::Arg::new("hidden_unsafe")
                .long("gpu-layers")
                .value_name("N")
                .hide(true),
        )
        .arg(
            clap::Arg::new("hidden_alias_host")
                .long("visible-primary")
                .alias("hidden-twin")
                .visible_alias("visible-twin")
                .action(clap::ArgAction::SetTrue),
        )
}

fn script() -> String {
    bash_completion_script("sample", "sample", &mut sample_command())
}

/// The spellings the script's `opts` list offers, as a set.
///
/// Read out of the emitted script rather than asserted with `contains`: a
/// substring check would report `--model` present merely because
/// `--model-path` is there, which is the opposite of what these tests pin.
fn offered_spellings(script: &str) -> std::collections::BTreeSet<String> {
    let line = script
        .lines()
        .find(|line| line.trim_start().starts_with("opts=\""))
        .expect("the script must carry an opts list");
    let inner = line
        .trim()
        .trim_start_matches("opts=\"")
        .trim_end_matches('"');
    inner.split_whitespace().map(str::to_owned).collect()
}

#[test]
fn the_function_name_is_a_legal_bash_identifier() {
    assert_eq!(function_name("mlxcel-server"), "_mlxcel_server_completions");
    assert_eq!(function_name("mlxcel"), "_mlxcel_completions");
    let name = function_name("weird name.1");
    assert!(
        name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
        "{name} is not a legal identifier"
    );
}

#[test]
fn visible_options_and_their_visible_aliases_are_offered() {
    let text = script();
    let offered = offered_spellings(&text);
    for spelling in [
        "-m",
        "--model",
        "--model-path",
        "--log-file",
        "--port",
        "-v",
        "--verbose",
        "--visible-primary",
        "--visible-twin",
    ] {
        assert!(
            offered.contains(spelling),
            "{spelling} is missing from the option list {offered:?}"
        );
    }
}

#[test]
fn hidden_arguments_never_reach_the_script() {
    let text = script();
    assert!(
        !text.contains("--gpu-layers"),
        "a hidden compatibility argument leaked into the completion script:\n{text}"
    );
    assert!(!offered_spellings(&text).contains("--gpu-layers"));
}

#[test]
fn hidden_aliases_of_a_visible_argument_never_reach_the_script() {
    // `clap::Arg::get_all_aliases` would return `--hidden-twin`; the
    // generator asks for `get_visible_aliases` precisely so it does not.
    let text = script();
    assert!(
        !text.contains("--hidden-twin"),
        "a hidden alias leaked into the completion script:\n{text}"
    );
    assert!(text.contains("--visible-twin"), "{text}");
}

#[test]
fn path_valued_options_get_file_completion() {
    let text = script();
    let case_arm = text
        .lines()
        .find(|line| line.contains("compgen -f"))
        .expect("a file-completion arm must exist");
    let patterns = text
        .lines()
        .find(|line| line.trim_end().ends_with(')') && line.contains("--model"))
        .expect("the path option pattern line");
    assert!(patterns.contains("--log-file"), "{patterns}");
    assert!(patterns.contains("-m"), "{patterns}");
    assert!(
        !patterns.contains("--port"),
        "a numeric option must not get file completion: {patterns}"
    );
    assert!(case_arm.contains("compgen -d"), "{case_arm}");
}

#[test]
fn value_less_flags_never_get_file_completion() {
    let text = script();
    let patterns = text
        .lines()
        .find(|line| line.trim_end().ends_with(')') && line.contains("--model"))
        .expect("the path option pattern line");
    assert!(!patterns.contains("--verbose"), "{patterns}");
}

#[test]
fn the_script_registers_the_completion_against_the_executable() {
    let text = script();
    assert!(
        text.trim_end()
            .ends_with("complete -F _sample_completions sample"),
        "{text}"
    );
}

#[test]
fn the_header_names_the_invocation_whose_options_were_harvested() {
    let text = bash_completion_script("mlxcel", "mlxcel serve", &mut sample_command());
    assert!(
        text.starts_with("# bash completion for `mlxcel serve`"),
        "{text}"
    );
    assert!(
        text.trim_end()
            .ends_with("complete -F _mlxcel_completions mlxcel"),
        "{text}"
    );
}

#[test]
fn the_output_is_byte_identical_across_runs() {
    // A completion script an operator sources from their profile must not
    // churn, and a test that pins output needs the ordering to be stable.
    assert_eq!(script(), script());
}

#[test]
fn every_brace_and_case_is_balanced() {
    // A cheap structural check for the environments where `bash` is absent;
    // `tests/llama_logging_presets.rs` runs the real `bash -n`.
    let text = script();
    assert_eq!(text.matches("_sample_completions() {").count(), 1, "{text}");
    assert_eq!(text.matches("    case \"$prev\" in").count(), 1, "{text}");
    assert_eq!(text.matches("    esac").count(), 1, "{text}");
    assert_eq!(
        text.matches('{').count(),
        text.matches('}').count(),
        "unbalanced braces:\n{text}"
    );
}
