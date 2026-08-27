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

//! End-to-end behavior of the b10621 chat-template, reasoning, and
//! output-parsing options on both server binaries (issue #1447).
//!
//! `src/cli/chat_compat_args_tests.rs` covers the resolution and
//! `src/server/reasoning_format_tests.rs` the placement. This file covers what
//! only the built binaries can answer: that every upstream spelling parses on
//! both, that an unsupported value stops startup with its own diagnostic, and
//! that none of it reaches the operator-facing `--help`.
//!
//! Every invocation fails (or prints help) before a weight is read, so the
//! file needs no checkpoint and no network.

use std::process::{Command, Output};

mod common;
use common::resolve_repo_binary;

const CLEARED: [&str; 12] = [
    "LLAMA_ARG_MODEL",
    "LLAMA_ARG_JINJA",
    "LLAMA_ARG_NO_JINJA",
    "LLAMA_ARG_REASONING",
    "LLAMA_ARG_THINK",
    "LLAMA_ARG_THINK_BUDGET_MESSAGE",
    "LLAMA_ARG_REASONING_EFFORT",
    "LLAMA_ARG_REASONING_PRESERVE",
    "LLAMA_ARG_NO_REASONING_PRESERVE",
    "LLAMA_ARG_SKIP_CHAT_PARSING",
    "LLAMA_ARG_PREFILL_ASSISTANT",
    "LLAMA_ARG_CHAT_TEMPLATE",
];

fn run(bin: &str, args: &[&str], env: &[(&str, &str)]) -> Output {
    let (path, resolution) = resolve_repo_binary(bin);
    let mut cmd = Command::new(&path);
    cmd.args(args);
    for key in CLEARED {
        cmd.env_remove(key);
    }
    for (key, value) in env {
        cmd.env(key, value);
    }
    cmd.output()
        .unwrap_or_else(|e| panic!("failed to spawn {bin} from {path:?}: {e}\n{resolution}"))
}

const ENTRY_POINTS: [(&str, &[&str]); 2] = [("mlxcel", &["serve"]), ("mlxcel-server", &[])];

fn expect_failure(entry: (&str, &[&str]), extra: &[&str]) -> String {
    let (bin, lead) = entry;
    let mut args: Vec<&str> = lead.to_vec();
    args.extend_from_slice(extra);
    let out = run(bin, &args, &[]);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "{bin} {args:?} unexpectedly succeeded; output: {text}"
    );
    text
}

// ── every upstream spelling parses ──────────────────────────────────────────

/// Every b10621 chat-template / reasoning spelling, with a value where the
/// option takes one.
const UPSTREAM_INVOCATIONS: &[&[&str]] = &[
    &["--jinja"],
    &["--no-jinja"],
    &["--reasoning", "on"],
    &["--reasoning", "off"],
    &["--reasoning", "auto"],
    &["--reasoning-format", "none"],
    &["--reasoning-format", "deepseek"],
    &["--reasoning-format", "deepseek-legacy"],
    &["--reasoning-format", "auto"],
    &["--reasoning-effort", "high"],
    &["--reasoning-effort", "default"],
    &["--reasoning-budget", "512"],
    &["--reasoning-budget-message", "(thinking budget spent)"],
    &["--reasoning-preserve"],
    &["--no-reasoning-preserve"],
    &["--skip-chat-parsing"],
    &["--no-skip-chat-parsing"],
    &["--prefill-assistant"],
    &["--no-prefill-assistant"],
    &["--escape"],
    &["--no-escape"],
    &["--special"],
    &["--chat-template-kwargs", "{\"enable_thinking\":true}"],
];

#[test]
fn every_b10621_chat_template_spelling_reaches_mlxcels_own_classification() {
    // Before #1447 thirteen of these were unknown arguments, so a
    // llama-server command line asking for a reasoning format died on clap
    // rather than on anything explaining the boundary.
    for invocation in UPSTREAM_INVOCATIONS {
        for entry in ENTRY_POINTS {
            let text = expect_failure(entry, invocation);
            assert!(
                !text.contains("unexpected argument"),
                "{} {invocation:?}: must not reach clap as an unknown token: {text}",
                entry.0
            );
        }
    }
}

// ── inert values do not stop startup for their own sake ─────────────────────

#[test]
fn inert_values_fail_only_for_the_missing_model() {
    for invocation in [
        &["--jinja"][..],
        &["--reasoning", "auto"][..],
        &["--reasoning-format", "auto"][..],
        &["--reasoning-format", "deepseek"][..],
        &["--reasoning-effort", "default"][..],
        &["--no-skip-chat-parsing"][..],
        &["--prefill-assistant"][..],
        &["--no-prefill-assistant"][..],
        &["--escape"][..],
        &["--no-escape"][..],
    ] {
        for entry in ENTRY_POINTS {
            let text = expect_failure(entry, invocation);
            assert!(
                text.contains("--model/-m is required"),
                "{} {invocation:?} is inert, so the only complaint must be the missing model: {text}",
                entry.0
            );
        }
    }
}

// ── unsupported values stop startup with their own diagnostic ───────────────

#[test]
fn unsupported_values_are_rejected_with_the_option_and_the_accepted_set() {
    for (invocation, marker) in [
        (&["--no-jinja"][..], "Jinja"),
        (&["--special"][..], "special tokens"),
        (&["--reasoning", "maybe"][..], "on, off, auto"),
        (
            &["--reasoning-format", "legacy"][..],
            "none, deepseek, deepseek-legacy, auto",
        ),
    ] {
        for entry in ENTRY_POINTS {
            let text = expect_failure(entry, invocation);
            assert!(
                text.contains(marker),
                "{} {invocation:?}: diagnostic must contain {marker:?}: {text}",
                entry.0
            );
        }
    }
}

#[test]
fn a_builtin_chat_template_name_is_refused_rather_than_taken_literally() {
    // b10621's `--chat-template` accepts either template text or one of its
    // built-in identifiers. mlxcel has no built-in library, so a bare name
    // would become the template itself and every prompt would render to the
    // literal string.
    let tmp = tempfile::tempdir().expect("tempdir");
    let model = tmp.path().join("checkpoint");
    std::fs::create_dir_all(&model).expect("model dir");
    std::fs::write(model.join("config.json"), br#"{"num_hidden_layers": 4}"#).expect("config");
    let model = model.to_str().expect("utf-8");

    for name in ["chatml", "llama3", "deepseek3", "mistral-v7"] {
        for entry in ENTRY_POINTS {
            let text = expect_failure(entry, &["-m", model, "--chat-template", name]);
            assert!(
                text.contains(name),
                "{} --chat-template {name}: must quote the name: {text}",
                entry.0
            );
            assert!(
                text.contains("tokenizer_config.json"),
                "{} --chat-template {name}: must say where mlxcel gets its template: {text}",
                entry.0
            );
            assert!(
                text.contains("--chat-template-file"),
                "{} --chat-template {name}: must offer the way to supply one: {text}",
                entry.0
            );
        }
    }
}

#[test]
fn actual_template_text_is_still_accepted() {
    // The guard must not catch a real template that happens to mention a
    // built-in name.
    let tmp = tempfile::tempdir().expect("tempdir");
    let model = tmp.path().join("checkpoint");
    std::fs::create_dir_all(&model).expect("model dir");
    std::fs::write(model.join("config.json"), br#"{"num_hidden_layers": 4}"#).expect("config");
    let model = model.to_str().expect("utf-8");

    for entry in ENTRY_POINTS {
        let text = expect_failure(
            entry,
            &[
                "-m",
                model,
                "--chat-template",
                "{% for m in messages %}chatml{{ m.content }}{% endfor %}",
            ],
        );
        assert!(
            !text.contains("built-in chat templates"),
            "{}: template text must not be mistaken for a built-in name: {text}",
            entry.0
        );
    }
}

// ── environment bindings ────────────────────────────────────────────────────

#[test]
fn the_reasoning_format_environment_variable_is_llama_arg_think() {
    for (bin, lead) in ENTRY_POINTS {
        let mut args: Vec<&str> = lead.to_vec();
        args.push("--offline");
        let out = run(bin, &args, &[("LLAMA_ARG_THINK", "legacy")]);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            text.contains("none, deepseek, deepseek-legacy, auto"),
            "{bin}: LLAMA_ARG_THINK must reach --reasoning-format: {text}"
        );
    }
}

#[test]
fn a_bool_pair_environment_value_reaches_the_resolution() {
    for (bin, lead) in ENTRY_POINTS {
        let mut args: Vec<&str> = lead.to_vec();
        args.push("--offline");
        // `LLAMA_ARG_JINJA=0` is the negative half, which mlxcel cannot do.
        let out = run(bin, &args, &[("LLAMA_ARG_JINJA", "0")]);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            text.contains("--no-jinja"),
            "{bin}: LLAMA_ARG_JINJA=0 must reach the --no-jinja rejection: {text}"
        );
    }
}

#[test]
fn an_unparseable_bool_pair_environment_value_stops_startup_like_upstream() {
    for (bin, lead) in ENTRY_POINTS {
        let mut args: Vec<&str> = lead.to_vec();
        args.push("--offline");
        let out = run(bin, &args, &[("LLAMA_ARG_SKIP_CHAT_PARSING", "perhaps")]);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!out.status.success(), "{bin}: expected failure: {text}");
        assert!(
            text.contains("LLAMA_ARG_SKIP_CHAT_PARSING") && text.contains("perhaps"),
            "{bin}: must name the variable and its value: {text}"
        );
    }
}

// ── the operator surface stays mlxcel's ─────────────────────────────────────

#[test]
fn no_compatibility_only_chat_option_appears_in_help() {
    for (bin, lead) in ENTRY_POINTS {
        let mut args: Vec<&str> = lead.to_vec();
        args.push("--help");
        let out = run(bin, &args, &[]);
        let help = String::from_utf8_lossy(&out.stdout);
        for hidden in [
            "--jinja",
            "--no-jinja",
            "--reasoning ",
            "--reasoning-format",
            "--reasoning-effort",
            "--reasoning-preserve",
            "--skip-chat-parsing",
            "--prefill-assistant",
            "--escape",
            "--special",
        ] {
            assert!(
                !help.contains(hidden),
                "{bin} --help advertises {hidden}, which is a compatibility surface"
            );
        }
        for visible in [
            "--chat-template",
            "--chat-template-file",
            "--reasoning-budget",
        ] {
            assert!(
                help.contains(visible),
                "{bin} --help must still document the mlxcel-native {visible}"
            );
        }
    }
}
