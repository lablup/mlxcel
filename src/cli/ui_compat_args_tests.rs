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

//! Unit tests for the b10621 Web UI / tools / MCP / agent group (#1435).

use clap::Parser;

use super::*;

#[derive(Parser)]
struct Probe {
    #[command(flatten)]
    ui: UiCompatArgs,
}

/// Parse an argv with the process environment held still: several fields
/// carry `LLAMA_ARG_*` clap bindings, so a concurrent env mutation in
/// another test would decide what this parse sees.
fn parse(argv: &[&str]) -> UiCompatArgs {
    let _env_guard = crate::test_support::env_lock::env_lock();
    let mut full = vec!["probe"];
    full.extend_from_slice(argv);
    Probe::try_parse_from(full)
        .unwrap_or_else(|e| panic!("{argv:?} must parse: {e}"))
        .ui
}

#[test]
fn an_empty_group_is_inert() {
    let args = parse(&[]);
    args.ensure_inert().expect("empty group must be inert");
}

#[test]
fn every_manifest_spelling_parses() {
    // Value-less forms.
    for flag in [
        "--ui",
        "--webui",
        "--no-ui",
        "--no-webui",
        "--ui-mcp-proxy",
        "--webui-mcp-proxy",
        "--no-ui-mcp-proxy",
        "--no-webui-mcp-proxy",
        "--agent",
        "--no-agent",
    ] {
        let _ = parse(&[flag]);
    }
    // Value-taking forms.
    for (flag, value) in [
        ("--ui-config", "{}"),
        ("--webui-config", "{}"),
        ("--ui-config-file", "ui.json"),
        ("--webui-config-file", "ui.json"),
        ("--path", "/srv/ui"),
        ("--tools", "read_file,grep_search"),
        ("--tools-runtime", "docker:alpine"),
        ("--mcp-servers-config", "mcp.json"),
        ("--mcp-servers-json", "{}"),
    ] {
        let _ = parse(&[flag, value]);
    }
}

#[test]
fn the_inert_forms_are_accepted_as_no_ops() {
    for argv in [
        &["--no-ui"][..],
        &["--no-webui"][..],
        &["--no-agent"][..],
        &["--no-ui-mcp-proxy"][..],
        &["--no-webui-mcp-proxy"][..],
        &["--no-ui", "--no-agent"][..],
    ] {
        parse(argv)
            .ensure_inert()
            .unwrap_or_else(|e| panic!("{argv:?} must be inert: {e}"));
    }
}

#[test]
fn every_enabling_form_is_rejected_with_its_flag_named() {
    for (argv, named) in [
        (&["--ui"][..], "--ui"),
        (&["--webui"][..], "--ui"),
        (&["--ui-config", "{}"][..], "--ui-config"),
        (&["--ui-config-file", "f"][..], "--ui-config-file"),
        (&["--path", "/srv/ui"][..], "--path"),
        (&["--tools", "read_file"][..], "--tools"),
        (&["--tools-runtime", "docker:x"][..], "--tools-runtime"),
        (&["--mcp-servers-config", "f"][..], "--mcp-servers-config"),
        (&["--mcp-servers-json", "{}"][..], "--mcp-servers-json"),
        (&["--ui-mcp-proxy"][..], "--ui-mcp-proxy"),
        (&["--webui-mcp-proxy"][..], "--ui-mcp-proxy"),
        (&["--agent"][..], "--agent"),
    ] {
        let err = parse(argv)
            .ensure_inert()
            .expect_err("an enabling form must be refused");
        assert!(
            err.contains(named),
            "{argv:?}: diagnostic must name {named}, got: {err}"
        );
        assert!(
            err.contains("not supported"),
            "{argv:?}: diagnostic must say the surface is unsupported: {err}"
        );
    }
}

#[test]
fn bool_pair_env_bindings_follow_b10621_rules() {
    let _guard = crate::test_support::env_lock::env_lock();
    // Truthy enables the positive form, which the startup check refuses.
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("LLAMA_ARG_UI", "on");
    }
    let mut args = UiCompatArgs::default();
    args.apply_env_bindings().expect("binding resolves");
    assert!(args.ui);
    assert!(args.ensure_inert().is_err());

    // Falsy selects the inert negative form.
    // SAFETY: as above.
    unsafe {
        std::env::set_var("LLAMA_ARG_UI", "off");
    }
    let mut args = UiCompatArgs::default();
    args.apply_env_bindings().expect("binding resolves");
    assert!(args.no_ui && !args.ui);
    args.ensure_inert().expect("falsy env form is inert");

    // The LLAMA_ARG_NO_* alias wins and means false.
    // SAFETY: as above.
    unsafe {
        std::env::set_var("LLAMA_ARG_UI", "on");
        std::env::set_var("LLAMA_ARG_NO_UI", "1");
    }
    let mut args = UiCompatArgs::default();
    args.apply_env_bindings().expect("binding resolves");
    assert!(args.no_ui && !args.ui);

    // A value parse_bool_value would throw on fails startup, naming the
    // variable.
    // SAFETY: as above.
    unsafe {
        std::env::remove_var("LLAMA_ARG_NO_UI");
        std::env::set_var("LLAMA_ARG_UI", "sometimes");
    }
    let mut args = UiCompatArgs::default();
    let (var, raw) = args.apply_env_bindings().expect_err("must throw");
    assert_eq!(var, "LLAMA_ARG_UI");
    assert_eq!(raw, "sometimes");

    // An explicit CLI flag outranks the environment.
    // SAFETY: as above.
    unsafe {
        std::env::remove_var("LLAMA_ARG_UI");
        std::env::set_var("LLAMA_ARG_AGENT", "on");
    }
    let mut args = UiCompatArgs {
        no_agent: true,
        ..Default::default()
    };
    args.apply_env_bindings().expect("binding resolves");
    assert!(args.no_agent && !args.agent);

    // SAFETY: as above.
    unsafe {
        std::env::remove_var("LLAMA_ARG_UI");
        std::env::remove_var("LLAMA_ARG_AGENT");
    }
}

#[test]
fn single_dash_agent_spellings_rewrite_to_the_long_forms() {
    use std::ffi::OsString;
    let mut cmd = clap::Command::new("probe")
        .arg(
            clap::Arg::new("agent")
                .long("agent")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            clap::Arg::new("no-agent")
                .long("no-agent")
                .action(clap::ArgAction::SetTrue),
        );
    let rewrite = |argv: &[&str], cmd: &mut clap::Command| -> Vec<String> {
        crate::cli::llama_short_flags::expand_llama_short_options(
            cmd,
            argv.iter().map(OsString::from).collect(),
            1,
        )
        .into_iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect()
    };
    assert_eq!(rewrite(&["probe", "-ag"], &mut cmd), ["probe", "--agent"]);
    assert_eq!(
        rewrite(&["probe", "-no-ag"], &mut cmd),
        ["probe", "--no-agent"]
    );
}
