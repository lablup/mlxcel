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

//! Unit tests for the b10621 WebUI flag and adjacent tools / MCP / agent group (#1435, #1838).

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
fn supported_webui_and_negative_forms_do_not_fail_the_compat_checker() {
    for argv in [
        &["--ui"][..],
        &["--webui"][..],
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
fn every_unsupported_enabling_form_is_rejected_with_its_flag_named() {
    for (argv, named) in [
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
            .expect_err("an unsupported enabling form must be refused");
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
fn webui_enable_disable_resolves_with_last_flag_winning() {
    assert!(parse(&["--ui"]).webui_enabled().expect("supported"));
    assert!(parse(&["--webui"]).webui_enabled().expect("supported"));
    assert!(!parse(&["--no-ui"]).webui_enabled().expect("supported"));
    assert!(
        !parse(&["--ui", "--no-webui"])
            .webui_enabled()
            .expect("supported")
    );
    assert!(
        parse(&["--no-ui", "--webui"])
            .webui_enabled()
            .expect("supported")
    );
}

/// #1900: `--webui` / `--no-webui` are the one visible pair of the group,
/// under their own help heading, and no other spelling of the group (the
/// b10621 `--ui` / `--no-ui` aliases and every refused surface) is listed.
#[test]
fn help_lists_only_the_webui_pair_under_its_heading() {
    use clap::CommandFactory;

    let command = Probe::command();
    let mut visible: Vec<&str> = command
        .get_arguments()
        .filter(|arg| !arg.is_hide_set())
        .filter_map(|arg| arg.get_long())
        .collect();
    visible.sort_unstable();
    assert_eq!(
        visible,
        [
            "mcp-max-output-bytes",
            "mcp-max-tool-rounds",
            "mcp-servers-config",
            "mcp-servers-json",
            "mcp-tool-timeout",
            "mcp-tools",
            "no-webui",
            "webui"
        ]
    );
    for arg in command.get_arguments().filter(|arg| !arg.is_hide_set()) {
        // The MCP tool loop flags (#1457) have their own section.
        let heading = if arg.get_long().is_some_and(|long| long.starts_with("mcp-")) {
            MCP_HELP_HEADING
        } else {
            WEBUI_HELP_HEADING
        };
        assert_eq!(
            arg.get_help_heading(),
            Some(heading),
            "{:?} must sit under the {heading} heading",
            arg.get_long()
        );
        assert!(
            arg.get_visible_aliases().unwrap_or_default().is_empty(),
            "{:?} must not list its b10621 alias",
            arg.get_long()
        );
    }

    let help = Probe::command().render_help().to_string();
    assert!(
        help.contains(&format!("\n{WEBUI_HELP_HEADING}:\n")),
        "rendered help must carry the {WEBUI_HELP_HEADING:?} section:\n{help}"
    );
    for hidden in [
        "--ui ",
        "--no-ui ",
        "--ui-config",
        "--path",
        "--tools",
        "--ui-mcp-proxy",
        "--agent",
    ] {
        assert!(
            !help.contains(hidden),
            "rendered help lists {hidden}:\n{help}"
        );
    }
}

#[test]
fn bool_pair_env_bindings_follow_b10621_rules() {
    let _guard = crate::test_support::env_lock::env_lock();
    // Truthy enables the positive form.
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("LLAMA_ARG_UI", "on");
    }
    let mut args = UiCompatArgs::default();
    args.apply_env_bindings().expect("binding resolves");
    assert!(args.ui);
    assert!(args.webui_enabled().expect("supported"));

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

/// #1457: the MCP server set is refused unless `--mcp-tools` is beside it,
/// whether it came from the command line or from its `LLAMA_ARG_*` variable.
#[test]
fn mcp_server_set_needs_the_mcp_tools_opt_in() {
    for argv in [
        &["--mcp-servers-config", "mcp.json"][..],
        &["--mcp-servers-json", "{}"][..],
    ] {
        let err = parse(argv)
            .ensure_supported()
            .expect_err("a server set alone must be refused");
        assert!(err.contains(argv[0]), "{err}");
        assert!(err.contains("without --mcp-tools"), "{err}");
        let mut with_opt_in = vec!["--mcp-tools"];
        with_opt_in.extend_from_slice(argv);
        parse(&with_opt_in)
            .ensure_supported()
            .expect("the opt-in accepts the server set");
    }

    let _guard = crate::test_support::env_lock::env_lock();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("LLAMA_ARG_MCP_SERVERS_JSON", "{}");
    }
    let parsed = Probe::try_parse_from(["probe"]).map(|probe| probe.ui);
    // SAFETY: as above.
    unsafe {
        std::env::remove_var("LLAMA_ARG_MCP_SERVERS_JSON");
    }
    let err = parsed
        .expect("the variable binds through clap")
        .ensure_supported()
        .expect_err("an inherited variable must not enable the tool loop");
    assert!(err.contains("--mcp-tools"), "{err}");
}

#[test]
fn mcp_bounds_default_to_the_documented_values() {
    let args = parse(&["--mcp-tools"]);
    assert!(args.mcp_tools);
    assert_eq!(args.mcp_max_tool_rounds, 8);
    assert_eq!(args.mcp_tool_timeout, 30);
    assert_eq!(args.mcp_max_output_bytes, 65536);
    let args = parse(&[
        "--mcp-max-tool-rounds",
        "3",
        "--mcp-tool-timeout",
        "5",
        "--mcp-max-output-bytes",
        "1024",
    ]);
    assert!(!args.mcp_tools);
    assert_eq!(
        (
            args.mcp_max_tool_rounds,
            args.mcp_tool_timeout,
            args.mcp_max_output_bytes
        ),
        (3, 5, 1024)
    );
}
