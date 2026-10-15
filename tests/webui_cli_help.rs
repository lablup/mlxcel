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

//! `--webui` discoverability on the two server entry points (issue #1900).
//!
//! `--webui` / `--no-webui` are mlxcel's own product flags for the bundled
//! WebUI, so both `mlxcel-server --help` and `mlxcel serve --help` must list
//! them. The rest of the llama-server b10621 UI / tools / MCP / proxy / agent
//! group (`src/cli/ui_compat_args.rs`) stays hidden and keeps failing startup
//! with its existing one-line diagnostic. The manifest records the same split
//! in `compat/llama-server/b10621/ui-tools-mcp-gcp.json`, and
//! `tests/llama_compat_manifest.rs` checks those `hidden` claims against the
//! dumped flag surface; this file checks what an operator actually sees.

use std::process::{Command, Output};

mod common;
use common::resolve_repo_binary;

const ENTRY_POINTS: [(&str, &[&str]); 2] = [("mlxcel", &["serve"]), ("mlxcel-server", &[])];

/// Variables that would otherwise switch the group on or off from the
/// developer's shell and change what the binaries report.
const CLEARED: [&str; 18] = [
    "LLAMA_ARG_UI",
    "LLAMA_ARG_NO_UI",
    "LLAMA_ARG_UI_CONFIG",
    "LLAMA_ARG_UI_CONFIG_FILE",
    "LLAMA_ARG_STATIC_PATH",
    "LLAMA_ARG_TOOLS",
    "LLAMA_ARG_TOOLS_RUNTIME",
    "LLAMA_ARG_MCP_SERVERS_CONFIG",
    "LLAMA_ARG_MCP_SERVERS_JSON",
    "LLAMA_ARG_UI_MCP_PROXY",
    "LLAMA_ARG_NO_UI_MCP_PROXY",
    "LLAMA_ARG_AGENT",
    "LLAMA_ARG_NO_AGENT",
    "LLAMA_ARG_MODEL",
    "LLAMA_ARG_MODELS_DIR",
    "LLAMA_ARG_CACHE_TYPE_K",
    "LLAMA_ARG_CACHE_TYPE_V",
    "MLXCEL_MODELS_DIR",
];

/// Long spellings of the group that must never be listed in `--help`.
const HIDDEN_SPELLINGS: [&str; 17] = [
    "--ui",
    "--no-ui",
    "--ui-config",
    "--webui-config",
    "--ui-config-file",
    "--webui-config-file",
    "--path",
    "--tools",
    "--tools-runtime",
    "--mcp-servers-config",
    "--mcp-servers-json",
    "--ui-mcp-proxy",
    "--webui-mcp-proxy",
    "--no-ui-mcp-proxy",
    "--no-webui-mcp-proxy",
    "--agent",
    "--no-agent",
];

/// Each refused enabling form and the flag its diagnostic names. The text is
/// the stable prefix `ensure_supported` emits: `<flag> is not supported: `.
const REFUSED: [(&[&str], &str); 10] = [
    (&["--ui-config", "{}"], "--ui-config is not supported: "),
    (
        &["--ui-config-file", "ui.json"],
        "--ui-config-file is not supported: ",
    ),
    (&["--path", "/srv/ui"], "--path is not supported: "),
    (&["--tools", "read_file"], "--tools is not supported: "),
    (
        &["--tools-runtime", "docker:alpine"],
        "--tools-runtime is not supported: ",
    ),
    (
        &["--mcp-servers-config", "mcp.json"],
        "--mcp-servers-config is not supported: ",
    ),
    (
        &["--mcp-servers-json", "{}"],
        "--mcp-servers-json is not supported: ",
    ),
    (
        &["--ui-mcp-proxy"],
        "--ui-mcp-proxy/--webui-mcp-proxy is not supported: ",
    ),
    (
        &["--webui-mcp-proxy"],
        "--ui-mcp-proxy/--webui-mcp-proxy is not supported: ",
    ),
    (&["--agent"], "--agent is not supported: "),
];

fn run(entry: (&str, &[&str]), extra: &[&str]) -> Output {
    let (bin, lead) = entry;
    let (path, resolution) = resolve_repo_binary(bin);
    let mut cmd = Command::new(&path);
    cmd.args(lead).args(extra);
    for key in CLEARED {
        cmd.env_remove(key);
    }
    cmd.output()
        .unwrap_or_else(|e| panic!("failed to spawn {bin} from {path:?}: {e}\n{resolution}"))
}

/// Every whitespace-, comma- or bracket-delimited `--token` in the help text.
fn listed_long_flags(help: &str) -> Vec<&str> {
    help.split(|c: char| c.is_whitespace() || c == ',' || c == '[' || c == ']')
        .filter(|token| token.starts_with("--") && token.len() > 2)
        .map(|token| token.trim_end_matches(['.', ':', ';']))
        .collect()
}

#[test]
fn webui_pair_is_listed_and_the_rest_of_the_group_stays_hidden() {
    for entry in ENTRY_POINTS {
        let out = run(entry, &["--help"]);
        let label = format!("{} {:?} --help", entry.0, entry.1);
        assert!(
            out.status.success(),
            "{label} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let help = String::from_utf8_lossy(&out.stdout);
        let listed = listed_long_flags(&help);

        let heading = help
            .find("\nWebUI Options:\n")
            .unwrap_or_else(|| panic!("{label} has no `WebUI Options:` section:\n{help}"));
        for flag in ["--webui", "--no-webui"] {
            assert!(listed.contains(&flag), "{label} must list {flag}:\n{help}");
            let entry_line = help
                .lines()
                .find(|line| line.trim_start().starts_with(flag))
                .unwrap_or_else(|| panic!("{label} has no option line for {flag}"));
            let at = help
                .find(entry_line)
                .expect("line comes from the same text");
            assert!(
                at > heading,
                "{label} lists {flag} outside the WebUI Options section"
            );
        }
        for hidden in HIDDEN_SPELLINGS {
            assert!(
                !listed.contains(&hidden),
                "{label} lists the hidden compatibility spelling {hidden}"
            );
        }
    }
}

#[test]
fn hidden_unsupported_forms_still_fail_startup_with_their_diagnostic() {
    for entry in ENTRY_POINTS {
        for (argv, diagnostic) in REFUSED {
            let out = run(entry, argv);
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(
                !out.status.success(),
                "{} {:?} {argv:?} unexpectedly started",
                entry.0,
                entry.1
            );
            assert!(
                stderr.contains(diagnostic),
                "{} {:?} {argv:?} must fail with {diagnostic:?}, stderr:\n{stderr}",
                entry.0,
                entry.1
            );
        }
    }
}
