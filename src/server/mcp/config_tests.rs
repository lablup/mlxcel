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

use super::*;

fn args(tools: bool, config: Option<&str>, json: Option<&str>) -> UiCompatArgs {
    UiCompatArgs {
        mcp_tools: tools,
        mcp_servers_config: config.map(str::to_string),
        mcp_servers_json: json.map(str::to_string),
        mcp_max_tool_rounds: DEFAULT_MAX_TOOL_ROUNDS,
        mcp_tool_timeout: DEFAULT_TOOL_TIMEOUT_SECS,
        mcp_max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
        ..UiCompatArgs::default()
    }
}

const ONE_SERVER: &str = r#"{"mcpServers": {"files": {"command": "srv", "args": ["--root", "/tmp"], "env": {"K": "v"}}}}"#;

#[test]
fn feature_is_off_without_any_mcp_flag() {
    assert_eq!(resolve_mcp_startup(&args(false, None, None)), Ok(None));
}

#[test]
fn server_set_without_opt_in_fails_and_names_the_opt_in() {
    for (config, json, flag) in [
        (Some("mcp.json"), None, "--mcp-servers-config"),
        (None, Some(ONE_SERVER), "--mcp-servers-json"),
    ] {
        let error = resolve_mcp_startup(&args(false, config, json)).expect_err("must be refused");
        assert!(error.contains(flag), "{error}");
        assert!(error.contains("--mcp-tools"), "{error}");
    }
}

#[test]
fn opt_in_without_a_server_set_fails() {
    let error = resolve_mcp_startup(&args(true, None, None)).expect_err("must be refused");
    assert!(error.contains("--mcp-servers-config"), "{error}");
}

#[test]
fn both_server_set_flags_are_refused() {
    let error = resolve_mcp_startup(&args(true, Some("mcp.json"), Some(ONE_SERVER)))
        .expect_err("must be refused");
    assert!(error.contains("exactly one"), "{error}");
}

#[test]
fn opt_in_with_inline_json_resolves_servers_and_default_limits() {
    let config = resolve_mcp_startup(&args(true, None, Some(ONE_SERVER)))
        .expect("resolves")
        .expect("enabled");
    assert_eq!(config.limits, McpLimits::default());
    assert_eq!(config.servers.len(), 1);
    let server = &config.servers[0];
    assert_eq!(server.name, "files");
    assert_eq!(server.command, "srv");
    assert_eq!(server.args, ["--root", "/tmp"]);
    assert_eq!(server.env.get("K").map(String::as_str), Some("v"));
}

#[test]
fn opt_in_reads_the_config_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("mcp.json");
    std::fs::write(&path, ONE_SERVER).expect("write");
    let config = resolve_mcp_startup(&args(true, path.to_str(), None))
        .expect("resolves")
        .expect("enabled");
    assert_eq!(config.servers[0].name, "files");

    let missing = dir.path().join("absent.json");
    let error = resolve_mcp_startup(&args(true, missing.to_str(), None)).expect_err("missing file");
    assert!(error.contains("cannot read"), "{error}");
}

#[test]
fn zero_bounds_are_refused() {
    let mut zero_rounds = args(true, None, Some(ONE_SERVER));
    zero_rounds.mcp_max_tool_rounds = 0;
    assert!(resolve_mcp_startup(&zero_rounds).is_err());
    let mut zero_timeout = args(true, None, Some(ONE_SERVER));
    zero_timeout.mcp_tool_timeout = 0;
    assert!(resolve_mcp_startup(&zero_timeout).is_err());
    let mut tiny_output = args(true, None, Some(ONE_SERVER));
    tiny_output.mcp_max_output_bytes = 8;
    assert!(resolve_mcp_startup(&tiny_output).is_err());
}

#[test]
fn non_stdio_and_unknown_entries_are_refused_with_the_server_named() {
    for (document, needle) in [
        (
            r#"{"mcpServers": {"remote": {"url": "https://example.com/mcp"}}}"#,
            "'url'",
        ),
        (
            r#"{"mcpServers": {"remote": {"command": "x", "type": "sse"}}}"#,
            "only \"stdio\"",
        ),
        (
            r#"{"mcpServers": {"remote": {"command": "x", "transport": "http"}}}"#,
            "'transport'",
        ),
        (
            r#"{"mcpServers": {"remote": {"command": "x", "cwd": "/"}}}"#,
            "unknown key 'cwd'",
        ),
        (r#"{"mcpServers": {"remote": {"args": []}}}"#, "'command'"),
        (
            r#"{"mcpServers": {"remote": {"command": "  "}}}"#,
            "'command'",
        ),
        (
            r#"{"mcpServers": {"remote": {"command": "x", "args": [1]}}}"#,
            "'args'",
        ),
        (
            r#"{"mcpServers": {"remote": {"command": "x", "env": {"A": 1}}}}"#,
            "'env'",
        ),
        (r#"{"mcpServers": {"remote": "x"}}"#, "JSON object"),
    ] {
        let error = parse_mcp_servers(document).expect_err(document);
        assert!(error.contains("remote"), "{document}: {error}");
        assert!(error.contains(needle), "{document}: {error}");
    }
}

#[test]
fn malformed_documents_are_refused() {
    for document in [
        "not json",
        "[]",
        "{}",
        r#"{"mcpServers": {}}"#,
        r#"{"mcpServers": {"a b": {"command": "x"}}}"#,
        r#"{"mcpServers": {"a": {"command": "x"}}, "servers": {}}"#,
    ] {
        assert!(parse_mcp_servers(document).is_err(), "{document}");
    }
}

#[test]
fn stdio_type_is_accepted() {
    let servers =
        parse_mcp_servers(r#"{"mcpServers": {"a": {"command": "x", "type": "stdio"}}}"#).unwrap();
    assert_eq!(servers[0].command, "x");
    assert!(servers[0].args.is_empty() && servers[0].env.is_empty());
}

#[test]
fn exposure_gate_needs_a_local_listener_or_api_keys() {
    let error = validate_mcp_exposure(ListenerScope::Network, "0.0.0.0", false)
        .expect_err("unauthenticated network listener");
    assert!(error.contains("--mcp-tools"), "{error}");
    assert!(error.contains("--api-key"), "{error}");
    assert!(error.contains("0.0.0.0"), "{error}");
    assert!(validate_mcp_exposure(ListenerScope::Network, "0.0.0.0", true).is_ok());
    assert!(validate_mcp_exposure(ListenerScope::LocalOnly, "127.0.0.1", false).is_ok());
    assert!(validate_mcp_exposure(ListenerScope::LocalOnly, "/tmp/mlxcel.sock", false).is_ok());
}

#[test]
fn message_cap_scales_with_the_output_cap() {
    let limits = McpLimits {
        max_output_bytes: 1000,
        ..McpLimits::default()
    };
    assert_eq!(limits.max_message_bytes(), 4000 + 64 * 1024);
}

#[test]
fn debug_output_hides_env_values() {
    let spec = McpServerSpec {
        name: "files".to_string(),
        command: "srv".to_string(),
        args: vec!["--root".to_string()],
        env: BTreeMap::from([("API_TOKEN".to_string(), "s3cret-value".to_string())]),
    };
    let text = format!("{spec:?}");
    assert!(text.contains("API_TOKEN"), "keys stay visible: {text}");
    assert!(!text.contains("s3cret-value"), "value leaked: {text}");
    let config = resolve_mcp_startup(&args(
        true,
        None,
        Some(r#"{"mcpServers": {"a": {"command": "x", "env": {"K": "hunter2"}}}}"#),
    ))
    .unwrap()
    .unwrap();
    assert!(!format!("{config:?}").contains("hunter2"));
}

#[test]
fn distributed_launches_are_refused() {
    assert_eq!(validate_mcp_single_node(None), Ok(()));
    for flag in ["--tp-size", "--pp-peer", "--node-role"] {
        let message = validate_mcp_single_node(Some(flag)).unwrap_err();
        assert!(message.contains(flag) && message.contains("--mcp-tools"));
        assert!(!message.contains('\n'));
    }
}
