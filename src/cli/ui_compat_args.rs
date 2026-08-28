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

//! llama-server b10621 Web UI, built-in tools, MCP, CORS-proxy, and agent
//! options (issue #1435, epic #1431).
//!
//! Every surface in this group exists for b10621's embedded browser UI:
//! `--ui` serves the SvelteKit bundle, `--path` replaces it with a
//! directory, `--tools` / `--tools-runtime` expose server-executed tools to
//! that UI through `/tools` (upstream's own developer docs mark the endpoint
//! UI-internal and tell applications not to use it), `--mcp-servers-config` /
//! `--mcp-servers-json` feed the same endpoint from stdio MCP child
//! processes, `--ui-mcp-proxy` opens the generic `/cors-proxy` URL proxy so
//! the browser can reach remote MCP servers, and `--agent` turns the proxy
//! and every built-in tool on at once. mlxcel ships no web UI and executes
//! nothing server-side on a model's behalf, so none of this has an
//! implementation to alias to; a server-side MCP tool loop is tracked
//! separately as a product feature (#1457), outside b10621 compatibility.
//!
//! The classification (issue #1435) is therefore uniform: the flags parse
//! (hidden), the forms that ask for nothing (`--no-ui`, `--no-webui`,
//! `--no-agent`, `--no-ui-mcp-proxy`, `--no-webui-mcp-proxy`) are accepted
//! as inert, and every form that would enable a surface fails startup with a
//! one-line diagnostic naming the supported alternative, before the model
//! load. The `/tools` and `/cors-proxy` routes are mounted as b10621's own
//! disabled-feature stubs: 403 with
//! `{"error":{"message":"this feature is disabled","type":"feature_disabled"}}`,
//! which is exactly what upstream answers when the features are off.
//! Because the CORS proxy is refused rather than implemented, no loopback,
//! link-local, metadata-service, or DNS-rebinding allowlist is needed: the
//! SSRF surface does not exist here by construction.
//!
//! Environment forms: the value-taking options bind their `LLAMA_ARG_*`
//! variable through clap. The three value-less bool pairs (`LLAMA_ARG_UI`,
//! `LLAMA_ARG_AGENT`, `LLAMA_ARG_UI_MCP_PROXY`) resolve at runtime through
//! [`crate::cli::ggml_compat_args::env_bool_pair`], b10621's
//! `parse_bool_value` rules with the `LLAMA_ARG_NO_*` alias meaning false,
//! so `LLAMA_ARG_UI=0` is the inert `--no-ui` and `LLAMA_ARG_UI=on` reaches
//! the same startup refusal `--ui` does.
//!
//! Used by: mlxcel serve, mlxcel-server.
//!
//! Upstream reference:
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>

use clap::Args;

use crate::cli::ggml_compat_args::env_bool_pair;

/// llama-server b10621 Web UI / tools / MCP / agent compatibility surface.
///
/// All hidden: these are compatibility arguments, not mlxcel features, and
/// rendering them in `--help` would imply a web UI that does not exist.
#[derive(Debug, Clone, Default, Args)]
pub struct UiCompatArgs {
    /// b10621 `--ui` / `--webui`: serve the embedded Web UI. mlxcel has
    /// none; rejected at startup.
    #[arg(long = "ui", alias = "webui", hide = true, default_value_t = false)]
    pub ui: bool,

    /// b10621 `--no-ui` / `--no-webui`: disable the Web UI. mlxcel never
    /// serves one, so this asks for what already holds; accepted as inert.
    #[arg(
        long = "no-ui",
        alias = "no-webui",
        hide = true,
        default_value_t = false
    )]
    pub no_ui: bool,

    /// b10621 `--ui-config` / `--webui-config`: inline JSON configuring the
    /// Web UI. Rejected at startup with the UI diagnostic.
    #[arg(
        long = "ui-config",
        alias = "webui-config",
        value_name = "JSON",
        env = "LLAMA_ARG_UI_CONFIG",
        hide = true
    )]
    pub ui_config: Option<String>,

    /// b10621 `--ui-config-file` / `--webui-config-file`: a file configuring
    /// the Web UI. Rejected at startup with the UI diagnostic.
    #[arg(
        long = "ui-config-file",
        alias = "webui-config-file",
        value_name = "PATH",
        env = "LLAMA_ARG_UI_CONFIG_FILE",
        hide = true
    )]
    pub ui_config_file: Option<String>,

    /// b10621 `--path`: serve a static directory at `/` in place of the
    /// embedded UI bundle (only effective there when the UI is on).
    /// Rejected at startup: mlxcel has no static file server to alias to.
    #[arg(
        long = "path",
        value_name = "PATH",
        env = "LLAMA_ARG_STATIC_PATH",
        hide = true
    )]
    pub static_path: Option<String>,

    /// b10621 `--tools`: comma-separated built-in tools executed by the
    /// server for its browser-UI agent. Rejected at startup.
    #[arg(
        long = "tools",
        value_name = "TOOL1,TOOL2,...",
        env = "LLAMA_ARG_TOOLS",
        hide = true
    )]
    pub tools: Option<String>,

    /// b10621 `--tools-runtime`: where those tools execute (`docker:`,
    /// `podman:`, `ssh:`). Rejected at startup.
    #[arg(
        long = "tools-runtime",
        value_name = "OPTION",
        env = "LLAMA_ARG_TOOLS_RUNTIME",
        hide = true
    )]
    pub tools_runtime: Option<String>,

    /// b10621 `--mcp-servers-config`: a file of stdio MCP servers exposed
    /// through `/tools` for the browser UI. Rejected at startup.
    #[arg(
        long = "mcp-servers-config",
        value_name = "PATH",
        env = "LLAMA_ARG_MCP_SERVERS_CONFIG",
        hide = true
    )]
    pub mcp_servers_config: Option<String>,

    /// b10621 `--mcp-servers-json`: the same MCP server set as inline JSON.
    /// Rejected at startup.
    #[arg(
        long = "mcp-servers-json",
        value_name = "JSON",
        env = "LLAMA_ARG_MCP_SERVERS_JSON",
        hide = true
    )]
    pub mcp_servers_json: Option<String>,

    /// b10621 `--ui-mcp-proxy` / `--webui-mcp-proxy`: enable the generic
    /// `/cors-proxy` URL proxy for the browser UI. Rejected at startup: a
    /// pure SSRF surface with no mlxcel counterpart.
    #[arg(
        long = "ui-mcp-proxy",
        alias = "webui-mcp-proxy",
        hide = true,
        default_value_t = false
    )]
    pub ui_mcp_proxy: bool,

    /// b10621 `--no-ui-mcp-proxy` / `--no-webui-mcp-proxy`: the proxy's off
    /// switch, upstream's default; accepted as inert.
    #[arg(
        long = "no-ui-mcp-proxy",
        alias = "no-webui-mcp-proxy",
        hide = true,
        default_value_t = false
    )]
    pub no_ui_mcp_proxy: bool,

    /// b10621 `--agent` (`-ag`): enable the CORS proxy plus every built-in
    /// tool for the browser-UI agent. Rejected at startup.
    #[arg(long = "agent", hide = true, default_value_t = false)]
    pub agent: bool,

    /// b10621 `--no-agent` (`-no-ag`): agent mode's off switch, upstream's
    /// default; accepted as inert.
    #[arg(long = "no-agent", hide = true, default_value_t = false)]
    pub no_agent: bool,
}

impl UiCompatArgs {
    /// Resolve the three value-less bool pairs from their `LLAMA_ARG_*`
    /// variables with b10621's `parse_bool_value` rules. A CLI flag wins
    /// over the environment; an unrecognized value fails startup exactly as
    /// upstream throws.
    pub fn apply_env_bindings(&mut self) -> Result<(), (&'static str, String)> {
        bind_pair("LLAMA_ARG_UI", &mut self.ui, &mut self.no_ui)?;
        bind_pair(
            "LLAMA_ARG_UI_MCP_PROXY",
            &mut self.ui_mcp_proxy,
            &mut self.no_ui_mcp_proxy,
        )?;
        bind_pair("LLAMA_ARG_AGENT", &mut self.agent, &mut self.no_agent)?;
        Ok(())
    }

    /// Refuse every form that would enable a surface mlxcel does not have.
    /// The inert forms (`--no-ui`, `--no-webui`, `--no-agent`,
    /// `--no-ui-mcp-proxy`) are accepted silently: they ask for the state
    /// the server is permanently in.
    pub fn ensure_inert(&self) -> Result<(), String> {
        if self.ui {
            return Err(reject(
                "--ui/--webui",
                "mlxcel ships no web UI; the server exposes the HTTP API only (use the /v1 routes; --no-ui and --no-webui are accepted as no-ops)",
            ));
        }
        if self.ui_config.is_some() {
            return Err(reject(
                "--ui-config",
                "it configures llama-server's web UI and mlxcel ships none",
            ));
        }
        if self.ui_config_file.is_some() {
            return Err(reject(
                "--ui-config-file",
                "it configures llama-server's web UI and mlxcel ships none",
            ));
        }
        if self.static_path.is_some() {
            return Err(reject(
                "--path",
                "it replaces llama-server's web UI bundle and mlxcel ships no web UI or static file server",
            ));
        }
        if self.tools.is_some() {
            return Err(reject(
                "--tools",
                "server-executed tools exist for llama-server's browser-UI agent (its /tools endpoint is UI-internal); mlxcel executes no server-side tools. Client-declared `tools` in chat requests are still parsed for tool calls",
            ));
        }
        if self.tools_runtime.is_some() {
            return Err(reject(
                "--tools-runtime",
                "it selects where llama-server executes its built-in tools, and mlxcel executes none",
            ));
        }
        if self.mcp_servers_config.is_some() {
            return Err(reject(
                "--mcp-servers-config",
                "llama-server spawns MCP processes for its browser UI; mlxcel spawns none (a server-side MCP tool loop is tracked separately in issue #1457)",
            ));
        }
        if self.mcp_servers_json.is_some() {
            return Err(reject(
                "--mcp-servers-json",
                "llama-server spawns MCP processes for its browser UI; mlxcel spawns none (a server-side MCP tool loop is tracked separately in issue #1457)",
            ));
        }
        if self.ui_mcp_proxy {
            return Err(reject(
                "--ui-mcp-proxy/--webui-mcp-proxy",
                "the CORS proxy is a generic URL proxy for llama-server's browser UI, an SSRF surface by construction, and mlxcel does not implement it; /cors-proxy answers b10621's 403 feature_disabled",
            ));
        }
        if self.agent {
            return Err(reject(
                "--agent",
                "agent mode enables llama-server's built-in tools plus its CORS proxy, and mlxcel implements neither (--no-agent is accepted as a no-op)",
            ));
        }
        Ok(())
    }
}

/// Resolve one `--x` / `--no-x` pair from its variable. A CLI flag
/// outranks the environment; `LLAMA_ARG_NO_*` wins inside
/// [`env_bool_pair`], matching upstream's `get_value_from_env`.
fn bind_pair(
    var: &'static str,
    positive: &mut bool,
    negative: &mut bool,
) -> Result<(), (&'static str, String)> {
    if *positive || *negative {
        return Ok(());
    }
    match env_bool_pair(var) {
        None => Ok(()),
        Some(Ok(true)) => {
            *positive = true;
            Ok(())
        }
        Some(Ok(false)) => {
            *negative = true;
            Ok(())
        }
        Some(Err(raw)) => Err((var, raw)),
    }
}

fn reject(flag: &str, reason: &str) -> String {
    format!("{flag} is not supported: {reason}.")
}

#[cfg(test)]
#[path = "ui_compat_args_tests.rs"]
mod tests;
