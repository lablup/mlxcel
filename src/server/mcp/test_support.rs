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

//! Shared fixtures for the MCP tests: the scripted stdio server in
//! `tests/fixtures/mcp/fake_mcp_server.py` (issue #1457).

use std::collections::BTreeMap;
use std::time::Duration;

use super::config::{McpLimits, McpServerSpec, McpStartupConfig};

/// The scripted server's path.
pub(crate) const FAKE_SERVER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mcp/fake_mcp_server.py"
);

/// A spec that starts the scripted server with `extra_args`.
pub(crate) fn fake_spec(name: &str, extra_args: &[&str]) -> McpServerSpec {
    let mut args = vec![FAKE_SERVER.to_string()];
    args.extend(extra_args.iter().map(|arg| (*arg).to_string()));
    McpServerSpec {
        name: name.to_string(),
        command: "python3".to_string(),
        args,
        env: BTreeMap::from([("FAKE_MCP_VISIBLE".to_string(), "1".to_string())]),
    }
}

/// Bounds small enough that a misbehaving tool fails a test quickly.
pub(crate) fn test_limits() -> McpLimits {
    McpLimits {
        max_tool_rounds: 4,
        tool_timeout: Duration::from_secs(3),
        max_output_bytes: 4096,
    }
}

/// One scripted server named `fake` under [`test_limits`].
pub(crate) fn fake_config() -> McpStartupConfig {
    McpStartupConfig {
        servers: vec![fake_spec("fake", &[])],
        limits: test_limits(),
    }
}
