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

#![allow(dead_code)]

use std::fmt;

use anyhow::{Result, bail};
use axum::http::HeaderValue;
use base64::Engine;

use super::WebUiSecurityPolicy;

/// A generated, restart-local bearer credential. `Debug` never prints it.
#[derive(PartialEq, Eq)]
pub(crate) struct WebUiSessionCredential(String);

impl WebUiSessionCredential {
    pub(crate) fn generate() -> Self {
        let mut bytes = Vec::with_capacity(48);
        for _ in 0..3 {
            bytes.extend_from_slice(uuid::Uuid::new_v4().as_bytes());
        }
        Self(format!(
            "mlxcel_webui_{}",
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
        ))
    }

    pub(crate) fn into_terminal_secret(self) -> String {
        self.0
    }
}

impl fmt::Debug for WebUiSessionCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("WebUiSessionCredential(<redacted>)")
    }
}

/// Startup inputs needed to decide whether WebUI mode is safe to enable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WebUiSecurityConfig {
    pub(crate) enabled: bool,
    pub(crate) listen: WebUiListenKind,
    pub(crate) tls_enabled: bool,
    pub(crate) configured_key_present: bool,
    pub(crate) interactive_terminal: bool,
    pub(crate) allowed_hosts: Vec<String>,
    pub(crate) allowed_origins: Vec<HeaderValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WebUiListenKind {
    LoopbackTcp,
    NonLoopbackTcp,
    UnixSocket,
}

#[derive(Debug)]
pub(crate) struct ResolvedWebUiSecurity {
    pub(crate) policy: WebUiSecurityPolicy,
    pub(crate) generated_credential: Option<WebUiSessionCredential>,
}

pub(crate) fn resolve_webui_security(
    config: WebUiSecurityConfig,
) -> Result<Option<ResolvedWebUiSecurity>> {
    if !config.enabled {
        return Ok(None);
    }
    if config.listen == WebUiListenKind::UnixSocket {
        bail!(
            "--webui requires a TCP listener; use a loopback TCP listener behind a reverse proxy instead of a Unix socket"
        );
    }
    let generated_credential = if config.configured_key_present {
        None
    } else if config.listen == WebUiListenKind::LoopbackTcp && config.interactive_terminal {
        Some(WebUiSessionCredential::generate())
    } else {
        bail!(
            "--webui without --api-key is only allowed on an interactive loopback terminal so the generated session key can be shown once"
        );
    };
    if config.listen == WebUiListenKind::NonLoopbackTcp
        && (!config.configured_key_present || !config.tls_enabled)
    {
        bail!(
            "non-loopback --webui requires an explicit API key and TLS; alternatively bind the WebUI backend to loopback behind a TLS reverse proxy"
        );
    }
    Ok(Some(ResolvedWebUiSecurity {
        policy: WebUiSecurityPolicy::new(config.allowed_hosts, config.allowed_origins)?,
        generated_credential,
    }))
}
