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

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use axum::http::{HeaderMap, HeaderValue, Uri, header};
use tokio::sync::Semaphore;

use super::canonical_authority;

pub(crate) const CONTENT_SECURITY_POLICY: &str = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'none'";
pub(crate) const PERMISSIONS_POLICY: &str = "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()";

pub(crate) const WEBUI_CONTROL_BODY_LIMIT_BYTES: usize = 2 * 1024 * 1024;
const DEFAULT_WEBUI_PREFIX: &str = "/webui";
const DEFAULT_WEBUI_API_PREFIX: &str = "/";
const DEFAULT_CONTROL_PERMITS: usize = 32;
const DEFAULT_SSE_PERMITS: usize = 16;
const DEFAULT_CONTROL_RATE_LIMIT: usize = 120;
const CONTROL_RATE_WINDOW: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub(crate) struct WebUiSecurityPolicy {
    pub(super) allowed_hosts: Arc<[String]>,
    pub(super) allowed_origins: Arc<[HeaderValue]>,
    pub(super) public_webui_prefix: Arc<str>,
    pub(super) api_prefix: Arc<str>,
    pub(super) control_permits: Arc<Semaphore>,
    pub(super) sse_permits: Arc<Semaphore>,
    control_rate: Arc<ControlRateLimit>,
}

impl fmt::Debug for WebUiSecurityPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WebUiSecurityPolicy")
            .field("allowed_hosts", &self.allowed_hosts)
            .field("allowed_origins", &self.allowed_origins)
            .field("public_webui_prefix", &self.public_webui_prefix)
            .field("api_prefix", &self.api_prefix)
            .field("control_permits", &self.control_permits.available_permits())
            .field("sse_permits", &self.sse_permits.available_permits())
            .field("control_rate_limit", &self.control_rate.max_requests)
            .finish()
    }
}

impl WebUiSecurityPolicy {
    pub(crate) fn new(
        allowed_hosts: Vec<String>,
        allowed_origins: Vec<HeaderValue>,
    ) -> Result<Self> {
        Self::with_prefixes_and_limits(
            allowed_hosts,
            allowed_origins,
            DEFAULT_WEBUI_PREFIX,
            DEFAULT_WEBUI_API_PREFIX,
            DEFAULT_CONTROL_PERMITS,
            DEFAULT_SSE_PERMITS,
        )
    }

    pub(crate) fn with_prefixes_and_limits(
        allowed_hosts: Vec<String>,
        allowed_origins: Vec<HeaderValue>,
        public_webui_prefix: &str,
        api_prefix: &str,
        control_limit: usize,
        sse_limit: usize,
    ) -> Result<Self> {
        Self::with_prefixes_limits_and_rate(
            allowed_hosts,
            allowed_origins,
            public_webui_prefix,
            api_prefix,
            control_limit,
            sse_limit,
            DEFAULT_CONTROL_RATE_LIMIT,
        )
    }

    pub(crate) fn with_prefixes_limits_and_rate(
        allowed_hosts: Vec<String>,
        allowed_origins: Vec<HeaderValue>,
        public_webui_prefix: &str,
        api_prefix: &str,
        control_limit: usize,
        sse_limit: usize,
        control_rate_limit: usize,
    ) -> Result<Self> {
        let hosts = allowed_hosts
            .into_iter()
            .map(|host| {
                canonical_authority(&host)
                    .ok_or_else(|| anyhow::anyhow!("invalid WebUI host authority '{host}'"))
            })
            .collect::<Result<Vec<_>>>()?;
        if hosts.is_empty() {
            bail!("WebUI security requires at least one allowed Host authority");
        }
        let origins = allowed_origins
            .into_iter()
            .map(validate_allowed_origin)
            .collect::<Result<Vec<_>>>()?;
        if origins.is_empty() {
            bail!("WebUI security requires at least one explicit allowed Origin");
        }
        Ok(Self {
            allowed_hosts: hosts.into(),
            allowed_origins: origins.into(),
            public_webui_prefix: normalize_prefix(public_webui_prefix, "WebUI public prefix")?
                .into(),
            api_prefix: normalize_prefix(api_prefix, "WebUI API prefix")?.into(),
            control_permits: Arc::new(Semaphore::new(control_limit)),
            sse_permits: Arc::new(Semaphore::new(sse_limit)),
            control_rate: Arc::new(ControlRateLimit::new(
                control_rate_limit,
                CONTROL_RATE_WINDOW,
            )),
        })
    }

    pub(super) fn try_record_control_request(&self) -> bool {
        self.control_rate.try_record()
    }
}

struct ControlRateLimit {
    max_requests: usize,
    window: Duration,
    hits: Mutex<VecDeque<Instant>>,
}

impl ControlRateLimit {
    fn new(max_requests: usize, window: Duration) -> Self {
        Self {
            max_requests,
            window,
            hits: Mutex::new(VecDeque::new()),
        }
    }

    fn try_record(&self) -> bool {
        if self.max_requests == 0 {
            return false;
        }
        let now = Instant::now();
        let mut hits = self.hits.lock().expect("control rate lock");
        while hits
            .front()
            .is_some_and(|hit| now.duration_since(*hit) >= self.window)
        {
            hits.pop_front();
        }
        if hits.len() >= self.max_requests {
            return false;
        }
        hits.push_back(now);
        true
    }
}

fn validate_allowed_origin(origin: HeaderValue) -> Result<HeaderValue> {
    let text = origin
        .to_str()
        .map_err(|_| anyhow::anyhow!("WebUI allowed Origin must be visible ASCII"))?;
    if text == "null" {
        bail!("WebUI allowed Origin cannot be null");
    }
    let uri: Uri = text
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid WebUI allowed Origin '{text}'"))?;
    let Some(scheme) = uri.scheme_str() else {
        bail!("WebUI allowed Origin requires http or https scheme");
    };
    if !matches!(scheme, "http" | "https") {
        bail!("WebUI allowed Origin scheme must be http or https");
    }
    let Some(authority) = uri.authority() else {
        bail!("WebUI allowed Origin requires an authority");
    };
    if canonical_authority(authority.as_str()).is_none() {
        bail!("WebUI allowed Origin authority is malformed");
    }
    if uri.path() != "/" || uri.query().is_some() {
        bail!("WebUI allowed Origin must not include a path or query");
    }
    Ok(origin)
}

fn normalize_prefix(prefix: &str, label: &str) -> Result<String> {
    let trimmed = prefix.trim_end_matches('/');
    let normalized = if trimmed.is_empty() { "/" } else { trimmed };
    if !normalized.starts_with('/')
        || normalized.contains('%')
        || normalized.contains('\\')
        || normalized.contains('?')
        || normalized.contains('#')
        || normalized
            .chars()
            .any(|ch| ch.is_control() || ch.is_whitespace())
    {
        bail!("{label} must be an absolute, undecoded path prefix");
    }
    Ok(normalized.to_string())
}
/// Add central browser security headers to static, API, and error responses.
pub(crate) fn apply_security_headers(headers: &mut HeaderMap) {
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(CONTENT_SECURITY_POLICY),
    );
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(
        header::HeaderName::from_static("permissions-policy"),
        HeaderValue::from_static(PERMISSIONS_POLICY),
    );
}
