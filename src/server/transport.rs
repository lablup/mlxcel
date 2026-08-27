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

//! HTTP transport configuration aligned with `llama-server` b10621 (#1432).
//!
//! This module owns the value resolution for the transport-layer options in
//! the pinned compatibility manifest shard
//! `compat/llama-server/b10621/transport-tls-cors.json`: the listen target,
//! the socket read/write timeout, the API path prefix, the SSE ping interval
//! and the HTTP worker-thread count. The bind/accept side lives in
//! [`crate::server::listen`], the TLS material in [`crate::server::tls`] and
//! the CORS policy in [`crate::server::cors`].
//!
//! ## The `--timeout` semantic collision
//!
//! Before #1432, mlxcel parsed `--timeout` / `LLAMA_ARG_TIMEOUT` as a
//! per-request decode watchdog with a 600-second default, while b10621
//! defines the same spelling as the HTTP socket read/write timeout with a
//! 3600-second default (upstream
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>
//! sets `timeout_read` and `timeout_write` from one value). Two different
//! controls answered to one name, so a deployment copied from a
//! `llama-server` command line silently got a decode cancellation instead of
//! a socket timeout.
//!
//! `--timeout` now means what b10621 means by it. The decode watchdog keeps
//! its behavior under the mlxcel-native spelling `--decode-timeout` /
//! `MLXCEL_DECODE_TIMEOUT`, still defaulting to 600 seconds. A startup
//! migration warning fires when `--timeout` is set without `--decode-timeout`,
//! because that is exactly the command line whose meaning changed.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, bail};

/// b10621 default for `--timeout` (`LLAMA_ARG_TIMEOUT`), in seconds.
///
/// Upstream sets `params.timeout_read` and `params.timeout_write` to this
/// value and hands both to `httplib::Server::set_read_timeout` /
/// `set_write_timeout`.
pub const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 3600;

/// Default for the mlxcel-native decode watchdog `--decode-timeout`
/// (`MLXCEL_DECODE_TIMEOUT`), in seconds.
///
/// This is the value `--timeout` carried before #1432, kept unchanged so an
/// operator who only renames the flag keeps the previous behavior.
pub const DEFAULT_DECODE_TIMEOUT_SECS: u64 = 600;

/// b10621 default for `--sse-ping-interval` (`LLAMA_ARG_SSE_PING_INTERVAL`),
/// in seconds. `-1` disables the pings.
pub const DEFAULT_SSE_PING_INTERVAL_SECS: i64 = 30;

/// b10621 default for `--cors-origins` (`LLAMA_ARG_CORS_ORIGINS`).
pub const DEFAULT_CORS_ORIGINS: &str = "*";

/// b10621 default for `--cors-methods` (`LLAMA_ARG_CORS_METHODS`).
pub const DEFAULT_CORS_METHODS: &str = "GET, POST, DELETE, OPTIONS";

/// b10621 default for `--cors-headers` (`LLAMA_ARG_CORS_HEADERS`).
pub const DEFAULT_CORS_HEADERS: &str = "*";

/// b10621 default for `--threads-http` (`LLAMA_ARG_THREADS_HTTP`). Any value
/// below 1 selects the automatic sizing in [`resolve_http_threads`].
pub const DEFAULT_THREADS_HTTP: i64 = -1;

/// Where the HTTP server listens.
///
/// b10621 selects a Unix domain socket when `--host` ends in `.sock`
/// (upstream `server_http_context::start`), and binds an ephemeral TCP port
/// when `--port 0`. mlxcel historically used `--port 0` to mean "treat
/// `--host` as a socket path", so that spelling is still honored, with a
/// deprecation warning, for a host that is a filesystem path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListenTarget {
    /// TCP listener. `port == 0` binds an ephemeral port, as b10621 does.
    Tcp { host: String, port: u16 },
    /// Unix domain socket at this path.
    Unix(PathBuf),
}

/// Resolution of `--host` / `--port` into a [`ListenTarget`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListenResolution {
    pub target: ListenTarget,
    /// Set when the legacy mlxcel `--port 0` socket spelling selected the
    /// Unix socket. The caller logs it once at startup.
    pub legacy_socket_warning: Option<String>,
}

/// Resolve the listen target from `--host` and `--port`.
///
/// Rules, in order:
///
/// 1. `--host` ending in `.sock` is a Unix domain socket path, whatever
///    `--port` says. This is the b10621 rule and the spelling to use.
/// 2. `--port 0` with a host that looks like a filesystem path (it contains a
///    `/`) is the legacy mlxcel socket spelling. It still works and returns a
///    deprecation warning for the caller to log.
/// 3. Anything else is TCP. `--port 0` then binds an ephemeral port, matching
///    b10621's `bind_to_any_port`.
///
/// An empty host is rejected rather than silently binding everywhere.
pub fn resolve_listen_target(host: &str, port: u16) -> Result<ListenResolution> {
    if host.trim().is_empty() {
        bail!(
            "--host must not be empty; pass an IP address, a hostname, or a path ending in .sock"
        );
    }

    if host.ends_with(".sock") {
        return Ok(ListenResolution {
            target: ListenTarget::Unix(PathBuf::from(host)),
            legacy_socket_warning: None,
        });
    }

    if port == 0 && host.contains('/') {
        return Ok(ListenResolution {
            target: ListenTarget::Unix(PathBuf::from(host)),
            legacy_socket_warning: Some(format!(
                "--host {host} --port 0 selected a Unix domain socket through the legacy mlxcel \
                 spelling. llama-server b10621 selects a socket from a --host ending in .sock and \
                 reads --port 0 as \"bind an ephemeral TCP port\"; rename the socket to end in \
                 .sock to keep this working when the legacy spelling is removed"
            )),
        });
    }

    Ok(ListenResolution {
        target: ListenTarget::Tcp {
            host: host.to_string(),
            port,
        },
        legacy_socket_warning: None,
    })
}

/// Socket read/write timeouts applied to every accepted connection.
///
/// b10621 hands one `--timeout` value to both `set_read_timeout` and
/// `set_write_timeout`, so the two are always equal here as well; the type
/// keeps them separate because the enforcement points differ (see
/// [`crate::server::http_timeout::TimeoutIo`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HttpTimeouts {
    pub read: Duration,
    pub write: Duration,
}

impl HttpTimeouts {
    /// Build the pair from the single `--timeout` value.
    ///
    /// `0` is preserved rather than reinterpreted as "no timeout": b10621
    /// passes it straight through to `select()`, where it closes a connection
    /// as soon as a read or write would block. Silently turning it into
    /// "disabled" would make the same command line behave in opposite ways on
    /// the two servers.
    pub fn from_secs(secs: u64) -> Self {
        let d = Duration::from_secs(secs);
        Self { read: d, write: d }
    }
}

impl Default for HttpTimeouts {
    fn default() -> Self {
        Self::from_secs(DEFAULT_HTTP_TIMEOUT_SECS)
    }
}

/// Validate and normalize `--api-prefix` (`LLAMA_ARG_API_PREFIX`).
///
/// b10621 concatenates the raw value in front of every registered path
/// (`srv->Get(params.api_prefix + "/health", ...)`) and does not validate it,
/// so a value with a trailing slash or without a leading one produces routes
/// no client can reach. mlxcel rejects those at startup instead, because the
/// alternative is a server that answers 404 on every endpoint with no
/// explanation.
///
/// Returns the empty string for the default (no prefix).
pub fn resolve_api_prefix(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(String::new());
    }
    let bad = |why: &str| -> anyhow::Error {
        anyhow::anyhow!(
            "invalid --api-prefix {raw:?}: {why}. The prefix is a path the server serves from, \
             written with a leading slash and without a trailing one, for example --api-prefix \
             /llama"
        )
    };
    if !trimmed.starts_with('/') {
        return Err(bad("it must start with '/'"));
    }
    if trimmed.len() > 1 && trimmed.ends_with('/') {
        return Err(bad("it must not end with '/'"));
    }
    if trimmed == "/" {
        return Err(bad(
            "'/' is the default route set, not a prefix; leave the flag unset instead",
        ));
    }
    if trimmed.contains("//") {
        return Err(bad("it must not contain an empty path segment ('//')"));
    }
    if trimmed.contains('?') || trimmed.contains('#') {
        return Err(bad("it must not contain a query string or a fragment"));
    }
    if trimmed.chars().any(|c| c.is_control() || c == ' ') {
        return Err(bad("it must not contain spaces or control characters"));
    }
    Ok(trimmed.to_string())
}

/// Resolve `--threads-http` into the number of HTTP worker threads.
///
/// b10621 sizes its `httplib::ThreadPool` with
/// `max(n_parallel + 4, hardware_concurrency() - 1)` when the flag is below
/// 1, reserving four threads above the slot count for monitoring and health
/// traffic. mlxcel serves HTTP on the Tokio runtime, so the resolved value
/// becomes the runtime's worker-thread count; the same formula keeps a copied
/// command line sized the same way on both servers.
///
/// The result is always at least 1: a zero-worker runtime cannot serve.
pub fn resolve_http_threads(requested: i64, n_parallel: usize) -> usize {
    if requested >= 1 {
        return usize::try_from(requested).unwrap_or(usize::MAX);
    }
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let reserved = n_parallel.saturating_add(4);
    reserved.max(available.saturating_sub(1)).max(1)
}

/// Build the Tokio runtime that serves HTTP, sized by `--threads-http`.
///
/// b10621 dedicates a thread pool of this size to HTTP request processing.
/// mlxcel serves on Tokio, so the same number becomes the runtime's
/// worker-thread count; inference itself runs on its own dedicated worker
/// threads either way, so this knob controls the same thing on both servers:
/// how much parallelism the HTTP side gets.
///
/// The runtime is built by the two binary entry points BEFORE any model work,
/// so `--threads-http` takes effect for the whole process rather than only for
/// the accept loop.
pub fn build_http_runtime(worker_threads: usize) -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads.max(1))
        .thread_name("mlxcel-http")
        .enable_all()
        .build()
}

/// Resolve `--sse-ping-interval` into a keepalive interval.
///
/// `-1` (and any negative value) disables the pings, matching b10621's
/// documented sentinel. `0` is rejected: upstream would busy-emit a comment
/// on every scheduler pass, which is a misconfiguration rather than a mode.
pub fn resolve_sse_ping_interval(secs: i64) -> Result<Option<Duration>> {
    if secs < 0 {
        return Ok(None);
    }
    if secs == 0 {
        bail!(
            "--sse-ping-interval 0 would emit an SSE comment continuously; pass a positive number \
             of seconds, or -1 to disable pings"
        );
    }
    Ok(Some(Duration::from_secs(secs as u64)))
}

/// The startup migration warning for the `--timeout` semantic change, or
/// `None` when the command line is unambiguous.
///
/// Fires when `--timeout` (or `LLAMA_ARG_TIMEOUT`) was given and
/// `--decode-timeout` was not, which is exactly the command line whose
/// meaning changed in #1432: it used to cancel a stalled decode and now
/// bounds socket reads and writes.
pub fn timeout_migration_warning(
    http_timeout_was_set: bool,
    decode_timeout_was_set: bool,
) -> Option<String> {
    if !http_timeout_was_set || decode_timeout_was_set {
        return None;
    }
    Some(format!(
        "--timeout / LLAMA_ARG_TIMEOUT now configures the HTTP socket read/write timeout \
         (llama-server b10621 semantics, default {DEFAULT_HTTP_TIMEOUT_SECS}s), not the decode \
         watchdog it configured before. The decode watchdog moved to --decode-timeout / \
         MLXCEL_DECODE_TIMEOUT and still defaults to {DEFAULT_DECODE_TIMEOUT_SECS}s. Pass \
         --decode-timeout explicitly to keep the previous behavior and silence this warning"
    ))
}

#[cfg(test)]
#[path = "transport_tests.rs"]
mod transport_tests;
