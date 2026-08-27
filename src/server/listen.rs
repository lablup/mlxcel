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

//! Binding and accepting HTTP connections (#1432).
//!
//! One accept loop serves all three transports (TCP, TLS over TCP, and a Unix
//! domain socket) so that `--timeout` is enforced identically on each: every
//! accepted stream is wrapped in
//! [`TimeoutIo`](crate::server::http_timeout::TimeoutIo) before hyper sees it.
//! `axum::serve` is deliberately not used, because it owns the accepted
//! `TcpStream` and leaves no seam to wrap the I/O in.
//!
//! Socket options follow upstream `server_http_context::start`: `SO_REUSEADDR`
//! is always set, `SO_REUSEPORT` only with `--reuse-port`, and a `--host`
//! ending in `.sock` binds an `AF_UNIX` socket instead.

use std::io;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use axum::Router;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ConnBuilder;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, UnixListener};
use tokio_rustls::TlsAcceptor;
use tokio_rustls::rustls::ServerConfig as RustlsServerConfig;
use tower::Service;

use super::http_timeout::TimeoutIo;
use super::read_budget::{BudgetBody, ReadBudget};
use super::transport::{HttpTimeouts, ListenTarget};

/// Backoff applied after a transient `accept()` failure so a descriptor
/// exhaustion does not spin the loop at full speed.
const ACCEPT_BACKOFF: Duration = Duration::from_millis(50);

/// Everything the accept loop needs beyond the router itself.
pub(crate) struct ServeOptions {
    pub timeouts: HttpTimeouts,
    pub reuse_port: bool,
    pub tls: Option<Arc<RustlsServerConfig>>,
}

/// Bind `target` and serve `app` on it forever.
///
/// `announce` is called once with the human-readable listening address after
/// the socket is bound and before the first connection is accepted, so a
/// `--port 0` bind can report the ephemeral port it actually got.
pub(crate) async fn serve(
    target: &ListenTarget,
    app: Router,
    options: ServeOptions,
    announce: impl FnOnce(&str),
) -> Result<()> {
    match target {
        ListenTarget::Tcp { host, port } => {
            let listener = bind_tcp(host, *port, options.reuse_port)?;
            let local = listener
                .local_addr()
                .context("failed to read the bound TCP address")?;
            let scheme = if options.tls.is_some() {
                "https"
            } else {
                "http"
            };
            announce(&format!("{scheme}://{local}"));
            accept_tcp(listener, app, options).await
        }
        ListenTarget::Unix(path) => {
            if options.tls.is_some() {
                bail!(
                    "--ssl-cert-file / --ssl-key-file cannot be combined with a Unix domain \
                     socket ({}); terminate TLS in front of the socket, or listen on TCP",
                    path.display()
                );
            }
            let listener = bind_unix(path)?;
            announce(&format!("unix://{}", path.display()));
            accept_unix(listener, app, options.timeouts).await
        }
    }
}

/// Bind a TCP listener, applying the b10621 socket options.
///
/// `SO_REUSEADDR` is unconditional, matching upstream's `set_socket_options`
/// callback; `SO_REUSEPORT` is applied only for `--reuse-port` and, where the
/// platform lacks it, produces the same warning upstream emits rather than a
/// startup failure.
fn bind_tcp(host: &str, port: u16, reuse_port: bool) -> Result<TcpListener> {
    let addr = resolve_socket_addr(host, port)?;
    let domain = socket2::Domain::for_address(addr);
    let socket = socket2::Socket::new(domain, socket2::Type::STREAM, Some(socket2::Protocol::TCP))
        .with_context(|| format!("failed to create a TCP socket for {addr}"))?;
    socket
        .set_reuse_address(true)
        .with_context(|| format!("failed to set SO_REUSEADDR on the listener for {addr}"))?;

    if reuse_port {
        #[cfg(unix)]
        socket.set_reuse_port(true).with_context(|| {
            format!(
                "--reuse-port: failed to set SO_REUSEPORT on the listener for {addr}. The option \
                 exists on this platform but the kernel refused it; drop --reuse-port to bind \
                 exclusively"
            )
        })?;
        #[cfg(not(unix))]
        tracing::warn!("SO_REUSEPORT is not supported on this platform; --reuse-port is ignored");
    }

    socket
        .set_nonblocking(true)
        .context("failed to put the listener in non-blocking mode")?;
    socket.bind(&addr.into()).with_context(|| {
        format!("failed to bind {addr}; another process may already hold the port")
    })?;
    socket
        .listen(1024)
        .with_context(|| format!("failed to listen on {addr}"))?;

    TcpListener::from_std(std::net::TcpListener::from(socket))
        .with_context(|| format!("failed to register the listener for {addr} with tokio"))
}

/// Resolve `host:port` to a single socket address.
///
/// A hostname with several records resolves to the first address, which is
/// what `bind_to_port` does upstream. An unresolvable host is a startup error
/// naming the value, not a silent bind to a different interface.
fn resolve_socket_addr(host: &str, port: u16) -> Result<SocketAddr> {
    let literal = if host.contains(':') && !host.starts_with('[') {
        // Bare IPv6 literal: bracket it so the port parses.
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    literal
        .to_socket_addrs()
        .with_context(|| format!("--host {host}: cannot resolve to an address"))?
        .next()
        .ok_or_else(|| anyhow::anyhow!("--host {host}: resolved to no addresses"))
}

/// Bind a Unix domain socket, clearing a stale path first.
fn bind_unix(path: &Path) -> Result<UnixListener> {
    if path.exists() {
        std::fs::remove_file(path)
            .with_context(|| format!("failed to remove stale socket: {}", path.display()))?;
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create socket directory: {}", parent.display()))?;
    }
    UnixListener::bind(path)
        .with_context(|| format!("failed to bind Unix socket: {}", path.display()))
}

async fn accept_tcp(listener: TcpListener, app: Router, options: ServeOptions) -> Result<()> {
    let acceptor = options.tls.map(TlsAcceptor::from);
    loop {
        let stream = match listener.accept().await {
            Ok((stream, _peer)) => stream,
            Err(e) if is_transient_accept_error(&e) => {
                tracing::warn!("transient accept error, retrying: {e}");
                tokio::time::sleep(ACCEPT_BACKOFF).await;
                continue;
            }
            Err(e) => return Err(e).context("TCP accept failed"),
        };
        let app = app.clone();
        let timeouts = options.timeouts;
        let budget = ReadBudget::new();
        match acceptor.clone() {
            Some(acceptor) => {
                tokio::spawn(async move {
                    match acceptor.accept(stream).await {
                        Ok(tls) => {
                            drive_connection(
                                TimeoutIo::new(tls, timeouts, budget.clone()),
                                app,
                                budget,
                            )
                            .await
                        }
                        Err(e) => tracing::debug!("TLS handshake failed: {e}"),
                    }
                });
            }
            None => {
                tokio::spawn(drive_connection(
                    TimeoutIo::new(stream, timeouts, budget.clone()),
                    app,
                    budget,
                ));
            }
        }
    }
}

async fn accept_unix(listener: UnixListener, app: Router, timeouts: HttpTimeouts) -> Result<()> {
    loop {
        let stream = match listener.accept().await {
            Ok((stream, _peer)) => stream,
            Err(e) if is_transient_accept_error(&e) => {
                tracing::warn!("transient accept error, retrying: {e}");
                tokio::time::sleep(ACCEPT_BACKOFF).await;
                continue;
            }
            Err(e) => return Err(e).context("Unix socket accept failed"),
        };
        let app = app.clone();
        let budget = ReadBudget::new();
        tokio::spawn(drive_connection(
            TimeoutIo::new(stream, timeouts, budget.clone()),
            app,
            budget,
        ));
    }
}

/// A descriptor or buffer shortage is a load condition, not a reason to stop
/// serving; the loop backs off and retries instead of returning from `serve`
/// and terminating the process.
fn is_transient_accept_error(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::Interrupted
            | io::ErrorKind::WouldBlock
            | io::ErrorKind::OutOfMemory
    ) || is_descriptor_exhaustion(e)
}

/// Descriptor exhaustion has no `io::ErrorKind`, so it is matched by errno.
#[cfg(unix)]
fn is_descriptor_exhaustion(e: &io::Error) -> bool {
    matches!(e.raw_os_error(), Some(libc::EMFILE) | Some(libc::ENFILE))
}

#[cfg(not(unix))]
fn is_descriptor_exhaustion(_e: &io::Error) -> bool {
    false
}

/// Serve one accepted connection.
///
/// Both bodies are wrapped in [`BudgetBody`] so the connection's
/// [`ReadBudget`] tracks exactly the window in which the server is waiting for
/// request bytes: the request body finishing stands the read deadline down for
/// the handler and the response, and the response body finishing arms it again
/// for the next request on a keep-alive connection. See
/// [`crate::server::read_budget`] for why an ungated deadline would cut off
/// slow generations.
async fn drive_connection<I>(io: I, app: Router, budget: Arc<ReadBudget>)
where
    I: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let service =
        hyper::service::service_fn(move |request: hyper::Request<hyper::body::Incoming>| {
            let mut app = app.clone();
            let budget = budget.clone();
            async move {
                let (parts, body) = request.into_parts();
                let request = hyper::Request::from_parts(
                    parts,
                    axum::body::Body::new(BudgetBody::request(body, budget.clone())),
                );
                let response = app.call(request).await?;
                let (parts, body) = response.into_parts();
                Ok::<_, std::convert::Infallible>(hyper::Response::from_parts(
                    parts,
                    axum::body::Body::new(BudgetBody::response(body, budget)),
                ))
            }
        });
    if let Err(e) = ConnBuilder::new(TokioExecutor::new())
        .serve_connection(TokioIo::new(io), service)
        .await
    {
        // A client that disconnects mid-stream, or one that hits the
        // `--timeout` budget, lands here. Both are ordinary, so this stays at
        // debug level.
        tracing::debug!("connection closed: {e}");
    }
}

#[cfg(test)]
#[path = "listen_tests.rs"]
mod listen_tests;
