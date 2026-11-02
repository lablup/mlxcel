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

//! TCP transport backend for inter-node communication.
//!
//! Implements the [`Transport`] trait using plain TCP sockets with a simple
//! length-prefixed framing protocol ([`super::tcp_frame`]):
//!
//! ```text
//! [kind: u8][payload_len: u64 BE][payload: bytes][tag, with auth]
//! ```
//!
//! For RPC, the `kind` distinguishes requests from responses and a 64-bit
//! request ID is prepended to the payload to correlate responses. The
//! response travels back on the connection the request went out on: the
//! accept side answers on its accepted socket, and `rpc_call` reads that
//! pooled outbound connection until the matching response arrives (#2712),
//! releasing the connection only then and dropping it on a timeout or any
//! error, so a late response is never left unread on a reused connection.
//!
//! # Security Notes
//!
//! With [`TcpTransportConfig::auth`] set, every connection runs the mutual
//! HMAC-SHA256 cluster handshake ([`super::cluster_handshake`]) before any
//! frame is read or sent: the accept side authenticates the peer in the
//! per-connection task before `handle_connection` dispatches anything, and
//! the connect side authenticates inside the connection pool when the
//! socket is opened. An unauthenticated connection never reaches `recv()`,
//! the RPC handler, or the pending-RPC table. The `from` address reported by
//! `recv()` is then the peer's authenticated listener identity (bound to its
//! TCP source IP), not the ephemeral `ip:port` of the accepted socket.
//!
//! With `auth` set, a connection that has not finished the handshake does not
//! hold one of the `max_connections` slots. It draws from a separate pending
//! pool ([`TcpTransportConfig::max_pending_handshakes`] in flight, at most
//! [`TcpTransportConfig::max_pending_handshakes_per_ip`] from one source IP)
//! for the duration of the handshake only; a socket beyond either bound is
//! closed at accept. A flood of sockets that connect and send nothing can
//! therefore delay new connections from any source while the pending pool is
//! full (each slot frees within the handshake timeout), but cannot take slots
//! from established authenticated connections or inject frames (#2655). A
//! zero bound is refused at `bind` when `auth` is set. The per-source bound
//! counts IPv4 sources per address and IPv6 sources per /64 (#2692). A failed
//! handshake is logged with its class (closed before the handshake at debug;
//! identity not owned, probable secret mismatch, altered transcript at warn,
//! rate limited through [`super::handshake_warn`] with a budget per class so
//! junk connections cannot hide those typed refusals). The `max_connections`
//! check happens inside the handshake, after the peer authenticated: a peer
//! that finds this node full gets the typed at-capacity verdict
//! ([`super::cluster_handshake::ServerAtCapacity`], not fatal) instead of an
//! accepted handshake followed by a closed socket.
//!
//! Without `auth` (the default, used by single-node serving) the transport
//! performs no authentication, and the reported `from` is the raw socket
//! peer address. Every distributed role, including the disaggregated prefill,
//! decode and router roles, sets `auth` (#2650, #2655).
//!
//! With `auth` set, every frame after the handshake is protected with keys
//! derived from that handshake ([`super::frame_protection`], #2656): the
//! default suite (AES-256-GCM) authenticates and encrypts the payload, binds
//! each frame to its connection and to an implicit per-direction sequence
//! number, and the receiver closes the connection on the first frame that
//! fails to verify (logged through the same rate limiter as failed
//! handshakes, never with payload bytes). An on-path attacker can therefore
//! still drop or delay traffic, but cannot read, alter, inject, replay or
//! reorder frames on an authenticated connection. Every send path (messages,
//! `send_stream`, RPC requests and responses) and the single inbound reader
//! go through [`ProtectedStream`]; there is no unprotected framing path with
//! `auth` set. In every deployment, also:
//! - Bind only to trusted network interfaces (e.g., `127.0.0.1` or a
//!   private VLAN)
//! - Use firewall rules to restrict access to the listening port

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use bytes::Bytes;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, Notify, Semaphore, mpsc};

use super::cluster_auth::ClusterAuth;
use super::cluster_handshake::{
    ConnectionLimitReached, HandshakeRejected, PeerClosedBeforeHandshake,
    server_handshake_admitting,
};
use super::connection_pool::{ConnectionPool, PoolConfig};
use super::frame_protection::{FrameChannel, FrameRejected, FrameSuite};
use super::handshake_warn::{HandshakeWarnClass, handshake_warn_allowed, handshake_warn_class};
use super::pending_handshakes::PendingHandshakeLimiter;
pub use super::pending_handshakes::{
    DEFAULT_MAX_PENDING_HANDSHAKES, DEFAULT_MAX_PENDING_HANDSHAKES_PER_IP,
};
use super::tcp_frame::{
    ProtectedStream, UnknownAuthenticatedKind, deserialize_message, encode_message_frame,
};
use super::transport::{
    BoxedAsyncRead, MessageKind, RpcHandler, Transport, TransportBackend, TransportMessage,
};

/// Default maximum number of concurrent inbound connections.
const DEFAULT_MAX_CONNECTIONS: usize = 512;

/// Largest RPC response frame `rpc_call` reads (#2726). `read_frame`
/// allocates the claimed length before the frame authenticates, so the
/// transport-wide `max_frame_size` (256 MiB by default) would let a peer,
/// or a party on the path, force that allocation per call. The only caller
/// (the latency benchmark) exchanges a few bytes.
const MAX_RPC_RESPONSE_FRAME: usize = 1 << 20;

/// TCP transport configuration.
#[derive(Debug, Clone)]
pub struct TcpTransportConfig {
    /// Address to bind the local listener to (e.g., `"0.0.0.0:9100"`).
    pub bind_address: String,
    /// Connection pool configuration.
    pub pool_config: PoolConfig,
    /// Maximum allowed frame size in bytes (default 256 MiB).
    pub max_frame_size: usize,
    /// Maximum concurrent inbound connections (DoS protection). With `auth`
    /// set, only authenticated connections count against it, and a peer that
    /// authenticates while the cap is full is refused with the typed
    /// at-capacity verdict inside the handshake.
    pub max_connections: usize,
    /// With `auth` set: maximum inbound connections that may be inside the
    /// cluster handshake at once, across all sources (default
    /// [`DEFAULT_MAX_PENDING_HANDSHAKES`]). Sockets beyond it are closed at
    /// accept. Must be non-zero when `auth` is set (`bind` fails otherwise).
    /// Ignored without `auth`.
    pub max_pending_handshakes: usize,
    /// With `auth` set: maximum inbound connections inside the handshake from
    /// one source (default [`DEFAULT_MAX_PENDING_HANDSHAKES_PER_IP`]). An
    /// IPv4 source is counted per address (IPv4-mapped IPv6 sources count
    /// against their IPv4 address); an IPv6 source is counted per /64, so a
    /// host cannot bypass the bound by rotating addresses inside its prefix.
    /// Cluster nodes inside one IPv6 /64 therefore share this budget, and a
    /// large IPv6 cluster in a single /64 may need a higher value. Must be
    /// non-zero when `auth` is set (`bind` fails otherwise). Ignored without
    /// `auth`.
    pub max_pending_handshakes_per_ip: usize,
    /// Cluster authentication. When `Some`, both inbound and outbound
    /// connections must complete the mutual cluster handshake before any
    /// frame is exchanged (see the module-level Security Notes). `None`
    /// keeps the unauthenticated behavior for single-node defaults.
    pub auth: Option<Arc<ClusterAuth>>,
    /// Per-frame protection suite used on every authenticated connection
    /// (#2656). Ignored without `auth`. Every node of a cluster must use the
    /// same suite; the handshake refuses a peer configured differently. The
    /// CLI never changes it from [`FrameSuite::default`]; the field exists so
    /// the activation benchmark can measure the alternatives on the real
    /// transport. There is no unprotected authenticated mode.
    pub frame_suite: FrameSuite,
}

impl Default for TcpTransportConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:0".to_string(),
            pool_config: PoolConfig::default(),
            max_frame_size: 256 * 1024 * 1024,
            max_connections: DEFAULT_MAX_CONNECTIONS,
            max_pending_handshakes: DEFAULT_MAX_PENDING_HANDSHAKES,
            max_pending_handshakes_per_ip: DEFAULT_MAX_PENDING_HANDSHAKES_PER_IP,
            auth: None,
            frame_suite: FrameSuite::default(),
        }
    }
}

/// TCP-based transport implementing the [`Transport`] trait.
///
/// Used by: distributed inference, pipeline-parallel communication,
/// tensor-parallel all-reduce.
pub struct TcpTransport {
    config: TcpTransportConfig,
    pool: ConnectionPool,
    /// Listener address, set once at construction. Not behind a lock since
    /// it is immutable after `bind()`.
    listener_addr: String,
    /// Channel for inbound messages from the accept loop.
    inbound_tx: mpsc::Sender<(String, TransportMessage)>,
    inbound_rx: Mutex<mpsc::Receiver<(String, TransportMessage)>>,
    /// Channel for inbound stream data.
    stream_tx: mpsc::Sender<(String, BoxedAsyncRead)>,
    stream_rx: Mutex<mpsc::Receiver<(String, BoxedAsyncRead)>>,
    /// Next RPC request ID (monotonically increasing).
    rpc_next_id: Mutex<u64>,
    /// RPC handler, shared with the accept loop so `serve_rpc` takes effect.
    rpc_handler: Arc<Mutex<Option<RpcHandler>>>,
    /// Shutdown flag (atomic for fast, lock-free checking).
    shutdown_flag: Arc<AtomicBool>,
    /// Shutdown signal for select-based waiters.
    shutdown: Arc<Notify>,
    /// Accept loop join handle.
    accept_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl TcpTransport {
    /// Create a new TCP transport and bind the listener.
    pub async fn bind(config: TcpTransportConfig) -> Result<Self> {
        // Validate the pending-handshake bounds before binding anything, so a
        // zero bound fails loudly instead of yielding a listener that closes
        // every socket at accept. Ignored without `auth`, as documented.
        let pending = match config.auth {
            Some(_) => Some(PendingHandshakeLimiter::new(
                config.max_pending_handshakes,
                config.max_pending_handshakes_per_ip,
            )?),
            None => None,
        };
        let listener = TcpListener::bind(&config.bind_address)
            .await
            .with_context(|| format!("failed to bind TCP listener on {}", config.bind_address))?;
        let local_addr = listener
            .local_addr()
            .context("failed to get local address")?
            .to_string();

        let pool = ConnectionPool::with_auth(
            config.pool_config.clone(),
            config.auth.clone(),
            &local_addr,
            config.frame_suite,
        );
        let (inbound_tx, inbound_rx) = mpsc::channel(256);
        let (stream_tx, stream_rx) = mpsc::channel(64);
        let shutdown = Arc::new(Notify::new());
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let rpc_handler: Arc<Mutex<Option<RpcHandler>>> = Arc::new(Mutex::new(None));

        let transport = Self {
            config: config.clone(),
            pool,
            listener_addr: local_addr,
            inbound_tx,
            inbound_rx: Mutex::new(inbound_rx),
            stream_tx,
            stream_rx: Mutex::new(stream_rx),
            rpc_next_id: Mutex::new(0),
            rpc_handler: rpc_handler.clone(),
            shutdown_flag: shutdown_flag.clone(),
            shutdown: shutdown.clone(),
            accept_handle: Mutex::new(None),
        };

        // Start the accept loop.
        let inbound_tx_clone = transport.inbound_tx.clone();
        let stream_tx_clone = transport.stream_tx.clone();
        let max_frame = config.max_frame_size;
        let max_conns = config.max_connections;
        let accept_auth = config
            .auth
            .clone()
            .zip(pending)
            .map(|(auth, pending)| AcceptAuth {
                auth,
                local_identity: transport.listener_addr.clone(),
                pending,
                suite: config.frame_suite,
            });

        let handle = tokio::spawn(accept_loop(
            listener,
            inbound_tx_clone,
            stream_tx_clone,
            rpc_handler,
            max_frame,
            max_conns,
            accept_auth,
            shutdown_flag,
            shutdown.clone(),
        ));

        *transport.accept_handle.lock().await = Some(handle);

        Ok(transport)
    }

    /// Whether this transport runs the cluster handshake on every connection
    /// ([`TcpTransportConfig::auth`] set). A role that must only ever talk to
    /// authenticated peers (the disaggregated router) checks this at build
    /// time instead of trusting its caller.
    pub fn is_authenticated(&self) -> bool {
        self.config.auth.is_some()
    }
}

impl Transport for TcpTransport {
    fn connect(
        &self,
        peers: &[String],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let peers = peers.to_vec();
        Box::pin(async move {
            // Pre-warm connections by acquiring and releasing one per peer.
            // A peer that is not up yet is a warning (it may start later);
            // a peer that rejected the cluster handshake is fatal, because
            // no retry can change a secret mismatch.
            for peer in &peers {
                match self.pool.acquire(peer).await {
                    Ok(stream) => {
                        self.pool.release(peer, stream).await;
                    }
                    Err(e) if e.downcast_ref::<HandshakeRejected>().is_some() => {
                        return Err(e).with_context(|| {
                            format!("cluster authentication with peer {peer} failed")
                        });
                    }
                    Err(e) => {
                        tracing::warn!("Failed to pre-connect to peer {peer}: {e}");
                    }
                }
            }
            Ok(())
        })
    }

    fn send(
        &self,
        peer: &str,
        message: TransportMessage,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let peer = peer.to_string();
        Box::pin(async move {
            // The message is encoded straight into its frame buffer and
            // sealed in place; no second copy of a 405 KB logits reply.
            let (kind, frame) = encode_message_frame(&message);
            let mut stream = self.pool.acquire(&peer).await?;
            stream.write_prepared_frame(kind, frame).await?;
            self.pool.release(&peer, stream).await;
            Ok(())
        })
    }

    fn recv(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(String, TransportMessage)>> + Send + '_>>
    {
        Box::pin(async move {
            let mut rx = self.inbound_rx.lock().await;
            rx.recv()
                .await
                .ok_or_else(|| anyhow::anyhow!("transport inbound channel closed"))
        })
    }

    fn send_stream(
        &self,
        peer: &str,
        data: Bytes,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let peer = peer.to_string();
        Box::pin(async move {
            let mut stream = self.pool.acquire(&peer).await?;
            // A stream chunk is a TensorData frame whose payload is the raw
            // bytes; it takes the same protected path as every other frame.
            stream.write_frame(MessageKind::TensorData, &data).await?;
            self.pool.release(&peer, stream).await;
            Ok(())
        })
    }

    fn recv_stream(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(String, BoxedAsyncRead)>> + Send + '_>>
    {
        Box::pin(async move {
            let mut rx = self.stream_rx.lock().await;
            rx.recv()
                .await
                .ok_or_else(|| anyhow::anyhow!("transport stream channel closed"))
        })
    }

    fn rpc_call(
        &self,
        peer: &str,
        request: &[u8],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>>> + Send + '_>> {
        let peer = peer.to_string();
        let request = request.to_vec();
        Box::pin(async move {
            let request_id = {
                let mut id = self.rpc_next_id.lock().await;
                let current = *id;
                *id = current.wrapping_add(1);
                current
            };

            // Frame: [RpcRequest kind][len][request_id: u64 BE][request bytes]
            let mut payload = Vec::with_capacity(8 + request.len());
            payload.extend_from_slice(&request_id.to_be_bytes());
            payload.extend_from_slice(&request);

            // The response comes back on this same connection (#2712): the
            // peer answers on the socket it accepted, which is this pooled
            // outbound one, so it is read here until the matching response
            // arrives. The connection is released only after that; on a
            // timeout or any error it is dropped, so a late response can
            // never be left unread on a connection the pool hands out again.
            let timeout = self.config.pool_config.connect_timeout;
            // A response is read under a reply-sized cap (#2726): the frame
            // body is allocated before it authenticates, and no caller needs
            // more than a few KiB.
            let max_frame_size = MAX_RPC_RESPONSE_FRAME.min(self.config.max_frame_size);
            let mut stream = self.pool.acquire(&peer).await?;
            let exchange = async {
                stream
                    .write_frame(MessageKind::RpcRequest, &payload)
                    .await?;
                loop {
                    let (kind, frame) = stream.read_frame(max_frame_size).await?;
                    if kind != MessageKind::RpcResponse {
                        anyhow::bail!(
                            "unexpected {kind:?} frame from {peer} while waiting for the RPC response"
                        );
                    }
                    if frame.len() < 8 {
                        anyhow::bail!("RPC response from {peer} is shorter than its request id");
                    }
                    let got = u64::from_be_bytes(frame[..8].try_into().expect("8-byte request id"));
                    if got == request_id {
                        return Ok(frame[8..].to_vec());
                    }
                    tracing::debug!(
                        "Ignoring RPC response {got} from {peer} while waiting for {request_id}"
                    );
                }
            };
            match tokio::time::timeout(timeout, exchange).await {
                Ok(Ok(response)) => {
                    self.pool.release(&peer, stream).await;
                    Ok(response)
                }
                Ok(Err(err)) => {
                    drop(stream);
                    Err(err)
                }
                Err(_) => {
                    drop(stream);
                    anyhow::bail!("RPC call to {peer} timed out after {timeout:?}")
                }
            }
        })
    }

    fn serve_rpc(
        &self,
        handler: RpcHandler,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move {
            *self.rpc_handler.lock().await = Some(handler);
            Ok(())
        })
    }

    fn shutdown(&self) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move {
            // Set the atomic flag first so spawned tasks see it immediately.
            self.shutdown_flag.store(true, Ordering::Release);
            self.shutdown.notify_waiters();
            self.pool.shutdown().await;
            if let Some(handle) = self.accept_handle.lock().await.take() {
                handle.abort();
                let _ = handle.await;
            }
            Ok(())
        })
    }

    fn backend(&self) -> TransportBackend {
        TransportBackend::Tcp
    }

    fn local_addr(&self) -> Result<String> {
        Ok(self.listener_addr.clone())
    }
}

/// Accept-side cluster authentication shared by every inbound connection.
#[derive(Clone)]
struct AcceptAuth {
    auth: Arc<ClusterAuth>,
    local_identity: String,
    /// Bound on connections inside the handshake (#2655 item a).
    pending: PendingHandshakeLimiter,
    /// Per-frame protection suite every accepted connection must use.
    suite: FrameSuite,
}

/// Background loop that accepts incoming TCP connections and dispatches
/// framed messages to the appropriate channel.
///
/// Uses a semaphore to limit concurrent connections (DoS protection). With
/// `accept_auth` set, each connection must pass the cluster handshake inside
/// its own task before `handle_connection` reads a single frame; a failed
/// handshake drops the socket and logs the peer IP (never secret material).
/// While inside the handshake a connection holds a pending-handshake slot
/// instead of a `max_connections` permit; the permit is taken inside the
/// handshake, once the peer is authenticated, and a full cap is answered
/// with the at-capacity verdict. Without `accept_auth` the permit is taken at
/// accept, as before. Every per-connection resource (the permit, the pending
/// slot, the socket) is released by drop, so a connection task that panics
/// (for example inside a registered RPC handler) leaks nothing; there is
/// deliberately no manual connection counter to keep in step.
#[allow(clippy::too_many_arguments)]
async fn accept_loop(
    listener: TcpListener,
    inbound_tx: mpsc::Sender<(String, TransportMessage)>,
    _stream_tx: mpsc::Sender<(String, BoxedAsyncRead)>,
    rpc_handler: Arc<Mutex<Option<RpcHandler>>>,
    max_frame_size: usize,
    max_connections: usize,
    accept_auth: Option<AcceptAuth>,
    shutdown_flag: Arc<AtomicBool>,
    shutdown: Arc<Notify>,
) {
    let conn_semaphore = Arc::new(Semaphore::new(max_connections));

    loop {
        tokio::select! {
            _ = shutdown.notified() => {
                tracing::info!("TCP accept loop shutting down");
                break;
            }
            accept_result = listener.accept() => {
                match accept_result {
                    Ok((mut stream, peer_addr)) => {
                        // Without auth, acquire the connection permit at
                        // accept. With auth, reserve a pending-handshake slot
                        // instead; the permit is taken after authentication.
                        let (permit, pending_slot) = match &accept_auth {
                            None => match conn_semaphore.clone().try_acquire_owned() {
                                Ok(permit) => (Some(permit), None),
                                Err(_) => {
                                    tracing::warn!(
                                        "Rejecting connection from {peer_addr}: max connections ({max_connections}) reached"
                                    );
                                    drop(stream);
                                    continue;
                                }
                            },
                            Some(accept_auth) => match accept_auth.pending.try_acquire(peer_addr.ip()) {
                                Ok(slot) => (None, Some(slot)),
                                Err(refusal) => {
                                    if handshake_warn_allowed(HandshakeWarnClass::Noise) {
                                        tracing::warn!(
                                            "Rejecting unauthenticated connection from {}: {refusal}",
                                            peer_addr.ip()
                                        );
                                    } else {
                                        tracing::debug!(
                                            "Rejecting unauthenticated connection from {}: {refusal}",
                                            peer_addr.ip()
                                        );
                                    }
                                    drop(stream);
                                    continue;
                                }
                            },
                        };

                        let mut peer = peer_addr.to_string();
                        let inbound_tx = inbound_tx.clone();
                        let rpc_handler = rpc_handler.clone();
                        let shutdown_flag = shutdown_flag.clone();
                        let shutdown = shutdown.clone();
                        let accept_auth = accept_auth.clone();
                        let conn_semaphore = conn_semaphore.clone();

                        tokio::spawn(async move {
                            // Held until the task exits: the accept-time
                            // permit without auth, the in-handshake permit
                            // with auth.
                            let (_permit, mut stream) = match accept_auth {
                                None => (permit, ProtectedStream::plain(stream)),
                                Some(accept_auth) => {
                                    // The connection cap is checked inside the
                                    // handshake, after the peer authenticated,
                                    // so a full cap is a typed verdict for the
                                    // peer and a stranger learns nothing.
                                    let outcome = server_handshake_admitting(
                                        &mut stream,
                                        &accept_auth.auth,
                                        &accept_auth.local_identity,
                                        accept_auth.suite,
                                        FrameChannel::Transport,
                                        || conn_semaphore.try_acquire_owned().ok(),
                                    )
                                    .await;
                                    // The pending slot covers the handshake only.
                                    drop(pending_slot);
                                    match outcome {
                                        Ok((identity, keys, acquired)) => {
                                            peer = identity;
                                            (Some(acquired), ProtectedStream::protected(stream, keys))
                                        }
                                        Err(err) => {
                                            if err.downcast_ref::<PeerClosedBeforeHandshake>().is_some() {
                                                // A connect-and-close liveness probe, not an attack.
                                                tracing::debug!(
                                                    "Connection from {} closed before the cluster handshake",
                                                    peer_addr.ip()
                                                );
                                            } else if let Some(full) =
                                                err.downcast_ref::<ConnectionLimitReached>()
                                            {
                                                // Only an authenticated peer
                                                // reaches this line, but a burst
                                                // of refused (and retried)
                                                // connects must not flood the log
                                                // (#2726): its own budget class.
                                                if handshake_warn_allowed(HandshakeWarnClass::AtCapacity) {
                                                    tracing::warn!(
                                                        "Rejecting connection from {}: max connections ({max_connections}) reached",
                                                        full.peer
                                                    );
                                                } else {
                                                    tracing::debug!(
                                                        "Rejecting connection from {}: max connections ({max_connections}) reached",
                                                        full.peer
                                                    );
                                                }
                                            } else if handshake_warn_allowed(handshake_warn_class(&err)) {
                                                tracing::warn!(
                                                    "Rejecting connection from {}: cluster handshake failed: {err}",
                                                    peer_addr.ip()
                                                );
                                            } else {
                                                tracing::debug!(
                                                    "Rejecting connection from {}: cluster handshake failed: {err}",
                                                    peer_addr.ip()
                                                );
                                            }
                                            return;
                                        }
                                    }
                                }
                            };
                            handle_connection(
                                &mut stream,
                                &peer,
                                &inbound_tx,
                                &rpc_handler,
                                max_frame_size,
                                &shutdown_flag,
                                &shutdown,
                            )
                            .await;
                        });
                    }
                    Err(e) => {
                        tracing::error!("TCP accept error: {e}");
                    }
                }
            }
        }
    }
}

/// Handle a single inbound TCP connection, reading frames in a loop.
///
/// The loop ends on the first read error. A frame that fails authentication
/// ([`FrameRejected`]) and an authenticated frame of a kind this build does
/// not know ([`UnknownAuthenticatedKind`], a version-skewed member, #2712)
/// are logged at warn through the handshake rate limiter's diagnostic budget
/// (debug once the window is exhausted), without payload bytes; any other
/// error (peer closed, malformed or oversize header) ends the loop quietly as
/// before.
async fn handle_connection(
    stream: &mut ProtectedStream,
    peer: &str,
    inbound_tx: &mpsc::Sender<(String, TransportMessage)>,
    rpc_handler: &Arc<Mutex<Option<RpcHandler>>>,
    max_frame_size: usize,
    shutdown_flag: &Arc<AtomicBool>,
    shutdown: &Arc<Notify>,
) {
    loop {
        // Check the atomic flag for fast shutdown detection even when
        // not blocked in select.
        if shutdown_flag.load(Ordering::Acquire) {
            break;
        }

        tokio::select! {
            _ = shutdown.notified() => break,
            frame = stream.read_frame(max_frame_size) => {
                match frame {
                    Ok((kind, payload)) => {
                        match kind {
                            MessageKind::TensorData | MessageKind::Control => {
                                if let Ok(msg) = deserialize_message(kind, &payload) {
                                    let _ = inbound_tx.send((peer.to_string(), msg)).await;
                                }
                            }
                            MessageKind::RpcResponse => {
                                // Responses are read by `rpc_call` on the
                                // connection its request went out on; none
                                // is expected on an accepted connection.
                                tracing::debug!(
                                    "Ignoring an RPC response from {peer} on an accepted connection"
                                );
                            }
                            MessageKind::RpcRequest => {
                                // Handle RPC request if a handler is registered.
                                if payload.len() >= 8 {
                                    let req_id = u64::from_be_bytes(
                                        payload[..8].try_into().unwrap(),
                                    );
                                    let request_data = payload[8..].to_vec();
                                    let handler = rpc_handler.lock().await;
                                    if let Some(ref h) = *handler {
                                        let response_data = h(&request_data);
                                        drop(handler); // Release lock before I/O.
                                        // Send the RPC response back on the same
                                        // (protected) connection.
                                        let mut resp_payload = Vec::with_capacity(8 + response_data.len());
                                        resp_payload.extend_from_slice(&req_id.to_be_bytes());
                                        resp_payload.extend_from_slice(&response_data);
                                        if let Err(e) = stream
                                            .write_frame(MessageKind::RpcResponse, &resp_payload)
                                            .await
                                        {
                                            tracing::warn!("Failed to send RPC response to {peer}: {e}");
                                            break;
                                        }
                                    } else {
                                        tracing::debug!(
                                            "Received RPC request from {peer} but no handler registered"
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Err(err) => {
                        if err.downcast_ref::<FrameRejected>().is_some() {
                            if handshake_warn_allowed(HandshakeWarnClass::Diagnostic) {
                                tracing::warn!(
                                    "Closing connection from {peer}: {err}"
                                );
                            } else {
                                tracing::debug!(
                                    "Closing connection from {peer}: {err}"
                                );
                            }
                        } else if err.downcast_ref::<UnknownAuthenticatedKind>().is_some() {
                            if handshake_warn_allowed(HandshakeWarnClass::Diagnostic) {
                                tracing::warn!(
                                    "Closing connection from {peer}: {err} (version skew?)"
                                );
                            } else {
                                tracing::debug!(
                                    "Closing connection from {peer}: {err} (version skew?)"
                                );
                            }
                        }
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "tcp_transport_tests.rs"]
mod tests;
