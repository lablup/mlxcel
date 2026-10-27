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
//! length-prefixed framing protocol. The framing format is:
//!
//! ```text
//! [kind: u8][payload_len: u64 BE][payload: bytes]
//! ```
//!
//! For RPC, the `kind` distinguishes requests from responses and a 64-bit
//! request ID is prepended to the payload to correlate responses.
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
//! therefore delay other unauthenticated peers, but cannot starve an
//! authenticated stage of a connection slot (#2655). A failed handshake is
//! logged with its class (closed before the handshake at debug; identity not
//! owned, probable secret mismatch, altered transcript at warn, rate
//! limited).
//!
//! Without `auth` (the default, used by single-node serving) the transport
//! performs no authentication, and the reported `from` is the raw socket
//! peer address. Every distributed role, including the disaggregated prefill,
//! decode and router roles, sets `auth` (#2650, #2655).
//!
//! The handshake authenticates connections; it does **not** encrypt or MAC
//! individual frames. An on-path attacker who can hijack an established TCP
//! stream is out of scope (TLS or a per-frame MAC is the follow-up). In every
//! deployment, also:
//! - Bind only to trusted network interfaces (e.g., `127.0.0.1` or a
//!   private VLAN)
//! - Use firewall rules to restrict access to the listening port

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Context, Result};
use bytes::{Bytes, BytesMut};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, Notify, Semaphore, mpsc};

use super::cluster_auth::ClusterAuth;
use super::cluster_handshake::{HandshakeRejected, PeerClosedBeforeHandshake, server_handshake};
use super::connection_pool::{ConnectionPool, PoolConfig};
use super::pending_handshakes::PendingHandshakeLimiter;
pub use super::pending_handshakes::{
    DEFAULT_MAX_PENDING_HANDSHAKES, DEFAULT_MAX_PENDING_HANDSHAKES_PER_IP,
};
use super::transport::{
    BoxedAsyncRead, MessageKind, RpcHandler, Transport, TransportBackend, TransportMessage,
};

/// Default maximum number of concurrent inbound connections.
const DEFAULT_MAX_CONNECTIONS: usize = 512;

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
    /// set, only authenticated connections count against it.
    pub max_connections: usize,
    /// With `auth` set: maximum inbound connections that may be inside the
    /// cluster handshake at once, across all sources (default
    /// [`DEFAULT_MAX_PENDING_HANDSHAKES`]). Sockets beyond it are closed at
    /// accept. Ignored without `auth`.
    pub max_pending_handshakes: usize,
    /// With `auth` set: maximum inbound connections inside the handshake from
    /// one source IP (default [`DEFAULT_MAX_PENDING_HANDSHAKES_PER_IP`]).
    /// IPv4-mapped IPv6 sources count against their IPv4 address. Ignored
    /// without `auth`.
    pub max_pending_handshakes_per_ip: usize,
    /// Cluster authentication. When `Some`, both inbound and outbound
    /// connections must complete the mutual cluster handshake before any
    /// frame is exchanged (see the module-level Security Notes). `None`
    /// keeps the unauthenticated behavior for single-node defaults.
    pub auth: Option<Arc<ClusterAuth>>,
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
    /// Pending RPC responses keyed by request ID.
    rpc_pending: Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<Vec<u8>>>>>,
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
        let listener = TcpListener::bind(&config.bind_address)
            .await
            .with_context(|| format!("failed to bind TCP listener on {}", config.bind_address))?;
        let local_addr = listener
            .local_addr()
            .context("failed to get local address")?
            .to_string();

        let pool =
            ConnectionPool::with_auth(config.pool_config.clone(), config.auth.clone(), &local_addr);
        let (inbound_tx, inbound_rx) = mpsc::channel(256);
        let (stream_tx, stream_rx) = mpsc::channel(64);
        let rpc_pending: Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<Vec<u8>>>>> =
            Arc::new(Mutex::new(HashMap::new()));
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
            rpc_pending: rpc_pending.clone(),
            rpc_next_id: Mutex::new(0),
            rpc_handler: rpc_handler.clone(),
            shutdown_flag: shutdown_flag.clone(),
            shutdown: shutdown.clone(),
            accept_handle: Mutex::new(None),
        };

        // Start the accept loop.
        let inbound_tx_clone = transport.inbound_tx.clone();
        let stream_tx_clone = transport.stream_tx.clone();
        let rpc_pending_clone = rpc_pending;
        let max_frame = config.max_frame_size;
        let max_conns = config.max_connections;
        let accept_auth = config.auth.clone().map(|auth| AcceptAuth {
            auth,
            local_identity: transport.listener_addr.clone(),
            pending: PendingHandshakeLimiter::new(
                config.max_pending_handshakes,
                config.max_pending_handshakes_per_ip,
            ),
        });

        let handle = tokio::spawn(accept_loop(
            listener,
            inbound_tx_clone,
            stream_tx_clone,
            rpc_pending_clone,
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

    /// Write a framed message to a TCP stream using vectored I/O to reduce
    /// syscall overhead.
    async fn write_frame(
        stream: &mut tokio::net::TcpStream,
        kind: MessageKind,
        payload: &[u8],
    ) -> Result<()> {
        let header = [kind as u8];
        let len_bytes = (payload.len() as u64).to_be_bytes();

        // Build a contiguous buffer for the frame to avoid multiple
        // write syscalls. For large payloads the copy cost is negligible
        // compared to the network I/O.
        let total = 1 + 8 + payload.len();
        let mut buf = Vec::with_capacity(total);
        buf.push(header[0]);
        buf.extend_from_slice(&len_bytes);
        buf.extend_from_slice(payload);
        stream.write_all(&buf).await?;
        stream.flush().await?;

        Ok(())
    }

    /// Serialize a [`TransportMessage`] to wire bytes.
    fn serialize_message(msg: &TransportMessage) -> (MessageKind, Vec<u8>) {
        match msg {
            TransportMessage::TensorData {
                tensor_id,
                shape,
                data,
            } => {
                // Format: [tensor_id_len: u32][tensor_id][shape_dims: u32][shape...][data]
                let id_bytes = tensor_id.as_bytes();
                let mut buf =
                    Vec::with_capacity(4 + id_bytes.len() + 4 + shape.len() * 8 + data.len());
                buf.extend_from_slice(&(id_bytes.len() as u32).to_be_bytes());
                buf.extend_from_slice(id_bytes);
                buf.extend_from_slice(&(shape.len() as u32).to_be_bytes());
                for &dim in shape {
                    buf.extend_from_slice(&(dim as u64).to_be_bytes());
                }
                buf.extend_from_slice(data);
                (MessageKind::TensorData, buf)
            }
            TransportMessage::Control { operation, payload } => {
                let op_bytes = operation.as_bytes();
                let mut buf = Vec::with_capacity(4 + op_bytes.len() + payload.len());
                buf.extend_from_slice(&(op_bytes.len() as u32).to_be_bytes());
                buf.extend_from_slice(op_bytes);
                buf.extend_from_slice(payload);
                (MessageKind::Control, buf)
            }
        }
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
            let (kind, payload) = Self::serialize_message(&message);
            let mut stream = self.pool.acquire(&peer).await?;
            Self::write_frame(&mut stream, kind, &payload).await?;
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
            // Use TensorData kind with a synthetic header for streaming.
            let header = [MessageKind::TensorData as u8];
            let len_bytes = (data.len() as u64).to_be_bytes();
            let total = 1 + 8 + data.len();
            let mut buf = Vec::with_capacity(total);
            buf.push(header[0]);
            buf.extend_from_slice(&len_bytes);
            buf.extend_from_slice(&data);
            stream.write_all(&buf).await?;
            stream.flush().await?;
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

            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            self.rpc_pending.lock().await.insert(request_id, resp_tx);

            // Frame: [RpcRequest kind][len][request_id: u64 BE][request bytes]
            let mut payload = Vec::with_capacity(8 + request.len());
            payload.extend_from_slice(&request_id.to_be_bytes());
            payload.extend_from_slice(&request);

            let mut stream = self.pool.acquire(&peer).await?;
            Self::write_frame(&mut stream, MessageKind::RpcRequest, &payload).await?;
            self.pool.release(&peer, stream).await;

            let result =
                tokio::time::timeout(self.config.pool_config.connect_timeout, resp_rx).await;

            // Clean up the pending entry on timeout or channel drop to
            // prevent unbounded growth of the rpc_pending map.
            match result {
                Ok(Ok(response)) => Ok(response),
                Ok(Err(_)) => {
                    self.rpc_pending.lock().await.remove(&request_id);
                    anyhow::bail!("RPC response channel dropped")
                }
                Err(_) => {
                    self.rpc_pending.lock().await.remove(&request_id);
                    anyhow::bail!("RPC call timed out")
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

/// Read a single framed message from a TCP stream.
///
/// Returns `(kind, payload)` or an error if the stream is closed or the
/// frame is malformed.
async fn read_frame(
    stream: &mut tokio::net::TcpStream,
    max_frame_size: usize,
) -> Result<(MessageKind, BytesMut)> {
    let mut header = [0u8; 1];
    stream.read_exact(&mut header).await.context("read kind")?;
    let kind = MessageKind::try_from(header[0])?;

    let mut len_buf = [0u8; 8];
    stream
        .read_exact(&mut len_buf)
        .await
        .context("read length")?;
    let len = u64::from_be_bytes(len_buf) as usize;

    if len > max_frame_size {
        anyhow::bail!("frame size {len} exceeds maximum {max_frame_size}");
    }

    let mut payload = BytesMut::zeroed(len);
    stream
        .read_exact(&mut payload)
        .await
        .context("read payload")?;

    Ok((kind, payload))
}

/// Deserialize a [`TransportMessage`] from a raw payload and its kind tag.
fn deserialize_message(kind: MessageKind, payload: &[u8]) -> Result<TransportMessage> {
    match kind {
        MessageKind::TensorData => {
            if payload.len() < 4 {
                anyhow::bail!("tensor message too short for id length");
            }
            let id_len = u32::from_be_bytes(payload[..4].try_into()?) as usize;
            let pos = 4;
            if payload.len() < pos + id_len + 4 {
                anyhow::bail!("tensor message too short for id + shape length");
            }
            let tensor_id = String::from_utf8(payload[pos..pos + id_len].to_vec())
                .context("invalid tensor id")?;
            let pos = pos + id_len;
            let ndims = u32::from_be_bytes(payload[pos..pos + 4].try_into()?) as usize;
            let pos = pos + 4;
            if payload.len() < pos + ndims * 8 {
                anyhow::bail!("tensor message too short for shape");
            }
            let mut shape = Vec::with_capacity(ndims);
            for i in 0..ndims {
                let offset = pos + i * 8;
                let dim = u64::from_be_bytes(payload[offset..offset + 8].try_into()?) as usize;
                shape.push(dim);
            }
            let data_start = pos + ndims * 8;
            let data = Bytes::copy_from_slice(&payload[data_start..]);
            Ok(TransportMessage::TensorData {
                tensor_id,
                shape,
                data,
            })
        }
        MessageKind::Control => {
            if payload.len() < 4 {
                anyhow::bail!("control message too short for operation length");
            }
            let op_len = u32::from_be_bytes(payload[..4].try_into()?) as usize;
            if payload.len() < 4 + op_len {
                anyhow::bail!("control message too short for operation");
            }
            let operation =
                String::from_utf8(payload[4..4 + op_len].to_vec()).context("invalid operation")?;
            let data = Bytes::copy_from_slice(&payload[4 + op_len..]);
            Ok(TransportMessage::Control {
                operation,
                payload: data,
            })
        }
        _ => anyhow::bail!("unexpected message kind for deserialization: {kind:?}"),
    }
}

/// Accept-side cluster authentication shared by every inbound connection.
#[derive(Clone)]
struct AcceptAuth {
    auth: Arc<ClusterAuth>,
    local_identity: String,
    /// Bound on connections inside the handshake (#2655 item a).
    pending: PendingHandshakeLimiter,
}

/// Background loop that accepts incoming TCP connections and dispatches
/// framed messages to the appropriate channel.
///
/// Uses a semaphore to limit concurrent connections (DoS protection). With
/// `accept_auth` set, each connection must pass the cluster handshake inside
/// its own task before `handle_connection` reads a single frame; a failed
/// handshake drops the socket and logs the peer IP (never secret material).
/// While inside the handshake a connection holds a pending-handshake slot
/// instead of a `max_connections` permit; the permit is taken only once the
/// peer is authenticated. Without `accept_auth` the permit is taken at
/// accept, as before.
#[allow(clippy::too_many_arguments)]
async fn accept_loop(
    listener: TcpListener,
    inbound_tx: mpsc::Sender<(String, TransportMessage)>,
    _stream_tx: mpsc::Sender<(String, BoxedAsyncRead)>,
    rpc_pending: Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<Vec<u8>>>>>,
    rpc_handler: Arc<Mutex<Option<RpcHandler>>>,
    max_frame_size: usize,
    max_connections: usize,
    accept_auth: Option<AcceptAuth>,
    shutdown_flag: Arc<AtomicBool>,
    shutdown: Arc<Notify>,
) {
    let conn_semaphore = Arc::new(Semaphore::new(max_connections));
    let active_conns = Arc::new(AtomicUsize::new(0));

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
                                    if handshake_warn_allowed() {
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
                        let rpc_pending = rpc_pending.clone();
                        let rpc_handler = rpc_handler.clone();
                        let shutdown_flag = shutdown_flag.clone();
                        let shutdown = shutdown.clone();
                        let active = active_conns.clone();
                        let accept_auth = accept_auth.clone();
                        let conn_semaphore = conn_semaphore.clone();
                        active.fetch_add(1, Ordering::Relaxed);

                        tokio::spawn(async move {
                            // Held until the task exits: the accept-time
                            // permit without auth, the post-handshake permit
                            // with auth.
                            let _permit = match accept_auth {
                                None => permit,
                                Some(accept_auth) => {
                                    let outcome = server_handshake(
                                        &mut stream,
                                        &accept_auth.auth,
                                        &accept_auth.local_identity,
                                    )
                                    .await;
                                    // The pending slot covers the handshake only.
                                    drop(pending_slot);
                                    match outcome {
                                        Ok(identity) => peer = identity,
                                        Err(err) => {
                                            if err.downcast_ref::<PeerClosedBeforeHandshake>().is_some() {
                                                // A connect-and-close liveness probe, not an attack.
                                                tracing::debug!(
                                                    "Connection from {} closed before the cluster handshake",
                                                    peer_addr.ip()
                                                );
                                            } else if handshake_warn_allowed() {
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
                                            active.fetch_sub(1, Ordering::Relaxed);
                                            return;
                                        }
                                    }
                                    // Authenticated: only now does it count
                                    // against the connection cap.
                                    match conn_semaphore.try_acquire_owned() {
                                        Ok(acquired) => Some(acquired),
                                        Err(_) => {
                                            tracing::warn!(
                                                "Rejecting connection from {peer}: max connections ({max_connections}) reached"
                                            );
                                            active.fetch_sub(1, Ordering::Relaxed);
                                            return;
                                        }
                                    }
                                }
                            };
                            handle_connection(
                                &mut stream,
                                &peer,
                                &inbound_tx,
                                &rpc_pending,
                                &rpc_handler,
                                max_frame_size,
                                &shutdown_flag,
                                &shutdown,
                            )
                            .await;
                            active.fetch_sub(1, Ordering::Relaxed);
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
async fn handle_connection(
    stream: &mut tokio::net::TcpStream,
    peer: &str,
    inbound_tx: &mpsc::Sender<(String, TransportMessage)>,
    rpc_pending: &Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<Vec<u8>>>>>,
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
            frame = read_frame(stream, max_frame_size) => {
                match frame {
                    Ok((kind, payload)) => {
                        match kind {
                            MessageKind::TensorData | MessageKind::Control => {
                                if let Ok(msg) = deserialize_message(kind, &payload) {
                                    let _ = inbound_tx.send((peer.to_string(), msg)).await;
                                }
                            }
                            MessageKind::RpcResponse => {
                                if payload.len() >= 8 {
                                    let req_id = u64::from_be_bytes(
                                        payload[..8].try_into().unwrap(),
                                    );
                                    let data = payload[8..].to_vec();
                                    if let Some(tx) = rpc_pending.lock().await.remove(&req_id) {
                                        let _ = tx.send(data);
                                    }
                                }
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
                                        // Send RPC response back.
                                        let mut resp_payload = Vec::with_capacity(8 + response_data.len());
                                        resp_payload.extend_from_slice(&req_id.to_be_bytes());
                                        resp_payload.extend_from_slice(&response_data);
                                        let header = [MessageKind::RpcResponse as u8];
                                        let len_bytes = (resp_payload.len() as u64).to_be_bytes();
                                        let total = 1 + 8 + resp_payload.len();
                                        let mut buf = Vec::with_capacity(total);
                                        buf.push(header[0]);
                                        buf.extend_from_slice(&len_bytes);
                                        buf.extend_from_slice(&resp_payload);
                                        if let Err(e) = stream.write_all(&buf).await {
                                            tracing::warn!("Failed to send RPC response to {peer}: {e}");
                                            break;
                                        }
                                        if let Err(e) = stream.flush().await {
                                            tracing::warn!("Failed to flush RPC response to {peer}: {e}");
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
                    Err(_) => break,
                }
            }
        }
    }
}

/// Failed inbound handshakes logged at warn per one-minute window. Beyond
/// this the rejection is logged at debug so an unauthenticated peer cannot
/// flood the log.
const HANDSHAKE_WARN_PER_WINDOW: u32 = 5;
const HANDSHAKE_WARN_WINDOW_SECS: u64 = 60;

/// Process-wide limiter for the failed-handshake warning.
fn handshake_warn_allowed() -> bool {
    use std::sync::Mutex;
    use std::time::Instant;
    static STATE: Mutex<Option<(Instant, u32)>> = Mutex::new(None);
    let now = Instant::now();
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    match guard.as_mut() {
        Some((start, count))
            if now.duration_since(*start).as_secs() < HANDSHAKE_WARN_WINDOW_SECS =>
        {
            *count = count.saturating_add(1);
            *count <= HANDSHAKE_WARN_PER_WINDOW
        }
        _ => {
            *guard = Some((now, 1));
            true
        }
    }
}

#[cfg(test)]
#[path = "tcp_transport_tests.rs"]
mod tests;
