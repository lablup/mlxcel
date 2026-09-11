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
//! This transport does **not** perform authentication or encryption.
//! In production deployments, callers should either:
//! - Bind only to trusted network interfaces (e.g., `127.0.0.1` or a
//!   private VLAN)
//! - Layer TLS on top using a wrapper transport
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

use super::connection_pool::{ConnectionPool, PoolConfig};
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
    /// Maximum concurrent inbound connections (DoS protection).
    pub max_connections: usize,
}

impl Default for TcpTransportConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:0".to_string(),
            pool_config: PoolConfig::default(),
            max_frame_size: 256 * 1024 * 1024,
            max_connections: DEFAULT_MAX_CONNECTIONS,
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

        let pool = ConnectionPool::new(config.pool_config.clone());
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

        let handle = tokio::spawn(accept_loop(
            listener,
            inbound_tx_clone,
            stream_tx_clone,
            rpc_pending_clone,
            rpc_handler,
            max_frame,
            max_conns,
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
            for peer in &peers {
                match self.pool.acquire(peer).await {
                    Ok(stream) => {
                        self.pool.release(peer, stream).await;
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

/// Background loop that accepts incoming TCP connections and dispatches
/// framed messages to the appropriate channel.
///
/// Uses a semaphore to limit concurrent connections (DoS protection).
async fn accept_loop(
    listener: TcpListener,
    inbound_tx: mpsc::Sender<(String, TransportMessage)>,
    _stream_tx: mpsc::Sender<(String, BoxedAsyncRead)>,
    rpc_pending: Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<Vec<u8>>>>>,
    rpc_handler: Arc<Mutex<Option<RpcHandler>>>,
    max_frame_size: usize,
    max_connections: usize,
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
                        // Acquire a permit to limit concurrent connections.
                        let permit = match conn_semaphore.clone().try_acquire_owned() {
                            Ok(permit) => permit,
                            Err(_) => {
                                tracing::warn!(
                                    "Rejecting connection from {peer_addr}: max connections ({max_connections}) reached"
                                );
                                drop(stream);
                                continue;
                            }
                        };

                        let peer = peer_addr.to_string();
                        let inbound_tx = inbound_tx.clone();
                        let rpc_pending = rpc_pending.clone();
                        let rpc_handler = rpc_handler.clone();
                        let shutdown_flag = shutdown_flag.clone();
                        let shutdown = shutdown.clone();
                        let active = active_conns.clone();
                        active.fetch_add(1, Ordering::Relaxed);

                        tokio::spawn(async move {
                            let _permit = permit; // Held until task exits.
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

#[cfg(test)]
#[path = "tcp_transport_tests.rs"]
mod tests;
