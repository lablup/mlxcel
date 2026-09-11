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

//! In-process mock transport for testing distributed logic without network.
//!
//! [`MockTransport`] implements the [`Transport`] trait using tokio channels
//! for message passing between simulated nodes. It supports:
//!
//! - Configurable latency simulation
//! - Message delivery control (drop, delay)
//! - Connection failure injection
//! - Per-peer message ordering guarantees
//!
//! Used by: TestCluster (test_harness.rs), integration tests

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Result, bail};
use bytes::Bytes;
use tokio::sync::{Mutex, RwLock, mpsc};

use super::transport::{BoxedAsyncRead, RpcHandler, Transport, TransportBackend, TransportMessage};

/// Configuration for mock transport behavior.
#[derive(Debug, Clone)]
pub struct MockTransportConfig {
    /// Simulated latency added to every send/recv operation.
    pub latency: Duration,
    /// Channel buffer size for message passing between nodes.
    pub channel_buffer: usize,
    /// Whether sends to disconnected peers should return errors.
    pub fail_on_disconnected: bool,
}

impl Default for MockTransportConfig {
    fn default() -> Self {
        Self {
            latency: Duration::ZERO,
            channel_buffer: 256,
            fail_on_disconnected: true,
        }
    }
}

/// Shared state for routing messages between MockTransport instances.
///
/// The router acts as a virtual network switch: each node registers its
/// receive channel, and sends are routed to the target node's channel.
#[derive(Clone)]
pub struct MockRouter {
    inner: Arc<RwLock<RouterInner>>,
}

struct RouterInner {
    /// Map from node address to its inbound message sender.
    senders: HashMap<String, mpsc::Sender<(String, TransportMessage)>>,
    /// Map from node address to its RPC handler.
    rpc_handlers: HashMap<String, Arc<Mutex<Option<RpcHandler>>>>,
    /// Set of addresses that are "partitioned" (cannot send/receive).
    partitioned: std::collections::HashSet<String>,
}

impl MockRouter {
    /// Create a new empty router.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RouterInner {
                senders: HashMap::new(),
                rpc_handlers: HashMap::new(),
                partitioned: std::collections::HashSet::new(),
            })),
        }
    }

    /// Register a node's receive channel with the router.
    async fn register(
        &self,
        addr: &str,
        sender: mpsc::Sender<(String, TransportMessage)>,
        rpc_slot: Arc<Mutex<Option<RpcHandler>>>,
    ) {
        let mut inner = self.inner.write().await;
        inner.senders.insert(addr.to_string(), sender);
        inner.rpc_handlers.insert(addr.to_string(), rpc_slot);
    }

    /// Route a message from `from_addr` to `to_addr`.
    async fn route(&self, from_addr: &str, to_addr: &str, msg: TransportMessage) -> Result<()> {
        let inner = self.inner.read().await;

        // Check partitions.
        if inner.partitioned.contains(from_addr) {
            bail!("node {from_addr} is partitioned");
        }
        if inner.partitioned.contains(to_addr) {
            bail!("node {to_addr} is partitioned");
        }

        let sender = inner
            .senders
            .get(to_addr)
            .ok_or_else(|| anyhow::anyhow!("peer {to_addr} not registered in mock router"))?;

        sender
            .send((from_addr.to_string(), msg))
            .await
            .map_err(|_| anyhow::anyhow!("peer {to_addr} channel closed"))?;

        Ok(())
    }

    /// Perform an RPC call from `from_addr` to `to_addr`.
    async fn rpc_route(&self, from_addr: &str, to_addr: &str, request: &[u8]) -> Result<Vec<u8>> {
        let inner = self.inner.read().await;

        if inner.partitioned.contains(from_addr) {
            bail!("node {from_addr} is partitioned");
        }
        if inner.partitioned.contains(to_addr) {
            bail!("node {to_addr} is partitioned");
        }

        let handler_slot = inner
            .rpc_handlers
            .get(to_addr)
            .ok_or_else(|| anyhow::anyhow!("peer {to_addr} not registered for RPC"))?;

        let handler = handler_slot.lock().await;
        let handler_fn = handler
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("peer {to_addr} has no RPC handler installed"))?;

        Ok(handler_fn(request))
    }

    /// Simulate a network partition: isolate a node so it cannot send or receive.
    pub async fn partition_node(&self, addr: &str) {
        let mut inner = self.inner.write().await;
        inner.partitioned.insert(addr.to_string());
    }

    /// Heal a network partition for a node.
    pub async fn heal_node(&self, addr: &str) {
        let mut inner = self.inner.write().await;
        inner.partitioned.remove(addr);
    }

    /// Check if a node is currently partitioned.
    pub async fn is_partitioned(&self, addr: &str) -> bool {
        let inner = self.inner.read().await;
        inner.partitioned.contains(addr)
    }
}

impl Default for MockRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// In-process mock transport for testing distributed logic without network.
///
/// Each MockTransport instance represents a single node. Messages are
/// delivered through a shared [`MockRouter`] using tokio channels.
pub struct MockTransport {
    /// This node's virtual address.
    local_addr: String,
    /// Shared router for message delivery.
    router: MockRouter,
    /// Configuration (latency, buffer size, etc.).
    config: MockTransportConfig,
    /// Inbound message receiver (wrapped in Mutex for &self recv).
    inbound_rx: Mutex<mpsc::Receiver<(String, TransportMessage)>>,
    /// RPC handler slot.
    rpc_handler: Arc<Mutex<Option<RpcHandler>>>,
    /// Whether this transport has been shut down.
    shutdown: AtomicBool,
    /// Counter for messages sent (useful for test assertions).
    pub messages_sent: AtomicU64,
    /// Counter for messages received.
    pub messages_received: AtomicU64,
}

impl MockTransport {
    /// Create a new mock transport with the given address and router.
    ///
    /// The transport must be registered with the router before it can
    /// send or receive messages (done automatically via `connect`).
    pub async fn new(local_addr: String, router: MockRouter, config: MockTransportConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.channel_buffer);
        let rpc_handler = Arc::new(Mutex::new(None));

        router.register(&local_addr, tx, rpc_handler.clone()).await;

        Self {
            local_addr,
            router,
            config,
            inbound_rx: Mutex::new(rx),
            rpc_handler,
            shutdown: AtomicBool::new(false),
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
        }
    }

    /// Check if this transport has been shut down.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Return the number of messages sent through this transport.
    pub fn sent_count(&self) -> u64 {
        self.messages_sent.load(Ordering::Relaxed)
    }

    /// Return the number of messages received through this transport.
    pub fn received_count(&self) -> u64 {
        self.messages_received.load(Ordering::Relaxed)
    }
}

impl Transport for MockTransport {
    fn connect(
        &self,
        _peers: &[String],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move {
            if self.shutdown.load(Ordering::Relaxed) {
                bail!("transport is shut down");
            }
            // In mock transport, peers are already connected via the shared router.
            // Connection is a no-op.
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
            if self.shutdown.load(Ordering::Relaxed) {
                bail!("transport is shut down");
            }

            // Simulate latency.
            if !self.config.latency.is_zero() {
                tokio::time::sleep(self.config.latency).await;
            }

            self.router.route(&self.local_addr, &peer, message).await?;
            self.messages_sent.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
    }

    fn recv(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(String, TransportMessage)>> + Send + '_>>
    {
        Box::pin(async move {
            if self.shutdown.load(Ordering::Relaxed) {
                bail!("transport is shut down");
            }

            let mut rx = self.inbound_rx.lock().await;
            let result = rx
                .recv()
                .await
                .ok_or_else(|| anyhow::anyhow!("all senders dropped"))?;
            self.messages_received.fetch_add(1, Ordering::Relaxed);
            Ok(result)
        })
    }

    fn send_stream(
        &self,
        peer: &str,
        data: Bytes,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        // Implement stream as a TensorData message for simplicity.
        let peer = peer.to_string();
        Box::pin(async move {
            if self.shutdown.load(Ordering::Relaxed) {
                bail!("transport is shut down");
            }

            if !self.config.latency.is_zero() {
                tokio::time::sleep(self.config.latency).await;
            }

            let msg = TransportMessage::TensorData {
                tensor_id: "__stream__".to_string(),
                shape: vec![data.len()],
                data,
            };
            self.router.route(&self.local_addr, &peer, msg).await?;
            self.messages_sent.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
    }

    fn recv_stream(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(String, BoxedAsyncRead)>> + Send + '_>>
    {
        Box::pin(async move {
            if self.shutdown.load(Ordering::Relaxed) {
                bail!("transport is shut down");
            }

            let mut rx = self.inbound_rx.lock().await;
            let (sender, msg) = rx
                .recv()
                .await
                .ok_or_else(|| anyhow::anyhow!("all senders dropped"))?;
            self.messages_received.fetch_add(1, Ordering::Relaxed);

            let data = match msg {
                TransportMessage::TensorData { data, .. } => data,
                TransportMessage::Control { payload, .. } => payload,
            };

            let reader: BoxedAsyncRead = Box::pin(std::io::Cursor::new(data.to_vec()));
            Ok((sender, reader))
        })
    }

    fn rpc_call(
        &self,
        peer: &str,
        request: &[u8],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>>> + Send + '_>> {
        let peer = peer.to_string();
        let req = request.to_vec();
        Box::pin(async move {
            if self.shutdown.load(Ordering::Relaxed) {
                bail!("transport is shut down");
            }

            if !self.config.latency.is_zero() {
                tokio::time::sleep(self.config.latency).await;
            }

            self.router.rpc_route(&self.local_addr, &peer, &req).await
        })
    }

    fn serve_rpc(
        &self,
        handler: RpcHandler,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move {
            let mut slot = self.rpc_handler.lock().await;
            *slot = Some(handler);
            Ok(())
        })
    }

    fn shutdown(&self) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move {
            self.shutdown.store(true, Ordering::Relaxed);
            Ok(())
        })
    }

    fn backend(&self) -> TransportBackend {
        // Return Tcp as a stand-in; the mock is backend-agnostic.
        TransportBackend::Tcp
    }

    fn local_addr(&self) -> Result<String> {
        Ok(self.local_addr.clone())
    }
}

#[cfg(test)]
#[path = "mock_transport_tests.rs"]
mod tests;
