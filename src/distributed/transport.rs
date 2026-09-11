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

//! Abstract transport layer for inter-node communication.
//!
//! Defines the [`Transport`] trait and message types used by both TCP and
//! Thunderbolt backends. Two communication patterns are supported:
//!
//! - **Streaming**: Large data transfers (tensors, KV caches, activations)
//! - **RPC**: Lightweight request/response control messages (heartbeats,
//!   scheduling commands, metadata exchanges)

use std::fmt;
use std::pin::Pin;

use anyhow::{Result, bail};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncRead;

/// Backend discriminator for transport implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TransportBackend {
    /// Standard TCP/IP networking.
    #[default]
    Tcp,
    /// Thunderbolt (IP-over-Thunderbolt Bridge) for local clusters.
    Thunderbolt,
    /// RDMA-aware zero-copy transport.
    ///
    /// Uses `io_uring` registered buffers on Linux, `kqueue` batched send with
    /// registered memory regions on macOS, and falls back transparently to TCP
    /// when the zero-copy primitive is unavailable or negotiation fails. A
    /// single-line log entry at coordinator startup names the reason when the
    /// fallback is taken (OS, driver, peer version, or capability mismatch).
    Rdma,
}

impl fmt::Display for TransportBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tcp => write!(f, "tcp"),
            Self::Thunderbolt => write!(f, "thunderbolt"),
            Self::Rdma => write!(f, "rdma"),
        }
    }
}

impl std::str::FromStr for TransportBackend {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "tcp" => Ok(Self::Tcp),
            "thunderbolt" => Ok(Self::Thunderbolt),
            "rdma" => Ok(Self::Rdma),
            other => {
                bail!(
                    "unknown transport backend '{other}'; \
                     expected one of: tcp, thunderbolt, rdma"
                )
            }
        }
    }
}

/// Message types that can be sent over the transport layer.
///
/// Each variant carries a purpose-specific payload. The framing protocol
/// prepends a header with the message kind and payload length so receivers
/// can dispatch without parsing the full body upfront.
#[derive(Debug, Clone)]
pub enum TransportMessage {
    /// Raw tensor data (activations, KV cache slices, weight shards).
    /// Uses `Bytes` for zero-copy reference counting.
    TensorData {
        /// Identifier for the tensor (e.g., layer name or shard ID).
        tensor_id: String,
        /// Tensor shape dimensions.
        shape: Vec<usize>,
        /// Raw bytes of the tensor payload.
        data: Bytes,
    },
    /// Lightweight control / RPC message (heartbeats, scheduling).
    Control {
        /// Application-defined operation tag (e.g., "heartbeat", "schedule").
        operation: String,
        /// JSON or binary payload.
        payload: Bytes,
    },
}

impl TransportMessage {
    /// Total byte size of the payload (not including framing overhead).
    pub fn payload_size(&self) -> usize {
        match self {
            Self::TensorData { data, .. } => data.len(),
            Self::Control { payload, .. } => payload.len(),
        }
    }
}

/// Wire-level message kind tag used in the framing protocol.
///
/// The first byte of every framed message is one of these discriminants.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    TensorData = 1,
    Control = 2,
    RpcRequest = 3,
    RpcResponse = 4,
}

impl TryFrom<u8> for MessageKind {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::TensorData),
            2 => Ok(Self::Control),
            3 => Ok(Self::RpcRequest),
            4 => Ok(Self::RpcResponse),
            other => anyhow::bail!("unknown message kind: {other}"),
        }
    }
}

/// Boxed async reader for streaming transfers.
pub type BoxedAsyncRead = Pin<Box<dyn AsyncRead + Send>>;

/// Async handler function type for RPC requests.
///
/// Receives the raw request bytes and returns the raw response bytes.
pub type RpcHandler = Box<dyn Fn(&[u8]) -> Vec<u8> + Send + Sync>;

/// Abstract transport interface for inter-node communication.
///
/// Implementations must be safe to share across tasks (`Send + Sync`) and
/// operate asynchronously on the tokio runtime.
///
/// # Communication Patterns
///
/// - **Message send/recv**: Fire-and-forget delivery of [`TransportMessage`]
///   values. The transport guarantees ordering per peer but not global ordering.
///
/// - **Streaming**: Large payloads transferred as an [`AsyncRead`] stream,
///   useful for tensor data that should not be buffered entirely in memory.
///
/// - **RPC**: Request/response pairs with a single roundtrip. Used for control
///   messages that need an acknowledgement or return value.
pub trait Transport: Send + Sync {
    /// Establish connections to the listed peer addresses.
    ///
    /// Addresses use the `host:port` format. The implementation should set up
    /// connection pooling and begin any necessary handshake.
    fn connect(
        &self,
        peers: &[String],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>>;

    /// Send a message to a specific peer.
    fn send(
        &self,
        peer: &str,
        message: TransportMessage,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>>;

    /// Receive the next inbound message, returning the sender address and the
    /// message.
    fn recv(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(String, TransportMessage)>> + Send + '_>>;

    /// Begin streaming data to a peer. The caller writes into the returned
    /// sink; the transport frames and sends the bytes incrementally.
    fn send_stream(
        &self,
        peer: &str,
        data: Bytes,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>>;

    /// Receive the next inbound stream, returning the sender address and a
    /// reader for the payload.
    fn recv_stream(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(String, BoxedAsyncRead)>> + Send + '_>>;

    /// Perform an RPC call: send `request` to `peer` and wait for the response.
    fn rpc_call(
        &self,
        peer: &str,
        request: &[u8],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>>> + Send + '_>>;

    /// Start serving RPC requests using the given handler.
    ///
    /// This is typically called once during initialization. The handler runs
    /// for each incoming RPC request and returns the response bytes.
    fn serve_rpc(
        &self,
        handler: RpcHandler,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>>;

    /// Gracefully shut down all connections and background tasks.
    fn shutdown(&self) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>>;

    /// Return the backend type of this transport.
    fn backend(&self) -> TransportBackend;

    /// Return the local listening address in `host:port` form.
    fn local_addr(&self) -> Result<String>;
}

#[cfg(test)]
#[path = "transport_tests.rs"]
mod tests;
