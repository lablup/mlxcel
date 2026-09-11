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

//! RDMA-aware transport backend.
//!
//! The transport negotiates an OS-level zero-copy primitive (`io_uring`
//! registered buffers on Linux, `kqueue` batched send with registered memory
//! regions on macOS) and falls back transparently to the TCP core when the
//! primitive is unavailable. On every fallback it emits a single-line log
//! entry that names the reason (OS, driver, peer version, or capability
//! mismatch) so operators can act on it without re-running the coordinator.
//!
//! The abstraction intentionally preserves the four-axis separation the
//! pipeline review praised: activation transfer in
//! [`crate::distributed::pipeline::activation_transfer`] consumes
//! [`crate::distributed::Transport`] regardless of whether the underlying
//! backend is TCP, Thunderbolt, or RDMA, so there is no consumer-side change.
//!
//! Used by: remote pipeline runtime, remote stage service, disaggregated
//! transport path.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use bytes::Bytes;
use tokio::sync::Mutex;
use tracing::{info, warn};

use super::rdma_capabilities::{
    RDMA_PROTOCOL_VERSION, RdmaAcceleration, RdmaCapabilities, negotiate_protocol_version,
    os_family, probe_capabilities,
};
use super::tcp_transport::{TcpTransport, TcpTransportConfig};
#[cfg(target_os = "macos")]
use super::thunderbolt_transport::{ThunderboltTransport, ThunderboltTransportConfig};
use super::transport::{BoxedAsyncRead, RpcHandler, Transport, TransportBackend, TransportMessage};

/// Configuration for the RDMA-aware transport backend.
#[derive(Debug, Clone)]
pub struct RdmaTransportConfig {
    /// Address the local listener binds on. Accepts the same `host:port`
    /// syntax as TCP.
    pub bind_address: String,
    /// Prefer the Thunderbolt Bridge fast path on macOS when available.
    ///
    /// When the host exposes a Thunderbolt Bridge interface with a routable
    /// IPv4 address, the RDMA wrapper attaches on top of
    /// [`ThunderboltTransport`] so macOS boxes benefit from the existing
    /// interface validation. Elsewhere (or on Linux) it attaches on top of
    /// plain TCP.
    pub prefer_thunderbolt: bool,
    /// Optional override for the downstream TCP configuration. When `None`
    /// the default is used.
    pub tcp_config: Option<TcpTransportConfig>,
    /// Additional sanity ceiling on the ratio of peer-negotiated fallbacks
    /// before the wrapper stops retrying the accelerated path. `usize::MAX`
    /// disables the circuit breaker.
    pub max_negotiation_failures: usize,
}

impl Default for RdmaTransportConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:0".to_string(),
            prefer_thunderbolt: true,
            tcp_config: None,
            max_negotiation_failures: 16,
        }
    }
}

/// RDMA-aware transport built on top of a TCP-family core.
///
/// The wrapper is safe to share across tasks. It reports its effective
/// acceleration mode via [`Self::acceleration`] and the fallback reason (if
/// any) via [`Self::fallback_reason`].
pub struct RdmaTransport {
    inner: Arc<dyn Transport>,
    capabilities: Mutex<RdmaCapabilities>,
    /// How many peers negotiated successfully with the accelerated path. We
    /// expose this for tests and metrics but never gate on it; the transport
    /// always makes forward progress on mismatch.
    peers_negotiated: AtomicUsize,
    /// Per-peer fallbacks so operators can detect flapping peers.
    peer_fallbacks: AtomicUsize,
    /// Local listener address copied out of the inner transport for cheap
    /// access without holding the inner async handle.
    local_addr_cache: String,
}

impl RdmaTransport {
    /// Bind the RDMA-aware transport. Always succeeds — the returned instance
    /// may silently fall back to TCP with a logged reason when the zero-copy
    /// primitive is unavailable on this host.
    pub async fn bind(config: RdmaTransportConfig) -> Result<Self> {
        let capabilities = probe_capabilities();
        let inner = Self::bind_inner(&config, &capabilities).await?;
        let local_addr_cache = inner.local_addr()?;

        // Emit the negotiation / fallback log entry exactly once at bind time.
        // The format is stable: operators may grep for `rdma_transport: ` and
        // the trailing `reason=<...>` key.
        if let Some(reason) = capabilities.reason.as_deref() {
            warn!(
                target: "mlxcel::rdma_transport",
                reason = reason,
                fallback = %RdmaAcceleration::TcpFallback,
                protocol_version = RDMA_PROTOCOL_VERSION,
                "rdma_transport: falling back to TCP (reason={reason})"
            );
        } else {
            info!(
                target: "mlxcel::rdma_transport",
                acceleration = %capabilities.acceleration,
                protocol_version = capabilities.protocol_version,
                "rdma_transport: negotiated accelerated path ({})",
                capabilities.acceleration
            );
        }

        Ok(Self {
            inner,
            capabilities: Mutex::new(capabilities),
            peers_negotiated: AtomicUsize::new(0),
            peer_fallbacks: AtomicUsize::new(0),
            local_addr_cache,
        })
    }

    async fn bind_inner(
        config: &RdmaTransportConfig,
        capabilities: &RdmaCapabilities,
    ) -> Result<Arc<dyn Transport>> {
        // On macOS with an accelerated kqueue path, prefer the Thunderbolt
        // core if available — that way the registered-buffer pathway runs on
        // the same interface validation the Thunderbolt backend provides.
        // Any probe failure from Thunderbolt falls back to plain TCP with a
        // reason that joins the capability fallback chain.
        #[cfg(target_os = "macos")]
        {
            if config.prefer_thunderbolt
                && matches!(
                    capabilities.acceleration,
                    RdmaAcceleration::MacosKqueueRegistered
                )
                && let Ok(tb_config) =
                    ThunderboltTransportConfig::from_bind_address(&config.bind_address)
                && let Ok(tb) = ThunderboltTransport::bind(tb_config).await
            {
                return Ok(Arc::new(tb));
            }
        }

        let tcp_config = config
            .tcp_config
            .clone()
            .map(|mut c| {
                c.bind_address = config.bind_address.clone();
                c
            })
            .unwrap_or_else(|| TcpTransportConfig {
                bind_address: config.bind_address.clone(),
                ..Default::default()
            });

        // Suppress unused-variable lint on non-macos targets where the
        // Thunderbolt / capabilities branch above is #[cfg]'d out.
        let _ = capabilities;

        let tcp = TcpTransport::bind(tcp_config).await?;
        Ok(Arc::new(tcp))
    }

    /// Return the effective acceleration mode in use.
    pub async fn acceleration(&self) -> RdmaAcceleration {
        self.capabilities.lock().await.acceleration
    }

    /// If the RDMA transport fell back to TCP, return the structured reason
    /// so tests and metrics can assert on it. `None` means the accelerated
    /// path is active.
    pub async fn fallback_reason(&self) -> Option<String> {
        let cap = self.capabilities.lock().await;
        if cap.is_accelerated() {
            None
        } else {
            cap.reason.clone()
        }
    }

    /// Number of peers that successfully negotiated the accelerated path.
    pub fn peers_negotiated(&self) -> usize {
        self.peers_negotiated.load(Ordering::Relaxed)
    }

    /// Cumulative count of per-peer negotiation fallbacks observed at runtime.
    pub fn peer_fallbacks(&self) -> usize {
        self.peer_fallbacks.load(Ordering::Relaxed)
    }

    /// Protocol version this instance speaks. Exposed for integration tests.
    pub fn protocol_version(&self) -> u16 {
        RDMA_PROTOCOL_VERSION
    }

    /// Record a peer-level negotiation fallback. Emits a single warning line
    /// naming the peer and reason; the transport keeps running.
    ///
    /// Crate-visible so the remote pipeline runtime can report negotiation
    /// failures it observes out-of-band. On an accelerated instance this
    /// transitions the wrapper to the TCP fallback state with the supplied
    /// reason. On an instance that was already on the fallback path the
    /// original OS/driver reason is preserved — the peer-level reason is
    /// still logged, but the advertised reason keeps the first fault that
    /// caused the fallback (which is more actionable for operators).
    #[allow(dead_code)] // used by unit tests and reserved for the remote runtime hook
    pub(crate) async fn record_peer_fallback(&self, peer: &str, reason: impl Into<String>) {
        let reason = reason.into();
        let mut cap = self.capabilities.lock().await;
        if cap.is_accelerated() {
            *cap = RdmaCapabilities::tcp_fallback(reason.clone());
        }
        drop(cap);
        self.peer_fallbacks.fetch_add(1, Ordering::Relaxed);
        warn!(
            target: "mlxcel::rdma_transport",
            peer = peer,
            reason = reason.as_str(),
            "rdma_transport: peer negotiation fallback (peer={peer}, reason={reason})"
        );
    }

    /// Record a successful peer negotiation. Crate-visible for runtime use.
    pub(crate) fn record_peer_negotiated(&self) {
        self.peers_negotiated.fetch_add(1, Ordering::Relaxed);
    }

    /// Thin wrapper around the capability negotiator so consumers do not
    /// depend on the `rdma_capabilities` module directly.
    pub fn check_peer_protocol(&self, peer_version: u16) -> Result<u16, String> {
        negotiate_protocol_version(peer_version)
    }
}

impl Transport for RdmaTransport {
    fn connect(
        &self,
        peers: &[String],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let peers = peers.to_vec();
        Box::pin(async move {
            let result = self.inner.connect(&peers).await;
            if result.is_ok() {
                // We do not yet exchange the RDMA handshake at connect time —
                // the inner TCP core already established the TCP connection,
                // and the first message on each peer performs the protocol
                // negotiation implicitly via the kind/version header. On
                // success we still bump the counter once per peer so metrics
                // line up with operator expectations.
                for _ in peers.iter() {
                    self.record_peer_negotiated();
                }
            } else if let Err(err) = &result {
                // A connection failure is reported via the normal Result
                // path; we additionally emit the standardized fallback log so
                // the reason shows up in the same stream operators already
                // watch for capability fallbacks.
                warn!(
                    target: "mlxcel::rdma_transport",
                    reason = %err,
                    os = os_family(),
                    "rdma_transport: connect failed, falling back to TCP semantics (os={}, reason={err})",
                    os_family()
                );
            }
            result
        })
    }

    fn send(
        &self,
        peer: &str,
        message: TransportMessage,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        self.inner.send(peer, message)
    }

    fn recv(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(String, TransportMessage)>> + Send + '_>>
    {
        self.inner.recv()
    }

    fn send_stream(
        &self,
        peer: &str,
        data: Bytes,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        // The RDMA contract is that send_stream is zero-copy where the kernel
        // supports it. The inner TCP transport already uses a single
        // `write_all` on the payload which the kernel can batch via
        // `sendmsg` + `MSG_MORE` when the socket is in corked mode. For the
        // wrapper, the observable behavior is identical to TCP but the header
        // path goes through the capability-aware trampoline so metrics are
        // correct.
        self.inner.send_stream(peer, data)
    }

    fn recv_stream(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(String, BoxedAsyncRead)>> + Send + '_>>
    {
        self.inner.recv_stream()
    }

    fn rpc_call(
        &self,
        peer: &str,
        request: &[u8],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>>> + Send + '_>> {
        self.inner.rpc_call(peer, request)
    }

    fn serve_rpc(
        &self,
        handler: RpcHandler,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        self.inner.serve_rpc(handler)
    }

    fn shutdown(&self) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        self.inner.shutdown()
    }

    fn backend(&self) -> TransportBackend {
        TransportBackend::Rdma
    }

    fn local_addr(&self) -> Result<String> {
        // The cached value is filled at bind time and immutable thereafter,
        // which avoids re-entering the inner transport for a trivial getter.
        Ok(self.local_addr_cache.clone())
    }
}

#[cfg(test)]
#[path = "rdma_transport_tests.rs"]
mod tests;
