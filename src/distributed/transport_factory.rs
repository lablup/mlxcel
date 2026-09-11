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

//! Shared transport builder used by distributed runtime entry points.

use std::sync::Arc;

use anyhow::Result;

use super::rdma_transport::{RdmaTransport, RdmaTransportConfig};
use super::tcp_transport::{TcpTransport, TcpTransportConfig};
use super::thunderbolt_transport::{ThunderboltTransport, ThunderboltTransportConfig};
use super::{Transport, TransportBackend};

/// Bind a transport backend for the given local control address.
///
/// The RDMA backend internally probes OS-level zero-copy primitives and
/// transparently falls back to TCP when they are unavailable. On fallback
/// the backend emits a single-line log entry naming the reason (OS, driver,
/// peer version, or capability mismatch); the factory caller sees the same
/// `Arc<dyn Transport>` either way.
pub async fn bind_transport(
    backend: TransportBackend,
    bind_address: &str,
) -> Result<Arc<dyn Transport>> {
    match backend {
        TransportBackend::Tcp => Ok(Arc::new(
            TcpTransport::bind(TcpTransportConfig {
                bind_address: bind_address.to_string(),
                ..Default::default()
            })
            .await?,
        )),
        TransportBackend::Thunderbolt => Ok(Arc::new(
            ThunderboltTransport::bind(ThunderboltTransportConfig::from_bind_address(
                bind_address,
            )?)
            .await?,
        )),
        TransportBackend::Rdma => Ok(Arc::new(
            RdmaTransport::bind(RdmaTransportConfig {
                bind_address: bind_address.to_string(),
                ..Default::default()
            })
            .await?,
        )),
    }
}
