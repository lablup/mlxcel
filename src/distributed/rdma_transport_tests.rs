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

//! Unit tests for the RDMA-aware transport wrapper.
//!
//! These tests exercise bind-time capability negotiation and runtime
//! fallback semantics without touching real hardware beyond what is
//! available on the CI worker.

use bytes::Bytes;

use super::{RdmaTransport, RdmaTransportConfig};
use crate::distributed::rdma_capabilities::{RDMA_PROTOCOL_VERSION, RdmaAcceleration};
use crate::distributed::transport::{Transport, TransportBackend, TransportMessage};

async fn new_local_rdma() -> RdmaTransport {
    RdmaTransport::bind(RdmaTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        prefer_thunderbolt: false,
        tcp_config: None,
        max_negotiation_failures: 4,
    })
    .await
    .expect("bind rdma transport")
}

#[tokio::test]
async fn bind_succeeds_and_reports_backend_rdma() {
    let transport = new_local_rdma().await;
    assert_eq!(transport.backend(), TransportBackend::Rdma);
    assert!(
        transport
            .local_addr()
            .expect("local addr")
            .starts_with("127.0.0.1:")
    );
    assert_eq!(transport.protocol_version(), RDMA_PROTOCOL_VERSION);
}

#[tokio::test]
async fn capability_negotiation_yields_stable_state() {
    let transport = new_local_rdma().await;
    let accel = transport.acceleration().await;

    // One of the three is always correct; it must never be some other value.
    match accel {
        RdmaAcceleration::LinuxIoUring
        | RdmaAcceleration::MacosKqueueRegistered
        | RdmaAcceleration::TcpFallback => {}
    }

    // Fallback reason is present iff we are on the TCP path.
    if accel == RdmaAcceleration::TcpFallback {
        let reason = transport.fallback_reason().await;
        assert!(
            reason.as_deref().is_some_and(|r| !r.is_empty()),
            "tcp fallback must carry a non-empty reason"
        );
    } else {
        assert!(transport.fallback_reason().await.is_none());
    }
}

#[tokio::test]
async fn record_peer_fallback_flips_state_and_logs_reason() {
    let transport = new_local_rdma().await;
    let before = transport.peer_fallbacks();
    let was_accelerated = transport.acceleration().await != RdmaAcceleration::TcpFallback;

    // Simulate a peer that replied with an incompatible protocol version.
    transport
        .record_peer_fallback("10.0.0.1:9000", "os=linux, peer_version=99: mismatch")
        .await;

    assert_eq!(transport.peer_fallbacks(), before + 1);
    let reason = transport.fallback_reason().await;
    assert!(reason.is_some(), "peer-level fallback must set a reason");

    if was_accelerated {
        // When we start on an accelerated path, a peer-level fallback
        // transitions us to TCP with the exact peer reason recorded.
        assert!(reason.unwrap().contains("peer_version=99"));
        assert_eq!(
            transport.acceleration().await,
            RdmaAcceleration::TcpFallback
        );
    } else {
        // On hosts where we already started on the TCP fallback path,
        // record_peer_fallback preserves the original OS/driver reason that
        // motivated the initial fallback — peer-level reasons are only
        // recorded on the first transition.
        assert_eq!(
            transport.acceleration().await,
            RdmaAcceleration::TcpFallback
        );
    }
}

#[tokio::test]
async fn check_peer_protocol_rejects_mismatch() {
    let transport = new_local_rdma().await;
    assert!(transport.check_peer_protocol(RDMA_PROTOCOL_VERSION).is_ok());
    let err = transport
        .check_peer_protocol(RDMA_PROTOCOL_VERSION.wrapping_add(1))
        .unwrap_err();
    assert!(err.contains("peer_version="));
}

#[tokio::test]
async fn small_message_send_loopback_matches_tcp_semantics() {
    // Spin up two RDMA wrappers bound to loopback, connect one to the
    // other, and exchange a tiny control message. The observable behavior
    // must match the TCP core regardless of the chosen acceleration path.
    let server = new_local_rdma().await;
    let server_addr = server.local_addr().unwrap();
    let client = new_local_rdma().await;

    client
        .connect(std::slice::from_ref(&server_addr))
        .await
        .expect("connect");

    let payload = Bytes::from_static(b"hello-rdma");
    client
        .send(
            &server_addr,
            TransportMessage::Control {
                operation: "probe".to_string(),
                payload: payload.clone(),
            },
        )
        .await
        .expect("send");

    let (from, msg) = tokio::time::timeout(std::time::Duration::from_secs(2), server.recv())
        .await
        .expect("recv timed out")
        .expect("recv");
    assert!(from.contains("127.0.0.1"));
    match msg {
        TransportMessage::Control { operation, payload } => {
            assert_eq!(operation, "probe");
            assert_eq!(payload, Bytes::from_static(b"hello-rdma"));
        }
        other => panic!("unexpected message: {other:?}"),
    }

    server.shutdown().await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn connect_counts_successful_peers() {
    // Pair two instances and verify that the per-peer negotiation counter
    // increments on a successful connect.
    let server = new_local_rdma().await;
    let server_addr = server.local_addr().unwrap();
    let client = new_local_rdma().await;

    assert_eq!(client.peers_negotiated(), 0);
    client.connect(&[server_addr]).await.expect("connect");
    assert_eq!(client.peers_negotiated(), 1);

    server.shutdown().await.unwrap();
    client.shutdown().await.unwrap();
}
