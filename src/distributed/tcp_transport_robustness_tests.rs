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

//! Transport robustness follow-ups (#2712): capacity refusals are retried
//! with backoff (item 2), a failed or cancelled send gives its connection
//! slot back (item 5), and `rpc_call` reads its response on the connection
//! the request went out on (item 6).
//!
//! Every wait is bounded so a regression shows up as a failed assertion, not
//! a hung test.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;

use crate::distributed::cluster_auth::{ClusterAuth, ClusterSecret};
use crate::distributed::cluster_handshake::{ServerAtCapacity, server_handshake};
use crate::distributed::connection_pool::{PoolConfig, PoolStats};
use crate::distributed::frame_protection::FrameSuite;
use crate::distributed::tcp_frame::ProtectedStream;
use crate::distributed::tcp_transport::{TcpTransport, TcpTransportConfig};
use crate::distributed::transport::{MessageKind, Transport, TransportMessage};

/// Upper bound for an operation that must settle.
const SETTLE: Duration = Duration::from_secs(10);

fn auth() -> Arc<ClusterAuth> {
    Arc::new(ClusterAuth::new(&ClusterSecret::random()))
}

async fn bind(
    auth: Option<Arc<ClusterAuth>>,
    pool_config: PoolConfig,
    max_connections: usize,
) -> TcpTransport {
    TcpTransport::bind(TcpTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        auth,
        pool_config,
        max_connections,
        ..Default::default()
    })
    .await
    .unwrap()
}

/// A pool that never retries a refused connect, so the tests below measure
/// only the behavior under test.
fn no_retry() -> PoolConfig {
    PoolConfig {
        connect_timeout: Duration::from_secs(2),
        reconnect_base_delay: Duration::from_millis(10),
        max_reconnect_attempts: 0,
        ..PoolConfig::default()
    }
}

fn ping() -> TransportMessage {
    TransportMessage::Control {
        operation: "ping".to_string(),
        payload: Bytes::from_static(b"ping"),
    }
}

/// A frame far larger than the loopback socket buffers, so a write to a peer
/// that does not read blocks, and a write to a peer that reset fails.
fn big_tensor() -> TransportMessage {
    TransportMessage::TensorData {
        tensor_id: "big".to_string(),
        shape: vec![32 << 20],
        data: Bytes::from(vec![0u8; 32 << 20]),
    }
}

/// A server at its connection cap and a client holding that one slot; the
/// returned triple is `(server, holder, server_addr)`.
async fn full_server(shared: &Arc<ClusterAuth>) -> (TcpTransport, TcpTransport, String) {
    let server = bind(Some(Arc::clone(shared)), PoolConfig::default(), 1).await;
    let server_addr = server.local_addr().unwrap();
    let holder = bind(Some(Arc::clone(shared)), no_retry(), 512).await;
    holder.send(&server_addr, ping()).await.unwrap();
    let (from, _) = tokio::time::timeout(Duration::from_secs(5), server.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(from, holder.local_addr().unwrap());
    // The holder keeps its pooled connection, so the single slot stays taken.
    (server, holder, server_addr)
}

/// Item 2: a peer that is at capacity now may free a slot within the retry
/// budget, so the pool backs off and retries instead of returning the
/// refusal at once. The holder releases the slot about 120 ms into client
/// B's acquire; with a 50 ms base delay and 3 retries, the same `send`
/// reaches the server.
#[tokio::test]
async fn server_at_capacity_is_retried_with_backoff() {
    let shared = auth();
    let (server, holder, server_addr) = full_server(&shared).await;

    let client_b = bind(
        Some(Arc::clone(&shared)),
        PoolConfig {
            max_reconnect_attempts: 3,
            reconnect_base_delay: Duration::from_millis(50),
            reconnect_max_delay: Duration::from_secs(1),
            connect_timeout: Duration::from_secs(2),
            ..PoolConfig::default()
        },
        512,
    )
    .await;
    let release = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(120)).await;
        holder.shutdown().await.unwrap();
    });

    let started = Instant::now();
    tokio::time::timeout(SETTLE, client_b.send(&server_addr, ping()))
        .await
        .expect("send must settle")
        .expect("the retried acquire must reach the server once the slot is free");
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_millis(100),
        "the send cannot have succeeded before the slot was freed: {elapsed:?}"
    );
    let (from, _) = tokio::time::timeout(Duration::from_secs(5), server.recv())
        .await
        .expect("client B's frame must arrive")
        .unwrap();
    assert_eq!(from, client_b.local_addr().unwrap());

    release.await.unwrap();
    client_b.shutdown().await.unwrap();
    server.shutdown().await.unwrap();
}

/// Item 2: when the slot is never freed, the refusal is returned after the
/// whole retry budget (20 ms + 40 ms of backoff here) with `ServerAtCapacity`
/// still in the error chain, so callers can keep distinguishing it from a
/// rejection.
#[tokio::test]
async fn server_at_capacity_exhausts_retry_budget() {
    let shared = auth();
    let (server, holder, server_addr) = full_server(&shared).await;

    let client_b = bind(
        Some(Arc::clone(&shared)),
        PoolConfig {
            max_reconnect_attempts: 2,
            reconnect_base_delay: Duration::from_millis(20),
            connect_timeout: Duration::from_secs(2),
            ..PoolConfig::default()
        },
        512,
    )
    .await;
    let started = Instant::now();
    let err = tokio::time::timeout(SETTLE, client_b.send(&server_addr, ping()))
        .await
        .expect("send must settle")
        .expect_err("the slot is never freed");
    let elapsed = started.elapsed();
    assert!(
        err.chain()
            .any(|cause| cause.downcast_ref::<ServerAtCapacity>().is_some()),
        "expected ServerAtCapacity in the chain, got: {err:#}"
    );
    assert!(
        elapsed >= Duration::from_millis(60),
        "the capacity refusal was returned without spending the retry budget: {elapsed:?}"
    );

    client_b.shutdown().await.unwrap();
    holder.shutdown().await.unwrap();
    server.shutdown().await.unwrap();
}

/// Item 5: a `send` whose write fails (the peer reset the connection after
/// the first bytes) must not leave the connection counted as checked out.
#[tokio::test]
async fn failed_send_releases_active_slot() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let peer = listener.local_addr().unwrap().to_string();
    let server = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut byte = [0u8; 1];
        let _ = sock.read_exact(&mut byte).await;
        // A zero linger turns the close into a reset.
        sock.set_linger(Some(Duration::ZERO)).unwrap();
        drop(sock);
    });

    let client = bind(None, no_retry(), 512).await;
    let result = tokio::time::timeout(SETTLE, client.send(&peer, big_tensor()))
        .await
        .expect("send must settle");
    assert!(
        result.is_err(),
        "writing 32 MiB into a reset connection must fail"
    );
    server.await.unwrap();

    assert_eq!(
        client.pool.stats().await,
        PoolStats {
            peer_count: 1,
            total_idle: 0,
            total_active: 0,
        },
        "the failed send must give its connection slot back"
    );
    client.shutdown().await.unwrap();
}

/// Item 5: a `send` future dropped mid-write (here by a timeout, as a
/// cancelled request does) must give its connection slot back too; an
/// explicit discard on the error path alone would miss this.
#[tokio::test]
async fn cancelled_send_releases_active_slot() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let peer = listener.local_addr().unwrap().to_string();
    let server = tokio::spawn(async move {
        // Accept and hold the socket without reading, so the write blocks.
        let (sock, _) = listener.accept().await.unwrap();
        tokio::time::sleep(SETTLE).await;
        drop(sock);
    });

    let client = bind(None, no_retry(), 512).await;
    let cancelled =
        tokio::time::timeout(Duration::from_millis(200), client.send(&peer, big_tensor())).await;
    assert!(
        cancelled.is_err(),
        "a 32 MiB write to a peer that never reads must still be in flight at 200 ms"
    );

    assert_eq!(
        client.pool.stats().await,
        PoolStats {
            peer_count: 1,
            total_idle: 0,
            total_active: 0,
        },
        "the cancelled send must give its connection slot back"
    );
    server.abort();
    client.shutdown().await.unwrap();
}

/// Item 6: the RPC response is read on the pooled connection the request
/// went out on. Two sequential calls on the same connection both receive the
/// handler's bytes, and the connection goes back to the pool afterwards.
#[tokio::test]
async fn rpc_call_round_trips_over_authenticated_tcp() {
    let shared = auth();
    let server = bind(Some(Arc::clone(&shared)), PoolConfig::default(), 512).await;
    server
        .serve_rpc(Box::new(|request| request.to_vec()))
        .await
        .unwrap();
    let server_addr = server.local_addr().unwrap();

    let client = bind(Some(shared), no_retry(), 512).await;
    for request in [&b"hello"[..], &b"again"[..]] {
        let response = tokio::time::timeout(SETTLE, client.rpc_call(&server_addr, request))
            .await
            .expect("rpc_call must settle")
            .expect("the echo handler's bytes must come back");
        assert_eq!(response, request);
    }
    assert_eq!(
        client.pool.stats().await,
        PoolStats {
            peer_count: 1,
            total_idle: 1,
            total_active: 0,
        },
        "both calls must share one pooled connection, returned after the response"
    );

    client.shutdown().await.unwrap();
    server.shutdown().await.unwrap();
}

/// Item 6: a call whose response does not arrive within `connect_timeout`
/// fails, and its connection is dropped rather than returned to the pool,
/// so a late response can never be left unread on a reused connection.
#[tokio::test]
async fn timed_out_rpc_call_drops_its_stream() {
    let shared = auth();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap().to_string();
    let server_auth = Arc::clone(&shared);
    let server_identity = server_addr.clone();
    // An authenticated peer that reads the request and never answers.
    let server = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let (_, keys) = server_handshake(
            &mut sock,
            &server_auth,
            &server_identity,
            FrameSuite::default(),
        )
        .await
        .unwrap();
        let mut stream = ProtectedStream::protected(sock, keys);
        let (kind, _) = stream.read_frame(1024 * 1024).await.unwrap();
        assert_eq!(kind, MessageKind::RpcRequest);
        tokio::time::sleep(SETTLE).await;
        drop(stream);
    });

    let client = bind(
        Some(shared),
        PoolConfig {
            connect_timeout: Duration::from_millis(300),
            ..no_retry()
        },
        512,
    )
    .await;
    let err = tokio::time::timeout(SETTLE, client.rpc_call(&server_addr, b"slow"))
        .await
        .expect("rpc_call must settle")
        .expect_err("a peer that never answers must time the call out");
    assert!(format!("{err:#}").contains("timed out"), "{err:#}");
    assert_eq!(
        client.pool.stats().await,
        PoolStats {
            peer_count: 1,
            total_idle: 0,
            total_active: 0,
        },
        "a timed-out call must drop its connection, not pool it"
    );

    server.abort();
    client.shutdown().await.unwrap();
}
