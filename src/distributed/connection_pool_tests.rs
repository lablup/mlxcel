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

use super::*;

#[test]
fn pool_config_default_values() {
    let config = PoolConfig::default();
    assert_eq!(config.max_connections_per_peer, 4);
    assert_eq!(config.connect_timeout, Duration::from_secs(5));
    assert!(config.tcp_nodelay);
    assert_eq!(config.max_reconnect_attempts, 5);
}

#[tokio::test]
async fn pool_stats_empty() {
    let pool = ConnectionPool::new(PoolConfig::default());
    let stats = pool.stats().await;
    assert_eq!(
        stats,
        PoolStats {
            peer_count: 0,
            total_idle: 0,
            total_active: 0,
        }
    );
}

#[tokio::test]
async fn pool_acquire_and_release() {
    // Start a local TCP listener.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();

    // Accept connections in the background (keep accepting indefinitely).
    let accept_handle = tokio::spawn(async move {
        let mut accepted = Vec::new();
        while let Ok((stream, _)) = listener.accept().await {
            accepted.push(stream);
        }
        accepted
    });

    let config = PoolConfig {
        max_connections_per_peer: 2,
        connect_timeout: Duration::from_secs(2),
        max_reconnect_attempts: 0,
        ..PoolConfig::default()
    };
    let pool = ConnectionPool::new(config);

    // Acquire a connection.
    let stream1 = pool.acquire(&addr).await.unwrap();
    let stats = pool.stats().await;
    assert_eq!(stats.total_active, 1);
    assert_eq!(stats.total_idle, 0);

    // Release it back.
    pool.release(&addr, stream1).await;
    let stats = pool.stats().await;
    assert_eq!(stats.total_active, 0);
    assert_eq!(stats.total_idle, 1);

    // Acquire again -- should reuse the idle connection.
    let _stream2 = pool.acquire(&addr).await.unwrap();
    let stats = pool.stats().await;
    assert_eq!(stats.total_active, 1);
    assert_eq!(stats.total_idle, 0);

    pool.shutdown().await;

    // Clean up the acceptor.
    accept_handle.abort();
    let _ = accept_handle.await;
}

#[tokio::test]
async fn pool_acquire_unreachable_fails() {
    let config = PoolConfig {
        max_connections_per_peer: 1,
        connect_timeout: Duration::from_millis(100),
        max_reconnect_attempts: 0,
        reconnect_base_delay: Duration::from_millis(10),
        ..PoolConfig::default()
    };
    let pool = ConnectionPool::new(config);

    // Connect to a port that is almost certainly not listening.
    let result = pool.acquire("127.0.0.1:1").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn pool_shutdown_clears_state() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();

    let accept_handle = tokio::spawn(async move {
        let _ = listener.accept().await;
    });

    let config = PoolConfig {
        max_reconnect_attempts: 0,
        ..PoolConfig::default()
    };
    let pool = ConnectionPool::new(config);

    let stream = pool.acquire(&addr).await.unwrap();
    pool.release(&addr, stream).await;

    let stats = pool.stats().await;
    assert_eq!(stats.peer_count, 1);

    pool.shutdown().await;

    let stats = pool.stats().await;
    assert_eq!(stats.peer_count, 0);
    assert_eq!(stats.total_idle, 0);

    let _ = accept_handle.await;
}

/// Issue #2695: an idle pooled connection whose peer has exited must not be
/// handed out again. Before the fix `acquire` checked `peer_addr()`, which
/// still succeeds for a socket in CLOSE_WAIT, so the stale stream came back
/// and the caller's first write "succeeded" into a dead socket.
#[tokio::test]
async fn acquire_discards_idle_connection_whose_peer_closed() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();

    let config = PoolConfig {
        max_connections_per_peer: 2,
        connect_timeout: Duration::from_secs(2),
        reconnect_base_delay: Duration::from_millis(10),
        max_reconnect_attempts: 0,
        ..PoolConfig::default()
    };
    let pool = ConnectionPool::new(config);

    let stream = pool.acquire(&addr).await.unwrap();
    pool.release(&addr, stream).await;
    assert_eq!(pool.stats().await.total_idle, 1);

    // The peer process exits: its side of the connection closes and nothing
    // listens on the address any more.
    let (server_side, _) = listener.accept().await.unwrap();
    drop(server_side);
    drop(listener);
    tokio::time::sleep(Duration::from_millis(50)).await;

    let result = pool.acquire(&addr).await;
    assert!(
        result.is_err(),
        "acquire handed out an idle connection whose peer had closed it"
    );
    let stats = pool.stats().await;
    assert_eq!(stats.total_idle, 0, "the dead connection must be dropped");
    assert_eq!(stats.total_active, 0);
}

/// The liveness probe behind [`ConnectionPool::acquire`] must not consume
/// bytes: a peer that already sent a frame on an idle connection is alive,
/// and that frame must still be readable in full afterwards (#2695).
#[tokio::test]
async fn peer_closed_probe_is_non_consuming() {
    use tokio::io::AsyncWriteExt;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let client = ProtectedStream::plain(TcpStream::connect(addr).await.unwrap());
    let (mut server_side, _) = listener.accept().await.unwrap();

    // Connected, nothing sent: alive.
    assert!(!client.peer_closed());

    // One complete plain Control frame waiting unread: alive, and the probe
    // leaves every byte in place.
    let payload = b"\x00\x00\x00\x04pingdata";
    let mut frame = vec![crate::distributed::transport::MessageKind::Control as u8];
    frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    frame.extend_from_slice(payload);
    server_side.write_all(&frame).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!client.peer_closed());
    assert!(!client.peer_closed());

    let mut client = client;
    let (kind, body) = client.read_frame(1024).await.unwrap();
    assert_eq!(kind, crate::distributed::transport::MessageKind::Control);
    assert_eq!(&body[..], payload);

    // Drained and then closed by the peer: gone.
    drop(server_side);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(client.peer_closed());
}

/// An idle pooled connection on which the peer has already written bytes is
/// not handed out again (#2712 item 5): an idle outbound connection has no
/// legitimate pending input (RPC responses are consumed by the call that
/// made the request), and stray bytes would desynchronize the next read.
/// The pool drops it and opens a new connection instead.
#[tokio::test]
async fn acquire_drops_idle_connection_with_pending_data() {
    use tokio::io::AsyncWriteExt;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let pool = ConnectionPool::new(PoolConfig {
        max_connections_per_peer: 2,
        connect_timeout: Duration::from_secs(2),
        reconnect_base_delay: Duration::from_millis(10),
        max_reconnect_attempts: 0,
        ..PoolConfig::default()
    });

    let stream = pool.acquire(&addr).await.unwrap();
    pool.release(&addr, stream).await;
    let (mut first_server_side, _) = listener.accept().await.unwrap();
    // The peer writes one complete plain Control frame onto the idle
    // connection; a FIN may or may not follow.
    first_server_side
        .write_all(b"\x02\x00\x00\x00\x00\x00\x00\x00\x04ping")
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let _second = pool.acquire(&addr).await.unwrap();
    let accepted = tokio::time::timeout(Duration::from_secs(1), listener.accept()).await;
    assert!(
        accepted.is_ok(),
        "acquire reused the idle connection with pending input instead of opening a new one"
    );
    let stats = pool.stats().await;
    assert_eq!(stats.total_idle, 0, "the connection with pending input must be dropped");
    assert_eq!(stats.total_active, 1);
}
