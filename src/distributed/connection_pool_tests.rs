// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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
