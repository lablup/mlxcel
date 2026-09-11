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

//! Connection pooling with automatic reconnection for persistent TCP streams.
//!
//! Each peer address maintains up to [`PoolConfig::max_connections_per_peer`]
//! idle connections. When a connection fails, the pool transparently retries
//! with exponential backoff.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

/// Configuration for the connection pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum idle connections per peer.
    pub max_connections_per_peer: usize,
    /// Timeout for establishing a new connection.
    pub connect_timeout: Duration,
    /// Initial delay before the first reconnection attempt.
    pub reconnect_base_delay: Duration,
    /// Maximum delay between reconnection attempts.
    pub reconnect_max_delay: Duration,
    /// Maximum number of reconnection attempts before giving up.
    pub max_reconnect_attempts: u32,
    /// Enable TCP_NODELAY (disable Nagle's algorithm) for lower latency.
    pub tcp_nodelay: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections_per_peer: 4,
            connect_timeout: Duration::from_secs(5),
            reconnect_base_delay: Duration::from_millis(100),
            reconnect_max_delay: Duration::from_secs(10),
            max_reconnect_attempts: 5,
            tcp_nodelay: true,
        }
    }
}

/// A pool of TCP connections to a single peer.
struct PeerPool {
    /// Idle connections ready for reuse.
    idle: Vec<TcpStream>,
    /// Number of connections currently checked out.
    active: usize,
}

impl PeerPool {
    fn new() -> Self {
        Self {
            idle: Vec::new(),
            active: 0,
        }
    }
}

/// Thread-safe connection pool managing TCP connections to multiple peers.
///
/// Used by [`super::tcp_transport::TcpTransport`] to maintain persistent
/// connections with automatic reconnection.
pub struct ConnectionPool {
    config: PoolConfig,
    pools: Arc<Mutex<HashMap<String, PeerPool>>>,
}

impl ConnectionPool {
    /// Create a new connection pool with the given configuration.
    pub fn new(config: PoolConfig) -> Self {
        Self {
            config,
            pools: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Acquire a connection to the given peer address.
    ///
    /// Tries to reuse an idle connection first; if none are available, opens
    /// a new one. On connection failure, retries with exponential backoff.
    pub async fn acquire(&self, peer: &str) -> Result<TcpStream> {
        // Try to reuse an idle connection.
        {
            let mut pools = self.pools.lock().await;
            if let Some(pool) = pools.get_mut(peer) {
                while let Some(stream) = pool.idle.pop() {
                    // Verify the connection is still alive by peeking.
                    if stream.peer_addr().is_ok() {
                        pool.active += 1;
                        return Ok(stream);
                    }
                    // Dead connection; drop and try next.
                }
            }
        }

        // No idle connection available; open a new one with retries.
        let stream = self.connect_with_retry(peer).await?;

        let mut pools = self.pools.lock().await;
        let pool = pools.entry(peer.to_string()).or_insert_with(PeerPool::new);
        pool.active += 1;

        Ok(stream)
    }

    /// Return a connection to the pool for reuse.
    ///
    /// If the pool for this peer is already at capacity, the connection is
    /// dropped instead.
    pub async fn release(&self, peer: &str, stream: TcpStream) {
        let mut pools = self.pools.lock().await;
        if let Some(pool) = pools.get_mut(peer) {
            pool.active = pool.active.saturating_sub(1);
            if pool.idle.len() < self.config.max_connections_per_peer {
                pool.idle.push(stream);
            }
            // else: drop the stream (over capacity)
        }
    }

    /// Open a new TCP connection with exponential-backoff retry.
    async fn connect_with_retry(&self, peer: &str) -> Result<TcpStream> {
        let mut delay = self.config.reconnect_base_delay;

        for attempt in 0..=self.config.max_reconnect_attempts {
            match tokio::time::timeout(self.config.connect_timeout, TcpStream::connect(peer)).await
            {
                Ok(Ok(stream)) => {
                    if self.config.tcp_nodelay {
                        let _ = stream.set_nodelay(true);
                    }
                    if attempt == 0 {
                        tracing::info!("Connected to peer {peer}");
                    } else {
                        tracing::info!("Reconnected to peer {peer} after {} attempts", attempt + 1);
                    }
                    return Ok(stream);
                }
                Ok(Err(e)) => {
                    if attempt == self.config.max_reconnect_attempts {
                        return Err(e).context(format!(
                            "failed to connect to {peer} after {} attempts",
                            attempt + 1
                        ));
                    }
                    tracing::warn!(
                        "Connection to {peer} failed (attempt {}/{}): {e}",
                        attempt + 1,
                        self.config.max_reconnect_attempts + 1
                    );
                }
                Err(_) => {
                    if attempt == self.config.max_reconnect_attempts {
                        anyhow::bail!(
                            "connection to {peer} timed out after {} attempts",
                            attempt + 1
                        );
                    }
                    tracing::warn!(
                        "Connection to {peer} timed out (attempt {}/{})",
                        attempt + 1,
                        self.config.max_reconnect_attempts + 1
                    );
                }
            }

            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(self.config.reconnect_max_delay);
        }

        anyhow::bail!("exhausted reconnection attempts to {peer}")
    }

    /// Close all connections in the pool and clear state.
    pub async fn shutdown(&self) {
        let mut pools = self.pools.lock().await;
        pools.clear();
    }

    /// Return a snapshot of pool statistics.
    pub async fn stats(&self) -> PoolStats {
        let pools = self.pools.lock().await;
        let mut total_idle = 0;
        let mut total_active = 0;
        let peer_count = pools.len();
        for pool in pools.values() {
            total_idle += pool.idle.len();
            total_active += pool.active;
        }
        PoolStats {
            peer_count,
            total_idle,
            total_active,
        }
    }
}

/// Snapshot of connection pool statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolStats {
    /// Number of distinct peers in the pool.
    pub peer_count: usize,
    /// Total idle connections across all peers.
    pub total_idle: usize,
    /// Total active (checked-out) connections across all peers.
    pub total_active: usize,
}

#[cfg(test)]
#[path = "connection_pool_tests.rs"]
mod tests;
