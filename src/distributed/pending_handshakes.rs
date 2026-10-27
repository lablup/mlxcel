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

//! Bound on inbound connections that have not finished the cluster handshake
//! (#2655 item a).
//!
//! An authenticated listener keeps two pools: the main connection cap
//! (`max_connections`), which only authenticated connections draw from, and
//! this pending pool, which every accepted socket occupies from accept until
//! `server_handshake` returns, at most [`HANDSHAKE_TIMEOUT`]. A flood of
//! sockets that connect and send nothing can therefore exhaust only the
//! pending pool, never the slots a real, authenticated peer needs. A per-IP
//! sub-limit keeps one source from taking the whole pending pool; it is not
//! sufficient on its own (real stages and an attacker can share a NAT IP),
//! which is why the global pending pool is separate from the main cap.
//!
//! Used by: `tcp_transport::accept_loop`. The type is `pub(crate)` with a
//! small API so another authenticated listener inside `distributed` (for
//! example a coordinator control channel) can reuse it.
//!
//! [`HANDSHAKE_TIMEOUT`]: super::cluster_handshake::HANDSHAKE_TIMEOUT

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, Mutex};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Default number of handshakes in flight across all sources.
pub const DEFAULT_MAX_PENDING_HANDSHAKES: usize = 64;
/// Default number of handshakes in flight from one source IP.
pub const DEFAULT_MAX_PENDING_HANDSHAKES_PER_IP: usize = 8;

/// Why a socket was refused a pending-handshake slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PendingSlotRefusal {
    /// The global pending pool is full.
    Global { max: usize },
    /// This source IP already holds its share of the pending pool.
    PerIp { ip: IpAddr, max: usize },
}

impl std::fmt::Display for PendingSlotRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Global { max } => write!(f, "max pending handshakes ({max}) reached"),
            Self::PerIp { ip, max } => {
                write!(f, "max pending handshakes from {ip} ({max}) reached")
            }
        }
    }
}

/// Shared pending-handshake accounting for one listener. Cheap to clone
/// (both pools are behind `Arc`).
#[derive(Clone)]
pub(crate) struct PendingHandshakeLimiter {
    global: Arc<Semaphore>,
    per_ip: Arc<Mutex<HashMap<IpAddr, usize>>>,
    max_total: usize,
    max_per_ip: usize,
}

impl PendingHandshakeLimiter {
    pub(crate) fn new(max_total: usize, max_per_ip: usize) -> Self {
        Self {
            global: Arc::new(Semaphore::new(max_total)),
            per_ip: Arc::new(Mutex::new(HashMap::new())),
            max_total,
            max_per_ip,
        }
    }

    /// Reserve a slot for a socket from `peer_ip`, or say why it cannot have
    /// one. Never waits: a full pool means the socket is dropped at accept.
    ///
    /// IPv4-mapped IPv6 sources count against their IPv4 key
    /// (`to_canonical`), the same key the handshake binds identities to.
    pub(crate) fn try_acquire(
        &self,
        peer_ip: IpAddr,
    ) -> Result<PendingHandshakeSlot, PendingSlotRefusal> {
        let ip = peer_ip.to_canonical();
        let per_ip_guard = {
            let mut counts = self.per_ip.lock().unwrap_or_else(|e| e.into_inner());
            let count = counts.entry(ip).or_insert(0);
            if *count >= self.max_per_ip {
                if *count == 0 {
                    counts.remove(&ip);
                }
                return Err(PendingSlotRefusal::PerIp {
                    ip,
                    max: self.max_per_ip,
                });
            }
            *count += 1;
            PerIpGuard {
                counts: Arc::clone(&self.per_ip),
                ip,
            }
        };
        let permit = Arc::clone(&self.global).try_acquire_owned().map_err(|_| {
            PendingSlotRefusal::Global {
                max: self.max_total,
            }
        })?;
        Ok(PendingHandshakeSlot {
            _permit: permit,
            _per_ip: per_ip_guard,
        })
    }

    /// Current number of pending handshakes from `peer_ip` (tests).
    #[cfg(test)]
    pub(crate) fn pending_from(&self, peer_ip: IpAddr) -> usize {
        self.per_ip
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&peer_ip.to_canonical())
            .copied()
            .unwrap_or(0)
    }

    /// Current number of pending handshakes across all sources (tests).
    #[cfg(test)]
    pub(crate) fn pending_total(&self) -> usize {
        self.max_total - self.global.available_permits()
    }
}

/// A reserved pending-handshake slot. Dropping it releases both the global
/// permit and the per-IP count; hold it only for the handshake itself.
#[derive(Debug)]
pub(crate) struct PendingHandshakeSlot {
    _permit: OwnedSemaphorePermit,
    _per_ip: PerIpGuard,
}

/// RAII decrement of one source IP's pending count; removes the entry when it
/// reaches zero so the map never grows with past sources.
#[derive(Debug)]
struct PerIpGuard {
    counts: Arc<Mutex<HashMap<IpAddr, usize>>>,
    ip: IpAddr,
}

impl Drop for PerIpGuard {
    fn drop(&mut self) {
        let mut counts = self.counts.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(count) = counts.get_mut(&self.ip) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                counts.remove(&self.ip);
            }
        }
    }
}

#[cfg(test)]
#[path = "pending_handshakes_tests.rs"]
mod tests;
