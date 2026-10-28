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
//! pending pool, never the slots of established authenticated connections.
//! While the pending pool is full, a new connection from any source
//! (including a real peer reconnecting) is closed at accept, so a flood from
//! enough source IPs delays new connections until slots free up, at most
//! [`HANDSHAKE_TIMEOUT`] per slot. A per-source sub-limit keeps one source
//! from taking the whole pending pool. The source key is the canonical IPv4
//! address, or the /64 prefix of an IPv6 address ([`pending_source_key`]): a
//! host that owns a /64 can otherwise open one pending slot per address and
//! bypass the per-source bound by rotating addresses (#2692). The flip side
//! is that cluster nodes inside one IPv6 /64 share one per-source budget, so
//! a large IPv6 cluster in a single /64 may need a higher
//! `max_pending_handshakes_per_ip`. The sub-limit is still not sufficient on
//! its own (real stages and an attacker can share a NAT IP or a /64), which
//! is why the global pending pool is separate from the main cap.
//!
//! Used by: `tcp_transport::accept_loop`. The type is `pub(crate)` with a
//! small API so another authenticated listener inside `distributed` (for
//! example a coordinator control channel) can reuse it.
//!
//! [`HANDSHAKE_TIMEOUT`]: super::cluster_handshake::HANDSHAKE_TIMEOUT

use std::collections::HashMap;
use std::net::{IpAddr, Ipv6Addr};
use std::sync::{Arc, Mutex};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Default number of handshakes in flight across all sources.
pub const DEFAULT_MAX_PENDING_HANDSHAKES: usize = 64;
/// Default number of handshakes in flight from one source (an IPv4 address
/// or an IPv6 /64, see [`pending_source_key`]).
pub const DEFAULT_MAX_PENDING_HANDSHAKES_PER_IP: usize = 8;

/// The key the per-source pending count is kept under.
///
/// The address is canonicalized first, so an IPv4-mapped IPv6 source keeps
/// counting as its IPv4 address (the same key the handshake binds identities
/// to). An IPv4 address is its own key. An IPv6 address is masked to its /64:
/// a single host or tenant normally holds a whole /64 and can pick any
/// address inside it, so counting per full address would let it take one
/// pending slot per address.
pub(crate) fn pending_source_key(ip: IpAddr) -> IpAddr {
    match ip.to_canonical() {
        v4 @ IpAddr::V4(_) => v4,
        IpAddr::V6(v6) => IpAddr::V6(Ipv6Addr::from(u128::from(v6) & !((1u128 << 64) - 1))),
    }
}

/// Why a socket was refused a pending-handshake slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PendingSlotRefusal {
    /// The global pending pool is full.
    Global { max: usize },
    /// This source already holds its share of the pending pool. `ip` is the
    /// source key ([`pending_source_key`]): the IPv4 address, or the /64
    /// prefix of the IPv6 source.
    PerIp { ip: IpAddr, max: usize },
}

impl std::fmt::Display for PendingSlotRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Global { max } => write!(f, "max pending handshakes ({max}) reached"),
            Self::PerIp {
                ip: ip @ IpAddr::V4(_),
                max,
            } => {
                write!(f, "max pending handshakes from {ip} ({max}) reached")
            }
            Self::PerIp {
                ip: ip @ IpAddr::V6(_),
                max,
            } => {
                write!(f, "max pending handshakes from {ip}/64 ({max}) reached")
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
    /// Build a limiter for one listener. Both bounds must be at least 1: a
    /// zero bound would close every inbound socket at accept, so no peer could
    /// ever authenticate, and that is refused here instead of producing a
    /// listener that silently accepts nobody.
    pub(crate) fn new(max_total: usize, max_per_ip: usize) -> anyhow::Result<Self> {
        if max_total == 0 || max_per_ip == 0 {
            anyhow::bail!(
                "max_pending_handshakes ({max_total}) and max_pending_handshakes_per_ip \
                 ({max_per_ip}) must both be at least 1; with 0 the authenticated listener \
                 would refuse every connection"
            );
        }
        Ok(Self {
            global: Arc::new(Semaphore::new(max_total)),
            per_ip: Arc::new(Mutex::new(HashMap::new())),
            max_total,
            max_per_ip,
        })
    }

    /// Reserve a slot for a socket from `peer_ip`, or say why it cannot have
    /// one. Never waits: a full pool means the socket is dropped at accept.
    ///
    /// The per-source count is keyed by [`pending_source_key`]: IPv4-mapped
    /// IPv6 sources count against their IPv4 address and IPv6 sources against
    /// their /64.
    pub(crate) fn try_acquire(
        &self,
        peer_ip: IpAddr,
    ) -> Result<PendingHandshakeSlot, PendingSlotRefusal> {
        let ip = pending_source_key(peer_ip);
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

    /// Current number of pending handshakes from `peer_ip`'s source key
    /// (tests).
    #[cfg(test)]
    pub(crate) fn pending_from(&self, peer_ip: IpAddr) -> usize {
        self.per_ip
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&pending_source_key(peer_ip))
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

/// RAII decrement of one source key's pending count; removes the entry when
/// it reaches zero so the map never grows with past sources.
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
