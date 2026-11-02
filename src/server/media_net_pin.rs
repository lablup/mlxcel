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

//! Pre-flight answers pinned for the connect (issue #2764).
//!
//! [`super::ensure_media_url_allowed`] resolves a hostname and checks every
//! address before the request; the media client's [`super::PolicyResolver`]
//! then needs the same name again to connect. Resolving twice took two DNS
//! permits per hop and let the answer change in between, so the pre-flight
//! pins the answer it approved and the connect reads it back. The connect
//! still runs the address filter on a pinned answer.
//!
//! Pins are written only when private addresses are refused and the host is a
//! name, and only after every address passed; with the opt-in the pre-flight
//! resolves nothing, so nothing is pinned.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// How long a pinned answer serves the connect.
const PIN_TTL: Duration = Duration::from_secs(30);

/// Pinned hosts kept at once; past it, expired pins go first, then the oldest.
const MAX_PINS: usize = 1024;

#[derive(Debug)]
struct Pin {
    addrs: Vec<SocketAddr>,
    expires: Instant,
}

/// A bounded map from lowercased host to its pinned answer.
#[derive(Debug)]
pub(super) struct PinStore {
    pins: Mutex<HashMap<String, Pin>>,
    ttl: Duration,
    capacity: usize,
}

impl PinStore {
    /// An empty store whose pins live `ttl` and hold at most `capacity` hosts.
    pub(super) fn new(ttl: Duration, capacity: usize) -> Self {
        Self {
            pins: Mutex::new(HashMap::new()),
            ttl,
            capacity,
        }
    }

    /// Pin `addrs` for `host` as of `now`.
    ///
    /// Ports are cleared to 0 so the connector substitutes the URL's own port,
    /// as it does for a fresh connect-time lookup. A new host past `capacity`
    /// first drops expired pins, then the oldest one.
    pub(super) fn pin_at(&self, host: &str, addrs: &[SocketAddr], now: Instant) {
        let addrs = addrs
            .iter()
            .map(|addr| SocketAddr::new(addr.ip(), 0))
            .collect();
        let key = host.to_ascii_lowercase();
        let mut pins = self.pins.lock().unwrap_or_else(PoisonError::into_inner);
        if !pins.contains_key(&key) && pins.len() >= self.capacity {
            pins.retain(|_, pin| pin.expires > now);
            if pins.len() >= self.capacity
                && let Some(oldest) = pins
                    .iter()
                    .min_by_key(|(_, pin)| pin.expires)
                    .map(|(host, _)| host.clone())
            {
                pins.remove(&oldest);
            }
        }
        pins.insert(
            key,
            Pin {
                addrs,
                expires: now + self.ttl,
            },
        );
    }

    /// The answer pinned for `host` that is still live at `now`, if any.
    pub(super) fn get_at(&self, host: &str, now: Instant) -> Option<Vec<SocketAddr>> {
        let key = host.to_ascii_lowercase();
        let mut pins = self.pins.lock().unwrap_or_else(PoisonError::into_inner);
        let pin = pins.get(&key)?;
        if pin.expires <= now {
            pins.remove(&key);
            return None;
        }
        Some(pin.addrs.clone())
    }

    /// Hosts held, expired ones included (tests only).
    #[cfg(test)]
    pub(super) fn host_count(&self) -> usize {
        self.pins
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

/// The process-wide store the pre-flight writes and the connect reads.
static PINS: LazyLock<PinStore> = LazyLock::new(|| PinStore::new(PIN_TTL, MAX_PINS));

/// Pin the pre-flight answer `addrs` for `host`.
pub(super) fn pin_preflight_answer(host: &str, addrs: &[SocketAddr]) {
    PINS.pin_at(host, addrs, Instant::now());
}

/// The live answer the pre-flight pinned for `host`, if any.
pub(super) fn pinned_answer(host: &str) -> Option<Vec<SocketAddr>> {
    PINS.get_at(host, Instant::now())
}
