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

//! Bounded name resolution for remote media fetches (issues #2728, #2764).
//!
//! Both the pre-flight and per-hop check ([`super::ensure_media_url_allowed`])
//! and the media client's connect-time [`super::PolicyResolver`] resolve
//! through [`lookup_bounded`] (the connect reuses the pre-flight answer when
//! one is pinned, see `media_net_pin.rs`). It keeps the system resolver, so
//! `/etc/hosts` and nsswitch apply as before, and adds a process-wide cap on
//! the blocking threads those lookups may hold, with a smaller cap per domain.
//!
//! # Per-domain fairness and its limit
//!
//! A domain is keyed by the last two labels of the lowercased host, so many
//! random subdomains of one attacker domain share one key and at most
//! [`MAX_CONCURRENT_LOOKUPS_PER_DOMAIN`] of the pool. Filling the whole pool
//! then takes at least `32 / 4 = 8` registered domains whose nameservers stay
//! silent, instead of one.
//!
//! Hosts under a multi-label public suffix (`example.co.uk`, `example.com.au`)
//! share the suffix's key (`co.uk`), so on those suffixes an attacker can make
//! other hosts wait for one of four slots. Keying by the public suffix list
//! was rejected: it adds a dependency for a bound that is approximate anyway.
//! Per-client fairness was rejected because the fetch layer, shared by the
//! chat, embeddings, rerank and image routes, has no client identity.

use std::collections::HashMap;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;

use super::MediaUrlError;

/// Upper bound on one name resolution once its permits are held, matching the
/// media client's connect timeout.
const DNS_LOOKUP_TIMEOUT: Duration = Duration::from_secs(5);

/// Upper bound on waiting for the domain and global permits, before
/// [`DNS_LOOKUP_TIMEOUT`] starts (issue #2764). Worst case per lookup: 7 s.
const DNS_PERMIT_WAIT: Duration = Duration::from_secs(2);

/// Media name resolutions that may occupy a blocking thread at once (issue #2728).
///
/// `getaddrinfo` cannot be cancelled: a timeout drops the caller's future, not
/// the thread, which stays busy until the system resolver gives up. Without a
/// bound, requests naming slow-resolving hosts could hold the whole Tokio
/// blocking pool and starve every other `spawn_blocking` user.
const MAX_CONCURRENT_MEDIA_DNS_LOOKUPS: usize = 32;

/// Of [`MAX_CONCURRENT_MEDIA_DNS_LOOKUPS`], how many one domain key may hold,
/// abandoned lookups included (issue #2764). See the module doc for the key.
const MAX_CONCURRENT_LOOKUPS_PER_DOMAIN: usize = 4;

/// Domain keys kept before idle ones are pruned.
const MAX_IDLE_DOMAIN_KEYS: usize = 1024;

/// The bounds one lookup runs under; [`DnsLimits::DEFAULT`] in production.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DnsLimits {
    /// How long to wait for the domain and global permits together.
    pub(super) permit_wait: Duration,
    /// How long the resolver gets once both permits are held.
    pub(super) lookup_timeout: Duration,
    /// Lookups one domain key may have in flight.
    pub(super) per_domain: usize,
}

impl DnsLimits {
    /// The production bounds.
    pub(super) const DEFAULT: Self = Self {
        permit_wait: DNS_PERMIT_WAIT,
        lookup_timeout: DNS_LOOKUP_TIMEOUT,
        per_domain: MAX_CONCURRENT_LOOKUPS_PER_DOMAIN,
    };
}

/// The permits lookups hold: one global pool and one semaphore per domain key.
#[derive(Debug)]
pub(super) struct DnsPool {
    global: Arc<Semaphore>,
    domains: Mutex<HashMap<String, Arc<Semaphore>>>,
}

impl DnsPool {
    /// A pool of `global` permits with no domain seen yet.
    pub(super) fn new(global: usize) -> Self {
        Self {
            global: Arc::new(Semaphore::new(global)),
            domains: Mutex::new(HashMap::new()),
        }
    }

    /// The semaphore for `key`, created with `per_domain` permits on first use.
    ///
    /// Before a new key is added past [`MAX_IDLE_DOMAIN_KEYS`], keys nobody
    /// holds or waits on are dropped: a held or awaited permit keeps a clone
    /// of the `Arc`, so a strong count of one with every permit available
    /// means the entry is idle and recreating it later loses nothing.
    pub(super) fn domain(&self, key: &str, per_domain: usize) -> Arc<Semaphore> {
        let mut domains = self.domains.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(semaphore) = domains.get(key) {
            return semaphore.clone();
        }
        if domains.len() >= MAX_IDLE_DOMAIN_KEYS {
            domains.retain(|_, semaphore| {
                Arc::strong_count(semaphore) > 1 || semaphore.available_permits() < per_domain
            });
        }
        let semaphore = Arc::new(Semaphore::new(per_domain));
        domains.insert(key.to_owned(), semaphore.clone());
        semaphore
    }

    /// Global permits not held by any lookup (tests only).
    #[cfg(test)]
    pub(super) fn available_global(&self) -> usize {
        self.global.available_permits()
    }

    /// Domain keys currently tracked (tests only).
    #[cfg(test)]
    pub(super) fn tracked_domains(&self) -> usize {
        self.domains
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

/// The process-wide pool behind [`MAX_CONCURRENT_MEDIA_DNS_LOOKUPS`].
fn media_dns_pool() -> &'static DnsPool {
    static POOL: OnceLock<DnsPool> = OnceLock::new();
    POOL.get_or_init(|| DnsPool::new(MAX_CONCURRENT_MEDIA_DNS_LOOKUPS))
}

/// The fairness key of `host`: its last two labels, lowercased, ignoring a
/// trailing dot (`A.Slow.Test.` keys as `slow.test`). A host of two labels or
/// fewer is its own key.
pub(super) fn domain_key(host: &str) -> String {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    match host.rmatch_indices('.').nth(1) {
        Some((dot, _)) => host[dot + 1..].to_owned(),
        None => host,
    }
}

/// The system resolver (`getaddrinfo`, honouring `/etc/hosts` and nsswitch).
fn system_resolve(host: &str, port: u16) -> std::io::Result<Vec<SocketAddr>> {
    #[cfg(test)]
    if let Some(answer) = test_answers::answer(host, port) {
        return Ok(answer);
    }
    (host, port).to_socket_addrs().map(Iterator::collect)
}

/// The system resolver under [`DnsLimits::DEFAULT`] and the process-wide
/// pool; see [`lookup_bounded_with`].
pub(super) async fn lookup_bounded(
    host: &str,
    port: u16,
) -> Result<Vec<SocketAddr>, MediaUrlError> {
    lookup_bounded_with(
        system_resolve,
        host,
        port,
        DnsLimits::DEFAULT,
        media_dns_pool(),
    )
    .await
}

/// Run the blocking `resolve` for `host:port` on a blocking thread while
/// holding a permit for `host`'s domain key and one from the global pool.
///
/// Both permits are acquired, domain first, within `limits.permit_wait`; the
/// resolver then gets the whole `limits.lookup_timeout`, so waiting for a
/// permit under contention does not shorten a healthy lookup. The permits are
/// moved into the blocking closure, so they are released only when `resolve`
/// returns, not when this future is dropped: a lookup whose caller already
/// gave up still counts against its domain and the pool while its thread is
/// busy.
///
/// # Errors
///
/// [`MediaUrlError::Unresolvable`] when no permit frees up in time, the
/// resolver fails, or the lookup outlives `limits.lookup_timeout`.
pub(super) async fn lookup_bounded_with<R>(
    resolve: R,
    host: &str,
    port: u16,
    limits: DnsLimits,
    pool: &DnsPool,
) -> Result<Vec<SocketAddr>, MediaUrlError>
where
    R: FnOnce(&str, u16) -> std::io::Result<Vec<SocketAddr>> + Send + 'static,
{
    let unresolvable = || MediaUrlError::Unresolvable {
        host: host.to_owned(),
    };
    let permit_deadline = Instant::now() + limits.permit_wait;
    let key = domain_key(host);
    let domain = pool.domain(&key, limits.per_domain);
    let Some(domain_permit) = acquire_by(permit_deadline, domain, host, "domain").await else {
        return Err(unresolvable());
    };
    let Some(global_permit) =
        acquire_by(permit_deadline, pool.global.clone(), host, "global").await
    else {
        return Err(unresolvable());
    };
    let owned_host = host.to_owned();
    let lookup = tokio::task::spawn_blocking(move || {
        let _permits = (domain_permit, global_permit);
        resolve(&owned_host, port)
    });
    match tokio::time::timeout(limits.lookup_timeout, lookup).await {
        Ok(Ok(Ok(addrs))) => Ok(addrs),
        Ok(Ok(Err(_))) | Ok(Err(_)) | Err(_) => Err(unresolvable()),
    }
}

/// One permit from `semaphore` by `deadline`, or `None` (logged at `debug`).
async fn acquire_by(
    deadline: Instant,
    semaphore: Arc<Semaphore>,
    host: &str,
    which: &'static str,
) -> Option<OwnedSemaphorePermit> {
    match tokio::time::timeout_at(deadline, semaphore.acquire_owned()).await {
        Ok(Ok(permit)) => Some(permit),
        Ok(Err(_closed)) => None,
        Err(_elapsed) => {
            tracing::debug!(
                host,
                pool = which,
                "media DNS lookup permits saturated; none freed up before the permit-wait deadline"
            );
            None
        }
    }
}

/// Canned answers for names no real resolver knows, and a count of how often
/// [`system_resolve`] was asked for each (tests only).
///
/// Keyed by the exact host, so a test that registers names of its own cannot
/// disturb another test's lookups.
#[cfg(test)]
pub(crate) mod test_answers {
    use std::collections::HashMap;
    use std::net::{IpAddr, SocketAddr};
    use std::sync::{LazyLock, Mutex, PoisonError};

    static ANSWERS: LazyLock<Mutex<HashMap<String, (Vec<IpAddr>, usize)>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    /// Answer every later lookup of `host` with `addrs`.
    pub(crate) fn register(host: &str, addrs: Vec<IpAddr>) {
        ANSWERS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(host.to_owned(), (addrs, 0));
    }

    /// How many times [`super::system_resolve`] was asked for `host`.
    pub(crate) fn lookups(host: &str) -> usize {
        ANSWERS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(host)
            .map_or(0, |(_, count)| *count)
    }

    pub(super) fn answer(host: &str, port: u16) -> Option<Vec<SocketAddr>> {
        let mut answers = ANSWERS.lock().unwrap_or_else(PoisonError::into_inner);
        let (addrs, count) = answers.get_mut(host)?;
        *count += 1;
        Some(addrs.iter().map(|ip| SocketAddr::new(*ip, port)).collect())
    }
}
