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

//! Bounded name resolution for remote media fetches (issue #2728).
//!
//! Both the pre-flight and per-hop check ([`super::ensure_media_url_allowed`])
//! and the media client's connect-time [`super::PolicyResolver`] resolve
//! through [`lookup_bounded`]. It keeps the system resolver, so `/etc/hosts`
//! and nsswitch apply as before, and adds a process-wide cap on the blocking
//! threads those lookups may hold.

use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use tokio::sync::Semaphore;
use tokio::time::Instant;

use super::MediaUrlError;

/// Upper bound on one name resolution, matching the media client's connect timeout.
const DNS_LOOKUP_TIMEOUT: Duration = Duration::from_secs(5);

/// Media name resolutions that may occupy a blocking thread at once (issue #2728).
///
/// `getaddrinfo` cannot be cancelled: a timeout drops the caller's future, not
/// the thread, which stays busy until the system resolver gives up. Without a
/// bound, requests naming slow-resolving hosts could hold the whole Tokio
/// blocking pool and starve every other `spawn_blocking` user.
const MAX_CONCURRENT_MEDIA_DNS_LOOKUPS: usize = 32;

/// The process-wide permit pool behind [`MAX_CONCURRENT_MEDIA_DNS_LOOKUPS`].
fn media_dns_permits() -> Arc<Semaphore> {
    static PERMITS: OnceLock<Arc<Semaphore>> = OnceLock::new();
    PERMITS
        .get_or_init(|| Arc::new(Semaphore::new(MAX_CONCURRENT_MEDIA_DNS_LOOKUPS)))
        .clone()
}

/// The system resolver (`getaddrinfo`, honouring `/etc/hosts` and nsswitch).
fn system_resolve(host: &str, port: u16) -> std::io::Result<Vec<SocketAddr>> {
    (host, port).to_socket_addrs().map(Iterator::collect)
}

/// The system resolver bounded by [`DNS_LOOKUP_TIMEOUT`] and the shared
/// permit pool; see [`lookup_bounded_with`].
pub(super) async fn lookup_bounded(
    host: &str,
    port: u16,
) -> Result<Vec<SocketAddr>, MediaUrlError> {
    lookup_bounded_with(
        system_resolve,
        host,
        port,
        DNS_LOOKUP_TIMEOUT,
        media_dns_permits(),
    )
    .await
}

/// Run the blocking `resolve` for `host:port` on a blocking thread, within
/// `timeout` and while holding one of `permits`.
///
/// The permit is acquired under the same deadline as the lookup, then moved
/// into the blocking closure, so it is released only when `resolve` returns,
/// not when this future is dropped: a lookup whose caller already gave up
/// still counts against the pool while its thread is busy.
///
/// # Errors
///
/// [`MediaUrlError::Unresolvable`] when no permit frees up in time, the
/// resolver fails, or the lookup outlives `timeout`.
pub(super) async fn lookup_bounded_with<R>(
    resolve: R,
    host: &str,
    port: u16,
    timeout: Duration,
    permits: Arc<Semaphore>,
) -> Result<Vec<SocketAddr>, MediaUrlError>
where
    R: FnOnce(&str, u16) -> std::io::Result<Vec<SocketAddr>> + Send + 'static,
{
    let unresolvable = || MediaUrlError::Unresolvable {
        host: host.to_owned(),
    };
    let deadline = Instant::now() + timeout;
    let permit = match tokio::time::timeout_at(deadline, permits.acquire_owned()).await {
        Ok(Ok(permit)) => permit,
        Ok(Err(_closed)) => return Err(unresolvable()),
        Err(_elapsed) => {
            tracing::debug!(
                host,
                "media DNS lookup pool saturated; no resolver thread freed up before the deadline"
            );
            return Err(unresolvable());
        }
    };
    let owned_host = host.to_owned();
    let lookup = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        resolve(&owned_host, port)
    });
    match tokio::time::timeout_at(deadline, lookup).await {
        Ok(Ok(Ok(addrs))) => Ok(addrs),
        Ok(Ok(Err(_))) | Ok(Err(_)) | Err(_) => Err(unresolvable()),
    }
}
