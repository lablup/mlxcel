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

//! Network-address policy for remote media URLs (issue #1451).
//!
//! A request may carry `{"type": "image_url", "image_url": {"url": "http://..."}}`
//! and the server fetches it. That makes the server a fetch proxy an
//! unauthenticated caller steers, which is the classic server-side request
//! forgery shape: `http://127.0.0.1:9090/admin`,
//! `http://169.254.169.254/latest/meta-data/iam/security-credentials/` on EC2,
//! `http://metadata.google.internal/` on GCE, or any address inside the
//! deployment's own network. b10621 fetches with `common_remote_get_content`
//! and applies no address policy at all; mlxcel refuses these addresses, which
//! is recorded as a divergence on the media entries it applies to.
//!
//! # The three checks
//!
//! 1. **Before the request.** The scheme must be `http` or `https`, the URL may
//!    not carry credentials, and a host written as an IP literal is checked
//!    directly. A host written as a name is resolved and every address it
//!    resolves to must be allowed, so a name that maps to `127.0.0.1` is
//!    refused before a socket is opened.
//! 2. **On every redirect.** The redirect policy is consulted with the *next*
//!    URL before it is followed, so a public origin cannot bounce the fetch to
//!    an internal one. Hop count stays bounded at five, as before.
//! 3. **After the connection.** [`reqwest::Response::remote_addr`] reports the
//!    peer the request actually reached. Re-checking it before the body is read
//!    closes the DNS-rebinding window that checks 1 and 2 leave open: a name
//!    that resolves publicly at check time and privately at connect time is
//!    caught here, after connecting but before any bytes are consumed.
//!
//! # The escape hatch
//!
//! A deployment that genuinely serves its media from an internal object store
//! sets `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1`, which turns every address check
//! into a pass. It is off by default because a fetch proxy that reaches the
//! private network has to be an explicit operator decision.
//!
//! Used by: chat / responses / anthropic image and audio parts, embeddings and
//! rerank image inputs, request-path video fetches.

use std::net::{IpAddr, SocketAddr};

/// Environment variable that disables the private-address refusal.
pub(crate) const ALLOW_PRIVATE_MEDIA_URLS_ENV: &str = "MLXCEL_ALLOW_PRIVATE_MEDIA_URLS";

/// Why a remote media URL was refused before, during, or after the fetch.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum MediaUrlError {
    #[error("media URL is not a valid absolute URL: {url}")]
    Malformed { url: String },
    #[error("media URL scheme {scheme:?} is not supported; only http and https are fetched")]
    Scheme { scheme: String },
    #[error("media URL must not carry credentials")]
    Credentials,
    #[error("media URL has no host")]
    NoHost,
    #[error(
        "media URL host {host} resolves to {address}, which is a loopback, link-local, private \
         or otherwise non-public address; set {ALLOW_PRIVATE_MEDIA_URLS_ENV}=1 to allow fetching \
         media from inside the deployment's own network"
    )]
    BlockedAddress { host: String, address: IpAddr },
    #[error("media URL host {host} does not resolve")]
    Unresolvable { host: String },
    #[error("media URL redirect chain exceeded 5 hops at {url}")]
    TooManyRedirects { url: String },
}

/// Whether private addresses are reachable, resolved once at startup.
///
/// Read from an atomic rather than from the environment on every fetch: the
/// setting is an operator decision that predates any request, and a per-request
/// `std::env::var` in the hot path would also be an unsynchronized environment
/// read next to whatever else a process is doing with `set_var`.
static ALLOW_PRIVATE_ADDRESSES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Record whether media fetches may reach private addresses.
pub fn configure_private_media_urls(allowed: bool) {
    ALLOW_PRIVATE_ADDRESSES.store(allowed, std::sync::atomic::Ordering::Relaxed);
}

/// Read the opt-in from the environment, for startup to hand to
/// [`configure_private_media_urls`].
#[must_use]
pub fn private_media_urls_allowed_from_env() -> bool {
    std::env::var(ALLOW_PRIVATE_MEDIA_URLS_ENV)
        .is_ok_and(|value| matches!(value.trim(), "1" | "true" | "on" | "yes"))
}

/// True when the operator opted into fetching media from private addresses.
fn private_addresses_allowed() -> bool {
    ALLOW_PRIVATE_ADDRESSES.load(std::sync::atomic::Ordering::Relaxed)
}

/// True when `addr` is a public unicast address a media fetch may reach.
///
/// Everything else is refused: loopback, the unspecified address, multicast and
/// broadcast, IPv4 private and link-local ranges (which is what makes
/// `169.254.169.254` unreachable), carrier-grade NAT, the documentation and
/// benchmark ranges, IPv6 unique-local and link-local, and any IPv4-mapped or
/// IPv4-compatible IPv6 form of a refused IPv4 address.
#[must_use]
pub(crate) fn is_public_address(addr: IpAddr) -> bool {
    match addr {
        IpAddr::V4(v4) => {
            {
                let [a, b, c, _] = v4.octets();
                // 100.64.0.0/10 carrier-grade NAT, 192.0.0.0/24 IETF protocol
                // assignments, 198.18.0.0/15 benchmarking, 192.0.2.0/24 /
                // 198.51.100.0/24 / 203.0.113.0/24 documentation, 240.0.0.0/4
                // reserved. `Ipv4Addr::is_global` and `is_documentation` cover
                // these on nightly only, so they are spelled out here, each at
                // its exact prefix so no ordinary public address is refused.
                let reserved = (a == 100 && (64..128).contains(&b))
                    || (a == 192 && b == 0 && c == 0)
                    || (a == 192 && b == 0 && c == 2)
                    || (a == 198 && (18..20).contains(&b))
                    || (a == 198 && b == 51 && c == 100)
                    || (a == 203 && b == 0 && c == 113)
                    || a >= 240;
                if reserved {
                    return false;
                }
            }
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_multicast()
                || v4.is_broadcast())
        }
        IpAddr::V6(v6) => {
            // The IPv6-native rules come first. `::1` and `::` both live inside
            // the deprecated IPv4-compatible `::/96` block, so converting to
            // IPv4 before testing them would turn `::1` into `0.0.0.1` and let
            // loopback through.
            if v6.is_loopback() || v6.is_unspecified() || v6.is_multicast() {
                return false;
            }
            // An IPv4 address written in IPv4-mapped form must be judged by its
            // IPv4 rules, or `::ffff:127.0.0.1` walks straight past them.
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_public_address(IpAddr::V4(mapped));
            }
            let segments = v6.segments();
            // The rest of `::/96` is the deprecated IPv4-compatible form, which
            // no reachable public host uses; refuse the whole block rather than
            // reasoning about the embedded address.
            if segments[..6] == [0, 0, 0, 0, 0, 0] {
                return false;
            }
            // fc00::/7 unique local, fe80::/10 link local, 2001:db8::/32
            // documentation.
            let unique_local = segments[0] & 0xfe00 == 0xfc00;
            let link_local = segments[0] & 0xffc0 == 0xfe80;
            let documentation = segments[0] == 0x2001 && segments[1] == 0x0db8;
            !(unique_local || link_local || documentation)
        }
    }
}

/// The IP address a host component names, when it is written as a literal.
///
/// `Url::host_str` renders an IPv6 literal in its bracketed form, so the
/// brackets are stripped before parsing.
#[must_use]
fn host_ip_literal(host: &str) -> Option<IpAddr> {
    let unbracketed = host
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(host);
    unbracketed.parse::<IpAddr>().ok()
}

/// Validate the shape of a media URL and its host when written as an IP literal.
///
/// Synchronous: this is the half that needs no name resolution, so it can run
/// inside a redirect policy callback.
///
/// # Errors
///
/// Returns the first violation among scheme, credentials, missing host, and a
/// blocked IP literal.
pub(crate) fn ensure_media_url_shape(url: &reqwest::Url) -> Result<(), MediaUrlError> {
    ensure_media_url_shape_with(url, private_addresses_allowed())
}

/// [`ensure_media_url_shape`] against an explicit opt-in value.
///
/// The policy tests call this rather than the global-reading wrapper, so a test
/// that flips the process-wide opt-in for its own loopback origin cannot make
/// them pass vacuously.
pub(crate) fn ensure_media_url_shape_with(
    url: &reqwest::Url,
    allow_private: bool,
) -> Result<(), MediaUrlError> {
    match url.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(MediaUrlError::Scheme {
                scheme: scheme.to_owned(),
            });
        }
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(MediaUrlError::Credentials);
    }
    let Some(host) = url.host_str() else {
        return Err(MediaUrlError::NoHost);
    };
    if allow_private {
        return Ok(());
    }
    if let Some(address) = host_ip_literal(host)
        && !is_public_address(address)
    {
        return Err(MediaUrlError::BlockedAddress {
            host: host.to_owned(),
            address,
        });
    }
    Ok(())
}

/// Validate a media URL's shape and every address its host resolves to.
///
/// # Errors
///
/// Returns the shape violations of [`ensure_media_url_shape`], plus a blocked
/// resolved address or an unresolvable host.
pub(crate) async fn ensure_media_url_allowed(raw: &str) -> Result<reqwest::Url, MediaUrlError> {
    let url = reqwest::Url::parse(raw).map_err(|_| MediaUrlError::Malformed {
        url: raw.to_owned(),
    })?;
    ensure_media_url_shape(&url)?;
    if private_addresses_allowed() {
        return Ok(url);
    }
    let Some(host) = url.host_str() else {
        return Err(MediaUrlError::NoHost);
    };
    if host_ip_literal(host).is_some() {
        // Already checked as a literal; there is nothing to resolve.
        return Ok(url);
    }
    let port = url.port_or_known_default().unwrap_or(80);
    let resolved = tokio::net::lookup_host((host.to_owned(), port))
        .await
        .map_err(|_| MediaUrlError::Unresolvable {
            host: host.to_owned(),
        })?;
    let mut any = false;
    for addr in resolved {
        any = true;
        ensure_socket_addr_allowed(host, addr)?;
    }
    if !any {
        return Err(MediaUrlError::Unresolvable {
            host: host.to_owned(),
        });
    }
    Ok(url)
}

/// Validate one concrete peer address, used both before and after connecting.
///
/// # Errors
///
/// Returns [`MediaUrlError::BlockedAddress`] for a non-public address.
pub(crate) fn ensure_socket_addr_allowed(
    host: &str,
    addr: SocketAddr,
) -> Result<(), MediaUrlError> {
    ensure_socket_addr_allowed_with(host, addr, private_addresses_allowed())
}

/// [`ensure_socket_addr_allowed`] against an explicit opt-in value.
pub(crate) fn ensure_socket_addr_allowed_with(
    host: &str,
    addr: SocketAddr,
    allow_private: bool,
) -> Result<(), MediaUrlError> {
    if allow_private || is_public_address(addr.ip()) {
        return Ok(());
    }
    Err(MediaUrlError::BlockedAddress {
        host: host.to_owned(),
        address: addr.ip(),
    })
}

/// Allow private media addresses for the duration of a test, restoring the
/// previous setting on drop.
///
/// Serialized against every other holder, so a test that stands up a loopback
/// origin cannot overlap one that asserts the refusal.
#[cfg(test)]
#[must_use]
pub(crate) fn allow_private_media_urls_in_tests() -> PrivateAddressTestGuard {
    private_media_urls_in_tests(true)
}

/// Refuse private media addresses for the duration of a test.
///
/// A test that asserts the refusal has to hold the same lock as one that opts
/// in, or the two overlap and the refusal never fires.
#[cfg(test)]
#[must_use]
pub(crate) fn deny_private_media_urls_in_tests() -> PrivateAddressTestGuard {
    private_media_urls_in_tests(false)
}

#[cfg(test)]
fn private_media_urls_in_tests(allowed: bool) -> PrivateAddressTestGuard {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let guard = LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let previous = private_addresses_allowed();
    configure_private_media_urls(allowed);
    PrivateAddressTestGuard {
        _lock: guard,
        previous,
    }
}

/// Restores the process-wide opt-in when dropped.
#[cfg(test)]
pub(crate) struct PrivateAddressTestGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    previous: bool,
}

#[cfg(test)]
impl Drop for PrivateAddressTestGuard {
    fn drop(&mut self) {
        configure_private_media_urls(self.previous);
    }
}

/// Redirect policy for the shared media client.
///
/// Bounds the chain at five hops, as before #1451, and additionally refuses a
/// hop whose target fails [`ensure_media_url_shape`], so a public origin cannot
/// redirect the fetch onto `http://169.254.169.254/` or a `file:` URL. Name
/// resolution is deliberately not attempted here: the policy callback is
/// synchronous, and the post-connect [`ensure_socket_addr_allowed`] check on
/// the final response covers the resolved address.
pub(crate) fn media_redirect_policy() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(|attempt| {
        if attempt.previous().len() >= 5 {
            let url = attempt.url().to_string();
            return attempt.error(MediaUrlError::TooManyRedirects { url });
        }
        match ensure_media_url_shape(attempt.url()) {
            Ok(()) => attempt.follow(),
            Err(err) => attempt.error(err),
        }
    })
}

#[cfg(test)]
#[path = "media_net_tests.rs"]
mod tests;
