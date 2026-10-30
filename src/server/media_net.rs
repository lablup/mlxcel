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
//! # The checks
//!
//! 1. **Before the request.** The scheme must be `http` or `https`, the URL may
//!    not carry credentials, and a host written as an IP literal is checked
//!    directly. A host written as a name is resolved and every address it
//!    resolves to must be allowed, so a name that maps to `127.0.0.1` is
//!    refused before a socket is opened.
//! 2. **At every name resolution.** The shared media client resolves names
//!    through [`PolicyResolver`], which drops every non-public address and
//!    fails the connection when none is left. Every connection the client
//!    opens to a name directly goes through it, which closes the window
//!    between a check's own resolution and the connect.
//! 3. **On every redirect.** The client follows no redirect itself;
//!    [`send_following_redirects`] does, and runs check 1 in full (shape,
//!    local resolution, every address) on each hop before a byte is sent
//!    toward it, whether or not a proxy carries the hop. A public origin that
//!    answers `302` with `http://intranet.internal/` or
//!    `http://169.254.169.254/` is refused there. The chain stays bounded at
//!    five requests and one shared deadline (issue #2693).
//! 4. **After the connection.** [`reqwest::Response::remote_addr`] reports the
//!    peer the final response actually came from, and it is re-checked before
//!    the body is read. With checks 2 and 3 in place this is defence in depth.
//!
//! # What the client is told
//!
//! A refused or unresolvable host renders as one sentence,
//! [`MEDIA_HOST_REFUSED`], whatever the reason, so a caller cannot use the
//! response to learn whether an internal name exists or what it resolves to
//! (issue #2644). The host, the resolved address and the opt-in hint go to the
//! server log at `debug` through [`MediaUrlError::log_detail`], the same split
//! #1612 made for `--media-path` candidates.
//!
//! # Proxies
//!
//! The media client honours the `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` /
//! `NO_PROXY` environment, as the corporate-proxy deployments
//! `MLXCEL_EXTRA_CA_CERTS` exists for need. Through a proxy, reqwest resolves
//! and dials only the proxy's own host, so checks 2 and 4 apply to the proxy
//! endpoint, and a proxy on a private address already needs the opt-in below.
//! Checks 1 and 3 do not depend on the transport: the first URL and every
//! redirect hop are resolved and checked locally before they are requested,
//! proxied or not. The proxy then resolves the name again itself, so a proxy
//! whose DNS view differs from the host's (split horizon) remains a matter for
//! the proxy's own egress policy. DNS rebinding is the same residual and
//! applies to the first URL as well: a low-TTL name can answer mlxcel with a
//! public address and the proxy with a private one, and the proxy's own egress
//! policy is the backstop. Because the check resolves locally, a host
//! that cannot resolve public names without the proxy already fails check 1;
//! that is unchanged.
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

#[path = "media_net_address.rs"]
mod address;
#[path = "media_net_dns.rs"]
mod dns;
#[path = "media_net_nat64.rs"]
mod nat64;
#[path = "media_net_redirect.rs"]
mod redirect;

pub(crate) use address::is_public_address;
#[cfg(test)]
pub(crate) use address::is_public_address_with;
use dns::lookup_bounded;
#[cfg(test)]
use dns::lookup_bounded_with;
pub use nat64::configure_media_nat64_prefixes_from_env;
#[cfg(test)]
pub(crate) use nat64::{MEDIA_NAT64_PREFIXES_ENV, Nat64Prefix, parse_media_nat64_prefixes};
#[cfg(test)]
pub(crate) use redirect::{MAX_MEDIA_REQUESTS, check_hop_with, send_following_redirects_within};
pub(crate) use redirect::{MEDIA_FETCH_TIMEOUT, MediaSendError, send_following_redirects};

/// Environment variable that disables the private-address refusal.
pub(crate) const ALLOW_PRIVATE_MEDIA_URLS_ENV: &str = "MLXCEL_ALLOW_PRIVATE_MEDIA_URLS";

/// The client-facing text for a host that is refused or does not resolve.
///
/// Both cases render identically, so the response does not say whether a name
/// exists, and it carries neither the host nor any resolved address.
pub(crate) const MEDIA_HOST_REFUSED: &str =
    "media URL host is not allowed or could not be resolved";

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
    /// `host` and `address` are for [`MediaUrlError::log_detail`] only.
    #[error("{MEDIA_HOST_REFUSED}")]
    BlockedAddress { host: String, address: IpAddr },
    /// `host` is for [`MediaUrlError::log_detail`] only.
    #[error("{MEDIA_HOST_REFUSED}")]
    Unresolvable { host: String },
    /// `url` is the hop the origin's `Location` named, for
    /// [`MediaUrlError::log_detail`] only (issue #2728).
    #[error("media URL redirect chain exceeded 5 hops")]
    TooManyRedirects { url: String },
}

impl MediaUrlError {
    /// Log what the client-facing text leaves out, at `debug`.
    ///
    /// Called at the fetch boundaries in `media.rs`, so the operator can still
    /// see which host was refused, what it resolved to, and how to opt in.
    pub(crate) fn log_detail(&self, url: &str) {
        match self {
            Self::BlockedAddress { host, address } => tracing::debug!(
                url,
                host = host.as_str(),
                %address,
                "media URL refused: host resolves to a loopback, link-local, private or \
                 otherwise non-public address; set {ALLOW_PRIVATE_MEDIA_URLS_ENV}=1 to allow \
                 fetching media from inside the deployment's own network"
            ),
            Self::Unresolvable { host } => tracing::debug!(
                url,
                host = host.as_str(),
                "media URL refused: host does not resolve"
            ),
            Self::TooManyRedirects { url: hop } => tracing::debug!(
                url,
                hop = hop.as_str(),
                "media URL refused: redirect chain exceeded 5 hops"
            ),
            _ => {}
        }
    }

    /// The policy refusal somewhere in `err`'s source chain, if any.
    ///
    /// A refusal from [`PolicyResolver`] reaches the caller wrapped in a
    /// `reqwest::Error`; this finds it so the caller reports the refusal
    /// instead of the transport error around it.
    #[must_use]
    pub(crate) fn find_in<'a>(err: &'a (dyn std::error::Error + 'static)) -> Option<&'a Self> {
        let mut current = Some(err);
        while let Some(error) = current {
            if let Some(refusal) = error.downcast_ref::<Self>() {
                return Some(refusal);
            }
            current = error.source();
        }
        None
    }
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
/// Synchronous: this is the half that needs no name resolution, and the first
/// step of [`ensure_media_url_allowed`].
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
    let resolved = lookup_bounded(host, port).await?;
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

#[cfg(test)]
#[path = "media_net_test_guard.rs"]
mod test_guard;
#[cfg(test)]
pub(crate) use test_guard::{allow_private_media_urls_in_tests, deny_private_media_urls_in_tests};

/// Resolve `host` and keep only the addresses a media fetch may reach.
///
/// With `allow_private` every resolved address is kept.
///
/// # Errors
///
/// [`MediaUrlError::Unresolvable`] when the name resolves to nothing, and
/// [`MediaUrlError::BlockedAddress`] when it resolves only to refused
/// addresses.
pub(crate) async fn resolve_allowed_addresses(
    host: &str,
    allow_private: bool,
) -> Result<Vec<SocketAddr>, MediaUrlError> {
    // Port 0: hyper-util substitutes the URL's port before connecting.
    let resolved = lookup_bounded(host, 0).await?;
    filter_allowed_addresses(host, resolved, allow_private)
}

/// Apply the address policy to a resolved answer set. Pure, no DNS.
///
/// # Errors
///
/// [`MediaUrlError::Unresolvable`] for an empty answer, and
/// [`MediaUrlError::BlockedAddress`] when every address is refused.
pub(crate) fn filter_allowed_addresses(
    host: &str,
    resolved: Vec<SocketAddr>,
    allow_private: bool,
) -> Result<Vec<SocketAddr>, MediaUrlError> {
    let Some(first) = resolved.first().copied() else {
        return Err(MediaUrlError::Unresolvable {
            host: host.to_owned(),
        });
    };
    if allow_private {
        return Ok(resolved);
    }
    let public: Vec<SocketAddr> = resolved
        .into_iter()
        .filter(|addr| is_public_address(addr.ip()))
        .collect();
    if public.is_empty() {
        return Err(MediaUrlError::BlockedAddress {
            host: host.to_owned(),
            address: first.ip(),
        });
    }
    Ok(public)
}

/// DNS resolver for the shared media client (check 2 in the module doc).
///
/// Reads the opt-in on every resolution rather than at construction, because
/// the client is built once per process.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PolicyResolver;

impl reqwest::dns::Resolve for PolicyResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_owned();
        Box::pin(async move {
            let addrs = resolve_allowed_addresses(&host, private_addresses_allowed()).await?;
            Ok(Box::new(addrs.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

#[cfg(test)]
#[path = "media_net_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "media_net_resolver_tests.rs"]
mod resolver_tests;
