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

//! Operator-configured NAT64 prefixes for the media address policy (issue #2693).
//!
//! RFC 8215 reserves `64:ff9b:1::/48` for local-use NAT64 at any RFC 6052
//! prefix length, and an operator may also translate through a network-specific
//! prefix carved from its own global space. Neither says where the IPv4 address
//! sits, so the address alone cannot be judged: by default the local-use block
//! is refused outright, and `MLXCEL_MEDIA_NAT64_PREFIXES` names the prefixes
//! whose addresses are judged by the IPv4 address they embed instead.

use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::{PoisonError, RwLock};

/// Comma-separated NAT64 prefixes (`addr/len`) whose embedded IPv4 decides.
pub(crate) const MEDIA_NAT64_PREFIXES_ENV: &str = "MLXCEL_MEDIA_NAT64_PREFIXES";

/// The prefix lengths RFC 6052 section 2.2 defines.
const ALLOWED_LENGTHS: [u8; 6] = [32, 40, 48, 56, 64, 96];

/// The RFC 6052 section 2.1 well-known prefix `64:ff9b::/96`, which is always decoded.
const WELL_KNOWN: Ipv6Addr = Ipv6Addr::new(0x64, 0xff9b, 0, 0, 0, 0, 0, 0);

/// Octet index of the RFC 6052 `u` octet (bits 64..72), which must be zero.
const U_OCTET: usize = 8;

/// One RFC 6052 NAT64 prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Nat64Prefix {
    network: Ipv6Addr,
    len: u8,
}

impl Nat64Prefix {
    /// Parse `addr/len`, with `len` one of the RFC 6052 lengths.
    ///
    /// # Errors
    ///
    /// A message naming the entry when it is not `addr/len`, the address does
    /// not parse as IPv6, the length is not an RFC 6052 length, bits past the
    /// length are set, or a /96 prefix sets the reserved `u` octet.
    pub(crate) fn parse(raw: &str) -> Result<Self, String> {
        let entry = raw.trim();
        let (addr, len) = entry
            .split_once('/')
            .ok_or_else(|| format!("{entry:?} is not an IPv6 prefix of the form addr/len"))?;
        let network: Ipv6Addr = addr
            .parse()
            .map_err(|_| format!("{entry:?} does not name an IPv6 address"))?;
        let len: u8 = len
            .parse()
            .ok()
            .filter(|len| ALLOWED_LENGTHS.contains(len))
            .ok_or_else(|| {
                format!("{entry:?} has a prefix length other than 32, 40, 48, 56, 64 or 96")
            })?;
        if u128::from(network) & !mask(len) != 0 {
            return Err(format!(
                "{entry:?} sets address bits past its /{len} length"
            ));
        }
        if len < 96 && u128::from(WELL_KNOWN) & mask(len) == u128::from(network) {
            return Err(format!(
                "{entry:?} covers the well-known prefix 64:ff9b::/96, which RFC 6052 \
                 section 2.1 defines at /96 only; a shorter length would misdecode it"
            ));
        }
        if network.octets()[U_OCTET] != 0 {
            return Err(format!(
                "{entry:?} sets bits 64..72, which RFC 6052 reserves as zero"
            ));
        }
        Ok(Self { network, len })
    }

    /// Build a prefix without the well-known overlap check (tests only), to
    /// model an operator prefix that predates it.
    #[cfg(test)]
    pub(crate) fn parse_unchecked_for_test(raw: &str) -> Self {
        let (addr, len) = raw.split_once('/').expect("addr/len");
        Self {
            network: addr.parse().expect("IPv6 address"),
            len: len.parse().expect("length"),
        }
    }

    /// True when `addr` falls inside this prefix.
    #[must_use]
    pub(crate) fn contains(&self, addr: Ipv6Addr) -> bool {
        u128::from(addr) & mask(self.len) == u128::from(self.network)
    }

    /// The IPv4 address `addr` carries under this prefix (RFC 6052 section 2.2).
    ///
    /// `None` when `addr` sets the `u` octet, which no conforming translator
    /// produces, so the caller refuses it rather than guessing.
    #[must_use]
    pub(crate) fn embedded_ipv4(&self, addr: Ipv6Addr) -> Option<Ipv4Addr> {
        let octets = addr.octets();
        if octets[U_OCTET] != 0 {
            return None;
        }
        let mut v4 = [0u8; 4];
        let positions = (usize::from(self.len) / 8..16).filter(|&index| index != U_OCTET);
        for (slot, index) in v4.iter_mut().zip(positions) {
            *slot = octets[index];
        }
        Some(Ipv4Addr::from(v4))
    }
}

impl std::fmt::Display for Nat64Prefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.network, self.len)
    }
}

/// The network mask of a `len`-bit prefix.
fn mask(len: u8) -> u128 {
    match len {
        0 => 0,
        len => u128::MAX << (128 - u32::from(len)),
    }
}

/// Parse the value of [`MEDIA_NAT64_PREFIXES_ENV`].
///
/// Whitespace around entries is trimmed and empty entries are ignored, so an
/// empty value means no prefixes.
///
/// # Errors
///
/// A message naming the variable and the first entry that does not parse.
pub(crate) fn parse_media_nat64_prefixes(raw: &str) -> Result<Vec<Nat64Prefix>, String> {
    raw.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            Nat64Prefix::parse(entry)
                .map_err(|reason| format!("invalid {MEDIA_NAT64_PREFIXES_ENV} entry: {reason}"))
        })
        .collect()
}

/// The configured prefixes, installed once at startup.
static PREFIXES: RwLock<Vec<Nat64Prefix>> = RwLock::new(Vec::new());

/// Install `prefixes` as the process-wide NAT64 configuration.
pub(crate) fn configure_media_nat64_prefixes(prefixes: Vec<Nat64Prefix>) {
    *PREFIXES.write().unwrap_or_else(PoisonError::into_inner) = prefixes;
}

/// Read [`MEDIA_NAT64_PREFIXES_ENV`] and install what it names.
///
/// Called once at startup next to `configure_private_media_urls`. Unset means
/// no prefixes, which leaves the local-use block refused.
///
/// # Errors
///
/// A message naming the variable when it is not valid UTF-8 or an entry does
/// not parse; startup fails rather than leaving a mistyped translator open.
pub fn configure_media_nat64_prefixes_from_env() -> Result<(), String> {
    let raw = match std::env::var(MEDIA_NAT64_PREFIXES_ENV) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => String::new(),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(format!("{MEDIA_NAT64_PREFIXES_ENV} is not valid UTF-8"));
        }
    };
    let prefixes = parse_media_nat64_prefixes(&raw)?;
    if !prefixes.is_empty() {
        let listed: Vec<String> = prefixes.iter().map(ToString::to_string).collect();
        tracing::info!(
            "media URL NAT64 prefixes ({MEDIA_NAT64_PREFIXES_ENV}): {}; addresses inside them \
             are judged by their embedded IPv4 address",
            listed.join(", ")
        );
    }
    configure_media_nat64_prefixes(prefixes);
    Ok(())
}

/// Run `judge` against the configured prefixes.
pub(crate) fn with_configured_prefixes<T>(judge: impl FnOnce(&[Nat64Prefix]) -> T) -> T {
    judge(&PREFIXES.read().unwrap_or_else(PoisonError::into_inner))
}
