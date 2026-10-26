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

//! Matching a configured peer address against the authenticated `from`
//! identity the transport reports (#2637).
//!
//! The pipeline runtime and the remote stage service authorize every inbound
//! frame by its source: a reply must come from the stage it was sent to, a
//! command must come from the coordinator, an activation from the adjacent
//! stage. Configured addresses may be hostnames while the transport reports
//! the peer's numeric listener identity, so a matcher resolves the configured
//! name once and compares socket addresses canonically, falling back to a
//! literal string comparison for addresses that do not parse (mock
//! transports in tests use symbolic names). A configured address with an
//! unspecified IP (`0.0.0.0:19000`, which `--pp-auto` emits for a coordinator
//! bound on every interface) names no single host, so it matches any
//! authenticated identity with that port; the handshake has already bound
//! that identity to the peer's real source IP.
//!
//! Used by: pipeline runtime, pipeline remote_service.

use std::net::SocketAddr;

/// One configured peer and the socket addresses it may legitimately appear as.
#[derive(Debug, Clone)]
pub struct PeerMatcher {
    configured: String,
    resolved: Vec<SocketAddr>,
}

fn canonical(addr: SocketAddr) -> SocketAddr {
    SocketAddr::new(addr.ip().to_canonical(), addr.port())
}

impl PeerMatcher {
    /// Build a matcher without DNS: only a literal socket address parses.
    pub fn literal(configured: &str) -> Self {
        let resolved = configured
            .parse::<SocketAddr>()
            .map(|addr| vec![canonical(addr)])
            .unwrap_or_default();
        Self {
            configured: configured.to_string(),
            resolved,
        }
    }

    /// Build a matcher, resolving a `host:port` name through the system
    /// resolver. Resolution failure is not fatal here (the connect path
    /// reports it with a better message); the matcher then only accepts the
    /// literal string.
    pub async fn resolve(configured: &str) -> Self {
        let mut matcher = Self::literal(configured);
        if matcher.resolved.is_empty()
            && let Ok(addrs) = tokio::net::lookup_host(configured).await
        {
            matcher.resolved = addrs.map(canonical).collect();
        }
        matcher
    }

    /// The address as configured (what `Transport::send` is given).
    pub fn configured(&self) -> &str {
        &self.configured
    }

    /// Whether `from` (an authenticated transport identity) is this peer.
    pub fn matches(&self, from: &str) -> bool {
        if from == self.configured {
            return true;
        }
        match from.parse::<SocketAddr>() {
            Ok(addr) => {
                let addr = canonical(addr);
                self.resolved.iter().any(|resolved| {
                    *resolved == addr
                        || (resolved.ip().is_unspecified() && resolved.port() == addr.port())
                })
            }
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_matches_canonical_socket_addresses_and_exact_strings() {
        let matcher = PeerMatcher::literal("127.0.0.1:19001");
        assert!(matcher.matches("127.0.0.1:19001"));
        assert!(matcher.matches("[::ffff:127.0.0.1]:19001"));
        assert!(!matcher.matches("127.0.0.1:19002"));
        assert!(!matcher.matches("127.0.0.2:19001"));
        assert!(!matcher.matches("stage:19001"));

        let any_interface = PeerMatcher::literal("0.0.0.0:19000");
        assert!(any_interface.matches("10.0.0.5:19000"));
        assert!(any_interface.matches("127.0.0.1:19000"));
        assert!(!any_interface.matches("10.0.0.5:19001"));

        let symbolic = PeerMatcher::literal("stage-0:9000");
        assert!(symbolic.matches("stage-0:9000"));
        assert!(!symbolic.matches("stage-1:9000"));
        assert!(!symbolic.matches("127.0.0.1:9000"));
    }

    #[tokio::test]
    async fn resolve_accepts_loopback_name() {
        let matcher = PeerMatcher::resolve("localhost:19001").await;
        assert!(matcher.matches("localhost:19001"));
        assert!(matcher.matches("127.0.0.1:19001") || matcher.matches("[::1]:19001"));
        assert!(!matcher.matches("127.0.0.1:19002"));
    }
}
