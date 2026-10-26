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

//! Authentication for the UDP discovery beacon (#2637).
//!
//! A [`DiscoveryBeacon`] is a plaintext JSON datagram broadcast on the LAN.
//! Before this module, the coordinator accepted any datagram whose
//! `cluster_name` matched, so one spoofed packet put an attacker-chosen
//! address into the generated cluster TOML as a pipeline stage. Now every
//! beacon carries a timestamp, a random nonce, and an HMAC-SHA256 tag (keyed
//! by the cluster secret) over a canonical byte encoding of all other fields,
//! and the coordinator's [`BeaconVerifier`] enforces, in order:
//!
//! 1. the datagram parses and names this cluster (anything else is noise
//!    from unrelated software on the same port),
//! 2. the `version` equals this build's version (mixed-version clusters are
//!    not supported: the stage wire protocol is not versioned),
//! 3. the MAC verifies in constant time,
//! 4. the timestamp is within [`BEACON_MAX_SKEW`] of the receiver's clock,
//! 5. the nonce has not been seen within that window (replay),
//! 6. the advertised `peer_addr` IP equals the datagram's source IP, or is
//!    unspecified and is replaced by it.
//!
//! The canonical encoding is length-prefixed field by field rather than a
//! JSON re-serialization, so key order and whitespace cannot change the MAC.
//!
//! Used by: cluster_init (`discover_peers`, `broadcast_beacon_loop`).

use std::collections::HashMap;
use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::cluster_auth::ClusterAuth;
use super::cluster_init::DiscoveryBeacon;

/// Domain-separation label for the beacon MAC.
const BEACON_LABEL: &[u8] = b"mlxcel-discovery-beacon-v1";

/// HMAC-SHA256 tag length in bytes (the hex `mac` field is twice this).
const BEACON_MAC_LEN: usize = 32;

/// Maximum accepted distance between a beacon's timestamp and the receiver's
/// clock, in either direction. Beacons are re-signed every tick (seconds), so
/// a minute tolerates clock drift without leaving a useful replay window.
pub const BEACON_MAX_SKEW: Duration = Duration::from_secs(60);

/// Milliseconds since the Unix epoch, saturating at zero before 1970.
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn push_prefixed(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    out.extend_from_slice(bytes);
}

/// Canonical byte encoding of every authenticated field (everything except
/// `mac` itself).
pub fn canonical_bytes(beacon: &DiscoveryBeacon) -> Vec<u8> {
    let mut out = Vec::with_capacity(128);
    push_prefixed(&mut out, beacon.version.as_bytes());
    push_prefixed(&mut out, beacon.cluster_name.as_bytes());
    push_prefixed(&mut out, beacon.peer_addr.to_string().as_bytes());
    match &beacon.node_id {
        Some(id) => {
            out.push(1);
            push_prefixed(&mut out, id.as_bytes());
        }
        None => out.push(0),
    }
    out.extend_from_slice(&beacon.pid.to_be_bytes());
    out.extend_from_slice(&beacon.timestamp_ms.to_be_bytes());
    out.extend_from_slice(&beacon.nonce.to_be_bytes());
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Decode the beacon's hex MAC. Works on bytes, never on `str` slices: the
/// `mac` field is unauthenticated input, and slicing a string that contains
/// a multi-byte character at an odd offset would panic the coordinator's
/// discovery loop on a single spoofed datagram. Anything other than exactly
/// one HMAC-SHA256 tag of ASCII hex is rejected before it is decoded.
fn unhex(s: &str) -> Option<Vec<u8>> {
    fn nibble(c: u8) -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    }
    let bytes = s.as_bytes();
    if bytes.len() != 2 * BEACON_MAC_LEN {
        return None;
    }
    bytes
        .chunks_exact(2)
        .map(|pair| Some((nibble(pair[0])? << 4) | nibble(pair[1])?))
        .collect()
}

/// Stamp `beacon` with the current time, a fresh nonce, and its MAC.
pub fn sign_beacon(beacon: &mut DiscoveryBeacon, auth: &ClusterAuth, now_ms: u64) {
    beacon.timestamp_ms = now_ms;
    beacon.nonce = auth.random_u64();
    beacon.mac = hex(&auth.sign(BEACON_LABEL, &canonical_bytes(beacon)));
}

/// Why a datagram was not accepted as a peer announcement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeaconRejection {
    /// Not a JSON `DiscoveryBeacon` at all (unrelated traffic on the port).
    Malformed,
    /// Names a different cluster.
    ClusterMismatch { got: String },
    /// Built from a different mlxcel version.
    VersionMismatch { got: String, expected: String },
    /// MAC missing or does not verify under the cluster secret.
    BadMac,
    /// Timestamp outside the skew window.
    StaleTimestamp { beacon_ms: u64, now_ms: u64 },
    /// Nonce already accepted within the skew window.
    ReplayedNonce,
    /// Advertised `peer_addr` IP differs from the datagram source IP.
    SourceMismatch { advertised: IpAddr, source: IpAddr },
}

impl fmt::Display for BeaconRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed => write!(f, "malformed datagram"),
            Self::ClusterMismatch { got } => write!(f, "beacon for other cluster '{got}'"),
            Self::VersionMismatch { got, expected } => {
                write!(
                    f,
                    "peer runs mlxcel {got}, this coordinator runs {expected}"
                )
            }
            Self::BadMac => write!(f, "beacon MAC missing or invalid (cluster secret mismatch)"),
            Self::StaleTimestamp { beacon_ms, now_ms } => write!(
                f,
                "beacon timestamp {beacon_ms} is outside the {}s window around {now_ms}",
                BEACON_MAX_SKEW.as_secs()
            ),
            Self::ReplayedNonce => write!(f, "beacon nonce was already accepted (replay)"),
            Self::SourceMismatch { advertised, source } => write!(
                f,
                "beacon advertises {advertised} but was sent from {source}"
            ),
        }
    }
}

/// A beacon that passed every check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedBeacon {
    /// The peer's address with its IP bound to the datagram source.
    pub peer_addr: SocketAddr,
    pub pid: u32,
    pub node_id: Option<String>,
}

/// Stateful receiver-side verifier (tracks nonces for replay detection).
pub struct BeaconVerifier<'a> {
    auth: &'a ClusterAuth,
    cluster_name: &'a str,
    expected_version: &'a str,
    /// Accepted nonces and the timestamp they arrived with, pruned once
    /// they fall out of the skew window.
    seen_nonces: HashMap<u64, u64>,
}

impl<'a> BeaconVerifier<'a> {
    pub fn new(auth: &'a ClusterAuth, cluster_name: &'a str, expected_version: &'a str) -> Self {
        Self {
            auth,
            cluster_name,
            expected_version,
            seen_nonces: HashMap::new(),
        }
    }

    /// Check one datagram received from `source_ip` at `now_ms`.
    pub fn accept(
        &mut self,
        datagram: &[u8],
        source_ip: IpAddr,
        now_ms: u64,
    ) -> Result<AcceptedBeacon, BeaconRejection> {
        let beacon = serde_json::from_slice::<DiscoveryBeacon>(datagram)
            .map_err(|_| BeaconRejection::Malformed)?;
        if beacon.cluster_name != self.cluster_name {
            return Err(BeaconRejection::ClusterMismatch {
                got: beacon.cluster_name,
            });
        }
        if beacon.version != self.expected_version {
            return Err(BeaconRejection::VersionMismatch {
                got: beacon.version,
                expected: self.expected_version.to_string(),
            });
        }
        let tag = unhex(&beacon.mac).ok_or(BeaconRejection::BadMac)?;
        if !self
            .auth
            .verify(BEACON_LABEL, &canonical_bytes(&beacon), &tag)
        {
            return Err(BeaconRejection::BadMac);
        }
        let skew_ms = BEACON_MAX_SKEW.as_millis() as u64;
        self.seen_nonces
            .retain(|_, seen_at| seen_at.abs_diff(now_ms) <= skew_ms);
        if beacon.timestamp_ms.abs_diff(now_ms) > skew_ms {
            return Err(BeaconRejection::StaleTimestamp {
                beacon_ms: beacon.timestamp_ms,
                now_ms,
            });
        }
        if self.seen_nonces.contains_key(&beacon.nonce) {
            return Err(BeaconRejection::ReplayedNonce);
        }
        let source_ip = source_ip.to_canonical();
        let advertised_ip = beacon.peer_addr.ip().to_canonical();
        let peer_addr = if advertised_ip.is_unspecified() {
            SocketAddr::new(source_ip, beacon.peer_addr.port())
        } else if advertised_ip == source_ip {
            SocketAddr::new(advertised_ip, beacon.peer_addr.port())
        } else {
            return Err(BeaconRejection::SourceMismatch {
                advertised: advertised_ip,
                source: source_ip,
            });
        };
        self.seen_nonces.insert(beacon.nonce, beacon.timestamp_ms);
        Ok(AcceptedBeacon {
            peer_addr,
            pid: beacon.pid,
            node_id: beacon.node_id,
        })
    }
}

#[cfg(test)]
#[path = "discovery_beacon_tests.rs"]
mod tests;
