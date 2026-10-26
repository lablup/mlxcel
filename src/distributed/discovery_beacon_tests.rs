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

use std::net::Ipv4Addr;

use super::*;
use crate::distributed::cluster_auth::ClusterSecret;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const NOW: u64 = 1_800_000_000_000;

fn src() -> IpAddr {
    IpAddr::V4(Ipv4Addr::new(10, 0, 0, 7))
}

fn unsigned(peer: &str) -> DiscoveryBeacon {
    DiscoveryBeacon {
        version: VERSION.to_string(),
        cluster_name: "mlxcel-cluster".to_string(),
        peer_addr: peer.parse().unwrap(),
        node_id: Some("stage-a".to_string()),
        pid: 4242,
        timestamp_ms: 0,
        nonce: 0,
        mac: String::new(),
    }
}

fn signed(auth: &ClusterAuth, peer: &str, now: u64) -> Vec<u8> {
    let mut beacon = unsigned(peer);
    sign_beacon(&mut beacon, auth, now);
    serde_json::to_vec(&beacon).unwrap()
}

#[test]
fn verifier_accepts_signed_beacon_and_binds_source() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let mut verifier = BeaconVerifier::new(&auth, "mlxcel-cluster", VERSION);
    let accepted = verifier
        .accept(&signed(&auth, "10.0.0.7:19001", NOW), src(), NOW + 500)
        .unwrap();
    assert_eq!(accepted.peer_addr.to_string(), "10.0.0.7:19001");
    assert_eq!(accepted.pid, 4242);
    // Unspecified advertised IP is replaced by the source IP.
    let accepted = verifier
        .accept(&signed(&auth, "0.0.0.0:19002", NOW), src(), NOW)
        .unwrap();
    assert_eq!(accepted.peer_addr.to_string(), "10.0.0.7:19002");
}

#[test]
fn verifier_rejects_spoofed_beacon_without_mac() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let mut verifier = BeaconVerifier::new(&auth, "mlxcel-cluster", VERSION);
    // The pre-fix wire shape: correct cluster_name, no authentication.
    let spoofed = serde_json::json!({
        "version": VERSION,
        "cluster_name": "mlxcel-cluster",
        "peer_addr": "10.0.0.7:9000",
        "node_id": null,
        "pid": 1
    });
    let err = verifier
        .accept(&serde_json::to_vec(&spoofed).unwrap(), src(), NOW)
        .unwrap_err();
    assert_eq!(err, BeaconRejection::BadMac);
}

#[test]
fn verifier_rejects_beacon_signed_with_other_secret() {
    let auth_a = ClusterAuth::new(&ClusterSecret::random());
    let auth_b = ClusterAuth::new(&ClusterSecret::random());
    let mut verifier = BeaconVerifier::new(&auth_b, "mlxcel-cluster", VERSION);
    let err = verifier
        .accept(&signed(&auth_a, "10.0.0.7:19001", NOW), src(), NOW)
        .unwrap_err();
    assert_eq!(err, BeaconRejection::BadMac);
    // And accepts one signed with its own secret, so the secret is not fixed.
    verifier
        .accept(&signed(&auth_b, "10.0.0.7:19001", NOW), src(), NOW)
        .unwrap();
}

#[test]
fn verifier_rejects_tampered_fields() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let mut verifier = BeaconVerifier::new(&auth, "mlxcel-cluster", VERSION);
    let mut beacon = unsigned("10.0.0.7:19001");
    sign_beacon(&mut beacon, &auth, NOW);
    beacon.peer_addr = "10.0.0.7:19999".parse().unwrap();
    let err = verifier
        .accept(&serde_json::to_vec(&beacon).unwrap(), src(), NOW)
        .unwrap_err();
    assert_eq!(err, BeaconRejection::BadMac);
}

#[test]
fn verifier_rejects_stale_and_future_timestamps() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let mut verifier = BeaconVerifier::new(&auth, "mlxcel-cluster", VERSION);
    let skew = BEACON_MAX_SKEW.as_millis() as u64;
    let err = verifier
        .accept(&signed(&auth, "10.0.0.7:19001", NOW - skew - 1), src(), NOW)
        .unwrap_err();
    assert!(
        matches!(err, BeaconRejection::StaleTimestamp { .. }),
        "{err}"
    );
    let err = verifier
        .accept(&signed(&auth, "10.0.0.7:19001", NOW + skew + 1), src(), NOW)
        .unwrap_err();
    assert!(
        matches!(err, BeaconRejection::StaleTimestamp { .. }),
        "{err}"
    );
    verifier
        .accept(&signed(&auth, "10.0.0.7:19001", NOW - skew), src(), NOW)
        .unwrap();
}

#[test]
fn verifier_rejects_replayed_nonce_within_window_and_forgets_after() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let mut verifier = BeaconVerifier::new(&auth, "mlxcel-cluster", VERSION);
    let datagram = signed(&auth, "10.0.0.7:19001", NOW);
    verifier.accept(&datagram, src(), NOW).unwrap();
    let err = verifier.accept(&datagram, src(), NOW + 10).unwrap_err();
    assert_eq!(err, BeaconRejection::ReplayedNonce);
    // Once the window has passed the nonce is pruned, but the timestamp
    // check rejects the old datagram anyway.
    let later = NOW + BEACON_MAX_SKEW.as_millis() as u64 + 1;
    let err = verifier.accept(&datagram, src(), later).unwrap_err();
    assert!(
        matches!(err, BeaconRejection::StaleTimestamp { .. }),
        "{err}"
    );
    assert!(verifier.seen_nonces.is_empty());
}

#[test]
fn verifier_rejects_source_ip_mismatch() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let mut verifier = BeaconVerifier::new(&auth, "mlxcel-cluster", VERSION);
    let err = verifier
        .accept(&signed(&auth, "10.0.0.99:19001", NOW), src(), NOW)
        .unwrap_err();
    assert!(
        matches!(err, BeaconRejection::SourceMismatch { .. }),
        "{err}"
    );
}

#[test]
fn verifier_enforces_version_and_cluster_name() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let mut verifier = BeaconVerifier::new(&auth, "mlxcel-cluster", VERSION);
    let mut beacon = unsigned("10.0.0.7:19001");
    beacon.version = "0.0.0-other".to_string();
    sign_beacon(&mut beacon, &auth, NOW);
    let err = verifier
        .accept(&serde_json::to_vec(&beacon).unwrap(), src(), NOW)
        .unwrap_err();
    assert!(
        matches!(err, BeaconRejection::VersionMismatch { .. }),
        "{err}"
    );

    let mut beacon = unsigned("10.0.0.7:19001");
    beacon.cluster_name = "other".to_string();
    sign_beacon(&mut beacon, &auth, NOW);
    let err = verifier
        .accept(&serde_json::to_vec(&beacon).unwrap(), src(), NOW)
        .unwrap_err();
    assert!(
        matches!(err, BeaconRejection::ClusterMismatch { .. }),
        "{err}"
    );

    assert_eq!(
        verifier.accept(b"not json", src(), NOW).unwrap_err(),
        BeaconRejection::Malformed
    );
}

#[test]
fn canonical_bytes_are_independent_of_json_layout() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let mut beacon = unsigned("10.0.0.7:19001");
    sign_beacon(&mut beacon, &auth, NOW);
    let reordered = format!(
        r#"{{"pid":{},"nonce":{},"mac":"{}","timestamp_ms":{},"node_id":"stage-a","peer_addr":"10.0.0.7:19001","cluster_name":"mlxcel-cluster","version":"{}"}}"#,
        beacon.pid, beacon.nonce, beacon.mac, beacon.timestamp_ms, VERSION
    );
    let mut verifier = BeaconVerifier::new(&auth, "mlxcel-cluster", VERSION);
    verifier.accept(reordered.as_bytes(), src(), NOW).unwrap();
}
