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

//! Unit tests for the model-free [`ServingCoordinator`] skeleton: serving-mode
//! binding and the transport seam (send/recv handoff) over an in-process
//! [`MockTransport`]. The real-model 2-role drive (prefill -> extract -> wire ->
//! ingest -> decode) lands with the scheduler driver (B2b) and is gated by the
//! real-model integration test (B2c).

use bytes::Bytes;

use super::ServingCoordinator;
use crate::distributed::disaggregated::serving::ServingMode;
use crate::distributed::mock_transport::{MockRouter, MockTransport, MockTransportConfig};
use crate::distributed::transport::TransportMessage;

/// Stand up a prefill and a decode coordinator sharing one in-process router.
/// The prefill node hands off to `"decode"`; the decode node is paired with
/// `"prefill"`.
async fn two_coordinators() -> (ServingCoordinator, ServingCoordinator) {
    let router = MockRouter::new();
    let prefill_transport = MockTransport::new(
        "prefill".to_string(),
        router.clone(),
        MockTransportConfig::default(),
    )
    .await;
    let decode_transport = MockTransport::new(
        "decode".to_string(),
        router.clone(),
        MockTransportConfig::default(),
    )
    .await;
    let prefill = ServingCoordinator::new(
        ServingMode::PrefillOnly,
        Box::new(prefill_transport),
        "decode",
    );
    let decode = ServingCoordinator::new(
        ServingMode::DecodeOnly,
        Box::new(decode_transport),
        "prefill",
    );
    (prefill, decode)
}

#[test]
fn coordinator_binds_mode_and_peer() {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(async {
        let (prefill, decode) = two_coordinators().await;

        assert_eq!(prefill.mode(), ServingMode::PrefillOnly);
        assert_eq!(prefill.peer(), "decode");
        assert_eq!(decode.mode(), ServingMode::DecodeOnly);
        assert_eq!(decode.peer(), "prefill");
    });
}

#[test]
fn coordinator_round_trips_a_handoff_frame() {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(async {
        let (prefill, decode) = two_coordinators().await;

        // An opaque serialized-cache stand-in; the coordinator seam moves bytes
        // and does not interpret them (the scheduler driver does, in B2b).
        let payload: Vec<u8> = (0..2048_u32).map(|i| (i % 251) as u8).collect();

        prefill
            .send_handoff(&payload)
            .await
            .expect("prefill coordinator sends the handoff frame");
        let (from, received) = decode
            .recv_handoff()
            .await
            .expect("decode coordinator receives the handoff frame");

        assert_eq!(from, "prefill", "sender must be the prefill node");
        assert_eq!(
            received, payload,
            "the decode coordinator must receive the prefill bytes unchanged"
        );
    });
}

#[test]
fn coordinator_recv_rejects_non_handoff_frame() {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(async {
        let (prefill, decode) = two_coordinators().await;

        // A control message is not a handoff frame; recv must reject it rather
        // than surface it as cache bytes for the (B2b) ingest path.
        prefill
            .transport()
            .send(
                "decode",
                TransportMessage::Control {
                    operation: "heartbeat".to_string(),
                    payload: Bytes::from_static(b"ping"),
                },
            )
            .await
            .expect("send control");

        let err = decode
            .recv_handoff()
            .await
            .expect_err("a control message must not be accepted as a handoff");
        assert!(
            err.to_string().contains(super::super::HANDOFF_TENSOR_ID),
            "rejection should name the expected handoff frame, got: {err}"
        );
    });
}

// ── decode_target allowlist (issue #389) ────────────────────────────────────

use super::{DecodeAllowlist, DecodeTargetDecision, decode_allowlist_from_env};

/// Parse a `host:port` test fixture into a `SocketAddr`.
fn addr(s: &str) -> std::net::SocketAddr {
    s.parse().expect("valid socket addr")
}

/// A router-chosen target that is on the allowlist is accepted, so router-driven
/// decode balancing keeps working for every decode node in the configured pool,
/// not just the prefill's first static peer (issue #201 / #389).
#[test]
fn allowlist_accepts_on_list_target() {
    let allow = DecodeAllowlist::from_peers(&[
        addr("10.0.0.1:9001"),
        addr("10.0.0.2:9002"),
        addr("10.0.0.3:9003"),
    ]);
    // The router may pick any decode node in the pool, including ones that are
    // not the prefill's first/static peer.
    assert_eq!(allow.decide("10.0.0.2:9002"), DecodeTargetDecision::Allow);
    assert_eq!(allow.decide("10.0.0.3:9003"), DecodeTargetDecision::Allow);
}

/// A target outside the configured allowlist is rejected (a forged frame cannot
/// redirect the KV handoff off-cluster).
#[test]
fn allowlist_rejects_off_list_target() {
    let allow = DecodeAllowlist::from_peers(&[addr("10.0.0.1:9001"), addr("10.0.0.2:9002")]);
    assert_eq!(allow.decide("10.9.9.9:9999"), DecodeTargetDecision::Reject);
    // A non-address string also cannot match any allowlisted peer.
    assert_eq!(
        allow.decide("attacker.example.com:443"),
        DecodeTargetDecision::Reject
    );
    assert_eq!(allow.decide("not-an-address"), DecodeTargetDecision::Reject);
}

/// When no allowlist source is configured (empty set), the prefill node stays
/// permissive-with-warning so router-driven balancing is never silently broken.
#[test]
fn allowlist_unconfigured_is_permissive() {
    let allow = DecodeAllowlist::from_peers(&[]);
    assert_eq!(
        allow.decide("10.0.0.7:9007"),
        DecodeTargetDecision::AllowUnchecked
    );
    // The default is likewise unconfigured/permissive.
    assert_eq!(
        DecodeAllowlist::default().decide("10.0.0.7:9007"),
        DecodeTargetDecision::AllowUnchecked
    );
}

/// Comparison is canonical (parsed `SocketAddr`), so equivalent spellings of the
/// same address match regardless of textual differences (e.g. an IPv6 address
/// with or without zero compression).
#[test]
fn allowlist_compares_canonical_addresses() {
    let allow = DecodeAllowlist::from_peers(&[addr("[2001:db8::1]:9001")]);
    // Uncompressed spelling of the same IPv6 address parses to the same value.
    assert_eq!(
        allow.decide("[2001:db8:0:0:0:0:0:1]:9001"),
        DecodeTargetDecision::Allow
    );
    // Same host, different port is a different node and is rejected.
    assert_eq!(
        allow.decide("[2001:db8::1]:9002"),
        DecodeTargetDecision::Reject
    );
}

/// A configured `MLXCEL_DECODE_ALLOWLIST` (the dedicated full-pool source, issue
/// #389) parses into an allowlist that accepts on-list targets and rejects
/// off-list ones, independent of the prefill's `--decode-peers` static fallback.
#[test]
fn allowlist_from_env_parses_configured_list() {
    let allow = decode_allowlist_from_env(Some("10.0.0.1:9001, 10.0.0.2:9002"));
    assert_eq!(allow.decide("10.0.0.1:9001"), DecodeTargetDecision::Allow);
    assert_eq!(allow.decide("10.0.0.2:9002"), DecodeTargetDecision::Allow);
    assert_eq!(allow.decide("10.0.0.9:9009"), DecodeTargetDecision::Reject);
}

/// An unset, empty, or whitespace-only `MLXCEL_DECODE_ALLOWLIST` yields a
/// permissive (`AllowUnchecked`) allowlist so router-driven balancing is never
/// silently broken when the operator has not configured the source.
#[test]
fn allowlist_from_env_unset_or_empty_is_permissive() {
    for raw in [None, Some(""), Some("   "), Some(",  ,")] {
        assert_eq!(
            decode_allowlist_from_env(raw).decide("10.0.0.7:9007"),
            DecodeTargetDecision::AllowUnchecked,
            "raw {raw:?} should yield a permissive allowlist"
        );
    }
}

/// An unparseable entry is skipped (warned, not fatal): a list with one good and
/// one garbage entry still allows the good address and is not empty, so a single
/// typo cannot silently fall back to fully permissive.
#[test]
fn allowlist_from_env_skips_unparseable_entries() {
    let allow = decode_allowlist_from_env(Some("not-an-address, 10.0.0.3:9003"));
    assert_eq!(allow.decide("10.0.0.3:9003"), DecodeTargetDecision::Allow);
    // The list is non-empty (the good entry parsed), so an off-list target is
    // rejected rather than permitted as unchecked.
    assert_eq!(allow.decide("10.0.0.9:9009"), DecodeTargetDecision::Reject);
}
