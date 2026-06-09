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
