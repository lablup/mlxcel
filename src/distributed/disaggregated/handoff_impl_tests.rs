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

//! Unit tests for the model-independent halves of the handoff mechanism: the
//! async serde<->transport byte bridge. The real-model extract/probe/ingest
//! parity (which loads a checkpoint and runs GPU forwards) lives in the
//! `tests/paged_handoff_parity.rs` integration test.

use bytes::Bytes;

use super::{HANDOFF_TENSOR_ID, recv_handoff_payload, send_handoff_payload};
use crate::distributed::mock_transport::{MockRouter, MockTransport, MockTransportConfig};
use crate::distributed::transport::{Transport, TransportMessage};

/// Stand up two in-process nodes sharing a [`MockRouter`].
async fn two_nodes() -> (MockTransport, MockTransport) {
    let router = MockRouter::new();
    let prefill = MockTransport::new(
        "prefill".to_string(),
        router.clone(),
        MockTransportConfig::default(),
    )
    .await;
    let decode = MockTransport::new(
        "decode".to_string(),
        router.clone(),
        MockTransportConfig::default(),
    )
    .await;
    (prefill, decode)
}

#[test]
fn handoff_payload_round_trips_over_mock_transport() {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(async {
        let (prefill, decode) = two_nodes().await;

        // A representative serialized cache frame (opaque bytes for the bridge).
        let payload: Vec<u8> = (0..4096_u32).map(|i| (i % 251) as u8).collect();

        send_handoff_payload(&prefill, "decode", &payload)
            .await
            .expect("send handoff payload");
        let (from, received) = recv_handoff_payload(&decode)
            .await
            .expect("recv handoff payload");

        assert_eq!(from, "prefill", "sender address must be the prefill node");
        assert_eq!(
            received, payload,
            "the decode node must receive the prefill bytes unchanged"
        );
    });
}

#[test]
fn recv_handoff_payload_rejects_non_handoff_message() {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(async {
        let (prefill, decode) = two_nodes().await;

        // A control message is not a handoff frame; the bridge must reject it
        // rather than mis-restore it as a cache state.
        prefill
            .send(
                "decode",
                TransportMessage::Control {
                    operation: "heartbeat".to_string(),
                    payload: Bytes::from_static(b"ping"),
                },
            )
            .await
            .expect("send control");

        let err = recv_handoff_payload(&decode)
            .await
            .expect_err("a control message must not be accepted as a handoff");
        assert!(
            err.to_string().contains(HANDOFF_TENSOR_ID),
            "rejection should name the expected handoff frame, got: {err}"
        );
    });
}

#[test]
fn recv_handoff_payload_rejects_mismatched_tensor_id() {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(async {
        let (prefill, decode) = two_nodes().await;

        prefill
            .send(
                "decode",
                TransportMessage::TensorData {
                    tensor_id: "activations".to_string(),
                    shape: vec![8],
                    data: Bytes::from_static(b"notacache"),
                },
            )
            .await
            .expect("send foreign tensor");

        let err = recv_handoff_payload(&decode)
            .await
            .expect_err("a foreign TensorData frame must be rejected");
        assert!(
            err.to_string().contains("activations"),
            "rejection should name the offending tensor id, got: {err}"
        );
    });
}
