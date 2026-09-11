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

use super::*;

#[test]
fn transport_message_payload_size() {
    let tensor_msg = TransportMessage::TensorData {
        tensor_id: "layer.0.weight".to_string(),
        shape: vec![128, 64],
        data: bytes::Bytes::from(vec![0u8; 1024]),
    };
    assert_eq!(tensor_msg.payload_size(), 1024);

    let ctrl_msg = TransportMessage::Control {
        operation: "heartbeat".to_string(),
        payload: bytes::Bytes::from(vec![1u8; 256]),
    };
    assert_eq!(ctrl_msg.payload_size(), 256);
}

#[test]
fn message_kind_roundtrip() {
    for (byte, expected) in [
        (1u8, MessageKind::TensorData),
        (2, MessageKind::Control),
        (3, MessageKind::RpcRequest),
        (4, MessageKind::RpcResponse),
    ] {
        let kind = MessageKind::try_from(byte).unwrap();
        assert_eq!(kind, expected);
        assert_eq!(kind as u8, byte);
    }
}

#[test]
fn message_kind_rejects_unknown() {
    assert!(MessageKind::try_from(0u8).is_err());
    assert!(MessageKind::try_from(5u8).is_err());
    assert!(MessageKind::try_from(255u8).is_err());
}

#[test]
fn transport_backend_display() {
    assert_eq!(TransportBackend::Tcp.to_string(), "tcp");
    assert_eq!(TransportBackend::Thunderbolt.to_string(), "thunderbolt");
    assert_eq!(TransportBackend::Rdma.to_string(), "rdma");
}

#[test]
fn transport_backend_from_str_accepts_rdma() {
    use std::str::FromStr;
    assert_eq!(
        TransportBackend::from_str("rdma").unwrap(),
        TransportBackend::Rdma
    );
    assert_eq!(
        TransportBackend::from_str("  RDMA  ").unwrap(),
        TransportBackend::Rdma
    );
}

#[test]
fn transport_backend_from_str_reports_valid_set_in_error() {
    use std::str::FromStr;
    let err = TransportBackend::from_str("infiniband").unwrap_err();
    let rendered = format!("{err}");
    assert!(rendered.contains("tcp"));
    assert!(rendered.contains("thunderbolt"));
    assert!(rendered.contains("rdma"));
}
