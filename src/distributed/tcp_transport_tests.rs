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

use bytes::Bytes;

use super::*;
use crate::distributed::transport::{Transport, TransportBackend, TransportMessage};

#[tokio::test]
async fn tcp_transport_bind_and_local_addr() {
    let config = TcpTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        ..Default::default()
    };
    let transport = TcpTransport::bind(config).await.unwrap();
    let addr = transport.local_addr().unwrap();
    assert!(addr.starts_with("127.0.0.1:"));
    assert_eq!(transport.backend(), TransportBackend::Tcp);
    transport.shutdown().await.unwrap();
}

#[tokio::test]
async fn tcp_transport_send_recv_control_message() {
    // Set up two transports: sender and receiver.
    let recv_transport = TcpTransport::bind(TcpTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        ..Default::default()
    })
    .await
    .unwrap();
    let recv_addr = recv_transport.local_addr().unwrap();

    let send_transport = TcpTransport::bind(TcpTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        ..Default::default()
    })
    .await
    .unwrap();

    // Connect sender to receiver.
    send_transport
        .connect(std::slice::from_ref(&recv_addr))
        .await
        .unwrap();

    // Send a control message.
    let msg = TransportMessage::Control {
        operation: "heartbeat".to_string(),
        payload: Bytes::from(b"ping".to_vec()),
    };
    send_transport.send(&recv_addr, msg).await.unwrap();

    // Receive the message.
    let (sender, received) = recv_transport.recv().await.unwrap();
    assert!(!sender.is_empty());
    match received {
        TransportMessage::Control { operation, payload } => {
            assert_eq!(operation, "heartbeat");
            assert_eq!(payload.as_ref(), b"ping");
        }
        _ => panic!("expected Control message"),
    }

    send_transport.shutdown().await.unwrap();
    recv_transport.shutdown().await.unwrap();
}

#[tokio::test]
async fn tcp_transport_send_recv_tensor_data() {
    let recv_transport = TcpTransport::bind(TcpTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        ..Default::default()
    })
    .await
    .unwrap();
    let recv_addr = recv_transport.local_addr().unwrap();

    let send_transport = TcpTransport::bind(TcpTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        ..Default::default()
    })
    .await
    .unwrap();

    send_transport
        .connect(std::slice::from_ref(&recv_addr))
        .await
        .unwrap();

    let tensor_data = vec![1u8; 1024];
    let msg = TransportMessage::TensorData {
        tensor_id: "layer.0.weight".to_string(),
        shape: vec![32, 32],
        data: Bytes::from(tensor_data.clone()),
    };
    send_transport.send(&recv_addr, msg).await.unwrap();

    let (_sender, received) = recv_transport.recv().await.unwrap();
    match received {
        TransportMessage::TensorData {
            tensor_id,
            shape,
            data,
        } => {
            assert_eq!(tensor_id, "layer.0.weight");
            assert_eq!(shape, vec![32, 32]);
            assert_eq!(data.len(), 1024);
            assert!(data.iter().all(|&b| b == 1));
        }
        _ => panic!("expected TensorData message"),
    }

    send_transport.shutdown().await.unwrap();
    recv_transport.shutdown().await.unwrap();
}

#[test]
fn serialize_deserialize_roundtrip_control() {
    let msg = TransportMessage::Control {
        operation: "test_op".to_string(),
        payload: Bytes::from(b"hello world".to_vec()),
    };
    let (kind, payload) = TcpTransport::serialize_message(&msg);
    assert_eq!(kind, MessageKind::Control);
    let deserialized = deserialize_message(kind, &payload).unwrap();
    match deserialized {
        TransportMessage::Control { operation, payload } => {
            assert_eq!(operation, "test_op");
            assert_eq!(payload.as_ref(), b"hello world");
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn serialize_deserialize_roundtrip_tensor() {
    let msg = TransportMessage::TensorData {
        tensor_id: "attn.k_proj".to_string(),
        shape: vec![128, 64, 32],
        data: Bytes::from(vec![42u8; 512]),
    };
    let (kind, payload) = TcpTransport::serialize_message(&msg);
    assert_eq!(kind, MessageKind::TensorData);
    let deserialized = deserialize_message(kind, &payload).unwrap();
    match deserialized {
        TransportMessage::TensorData {
            tensor_id,
            shape,
            data,
        } => {
            assert_eq!(tensor_id, "attn.k_proj");
            assert_eq!(shape, vec![128, 64, 32]);
            assert_eq!(data.len(), 512);
            assert!(data.iter().all(|&b| b == 42));
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn deserialize_rejects_truncated_control() {
    // Too short for operation length.
    assert!(deserialize_message(MessageKind::Control, &[0, 0]).is_err());
}

#[test]
fn deserialize_rejects_truncated_tensor() {
    // Too short for id length.
    assert!(deserialize_message(MessageKind::TensorData, &[0]).is_err());
}
