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

use std::time::Duration;

use bytes::Bytes;

use super::*;

async fn make_pair() -> (MockTransport, MockTransport, MockRouter) {
    let router = MockRouter::new();
    let config = MockTransportConfig::default();
    let a = MockTransport::new("node-a:8080".to_string(), router.clone(), config.clone()).await;
    let b = MockTransport::new("node-b:8081".to_string(), router.clone(), config).await;
    (a, b, router)
}

#[tokio::test]
async fn send_and_recv_control_message() {
    let (a, b, _router) = make_pair().await;

    let msg = TransportMessage::Control {
        operation: "test".to_string(),
        payload: Bytes::from("hello"),
    };
    a.send("node-b:8081", msg).await.unwrap();

    let (sender, received) = b.recv().await.unwrap();
    assert_eq!(sender, "node-a:8080");
    match received {
        TransportMessage::Control { operation, payload } => {
            assert_eq!(operation, "test");
            assert_eq!(&payload[..], b"hello");
        }
        _ => panic!("expected Control message"),
    }

    assert_eq!(a.sent_count(), 1);
    assert_eq!(b.received_count(), 1);
}

#[tokio::test]
async fn send_and_recv_tensor_data() {
    let (a, b, _router) = make_pair().await;

    let data = Bytes::from(vec![42u8; 1024]);
    let msg = TransportMessage::TensorData {
        tensor_id: "layer.0".to_string(),
        shape: vec![32, 32],
        data: data.clone(),
    };
    a.send("node-b:8081", msg).await.unwrap();

    let (sender, received) = b.recv().await.unwrap();
    assert_eq!(sender, "node-a:8080");
    match received {
        TransportMessage::TensorData {
            tensor_id,
            shape,
            data: recv_data,
        } => {
            assert_eq!(tensor_id, "layer.0");
            assert_eq!(shape, vec![32, 32]);
            assert_eq!(recv_data, data);
        }
        _ => panic!("expected TensorData message"),
    }
}

#[tokio::test]
async fn rpc_call_and_response() {
    let (a, b, _router) = make_pair().await;

    // Install echo handler on node B.
    b.serve_rpc(Box::new(|req| {
        let mut response = b"echo: ".to_vec();
        response.extend_from_slice(req);
        response
    }))
    .await
    .unwrap();

    let response = a.rpc_call("node-b:8081", b"ping").await.unwrap();
    assert_eq!(&response, b"echo: ping");
}

#[tokio::test]
async fn rpc_fails_without_handler() {
    let (a, _b, _router) = make_pair().await;

    let result = a.rpc_call("node-b:8081", b"test").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("no RPC handler"));
}

#[tokio::test]
async fn partition_blocks_messages() {
    let (a, b, router) = make_pair().await;

    router.partition_node("node-b:8081").await;

    let msg = TransportMessage::Control {
        operation: "test".to_string(),
        payload: Bytes::from("blocked"),
    };
    let result = a.send("node-b:8081", msg).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("partitioned"));

    // Heal and verify communication resumes.
    router.heal_node("node-b:8081").await;

    let msg = TransportMessage::Control {
        operation: "test".to_string(),
        payload: Bytes::from("healed"),
    };
    a.send("node-b:8081", msg).await.unwrap();

    let (_, received) = b.recv().await.unwrap();
    match received {
        TransportMessage::Control { payload, .. } => {
            assert_eq!(&payload[..], b"healed");
        }
        _ => panic!("expected Control message"),
    }
}

#[tokio::test]
async fn shutdown_prevents_further_operations() {
    let (a, _b, _router) = make_pair().await;

    a.shutdown().await.unwrap();
    assert!(a.is_shutdown());

    let msg = TransportMessage::Control {
        operation: "test".to_string(),
        payload: Bytes::from("after_shutdown"),
    };
    let result = a.send("node-b:8081", msg).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("shut down"));
}

#[tokio::test]
async fn connect_is_noop() {
    let (a, _b, _router) = make_pair().await;

    // Connect should succeed (no-op for mock).
    a.connect(&["node-b:8081".to_string()]).await.unwrap();
}

#[tokio::test]
async fn local_addr_returns_configured_address() {
    let router = MockRouter::new();
    let config = MockTransportConfig::default();
    let transport = MockTransport::new("test-node:9999".to_string(), router, config).await;

    assert_eq!(transport.local_addr().unwrap(), "test-node:9999");
}

#[tokio::test]
async fn backend_returns_tcp() {
    let router = MockRouter::new();
    let config = MockTransportConfig::default();
    let transport = MockTransport::new("test:1234".to_string(), router, config).await;

    assert_eq!(transport.backend(), TransportBackend::Tcp);
}

#[tokio::test]
async fn latency_simulation() {
    let router = MockRouter::new();
    let config = MockTransportConfig {
        latency: Duration::from_millis(50),
        ..Default::default()
    };
    let a = MockTransport::new("a:1".to_string(), router.clone(), config.clone()).await;
    let b = MockTransport::new("b:2".to_string(), router, MockTransportConfig::default()).await;

    let start = std::time::Instant::now();
    let msg = TransportMessage::Control {
        operation: "latency_test".to_string(),
        payload: Bytes::from("data"),
    };
    a.send("b:2", msg).await.unwrap();
    let elapsed = start.elapsed();

    // Should have taken at least 50ms due to simulated latency.
    assert!(elapsed >= Duration::from_millis(40)); // Allow small timing slack.

    let _ = b.recv().await.unwrap();
}

#[tokio::test]
async fn send_to_unknown_peer_fails() {
    let router = MockRouter::new();
    let config = MockTransportConfig::default();
    let a = MockTransport::new("a:1".to_string(), router, config).await;

    let msg = TransportMessage::Control {
        operation: "test".to_string(),
        payload: Bytes::from("orphan"),
    };
    let result = a.send("nonexistent:9999", msg).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not registered"));
}

#[tokio::test]
async fn multiple_messages_maintain_order() {
    let (a, b, _router) = make_pair().await;

    for i in 0..10u32 {
        let msg = TransportMessage::Control {
            operation: format!("msg-{i}"),
            payload: Bytes::from(i.to_be_bytes().to_vec()),
        };
        a.send("node-b:8081", msg).await.unwrap();
    }

    for i in 0..10u32 {
        let (_, received) = b.recv().await.unwrap();
        match received {
            TransportMessage::Control { operation, .. } => {
                assert_eq!(operation, format!("msg-{i}"));
            }
            _ => panic!("expected Control message"),
        }
    }
}

#[tokio::test]
async fn bidirectional_communication() {
    let (a, b, _router) = make_pair().await;

    // A -> B
    let msg = TransportMessage::Control {
        operation: "from_a".to_string(),
        payload: Bytes::new(),
    };
    a.send("node-b:8081", msg).await.unwrap();

    // B -> A
    let msg = TransportMessage::Control {
        operation: "from_b".to_string(),
        payload: Bytes::new(),
    };
    b.send("node-a:8080", msg).await.unwrap();

    let (sender_to_b, msg_at_b) = b.recv().await.unwrap();
    assert_eq!(sender_to_b, "node-a:8080");
    match msg_at_b {
        TransportMessage::Control { operation, .. } => assert_eq!(operation, "from_a"),
        _ => panic!("expected Control"),
    }

    let (sender_to_a, msg_at_a) = a.recv().await.unwrap();
    assert_eq!(sender_to_a, "node-b:8081");
    match msg_at_a {
        TransportMessage::Control { operation, .. } => assert_eq!(operation, "from_b"),
        _ => panic!("expected Control"),
    }
}

#[tokio::test]
async fn stream_send_and_recv() {
    let (a, b, _router) = make_pair().await;

    let data = Bytes::from(vec![0xAB; 512]);
    a.send_stream("node-b:8081", data.clone()).await.unwrap();

    let (sender, mut reader) = b.recv_stream().await.unwrap();
    assert_eq!(sender, "node-a:8080");

    use tokio::io::AsyncReadExt;
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).await.unwrap();
    assert_eq!(buf, data.to_vec());
}

#[tokio::test]
async fn router_partition_and_heal() {
    let router = MockRouter::new();
    assert!(!router.is_partitioned("node-x").await);

    router.partition_node("node-x").await;
    assert!(router.is_partitioned("node-x").await);

    router.heal_node("node-x").await;
    assert!(!router.is_partitioned("node-x").await);
}
