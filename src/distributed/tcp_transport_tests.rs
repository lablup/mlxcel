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
    let (kind, payload) = serialize_message(&msg);
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
    let (kind, payload) = serialize_message(&msg);
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

mod cluster_auth {
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use crate::distributed::cluster_auth::{ClusterAuth, ClusterSecret};
    use crate::distributed::cluster_handshake::ServerAtCapacity;
    use crate::distributed::connection_pool::PoolConfig;
    use crate::distributed::tcp_transport::{TcpTransport, TcpTransportConfig};
    use crate::distributed::transport::{MessageKind, Transport, TransportMessage};

    fn auth() -> Arc<ClusterAuth> {
        Arc::new(ClusterAuth::new(&ClusterSecret::random()))
    }

    async fn bind(auth: Option<Arc<ClusterAuth>>) -> TcpTransport {
        TcpTransport::bind(TcpTransportConfig {
            bind_address: "127.0.0.1:0".to_string(),
            auth,
            ..Default::default()
        })
        .await
        .unwrap()
    }

    fn ping() -> TransportMessage {
        TransportMessage::Control {
            operation: "ping".to_string(),
            payload: Bytes::from_static(b"ping"),
        }
    }

    /// Encode one Control frame the way `TcpTransport::send` does.
    fn raw_control_frame(operation: &str, payload: &[u8]) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&(operation.len() as u32).to_be_bytes());
        body.extend_from_slice(operation.as_bytes());
        body.extend_from_slice(payload);
        let mut frame = vec![MessageKind::Control as u8];
        frame.extend_from_slice(&(body.len() as u64).to_be_bytes());
        frame.extend_from_slice(&body);
        frame
    }

    #[tokio::test]
    async fn authenticated_peers_exchange_frames_and_report_listener_identity() {
        let shared = auth();
        let server = bind(Some(Arc::clone(&shared))).await;
        let client = bind(Some(shared)).await;
        let server_addr = server.local_addr().unwrap();
        let client_addr = client.local_addr().unwrap();

        client.send(&server_addr, ping()).await.unwrap();
        let (from, msg) = tokio::time::timeout(Duration::from_secs(5), server.recv())
            .await
            .unwrap()
            .unwrap();
        // `from` is the client's authenticated listener identity, not the
        // ephemeral source port of the accepted socket.
        assert_eq!(from, client_addr);
        assert!(matches!(msg, TransportMessage::Control { operation, .. } if operation == "ping"));

        // And the reverse direction authenticates too (mutual).
        server.send(&from, ping()).await.unwrap();
        let (from, _) = tokio::time::timeout(Duration::from_secs(5), client.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(from, server_addr);

        client.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn unauthenticated_raw_connection_never_reaches_recv() {
        let server = bind(Some(auth())).await;
        let server_addr = server.local_addr().unwrap();

        let mut raw = tokio::net::TcpStream::connect(&server_addr).await.unwrap();
        let _ = raw
            .write_all(&raw_control_frame("ping", &[0x41u8; 64]))
            .await;
        let _ = raw.flush().await;
        // The server drops the socket after the failed handshake.
        let mut buf = [0u8; 1];
        let closed = tokio::time::timeout(Duration::from_secs(5), raw.read(&mut buf)).await;
        assert!(matches!(closed, Ok(Ok(0)) | Ok(Err(_))), "{closed:?}");

        assert!(
            tokio::time::timeout(Duration::from_millis(300), server.recv())
                .await
                .is_err(),
            "a frame from an unauthenticated connection was dispatched"
        );
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn client_with_wrong_secret_is_rejected_in_both_directions() {
        let server = bind(Some(auth())).await;
        let client = bind(Some(auth())).await;
        let server_addr = server.local_addr().unwrap();

        // Pre-connect reports the mismatch instead of warning and continuing.
        let err = client
            .connect(std::slice::from_ref(&server_addr))
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("secret mismatch"), "{err:#}");
        let err = client.send(&server_addr, ping()).await.unwrap_err();
        assert!(format!("{err:#}").contains("cluster handshake"), "{err:#}");
        assert!(
            tokio::time::timeout(Duration::from_millis(300), server.recv())
                .await
                .is_err()
        );
        client.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn unauthenticated_client_cannot_talk_to_authenticated_server() {
        let server = bind(Some(auth())).await;
        let client = bind(None).await;
        let server_addr = server.local_addr().unwrap();
        // The plain client's first frame is not a handshake; the server
        // drops the connection and the frame is never dispatched.
        let _ = client.send(&server_addr, ping()).await;
        assert!(
            tokio::time::timeout(Duration::from_millis(300), server.recv())
                .await
                .is_err()
        );
        client.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn authenticated_client_refuses_plain_server() {
        let server = bind(None).await;
        let client = bind(Some(auth())).await;
        let server_addr = server.local_addr().unwrap();
        let err = client.send(&server_addr, ping()).await.unwrap_err();
        assert!(format!("{err:#}").contains("cluster handshake"), "{err:#}");
        client.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }

    /// Sockets that connect and never start the handshake must not hold a
    /// `max_connections` permit (#2655 item a): with the cap at 2 and two
    /// silent sockets open, an authenticated peer still gets through.
    #[tokio::test]
    async fn pending_handshakes_do_not_consume_connection_slots() {
        let shared = auth();
        let server = TcpTransport::bind(TcpTransportConfig {
            bind_address: "127.0.0.1:0".to_string(),
            auth: Some(Arc::clone(&shared)),
            max_connections: 2,
            ..Default::default()
        })
        .await
        .unwrap();
        let server_addr = server.local_addr().unwrap();

        let _silent_a = tokio::net::TcpStream::connect(&server_addr).await.unwrap();
        let _silent_b = tokio::net::TcpStream::connect(&server_addr).await.unwrap();
        // Let the accept loop pick both sockets up before the real peer dials.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let client = bind(Some(shared)).await;
        client
            .connect(std::slice::from_ref(&server_addr))
            .await
            .unwrap();
        client.send(&server_addr, ping()).await.unwrap();
        let (from, _) = tokio::time::timeout(Duration::from_secs(2), server.recv())
            .await
            .expect("authenticated frame must arrive while silent sockets are pending")
            .unwrap();
        assert_eq!(from, client.local_addr().unwrap());

        client.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }

    /// Silent sockets from one source are bounded per IP: the third one is
    /// closed at accept while the first two keep their pending slots.
    #[tokio::test]
    async fn per_ip_pending_limit_closes_excess_sockets() {
        let server = TcpTransport::bind(TcpTransportConfig {
            bind_address: "127.0.0.1:0".to_string(),
            auth: Some(auth()),
            max_pending_handshakes_per_ip: 2,
            ..Default::default()
        })
        .await
        .unwrap();
        let server_addr = server.local_addr().unwrap();

        let mut first = tokio::net::TcpStream::connect(&server_addr).await.unwrap();
        let mut second = tokio::net::TcpStream::connect(&server_addr).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        let mut third = tokio::net::TcpStream::connect(&server_addr).await.unwrap();

        let mut buf = [0u8; 1];
        let closed = tokio::time::timeout(Duration::from_millis(500), third.read(&mut buf)).await;
        assert!(
            matches!(closed, Ok(Ok(0)) | Ok(Err(_))),
            "third silent socket must be closed at accept: {closed:?}"
        );
        // The first two are still inside their handshake window (no EOF yet).
        for open in [&mut first, &mut second] {
            let pending =
                tokio::time::timeout(Duration::from_millis(100), open.read(&mut buf)).await;
            assert!(
                pending.is_err(),
                "pending socket was closed early: {pending:?}"
            );
        }
        server.shutdown().await.unwrap();
    }

    /// A pending slot is released as soon as the handshake finishes, so an
    /// authenticated connection does not keep occupying it: with one slot per
    /// IP, two authenticated clients from 127.0.0.1 connect one after the
    /// other and both get through.
    #[tokio::test]
    async fn pending_slot_is_released_after_handshake() {
        let shared = auth();
        let server = TcpTransport::bind(TcpTransportConfig {
            bind_address: "127.0.0.1:0".to_string(),
            auth: Some(Arc::clone(&shared)),
            max_pending_handshakes_per_ip: 1,
            ..Default::default()
        })
        .await
        .unwrap();
        let server_addr = server.local_addr().unwrap();

        let first = bind(Some(Arc::clone(&shared))).await;
        let second = bind(Some(shared)).await;
        for client in [&first, &second] {
            client.send(&server_addr, ping()).await.unwrap();
            let (from, _) = tokio::time::timeout(Duration::from_secs(5), server.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(from, client.local_addr().unwrap());
        }
        first.shutdown().await.unwrap();
        second.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }

    /// A zero pending-handshake bound would close every socket at accept, so
    /// an authenticated bind refuses it up front; without auth the bounds are
    /// unused and a zero stays accepted.
    #[tokio::test]
    async fn bind_refuses_zero_pending_bounds_with_auth() {
        for (total, per_ip) in [(0, 8), (64, 0)] {
            let err = TcpTransport::bind(TcpTransportConfig {
                bind_address: "127.0.0.1:0".to_string(),
                auth: Some(auth()),
                max_pending_handshakes: total,
                max_pending_handshakes_per_ip: per_ip,
                ..Default::default()
            })
            .await
            .err()
            .unwrap_or_else(|| panic!("({total}, {per_ip}) must be refused"))
            .to_string();
            assert!(err.contains("at least 1"), "{err}");
        }
        let plain = TcpTransport::bind(TcpTransportConfig {
            bind_address: "127.0.0.1:0".to_string(),
            max_pending_handshakes: 0,
            max_pending_handshakes_per_ip: 0,
            ..Default::default()
        })
        .await
        .unwrap();
        plain.shutdown().await.unwrap();
    }

    /// Regression guard for the connection-slot accounting (#2692 item 2):
    /// every per-connection resource is released by drop, so a connection
    /// task that panics inside a registered RPC handler frees its
    /// `max_connections` slot. With one slot, client A's request makes the
    /// handler panic, and client B still gets through afterwards.
    ///
    /// Client A's `rpc_call` fails by timing out: `TcpTransport` reads RPC
    /// responses only on accepted connections, never on the pooled outbound
    /// one the request went out on, so no echo could reach A even without the
    /// panic. B's observable is therefore a plain `send` reaching the server.
    #[tokio::test]
    async fn panicking_rpc_handler_releases_its_connection_slot() {
        let shared = auth();
        let server = TcpTransport::bind(TcpTransportConfig {
            bind_address: "127.0.0.1:0".to_string(),
            auth: Some(Arc::clone(&shared)),
            max_connections: 1,
            ..Default::default()
        })
        .await
        .unwrap();
        server
            .serve_rpc(Box::new(|request| {
                if request == b"boom" {
                    panic!("rpc handler panic (expected by the test)");
                }
                request.to_vec()
            }))
            .await
            .unwrap();
        let server_addr = server.local_addr().unwrap();

        let client_a = TcpTransport::bind(TcpTransportConfig {
            bind_address: "127.0.0.1:0".to_string(),
            auth: Some(Arc::clone(&shared)),
            pool_config: PoolConfig {
                connect_timeout: Duration::from_secs(1),
                ..PoolConfig::default()
            },
            ..Default::default()
        })
        .await
        .unwrap();
        let failed = tokio::time::timeout(
            Duration::from_secs(5),
            client_a.rpc_call(&server_addr, b"boom"),
        )
        .await
        .expect("rpc_call must return");
        assert!(failed.is_err(), "the panicking handler cannot answer");

        let client_b = bind(Some(shared)).await;
        client_b.send(&server_addr, ping()).await.unwrap();
        let (from, _) = tokio::time::timeout(Duration::from_secs(2), server.recv())
            .await
            .expect("client B must get the slot the panicked connection task held")
            .unwrap();
        assert_eq!(from, client_b.local_addr().unwrap());

        client_b.shutdown().await.unwrap();
        client_a.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }

    /// A full connection cap is answered inside the handshake: client B's
    /// `send` fails with the typed, retryable `ServerAtCapacity` instead of
    /// an accepted handshake followed by a closed socket.
    #[tokio::test]
    async fn full_connection_cap_reports_capacity_to_the_client() {
        let shared = auth();
        let server = TcpTransport::bind(TcpTransportConfig {
            bind_address: "127.0.0.1:0".to_string(),
            auth: Some(Arc::clone(&shared)),
            max_connections: 1,
            ..Default::default()
        })
        .await
        .unwrap();
        let server_addr = server.local_addr().unwrap();

        let client_a = bind(Some(Arc::clone(&shared))).await;
        client_a.send(&server_addr, ping()).await.unwrap();
        let (from, _) = tokio::time::timeout(Duration::from_secs(5), server.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(from, client_a.local_addr().unwrap());
        // A keeps its pooled connection, so the single slot stays taken.

        let client_b = bind(Some(shared)).await;
        let err = tokio::time::timeout(Duration::from_secs(1), client_b.send(&server_addr, ping()))
            .await
            .expect("send must fail within 1 s, not hang on a closed socket")
            .expect_err("the server is at its connection cap");
        assert!(
            err.chain()
                .any(|cause| cause.downcast_ref::<ServerAtCapacity>().is_some()),
            "expected ServerAtCapacity in the chain, got: {err:#}"
        );

        // Once A releases its connection the slot is free for B.
        client_a.shutdown().await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        client_b.send(&server_addr, ping()).await.unwrap();
        let (from, _) = tokio::time::timeout(Duration::from_secs(5), server.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(from, client_b.local_addr().unwrap());

        client_b.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn default_config_stays_unauthenticated() {
        let server = bind(None).await;
        let client = bind(None).await;
        let server_addr = server.local_addr().unwrap();
        client.send(&server_addr, ping()).await.unwrap();
        let (from, _) = tokio::time::timeout(Duration::from_secs(5), server.recv())
            .await
            .unwrap()
            .unwrap();
        // Without auth the raw socket peer address is reported, as before.
        assert_ne!(from, client.local_addr().unwrap());
        assert!(from.starts_with("127.0.0.1:"));
        client.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }
}

/// Issue #2695, transport-level twin of the pool regression test: `send` to a
/// peer that closed its pooled connection and stopped listening must return
/// `Err`, so a caller such as the disaggregated router fails over. Before the
/// fix the pool reused the dead connection, the write landed in the kernel
/// buffer of a CLOSE_WAIT socket, and `send` returned `Ok` for a frame
/// nobody would ever read.
#[tokio::test]
async fn send_to_peer_that_closed_its_pooled_connection_fails() {
    use std::time::Duration;

    use tokio::io::AsyncReadExt;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let peer = listener.local_addr().unwrap().to_string();

    let client = TcpTransport::bind(TcpTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        pool_config: PoolConfig {
            connect_timeout: Duration::from_secs(2),
            reconnect_base_delay: Duration::from_millis(10),
            max_reconnect_attempts: 0,
            ..Default::default()
        },
        ..Default::default()
    })
    .await
    .unwrap();

    let msg = || TransportMessage::Control {
        operation: "ping".to_string(),
        payload: Bytes::from_static(b"ping"),
    };
    let (kind, payload) = serialize_message(&msg());
    let frame_len = 1 + 8 + payload.len();
    assert_eq!(kind, MessageKind::Control);

    // The first send opens the pooled connection and the peer reads the
    // whole frame, so its later close is an orderly FIN, as when a node
    // process that has drained its socket exits.
    client.send(&peer, msg()).await.unwrap();
    let (mut server_side, _) = listener.accept().await.unwrap();
    let mut received = vec![0u8; frame_len];
    tokio::time::timeout(
        Duration::from_secs(5),
        server_side.read_exact(&mut received),
    )
    .await
    .expect("the first frame arrives")
    .unwrap();

    drop(server_side);
    drop(listener);
    tokio::time::sleep(Duration::from_millis(50)).await;

    let result = tokio::time::timeout(Duration::from_secs(10), client.send(&peer, msg()))
        .await
        .expect("send to a dead peer must not hang");
    assert!(
        result.is_err(),
        "send to a peer that closed its pooled connection returned Ok"
    );
    client.shutdown().await.unwrap();
}

#[path = "tcp_transport_frame_tests.rs"]
mod frame_protection;
