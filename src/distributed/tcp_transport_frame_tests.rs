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

//! Per-frame protection acceptance tests (#2656).
//!
//! Each test puts an in-test TCP proxy between two authenticated transports.
//! The proxy forwards the handshake untouched and then edits, replays,
//! reorders or splices frames on the client-to-server direction. None of the
//! tests depends on the protected frame layout: the proxy learns the length
//! of whatever trails the `[kind][len][body]` header from one calibration
//! frame, so the same tests run against the unprotected transport (where they
//! fail: every edited frame reaches `recv()`) and against the protected one.
//!
//! Every wait is bounded so a regression shows up as a failed assertion, not
//! a hung test.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, Notify};

use crate::distributed::cluster_auth::{ClusterAuth, ClusterSecret};
use crate::distributed::tcp_transport::{TcpTransport, TcpTransportConfig};
use crate::distributed::transport::{Transport, TransportMessage};

/// `[kind: u8][len: u64 BE]`, common to the plain and the protected frame.
const HEADER_LEN: usize = 9;
/// Upper bound for a message that must arrive.
const ARRIVAL: Duration = Duration::from_secs(5);
/// How long a message that must not arrive is waited for.
const SILENCE: Duration = Duration::from_millis(500);

/// What the proxy does with the client-to-server bytes of one connection.
#[derive(Clone, Copy)]
enum Plan {
    /// Forward unchanged.
    Forward,
    /// Flip the byte at this absolute client-to-server offset, then forward.
    FlipAt(usize),
    /// Hold complete frames back instead of forwarding them.
    Hold,
}

struct ConnState {
    plan: Plan,
    /// Every client-to-server byte read so far, unmodified, in order.
    recorded: Vec<u8>,
    /// Frames held back by [`Plan::Hold`], oldest first.
    held: Vec<Vec<u8>>,
    /// Bytes after the body of each frame, once calibrated. `None` means the
    /// proxy is still in byte mode (handshake or calibration).
    trailer: Option<usize>,
}

/// One proxied connection, in accept order.
struct ProxyConn {
    state: Arc<Mutex<ConnState>>,
    /// The server-side write half, shared with the forwarding task so a test
    /// can inject bytes exactly where the next frame would go.
    to_server: Arc<Mutex<OwnedWriteHalf>>,
    server_closed: Arc<AtomicBool>,
    closed_notify: Arc<Notify>,
}

impl ProxyConn {
    async fn set_plan(&self, plan: Plan) {
        self.state.lock().await.plan = plan;
    }

    async fn recorded_len(&self) -> usize {
        self.state.lock().await.recorded.len()
    }

    async fn recorded_from(&self, start: usize) -> Vec<u8> {
        self.state.lock().await.recorded[start..].to_vec()
    }

    async fn inject(&self, bytes: &[u8]) {
        let mut w = self.to_server.lock().await;
        w.write_all(bytes).await.unwrap();
        w.flush().await.unwrap();
    }

    /// Wait until the proxy has `n` complete frames held back.
    async fn wait_held(&self, n: usize) -> Vec<Vec<u8>> {
        let deadline = tokio::time::Instant::now() + ARRIVAL;
        loop {
            let held = self.state.lock().await.held.clone();
            if held.len() >= n {
                return held;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "proxy holds {} frame(s), expected {n}",
                held.len()
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// The server closed its side of this connection.
    async fn wait_server_closed(&self) {
        let notified = self.closed_notify.notified();
        if self.server_closed.load(Ordering::Acquire) {
            return;
        }
        tokio::time::timeout(ARRIVAL, notified)
            .await
            .expect("server must close the connection after a rejected frame");
    }

    fn server_is_closed(&self) -> bool {
        self.server_closed.load(Ordering::Acquire)
    }
}

/// A TCP proxy in front of one server transport.
struct Proxy {
    addr: String,
    conns: Arc<Mutex<Vec<Arc<ProxyConn>>>>,
}

impl Proxy {
    async fn spawn(server_addr: String) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let conns: Arc<Mutex<Vec<Arc<ProxyConn>>>> = Arc::new(Mutex::new(Vec::new()));
        let registry = Arc::clone(&conns);
        tokio::spawn(async move {
            while let Ok((client, _)) = listener.accept().await {
                let server = TcpStream::connect(&server_addr).await.unwrap();
                let _ = client.set_nodelay(true);
                let _ = server.set_nodelay(true);
                let (client_rd, client_wr) = client.into_split();
                let (server_rd, server_wr) = server.into_split();
                let conn = Arc::new(ProxyConn {
                    state: Arc::new(Mutex::new(ConnState {
                        plan: Plan::Forward,
                        recorded: Vec::new(),
                        held: Vec::new(),
                        trailer: None,
                    })),
                    to_server: Arc::new(Mutex::new(server_wr)),
                    server_closed: Arc::new(AtomicBool::new(false)),
                    closed_notify: Arc::new(Notify::new()),
                });
                registry.lock().await.push(Arc::clone(&conn));
                tokio::spawn(forward_client_to_server(client_rd, Arc::clone(&conn)));
                tokio::spawn(forward_server_to_client(server_rd, client_wr, conn));
            }
        });
        Self { addr, conns }
    }

    /// The `index`-th accepted connection (waits for it to exist).
    async fn conn(&self, index: usize) -> Arc<ProxyConn> {
        let deadline = tokio::time::Instant::now() + ARRIVAL;
        loop {
            if let Some(conn) = self.conns.lock().await.get(index) {
                return Arc::clone(conn);
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "proxy connection {index} never arrived"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// Server-to-client bytes pass through verbatim; EOF or an error from the
/// server marks the connection closed.
async fn forward_server_to_client(
    mut server_rd: OwnedReadHalf,
    mut client_wr: OwnedWriteHalf,
    conn: Arc<ProxyConn>,
) {
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        match server_rd.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if client_wr.write_all(&buf[..n]).await.is_err() {
                    break;
                }
                let _ = client_wr.flush().await;
            }
        }
    }
    conn.server_closed.store(true, Ordering::Release);
    conn.closed_notify.notify_waiters();
}

/// Read until `pending` holds at least `need` bytes. `false` on EOF.
async fn fill(client_rd: &mut OwnedReadHalf, pending: &mut Vec<u8>, need: usize) -> bool {
    let mut buf = vec![0u8; 64 * 1024];
    while pending.len() < need {
        match client_rd.read(&mut buf).await {
            Ok(0) | Err(_) => return false,
            Ok(n) => pending.extend_from_slice(&buf[..n]),
        }
    }
    true
}

/// Client-to-server bytes: byte mode until calibrated, then one frame at a
/// time with the connection's current [`Plan`] applied.
async fn forward_client_to_server(mut client_rd: OwnedReadHalf, conn: Arc<ProxyConn>) {
    let mut pending: Vec<u8> = Vec::new();
    loop {
        let trailer = conn.state.lock().await.trailer;
        let unit: Vec<u8> = match trailer {
            None => {
                if pending.is_empty() && !fill(&mut client_rd, &mut pending, 1).await {
                    return;
                }
                std::mem::take(&mut pending)
            }
            Some(trailer) => {
                if !fill(&mut client_rd, &mut pending, HEADER_LEN).await {
                    return;
                }
                let len = u64::from_be_bytes(pending[1..HEADER_LEN].try_into().unwrap()) as usize;
                let total = HEADER_LEN + len + trailer;
                if !fill(&mut client_rd, &mut pending, total).await {
                    return;
                }
                let rest = pending.split_off(total);
                std::mem::replace(&mut pending, rest)
            }
        };

        let mut state = conn.state.lock().await;
        let start = state.recorded.len();
        state.recorded.extend_from_slice(&unit);
        let mut out = unit;
        match state.plan {
            Plan::Forward => {}
            Plan::FlipAt(offset) => {
                if offset >= start && offset < start + out.len() {
                    out[offset - start] ^= 0x01;
                    state.plan = Plan::Forward;
                }
            }
            Plan::Hold => {
                if trailer.is_some() {
                    state.held.push(out);
                    continue;
                }
            }
        }
        drop(state);
        let mut w = conn.to_server.lock().await;
        if w.write_all(&out).await.is_err() {
            return;
        }
        let _ = w.flush().await;
    }
}

fn auth() -> Arc<ClusterAuth> {
    Arc::new(ClusterAuth::new(&ClusterSecret::random()))
}

async fn bind(auth: Arc<ClusterAuth>) -> TcpTransport {
    TcpTransport::bind(TcpTransportConfig {
        bind_address: "127.0.0.1:0".to_string(),
        auth: Some(auth),
        ..Default::default()
    })
    .await
    .unwrap()
}

fn control(payload: &[u8]) -> TransportMessage {
    TransportMessage::Control {
        operation: "ping".to_string(),
        payload: Bytes::copy_from_slice(payload),
    }
}

/// Offset of the first byte of a Control message's payload inside its frame:
/// header, `op_len` u32, the operation bytes.
fn control_payload_offset() -> usize {
    HEADER_LEN + 4 + "ping".len()
}

/// Server, proxy, and one authenticated client whose only peer is the proxy.
struct Rig {
    auth: Arc<ClusterAuth>,
    server: TcpTransport,
    client: TcpTransport,
    proxy: Proxy,
    proxy_addr: String,
}

impl Rig {
    async fn new() -> Self {
        let shared = auth();
        let server = bind(Arc::clone(&shared)).await;
        let proxy = Proxy::spawn(server.local_addr().unwrap()).await;
        let proxy_addr = proxy.addr.clone();
        let client = bind(Arc::clone(&shared)).await;
        Self {
            auth: shared,
            server,
            client,
            proxy,
            proxy_addr,
        }
    }

    /// Open (and authenticate) one pooled connection to the proxy, then send a
    /// calibration frame through it so the proxy learns the frame trailer
    /// length and switches to frame mode. Returns the proxy connection.
    async fn connect_and_calibrate(&self, index: usize) -> Arc<ProxyConn> {
        self.client
            .connect(std::slice::from_ref(&self.proxy_addr))
            .await
            .unwrap();
        let conn = self.proxy.conn(index).await;
        let before = conn.recorded_len().await;
        self.expect_delivery(b"calibrate").await;
        let frame = conn.recorded_from(before).await;
        let body_len = u64::from_be_bytes(frame[1..HEADER_LEN].try_into().unwrap()) as usize;
        let trailer = frame.len() - HEADER_LEN - body_len;
        conn.state.lock().await.trailer = Some(trailer);
        conn
    }

    /// Send one Control frame through the proxy and require it at the server.
    async fn expect_delivery(&self, payload: &[u8]) {
        self.client
            .send(&self.proxy_addr, control(payload))
            .await
            .unwrap();
        let (_, msg) = tokio::time::timeout(ARRIVAL, self.server.recv())
            .await
            .expect("frame must arrive within the bound")
            .unwrap();
        match msg {
            TransportMessage::Control { payload: got, .. } => assert_eq!(got.as_ref(), payload),
            other => panic!("expected Control, got {other:?}"),
        }
    }

    /// Nothing may reach `recv()` within [`SILENCE`].
    async fn expect_silence(&self, what: &str) {
        let got = tokio::time::timeout(SILENCE, self.server.recv()).await;
        assert!(got.is_err(), "{what} reached recv(): {got:?}");
    }

    async fn shutdown(self) {
        self.client.shutdown().await.unwrap();
        self.server.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn tampered_frame_is_rejected() {
    let rig = Rig::new().await;
    let conn = rig.connect_and_calibrate(0).await;

    // Flip one byte inside the payload of the next frame.
    let next = conn.recorded_len().await;
    conn.set_plan(Plan::FlipAt(next + control_payload_offset() + 2))
        .await;
    rig.client
        .send(&rig.proxy_addr, control(b"integrity"))
        .await
        .unwrap();

    rig.expect_silence("a tampered frame").await;
    conn.wait_server_closed().await;
    rig.shutdown().await;
}

#[tokio::test]
async fn replayed_frame_is_rejected() {
    let rig = Rig::new().await;
    let conn = rig.connect_and_calibrate(0).await;

    let start = conn.recorded_len().await;
    rig.expect_delivery(b"once").await;
    let frame = conn.recorded_from(start).await;

    conn.inject(&frame).await;
    rig.expect_silence("a replayed frame").await;
    conn.wait_server_closed().await;
    rig.shutdown().await;
}

#[tokio::test]
async fn reordered_frames_are_rejected() {
    let rig = Rig::new().await;
    let conn = rig.connect_and_calibrate(0).await;
    rig.expect_delivery(b"first").await;

    // Hold the second frame, let the third through, then release the second.
    conn.set_plan(Plan::Hold).await;
    rig.client
        .send(&rig.proxy_addr, control(b"second"))
        .await
        .unwrap();
    let held = conn.wait_held(1).await;
    conn.set_plan(Plan::Forward).await;
    rig.client
        .send(&rig.proxy_addr, control(b"third"))
        .await
        .unwrap();
    rig.expect_silence("an out-of-order frame").await;
    conn.inject(&held[0]).await;
    rig.expect_silence("a late frame").await;
    conn.wait_server_closed().await;
    rig.shutdown().await;
}

#[tokio::test]
async fn frame_from_other_connection_is_rejected() {
    let rig = Rig::new().await;
    let conn_a = rig.connect_and_calibrate(0).await;

    // A second authenticated client: proxy connection 1.
    let client_b = bind(Arc::clone(&rig.auth)).await;
    client_b
        .connect(std::slice::from_ref(&rig.proxy_addr))
        .await
        .unwrap();
    let conn_b = rig.proxy.conn(1).await;

    // Record a frame on connection A and splice it into connection B.
    let start = conn_a.recorded_len().await;
    rig.expect_delivery(b"from-a").await;
    let frame = conn_a.recorded_from(start).await;
    conn_b.inject(&frame).await;

    rig.expect_silence("a frame spliced from another connection")
        .await;
    conn_b.wait_server_closed().await;
    assert!(
        !conn_a.server_is_closed(),
        "the originating connection must stay open"
    );
    client_b.shutdown().await.unwrap();
    rig.shutdown().await;
}

#[tokio::test]
async fn activation_bytes_are_not_visible_on_the_wire() {
    let rig = Rig::new().await;
    let conn = rig.connect_and_calibrate(0).await;

    let pattern: Vec<u8> = b"MLXCEL-ACTIVATION-PATTERN-".repeat(64);
    let start = conn.recorded_len().await;
    rig.client
        .send(
            &rig.proxy_addr,
            TransportMessage::TensorData {
                tensor_id: "hidden".to_string(),
                shape: vec![pattern.len()],
                data: Bytes::from(pattern.clone()),
            },
        )
        .await
        .unwrap();
    let (_, msg) = tokio::time::timeout(ARRIVAL, rig.server.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(msg, TransportMessage::TensorData { data, .. } if data == pattern));

    let wire = conn.recorded_from(start).await;
    let probe = &pattern[..64];
    assert!(
        !wire.windows(probe.len()).any(|w| w == probe),
        "activation bytes appear in clear text on the wire"
    );
    rig.shutdown().await;
}

/// A header announcing more than the receiver's maximum is refused from the
/// header alone: the body is never read and no cryptographic work is done,
/// and the connection is closed.
#[tokio::test]
async fn protected_stream_refuses_oversize_header_before_body() {
    use crate::distributed::cluster_handshake::{client_handshake, server_handshake};
    use crate::distributed::frame_protection::{FrameRejected, FrameSuite};
    use crate::distributed::tcp_frame::ProtectedStream;
    use crate::distributed::transport::MessageKind;

    let auth = auth();
    let listener = TcpListener::bind("127.0.0.1:19551").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_auth = Arc::clone(&auth);
    let server = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let (_, keys) = server_handshake(
            &mut sock,
            &server_auth,
            "127.0.0.1:19552",
            FrameSuite::default(),
        )
        .await
        .unwrap();
        let mut stream = ProtectedStream::protected(sock, keys);
        // A tiny maximum; the peer sends only a header, so a receiver that
        // tried to read the announced body would block until the timeout.
        let res = tokio::time::timeout(ARRIVAL, stream.read_frame(1024)).await;
        (res, stream)
    });

    let mut sock = TcpStream::connect(addr).await.unwrap();
    let identity = sock.local_addr().unwrap().to_string();
    let _keys = client_handshake(&mut sock, &auth, &identity, FrameSuite::default())
        .await
        .unwrap();
    let mut header = vec![MessageKind::Control as u8];
    header.extend_from_slice(&(1u64 << 40).to_be_bytes());
    sock.write_all(&header).await.unwrap();
    sock.flush().await.unwrap();

    let (res, stream) = server.await.unwrap();
    let err = res
        .expect("the oversize header must be refused without waiting for a body")
        .expect_err("an oversize frame must be an error");
    assert!(
        err.to_string().contains("exceeds maximum 1024"),
        "unexpected error: {err:#}"
    );
    assert!(
        err.downcast_ref::<FrameRejected>().is_none(),
        "the size check runs before authentication, not as an auth failure"
    );

    // The caller closes the connection on error; the peer sees the close.
    drop(stream);
    let mut buf = [0u8; 1];
    let closed = tokio::time::timeout(ARRIVAL, sock.read(&mut buf))
        .await
        .expect("the connection must be closed");
    assert!(matches!(closed, Ok(0) | Err(_)));
}

/// Shared by the kind-byte tests: a protected server reading one frame with a
/// 1 MiB bound, and an authenticated client socket plus its sealer.
async fn kind_byte_rig() -> (
    tokio::task::JoinHandle<anyhow::Result<(crate::distributed::transport::MessageKind, Vec<u8>)>>,
    TcpStream,
    crate::distributed::frame_protection::FrameSealer,
) {
    use crate::distributed::cluster_handshake::{client_handshake, server_handshake};
    use crate::distributed::frame_protection::FrameSuite;
    use crate::distributed::tcp_frame::ProtectedStream;

    let auth = auth();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_auth = Arc::clone(&auth);
    let server = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let (_, keys) = server_handshake(
            &mut sock,
            &server_auth,
            "127.0.0.1:1",
            FrameSuite::default(),
        )
        .await
        .unwrap();
        let mut stream = ProtectedStream::protected(sock, keys);
        tokio::time::timeout(ARRIVAL, stream.read_frame(1024 * 1024))
            .await
            .expect("read_frame must settle within the bound")
            .map(|(kind, payload)| (kind, payload.to_vec()))
    });

    let mut sock = TcpStream::connect(addr).await.unwrap();
    let identity = sock.local_addr().unwrap().to_string();
    let keys = client_handshake(&mut sock, &auth, &identity, FrameSuite::default())
        .await
        .unwrap();
    let (sealer, _) = keys.into_codec();
    (server, sock, sealer)
}

/// A kind byte flipped to an invalid value (0xFF) after sealing is an
/// authentication failure: the header is associated data, so the receiver
/// must read the body, fail `open`, and report `FrameRejected` (which the
/// transport and the assignment listener log through the rate-limited
/// diagnostic budget), instead of closing silently on the header alone.
#[tokio::test]
async fn protected_stream_rejects_tampered_invalid_kind_as_frame_rejected() {
    use crate::distributed::frame_protection::FrameRejected;
    use crate::distributed::tcp_frame::{encode_message_frame, finish_frame};

    let (server, mut sock, mut sealer) = kind_byte_rig().await;
    let (kind, mut frame) = encode_message_frame(&control(b"kind-flip"));
    finish_frame(Some(&mut sealer), kind, &mut frame).unwrap();
    frame[0] = 0xFF;
    sock.write_all(&frame).await.unwrap();
    sock.flush().await.unwrap();

    let err = server
        .await
        .unwrap()
        .expect_err("a tampered kind byte must be an error");
    assert!(
        err.downcast_ref::<FrameRejected>().is_some(),
        "a tampered invalid kind must surface as FrameRejected, got: {err:#}"
    );
}

/// An invalid kind byte that the peer really sealed (a version-skewed member
/// speaking a newer frame kind) authenticates and is reported as such, not
/// as a tampered frame.
#[tokio::test]
async fn protected_stream_reports_authenticated_unknown_kind_distinctly() {
    use crate::distributed::frame_protection::{FRAME_HEADER_LEN, FrameRejected};

    let (server, mut sock, mut sealer) = kind_byte_rig().await;
    let payload = b"future-kind";
    let mut frame = vec![0xFFu8];
    frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    frame.extend_from_slice(payload);
    assert_eq!(frame.len(), FRAME_HEADER_LEN + payload.len());
    sealer.seal(&mut frame).unwrap();
    sock.write_all(&frame).await.unwrap();
    sock.flush().await.unwrap();

    let err = server
        .await
        .unwrap()
        .expect_err("an unknown kind must be an error");
    assert!(
        err.downcast_ref::<FrameRejected>().is_none(),
        "an authenticated unknown kind is not an auth failure: {err:#}"
    );
    assert!(
        err.to_string()
            .contains("unknown message kind 255 from an authenticated peer"),
        "unexpected error: {err:#}"
    );
}

/// A plain stream keeps parsing the kind first: an invalid kind byte is a
/// plain error and never `FrameRejected`.
#[tokio::test]
async fn plain_stream_invalid_kind_is_not_frame_rejected() {
    use crate::distributed::frame_protection::FrameRejected;
    use crate::distributed::tcp_frame::ProtectedStream;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (sock, _) = listener.accept().await.unwrap();
        let mut stream = ProtectedStream::plain(sock);
        tokio::time::timeout(ARRIVAL, stream.read_frame(1024))
            .await
            .expect("read_frame must settle within the bound")
    });

    let mut sock = TcpStream::connect(addr).await.unwrap();
    let mut frame = vec![0xFFu8];
    frame.extend_from_slice(&4u64.to_be_bytes());
    frame.extend_from_slice(b"ping");
    sock.write_all(&frame).await.unwrap();
    sock.flush().await.unwrap();

    let err = server
        .await
        .unwrap()
        .expect_err("an invalid kind must be an error on a plain stream");
    assert!(err.downcast_ref::<FrameRejected>().is_none(), "{err:#}");
    assert!(
        err.to_string().contains("unknown message kind"),
        "unexpected error: {err:#}"
    );
}

/// The same rule on the transport: an oversize header from an authenticated
/// peer makes the server close the connection without reading a body.
#[tokio::test]
async fn transport_closes_connection_on_oversize_header() {
    use crate::distributed::cluster_handshake::client_handshake;
    use crate::distributed::frame_protection::FrameSuite;
    use crate::distributed::transport::MessageKind;

    let auth = auth();
    let server = TcpTransport::bind(TcpTransportConfig {
        bind_address: "127.0.0.1:19553".to_string(),
        auth: Some(Arc::clone(&auth)),
        max_frame_size: 1024,
        ..Default::default()
    })
    .await
    .unwrap();

    let mut sock = TcpStream::connect(server.local_addr().unwrap())
        .await
        .unwrap();
    let identity = sock.local_addr().unwrap().to_string();
    let _keys = client_handshake(&mut sock, &auth, &identity, FrameSuite::default())
        .await
        .unwrap();
    let mut header = vec![MessageKind::Control as u8];
    header.extend_from_slice(&(1u64 << 40).to_be_bytes());
    sock.write_all(&header).await.unwrap();
    sock.flush().await.unwrap();

    let mut buf = [0u8; 1];
    let closed = tokio::time::timeout(ARRIVAL, sock.read(&mut buf))
        .await
        .expect("the server must close instead of waiting for the body");
    assert!(matches!(closed, Ok(0) | Err(_)));
    let got = tokio::time::timeout(SILENCE, server.recv()).await;
    assert!(got.is_err(), "nothing may reach recv(): {got:?}");
    server.shutdown().await.unwrap();
}
