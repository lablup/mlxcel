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

use tokio::net::TcpListener;

use super::*;
use crate::distributed::cluster_auth::ClusterSecret;
use crate::distributed::frame_protection::FrameSuite;

async fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let client = TcpStream::connect(addr);
    let (server, client) = tokio::join!(listener.accept(), client);
    (server.unwrap().0, client.unwrap())
}

#[tokio::test]
async fn handshake_succeeds_with_shared_secret_and_binds_identities() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let (server_res, client_res) = tokio::join!(
        server_handshake(&mut server, &auth, "0.0.0.0:19001", FrameSuite::default()),
        client_handshake(&mut client, &auth, "127.0.0.1:19000", FrameSuite::default()),
    );
    assert_eq!(server_res.unwrap().0, "127.0.0.1:19000");
    client_res.unwrap();
}

#[tokio::test]
async fn client_accepts_server_advertising_a_forwarded_address() {
    // A stage reached through NAT or a port forward advertises its own
    // listener address, which is not the IP the coordinator connected to.
    // The server's authenticator already proves it holds the secret, so the
    // connect side must not refuse it over the advertised address.
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let (server_res, client_res) = tokio::join!(
        server_handshake(&mut server, &auth, "10.9.8.7:19001", FrameSuite::default()),
        client_handshake(&mut client, &auth, "127.0.0.1:19000", FrameSuite::default()),
    );
    assert_eq!(server_res.unwrap().0, "127.0.0.1:19000");
    client_res.unwrap();
}

#[tokio::test]
async fn handshake_rejects_client_with_wrong_secret() {
    let server_auth = ClusterAuth::new(&ClusterSecret::random());
    let client_auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let (server_res, client_res) = tokio::join!(
        server_handshake(
            &mut server,
            &server_auth,
            "127.0.0.1:1",
            FrameSuite::default()
        ),
        client_handshake(
            &mut client,
            &client_auth,
            "127.0.0.1:2",
            FrameSuite::default()
        ),
    );
    // Mutual: the client detects the server's bad authenticator first and
    // never sends its own, so the server sees a closed connection.
    let client_err = client_res.unwrap_err().to_string();
    assert!(client_err.contains("secret mismatch"), "{client_err}");
    assert!(server_res.is_err());
}

#[tokio::test]
async fn handshake_rejects_forged_client_tag() {
    let auth = std::sync::Arc::new(ClusterAuth::new(&ClusterSecret::random()));
    let (mut server, mut client) = pair().await;
    let server_auth = std::sync::Arc::clone(&auth);
    let server_task = tokio::spawn(async move {
        let res = server_handshake(
            &mut server,
            &server_auth,
            "127.0.0.1:1",
            FrameSuite::default(),
        )
        .await;
        res.map(|_| ())
    });
    // Hand-roll a client that sends a valid hello and then a garbage tag.
    // Even the reject verdict comes as the authenticated 33-byte record
    // (#2714 item 1).
    let transcript = handroll_client_through_tag_c(&mut client, &auth, "127.0.0.1:2", true).await;
    let verdict = read_authenticated_verdict(&mut client, &auth, &transcript).await;
    assert_eq!(verdict, VERDICT_REJECT);
    let err = server_task.await.unwrap().unwrap_err().to_string();
    assert!(err.contains("secret mismatch"), "{err}");
}

#[tokio::test]
async fn handshake_rejects_client_advertising_foreign_ip() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let (server_res, client_res) = tokio::join!(
        server_handshake(&mut server, &auth, "127.0.0.1:1", FrameSuite::default()),
        client_handshake(&mut client, &auth, "10.9.8.7:2", FrameSuite::default()),
    );
    let err = server_res.unwrap_err();
    assert!(err.downcast_ref::<IdentityNotOwned>().is_some(), "{err:#}");
    assert!(err.to_string().contains("does not own"), "{err}");
    // The client gets the typed rejection, not a bare EOF or a timeout.
    let client_err = client_res.unwrap_err();
    assert!(
        client_err.downcast_ref::<HandshakeRejected>().is_some(),
        "{client_err:#}"
    );
}

/// Identity binding failure is an explicit, typed rejection on the client
/// (#2655 item b): the client learns within 1 s that its advertised identity
/// was refused, not a bare EOF or the 5 s handshake timeout.
#[tokio::test]
async fn client_reports_identity_rejection_as_typed_rejection() {
    let secret = ClusterSecret::random();
    let server_auth = ClusterAuth::new(&secret);
    let client_auth = ClusterAuth::new(&secret);
    let (mut server, mut client) = pair().await;
    let server_task = tokio::spawn(async move {
        server_handshake(
            &mut server,
            &server_auth,
            "127.0.0.1:1",
            FrameSuite::default(),
        )
        .await
        .map(|_| ())
    });
    let client_err = tokio::time::timeout(
        Duration::from_secs(1),
        client_handshake(
            &mut client,
            &client_auth,
            "10.9.8.7:2",
            FrameSuite::default(),
        ),
    )
    .await
    .expect("identity rejection must reach the client within 1 s")
    .unwrap_err();
    let rejected = client_err
        .downcast_ref::<HandshakeRejected>()
        .unwrap_or_else(|| panic!("expected HandshakeRejected, got: {client_err:#}"));
    assert!(rejected.0.contains("10.9.8.7:2"), "{rejected}");
    assert!(rejected.0.contains("advertised IP"), "{rejected}");
    assert!(server_task.await.unwrap().is_err());
}

/// A client holding another secret detects the server's bad authenticator
/// first and closes; the server must report that as a probable secret
/// mismatch, not as a generic closed connection (#2655 item b).
#[tokio::test]
async fn server_reports_client_abort_after_proof_as_probable_secret_mismatch() {
    let server_auth = ClusterAuth::new(&ClusterSecret::random());
    let client_auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let server_task = tokio::spawn(async move {
        server_handshake(
            &mut server,
            &server_auth,
            "127.0.0.1:1",
            FrameSuite::default(),
        )
        .await
        .map(|_| ())
    });
    let client_err = client_handshake(
        &mut client,
        &client_auth,
        "127.0.0.1:2",
        FrameSuite::default(),
    )
    .await
    .unwrap_err();
    assert!(
        client_err.to_string().contains("secret mismatch"),
        "{client_err}"
    );
    // The connection pool drops the socket on a failed handshake; do the same.
    drop(client);
    let server_err = tokio::time::timeout(Duration::from_secs(1), server_task)
        .await
        .expect("server must notice the abort well before the handshake timeout")
        .unwrap()
        .unwrap_err();
    assert!(
        server_err
            .downcast_ref::<ClientAbortedAfterServerProof>()
            .is_some(),
        "{server_err:#}"
    );
    assert!(
        server_err.to_string().contains("different cluster secret"),
        "{server_err:#}"
    );
}

/// The identity rejection path sends the 9-byte rejection record and nothing
/// else: no server hello, no nonce, no authenticator (#2655 item b).
#[tokio::test]
async fn identity_rejection_sends_only_the_reject_record() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let server_task = tokio::spawn(async move {
        server_handshake(&mut server, &auth, "127.0.0.1:1", FrameSuite::default())
            .await
            .map(|_| ())
    });
    let hello = encode_hello(&[9u8; NONCE_LEN], "10.9.8.7:2", FrameSuite::default()).unwrap();
    client.write_all(&hello).await.unwrap();
    client.flush().await.unwrap();
    let mut received = Vec::new();
    tokio::time::timeout(Duration::from_secs(1), client.read_to_end(&mut received))
        .await
        .expect("server must close the connection after the rejection record")
        .unwrap();
    let mut expected = REJECT_MAGIC.to_vec();
    expected.push(REJECT_IDENTITY_NOT_OWNED);
    assert_eq!(received, expected);
    let err = server_task.await.unwrap().unwrap_err();
    let typed = err
        .downcast_ref::<IdentityNotOwned>()
        .unwrap_or_else(|| panic!("expected IdentityNotOwned, got: {err:#}"));
    assert_eq!(typed.advertised, "10.9.8.7:2");
    assert_eq!(typed.source_ip, IpAddr::V4(Ipv4Addr::LOCALHOST));
}

/// A hello whose identity is not a socket address at all is a protocol
/// error, not an identity-ownership refusal: the server closes without the
/// rejection record (which would tell the client to fix an address it never
/// sent) and logs the identity escaped, since it is unauthenticated input.
#[tokio::test]
async fn unparseable_identity_closes_without_a_reject_record() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let server_task = tokio::spawn(async move {
        server_handshake(&mut server, &auth, "127.0.0.1:1", FrameSuite::default())
            .await
            .map(|_| ())
    });
    let hello = encode_hello(
        &[9u8; NONCE_LEN],
        "node-a\nforged log line",
        FrameSuite::default(),
    )
    .unwrap();
    client.write_all(&hello).await.unwrap();
    client.flush().await.unwrap();
    let mut received = Vec::new();
    tokio::time::timeout(Duration::from_secs(1), client.read_to_end(&mut received))
        .await
        .expect("server must close the connection on a malformed identity")
        .unwrap();
    assert!(received.is_empty(), "server replied with {received:?}");
    let err = server_task.await.unwrap().unwrap_err();
    assert!(err.downcast_ref::<IdentityNotOwned>().is_none(), "{err:#}");
    let message = format!("{err:#}");
    assert!(message.contains("is not a socket address"), "{message}");
    assert!(
        !message.contains('\n'),
        "identity must be escaped: {message}"
    );
}

/// A client that aborts with a reset (unread data or `SO_LINGER` 0) after
/// the server's proof is the same probable secret mismatch as an orderly
/// close, not a generic I/O error (#2655 item b).
#[tokio::test]
async fn server_reports_client_reset_after_proof_as_probable_secret_mismatch() {
    let server_auth = ClusterAuth::new(&ClusterSecret::random());
    let client_auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let server_task = tokio::spawn(async move {
        server_handshake(
            &mut server,
            &server_auth,
            "127.0.0.1:1",
            FrameSuite::default(),
        )
        .await
        .map(|_| ())
    });
    client_handshake(
        &mut client,
        &client_auth,
        "127.0.0.1:2",
        FrameSuite::default(),
    )
    .await
    .unwrap_err();
    client.set_zero_linger().unwrap();
    drop(client);
    let server_err = tokio::time::timeout(Duration::from_secs(1), server_task)
        .await
        .expect("server must notice the reset well before the handshake timeout")
        .unwrap()
        .unwrap_err();
    assert!(
        server_err
            .downcast_ref::<ClientAbortedAfterServerProof>()
            .is_some(),
        "{server_err:#}"
    );
}

#[test]
fn peer_abort_classifies_eof_reset_and_broken_pipe() {
    use std::io::{Error, ErrorKind};
    for kind in [
        ErrorKind::UnexpectedEof,
        ErrorKind::ConnectionReset,
        ErrorKind::ConnectionAborted,
        ErrorKind::BrokenPipe,
    ] {
        let err = anyhow::Error::from(Error::from(kind)).context("reading the client tag");
        assert!(is_peer_abort(&err), "{kind:?}");
    }
    let other = anyhow::Error::from(Error::from(ErrorKind::PermissionDenied)).context("x");
    assert!(!is_peer_abort(&other));
}

#[tokio::test]
async fn handshake_rejects_peer_that_speaks_plain_frames() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    // A pre-auth transport frame: kind=Control(?) plus a length header.
    let mut garbage = vec![0x02u8];
    garbage.extend_from_slice(&(100u64).to_be_bytes());
    garbage.extend_from_slice(&[0u8; 64]);
    client.write_all(&garbage).await.unwrap();
    let err = server_handshake(&mut server, &auth, "127.0.0.1:1", FrameSuite::default())
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("did not start the cluster handshake"), "{err}");
}

#[tokio::test]
async fn handshake_reports_probe_that_closes_without_bytes() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, client) = pair().await;
    drop(client);
    let err = server_handshake(&mut server, &auth, "127.0.0.1:1", FrameSuite::default())
        .await
        .unwrap_err();
    assert!(
        err.downcast_ref::<PeerClosedBeforeHandshake>().is_some(),
        "{err}"
    );
}

#[tokio::test]
async fn handshake_times_out_on_silent_peer() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, _client) = pair().await;
    tokio::time::pause();
    let fut = server_handshake(&mut server, &auth, "127.0.0.1:1", FrameSuite::default());
    tokio::pin!(fut);
    tokio::time::advance(HANDSHAKE_TIMEOUT + Duration::from_millis(1)).await;
    let err = fut.await.unwrap_err().to_string();
    assert!(err.contains("timed out"), "{err}");
}

#[test]
fn bind_identity_substitutes_unspecified_and_rejects_mismatch() {
    let src = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5));
    assert_eq!(
        bind_advertised_identity("0.0.0.0:9000", src).unwrap(),
        "10.0.0.5:9000"
    );
    assert_eq!(
        bind_advertised_identity("10.0.0.5:9000", src).unwrap(),
        "10.0.0.5:9000"
    );
    assert!(bind_advertised_identity("10.0.0.6:9000", src).is_err());
    assert!(bind_advertised_identity("not-an-addr", src).is_err());
    // IPv4-mapped IPv6 source canonicalizes to the IPv4 form.
    let mapped: IpAddr = "::ffff:10.0.0.5".parse().unwrap();
    assert_eq!(
        bind_advertised_identity("10.0.0.5:9000", mapped).unwrap(),
        "10.0.0.5:9000"
    );
}

/// Both ends derive the same directional keys: what the client seals under
/// its send key, the server opens under its receive key, and the other way
/// round (#2656).
#[tokio::test]
async fn handshake_derives_matching_frame_keys_on_both_sides() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let (server_res, client_res) = tokio::join!(
        server_handshake(&mut server, &auth, "127.0.0.1:1", FrameSuite::default()),
        client_handshake(&mut client, &auth, "127.0.0.1:2", FrameSuite::default()),
    );
    let (_, server_keys) = server_res.unwrap();
    let client_keys = client_res.unwrap();
    let (mut c_seal, mut c_open) = client_keys.into_codec();
    let (mut s_seal, mut s_open) = server_keys.into_codec();
    for (seal, open) in [(&mut c_seal, &mut s_open), (&mut s_seal, &mut c_open)] {
        let mut frame = vec![2u8];
        frame.extend_from_slice(&5u64.to_be_bytes());
        frame.extend_from_slice(b"hello");
        seal.seal(&mut frame).unwrap();
        let header: [u8; 9] = frame[..9].try_into().unwrap();
        let payload_len = open.open(&header, &mut frame[9..]).unwrap();
        assert_eq!(&frame[9..9 + payload_len], b"hello");
    }
}

/// Two handshakes under the same secret yield different keys (fresh nonces),
/// so a frame from one connection does not verify on another (#2656).
#[tokio::test]
async fn handshake_keys_are_per_connection() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let mut codecs = Vec::new();
    for _ in 0..2 {
        let (mut server, mut client) = pair().await;
        let (server_res, client_res) = tokio::join!(
            server_handshake(&mut server, &auth, "127.0.0.1:1", FrameSuite::default()),
            client_handshake(&mut client, &auth, "127.0.0.1:2", FrameSuite::default()),
        );
        codecs.push((
            client_res.unwrap().into_codec(),
            server_res.unwrap().1.into_codec(),
        ));
    }
    let (mut a_seal, _) = codecs.remove(0).0;
    let (_, mut b_open) = codecs.remove(0).1;
    let mut frame = vec![2u8];
    frame.extend_from_slice(&3u64.to_be_bytes());
    frame.extend_from_slice(b"abc");
    a_seal.seal(&mut frame).unwrap();
    let header: [u8; 9] = frame[..9].try_into().unwrap();
    assert!(b_open.open(&header, &mut frame[9..]).is_err());
}

/// A suite mismatch is refused with the explicit record before any nonce or
/// authenticator, and both sides report it as a typed, terminal rejection.
#[tokio::test]
async fn handshake_rejects_frame_suite_mismatch_with_typed_errors() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let (server_res, client_res) = tokio::join!(
        server_handshake(&mut server, &auth, "127.0.0.1:1", FrameSuite::HmacSha256),
        client_handshake(&mut client, &auth, "127.0.0.1:2", FrameSuite::Aes256Gcm),
    );
    let server_err = server_res.unwrap_err();
    let typed = server_err
        .downcast_ref::<FrameSuiteMismatch>()
        .unwrap_or_else(|| panic!("expected FrameSuiteMismatch, got: {server_err:#}"));
    assert_eq!(typed.peer_suite_id, FrameSuite::Aes256Gcm.wire_id());
    assert_eq!(typed.local, FrameSuite::HmacSha256);
    let client_err = client_res.unwrap_err();
    let rejected = client_err
        .downcast_ref::<HandshakeRejected>()
        .unwrap_or_else(|| panic!("expected HandshakeRejected, got: {client_err:#}"));
    assert!(rejected.0.contains("frame protection suite"), "{rejected}");
}

/// A full connection cap is reported inside the handshake, after the client
/// authenticated: the client gets the typed, retryable `ServerAtCapacity`
/// (not `HandshakeRejected`) and the server `ConnectionLimitReached` naming
/// the peer it turned away.
#[tokio::test]
async fn server_at_capacity_sends_capacity_verdict() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let (server_res, client_res) = tokio::join!(
        server_handshake_admitting::<()>(
            &mut server,
            &auth,
            "127.0.0.1:1",
            FrameSuite::default(),
            FrameChannel::Transport,
            || None,
        ),
        client_handshake(&mut client, &auth, "127.0.0.1:2", FrameSuite::default()),
    );
    let client_err = client_res.unwrap_err();
    assert!(
        client_err.downcast_ref::<ServerAtCapacity>().is_some(),
        "expected ServerAtCapacity, got: {client_err:#}"
    );
    assert!(client_err.downcast_ref::<HandshakeRejected>().is_none());
    assert!(
        client_err.to_string().contains("retry later"),
        "{client_err}"
    );
    let server_err = server_res.unwrap_err();
    let full = server_err
        .downcast_ref::<ConnectionLimitReached>()
        .unwrap_or_else(|| panic!("expected ConnectionLimitReached, got: {server_err:#}"));
    assert_eq!(full.peer, "127.0.0.1:2");
}

/// `admit` runs only after the client's authenticator verified: a client with
/// another secret never triggers it, so it learns nothing about capacity.
#[tokio::test]
async fn admission_is_not_consulted_before_the_client_authenticates() {
    let server_auth = ClusterAuth::new(&ClusterSecret::random());
    let client_auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let admitted = std::sync::atomic::AtomicBool::new(false);
    let (server_res, client_res) = tokio::join!(
        server_handshake_admitting::<()>(
            &mut server,
            &server_auth,
            "127.0.0.1:1",
            FrameSuite::default(),
            FrameChannel::Transport,
            || {
                admitted.store(true, std::sync::atomic::Ordering::SeqCst);
                None
            },
        ),
        client_handshake(
            &mut client,
            &client_auth,
            "127.0.0.1:2",
            FrameSuite::default()
        ),
    );
    assert!(server_res.is_err());
    assert!(client_res.is_err());
    assert!(!admitted.load(std::sync::atomic::Ordering::SeqCst));
}

// ---------------------------------------------------------------------------
// Authenticated verdict (#2714 item 1)
// ---------------------------------------------------------------------------

/// Hand-rolled connect side up to and including `tag_c`: sends a hello,
/// reads the server hello and `tag_s`, sends `tag_c` (forged when `forge_tag`
/// is set) and returns the transcript `client_hello || server_hello || tag_s
/// || tag_c`, which is what the verdict record is signed over.
async fn handroll_client_through_tag_c(
    client: &mut TcpStream,
    auth: &ClusterAuth,
    identity: &str,
    forge_tag: bool,
) -> Vec<u8> {
    let hello = encode_hello(&[7u8; NONCE_LEN], identity, FrameSuite::default()).unwrap();
    client.write_all(&hello).await.unwrap();
    let server_hello = read_hello(
        client,
        HelloFrom::Server {
            local_identity: identity,
        },
    )
    .await
    .unwrap();
    let tag_s = read_tag(client, "server").await.unwrap();
    let mut transcript = hello;
    transcript.extend_from_slice(&server_hello.raw);
    assert!(auth.verify(SERVER_LABEL, &transcript, &tag_s));
    transcript.extend_from_slice(&tag_s);
    let tag_c = if forge_tag {
        [0u8; TAG_LEN]
    } else {
        auth.sign(CLIENT_LABEL, &transcript)
    };
    client.write_all(&tag_c).await.unwrap();
    client.flush().await.unwrap();
    transcript.extend_from_slice(&tag_c);
    transcript
}

/// Read the verdict record and check its authenticator against
/// `transcript`; returns the verdict byte.
async fn read_authenticated_verdict(
    client: &mut TcpStream,
    auth: &ClusterAuth,
    transcript: &[u8],
) -> u8 {
    let mut record = [0u8; VERDICT_RECORD_LEN];
    client
        .read_exact(&mut record)
        .await
        .expect("the verdict record must be verdict || tag_v");
    let mut signed = transcript.to_vec();
    signed.push(record[0]);
    assert!(
        auth.verify(VERDICT_LABEL, &signed, &record[1..]),
        "verdict {:#04x} does not verify under the cluster secret",
        record[0]
    );
    record[0]
}

/// A TCP proxy between the connect side and the accept side that XORs
/// `flip_mask` into byte `flip_offset` of the server-to-client stream and
/// passes every other byte through unchanged (the on-path attacker of #2714
/// item 1).
///
/// Returns the streams the two handshake halves run on: the accept side's
/// socket (connected from the proxy) and the connect side's socket (dialed
/// into the proxy).
async fn tampering_proxy_pair(flip_offset: usize, flip_mask: u8) -> (TcpStream, TcpStream) {
    let server_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = server_listener.local_addr().unwrap();
    let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_addr = proxy_listener.local_addr().unwrap();

    let client = TcpStream::connect(proxy_addr);
    let (proxy_in, client) = tokio::join!(proxy_listener.accept(), client);
    let (proxy_in, _) = proxy_in.unwrap();
    let client = client.unwrap();
    let proxy_out = TcpStream::connect(server_addr);
    let (server, proxy_out) = tokio::join!(server_listener.accept(), proxy_out);
    let (server, _) = server.unwrap();
    let proxy_out = proxy_out.unwrap();

    let (mut in_r, mut in_w) = proxy_in.into_split();
    let (mut out_r, mut out_w) = proxy_out.into_split();
    // Client to server: pass through.
    tokio::spawn(async move {
        let _ = tokio::io::copy(&mut in_r, &mut out_w).await;
    });
    // Server to client: flip the one byte.
    tokio::spawn(async move {
        let mut seen = 0usize;
        let mut buf = [0u8; 1024];
        loop {
            let n = match out_r.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            if flip_offset >= seen && flip_offset < seen + n {
                buf[flip_offset - seen] ^= flip_mask;
            }
            seen += n;
            if in_w.write_all(&buf[..n]).await.is_err() {
                break;
            }
            let _ = in_w.flush().await;
        }
    });
    (server, client)
}

const SERVER_IDENTITY: &str = "127.0.0.1:19901";
const CLIENT_IDENTITY: &str = "127.0.0.1:19902";

/// Offset of the verdict byte in the server-to-client stream.
fn verdict_offset() -> usize {
    HELLO_HEAD_LEN + SERVER_IDENTITY.len() + TAG_LEN
}

/// An on-path attacker turns an accept (0x01) into an at-capacity verdict
/// (0x02). Without the authenticator the client would report a healthy
/// server as full and back off; with it the client sees the tampering as a
/// terminal rejection, not as `ServerAtCapacity`.
#[tokio::test]
async fn flipped_accept_to_at_capacity_verdict_is_rejected_not_retried() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) =
        tampering_proxy_pair(verdict_offset(), VERDICT_ACCEPT ^ VERDICT_AT_CAPACITY).await;
    let (_server_res, client_res) = tokio::join!(
        server_handshake(&mut server, &auth, SERVER_IDENTITY, FrameSuite::default()),
        client_handshake(&mut client, &auth, CLIENT_IDENTITY, FrameSuite::default()),
    );
    let client_err = client_res.expect_err("a tampered verdict must not be accepted");
    assert!(
        client_err.downcast_ref::<ServerAtCapacity>().is_none(),
        "the client took the forged capacity verdict at face value: {client_err:#}"
    );
    let rejected = client_err
        .downcast_ref::<HandshakeRejected>()
        .unwrap_or_else(|| panic!("expected HandshakeRejected, got: {client_err:#}"));
    assert!(rejected.0.contains("verdict"), "{rejected}");
}

/// The reverse flip: the server is full (0x02) and the attacker turns it into
/// an accept (0x01). The client must not come up on a connection the server
/// is about to close.
#[tokio::test]
async fn flipped_at_capacity_to_accept_verdict_is_rejected_not_accepted() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) =
        tampering_proxy_pair(verdict_offset(), VERDICT_ACCEPT ^ VERDICT_AT_CAPACITY).await;
    let (server_res, client_res) = tokio::join!(
        server_handshake_admitting::<()>(
            &mut server,
            &auth,
            SERVER_IDENTITY,
            FrameSuite::default(),
            FrameChannel::Transport,
            || None,
        ),
        client_handshake(&mut client, &auth, CLIENT_IDENTITY, FrameSuite::default()),
    );
    assert!(
        server_res
            .unwrap_err()
            .downcast_ref::<ConnectionLimitReached>()
            .is_some()
    );
    let client_err = client_res.expect_err("a forged accept must not open the connection");
    assert!(
        client_err.downcast_ref::<HandshakeRejected>().is_some(),
        "expected HandshakeRejected, got: {client_err:#}"
    );
}

/// One altered byte of `tag_v` is a rejection even when the verdict byte
/// itself is untouched.
#[tokio::test]
async fn flipped_verdict_tag_byte_is_rejected() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = tampering_proxy_pair(verdict_offset() + 1, 0x80).await;
    let (_server_res, client_res) = tokio::join!(
        server_handshake(&mut server, &auth, SERVER_IDENTITY, FrameSuite::default()),
        client_handshake(&mut client, &auth, CLIENT_IDENTITY, FrameSuite::default()),
    );
    let client_err = client_res.expect_err("a tampered verdict tag must not be accepted");
    let rejected = client_err
        .downcast_ref::<HandshakeRejected>()
        .unwrap_or_else(|| panic!("expected HandshakeRejected, got: {client_err:#}"));
    assert!(rejected.0.contains("verdict"), "{rejected}");
}

/// The at-capacity verdict is the 33-byte authenticated record, signed over
/// the whole transcript including the client's own authenticator.
#[tokio::test]
async fn server_at_capacity_sends_authenticated_capacity_verdict_record() {
    let auth = std::sync::Arc::new(ClusterAuth::new(&ClusterSecret::random()));
    let (mut server, mut client) = pair().await;
    let server_auth = std::sync::Arc::clone(&auth);
    let server_task = tokio::spawn(async move {
        server_handshake_admitting::<()>(
            &mut server,
            &server_auth,
            "127.0.0.1:1",
            FrameSuite::default(),
            FrameChannel::Transport,
            || None,
        )
        .await
        .map(|_| ())
    });
    let transcript = handroll_client_through_tag_c(&mut client, &auth, "127.0.0.1:2", false).await;
    let verdict = read_authenticated_verdict(&mut client, &auth, &transcript).await;
    assert_eq!(verdict, VERDICT_AT_CAPACITY);
    let server_err = server_task.await.unwrap().unwrap_err();
    assert!(
        server_err
            .downcast_ref::<ConnectionLimitReached>()
            .is_some(),
        "{server_err:#}"
    );
}

/// A hello from a version-4 build (no verdict authenticator) is refused with
/// the explicit version message, not with a confusing short read later.
#[tokio::test]
async fn version_4_hello_is_refused_with_the_version_message() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let mut hello = encode_hello(&[3u8; NONCE_LEN], "127.0.0.1:2", FrameSuite::default()).unwrap();
    hello[8..10].copy_from_slice(&4u16.to_be_bytes());
    client.write_all(&hello).await.unwrap();
    client.flush().await.unwrap();
    let err = server_handshake(&mut server, &auth, "127.0.0.1:1", FrameSuite::default())
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("speaks cluster handshake version 4, this build expects 5"),
        "{err}"
    );
}
