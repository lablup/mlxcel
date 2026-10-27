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
        server_handshake(&mut server, &auth, "0.0.0.0:19001"),
        client_handshake(&mut client, &auth, "127.0.0.1:19000"),
    );
    assert_eq!(server_res.unwrap(), "127.0.0.1:19000");
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
        server_handshake(&mut server, &auth, "10.9.8.7:19001"),
        client_handshake(&mut client, &auth, "127.0.0.1:19000"),
    );
    assert_eq!(server_res.unwrap(), "127.0.0.1:19000");
    client_res.unwrap();
}

#[tokio::test]
async fn handshake_rejects_client_with_wrong_secret() {
    let server_auth = ClusterAuth::new(&ClusterSecret::random());
    let client_auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let (server_res, client_res) = tokio::join!(
        server_handshake(&mut server, &server_auth, "127.0.0.1:1"),
        client_handshake(&mut client, &client_auth, "127.0.0.1:2"),
    );
    // Mutual: the client detects the server's bad authenticator first and
    // never sends its own, so the server sees a closed connection.
    let client_err = client_res.unwrap_err().to_string();
    assert!(client_err.contains("secret mismatch"), "{client_err}");
    assert!(server_res.is_err());
}

#[tokio::test]
async fn handshake_rejects_forged_client_tag() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let server_task = tokio::spawn(async move {
        let res = server_handshake(&mut server, &auth, "127.0.0.1:1").await;
        res.map(|_| ())
    });
    // Hand-roll a client that sends a valid hello and then a garbage tag.
    let hello = encode_hello(&[7u8; NONCE_LEN], "127.0.0.1:2").unwrap();
    client.write_all(&hello).await.unwrap();
    let (_server_hello, _identity) = read_hello(
        &mut client,
        HelloFrom::Server {
            local_identity: "127.0.0.1:2",
        },
    )
    .await
    .unwrap();
    let _tag_s = read_tag(&mut client, "server").await.unwrap();
    client.write_all(&[0u8; TAG_LEN]).await.unwrap();
    let mut verdict = [0u8; 1];
    client.read_exact(&mut verdict).await.unwrap();
    assert_eq!(verdict[0], VERDICT_REJECT);
    let err = server_task.await.unwrap().unwrap_err().to_string();
    assert!(err.contains("secret mismatch"), "{err}");
}

#[tokio::test]
async fn handshake_rejects_client_advertising_foreign_ip() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    let (server_res, client_res) = tokio::join!(
        server_handshake(&mut server, &auth, "127.0.0.1:1"),
        client_handshake(&mut client, &auth, "10.9.8.7:2"),
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
        server_handshake(&mut server, &server_auth, "127.0.0.1:1")
            .await
            .map(|_| ())
    });
    let client_err = tokio::time::timeout(
        Duration::from_secs(1),
        client_handshake(&mut client, &client_auth, "10.9.8.7:2"),
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
        server_handshake(&mut server, &server_auth, "127.0.0.1:1")
            .await
            .map(|_| ())
    });
    let client_err = client_handshake(&mut client, &client_auth, "127.0.0.1:2")
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
        server_handshake(&mut server, &auth, "127.0.0.1:1")
            .await
            .map(|_| ())
    });
    let hello = encode_hello(&[9u8; NONCE_LEN], "10.9.8.7:2").unwrap();
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

#[tokio::test]
async fn handshake_rejects_peer_that_speaks_plain_frames() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (mut server, mut client) = pair().await;
    // A pre-auth transport frame: kind=Control(?) plus a length header.
    let mut garbage = vec![0x02u8];
    garbage.extend_from_slice(&(100u64).to_be_bytes());
    garbage.extend_from_slice(&[0u8; 64]);
    client.write_all(&garbage).await.unwrap();
    let err = server_handshake(&mut server, &auth, "127.0.0.1:1")
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
    let err = server_handshake(&mut server, &auth, "127.0.0.1:1")
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
    let fut = server_handshake(&mut server, &auth, "127.0.0.1:1");
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
