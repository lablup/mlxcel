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

//! Connect-time mutual challenge-response for the TCP transport core (#2637).
//!
//! Runs once per TCP connection, before any transport frame is read or
//! written, on both the accept side ([`server_handshake`]) and the connect
//! side ([`client_handshake`]). Both ends prove possession of the cluster
//! secret with HMAC-SHA256 over the full transcript, so a peer with the wrong
//! secret is rejected whichever side it is on. Wire shape, all big-endian and
//! length-prefixed:
//!
//! ```text
//! client -> server : MAGIC(8) | version u16 | suite u8 | nonce_c [32] | id_len u16 | identity
//! server -> client : MAGIC(8) | version u16 | suite u8 | nonce_s [32] | id_len u16 | identity
//!                    | tag_s = HMAC(K, "mlxcel-cluster-auth-v1/server", transcript)
//! client -> server : tag_c = HMAC(K, "mlxcel-cluster-auth-v1/client", transcript || tag_s)
//! server -> client : verdict u8 (0x01 accept; anything else or EOF is a rejection)
//! ```
//!
//! `identity` is the sender's listener address (`host:port`). The accept side
//! binds the client's identity to the TCP source IP
//! ([`bind_advertised_identity`]): an unspecified IP (`0.0.0.0`) is replaced
//! by the source IP, a concrete IP must equal it. The bound identity is what
//! the transport reports as the `from` of every frame on that connection,
//! which is what the pipeline runtime and the remote stage service authorize
//! against. The connect side does not bind the server's identity: it already
//! knows which address it dialed, the server's authenticator proves the peer
//! holds the secret, and a server reached through NAT or a port forward
//! legitimately advertises an address other than the one the client sees.
//!
//! `suite` names the per-frame protection suite
//! ([`super::frame_protection::FrameSuite`]) the sender will use after the
//! handshake. Both hellos are inside the authenticated transcript, so the
//! suite cannot be downgraded in transit; a server whose suite differs from
//! the client's answers with a rejection record instead of a hello.
//!
//! On success both sides hold the same two 32-byte nonces, and each derives
//! the connection's directional frame keys from them and the secret
//! ([`super::frame_protection::SessionKeys::derive`], HKDF-SHA256). The keys
//! are returned to the caller, which wraps the socket in a
//! [`super::tcp_frame::ProtectedStream`]; every frame after the handshake is
//! authenticated (and, with an AEAD suite, encrypted) under them (#2656).
//!
//! When the client's advertised identity is not its source IP, or its suite
//! is not this server's, the server answers with an explicit rejection record
//! instead of a server hello (#2655 item b):
//!
//! ```text
//! server -> client : REJECT_MAGIC(8) | reason u8 (0x01 identity not owned, 0x02 suite mismatch)
//! ```
//!
//! It still sends no nonce and no authenticator, so a client that lies about
//! its address never learns a valid server proof; the record only tells the
//! client what it can see for itself. The client turns it into a typed
//! [`HandshakeRejected`] so `Transport::connect` fails instead of warning.
//!
//! Failure classes the accept side distinguishes for the operator log:
//! [`PeerClosedBeforeHandshake`] (a liveness probe, debug only),
//! [`IdentityNotOwned`] (wrong bind or advertised address, or a NAT between
//! the nodes), [`ClientAbortedAfterServerProof`] (the client most likely holds
//! a different secret: it verified this server's authenticator, failed, and
//! closed), and [`HandshakeRejected`] (a client authenticator that does not
//! verify although the server's own did, i.e. the transcript was altered).
//!
//! Used by: tcp_transport (accept side), connection_pool (connect side),
//! stage_assignment (both sides). Both entry points take a bare `TcpStream`,
//! the cluster auth, the caller's own listener identity and the frame suite,
//! so any other TCP channel inside the cluster can reuse them as-is.

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use super::cluster_auth::ClusterAuth;
use super::frame_protection::{FrameChannel, FrameSuite, HandshakeRole, SessionKeys};

const MAGIC: &[u8; 8] = b"MLXCAUTH";
/// Wire version of the handshake. 2 added the identity rejection record
/// (#2655); 3 added the frame suite byte and the derived per-frame keys
/// (#2656), so a build without frame protection fails with the explicit
/// version message (mixed versions are unsupported; discovery beacons already
/// pin `CARGO_PKG_VERSION`).
pub const PROTOCOL_VERSION: u16 = 3;
const NONCE_LEN: usize = 32;
const TAG_LEN: usize = 32;
const MAX_IDENTITY_LEN: usize = 255;
/// `MAGIC | version | suite | nonce | id_len`.
const HELLO_HEAD_LEN: usize = 8 + 2 + 1 + NONCE_LEN + 2;
const NONCE_OFFSET: usize = 8 + 2 + 1;
const SERVER_LABEL: &[u8] = b"mlxcel-cluster-auth-v1/server";
const CLIENT_LABEL: &[u8] = b"mlxcel-cluster-auth-v1/client";
const VERDICT_ACCEPT: u8 = 0x01;
const VERDICT_REJECT: u8 = 0x00;

/// First 8 bytes of the server's reply when it refuses the client's advertised
/// identity or frame suite instead of sending a server hello. Followed by one
/// reason byte.
pub const REJECT_MAGIC: &[u8; 8] = b"MLXCREJ\0";
/// Reason byte: the advertised listener IP is not the TCP source IP.
pub const REJECT_IDENTITY_NOT_OWNED: u8 = 0x01;
/// Reason byte: the client's frame protection suite is not the server's.
pub const REJECT_FRAME_SUITE: u8 = 0x02;

/// Upper bound for the whole handshake on either side. A peer that connects
/// and then stalls holds one pending-handshake slot for at most this long.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// The peer completed enough of the protocol to be refused on authentication
/// grounds (a secret mismatch detected by this side, an altered transcript,
/// or an identity or suite refused by the server). Unlike an I/O error or a
/// timeout this is terminal: retrying the connection cannot fix it, so the
/// transport's pre-connect treats it as fatal instead of warning and
/// continuing.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct HandshakeRejected(pub String);

/// The peer connected and closed without sending a single byte. Liveness
/// probes (`probe_peers`, the router health monitor) do exactly this, so the
/// accept side logs it at debug rather than as a rejected handshake.
#[derive(Debug, thiserror::Error)]
#[error("peer closed the connection without starting the cluster handshake")]
pub struct PeerClosedBeforeHandshake;

/// Accept side: the client advertised a listener IP that is not the IP its
/// connection arrived from. The server answers with the rejection record and
/// sends no nonce or authenticator.
#[derive(Debug, thiserror::Error)]
#[error(
    "peer advertised identity {advertised} but connected from {source_ip}; refusing to bind a \
     listener address the peer does not own"
)]
pub struct IdentityNotOwned {
    pub advertised: String,
    pub source_ip: IpAddr,
}

/// Accept side: the client announced a frame protection suite other than
/// this server's. A configuration error on one side, never fixed by a retry.
#[derive(Debug, thiserror::Error)]
#[error(
    "peer uses frame protection suite id {peer_suite_id} but this node is configured for \
     {local} (id {}); every node of a cluster must use the same suite",
    local.wire_id()
)]
pub struct FrameSuiteMismatch {
    pub peer_suite_id: u8,
    pub local: FrameSuite,
}

/// Accept side: the server sent its authenticator and the client closed the
/// connection instead of answering with its own. A client that holds a
/// different secret does exactly this (it verifies the server's proof first,
/// fails, and hangs up), so this is reported as a probable secret mismatch.
#[derive(Debug, thiserror::Error)]
#[error(
    "client closed the connection after receiving this server's authenticator; it most likely \
     holds a different cluster secret"
)]
pub struct ClientAbortedAfterServerProof;

/// Bind an advertised listener identity to the TCP source IP it arrived from.
///
/// Returns the identity the transport should report for the connection. A
/// concrete advertised IP that differs from the source IP is an
/// [`IdentityNotOwned`] error.
pub fn bind_advertised_identity(advertised: &str, source_ip: IpAddr) -> Result<String> {
    let advertised_addr: SocketAddr = advertised.parse().with_context(|| {
        // Debug formatting escapes control characters: the identity is
        // unauthenticated peer input and ends up in the operator log.
        format!("peer advertised identity {advertised:?} is not a socket address")
    })?;
    let source_ip = source_ip.to_canonical();
    let advertised_ip = advertised_addr.ip().to_canonical();
    if advertised_ip.is_unspecified() {
        return Ok(SocketAddr::new(source_ip, advertised_addr.port()).to_string());
    }
    if advertised_ip != source_ip {
        return Err(IdentityNotOwned {
            advertised: advertised.to_string(),
            source_ip,
        }
        .into());
    }
    Ok(SocketAddr::new(advertised_ip, advertised_addr.port()).to_string())
}

fn encode_hello(nonce: &[u8; NONCE_LEN], identity: &str, suite: FrameSuite) -> Result<Vec<u8>> {
    let id = identity.as_bytes();
    if id.len() > MAX_IDENTITY_LEN {
        bail!(
            "local transport identity '{identity}' exceeds {MAX_IDENTITY_LEN} bytes; use a \
             numeric host:port bind address"
        );
    }
    let mut buf = Vec::with_capacity(HELLO_HEAD_LEN + id.len());
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
    buf.push(suite.wire_id());
    buf.extend_from_slice(nonce);
    buf.extend_from_slice(&(id.len() as u16).to_be_bytes());
    buf.extend_from_slice(id);
    Ok(buf)
}

/// Which end sent the hello being read. Only a hello from the server may be
/// replaced by the identity rejection record.
#[derive(Clone, Copy)]
enum HelloFrom<'a> {
    /// The connect side reading the server's hello; `local_identity` is the
    /// identity this client advertised, named in the rejection message.
    Server { local_identity: &'a str },
    /// The accept side reading the client's hello.
    Client,
}

impl HelloFrom<'_> {
    fn role(self) -> &'static str {
        match self {
            Self::Server { .. } => "server",
            Self::Client => "client",
        }
    }
}

/// One parsed hello: its exact bytes (for the transcript) and its fields.
struct Hello {
    raw: Vec<u8>,
    identity: String,
    nonce: [u8; NONCE_LEN],
    suite_id: u8,
}

/// Read one hello (without any trailing tag).
///
/// The first 8 bytes are read before the rest so the connect side can tell a
/// rejection record from a hello; 0 bytes is still
/// [`PeerClosedBeforeHandshake`].
async fn read_hello(stream: &mut TcpStream, from: HelloFrom<'_>) -> Result<Hello> {
    let role = from.role();
    let mut head = [0u8; HELLO_HEAD_LEN];
    let first = stream
        .read(&mut head[..8])
        .await
        .with_context(|| format!("{role} connection failed before the cluster handshake"))?;
    if first == 0 {
        return Err(PeerClosedBeforeHandshake.into());
    }
    stream
        .read_exact(&mut head[first..8])
        .await
        .with_context(|| {
            format!("{role} closed the connection before completing the cluster handshake")
        })?;
    if let HelloFrom::Server { local_identity } = from
        && &head[..8] == REJECT_MAGIC
    {
        return Err(read_rejection_reason(stream, local_identity).await.into());
    }
    if &head[..8] != MAGIC {
        bail!(
            "{role} did not start the cluster handshake (unexpected bytes); the peer is either \
             not an mlxcel cluster transport or runs a build without cluster authentication"
        );
    }
    stream.read_exact(&mut head[8..]).await.with_context(|| {
        format!("{role} closed the connection before completing the cluster handshake")
    })?;
    let version = u16::from_be_bytes([head[8], head[9]]);
    if version != PROTOCOL_VERSION {
        bail!(
            "{role} speaks cluster handshake version {version}, this build expects \
             {PROTOCOL_VERSION}"
        );
    }
    let suite_id = head[10];
    let mut nonce = [0u8; NONCE_LEN];
    nonce.copy_from_slice(&head[NONCE_OFFSET..NONCE_OFFSET + NONCE_LEN]);
    let id_len = u16::from_be_bytes([head[HELLO_HEAD_LEN - 2], head[HELLO_HEAD_LEN - 1]]) as usize;
    if id_len > MAX_IDENTITY_LEN {
        bail!("{role} advertised an identity longer than {MAX_IDENTITY_LEN} bytes");
    }
    let mut id = vec![0u8; id_len];
    stream
        .read_exact(&mut id)
        .await
        .with_context(|| format!("{role} truncated its cluster handshake identity"))?;
    let identity = String::from_utf8(id.clone())
        .with_context(|| format!("{role} advertised a non-UTF-8 identity"))?;
    let mut raw = head.to_vec();
    raw.extend_from_slice(&id);
    Ok(Hello {
        raw,
        identity,
        nonce,
        suite_id,
    })
}

/// Connect side: the server answered with the rejection record. Read the
/// reason byte and describe it. A truncated record is still a rejection.
async fn read_rejection_reason(stream: &mut TcpStream, local_identity: &str) -> HandshakeRejected {
    let mut reason = [0u8; 1];
    let reason = match stream.read_exact(&mut reason).await {
        Ok(_) => Some(reason[0]),
        Err(_) => None,
    };
    match reason {
        Some(REJECT_IDENTITY_NOT_OWNED) | None => HandshakeRejected(format!(
            "server refused this node's advertised identity {local_identity}: the advertised IP \
             must be the address the server sees this connection come from (check the bind or \
             advertised address and any NAT or port forward between the nodes)"
        )),
        Some(REJECT_FRAME_SUITE) => HandshakeRejected(
            "server refused this node's frame protection suite: every node of a cluster must be \
             configured for the same suite"
                .to_string(),
        ),
        Some(code) => HandshakeRejected(format!(
            "server refused this node's cluster handshake with reason code {code:#04x}"
        )),
    }
}

async fn read_tag(stream: &mut TcpStream, what: &str) -> Result<[u8; TAG_LEN]> {
    let mut tag = [0u8; TAG_LEN];
    stream
        .read_exact(&mut tag)
        .await
        .with_context(|| format!("connection closed before the {what} authenticator arrived"))?;
    Ok(tag)
}

async fn write_rejection(stream: &mut TcpStream, reason: u8) {
    let _ = stream
        .write_all(&[REJECT_MAGIC.as_slice(), &[reason]].concat())
        .await;
    let _ = stream.flush().await;
}

/// Connect side. Proves possession of the secret to the server and verifies
/// the server's proof (mutual authentication). Returns the per-connection
/// frame keys for `suite`, which the caller must use for every frame on this
/// connection.
pub async fn client_handshake(
    stream: &mut TcpStream,
    auth: &ClusterAuth,
    local_identity: &str,
    suite: FrameSuite,
) -> Result<SessionKeys> {
    client_handshake_on(stream, auth, local_identity, suite, FrameChannel::Transport).await
}

/// [`client_handshake`] for a named [`FrameChannel`]: the returned frame keys
/// are bound to `channel`, so both ends must name the same one.
pub async fn client_handshake_on(
    stream: &mut TcpStream,
    auth: &ClusterAuth,
    local_identity: &str,
    suite: FrameSuite,
    channel: FrameChannel,
) -> Result<SessionKeys> {
    tokio::time::timeout(
        HANDSHAKE_TIMEOUT,
        client_handshake_inner(stream, auth, local_identity, suite, channel),
    )
    .await
    .map_err(|_| anyhow::anyhow!("cluster handshake timed out after {HANDSHAKE_TIMEOUT:?}"))?
}

async fn client_handshake_inner(
    stream: &mut TcpStream,
    auth: &ClusterAuth,
    local_identity: &str,
    suite: FrameSuite,
    channel: FrameChannel,
) -> Result<SessionKeys> {
    let client_nonce = auth.random_nonce();
    let client_hello = encode_hello(&client_nonce, local_identity, suite)?;
    stream.write_all(&client_hello).await?;
    stream.flush().await?;

    let server_hello = read_hello(stream, HelloFrom::Server { local_identity }).await?;
    let tag_s = read_tag(stream, "server").await?;
    let mut transcript = client_hello;
    transcript.extend_from_slice(&server_hello.raw);
    if !auth.verify(SERVER_LABEL, &transcript, &tag_s) {
        return Err(HandshakeRejected(
            "server failed the cluster handshake: its authenticator does not verify under the \
             configured cluster secret (secret mismatch)"
                .to_string(),
        )
        .into());
    }
    // The server echoes the suite it will use; it refused ours otherwise.
    if server_hello.suite_id != suite.wire_id() {
        return Err(HandshakeRejected(format!(
            "server announced frame protection suite id {} but this node is configured for \
             {suite}; every node of a cluster must use the same suite",
            server_hello.suite_id
        ))
        .into());
    }
    transcript.extend_from_slice(&tag_s);
    let tag_c = auth.sign(CLIENT_LABEL, &transcript);
    stream.write_all(&tag_c).await?;
    stream.flush().await?;

    let mut verdict = [0u8; 1];
    stream
        .read_exact(&mut verdict)
        .await
        .context("server closed the connection instead of accepting the cluster handshake")?;
    if verdict[0] != VERDICT_ACCEPT {
        return Err(HandshakeRejected(
            "server rejected this node's authenticator although the server's own authenticator \
             verified; the handshake was altered in transit"
                .to_string(),
        )
        .into());
    }
    Ok(SessionKeys::derive(
        auth,
        channel,
        suite,
        &client_nonce,
        &server_hello.nonce,
        HandshakeRole::Client,
    ))
}

/// Accept side. Verifies the client's proof and proves possession of the
/// secret back. Returns the client's bound identity, which the transport
/// reports as the `from` of every frame on this connection, and the
/// per-connection frame keys.
///
/// Writes the rejection record itself when the client's advertised identity
/// is not its source IP or its suite is not `suite`, so every caller (not only
/// `TcpTransport`) gets that behavior.
pub async fn server_handshake(
    stream: &mut TcpStream,
    auth: &ClusterAuth,
    local_identity: &str,
    suite: FrameSuite,
) -> Result<(String, SessionKeys)> {
    server_handshake_on(stream, auth, local_identity, suite, FrameChannel::Transport).await
}

/// [`server_handshake`] for a named [`FrameChannel`]: the returned frame keys
/// are bound to `channel`, so both ends must name the same one.
pub async fn server_handshake_on(
    stream: &mut TcpStream,
    auth: &ClusterAuth,
    local_identity: &str,
    suite: FrameSuite,
    channel: FrameChannel,
) -> Result<(String, SessionKeys)> {
    tokio::time::timeout(
        HANDSHAKE_TIMEOUT,
        server_handshake_inner(stream, auth, local_identity, suite, channel),
    )
    .await
    .map_err(|_| anyhow::anyhow!("cluster handshake timed out after {HANDSHAKE_TIMEOUT:?}"))?
}

async fn server_handshake_inner(
    stream: &mut TcpStream,
    auth: &ClusterAuth,
    local_identity: &str,
    suite: FrameSuite,
    channel: FrameChannel,
) -> Result<(String, SessionKeys)> {
    let client_hello = read_hello(stream, HelloFrom::Client).await?;
    let source_ip = stream.peer_addr().context("peer address")?.ip();
    // Bind before replying so a client that lies about its address never
    // learns a valid server authenticator: it gets the rejection record and
    // nothing else (no nonce, no proof). Only a well-formed identity naming
    // another IP gets the record; an identity that is not a socket address
    // at all comes from no mlxcel client (each advertises its listener
    // `SocketAddr`), so it is a protocol error and the connection is closed
    // without a reply, like any other malformed hello.
    let bound = match bind_advertised_identity(&client_hello.identity, source_ip) {
        Ok(bound) => bound,
        Err(err) => {
            if err.downcast_ref::<IdentityNotOwned>().is_some() {
                write_rejection(stream, REJECT_IDENTITY_NOT_OWNED).await;
            }
            return Err(err);
        }
    };
    if client_hello.suite_id != suite.wire_id() {
        write_rejection(stream, REJECT_FRAME_SUITE).await;
        return Err(FrameSuiteMismatch {
            peer_suite_id: client_hello.suite_id,
            local: suite,
        }
        .into());
    }

    let server_nonce = auth.random_nonce();
    let server_hello = encode_hello(&server_nonce, local_identity, suite)?;
    let mut transcript = client_hello.raw;
    transcript.extend_from_slice(&server_hello);
    let tag_s = auth.sign(SERVER_LABEL, &transcript);
    let mut reply = server_hello;
    reply.extend_from_slice(&tag_s);
    stream.write_all(&reply).await?;
    stream.flush().await?;

    let tag_c = match read_tag(stream, "client").await {
        Ok(tag) => tag,
        Err(err) if is_peer_abort(&err) => return Err(ClientAbortedAfterServerProof.into()),
        Err(err) => return Err(err),
    };
    transcript.extend_from_slice(&tag_s);
    if !auth.verify(CLIENT_LABEL, &transcript, &tag_c) {
        let _ = stream.write_all(&[VERDICT_REJECT]).await;
        let _ = stream.flush().await;
        return Err(HandshakeRejected(
            "client failed the cluster handshake: its authenticator does not verify under the \
             configured cluster secret (secret mismatch)"
                .to_string(),
        )
        .into());
    }
    stream.write_all(&[VERDICT_ACCEPT]).await?;
    stream.flush().await?;
    let keys = SessionKeys::derive(
        auth,
        channel,
        suite,
        &client_hello.nonce,
        &server_nonce,
        HandshakeRole::Server,
    );
    Ok((bound, keys))
}

/// Whether an I/O error (possibly wrapped in context) is the peer ending the
/// stream mid-read: an orderly close (`UnexpectedEof`) or an abortive one (a
/// reset, for example a client that closes with unread data or with
/// `SO_LINGER` 0). Both mean the client gave up on the handshake.
fn is_peer_abort(err: &anyhow::Error) -> bool {
    use std::io::ErrorKind;
    err.chain().any(|cause| {
        cause.downcast_ref::<std::io::Error>().is_some_and(|io| {
            matches!(
                io.kind(),
                ErrorKind::UnexpectedEof
                    | ErrorKind::ConnectionReset
                    | ErrorKind::ConnectionAborted
                    | ErrorKind::BrokenPipe
            )
        })
    })
}

#[cfg(test)]
#[path = "cluster_handshake_tests.rs"]
mod tests;
