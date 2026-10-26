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
//! client -> server : MAGIC(8) | version u16 | nonce_c [32] | id_len u16 | identity
//! server -> client : MAGIC(8) | version u16 | nonce_s [32] | id_len u16 | identity
//!                    | tag_s = HMAC(K, "mlxcel-cluster-auth-v1/server", transcript)
//! client -> server : tag_c = HMAC(K, "mlxcel-cluster-auth-v1/client", transcript || tag_s)
//! server -> client : verdict u8 (0x01 accept; anything else or EOF is a rejection)
//! ```
//!
//! `identity` is the sender's listener address (`host:port`). The receiver
//! binds it to the TCP source IP ([`bind_advertised_identity`]): an
//! unspecified IP (`0.0.0.0`) is replaced by the source IP, a concrete IP
//! must equal it. The bound identity is what the transport reports as the
//! `from` of every frame on that connection, which is what the pipeline
//! runtime and the remote stage service authorize against.
//!
//! Residual risk: this authenticates the connection, not each frame. An
//! on-path attacker who can hijack an established TCP stream is out of
//! scope; TLS or a per-frame MAC is the follow-up.
//!
//! Used by: tcp_transport (accept side), connection_pool (connect side).

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use super::cluster_auth::ClusterAuth;

const MAGIC: &[u8; 8] = b"MLXCAUTH";
const PROTOCOL_VERSION: u16 = 1;
const NONCE_LEN: usize = 32;
const TAG_LEN: usize = 32;
const MAX_IDENTITY_LEN: usize = 255;
const SERVER_LABEL: &[u8] = b"mlxcel-cluster-auth-v1/server";
const CLIENT_LABEL: &[u8] = b"mlxcel-cluster-auth-v1/client";
const VERDICT_ACCEPT: u8 = 0x01;
const VERDICT_REJECT: u8 = 0x00;

/// Upper bound for the whole handshake on either side. A peer that connects
/// and then stalls holds one connection slot for at most this long.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// The peer completed the protocol but failed authentication (secret
/// mismatch or an identity it does not own). Unlike an I/O error or a
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

/// Bind an advertised listener identity to the TCP source IP it arrived from.
///
/// Returns the identity the transport should report for the connection.
pub fn bind_advertised_identity(advertised: &str, source_ip: IpAddr) -> Result<String> {
    let advertised_addr: SocketAddr = advertised.parse().with_context(|| {
        format!("peer advertised identity '{advertised}' is not a socket address")
    })?;
    let source_ip = source_ip.to_canonical();
    let advertised_ip = advertised_addr.ip().to_canonical();
    if advertised_ip.is_unspecified() {
        return Ok(SocketAddr::new(source_ip, advertised_addr.port()).to_string());
    }
    if advertised_ip != source_ip {
        bail!(
            "peer advertised identity {advertised} but connected from {source_ip}; refusing to \
             bind a listener address the peer does not own"
        );
    }
    Ok(SocketAddr::new(advertised_ip, advertised_addr.port()).to_string())
}

fn encode_hello(nonce: &[u8; NONCE_LEN], identity: &str) -> Result<Vec<u8>> {
    let id = identity.as_bytes();
    if id.len() > MAX_IDENTITY_LEN {
        bail!(
            "local transport identity '{identity}' exceeds {MAX_IDENTITY_LEN} bytes; use a \
             numeric host:port bind address"
        );
    }
    let mut buf = Vec::with_capacity(MAGIC.len() + 2 + NONCE_LEN + 2 + id.len());
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
    buf.extend_from_slice(nonce);
    buf.extend_from_slice(&(id.len() as u16).to_be_bytes());
    buf.extend_from_slice(id);
    Ok(buf)
}

/// Read one hello (without any trailing tag) and return its exact bytes plus
/// the advertised identity.
async fn read_hello(stream: &mut TcpStream, role: &str) -> Result<(Vec<u8>, String)> {
    let mut head = [0u8; 8 + 2 + NONCE_LEN + 2];
    let first = stream
        .read(&mut head)
        .await
        .with_context(|| format!("{role} connection failed before the cluster handshake"))?;
    if first == 0 {
        return Err(PeerClosedBeforeHandshake.into());
    }
    stream
        .read_exact(&mut head[first..])
        .await
        .with_context(|| {
            format!("{role} closed the connection before completing the cluster handshake")
        })?;
    if &head[..8] != MAGIC {
        bail!(
            "{role} did not start the cluster handshake (unexpected bytes); the peer is either \
             not an mlxcel cluster transport or runs a build without cluster authentication"
        );
    }
    let version = u16::from_be_bytes([head[8], head[9]]);
    if version != PROTOCOL_VERSION {
        bail!(
            "{role} speaks cluster handshake version {version}, this build expects \
             {PROTOCOL_VERSION}"
        );
    }
    let id_len =
        u16::from_be_bytes([head[8 + 2 + NONCE_LEN], head[8 + 2 + NONCE_LEN + 1]]) as usize;
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
    Ok((raw, identity))
}

async fn read_tag(stream: &mut TcpStream, what: &str) -> Result<[u8; TAG_LEN]> {
    let mut tag = [0u8; TAG_LEN];
    stream
        .read_exact(&mut tag)
        .await
        .with_context(|| format!("connection closed before the {what} authenticator arrived"))?;
    Ok(tag)
}

/// Connect side. Proves possession of the secret to the server and verifies
/// the server's proof. Returns the server's bound identity.
pub async fn client_handshake(
    stream: &mut TcpStream,
    auth: &ClusterAuth,
    local_identity: &str,
) -> Result<String> {
    tokio::time::timeout(
        HANDSHAKE_TIMEOUT,
        client_handshake_inner(stream, auth, local_identity),
    )
    .await
    .map_err(|_| anyhow::anyhow!("cluster handshake timed out after {HANDSHAKE_TIMEOUT:?}"))?
}

async fn client_handshake_inner(
    stream: &mut TcpStream,
    auth: &ClusterAuth,
    local_identity: &str,
) -> Result<String> {
    let client_hello = encode_hello(&auth.random_nonce(), local_identity)?;
    stream.write_all(&client_hello).await?;
    stream.flush().await?;

    let (server_hello, server_identity) = read_hello(stream, "server").await?;
    let tag_s = read_tag(stream, "server").await?;
    let mut transcript = client_hello;
    transcript.extend_from_slice(&server_hello);
    if !auth.verify(SERVER_LABEL, &transcript, &tag_s) {
        return Err(HandshakeRejected(
            "server failed the cluster handshake: its authenticator does not verify under the \
             configured cluster secret (secret mismatch)"
                .to_string(),
        )
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
            "server rejected the cluster handshake (secret mismatch or identity binding failed)"
                .to_string(),
        )
        .into());
    }
    let source_ip = stream.peer_addr().context("peer address")?.ip();
    bind_advertised_identity(&server_identity, source_ip)
}

/// Accept side. Verifies the client's proof and proves possession of the
/// secret back. Returns the client's bound identity, which the transport
/// reports as the `from` of every frame on this connection.
pub async fn server_handshake(
    stream: &mut TcpStream,
    auth: &ClusterAuth,
    local_identity: &str,
) -> Result<String> {
    tokio::time::timeout(
        HANDSHAKE_TIMEOUT,
        server_handshake_inner(stream, auth, local_identity),
    )
    .await
    .map_err(|_| anyhow::anyhow!("cluster handshake timed out after {HANDSHAKE_TIMEOUT:?}"))?
}

async fn server_handshake_inner(
    stream: &mut TcpStream,
    auth: &ClusterAuth,
    local_identity: &str,
) -> Result<String> {
    let (client_hello, client_identity) = read_hello(stream, "client").await?;
    let source_ip = stream.peer_addr().context("peer address")?.ip();
    // Bind before replying so a client that lies about its address never
    // learns a valid server authenticator.
    let bound = bind_advertised_identity(&client_identity, source_ip);

    let server_hello = encode_hello(&auth.random_nonce(), local_identity)?;
    let mut transcript = client_hello;
    transcript.extend_from_slice(&server_hello);
    let tag_s = auth.sign(SERVER_LABEL, &transcript);
    // An identity the client does not own is a protocol violation, not a
    // credential to be checked: do not answer at all.
    let bound = bound?;
    let mut reply = server_hello.clone();
    reply.extend_from_slice(&tag_s);
    stream.write_all(&reply).await?;
    stream.flush().await?;

    let tag_c = read_tag(stream, "client").await?;
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
    Ok(bound)
}

#[cfg(test)]
#[path = "cluster_handshake_tests.rs"]
mod tests;
