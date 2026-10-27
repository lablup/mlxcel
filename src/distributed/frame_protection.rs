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

//! Per-frame protection for the cluster TCP core (#2656).
//!
//! The connect-time handshake ([`super::cluster_handshake`]) proves that both
//! ends hold the cluster secret; this module protects every frame exchanged
//! afterwards on that connection. Key material comes from the handshake
//! through HKDF-SHA256: IKM is the cluster secret, the salt is the client
//! nonce followed by the server nonce, and the info strings
//! [`FRAME_INFO_C2S`] and [`FRAME_INFO_S2C`] give one key per direction. Keys
//! are therefore per connection: a frame recorded on one connection cannot
//! verify on another, and a relay that forwards a real handshake cannot
//! derive them because it does not hold the secret.
//!
//! Each direction carries an implicit 64-bit sequence number that starts at
//! 0, is never sent on the wire, and never wraps (a sender that reaches the
//! end refuses to send, which closes the connection). The receiver verifies
//! frame `n` under sequence `n`, so a replayed, reordered, dropped or injected
//! frame fails authentication and the connection is closed.
//!
//! Three suites share the codec. [`FrameSuite::HmacSha256`] appends
//! `HMAC-SHA256(K_dir, seq || kind || len || payload)` and leaves the payload
//! in clear text. The AEAD suites ([`FrameSuite::Aes256Gcm`],
//! [`FrameSuite::ChaCha20Poly1305`]) encrypt the payload in place with the
//! 96-bit nonce `0x00000000 || seq` and the `kind || len` header as
//! associated data, and append the 16-byte tag. The header stays in clear so
//! the receiver can refuse an oversize frame before any cryptographic work.
//! The suite in production use is [`FrameSuite::default`]; the others exist
//! so the activation benchmark can measure every candidate on the real
//! transport, and the handshake refuses a peer configured with another suite.
//!
//! Used by: cluster_handshake (key derivation), tcp_frame (the one protected
//! read and write path), connection_pool and tcp_transport (per-connection
//! state), stage_assignment (same codec on the assignment channel).

use std::fmt;

use ring::{aead, hkdf, hmac};
use subtle::ConstantTimeEq;

use super::cluster_auth::ClusterAuth;

/// HKDF info for the client-to-server frame key.
pub const FRAME_INFO_C2S: &[u8] = b"mlxcel-cluster-frame-v1/c2s";
/// HKDF info for the server-to-client frame key.
pub const FRAME_INFO_S2C: &[u8] = b"mlxcel-cluster-frame-v1/s2c";

/// Length of the clear-text frame header the tag or AEAD covers:
/// `[kind: u8][len: u64 BE]`.
pub const FRAME_HEADER_LEN: usize = 9;

const KEY_LEN: usize = 32;
const AEAD_NONCE_LEN: usize = 12;

/// Which per-frame construction a connection uses.
///
/// Every node of one cluster must use the same suite; the handshake carries
/// the suite id (covered by both authenticators) and refuses a mismatch with
/// an explicit error instead of failing on the first frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum FrameSuite {
    /// 32-byte HMAC-SHA256 tag per frame: integrity, replay and reorder
    /// rejection, connection binding. Payloads travel in clear text.
    HmacSha256,
    /// AES-256-GCM: the guarantees of [`Self::HmacSha256`] plus
    /// confidentiality, 16-byte tag. The production suite (#2656 decision).
    #[default]
    Aes256Gcm,
    /// ChaCha20-Poly1305: same guarantees and tag length as
    /// [`Self::Aes256Gcm`], for hosts without AES instructions.
    ChaCha20Poly1305,
}

impl FrameSuite {
    /// Every suite, in wire-id order (benchmark enumeration).
    pub const ALL: [FrameSuite; 3] = [
        FrameSuite::HmacSha256,
        FrameSuite::Aes256Gcm,
        FrameSuite::ChaCha20Poly1305,
    ];

    /// One-byte identifier carried in the handshake hello.
    pub const fn wire_id(self) -> u8 {
        match self {
            Self::HmacSha256 => 1,
            Self::Aes256Gcm => 2,
            Self::ChaCha20Poly1305 => 3,
        }
    }

    /// Inverse of [`Self::wire_id`].
    pub const fn from_wire_id(id: u8) -> Option<Self> {
        match id {
            1 => Some(Self::HmacSha256),
            2 => Some(Self::Aes256Gcm),
            3 => Some(Self::ChaCha20Poly1305),
            _ => None,
        }
    }

    /// Bytes appended after the payload of every frame.
    pub const fn tag_len(self) -> usize {
        match self {
            Self::HmacSha256 => 32,
            Self::Aes256Gcm | Self::ChaCha20Poly1305 => 16,
        }
    }

    /// Whether payloads are encrypted (true for the AEAD suites).
    pub const fn encrypts(self) -> bool {
        !matches!(self, Self::HmacSha256)
    }

    /// Stable lower-case name for logs and benchmark tables.
    pub const fn label(self) -> &'static str {
        match self {
            Self::HmacSha256 => "hmac-sha256",
            Self::Aes256Gcm => "aes-256-gcm",
            Self::ChaCha20Poly1305 => "chacha20-poly1305",
        }
    }
}

impl fmt::Display for FrameSuite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Which end of the handshake the holder of a [`SessionKeys`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeRole {
    Client,
    Server,
}

/// A receiver refused a frame: the tag did not verify, or the ciphertext did
/// not decrypt. The transport closes the connection and logs this without
/// any payload bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("frame authentication failed (altered, replayed, reordered or foreign frame)")]
pub struct FrameRejected;

/// A direction used its last sequence number. The connection is closed; a
/// fresh connection starts at 0 again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("per-frame sequence number exhausted; reconnect to continue")]
pub struct SequenceExhausted;

/// Best-effort scrub of key bytes that are about to leave scope.
fn zeroize(bytes: &mut [u8]) {
    for b in bytes.iter_mut() {
        // SAFETY: `b` is a valid, exclusively borrowed `u8`; a volatile zero
        // write through it is sound and only keeps the compiler from eliding
        // the scrub.
        unsafe { std::ptr::write_volatile(b, 0) };
    }
}

enum DirectionKey {
    Hmac(hmac::Key),
    Aead(aead::LessSafeKey),
}

impl DirectionKey {
    fn new(suite: FrameSuite, raw: &mut [u8; KEY_LEN]) -> Self {
        let key = match suite {
            FrameSuite::HmacSha256 => Self::Hmac(hmac::Key::new(hmac::HMAC_SHA256, raw)),
            FrameSuite::Aes256Gcm => Self::Aead(aead::LessSafeKey::new(
                aead::UnboundKey::new(&aead::AES_256_GCM, raw).expect("32-byte AES-256-GCM key"),
            )),
            FrameSuite::ChaCha20Poly1305 => Self::Aead(aead::LessSafeKey::new(
                aead::UnboundKey::new(&aead::CHACHA20_POLY1305, raw)
                    .expect("32-byte ChaCha20-Poly1305 key"),
            )),
        };
        zeroize(raw);
        key
    }
}

/// Directional frame keys of one connection, oriented from the holder's
/// point of view (`send` protects what it writes, `recv` verifies what it
/// reads). Produced by the handshake, consumed by
/// [`super::tcp_frame::ProtectedStream`].
pub struct SessionKeys {
    suite: FrameSuite,
    send: DirectionKey,
    recv: DirectionKey,
}

impl SessionKeys {
    /// Derive both directional keys from the handshake inputs.
    ///
    /// `client_nonce` and `server_nonce` are the 32-byte hello nonces, which
    /// both authenticators cover; the caller runs this only after its peer's
    /// authenticator verified.
    pub fn derive(
        auth: &ClusterAuth,
        suite: FrameSuite,
        client_nonce: &[u8; 32],
        server_nonce: &[u8; 32],
        role: HandshakeRole,
    ) -> Self {
        let mut salt = [0u8; 64];
        salt[..32].copy_from_slice(client_nonce);
        salt[32..].copy_from_slice(server_nonce);
        let prk = auth.hkdf_extract(&salt);
        let mut c2s = expand(&prk, FRAME_INFO_C2S);
        let mut s2c = expand(&prk, FRAME_INFO_S2C);
        let (send, recv) = match role {
            HandshakeRole::Client => (&mut c2s, &mut s2c),
            HandshakeRole::Server => (&mut s2c, &mut c2s),
        };
        Self {
            suite,
            send: DirectionKey::new(suite, send),
            recv: DirectionKey::new(suite, recv),
        }
    }

    pub fn suite(&self) -> FrameSuite {
        self.suite
    }

    /// Split into the sender and receiver halves, each with its own sequence.
    pub fn into_codec(self) -> (FrameSealer, FrameOpener) {
        (
            FrameSealer {
                suite: self.suite,
                key: self.send,
                seq: 0,
            },
            FrameOpener {
                suite: self.suite,
                key: self.recv,
                seq: 0,
            },
        )
    }
}

impl fmt::Debug for SessionKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SessionKeys({}, <redacted>)", self.suite)
    }
}

fn expand(prk: &hkdf::Prk, info: &[u8]) -> [u8; KEY_LEN] {
    let mut out = [0u8; KEY_LEN];
    prk.expand(&[info], hkdf::HKDF_SHA256)
        .and_then(|okm| okm.fill(&mut out))
        .expect("HKDF-SHA256 expand to 32 bytes");
    out
}

/// Sequence `seq` as the 96-bit AEAD nonce (directional keys make the
/// counter unique per key).
fn aead_nonce(seq: u64) -> aead::Nonce {
    let mut nonce = [0u8; AEAD_NONCE_LEN];
    nonce[4..].copy_from_slice(&seq.to_be_bytes());
    aead::Nonce::assume_unique_for_key(nonce)
}

/// Sending half: protects frames in place and advances the send sequence.
pub struct FrameSealer {
    suite: FrameSuite,
    key: DirectionKey,
    seq: u64,
}

impl FrameSealer {
    pub fn suite(&self) -> FrameSuite {
        self.suite
    }

    /// Protect one frame. `frame` holds `[kind][len][payload]` on entry; on
    /// return the payload is encrypted in place (AEAD suites) and the tag is
    /// appended. Fails only when the sequence is exhausted, in which case the
    /// frame is left untouched and must not be sent.
    pub fn seal(&mut self, frame: &mut Vec<u8>) -> Result<(), SequenceExhausted> {
        debug_assert!(frame.len() >= FRAME_HEADER_LEN);
        if self.seq == u64::MAX {
            return Err(SequenceExhausted);
        }
        let seq = self.seq;
        match &self.key {
            DirectionKey::Hmac(key) => {
                let mut ctx = hmac::Context::with_key(key);
                ctx.update(&seq.to_be_bytes());
                ctx.update(frame);
                let tag = ctx.sign();
                frame.extend_from_slice(tag.as_ref());
            }
            DirectionKey::Aead(key) => {
                let (header, payload) = frame.split_at_mut(FRAME_HEADER_LEN);
                let tag = key
                    .seal_in_place_separate_tag(aead_nonce(seq), aead::Aad::from(&*header), payload)
                    .expect("AEAD seal cannot fail for an in-range payload");
                frame.extend_from_slice(tag.as_ref());
            }
        }
        self.seq += 1;
        Ok(())
    }
}

/// Receiving half: verifies frames in place and advances the receive
/// sequence.
pub struct FrameOpener {
    suite: FrameSuite,
    key: DirectionKey,
    seq: u64,
}

impl FrameOpener {
    pub fn suite(&self) -> FrameSuite {
        self.suite
    }

    /// Bytes that follow the payload of every frame in this suite.
    pub fn tag_len(&self) -> usize {
        self.suite.tag_len()
    }

    /// Verify one frame. `header` is the clear-text `[kind][len]` header and
    /// `body` holds `payload || tag`. On success the payload (decrypted in
    /// place for the AEAD suites) is the first `body.len() - tag_len` bytes
    /// and the receive sequence advances. On failure nothing about the
    /// payload is revealed and the sequence does not advance; the caller
    /// closes the connection.
    pub fn open(
        &mut self,
        header: &[u8; FRAME_HEADER_LEN],
        body: &mut [u8],
    ) -> Result<usize, FrameRejected> {
        let tag_len = self.tag_len();
        if body.len() < tag_len || self.seq == u64::MAX {
            return Err(FrameRejected);
        }
        let payload_len = body.len() - tag_len;
        let seq = self.seq;
        match &self.key {
            DirectionKey::Hmac(key) => {
                let (payload, tag) = body.split_at(payload_len);
                let mut ctx = hmac::Context::with_key(key);
                ctx.update(&seq.to_be_bytes());
                ctx.update(header);
                ctx.update(payload);
                let expected = ctx.sign();
                if !bool::from(expected.as_ref().ct_eq(tag)) {
                    return Err(FrameRejected);
                }
            }
            DirectionKey::Aead(key) => {
                key.open_in_place(aead_nonce(seq), aead::Aad::from(&header[..]), body)
                    .map_err(|_| FrameRejected)?;
            }
        }
        self.seq += 1;
        Ok(payload_len)
    }
}

#[cfg(test)]
#[path = "frame_protection_tests.rs"]
mod tests;
