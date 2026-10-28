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

//! Wire framing of the TCP transport core and its one protected read and
//! write path (#2656).
//!
//! ```text
//! plain      : [kind: u8][len: u64 BE][payload: len bytes]
//! protected  : [kind: u8][len: u64 BE][payload or ciphertext: len bytes][tag]
//! ```
//!
//! [`ProtectedStream`] owns a connection's socket together with its
//! per-connection codec state (directional keys and sequence numbers from
//! [`super::frame_protection`]). Every frame the transport sends, whether a
//! message, a stream chunk, an RPC request or an RPC response, is sealed by
//! [`finish_frame`] and written by [`ProtectedStream::write_prepared_frame`]
//! ([`ProtectedStream::write_frame`] is the borrowed-payload front of the
//! same path), and every inbound frame goes through
//! [`ProtectedStream::read_frame`]; the stage assignment channel uses the
//! same methods. There is no second framing or sealing path. A stream built
//! with [`ProtectedStream::plain`] (no cluster auth, single-node defaults)
//! uses the plain layout.
//!
//! A message is encoded straight into its frame buffer
//! ([`encode_message_frame`]): the header bytes are reserved in front of the
//! payload and [`super::frame_protection::MAX_TAG_LEN`] bytes behind it, so
//! sealing writes the header and appends the tag in place. The per-token
//! logits reply of a pipeline is about 405 KB; the earlier payload-then-frame
//! copy of that buffer on every send is gone (#2694).
//!
//! The receiver reads the header first and refuses a `len` above the
//! configured maximum before reading the body or doing any cryptographic
//! work, so an oversize announcement costs nothing but the header read.
//!
//! Used by: tcp_transport, connection_pool, stage_assignment.

use std::io;
use std::net::SocketAddr;

use anyhow::{Context, Result};
use bytes::{Bytes, BytesMut};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use super::frame_protection::{
    FRAME_HEADER_LEN, FrameOpener, FrameSealer, MAX_TAG_LEN, SessionKeys,
};
use super::transport::{MessageKind, TransportMessage};

/// An authenticated frame whose kind byte this build does not know. The
/// peer sealed it, so this is not tampering but a version-skewed member
/// (#2708); the transport closes the connection and logs it through the
/// rate-limited diagnostic budget (#2712) so the skewed member is visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown message kind {kind} from an authenticated peer")]
pub struct UnknownAuthenticatedKind {
    /// The kind byte as received.
    pub kind: u8,
}

/// What a non-consuming probe of an idle connection found (#2712).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdleProbe {
    /// No input waiting: the peer is there and the connection is reusable.
    Idle,
    /// Unread bytes wait on the connection; a FIN may follow them. An idle
    /// outbound connection has no legitimate pending input, so the pool
    /// drops it rather than hand it out with a desynchronized read side.
    PendingData,
    /// The peer closed or reset the connection.
    Closed,
}

/// A TCP connection plus the codec state that protects its frames.
pub struct ProtectedStream {
    stream: TcpStream,
    codec: Option<(FrameSealer, FrameOpener)>,
}

impl ProtectedStream {
    /// A connection without per-frame protection (no cluster auth).
    pub fn plain(stream: TcpStream) -> Self {
        Self {
            stream,
            codec: None,
        }
    }

    /// A connection whose frames are protected with the keys the handshake
    /// derived for it.
    pub fn protected(stream: TcpStream, keys: SessionKeys) -> Self {
        Self {
            stream,
            codec: Some(keys.into_codec()),
        }
    }

    /// Whether frames on this connection carry a tag (and, for an AEAD
    /// suite, an encrypted payload).
    pub fn is_protected(&self) -> bool {
        self.codec.is_some()
    }

    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.stream.peer_addr()
    }

    /// Probe an idle connection without consuming anything, for a pool
    /// deciding whether it is still worth reusing (#2695, #2712).
    ///
    /// `peer_addr` cannot answer this: it keeps succeeding for a socket in
    /// CLOSE_WAIT whose peer process has exited, and the first write to such
    /// a socket also succeeds, so a stale pooled connection would swallow a
    /// frame without any error. This issues one non-blocking `MSG_PEEK` of at
    /// most one byte directly on the socket: it never waits, never consumes a
    /// byte, and never touches the codec state, so the frame stream and its
    /// sequence numbers are exactly as they were. Calling the socket directly
    /// rather than through the reactor's readiness makes the answer
    /// independent of whether the runtime has already observed the FIN.
    ///
    /// No data waiting (would block) is [`IdleProbe::Idle`]; unread bytes are
    /// [`IdleProbe::PendingData`] even when a FIN follows them; a zero-length
    /// read (orderly shutdown) or a socket error such as a reset is
    /// [`IdleProbe::Closed`]. A peer that dies after this check is not
    /// detected here; the write or the read that follows reports it.
    pub fn probe_idle(&self) -> IdleProbe {
        let mut probe = [std::mem::MaybeUninit::<u8>::uninit(); 1];
        match socket2::SockRef::from(&self.stream).peek(&mut probe) {
            Ok(0) => IdleProbe::Closed,
            Ok(_) => IdleProbe::PendingData,
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                IdleProbe::Idle
            }
            Err(_) => IdleProbe::Closed,
        }
    }

    /// Whether the peer has closed or reset this connection
    /// ([`Self::probe_idle`] found [`IdleProbe::Closed`]).
    pub fn peer_closed(&self) -> bool {
        self.probe_idle() == IdleProbe::Closed
    }

    /// Write one frame from a borrowed payload: the payload is copied into a
    /// prepared frame buffer ([`prepared_frame`]) and sent through
    /// [`Self::write_prepared_frame`]. Callers that build the payload
    /// themselves ([`encode_message_frame`]) skip the copy.
    pub async fn write_frame(&mut self, kind: MessageKind, payload: &[u8]) -> Result<()> {
        let mut frame = prepared_frame(payload.len());
        frame.extend_from_slice(payload);
        self.write_prepared_frame(kind, frame).await
    }

    /// Write one prepared frame: `frame` holds [`FRAME_HEADER_LEN`] reserved
    /// bytes followed by the payload (see [`prepared_frame`]). The header is
    /// filled in, the frame is sealed in place when the connection is
    /// protected ([`finish_frame`]), and header, payload and tag go out as a
    /// single buffer so small frames cost one write syscall.
    pub(crate) async fn write_prepared_frame(
        &mut self,
        kind: MessageKind,
        mut frame: Vec<u8>,
    ) -> Result<()> {
        finish_frame(
            self.codec.as_mut().map(|(sealer, _)| sealer),
            kind,
            &mut frame,
        )?;
        self.stream.write_all(&frame).await?;
        self.stream.flush().await?;
        Ok(())
    }

    /// Read one frame. Returns `(kind, payload)`; the payload is verified
    /// (and decrypted) when the connection is protected. Errors on a closed
    /// stream, a malformed or oversize header, or a frame that fails
    /// authentication ([`super::frame_protection::FrameRejected`], which the
    /// caller can downcast); after any error the connection must be closed.
    ///
    /// On a protected stream the kind byte is interpreted only after the
    /// frame authenticated (#2694): the header is associated data, so a kind
    /// byte altered in flight, to a valid or an invalid value, fails `open`
    /// and is reported as `FrameRejected`, which the callers log through the
    /// rate-limited diagnostic budget. An invalid kind that does authenticate
    /// (a version-skewed member) is the distinct, typed
    /// [`UnknownAuthenticatedKind`] error.
    /// The oversize check still runs on the header alone and is deliberately
    /// not an authentication failure. A plain stream parses the kind first.
    pub async fn read_frame(&mut self, max_frame_size: usize) -> Result<(MessageKind, BytesMut)> {
        let mut header = [0u8; FRAME_HEADER_LEN];
        self.stream
            .read_exact(&mut header)
            .await
            .context("read frame header")?;
        let raw_kind = header[0];
        let plain_kind = if self.codec.is_none() {
            Some(MessageKind::try_from(raw_kind)?)
        } else {
            None
        };
        let len = u64::from_be_bytes(header[1..].try_into().expect("8-byte length")) as usize;
        if len > max_frame_size {
            anyhow::bail!("frame size {len} exceeds maximum {max_frame_size}");
        }
        let tag_len = self
            .codec
            .as_ref()
            .map_or(0, |(_, opener)| opener.tag_len());
        let mut body = BytesMut::zeroed(len + tag_len);
        self.stream
            .read_exact(&mut body)
            .await
            .context("read frame payload")?;
        let kind = match (self.codec.as_mut(), plain_kind) {
            (Some((_, opener)), _) => {
                let payload_len = opener.open(&header, &mut body)?;
                body.truncate(payload_len);
                MessageKind::try_from(raw_kind)
                    .map_err(|_| UnknownAuthenticatedKind { kind: raw_kind })?
            }
            (None, Some(kind)) => kind,
            (None, None) => unreachable!("a plain stream parsed its kind before the body"),
        };
        Ok((kind, body))
    }
}

/// A frame buffer for a payload of `payload_len` bytes: [`FRAME_HEADER_LEN`]
/// zero bytes that [`finish_frame`] overwrites with the header, room for the
/// payload, and [`MAX_TAG_LEN`] spare bytes so sealing never reallocates.
/// The caller appends the payload.
fn prepared_frame(payload_len: usize) -> Vec<u8> {
    let mut frame = Vec::with_capacity(FRAME_HEADER_LEN + payload_len + MAX_TAG_LEN);
    frame.resize(FRAME_HEADER_LEN, 0);
    frame
}

/// Finish a prepared frame in place: write `kind` and the big-endian payload
/// length into the reserved header and, when `sealer` is given, seal the
/// frame (encrypt the payload for an AEAD suite and append the tag). This is
/// the one sealing path of the codec. The buffer does not reallocate when it
/// came from [`prepared_frame`] or [`encode_message_frame`]. Fails only when
/// the buffer is shorter than the reserved header or the sealer's sequence is
/// exhausted; the frame must then not be sent.
pub(crate) fn finish_frame(
    sealer: Option<&mut FrameSealer>,
    kind: MessageKind,
    frame: &mut Vec<u8>,
) -> Result<()> {
    if frame.len() < FRAME_HEADER_LEN {
        anyhow::bail!(
            "prepared frame is {} bytes, shorter than the {FRAME_HEADER_LEN}-byte reserved header",
            frame.len()
        );
    }
    let payload_len = frame.len() - FRAME_HEADER_LEN;
    frame[0] = kind as u8;
    frame[1..FRAME_HEADER_LEN].copy_from_slice(&(payload_len as u64).to_be_bytes());
    if let Some(sealer) = sealer {
        sealer.seal(frame)?;
    }
    Ok(())
}

/// Encode a [`TransportMessage`] straight into a prepared frame: the frame
/// kind plus a buffer holding the reserved header and the payload, with
/// [`MAX_TAG_LEN`] spare bytes, for [`ProtectedStream::write_prepared_frame`].
pub(crate) fn encode_message_frame(msg: &TransportMessage) -> (MessageKind, Vec<u8>) {
    match msg {
        TransportMessage::TensorData {
            tensor_id,
            shape,
            data,
        } => {
            // Format: [tensor_id_len: u32][tensor_id][shape_dims: u32][shape...][data]
            let id_bytes = tensor_id.as_bytes();
            let mut buf = prepared_frame(4 + id_bytes.len() + 4 + shape.len() * 8 + data.len());
            buf.extend_from_slice(&(id_bytes.len() as u32).to_be_bytes());
            buf.extend_from_slice(id_bytes);
            buf.extend_from_slice(&(shape.len() as u32).to_be_bytes());
            for &dim in shape {
                buf.extend_from_slice(&(dim as u64).to_be_bytes());
            }
            buf.extend_from_slice(data);
            (MessageKind::TensorData, buf)
        }
        TransportMessage::Control { operation, payload } => {
            let op_bytes = operation.as_bytes();
            let mut buf = prepared_frame(4 + op_bytes.len() + payload.len());
            buf.extend_from_slice(&(op_bytes.len() as u32).to_be_bytes());
            buf.extend_from_slice(op_bytes);
            buf.extend_from_slice(payload);
            (MessageKind::Control, buf)
        }
    }
}

/// Serialize a [`TransportMessage`] to its frame kind and payload bytes: the
/// payload slice of [`encode_message_frame`], for tests that inspect or
/// round-trip the encoding without a stream.
#[cfg(test)]
pub(crate) fn serialize_message(msg: &TransportMessage) -> (MessageKind, Vec<u8>) {
    let (kind, frame) = encode_message_frame(msg);
    (kind, frame[FRAME_HEADER_LEN..].to_vec())
}

/// Deserialize a [`TransportMessage`] from a raw payload and its kind tag.
pub(crate) fn deserialize_message(kind: MessageKind, payload: &[u8]) -> Result<TransportMessage> {
    match kind {
        MessageKind::TensorData => {
            if payload.len() < 4 {
                anyhow::bail!("tensor message too short for id length");
            }
            let id_len = u32::from_be_bytes(payload[..4].try_into()?) as usize;
            let pos = 4;
            if payload.len() < pos + id_len + 4 {
                anyhow::bail!("tensor message too short for id + shape length");
            }
            let tensor_id = String::from_utf8(payload[pos..pos + id_len].to_vec())
                .context("invalid tensor id")?;
            let pos = pos + id_len;
            let ndims = u32::from_be_bytes(payload[pos..pos + 4].try_into()?) as usize;
            let pos = pos + 4;
            if payload.len() < pos + ndims * 8 {
                anyhow::bail!("tensor message too short for shape");
            }
            let mut shape = Vec::with_capacity(ndims);
            for i in 0..ndims {
                let offset = pos + i * 8;
                let dim = u64::from_be_bytes(payload[offset..offset + 8].try_into()?) as usize;
                shape.push(dim);
            }
            let data_start = pos + ndims * 8;
            let data = Bytes::copy_from_slice(&payload[data_start..]);
            Ok(TransportMessage::TensorData {
                tensor_id,
                shape,
                data,
            })
        }
        MessageKind::Control => {
            if payload.len() < 4 {
                anyhow::bail!("control message too short for operation length");
            }
            let op_len = u32::from_be_bytes(payload[..4].try_into()?) as usize;
            if payload.len() < 4 + op_len {
                anyhow::bail!("control message too short for operation");
            }
            let operation =
                String::from_utf8(payload[4..4 + op_len].to_vec()).context("invalid operation")?;
            let data = Bytes::copy_from_slice(&payload[4 + op_len..]);
            Ok(TransportMessage::Control {
                operation,
                payload: data,
            })
        }
        _ => anyhow::bail!("unexpected message kind for deserialization: {kind:?}"),
    }
}

#[cfg(test)]
#[path = "tcp_frame_tests.rs"]
mod tests;
