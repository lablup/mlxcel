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
//! message, a stream chunk, an RPC request or an RPC response, goes through
//! [`ProtectedStream::write_frame`], and every inbound frame through
//! [`ProtectedStream::read_frame`]; the stage assignment channel uses the
//! same two methods. There is no second framing path. A stream built with
//! [`ProtectedStream::plain`] (no cluster auth, single-node defaults) uses the
//! plain layout.
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

use super::frame_protection::{FRAME_HEADER_LEN, FrameOpener, FrameSealer, SessionKeys};
use super::transport::{MessageKind, TransportMessage};

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

    /// Whether the peer has closed or reset this connection, for a pool
    /// deciding if an idle connection is still worth reusing (#2695).
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
    /// No data waiting (would block) or unread bytes mean the peer is still
    /// there. A zero-length read (orderly shutdown) or a socket error such as
    /// a reset means it is gone. A peer that dies after this check is not
    /// detected here; the write or the read that follows reports it.
    pub fn peer_closed(&self) -> bool {
        let mut probe = [std::mem::MaybeUninit::<u8>::uninit(); 1];
        match socket2::SockRef::from(&self.stream).peek(&mut probe) {
            Ok(0) => true,
            Ok(_) => false,
            Err(e) => !matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            ),
        }
    }

    /// Write one frame: header, payload, and (when protected) the tag, as a
    /// single buffer so small frames cost one write syscall.
    pub async fn write_frame(&mut self, kind: MessageKind, payload: &[u8]) -> Result<()> {
        let tag_len = self
            .codec
            .as_ref()
            .map_or(0, |(sealer, _)| sealer.suite().tag_len());
        let mut buf = Vec::with_capacity(FRAME_HEADER_LEN + payload.len() + tag_len);
        buf.push(kind as u8);
        buf.extend_from_slice(&(payload.len() as u64).to_be_bytes());
        buf.extend_from_slice(payload);
        if let Some((sealer, _)) = self.codec.as_mut() {
            sealer.seal(&mut buf)?;
        }
        self.stream.write_all(&buf).await?;
        self.stream.flush().await?;
        Ok(())
    }

    /// Read one frame. Returns `(kind, payload)`; the payload is verified
    /// (and decrypted) when the connection is protected. Errors on a closed
    /// stream, a malformed or oversize header, or a frame that fails
    /// authentication ([`super::frame_protection::FrameRejected`], which the
    /// caller can downcast); after any error the connection must be closed.
    pub async fn read_frame(&mut self, max_frame_size: usize) -> Result<(MessageKind, BytesMut)> {
        let mut header = [0u8; FRAME_HEADER_LEN];
        self.stream
            .read_exact(&mut header)
            .await
            .context("read frame header")?;
        let kind = MessageKind::try_from(header[0])?;
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
        if let Some((_, opener)) = self.codec.as_mut() {
            let payload_len = opener.open(&header, &mut body)?;
            body.truncate(payload_len);
        }
        Ok((kind, body))
    }
}

/// Serialize a [`TransportMessage`] to its frame kind and payload bytes.
pub(crate) fn serialize_message(msg: &TransportMessage) -> (MessageKind, Vec<u8>) {
    match msg {
        TransportMessage::TensorData {
            tensor_id,
            shape,
            data,
        } => {
            // Format: [tensor_id_len: u32][tensor_id][shape_dims: u32][shape...][data]
            let id_bytes = tensor_id.as_bytes();
            let mut buf = Vec::with_capacity(4 + id_bytes.len() + 4 + shape.len() * 8 + data.len());
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
            let mut buf = Vec::with_capacity(4 + op_bytes.len() + payload.len());
            buf.extend_from_slice(&(op_bytes.len() as u32).to_be_bytes());
            buf.extend_from_slice(op_bytes);
            buf.extend_from_slice(payload);
            (MessageKind::Control, buf)
        }
    }
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
