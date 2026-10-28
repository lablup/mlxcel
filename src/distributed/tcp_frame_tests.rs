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

//! The in-place sealing path of the frame codec (#2694, item 2): a message
//! encoded into its frame buffer is sealed without a second allocation, and
//! a frame of the size of a per-token logits reply round-trips byte for byte
//! over a protected and over a plain loopback connection.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tokio::net::{TcpListener, TcpStream};

use super::*;
use crate::distributed::cluster_auth::{ClusterAuth, ClusterSecret};
use crate::distributed::cluster_handshake::{client_handshake, server_handshake};
use crate::distributed::frame_protection::{FrameChannel, FrameSuite, HandshakeRole};

/// The size of a qwen3 per-token logits reply (#2684).
const LOGITS_REPLY_BYTES: usize = 405 * 1024;
const ARRIVAL: Duration = Duration::from_secs(5);

fn large_control() -> TransportMessage {
    let payload: Vec<u8> = (0..LOGITS_REPLY_BYTES).map(|i| (i % 251) as u8).collect();
    TransportMessage::Control {
        operation: "pipeline_stage_response".to_string(),
        payload: Bytes::from(payload),
    }
}

fn sealer_for(suite: FrameSuite) -> FrameSealer {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (sealer, _) = SessionKeys::derive(
        &auth,
        FrameChannel::Transport,
        suite,
        &[1u8; 32],
        &[2u8; 32],
        HandshakeRole::Client,
    )
    .into_codec();
    sealer
}

#[test]
fn finish_frame_seals_without_reallocating() {
    let suites: Vec<Option<FrameSuite>> = FrameSuite::ALL
        .iter()
        .copied()
        .map(Some)
        .chain(std::iter::once(None))
        .collect();
    for suite in suites {
        let (kind, mut frame) = encode_message_frame(&large_control());
        assert_eq!(kind, MessageKind::Control);
        let before_ptr = frame.as_ptr();
        let before_len = frame.len();
        let before_cap = frame.capacity();
        let payload_len = before_len - FRAME_HEADER_LEN;

        let mut sealer = suite.map(sealer_for);
        finish_frame(sealer.as_mut(), kind, &mut frame).unwrap();

        let tag_len = suite.map_or(0, FrameSuite::tag_len);
        assert_eq!(
            frame.as_ptr(),
            before_ptr,
            "{suite:?}: sealing reallocated the frame buffer"
        );
        assert_eq!(frame.capacity(), before_cap, "{suite:?}: capacity changed");
        assert_eq!(
            frame.len(),
            before_len + tag_len,
            "{suite:?}: the frame must grow by exactly the tag length"
        );
        assert_eq!(frame[0], MessageKind::Control as u8);
        assert_eq!(
            u64::from_be_bytes(frame[1..FRAME_HEADER_LEN].try_into().unwrap()),
            payload_len as u64,
            "{suite:?}: the header length must be the payload length"
        );
    }
}

#[test]
fn max_tag_len_covers_every_suite() {
    let widest = FrameSuite::ALL
        .iter()
        .map(|suite| suite.tag_len())
        .max()
        .unwrap();
    assert_eq!(widest, MAX_TAG_LEN, "MAX_TAG_LEN must track the widest tag");
}

/// Server side of one loopback pair: accepts, optionally authenticates, reads
/// one frame with a 1 MiB bound and returns the decoded message.
async fn serve_one(
    listener: TcpListener,
    auth: Option<Arc<ClusterAuth>>,
) -> (MessageKind, TransportMessage) {
    let (mut sock, _) = listener.accept().await.unwrap();
    let mut stream = match auth {
        Some(auth) => {
            let (_, keys) =
                server_handshake(&mut sock, &auth, "127.0.0.1:1", FrameSuite::default())
                    .await
                    .unwrap();
            ProtectedStream::protected(sock, keys)
        }
        None => ProtectedStream::plain(sock),
    };
    let (kind, payload) = tokio::time::timeout(ARRIVAL, stream.read_frame(1024 * 1024))
        .await
        .expect("the frame must arrive")
        .unwrap();
    assert_eq!(
        payload.len(),
        4 + "pipeline_stage_response".len() + LOGITS_REPLY_BYTES
    );
    (kind, deserialize_message(kind, &payload).unwrap())
}

async fn round_trip(protected: bool) {
    let auth = protected.then(|| Arc::new(ClusterAuth::new(&ClusterSecret::random())));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(serve_one(listener, auth.clone()));

    let mut sock = TcpStream::connect(addr).await.unwrap();
    let mut client = match auth {
        Some(auth) => {
            let identity = sock.local_addr().unwrap().to_string();
            let keys = client_handshake(&mut sock, &auth, &identity, FrameSuite::default())
                .await
                .unwrap();
            ProtectedStream::protected(sock, keys)
        }
        None => ProtectedStream::plain(sock),
    };
    assert_eq!(client.is_protected(), protected);

    let sent = large_control();
    let (kind, frame) = encode_message_frame(&sent);
    client.write_prepared_frame(kind, frame).await.unwrap();

    let (got_kind, got) = server.await.unwrap();
    assert_eq!(got_kind, MessageKind::Control);
    match (sent, got) {
        (
            TransportMessage::Control {
                operation: want_op,
                payload: want,
            },
            TransportMessage::Control {
                operation: got_op,
                payload: got,
            },
        ) => {
            assert_eq!(got_op, want_op);
            assert!(got == want, "payload bytes differ after the round trip");
        }
        other => panic!("unexpected message pair: {other:?}"),
    }
}

#[tokio::test]
async fn prepared_frame_round_trips_over_protected_loopback() {
    round_trip(true).await;
}

#[tokio::test]
async fn prepared_frame_round_trips_over_plain_loopback() {
    round_trip(false).await;
}
