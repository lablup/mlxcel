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

use super::*;
use crate::distributed::cluster_auth::ClusterSecret;

fn keys(
    auth: &ClusterAuth,
    suite: FrameSuite,
    nonces: ([u8; 32], [u8; 32]),
) -> (SessionKeys, SessionKeys) {
    (
        SessionKeys::derive(auth, suite, &nonces.0, &nonces.1, HandshakeRole::Client),
        SessionKeys::derive(auth, suite, &nonces.0, &nonces.1, HandshakeRole::Server),
    )
}

fn frame(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut f = vec![kind];
    f.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    f.extend_from_slice(payload);
    f
}

fn header(frame: &[u8]) -> [u8; FRAME_HEADER_LEN] {
    frame[..FRAME_HEADER_LEN].try_into().unwrap()
}

#[test]
fn suite_wire_ids_round_trip_and_tag_lengths_match() {
    for suite in FrameSuite::ALL {
        assert_eq!(FrameSuite::from_wire_id(suite.wire_id()), Some(suite));
        assert_eq!(suite.tag_len(), if suite.encrypts() { 16 } else { 32 });
    }
    assert_eq!(FrameSuite::from_wire_id(0), None);
    assert_eq!(FrameSuite::from_wire_id(9), None);
}

#[test]
fn every_suite_round_trips_in_both_directions() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    for suite in FrameSuite::ALL {
        let (client, server) = keys(&auth, suite, ([1u8; 32], [2u8; 32]));
        let (mut c_seal, mut c_open) = client.into_codec();
        let (mut s_seal, mut s_open) = server.into_codec();
        for (seal, open) in [(&mut c_seal, &mut s_open), (&mut s_seal, &mut c_open)] {
            for i in 0..3u8 {
                let payload = vec![0x40 + i; 100 + i as usize];
                let mut f = frame(1, &payload);
                seal.seal(&mut f).unwrap();
                assert_eq!(f.len(), FRAME_HEADER_LEN + payload.len() + suite.tag_len());
                if suite.encrypts() {
                    assert_ne!(
                        &f[FRAME_HEADER_LEN..FRAME_HEADER_LEN + payload.len()],
                        &payload[..]
                    );
                } else {
                    assert_eq!(
                        &f[FRAME_HEADER_LEN..FRAME_HEADER_LEN + payload.len()],
                        &payload[..]
                    );
                }
                let h = header(&f);
                let n = open.open(&h, &mut f[FRAME_HEADER_LEN..]).unwrap();
                assert_eq!(n, payload.len());
                assert_eq!(&f[FRAME_HEADER_LEN..FRAME_HEADER_LEN + n], &payload[..]);
            }
        }
    }
}

#[test]
fn altered_payload_header_or_tag_is_rejected() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    for suite in FrameSuite::ALL {
        let (client, server) = keys(&auth, suite, ([3u8; 32], [4u8; 32]));
        let (mut seal, _) = client.into_codec();
        let (_, mut open) = server.into_codec();
        let mut sealed = frame(2, b"payload-bytes");
        seal.seal(&mut sealed).unwrap();

        // Payload byte.
        let mut f = sealed.clone();
        f[FRAME_HEADER_LEN + 2] ^= 0x80;
        assert_eq!(
            open.open(&header(&f), &mut f[FRAME_HEADER_LEN..]),
            Err(FrameRejected)
        );
        // Header (kind) byte.
        let mut f = sealed.clone();
        f[0] = 1;
        assert_eq!(
            open.open(&header(&f), &mut f[FRAME_HEADER_LEN..]),
            Err(FrameRejected)
        );
        // Tag byte.
        let mut f = sealed.clone();
        let last = f.len() - 1;
        f[last] ^= 0x01;
        assert_eq!(
            open.open(&header(&f), &mut f[FRAME_HEADER_LEN..]),
            Err(FrameRejected)
        );
        // Too short to hold a tag.
        let mut short = vec![0u8; suite.tag_len() - 1];
        assert_eq!(open.open(&header(&sealed), &mut short), Err(FrameRejected));
        // A rejected frame does not advance the sequence: the untouched frame still opens.
        let mut f = sealed;
        assert!(open.open(&header(&f), &mut f[FRAME_HEADER_LEN..]).is_ok());
    }
}

#[test]
fn replay_and_reorder_fail_under_the_implicit_sequence() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    for suite in FrameSuite::ALL {
        let (client, server) = keys(&auth, suite, ([5u8; 32], [6u8; 32]));
        let (mut seal, _) = client.into_codec();
        let (_, mut open) = server.into_codec();
        let mut first = frame(2, b"first");
        seal.seal(&mut first).unwrap();
        let mut second = frame(2, b"second");
        seal.seal(&mut second).unwrap();

        // Reorder: second before first.
        let mut f = second.clone();
        assert_eq!(
            open.open(&header(&f), &mut f[FRAME_HEADER_LEN..]),
            Err(FrameRejected)
        );
        // In order both open.
        let mut f = first.clone();
        assert!(open.open(&header(&f), &mut f[FRAME_HEADER_LEN..]).is_ok());
        let mut f = second;
        assert!(open.open(&header(&f), &mut f[FRAME_HEADER_LEN..]).is_ok());
        // Replay of the first.
        let mut f = first;
        assert_eq!(
            open.open(&header(&f), &mut f[FRAME_HEADER_LEN..]),
            Err(FrameRejected)
        );
    }
}

#[test]
fn keys_depend_on_nonces_direction_and_secret() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let other_auth = ClusterAuth::new(&ClusterSecret::random());
    let suite = FrameSuite::default();
    let (client, _) = keys(&auth, suite, ([7u8; 32], [8u8; 32]));
    let (mut seal, _) = client.into_codec();
    let mut sealed = frame(1, b"x");
    seal.seal(&mut sealed).unwrap();

    // Other nonces, same secret.
    let (_, server) = keys(&auth, suite, ([7u8; 32], [9u8; 32]));
    let (_, mut open) = server.into_codec();
    let mut f = sealed.clone();
    assert_eq!(
        open.open(&header(&f), &mut f[FRAME_HEADER_LEN..]),
        Err(FrameRejected)
    );
    // Same nonces, other secret.
    let (_, server) = keys(&other_auth, suite, ([7u8; 32], [8u8; 32]));
    let (_, mut open) = server.into_codec();
    let mut f = sealed.clone();
    assert_eq!(
        open.open(&header(&f), &mut f[FRAME_HEADER_LEN..]),
        Err(FrameRejected)
    );
    // Wrong direction: the client's own receive key does not open its sends.
    let (client, _) = keys(&auth, suite, ([7u8; 32], [8u8; 32]));
    let (_, mut own_open) = client.into_codec();
    let mut f = sealed;
    assert_eq!(
        own_open.open(&header(&f), &mut f[FRAME_HEADER_LEN..]),
        Err(FrameRejected)
    );
}

#[test]
fn sequence_exhaustion_refuses_to_seal_and_leaves_the_frame_untouched() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (client, _) = keys(&auth, FrameSuite::default(), ([1u8; 32], [1u8; 32]));
    let (mut seal, _) = client.into_codec();
    seal.seq = u64::MAX;
    let mut f = frame(1, b"last");
    let before = f.clone();
    assert_eq!(seal.seal(&mut f), Err(SequenceExhausted));
    assert_eq!(f, before);
}

#[test]
fn debug_output_redacts_keys() {
    let auth = ClusterAuth::new(&ClusterSecret::random());
    let (client, _) = keys(&auth, FrameSuite::default(), ([1u8; 32], [2u8; 32]));
    let text = format!("{client:?}");
    assert!(text.contains("redacted"), "{text}");
}
