// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Unit tests for the resumable stream-session store (#1444).

use super::*;

fn manager() -> StreamSessionManager {
    StreamSessionManager::new()
}

// -- buffer semantics --

#[test]
fn append_and_read_from_zero_returns_everything() {
    let s = StreamSession::new("c".into(), STREAM_SESSION_MAX_BYTES);
    assert!(s.append(b"data: a\n\n"));
    assert!(s.append(b"data: b\n\n"));
    match s.read_chunk(0) {
        ReadChunk::Data(d) => assert_eq!(d, b"data: a\n\ndata: b\n\n"),
        other => panic!("expected data, got {other:?}"),
    }
    assert_eq!(s.total_size(), 18);
    assert_eq!(s.dropped_prefix(), 0);
}

#[test]
fn read_at_live_edge_is_pending_then_eof_after_finalize() {
    let s = StreamSession::new("c".into(), STREAM_SESSION_MAX_BYTES);
    s.append(b"xyz");
    assert_eq!(s.read_chunk(3), ReadChunk::Pending);
    assert!(!s.is_done());
    assert_eq!(s.completed_at(), 0);
    s.finalize();
    assert!(s.is_done());
    assert!(s.completed_at() > 0);
    assert_eq!(s.read_chunk(3), ReadChunk::Eof);
    // Finalize is idempotent and blocks further appends.
    let first_completed = s.completed_at();
    s.finalize();
    assert_eq!(s.completed_at(), first_completed);
    assert!(!s.append(b"late"));
    assert_eq!(s.total_size(), 3);
}

#[test]
fn cap_drops_the_front_and_replay_below_the_prefix_is_lost() {
    let s = StreamSession::new("c".into(), 8);
    s.append(b"AAAA");
    s.append(b"BBBB");
    s.append(b"CC");
    // Cap 8: total 10, front 2 bytes dropped.
    assert_eq!(s.total_size(), 10);
    assert_eq!(s.dropped_prefix(), 2);
    assert_eq!(s.read_chunk(0), ReadChunk::OffsetLost);
    assert_eq!(s.read_chunk(1), ReadChunk::OffsetLost);
    match s.read_chunk(2) {
        ReadChunk::Data(d) => assert_eq!(d, b"AABBBBCC"),
        other => panic!("expected data, got {other:?}"),
    }
    match s.read_chunk(6) {
        ReadChunk::Data(d) => assert_eq!(d, b"BBCC"),
        other => panic!("expected data, got {other:?}"),
    }
}

#[test]
fn single_chunk_larger_than_the_cap_keeps_only_the_tail() {
    let s = StreamSession::new("c".into(), 4);
    s.append(b"0123456789");
    assert_eq!(s.total_size(), 10);
    assert_eq!(s.dropped_prefix(), 6);
    match s.read_chunk(6) {
        ReadChunk::Data(d) => assert_eq!(d, b"6789"),
        other => panic!("expected data, got {other:?}"),
    }
}

// -- SSE framing --

#[test]
fn frame_sse_matches_axum_event_wire_format() {
    assert_eq!(frame_sse(None, "{\"a\":1}"), b"data: {\"a\":1}\n\n");
    assert_eq!(
        frame_sse(Some("message_start"), "{}"),
        b"event: message_start\ndata: {}\n\n"
    );
    // Multi-line payloads become one data: line per line, as axum's Event
    // serializes them.
    assert_eq!(frame_sse(None, "a\nb"), b"data: a\ndata: b\n\n");
}

#[test]
fn resume_tee_frames_and_finalizes_on_drop() {
    let m = manager();
    let session = m.create_or_replace("conv", None);
    {
        let tee = ResumeTee::new(session.clone());
        tee.write_data("[DONE]");
        tee.write_event("ping", "{}");
    }
    // Tee dropped: session finalized.
    assert!(session.is_done());
    match session.read_chunk(0) {
        ReadChunk::Data(d) => {
            assert_eq!(d, b"data: [DONE]\n\nevent: ping\ndata: {}\n\n");
        }
        other => panic!("expected data, got {other:?}"),
    }
}

// -- manager keying, replacement, ownership --

#[test]
fn create_or_replace_cancels_and_finalizes_the_previous_session() {
    let m = manager();
    let first = m.create_or_replace("conv", None);
    let token = first.cancellation_token();
    first.append(b"old");
    let second = m.create_or_replace("conv", None);
    assert!(first.is_cancelled());
    assert!(token.load(std::sync::atomic::Ordering::Acquire));
    assert!(first.is_done());
    assert!(!second.is_cancelled());
    // The map now resolves to the fresh session.
    let got = m.get("conv", &None).expect("session");
    assert!(std::ptr::eq(Arc::as_ptr(&got), Arc::as_ptr(&second)));
}

#[test]
fn same_conversation_id_under_different_owners_is_two_sessions() {
    let m = manager();
    let a = m.create_or_replace("conv", Some("key-a".into()));
    let b = m.create_or_replace("conv", Some("key-b".into()));
    // Key B's create did not cancel key A's stream: no cross-key control.
    assert!(!a.is_cancelled());
    assert!(!b.is_cancelled());
    assert!(m.get("conv", &Some("key-a".into())).is_some());
    assert!(m.get("conv", &Some("key-b".into())).is_some());
    // And a third key sees neither.
    assert!(m.get("conv", &Some("key-c".into())).is_none());
    assert!(m.get("conv", &None).is_none());
}

#[test]
fn lookup_matches_exact_and_per_model_variants_for_the_owner_only() {
    let m = manager();
    m.create_or_replace("conv-1", None);
    m.create_or_replace("conv-1::org/model", None);
    m.create_or_replace("conv-10", None);
    m.create_or_replace("conv-1", Some("key-a".into()));

    let found = m.lookup(&["conv-1".to_string()], &None);
    let mut ids: Vec<&str> = found.iter().map(|s| s.conversation_id()).collect();
    ids.sort_unstable();
    // Exact + `::` variant; NOT the `conv-10` prefix and NOT key-a's session.
    assert_eq!(ids, vec!["conv-1", "conv-1::org/model"]);

    // Empty request answers nothing; unknown ids answer nothing.
    assert!(m.lookup(&[], &None).is_empty());
    assert!(m.lookup(&["missing".to_string()], &None).is_empty());
    // The other owner sees only its own.
    let found = m.lookup(&["conv-1".to_string()], &Some("key-a".into()));
    assert_eq!(found.len(), 1);
}

#[test]
fn evict_and_cancel_is_idempotent_and_owner_scoped() {
    let m = manager();
    let s = m.create_or_replace("conv", Some("key-a".into()));

    // Wrong owner: no-op, session untouched.
    m.evict_and_cancel("conv", &Some("key-b".into()));
    assert!(!s.is_cancelled());
    assert!(m.get("conv", &Some("key-a".into())).is_some());

    // Right owner: cancelled, finalized, gone.
    m.evict_and_cancel("conv", &Some("key-a".into()));
    assert!(s.is_cancelled());
    assert!(s.is_done());
    assert!(m.get("conv", &Some("key-a".into())).is_none());

    // Second delete of the same id: still fine.
    m.evict_and_cancel("conv", &Some("key-a".into()));
}

// -- retention bounds --

#[test]
fn sweep_evicts_completed_sessions_past_the_ttl_and_keeps_live_ones() {
    let m = manager();
    let done = m.create_or_replace("done", None);
    done.finalize();
    let live = m.create_or_replace("live", None);

    // Within the TTL: both retained.
    m.sweep(done.completed_at() + STREAM_SESSION_TTL_SECONDS - 1);
    assert!(m.get("done", &None).is_some());

    // Past the TTL: the completed one is evicted, the live one stays.
    m.sweep(done.completed_at() + STREAM_SESSION_TTL_SECONDS + 1);
    assert!(m.get("done", &None).is_none());
    assert!(m.get("live", &None).is_some());
    assert!(!live.is_done());
}

#[test]
fn retained_completed_sessions_are_capped() {
    let m = manager();
    for i in 0..(STREAM_SESSION_MAX_RETAINED + 40) {
        let s = m.create_or_replace(&format!("conv-{i}"), None);
        s.finalize();
    }
    m.sweep(now_seconds());
    let mut retained = 0usize;
    for i in 0..(STREAM_SESSION_MAX_RETAINED + 40) {
        if m.get(&format!("conv-{i}"), &None).is_some() {
            retained += 1;
        }
    }
    assert_eq!(retained, STREAM_SESSION_MAX_RETAINED);
}

// -- async reader wakeups --

#[tokio::test]
async fn wait_for_change_wakes_on_append_and_finalize() {
    let s = Arc::new(StreamSession::new("c".into(), STREAM_SESSION_MAX_BYTES));
    let reader = s.clone();
    let handle = tokio::spawn(async move {
        let mut offset = 0usize;
        let mut collected: Vec<u8> = Vec::new();
        loop {
            match reader.read_chunk(offset) {
                ReadChunk::Data(d) => {
                    offset += d.len();
                    collected.extend_from_slice(&d);
                }
                ReadChunk::Pending => reader.wait_for_change().await,
                ReadChunk::Eof => return collected,
                ReadChunk::OffsetLost => panic!("offset lost"),
            }
        }
    });
    // Appends run from a blocking context, as the generation task does.
    let writer = s.clone();
    tokio::task::spawn_blocking(move || {
        writer.append(b"data: 1\n\n");
        writer.append(b"data: 2\n\n");
        writer.finalize();
    })
    .await
    .expect("writer task");
    let collected = tokio::time::timeout(std::time::Duration::from_secs(5), handle)
        .await
        .expect("reader finishes")
        .expect("reader task");
    assert_eq!(collected, b"data: 1\n\ndata: 2\n\n");
}
