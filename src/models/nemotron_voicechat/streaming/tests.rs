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

//! Unit tests for the streaming session's model-free parts: frame
//! buffering, the text-delta rule, events, and the profiler summary.

use super::buffer::{FrameBuffer, InputLifecycle};
use super::{FrameTiming, TokenAccumulator, VoiceChatEvent, VoiceChatProfile, percentile};

/// Push through an `InputLifecycle` whose per-frame step only counts.
fn counting_push(
    lifecycle: &mut InputLifecycle,
    n: usize,
    frames: &mut usize,
) -> Result<usize, &'static str> {
    lifecycle
        .push(&vec![0.5; n], "closed", |_| {
            *frames += 1;
            Ok::<_, &'static str>(vec![()])
        })
        .map(|events| events.len())
}

#[test]
fn push_audio_buffers_arbitrary_chunk_boundaries() {
    let mut lifecycle = InputLifecycle::new(1280);
    let mut frames = 0;
    let counts: Vec<usize> = [300, 1000, 1280, 2000]
        .iter()
        .map(|&n| counting_push(&mut lifecycle, n, &mut frames).unwrap())
        .collect();
    assert_eq!(counts, [0, 1, 1, 1]);
    let flushed = lifecycle
        .flush(true, |frame| {
            assert_eq!(frame.len(), 1280);
            frames += 1;
            Ok::<_, &'static str>(vec![()])
        })
        .expect("first flush runs")
        .unwrap();
    assert_eq!(flushed.len(), 1, "flush(pad) runs the partial frame");
    assert_eq!(frames, 4);
}

#[test]
fn flush_without_padding_drops_partial_frame() {
    let mut lifecycle = InputLifecycle::new(1280);
    let mut frames = 0;
    assert_eq!(counting_push(&mut lifecycle, 700, &mut frames), Ok(0));
    let flushed = lifecycle
        .flush(false, |_| Ok::<Vec<()>, &'static str>(vec![()]))
        .unwrap()
        .unwrap();
    assert!(flushed.is_empty());
}

#[test]
fn closed_session_rejects_push_and_close_is_idempotent() {
    let mut lifecycle = InputLifecycle::new(1280);
    let mut frames = 0;
    assert!(
        lifecycle
            .flush(true, |_| Ok::<Vec<()>, &'static str>(vec![]))
            .is_some()
    );
    assert!(lifecycle.is_closed());
    assert_eq!(
        counting_push(&mut lifecycle, 1280, &mut frames),
        Err("closed")
    );
    assert!(
        lifecycle
            .flush(true, |_| Ok::<Vec<()>, &'static str>(vec![]))
            .is_none()
    );
    assert!(!lifecycle.cancel(), "cancel after close is a no-op");

    let mut other = InputLifecycle::new(1280);
    assert!(other.cancel());
    assert!(!other.cancel());
    assert_eq!(counting_push(&mut other, 10, &mut frames), Err("closed"));
    assert_eq!(frames, 0);
}

#[test]
fn failing_frame_keeps_later_frames_pending() {
    let mut lifecycle = InputLifecycle::new(4);
    let mut seen = Vec::new();
    let mut calls = 0;
    // Three whole frames in one push; the second one fails (a context limit).
    let result = lifecycle.push(&[1.0; 12], "limit", |frame| {
        calls += 1;
        if calls == 2 {
            return Err("limit");
        }
        seen.push(frame.to_vec());
        Ok(vec![()])
    });
    assert_eq!(result, Err("limit"));
    // The third frame was not consumed and runs on the next push.
    let next = lifecycle.push(&[], "closed", |frame| {
        seen.push(frame.to_vec());
        Ok::<_, &'static str>(vec![()])
    });
    assert_eq!(next.map(|e| e.len()), Ok(1));
    assert_eq!(seen.len(), 2);
}

#[test]
fn frames_keep_sample_order_across_pushes() {
    let mut buffer = FrameBuffer::new(4);
    let samples: Vec<f32> = (0..10).map(|i| i as f32).collect();
    buffer.append(&samples[..3]);
    assert!(buffer.pop_frame().is_none());
    buffer.append(&samples[3..]);
    assert_eq!(buffer.pop_frame(), Some(vec![0.0, 1.0, 2.0, 3.0]));
    assert_eq!(buffer.pop_frame(), Some(vec![4.0, 5.0, 6.0, 7.0]));
    assert_eq!(buffer.pop_frame(), None);
    assert_eq!(buffer.pending_len(), 2);
    assert_eq!(buffer.take_partial(true), Some(vec![8.0, 9.0, 0.0, 0.0]));
    assert_eq!(buffer.pending_len(), 0);
}

#[test]
fn token_accumulator_delta_rule() {
    // A toy decoder: ids map to letters, except that id 9 revises the
    // previous character (as a tokenizer completing a multibyte sequence).
    let decode = |ids: &[i32]| -> Result<String, String> {
        let mut out = String::new();
        for &id in ids {
            if id == 9 {
                out.pop();
                out.push('Z');
            } else {
                out.push((b'a' + id as u8) as char);
            }
        }
        Ok(out)
    };
    let mut acc = TokenAccumulator::new(&[12, 11, 1, 2]);
    assert_eq!(
        acc.append(12, decode).unwrap(),
        None,
        "special ids are skipped"
    );
    assert_eq!(
        acc.append(0, decode).unwrap(),
        Some(("a".to_string(), "a".to_string()))
    );
    assert_eq!(
        acc.append(3, decode).unwrap(),
        Some(("d".to_string(), "ad".to_string()))
    );
    // "ad" + id 9 decodes to "aZ", which does not extend "ad": the delta
    // falls back to the single-token decode and the text stays cumulative.
    assert_eq!(
        acc.append(9, decode).unwrap(),
        Some(("Z".to_string(), "aZ".to_string()))
    );
    assert_eq!(acc.text(), "aZ");
    assert_eq!(acc.tokens(), &[0, 3, 9]);
}

#[test]
fn profile_summary_percentiles() {
    assert_eq!(percentile(&[], 0.5), 0.0);
    assert_eq!(percentile(&[4.0, 1.0, 3.0, 2.0], 0.5), 2.5);
    assert!((percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.95) - 4.8).abs() < 1e-12);

    let mut profile = VoiceChatProfile::new(80.0);
    for (i, total) in [500.0, 40.0, 60.0, 80.0, 100.0].iter().enumerate() {
        profile.frames.push(FrameTiming {
            frame_index: i as u64,
            perception_ms: total / 4.0,
            rnnt_ms: 1.0,
            language_ms: total / 2.0,
            tts_ms: total / 8.0,
            codec_ms: 2.0,
            total_ms: *total,
            host_syncs: i as u32 + 7,
            sub_stages: [("tts.backbone".to_string(), total / 16.0)].into(),
        });
    }
    let summary = profile.summary(1);
    assert_eq!(summary.frames, 4);
    assert_eq!(summary.dropped_cold_frames, 1);
    assert_eq!(summary.total.mean_ms, 70.0);
    assert_eq!(summary.total.p50_ms, 70.0);
    assert_eq!(summary.total.max_ms, 100.0);
    assert!((summary.realtime_factor - 70.0 / 80.0).abs() < 1e-12);
    assert!((summary.processing_frames_per_second - 1000.0 / 70.0).abs() < 1e-9);
    // The cold frame (7 syncs) is dropped; the rest have 8, 9, 10, 11.
    assert_eq!(summary.host_syncs.mean, 9.5);
    assert_eq!(summary.host_syncs.max, 11);
    assert_eq!(summary.sub_stages["tts.backbone"].mean_ms, 70.0 / 16.0);
    let json = serde_json::to_value(&summary).unwrap();
    assert!(json.get("sub_stages").is_some());
    let mut plain = profile.clone();
    for f in &mut plain.frames {
        f.sub_stages.clear();
    }
    let json = serde_json::to_value(plain.summary(1)).unwrap();
    assert!(
        json.get("sub_stages").is_none(),
        "sub_stages stays out of plain --profile output"
    );
    assert_eq!(profile.summary(10).frames, 0);
}

#[test]
fn event_frame_index_accessor() {
    let events = [
        VoiceChatEvent::Done { frame_index: 7 },
        VoiceChatEvent::Cancelled { frame_index: 3 },
        VoiceChatEvent::UserTranscriptDelta {
            frame_index: 5,
            delta: "hi".into(),
            text: "hi".into(),
        },
    ];
    let idx: Vec<u64> = events.iter().map(VoiceChatEvent::frame_index).collect();
    assert_eq!(idx, [7, 3, 5]);
}
