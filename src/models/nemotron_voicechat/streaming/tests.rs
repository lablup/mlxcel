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

use super::buffer::FrameBuffer;
use super::{FrameTiming, TokenAccumulator, VoiceChatEvent, VoiceChatProfile, percentile};

#[test]
fn push_audio_buffers_arbitrary_chunk_boundaries() {
    let mut buffer = FrameBuffer::new(1280);
    let counts: Vec<usize> = [300, 1000, 1280, 2000]
        .iter()
        .map(|&n| buffer.push(&vec![0.5; n]).len())
        .collect();
    assert_eq!(counts, [0, 1, 1, 1]);
    assert_eq!(buffer.pending_len(), 4580 - 3 * 1280);
    let tail = buffer.take_partial(true).expect("padded partial frame");
    assert_eq!(tail.len(), 1280);
    assert_eq!(tail[..1].to_vec(), vec![0.5]);
    assert_eq!(tail[1279], 0.0);
    assert_eq!(buffer.pending_len(), 0);
}

#[test]
fn flush_without_padding_drops_partial_frame() {
    let mut buffer = FrameBuffer::new(1280);
    assert!(buffer.push(&[0.1; 700]).is_empty());
    assert!(buffer.take_partial(false).is_none());
    assert_eq!(buffer.pending_len(), 0);
    assert!(
        buffer.take_partial(true).is_none(),
        "empty buffer yields no frame"
    );
}

#[test]
fn frames_keep_sample_order_across_pushes() {
    let mut buffer = FrameBuffer::new(4);
    let samples: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let mut frames = buffer.push(&samples[..3]);
    frames.extend(buffer.push(&samples[3..]));
    assert_eq!(
        frames,
        vec![vec![0.0, 1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0, 7.0]]
    );
    assert_eq!(buffer.take_partial(true), Some(vec![8.0, 9.0, 0.0, 0.0]));
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
