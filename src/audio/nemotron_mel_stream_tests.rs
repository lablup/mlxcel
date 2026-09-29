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

use mlxcel_core::utils::array_to_vec_f32;

use super::*;

fn signal(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            0.4 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 1_730.0 * t + 0.3).sin()
                + 0.05 * ((i * 7919 % 1013) as f32 / 1013.0 - 0.5)
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn stream_all(x: &[f32], chunk: usize, lookahead: Option<usize>) -> (Vec<f32>, usize, usize) {
    let args = MelArgs::default();
    let mut stream = StreamingLogMel::new(&args, lookahead).unwrap();
    let mut joined = Vec::new();
    let mut frames = 0;
    let mut max_buffer = 0;
    for piece in x.chunks(chunk) {
        let out = stream.push(piece).unwrap();
        let shape = mlxcel_core::array_shape(&out);
        assert_eq!((shape[0], shape[2]), (1, 128));
        frames += shape[1] as usize;
        joined.extend(array_to_vec_f32(&out));
        max_buffer = max_buffer.max(stream.buffered_samples());
    }
    let tail = stream.flush().unwrap();
    frames += mlxcel_core::array_shape(&tail)[1] as usize;
    joined.extend(array_to_vec_f32(&tail));
    assert!(stream.is_closed());
    (joined, frames, max_buffer)
}

#[test]
fn streaming_mel_joined_outputs_equal_offline() {
    let x = signal(16_000);
    let (offline, offline_frames) = log_mel_spectrogram(&x, &MelArgs::default()).unwrap();
    let (joined, frames, max_buffer) = stream_all(&x, 300, None);
    assert_eq!(frames, offline_frames);
    let d = max_abs_diff(&joined, &offline);
    assert!(d <= 1e-5, "streamed log-mel drifted by {d}");
    // Bounded state: look-behind (2 hops) + lookahead + one chunk.
    assert!(
        max_buffer <= 2 * 160 + 256 + 300 + 160,
        "buffer grew to {max_buffer}"
    );
}

#[test]
fn streaming_mel_with_frame_lookahead_emits_eight_hops_per_frame() {
    let x = signal(1280 * 6 + 700);
    let args = MelArgs::default();
    let mut stream = StreamingLogMel::new(&args, Some(1280)).unwrap();
    let mut counts = Vec::new();
    let mut joined = Vec::new();
    for piece in x.chunks(1280) {
        let (values, n) = stream.push_values(piece, false).unwrap();
        counts.push(n);
        joined.extend(values);
    }
    // First push: center 0 only; then 8 hops per 1280-sample push; the
    // trailing 700 samples release centers up to 8380 - 1280 = 7100.
    assert_eq!(counts, vec![1, 8, 8, 8, 8, 8, 4]);
    let (tail, _) = stream.push_values(&[], true).unwrap();
    joined.extend(tail);
    let (offline, _) = log_mel_spectrogram(&x, &args).unwrap();
    assert!(max_abs_diff(&joined, &offline) <= 1e-5);
}

#[test]
fn streaming_mel_rejects_bad_settings_and_closed_push() {
    let args = MelArgs::default();
    assert!(StreamingLogMel::new(&args, Some(100)).is_err());
    let padded = MelArgs {
        pad_to: 16,
        ..MelArgs::default()
    };
    assert!(StreamingLogMel::new(&padded, None).is_err());
    let normalized = MelArgs {
        normalize: "per_feature".to_string(),
        ..MelArgs::default()
    };
    assert!(StreamingLogMel::new(&normalized, None).is_err());

    let mut stream = StreamingLogMel::new(&args, None).unwrap();
    let out = stream.push(&signal(100)).unwrap();
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 0, 128]);
    stream.flush().unwrap();
    assert!(stream.push(&signal(10)).is_err());
}
