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

use super::tests::{mel, tiny_args, tiny_weights};
use super::*;

fn encoder() -> FastConformerEncoder {
    FastConformerEncoder::from_weights(&tiny_weights("enc"), "enc", &tiny_args(), (64, 4)).unwrap()
}

fn frame(x: &MlxArray, t: i32) -> Vec<f32> {
    let s = mlxcel_core::array_shape(x);
    array_to_vec_f32(&mlxcel_core::slice(x, &[0, t, 0], &[1, t + 1, s[2]]))
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn mel_slice(m: &MlxArray, start: i32, end: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::slice(m, &[0, start, 0], &[1, end, 16])
}

#[test]
fn streamed_encoder_matches_offline_chunked_limited() {
    // Layer stack: per-frame streaming with attention/conv caches equals one
    // offline `chunked_limited` call with right context 0. A left context of
    // 2 makes the attention cache trim on every frame after the third.
    let enc = encoder();
    let input = mel(65, 16);
    let context = [2, 0];
    let (offline, frames) = enc.forward_with_context(&input, 65, context).unwrap();
    assert_eq!(frames, 9);
    let sub = enc.pre_encode.forward(&input).unwrap();
    let mut state = ConformerStreamingState::new(&enc, 1, context).unwrap();
    for t in 0..frames as i32 {
        let h = mlxcel_core::slice(&sub, &[0, t, 0], &[1, t + 1, 8]);
        let out = state.stream_layers(&h).unwrap();
        state.materialize(&[&out]);
        let d = max_abs(&array_to_vec_f32(&out), &frame(&offline, t));
        assert!(d < 1e-3, "frame {t}: streamed layers drifted by {d}");
    }
}

#[test]
fn push_of_eight_mel_frames_emits_one_encoder_frame() {
    // The session pattern: the streaming log-mel yields 1 frame for the first
    // 1280-sample push and 8 afterwards; each push must give exactly one
    // encoder frame. The first three frames see an 8-aligned subsampling
    // window and equal the offline encoder.
    let enc = encoder();
    let input = mel(65, 16);
    let (offline, _) = enc.forward_with_context(&input, 65, [70, 0]).unwrap();
    let mut state = ConformerStreamingState::new(&enc, 1, [70, 0]).unwrap();
    let mut bounds = vec![(0, 1)];
    bounds.extend((0..8).map(|k| (1 + 8 * k, 9 + 8 * k)));
    for (t, (start, end)) in bounds.into_iter().enumerate() {
        let out = state
            .push(&mel_slice(&input, start, end), false, true)
            .unwrap();
        assert_eq!(out.len(), 1, "push {t}");
        assert_eq!(mlxcel_core::array_shape(&out[0]), vec![1, 1, 8]);
        state.materialize(&[&out[0]]);
        let v = array_to_vec_f32(&out[0]);
        assert!(v.iter().all(|x| x.is_finite()));
        if t < 3 {
            let d = max_abs(&v, &frame(&offline, t as i32));
            assert!(d < 1e-3, "frame {t}: {d}");
        }
    }
    assert_eq!(state.emitted_frames(), 9);
    assert!(!state.is_closed());

    // Reference behavior: an 8-frame first push already holds the boundary
    // frame at mel index 8, so it emits two frames.
    let mut fresh = ConformerStreamingState::new(&enc, 1, [70, 0]).unwrap();
    let out = fresh.push(&mel_slice(&input, 0, 8), false, true).unwrap();
    let total: i32 = out.iter().map(|o| mlxcel_core::array_shape(o)[1]).sum();
    assert_eq!(total, 2);
}

#[test]
fn full_chunks_emit_without_partial_and_final_closes() {
    let enc = encoder();
    let input = mel(16, 16);
    let mut state = ConformerStreamingState::new(&enc, 1, [70, 0]).unwrap();
    let mut emitted = 0;
    for k in 0..4 {
        let out = state
            .push(&mel_slice(&input, 4 * k, 4 * k + 4), false, false)
            .unwrap();
        emitted += out.len();
        if k % 2 == 0 {
            assert!(out.is_empty(), "half chunk must wait");
        }
    }
    assert_eq!(emitted, 2);
    let empty = mlxcel_core::zeros(&[1, 0, 16], mlxcel_core::dtype::FLOAT32);
    assert!(state.push(&empty, true, false).unwrap().is_empty());
    assert!(state.is_closed());
    assert!(state.push(&mel_slice(&input, 0, 1), false, true).is_err());
}

#[test]
fn streaming_rejects_bad_inputs() {
    let enc = encoder();
    assert!(ConformerStreamingState::new(&enc, 0, [70, 0]).is_err());
    assert!(ConformerStreamingState::new(&enc, 1, [-1, 0]).is_err());
    let mut state = ConformerStreamingState::new(&enc, 1, [70, 0]).unwrap();
    assert!(state.push(&mel(4, 12), false, true).is_err());
    // A 2-D `[n, feat]` chunk is accepted.
    let flat = mlxcel_core::reshape(&mel(1, 16), &[1, 16]);
    assert_eq!(state.push(&flat, false, true).unwrap().len(), 1);
}
