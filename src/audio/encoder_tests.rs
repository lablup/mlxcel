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

use super::AudioLinear;
use mlxcel_core::weights::WeightMap;

fn identity_weight(dim: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut data = vec![0.0f32; dim * dim];
    for i in 0..dim {
        data[i * dim + i] = 1.0;
    }
    mlxcel_core::from_slice_f32(&data, &[dim as i32, dim as i32])
}

fn to_vec_f32(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    let arr = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&arr);
    mlxcel_core::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// The Gemma 4 audio checkpoints ship finite input/output clamp bounds for
/// every ClippableLinear, and the reference implementation applies them at
/// inference. Skipping them decorrelates the Conformer stack (issue #782):
/// the per-block error compounds from ~8% relative RMS after block 0 to ~95%
/// after block 11 on a real clip. This test locks the clamp semantics in.
#[test]
fn audio_linear_applies_checkpoint_clip_bounds() {
    let mut weights = WeightMap::new();
    weights.insert(
        "clip.linear.weight".to_string(),
        mlxcel_core::copy(&identity_weight(4)),
    );
    weights.insert(
        "clip.input_min".to_string(),
        mlxcel_core::from_slice_f32(&[-1.0], &[1]),
    );
    weights.insert(
        "clip.input_max".to_string(),
        mlxcel_core::from_slice_f32(&[1.0], &[1]),
    );
    weights.insert(
        "clip.output_min".to_string(),
        mlxcel_core::from_slice_f32(&[-0.5], &[1]),
    );
    weights.insert(
        "clip.output_max".to_string(),
        mlxcel_core::from_slice_f32(&[0.5], &[1]),
    );

    let layer = AudioLinear::from_weights(&weights, "clip", 64, 4).expect("clippable linear");
    let x = mlxcel_core::from_slice_f32(&[2.0, -3.0, 0.3, 0.9], &[1, 4]);
    let y = to_vec_f32(&layer.forward(&x));

    // Input clamps to [-1, 1] -> identity matmul -> output clamps to [-0.5, 0.5]:
    // 2.0 -> 1.0 -> 0.5; -3.0 -> -1.0 -> -0.5; 0.3 passes through; 0.9 -> 0.5.
    let expected = [0.5, -0.5, 0.3, 0.5];
    for (actual, expected) in y.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "clipped output {y:?} != expected {expected:?}"
        );
    }
}

/// Without bound tensors in the checkpoint the layer must behave as a plain
/// linear (clamping disabled), so non-clippable checkpoints keep working.
#[test]
fn audio_linear_without_bounds_is_plain_linear() {
    let mut weights = WeightMap::new();
    weights.insert(
        "plain.linear.weight".to_string(),
        mlxcel_core::copy(&identity_weight(4)),
    );

    let layer = AudioLinear::from_weights(&weights, "plain", 64, 4).expect("plain linear");
    let x = mlxcel_core::from_slice_f32(&[2.0, -3.0, 0.3, 0.9], &[1, 4]);
    let y = to_vec_f32(&layer.forward(&x));

    let expected = [2.0, -3.0, 0.3, 0.9];
    for (actual, expected) in y.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "passthrough output {y:?} != expected {expected:?}"
        );
    }
}
