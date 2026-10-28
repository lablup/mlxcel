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
use crate::imagegen::llada_image::conditioning::tests::host;

#[test]
fn draws_follow_the_global_key_sequence() {
    // `mx.random.seed(s); mx.random.normal(shape)` draws from
    // `split(key(s))[1]` and keeps `split(key(s))[0]` as the state.
    let shape = [1, 4, 8];
    let normal = |key: &MlxArray| {
        // SAFETY: `key` is alive for the duration of the call.
        let draw = unsafe { mlxcel_core::random_normal(&shape, dtype::FLOAT32, key) };
        host(&draw)
    };
    let row = |keys: &MlxArray, r: i32| {
        mlxcel_core::reshape(&mlxcel_core::slice(keys, &[r, 0], &[r + 1, 2]), &[2])
    };
    let first = mlxcel_core::random_split_key(&mlxcel_core::random_key(42), 2);
    let second = mlxcel_core::random_split_key(&row(&first, 0), 2);
    let mut rng = KeySequence::new(42);
    assert_eq!(host(&rng.normal_f32(&shape)), normal(&row(&first, 1)));
    assert_eq!(host(&rng.normal_f32(&shape)), normal(&row(&second, 1)));
}

#[test]
fn pixels_normalize_and_halve() {
    let rgb: Vec<u8> = (0..4 * 4 * 3).map(|v| (v * 5) as u8).collect();
    let pixels = normalized_pixels(&rgb, 4, 4);
    let values = host(&pixels);
    assert!((values[0] + 1.0).abs() < 1e-7);
    assert!((values[3] - (15.0 / 127.5 - 1.0)).abs() < 1e-6);
    let half = half_resolution(&pixels);
    assert_eq!(mlxcel_core::array_shape(&half), vec![1, 2, 2, 3]);
    // Top-left output, red: pixels (0,0) (0,1) (1,0) (1,1) are values 0, 15, 60, 75.
    let want = ((0.0 + 15.0 + 60.0 + 75.0) / 4.0) / 127.5 - 1.0;
    assert!((host(&half)[0] - want as f32).abs() < 1e-6);
}

#[test]
fn latent_shape_is_one_sixteenth() {
    assert_eq!(latent_shape(128, 1024, 768), [1, 128, 48, 64]);
}
