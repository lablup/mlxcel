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

/// A purely local "decoder": each latent cell becomes an 8x8 block whose value
/// depends only on that cell, so a correct tiled blend equals the whole decode.
fn local_decode(latent: &MlxArray) -> Result<UniquePtr<MlxArray>> {
    let s = mlxcel_core::array_shape(latent);
    let (h, w) = (s[2], s[3]);
    // Mix channels into 3 outputs: c0 + 2 c1, c1 - c0, 0.5 c0.
    let c = |i: i32| mlxcel_core::slice(latent, &[0, i, 0, 0], &[1, i + 1, h, w]);
    let two = mlxcel_core::full_f32(&[1], 2.0, dtype::FLOAT32);
    let half = mlxcel_core::full_f32(&[1], 0.5, dtype::FLOAT32);
    let rgb = mlxcel_core::concatenate_many(
        &[
            &mlxcel_core::add(&c(0), &mlxcel_core::multiply(&c(1), &two)),
            &mlxcel_core::subtract(&c(1), &c(0)),
            &mlxcel_core::multiply(&c(0), &half),
        ],
        1,
    );
    // Nearest 8x upsample.
    let x = mlxcel_core::reshape(&rgb, &[1, 3, h, 1, w, 1]);
    let x = mlxcel_core::broadcast_to(&x, &[1, 3, h, 8, w, 8]);
    Ok(mlxcel_core::reshape(&x, &[1, 3, h * 8, w * 8]))
}

fn latent(h: i32, w: i32) -> UniquePtr<MlxArray> {
    let n = 2 * h * w;
    let values: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin()).collect();
    mlxcel_core::from_slice_f32(&values, &[1, 2, h, w])
}

#[test]
fn tiled_decode_equals_untiled_for_a_local_decoder() {
    // 64x48 latent cells = a 512x384 image; FLUX.2 tiles of 128 px (16 cells)
    // with an 8-cell (64 px) overlap.
    let z = latent(64, 48);
    let whole = mlxcel_core::utils::array_to_vec_f32(&local_decode(&z).unwrap());
    let config = TilingConfig::default();
    let tiled = decode_image_tiled(&z, &local_decode, (128, 128), (64, 64), 8).unwrap();
    assert_eq!(mlxcel_core::array_shape(&tiled), [1, 3, 512, 384]);
    let tiled = mlxcel_core::utils::array_to_vec_f32(&tiled);
    let max = whole
        .iter()
        .zip(&tiled)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max < 1e-3, "max abs diff {max}");
    assert!(config.enabled());
}

#[test]
fn a_latent_that_fits_one_tile_is_decoded_whole() {
    let z = latent(8, 8);
    let out = decode_image_tiled(&z, &local_decode, (128, 128), (64, 64), 8).unwrap();
    assert_eq!(mlxcel_core::array_shape(&out), [1, 3, 64, 64]);
}

#[test]
fn auto_policy_tiles_from_twice_the_tile_size() {
    assert_eq!(TilingConfig::resolve(255, 128, None), None);
    assert!(TilingConfig::resolve(256, 128, None).is_some());
    assert!(TilingConfig::resolve(512, 512, None).is_some());
    assert_eq!(TilingConfig::resolve(1024, 1024, Some(false)), None);
    assert!(TilingConfig::resolve(64, 64, Some(true)).is_some());
}

#[test]
fn cos_ramp_matches_numpy() {
    // np.float32(0.5 - 0.5 * np.cos(np.linspace(0, 1, 5, dtype=np.float32) * np.pi))
    let r = cos_ramp(5);
    let want = [0.0_f32, 0.146_446_6, 0.5, 0.853_553_4, 1.0];
    for (g, w) in r.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "{r:?}");
    }
    assert!(cos_ramp(0).is_empty());
}
