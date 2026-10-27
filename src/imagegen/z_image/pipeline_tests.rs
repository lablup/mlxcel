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
use crate::imagegen::scheduler::FlowMatchEulerDiscrete;

fn values(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(a, dtype::FLOAT32))
}

#[test]
fn noise_has_the_latent_shape_and_depends_only_on_the_seed() {
    let a = initial_noise(42, 16, 512, 256);
    assert_eq!(mlxcel_core::array_shape(&a), [1, 16, 1, 32, 64]);
    assert_eq!(mlxcel_core::array_dtype(&a), dtype::FLOAT32);
    let b = initial_noise(42, 16, 512, 256);
    let c = initial_noise(43, 16, 512, 256);
    assert_eq!(values(&a), values(&b));
    assert_ne!(values(&a), values(&c));
}

#[test]
fn noised_reference_mixes_with_the_start_sigma() {
    let schedule = FlowMatchEulerDiscrete::static_shift(8, 3.0).unwrap();
    let image = mlxcel_core::full_f32(&[1, 2, 1, 2, 2], 0.5, dtype::FLOAT32);
    let start = 3;
    let mixed = noised_reference(&image, &schedule, start, 7).unwrap();
    let noise = noise_like(7, &[1, 2, 1, 2, 2]);
    let sigma = schedule.sigmas()[start];
    for (m, n) in values(&mixed).iter().zip(values(&noise)) {
        let want = (1.0 - sigma) * 0.5 + sigma * n;
        assert!((m - want).abs() < 1e-6, "{m} vs {want}");
    }
    assert!(noised_reference(&image, &schedule, 8, 7).is_err());
}

#[test]
fn reference_pixels_map_bytes_to_unit_range_in_nchw() {
    let mut img = image::RgbImage::new(16, 16);
    img.put_pixel(1, 0, image::Rgb([0, 255, 51]));
    let pixels = reference_pixels(&image::DynamicImage::ImageRgb8(img.clone()), 16, 16);
    assert_eq!(mlxcel_core::array_shape(&pixels), [1, 3, 16, 16]);
    let v = values(&pixels);
    // (channel, y=0, x=1) of a 16x16 plane.
    assert_eq!(v[1], -1.0);
    assert_eq!(v[256 + 1], 1.0);
    assert!((v[512 + 1] - (51.0 / 127.5 - 1.0)).abs() < 1e-7);
    // A different target size resizes.
    let resized = reference_pixels(&image::DynamicImage::ImageRgb8(img), 32, 16);
    assert_eq!(mlxcel_core::array_shape(&resized), [1, 3, 16, 32]);
}
