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
//
// Portions of this file are derived from turboquant_plus
// (https://github.com/TheTom/turboquant_plus), Copyright 2026 Tom Turney,
// licensed under the Apache License, Version 2.0. See the top-level NOTICE
// file for the attribution carried forward under Apache-2.0 Section 4(d).

//! Pre-device Turbo4 quantizer for byte and scale comparisons.
use super::*;
use crate::cache::turbo::codebook::nearest_centroid_indices_with_boundaries;

pub(crate) fn quantize(
    x: &MlxArray,
    params: &TurboQuantParams,
    signs1: &[f32],
    signs2: &[f32],
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let shape = ffi::array_shape(x);
    let [b, h, t, d]: [i32; 4] = shape.try_into().unwrap();
    assert_eq!(d as usize, signs1.len());
    assert_eq!(d as usize, signs2.len());
    let x = ffi::astype(x, dtype::FLOAT32);
    let norm = ffi::sqrt(&ffi::sum_axis(&ffi::multiply(&x, &x), -1, true));
    let zero = ffi::full_f32(&[1], 0.0, dtype::FLOAT32);
    let one = ffi::full_f32(&[1], 1.0, dtype::FLOAT32);
    let safe = ffi::where_cond(&ffi::greater(&norm, &zero), &norm, &one);
    let unit = ffi::divide(&x, &safe);
    let first = ffi::from_slice_f32(signs1, &[1, 1, 1, d]);
    let second = ffi::from_slice_f32(signs2, &[1, 1, 1, d]);
    let rotated = ffi::multiply(&wht(&ffi::multiply(&unit, &first)), &second);
    let floats = |a: &MlxArray| {
        ffi::eval(a);
        ffi::array_to_raw_bytes(a)
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let coords = floats(&rotated);
    let cb = &params.codebook;
    let indices =
        nearest_centroid_indices_with_boundaries(&coords, &cb.boundaries, cb.centroids.len());
    let packed: Vec<u8> = indices
        .chunks_exact(2)
        .map(|p| (p[0] | (p[1] << 4)) as u8)
        .collect();
    let rescale: Vec<f32> = floats(&norm)
        .iter()
        .zip(indices.chunks_exact(d as usize))
        .map(|(n, row)| {
            let mut sq = 0.0;
            for &i in row {
                let c = cb.centroids[i];
                sq += c * c;
            }
            n / sq.sqrt().max(1e-10)
        })
        .collect();
    (
        ffi::from_bytes(&packed, &[b, h, t, d / 2], dtype::UINT8),
        ffi::astype(&norm, dtype::FLOAT16),
        ffi::astype(
            &ffi::from_slice_f32(&rescale, &[b, h, t, 1]),
            dtype::FLOAT16,
        ),
    )
}
