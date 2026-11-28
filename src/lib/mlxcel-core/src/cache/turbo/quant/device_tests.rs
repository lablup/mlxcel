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

use super::*;

fn bytes(a: &MlxArray) -> Vec<u8> {
    ffi::eval(a);
    ffi::array_to_raw_bytes(a)
}

#[test]
fn device_rows_match_host_bytes_norms_and_scales() {
    let mut tested_scales = 0;
    let mut rounded_differently = 0;
    let mut max_relative = 0.0_f32;
    for d in [64, 128, 256] {
        let params = TurboQuantParams::new(d, 0xfedc_4321);
        for t in [1usize, 4096] {
            let mut state = 42u32;
            let mut data: Vec<_> = (0..t * d as usize)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    (state >> 8) as f32 / 8388608.0 - 1.0
                })
                .collect();
            if t > 1 {
                data[..d as usize].fill(0.0);
                data[d as usize..2 * d as usize].fill(0.00001);
                data[2 * d as usize..3 * d as usize].fill(1024.0);
            }
            let x = ffi::from_slice_f32(&data, &[1, 1, t as i32, d as i32]);
            for (s1, s2) in [
                (&params.signs1, &params.signs2),
                (&params.k_signs1, &params.k_signs2),
            ] {
                let (p, n, r) = quantize_into_packed(&x, &params, s1, s2);
                let (wp, wn, wr) = host_reference::quantize(&x, &params, s1, s2);
                assert_eq!(ffi::array_shape(&p), [1, 1, t as i32, d as i32 / 2]);
                assert_eq!(ffi::array_dtype(&p), dtype::UINT8);
                assert_eq!(ffi::array_dtype(&n), dtype::FLOAT16);
                assert_eq!(ffi::array_dtype(&r), dtype::FLOAT16);
                assert!(bytes(&p) == bytes(&wp), "packed d={d} t={t}");
                assert!(bytes(&n) == bytes(&wn), "norms d={d} t={t}");
                let floats = |a: &MlxArray| {
                    bytes(&ffi::astype(a, dtype::FLOAT32))
                        .chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                        .collect::<Vec<_>>()
                };
                let actual = floats(&r);
                let expected = floats(&wr);
                assert_eq!(actual.len(), expected.len());
                for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
                    tested_scales += 1;
                    rounded_differently += usize::from(*a != b);
                    if b != 0.0 {
                        max_relative = max_relative.max((a - b).abs() / b.abs());
                    }
                    assert!(
                        (a - b).abs() <= 1e-3 * b.abs(),
                        "rescale d={d} t={t} i={i}: {a} vs {b}"
                    );
                }
            }
        }
    }
    eprintln!(
        "Turbo4 K/V rescale: {rounded_differently}/{tested_scales} rounded differently, maximum relative error {max_relative:e}"
    );
}

#[test]
fn device_pack_handles_boundary_ties_and_extremes() {
    for d in [64, 128, 256] {
        let params = TurboQuantParams::new(d, 42);
        let cb = &params.codebook;
        let mut coords = Vec::new();
        for &b in cb.boundaries.iter() {
            coords.extend([b.next_down(), b, b.next_up()]);
        }
        coords.extend([
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            f32::MIN,
            f32::MAX,
            -0.0,
            0.0,
        ]);
        coords.resize(d as usize, 0.0);
        let idx = crate::cache::turbo::codebook::nearest_centroid_indices_with_boundaries(
            &coords,
            &cb.boundaries,
            16,
        );
        let expected: Vec<_> = idx
            .chunks_exact(2)
            .map(|p| (p[0] | (p[1] << 4)) as u8)
            .collect();
        for t in [1, 4096] {
            let unit = ffi::from_slice_f32(&coords.repeat(t), &[1, 1, t as i32, d as i32]);
            let (packed, rescale) = pack_rotated_turbo4(
                &unit,
                &ffi::zeros(&[1, 1, t as i32, 1], dtype::FLOAT32),
                &ffi::from_slice_f32(&cb.centroids, &[16]),
                &ffi::from_slice_f32(&cb.boundaries, &[15]),
            );
            assert!(
                bytes(&packed) == expected.repeat(t),
                "boundary indices d={d}, t={t}"
            );
            assert!(bytes(&rescale).iter().all(|&b| b == 0));
        }
    }
}

#[test]
fn device_quantize_empty_rows_have_the_stored_shapes() {
    let params = TurboQuantParams::new(64, 42);
    for [b, h, t] in [[1, 1, 0], [1, 0, 1], [0, 1, 1]] {
        let x = ffi::zeros(&[b, h, t, 64], dtype::FLOAT16);
        let (packed, norms, rescale) = quantize_v_turbo4(&x, &params);
        for (a, width) in [(packed, 32), (norms, 1), (rescale, 1)] {
            assert_eq!(ffi::array_shape(&a), [b, h, t, width]);
            assert!(bytes(&a).is_empty());
        }
    }
}

#[test]
fn turbo4_quantize_stays_on_device() {
    let sources = [
        (include_str!("../quant.rs"), "fn quantize_into_packed("),
        (include_str!("../quant.rs"), "fn pack_rotated_turbo4("),
        (
            include_str!("../codebook.rs"),
            "fn boundary_indices_on_device(",
        ),
    ];
    for (source, signature) in sources {
        let (_, rest) = source.split_once(signature).expect("quantizer function");
        let (body, _) = rest.split_once("\n}").expect("end of function");
        for forbidden in ["ffi::eval", "array_to_raw_bytes"] {
            assert!(
                !body.contains(forbidden),
                "{signature} contains {forbidden}"
            );
        }
    }
}

#[test]
fn all_zero_decode_token_keeps_the_zero_region_and_zero_sidecars() {
    for d in [64, 128, 256] {
        let params = TurboQuantParams::new(d, 42);
        let x = ffi::zeros(&[1, 1, 1, d as i32], dtype::FLOAT16);
        let (p, n, r) = quantize_v_turbo4(&x, &params);
        let zero_index = params.codebook.boundaries.partition_point(|&v| v < 0.0) as u8;
        assert_eq!(bytes(&p), vec![zero_index * 17; d as usize / 2]);
        assert_eq!(bytes(&n), [0, 0]);
        assert_eq!(bytes(&r), [0, 0]);
    }
}
