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

//! Tests for the shared interpolation matrices and the two position-grid
//! resample modes (#2388).
//!
//! The antialiased-bilinear references come from
//! `tests/fixtures/lfm2_vl/reference.json`, generated out of band by
//! `tests/fixtures/lfm2_vl/generate_reference.py` with PyTorch's
//! `F.interpolate` and mlx-vlm's LFM2-VL tower code (the versions are recorded
//! in the file). The antialiased-bicubic references come from
//! `tests/fixtures/deepseekocr_sam/reference.json`, generated the same way
//! with torch's `F.interpolate(mode="bicubic", antialias=True)` and mlx-vlm's
//! `resize_bicubic_nhwc` (issue #2465). The MoonViT bicubic matrices are
//! pinned bit for bit from the implementation before the refactor.

use super::{ResampleKernel, interp_matrix};
use crate::vision::encoders::deepseekocr_sam::bicubic_resample;
use crate::vision::encoders::kimi_vl::pos_emb::{
    Learnable2DInterpPosEmb, PosEmbResample, bicubic_interp_matrix,
};

const REFERENCE: &str = include_str!("../../../tests/fixtures/lfm2_vl/reference.json");

fn reference() -> serde_json::Value {
    serde_json::from_str(REFERENCE).expect("reference.json parses")
}

fn floats(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_f64().expect("number") as f32)
        .collect()
}

/// The antialiased bicubic matrices and grids in
/// `tests/fixtures/deepseekocr_sam/reference.json`, generated out of band by
/// `tests/fixtures/deepseekocr_sam/generate_reference.py` with torch's
/// `F.interpolate(mode="bicubic", antialias=True)` and mlx-vlm 0.7.8's
/// `resize_bicubic_nhwc` (the versions are recorded in the file). `Cubic` must
/// pick ATen's antialiased coefficient `a = -0.5` (issue #2465).
const SAM_REFERENCE: &str = include_str!("../../../tests/fixtures/deepseekocr_sam/reference.json");

fn sam_reference() -> serde_json::Value {
    serde_json::from_str(SAM_REFERENCE).expect("reference.json parses")
}

#[test]
fn bicubic_antialias_matrix_matches_pytorch() {
    for m in sam_reference()["matrices"].as_array().expect("matrices") {
        let (input, output) = (
            m["in"].as_i64().unwrap() as i32,
            m["out"].as_i64().unwrap() as i32,
        );
        let want = floats(&m["values"]);
        let got = interp_matrix(input, output, ResampleKernel::Cubic, true);
        assert_eq!(got.len(), want.len());
        for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-6,
                "{input} -> {output} entry {idx}: got {g}, PyTorch {w}"
            );
        }
    }
}

/// The real DeepSeek-OCR SAM preprocessing path resamples its position grid
/// with `bicubic_resample(..., antialias = true)`; it must reproduce torch's
/// direct 2-D antialiased bicubic (issue #2465).
#[test]
fn sam_grid_resample_matches_torch() {
    let side = sam_reference()["side"].as_i64().unwrap() as i32;
    let dim = sam_reference()["dim"].as_i64().unwrap() as i32;
    let table = mlxcel_core::from_slice_f32(&fixture_table(side, dim), &[side, side, dim]);
    for grid in sam_reference()["grids"].as_array().expect("grids") {
        let (h, w) = (
            grid["h"].as_i64().unwrap() as i32,
            grid["w"].as_i64().unwrap() as i32,
        );
        let want = floats(&grid["values"]);
        let got = bicubic_resample(&table, side, side, dim, h, w, true);
        let got = mlxcel_core::utils::array_to_vec_f32(&got);
        assert_eq!(got.len(), want.len(), "{h}x{w}");
        for (idx, (g, r)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - r).abs() <= 1e-5,
                "{h}x{w} entry {idx}: got {g}, torch {r}"
            );
        }
    }
}

/// The non-antialiased cubic matrix, as the pre-#2465 implementation built it:
/// ATen's non-AA upsample path and MoonViT's edge-replicated grid keep Keys'
/// `a = -0.75`, so the `antialias = false` output must stay bit for bit.
const NONANTIALIASED_CUBIC_BITS: &[(i32, i32, &[u32])] = &[
    (
        16,
        5,
        &[
            0xbd78d588, 0x3f7a8f64, 0x3db5c29e, 0xbbdd3007, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xbde1cac9, 0x3f54ccd2,
            0x3ea70a4f, 0xbd418a08, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0xbdc00000, 0x3f180000, 0x3f180000, 0xbdc00000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0xbd418a08, 0x3ea70a4f, 0x3f54ccd2, 0xbde1cac9, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xbbdd3000,
            0x3db5c238, 0x3f7a8f60, 0xbd78d480,
        ],
    ),
    (
        16,
        8,
        &[
            0x3f0af8b0, 0x3f0af8b0, 0xbdaf8af9, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0xbdc00000, 0x3f180000, 0x3f180000, 0xbdc00000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0xbdc00000, 0x3f180000, 0x3f180000, 0xbdc00000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xbdc00000, 0x3f180000, 0x3f180000,
            0xbdc00000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0xbdc00000, 0x3f180000, 0x3f180000, 0xbdc00000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xbdc00000, 0x3f180000,
            0x3f180000, 0xbdc00000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0xbdc00000, 0x3f180000, 0x3f180000, 0xbdc00000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xbdaf8af9,
            0x3f0af8b0, 0x3f0af8b0,
        ],
    ),
];

/// A same-size resample is the identity whatever the coefficient: the taps sit
/// exactly on input pixels, where both `a` values weight the pixel itself 1.
#[test]
fn nonantialiased_cubic_matrix_is_unchanged() {
    for &(input, output, bits) in NONANTIALIASED_CUBIC_BITS {
        let got: Vec<u32> = interp_matrix(input, output, ResampleKernel::Cubic, false)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got, bits, "{input} -> {output}");
    }
    for antialias in [false, true] {
        let m = interp_matrix(9, 9, ResampleKernel::Cubic, antialias);
        for o in 0..9 {
            for i in 0..9 {
                let want = if o == i { 1.0 } else { 0.0 };
                assert_eq!(m[(o * 9 + i) as usize], want);
            }
        }
    }
}

#[test]
fn bilinear_antialias_matrix_matches_pytorch() {
    let reference = reference();
    let matrices = reference["matrices"].as_array().expect("matrices");
    let outs: Vec<i64> = matrices
        .iter()
        .map(|m| m["out"].as_i64().unwrap())
        .collect();
    // The issue's two shapes, a 16 -> 8 downsample and a 16 -> 24 upsample,
    // must be among them.
    assert!(outs.contains(&8) && outs.contains(&24), "{outs:?}");
    for m in matrices {
        let (input, output) = (
            m["in"].as_i64().unwrap() as i32,
            m["out"].as_i64().unwrap() as i32,
        );
        let want = floats(&m["values"]);
        let got = interp_matrix(input, output, ResampleKernel::Triangle, true);
        assert_eq!(got.len(), want.len());
        for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-6,
                "{input} -> {output} entry {idx}: got {g}, PyTorch {w}"
            );
        }
    }
}

#[test]
fn triangle_rows_sum_to_one_and_same_size_is_identity() {
    for (input, output) in [(16, 5), (16, 16), (7, 31), (40, 3)] {
        let m = interp_matrix(input, output, ResampleKernel::Triangle, true);
        for row in 0..output {
            let s: f32 = (0..input).map(|k| m[(row * input + k) as usize]).sum();
            assert!(
                (s - 1.0).abs() < 1e-5,
                "{input} -> {output} row {row} sums to {s}"
            );
        }
    }
    let m = interp_matrix(9, 9, ResampleKernel::Triangle, true);
    for o in 0..9 {
        for i in 0..9 {
            let want = if o == i { 1.0 } else { 0.0 };
            assert_eq!(m[(o * 9 + i) as usize], want);
        }
    }
}

/// MoonViT's edge-replicated bicubic matrices, as the pre-#2388
/// `kimi_vl_pos_emb::interp_matrix` built them. `Bicubic` must stay bit for bit.
const KIMI_BICUBIC_BITS: &[(i32, i32, &[u32])] = &[
    (
        4,
        6,
        &[
            0x3f8b1c72, 0xbdb1c720, 0x00000000, 0x00000000, 0x3f000000, 0x3f180000, 0xbdc00000,
            0x00000000, 0xbdb1c700, 0x3f717b42, 0x3e24bda8, 0xbc8e3b00, 0xbc8e3900, 0x3e24bd94,
            0x3f717b44, 0xbdb1c700, 0x00000000, 0xbdc00000, 0x3f180000, 0x3f000000, 0x00000000,
            0x00000000, 0xbdb1c700, 0x3f8b1c6a,
        ],
    ),
    (
        6,
        4,
        &[
            0x3f460000, 0x3e860000, 0xbd100000, 0x00000000, 0x00000000, 0x00000000, 0xbd100000,
            0x3e860000, 0x3f610000, 0xbdd80000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0xbdd80000, 0x3f610000, 0x3e860000, 0xbd100000, 0x00000000, 0x00000000, 0x00000000,
            0xbd100000, 0x3e860000, 0x3f460000,
        ],
    ),
    (
        16,
        9,
        &[
            0x3f1fd30e, 0x3ee3d744, 0xbd8df580, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0xbdb1c700, 0x3f717b42, 0x3e24bda8, 0xbc8e3b00,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xbb0f4000,
            0x3d3cc020, 0x3f7e46f0, 0xbd183b00, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0xbd2b3100, 0x3e98355e, 0x3f5a6a0f, 0xbdde8d40,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0xbdc00000, 0x3f180000, 0x3f180000, 0xbdc00000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xbdde8d40, 0x3f5a6a0f, 0x3e98355e,
            0xbd2b3100, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0xbd183b00, 0x3f7e46f0, 0x3d3cc020, 0xbb0f4000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xbc8e3700, 0x3e24bdb8, 0x3f717b3f,
            0xbdb1c740, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0xbd8df580, 0x3ee3d72c, 0x3f1fd310,
        ],
    ),
    (
        5,
        5,
        &[
            0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x3f800000,
        ],
    ),
];

#[test]
fn moonvit_bicubic_matrix_is_bit_identical() {
    for &(input, output, bits) in KIMI_BICUBIC_BITS {
        let got: Vec<u32> = bicubic_interp_matrix(input, output)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got, bits, "{input} -> {output}");
    }
}

/// The deterministic `[16, 16, 4]` table the fixture generator resampled.
fn fixture_table(side: i32, dim: i32) -> Vec<f32> {
    let mut table = Vec::with_capacity((side * side * dim) as usize);
    for i in 0..side {
        for j in 0..side {
            for d in 0..dim {
                let (i, j, d) = (f64::from(i), f64::from(j), f64::from(d));
                table.push(((0.37 * i + 0.11 * j * (d + 1.0)).sin() + 0.05 * d) as f32);
            }
        }
    }
    table
}

#[test]
fn lfm2_position_grid_matches_mlx_vlm() {
    let reference = reference();
    let side = reference["side"].as_i64().unwrap() as i32;
    let dim = reference["dim"].as_i64().unwrap() as i32;
    let table = mlxcel_core::from_slice_f32(&fixture_table(side, dim), &[side, side, dim]);
    // The mode the LFM2-VL tower actually constructs.
    let pos = Learnable2DInterpPosEmb::from_array(table, side, side, dim)
        .with_resample(crate::vision::encoders::lfm2_vl::POSITION_RESAMPLE);
    for grid in reference["grids"].as_array().expect("grids") {
        let (h, w) = (
            grid["h"].as_i64().unwrap() as i32,
            grid["w"].as_i64().unwrap() as i32,
        );
        let want = floats(&grid["values"]);
        let got = mlxcel_core::utils::array_to_vec_f32(&pos.pos_for(h, w));
        assert_eq!(got.len(), want.len(), "{h}x{w}");
        for (idx, (g, r)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - r).abs() <= 1e-5,
                "{h}x{w} entry {idx}: got {g}, mlx-vlm {r}"
            );
        }
    }
}

#[test]
fn bilinear_antialias_returns_the_table_dtype() {
    let table = mlxcel_core::from_slice_f32(&fixture_table(4, 2), &[4, 4, 2]);
    let table = mlxcel_core::astype(&table, mlxcel_core::dtype::BFLOAT16);
    let pos = Learnable2DInterpPosEmb::from_array(table, 4, 4, 2)
        .with_resample(PosEmbResample::BilinearAntialias);
    let out = pos.pos_for(3, 6);
    assert_eq!(mlxcel_core::array_shape(&out), vec![18, 2]);
    assert_eq!(mlxcel_core::array_dtype(&out), mlxcel_core::dtype::BFLOAT16);
}

#[test]
fn the_two_modes_differ_off_the_learned_grid() {
    let table = mlxcel_core::from_slice_f32(&fixture_table(16, 4), &[16, 16, 4]);
    let bilinear = Learnable2DInterpPosEmb::from_array(mlxcel_core::copy(&table), 16, 16, 4)
        .with_resample(PosEmbResample::BilinearAntialias);
    let bicubic = Learnable2DInterpPosEmb::from_array(table, 16, 16, 4);
    let a = mlxcel_core::utils::array_to_vec_f32(&bilinear.pos_for(28, 28));
    let b = mlxcel_core::utils::array_to_vec_f32(&bicubic.pos_for(28, 28));
    let max_diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff > 1e-3, "modes coincide: {max_diff}");
    // On the learned grid both take the fast path.
    let a = mlxcel_core::utils::array_to_vec_f32(&bilinear.pos_for(16, 16));
    let b = mlxcel_core::utils::array_to_vec_f32(&bicubic.pos_for(16, 16));
    assert_eq!(a, b);
}
