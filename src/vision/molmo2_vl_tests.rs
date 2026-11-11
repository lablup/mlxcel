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

//! The additive image-feature merge of Molmo2, on synthetic arrays.
//!
//! The shapes are small, the values are the ones that matter: the projector
//! output of `mlx-community/Molmo2-8B-4bit` reaches about 8.2e4 on two values,
//! past the `float16` maximum, and the checkpoint's text embeddings are
//! `float16`.

use super::{
    MAX_FLOAT16_IMAGE_FEATURE, clip_image_features, image_feature_clip_limit, merge_image_features,
};
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

const TOKENS: i32 = 5;
const HIDDEN: i32 = 4;
/// The largest projected value measured on the 8B checkpoint.
const OVER_RANGE: f32 = 81998.11;
/// `float16` spacing is 32 between 2^15 and 2^16, so 65000 lands on 64992.
const FLOAT16_OF_LIMIT: f32 = 64992.0;
/// `finfo(bfloat16).max`, (2 - 2^-7) * 2^127.
const BFLOAT16_MAX_BITS: u32 = 0x7F7F_0000;

fn array(values: &[f32], shape: &[i32], dtype: i32) -> UniquePtr<MlxArray> {
    let array = mlxcel_core::from_slice_f32(values, shape);
    if dtype == dtype::FLOAT32 {
        array
    } else {
        mlxcel_core::astype(&array, dtype)
    }
}

/// Text embeddings `[1, TOKENS, HIDDEN]`: small values, exact in every dtype,
/// with zeros where the tests add an out-of-range feature.
fn text_embeddings(dtype: i32) -> UniquePtr<MlxArray> {
    #[rustfmt::skip]
    let values = [
        0.5,  -0.25, 0.125, 0.75,
        0.25,  0.0,  0.5,   0.0,
        -0.5,  1.0,  0.25, -0.125,
        0.0,   0.0,  0.75,  0.5,
        1.5,  -1.0,  0.0,   0.25,
    ];
    array(&values, &[1, TOKENS, HIDDEN], dtype)
}

/// The merge as it was before the clamp: cast, then the same one-hot scatter.
fn merge_without_the_clamp(
    x: &MlxArray,
    image_features: &MlxArray,
    positions: &[i32],
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let txt_dtype = mlxcel_core::array_dtype(x);
    let image_features = mlxcel_core::astype(image_features, txt_dtype);
    let flat_x = mlxcel_core::reshape(x, &[-1, HIDDEN]);
    let rows: Vec<i32> = (0..TOKENS).collect();
    let row_arr = mlxcel_core::from_slice_i32(&rows, &[TOKENS, 1]);
    let pos_arr = mlxcel_core::from_slice_i32(positions, &[1, positions.len() as i32]);
    let one_hot = mlxcel_core::astype(&mlxcel_core::equal(&row_arr, &pos_arr), txt_dtype);
    let spread = mlxcel_core::matmul(&one_hot, &image_features);
    mlxcel_core::reshape(&mlxcel_core::add(&flat_x, &spread), &shape)
}

fn row(values: &[f32], index: usize) -> &[f32] {
    let hidden = HIDDEN as usize;
    &values[index * hidden..(index + 1) * hidden]
}

#[test]
fn the_clip_limit_is_mlx_vlms_per_dtype() {
    assert_eq!(MAX_FLOAT16_IMAGE_FEATURE, 65000.0);
    assert_eq!(image_feature_clip_limit(dtype::FLOAT16), 65000.0);
    assert_eq!(
        image_feature_clip_limit(dtype::BFLOAT16).to_bits(),
        BFLOAT16_MAX_BITS
    );
    assert_eq!(image_feature_clip_limit(dtype::FLOAT32), f32::MAX);
}

#[test]
fn a_feature_past_the_float16_maximum_stays_finite_and_local() {
    let x = text_embeddings(dtype::FLOAT16);
    #[rustfmt::skip]
    let features = array(
        &[
            1.0,  OVER_RANGE, -2.0,     0.5,
            3.0, -90000.0,     f32::NAN, 4.0,
        ],
        &[2, HIDDEN],
        dtype::FLOAT32,
    );
    let positions = [1, 3];
    let before = array_to_vec_f32(&x);

    // The trigger: the plain cast to the text dtype makes the two out-of-range
    // values infinite. The one-hot matmul of the scatter then multiplies them
    // by the zeros of every other row, which is how one value used to turn a
    // whole hidden dimension of the sequence into NaN.
    let cast = array_to_vec_f32(&mlxcel_core::astype(&features, dtype::FLOAT16));
    assert_eq!(cast[1], f32::INFINITY);
    assert_eq!(cast[5], f32::NEG_INFINITY);
    assert!(cast[6].is_nan());

    let merged = merge_image_features(&x, &features, &positions);
    assert_eq!(mlxcel_core::array_dtype(&merged), dtype::FLOAT16);
    assert_eq!(mlxcel_core::array_shape(&merged), [1, TOKENS, HIDDEN]);
    let merged = array_to_vec_f32(&merged);
    assert!(
        merged.iter().all(|value| value.is_finite()),
        "merged embeddings must be finite: {merged:?}"
    );

    // Rows that hold no image patch are the text embeddings, bit for bit.
    for token in [0, 2, 4] {
        assert_eq!(row(&merged, token), row(&before, token), "token {token}");
    }
    // Image rows: text + feature, with the feature clamped to the float16 limit
    // and NaN read as zero.
    assert_eq!(row(&merged, 1), [1.25, FLOAT16_OF_LIMIT, -1.5, 0.5]);
    assert_eq!(row(&merged, 3), [3.0, -FLOAT16_OF_LIMIT, 0.75, 4.5]);
}

#[test]
fn features_inside_the_range_merge_exactly_as_before() {
    #[rustfmt::skip]
    let in_range = [
        1.0,   -1000.5,  0.03125, 250.0,
        -7.25,  64000.0, 2.5,    -0.5,
    ];
    for text_dtype in [dtype::FLOAT16, dtype::BFLOAT16, dtype::FLOAT32] {
        let x = text_embeddings(text_dtype);
        let features = array(&in_range, &[2, HIDDEN], dtype::FLOAT32);
        let merged = merge_image_features(&x, &features, &[0, 4]);
        let unclamped = merge_without_the_clamp(&x, &features, &[0, 4]);
        assert_eq!(mlxcel_core::array_dtype(&merged), text_dtype);
        let merged = array_to_vec_f32(&merged);
        let unclamped = array_to_vec_f32(&unclamped);
        assert_eq!(
            merged.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            unclamped.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "text dtype {text_dtype}"
        );
    }
}

#[test]
fn wide_text_dtypes_keep_large_features_and_only_lose_non_finite_ones() {
    #[rustfmt::skip]
    let values = [
        1.0, OVER_RANGE,    -2.0,              0.5,
        3.0, f32::INFINITY,  f32::NEG_INFINITY, f32::NAN,
    ];
    for (text_dtype, limit) in [
        (dtype::BFLOAT16, f32::from_bits(BFLOAT16_MAX_BITS)),
        (dtype::FLOAT32, f32::MAX),
    ] {
        let features = array(&values, &[2, HIDDEN], dtype::FLOAT32);
        let clipped = clip_image_features(&features, text_dtype);
        assert_eq!(mlxcel_core::array_dtype(&clipped), dtype::FLOAT32);
        let clipped = array_to_vec_f32(&clipped);
        assert_eq!(
            clipped,
            [1.0, OVER_RANGE, -2.0, 0.5, 3.0, limit, -limit, 0.0],
            "text dtype {text_dtype}"
        );

        // Merged into zeros, the clamped features land in the text dtype finite.
        let zeros = array(&[0.0; 20], &[1, TOKENS, HIDDEN], text_dtype);
        let merged = array_to_vec_f32(&merge_image_features(&zeros, &features, &[1, 3]));
        assert!(merged.iter().all(|value| value.is_finite()), "{merged:?}");
        assert_eq!(row(&merged, 3)[1], limit);
        assert_eq!(row(&merged, 3)[2], -limit);
        assert_eq!(row(&merged, 3)[3], 0.0);
    }
}

#[test]
fn float16_features_are_clamped_in_their_own_dtype() {
    // A tower that already ran in float16 hands over +inf where it overflowed;
    // the clamp brings it back to the float16 limit whatever the text dtype is.
    let features = array(
        &[1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
        &[1, HIDDEN],
        dtype::FLOAT16,
    );
    for text_dtype in [dtype::FLOAT16, dtype::FLOAT32] {
        let clipped = clip_image_features(&features, text_dtype);
        assert_eq!(mlxcel_core::array_dtype(&clipped), dtype::FLOAT16);
        assert_eq!(
            array_to_vec_f32(&clipped),
            [1.0, FLOAT16_OF_LIMIT, -FLOAT16_OF_LIMIT, 0.0],
            "text dtype {text_dtype}"
        );
    }
}
