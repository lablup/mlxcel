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

//! Qwen2.5-VL vision encoder tests.
//!
//! The patch-embedding layout normalizer is tested here rather than in a
//! consumer's module because it guards the encoder's own contract: both the
//! `mlxcel generate` loader and ColQwen2.5 call it with their own prefix, and
//! the detection rule should be stated once.
//!
//! Every test that drives MLX takes
//! [`crate::models::embedding_test_support::mlx_test_guard`].

use mlxcel_core::weights::WeightMap;

use super::{invert_window_index, normalize_patch_embed_layout, window_index_for_grid};
use crate::models::embedding_test_support::{mlx_test_guard, to_vec};

/// A `[out, a, b, c, d]` tensor filled with `0..count`, so a permutation is
/// distinguishable from a reinterpretation of the same buffer.
fn ramp(shape: &[i32]) -> (mlxcel_core::UniquePtr<mlxcel_core::MlxArray>, usize) {
    let count: i32 = shape.iter().product();
    let values: Vec<f32> = (0..count).map(|i| i as f32).collect();
    (mlxcel_core::from_slice_f32(&values, shape), count as usize)
}

#[test]
fn patch_embed_layout_is_converted_from_the_pytorch_conv3d_form() {
    let _guard = mlx_test_guard();
    // Raw HuggingFace: [out, in, kT, kH, kW]. The encoder wants the mlx
    // conversion's [out, kT, kH, kW, in], so this must be permuted.
    let (out, channels, kt, k) = (4i32, 3i32, 2i32, 2i32);
    let (weight, count) = ramp(&[out, channels, kt, k, k]);
    let mut weights = WeightMap::new();
    weights.insert("vision_tower.patch_embed.proj.weight".to_string(), weight);

    assert!(normalize_patch_embed_layout(
        &mut weights,
        "vision_tower",
        3
    ));
    let converted = &weights["vision_tower.patch_embed.proj.weight"];
    assert_eq!(
        mlxcel_core::array_shape(converted),
        vec![out, kt, k, k, channels]
    );
    // Element count is preserved and the permutation is the transpose, not a
    // reinterpretation: element [0, 0, 0, 0, 1] of the output is element
    // [0, 1, 0, 0, 0] of the input, which is `kt * k * k` = 8.
    mlxcel_core::eval(converted);
    let flat = to_vec(converted);
    assert_eq!(flat.len(), count);
    assert_eq!(flat[1], 8.0);

    // Idempotent: a second call sees the converted layout and declines.
    assert!(!normalize_patch_embed_layout(
        &mut weights,
        "vision_tower",
        3
    ));
    assert_eq!(
        mlxcel_core::array_shape(&weights["vision_tower.patch_embed.proj.weight"]),
        vec![out, kt, k, k, channels]
    );
}

#[test]
fn patch_embed_layout_leaves_the_mlx_conversion_untouched() {
    let _guard = mlx_test_guard();
    let (out, channels, kt, k) = (4i32, 3i32, 2i32, 2i32);
    let (weight, count) = ramp(&[out, kt, k, k, channels]);
    let before = {
        mlxcel_core::eval(&weight);
        to_vec(&weight)
    };
    let mut weights = WeightMap::new();
    weights.insert("vision_tower.patch_embed.proj.weight".to_string(), weight);

    assert!(!normalize_patch_embed_layout(
        &mut weights,
        "vision_tower",
        3
    ));
    let untouched = &weights["vision_tower.patch_embed.proj.weight"];
    assert_eq!(
        mlxcel_core::array_shape(untouched),
        vec![out, kt, k, k, channels]
    );
    mlxcel_core::eval(untouched);
    let after = to_vec(untouched);
    assert_eq!(after.len(), count);
    assert_eq!(before, after, "an mlx conversion must load bit-identically");
}

#[test]
fn patch_embed_layout_normalizer_is_prefix_parameterized() {
    let _guard = mlx_test_guard();
    let (out, channels, kt, k) = (4i32, 3i32, 2i32, 2i32);
    let (weight, _) = ramp(&[out, channels, kt, k, k]);
    let mut weights = WeightMap::new();
    // ColQwen2.5 and the generation loader both pass "vision_tower" today, but
    // the prefix is an argument, so a tower stored anywhere else converts too
    // and a tower under a different prefix is not touched by mistake.
    weights.insert("visual.patch_embed.proj.weight".to_string(), weight);

    assert!(!normalize_patch_embed_layout(
        &mut weights,
        "vision_tower",
        3
    ));
    assert_eq!(
        mlxcel_core::array_shape(&weights["visual.patch_embed.proj.weight"]),
        vec![out, channels, kt, k, k]
    );

    assert!(normalize_patch_embed_layout(&mut weights, "visual", 3));
    assert_eq!(
        mlxcel_core::array_shape(&weights["visual.patch_embed.proj.weight"]),
        vec![out, kt, k, k, channels]
    );
}

#[test]
fn patch_embed_layout_passes_through_shapes_it_cannot_classify() {
    let _guard = mlx_test_guard();
    // A checkpoint without the tower is not an error: the encoder reports the
    // missing key itself.
    let mut empty = WeightMap::new();
    assert!(!normalize_patch_embed_layout(&mut empty, "vision_tower", 3));

    // An already-flattened 2-D weight is the encoder's other accepted form.
    let (flat, _) = ramp(&[4, 3 * 2 * 2 * 2]);
    let mut flattened = WeightMap::new();
    flattened.insert("vision_tower.patch_embed.proj.weight".to_string(), flat);
    assert!(!normalize_patch_embed_layout(
        &mut flattened,
        "vision_tower",
        3
    ));
    assert_eq!(
        mlxcel_core::array_shape(&flattened["vision_tower.patch_embed.proj.weight"]),
        vec![4, 24]
    );

    // Both candidate axes carrying `in_channels` makes the two layouts
    // indistinguishable from shape alone, so the normalizer refuses to guess.
    let (ambiguous, _) = ramp(&[4, 3, 2, 3, 3]);
    let mut ambiguous_map = WeightMap::new();
    ambiguous_map.insert(
        "vision_tower.patch_embed.proj.weight".to_string(),
        ambiguous,
    );
    assert!(!normalize_patch_embed_layout(
        &mut ambiguous_map,
        "vision_tower",
        3
    ));
    assert_eq!(
        mlxcel_core::array_shape(&ambiguous_map["vision_tower.patch_embed.proj.weight"]),
        vec![4, 3, 2, 3, 3]
    );
}

/// The tower's own configuration: `window_size / spatial_merge_size /
/// patch_size` is `112 / 2 / 14 = 4`, so a merged grid is cut into 4x4 windows.
const SPATIAL_MERGE_SIZE: usize = 2;
const WINDOW_SIZE: usize = 112;
const PATCH_SIZE: usize = 14;

fn window_index_for(grid_thw: &[(i32, i32, i32)]) -> Vec<i32> {
    window_index_for_grid(grid_thw, SPATIAL_MERGE_SIZE, WINDOW_SIZE, PATCH_SIZE).0
}

fn is_identity(permutation: &[i32]) -> bool {
    permutation.iter().enumerate().all(|(i, &v)| v == i as i32)
}

/// `window_index` composed with its inverse, in the order the encoder applies
/// them: the tower reads token `window_index[rank]` at `rank`, then the
/// un-reorder puts `rank` back at `window_index[rank]`.
fn round_trip(window_index: &[i32]) -> Vec<i32> {
    let inverse = invert_window_index(window_index);
    window_index.iter().map(|&v| inverse[v as usize]).collect()
}

#[test]
fn window_index_inverse_is_argsort_not_the_permutation_itself() {
    // The smallest permutation that is not its own inverse. Writing
    // `out[rank] = window_index[rank]` would return `[2, 0, 1]` unchanged.
    assert_eq!(invert_window_index(&[2, 0, 1]), vec![1, 2, 0]);
    assert!(is_identity(&round_trip(&[2, 0, 1])));

    // An involution is its own inverse, which is the case that hid the defect.
    assert_eq!(invert_window_index(&[1, 0, 3, 2]), vec![1, 0, 3, 2]);
}

#[test]
fn window_index_inverse_restores_raster_order_on_a_non_involution_grid() {
    // 336x336 gives a 24x24 patch grid and a 12x12 merged grid, which is 3x3
    // windows of 4x4 merged tokens. That permutation is not an involution, so
    // applying `window_index` twice does not restore raster order.
    let window_index = window_index_for(&[(1, 24, 24)]);
    assert_eq!(window_index.len(), 144);
    let applied_twice: Vec<i32> = window_index
        .iter()
        .map(|&v| window_index[v as usize])
        .collect();
    assert!(
        !is_identity(&applied_twice),
        "a (1, 24, 24) grid must exercise the non-involution case"
    );
    let misplaced = applied_twice
        .iter()
        .enumerate()
        .filter(|&(i, &v)| v != i as i32)
        .count();
    assert_eq!(
        misplaced, 120,
        "the pre-#1596 un-reorder misplaced this many merged tokens at 336x336"
    );

    assert!(is_identity(&round_trip(&window_index)));
}

#[test]
fn window_index_inverse_is_the_identity_case_that_hid_the_defect() {
    // 448x448 gives a 32x32 patch grid and a 16x16 merged grid, which is 4x4
    // windows of 4x4 merged tokens. Here the permutation IS an involution, so
    // the incorrect un-reorder produced correct output and the defect stayed
    // invisible at the size issue #1596 was first measured at.
    let window_index = window_index_for(&[(1, 32, 32)]);
    assert_eq!(window_index.len(), 256);
    let applied_twice: Vec<i32> = window_index
        .iter()
        .map(|&v| window_index[v as usize])
        .collect();
    assert!(is_identity(&applied_twice));
    assert!(is_identity(&round_trip(&window_index)));
}

#[test]
fn window_index_inverse_round_trips_a_padded_and_a_multi_image_grid() {
    // A merged grid that is not a multiple of the 4-token window (37x37 merged
    // from a 74x74 patch grid) pads the last window, so `window_index` is still
    // a permutation of the unpadded tokens only.
    let padded = window_index_for(&[(1, 74, 74)]);
    assert_eq!(padded.len(), 37 * 37);
    assert!(is_identity(&round_trip(&padded)));

    // Two images in one prompt: `window_index_id` offsets the second image's
    // indices, so the concatenation is one permutation over both.
    let two_images = window_index_for(&[(1, 32, 32), (1, 16, 32)]);
    assert_eq!(two_images.len(), 16 * 16 + 8 * 16);
    let mut sorted = two_images.clone();
    sorted.sort_unstable();
    assert!(is_identity(&sorted), "must be a permutation of 0..n");
    assert!(is_identity(&round_trip(&two_images)));
}

/// A row whose squared sum crosses the float16 ceiling of 65504.
///
/// `rms_norm` reduces `sum(x * x)` over the feature axis. In float16 that sum
/// saturates to infinity and the norm returns `rsqrt(inf) = 0`, erasing the
/// row. Qwen2.5-VL's tower reaches this on real images: on a 448x448 input one
/// channel of the residual stream carries 5.4e2 by block 15 and 2.3e4 by block
/// 31 (issue #1596).
#[test]
fn vision_rms_norm_survives_a_float16_row_that_overflows_the_reduction() {
    let _guard = mlx_test_guard();
    let dim = 1280usize;
    let mut row = vec![0.05f32; dim];
    row[849] = 536.0;
    let x = mlxcel_core::from_slice_f32(&row, &[1, dim as i32]);

    let mut weights = WeightMap::new();
    weights.insert(
        "norm.weight".to_string(),
        mlxcel_core::from_slice_f32(&vec![1.0f32; dim], &[dim as i32]),
    );
    let norm = super::VisionRMSNorm::from_weights(&weights, "norm", 1e-6).expect("norm weights");

    let reference = norm.forward(&x);
    mlxcel_core::eval(&reference);
    let reference = to_vec(&reference);

    let x16 = mlxcel_core::astype(&x, mlxcel_core::dtype::FLOAT16);
    let normed = norm.forward(&x16);
    mlxcel_core::eval(&normed);
    assert_eq!(
        mlxcel_core::array_dtype(&normed),
        mlxcel_core::dtype::FLOAT16,
        "the norm must hand the tower back its own dtype"
    );
    let normed = to_vec(&normed);

    let peak = normed[849].abs();
    assert!(
        peak > 1.0,
        "float16 reduction overflowed and erased the row: peak {peak}"
    );
    for (i, (&got, &want)) in normed.iter().zip(reference.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-2 * want.abs().max(1e-2),
            "element {i}: got {got}, want {want}"
        );
    }
}
