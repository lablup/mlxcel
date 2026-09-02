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

use super::normalize_patch_embed_layout;
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
