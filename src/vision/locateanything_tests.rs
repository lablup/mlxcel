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

//! Unit tests for the LocateAnything connector.

use super::LocateAnythingConnector;
use mlxcel_core::weights::WeightMap;

fn insert(wm: &mut WeightMap, key: &str, data: &[f32], shape: &[i32]) {
    wm.insert(key.to_string(), mlxcel_core::from_slice_f32(data, shape));
}

/// Build a connector whose LayerNorm spans the flattened merged patch
/// (`vision_hidden * kh * kw`), which is what the real checkpoint stores.
fn build(input_dim: i32, text_hidden: i32) -> LocateAnythingConnector {
    let mut wm = WeightMap::new();
    insert(
        &mut wm,
        "mmp.layer_norm.weight",
        &vec![1.0; input_dim as usize],
        &[input_dim],
    );
    insert(
        &mut wm,
        "mmp.layer_norm.bias",
        &vec![0.0; input_dim as usize],
        &[input_dim],
    );
    insert(
        &mut wm,
        "mmp.linear_1.weight",
        &vec![0.05; (text_hidden * input_dim) as usize],
        &[text_hidden, input_dim],
    );
    insert(
        &mut wm,
        "mmp.linear_1.bias",
        &vec![0.0; text_hidden as usize],
        &[text_hidden],
    );
    insert(
        &mut wm,
        "mmp.linear_2.weight",
        &vec![0.05; (text_hidden * text_hidden) as usize],
        &[text_hidden, text_hidden],
    );
    insert(
        &mut wm,
        "mmp.linear_2.bias",
        &vec![0.0; text_hidden as usize],
        &[text_hidden],
    );

    LocateAnythingConnector::from_weights(&wm, "mmp", input_dim, 64, 4).expect("build connector")
}

#[test]
fn projects_merged_patches_into_the_text_hidden_size() {
    // vision_hidden d = 4, merge 2x2 -> input_dim = 16; text_hidden = 3.
    let d = 4i32;
    let input_dim = 16i32;
    let text_hidden = 3i32;
    let conn = build(input_dim, text_hidden);

    // MoonViT patch-merger output: [total_merged = 2, kh*kw = 4, d = 4].
    let total_merged = 2i32;
    let data: Vec<f32> = (0..(total_merged * 4 * d)).map(|i| i as f32).collect();
    let feats = mlxcel_core::from_slice_f32(&data, &[total_merged, 4, d]);

    let out = conn.forward(&feats);
    mlxcel_core::eval(&out);
    assert_eq!(
        mlxcel_core::array_shape(&out),
        vec![total_merged, text_hidden],
        "one feature row per merged patch"
    );

    let mx = mlxcel_core::max_all(&out);
    mlxcel_core::eval(&mx);
    assert!(
        mlxcel_core::item_f32(&mx).is_finite(),
        "connector output must be finite"
    );
}

#[test]
fn feature_row_count_tracks_the_merged_patch_count() {
    // A different image grid must produce exactly as many rows as there will be
    // <IMG_CONTEXT> placeholders. 9 merged patches -> 9 rows.
    let input_dim = 16i32;
    let text_hidden = 5i32;
    let conn = build(input_dim, text_hidden);

    let total_merged = 9i32;
    let data: Vec<f32> = (0..(total_merged * 16)).map(|i| (i % 7) as f32).collect();
    let feats = mlxcel_core::from_slice_f32(&data, &[total_merged, 4, 4]);

    let out = conn.forward(&feats);
    mlxcel_core::eval(&out);
    assert_eq!(
        mlxcel_core::array_shape(&out),
        vec![total_merged, text_hidden]
    );
}

#[test]
fn layer_norm_spans_the_flattened_merged_patch() {
    // The connector flattens `[total_merged, kh*kw, vision_hidden]` to
    // `[total_merged, vision_hidden*kh*kw]` BEFORE normalizing, matching the
    // real checkpoint whose `layer_norm.weight` is 4608 wide (= 1152 * 2 * 2)
    // rather than 1152. Normalizing over the 4-wide vision dim instead would be
    // a different function, and this test separates the two.
    //
    // Input `[1, 4, 4]` where the kh*kw group `g` is filled with the constant
    // `g`. Under a per-group normalization every group is constant, so each
    // normalized value is exactly 0. Under the flattened normalization the 16
    // values are `[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]`, mean 1.5, so element 0
    // normalizes to a clearly negative number.
    //
    // `linear_1` is a one-hot row that selects element 0 (an all-equal weight
    // row would sum the normalized vector to 0 either way and prove nothing),
    // and `linear_2` is the 1x1 identity, so the observed value is
    // `gelu(normalized[0])`.
    let input_dim = 16i32;
    let mut wm = WeightMap::new();
    insert(&mut wm, "mmp.layer_norm.weight", &[1.0; 16], &[input_dim]);
    insert(&mut wm, "mmp.layer_norm.bias", &[0.0; 16], &[input_dim]);
    let mut one_hot = vec![0.0f32; 16];
    one_hot[0] = 1.0;
    insert(&mut wm, "mmp.linear_1.weight", &one_hot, &[1, input_dim]);
    insert(&mut wm, "mmp.linear_1.bias", &[0.0], &[1]);
    insert(&mut wm, "mmp.linear_2.weight", &[1.0], &[1, 1]);
    insert(&mut wm, "mmp.linear_2.bias", &[0.0], &[1]);
    let conn =
        LocateAnythingConnector::from_weights(&wm, "mmp", input_dim, 64, 4).expect("connector");

    let mut data = Vec::with_capacity(16);
    for g in 0..4 {
        for _ in 0..4 {
            data.push(g as f32);
        }
    }
    let feats = mlxcel_core::from_slice_f32(&data, &[1, 4, 4]);
    let out = conn.forward(&feats);
    mlxcel_core::eval(&out);

    // normalized[0] = (0 - 1.5) / sqrt(1.25) = -1.34164; gelu(-1.34164) =
    // -1.34164 * 0.5 * (1 + erf(-0.94868)) = -0.12055.
    let value = mlxcel_core::item_f32(&out);
    assert!(
        (value - -0.120_55).abs() < 1e-3,
        "expected gelu((0-1.5)/sqrt(1.25)) = -0.12055 from a flattened LayerNorm, got {value}"
    );
    assert!(
        value.abs() > 1e-3,
        "a per-group LayerNorm would collapse this input to 0"
    );
}
