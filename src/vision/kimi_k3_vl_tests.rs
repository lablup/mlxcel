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
use mlxcel_core::dtype;

fn insert(wm: &mut WeightMap, key: &str, data: &[f32], shape: &[i32]) {
    wm.insert(key.to_string(), mlxcel_core::from_slice_f32(data, shape));
}

/// `vision_hidden = 2`, merge 2x2 -> `merged_hidden = 8`, projector width 4,
/// `text_hidden = 6`.
fn tiny_projector() -> KimiK3Projector {
    let mut wm = WeightMap::new();
    let proj_0: Vec<f32> = (0..32).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    insert(&mut wm, "mm_projector.proj.0.weight", &proj_0, &[4, 8]);
    let proj_2: Vec<f32> = (0..24).map(|i| ((i % 3) as f32 - 1.0) * 0.5).collect();
    insert(&mut wm, "mm_projector.proj.2.weight", &proj_2, &[6, 4]);
    insert(&mut wm, "mm_projector.post_norm.weight", &[1.0; 6], &[6]);
    KimiK3Projector::from_weights(&wm, "mm_projector", 8, 1e-5).expect("projector")
}

#[test]
fn projector_shapes_and_merge_count_mismatch_errors() {
    let projector = tiny_projector();

    // Three merged tokens of [4, 2] flatten to [3, 8] and project to [3, 6].
    let merged: Vec<f32> = (0..24).map(|i| i as f32 * 0.05).collect();
    let merged = mlxcel_core::from_slice_f32(&merged, &[3, 4, 2]);
    let projected = projector.forward(&merged).expect("projector forward");
    mlxcel_core::eval(&projected);
    assert_eq!(mlxcel_core::array_shape(&projected), vec![3, 6]);
    let values = mlxcel_core::utils::array_to_vec_f32(&projected);
    assert!(values.iter().all(|v| v.is_finite()));
    // The trailing RMSNorm (unit weight) leaves each row at RMS 1, less the
    // eps term, which is visible here because the synthetic rows are small.
    for row in values.chunks(6) {
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / 6.0).sqrt();
        assert!(rms > 0.9 && rms <= 1.0 + 1e-4, "row rms {rms}");
    }

    // A merged token of the wrong width is refused, not reshaped.
    let wrong = mlxcel_core::from_slice_f32(&[0.0; 18], &[3, 3, 2]);
    assert!(projector.forward(&wrong).is_err());

    // A rank-1 array is refused by the same check rather than indexed into
    // a shape that is not there.
    let rank1 = mlxcel_core::from_slice_f32(&[0.0; 8], &[8]);
    assert!(projector.forward(&rank1).is_err());

    // Merge: 3 rows need exactly 3 placeholders (id 99).
    let embeds = mlxcel_core::zeros(&[1, 5, 6], dtype::FLOAT32);
    let ids_ok = mlxcel_core::from_slice_i32(&[1, 99, 99, 99, 2], &[1, 5]);
    let merged_ok = merge_media_features(99, &projected, &embeds, &ids_ok).expect("3 == 3");
    mlxcel_core::eval(&merged_ok.inputs_embeds);
    assert_eq!(
        mlxcel_core::array_shape(&merged_ok.inputs_embeds),
        vec![1, 5, 6]
    );
    let out = mlxcel_core::utils::array_to_vec_f32(&merged_ok.inputs_embeds);
    // Row 1 of the stream is projected row 0; rows 0 and 4 stay zero.
    assert_eq!(&out[6..12], &values[0..6]);
    assert!(out[0..6].iter().all(|&v| v == 0.0));
    assert!(out[24..30].iter().all(|&v| v == 0.0));

    let ids_short = mlxcel_core::from_slice_i32(&[1, 99, 99, 3, 2], &[1, 5]);
    let err = match merge_media_features(99, &projected, &embeds, &ids_short) {
        Err(err) => err,
        Ok(_) => panic!("2 placeholders for 3 rows must be refused"),
    };
    assert!(err.contains("2 <|media_pad|>"), "{err}");
    assert!(err.contains("3 feature rows"), "{err}");

    let ids_long = mlxcel_core::from_slice_i32(&[99, 99, 99, 99, 2], &[1, 5]);
    assert!(merge_media_features(99, &projected, &embeds, &ids_long).is_err());
}

#[test]
fn vision_sanitize_transposes_the_patch_kernel_once() {
    let mut raw = WeightMap::new();
    raw.insert(
        "vision_tower.patch_embed.proj.weight".to_string(),
        mlxcel_core::zeros(&[8, 3, 2, 2], dtype::FLOAT32),
    );
    raw.insert(
        "vision_tower.encoder.blocks.0.wqkv.weight".to_string(),
        mlxcel_core::zeros(&[6, 4], dtype::FLOAT32),
    );
    let once = sanitize_kimi_k3_vision_weights(raw);
    assert_eq!(
        mlxcel_core::array_shape(&once["vision_tower.patch_embed.proj.weight"]),
        vec![8, 2, 2, 3]
    );
    assert!(once.contains_key("vision_tower.encoder.blocks.0.wqkv.weight"));
    let twice = sanitize_kimi_k3_vision_weights(once);
    assert_eq!(
        mlxcel_core::array_shape(&twice["vision_tower.patch_embed.proj.weight"]),
        vec![8, 2, 2, 3]
    );
}
