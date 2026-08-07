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

//! Falcon-OCR weight-sanitization tests.

use super::*;

fn read(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::array_evaluated_bytes(array)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The fused projection alternates gate and up **rows**, not halves. Getting
/// this backwards yields a model that loads cleanly and generates garbage, so
/// the row assignment is pinned against the reference kernel's indexing
/// (`gate = 2*i`, `up = 2*i + 1`).
#[test]
fn the_fused_w13_splits_on_alternating_rows() {
    let mut weights = WeightMap::new();
    // 3 gate rows and 3 up rows, interleaved, width 2.
    let data: Vec<f32> = vec![
        10.0, 11.0, // gate row 0
        20.0, 21.0, // up   row 0
        12.0, 13.0, // gate row 1
        22.0, 23.0, // up   row 1
        14.0, 15.0, // gate row 2
        24.0, 25.0, // up   row 2
    ];
    weights.insert(
        "layers.0.feed_forward.w13.weight".to_string(),
        mlxcel_core::from_slice_f32(&data, &[6, 2]),
    );

    let out = sanitize_falcon_ocr_weights(weights).expect("sanitize succeeds");
    assert!(!out.contains_key("layers.0.feed_forward.w13.weight"));

    let gate = out
        .get("layers.0.feed_forward.w1.weight")
        .expect("gate rows");
    let up = out.get("layers.0.feed_forward.w3.weight").expect("up rows");
    assert_eq!(mlxcel_core::array_shape(gate), vec![3, 2]);
    assert_eq!(read(gate), vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    assert_eq!(read(up), vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0]);
}

/// Only `w13` is rewritten; the rest of the non-HF naming is already what the
/// decoder expects and must survive untouched.
#[test]
fn the_other_checkpoint_keys_pass_through_unchanged() {
    let mut weights = WeightMap::new();
    for key in [
        "tok_embeddings.weight",
        "img_projector.weight",
        "freqs_cis_golden",
        "norm.weight",
        "output.weight",
        "layers.0.attention.wqkv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.attention.sinks",
        "layers.0.feed_forward.w2.weight",
    ] {
        weights.insert(key.to_string(), mlxcel_core::from_slice_f32(&[1.0], &[1]));
    }
    let out = sanitize_falcon_ocr_weights(weights).expect("sanitize succeeds");
    assert_eq!(out.len(), 9);
    assert!(out.contains_key("layers.0.attention.sinks"));
    assert!(out.contains_key("freqs_cis_golden"));
}

/// A quantized conversion carries `scales` / `biases` on the same row axis, so
/// the split has to follow them too or the halves desync from their weights.
#[test]
fn quantization_sidecars_split_with_their_weight() {
    let mut weights = WeightMap::new();
    for suffix in ["weight", "scales", "biases"] {
        weights.insert(
            format!("layers.3.feed_forward.w13.{suffix}"),
            mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1]),
        );
    }
    let out = sanitize_falcon_ocr_weights(weights).expect("sanitize succeeds");
    for suffix in ["weight", "scales", "biases"] {
        let gate = out
            .get(&format!("layers.3.feed_forward.w1.{suffix}"))
            .unwrap_or_else(|| panic!("missing gate {suffix}"));
        let up = out
            .get(&format!("layers.3.feed_forward.w3.{suffix}"))
            .unwrap_or_else(|| panic!("missing up {suffix}"));
        assert_eq!(read(gate), vec![1.0, 3.0]);
        assert_eq!(read(up), vec![2.0, 4.0]);
    }
}

#[test]
fn an_odd_row_count_is_rejected_rather_than_silently_truncated() {
    let mut weights = WeightMap::new();
    weights.insert(
        "layers.0.feed_forward.w13.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0], &[3, 1]),
    );
    assert!(sanitize_falcon_ocr_weights(weights).is_err());
}

#[test]
fn the_key_splitter_only_matches_the_fused_projection() {
    assert_eq!(
        split_w13_key("layers.7.feed_forward.w13.weight"),
        Some(("layers.7.feed_forward", "weight"))
    );
    assert_eq!(split_w13_key("layers.7.feed_forward.w2.weight"), None);
    assert_eq!(split_w13_key("layers.7.attention.wqkv.weight"), None);
}
