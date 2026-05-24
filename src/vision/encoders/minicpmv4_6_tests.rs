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

//! Pure-host unit tests for the MiniCPM-V 4.6 vision encoder config and
//! Merger geometry. No real model checkpoint is required.

use super::{MiniCPMV46Config, MiniCPMV46Merger, MiniCPMV46VisionConfig};
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;

// ── Config-default deserialization (L3b) ─────────────────────────────────────

#[test]
fn minicpmv4_6_config_applies_documented_defaults_when_fields_absent() {
    // Real MiniCPM-V-4.6 checkpoints omit these top-level fields, so the serde
    // defaults must match the upstream reference values.
    let cfg: MiniCPMV46Config = serde_json::from_value(serde_json::json!({})).unwrap();
    assert_eq!(cfg.insert_layer_id, 6);
    assert_eq!(cfg.merge_kernel_size, [2, 2]);
    assert_eq!(cfg.merger_times, 1);
}

#[test]
fn minicpmv4_6_config_honors_explicit_overrides() {
    let cfg: MiniCPMV46Config = serde_json::from_value(serde_json::json!({
        "insert_layer_id": 8,
        "merge_kernel_size": [4, 4],
        "merger_times": 2,
    }))
    .unwrap();
    assert_eq!(cfg.insert_layer_id, 8);
    assert_eq!(cfg.merge_kernel_size, [4, 4]);
    assert_eq!(cfg.merger_times, 2);
}

#[test]
fn minicpmv4_6_vision_config_applies_documented_defaults() {
    // num_channels / image_size / patch_size / layer_norm_eps / window_kernel_size
    // all default; only the four required fields are supplied.
    let cfg: MiniCPMV46VisionConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
    }))
    .unwrap();
    assert_eq!(cfg.num_channels, 3);
    assert_eq!(cfg.image_size, 448);
    assert_eq!(cfg.patch_size, 14);
    assert!((cfg.layer_norm_eps - 1e-6).abs() < 1e-12);
    assert_eq!(cfg.window_kernel_size, [2, 2]);
}

// ── Merger geometry / shape (L3c) ────────────────────────────────────────────

/// Build a non-quantized `MergerBlock`'s weights at `{prefix}.mlp.{idx}`:
///   pre_norm.weight : [in_dim]
///   linear_1        : [mid_dim, in_dim]   (out, in)
///   linear_2        : [out_dim, mid_dim]
/// where `in_dim = inner_dim * merge_tokens`.
fn insert_merger_block_weights(
    weights: &mut WeightMap,
    prefix: &str,
    idx: usize,
    in_dim: i32,
    mid_dim: i32,
    out_dim: i32,
) {
    let base = format!("{}.mlp.{}", prefix, idx);
    weights.insert(
        format!("{}.pre_norm.weight", base),
        mlxcel_core::ones(&[in_dim], dtype::FLOAT32),
    );
    weights.insert(
        format!("{}.linear_1.weight", base),
        mlxcel_core::ones(&[mid_dim, in_dim], dtype::FLOAT32),
    );
    weights.insert(
        format!("{}.linear_2.weight", base),
        mlxcel_core::ones(&[out_dim, mid_dim], dtype::FLOAT32),
    );
}

#[test]
fn merger_two_rounds_2x2_reduces_32x32_grid_to_64_tokens() {
    // 32x32 = 1024 tokens -> round0 (2x2) -> 16x16 = 256 -> round1 (2x2) -> 8x8 = 64.
    let inner_dim = 2i32;
    let merge_tokens = 4i32; // 2 * 2
    let in_dim = inner_dim * merge_tokens; // pre_norm + linear_1 input
    let mid_dim = 2i32;
    let out_dim = 2i32; // chain dim stays constant so two rounds compose

    let mut weights = WeightMap::new();
    insert_merger_block_weights(&mut weights, "merger", 0, in_dim, mid_dim, out_dim);
    insert_merger_block_weights(&mut weights, "merger", 1, in_dim, mid_dim, out_dim);

    let merger = MiniCPMV46Merger::from_weights(
        &weights,
        "merger",
        /*merger_times*/ 2,
        /*merge_kernel_size*/ [2, 2],
        /*eps*/ 1e-6,
        /*group_size*/ 0,
        /*bits*/ 0,
    )
    .unwrap();

    let x = mlxcel_core::ones(&[1024, inner_dim], dtype::FLOAT32);
    let (tokens, h, w) = merger.forward(&x, 32, 32);
    mlxcel_core::eval(&tokens);

    assert_eq!(mlxcel_core::array_shape(&tokens), vec![64, out_dim]);
    assert_eq!((h, w), (8, 8));
}

#[test]
fn merger_single_round_honors_4x4_merge_kernel_size() {
    // M1 regression: a single 4x4 merge round collapses 32x32 -> 8x8 = 64 tokens.
    // With the previously hardcoded 2x2 this would have produced 16x16 = 256.
    let inner_dim = 2i32;
    let merge_tokens = 16i32; // 4 * 4
    let in_dim = inner_dim * merge_tokens;
    let mid_dim = 2i32;
    let out_dim = 3i32;

    let mut weights = WeightMap::new();
    insert_merger_block_weights(&mut weights, "merger", 0, in_dim, mid_dim, out_dim);

    let merger = MiniCPMV46Merger::from_weights(
        &weights,
        "merger",
        /*merger_times*/ 1,
        /*merge_kernel_size*/ [4, 4],
        /*eps*/ 1e-6,
        /*group_size*/ 0,
        /*bits*/ 0,
    )
    .unwrap();

    let x = mlxcel_core::ones(&[1024, inner_dim], dtype::FLOAT32);
    let (tokens, h, w) = merger.forward(&x, 32, 32);
    mlxcel_core::eval(&tokens);

    assert_eq!(mlxcel_core::array_shape(&tokens), vec![64, out_dim]);
    assert_eq!((h, w), (8, 8));
}

#[test]
fn merger_output_grid_size_rejects_non_divisible_spatial_grid() {
    let inner_dim = 2i32;
    let merge_tokens = 4i32;
    let in_dim = inner_dim * merge_tokens;
    let mid_dim = 2i32;
    let out_dim = 2i32;

    let mut weights = WeightMap::new();
    insert_merger_block_weights(&mut weights, "merger", 0, in_dim, mid_dim, out_dim);

    let merger = MiniCPMV46Merger::from_weights(&weights, "merger", 1, [2, 2], 1e-6, 0, 0).unwrap();

    assert_eq!(merger.output_grid_size(16, 12).unwrap(), (8, 6));
    assert!(merger.output_grid_size(15, 12).is_err());
}

#[test]
fn merger_rejects_zero_merge_kernel_size() {
    let weights = WeightMap::new();
    let err = match MiniCPMV46Merger::from_weights(&weights, "merger", 0, [0, 2], 1e-6, 0, 0) {
        Ok(_) => panic!("expected zero merge_kernel_size to be rejected"),
        Err(err) => err,
    };

    assert!(err.contains("merge_kernel_size"));
}
