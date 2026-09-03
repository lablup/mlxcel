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

//! Unit tests for the Youtu-VL vision encoder.
//!
//! These exercise the windowed-attention bookkeeping (`get_window_index` and
//! `cu_window_seqlens`) along with a synthetic-weight forward pass. We avoid
//! coupling to a HuggingFace checkpoint so the tests run without network
//! access on Linux/CUDA CI.

use super::*;
use mlxcel_core::weights::WeightMap;

fn small_config() -> YoutuVisionConfig {
    YoutuVisionConfig {
        model_type: "siglip2_vision_model".to_string(),
        hidden_size: 32,
        out_hidden_size: 16,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_channels: 3,
        num_patches: 1024,
        patch_size: 2,
        layer_norm_eps: 1e-6,
        spatial_merge_size: 2,
        // window_size in pixel units → window_tokens = window_size / merge / patch
        // = 16 / 2 / 2 = 4 merged tokens per window edge.
        window_size: 16,
        // With num_hidden_layers = 2, only the last block runs full attention.
        fullatt_block_indexes: vec![1],
        quant_group_size: 0,
        quant_bits: 0,
    }
}

fn put(weights: &mut WeightMap, key: &str, shape: &[i32]) {
    let total: i64 = shape.iter().map(|&d| d as i64).product();
    let arr = mlxcel_core::arange_f32(0.0, total as f32, 1.0);
    let arr = mlxcel_core::reshape(&arr, shape);
    let arr = mlxcel_core::utils::silu(&arr); // bound the values to a small range
    weights.insert(key.to_string(), mlxcel_core::copy(&arr));
}

fn put_norm(weights: &mut WeightMap, key: &str, dim: i32) {
    let w = mlxcel_core::ones(&[dim], mlxcel_core::dtype::FLOAT32);
    weights.insert(format!("{}.weight", key), mlxcel_core::copy(&w));
    let b = mlxcel_core::zeros(&[dim], mlxcel_core::dtype::FLOAT32);
    weights.insert(format!("{}.bias", key), mlxcel_core::copy(&b));
}

fn put_rms(weights: &mut WeightMap, key: &str, dim: i32) {
    let w = mlxcel_core::ones(&[dim], mlxcel_core::dtype::FLOAT32);
    weights.insert(format!("{}.weight", key), mlxcel_core::copy(&w));
}

fn put_linear(weights: &mut WeightMap, key: &str, out_dim: i32, in_dim: i32, with_bias: bool) {
    put(weights, &format!("{}.weight", key), &[out_dim, in_dim]);
    if with_bias {
        put(weights, &format!("{}.bias", key), &[out_dim]);
    }
}

fn build_synthetic_weights(prefix: &str, config: &YoutuVisionConfig) -> WeightMap {
    let mut w = WeightMap::new();

    let h = config.hidden_size as i32;
    let i = config.intermediate_size as i32;
    let patch_dim = (config.patch_size * config.patch_size * config.num_channels) as i32;

    // patch embedding
    put_linear(
        &mut w,
        &format!("{}.embeddings.patch_embedding", prefix),
        h,
        patch_dim,
        true,
    );

    // encoder layers
    for layer in 0..config.num_hidden_layers {
        let lp = format!("{}.encoder.layers.{}", prefix, layer);
        put_norm(&mut w, &format!("{}.layer_norm1", lp), h);
        put_norm(&mut w, &format!("{}.layer_norm2", lp), h);
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            put_linear(&mut w, &format!("{}.self_attn.{}", lp, proj), h, h, true);
        }
        put_linear(&mut w, &format!("{}.mlp.fc1", lp), i, h, true);
        put_linear(&mut w, &format!("{}.mlp.fc2", lp), h, i, true);
    }

    // post LN
    put_norm(&mut w, &format!("{}.post_layernorm", prefix), h);

    // merger
    let merged_dim = config.hidden_size as i32
        * config.spatial_merge_size as i32
        * config.spatial_merge_size as i32;
    put_rms(&mut w, &format!("{}.merger.ln_q", prefix), h);
    put_linear(
        &mut w,
        &format!("{}.merger.mlp.0", prefix),
        merged_dim,
        merged_dim,
        true,
    );
    put_linear(
        &mut w,
        &format!("{}.merger.mlp.2", prefix),
        config.out_hidden_size as i32,
        merged_dim,
        true,
    );

    w
}

#[test]
fn config_defaults_match_upstream() {
    let raw = serde_json::json!({});
    let cfg: YoutuVisionConfig = serde_json::from_value(raw).unwrap();
    assert_eq!(cfg.hidden_size, 1152);
    assert_eq!(cfg.out_hidden_size, 2560);
    assert_eq!(cfg.num_hidden_layers, 27);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.patch_size, 16);
    assert_eq!(cfg.spatial_merge_size, 2);
    assert_eq!(cfg.window_size, 256);
    assert_eq!(cfg.fullatt_block_indexes, vec![7, 15, 23, 26]);
    assert_eq!(cfg.layer_norm_eps, 1e-6);
}

#[test]
fn window_index_partitions_token_set() {
    // Build an encoder just to hit `get_window_index` with a small config.
    // Using h=8, w=8 → llm_h=4, llm_w=4. With window_size=16, patch=2,
    // merge=2 we have merger_window_size = 16/2/2 = 4 → exactly one window.
    let cfg = small_config();
    let prefix = "vision_tower";
    let weights = build_synthetic_weights(prefix, &cfg);
    let encoder = YoutuVLVisionEncoder::from_weights(&weights, &cfg, prefix).unwrap();

    let spatial = vec![(8, 8)];
    let (window_index, cu_window_seqlens) = encoder.get_window_index(&spatial);

    let expected_total = (8 / 2) * (8 / 2);
    assert_eq!(
        window_index.len() as i32,
        expected_total,
        "window_index must cover every merged-patch index exactly once"
    );

    // Each merged-patch index in [0, expected_total) must appear exactly once
    // (i.e. window_index is a permutation of 0..expected_total).
    let mut sorted = window_index.clone();
    sorted.sort();
    let expected: Vec<i32> = (0..expected_total).collect();
    assert_eq!(sorted, expected);

    // cu_window_seqlens must be non-decreasing and end at the full sequence
    // length (in pre-merge token units). The pre-merge sequence length is
    // h*w = 64.
    let pre_merge_total = 8 * 8;
    assert_eq!(*cu_window_seqlens.first().unwrap(), 0);
    assert_eq!(*cu_window_seqlens.last().unwrap(), pre_merge_total);
    for w in cu_window_seqlens.windows(2) {
        assert!(w[0] <= w[1], "cu_window_seqlens must be non-decreasing");
    }
}

#[test]
fn window_index_handles_padding() {
    // h=10, w=8 with merger_window_size=4 forces vertical padding (10 not
    // divisible by 4 in merged-patch space → llm_h=5, pad to 8). The
    // permutation must still cover every real merged-patch index exactly once
    // (padding cells are stripped from the index list).
    let cfg = small_config();
    let prefix = "vision_tower";
    let weights = build_synthetic_weights(prefix, &cfg);
    let encoder = YoutuVLVisionEncoder::from_weights(&weights, &cfg, prefix).unwrap();

    let spatial = vec![(10, 8)];
    let (window_index, cu) = encoder.get_window_index(&spatial);

    let llm_h = 10 / 2;
    let llm_w = 8 / 2;
    let total = llm_h * llm_w;

    // window_index should be a permutation of 0..total.
    let mut sorted = window_index.clone();
    sorted.sort();
    let expected: Vec<i32> = (0..total).collect();
    assert_eq!(sorted, expected);

    // cu must end at h * w (pre-merge tokens).
    assert_eq!(*cu.last().unwrap(), 10 * 8);
}

#[test]
fn forward_with_spatial_produces_expected_shape() {
    // End-to-end synthetic-weight forward pass. With h=4, w=4, patch=2:
    // - num_patches = 4 (h_patches=2, w_patches=2)
    // - after merger: 1 merged token (spatial_merge_size=2 collapses 2x2)
    // - output dim = out_hidden_size = 16
    let cfg = small_config();
    let prefix = "vision_tower";
    let weights = build_synthetic_weights(prefix, &cfg);
    let encoder = YoutuVLVisionEncoder::from_weights(&weights, &cfg, prefix).unwrap();

    let h_patches = 4i32;
    let w_patches = 4i32;
    let num_patches = h_patches * w_patches;
    let patch_dim = (cfg.patch_size * cfg.patch_size * cfg.num_channels) as i32;

    let pixel_total = (num_patches * patch_dim) as usize;
    let pixel_data: Vec<f32> = (0..pixel_total).map(|i| (i as f32) * 1e-3).collect();
    let pixel_values = mlxcel_core::from_slice_f32(&pixel_data, &[num_patches, patch_dim]);

    let out = encoder.forward_with_spatial(&pixel_values, &[(h_patches, w_patches)]);
    let shape = mlxcel_core::array_shape(&out.hidden_states);

    let expected_merged_tokens =
        (h_patches / cfg.spatial_merge_size as i32) * (w_patches / cfg.spatial_merge_size as i32);
    assert_eq!(
        shape,
        vec![expected_merged_tokens, cfg.out_hidden_size as i32]
    );
}

#[test]
fn forward_is_deterministic_for_fixed_weights() {
    let cfg = small_config();
    let prefix = "vision_tower";
    let weights = build_synthetic_weights(prefix, &cfg);
    let encoder = YoutuVLVisionEncoder::from_weights(&weights, &cfg, prefix).unwrap();

    let h_patches = 4i32;
    let w_patches = 4i32;
    let num_patches = h_patches * w_patches;
    let patch_dim = (cfg.patch_size * cfg.patch_size * cfg.num_channels) as i32;
    let pixel_total = (num_patches * patch_dim) as usize;
    let pixel_data: Vec<f32> = (0..pixel_total).map(|i| (i as f32) * 1e-3).collect();
    let pixel_values = mlxcel_core::from_slice_f32(&pixel_data, &[num_patches, patch_dim]);

    let out1 = encoder.forward_with_spatial(&pixel_values, &[(h_patches, w_patches)]);
    let out2 = encoder.forward_with_spatial(&pixel_values, &[(h_patches, w_patches)]);

    mlxcel_core::eval(&out1.hidden_states);
    mlxcel_core::eval(&out2.hidden_states);

    let diff = mlxcel_core::subtract(&out1.hidden_states, &out2.hidden_states);
    let abs_diff = mlxcel_core::abs(&diff);
    let max_diff = mlxcel_core::max_all(&abs_diff);
    mlxcel_core::eval(&max_diff);
    assert!(
        mlxcel_core::item_f32(&max_diff) < 1e-5,
        "forward should be deterministic across calls (max diff observed = {})",
        mlxcel_core::item_f32(&max_diff)
    );
}

#[test]
fn merger_window_size_rejects_zero_divisor_config() {
    // M3 hardening: from_weights must return Err (not divide-by-zero panic)
    // when spatial_merge_size or patch_size is zero, or when window_size is
    // not an exact multiple of their product.
    let prefix = "vision_tower";
    let weights = build_synthetic_weights(prefix, &small_config());

    // spatial_merge_size = 0
    let mut cfg = small_config();
    cfg.spatial_merge_size = 0;
    let result = YoutuVLVisionEncoder::from_weights(&weights, &cfg, prefix);
    assert!(
        result.is_err(),
        "expected Err for spatial_merge_size=0, got Ok"
    );
    assert!(
        result.err().unwrap().contains("spatial_merge_size"),
        "error should mention spatial_merge_size"
    );

    // patch_size = 0
    let mut cfg = small_config();
    cfg.patch_size = 0;
    let result = YoutuVLVisionEncoder::from_weights(&weights, &cfg, prefix);
    assert!(result.is_err(), "expected Err for patch_size=0, got Ok");
    assert!(
        result.err().unwrap().contains("patch_size"),
        "error should mention patch_size"
    );

    // window_size not divisible by spatial_merge_size * patch_size
    // small_config: spatial_merge_size=2, patch_size=2 → divisor=4.
    // Setting window_size=7 is not divisible by 4.
    let mut cfg = small_config();
    cfg.window_size = 7;
    let result = YoutuVLVisionEncoder::from_weights(&weights, &cfg, prefix);
    assert!(
        result.is_err(),
        "expected Err for window_size not divisible by sms*ps, got Ok"
    );
    assert!(
        result.err().unwrap().contains("window_size"),
        "error should mention window_size"
    );
}

// ── Window inverse (issue #1600) ─────────────────────────────────────────────
//
// Upstream defaults: `patch_size` 16, `spatial_merge_size` 2, `window_size`
// 256, so a window is 8x8 merged tokens. `get_window_index` is a permutation of
// `0..N`; the encoder must un-reorder the merger output with its inverse.

/// `window_index` for a batch of `(h_patches, w_patches)` grids at the
/// upstream defaults.
fn window_index_at_defaults(spatial_shapes: &[(i32, i32)]) -> Vec<i32> {
    window::get_window_index(spatial_shapes, 2, 256, 16, 4).0
}

fn is_identity(permutation: &[i32]) -> bool {
    permutation.iter().enumerate().all(|(i, &v)| v == i as i32)
}

/// Apply `window_index` and then its inverse, as the encoder does around the
/// merger. Returns the composed permutation, which must be the identity.
fn round_trip(window_index: &[i32]) -> Vec<i32> {
    let inverse = window::reverse_window_indices(window_index);
    inverse.iter().map(|&i| window_index[i as usize]).collect()
}

/// How many positions `window_index` applied twice leaves out of place, which
/// is what the pre-#1600 un-reorder did.
fn misplaced_when_applied_twice(window_index: &[i32]) -> usize {
    window_index
        .iter()
        .enumerate()
        .filter(|&(i, &v)| window_index[v as usize] != i as i32)
        .count()
}

#[test]
fn reverse_window_indices_is_argsort_not_the_permutation_itself() {
    // The minimal non-involution: the old construction returned [2, 0, 1].
    assert_eq!(window::reverse_window_indices(&[2, 0, 1]), vec![1, 2, 0]);
    // An involution is its own inverse, which is the case that hid the defect.
    assert_eq!(
        window::reverse_window_indices(&[1, 0, 3, 2]),
        vec![1, 0, 3, 2]
    );
    // A malformed index is skipped rather than panicking.
    assert_eq!(window::reverse_window_indices(&[0, 7, 1]), vec![0, 2, 0]);
}

#[test]
fn reverse_window_indices_restores_raster_order_on_a_non_involution_grid() {
    // 512x512 pixels: a 32x32 patch grid, a 16x16 merged grid, 2x2 windows of
    // 8x8 merged tokens. Unlike Qwen2.5-VL's 4-wide window, where 16x16 is an
    // involution, this permutation is not one, so applying it twice scrambles
    // most of the image.
    let window_index = window_index_at_defaults(&[(32, 32)]);
    assert_eq!(window_index.len(), 256);
    assert_eq!(misplaced_when_applied_twice(&window_index), 192);
    assert!(is_identity(&round_trip(&window_index)));
}

#[test]
fn reverse_window_indices_is_the_identity_case_that_hid_the_defect() {
    // 256x256 pixels: one 8x8 window, so `window_index` is the identity and
    // the old construction was accidentally right. Keep the case so a change
    // to the window bookkeeping that breaks it is caught too.
    let window_index = window_index_at_defaults(&[(16, 16)]);
    assert_eq!(window_index.len(), 64);
    assert!(is_identity(&window_index));
    assert!(is_identity(&round_trip(&window_index)));
}

#[test]
fn reverse_window_indices_round_trips_a_multi_image_batch() {
    // Two images of different sizes; the second grid needs padding on both
    // edges. The inverse must respect the per-image offset `get_window_index`
    // adds.
    let window_index = window_index_at_defaults(&[(32, 32), (24, 16)]);
    assert_eq!(window_index.len(), 256 + 96);
    assert_eq!(misplaced_when_applied_twice(&window_index), 192);
    assert!(is_identity(&round_trip(&window_index)));
}
