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

fn tiny_oracle_config() -> YoutuVisionConfig {
    YoutuVisionConfig {
        hidden_size: 16,
        out_hidden_size: 12,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_channels: 3,
        num_patches: 64,
        patch_size: 2,
        spatial_merge_size: 2,
        // Three merged groups per edge makes the `(4x8, 8x4)` packed fixture
        // exercise a non-self-inverse window permutation.
        window_size: 12,
        fullatt_block_indexes: vec![1],
        ..small_config()
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

fn array_to_vec_f32(array: &MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&array);
    mlxcel_core::array_to_raw_bytes(&array)
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
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
fn tiny_two_layer_oracle_exposes_non_vacuous_window_and_restore_stages() {
    let config = tiny_oracle_config();
    let prefix = "vision_tower";
    let weights = build_synthetic_weights(prefix, &config);
    let encoder = YoutuVLVisionEncoder::from_weights(&weights, &config, prefix).unwrap();
    let patch_width = config.num_channels * config.patch_size * config.patch_size;
    let mut values = vec![0.0f32; 64 * patch_width];
    for patch in 0..64 {
        for column in 0..patch_width {
            values[patch * patch_width + column] = patch as f32 * 0.01 + column as f32 * 0.0001;
        }
    }
    let patches = mlxcel_core::from_slice_f32(&values, &[64, patch_width as i32]);
    let capture = encoder
        .forward_with_spatial_diagnostics(&patches, &[(4, 8), (8, 4)])
        .unwrap();

    assert_eq!(
        mlxcel_core::array_shape(&capture.patch_projection),
        vec![64, 16]
    );
    assert_eq!(
        mlxcel_core::array_shape(&capture.window_layer0),
        vec![64, 16]
    );
    assert_eq!(mlxcel_core::array_shape(&capture.full_layer1), vec![64, 16]);
    assert_eq!(
        mlxcel_core::array_shape(&capture.merger_window_order),
        vec![16, 12]
    );
    assert_eq!(
        mlxcel_core::array_shape(&capture.restored_output),
        vec![16, 12]
    );
    assert_ne!(
        capture.window_index,
        (0..capture.window_index.len() as i32).collect::<Vec<_>>()
    );
    assert_ne!(capture.window_index, capture.reverse_indices);

    let patch_projection = array_to_vec_f32(&capture.patch_projection);
    let layer0 = array_to_vec_f32(&capture.window_layer0);
    let layer1 = array_to_vec_f32(&capture.full_layer1);
    let merger_window = array_to_vec_f32(&capture.merger_window_order);
    let restored = array_to_vec_f32(&capture.restored_output);
    for stage in [
        &patch_projection,
        &layer0,
        &layer1,
        &merger_window,
        &restored,
    ] {
        assert!(stage.iter().all(|value| value.is_finite()));
    }
    assert_ne!(patch_projection, layer0);
    assert_ne!(layer0, layer1);
    assert_ne!(merger_window, restored);
    for (original_group, &window_position) in capture.reverse_indices.iter().enumerate() {
        assert_eq!(
            &restored[original_group * 12..original_group * 12 + 12],
            &merger_window[window_position as usize * 12..window_position as usize * 12 + 12]
        );
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
