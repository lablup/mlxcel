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

//! Checkpoint-free unit tests for MiniMax-M3-VL.
//!
//! The 427B checkpoint cannot be loaded on the development machine, so these
//! cover the surface reachable without weights: real nested-config parsing
//! (`vision_config` and `text_config`), the sanitizer's vision/projector skip
//! against verbatim checkpoint keys, the projector fold ordering
//! (`merge^2 * projection_dim`), a tiny synthetic tower forward (patch embed ->
//! pre_layrnorm -> CLIP layers with `cu_seqlens` + 3D vision RoPE -> two-stage
//! projector), the 3D vision RoPE axis/rot-dim arithmetic and the emitted
//! `rot_pos_emb` shape with its all-zero temporal section for `grid_t == 1`,
//! and the placeholder-count invariant against the shared Qwen-VL insertion
//! helper. Run serially (`--test-threads=1`); the MLX ops touch the device.

use crate::models::minimax_m3;
use crate::vision::encoders::minimax_m3_vl::{
    MiniMaxM3VisionConfig, MiniMaxM3VisionEncoder, rope_axis_dim, rope_rot_dim,
};
use mlxcel_core::MlxArray;
use mlxcel_core::weights::WeightMap;

fn filled(shape: &[i32], val: f32) -> mlxcel_core::UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![val; n as usize], shape)
}

fn reduce_max_abs(a: &MlxArray) -> f32 {
    let flat = mlxcel_core::reshape(a, &[-1]);
    let m = mlxcel_core::max_axis(&mlxcel_core::abs(&flat), 0, false);
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

// A reduced tower config: hidden 128, 2 heads (head_dim 64, a fused-SDPA
// supported width), 1 layer, patch 2, projection_dim 16, merge 2,
// temporal_patch 2. in_features = 3*2*2*2 = 24.
const TINY_VISION_CONFIG: &str = r#"{
    "model_type": "clip_vision_model",
    "hidden_size": 128,
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "intermediate_size": 64,
    "patch_size": 2,
    "projection_dim": 16,
    "rope_theta": 10000.0,
    "layer_norm_eps": 1e-05,
    "img_token_compression_config": {
        "image_token_compression_method": "patch_merge",
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    }
}"#;

fn tiny_vision_config() -> MiniMaxM3VisionConfig {
    serde_json::from_str(TINY_VISION_CONFIG).expect("tiny vision config parses")
}

/// Verbatim-named synthetic weights for the tiny tower.
fn tiny_tower_weights(cfg: &MiniMaxM3VisionConfig) -> WeightMap {
    let hidden = cfg.hidden_size as i32;
    let inter = cfg.intermediate_size as i32;
    let proj = cfg.projection_dim as i32;
    let in_features =
        (cfg.in_channels * cfg.temporal_patch_size() * cfg.patch_size * cfg.patch_size) as i32;
    let fold = (cfg.spatial_merge_size() * cfg.spatial_merge_size()) as i32;

    let vt = "vision_tower.vision_model";
    let mut w = WeightMap::new();

    // Patch embedding (2D form) + pre_layrnorm.
    w.insert(
        format!("{vt}.embeddings.patch_embedding.weight"),
        filled(&[hidden, in_features], 0.02),
    );
    w.insert(format!("{vt}.pre_layrnorm.weight"), filled(&[hidden], 1.0));
    w.insert(format!("{vt}.pre_layrnorm.bias"), filled(&[hidden], 0.0));

    // Encoder layer 0.
    let l = format!("{vt}.encoder.layers.0");
    for ln in ["layer_norm1", "layer_norm2"] {
        w.insert(format!("{l}.{ln}.weight"), filled(&[hidden], 1.0));
        w.insert(format!("{l}.{ln}.bias"), filled(&[hidden], 0.0));
    }
    for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        w.insert(
            format!("{l}.self_attn.{p}.weight"),
            filled(&[hidden, hidden], 0.05),
        );
        w.insert(format!("{l}.self_attn.{p}.bias"), filled(&[hidden], 0.0));
    }
    w.insert(
        format!("{l}.mlp.fc1.weight"),
        filled(&[inter, hidden], 0.03),
    );
    w.insert(format!("{l}.mlp.fc1.bias"), filled(&[inter], 0.0));
    w.insert(
        format!("{l}.mlp.fc2.weight"),
        filled(&[hidden, inter], 0.03),
    );
    w.insert(format!("{l}.mlp.fc2.bias"), filled(&[hidden], 0.0));

    // Two-stage projector.
    w.insert(
        "multi_modal_projector.linear_1.weight".into(),
        filled(&[proj, hidden], 0.04),
    );
    w.insert(
        "multi_modal_projector.linear_1.bias".into(),
        filled(&[proj], 0.0),
    );
    w.insert(
        "multi_modal_projector.linear_2.weight".into(),
        filled(&[proj, proj], 0.04),
    );
    w.insert(
        "multi_modal_projector.linear_2.bias".into(),
        filled(&[proj], 0.0),
    );
    w.insert(
        "patch_merge_mlp.linear_1.weight".into(),
        filled(&[proj, fold * proj], 0.02),
    );
    w.insert("patch_merge_mlp.linear_1.bias".into(), filled(&[proj], 0.0));
    w.insert(
        "patch_merge_mlp.linear_2.weight".into(),
        filled(&[proj, proj], 0.02),
    );
    w.insert("patch_merge_mlp.linear_2.bias".into(), filled(&[proj], 0.0));

    w
}

#[test]
fn tower_loads_verbatim_keys_and_emits_merged_token_shape() {
    // A single 2x2-patch-grid image (grid (1,2,2)) -> 4 patches -> 1 merged
    // token after the merge^2 fold. Exercises patch embed, pre_layrnorm, the
    // CLIP layer (cu_seqlens attention + 3D vision RoPE), and both projector
    // stages, and proves the tower resolves every verbatim key (including the
    // `pre_layrnorm` spelling).
    let cfg = tiny_vision_config();
    let weights = tiny_tower_weights(&cfg);
    let encoder = MiniMaxM3VisionEncoder::from_weights(&weights, &cfg).expect("tiny tower loads");

    let in_features =
        (cfg.in_channels * cfg.temporal_patch_size() * cfg.patch_size * cfg.patch_size) as i32;
    let grid = vec![(1i32, 2i32, 2i32)];
    let num_patches = 4i32;
    let pixel_values = filled(&[num_patches, in_features], 0.1);

    let out = encoder.forward_with_grid(&pixel_values, &grid);
    mlxcel_core::eval(&out.hidden_states);

    let merge = cfg.spatial_merge_size() as i32;
    let (t, h, w) = grid[0];
    let expected_tokens = t * (h / merge) * (w / merge); // grid.prod()/merge^2 = 1
    assert_eq!(
        mlxcel_core::array_shape(&out.hidden_states),
        vec![expected_tokens, cfg.projection_dim as i32]
    );
    assert!(
        reduce_max_abs(&out.hidden_states).is_finite(),
        "tower forward must be finite"
    );
}

#[test]
fn rope_axis_and_rot_dim_match_reference_arithmetic() {
    // Pins the 3D vision RoPE split introduced by the review fix: axis_dim =
    // 2 * ((head_dim / 2) / 3 / 2) head dims per (t, h, w) axis, rot_dim =
    // 3 * axis_dim rotated, with head_dim - rot_dim trailing dims untouched.
    // head_dim 80 is the real MiniMaxAI/MiniMax-M3 vision head_dim (hidden
    // 1280 / 16 heads); head_dim 64 is the reduced tiny test config's
    // (hidden 128 / 2 heads).
    assert_eq!(rope_axis_dim(80), 26);
    assert_eq!(rope_rot_dim(80), 78);
    assert_eq!(rope_axis_dim(64), 20);
    assert_eq!(rope_rot_dim(64), 60);

    // rot_pos_emb must emit [total_tokens, rot_dim / 2] (the concatenated t, h,
    // w frequency sections), and for grid_t == 1 (images) the temporal ids are
    // all-zero, so the leading axis_dim/2 columns (the t section) must be
    // exactly zero even though the axis still reserves its slice of head_dim.
    let cfg = tiny_vision_config();
    let weights = tiny_tower_weights(&cfg);
    let encoder = MiniMaxM3VisionEncoder::from_weights(&weights, &cfg).expect("tiny tower loads");

    let head_dim = cfg.head_dim() as i32;
    let axis_dim = rope_axis_dim(head_dim);
    let rot_dim = rope_rot_dim(head_dim);
    assert_eq!(head_dim, 64);
    assert_eq!(axis_dim, 20);
    assert_eq!(rot_dim, 60);

    let grid = vec![(1i32, 2i32, 2i32)]; // grid_t == 1: temporal section is inert.
    let (t, h, w) = grid[0];
    let total_tokens = t * h * w;

    let freqs = encoder.rot_pos_emb(&grid);
    mlxcel_core::eval(&freqs);
    assert_eq!(
        mlxcel_core::array_shape(&freqs),
        vec![total_tokens, rot_dim / 2]
    );

    let half_axis = axis_dim / 2;
    let t_section = mlxcel_core::slice(&freqs, &[0, 0], &[total_tokens, half_axis]);
    assert_eq!(
        reduce_max_abs(&t_section),
        0.0,
        "t-section of rot_pos_emb must be exactly zero for grid_t == 1"
    );
}

#[test]
fn projector_fold_concatenates_four_adjacent_patches_in_order() {
    // The patch-merge fold is a row-major reshape [P, D] -> [P/4, 4*D]. The four
    // patches of a 2x2 merge cell are contiguous in the processor's patch
    // order, so the reshape must place patch i's features in the i-th D-slice.
    let d = 3i32;
    let rows = 4i32;
    // Row i is the constant (i+1); after fold the 4*D vector must read
    // [1,1,1, 2,2,2, 3,3,3, 4,4,4].
    let mut data = Vec::new();
    for i in 0..rows {
        for _ in 0..d {
            data.push((i + 1) as f32);
        }
    }
    let patches = mlxcel_core::from_slice_f32(&data, &[rows, d]);
    let folded = mlxcel_core::reshape(&patches, &[-1, 4 * d]);
    mlxcel_core::eval(&folded);
    assert_eq!(mlxcel_core::array_shape(&folded), vec![1, 4 * d]);

    for (patch_idx, &expected) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
        let start = patch_idx as i32 * d;
        let slice = mlxcel_core::slice(&folded, &[0, start], &[1, start + d]);
        let diff = mlxcel_core::subtract(&slice, &filled(&[1, d], expected));
        assert!(
            reduce_max_abs(&diff) < 1e-6,
            "patch {patch_idx} features must occupy fold slice {patch_idx}"
        );
    }
}

#[test]
fn vision_config_parses_real_nested_values() {
    // The real MiniMaxAI/MiniMax-M3 vision_config (with the vestigial LLaVA-style
    // keys present, which must be ignored).
    let json = r#"{
        "model_type": "clip_vision_model",
        "hidden_size": 1280,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "intermediate_size": 5120,
        "patch_size": 14,
        "image_size": 2016,
        "projection_dim": 6144,
        "position_embedding_type": "rope",
        "rope_mode": "3d",
        "rope_theta": 10000.0,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-05,
        "img_token_compression_config": {
            "image_token_compression_method": "patch_merge",
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        },
        "vision_segment_max_frames": 4,
        "image_grid_pinpoints": [[336, 336]],
        "vision_feature_layer": -1,
        "vision_feature_select_strategy": "full",
        "image_seq_length": 576
    }"#;
    let cfg: MiniMaxM3VisionConfig = serde_json::from_str(json).expect("real vision_config parses");
    assert_eq!(cfg.hidden_size, 1280);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_hidden_layers, 32);
    assert_eq!(cfg.intermediate_size, 5120);
    assert_eq!(cfg.patch_size, 14);
    assert_eq!(cfg.projection_dim, 6144);
    assert_eq!(cfg.head_dim(), 80);
    assert!((cfg.rope_theta - 10000.0).abs() < 1.0);
    assert!((cfg.layer_norm_eps - 1e-5).abs() < 1e-9);
    assert_eq!(cfg.spatial_merge_size(), 2);
    assert_eq!(cfg.temporal_patch_size(), 2);
    // 24576 = merge^2 * projection_dim, the patch-merge fold width.
    assert_eq!(
        cfg.spatial_merge_size() * cfg.spatial_merge_size() * cfg.projection_dim,
        24576
    );
}

#[test]
fn nested_config_parses_text_and_vision_blocks() {
    // The top-level minimax_m3_vl config nests text_config + vision_config.
    let json = r#"{
        "model_type": "minimax_m3_vl",
        "text_config": {
            "model_type": "minimax_m3",
            "hidden_size": 6144,
            "intermediate_size": 3072,
            "num_hidden_layers": 60,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 200064,
            "num_local_experts": 128,
            "num_experts_per_tok": 4
        },
        "vision_config": {
            "model_type": "clip_vision_model",
            "hidden_size": 1280,
            "num_attention_heads": 16,
            "num_hidden_layers": 32,
            "intermediate_size": 5120,
            "patch_size": 14,
            "projection_dim": 6144
        }
    }"#;
    let full: serde_json::Value = serde_json::from_str(json).expect("nested config parses");

    let text_args: minimax_m3::ModelArgs =
        serde_json::from_value(full.get("text_config").cloned().unwrap())
            .expect("text_config parses into MiniMax-M3 ModelArgs");
    assert_eq!(text_args.hidden_size, 6144);
    assert_eq!(text_args.num_hidden_layers, 60);
    assert_eq!(text_args.num_local_experts, 128);

    let vcfg: MiniMaxM3VisionConfig =
        serde_json::from_value(full.get("vision_config").cloned().unwrap())
            .expect("vision_config parses");
    assert_eq!(vcfg.hidden_size, 1280);
    assert_eq!(vcfg.projection_dim, 6144);
    // Vision projector output must equal the text hidden size for the merge.
    assert_eq!(vcfg.projection_dim, text_args.hidden_size);
}

#[test]
fn text_sanitizer_drops_verbatim_vision_and_projector_keys() {
    // The VL loader builds the tower from the raw keys, then hands the map to
    // the MiniMax-M3 text sanitizer, which must drop every vision/projector
    // tensor (verbatim names, including the pre_layrnorm spelling) and rewrite
    // language_model.model.* -> model.*.
    let mut weights = WeightMap::new();
    for key in [
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
        "vision_tower.vision_model.pre_layrnorm.weight",
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "multi_modal_projector.linear_1.weight",
        "patch_merge_mlp.linear_1.weight",
    ] {
        weights.insert(key.to_string(), filled(&[1], 0.0));
    }

    let args: minimax_m3::ModelArgs = serde_json::from_str(
        r#"{"model_type":"minimax_m3","hidden_size":8,"intermediate_size":8,"num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":4,"vocab_size":16,"num_local_experts":4,"num_experts_per_tok":2}"#,
    )
    .expect("args parse");
    let out = minimax_m3::sanitize_weights(weights, &args);

    assert!(out.contains_key("model.embed_tokens.weight"));
    assert!(out.contains_key("model.norm.weight"));
    assert!(out.contains_key("model.lm_head.weight"));
    assert!(!out.contains_key("vision_tower.vision_model.pre_layrnorm.weight"));
    assert!(!out.contains_key("vision_tower.vision_model.embeddings.patch_embedding.weight"));
    assert!(
        !out.contains_key("vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight")
    );
    assert!(!out.contains_key("multi_modal_projector.linear_1.weight"));
    assert!(!out.contains_key("patch_merge_mlp.linear_1.weight"));
    // Only the 3 text tensors survive.
    assert_eq!(out.len(), 3);
}

#[test]
fn placeholder_expansion_matches_merged_token_count() {
    // The shared Qwen-VL insertion helper expands one placeholder per image to
    // t*(h/merge)*(w/merge) image tokens, which must equal the tower's merged
    // token count (grid.prod()/merge^2).
    let merge = 2usize;
    let grid = vec![(1i32, 6i32, 8i32)];
    let image_token_id = 200025;
    let vision_start = 200029;
    let mut prompt = vec![1i32, image_token_id, 42i32];

    let stats = crate::qwen_vl::insert_qwen_vl_image_tokens(
        &mut prompt,
        &grid,
        merge,
        vision_start,
        image_token_id,
    )
    .expect("insertion succeeds for a one-placeholder-per-image prompt");

    let (t, h, w) = grid[0];
    let merged_tokens = t * (h / merge as i32) * (w / merge as i32);
    assert_eq!(stats.total_image_tokens, merged_tokens);
    // Expanded prompt: bos + merged_tokens image tokens + trailing text.
    let placeholders = prompt.iter().filter(|&&t| t == image_token_id).count() as i32;
    assert_eq!(placeholders, merged_tokens);
}
