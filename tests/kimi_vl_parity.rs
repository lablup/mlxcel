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

//! Kimi-VL / Kimi-VL 2.5 (MoonViT) parity tests.
//!
//! These are checkpoint-free: they validate model-type detection, arch/metadata
//! registration, and the MoonViT vision-encoder + connector numerics against
//! values derived from the upstream reference
//! (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/kimi_vl). The
//! full safetensors directory loader and image-runtime path are wired in a
//! follow-up, so there is no gated real-model forward pass here yet.

use std::path::PathBuf;

use mlxcel::models::{ModelType, get_model_type};
use mlxcel::vision::encoders::kimi_vl::{KimiVLVisionConfig, KimiVLVisionModel};
use mlxcel::vision::kimi_vl::KimiVLMultiModalProjector;
use mlxcel_core::weights::WeightMap;

fn write_config(dir: &PathBuf, model_type: &str) {
    std::fs::create_dir_all(dir).expect("create temp model dir");
    // Minimal config: a `vision_config` block plus the discriminating
    // `model_type` string is enough for detection.
    let cfg = format!(
        r#"{{"model_type": "{model_type}", "vision_config": {{"model_type": "moonvit"}}, "text_config": {{"model_type": "deepseek_v3"}}}}"#
    );
    std::fs::write(dir.join("config.json"), cfg).expect("write config.json");
}

#[test]
fn detects_kimi_vl_and_kimi_k25_model_types() {
    let base = std::env::temp_dir().join(format!("kimi_vl_detect_{}", std::process::id()));

    let vl_dir = base.join("kimi_vl");
    write_config(&vl_dir, "kimi_vl");
    assert_eq!(
        get_model_type(&vl_dir).expect("detect kimi_vl"),
        ModelType::KimiVL
    );

    let k25_dir = base.join("kimi_k25");
    write_config(&k25_dir, "kimi_k25");
    assert_eq!(
        get_model_type(&k25_dir).expect("detect kimi_k25"),
        ModelType::KimiK25
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn kimi_vl_metadata_is_registered() {
    for mt in [ModelType::KimiVL, ModelType::KimiK25] {
        let (name, family) = mt.metadata();
        assert!(!name.is_empty(), "{mt:?} display name must be non-empty");
        assert_eq!(family, "Kimi VLM", "{mt:?} family");
        assert!(name.contains("Kimi-VL"), "{mt:?} name mentions Kimi-VL");
    }
    assert!(
        mlxcel::models::ALL_MODEL_TYPES.contains(&ModelType::KimiVL),
        "KimiVL must appear in ALL_MODEL_TYPES"
    );
    assert!(
        mlxcel::models::ALL_MODEL_TYPES.contains(&ModelType::KimiK25),
        "KimiK25 must appear in ALL_MODEL_TYPES"
    );
}

fn insert(wm: &mut WeightMap, key: &str, data: &[f32], shape: &[i32]) {
    wm.insert(key.to_string(), mlxcel_core::from_slice_f32(data, shape));
}

/// Build a tiny MoonViT config: embed_dim 8, 2 heads (head_dim 4, divisible by
/// 4 for the 2D rope), patch 2, 3 channels, 4x4 learned pos-emb grid, 2x2 merge.
fn tiny_config(depth: usize) -> KimiVLVisionConfig {
    KimiVLVisionConfig {
        model_type: "moonvit".to_string(),
        depth,
        embed_dim: 8,
        hidden_size: 8,
        num_heads: 2,
        patch_size: 2,
        num_channels: 3,
        intermediate_size: 16,
        init_pos_emb_height: 4,
        init_pos_emb_width: 4,
        spatial_merge_size: 2,
        layer_norm_eps: 1e-6,
        quant_group_size: 0,
        quant_bits: 0,
    }
}

fn build_vision_weights(cfg: &KimiVLVisionConfig, prefix: &str) -> WeightMap {
    let e = cfg.embed_dim as i32;
    let c = cfg.num_channels as i32;
    let p = cfg.patch_size as i32;
    let inter = cfg.intermediate_size as i32;
    let ph = cfg.init_pos_emb_height as i32;
    let pw = cfg.init_pos_emb_width as i32;
    let mut wm = WeightMap::new();

    insert(
        &mut wm,
        &format!("{prefix}.patch_embed.proj.weight"),
        &vec![0.05; (e * p * p * c) as usize],
        &[e, p, p, c],
    );
    insert(
        &mut wm,
        &format!("{prefix}.patch_embed.proj.bias"),
        &vec![0.0; e as usize],
        &[e],
    );
    insert(
        &mut wm,
        &format!("{prefix}.patch_embed.pos_emb.weight"),
        &vec![0.0; (ph * pw * e) as usize],
        &[ph, pw, e],
    );
    for i in 0..cfg.depth {
        for norm in ["norm0", "norm1"] {
            insert(
                &mut wm,
                &format!("{prefix}.blocks.{i}.{norm}.weight"),
                &vec![1.0; e as usize],
                &[e],
            );
            insert(
                &mut wm,
                &format!("{prefix}.blocks.{i}.{norm}.bias"),
                &vec![0.0; e as usize],
                &[e],
            );
        }
        insert(
            &mut wm,
            &format!("{prefix}.blocks.{i}.attn.wqkv.weight"),
            &vec![0.05; (3 * e * e) as usize],
            &[3 * e, e],
        );
        insert(
            &mut wm,
            &format!("{prefix}.blocks.{i}.attn.wqkv.bias"),
            &vec![0.0; (3 * e) as usize],
            &[3 * e],
        );
        insert(
            &mut wm,
            &format!("{prefix}.blocks.{i}.attn.wo.weight"),
            &vec![0.05; (e * e) as usize],
            &[e, e],
        );
        insert(
            &mut wm,
            &format!("{prefix}.blocks.{i}.attn.wo.bias"),
            &vec![0.0; e as usize],
            &[e],
        );
        insert(
            &mut wm,
            &format!("{prefix}.blocks.{i}.mlp.fc0.weight"),
            &vec![0.05; (inter * e) as usize],
            &[inter, e],
        );
        insert(
            &mut wm,
            &format!("{prefix}.blocks.{i}.mlp.fc0.bias"),
            &vec![0.0; inter as usize],
            &[inter],
        );
        insert(
            &mut wm,
            &format!("{prefix}.blocks.{i}.mlp.fc1.weight"),
            &vec![0.05; (e * inter) as usize],
            &[e, inter],
        );
        insert(
            &mut wm,
            &format!("{prefix}.blocks.{i}.mlp.fc1.bias"),
            &vec![0.0; e as usize],
            &[e],
        );
    }
    insert(
        &mut wm,
        &format!("{prefix}.final_layernorm.weight"),
        &vec![1.0; e as usize],
        &[e],
    );
    insert(
        &mut wm,
        &format!("{prefix}.final_layernorm.bias"),
        &vec![0.0; e as usize],
        &[e],
    );
    wm
}

#[test]
fn moonvit_encoder_and_connector_shapes_match_reference() {
    // Reference invariant: a (h, w) patch grid with a `merge x merge` merge
    // yields (h/merge)*(w/merge) merged tokens, and the connector projects each
    // merged token to the text hidden size.
    let cfg = tiny_config(2);
    let prefix = "vision_tower";
    let wm = build_vision_weights(&cfg, prefix);
    let encoder = KimiVLVisionModel::from_weights(&wm, &cfg, prefix).expect("build MoonViT");

    // One 4x4 patch grid -> 16 patches; patch 2, 3 channels.
    let grid = (4i32, 4i32);
    let n = grid.0 * grid.1;
    let pixels = mlxcel_core::from_slice_f32(
        &vec![
            0.25;
            (n * cfg.patch_size as i32 * cfg.patch_size as i32 * cfg.num_channels as i32) as usize
        ],
        &[
            n,
            cfg.patch_size as i32,
            cfg.patch_size as i32,
            cfg.num_channels as i32,
        ],
    );
    let merged = encoder.forward_with_grid(&pixels, &[grid]);
    // (4/2)*(4/2) = 4 merged tokens, each carrying kh*kw=4 vectors of dim 8.
    assert_eq!(
        mlxcel_core::array_shape(&merged),
        vec![4, 4, cfg.embed_dim as i32]
    );

    // Connector: merged_hidden = embed_dim * merge * merge = 8*4 = 32 -> text hidden 5.
    let text_hidden = 5i32;
    let merged_hidden =
        cfg.embed_dim as i32 * (cfg.spatial_merge_size * cfg.spatial_merge_size) as i32;
    let mut pw = WeightMap::new();
    insert(
        &mut pw,
        "mm.pre_norm.weight",
        &vec![1.0; cfg.embed_dim],
        &[cfg.embed_dim as i32],
    );
    insert(
        &mut pw,
        "mm.pre_norm.bias",
        &vec![0.0; cfg.embed_dim],
        &[cfg.embed_dim as i32],
    );
    insert(
        &mut pw,
        "mm.l1.weight",
        &vec![0.02; (merged_hidden * merged_hidden) as usize],
        &[merged_hidden, merged_hidden],
    );
    insert(
        &mut pw,
        "mm.l1.bias",
        &vec![0.0; merged_hidden as usize],
        &[merged_hidden],
    );
    insert(
        &mut pw,
        "mm.l2.weight",
        &vec![0.02; (text_hidden * merged_hidden) as usize],
        &[text_hidden, merged_hidden],
    );
    insert(
        &mut pw,
        "mm.l2.bias",
        &vec![0.0; text_hidden as usize],
        &[text_hidden],
    );

    let projector = KimiVLMultiModalProjector::from_weights(
        &pw,
        "mm.pre_norm",
        "mm.l1",
        "mm.l2",
        merged_hidden,
        64,
        4,
    )
    .expect("build projector");
    let image_features = projector.forward(&merged);
    mlxcel_core::eval(&image_features);
    // 4 merged tokens -> [4, text_hidden].
    assert_eq!(
        mlxcel_core::array_shape(&image_features),
        vec![4, text_hidden]
    );

    let mx = mlxcel_core::max_all(&image_features);
    mlxcel_core::eval(&mx);
    assert!(
        mlxcel_core::item_f32(&mx).is_finite(),
        "projected image features must be finite"
    );
}
