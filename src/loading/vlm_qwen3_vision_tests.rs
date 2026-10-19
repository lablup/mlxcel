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

//! The Qwen3-VL Conv3d layout guard (#2388): a PyTorch-layout patch embed and
//! its MLX-layout twin must build the same tower, an MLX-layout weight must be
//! left as it is, and every Qwen3-VL tower site must go through the guard.

use mlxcel_core::weights::WeightMap;

use super::load_qwen3_vl_vision_encoder;
use crate::vision::encoders::qwen3_vl::Qwen3VLVisionConfig;

const HIDDEN: usize = 4;
const IN_CHANNELS: usize = 3;
const T: usize = 2;
const P: usize = 2;
const PATCH_KEY: &str = "vision_tower.patch_embed.proj.weight";

fn config() -> Qwen3VLVisionConfig {
    Qwen3VLVisionConfig {
        depth: 0,
        hidden_size: HIDDEN,
        intermediate_size: 8,
        out_hidden_size: HIDDEN,
        num_heads: 1,
        patch_size: P,
        spatial_merge_size: 1,
        temporal_patch_size: T,
        in_channels: IN_CHANNELS,
        num_position_embeddings: 4,
        deepstack_visual_indexes: Vec::new(),
        quant_group_size: 0,
        quant_bits: 0,
    }
}

fn insert(wm: &mut WeightMap, key: &str, data: &[f32], shape: &[i32]) {
    wm.insert(key.to_string(), mlxcel_core::from_slice_f32(data, shape));
}

/// A depth-0 tower: patch embed, position table and merger.
fn tower(patch_weight: mlxcel_core::UniquePtr<mlxcel_core::MlxArray>) -> WeightMap {
    let h = HIDDEN as i32;
    let mut identity = vec![0.0f32; HIDDEN * HIDDEN];
    for i in 0..HIDDEN {
        identity[i * HIDDEN + i] = 1.0;
    }
    let mut wm = WeightMap::new();
    wm.insert(PATCH_KEY.to_string(), patch_weight);
    insert(
        &mut wm,
        "vision_tower.patch_embed.proj.bias",
        &[0.0; HIDDEN],
        &[h],
    );
    insert(&mut wm, "vision_tower.pos_embed.weight", &identity, &[4, h]);
    insert(
        &mut wm,
        "vision_tower.merger.norm.weight",
        &[1.0; HIDDEN],
        &[h],
    );
    insert(
        &mut wm,
        "vision_tower.merger.norm.bias",
        &[0.0; HIDDEN],
        &[h],
    );
    for fc in ["linear_fc1", "linear_fc2"] {
        insert(
            &mut wm,
            &format!("vision_tower.merger.{fc}.weight"),
            &identity,
            &[h, h],
        );
        insert(
            &mut wm,
            &format!("vision_tower.merger.{fc}.bias"),
            &[0.0; HIDDEN],
            &[h],
        );
    }
    wm
}

/// The mlx-vlm `[out, kT, kH, kW, in]` patch weight, every element distinct.
fn mlx_layout() -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n = HIDDEN * T * P * P * IN_CHANNELS;
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 0.3).collect();
    mlxcel_core::from_slice_f32(
        &data,
        &[
            HIDDEN as i32,
            T as i32,
            P as i32,
            P as i32,
            IN_CHANNELS as i32,
        ],
    )
}

fn values(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

#[test]
fn hub_layout_and_mlx_layout_build_the_same_patch_embed() {
    let cfg = config();
    let mut mlx = tower(mlx_layout());
    // PyTorch's Conv3d order, [out, in, kT, kH, kW] = [4, 3, 2, 2, 2].
    let hub_weight = mlxcel_core::transpose_axes(&mlx_layout(), &[0, 4, 1, 2, 3]);
    assert_eq!(
        mlxcel_core::array_shape(&hub_weight),
        vec![
            HIDDEN as i32,
            IN_CHANNELS as i32,
            T as i32,
            P as i32,
            P as i32
        ]
    );
    let mut hub = tower(hub_weight);

    let from_mlx = load_qwen3_vl_vision_encoder(&mut mlx, &cfg, "vision_tower").expect("mlx");
    let from_hub = load_qwen3_vl_vision_encoder(&mut hub, &cfg, "vision_tower").expect("hub");
    let want = values(from_mlx.patch_embed_weight());
    assert_eq!(
        mlxcel_core::array_shape(from_hub.patch_embed_weight()),
        vec![HIDDEN as i32, (IN_CHANNELS * T * P * P) as i32]
    );
    assert_eq!(values(from_hub.patch_embed_weight()), want);
    // The guard converted the map entry itself.
    assert_eq!(
        mlxcel_core::array_shape(hub.get(PATCH_KEY).unwrap()),
        vec![
            HIDDEN as i32,
            T as i32,
            P as i32,
            P as i32,
            IN_CHANNELS as i32
        ]
    );
}

#[test]
fn an_already_normalized_layout_is_left_unchanged() {
    let cfg = config();
    let original = mlx_layout();
    let mut weights = tower(mlxcel_core::copy(&original));
    load_qwen3_vl_vision_encoder(&mut weights, &cfg, "vision_tower").expect("mlx layout loads");
    let kept = weights.get(PATCH_KEY).unwrap();
    assert_eq!(
        mlxcel_core::array_shape(kept),
        mlxcel_core::array_shape(&original)
    );
    assert_eq!(values(kept), values(&original));
}

/// Every loader that builds a Qwen3-VL tower: Qwen3-VL, Qwen3-VL-MoE, Qwen 3.5
/// / 3.8 and Qwen3-Omni in `vlm_qwen.rs`, Cohere Compass in
/// `vlm_cohere_compass.rs`. A new site that calls the encoder directly would
/// skip the layout guard, so none may.
#[test]
fn every_qwen3_vl_tower_site_goes_through_the_guard() {
    let sites = [
        ("vlm_qwen.rs", include_str!("vlm_qwen.rs"), 4),
        (
            "vlm_cohere_compass.rs",
            include_str!("vlm_cohere_compass.rs"),
            1,
        ),
    ];
    for (name, source, expected) in sites {
        assert!(
            !source.contains("Qwen3VLVisionEncoder::from_weights("),
            "{name} builds a Qwen3-VL tower without the Conv3d layout guard"
        );
        assert_eq!(
            source.matches("load_qwen3_vl_vision_encoder(&mut").count(),
            expected,
            "{name}: Qwen3-VL tower sites"
        );
    }
}
