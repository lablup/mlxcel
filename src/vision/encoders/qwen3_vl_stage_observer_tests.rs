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

use super::{
    PatchEmbed, PositionEmbedding, Qwen3VLVisionConfig, Qwen3VLVisionEncoder, Qwen3VLVisionStage,
};
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;

fn insert(wm: &mut WeightMap, key: &str, data: &[f32], shape: &[i32]) {
    wm.insert(key.to_string(), mlxcel_core::from_slice_f32(data, shape));
}

fn cast_weight(wm: &mut WeightMap, key: &str, dtype: i32) {
    let casted = mlxcel_core::astype(
        wm.get(key)
            .unwrap_or_else(|| panic!("missing tiny weight {key}")),
        dtype,
    );
    wm.insert(key.to_string(), casted);
}

fn tiny_config(depth: usize) -> Qwen3VLVisionConfig {
    Qwen3VLVisionConfig {
        depth,
        hidden_size: 4,
        intermediate_size: 8,
        out_hidden_size: 4,
        num_heads: 1,
        patch_size: 1,
        spatial_merge_size: 1,
        temporal_patch_size: 1,
        in_channels: 1,
        num_position_embeddings: 4,
        deepstack_visual_indexes: (0..depth).collect(),
        quant_group_size: 0,
        quant_bits: 0,
    }
}

fn identity(rows: usize, cols: usize) -> Vec<f32> {
    let mut data = vec![0.0; rows * cols];
    for i in 0..rows.min(cols) {
        data[i * cols + i] = 1.0;
    }
    data
}

fn tiny_weights(cfg: &Qwen3VLVisionConfig, prefix: &str) -> WeightMap {
    let mut wm = WeightMap::new();
    insert(
        &mut wm,
        &format!("{prefix}.patch_embed.proj.weight"),
        &[1.0, 0.5, -0.25, 0.125],
        &[cfg.hidden_size as i32, 1],
    );
    insert(
        &mut wm,
        &format!("{prefix}.patch_embed.proj.bias"),
        &[0.0; 4],
        &[cfg.hidden_size as i32],
    );
    insert(
        &mut wm,
        &format!("{prefix}.pos_embed.weight"),
        &identity(4, 4),
        &[4, cfg.hidden_size as i32],
    );

    for layer in 0..cfg.depth {
        let block = format!("{prefix}.blocks.{layer}");
        for norm in ["norm1", "norm2"] {
            insert(
                &mut wm,
                &format!("{block}.{norm}.weight"),
                &[1.0; 4],
                &[cfg.hidden_size as i32],
            );
            insert(
                &mut wm,
                &format!("{block}.{norm}.bias"),
                &[0.0; 4],
                &[cfg.hidden_size as i32],
            );
        }
        insert(
            &mut wm,
            &format!("{block}.attn.qkv.weight"),
            &[0.0; 48],
            &[(cfg.hidden_size * 3) as i32, cfg.hidden_size as i32],
        );
        insert(
            &mut wm,
            &format!("{block}.attn.qkv.bias"),
            &[0.0; 12],
            &[(cfg.hidden_size * 3) as i32],
        );
        insert(
            &mut wm,
            &format!("{block}.attn.proj.weight"),
            &[0.0; 16],
            &[cfg.hidden_size as i32, cfg.hidden_size as i32],
        );
        insert(
            &mut wm,
            &format!("{block}.attn.proj.bias"),
            &[0.0; 4],
            &[cfg.hidden_size as i32],
        );
        insert(
            &mut wm,
            &format!("{block}.mlp.linear_fc1.weight"),
            &[0.0; 32],
            &[cfg.intermediate_size as i32, cfg.hidden_size as i32],
        );
        insert(
            &mut wm,
            &format!("{block}.mlp.linear_fc1.bias"),
            &[0.0; 8],
            &[cfg.intermediate_size as i32],
        );
        insert(
            &mut wm,
            &format!("{block}.mlp.linear_fc2.weight"),
            &[0.0; 32],
            &[cfg.hidden_size as i32, cfg.intermediate_size as i32],
        );
        insert(
            &mut wm,
            &format!("{block}.mlp.linear_fc2.bias"),
            &[0.0; 4],
            &[cfg.hidden_size as i32],
        );
    }

    for merger in std::iter::once(format!("{prefix}.merger")).chain(
        (0..cfg.deepstack_visual_indexes.len())
            .map(|i| format!("{prefix}.deepstack_merger_list.{i}")),
    ) {
        insert(
            &mut wm,
            &format!("{merger}.norm.weight"),
            &[1.0; 4],
            &[cfg.hidden_size as i32],
        );
        insert(
            &mut wm,
            &format!("{merger}.norm.bias"),
            &[0.0; 4],
            &[cfg.hidden_size as i32],
        );
        insert(
            &mut wm,
            &format!("{merger}.linear_fc1.weight"),
            &identity(4, 4),
            &[cfg.hidden_size as i32, cfg.hidden_size as i32],
        );
        insert(
            &mut wm,
            &format!("{merger}.linear_fc1.bias"),
            &[0.0; 4],
            &[cfg.hidden_size as i32],
        );
        insert(
            &mut wm,
            &format!("{merger}.linear_fc2.weight"),
            &identity(4, 4),
            &[cfg.out_hidden_size as i32, cfg.hidden_size as i32],
        );
        insert(
            &mut wm,
            &format!("{merger}.linear_fc2.bias"),
            &[0.0; 4],
            &[cfg.out_hidden_size as i32],
        );
    }

    wm
}

fn as_f32_vec(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "mismatch at {idx}: got {g}, want {w}, all got={got:?}, want={want:?}"
        );
    }
}

fn bilinear_align_corners_false_reference(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(out_h * out_w);
    for oh in 0..out_h {
        let ih_f = ((oh as f32 + 0.5) * in_h as f32 / out_h as f32 - 0.5).max(0.0);
        let ih0 = ih_f.floor() as usize;
        let ih1 = (ih0 + 1).min(in_h - 1);
        let wh = ih_f - ih0 as f32;
        for ow in 0..out_w {
            let iw_f = ((ow as f32 + 0.5) * in_w as f32 / out_w as f32 - 0.5).max(0.0);
            let iw0 = iw_f.floor() as usize;
            let iw1 = (iw0 + 1).min(in_w - 1);
            let ww = iw_f - iw0 as f32;
            let top = src[ih0 * in_w + iw0] * (1.0 - ww) + src[ih0 * in_w + iw1] * ww;
            let bottom = src[ih1 * in_w + iw0] * (1.0 - ww) + src[ih1 * in_w + iw1] * ww;
            out.push(top * (1.0 - wh) + bottom * wh);
        }
    }
    out
}

#[test]
fn position_interpolation_keeps_fractional_weights_in_float32() {
    let cfg = Qwen3VLVisionConfig {
        hidden_size: 1,
        num_position_embeddings: 4,
        ..tiny_config(0)
    };
    let mut wm = WeightMap::new();
    let bf16_weight = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[4, 1]),
        dtype::BFLOAT16,
    );
    wm.insert("pos.weight".to_string(), bf16_weight);
    let pos = PositionEmbedding::from_weights(&wm, "pos", &cfg).expect("position embedding");

    let got = pos.fast_pos_embed_interpolate(&[(1, 4, 4)], 1);
    assert_eq!(mlxcel_core::array_dtype(&got), dtype::FLOAT32);
    let values = as_f32_vec(&got);
    assert_close(&[values[1]], &[1.0 / 3.0], 1e-5);
}

#[test]
fn position_interpolation_uses_transformers_align_corners_true_convention() {
    let mut wm = WeightMap::new();
    insert(
        &mut wm,
        "pos.weight",
        &[0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        &[9, 1],
    );
    let cfg = Qwen3VLVisionConfig {
        num_position_embeddings: 9,
        hidden_size: 1,
        ..tiny_config(0)
    };
    let pos = PositionEmbedding::from_weights(&wm, "pos", &cfg).expect("position embedding");

    let out = pos.fast_pos_embed_interpolate(&[(1, 2, 4)], 1);
    let got = as_f32_vec(&out);
    let align_corners_true = [
        0.0,
        2.0 / 3.0,
        4.0 / 3.0,
        2.0,
        20.0,
        20.0 + 2.0 / 3.0,
        21.0 + 1.0 / 3.0,
        22.0,
    ];
    assert_close(&got, &align_corners_true, 1e-5);

    let align_corners_false = bilinear_align_corners_false_reference(
        &[0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        3,
        3,
        2,
        4,
    );
    let max_false_gap = got
        .iter()
        .zip(align_corners_false.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_false_gap > 2.0,
        "the Qwen3-VL/Compass transformer source uses align_corners=True; the false-convention fixture should not be accidentally equal"
    );
}

#[test]
fn patch_embed_flattens_conv3d_weights_in_processor_temporal_channel_order() {
    let mut wm = WeightMap::new();
    insert(
        &mut wm,
        "patch.proj.weight",
        &[1.0, 2.0, 10.0, 20.0],
        &[1, 2, 1, 1, 2],
    );
    insert(&mut wm, "patch.proj.bias", &[0.0], &[1]);
    let cfg = Qwen3VLVisionConfig {
        hidden_size: 1,
        in_channels: 2,
        temporal_patch_size: 2,
        patch_size: 1,
        ..tiny_config(0)
    };
    let patch = PatchEmbed::from_weights(&wm, &cfg, "patch").expect("patch embed");
    let input = mlxcel_core::from_slice_f32(&[2.0, 3.0, 5.0, 7.0], &[2, 2]);
    let got = as_f32_vec(&patch.forward(&input));

    // The Rust Qwen image processor emits each flattened patch in T,C,H,W order.
    // Keep the MLX `[out, T, H, W, C]` kernel paired with that input layout.
    assert_close(&got, &[198.0], 1e-6);
}

#[test]
fn observed_forward_casts_position_to_patch_dtype_before_residual_add() {
    let cfg = tiny_config(0);
    let mut weights = tiny_weights(&cfg, "vision_tower");
    cast_weight(
        &mut weights,
        "vision_tower.patch_embed.proj.weight",
        dtype::BFLOAT16,
    );
    cast_weight(
        &mut weights,
        "vision_tower.patch_embed.proj.bias",
        dtype::BFLOAT16,
    );
    cast_weight(
        &mut weights,
        "vision_tower.pos_embed.weight",
        dtype::BFLOAT16,
    );
    let encoder =
        Qwen3VLVisionEncoder::from_weights(&weights, &cfg, "vision_tower").expect("tiny encoder");
    let input = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1]),
        dtype::BFLOAT16,
    );

    let mut patch_dtype = None;
    let mut pos_dtype = None;
    let mut after_add_dtype = None;
    let output = encoder.forward_with_grid_observer(&input, &[(1, 2, 2)], |stage, tensor| {
        let tensor_dtype = mlxcel_core::array_dtype(tensor);
        match stage {
            Qwen3VLVisionStage::PatchEmbed => patch_dtype = Some(tensor_dtype),
            Qwen3VLVisionStage::PositionEmbedding => pos_dtype = Some(tensor_dtype),
            Qwen3VLVisionStage::AfterPositionEmbedding => after_add_dtype = Some(tensor_dtype),
            _ => {}
        }
    });
    mlxcel_core::eval(&output.hidden_states);

    assert_eq!(pos_dtype, Some(dtype::FLOAT32));
    assert_eq!(after_add_dtype, patch_dtype);
    assert_ne!(patch_dtype, Some(dtype::FLOAT32));
}

#[test]
fn observed_forward_reports_all_tower_stages_without_production_path_flags() {
    let cfg = tiny_config(2);
    let weights = tiny_weights(&cfg, "vision_tower");
    let encoder =
        Qwen3VLVisionEncoder::from_weights(&weights, &cfg, "vision_tower").expect("tiny encoder");
    let input = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1]);
    let mut stages = Vec::new();
    let output = encoder.forward_with_grid_observer(&input, &[(1, 2, 2)], |stage, tensor| {
        mlxcel_core::eval(tensor);
        stages.push((stage.label(), mlxcel_core::array_shape(tensor)));
    });
    mlxcel_core::eval(&output.hidden_states);

    let names: Vec<&str> = stages.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "input_patches",
            "patch_embed",
            "position_embedding",
            "after_position_embedding",
            "rotary_position_embedding",
            "block_00_attention",
            "block_00_after_attention",
            "block_00_mlp",
            "block_00_output",
            "deepstack_0_after_block_00",
            "block_01_attention",
            "block_01_after_attention",
            "block_01_mlp",
            "block_01_output",
            "deepstack_1_after_block_01",
            "pre_merger",
            "post_merger",
        ]
    );
    assert!(stages.iter().all(|(_, shape)| !shape.is_empty()));
    assert_eq!(mlxcel_core::array_shape(&output.hidden_states), vec![4, 4]);
    assert_eq!(output.deepstack_features.len(), 2);
}
