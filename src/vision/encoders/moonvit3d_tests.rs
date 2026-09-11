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

use super::pos_emb::{DividedFixedPosEmb, bilinear_matrix, bilinear_resample, time_embedding};
use super::*;
use mlxcel_core::utils::array_to_vec_f32;

fn to_vec(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(a);
    array_to_vec_f32(a)
}

/// The closed form of the issue: `src = (i + 0.5) * in / out - 0.5` clipped,
/// two taps, rows then columns.
fn bilinear_closed_form(
    grid: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let tap = |i: usize, in_size: usize, out_size: usize| -> (usize, usize, f32) {
        let src = ((i as f32 + 0.5) * in_size as f32 / out_size as f32 - 0.5)
            .clamp(0.0, (in_size - 1) as f32);
        let i0 = src.floor() as usize;
        (i0, (i0 + 1).min(in_size - 1), src - i0 as f32)
    };
    let mut rows = vec![0.0f32; out_h * in_w];
    for y in 0..out_h {
        let (y0, y1, fy) = tap(y, in_h, out_h);
        for x in 0..in_w {
            rows[y * in_w + x] = (1.0 - fy) * grid[y0 * in_w + x] + fy * grid[y1 * in_w + x];
        }
    }
    let mut out = vec![0.0f32; out_h * out_w];
    for y in 0..out_h {
        for x in 0..out_w {
            let (x0, x1, fx) = tap(x, in_w, out_w);
            out[y * out_w + x] = (1.0 - fx) * rows[y * in_w + x0] + fx * rows[y * in_w + x1];
        }
    }
    out
}

#[test]
fn pos_emb_bilinear_matches_half_pixel_formula() {
    // A [4, 4, 1] ramp: value = 4 * y + x.
    let ramp: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let grid = mlxcel_core::from_slice_f32(&ramp, &[4, 4, 1]);

    // Downsample to [2, 2]: src = (i + 0.5) * 2 - 0.5 = 0.5, 2.5, so each
    // output is the mean of a 2x2 block: 2.5, 4.5, 10.5, 12.5.
    let out = to_vec(&bilinear_resample(&grid, 4, 4, 1, 2, 2));
    assert_eq!(out, vec![2.5, 4.5, 10.5, 12.5]);
    assert_eq!(out, bilinear_closed_form(&ramp, 4, 4, 2, 2));

    // Upsample to [8, 8]: src = (i + 0.5) / 2 - 0.5 = -0.25 (clipped to 0),
    // 0.25, 0.75, ..., 3.25 (clipped to 3).
    let out = to_vec(&bilinear_resample(&grid, 4, 4, 1, 8, 8));
    let expect = bilinear_closed_form(&ramp, 4, 4, 8, 8);
    assert_eq!(out.len(), 64);
    for (i, (a, b)) in out.iter().zip(&expect).enumerate() {
        assert!((a - b).abs() < 1e-5, "index {i}: {a} vs {b}");
    }
    // Corner and edge behaviour: (0, 0) clips to grid[0][0] = 0, (7, 7) to
    // grid[3][3] = 15, and (0, 1) sits at src_x = 0.25 -> 0.25.
    assert_eq!(out[0], 0.0);
    assert_eq!(out[63], 15.0);
    assert!((out[1] - 0.25).abs() < 1e-6);

    // Every interpolation row sums to one, so a constant grid is preserved.
    for in_size in [1, 3, 64] {
        for out_size in [1, 2, 7, 64, 100] {
            let m = bilinear_matrix(in_size, out_size);
            for o in 0..out_size as usize {
                let s: f32 = m[o * in_size as usize..(o + 1) * in_size as usize]
                    .iter()
                    .sum();
                assert!(
                    (s - 1.0).abs() < 1e-6,
                    "in {in_size} out {out_size} row {o}: {s}"
                );
            }
        }
    }

    // The same size is the identity, and the module returns [h*w, dim].
    let pe =
        DividedFixedPosEmb::from_array(mlxcel_core::from_slice_f32(&ramp, &[4, 4, 1]), 4, 4, 1);
    assert_eq!(to_vec(&pe.pos_for(MoonViT3DGrid::image(4, 4))), ramp);
    assert_eq!(
        to_vec(&pe.pos_for(MoonViT3DGrid::image(2, 2))),
        vec![2.5, 4.5, 10.5, 12.5]
    );
}

#[test]
fn time_embedding_sincos() {
    let dim = 1024;
    let table = to_vec(&time_embedding(3, dim));
    assert_eq!(table.len(), 3 * dim as usize);
    let half = (dim / 2) as usize;
    // time[0] == concat(0 x 512, 1 x 512).
    assert!(table[..half].iter().all(|&v| v == 0.0));
    assert!(table[half..dim as usize].iter().all(|&v| v == 1.0));
    // time[1][0] == sin(1 * omega_0) == sin(1); time[1][512] == cos(1).
    let row1 = &table[dim as usize..2 * dim as usize];
    assert!((row1[0] - 1.0f32.sin()).abs() < 1e-6);
    assert!((row1[half] - 1.0f32.cos()).abs() < 1e-6);
    // omega[j] = 10000^(-j / 512): time[2][j] == sin(2 * omega[j]).
    let j = 100usize;
    let omega = 10_000f32.powf(-(j as f32) / half as f32);
    let row2 = &table[2 * dim as usize..];
    assert!((row2[j] - (2.0 * omega).sin()).abs() < 1e-5);
    assert!((row2[half + j] - (2.0 * omega).cos()).abs() < 1e-5);

    // t > 1 tiles the spatial table and adds time[k] to frame k; t == 1 adds
    // nothing (frame 0 would otherwise pick up the cos(0) = 1 constant).
    let spatial: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect(); // [2, 2, 2]
    let pe =
        DividedFixedPosEmb::from_array(mlxcel_core::from_slice_f32(&spatial, &[2, 2, 2]), 2, 2, 2);
    let still = to_vec(&pe.pos_for(MoonViT3DGrid::image(2, 2)));
    assert_eq!(still, spatial);
    let clip = to_vec(&pe.pos_for(MoonViT3DGrid { t: 2, h: 2, w: 2 }));
    assert_eq!(clip.len(), 16);
    // dim 2 -> omega = [1]; time[0] = [0, 1], time[1] = [sin 1, cos 1].
    for (i, v) in spatial.iter().enumerate() {
        let expect0 = v + if i % 2 == 0 { 0.0 } else { 1.0 };
        assert!((clip[i] - expect0).abs() < 1e-6, "frame 0 index {i}");
        let expect1 = v + if i % 2 == 0 {
            1.0f32.sin()
        } else {
            1.0f32.cos()
        };
        assert!((clip[8 + i] - expect1).abs() < 1e-6, "frame 1 index {i}");
    }
}

#[test]
fn rope2d_angle_layout() {
    // head_dim 128 -> 32 frequencies f_j = 10000^(-4j / 128), 64 angles per
    // token alternating x * f_j, y * f_j. For h = 2, w = 3 the patch (y = 1,
    // x = 2) is token 1 * 3 + 2 = 5.
    let rope = Rope2DPosEmb::new(128);
    let angles = to_vec(&rope.angles(&[KimiMediaGrid::Image { h: 2, w: 3 }]));
    assert_eq!(angles.len(), 6 * 64);
    let token = &angles[5 * 64..6 * 64];
    for j in 0..32 {
        let f = 10_000f32.powf(-4.0 * j as f32 / 128.0);
        assert!(
            (token[2 * j] - 2.0 * f).abs() < 1e-5 * (1.0 + 2.0 * f),
            "x angle j={j}"
        );
        assert!(
            (token[2 * j + 1] - f).abs() < 1e-5 * (1.0 + f),
            "y angle j={j}"
        );
    }
    // Token 0 is patch (0, 0): every angle is zero.
    assert!(angles[..64].iter().all(|&a| a == 0.0));
    // A clip tiles the per-frame stream: frame 1's token 5 equals frame 0's.
    let tiled = to_vec(&rope.angles(&[KimiMediaGrid::Video { t: 2, h: 2, w: 3 }]));
    assert_eq!(tiled.len(), 12 * 64);
    assert_eq!(&tiled[11 * 64..12 * 64], token);

    // Applied as interleaved pairs: (e, o) -> (e cos - o sin, e sin + o cos).
    let (cos, sin) = rope.cos_sin(&[KimiMediaGrid::Image { h: 2, w: 3 }]);
    let mut q = vec![0.0f32; 6 * 128];
    q[5 * 128] = 1.0; // token 5, head 0, dim 0 (pair 0, even half)
    q[5 * 128 + 3] = 1.0; // pair 1, odd half
    let q = mlxcel_core::from_slice_f32(&q, &[6, 1, 128]);
    let (rotated, _) = apply_rope(&q, &q, &cos, &sin);
    let r = to_vec(&rotated);
    let a0 = token[0];
    let a1 = token[1];
    assert!((r[5 * 128] - a0.cos()).abs() < 1e-5);
    assert!((r[5 * 128 + 1] - a0.sin()).abs() < 1e-5);
    assert!((r[5 * 128 + 2] + a1.sin()).abs() < 1e-5);
    assert!((r[5 * 128 + 3] - a1.cos()).abs() < 1e-5);
}

#[test]
fn tpool_merge_averages_over_time_and_groups_2x2() {
    // t = 2, h = w = 4, dim = 1, value = 100 * t + 4 * y + x.
    let mut values = Vec::with_capacity(32);
    for t in 0..2 {
        for y in 0..4 {
            for x in 0..4 {
                values.push((100 * t + 4 * y + x) as f32);
            }
        }
    }
    let x = mlxcel_core::from_slice_f32(&values, &[32, 1]);
    let grid = MoonViT3DGrid { t: 2, h: 4, w: 4 };
    let merged = tpool_merge(&x, grid, (2, 2)).expect("merge");
    assert_eq!(mlxcel_core::array_shape(&merged), vec![4, 4, 1]);
    let out = to_vec(&merged);
    // Token (0, 0) groups (0,0) (0,1) (1,0) (1,1) averaged over t: +50.
    assert_eq!(&out[0..4], &[50.0, 51.0, 54.0, 55.0]);
    assert_eq!(&out[4..8], &[52.0, 53.0, 56.0, 57.0]);
    assert_eq!(&out[8..12], &[58.0, 59.0, 62.0, 63.0]);
    assert_eq!(&out[12..16], &[60.0, 61.0, 64.0, 65.0]);

    // A still image is the same grouping without the average.
    let still = mlxcel_core::from_slice_f32(&values[..16], &[16, 1]);
    let out = to_vec(&tpool_merge(&still, MoonViT3DGrid::image(4, 4), (2, 2)).unwrap());
    assert_eq!(&out[0..4], &[0.0, 1.0, 4.0, 5.0]);

    // Odd grids and wrong token counts are refused.
    assert!(tpool_merge(&still, MoonViT3DGrid::image(3, 4), (2, 2)).is_err());
    assert!(tpool_merge(&still, MoonViT3DGrid::image(4, 4), (3, 2)).is_err());
    assert!(tpool_merge(&still, MoonViT3DGrid::image(2, 4), (2, 2)).is_err());
}

#[test]
fn config_parses_published_fields_and_refuses_other_variants() {
    let published = serde_json::json!({
        "_attn_implementation": "flash_attention_2",
        "activation_func": "gelu_pytorch_tanh",
        "attn_bias": false,
        "init_pos_emb_height": 64,
        "init_pos_emb_time": 4,
        "init_pos_emb_width": 64,
        "linear_bias": false,
        "merge_kernel_size": [2, 2],
        "merge_type": "sd2_tpool",
        "mlp_type": "mlp2",
        "mm_hidden_size": 1024,
        "mm_projector_type": "patchmergerv2",
        "norm_type": "rmsnorm",
        "patch_embed_proj_bias": false,
        "patch_size": 14,
        "pos_emb_interpolation_mode": "bilinear",
        "pos_emb_type": "divided_fixed",
        "projector_hidden_act": "gelu",
        "projector_ln_eps": 1e-05,
        "qkv_hidden_size": 1536,
        "text_hidden_size": 7168,
        "vt_hidden_size": 1024,
        "vt_intermediate_size": 4096,
        "vt_num_attention_heads": 12,
        "vt_num_hidden_layers": 27
    });
    let cfg: MoonViT3DConfig = serde_json::from_value(published).unwrap();
    cfg.validate().unwrap();
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.merged_hidden(), 4096);
    assert_eq!(cfg.merge(), (2, 2));
    assert_eq!(MOONVIT3D_NORM_EPS, 2f32.powi(-7));

    let mut layernorm = cfg.clone();
    layernorm.norm_type = "layernorm".to_string();
    assert!(layernorm.validate().unwrap_err().contains("norm_type"));
    let mut heads = cfg.clone();
    heads.vt_num_attention_heads = 7;
    assert!(heads.validate().is_err());
    let mut bicubic = cfg;
    bicubic.pos_emb_interpolation_mode = "bicubic".to_string();
    assert!(
        bicubic
            .validate()
            .unwrap_err()
            .contains("pos_emb_interpolation_mode")
    );
}

/// Synthetic end-to-end shape check: a 2-block tower with tiny widths over
/// two images of different grids, then the projector.
#[test]
fn tower_runs_per_image_and_merges_to_projector_rows() {
    use mlxcel_core::weights::WeightMap;
    let cfg = MoonViT3DConfig {
        vt_num_hidden_layers: 2,
        vt_hidden_size: 8,
        mm_hidden_size: 8,
        vt_intermediate_size: 16,
        qkv_hidden_size: 8,
        vt_num_attention_heads: 2,
        init_pos_emb_height: 4,
        init_pos_emb_width: 4,
        patch_size: 2,
        text_hidden_size: 6,
        ..serde_json::from_value(serde_json::json!({})).unwrap()
    };
    cfg.validate().unwrap();
    let mut wm = WeightMap::new();
    let fill = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * scale)
            .collect()
    };
    let put = |wm: &mut WeightMap, key: &str, data: Vec<f32>, shape: &[i32]| {
        wm.insert(key.to_string(), mlxcel_core::from_slice_f32(&data, shape));
    };
    put(
        &mut wm,
        "vision_tower.patch_embed.proj.weight",
        fill(8 * 2 * 2 * 3, 0.05),
        &[8, 2, 2, 3],
    );
    put(
        &mut wm,
        "vision_tower.patch_embed.pos_emb.weight",
        fill(4 * 4 * 8, 0.1),
        &[4, 4, 8],
    );
    for i in 0..2 {
        let p = format!("vision_tower.encoder.blocks.{i}");
        put(&mut wm, &format!("{p}.norm0.weight"), vec![1.0; 8], &[8]);
        put(&mut wm, &format!("{p}.norm1.weight"), vec![1.0; 8], &[8]);
        put(
            &mut wm,
            &format!("{p}.wqkv.weight"),
            fill(24 * 8, 0.05),
            &[24, 8],
        );
        put(
            &mut wm,
            &format!("{p}.wo.weight"),
            fill(8 * 8, 0.05),
            &[8, 8],
        );
        put(
            &mut wm,
            &format!("{p}.mlp.fc0.weight"),
            fill(16 * 8, 0.05),
            &[16, 8],
        );
        put(
            &mut wm,
            &format!("{p}.mlp.fc1.weight"),
            fill(8 * 16, 0.05),
            &[8, 16],
        );
    }
    put(
        &mut wm,
        "vision_tower.encoder.final_layernorm.weight",
        vec![1.0; 8],
        &[8],
    );
    put(
        &mut wm,
        "mm_projector.proj.0.weight",
        fill(32 * 32, 0.02),
        &[32, 32],
    );
    put(
        &mut wm,
        "mm_projector.proj.2.weight",
        fill(6 * 32, 0.05),
        &[6, 32],
    );
    put(&mut wm, "mm_projector.post_norm.weight", vec![1.0; 6], &[6]);

    let tower = MoonViT3DVisionModel::from_weights(&wm, &cfg, "vision_tower").expect("tower");
    let projector = crate::vision::kimi_k3_vl::KimiK3VLModel::projector_from_weights(&wm, &cfg)
        .expect("projector");

    // Image A: grid (2, 4) = 8 patches; image B: grid (4, 2) = 8 patches.
    let grids = [MoonViT3DGrid::image(2, 4), MoonViT3DGrid::image(4, 2)];
    let patches = fill(16 * 2 * 2 * 3, 0.1);
    let patches = mlxcel_core::from_slice_f32(&patches, &[16, 2, 2, 3]);
    let per_image = tower.forward(&patches, &grids).expect("forward");
    assert_eq!(per_image.len(), 2);
    assert_eq!(mlxcel_core::array_shape(&per_image[0]), vec![8, 8]);
    let merged = tower.forward_merged(&patches, &grids).expect("merged");
    assert_eq!(mlxcel_core::array_shape(&merged), vec![4, 4, 8]);
    let projected = projector.forward(&merged).expect("projected");
    let out = to_vec(&projected);
    assert_eq!(mlxcel_core::array_shape(&projected), vec![4, 6]);
    assert!(out.iter().all(|v| v.is_finite()));

    // Per-image attention: image A's rows do not change when image B's
    // patches change.
    let mut altered = fill(16 * 2 * 2 * 3, 0.1);
    for v in &mut altered[8 * 12..] {
        *v += 1.0;
    }
    let altered = mlxcel_core::from_slice_f32(&altered, &[16, 2, 2, 3]);
    let again = tower.forward(&altered, &grids).expect("forward");
    assert_eq!(to_vec(&per_image[0]), to_vec(&again[0]));
    assert_ne!(to_vec(&per_image[1]), to_vec(&again[1]));

    let bad = mlxcel_core::from_slice_f32(&fill(15 * 12, 0.1), &[15, 2, 2, 3]);
    assert!(tower.forward(&bad, &grids).is_err());
}
