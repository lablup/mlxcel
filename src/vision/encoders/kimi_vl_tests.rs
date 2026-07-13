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

//! Pure-numeric parity/unit tests for the MoonViT encoder pieces. These run
//! without any checkpoint: every expected value is derived by hand from the
//! upstream reference math.

use super::pos_emb::{Learnable2DInterpPosEmb, temporal_sinusoid};
use super::rope::{Rope2DPosEmb, apply_rope};
use super::{KimiMediaGrid, KimiVLVisionConfig, KimiVLVisionModel, cu_seqlens, patch_merger};
use mlxcel_core::weights::WeightMap;

fn to_vec(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::eval(a);
    let n: i32 = mlxcel_core::array_shape(a).iter().product();
    let flat = mlxcel_core::reshape(a, &[n]);
    (0..n)
        .map(|i| {
            let s = mlxcel_core::slice(&flat, &[i], &[i + 1]);
            mlxcel_core::item_f32(&s)
        })
        .collect()
}

fn assert_close(a: f32, b: f32, msg: &str) {
    assert!((a - b).abs() < 1e-4, "{msg}: {a} vs {b}");
}

#[test]
fn rope_cos_sin_matches_reference_2x2() {
    // dim = 4 -> dim/4 = 1 base frequency, freq[0] = theta^0 = 1.0.
    // Grid (2, 2), row-major tokens (row y, col x): (0,0),(0,1),(1,0),(1,1).
    // angle[t] = [x_angle=col*1, y_angle=row*1]  (interleaved x then y).
    let rope = Rope2DPosEmb::new(4);
    let (cos, sin) = rope.cos_sin(&[KimiMediaGrid::Image { h: 2, w: 2 }]);
    assert_eq!(mlxcel_core::array_shape(&cos), vec![4, 2]);

    let cos = to_vec(&cos); // [t0(x,y), t1, t2, t3]
    let sin = to_vec(&sin);
    let c1 = 1.0f32.cos();
    let s1 = 1.0f32.sin();

    // token 0 (col 0, row 0): angles [0, 0] -> cos [1,1], sin [0,0]
    assert_close(cos[0], 1.0, "cos t0 x");
    assert_close(cos[1], 1.0, "cos t0 y");
    // token 1 (col 1, row 0): angles [1, 0] -> cos [cos1, 1], sin [sin1, 0]
    assert_close(cos[2], c1, "cos t1 x");
    assert_close(cos[3], 1.0, "cos t1 y");
    assert_close(sin[2], s1, "sin t1 x");
    assert_close(sin[3], 0.0, "sin t1 y");
    // token 2 (col 0, row 1): angles [0, 1] -> cos [1, cos1]
    assert_close(cos[4], 1.0, "cos t2 x");
    assert_close(cos[5], c1, "cos t2 y");
    // token 3 (col 1, row 1): angles [1, 1] -> cos [cos1, cos1]
    assert_close(cos[6], c1, "cos t3 x");
    assert_close(cos[7], c1, "cos t3 y");
}

#[test]
fn apply_rope_interleaved_rotation() {
    // q = [1,2,3,4]; pair 0 rotated by 90deg (cos0=0, sin0=1) -> [-2, 1];
    // pair 1 rotated by 0deg (cos1=1, sin1=0) -> [3, 4].
    let q = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let cos = mlxcel_core::from_slice_f32(&[0.0, 1.0], &[1, 2]);
    let sin = mlxcel_core::from_slice_f32(&[1.0, 0.0], &[1, 2]);
    let (out, _) = apply_rope(&q, &q, &cos, &sin);
    let out = to_vec(&out);
    assert_close(out[0], -2.0, "out0");
    assert_close(out[1], 1.0, "out1");
    assert_close(out[2], 3.0, "out2");
    assert_close(out[3], 4.0, "out3");
}

#[test]
fn patch_merger_2x2_groups_neighbours() {
    // A single (2,2) image, dim=1, patch values 0..4 in row-major order.
    // A 2x2 merge collapses to one token carrying [0,1,2,3] (kh,kw order).
    let x = mlxcel_core::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[4, 1]);
    let out = patch_merger(&x, &[KimiMediaGrid::Image { h: 2, w: 2 }], 2);
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 4, 1]);
    assert_eq!(to_vec(&out), vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn bicubic_reproduces_constant_and_exact_grid() {
    // Constant grid must resample to the same constant (cubic weights sum to 1).
    let dim = 2i32;
    let weight = mlxcel_core::from_slice_f32(&[5.0; 8], &[2, 2, dim]); // [2,2,2] all 5.0
    let emb = Learnable2DInterpPosEmb::from_array(mlxcel_core::copy(&weight), 2, 2, dim);

    // Exact-grid fast path: pos_for(2,2) is the flattened weight.
    let exact = emb.pos_for(2, 2);
    assert_eq!(mlxcel_core::array_shape(&exact), vec![4, dim]);
    for v in to_vec(&exact) {
        assert_close(v, 5.0, "exact");
    }

    // Interpolated path 2 -> 3: partition of unity keeps the constant.
    let up = emb.pos_for(3, 3);
    assert_eq!(mlxcel_core::array_shape(&up), vec![9, dim]);
    for v in to_vec(&up) {
        assert_close(v, 5.0, "bicubic constant");
    }
}

fn tiny_config() -> KimiVLVisionConfig {
    KimiVLVisionConfig {
        model_type: "moonvit".to_string(),
        depth: 1,
        embed_dim: 4,
        hidden_size: 4,
        num_heads: 1, // head_dim = 4 (divisible by 4 for the 2D rope)
        patch_size: 1,
        num_channels: 2,
        intermediate_size: 8,
        init_pos_emb_height: 2,
        init_pos_emb_width: 2,
        spatial_merge_size: 2,
        temporal_patch_size: 2,
        layer_norm_eps: 1e-6,
        quant_group_size: 0,
        quant_bits: 0,
    }
}

fn insert(wm: &mut WeightMap, key: &str, data: &[f32], shape: &[i32]) {
    wm.insert(key.to_string(), mlxcel_core::from_slice_f32(data, shape));
}

#[test]
fn encoder_forward_smoke_synthetic_weights() {
    // A single (2,2) image = 4 patches, patch_size 1, C 2, embed_dim 4.
    let cfg = tiny_config();
    let p = "vision_tower";
    let mut wm = WeightMap::new();

    // Conv patch embed weight [embed_dim, p, p, C] = [4,1,1,2], bias [4].
    insert(
        &mut wm,
        &format!("{p}.patch_embed.proj.weight"),
        &[0.1; 8],
        &[4, 1, 1, 2],
    );
    insert(
        &mut wm,
        &format!("{p}.patch_embed.proj.bias"),
        &[0.0; 4],
        &[4],
    );
    // Learned pos emb grid [2,2,4].
    insert(
        &mut wm,
        &format!("{p}.patch_embed.pos_emb.weight"),
        &[0.0; 16],
        &[2, 2, 4],
    );
    // Block 0 norms.
    for norm in ["norm0", "norm1"] {
        insert(
            &mut wm,
            &format!("{p}.blocks.0.{norm}.weight"),
            &[1.0; 4],
            &[4],
        );
        insert(
            &mut wm,
            &format!("{p}.blocks.0.{norm}.bias"),
            &[0.0; 4],
            &[4],
        );
    }
    // Attention wqkv [3*embed_dim, embed_dim] = [12,4], wo [4,4].
    insert(
        &mut wm,
        &format!("{p}.blocks.0.attn.wqkv.weight"),
        &[0.1; 48],
        &[12, 4],
    );
    insert(
        &mut wm,
        &format!("{p}.blocks.0.attn.wqkv.bias"),
        &[0.0; 12],
        &[12],
    );
    insert(
        &mut wm,
        &format!("{p}.blocks.0.attn.wo.weight"),
        &[0.1; 16],
        &[4, 4],
    );
    insert(
        &mut wm,
        &format!("{p}.blocks.0.attn.wo.bias"),
        &[0.0; 4],
        &[4],
    );
    // MLP fc0 [8,4], fc1 [4,8].
    insert(
        &mut wm,
        &format!("{p}.blocks.0.mlp.fc0.weight"),
        &[0.1; 32],
        &[8, 4],
    );
    insert(
        &mut wm,
        &format!("{p}.blocks.0.mlp.fc0.bias"),
        &[0.0; 8],
        &[8],
    );
    insert(
        &mut wm,
        &format!("{p}.blocks.0.mlp.fc1.weight"),
        &[0.1; 32],
        &[4, 8],
    );
    insert(
        &mut wm,
        &format!("{p}.blocks.0.mlp.fc1.bias"),
        &[0.0; 4],
        &[4],
    );
    // Final layer norm.
    insert(
        &mut wm,
        &format!("{p}.final_layernorm.weight"),
        &[1.0; 4],
        &[4],
    );
    insert(
        &mut wm,
        &format!("{p}.final_layernorm.bias"),
        &[0.0; 4],
        &[4],
    );

    let model = KimiVLVisionModel::from_weights(&wm, &cfg, p).expect("build encoder");

    // pixel_values [num_patches, p, p, C] = [4, 1, 1, 2].
    let pixels = mlxcel_core::from_slice_f32(&[0.5; 8], &[4, 1, 1, 2]);
    let out = model.forward_with_grid(&pixels, &[KimiMediaGrid::Image { h: 2, w: 2 }]);
    // (2,2) grid, 2x2 merge -> 1 merged token of [kh*kw=4, dim=4].
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 4, 4]);

    let values = to_vec(&out);
    assert!(
        values.iter().all(|v| v.is_finite()),
        "encoder output must be finite"
    );
}

// ── Video (3D MoonViT) tests ────────────────────────────────────────────────

#[test]
fn temporal_sinusoid_shape_row0_and_closed_form() {
    // Real embed_dim so the col split lands exactly on 576.
    let dim = 1152i32;
    let half = dim / 2; // 576
    let t = 3i32;
    let table = temporal_sinusoid(t, dim);
    assert_eq!(mlxcel_core::array_shape(&table), vec![t, dim]);

    let v = to_vec(&table); // row-major [t, dim]
    // Row 0: sin(0)=0 for cols 0..half, cos(0)=1 for cols half..dim.
    for c in 0..half {
        assert_close(v[c as usize], 0.0, "row0 sin col");
    }
    for c in half..dim {
        assert_close(v[c as usize], 1.0, "row0 cos col");
    }

    // Spot-check frame f=2 against the closed form for a few columns j.
    let ln_theta = 10_000.0f32.ln();
    let f = 2i32;
    let base = (f * dim) as usize;
    for j in [0i32, 1, 100, 575] {
        let freq = (-ln_theta * j as f32 / half as f32).exp();
        let ang = f as f32 * freq;
        assert_close(v[base + j as usize], ang.sin(), "sin closed form");
        assert_close(v[base + (half + j) as usize], ang.cos(), "cos closed form");
    }
}

#[test]
fn pos_emb_video_t1_equals_image_plus_frame0_temporal() {
    // Video table for grid (1, h, w) == image table for (h, w) plus the frame-0
    // temporal row broadcast to all h*w rows.
    let dim = 8i32;
    let (ih, iw) = (2i32, 2i32);
    let grid_vals: Vec<f32> = (0..(ih * iw * dim)).map(|i| i as f32 * 0.1).collect();
    let weight = mlxcel_core::from_slice_f32(&grid_vals, &[ih, iw, dim]);
    let emb = Learnable2DInterpPosEmb::from_array(mlxcel_core::copy(&weight), ih, iw, dim);

    let (h, w) = (2i32, 2i32);
    // Recover each position table as add_to_media(zeros).
    let zeros_img = mlxcel_core::zeros(&[h * w, dim], mlxcel_core::dtype::FLOAT32);
    let image_tbl = emb.add_to_media(&zeros_img, &[KimiMediaGrid::Image { h, w }]);
    let zeros_vid = mlxcel_core::zeros(&[h * w, dim], mlxcel_core::dtype::FLOAT32);
    let video_tbl = emb.add_to_media(&zeros_vid, &[KimiMediaGrid::Video { t: 1, h, w }]);

    let frame0 = to_vec(&temporal_sinusoid(1, dim)); // dim values (row 0)
    let img = to_vec(&image_tbl);
    let vid = to_vec(&video_tbl);
    assert_eq!(img.len(), (h * w * dim) as usize);
    for r in 0..(h * w) as usize {
        let base = r * dim as usize;
        for (c, &f0) in frame0.iter().enumerate() {
            let idx = base + c;
            assert_close(
                vid[idx],
                img[idx] + f0,
                "video(1,h,w) == image + frame0 temporal",
            );
        }
    }
}

#[test]
fn rope_video_table_tiles_image_table() {
    // head_dim=4 -> dim/2 = 2 angle columns. The (t*h*w, 2) video table equals
    // the (h*w, 2) image table repeated t times along axis 0.
    let rope = Rope2DPosEmb::new(4);
    let (h, w) = (2i32, 3i32);
    let t = 3i32;
    let half = 2i32; // head_dim/2

    let (cos_img, sin_img) = rope.cos_sin(&[KimiMediaGrid::Image { h, w }]);
    let (cos_vid, sin_vid) = rope.cos_sin(&[KimiMediaGrid::Video { t, h, w }]);
    assert_eq!(mlxcel_core::array_shape(&cos_vid), vec![t * h * w, half]);

    let ci = to_vec(&cos_img);
    let si = to_vec(&sin_img);
    let cv = to_vec(&cos_vid);
    let sv = to_vec(&sin_vid);
    let per = (h * w * half) as usize;
    for f in 0..t as usize {
        for k in 0..per {
            assert_close(cv[f * per + k], ci[k], "cos tile");
            assert_close(sv[f * per + k], si[k], "sin tile");
        }
    }
}

#[test]
fn merger_temporal_pool_is_frame_mean_then_merge() {
    // dim=1, grid (2,2), t=2 with distinct per-frame values.
    // frame-major packing: frame0=[0,1,2,3], frame1=[10,11,12,13].
    let dim = 1i32;
    let (h, w) = (2i32, 2i32);
    let t = 2i32;
    let data: Vec<f32> = vec![0., 1., 2., 3., 10., 11., 12., 13.];
    let x = mlxcel_core::from_slice_f32(&data, &[t * h * w, dim]);

    let out = patch_merger(&x, &[KimiMediaGrid::Video { t, h, w }], 2);
    // Output shape ((h/2)*(w/2), 4, dim) for any t.
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 4, dim]);
    // Frame mean [5,6,7,8], then the 2x2 merge groups them into one token.
    assert_eq!(to_vec(&out), vec![5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn cu_seqlens_mixes_video_and_image_segments() {
    // Media order: video (t=3,2,2)=12, image (2,2)=4, video (t=2,3,3)=18.
    let grids = [
        KimiMediaGrid::Video { t: 3, h: 2, w: 2 },
        KimiMediaGrid::Image { h: 2, w: 2 },
        KimiMediaGrid::Video { t: 2, h: 3, w: 3 },
    ];
    assert_eq!(cu_seqlens(&grids), vec![0, 12, 16, 34]);
}

/// Build the tiny synthetic MoonViT encoder weights with a configurable
/// patch-embed bias and a non-trivial learned position grid.
fn tiny_encoder_weights(p: &str, proj_bias: &[f32]) -> WeightMap {
    let mut wm = WeightMap::new();
    insert(
        &mut wm,
        &format!("{p}.patch_embed.proj.weight"),
        &[0.1; 8],
        &[4, 1, 1, 2],
    );
    insert(
        &mut wm,
        &format!("{p}.patch_embed.proj.bias"),
        proj_bias,
        &[4],
    );
    // Non-trivial learned pos grid [2,2,4].
    let pos: Vec<f32> = (0..16).map(|i| i as f32 * 0.01).collect();
    insert(
        &mut wm,
        &format!("{p}.patch_embed.pos_emb.weight"),
        &pos,
        &[2, 2, 4],
    );
    for norm in ["norm0", "norm1"] {
        insert(
            &mut wm,
            &format!("{p}.blocks.0.{norm}.weight"),
            &[1.0; 4],
            &[4],
        );
        insert(
            &mut wm,
            &format!("{p}.blocks.0.{norm}.bias"),
            &[0.0; 4],
            &[4],
        );
    }
    insert(
        &mut wm,
        &format!("{p}.blocks.0.attn.wqkv.weight"),
        &[0.1; 48],
        &[12, 4],
    );
    insert(
        &mut wm,
        &format!("{p}.blocks.0.attn.wqkv.bias"),
        &[0.0; 12],
        &[12],
    );
    insert(
        &mut wm,
        &format!("{p}.blocks.0.attn.wo.weight"),
        &[0.1; 16],
        &[4, 4],
    );
    insert(
        &mut wm,
        &format!("{p}.blocks.0.attn.wo.bias"),
        &[0.0; 4],
        &[4],
    );
    insert(
        &mut wm,
        &format!("{p}.blocks.0.mlp.fc0.weight"),
        &[0.1; 32],
        &[8, 4],
    );
    insert(
        &mut wm,
        &format!("{p}.blocks.0.mlp.fc0.bias"),
        &[0.0; 8],
        &[8],
    );
    insert(
        &mut wm,
        &format!("{p}.blocks.0.mlp.fc1.weight"),
        &[0.1; 32],
        &[4, 8],
    );
    insert(
        &mut wm,
        &format!("{p}.blocks.0.mlp.fc1.bias"),
        &[0.0; 4],
        &[4],
    );
    insert(
        &mut wm,
        &format!("{p}.final_layernorm.weight"),
        &[1.0; 4],
        &[4],
    );
    insert(
        &mut wm,
        &format!("{p}.final_layernorm.bias"),
        &[0.0; 4],
        &[4],
    );
    wm
}

#[test]
fn video_t1_equals_image_with_frame0_temporal_folded_into_bias() {
    // A t=1 clip through the full tower equals the image-path output once the
    // frame-0 temporal constant is accounted for. We account for it by folding
    // the frame-0 temporal row into the image model's patch-embed bias, so both
    // block inputs are identical (addition is commutative). Token counts and
    // every output value must then match exactly.
    let cfg = tiny_config();
    let p = "vision_tower";
    let dim = cfg.embed_dim as i32; // 4

    let frame0 = to_vec(&temporal_sinusoid(1, dim)); // [dim]
    let base_bias = vec![0.0f32; dim as usize];
    let img_bias: Vec<f32> = base_bias.iter().zip(&frame0).map(|(b, t)| b + t).collect();

    let video_model =
        KimiVLVisionModel::from_weights(&tiny_encoder_weights(p, &base_bias), &cfg, p).unwrap();
    let image_model =
        KimiVLVisionModel::from_weights(&tiny_encoder_weights(p, &img_bias), &cfg, p).unwrap();

    let pixels = mlxcel_core::from_slice_f32(&[0.5; 8], &[4, 1, 1, 2]);
    let vid_out =
        video_model.forward_with_grid(&pixels, &[KimiMediaGrid::Video { t: 1, h: 2, w: 2 }]);
    let img_out = image_model.forward_with_grid(&pixels, &[KimiMediaGrid::Image { h: 2, w: 2 }]);

    assert_eq!(
        mlxcel_core::array_shape(&vid_out),
        mlxcel_core::array_shape(&img_out),
        "token counts must match exactly"
    );
    let v = to_vec(&vid_out);
    let i = to_vec(&img_out);
    for (a, b) in v.iter().zip(i.iter()) {
        assert_close(*a, *b, "t=1 video == image + frame0 temporal");
    }
}
