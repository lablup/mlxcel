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

//! dots.ocr vision-tower synthetic parity checks.
//!
//! Builds a small `DotsVisionEncoder` from a deterministic in-memory weight map
//! (no checkpoint) and runs the full patch-embed -> blocks -> post-trunk-norm ->
//! merger pipeline, exercising the pieces unique to `dots_vit`: the OHWI patch
//! projection normalization, RMSNorm blocks, the SwiGLU vision MLP, block-
//! diagonal `cu_seqlens` attention, and the 2x2 merger grouping. A non-square
//! grid (`h != w`) pins down the row-order / axis handling. The real-checkpoint
//! numerics are validated by end-to-end OCR generation.

use mlxcel::vision::encoders::dots_ocr::{DotsVisionConfig, DotsVisionEncoder};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

fn arr(shape: &[i32], seed: i32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (((i * 5 + seed) % 11) as f32 / 11.0 - 0.5) * 0.1)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn small_config() -> DotsVisionConfig {
    DotsVisionConfig {
        embed_dim: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        patch_size: 2,
        temporal_patch_size: 1,
        spatial_merge_size: 2,
        rms_norm_eps: 1e-5,
        post_norm: true,
        quant_group_size: 0,
        quant_bits: 0,
    }
}

fn build_weights(cfg: &DotsVisionConfig) -> WeightMap {
    let mut w = WeightMap::new();
    let mut put = |name: String, a: UniquePtr<MlxArray>| {
        w.insert(name, a);
    };
    let dim = cfg.embed_dim;
    let feat = 3 * cfg.patch_size * cfg.patch_size; // 12
    let qkv = 3 * dim;
    let inter = cfg.intermediate_size;
    let merged = dim * cfg.spatial_merge_size * cfg.spatial_merge_size; // 32

    // Patch embed: 2-D (out, feat) linear + bias, trailing RMSNorm.
    put(
        "vision_tower.patch_embed.patchifier.proj.weight".into(),
        arr(&[dim, feat], 1),
    );
    put(
        "vision_tower.patch_embed.patchifier.proj.bias".into(),
        arr(&[dim], 2),
    );
    put(
        "vision_tower.patch_embed.patchifier.norm.weight".into(),
        arr(&[dim], 3),
    );

    for l in 0..cfg.num_hidden_layers {
        let p = format!("vision_tower.blocks.{l}");
        let s = (l as i32 + 1) * 13;
        put(format!("{p}.norm1.weight"), arr(&[dim], s + 1));
        put(format!("{p}.norm2.weight"), arr(&[dim], s + 2));
        put(format!("{p}.attn.qkv.weight"), arr(&[qkv, dim], s + 3));
        put(format!("{p}.attn.proj.weight"), arr(&[dim, dim], s + 4));
        put(format!("{p}.mlp.fc1.weight"), arr(&[inter, dim], s + 5));
        put(format!("{p}.mlp.fc3.weight"), arr(&[inter, dim], s + 6));
        put(format!("{p}.mlp.fc2.weight"), arr(&[dim, inter], s + 7));
    }

    put(
        "vision_tower.post_trunk_norm.weight".into(),
        arr(&[dim], 90),
    );
    put("vision_tower.merger.ln_q.weight".into(), arr(&[dim], 91));
    put("vision_tower.merger.ln_q.bias".into(), arr(&[dim], 92));
    put(
        "vision_tower.merger.mlp.0.weight".into(),
        arr(&[merged, merged], 93),
    );
    put("vision_tower.merger.mlp.0.bias".into(), arr(&[merged], 94));
    put(
        "vision_tower.merger.mlp.2.weight".into(),
        arr(&[dim, merged], 95),
    );
    put("vision_tower.merger.mlp.2.bias".into(), arr(&[dim], 96));
    w
}

fn absmean(a: &MlxArray) -> f32 {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    let flat = mlxcel_core::reshape(&f, &[-1]);
    let am = mlxcel_core::mean_axis(&mlxcel_core::abs(&flat), 0, false);
    mlxcel_core::eval(&am);
    mlxcel_core::item_f32(&am)
}

fn run(enc: &DotsVisionEncoder, grid: &[(i32, i32, i32)], feat: i32) -> UniquePtr<MlxArray> {
    let s: i32 = grid.iter().map(|&(t, h, w)| t * h * w).sum();
    let n = s * feat;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32 - 0.5) * 0.2).collect();
    let pixels = mlxcel_core::from_slice_f32(&data, &[s, feat]);
    enc.forward_with_grid(&pixels, grid).hidden_states
}

#[test]
fn vision_tower_merges_non_square_grid_to_text_width() {
    let cfg = small_config();
    let weights = build_weights(&cfg);
    let enc = DotsVisionEncoder::from_weights(&weights, &cfg, "vision_tower").expect("build");
    let feat = 3 * cfg.patch_size * cfg.patch_size;

    // Non-square grid 4x6 (both > 2 merge blocks): 24 patches -> 6 merged rows.
    let grid = [(1i32, 4i32, 6i32)];
    let out = run(&enc, &grid, feat);
    let merged_rows = (4 / cfg.spatial_merge_size) * (6 / cfg.spatial_merge_size);
    assert_eq!(
        mlxcel_core::array_shape(&out),
        vec![merged_rows, cfg.embed_dim],
        "merger should emit (h/2)*(w/2) rows at text width"
    );

    // Finite and deterministic across runs.
    let m = absmean(&out);
    assert!(m.is_finite() && m > 0.0, "vision output absmean={m}");
    let out2 = run(&enc, &grid, feat);
    assert!(
        (absmean(&out2) - m).abs() < 1e-6,
        "vision tower is nondeterministic"
    );
}

#[test]
fn vision_tower_handles_multi_image_block_diagonal() {
    let cfg = small_config();
    let weights = build_weights(&cfg);
    let enc = DotsVisionEncoder::from_weights(&weights, &cfg, "vision_tower").expect("build");
    let feat = 3 * cfg.patch_size * cfg.patch_size;

    // Two images -> block-diagonal attention over cu_seqlens; total merged rows
    // is the sum of per-image (h/2)*(w/2).
    let grid = [(1i32, 4i32, 4i32), (1i32, 2i32, 6i32)];
    let out = run(&enc, &grid, feat);
    let rows: i32 = grid.iter().map(|&(t, h, w)| t * (h / 2) * (w / 2)).sum();
    assert_eq!(mlxcel_core::array_shape(&out), vec![rows, cfg.embed_dim]);
}
