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

use super::{Downsample, PatchEmbed, merge_window_perm};
use crate::vision::encoders::glm4v::Glm4vVisionConfig;
use mlxcel_core::MlxArray;
use mlxcel_core::weights::WeightMap;

fn vision_config(
    hidden_size: usize,
    out_hidden: usize,
    patch: usize,
    in_ch: usize,
    temporal: usize,
    merge: usize,
) -> Glm4vVisionConfig {
    serde_json::from_value(serde_json::json!({
        "depth": 1,
        "hidden_size": hidden_size,
        "intermediate_size": 4,
        "num_heads": 1,
        "patch_size": patch,
        "out_hidden_size": out_hidden,
        "spatial_merge_size": merge,
        "temporal_patch_size": temporal,
        "in_channels": in_ch,
        "rms_norm_eps": 1e-5
    }))
    .unwrap()
}

/// Flatten an array to host f32 for element-wise comparison.
fn to_vec(x: &MlxArray) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(x);
    let n: i32 = shape.iter().product();
    let flat = mlxcel_core::reshape(x, &[n]);
    mlxcel_core::eval(&flat);
    (0..n)
        .map(|i| {
            let cell = mlxcel_core::slice(&flat, &[i], &[i + 1]);
            mlxcel_core::eval(&cell);
            mlxcel_core::item_f32(&cell)
        })
        .collect()
}

fn assert_close(a: &MlxArray, b: &MlxArray) {
    let va = to_vec(a);
    let vb = to_vec(b);
    assert_eq!(va.len(), vb.len(), "length mismatch");
    for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
        assert!((x - y).abs() < 1e-5, "element {i}: {x} vs {y}");
    }
}

#[test]
fn patch_embed_both_conv_layouts_yield_identical_weight() {
    // Canonical flatten order for the linear: (out, kT, in, kH, kW).
    let (out, kt, inc, kh, kw) = (2i32, 2i32, 3i32, 2i32, 2i32);
    let n = (out * kt * inc * kh * kw) as usize;
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let canonical = mlxcel_core::from_slice_f32(&data, &[out, kt, inc, kh, kw]);
    let in_features = kt * inc * kh * kw;
    let expected = mlxcel_core::reshape(&canonical, &[out, in_features]);

    let cfg = vision_config(out as usize, 4, kh as usize, inc as usize, kt as usize, 2);

    // Channels-last export layout [out, kT, kH, kW, in] = canonical[0,1,3,4,2].
    let cl = mlxcel_core::transpose_axes(&canonical, &[0, 1, 3, 4, 2]);
    mlxcel_core::eval(&cl);
    let mut w_cl = WeightMap::new();
    w_cl.insert("vt.proj.weight".to_string(), mlxcel_core::copy(&cl));
    let pe_cl = PatchEmbed::from_weights(&w_cl, &cfg, "vt").unwrap();
    assert_close(&pe_cl.proj_weight, &expected);

    // Raw channels-second layout [out, in, kT, kH, kW] = canonical[0,2,1,3,4].
    let cs = mlxcel_core::transpose_axes(&canonical, &[0, 2, 1, 3, 4]);
    mlxcel_core::eval(&cs);
    let mut w_cs = WeightMap::new();
    w_cs.insert("vt.proj.weight".to_string(), mlxcel_core::copy(&cs));
    let pe_cs = PatchEmbed::from_weights(&w_cs, &cfg, "vt").unwrap();
    assert_close(&pe_cs.proj_weight, &expected);
}

#[test]
fn downsample_both_conv_layouts_yield_identical_weight() {
    // Canonical flatten order: (out, kH, kW, in) with in == hidden_size.
    let (out, kh, kw, hidden) = (2i32, 2i32, 2i32, 3i32);
    let n = (out * kh * kw * hidden) as usize;
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let canonical = mlxcel_core::from_slice_f32(&data, &[out, kh, kw, hidden]);
    let block_features = kh * kw * hidden;
    let expected = mlxcel_core::reshape(&canonical, &[out, block_features]);

    // hidden_size = downsample in-channels; out_hidden = downsample out; merge 2.
    let cfg = vision_config(hidden as usize, out as usize, 2, 3, 2, 2);

    // Channels-last layout [out, kH, kW, in] = canonical.
    let mut w_cl = WeightMap::new();
    w_cl.insert("ds.weight".to_string(), mlxcel_core::copy(&canonical));
    let ds_cl = Downsample::from_weights(&w_cl, &cfg, "ds").unwrap();
    assert_close(&ds_cl.weight, &expected);

    // Raw channels-second layout [out, in, kH, kW] = canonical[0,3,1,2].
    let cs = mlxcel_core::transpose_axes(&canonical, &[0, 3, 1, 2]);
    mlxcel_core::eval(&cs);
    let mut w_cs = WeightMap::new();
    w_cs.insert("ds.weight".to_string(), mlxcel_core::copy(&cs));
    let ds_cs = Downsample::from_weights(&w_cs, &cfg, "ds").unwrap();
    assert_close(&ds_cs.weight, &expected);
}

#[test]
fn merge_window_perm_matches_reference_4x4() {
    // 4x4 patch grid, merge 2 -> four 2x2 windows in row-major window order.
    let perm = merge_window_perm(&[(1, 4, 4)], 2);
    let expected = vec![0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
    assert_eq!(perm, expected);
}

#[test]
fn merge_window_perm_offsets_multiple_images() {
    // Two 2x2 images: each is a single window; second image offset by 4.
    let perm = merge_window_perm(&[(1, 2, 2), (1, 2, 2)], 2);
    assert_eq!(perm, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}
