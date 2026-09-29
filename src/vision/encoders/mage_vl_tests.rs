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

//! Mage-ViT gates: block-order positions, the interleaved rotation with the
//! `concat(freqs, freqs)` pairing and the 4:6:6 `i/size` exponents, the patch
//! kernel layout, and a random-weight forward that checks the merged row
//! count and that images do not attend to each other.

use std::collections::HashSet;

use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;

use super::super::mage_vl_rope::{
    apply_rotary, attention_windows, inv_freqs, positions_from_grid, rotary_angles,
    rotate_half_interleaved,
};
use super::*;

#[test]
fn positions_follow_2x2_block_order() {
    let pos = positions_from_grid(&[(1, 4, 4)], 2);
    assert_eq!(
        &pos[..8],
        &[
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 1, 2],
            [0, 1, 3],
        ]
    );
    assert_eq!(pos[8], [0, 2, 0]);
}

#[test]
fn every_group_of_four_shares_one_merge_cell() {
    let pos = positions_from_grid(&[(1, 6, 8)], 2);
    for group in pos.chunks(4) {
        let cell = (group[0][1] / 2, group[0][2] / 2);
        for p in group {
            assert_eq!((p[1] / 2, p[2] / 2), cell, "{group:?}");
        }
    }
}

#[test]
fn positions_cover_grid_exactly_once() {
    let pos = positions_from_grid(&[(2, 8, 6)], 2);
    assert_eq!(pos.len(), 2 * 8 * 6);
    let unique: HashSet<[i32; 3]> = pos.iter().copied().collect();
    assert_eq!(unique.len(), pos.len());
    for t in 0..2 {
        for h in 0..8 {
            for w in 0..6 {
                assert!(unique.contains(&[t, h, w]));
            }
        }
    }
    // Frames are contiguous: all t=0 rows precede all t=1 rows.
    assert!(pos[..48].iter().all(|p| p[0] == 0));
    assert!(pos[48..].iter().all(|p| p[0] == 1));
}

#[test]
fn rotate_half_is_interleaved() {
    let x = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let out = array_to_vec_f32(&rotate_half_interleaved(&x));
    assert_eq!(out, vec![-2.0, 1.0, -4.0, 3.0]);

    // Leading axes are preserved.
    let x = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 1, 4]);
    let out = array_to_vec_f32(&rotate_half_interleaved(&x));
    assert_eq!(out, vec![-2.0, 1.0, -4.0, 3.0, -6.0, 5.0, -8.0, 7.0]);
}

#[test]
fn inv_freq_sizes_are_8_12_12_for_head_dim_64() {
    let [t, h, w] = inv_freqs(64, 10000.0);
    assert_eq!((t.len(), h.len(), w.len()), (8, 12, 12));
    // Exponent is i/size, not 2i/size.
    assert!((t[1] - 10000f32.powf(-1.0 / 8.0)).abs() < 1e-7);
    assert!((h[1] - 10000f32.powf(-1.0 / 12.0)).abs() < 1e-7);
    assert!((w[11] - 10000f32.powf(-11.0 / 12.0)).abs() < 1e-7);
    assert_eq!(t[0], 1.0);
}

#[test]
fn rotary_angles_concatenate_the_half_table_with_itself() {
    let inv = inv_freqs(64, 10000.0);
    let angles = rotary_angles(&[[1, 2, 3]], &inv);
    assert_eq!(angles.len(), 64);
    let (first, second) = angles.split_at(32);
    assert_eq!(first, second);
    assert!((first[1] - inv[0][1]).abs() < 1e-7); // t = 1
    assert!((first[8 + 1] - 2.0 * inv[1][1]).abs() < 1e-6); // h = 2
    assert!((first[20 + 1] - 3.0 * inv[2][1]).abs() < 1e-6); // w = 3
    // Interleaved pairing: lanes 0 and 1 carry different frequencies.
    assert_ne!(first[0], first[1]);
}

#[test]
fn apply_rotary_matches_the_upstream_formula() {
    // Single lane pair with angles (a0, a1): out = x*cos + rot(x)*sin where
    // rot((x0, x1)) = (-x1, x0).
    let (a0, a1) = (0.3f32, 1.1f32);
    let x = mlxcel_core::from_slice_f32(&[2.0, 5.0], &[1, 2]);
    let angles = mlxcel_core::from_slice_f32(&[a0, a1], &[1, 2]);
    let out = array_to_vec_f32(&apply_rotary(
        &x,
        &mlxcel_core::cos(&angles),
        &mlxcel_core::sin(&angles),
    ));
    let expected = [
        2.0 * a0.cos() - 5.0 * a0.sin(),
        5.0 * a1.cos() + 2.0 * a1.sin(),
    ];
    assert!((out[0] - expected[0]).abs() < 1e-5, "{out:?}");
    assert!((out[1] - expected[1]).abs() < 1e-5, "{out:?}");
}

#[test]
fn attention_windows_split_long_clips_only() {
    assert_eq!(attention_windows(&[(1, 14, 14)], 4), vec![196]);
    assert_eq!(attention_windows(&[(1, 2, 2), (1, 4, 2)], 4), vec![4, 8]);
    assert_eq!(attention_windows(&[(10, 2, 2)], 4), vec![16, 16, 8]);
}

#[test]
fn patch_kernel_channels_last_matches_processor_row_order() {
    // out=1, C=3, P=2. Put a 1 at channels-last position (dy=1, dx=0, c=2).
    let (c, p) = (3usize, 2usize);
    let mut data = vec![0f32; p * p * c];
    data[p * c + 2] = 1.0;
    let kernel = mlxcel_core::from_slice_f32(&data, &[1, p as i32, p as i32, c as i32]);
    let linear = array_to_vec_f32(&patch_kernel_as_linear(&kernel, c, p).unwrap());
    // Processor rows are (c, dy, dx): index 2*P*P + 1*P + 0.
    let hot: Vec<usize> = linear
        .iter()
        .enumerate()
        .filter(|(_, v)| **v == 1.0)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(hot, vec![2 * p * p + p]);

    // The torch layout already is (c, dy, dx).
    let torch = mlxcel_core::from_slice_f32(&linear, &[1, c as i32, p as i32, p as i32]);
    assert_eq!(
        array_to_vec_f32(&patch_kernel_as_linear(&torch, c, p).unwrap()),
        linear
    );
    let bad = mlxcel_core::from_slice_f32(&[0.0; 10], &[1, 10]);
    assert!(patch_kernel_as_linear(&bad, c, p).is_err());
}

/// Deterministic pseudo-random weights in [-scale, scale].
fn noise(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn tiny_config() -> MageVlVisionConfig {
    serde_json::from_value(serde_json::json!({
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 96,
        "patch_size": 4,
        "out_hidden_size": 40,
    }))
    .unwrap()
}

fn tiny_weights(config: &MageVlVisionConfig) -> WeightMap {
    let mut w = WeightMap::new();
    let mut seed = 1u64;
    let mut put = |w: &mut WeightMap, name: String, shape: &[i32], scale: f32| {
        let len: i32 = shape.iter().product();
        seed += 1;
        w.insert(
            name,
            mlxcel_core::from_slice_f32(&noise(len as usize, seed, scale), shape),
        );
    };
    let d = config.hidden_size as i32;
    let ff = config.intermediate_size as i32;
    let p = config.patch_size as i32;
    let merged = d * 4;
    let out = config.out_hidden_size as i32;
    let pre = "vision_tower";
    put(
        &mut w,
        format!("{pre}.embeddings.patch_embedding.weight"),
        &[d, p, p, 3],
        0.2,
    );
    for norm in ["layernorm_pre", "merger.ln_q"] {
        put(&mut w, format!("{pre}.{norm}.weight"), &[d], 1.0);
        put(&mut w, format!("{pre}.{norm}.bias"), &[d], 0.1);
    }
    for i in 0..config.num_hidden_layers {
        let l = format!("{pre}.encoder.layers.{i}");
        for norm in ["layer_norm1", "layer_norm2"] {
            put(&mut w, format!("{l}.{norm}.weight"), &[d], 1.0);
            put(&mut w, format!("{l}.{norm}.bias"), &[d], 0.1);
        }
        for (name, o, i_) in [
            ("self_attn.qkv", 3 * d, d),
            ("self_attn.proj", d, d),
            ("mlp.fc1", ff, d),
            ("mlp.fc2", d, ff),
        ] {
            put(&mut w, format!("{l}.{name}.weight"), &[o, i_], 0.15);
            put(&mut w, format!("{l}.{name}.bias"), &[o], 0.05);
        }
    }
    for (name, o, i_) in [
        ("merger.mlp.0", merged, merged),
        ("merger.mlp.2", out, merged),
    ] {
        put(&mut w, format!("{pre}.{name}.weight"), &[o, i_], 0.05);
        put(&mut w, format!("{pre}.{name}.bias"), &[o], 0.05);
    }
    w
}

fn rows_for(grids: &[(i32, i32, i32)], patch: usize, seed: u64) -> UniquePtr<MlxArray> {
    let rows: i32 = grids.iter().map(|&(t, h, w)| t * h * w).sum();
    let width = (3 * patch * patch) as i32;
    mlxcel_core::from_slice_f32(&noise((rows * width) as usize, seed, 1.0), &[rows, width])
}

#[test]
fn forward_output_rows_equal_patches_over_four() {
    let config = tiny_config();
    let encoder =
        MageVlVisionEncoder::from_weights(&tiny_weights(&config), &config, "vision_tower", 64, 4)
            .unwrap();
    let grids = [(1, 4, 6)];
    let out = encoder
        .forward(&rows_for(&grids, config.patch_size, 7), &grids)
        .unwrap();
    assert_eq!(mlxcel_core::array_shape(&out), vec![6, 40]);
    assert!(array_to_vec_f32(&out).iter().all(|v| v.is_finite()));

    // A grid that disagrees with the row count is an error, not a panic.
    assert!(
        encoder
            .forward(&rows_for(&grids, config.patch_size, 7), &[(1, 4, 4)])
            .is_err()
    );
    assert!(
        encoder
            .forward(&rows_for(&[(1, 3, 4)], config.patch_size, 7), &[(1, 3, 4)])
            .is_err()
    );
}

#[test]
fn images_do_not_attend_to_each_other() {
    let config = tiny_config();
    let encoder =
        MageVlVisionEncoder::from_weights(&tiny_weights(&config), &config, "vision_tower", 64, 4)
            .unwrap();
    let first = [(1, 4, 4)];
    let alone = array_to_vec_f32(
        &encoder
            .forward(&rows_for(&first, config.patch_size, 11), &first)
            .unwrap(),
    );

    // Same first image followed by two different second images.
    let width = (3 * config.patch_size * config.patch_size) as i32;
    let pair = [(1, 4, 4), (1, 2, 4)];
    let run = |seed: u64| {
        let a = rows_for(&first, config.patch_size, 11);
        let b = rows_for(&[(1, 2, 4)], config.patch_size, seed);
        let rows = mlxcel_core::concatenate(&a, &b, 0);
        assert_eq!(mlxcel_core::array_shape(&rows), vec![24, width]);
        array_to_vec_f32(&encoder.forward(&rows, &pair).unwrap())
    };
    let (x, y) = (run(21), run(22));
    assert_eq!(x.len(), (4 + 2) * 40);
    let n = alone.len();
    for i in 0..n {
        assert!(
            (x[i] - alone[i]).abs() < 1e-4,
            "row feature {i} leaked across images"
        );
        assert!(
            (y[i] - alone[i]).abs() < 1e-4,
            "row feature {i} leaked across images"
        );
    }
    assert!(
        x[n..]
            .iter()
            .zip(&y[n..])
            .any(|(a, b)| (a - b).abs() > 1e-4)
    );
}
