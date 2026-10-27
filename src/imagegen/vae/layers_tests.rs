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

//! The VAE attention gives the same rows when its score matrix is chunked.

use super::*;
use crate::imagegen::testing::{det, with_forced_attention_chunking};

const CHANNELS: i32 = 32;
/// `30 * 30` tokens: one head of `900^2` f32 scores is about 3.1 MiB, so a
/// 1 MiB budget splits the queries into four chunks.
const SIDE: i32 = 30;

fn attention_output() -> Vec<f32> {
    let mut w = WeightMap::new();
    let prefix = "attn";
    w.insert(
        format!("{prefix}.group_norm.weight"),
        det("gn.w", &[CHANNELS], 0.2, 1.0),
    );
    w.insert(
        format!("{prefix}.group_norm.bias"),
        det("gn.b", &[CHANNELS], 0.1, 0.0),
    );
    for name in ["to_q", "to_k", "to_v", "to_out"] {
        let scale = 0.8 / f64::from(CHANNELS).sqrt();
        w.insert(
            format!("{prefix}.{name}.weight"),
            det(name, &[CHANNELS, CHANNELS], scale, 0.0),
        );
        w.insert(
            format!("{prefix}.{name}.bias"),
            det(&format!("{name}.b"), &[CHANNELS], 0.1, 0.0),
        );
    }
    let block = AttentionBlock::load(&w, prefix, 8, 1e-6).unwrap();
    let x = det("x", &[1, SIDE, SIDE, CHANNELS], 1.0, 0.0);
    let out = block.forward(&x);
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(&out, dtype::FLOAT32))
}

#[test]
fn attention_block_is_unchanged_by_query_chunked_attention() {
    let Some((full, chunked)) = with_forced_attention_chunking(
        "imagegen::vae::layers::tests::attention_block_is_unchanged_by_query_chunked_attention",
        attention_output,
    ) else {
        return;
    };
    assert_eq!(full.len(), (SIDE * SIDE * CHANNELS) as usize);
    assert_eq!(full.len(), chunked.len());
    let max_diff = full
        .iter()
        .zip(&chunked)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let peak = full.iter().map(|a| a.abs()).fold(0.0_f32, f32::max);
    eprintln!("chunked vs single SDPA: max |diff| {max_diff:.3e} (peak {peak:.3e})");
    assert!(max_diff <= 1e-5 * peak.max(1.0), "max |diff| {max_diff}");
}
