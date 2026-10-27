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

//! The DiT attention gives the same rows when its score matrix is chunked.

use super::*;
use crate::imagegen::testing::with_forced_attention_chunking;

/// Tokens: `HEADS * L * L * 4 B` is about 3.4 MiB of f32 scores, so a 1 MiB
/// budget splits the queries into four uneven chunks.
const L: i32 = 650;

fn long_positions() -> UniquePtr<MlxArray> {
    let values: Vec<f32> = (0..L)
        .flat_map(|i| [(i % 7) as f32, ((i / 7) % 5) as f32, (i % 3) as f32])
        .collect();
    mlxcel_core::from_slice_f32(&values, &[1, L, 3])
}

fn block_outputs() -> Vec<f32> {
    let w = block_weights("layers.0", true);
    let block = ZImageTransformerBlock::load(
        &w,
        "layers.0",
        HEADS,
        HEAD_DIM,
        true,
        RopePrecision::KeepPromoted,
    )
    .unwrap();
    let (cos, sin) = rope().compute_freqs(&long_positions()).unwrap();
    let x = det("x_long", &[1, L, DIM], 1.0, 0.0);
    let adaln = det("adaln", &[1, ADALN], 1.0, 0.0);
    // A query-dependent additive mask, so the per-chunk mask row slicing runs.
    let mut mask = vec![0.0_f32; (L * L) as usize];
    for r in 0..L as usize {
        for c in (r + 1)..L as usize {
            mask[r * L as usize + c] = -1e4;
        }
    }
    let mask = mlxcel_core::from_slice_f32(&mask, &[1, 1, L, L]);
    let masked = block
        .forward(&x, &cos, &sin, Some(&adaln), Some(&mask))
        .unwrap();
    let unmasked = block.forward(&x, &cos, &sin, Some(&adaln), None).unwrap();
    let mut out = to_vec(&masked);
    out.extend(to_vec(&unmasked));
    out
}

#[test]
fn block_output_is_unchanged_by_query_chunked_attention() {
    let Some((full, chunked)) = with_forced_attention_chunking(
        "imagegen::dit_blocks::tests::chunk_tests::block_output_is_unchanged_by_query_chunked_attention",
        block_outputs,
    ) else {
        return;
    };
    assert_eq!(full.len(), chunked.len());
    assert_eq!(full.len(), (2 * L * DIM) as usize);
    let max_diff = full
        .iter()
        .zip(&chunked)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let peak = full.iter().map(|a| a.abs()).fold(0.0_f32, f32::max);
    eprintln!("chunked vs single SDPA: max |diff| {max_diff:.3e} (peak {peak:.3e})");
    assert!(max_diff <= 1e-5 * peak.max(1.0), "max |diff| {max_diff}");
}
