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

//! The routed experts of a Gemma 4 MoE layer inside an MTP verify block
//! against single-token decode (issue #3418).
//!
//! `Experts::forward` picks its kernel by the token count: the fused GeGLU
//! kernel for one token, the compiled `gather_qmm` graph below 64 routed
//! pairs, the sorted gather from there. On `gemma-4-26b-a4b-it-4bit` that put
//! the first block-versus-chain divergence at the layer 0 expert output and
//! kept the checkpoint out of the exactness probe on both quantized kernels.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype, utils::slice_axis};

use super::{Experts, SwitchGeGLU};
use crate::models::switch_layers::{HONEST_EXPERT_GROUP_SIZE, insert_honest_affine_swiglu_experts};

const HIDDEN: i32 = 128;
const DFF: i32 = 64;
const EXPERTS: i32 = 16;
const TOP_K: i32 = 8;

/// Affine 4-bit expert planes with real codes, the layout the 26B A4B
/// checkpoint ships.
fn experts() -> Experts {
    let mut weights = WeightMap::new();
    insert_honest_affine_swiglu_experts(&mut weights, "experts.switch_glu", EXPERTS, HIDDEN, DFF);
    Experts {
        switch_geglu: SwitchGeGLU::from_weights(
            &weights,
            "experts.switch_glu",
            HONEST_EXPERT_GROUP_SIZE,
            4,
        )
        .expect("tiny experts must load"),
    }
}

/// Hidden states, routed expert ids and routing weights of a `width`-token
/// block, as the layer's router would hand them over.
fn block(
    width: i32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let mut state = 0x3418_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let x: Vec<f32> = (0..width * HIDDEN).map(|_| next() * 4.0).collect();
    let x = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&x, &[1, width, HIDDEN]),
        dtype::BFLOAT16,
    );
    // Eight distinct experts per token, a different set for each token.
    let indices: Vec<i32> = (0..width)
        .flat_map(|t| (0..TOP_K).map(move |j| (t * 3 + j * 2 + (t % 2)) % EXPERTS))
        .collect();
    let indices = mlxcel_core::astype(
        &mlxcel_core::from_slice_i32(&indices, &[1, width, TOP_K]),
        dtype::UINT32,
    );
    let weights: Vec<f32> = (0..width * TOP_K)
        .map(|i| 0.05 + ((i * 7) % 11) as f32 * 0.02)
        .collect();
    let weights = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&weights, &[1, width, TOP_K]),
        dtype::BFLOAT16,
    );
    (x, indices, weights)
}

fn row_bytes(out: &MlxArray, row: i32) -> Vec<u8> {
    mlxcel_core::array_to_raw_bytes(&slice_axis(out, 1, row, row + 1))
}

/// The expert output of every token alone: what `width` decode steps compute.
fn chain(experts: &Experts, width: i32) -> Vec<Vec<u8>> {
    let (x, indices, weights) = block(width);
    (0..width)
        .map(|t| {
            let out = experts.forward(
                &slice_axis(&x, 1, t, t + 1),
                &slice_axis(&indices, 1, t, t + 1),
                &slice_axis(&weights, 1, t, t + 1),
            );
            row_bytes(&out, 0)
        })
        .collect()
}

fn differing_rows(out: &MlxArray, chain: &[Vec<u8>]) -> Vec<usize> {
    (0..chain.len())
        .filter(|&row| row_bytes(out, row as i32) != chain[row])
        .collect()
}

#[test]
fn verify_block_experts_match_single_token_decode_on_metal() {
    if !mlxcel_core::metal_is_available() {
        return;
    }
    let experts = experts();
    // 2 and 4 tokens stay below 64 routed pairs, 8 and 12 reach the sorted
    // gather.
    for width in [2, 4, 8, 12] {
        let (x, indices, weights) = block(width);
        let out = experts.forward_verify_block(&x, &indices, &weights);
        assert_eq!(
            mlxcel_core::array_shape(&out),
            vec![1, width, HIDDEN],
            "width {width}"
        );
        assert_eq!(
            differing_rows(&out, &chain(&experts, width)),
            Vec::<usize>::new(),
            "width {width}: verify block rows differ from single-token decode"
        );
    }
}

/// What the per-position split fixes: the block-wide expert product is not
/// the single-token kernel's result.
#[test]
fn block_wide_experts_differ_from_single_token_decode_on_metal() {
    if !mlxcel_core::metal_is_available() {
        return;
    }
    let experts = experts();
    let differing: usize = [2, 8]
        .into_iter()
        .map(|width| {
            let (x, indices, weights) = block(width);
            let out = experts.forward(&x, &indices, &weights);
            differing_rows(&out, &chain(&experts, width)).len()
        })
        .sum();
    assert!(
        differing > 0,
        "the block-wide expert product already equals single-token decode"
    );
}

/// A batch of sequences is not a verify block: it keeps the block product.
#[test]
fn a_batch_keeps_the_block_wide_product() {
    let experts = experts();
    let (x, indices, weights) = block(4);
    let as_batch = |a: &MlxArray, last: i32| mlxcel_core::reshape(a, &[2, 2, last]);
    let (x, indices, weights) = (
        as_batch(&x, HIDDEN),
        as_batch(&indices, TOP_K),
        as_batch(&weights, TOP_K),
    );
    let verify = experts.forward_verify_block(&x, &indices, &weights);
    let plain = experts.forward(&x, &indices, &weights);
    assert_eq!(
        mlxcel_core::array_to_raw_bytes(&verify),
        mlxcel_core::array_to_raw_bytes(&plain)
    );
}
