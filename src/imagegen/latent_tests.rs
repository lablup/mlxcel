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

use super::*;

fn arange(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let flat = mlxcel_core::arange_f32(0.0, n as f32, 1.0);
    mlxcel_core::reshape(&flat, shape)
}

fn values(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

fn shape(a: &MlxArray) -> Vec<i32> {
    mlxcel_core::array_shape(a)
}

#[test]
fn patchify_folds_each_2x2_block_into_channels() {
    // One channel, 2x4 grid:  0 1 2 3
    //                         4 5 6 7
    let x = arange(&[1, 1, 2, 4]);
    let p = patchify_latents(&x).unwrap();
    assert_eq!(shape(&p), [1, 4, 1, 2]);
    // Channel order is (row-in-block, col-in-block): block 0 holds 0,1,4,5.
    assert_eq!(values(&p), [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
}

#[test]
fn patchify_then_unpatchify_is_the_identity() {
    let x = arange(&[2, 3, 6, 4]);
    let round = unpatchify_latents(&patchify_latents(&x).unwrap()).unwrap();
    assert_eq!(shape(&round), [2, 3, 6, 4]);
    assert_eq!(values(&round), values(&x));
}

#[test]
fn patchify_accepts_a_singleton_frame_axis() {
    let x = arange(&[1, 2, 1, 4, 4]);
    let flat = arange(&[1, 2, 4, 4]);
    assert_eq!(
        values(&patchify_latents(&x).unwrap()),
        values(&patchify_latents(&flat).unwrap())
    );
}

#[test]
fn patchify_rejects_bad_shapes() {
    assert!(patchify_latents(&arange(&[4, 4])).is_err());
    assert!(patchify_latents(&arange(&[1, 1, 3, 4])).is_err());
    assert!(patchify_latents(&arange(&[1, 1, 2, 4, 4])).is_err());
}

#[test]
fn unpatchify_needs_a_channel_count_divisible_by_four() {
    assert!(unpatchify_latents(&arange(&[1, 6, 2, 2])).is_err());
    assert!(unpatchify_latents(&arange(&[6, 2, 2])).is_err());
}

#[test]
fn pack_flattens_the_grid_to_tokens() {
    // [B=1, C=2, H=2, W=2] -> [1, 4, 2]; token t holds (c0[t], c1[t]).
    let x = arange(&[1, 2, 2, 2]);
    let packed = pack_latents(&x).unwrap();
    assert_eq!(shape(&packed), [1, 4, 2]);
    assert_eq!(values(&packed), [0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]);
}

#[test]
fn pack_then_unpack_is_the_identity() {
    let x = arange(&[2, 5, 3, 4]);
    let round = unpack_latents(&pack_latents(&x).unwrap(), 3, 4).unwrap();
    assert_eq!(shape(&round), [2, 5, 3, 4]);
    assert_eq!(values(&round), values(&x));
}

#[test]
fn unpack_leaves_a_grid_alone_and_checks_the_token_count() {
    let grid = arange(&[1, 2, 3, 4]);
    assert_eq!(values(&unpack_latents(&grid, 9, 9).unwrap()), values(&grid));
    let tokens = arange(&[1, 12, 2]);
    assert!(unpack_latents(&tokens, 3, 5).is_err());
    assert!(unpack_latents(&arange(&[12, 2]), 3, 4).is_err());
}

#[test]
fn the_full_pipeline_layout_round_trips() {
    // latents -> patchify -> pack (what the DiT sees) -> unpack -> unpatchify
    let latents = arange(&[1, 32, 8, 8]);
    let packed = pack_latents(&patchify_latents(&latents).unwrap()).unwrap();
    assert_eq!(shape(&packed), [1, 16, 128]);
    let grid = unpack_latents(&packed, 4, 4).unwrap();
    let back = unpatchify_latents(&grid).unwrap();
    assert_eq!(values(&back), values(&latents));
}
