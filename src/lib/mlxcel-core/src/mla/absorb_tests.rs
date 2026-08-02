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

//! Tests for the `kv_b_proj` fold (issue #907).
//!
//! The fold is a pure reshape-and-slice of the checkpoint matrix, so the tests
//! check that the two halves land in the right place and that a shape or
//! geometry that does not describe the matrix is refused rather than
//! reinterpreted.

use super::*;
use crate::dtype;
use crate::mla::testkit::{MlaFixture, TINY, to_vec_f32};

#[test]
fn fold_splits_kv_b_proj_into_the_two_identity_operands() {
    let fx = MlaFixture::new(TINY, 1, 1, 4, 0xA11A);
    let weight = fx.kv_b_array(dtype::FLOAT32);
    let proj = MlaAbsorbedProjections::from_dense(&weight, TINY).unwrap();

    let h = TINY.num_heads;
    let r = TINY.kv_lora_rank;
    let nope = TINY.qk_nope_head_dim;
    let v = TINY.v_head_dim;
    let rows = TINY.kv_b_rows_per_head();

    assert_eq!(
        crate::ffi::array_shape(proj.w_uk()),
        vec![1, h as i32, nope as i32, r as i32]
    );
    assert_eq!(
        crate::ffi::array_shape(proj.w_uv()),
        vec![1, h as i32, r as i32, v as i32]
    );

    // W_UK[h, d, i] == kv_b[h * rows + d, i], no transpose.
    let w_uk = to_vec_f32(proj.w_uk());
    for head in 0..h {
        for d in 0..nope {
            for i in 0..r {
                assert_eq!(
                    w_uk[(head * nope + d) * r + i],
                    fx.kv_b[(head * rows + d) * r + i],
                    "w_uk mismatch at head {head} row {d} col {i}"
                );
            }
        }
    }

    // W_UV is stored transposed: w_uv[h, i, e] == kv_b[h * rows + nope + e, i].
    let w_uv = to_vec_f32(proj.w_uv());
    for head in 0..h {
        for i in 0..r {
            for e in 0..v {
                assert_eq!(
                    w_uv[(head * r + i) * v + e],
                    fx.kv_b[(head * rows + nope + e) * r + i],
                    "w_uv mismatch at head {head} rank {i} value dim {e}"
                );
            }
        }
    }
}

#[test]
fn fold_reports_the_weight_memory_it_costs() {
    let fx = MlaFixture::new(TINY, 1, 1, 4, 7);
    let proj = MlaAbsorbedProjections::from_dense(&fx.kv_b_array(dtype::FLOAT32), TINY).unwrap();
    // Exactly kv_b_proj's own element count: the fold partitions its rows.
    assert_eq!(
        proj.element_count(),
        TINY.num_heads * TINY.kv_b_rows_per_head() * TINY.kv_lora_rank
    );
    assert_eq!(proj.element_count(), fx.kv_b.len());
}

#[test]
fn fold_refuses_a_geometry_that_does_not_describe_the_matrix() {
    let fx = MlaFixture::new(TINY, 1, 1, 4, 11);
    let weight = fx.kv_b_array(dtype::FLOAT32);

    // A head count the matrix cannot be partitioned by. Reinterpreting it would
    // silently mix one head's rows into another's fold, which produces plausible
    // numbers and wrong tokens.
    let mut wrong = TINY;
    wrong.num_heads = TINY.num_heads + 1;
    let err = MlaAbsorbedProjections::from_dense(&weight, wrong).unwrap_err();
    assert!(err.contains("disagrees with geometry"), "{err}");

    // A rank that is not the matrix's second axis.
    let mut wrong_rank = TINY;
    wrong_rank.kv_lora_rank = TINY.kv_lora_rank * 2;
    let err = MlaAbsorbedProjections::from_dense(&weight, wrong_rank).unwrap_err();
    assert!(err.contains("disagrees with geometry"), "{err}");
}

#[test]
fn fold_refuses_a_non_matrix_weight() {
    let three_d = crate::ffi::zeros(&[2, 3, 4], dtype::FLOAT32);
    let err = MlaAbsorbedProjections::from_dense(&three_d, TINY).unwrap_err();
    assert!(err.contains("must be 2-D"), "{err}");
}

#[test]
fn fold_refuses_a_biased_kv_b_proj() {
    // DeepSeek's kv_b_proj has no bias, and the absorption identity has no term
    // for one: `q . (Wc + b)` does not factor through the latent. A checkpoint
    // that grew one must fall back rather than drop the bias.
    let fx = MlaFixture::new(TINY, 1, 1, 4, 13);
    let rows = (TINY.num_heads * TINY.kv_b_rows_per_head()) as i32;
    let linear = crate::layers::UnifiedLinear::Regular(crate::layers::Linear::new(
        fx.kv_b_array(dtype::FLOAT32),
        Some(crate::ffi::zeros(&[rows], dtype::FLOAT32)),
    ));
    let err = MlaAbsorbedProjections::from_kv_b_proj(&linear, TINY).unwrap_err();
    assert!(err.contains("bias"), "{err}");
}

#[test]
fn fold_accepts_an_unquantized_kv_b_proj_layer() {
    let fx = MlaFixture::new(TINY, 1, 1, 4, 17);
    let linear = crate::layers::UnifiedLinear::Regular(crate::layers::Linear::new(
        fx.kv_b_array(dtype::FLOAT32),
        None,
    ));
    let proj = MlaAbsorbedProjections::from_kv_b_proj(&linear, TINY).unwrap();
    assert_eq!(proj.geometry(), TINY);
    let direct = MlaAbsorbedProjections::from_dense(&fx.kv_b_array(dtype::FLOAT32), TINY).unwrap();
    assert_eq!(to_vec_f32(proj.w_uk()), to_vec_f32(direct.w_uk()));
    assert_eq!(to_vec_f32(proj.w_uv()), to_vec_f32(direct.w_uv()));
}
