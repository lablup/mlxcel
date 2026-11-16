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

//! Tests for the `kv_b_proj` fold (issue #907).
//!
//! The fold is a pure reshape-and-slice of the checkpoint matrix, so the tests
//! check that the two halves land in the right place and that a shape or
//! geometry that does not describe the matrix is refused rather than
//! reinterpreted.

use super::*;
use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::mla::testkit::{MlaFixture, TINY, max_rel_error, to_vec_f32};
use cxx::UniquePtr;

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
        crate::ffi::array_shape(proj.w_uk().unwrap()),
        vec![1, h as i32, nope as i32, r as i32]
    );
    assert_eq!(
        crate::ffi::array_shape(proj.w_uv().unwrap()),
        vec![1, h as i32, r as i32, v as i32]
    );

    // W_UK[h, d, i] == kv_b[h * rows + d, i], no transpose.
    let w_uk = to_vec_f32(proj.w_uk().unwrap());
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
    let w_uv = to_vec_f32(proj.w_uv().unwrap());
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
    assert_eq!(
        to_vec_f32(proj.w_uk().unwrap()),
        to_vec_f32(direct.w_uk().unwrap())
    );
    assert_eq!(
        to_vec_f32(proj.w_uv().unwrap()),
        to_vec_f32(direct.w_uv().unwrap())
    );
}

/// A quantized `kv_b_proj` and the dense matrix it dequantizes to, as a layer
/// and as that dense matrix.
fn quantized_kv_b(fx: &MlaFixture) -> (crate::layers::UnifiedLinear, UniquePtr<MlxArray>) {
    let prefix = "kv_b_proj";
    let q = crate::quantize_weights(&fx.kv_b_array(dtype::FLOAT32), 32, 4);
    let mut weights = crate::weights::WeightMap::new();
    weights.insert(format!("{prefix}.weight"), crate::quantized_weights_w(&q));
    weights.insert(
        format!("{prefix}.scales"),
        crate::quantized_weights_scales(&q),
    );
    weights.insert(
        format!("{prefix}.biases"),
        crate::quantized_weights_biases(&q),
    );
    let linear = crate::layers::UnifiedLinear::from_weights(&weights, prefix, 32, 4).unwrap();
    let dense = unsafe {
        ffi::dequantize(
            &crate::quantized_weights_w(&q),
            &crate::quantized_weights_scales(&q),
            &*crate::quantized_weights_biases(&q) as *const MlxArray,
            32,
            4,
            "affine",
        )
    };
    (linear, dense)
}

#[test]
fn quantized_fold_keeps_the_packed_planes_and_matches_the_dequantized_fold() {
    // Issue #3121: a quantized kv_b_proj folds without dequantizing, and its
    // three contractions agree with the dense fold of the same dequantized
    // matrix, so the only difference is where the dequantization happens.
    let fx = MlaFixture::new(TINY, 2, 1, 5, 0x3121);
    let (linear, dense) = quantized_kv_b(&fx);
    let quantized = MlaAbsorbedProjections::from_kv_b_proj(&linear, TINY).unwrap();
    let reference = MlaAbsorbedProjections::from_dense(&dense, TINY).unwrap();
    assert!(quantized.is_quantized());
    assert!(!reference.is_quantized());
    assert!(quantized.w_uk().is_none() && quantized.w_uv().is_none());
    // 4-bit packed rows plus per-group scales and biases, against f32 dense.
    assert!(
        quantized.weight_bytes() * 4 < reference.weight_bytes(),
        "quantized fold {} bytes vs dense {}",
        quantized.weight_bytes(),
        reference.weight_bytes()
    );

    let q_nope = fx.q_nope_array(dtype::FLOAT32);
    let got = to_vec_f32(&quantized.absorb(&q_nope));
    let want = to_vec_f32(&reference.absorb(&q_nope));
    assert!(
        max_rel_error(&got, &want) < 1e-4,
        "absorb {}",
        max_rel_error(&got, &want)
    );

    let o_latent = crate::ffi::astype(
        &crate::ffi::from_slice_f32(
            &fx.q_nope
                .iter()
                .cycle()
                .take(2 * TINY.num_heads * TINY.kv_lora_rank)
                .copied()
                .collect::<Vec<_>>(),
            &[2, TINY.num_heads as i32, 1, TINY.kv_lora_rank as i32],
        ),
        dtype::FLOAT32,
    );
    let got = to_vec_f32(&quantized.unabsorb(&o_latent));
    let want = to_vec_f32(&reference.unabsorb(&o_latent));
    assert!(
        max_rel_error(&got, &want) < 1e-4,
        "unabsorb {}",
        max_rel_error(&got, &want)
    );

    let ckv = fx.ckv_array(dtype::FLOAT32);
    let (gk, gv) = quantized.expand(&ckv);
    let (wk, wv) = reference.expand(&ckv);
    assert_eq!(crate::ffi::array_shape(&gk), crate::ffi::array_shape(&wk));
    assert_eq!(crate::ffi::array_shape(&gv), crate::ffi::array_shape(&wv));
    assert!(max_rel_error(&to_vec_f32(&gk), &to_vec_f32(&wk)) < 1e-4);
    assert!(max_rel_error(&to_vec_f32(&gv), &to_vec_f32(&wv)) < 1e-4);
}
