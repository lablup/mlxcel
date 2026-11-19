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

//! Parity tests for the rope-folded multi-token MLA attention (issue #3071).
//!
//! The reference is [`MlaFixture::decompressed_reference`], the host f64
//! computation of `softmax(scale * (q_nope . k_nope + q_pe . k_pe)) v`, so the
//! tests pin the folded call to the MLA score itself rather than to the
//! rope-score-mask form it replaces.

use cxx::UniquePtr;

use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::mla::absorb::MlaAbsorbedProjections;
use crate::mla::decode::{absorb_queries, unabsorb_output};
use crate::mla::prefill::{FoldedMask, rope_folded_attention};
use crate::mla::testkit::{MlaFixture, TINY, max_rel_error, serial, to_vec_f32};
use crate::utils::slice_axis;

const F32_TOL: f32 = 2e-5;
const F16_TOL: f32 = 6e-3;

/// Per-head `(k_nope, v)` up-projected from the fixture latent, each
/// `[B, H, S, D]`: what the materializing prefill hands SDPA.
fn materialized_kv(fx: &MlaFixture, dt: i32) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let g = fx.geometry;
    let (b, s) = (fx.batch as i32, fx.kv_len as i32);
    let ckv = ffi::reshape(&fx.ckv_array(dt), &[b, s, g.kv_lora_rank as i32]);
    let w_t = ffi::transpose_axes(&fx.kv_b_array(dt), &[1, 0]);
    let kv = ffi::matmul(&ckv, &w_t);
    let kv = ffi::reshape(&kv, &[b, s, g.num_heads as i32, -1]);
    let kv = ffi::transpose_axes(&kv, &[0, 2, 1, 3]);
    let nope = g.qk_nope_head_dim as i32;
    (slice_axis(&kv, -1, 0, nope), slice_axis(&kv, -1, nope, -1))
}

fn run_materialized(fx: &MlaFixture, dt: i32, mask: FoldedMask<'_>) -> Vec<f32> {
    let (k_nope, v) = materialized_kv(fx, dt);
    let out = rope_folded_attention(
        &fx.q_nope_array(dt),
        &fx.q_pe_array(dt),
        &k_nope,
        &fx.kpe_array(dt),
        &v,
        fx.scale,
        mask,
    );
    to_vec_f32(&out)
}

#[test]
fn folded_materialized_prefill_matches_the_reference_in_causal_mode() {
    let _guard = serial();
    // q_len < kv_len: a later prefill chunk behind a cache, where causal mode
    // must align bottom-right exactly like the old `create_causal_mask(l,
    // live_before)` did.
    let fx = MlaFixture::new(TINY, 2, 5, 13, 0x3071);
    let want = fx.decompressed_reference(true);
    let got = run_materialized(&fx, dtype::FLOAT32, FoldedMask::Causal);
    let err = max_rel_error(&got, &want);
    assert!(err < F32_TOL, "causal folded prefill drifted by {err}");

    // Positive control: unmasked must not match the causal reference.
    let unmasked = run_materialized(&fx, dtype::FLOAT32, FoldedMask::None);
    assert!(
        max_rel_error(&unmasked, &want) > 1e-2,
        "causal mode made no difference, so the causal test proves nothing"
    );
    let err = max_rel_error(&unmasked, &fx.decompressed_reference(false));
    assert!(err < F32_TOL, "unmasked folded prefill drifted by {err}");
}

#[test]
fn folded_materialized_prefill_applies_an_array_mask() {
    let _guard = serial();
    let fx = MlaFixture::new(TINY, 1, 6, 6, 0xF01D);
    let mask = fx.causal_mask(dtype::FLOAT32);
    let got = run_materialized(&fx, dtype::FLOAT32, FoldedMask::Array(&mask));
    let err = max_rel_error(&got, &fx.decompressed_reference(true));
    assert!(
        err < F32_TOL,
        "array-masked folded prefill drifted by {err}"
    );
}

#[test]
fn folded_materialized_prefill_holds_in_f16() {
    let _guard = serial();
    let fx = MlaFixture::new(TINY, 2, 7, 19, 0xF16);
    let got = run_materialized(&fx, dtype::FLOAT16, FoldedMask::Causal);
    let err = max_rel_error(&got, &fx.decompressed_reference(true));
    assert!(err < F16_TOL, "f16 folded prefill drifted by {err}");
}

#[test]
fn folded_absorbed_prefill_matches_the_reference() {
    let _guard = serial();
    // The latent is a single KV head (`Hk = 1`) whose value width
    // (`kv_lora_rank`) differs from the folded key width (`kv_lora_rank +
    // qk_rope`): the shape the absorbed multi-token step hands SDPA.
    let fx = MlaFixture::new(TINY, 2, 4, 11, 0xAB50);
    let dt = dtype::FLOAT32;
    let proj = MlaAbsorbedProjections::from_dense(&fx.kv_b_array(dt), fx.geometry).unwrap();
    let q_abs = absorb_queries(&fx.q_nope_array(dt), &proj);
    let ckv = fx.ckv_array(dt);
    let o_latent = rope_folded_attention(
        &q_abs,
        &fx.q_pe_array(dt),
        &ckv,
        &fx.kpe_array(dt),
        &ckv,
        fx.scale,
        FoldedMask::Causal,
    );
    let out = unabsorb_output(&o_latent, &proj);
    let err = max_rel_error(&to_vec_f32(&out), &fx.decompressed_reference(true));
    assert!(err < F32_TOL, "absorbed folded prefill drifted by {err}");
}

#[test]
fn query_chunked_fallback_reproduces_the_folded_latent_shape() {
    let _guard = serial();
    // The folded latent call is not a shape MLX fuses (qk width != v width),
    // so a long prefill reaches the query-chunked causal fallback. Run that
    // fallback directly with a chunk that does not divide the query count.
    let fx = MlaFixture::new(TINY, 1, 7, 16, 0xC4C4);
    let dt = dtype::FLOAT32;
    let proj = MlaAbsorbedProjections::from_dense(&fx.kv_b_array(dt), fx.geometry).unwrap();
    let q = crate::concatenate(
        &absorb_queries(&fx.q_nope_array(dt), &proj),
        &fx.q_pe_array(dt),
        -1,
    );
    let ckv = fx.ckv_array(dt);
    let k = crate::concatenate(&ckv, &fx.kpe_array(dt), -1);
    let o_latent = crate::layers::chunked_causal_attention(&q, &k, &ckv, fx.scale, 3);
    let out = unabsorb_output(&o_latent, &proj);
    let err = max_rel_error(&to_vec_f32(&out), &fx.decompressed_reference(true));
    assert!(err < F32_TOL, "chunked folded prefill drifted by {err}");
}
