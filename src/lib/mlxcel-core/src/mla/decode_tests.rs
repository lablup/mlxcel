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

//! Parity tests for Stage 1 absorbed MLA decode (issue #907).
//!
//! The gate the issue asks for is that absorption is *mathematically exact*, so
//! the reference is the pre-absorption computation itself: up-project the latent
//! into per-head K and V, concatenate the rope stream into every head, and run
//! dense attention. That reference is computed on the host in f64 by
//! [`MlaFixture::decompressed_reference`], which is the same arithmetic
//! `deepseek_v2::MLAAttention::forward` performs today.
//!
//! Synthetic tensors only. No MLA checkpoint fits the development host, so
//! these tests prove the identity and its implementation, not token parity on a
//! real DeepSeek model.

use super::*;
use crate::dtype;
use crate::mla::absorb::MlaAbsorbedProjections;
use crate::mla::testkit::{MlaFixture, TINY, max_rel_error, serial, to_vec_f32};
use crate::mla::{MlaGeometry, cache::MlaLatentCache};

/// f32 end to end: the only error left is the GPU's own accumulation order
/// against the host's f64, so the bound is tight enough that a genuinely wrong
/// operand cannot hide under it.
const F32_TOL: f32 = 2e-5;

/// f16 latent and weights, which is what a real checkpoint stores. Looser
/// because the absorbed score sums `kv_lora_rank` products where the
/// decompressed one sums `qk_nope_head_dim`, so the two reassociate differently.
const F16_TOL: f32 = 6e-3;

/// Runs one absorbed step. Callers hold [`serial`] for the whole test body,
/// because this records into the process-global dispatch counters.
fn run_absorbed(fx: &MlaFixture, dt: i32, causal: bool) -> Vec<f32> {
    let proj = MlaAbsorbedProjections::from_dense(&fx.kv_b_array(dt), fx.geometry).unwrap();
    let mask = causal.then(|| fx.causal_mask(dt));
    let out = absorbed_decode(
        &fx.q_nope_array(dt),
        &fx.q_pe_array(dt),
        &fx.ckv_array(dt),
        &fx.kpe_array(dt),
        &proj,
        fx.scale,
        mask.as_deref(),
    );
    to_vec_f32(&out)
}

#[test]
fn absorbed_decode_matches_the_decompressed_reference_in_f32() {
    let _guard = serial();
    let fx = MlaFixture::new(TINY, 2, 1, 37, 0xD00D);
    let got = run_absorbed(&fx, dtype::FLOAT32, false);
    let want = fx.decompressed_reference(false);
    let err = max_rel_error(&got, &want);
    assert!(err < F32_TOL, "absorbed decode drifted by {err}");
}

#[test]
fn absorbed_decode_matches_the_decompressed_reference_in_f16() {
    let _guard = serial();
    let fx = MlaFixture::new(TINY, 2, 1, 37, 0xBEEF);
    let got = run_absorbed(&fx, dtype::FLOAT16, false);
    let want = fx.decompressed_reference(false);
    let err = max_rel_error(&got, &want);
    assert!(err < F16_TOL, "absorbed decode drifted by {err} in f16");
}

#[test]
fn absorbed_attention_respects_an_additive_causal_mask() {
    let _guard = serial();
    // The rope half of the score is passed to SDPA *as* the mask, so a caller's
    // causal mask has to be folded into it. If it were dropped, a multi-token
    // step would be fully bidirectional and every layer above the first would
    // write future-contaminated latents into the cache. That is exactly the bug
    // deepseek_v3.rs carries a comment about; this pins it for the shared path.
    let fx = MlaFixture::new(TINY, 2, 4, 12, 0xCA5E);
    let got = run_absorbed(&fx, dtype::FLOAT32, true);
    let want = fx.decompressed_reference(true);
    let err = max_rel_error(&got, &want);
    assert!(err < F32_TOL, "masked absorbed attention drifted by {err}");

    // Positive control: without the mask the same fixture must NOT match the
    // causal reference, otherwise the test above would pass on a no-op mask.
    let unmasked = run_absorbed(&fx, dtype::FLOAT32, false);
    assert!(
        max_rel_error(&unmasked, &want) > 1e-2,
        "the causal mask made no difference, so the masked test proves nothing"
    );
}

#[test]
fn expand_latent_reproduces_the_up_projection() {
    // The prefill direction of the same fold. `k_nope` and `v` must equal what
    // `kv_b_proj` would have produced, or prefill and decode disagree about the
    // cache contents.
    let fx = MlaFixture::new(TINY, 1, 1, 5, 0x3417);
    let proj = MlaAbsorbedProjections::from_dense(&fx.kv_b_array(dtype::FLOAT32), TINY).unwrap();
    let (k_nope, v) = expand_latent(&fx.ckv_array(dtype::FLOAT32), &proj);

    let h = TINY.num_heads;
    let r = TINY.kv_lora_rank;
    let nope = TINY.qk_nope_head_dim;
    let v_dim = TINY.v_head_dim;
    let rows = TINY.kv_b_rows_per_head();
    assert_eq!(
        crate::ffi::array_shape(&k_nope),
        vec![1, h as i32, fx.kv_len as i32, nope as i32]
    );

    let got_k = to_vec_f32(&k_nope);
    let got_v = to_vec_f32(&v);
    for head in 0..h {
        for t in 0..fx.kv_len {
            let c = &fx.ckv[t * r..][..r];
            for d in 0..nope {
                let w = &fx.kv_b[(head * rows + d) * r..][..r];
                let want: f64 = (0..r).map(|i| w[i] as f64 * c[i] as f64).sum();
                let got = got_k[(head * fx.kv_len + t) * nope + d] as f64;
                assert!(
                    (got - want).abs() < 1e-4,
                    "k_nope[{head}][{t}][{d}] {got} != {want}"
                );
            }
            for e in 0..v_dim {
                let w = &fx.kv_b[(head * rows + nope + e) * r..][..r];
                let want: f64 = (0..r).map(|i| w[i] as f64 * c[i] as f64).sum();
                let got = got_v[(head * fx.kv_len + t) * v_dim + e] as f64;
                assert!(
                    (got - want).abs() < 1e-4,
                    "v[{head}][{t}][{e}] {got} != {want}"
                );
            }
        }
    }
}

#[test]
fn latent_cache_round_trips_the_two_asymmetric_streams() {
    // The packing claim in `mla::cache`: an FP16 KVCache holds a 512-wide "key"
    // and a 64-wide "value" without either shape leaking into the other. This
    // is what makes the whole design possible without a new cache type, so it is
    // tested against the cache rather than assumed from reading it.
    let geometry = MlaGeometry {
        num_heads: 4,
        kv_lora_rank: 32,
        qk_nope_head_dim: 16,
        qk_rope_head_dim: 8,
        v_head_dim: 12,
    };
    let mut cache = crate::cache::KVCache::new();
    let mut view = MlaLatentCache::wrap(&mut cache, geometry).unwrap();

    let prefill = MlaFixture::new(geometry, 1, 1, 5, 0x77);
    let (ckv_all, kpe_all) = view.update_and_fetch(
        prefill.ckv_array(dtype::FLOAT16),
        prefill.kpe_array(dtype::FLOAT16),
    );
    assert_eq!(crate::ffi::array_shape(&ckv_all), vec![1, 1, 5, 32]);
    assert_eq!(crate::ffi::array_shape(&kpe_all), vec![1, 1, 5, 8]);
    assert_eq!(view.seq_len(), 5);

    let step = MlaFixture::new(geometry, 1, 1, 1, 0x78);
    let (ckv_all, kpe_all) = view.update_and_fetch(
        step.ckv_array(dtype::FLOAT16),
        step.kpe_array(dtype::FLOAT16),
    );
    assert_eq!(crate::ffi::array_shape(&ckv_all), vec![1, 1, 6, 32]);
    assert_eq!(crate::ffi::array_shape(&kpe_all), vec![1, 1, 6, 8]);
    assert_eq!(view.seq_len(), 6);

    // The appended rows must be the ones just written, at the tail.
    let ckv_host = to_vec_f32(&ckv_all);
    for (i, want) in step.ckv.iter().enumerate() {
        let got = ckv_host[5 * 32 + i];
        assert!(
            (got - want).abs() < 1e-2,
            "latent tail row {i}: {got} != {want}"
        );
    }
}

#[test]
fn stage_1_records_the_path_it_took() {
    let _guard = serial();
    // The #899 trap: a benchmark arm that silently ran the fallback. A decode
    // step must land on `absorbed_composed` and a multi-token step on
    // `absorbed_prefill`, so an arm's counters name what actually ran.
    let _ = crate::mla::stats::take();
    let decode = MlaFixture::new(TINY, 1, 1, 8, 0x91);
    run_absorbed(&decode, dtype::FLOAT32, false);
    let counts = crate::mla::stats::take();
    assert_eq!(counts.absorbed_composed, 1, "{}", counts.summary());
    assert_eq!(counts.absorbed_prefill, 0, "{}", counts.summary());

    let prefill = MlaFixture::new(TINY, 1, 3, 8, 0x92);
    run_absorbed(&prefill, dtype::FLOAT32, true);
    let counts = crate::mla::stats::take();
    assert_eq!(counts.absorbed_prefill, 1, "{}", counts.summary());
    assert_eq!(counts.absorbed_composed, 0, "{}", counts.summary());
}
