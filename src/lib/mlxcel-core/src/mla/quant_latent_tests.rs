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

//! Affine-quantized MLA latent: codec round trip, composed decode parity, and
//! the cache bookkeeping a quantized latent shares with per-head INT8
//! (issue #3155).
//!
//! Two references are used. The dequantized window is what the stored rows
//! mean, so the quantized decode must match the dense absorbed attention over
//! it to float accuracy: that isolates the `quantized_matmul` composition from
//! the codec. The original FP16 latent is the quality reference, held to a
//! looser bound that is the codec's own error.

use super::*;
use crate::cache::KVCache;
use crate::mla::cache::MlaLatentCache;
use crate::mla::decode::absorbed_attention;
use crate::mla::testkit::{MlaFixture, max_rel_error, serial, to_vec_f32};

/// A rank of two affine groups, wide enough to exercise per-group scales.
const G: MlaGeometry = MlaGeometry {
    num_heads: 4,
    kv_lora_rank: 128,
    qk_nope_head_dim: 16,
    qk_rope_head_dim: 32,
    v_head_dim: 12,
};

/// Quantized decode against dense attention over the dequantized window.
const COMPOSITION_TOL: f32 = 2e-3;
/// Quantized decode against the unquantized latent, per bit width: the affine
/// step is `range / (2^bits - 1)` per element and the output is a
/// softmax-weighted mean of latent rows.
const INT8_TOL: f32 = 2e-2;
const INT4_TOL: f32 = 2.5e-1;

/// The one mode that stores a quantized latent, and its bit width.
const MODE: KVCacheMode = KVCacheMode::Int8;
const BITS: i32 = 8;

fn q_absorbed(fx: &MlaFixture, seed: u64) -> UniquePtr<MlxArray> {
    let mut rng = crate::mla::testkit::Rng::new(seed);
    let (b, h, l, r) = (
        fx.batch as i32,
        fx.geometry.num_heads as i32,
        fx.q_len as i32,
        fx.geometry.kv_lora_rank as i32,
    );
    let v = rng.vec((b * h * l * r) as usize, 0.5);
    ffi::astype(&ffi::from_slice_f32(&v, &[b, h, l, r]), dtype::FLOAT16)
}

fn rows(a: &MlxArray, from: i32, to: i32) -> UniquePtr<MlxArray> {
    let s = ffi::array_shape(a);
    ffi::slice(a, &[0, 0, from, 0], &[s[0], s[1], to, s[3]])
}

/// Dense absorbed attention, with the fused kernel kept out of the reference
/// by a zero additive mask.
fn dense_reference(
    q: &MlxArray,
    q_pe: &MlxArray,
    ckv: &MlxArray,
    kpe: &MlxArray,
    scale: f32,
) -> Vec<f32> {
    let s = ffi::array_shape(ckv)[2];
    let zero = ffi::zeros(&[1, 1, ffi::array_shape(q)[2], s], dtype::FLOAT32);
    let f32q = ffi::astype(q, dtype::FLOAT32);
    let out = absorbed_attention(
        &f32q,
        &ffi::astype(q_pe, dtype::FLOAT32),
        &ffi::astype(ckv, dtype::FLOAT32),
        &ffi::astype(kpe, dtype::FLOAT32),
        scale,
        Some(&zero),
    );
    to_vec_f32(&out)
}

#[test]
fn quantized_bytes_per_token_matches_the_issue_figures() {
    assert_eq!(quantized_latent_bytes_per_token(512, 64, 8, 64), 672);
    assert_eq!(quantized_latent_bytes_per_token(512, 64, 4, 64), 416);
}

#[test]
fn affine_round_trip_stays_within_one_step_and_keeps_zero_rows_zero() {
    let _guard = serial();
    let fx = MlaFixture::new(G, 1, 1, 9, 0x3155);
    // Row 4 all zero: its groups have range zero and must read back as zero.
    let mut ckv = fx.ckv.clone();
    let r = G.kv_lora_rank;
    ckv[4 * r..5 * r].fill(0.0);
    let ckv_arr = ffi::from_slice_f32(&ckv, &[1, 1, 9, r as i32]);
    // The codec is bit-width generic. 4-bit has no cache mode (it failed the
    // #3155 quality gate) but stays covered here, since the window carries
    // `bits` for the follow-ups that revisit it.
    for bits in [8, 4] {
        let quantized = quantize_latent_rows(&ckv_arr, fx.kpe_array(dtype::FLOAT16), bits);
        assert_eq!(
            ffi::array_shape(&quantized.packed),
            vec![1, 1, 9, r as i32 * bits / 32]
        );
        assert_eq!(ffi::array_dtype(&quantized.packed), dtype::UINT32);
        assert_eq!(ffi::array_shape(&quantized.scales), vec![1, 1, 9, 2]);
        assert_eq!(ffi::array_dtype(&quantized.scales), dtype::FLOAT16);
        assert_eq!(ffi::array_shape(&quantized.biases), vec![1, 1, 9, 2]);
        let (dense, _) = LatentWindow::from_rows(quantized, bits).into_dense();
        let got = to_vec_f32(&dense);
        // Fixture amplitude is 0.5, so a group spans at most 1.0 and one
        // affine step is at most `1 / (2^bits - 1)`. Half a step of rounding
        // plus FP16 scale and bias rounding stays under one step.
        let step = 1.0 / ((1 << bits) - 1) as f32;
        for (i, (g, w)) in got.iter().zip(&ckv).enumerate() {
            assert!((g - w).abs() <= step, "bits {bits} element {i}: {g} vs {w}");
        }
        assert!(
            got[4 * r..5 * r].iter().all(|v| *v == 0.0),
            "bits {bits}: an all-zero row did not read back as zero"
        );
    }
}

#[test]
fn composed_quantized_decode_matches_dense_attention_over_the_same_rows() {
    let _guard = serial();
    for batch in [1usize, 2] {
        let fx = MlaFixture::new(G, batch, 1, 41, 0xC0DE + batch as u64);
        let dt = dtype::FLOAT16;
        let (ckv, kpe) = (fx.ckv_array(dt), fx.kpe_array(dt));
        let q = q_absorbed(&fx, 0x51);
        let q_pe = fx.q_pe_array(dt);
        let fp16_reference = dense_reference(&q, &q_pe, &ckv, &kpe, fx.scale);
        {
            let (mode, bits, codec_tol) = (MODE, BITS, INT8_TOL);
            let mut inner = KVCache::new_with_mode(mode);
            let mut cache = MlaLatentCache::wrap(&mut inner, G).unwrap();
            // Prefill 40 rows, then decode one: with two batch rows the live
            // window is a strided slice of the step-padded buffer.
            let window = cache.update_and_fetch(rows(&ckv, 0, 40), rows(&kpe, 0, 40));
            assert!(window.is_quantized());
            assert_eq!(window.seq_len(), 40);
            let _ = stats::take();
            let out = cache.attend_preprojected(
                &q,
                &q_pe,
                rows(&ckv, 40, 41),
                rows(&kpe, 40, 41),
                fx.scale,
                None,
            );
            let counts = stats::take();
            assert_eq!(counts.absorbed_composed, 1, "{}", counts.summary());
            assert_eq!(counts.total(), 1, "{}", counts.summary());
            assert_eq!(ffi::array_dtype(&out), dt);
            let got = to_vec_f32(&out);

            let (stored, stored_kpe) = inner_window(&inner, bits).into_dense();
            let same_rows = dense_reference(&q, &q_pe, &stored, &stored_kpe, fx.scale);
            let err = max_rel_error(&got, &same_rows);
            assert!(
                err < COMPOSITION_TOL,
                "{mode} batch {batch}: composed decode off the stored rows by {err}"
            );
            let err = max_rel_error(&got, &fp16_reference);
            assert!(
                err < codec_tol,
                "{mode} batch {batch}: quantized decode off the fp16 latent by {err}"
            );
        }
    }
}

fn inner_window(cache: &KVCache, bits: i32) -> LatentWindow {
    LatentWindow::from_rows(cache.latent_quantized_window().unwrap(), bits)
}

#[test]
fn a_masked_single_query_stays_on_the_quantized_path() {
    let _guard = serial();
    let fx = MlaFixture::new(G, 1, 1, 12, 0xA5);
    let dt = dtype::FLOAT16;
    let q = q_absorbed(&fx, 0x52);
    let q_pe = fx.q_pe_array(dt);
    // Hide the first five keys.
    let mut m = vec![0.0f32; 12];
    m[..5].fill(f32::NEG_INFINITY);
    let mask = ffi::from_slice_f32(&m, &[1, 1, 1, 12]);
    let mut inner = KVCache::new_with_mode(KVCacheMode::Int8);
    let mut cache = MlaLatentCache::wrap(&mut inner, G).unwrap();
    let out = cache.attend_preprojected(
        &q,
        &q_pe,
        fx.ckv_array(dt),
        fx.kpe_array(dt),
        fx.scale,
        Some(&mask),
    );
    let (stored, stored_kpe) = inner_window(&inner, 8).into_dense();
    let want = dense_reference(
        &q,
        &q_pe,
        &rows(&stored, 5, 12),
        &rows(&stored_kpe, 5, 12),
        fx.scale,
    );
    let err = max_rel_error(&to_vec_f32(&out), &want);
    assert!(
        err < COMPOSITION_TOL,
        "masked quantized decode off by {err}"
    );
}

#[test]
fn a_multi_token_step_attends_over_the_dequantized_window() {
    let _guard = serial();
    let fx = MlaFixture::new(G, 1, 3, 20, 0xBEEF);
    let dt = dtype::FLOAT16;
    let (ckv, kpe) = (fx.ckv_array(dt), fx.kpe_array(dt));
    let q = q_absorbed(&fx, 0x53);
    let q_pe = fx.q_pe_array(dt);
    let mask = fx.causal_mask(dt);
    {
        let (mode, bits) = (MODE, BITS);
        let mut inner = KVCache::new_with_mode(mode);
        let mut cache = MlaLatentCache::wrap(&mut inner, G).unwrap();
        let _ = cache.update_and_fetch(rows(&ckv, 0, 17), rows(&kpe, 0, 17));
        let _ = stats::take();
        let out = cache.attend_preprojected(
            &q,
            &q_pe,
            rows(&ckv, 17, 20),
            rows(&kpe, 17, 20),
            fx.scale,
            Some(&mask),
        );
        let counts = stats::take();
        assert_eq!(counts.absorbed_prefill, 1, "{}", counts.summary());
        assert_eq!(counts.total(), 1, "{}", counts.summary());

        let (stored, stored_kpe) = inner_window(&inner, bits).into_dense();
        let want = absorbed_attention(&q, &q_pe, &stored, &stored_kpe, fx.scale, Some(&mask));
        let _ = stats::take();
        let err = max_rel_error(&to_vec_f32(&out), &to_vec_f32(&want));
        assert!(
            err < 1e-6,
            "{mode}: multi-token step off its own window by {err}"
        );
    }
}

/// Decode one fixed query against whatever `cache` holds, appending `step`.
fn decode_step(cache: &mut KVCache, fx: &MlaFixture, at: i32) -> Vec<f32> {
    let dt = dtype::FLOAT16;
    let mut view = MlaLatentCache::wrap(cache, G).unwrap();
    let out = view.attend_preprojected(
        &q_absorbed(fx, 0x54),
        &fx.q_pe_array(dt),
        rows(&fx.ckv_array(dt), at, at + 1),
        rows(&fx.kpe_array(dt), at, at + 1),
        fx.scale,
        None,
    );
    to_vec_f32(&out)
}

fn prefill(cache: &mut KVCache, fx: &MlaFixture, from: i32, to: i32) {
    let dt = dtype::FLOAT16;
    let mut view = MlaLatentCache::wrap(cache, G).unwrap();
    let _ = view.update_and_fetch(
        rows(&fx.ckv_array(dt), from, to),
        rows(&fx.kpe_array(dt), from, to),
    );
}

#[test]
fn trim_rewinds_a_quantized_latent_mid_buffer() {
    let _guard = serial();
    // 300 rows crosses one 256-row growth step, so the rewind lands inside a
    // step-padded buffer whose length is not a multiple of the step.
    let fx = MlaFixture::new(G, 1, 1, 301, 0x7121);
    {
        let mode = MODE;
        let mut reference = KVCache::new_with_mode(mode);
        prefill(&mut reference, &fx, 0, 300);
        let want = decode_step(&mut reference, &fx, 300);

        // Speculative shape: write three rows the verifier then rejects.
        let mut cache = KVCache::new_with_mode(mode);
        prefill(&mut cache, &fx, 0, 297);
        prefill(&mut cache, &fx, 100, 103);
        assert_eq!(cache.trim(3), 3);
        assert_eq!(cache.seq_len(), 297);
        prefill(&mut cache, &fx, 297, 300);
        let got = decode_step(&mut cache, &fx, 300);
        assert_eq!(got, want, "{mode}: decode after a mid-buffer trim differs");
    }
}

#[test]
fn trim_front_drops_the_oldest_quantized_rows_with_their_scales() {
    let _guard = serial();
    let fx = MlaFixture::new(G, 1, 1, 31, 0x7122);
    {
        let (mode, bits) = (MODE, BITS);
        let mut cache = KVCache::new_with_mode(mode);
        prefill(&mut cache, &fx, 0, 30);
        let (full, full_kpe) = inner_window(&cache, bits).into_dense();
        assert_eq!(cache.trim_front(7), 7);
        assert_eq!(cache.seq_len(), 23);
        assert_eq!(cache.offset, 30, "trim_front keeps the monotonic offset");
        let (kept, kept_kpe) = inner_window(&cache, bits).into_dense();
        assert_eq!(to_vec_f32(&kept), to_vec_f32(&rows(&full, 7, 30)), "{mode}");
        assert_eq!(to_vec_f32(&kept_kpe), to_vec_f32(&rows(&full_kpe, 7, 30)));

        // A sink-preserving front trim moves all four buffers the same way.
        assert_eq!(cache.trim_front_keep_sink(4, 2), 4);
        let (sunk, _) = inner_window(&cache, bits).into_dense();
        let mut want = to_vec_f32(&rows(&full, 7, 9));
        want.extend(to_vec_f32(&rows(&full, 13, 30)));
        assert_eq!(to_vec_f32(&sunk), want, "{mode}");

        // And the cache still decodes: the new row lands after the survivors.
        let got = decode_step(&mut cache, &fx, 30);
        assert_eq!(cache.seq_len(), 20);
        assert!(got.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn a_detached_and_adopted_quantized_latent_decodes_identically() {
    let _guard = serial();
    // The prompt-cache store and the scheduler handoff both park a cache as a
    // detached handle and install it into a fresh one.
    let fx = MlaFixture::new(G, 1, 1, 26, 0x7123);
    {
        let mode = MODE;
        let mut original = KVCache::new_with_mode(mode);
        prefill(&mut original, &fx, 0, 25);
        let mut twin = KVCache::new_with_mode(mode);
        prefill(&mut twin, &fx, 0, 25);
        let want = decode_step(&mut twin, &fx, 25);

        let handle = original.clone_handle();
        assert_eq!(handle.mode(), mode);
        assert!(original.is_empty());
        let mut restored = KVCache::new_with_mode(mode);
        restored.install_detached(handle).unwrap();
        assert!(restored.holds_quantized_latent());
        assert_eq!(restored.seq_len(), 25);
        assert_eq!(decode_step(&mut restored, &fx, 25), want, "{mode}");

        // A partial adopt (APC block-aligned prefix) trims the handle first.
        let mut source = KVCache::new_with_mode(mode);
        prefill(&mut source, &fx, 0, 25);
        let mut handle = source.clone_handle();
        handle.trim_to(16).unwrap();
        let mut partial = KVCache::new_with_mode(mode);
        partial.install_detached(handle).unwrap();
        let mut short = KVCache::new_with_mode(mode);
        prefill(&mut short, &fx, 0, 16);
        assert_eq!(
            decode_step(&mut partial, &fx, 16),
            decode_step(&mut short, &fx, 16),
            "{mode}: decode after a truncated adopt differs"
        );
    }
}

#[test]
fn runtime_bytes_per_token_are_672_on_v2_lite() {
    let _guard = serial();
    const LITE: MlaGeometry = MlaGeometry {
        num_heads: 16,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
    };
    let mut cache = KVCache::new_with_mode(MODE);
    let mut view = MlaLatentCache::wrap(&mut cache, LITE).unwrap();
    let _ = view.update_and_fetch(
        ffi::zeros(&[1, 1, 10, 512], dtype::FLOAT16),
        ffi::zeros(&[1, 1, 10, 64], dtype::FLOAT16),
    );
    assert_eq!(view.bytes_per_token(2), 672);
    // One growth step is reserved, and every reserved row costs the rate.
    assert_eq!(cache.bytes_per_reserved_token(), 672);
    assert_eq!(cache.nbytes(), 672 * 256);
}

/// The composed decode at 4 bits, called on the window directly: no cache
/// mode stores a 4-bit latent, and this keeps the `bits` the window carries
/// honest for the consumers that will read it.
#[test]
fn composed_decode_reads_the_bit_width_from_the_window() {
    let _guard = serial();
    let fx = MlaFixture::new(G, 1, 1, 33, 0x4B17);
    let dt = dtype::FLOAT16;
    let (ckv, kpe) = (fx.ckv_array(dt), fx.kpe_array(dt));
    let q = q_absorbed(&fx, 0x55);
    let q_pe = fx.q_pe_array(dt);
    let fp16_reference = dense_reference(&q, &q_pe, &ckv, &kpe, fx.scale);
    let rows4 = quantize_latent_rows(&ckv, fx.kpe_array(dt), 4);
    let LatentWindow::Quantized {
        ckv_q,
        scales,
        biases,
        kpe: kpe_q,
        bits,
        group_size,
    } = LatentWindow::from_rows(rows4, 4)
    else {
        panic!("from_rows builds a quantized window");
    };
    assert_eq!((bits, group_size), (4, 64));
    let out = quantized_absorbed_attention(
        &q, &q_pe, &ckv_q, &scales, &biases, &kpe_q, bits, group_size, fx.scale, None,
    );
    let _ = stats::take();
    let got = to_vec_f32(&out);
    let (stored, stored_kpe) = LatentWindow::Quantized {
        ckv_q,
        scales,
        biases,
        kpe: kpe_q,
        bits,
        group_size,
    }
    .into_dense();
    let same_rows = dense_reference(&q, &q_pe, &stored, &stored_kpe, fx.scale);
    let err = max_rel_error(&got, &same_rows);
    assert!(
        err < COMPOSITION_TOL,
        "4-bit composed decode off its rows by {err}"
    );
    let err = max_rel_error(&got, &fp16_reference);
    assert!(err < INT4_TOL, "4-bit decode off the fp16 latent by {err}");
}

#[test]
fn quantized_bytes_per_token_saturates_instead_of_wrapping() {
    // `kv_arch` passes `config.json` values straight in; a wrapped product
    // would undercount the cache.
    assert_eq!(
        quantized_latent_bytes_per_token(usize::MAX / 2, 64, 8, 64),
        usize::MAX
    );
    assert_eq!(
        quantized_latent_bytes_per_token(512, usize::MAX, 8, 64),
        usize::MAX
    );
    assert_eq!(quantized_latent_bytes_per_token(512, 64, 8, 0), usize::MAX);
}
