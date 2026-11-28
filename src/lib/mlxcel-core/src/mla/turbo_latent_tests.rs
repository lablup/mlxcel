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

//! Rotated-domain Turbo-4 latent (issue #3157): codec round trip, decode
//! parity against dequantize-then-`absorbed_attention`, zero rows, trim and
//! restore of rotated rows, and bf16 inputs.

use super::*;
use crate::cache::TURBO_DEFAULT_SEED;
use crate::cache::latent_quant::LATENT_QUANT_GROUP_SIZE;
use crate::mla::MlaGeometry;
use crate::mla::decode::absorbed_attention;
use crate::mla::quant_latent::LatentWindow;
use crate::mla::testkit::{MlaFixture, Rng, max_rel_error, serial, to_vec_f32};

/// Power-of-two rank (the WHT's requirement), two affine groups.
const G: MlaGeometry = MlaGeometry {
    num_heads: 4,
    kv_lora_rank: 128,
    qk_nope_head_dim: 16,
    qk_rope_head_dim: 32,
    v_head_dim: 12,
};

/// Rotated-domain decode against dense attention over the dequantized rows.
const COMPOSITION_TOL: f32 = 2e-3;

fn q_absorbed(fx: &MlaFixture, seed: u64, dt: i32) -> UniquePtr<MlxArray> {
    let mut rng = Rng::new(seed);
    let (b, h, r) = (fx.batch as i32, G.num_heads as i32, G.kv_lora_rank as i32);
    let v = rng.vec((b * h * r) as usize, 0.5);
    ffi::astype(&ffi::from_slice_f32(&v, &[b, h, 1, r]), dt)
}

fn rows(a: &MlxArray, from: i32, to: i32) -> UniquePtr<MlxArray> {
    let s = ffi::array_shape(a);
    ffi::slice(a, &[0, 0, from, 0], &[s[0], s[1], to, s[3]])
}

/// f32 dense absorbed attention with the fused kernel kept out by a zero mask.
fn dense(q: &MlxArray, q_pe: &MlxArray, ckv: &MlxArray, kpe: &MlxArray, scale: f32) -> Vec<f32> {
    let s = ffi::array_shape(ckv)[2];
    let zero = ffi::zeros(&[1, 1, 1, s], dtype::FLOAT32);
    let f = |a: &MlxArray| ffi::astype(a, dtype::FLOAT32);
    let out = absorbed_attention(&f(q), &f(q_pe), &f(ckv), &f(kpe), scale, Some(&zero));
    to_vec_f32(&out)
}

fn bytes(a: &MlxArray) -> Vec<u8> {
    ffi::eval(a);
    ffi::array_to_raw_bytes(a)
}

#[test]
fn turbo4_latent_costs_386_bytes_per_token_on_v2_lite() {
    assert_eq!(turbo4_latent_bytes_per_token(512, 64), 386);
}

#[test]
fn rotation_is_orthogonal_and_round_trip_stays_within_the_codebook_error() {
    let _guard = serial();
    let codec = TurboLatentCodec::new(G.kv_lora_rank, TURBO_DEFAULT_SEED);
    let fx = MlaFixture::new(G, 1, 1, 33, 0x3157);
    let r = G.kv_lora_rank;
    let mut ckv = fx.ckv.clone();
    ckv[7 * r..8 * r].fill(0.0);
    let x = ffi::from_slice_f32(&ckv, &[1, 1, 33, r as i32]);

    let back = to_vec_f32(&codec.inverse_rotate(&codec.rotate(&x), dtype::FLOAT32));
    assert!(
        max_rel_error(&back, &ckv) < 1e-5,
        "R^T R is not the identity"
    );

    let q = codec.quantize(&x);
    assert_eq!(ffi::array_shape(&q.packed), vec![1, 1, 33, r as i32 / 2]);
    assert_eq!(ffi::array_dtype(&q.packed), dtype::UINT8);
    assert_eq!(ffi::array_shape(&q.rescale), vec![1, 1, 33, 1]);
    assert_eq!(ffi::array_dtype(&q.rescale), dtype::FLOAT16);
    let got = to_vec_f32(&codec.dequantize(&q, dtype::FLOAT32));
    // 4-bit Lloyd-Max on a whitened row: relative MSE about 0.01, so a
    // per-row relative L2 error well under 0.2.
    for t in (0..33).filter(|&t| t != 7) {
        let (g, w) = (&got[t * r..(t + 1) * r], &ckv[t * r..(t + 1) * r]);
        let err: f32 = g
            .iter()
            .zip(w)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let norm: f32 = w.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(err / norm < 0.2, "row {t}: relative error {}", err / norm);
        // The rescale restores the norm to FP16 accuracy.
        let gn: f32 = g.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (gn - norm).abs() / norm < 2e-3,
            "row {t}: norm {gn} vs {norm}"
        );
    }
    assert!(
        got[7 * r..8 * r].iter().all(|v| *v == 0.0),
        "an all-zero row did not read back as zero"
    );
}

#[test]
fn rotated_domain_decode_matches_dequantize_then_absorbed_attention() {
    let _guard = serial();
    let codec = TurboLatentCodec::new(G.kv_lora_rank, TURBO_DEFAULT_SEED);
    for batch in [1usize, 2] {
        let fx = MlaFixture::new(G, batch, 1, 41, 0xC0DE + batch as u64);
        let dt = dtype::FLOAT16;
        let (ckv, kpe) = (fx.ckv_array(dt), fx.kpe_array(dt));
        let q = q_absorbed(&fx, 0x51, dt);
        let q_pe = fx.q_pe_array(dt);
        let stored = codec.quantize(&ckv);
        let _ = stats::take();
        let out = codec.absorbed_attention(&q, &q_pe, &stored, &kpe, fx.scale, None);
        let counts = stats::take();
        assert_eq!(counts.absorbed_composed, 1, "{}", counts.summary());
        assert_eq!(ffi::array_dtype(&out), dt);
        let got = to_vec_f32(&out);

        let want = dense(&q, &q_pe, &codec.dequantize(&stored, dt), &kpe, fx.scale);
        let err = max_rel_error(&got, &want);
        assert!(
            err < COMPOSITION_TOL,
            "batch {batch}: off its own rows by {err}"
        );
        let err = max_rel_error(&got, &dense(&q, &q_pe, &ckv, &kpe, fx.scale));
        assert!(err < 0.25, "batch {batch}: off the fp16 latent by {err}");
    }
}

#[test]
fn a_masked_query_and_bf16_inputs_stay_on_the_rotated_path() {
    let _guard = serial();
    let codec = TurboLatentCodec::new(G.kv_lora_rank, TURBO_DEFAULT_SEED);
    let fx = MlaFixture::new(G, 1, 1, 12, 0xA5);
    let dt = dtype::BFLOAT16;
    let (ckv, kpe) = (fx.ckv_array(dt), fx.kpe_array(dt));
    let q = q_absorbed(&fx, 0x52, dt);
    let q_pe = fx.q_pe_array(dt);
    let mut m = vec![0.0f32; 12];
    m[..5].fill(f32::NEG_INFINITY);
    let mask = ffi::from_slice_f32(&m, &[1, 1, 1, 12]);
    let stored = codec.quantize(&ckv);
    let out = codec.absorbed_attention(&q, &q_pe, &stored, &kpe, fx.scale, Some(&mask));
    assert_eq!(ffi::array_dtype(&out), dt);
    let dq = codec.dequantize(&stored, dtype::FLOAT32);
    let want = dense(&q, &q_pe, &rows(&dq, 5, 12), &rows(&kpe, 5, 12), fx.scale);
    // bf16 output rounding (8 mantissa bits) dominates the bound.
    let err = max_rel_error(&to_vec_f32(&out), &want);
    assert!(err < 1e-2, "masked bf16 decode off by {err}");
}

#[test]
fn rows_are_token_local_so_trim_and_restore_reproduce_the_stored_bytes() {
    let _guard = serial();
    let fx = MlaFixture::new(G, 1, 1, 20, 0x7121);
    let ckv = fx.ckv_array(dtype::FLOAT16);
    let codec = TurboLatentCodec::new(G.kv_lora_rank, TURBO_DEFAULT_SEED);
    let whole = codec.quantize(&ckv);

    // Prefill 17 rows, write 3 a verifier rejects, trim back to 17, then
    // rewrite the accepted continuation: the window is the whole quantized
    // window byte for byte, since no row depends on another.
    let head = codec.quantize(&rows(&ckv, 0, 17));
    let tail = codec.quantize(&rows(&ckv, 17, 20));
    let rejected = codec.quantize(&rows(&fx.ckv_array(dtype::FLOAT32), 0, 3));
    let spec = crate::concatenate(&head.packed, &rejected.packed, 2);
    let trimmed = rows(&spec, 0, 17);
    let rewound = crate::concatenate(&trimmed, &tail.packed, 2);
    assert_eq!(bytes(&rewound), bytes(&whole.packed));
    let scales = crate::concatenate(&head.rescale, &tail.rescale, 2);
    assert_eq!(bytes(&scales), bytes(&whole.rescale));

    // Restore: a codec rebuilt from the same seed reads the parked rows back
    // identically; another seed does not.
    let restored = TurboLatentCodec::new(G.kv_lora_rank, TURBO_DEFAULT_SEED);
    let a = bytes(&codec.dequantize(&whole, dtype::FLOAT32));
    assert_eq!(a, bytes(&restored.dequantize(&whole, dtype::FLOAT32)));
    let other = TurboLatentCodec::new(G.kv_lora_rank, TURBO_DEFAULT_SEED ^ 1);
    assert_ne!(a, bytes(&other.dequantize(&whole, dtype::FLOAT32)));
}

#[test]
fn rotated_affine_decode_matches_its_dequantized_window() {
    let _guard = serial();
    let codec = TurboLatentCodec::new(G.kv_lora_rank, TURBO_DEFAULT_SEED);
    let fx = MlaFixture::new(G, 1, 1, 41, 0xAFF);
    let dt = dtype::FLOAT16;
    let (ckv, kpe) = (fx.ckv_array(dt), fx.kpe_array(dt));
    let q = q_absorbed(&fx, 0x53, dt);
    let q_pe = fx.q_pe_array(dt);
    let arm = RotatedAffine {
        codec: &codec,
        bits: 4,
    };
    let stored = arm.quantize(&ckv, fx.kpe_array(dt));
    let out = arm.absorbed_attention(&q, &q_pe, &stored, LATENT_QUANT_GROUP_SIZE, fx.scale, None);
    assert_eq!(ffi::array_dtype(&out), dt);
    let (rot, _) = LatentWindow::from_rows(arm.quantize(&ckv, fx.kpe_array(dt)), 4).into_dense();
    let window = codec.inverse_rotate(&rot, dt);
    let want = dense(&q, &q_pe, &window, &kpe, fx.scale);
    let err = max_rel_error(&to_vec_f32(&out), &want);
    assert!(
        err < COMPOSITION_TOL,
        "rotated affine decode off its own rows by {err}"
    );
}

#[test]
fn on_device_codec_matches_the_host_turbo4_codec() {
    let _guard = serial();
    let fx = MlaFixture::new(G, 2, 1, 9, 0x4747);
    let x = fx.ckv_array(dtype::FLOAT32);
    let codec = TurboLatentCodec::new(G.kv_lora_rank, TURBO_DEFAULT_SEED);
    let params = TurboQuantParams::new(G.kv_lora_rank as u32, TURBO_DEFAULT_SEED);
    let (packed, _norms, rescale) = crate::cache::turbo::quant::host_reference::quantize(
        &x,
        &params,
        &params.signs1,
        &params.signs2,
    );
    let ours = codec.quantize(&x);
    assert_eq!(bytes(&ours.packed), bytes(&packed));
    let err = max_rel_error(&to_vec_f32(&ours.rescale), &to_vec_f32(&rescale));
    assert!(err < 1e-3, "rescale off the host codec by {err}");
}

#[test]
fn turbo_latent_quantizes_an_empty_sequence() {
    let _guard = serial();
    let codec = TurboLatentCodec::new(128, 42);
    let rows = codec.quantize(&ffi::zeros(&[1, 1, 0, 128], dtype::FLOAT16));
    assert_eq!(ffi::array_shape(&rows.packed), [1, 1, 0, 64]);
    assert_eq!(ffi::array_shape(&rows.rescale), [1, 1, 0, 1]);
    assert_eq!(ffi::array_size(&rows.packed), 0);
    assert_eq!(ffi::array_size(&rows.rescale), 0);
}
