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

//! The 4-bit latent `--mla-latent-quant int4` stores (issue #3179): the
//! uniform slot of every non-DSA family, the split slot of a DSA key
//! (`[4-bit latent | 8-bit indexer key]`), and the decode that reads them.
//!
//! The reference is the dequantized rows, as in `quant_latent_tests`: what a
//! packed read returns must be what the same read returns after dequantizing,
//! and a single-query decode must match dense attention over those rows.

use cxx::UniquePtr;

use crate::cache::{KVCache, KVCacheMode};
use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::mla::cache::MlaLatentCache;
use crate::mla::decode::absorbed_attention;
use crate::mla::latent_keys::{SIDECAR_KEY_BITS, quantized_key_slot_bytes_per_token};
use crate::mla::latent_option::{MlaLatentQuant, with_mla_latent_quant};
use crate::mla::quant_latent::{LatentWindow, quantize_latent_rows};
use crate::mla::stats;
use crate::mla::testkit::{MlaFixture, Rng, max_rel_error, serial, to_vec_f32};
use crate::mla::{MlaGeometry, quantized_absorbed_attention};

/// Two latent groups; with [`SIDECAR`] one sidecar group, the DSA layout at a
/// quarter of the shipping width (512 + 128).
const G: MlaGeometry = MlaGeometry {
    num_heads: 4,
    kv_lora_rank: 128,
    qk_nope_head_dim: 16,
    qk_rope_head_dim: 32,
    v_head_dim: 12,
};
const RANK: i32 = 128;
const SIDECAR: i32 = 64;
const KEY_WIDTH: i32 = RANK + SIDECAR;
const SEQ: i32 = 9;
const COMPOSITION_TOL: f32 = 2e-3;

fn rows(seed: u64, width: i32, amplitude: f32) -> UniquePtr<MlxArray> {
    let v = Rng::new(seed).vec((SEQ * width) as usize, amplitude);
    ffi::astype(
        &ffi::from_slice_f32(&v, &[1, 1, SEQ, width]),
        dtype::FLOAT16,
    )
}

fn last_dim(a: &MlxArray) -> i32 {
    *ffi::array_shape(a).last().unwrap()
}

fn columns(a: &MlxArray, from: i32, to: i32) -> Vec<f32> {
    let s = ffi::array_shape(a);
    to_vec_f32(&ffi::slice(a, &[0, 0, 0, from], &[s[0], s[1], s[2], to]))
}

/// `values` quantized on their own at `bits` and read back.
fn round_trip(values: &MlxArray, bits: i32) -> Vec<f32> {
    let kpe = rows(1, G.qk_rope_head_dim as i32, 1.0);
    let (dense, _) =
        LatentWindow::from_rows(quantize_latent_rows(values, kpe, bits), bits).into_dense();
    to_vec_f32(&dense)
}

fn int4<R>(f: impl FnOnce() -> R) -> R {
    with_mla_latent_quant(MlaLatentQuant::Int4, f)
}

#[test]
fn the_sizing_rates_match_the_issue() {
    // Rank 512, rope 64: 416 at 4 bits against 672 at 8 and 1152 fp16.
    assert_eq!(quantized_key_slot_bytes_per_token(512, 0, 64, 4, 64), 416);
    assert_eq!(quantized_key_slot_bytes_per_token(512, 0, 64, 8, 64), 672);
    // DSA: the 128-wide indexer key stays int8 (128 packed + 2 groups x 4).
    assert_eq!(
        quantized_key_slot_bytes_per_token(512, 128, 64, 4, 64),
        416 + 136
    );
    assert_eq!(quantized_key_slot_bytes_per_token(512, 128, 64, 8, 64), 808);
    assert_eq!(
        quantized_key_slot_bytes_per_token(usize::MAX, 128, 64, 4, 64),
        usize::MAX
    );
}

#[test]
fn int4_stores_four_bit_rows_that_read_back_as_the_codec_round_trip() {
    let _guard = serial();
    let latent = rows(0x3179, RANK, 0.5);
    let kpe = rows(0x64, G.qk_rope_head_dim as i32, 1.0);
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let window = int4(|| {
        let mut view = MlaLatentCache::wrap(&mut cache, G).unwrap();
        assert_eq!(view.latent_bits(), Some(4));
        assert_eq!(
            view.bytes_per_token(2),
            quantized_key_slot_bytes_per_token(128, 0, 32, 4, 64)
        );
        view.update_and_fetch(ffi::copy(&latent), ffi::copy(&kpe))
    });
    let LatentWindow::Quantized {
        ckv_q,
        scales,
        bits,
        ..
    } = &window
    else {
        panic!("an int4 latent is a quantized window");
    };
    assert_eq!(*bits, 4);
    assert_eq!(last_dim(ckv_q), RANK * 4 / 32);
    assert_eq!(last_dim(scales), RANK / 64);
    assert!(cache.holds_quantized_latent());
    assert_eq!(cache.stored_latent_bits(), Some(4));
    let (dense, _) = window.into_dense();
    assert_eq!(to_vec_f32(&dense), round_trip(&latent, 4));

    // The cache keeps its width once it holds rows, whatever the option says
    // afterwards, so a later write cannot mix widths in one buffer.
    let view = MlaLatentCache::wrap(&mut cache, G).unwrap();
    assert_eq!(view.latent_bits(), Some(4));
    let mut fresh = KVCache::new_with_mode(KVCacheMode::Int8);
    assert_eq!(
        MlaLatentCache::wrap(&mut fresh, G).unwrap().latent_bits(),
        Some(8)
    );
    let mut fp16 = KVCache::new();
    assert_eq!(
        int4(|| MlaLatentCache::wrap(&mut fp16, G).unwrap().latent_bits()),
        None
    );
}

#[test]
fn an_int4_single_query_decodes_on_the_composed_path_against_the_stored_rows() {
    let _guard = serial();
    let fx = MlaFixture::new(G, 1, 1, 24, 0x3179);
    let dt = dtype::FLOAT16;
    let mut rng = Rng::new(0x51);
    let q = ffi::astype(
        &ffi::from_slice_f32(&rng.vec(4 * 128, 0.5), &[1, 4, 1, 128]),
        dt,
    );
    let q_pe = fx.q_pe_array(dt);
    let mut inner = KVCache::new_with_mode(KVCacheMode::Int8);
    let _ = stats::take();
    let out = int4(|| {
        MlaLatentCache::wrap(&mut inner, G)
            .unwrap()
            .attend_preprojected(
                &q,
                &q_pe,
                fx.ckv_array(dt),
                fx.kpe_array(dt),
                fx.scale,
                None,
            )
    });
    // The fused kernel reads 8-bit codes only, so an unmasked int4 step is
    // composed on every backend.
    let counts = stats::take();
    assert_eq!(counts.absorbed_composed, 1, "{}", counts.summary());
    assert_eq!(counts.total(), 1, "{}", counts.summary());

    assert_eq!(inner.stored_latent_bits(), Some(4));
    let window = LatentWindow::from_rows(inner.latent_quantized_window().unwrap(), 4);
    let (stored, stored_kpe) = window.into_dense();
    let f32of = |a: &MlxArray| ffi::astype(a, dtype::FLOAT32);
    let zero = ffi::zeros(&[1, 1, 1, 24], dtype::FLOAT32);
    let want = absorbed_attention(
        &f32of(&q),
        &f32of(&q_pe),
        &f32of(&stored),
        &f32of(&stored_kpe),
        fx.scale,
        Some(&zero),
    );
    let _ = stats::take();
    let err = max_rel_error(&to_vec_f32(&out), &to_vec_f32(&want));
    assert!(
        err < COMPOSITION_TOL,
        "int4 composed decode off its stored rows by {err}"
    );
}

/// A DSA key slot under `int4`: `[latent | sidecar]` rows with the sidecar on
/// a different scale, so a group that mixed the two parts would show.
fn split_window(cache: &mut KVCache) -> (LatentWindow, UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let latent = rows(0x3163, RANK, 0.5);
    let sidecar = rows(0x0640, SIDECAR, 6.0);
    let keys = crate::concatenate(&latent, &sidecar, -1);
    let kpe = rows(0x0064, G.qk_rope_head_dim as i32, 1.0);
    let window = int4(|| {
        let mut view = MlaLatentCache::wrap_with_key_width(cache, G, KEY_WIDTH as usize).unwrap();
        assert_eq!(
            view.bytes_per_token(2),
            quantized_key_slot_bytes_per_token(128, 64, 32, 4, 64)
        );
        view.update_and_fetch(keys, kpe)
    });
    (window, latent, sidecar)
}

#[test]
fn a_dsa_slot_under_int4_keeps_the_indexer_key_at_eight_bits() {
    let _guard = serial();
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let (window, latent, sidecar) = split_window(&mut cache);
    let LatentWindow::Quantized {
        ckv_q,
        scales,
        bits,
        ..
    } = &window
    else {
        panic!("an int4 key slot is a quantized window");
    };
    assert_eq!(*bits, 4);
    // Two latent groups at 8 words each, one sidecar group at 16.
    assert_eq!(
        last_dim(ckv_q),
        RANK * 4 / 32 + SIDECAR * SIDECAR_KEY_BITS / 32
    );
    assert_eq!(last_dim(scales), KEY_WIDTH / 64);
    assert_eq!(window.key_width(), KEY_WIDTH);
    assert_eq!(cache.stored_latent_bits(), Some(4));
    // The runtime footprint is the sizing rate, over the rows the buffer
    // reserves (it grows in whole steps).
    let rate = quantized_key_slot_bytes_per_token(128, 64, 32, 4, 64);
    let reserved = cache.nbytes() / rate;
    assert_eq!(cache.nbytes(), reserved * rate);
    assert!(reserved >= SEQ as usize, "{reserved} rows reserved");

    // The indexer reads its columns at 8 bits, the attention the latent at 4.
    let want_sidecar = round_trip(&sidecar, SIDECAR_KEY_BITS);
    let want_latent = round_trip(&latent, 4);
    assert_ne!(
        want_sidecar,
        round_trip(&sidecar, 4),
        "the widths must differ to be told apart"
    );
    assert_eq!(
        to_vec_f32(&window.dense_key_columns(RANK, KEY_WIDTH)),
        want_sidecar
    );
    assert_eq!(to_vec_f32(&window.dense_key_columns(0, RANK)), want_latent);

    // Row selection keeps the split, and narrowing drops the sidecar.
    let picked = [7, 2, 5, 0];
    let indices = ffi::from_slice_i32(&picked, &[1, 1, 1, 4]);
    let gathered = window.gather_rows(&indices);
    assert_eq!(gathered.key_width(), KEY_WIDTH);
    let gathered_keys = gathered.dense_key_columns(RANK, KEY_WIDTH);
    let narrowed = gathered.narrow_keys(0, RANK);
    let LatentWindow::Quantized { ckv_q, bits, .. } = &narrowed else {
        panic!("narrowing keeps the quantized form");
    };
    assert_eq!((*bits, last_dim(ckv_q)), (4, RANK * 4 / 32));
    let (narrow_dense, _) = narrowed.into_dense();
    let (all, _) = window.into_dense();
    assert_eq!(ffi::array_shape(&all), vec![1, 1, SEQ, KEY_WIDTH]);
    assert_eq!(columns(&all, 0, RANK), want_latent);
    assert_eq!(columns(&all, RANK, KEY_WIDTH), want_sidecar);
    let row = |values: &[f32], width: i32, r: i32| {
        values[(r * width) as usize..((r + 1) * width) as usize].to_vec()
    };
    let (narrow_dense, gathered_keys) = (to_vec_f32(&narrow_dense), to_vec_f32(&gathered_keys));
    for (slot, &position) in picked.iter().enumerate() {
        assert_eq!(
            row(&narrow_dense, RANK, slot as i32),
            row(&want_latent, RANK, position)
        );
        assert_eq!(
            row(&gathered_keys, SIDECAR, slot as i32),
            row(&want_sidecar, SIDECAR, position)
        );
    }
}

#[test]
fn the_narrowed_latent_of_a_split_slot_decodes_against_its_dequantized_rows() {
    let _guard = serial();
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let (window, _, _) = split_window(&mut cache);
    let (all, kpe_dense) =
        LatentWindow::from_rows(cache.latent_quantized_window().unwrap(), 4).into_dense();
    let LatentWindow::Quantized {
        ckv_q,
        scales,
        biases,
        kpe,
        bits,
        group_size,
    } = window.narrow_keys(0, RANK)
    else {
        panic!("narrowing keeps the quantized form");
    };
    let mut rng = Rng::new(0x52);
    let q = ffi::from_slice_f32(&rng.vec(4 * 128, 0.5), &[1, 4, 1, 128]);
    let q_pe = ffi::from_slice_f32(&rng.vec(4 * 32, 0.5), &[1, 4, 1, 32]);
    let out = quantized_absorbed_attention(
        &q, &q_pe, &ckv_q, &scales, &biases, &kpe, bits, group_size, 0.1, None,
    );
    let f32of = |a: &MlxArray| ffi::astype(a, dtype::FLOAT32);
    let s = ffi::array_shape(&all);
    let latent = ffi::slice(&all, &[0, 0, 0, 0], &[s[0], s[1], s[2], RANK]);
    let zero = ffi::zeros(&[1, 1, 1, SEQ], dtype::FLOAT32);
    let want = absorbed_attention(
        &q,
        &q_pe,
        &f32of(&latent),
        &f32of(&kpe_dense),
        0.1,
        Some(&zero),
    );
    let _ = stats::take();
    let err = max_rel_error(&to_vec_f32(&out), &to_vec_f32(&want));
    assert!(
        err < COMPOSITION_TOL,
        "split-slot latent decode off by {err}"
    );
}

#[test]
fn a_split_slot_survives_growth_trim_and_detach() {
    let _guard = serial();
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let (window, _, _) = split_window(&mut cache);
    let (before, _) = window.into_dense();
    // A second step appends at the same widths.
    let (window, _, _) = split_window(&mut cache);
    assert_eq!(window.seq_len(), 2 * SEQ);
    cache.trim(SEQ);
    let detached = cache.clone_handle();
    assert_eq!(detached.stored_latent_bits(), Some(4));
    let mut restored = KVCache::new_with_mode(KVCacheMode::Int8);
    restored.install_detached(detached).unwrap();
    assert_eq!(restored.stored_latent_bits(), Some(4));
    let (after, _) =
        LatentWindow::from_rows(restored.latent_quantized_window().unwrap(), 4).into_dense();
    assert_eq!(to_vec_f32(&after), to_vec_f32(&before));

    // An int8 slot reports 8, and an fp16 or per-head cache reports none.
    let mut int8 = KVCache::new_with_mode(KVCacheMode::Int8);
    let keys = crate::concatenate(&rows(1, RANK, 0.5), &rows(2, SIDECAR, 6.0), -1);
    let _ = MlaLatentCache::wrap_with_key_width(&mut int8, G, KEY_WIDTH as usize)
        .unwrap()
        .update_and_fetch(keys, rows(3, 32, 1.0));
    assert_eq!(int8.stored_latent_bits(), Some(8));
    assert_eq!(int8.clone_handle().stored_latent_bits(), Some(8));
    assert_eq!(KVCache::new().stored_latent_bits(), None);
}
