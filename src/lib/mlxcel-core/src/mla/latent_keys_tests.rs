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

//! A latent "K" slot with a sidecar: the stored-width rule, the column reads
//! and the packed row gather (issue #3163).
//!
//! The reference throughout is the dequantized window: a column read or a row
//! gather on the packed triple must give exactly the values the same read
//! gives after `into_dense`, because a stored row carries its own scales and
//! biases and a group is a contiguous run of packed words.

use super::*;
use crate::cache::KVCache;
use crate::dtype;
use crate::mla::MlaGeometry;
use crate::mla::cache::MlaLatentCache;
use crate::mla::testkit::{Rng, serial, to_vec_f32};

/// Two latent groups and one sidecar group: the DSA layout at a quarter of
/// the shipping width (512 + 128).
const G: MlaGeometry = MlaGeometry {
    num_heads: 4,
    kv_lora_rank: 128,
    qk_nope_head_dim: 16,
    qk_rope_head_dim: 32,
    v_head_dim: 12,
};
const SIDECAR: usize = 64;
const KEY_WIDTH: i32 = 192;
const SEQ: i32 = 9;

fn rows(seed: u64, width: i32, amplitude: f32) -> UniquePtr<MlxArray> {
    let v = Rng::new(seed).vec((SEQ * width) as usize, amplitude);
    ffi::astype(
        &ffi::from_slice_f32(&v, &[1, 1, SEQ, width]),
        dtype::FLOAT16,
    )
}

/// A quantized window over `SEQ` rows of `[latent | sidecar]`, with the
/// sidecar on a different scale from the latent so a group that mixed the two
/// would show.
fn quantized_window(cache: &mut KVCache) -> LatentWindow {
    let latent = rows(0x3163, G.kv_lora_rank as i32, 0.5);
    let sidecar = rows(0x0640, SIDECAR as i32, 6.0);
    let keys = crate::concatenate(&latent, &sidecar, -1);
    let kpe = rows(0x0064, G.qk_rope_head_dim as i32, 1.0);
    let mut view =
        MlaLatentCache::wrap_with_key_width(cache, G, KEY_WIDTH as usize).expect("192 is 3 groups");
    view.update_and_fetch(keys, kpe)
}

fn columns(a: &MlxArray, from: i32, to: i32) -> Vec<f32> {
    let s = ffi::array_shape(a);
    to_vec_f32(&ffi::slice(a, &[0, 0, 0, from], &[s[0], s[1], s[2], to]))
}

fn dense(window: LatentWindow) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    window.into_dense()
}

#[test]
fn stored_key_width_must_be_whole_groups_on_both_sides_of_the_latent() {
    let int8 = KVCacheMode::Int8;
    assert!(check_stored_key_width(512, 640, int8).is_ok());
    assert!(check_stored_key_width(512, 512, int8).is_ok());
    // A sidecar that is not a whole group leaves a partial group at the end.
    let err = check_stored_key_width(512, 612, int8).unwrap_err();
    assert!(err.contains("612") && err.contains("100 sidecar"), "{err}");
    // A rank that is not a whole group would put a group across the boundary,
    // even when the total happens to be a multiple of 64.
    assert!(check_stored_key_width(500, 640, int8).is_err());
    // fp16 stores any width; nothing stores a slot narrower than the latent.
    assert!(check_stored_key_width(500, 612, KVCacheMode::Fp16).is_ok());
    assert!(check_stored_key_width(512, 256, KVCacheMode::Fp16).is_err());
}

#[test]
fn dsa_families_are_latent_pair_families() {
    for &family in DSA_INDEXER_KEY_FAMILIES {
        assert!(caches_dsa_indexer_key(family), "{family}");
        assert!(crate::mla::caches_mla_latent_pair(family), "{family}");
    }
    assert!(caches_dsa_indexer_key(" DeepSeek_V3.2 "));
    assert!(!caches_dsa_indexer_key("deepseek_v3"));
    assert!(!caches_dsa_indexer_key("glm4_moe_lite"));
}

/// The DSA families are not in `QUANTIZED_LATENT_FAMILIES`: on DeepSeek-V3.2
/// the int8 slot passed the trace gate and missed the top-k overlap gate
/// (`docs/benchmark_results/mla-quantized-latent-dsa-m3ultra-2026-11-19.md`),
/// so every quantized request still resolves to fp16 with a warning.
#[test]
fn dsa_families_still_resolve_every_quantized_mode_to_fp16() {
    use crate::cache::turbo::resolve_kv_cache_mode_for_model_with;
    for &family in DSA_INDEXER_KEY_FAMILIES {
        assert!(
            !crate::mla::stores_quantized_latent_pair(family),
            "{family}"
        );
        for absorbed in [false, true] {
            for requested in [
                KVCacheMode::Int8,
                KVCacheMode::Turbo4Asym,
                KVCacheMode::Turbo4,
            ] {
                let (effective, warning) =
                    resolve_kv_cache_mode_for_model_with(requested, family, absorbed);
                assert_eq!(effective, KVCacheMode::Fp16, "{family} {requested}");
                assert!(warning.is_some(), "{family} {requested}");
            }
            let (effective, warning) =
                resolve_kv_cache_mode_for_model_with(KVCacheMode::Fp16, family, absorbed);
            assert_eq!(effective, KVCacheMode::Fp16);
            assert!(warning.is_none(), "{family}: {warning:?}");
        }
    }
}

#[test]
fn the_view_stores_a_sidecar_width_and_refuses_one_it_cannot_group() {
    let _guard = serial();
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    assert!(MlaLatentCache::supports_with_key_width(&cache, G, 192).is_ok());
    assert!(MlaLatentCache::supports_with_key_width(&cache, G, 160).is_err());
    assert!(MlaLatentCache::supports_with_key_width(&cache, G, 64).is_err());

    let window = quantized_window(&mut cache);
    assert!(window.is_quantized());
    assert_eq!(window.key_width(), KEY_WIDTH);
    assert_eq!(window.seq_len(), SEQ);
    // 192 * 8 / 8 packed, 3 groups of fp16 scale and bias, fp16 rope keys.
    let rate = 192 + 3 * 4 + 32 * 2;
    assert_eq!(
        MlaLatentCache::wrap_with_key_width(&mut cache, G, 192)
            .unwrap()
            .bytes_per_token(2),
        rate
    );
    assert_eq!(cache.bytes_per_reserved_token(), rate);

    // An fp16 cache holds the same slot at the element width.
    let mut fp16 = KVCache::new();
    let view = MlaLatentCache::wrap_with_key_width(&mut fp16, G, 192).unwrap();
    assert_eq!(view.bytes_per_token(2), (192 + 32) * 2);
}

#[test]
fn sidecar_columns_dequantize_alone_to_what_the_whole_window_dequantizes_to() {
    let _guard = serial();
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let window = quantized_window(&mut cache);
    let rank = G.kv_lora_rank as i32;

    let sidecar = window.dense_key_columns(rank, KEY_WIDTH);
    assert_eq!(ffi::array_shape(&sidecar), vec![1, 1, SEQ, SIDECAR as i32]);
    let latent = window.dense_key_columns(0, rank);
    let (all, _) = dense(window);
    assert_eq!(ffi::array_dtype(&sidecar), ffi::array_dtype(&all));
    assert_eq!(to_vec_f32(&sidecar), columns(&all, rank, KEY_WIDTH));
    assert_eq!(to_vec_f32(&latent), columns(&all, 0, rank));
}

#[test]
fn narrowing_keeps_the_latent_packed_and_its_values_unchanged() {
    let _guard = serial();
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let rank = G.kv_lora_rank as i32;
    let (all, kpe_all) = dense(quantized_window(&mut cache));

    let narrowed = cache_window(&cache).narrow_keys(0, rank);
    let LatentWindow::Quantized {
        ckv_q,
        scales,
        biases,
        ..
    } = &narrowed
    else {
        panic!("narrowing must keep the quantized form");
    };
    assert_eq!(ffi::array_dtype(ckv_q), dtype::UINT32);
    assert_eq!(ffi::array_shape(ckv_q), vec![1, 1, SEQ, rank * 8 / 32]);
    assert_eq!(ffi::array_shape(scales), vec![1, 1, SEQ, 2]);
    assert_eq!(ffi::array_shape(biases), vec![1, 1, SEQ, 2]);
    assert_eq!(narrowed.key_width(), rank);
    let (latent, kpe) = dense(narrowed);
    assert_eq!(to_vec_f32(&latent), columns(&all, 0, rank));
    assert_eq!(to_vec_f32(&kpe), to_vec_f32(&kpe_all));

    // The full range is the window itself.
    let same = cache_window(&cache).narrow_keys(0, KEY_WIDTH);
    assert_eq!(same.key_width(), KEY_WIDTH);
}

#[test]
#[should_panic(expected = "group boundaries")]
fn a_column_range_inside_a_group_is_refused_on_a_quantized_window() {
    let _guard = serial();
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let _ = quantized_window(&mut cache).dense_key_columns(100, KEY_WIDTH);
}

/// The live window of a cache `quantized_window` already filled.
fn cache_window(cache: &KVCache) -> LatentWindow {
    LatentWindow::from_rows(
        cache
            .latent_quantized_window()
            .expect("the cache holds quantized latent rows"),
        8,
    )
}

#[test]
fn the_packed_gather_equals_gathering_after_dequantize() {
    let _guard = serial();
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let (all, kpe_all) = dense(quantized_window(&mut cache));
    // Out of order and not contiguous, as an argpartition selection is.
    let picked = [7, 2, 5, 0];
    let indices = ffi::from_slice_i32(&picked, &[1, 1, 1, picked.len() as i32]);

    let gathered = cache_window(&cache).gather_rows(&indices);
    assert!(gathered.is_quantized(), "the gather must not dequantize");
    assert_eq!(gathered.seq_len(), picked.len() as i32);
    let (got, got_kpe) = dense(gathered);

    let take = |a: &MlxArray| -> Vec<f32> {
        let s = ffi::array_shape(a);
        picked
            .iter()
            .flat_map(|&row| to_vec_f32(&ffi::slice(a, &[0, 0, row, 0], &[1, 1, row + 1, s[3]])))
            .collect()
    };
    assert_eq!(to_vec_f32(&got), take(&all));
    assert_eq!(to_vec_f32(&got_kpe), take(&kpe_all));

    // The dense form gathers the same rows.
    let dense_gather = LatentWindow::Dense {
        ckv: ffi::copy(&all),
        kpe: ffi::copy(&kpe_all),
    }
    .gather_rows(&indices);
    let (dense_got, dense_kpe) = dense(dense_gather);
    assert_eq!(to_vec_f32(&dense_got), take(&all));
    assert_eq!(to_vec_f32(&dense_kpe), take(&kpe_all));
}

/// Sparse decode's shape: narrow to the latent, gather, then decode. The
/// quantized decode over the gathered packed rows equals dense absorbed
/// attention over the same rows gathered after dequantize.
#[test]
fn decode_over_gathered_packed_rows_matches_dense_decode_over_the_dequantized_rows() {
    let _guard = serial();
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let _ = quantized_window(&mut cache);
    let rank = G.kv_lora_rank as i32;
    let picked = [6, 1, 3];
    let indices = ffi::from_slice_i32(&picked, &[1, 1, 1, picked.len() as i32]);
    let selected = cache_window(&cache)
        .narrow_keys(0, rank)
        .gather_rows(&indices);

    let heads = G.num_heads as i32;
    let rope = G.qk_rope_head_dim as i32;
    let mut rng = Rng::new(0x51EC);
    let q = ffi::from_slice_f32(&rng.vec((heads * rank) as usize, 0.5), &[1, heads, 1, rank]);
    let q_pe = ffi::from_slice_f32(&rng.vec((heads * rope) as usize, 0.5), &[1, heads, 1, rope]);
    let scale = 0.25;

    let LatentWindow::Quantized {
        ckv_q,
        scales,
        biases,
        kpe,
        bits,
        group_size,
    } = &selected
    else {
        panic!("quantized window expected");
    };
    let got = to_vec_f32(&crate::mla::quantized_absorbed_attention(
        &ffi::astype(&q, dtype::FLOAT16),
        &ffi::astype(&q_pe, dtype::FLOAT16),
        ckv_q,
        scales,
        biases,
        kpe,
        *bits,
        *group_size,
        scale,
        None,
    ));

    // Reference: dequantize the whole window, slice the latent, gather the
    // rows, and run dense attention in f32. The zero mask keeps the fused
    // kernel out of the reference.
    let dense_rows = LatentWindow::Dense {
        ckv: cache_window(&cache).dense_key_columns(0, rank),
        kpe: dense(cache_window(&cache)).1,
    }
    .gather_rows(&indices);
    let (ckv, kpe) = dense(dense_rows);
    let zero = ffi::zeros(&[1, 1, 1, picked.len() as i32], dtype::FLOAT32);
    let want = to_vec_f32(&crate::mla::decode::absorbed_attention(
        &q,
        &q_pe,
        &ffi::astype(&ckv, dtype::FLOAT32),
        &ffi::astype(&kpe, dtype::FLOAT32),
        scale,
        Some(&zero),
    ));
    let err = crate::mla::testkit::max_rel_error(&got, &want);
    assert!(
        err < 2e-3,
        "gathered packed decode vs dense reference: {err}"
    );
}
