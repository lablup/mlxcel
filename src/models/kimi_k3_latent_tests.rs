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

//! The affine int8 MLA latent on Kimi K3 (issue #3162), on the tiny
//! four-layer checkpoint of `kimi_k3::tests` with its latent widened to one
//! affine group (`kv_lora_rank` 64). Same bounds as
//! `kimi_linear::latent_tests`.

use mlxcel_core::cache::{KVCacheMode, SequenceId};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::mla::MlaDecodePath;
use mlxcel_core::{MlxArray, UniquePtr};

use super::tests::{noise_arr, sanitized, tiny_text_config};
use super::{KimiK3LayerCache, KimiK3MLAAttention, KimiK3Model, KimiK3TextConfig};

const INT8_LAYER_TOL: f32 = 5e-2;
const HIDDEN: i32 = 16;

fn config() -> KimiK3TextConfig {
    KimiK3TextConfig {
        kv_lora_rank: 64,
        ..tiny_text_config()
    }
}

fn model(mode: Option<KVCacheMode>) -> KimiK3Model {
    let config = config();
    let model = KimiK3Model::from_weights(&sanitized(&config), &config).expect("tiny model loads");
    if let Some(mode) = mode {
        model.set_kv_cache_layer_modes(vec![mode; config.num_hidden_layers]);
    }
    model
}

/// Layer 1, the first MLA layer of the tiny stack.
fn attention() -> KimiK3MLAAttention {
    let config = config();
    KimiK3MLAAttention::from_weights(&sanitized(&config), &config, "model.layers.1.self_attn")
        .expect("MLA layer loads")
}

fn values(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

fn max_rel_error(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        .max(1e-6);
    got.iter()
        .zip(want)
        .fold(0.0f32, |acc, (g, w)| acc.max((g - w).abs() / scale))
}

fn block_key_one(keys: i32) -> UniquePtr<MlxArray> {
    let row: Vec<f32> = (0..keys)
        .map(|k| if k == 1 { f32::NEG_INFINITY } else { 0.0 })
        .collect();
    mlxcel_core::from_slice_f32(&row, &[1, 1, 1, keys])
}

/// Prefill `prefill` tokens, then decode one at a time up to `total`, masking
/// each decode step when `masked`. Returns each step's output.
fn decode(
    attn: &KimiK3MLAAttention,
    cache: &mut KVCache,
    prefill: i32,
    total: i32,
    masked: bool,
) -> Vec<Vec<f32>> {
    let x = noise_arr(&[1, total, HIDDEN], 41, 2.0);
    let head = mlxcel_core::slice(&x, &[0, 0, 0], &[1, prefill, HIDDEN]);
    mlxcel_core::eval(&attn.forward(&head, None, cache));
    (prefill..total)
        .map(|t| {
            let tok = mlxcel_core::slice(&x, &[0, t, 0], &[1, t + 1, HIDDEN]);
            let mask = masked.then(|| block_key_one(t + 1));
            values(&attn.forward(&tok, mask.as_deref(), cache))
        })
        .collect()
}

/// Unmasked int8 decode steps are absorbed (fused or composed), none is
/// decompressed, and the output tracks the fp16 latent; masked steps take the
/// composed path, track the dense masked result and differ from unmasked.
#[test]
fn int8_decode_tracks_the_fp16_latent_masked_and_unmasked() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let attn = attention();
    for masked in [false, true] {
        let want = decode(&attn, &mut KVCache::new(), 4, 8, masked);
        let mut int8 = KVCache::new_with_mode(KVCacheMode::Int8);
        let before = mlxcel_core::mla::stats::snapshot();
        let got = decode(&attn, &mut int8, 4, 8, masked);
        let after = mlxcel_core::mla::stats::snapshot();
        assert!(int8.holds_quantized_latent());
        let delta = |path| after.get(path).saturating_sub(before.get(path));
        let absorbed = delta(MlaDecodePath::AbsorbedFused) + delta(MlaDecodePath::AbsorbedComposed);
        assert!(absorbed >= 4, "masked={masked}: every decode step absorbed");
        if masked {
            assert!(delta(MlaDecodePath::AbsorbedComposed) >= 4);
        }
        for (step, (got, want)) in got.iter().zip(&want).enumerate() {
            assert!(got.iter().all(|v| v.is_finite()), "step {step}");
            let err = max_rel_error(got, want);
            assert!(err > 0.0, "masked={masked} step {step}: int8 is not fp16");
            assert!(
                err < INT8_LAYER_TOL,
                "masked={masked} step {step}: relative error {err}"
            );
        }
        if masked {
            let open = decode(
                &attn,
                &mut KVCache::new_with_mode(KVCacheMode::Int8),
                4,
                8,
                false,
            );
            for (step, (got, open)) in got.iter().zip(&open).enumerate() {
                assert!(
                    max_rel_error(got, open) > INT8_LAYER_TOL,
                    "step {step}: the mask must change the int8 output"
                );
            }
        }
    }
}

fn kda_state(cache: &KimiK3LayerCache) -> Vec<Vec<f32>> {
    match cache {
        KimiK3LayerCache::Delta(state) => [&state.conv, &state.ssm]
            .iter()
            .map(|field| values(field.as_deref().expect("prefill fills the KDA state")))
            .collect(),
        KimiK3LayerCache::Attn(_) => panic!("expected a KDA layer"),
    }
}

fn attn_cache(cache: &KimiK3LayerCache) -> &KVCache {
    match cache {
        KimiK3LayerCache::Attn(kv) => kv,
        KimiK3LayerCache::Delta(_) => panic!("expected an MLA layer"),
    }
}

/// An int8 table builds int8 caches on the MLA layers (1 and 3), which store
/// the quantized latent, reports no KV on the KDA layers (0 and 2), and leaves
/// the first KDA layer's state exactly as the fp16 model computes it. A Turbo
/// table builds fp16.
#[test]
fn an_int8_table_builds_int8_mla_caches_and_leaves_kda_state_unchanged() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let fp16 = model(None);
    let int8 = model(Some(KVCacheMode::Int8));
    assert_eq!(
        int8.kv_cache_layer_modes(),
        Some(vec![KVCacheMode::Int8; 4])
    );
    let int8_storage = Some(vec![
        None,
        Some(KVCacheMode::Int8),
        None,
        Some(KVCacheMode::Int8),
    ]);
    assert_eq!(int8.kv_cache_storage_modes(), int8_storage);
    assert_eq!(
        fp16.kv_cache_storage_modes(),
        Some(vec![
            None,
            Some(KVCacheMode::Fp16),
            None,
            Some(KVCacheMode::Fp16)
        ])
    );

    let ids = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5, 6], &[1, 6]);
    let mut fp16_caches = fp16.make_layer_caches();
    let mut int8_caches = int8.make_layer_caches();
    mlxcel_core::eval(&fp16.forward(&ids, &mut fp16_caches));
    mlxcel_core::eval(&int8.forward(&ids, &mut int8_caches));
    for layer in [1, 3] {
        assert_eq!(attn_cache(&int8_caches[layer]).mode, KVCacheMode::Int8);
        assert!(attn_cache(&int8_caches[layer]).holds_quantized_latent());
        assert!(!attn_cache(&fp16_caches[layer]).holds_quantized_latent());
    }
    assert_eq!(kda_state(&int8_caches[0]), kda_state(&fp16_caches[0]));

    let turbo = model(Some(KVCacheMode::Turbo4));
    assert_eq!(
        attn_cache(&turbo.make_layer_caches()[1]).mode,
        KVCacheMode::Fp16
    );
}

/// Batched decode over quantized rows matches single-row decode (shared
/// model-owned harness), and the decoded MLA caches hold the quantized latent
/// and are never paged-backed.
#[test]
fn batched_int8_decode_matches_single_row_decode() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let make = || model(Some(KVCacheMode::Int8));
    crate::models::batched_decode_harness::assert_model_owned_multistep_with_paged_lengths_prepared_b1(
        make,
        &[0, 3, 0, 3],
    );
    crate::models::batched_decode_harness::assert_model_owned_fresh_and_mixed(make);

    let m = make();
    let id = SequenceId::from_raw(3162);
    m.prepare_sequence_state(id);
    let ids = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    mlxcel_core::eval(&m.forward_with_sequence_id(&ids, Some(id), &mut [], None));
    let next = mlxcel_core::from_slice_i32(&[4], &[1, 1]);
    mlxcel_core::eval(&m.forward_with_sequence_id(&next, Some(id), &mut [], None));
    m.sequence_state
        .with_sequence_state_ref(id, |caches| {
            for layer in [1, 3] {
                let kv = attn_cache(&caches[layer]);
                assert!(kv.holds_quantized_latent());
                assert!(!kv.is_paged_backed());
                assert_eq!(kv.seq_len(), 4);
            }
        })
        .expect("the prepared sequence exists");
}
