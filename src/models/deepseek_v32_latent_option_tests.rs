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

//! `--mla-latent-quant` on the DSA attention (issue #3179): a forced `int8`
//! stores the whole `[kv_latent | indexer key]` slot at 8 bits, which the
//! default gate keeps off, and `int4` stores the latent at 4 bits with the
//! indexer key still at 8.

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::cache::turbo::resolve_kv_cache_mode_for_model_with_option;
use mlxcel_core::layers::KVCache;
use mlxcel_core::mla::{MlaDecodePath, MlaLatentQuant, with_mla_latent_quant};

use super::{INDEX_DIM, RANK, ROPE, attention, config, hidden_rows, run_layer, values};
use crate::models::mla_latent_test_support as support;

const HIDDEN: usize = 16;

/// The resolver hands a DSA type `int8` storage only under an explicit value.
#[test]
fn the_resolver_builds_int8_storage_for_a_dsa_type_only_when_forced() {
    for family in mlxcel_core::mla::DSA_INDEXER_KEY_FAMILIES {
        let resolve = |option| {
            resolve_kv_cache_mode_for_model_with_option(KVCacheMode::Int8, family, true, option).0
        };
        assert_eq!(resolve(MlaLatentQuant::Auto), KVCacheMode::Fp16, "{family}");
        assert_eq!(resolve(MlaLatentQuant::Int8), KVCacheMode::Int8, "{family}");
        assert_eq!(resolve(MlaLatentQuant::Int4), KVCacheMode::Int8, "{family}");
        assert_eq!(resolve(MlaLatentQuant::Fp16), KVCacheMode::Fp16, "{family}");
    }
}

#[test]
fn a_forced_int8_slot_is_one_eight_bit_run() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let attn = attention(&config(64, INDEX_DIM));
    let mut fp16 = KVCache::new();
    let _ = run_layer(&attn, &mut fp16, 5, 9);
    with_mla_latent_quant(MlaLatentQuant::Int8, || {
        let mut int8 = KVCache::new_with_mode(KVCacheMode::Int8);
        let _ = run_layer(&attn, &mut int8, 5, 9);
        assert_eq!(int8.stored_latent_bits(), Some(8));
        assert_eq!(
            int8.bytes_per_reserved_token(),
            mlxcel_core::mla::quantized_key_slot_bytes_per_token(64, 64, ROPE as usize, 8, 64)
        );
        support::assert_stored_columns(&int8, &fp16, 0, RANK, support::INT8_ROW_TOL);
        support::assert_stored_columns(&int8, &fp16, RANK, RANK + INDEX_DIM, support::INT8_ROW_TOL);
    });
}

/// The split slot end to end: the latent columns read back at the 4-bit
/// bound and the indexer key at the 8-bit bound, the footprint is the sizing
/// rate, decode stays on the composed path, and a decode step matches the
/// dense step over the same rows dequantized.
#[test]
fn a_forced_int4_slot_keeps_the_indexer_key_at_eight_bits() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let attn = attention(&config(64, INDEX_DIM));
    let mut fp16 = KVCache::new();
    let _ = run_layer(&attn, &mut fp16, 5, 9);

    with_mla_latent_quant(MlaLatentQuant::Int4, || {
        let mut int4 = KVCache::new_with_mode(KVCacheMode::Int8);
        mlxcel_core::eval(&attn.forward(&hidden_rows(0, 5), None, &mut int4));
        let before = mlxcel_core::mla::stats::snapshot();
        let got: Vec<Vec<f32>> = (5..9)
            .map(|t| values(&attn.forward(&hidden_rows(t, t + 1), None, &mut int4)))
            .collect();
        let after = mlxcel_core::mla::stats::snapshot();
        let delta = |path| after.get(path).saturating_sub(before.get(path));
        assert_eq!(delta(MlaDecodePath::AbsorbedComposed), 4);
        assert_eq!(delta(MlaDecodePath::AbsorbedFused), 0);
        assert_eq!(delta(MlaDecodePath::Decompressed), 0);

        assert_eq!(int4.stored_latent_bits(), Some(4));
        // 32 packed latent bytes and 64 packed indexer bytes, two groups of
        // fp16 scale and bias, fp16 rope keys.
        let rate =
            mlxcel_core::mla::quantized_key_slot_bytes_per_token(64, 64, ROPE as usize, 4, 64);
        assert_eq!(rate, 32 + 64 + 8 + 2 * ROPE as usize);
        assert_eq!(int4.bytes_per_reserved_token(), rate);
        support::assert_stored_columns(&int4, &fp16, 0, RANK, support::INT4_ROW_TOL);
        support::assert_stored_columns(&int4, &fp16, RANK, RANK + INDEX_DIM, support::INT8_ROW_TOL);

        let mut twin = KVCache::new_with_mode(KVCacheMode::Int8);
        mlxcel_core::eval(&attn.forward(&hidden_rows(0, 7), None, &mut twin));
        let pair = values(&attn.forward(&hidden_rows(7, 9), None, &mut twin));
        let want = support::last_position(&pair, HIDDEN);
        let err = support::max_rel_error(&got[3], &want);
        assert!(
            err < support::SAME_ROWS_TOL,
            "int4 decode off its stored rows by {err}"
        );
    });
}

/// With a top-k in force the sparse forward gathers packed rows from the
/// split slot and stays finite.
#[test]
fn a_forced_int4_slot_runs_sparse_prefill_and_decode() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let attn = attention(&config(3, INDEX_DIM));
    with_mla_latent_quant(MlaLatentQuant::Int4, || {
        let mut int4 = KVCache::new_with_mode(KVCacheMode::Int8);
        let steps = run_layer(&attn, &mut int4, 6, 10);
        assert_eq!(int4.stored_latent_bits(), Some(4));
        assert_eq!(int4.seq_len(), 10);
        assert!(steps.iter().flatten().all(|v| v.is_finite()));
    });
}
