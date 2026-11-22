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

//! `--mla-latent-quant int4` on DeepSeek V2 under absorbed decode (issue
//! #3179): the #3155 path with the latent packed at 4 bits.

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::mla::{
    MlaAbsorbedProjections, MlaDecodePath, MlaGeometry, MlaLatentQuant, with_mla_latent_quant,
};

use super::{
    ATTN_PREFIX, HIDDEN, KVCache, load_mla_attention, mla_test_args, mla_test_weights, xorshift,
};
use crate::models::mla_latent_test_support as support;

/// The `int8` storage packs the latent at 4 bits, the rows read back within
/// the 4-bit codec step of the fp16 latent, a decode step takes the composed
/// path and matches the dense step over the same rows dequantized.
#[test]
fn a_forced_int4_latent_stores_four_bit_rows_and_decodes_composed() {
    let _runtime = crate::initialize_runtime();
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut args = mla_test_args();
    args.kv_lora_rank = 64;
    let weights = mla_test_weights(&args, 0x3179);
    let geometry = MlaGeometry {
        num_heads: args.num_attention_heads,
        kv_lora_rank: args.kv_lora_rank,
        qk_nope_head_dim: args.qk_nope_head_dim,
        qk_rope_head_dim: args.qk_rope_head_dim,
        v_head_dim: args.v_head_dim,
    };
    let mut attn = load_mla_attention(&weights, &args, ATTN_PREFIX).expect("load absorbed");
    attn.absorbed =
        Some(MlaAbsorbedProjections::from_kv_b_proj(&attn.kv_b_proj, geometry).expect("fold"));

    let mut state = 0x3179u64;
    let all = mlxcel_core::from_slice_f32(
        &xorshift(&mut state, 8 * HIDDEN, 0.25),
        &[1, 8, HIDDEN as i32],
    );
    let rows =
        |from: i32, to: i32| mlxcel_core::slice(&all, &[0, from, 0], &[1, to, HIDDEN as i32]);

    let mut fp16 = KVCache::new();
    mlxcel_core::eval(&attn.forward(&rows(0, 8), None, &mut fp16));

    with_mla_latent_quant(MlaLatentQuant::Int4, || {
        let mut int4 = KVCache::new_with_mode(KVCacheMode::Int8);
        mlxcel_core::eval(&attn.forward(&rows(0, 5), None, &mut int4));
        let before = mlxcel_core::mla::stats::snapshot();
        let got: Vec<Vec<f32>> = (5..8)
            .map(|t| support::f32_values(&attn.forward(&rows(t, t + 1), None, &mut int4)))
            .collect();
        let after = mlxcel_core::mla::stats::snapshot();
        let delta = |path| after.get(path).saturating_sub(before.get(path));
        assert_eq!(delta(MlaDecodePath::AbsorbedComposed), 3);
        assert_eq!(delta(MlaDecodePath::AbsorbedFused), 0);

        assert_eq!(int4.stored_latent_bits(), Some(4));
        assert_eq!(
            int4.bytes_per_reserved_token(),
            mlxcel_core::mla::quantized_latent_bytes_per_token(64, 8, 4, 64)
        );
        support::assert_stored_columns(&int4, &fp16, 0, 64, support::INT4_ROW_TOL);

        let mut twin = KVCache::new_with_mode(KVCacheMode::Int8);
        mlxcel_core::eval(&attn.forward(&rows(0, 6), None, &mut twin));
        let pair = support::f32_values(&attn.forward(&rows(6, 8), None, &mut twin));
        let want = support::last_position(&pair, HIDDEN);
        let err = support::max_rel_error(&got[2], &want);
        assert!(
            err < support::SAME_ROWS_TOL,
            "int4 decode off its stored rows by {err}"
        );
    });
}
