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

//! The affine int8 MLA latent on Kimi Linear (issue #3162).
//!
//! The quality reference is the fp16 latent of the same weights, held to the
//! #3155 codec bound widened for a full layer's output, as in
//! `deepseek_v3::latent_tests`. Paths that read the same stored rows (the
//! batched and single-row decode) are held to float accuracy by the shared
//! model-owned harness.

use mlxcel_core::cache::{KVCacheMode, SequenceId};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::mla::MlaDecodePath;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::tests::{hybrid_batching_config, hybrid_batching_weights, mla_config, noise};
use super::{KimiLinearCache, KimiLinearModel, KimiMLAAttention};

/// The #3155 int8 codec bound (2e-2 relative on the latent attention output)
/// after the layer's `unembed_out` and `o_proj`, which cancel part of the
/// output and so raise the relative error, as `deepseek_v3::latent_tests`
/// found on its fixture.
const INT8_LAYER_TOL: f32 = 5e-2;
const HIDDEN: i32 = 32;

/// An MLA layer on pseudo-random weights with `kv_lora_rank` `rank` (64 is
/// one affine group) and rope width `rope`.
fn attention(rank: usize, rope: usize) -> KimiMLAAttention {
    attention_in(rank, rope, mlxcel_core::dtype::FLOAT32)
}

/// [`attention`] with every weight cast to `dtype`.
fn attention_in(rank: usize, rope: usize, dtype: i32) -> KimiMLAAttention {
    let mut config = mla_config(rank);
    config.qk_rope_head_dim = Some(rope);
    let heads = config.num_attention_heads as i32;
    let q_head = config.q_head_dim() as i32;
    let nope = config.qk_nope() as i32;
    let value = config.v_head() as i32;
    let (rank, rope) = (rank as i32, rope as i32);
    let prefix = "model.layers.0.self_attn";
    let mut weights = WeightMap::new();
    let mut seed = 3162u64;
    let mut put = |name: &str, shape: &[i32], scale: f32| {
        seed += 1;
        let n = shape.iter().product::<i32>() as usize;
        weights.insert(
            format!("{prefix}.{name}.weight"),
            mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(&noise(seed, n, scale), shape),
                dtype,
            ),
        );
    };
    put("q_proj", &[heads * q_head, HIDDEN], 0.4);
    put("kv_a_proj_with_mqa", &[rank + rope, HIDDEN], 0.4);
    put("embed_q", &[heads, rank, nope], 0.6);
    put("unembed_out", &[heads, value, rank], 0.6);
    put("o_proj", &[HIDDEN, heads * value], 0.4);
    weights.insert(
        format!("{prefix}.kv_a_layernorm.weight"),
        mlxcel_core::ones(&[rank], dtype),
    );
    KimiMLAAttention::from_weights(&weights, &config, prefix)
        .unwrap_or_else(|e| panic!("random MLA fixture must load: {e}"))
}

fn hidden(tokens: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(
        &noise(11, (tokens * HIDDEN) as usize, 1.0),
        &[1, tokens, HIDDEN],
    )
}

fn token(x: &MlxArray, i: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::slice(x, &[0, i, 0], &[1, i + 1, HIDDEN])
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

/// An additive `[1, 1, 1, keys]` mask that blocks key 1 and leaves the rest.
fn block_key_one(keys: i32) -> UniquePtr<MlxArray> {
    let row: Vec<f32> = (0..keys)
        .map(|k| if k == 1 { f32::NEG_INFINITY } else { 0.0 })
        .collect();
    mlxcel_core::from_slice_f32(&row, &[1, 1, 1, keys])
}

/// Prefill `prefill` tokens, then decode one token at a time up to `total`,
/// masking each decode step when `masked`. Returns each step's output.
fn decode(
    attn: &KimiMLAAttention,
    cache: &mut KVCache,
    prefill: i32,
    total: i32,
    masked: bool,
) -> Vec<Vec<f32>> {
    let x = hidden(total);
    let head = mlxcel_core::slice(&x, &[0, 0, 0], &[1, prefill, HIDDEN]);
    mlxcel_core::eval(&attn.forward(&head, None, cache));
    (prefill..total)
        .map(|t| {
            let mask = masked.then(|| block_key_one(t + 1));
            values(&attn.forward(&token(&x, t), mask.as_deref(), cache))
        })
        .collect()
}

fn path_delta(
    before: &mlxcel_core::mla::MlaDispatchCounts,
    after: &mlxcel_core::mla::MlaDispatchCounts,
    path: MlaDecodePath,
) -> u64 {
    after.get(path).saturating_sub(before.get(path))
}

/// An int8 cache stores the quantized latent, every single-token step takes
/// an absorbed quantized path (fused or composed) and none is recorded as
/// decompressed, and the output tracks the fp16 latent within the codec bound.
#[test]
fn unmasked_int8_decode_tracks_the_fp16_latent_on_the_absorbed_path() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let attn = attention(64, 4);
    let mut fp16 = KVCache::new();
    let want = decode(&attn, &mut fp16, 5, 9, false);
    assert!(!fp16.holds_quantized_latent());

    let mut int8 = KVCache::new_with_mode(KVCacheMode::Int8);
    let x = hidden(9);
    let head = mlxcel_core::slice(&x, &[0, 0, 0], &[1, 5, HIDDEN]);
    mlxcel_core::eval(&attn.forward(&head, None, &mut int8));
    let before = mlxcel_core::mla::stats::snapshot();
    let got: Vec<Vec<f32>> = (5..9)
        .map(|t| values(&attn.forward(&token(&x, t), None, &mut int8)))
        .collect();
    let after = mlxcel_core::mla::stats::snapshot();
    assert!(int8.holds_quantized_latent());
    assert_eq!(int8.seq_len(), 9);
    let absorbed = path_delta(&before, &after, MlaDecodePath::AbsorbedFused)
        + path_delta(&before, &after, MlaDecodePath::AbsorbedComposed);
    assert!(absorbed >= 4, "every int8 decode step must be absorbed");
    assert_eq!(
        path_delta(&before, &after, MlaDecodePath::Decompressed),
        0,
        "no int8 decode step may run decompressed"
    );

    for (step, (got, want)) in got.iter().zip(&want).enumerate() {
        assert!(got.iter().all(|v| v.is_finite()), "step {step}");
        let err = max_rel_error(got, want);
        assert!(
            err > 0.0,
            "step {step}: the int8 arm must not be the fp16 arm"
        );
        assert!(err < INT8_LAYER_TOL, "step {step}: relative error {err}");
    }
}

/// A masked single-token step over a quantized window folds the additive mask
/// into the rope-score term: it takes the composed path, tracks the dense
/// masked result within the codec bound, and differs from the unmasked step.
#[test]
fn masked_int8_decode_matches_the_dense_masked_result() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let attn = attention(64, 4);
    let want = decode(&attn, &mut KVCache::new(), 4, 8, true);
    let before = mlxcel_core::mla::stats::snapshot();
    let got = decode(
        &attn,
        &mut KVCache::new_with_mode(KVCacheMode::Int8),
        4,
        8,
        true,
    );
    let after = mlxcel_core::mla::stats::snapshot();
    assert!(
        path_delta(&before, &after, MlaDecodePath::AbsorbedComposed) >= 4,
        "a masked int8 step must take the composed quantized path"
    );
    let unmasked = decode(
        &attn,
        &mut KVCache::new_with_mode(KVCacheMode::Int8),
        4,
        8,
        false,
    );
    for (step, ((got, want), open)) in got.iter().zip(&want).zip(&unmasked).enumerate() {
        let err = max_rel_error(got, want);
        assert!(err < INT8_LAYER_TOL, "step {step}: relative error {err}");
        assert!(
            max_rel_error(got, open) > INT8_LAYER_TOL,
            "step {step}: the mask must change the int8 output"
        );
    }
}

/// The two-layer KDA + MLA model (layer 0 KDA, layer 1 MLA) at `rank`, with
/// `mode` injected as a uniform table when given.
fn model(rank: usize, mode: Option<KVCacheMode>) -> KimiLinearModel {
    let mut config = hybrid_batching_config();
    config.kv_lora_rank = rank;
    let model = KimiLinearModel::from_weights(&hybrid_batching_weights(&config), &config)
        .expect("two-layer KDA/MLA model loads");
    if let Some(mode) = mode {
        model.set_kv_cache_layer_modes(vec![mode; 2]);
    }
    model
}

fn mla_cache(caches: &[KimiLinearCache]) -> &KVCache {
    match &caches[1] {
        KimiLinearCache::MLA(kv) => kv,
        KimiLinearCache::Delta(_) => panic!("layer 1 must be MLA"),
    }
}

fn kda_state(caches: &[KimiLinearCache]) -> Vec<Vec<f32>> {
    match &caches[0] {
        KimiLinearCache::Delta(state) => [
            &state.q_conv_state,
            &state.k_conv_state,
            &state.v_conv_state,
            &state.ssm_state,
        ]
        .iter()
        .map(|field| values(field.as_deref().expect("prefill fills every KDA field")))
        .collect(),
        KimiLinearCache::MLA(_) => panic!("layer 0 must be KDA"),
    }
}

/// An int8 table builds an int8 MLA cache that stores the quantized latent,
/// reports int8 for the MLA layer and no KV for the KDA layer, and leaves the
/// KDA state exactly as the fp16 model computes it. A Turbo table builds fp16.
#[test]
fn an_int8_table_builds_int8_mla_caches_and_leaves_kda_state_unchanged() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let fp16 = model(64, None);
    let int8 = model(64, Some(KVCacheMode::Int8));
    assert_eq!(fp16.kv_cache_layer_modes(), None);
    assert_eq!(
        int8.kv_cache_layer_modes(),
        Some(vec![KVCacheMode::Int8; 2])
    );
    assert_eq!(
        fp16.kv_cache_storage_modes(),
        Some(vec![None, Some(KVCacheMode::Fp16)])
    );
    assert_eq!(
        int8.kv_cache_storage_modes(),
        Some(vec![None, Some(KVCacheMode::Int8)])
    );

    let ids = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5], &[1, 5]);
    let mut fp16_caches = fp16.make_kimi_caches();
    let mut int8_caches = int8.make_kimi_caches();
    assert_eq!(mla_cache(&int8_caches).mode, KVCacheMode::Int8);
    mlxcel_core::eval(&fp16.forward(&ids, &mut fp16_caches));
    mlxcel_core::eval(&int8.forward(&ids, &mut int8_caches));
    assert!(mla_cache(&int8_caches).holds_quantized_latent());
    assert!(!mla_cache(&fp16_caches).holds_quantized_latent());
    assert_eq!(kda_state(&int8_caches), kda_state(&fp16_caches));

    let turbo = model(64, Some(KVCacheMode::Turbo4Asym));
    assert_eq!(mla_cache(&turbo.make_kimi_caches()).mode, KVCacheMode::Fp16);
    assert_eq!(
        turbo.kv_cache_storage_modes(),
        Some(vec![None, Some(KVCacheMode::Fp16)])
    );
}

/// Batched decode over quantized rows matches single-row decode through the
/// shared model-owned harness (multi-step, paged-length sync, fresh and mixed
/// rows), and the model-owned MLA caches it decodes from hold the quantized
/// latent and are never paged-backed (`MlaLatentCache::wrap` refuses those).
#[test]
fn batched_int8_decode_matches_single_row_decode() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let make = || model(64, Some(KVCacheMode::Int8));
    crate::models::batched_decode_harness::assert_model_owned_multistep_with_paged_lengths_prepared_b1(
        make,
        &[0, 3],
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
            let kv = mla_cache(caches);
            assert!(kv.holds_quantized_latent());
            assert!(!kv.is_paged_backed());
            assert_eq!(kv.seq_len(), 4);
        })
        .expect("the prepared sequence exists");
}

/// At rank 512 and rope 64 (the Kimi-Linear-48B geometry) the runtime MLA
/// cache allocates 672 bytes per reserved row, and `kv_arch`, which
/// `mlxcel inspect` reports, charges that rate on the MLA layers only.
#[test]
fn runtime_bytes_match_the_inspect_rate_on_the_mla_layers() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let attn = attention(512, 64);
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    mlxcel_core::eval(&attn.forward(&hidden(10), None, &mut cache));
    assert!(cache.holds_quantized_latent());
    assert_eq!(cache.bytes_per_reserved_token(), 672);

    let config = serde_json::json!({
        "model_type": "kimi_linear",
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "hidden_size": HIDDEN,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 4,
        "v_head_dim": 4,
        "linear_attn_config": { "kda_layers": [1, 2, 3] },
    });
    let e = crate::execution::kv_arch::estimate_kv_arch_from_config(
        &config,
        2048,
        KVCacheMode::Int8,
        1,
    )
    .expect("estimate");
    assert_eq!(
        e.marginal_bytes_per_token,
        cache.bytes_per_reserved_token() as u64
    );
}

/// A bf16 layer over an int8 latent keeps bf16 activations. The window
/// dequantizes to FP16, and bf16 against FP16 promotes to float32 in MLX, so
/// without the cast back a prefill returned float32, which on the real
/// checkpoint carried float32 into the KDA state and ran every later decode
/// step in float32 (2.5x slower decode on Kimi-Linear-48B).
#[test]
fn a_bf16_layer_keeps_bf16_over_an_int8_latent() {
    let bf16 = mlxcel_core::dtype::BFLOAT16;
    let attn = attention_in(64, 64, bf16);
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let prefill = attn.forward(&mlxcel_core::astype(&hidden(5), bf16), None, &mut cache);
    assert_eq!(
        mlxcel_core::array_dtype(&prefill),
        bf16,
        "prefill output dtype"
    );
    let step = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&noise(29, HIDDEN as usize, 1.0), &[1, 1, HIDDEN]),
        bf16,
    );
    let decode = attn.forward(&step, None, &mut cache);
    assert_eq!(
        mlxcel_core::array_dtype(&decode),
        bf16,
        "decode output dtype"
    );
}

/// Only the MLA index of the table decides the cache mode: a table that
/// differs from fp16 on the KDA index alone builds fp16 MLA caches and no
/// quantized latent, and a table that is int8 on the MLA index alone builds an
/// int8 cache whatever the KDA index holds.
#[test]
fn a_table_differing_on_the_kda_index_alone_does_not_change_the_mla_cache() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let kda_only = model(64, None);
    kda_only.set_kv_cache_layer_modes(vec![KVCacheMode::Int8, KVCacheMode::Fp16]);
    assert_eq!(
        kda_only.kv_cache_storage_modes(),
        Some(vec![None, Some(KVCacheMode::Fp16)])
    );
    assert_eq!(
        mla_cache(&kda_only.make_kimi_caches()).mode,
        KVCacheMode::Fp16
    );

    let mla_only = model(64, None);
    mla_only.set_kv_cache_layer_modes(vec![KVCacheMode::Fp16, KVCacheMode::Int8]);
    assert_eq!(
        mla_only.kv_cache_storage_modes(),
        Some(vec![None, Some(KVCacheMode::Int8)])
    );
    assert_eq!(
        mla_cache(&mla_only.make_kimi_caches()).mode,
        KVCacheMode::Int8
    );
}

/// `--mla-latent-quant int4` on the Kimi Linear MLA layer (issue #3179): the
/// `int8` storage packs the latent at 4 bits, the rows read back within the
/// 4-bit codec step of the fp16 latent, decode stays on the composed path,
/// and a decode step matches the dense step over the same rows dequantized.
/// The KDA layer holds no KV and is untouched (the model-level test above).
#[test]
fn a_forced_int4_latent_stores_four_bit_rows_and_decodes_composed() {
    use crate::models::mla_latent_test_support as support;
    use mlxcel_core::mla::{MlaLatentQuant, with_mla_latent_quant};

    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let attn = attention(64, 4);
    let mut fp16 = KVCache::new();
    let _ = decode(&attn, &mut fp16, 5, 9, false);
    let x = hidden(9);
    let rows = |from: i32, to: i32| mlxcel_core::slice(&x, &[0, from, 0], &[1, to, HIDDEN]);

    with_mla_latent_quant(MlaLatentQuant::Int4, || {
        let mut int4 = KVCache::new_with_mode(KVCacheMode::Int8);
        mlxcel_core::eval(&attn.forward(&rows(0, 5), None, &mut int4));
        let before = mlxcel_core::mla::stats::snapshot();
        let got: Vec<Vec<f32>> = (5..9)
            .map(|t| values(&attn.forward(&token(&x, t), None, &mut int4)))
            .collect();
        let after = mlxcel_core::mla::stats::snapshot();
        assert_eq!(
            path_delta(&before, &after, MlaDecodePath::AbsorbedComposed),
            4
        );
        assert_eq!(path_delta(&before, &after, MlaDecodePath::AbsorbedFused), 0);

        assert_eq!(int4.stored_latent_bits(), Some(4));
        assert_eq!(
            int4.bytes_per_reserved_token(),
            mlxcel_core::mla::quantized_latent_bytes_per_token(64, 4, 4, 64)
        );
        support::assert_stored_columns(&int4, &fp16, 0, 64, support::INT4_ROW_TOL);

        let mut twin = KVCache::new_with_mode(KVCacheMode::Int8);
        mlxcel_core::eval(&attn.forward(&rows(0, 7), None, &mut twin));
        let pair = values(&attn.forward(&rows(7, 9), None, &mut twin));
        let want = support::last_position(&pair, HIDDEN as usize);
        let err = support::max_rel_error(&got[3], &want);
        assert!(
            err < support::SAME_ROWS_TOL,
            "int4 decode off its stored rows by {err}"
        );
    });
}
