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

//! The affine int8 `[kv_latent | indexer key]` slot on the DSA attention
//! (issue #3163).
//!
//! Three references are used. A quantized window's own dequantized form is
//! what its stored rows mean, so selection and the packed gather must match it
//! exactly and the decode to float accuracy. Paths that read the same stored
//! rows (batched and single-row decode) are held to float accuracy. The fp16
//! latent of the same weights is the quality reference, held to the #3155
//! codec bound, and only where the indexer selects nothing: with a top-k of
//! two over synthetic keys the two arms may legitimately select different
//! rows.

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::mla::{LatentWindow, MlaDecodePath, MlaLatentCache};
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::deepseek_v32_tests::{f32_weight, synthetic_mla_weights};
use super::{DeepSeekV32Model, MLAAttention, ModelArgs, gather_topk_kv, load_mla_attention};

/// Quantized decode against dense attention over the dequantized window.
const COMPOSITION_TOL: f32 = 2e-3;
/// The #3155 int8 codec bound after the layer's `unembed_out` and `o_proj`
/// (`deepseek_v3::latent_tests::INT8_LAYER_TOL`).
const INT8_LAYER_TOL: f32 = 5e-2;

const RANK: i32 = 64;
const INDEX_DIM: i32 = 64;
const ROPE: i32 = 8;
const HIDDEN: usize = 16;

/// One latent group and one indexer-key group: the DSA slot at a tenth of the
/// shipping width (512 + 128).
fn config(index_topk: i32, index_head_dim: i32) -> ModelArgs {
    let json = format!(
        r#"{{
        "model_type": "deepseek_v32",
        "vocab_size": 16,
        "hidden_size": {HIDDEN},
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "q_lora_rank": 8,
        "kv_lora_rank": {RANK},
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": {ROPE},
        "v_head_dim": 8,
        "index_topk": {index_topk},
        "index_n_heads": 2,
        "index_head_dim": {index_head_dim}
    }}"#
    );
    serde_json::from_str(&json).expect("parse tiny DSA config")
}

fn attention(args: &ModelArgs) -> MLAAttention {
    let prefix = "model.layers.0.self_attn";
    let weights = synthetic_mla_weights(args, prefix, true);
    let weights = DeepSeekV32Model::sanitize_weights(weights, args).expect("sanitize");
    load_mla_attention(&weights, args, prefix).expect("tiny DSA attention must load")
}

fn model(args: &ModelArgs) -> DeepSeekV32Model {
    let mut weights: WeightMap = synthetic_mla_weights(args, "model.layers.0.self_attn", true);
    let hidden = HIDDEN as i32;
    let mut put = |key: &str, shape: &[i32]| {
        weights.insert(key.to_string(), f32_weight(shape));
    };
    put("model.embed_tokens.weight", &[16, hidden]);
    put("model.norm.weight", &[hidden]);
    put("lm_head.weight", &[16, hidden]);
    put("model.layers.0.input_layernorm.weight", &[hidden]);
    put("model.layers.0.post_attention_layernorm.weight", &[hidden]);
    put("model.layers.0.mlp.gate_proj.weight", &[16, hidden]);
    put("model.layers.0.mlp.up_proj.weight", &[16, hidden]);
    put("model.layers.0.mlp.down_proj.weight", &[hidden, 16]);
    let weights = DeepSeekV32Model::sanitize_weights(weights, args).expect("sanitize");
    DeepSeekV32Model::from_weights(&weights, args).expect("tiny DSA model loads")
}

/// Deterministic, varied, finite fill so every row and group differs.
fn ramp(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i + salt) * 37 + 11) % 97) as f32 - 48.0) * 0.01)
        .collect()
}

fn tensor(shape: &[i32], salt: usize) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&ramp(n as usize, salt), shape)
}

/// Token `t`'s hidden state, varied per token so every latent row differs.
fn hidden_rows(from: usize, to: usize) -> UniquePtr<MlxArray> {
    let all = ramp((to + 3) * HIDDEN * 3, 0);
    let rows: Vec<f32> = (from..to)
        .flat_map(|t| all[t * 3 * HIDDEN..t * 3 * HIDDEN + HIDDEN].iter().copied())
        .collect();
    mlxcel_core::from_slice_f32(&rows, &[1, (to - from) as i32, HIDDEN as i32])
}

fn scaled(a: &MlxArray, factor: f32) -> UniquePtr<MlxArray> {
    let factor = mlxcel_core::full_f32(&[1], factor, mlxcel_core::array_dtype(a));
    mlxcel_core::multiply(a, &factor)
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

/// A quantized `[kv_latent | indexer key]` window of `seq` rows, the way the
/// attention stores one. Built twice per test (the window is consumed), from
/// the same rows, so both copies hold identical codes.
fn quantized_window(attn: &MLAAttention, seq: i32) -> LatentWindow {
    let latent = tensor(&[1, 1, seq, RANK], 3);
    // A different scale from the latent, so a group mixing the two would show.
    let index_keys = scaled(&tensor(&[1, 1, seq, INDEX_DIM], 29), 9.0);
    let keys = mlxcel_core::concatenate(&latent, &index_keys, -1);
    let k_pe = tensor(&[1, 1, seq, ROPE], 53);
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    MlaLatentCache::wrap_with_key_width(&mut cache, attn.geometry(), (RANK + INDEX_DIM) as usize)
        .expect("a 128-wide slot is two groups")
        .update_and_fetch(keys, k_pe)
}

/// One decode query's inputs: absorbed query, rope query, indexer query and
/// indexer weights.
struct Query {
    q_projected: UniquePtr<MlxArray>,
    q_pe: UniquePtr<MlxArray>,
    index_q: UniquePtr<MlxArray>,
    index_w: UniquePtr<MlxArray>,
}

fn query() -> Query {
    Query {
        q_projected: tensor(&[1, 2, 1, RANK], 7),
        q_pe: tensor(&[1, 2, 1, ROPE], 17),
        index_q: tensor(&[1, 2, 1, INDEX_DIM], 41),
        index_w: tensor(&[1, 1, 2], 61),
    }
}

/// The fp16 sparse decode of `MLAAttention::forward`, run over a dense
/// window: slice, select, gather, attend. Returns the latent-space output and
/// the selected positions.
fn dense_sparse_decode(
    attn: &MLAAttention,
    keys: &MlxArray,
    k_pe: &MlxArray,
    q: &Query,
) -> (Vec<f32>, Vec<f32>) {
    let latent = slice_axis(keys, -1, 0, RANK);
    let index_keys = slice_axis(keys, -1, RANK, -1);
    let indices = attn
        .indexer
        .as_ref()
        .expect("the fixture has an indexer")
        .top_indices(&q.index_q, &index_keys, &q.index_w, None)
        .expect("more keys than the top-k");
    let (latent_g, k_pe_g) = gather_topk_kv(&latent, k_pe, &indices, RANK, ROPE);
    let latent_g = mlxcel_core::astype(&latent_g, mlxcel_core::dtype::FLOAT32);
    let k_pe_g = mlxcel_core::astype(&k_pe_g, mlxcel_core::dtype::FLOAT32);
    let q_pe_scaled = scaled(&q.q_pe, attn.scale);
    let pe_scores = mlxcel_core::matmul(
        &q_pe_scaled,
        &mlxcel_core::transpose_axes(&k_pe_g, &[0, 1, 3, 2]),
    );
    // SAFETY: `pe_scores` is a live local for the duration of the call.
    let output = unsafe {
        mlxcel_core::layers::attention_from_ptr(
            &q.q_projected,
            &latent_g,
            &latent_g,
            attn.scale,
            &*pe_scores as *const MlxArray,
            0.0,
            0,
        )
    };
    let indices = mlxcel_core::astype(&indices, mlxcel_core::dtype::FLOAT32);
    (values(&output), values(&indices))
}

/// Sparse decode over a quantized window equals sparse decode over its
/// dequantized window: the same rows are selected, and the attention over the
/// packed rows matches dense attention over the gathered dense rows.
#[test]
fn sparse_decode_over_a_quantized_window_equals_sparse_decode_over_its_dequantized_window() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = config(2, INDEX_DIM);
    let attn = attention(&args);
    let q = query();

    let before = mlxcel_core::mla::stats::snapshot();
    let got = values(&attn.quantized_decode(
        quantized_window(&attn, 6),
        &q.q_projected,
        &q.q_pe,
        Some((&q.index_q, &q.index_w)),
        None,
    ));
    let after = mlxcel_core::mla::stats::snapshot();
    let absorbed = |counts: &mlxcel_core::mla::MlaDispatchCounts| {
        counts.get(MlaDecodePath::AbsorbedComposed) + counts.get(MlaDecodePath::AbsorbedFused)
    };
    assert!(
        absorbed(&after) > absorbed(&before),
        "the quantized decode must record an absorbed path"
    );

    let (keys, k_pe) = quantized_window(&attn, 6).into_dense();
    let (want, _) = dense_sparse_decode(&attn, &keys, &k_pe, &q);
    assert_eq!(got.len(), 2 * RANK as usize);
    let err = max_rel_error(&got, &want);
    assert!(
        err < COMPOSITION_TOL,
        "quantized vs dequantized sparse decode: {err}"
    );
}

/// The packed gather equals gathering after dequantize, bit for bit, and the
/// indexer reads only its own columns: the selected window holds the latent
/// columns of exactly the rows the dense selection picks.
#[test]
fn the_packed_gather_equals_gathering_after_dequantize() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = config(2, INDEX_DIM);
    let attn = attention(&args);
    let q = query();

    let (selected, sparse) = attn.select_quantized_window(
        quantized_window(&attn, 6),
        Some((&q.index_q, &q.index_w)),
        None,
    );
    assert!(sparse, "six keys behind a top-k of two must select");
    assert!(selected.is_quantized(), "the gather must not dequantize");
    assert_eq!(
        selected.key_width(),
        RANK,
        "the indexer columns are dropped"
    );
    assert_eq!(selected.seq_len(), 2);
    let (got_latent, got_k_pe) = selected.into_dense();

    let (keys, k_pe) = quantized_window(&attn, 6).into_dense();
    let latent = slice_axis(&keys, -1, 0, RANK);
    let index_keys = slice_axis(&keys, -1, RANK, -1);
    let indices = attn
        .indexer
        .as_ref()
        .unwrap()
        .top_indices(&q.index_q, &index_keys, &q.index_w, None)
        .unwrap();
    let (want_latent, want_k_pe) = gather_topk_kv(&latent, &k_pe, &indices, RANK, ROPE);
    assert_eq!(values(&got_latent), values(&want_latent));
    assert_eq!(values(&got_k_pe), values(&want_k_pe));
}

/// At or under the top-k the indexer selects nothing and the whole latent is
/// attended, still packed.
#[test]
fn a_short_quantized_window_is_attended_whole() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = config(8, INDEX_DIM);
    let attn = attention(&args);
    let q = query();
    let (selected, sparse) = attn.select_quantized_window(
        quantized_window(&attn, 6),
        Some((&q.index_q, &q.index_w)),
        None,
    );
    assert!(!sparse);
    assert!(selected.is_quantized());
    assert_eq!((selected.key_width(), selected.seq_len()), (RANK, 6));
}

/// Prefill `prefill` tokens, then decode one token at a time up to `total`.
fn run_layer(
    attn: &MLAAttention,
    cache: &mut KVCache,
    prefill: usize,
    total: usize,
) -> Vec<Vec<f32>> {
    let out = attn.forward(&hidden_rows(0, prefill), None, cache);
    mlxcel_core::eval(&out);
    (prefill..total)
        .map(|t| values(&attn.forward(&hidden_rows(t, t + 1), None, cache)))
        .collect()
}

/// The single-sequence forward stores the whole slot quantized, decodes on an
/// absorbed path with no decompressed decode step, and tracks the fp16 latent
/// within the codec bound where no top-k is applied.
#[test]
fn int8_forward_stores_the_slot_quantized_and_tracks_the_fp16_latent() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = config(64, INDEX_DIM);
    let attn = attention(&args);

    let mut fp16 = KVCache::new();
    let want = run_layer(&attn, &mut fp16, 5, 9);
    assert!(!fp16.holds_quantized_latent());

    let mut int8 = KVCache::new_with_mode(KVCacheMode::Int8);
    let prefill = attn.forward(&hidden_rows(0, 5), None, &mut int8);
    mlxcel_core::eval(&prefill);
    let before = mlxcel_core::mla::stats::snapshot();
    let got: Vec<Vec<f32>> = (5..9)
        .map(|t| values(&attn.forward(&hidden_rows(t, t + 1), None, &mut int8)))
        .collect();
    let after = mlxcel_core::mla::stats::snapshot();
    assert!(int8.holds_quantized_latent());
    assert_eq!(int8.seq_len(), 9);
    // 128 packed bytes, two groups of fp16 scale and bias, fp16 rope keys.
    assert_eq!(
        int8.bytes_per_reserved_token(),
        mlxcel_core::mla::quantized_latent_bytes_per_token(128, ROPE as usize, 8, 64)
    );
    let delta = |path| after.get(path).saturating_sub(before.get(path));
    assert!(
        delta(MlaDecodePath::AbsorbedComposed) + delta(MlaDecodePath::AbsorbedFused) >= 4,
        "every int8 decode step must take an absorbed quantized path"
    );
    assert_eq!(
        delta(MlaDecodePath::Decompressed),
        0,
        "no int8 decode step may decompress the latent"
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

/// With a top-k in force the sparse forward over an int8 cache runs end to
/// end: prefill over the dequantized window, then sparse decode steps on the
/// packed rows.
#[test]
fn int8_sparse_forward_runs_prefill_and_decode() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = config(2, INDEX_DIM);
    let attn = attention(&args);
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let steps = run_layer(&attn, &mut cache, 5, 8);
    assert!(cache.holds_quantized_latent());
    assert_eq!(cache.seq_len(), 8);
    for step in &steps {
        assert_eq!(step.len(), HIDDEN);
        assert!(step.iter().all(|v| v.is_finite()));
    }
}

/// Batched decode with the selector over quantized rows matches single-row
/// decode over the same rows. Row lengths straddle the top-k, so one row
/// stays dense while the others gather packed rows.
#[test]
fn batched_decode_with_the_selector_over_quantized_rows_matches_single_row_decode() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = config(2, INDEX_DIM);
    let model = model(&args);
    let prompts: [&[i32]; 3] = [&[1], &[2, 3, 4], &[5, 6, 7, 8, 9]];
    let next = [10, 11, 12];

    let prefilled = || -> Vec<Vec<KVCache>> {
        prompts
            .iter()
            .map(|prompt| {
                let mut caches: Vec<KVCache> = (0..model.num_layers())
                    .map(|_| KVCache::new_with_mode(KVCacheMode::Int8))
                    .collect();
                let ids = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
                mlxcel_core::eval(&model.forward(&ids, &mut caches, None));
                caches
            })
            .collect()
    };

    let mut rows = prefilled();
    let batched = {
        let mut refs: Vec<&mut [KVCache]> = rows.iter_mut().map(Vec::as_mut_slice).collect();
        let ids = mlxcel_core::from_slice_i32(&next, &[next.len() as i32, 1]);
        values(&model.forward_batched(&ids, &mut refs, None))
    };
    assert!(rows.iter().flatten().all(KVCache::holds_quantized_latent));

    let mut rows = prefilled();
    let sequential: Vec<f32> = rows
        .iter_mut()
        .zip(next)
        .flat_map(|(caches, token)| {
            let ids = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
            values(&model.forward(&ids, caches, None))
        })
        .collect();
    assert!(batched.iter().all(|v| v.is_finite()));
    let err = max_rel_error(&batched, &sequential);
    assert!(
        err < 1e-5,
        "batched vs single-row decode over int8 rows: {err}"
    );
}

/// A key slot that is not a whole number of groups (64 + 16) is declined by
/// the latent view: the cache keeps its per-token rows and a dense window,
/// and the forward still runs.
#[test]
fn a_slot_the_codec_cannot_group_keeps_the_per_token_path() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = config(2, 16);
    let attn = attention(&args);
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let steps = run_layer(&attn, &mut cache, 5, 7);
    assert!(!cache.holds_quantized_latent());
    assert_eq!(cache.seq_len(), 7);
    assert!(steps.iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn the_trace_layer_index_is_read_from_the_weight_prefix() {
    assert_eq!(
        super::layer_index_from_prefix("model.layers.17.self_attn"),
        17
    );
    assert_eq!(
        super::layer_index_from_prefix("language_model.model.layers.3.self_attn"),
        3
    );
    assert_eq!(super::layer_index_from_prefix("self_attn"), -1);
}
