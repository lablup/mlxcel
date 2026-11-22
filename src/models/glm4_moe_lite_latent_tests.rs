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

//! The affine int8 MLA latent on GLM-4.7-Flash (issue #3161).
//!
//! The port is in place but off in production: `glm4_moe_lite` is not in
//! `mla::QUANTIZED_LATENT_FAMILIES` because the real-checkpoint trace showed
//! decided-position disagreements. These tests opt the layers in.
//!
//! The quality reference is the fp16 latent of the same weights, held to the
//! #3155 codec bound widened for full-model logits. Paths that read the same
//! stored rows (batched and sequential decode, the MTP verify block and the
//! decode chain, a rolled-back slot and a cache that never saw the rejected
//! rows) are held to float accuracy.

use mlxcel_core::cache::{KVCacheMode, SequenceId};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::mla::MlaDecodePath;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{Glm4MoeLiteModel, ModelArgs};

#[path = "glm4_moe_lite_latent_option_tests.rs"]
mod latent_option;

const VOCAB: i32 = 32;
const HIDDEN: i32 = 16;
const HEADS: i32 = 2;
const NOPE: i32 = 8;
const ROPE: i32 = 8;
const V_HEAD: i32 = 8;
const INTER: i32 = 16;
const LAYERS: usize = 2;
/// One affine group: the smallest rank the int8 latent stores.
const RANK: i32 = 64;

/// The #3155 codec bound on full-model logits, relative to the largest fp16
/// logit of the step.
const INT8_LOGIT_TOL: f32 = 5e-2;
/// Float accumulation-order noise between two paths over the same stored rows.
const SAME_ROWS_TOL: f32 = 1e-4;

fn args(rank: i32, rope: i32) -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "glm4_moe_lite",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "moe_intermediate_size": INTER,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": HEADS,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "kv_lora_rank": rank,
        "q_lora_rank": null,
        "qk_rope_head_dim": rope,
        "qk_nope_head_dim": NOPE,
        "v_head_dim": V_HEAD,
        "n_routed_experts": null,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "first_k_dense_replace": LAYERS,
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.0
    }))
    .expect("tiny args parse")
}

/// A dense `LAYERS`-deep model on byte-reproducible pseudo-random weights.
fn model_with(rank: i32, rope: i32) -> Glm4MoeLiteModel {
    model_in(rank, rope, mlxcel_core::dtype::FLOAT32)
}

/// [`model_with`] with every weight stored in `dtype`, so the activations run
/// in it.
fn model_in(rank: i32, rope: i32, dtype: i32) -> Glm4MoeLiteModel {
    let _runtime = crate::initialize_runtime();
    let mut state = 0x3161u64;
    let mut weights = WeightMap::new();
    let mut put = |key: String, shape: &[i32]| {
        let n = shape.iter().product::<i32>() as usize;
        let norm = key.ends_with("norm.weight");
        let data: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let v = (((state >> 33) as u32) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
                if norm { 1.0 + 0.2 * v } else { 0.5 * v }
            })
            .collect();
        weights.insert(
            key,
            mlxcel_core::astype(&mlxcel_core::from_slice_f32(&data, shape), dtype),
        );
    };
    put("model.embed_tokens.weight".into(), &[VOCAB, HIDDEN]);
    put("model.norm.weight".into(), &[HIDDEN]);
    put("lm_head.weight".into(), &[VOCAB, HIDDEN]);
    for layer in 0..LAYERS {
        let p = format!("model.layers.{layer}");
        put(format!("{p}.input_layernorm.weight"), &[HIDDEN]);
        put(format!("{p}.post_attention_layernorm.weight"), &[HIDDEN]);
        let a = format!("{p}.self_attn");
        put(
            format!("{a}.q_proj.weight"),
            &[HEADS * (NOPE + rope), HIDDEN],
        );
        put(
            format!("{a}.kv_a_proj_with_mqa.weight"),
            &[rank + rope, HIDDEN],
        );
        put(format!("{a}.kv_a_layernorm.weight"), &[rank]);
        put(format!("{a}.embed_q.weight"), &[HEADS, rank, NOPE]);
        put(format!("{a}.unembed_out.weight"), &[HEADS, V_HEAD, rank]);
        put(format!("{a}.o_proj.weight"), &[HIDDEN, HEADS * V_HEAD]);
        put(format!("{p}.mlp.gate_proj.weight"), &[INTER, HIDDEN]);
        put(format!("{p}.mlp.up_proj.weight"), &[INTER, HIDDEN]);
        put(format!("{p}.mlp.down_proj.weight"), &[HIDDEN, INTER]);
    }
    Glm4MoeLiteModel::from_weights(&weights, &args(rank, rope)).expect("tiny GLM builds")
}

/// [`model_with`] with the quantizing view switched on in every layer. The
/// production model leaves it off, since `glm4_moe_lite` is not in
/// `mla::QUANTIZED_LATENT_FAMILIES` (its real-checkpoint trace failed the gate);
/// these tests keep the ported path working for a later fix.
fn opted_in(rank: i32, rope: i32) -> Glm4MoeLiteModel {
    let mut model = model_with(rank, rope);
    for layer in &mut model.layers {
        layer.self_attn.quantized_latent = true;
    }
    model
}

fn model() -> Glm4MoeLiteModel {
    opted_in(RANK, ROPE)
}

/// The production model follows the allowlist: GLM-4.7-Flash is not in it, so
/// an `int8` cache keeps its own per-token rows and no layer stores the affine
/// latent.
#[test]
fn production_glm_keeps_the_per_token_path_under_int8() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    assert!(!mlxcel_core::mla::stores_quantized_latent_pair(
        "glm4_moe_lite"
    ));
    let model = model_with(RANK, ROPE);
    assert!(model.layers.iter().all(|l| !l.self_attn.quantized_latent));
    let mut caches = caches(KVCacheMode::Int8);
    mlxcel_core::eval(&model.forward(&ids(&PROMPT), &mut caches, None));
    mlxcel_core::eval(&model.forward(&ids(&[DECODE[0]]), &mut caches, None));
    assert!(caches.iter().all(|c| !c.holds_quantized_latent()));
}

fn ids(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn values(a: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::utils::array_to_vec_f32(&a)
}

/// Logits of position `i` of a `[1, L, V]` output.
fn row(logits: &MlxArray, i: i32) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(logits);
    values(&mlxcel_core::slice(
        logits,
        &[0, i, 0],
        &[shape[0], i + 1, shape[2]],
    ))
}

fn caches(mode: KVCacheMode) -> Vec<KVCache> {
    (0..LAYERS).map(|_| KVCache::new_with_mode(mode)).collect()
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

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

const PROMPT: [i32; 5] = [3, 17, 9, 25, 4];
const DECODE: [i32; 4] = [30, 12, 7, 21];

/// Int8 decode steps take the absorbed quantized path and never the
/// decompressed one, the cache holds the quantized latent, and the logits
/// track the fp16 latent within the codec bound. The fp16 arm holds no
/// quantized rows.
#[test]
fn int8_decode_tracks_the_fp16_latent_on_the_absorbed_path() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model();
    let run = |mode: KVCacheMode| {
        let mut caches = caches(mode);
        mlxcel_core::eval(&model.forward(&ids(&PROMPT), &mut caches, None));
        let before = mlxcel_core::mla::stats::snapshot();
        let steps: Vec<Vec<f32>> = DECODE
            .iter()
            .map(|t| row(&model.forward(&ids(&[*t]), &mut caches, None), 0))
            .collect();
        let after = mlxcel_core::mla::stats::snapshot();
        let delta = move |path| after.get(path).saturating_sub(before.get(path));
        (steps, caches, delta)
    };
    let (want, fp16_caches, _) = run(KVCacheMode::Fp16);
    assert!(fp16_caches.iter().all(|c| !c.holds_quantized_latent()));

    let (got, int8_caches, delta) = run(KVCacheMode::Int8);
    assert!(int8_caches.iter().all(KVCache::holds_quantized_latent));
    let absorbed = delta(MlaDecodePath::AbsorbedComposed) + delta(MlaDecodePath::AbsorbedFused);
    assert!(
        absorbed >= (DECODE.len() * LAYERS) as u64,
        "absorbed steps {absorbed}"
    );
    assert_eq!(delta(MlaDecodePath::Decompressed), 0);

    for (step, (got, want)) in got.iter().zip(&want).enumerate() {
        let err = max_rel_error(got, want);
        assert!(
            err > 0.0,
            "step {step}: the int8 arm must not be the fp16 arm"
        );
        assert!(err < INT8_LOGIT_TOL, "step {step}: relative error {err}");
    }
}

/// A bf16 model keeps bf16 activations on the int8 latent: a multi-token step
/// reads the dequantized window in the queries' dtype, so the prefill and a
/// multi-token continuation return logits in the fp16 arm's dtype instead of
/// promoting the rest of the network to f32 (#3161).
#[test]
fn a_bf16_model_stays_bf16_over_the_int8_latent() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut model = model_in(RANK, ROPE, mlxcel_core::dtype::BFLOAT16);
    for layer in &mut model.layers {
        layer.self_attn.quantized_latent = true;
    }
    for mode in [KVCacheMode::Fp16, KVCacheMode::Int8] {
        let mut caches = caches(mode);
        let prefill = model.forward(&ids(&PROMPT), &mut caches, None);
        let decode = model.forward(&ids(&[DECODE[0]]), &mut caches, None);
        let continuation = model.forward(&ids(&DECODE[1..]), &mut caches, None);
        assert_eq!(
            caches.iter().all(KVCache::holds_quantized_latent),
            mode == KVCacheMode::Int8
        );
        for (what, logits) in [
            ("prefill", &prefill),
            ("decode", &decode),
            ("continuation", &continuation),
        ] {
            assert_eq!(
                mlxcel_core::array_dtype(logits),
                mlxcel_core::dtype::BFLOAT16,
                "{mode}: {what} logits left bf16"
            );
        }
    }
}

/// The batched decode over quantized rows equals the sequential decode over
/// the same rows and tracks the fp16-latent batch within the codec bound.
#[test]
fn batched_int8_rows_match_sequential_and_track_the_fp16_latent() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model();
    let prompts: [&[i32]; 3] = [&[1], &[2, 3], &[4, 5, 6, 7]];
    let next = [7, 8, 9];
    let prefilled = |mode: KVCacheMode| -> Vec<Vec<KVCache>> {
        prompts
            .iter()
            .map(|prompt| {
                let mut c = caches(mode);
                mlxcel_core::eval(&model.forward(&ids(prompt), &mut c, None));
                c
            })
            .collect()
    };
    let batched = |mode: KVCacheMode| -> Vec<f32> {
        let mut rows = prefilled(mode);
        let mut refs: Vec<&mut [KVCache]> = rows.iter_mut().map(Vec::as_mut_slice).collect();
        let input = mlxcel_core::from_slice_i32(&next, &[next.len() as i32, 1]);
        let logits = model.forward_batched(&input, &mut refs, None);
        assert!(
            rows.iter()
                .flatten()
                .all(|c| { c.holds_quantized_latent() == (mode == KVCacheMode::Int8) })
        );
        values(&logits)
    };
    let int8_batched = batched(KVCacheMode::Int8);
    let mut int8_rows = prefilled(KVCacheMode::Int8);
    let int8_sequential: Vec<f32> = int8_rows
        .iter_mut()
        .zip(next)
        .flat_map(|(c, token)| values(&model.forward(&ids(&[token]), c, None)))
        .collect();
    let err = max_rel_error(&int8_batched, &int8_sequential);
    assert!(err < 1e-5, "batched vs sequential over int8 rows: {err}");
    let err = max_rel_error(&int8_batched, &batched(KVCacheMode::Fp16));
    assert!(err < INT8_LOGIT_TOL, "int8 vs fp16 latent batch: {err}");
}

/// The MTP verify block over an int8 slot reads the same quantized rows the
/// decode chain reads, so its logits and argmax match the int8 chain (the
/// premise of an int8 MTP run emitting the int8 greedy stream), and the slot
/// stores the quantized latent.
#[test]
fn int8_verify_block_matches_the_int8_decode_chain() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model();
    LanguageModel::set_kv_cache_layer_modes(&model, vec![KVCacheMode::Int8; LAYERS]);
    let block = [9, 4, 13];

    let mut chain = model.make_configured_caches();
    mlxcel_core::eval(&model.forward(&ids(&PROMPT), &mut chain, None));
    let chain_rows: Vec<Vec<f32>> = block
        .iter()
        .map(|t| row(&model.forward(&ids(&[*t]), &mut chain, None), 0))
        .collect();

    let mut blocked = model.make_configured_caches();
    mlxcel_core::eval(&model.forward(&ids(&PROMPT), &mut blocked, None));
    let (logits, _hidden) = model.forward_verify(&ids(&block), &mut blocked);
    assert!(blocked.iter().all(KVCache::holds_quantized_latent));
    for (i, chain_row) in chain_rows.iter().enumerate() {
        let block_row = row(&logits, i as i32);
        let err = max_rel_error(&block_row, chain_row);
        assert!(err < SAME_ROWS_TOL, "position {i}: block vs chain {err}");
        assert_eq!(argmax(&block_row), argmax(chain_row), "position {i} argmax");
    }
}

/// A verify block plus rollback over an int8 slot leaves the live window the
/// fp16 rollback leaves (the same length on every layer, still the quantized
/// latent), and the next decode step reads exactly the rows a cache that
/// never saw the rejected drafts holds.
#[test]
fn int8_verify_and_rollback_leave_the_accepted_window() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let block = [9, 4, 13];
    let accepted = 1usize;
    let rejected = (block.len() - accepted - 1) as i32;
    let after_rollback = |mode: KVCacheMode| {
        let model = model();
        LanguageModel::set_kv_cache_layer_modes(&model, vec![mode; LAYERS]);
        let seq = Some(SequenceId::from_raw(11));
        model.reset_mtp_sequence_state(seq);
        let _ = model.forward_prefill_with_last_hidden_for_sequence(&ids(&PROMPT), seq);
        let _ = model.forward_verify_for_sequence(&ids(&block), seq);
        let trimmed = model.rollback_speculative_cache_for_sequence(seq, rejected);
        assert_eq!(trimmed, rejected, "{mode}");
        let (lens, quantized) = model.mtp_sequence_state.with_sequence_state(seq, |c| {
            (
                c.iter().map(KVCache::seq_len).collect::<Vec<_>>(),
                c.iter().all(KVCache::holds_quantized_latent),
            )
        });
        let next = model
            .mtp_sequence_state
            .with_sequence_state(seq, |c| row(&model.forward(&ids(&[21]), c, None), 0));

        // A cache that took the prompt and the accepted rows as a verify
        // block of their own, then the same next token.
        let mut reference = model.make_configured_caches();
        mlxcel_core::eval(&model.forward(&ids(&PROMPT), &mut reference, None));
        let _ = model.forward_verify(&ids(&block[..=accepted]), &mut reference);
        let want = row(&model.forward(&ids(&[21]), &mut reference, None), 0);
        (lens, quantized, next, want)
    };

    let (fp16_lens, fp16_quantized, fp16_next, _) = after_rollback(KVCacheMode::Fp16);
    let (int8_lens, int8_quantized, int8_next, int8_want) = after_rollback(KVCacheMode::Int8);
    let expected = (PROMPT.len() + accepted + 1) as i32;
    assert_eq!(fp16_lens, vec![expected; LAYERS]);
    assert_eq!(int8_lens, fp16_lens);
    assert!(!fp16_quantized);
    assert!(int8_quantized);
    let err = max_rel_error(&int8_next, &int8_want);
    assert!(
        err < SAME_ROWS_TOL,
        "rolled-back vs never-drafted int8 window: {err}"
    );
    let err = max_rel_error(&int8_next, &fp16_next);
    assert!(err < INT8_LOGIT_TOL, "int8 vs fp16 rollback: {err}");
}

/// At the GLM-4.7-Flash latent geometry (rank 512, rope 64) an opted-in
/// runtime cache allocates 672 bytes per reserved row on every layer, the
/// `mla::quantized_latent_bytes_per_token` rate `kv_arch` would report once
/// the family is listed.
#[test]
fn runtime_bytes_per_token_are_the_affine_rate_at_rank_512() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = opted_in(512, 64);
    let mut caches = caches(KVCacheMode::Int8);
    mlxcel_core::eval(&model.forward(&ids(&PROMPT), &mut caches, None));
    for c in &caches {
        assert!(c.holds_quantized_latent());
        assert_eq!(
            c.bytes_per_reserved_token(),
            mlxcel_core::mla::quantized_latent_bytes_per_token(512, 64, 8, 64)
        );
        assert_eq!(c.bytes_per_reserved_token(), 672);
    }
}
