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

//! LongCat-Flash-Lite-Sparse (#2362): the n-gram fusion, the LSA selection
//! rule, the HF and mlx-community layouts, and a tiny model on the engine path
//! with LSA active, in fp16 and over the quantized MLA latent.

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::engine::DirectEngine;
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::mla::{MlaDecodePath, MlaLatentQuant, with_mla_latent_quant};
use mlxcel_core::utils::{array_to_vec_f32, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::*;

const VOCAB: i32 = 32;
const HIDDEN: i32 = 16;
const HEADS: i32 = 2;
const Q_LORA: i32 = 8;
const ROPE: i32 = 4;
const NOPE: i32 = 4;
const V_DIM: i32 = 4;
const FFN: i32 = 24;
const EXPERT_FFN: i32 = 8;
const ROUTED: i32 = 2;
const ZERO: i32 = 1;
const INDEX_HEADS: i32 = 2;
const INDEX_DIM: i32 = 8;
/// Small enough that the 11-token fixture runs with the selection active
/// from its fifth key on.
const INDEX_TOPK: i32 = 4;
/// One affine group: the smallest rank the quantized latent stores.
const QUANTIZABLE_RANK: i32 = 64;

const TOKENS: [i32; 11] = [3, 17, 9, 25, 4, 30, 12, 7, 21, 1, 14];

fn tiny_config(kv_lora: i32, init: i32, local: i32) -> LongcatFlashSparseConfig {
    let json = format!(
        r#"{{
            "architectures": ["LongcatCausalLM"],
            "attention_method": "LSA",
            "hidden_size": {HIDDEN},
            "ffn_hidden_size": {FFN},
            "moe_topk": 2,
            "expert_ffn_hidden_size": {EXPERT_FFN},
            "n_routed_experts": {ROUTED},
            "zero_expert_num": {ZERO},
            "num_layers": 2,
            "vocab_size": {VOCAB},
            "max_position_embeddings": 256,
            "num_attention_heads": {HEADS},
            "kv_lora_rank": {kv_lora},
            "q_lora_rank": {Q_LORA},
            "qk_rope_head_dim": {ROPE},
            "qk_nope_head_dim": {NOPE},
            "v_head_dim": {V_DIM},
            "routed_scaling_factor": 1.0,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "rope_scaling": {{
                "rope_type": "deepseek_yarn", "factor": 4, "original_max_position_embeddings": 32,
                "beta_fast": 32, "beta_slow": 1, "mscale": 1, "mscale_all_dim": 1
            }},
            "index_n_heads": {INDEX_HEADS},
            "index_head_dim": {INDEX_DIM},
            "index_topk": {INDEX_TOPK},
            "index_init_tokens": {init},
            "index_local_tokens": {local},
            "index_k_norm_type": "rms",
            "indexer_rope_interleave": true,
            "cli_factor": 2,
            "oe_vocab_size_ratio": 2,
            "oe_neighbor_num": 3,
            "oe_split_num": 2
        }}"#
    );
    serde_json::from_str(&json).expect("parse tiny longcat_flash_sparse config")
}

fn pseudo_random(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 33) as u32) as f32 / (1u64 << 31) as f32;
            (unit * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Original HF layout under the mlx-community `language_model.` root:
/// per-expert MLPs, fused `kv_b_proj`, `model.embed_tokens`, and the n-gram
/// tables and per-embedder projections as `oe_embed_tokens{i}` /
/// `oe_embed_proj{i}`. Fixed insertion order so every run builds the same
/// model.
fn tiny_weights(args: &LongcatFlashSparseConfig, kv_lora: i32) -> WeightMap {
    let mut weights = WeightMap::new();
    let mut seed = 211u64;
    let mut put = |name: String, shape: &[i32], scale: f32| {
        seed += 1;
        let n: i32 = shape.iter().product();
        let values = pseudo_random(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), n as usize, scale);
        weights.insert(
            format!("language_model.{name}"),
            mlxcel_core::from_slice_f32(&values, shape),
        );
    };
    put("model.embed_tokens.weight".into(), &[VOCAB, HIDDEN], 0.8);
    put("lm_head.weight".into(), &[VOCAB, HIDDEN], 0.6);
    let m = (args.oe_vocab_size_ratio * args.vocab_size) as i32;
    let embedders = (args.oe_split_num * (args.oe_neighbor_num - 1)) as i32;
    let emb_dim = HIDDEN / embedders;
    for i in 0..embedders {
        put(
            format!("model.oe_embed_tokens{i}.weight"),
            &[m + 2 * i + 1, emb_dim],
            0.8,
        );
        put(
            format!("model.oe_embed_proj{i}.weight"),
            &[HIDDEN, emb_dim],
            0.5,
        );
    }
    for l in 0..args.num_layers {
        let p = format!("model.layers.{l}");
        for i in 0..2 {
            let a = format!("{p}.self_attn.{i}");
            put(format!("{a}.q_a_proj.weight"), &[Q_LORA, HIDDEN], 0.5);
            put(
                format!("{a}.q_b_proj.weight"),
                &[HEADS * (NOPE + ROPE), Q_LORA],
                0.5,
            );
            put(
                format!("{a}.kv_a_proj_with_mqa.weight"),
                &[kv_lora + ROPE, HIDDEN],
                0.5,
            );
            put(
                format!("{a}.kv_b_proj.weight"),
                &[HEADS * (NOPE + V_DIM), kv_lora],
                0.5,
            );
            put(format!("{a}.o_proj.weight"), &[HIDDEN, HEADS * V_DIM], 0.4);
            put(
                format!("{p}.mlps.{i}.gate_proj.weight"),
                &[FFN, HIDDEN],
                0.3,
            );
            put(format!("{p}.mlps.{i}.up_proj.weight"), &[FFN, HIDDEN], 0.3);
            put(
                format!("{p}.mlps.{i}.down_proj.weight"),
                &[HIDDEN, FFN],
                0.3,
            );
        }
        let ix = format!("{p}.self_attn.0.indexer");
        put(
            format!("{ix}.wq_b.weight"),
            &[INDEX_HEADS * INDEX_DIM, Q_LORA],
            0.8,
        );
        put(format!("{ix}.wk.weight"), &[INDEX_DIM, HIDDEN], 0.8);
        put(
            format!("{ix}.weights_proj.weight"),
            &[INDEX_HEADS, HIDDEN],
            0.8,
        );
        put(
            format!("{p}.mlp.router.classifier.weight"),
            &[ROUTED + ZERO, HIDDEN],
            0.8,
        );
        put(
            format!("{p}.mlp.router.e_score_correction_bias"),
            &[ROUTED + ZERO],
            0.05,
        );
        for e in 0..ROUTED {
            let x = format!("{p}.mlp.experts.{e}");
            put(format!("{x}.gate_proj.weight"), &[EXPERT_FFN, HIDDEN], 0.4);
            put(format!("{x}.up_proj.weight"), &[EXPERT_FFN, HIDDEN], 0.4);
            put(format!("{x}.down_proj.weight"), &[HIDDEN, EXPERT_FFN], 0.4);
        }
    }
    let mut norms = vec![("model.norm.weight".to_string(), HIDDEN)];
    for l in 0..args.num_layers {
        let p = format!("model.layers.{l}");
        norms.push((format!("{p}.self_attn.0.indexer.k_norm.weight"), INDEX_DIM));
        for i in 0..2 {
            norms.push((format!("{p}.self_attn.{i}.q_a_layernorm.weight"), Q_LORA));
            norms.push((format!("{p}.self_attn.{i}.kv_a_layernorm.weight"), kv_lora));
            norms.push((format!("{p}.input_layernorm.{i}.weight"), HIDDEN));
            norms.push((format!("{p}.post_attention_layernorm.{i}.weight"), HIDDEN));
        }
    }
    for (name, dim) in &norms {
        put(name.clone(), &[*dim], 0.1);
    }
    let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
    for (name, _) in norms {
        let key = format!("language_model.{name}");
        let w = weights.remove(&key).expect("norm weight present");
        weights.insert(key, mlxcel_core::add(&w, &one));
    }
    weights
}

fn tiny_model_with(kv_lora: i32, init: i32, local: i32) -> LongcatFlashSparseModel {
    let args = tiny_config(kv_lora, init, local);
    let weights = sanitize_weights(tiny_weights(&args, kv_lora), &args).expect("sanitize");
    LongcatFlashSparseModel::from_weights(&weights, &args).expect("build tiny model")
}

fn tiny_model() -> LongcatFlashSparseModel {
    tiny_model_with(8, 1, 2)
}

fn ids(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn row(logits: &MlxArray, pos: i32) -> Vec<f32> {
    let r = slice_axis(logits, 1, pos, pos + 1);
    let r = mlxcel_core::astype(&r, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&r);
    array_to_vec_f32(&r)
}

fn to_vec(a: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    array_to_vec_f32(&a)
}

fn to_vec_i32(a: &MlxArray) -> Vec<i32> {
    to_vec(a).into_iter().map(|v| v as i32).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The fusion keeps the word embedding at full scale: on synthetic tables
/// `word + proj / (1 + n)` holds and the ngram family's `(word + proj) /
/// (1 + n)` does not.
#[test]
fn ngram_fusion_divides_only_the_projection() {
    let word = mlxcel_core::from_slice_f32(&[10.0, -4.0, 2.0, 0.5], &[1, 1, 4]);
    let proj = mlxcel_core::from_slice_f32(&[13.0, 26.0, -39.0, 6.5], &[1, 1, 4]);
    let n = 12;
    let got = to_vec(&fuse_ngram(&word, &proj, n));
    let want = [11.0, -2.0, -1.0, 1.0];
    assert!(max_abs_diff(&got, &want) < 1e-6, "{got:?}");
    let whole_sum: Vec<f32> = [10.0f32 + 13.0, -4.0 + 26.0, 2.0 - 39.0, 0.5 + 6.5]
        .iter()
        .map(|v| v / 13.0)
        .collect();
    assert!(
        max_abs_diff(&got, &whole_sum) > 1.0,
        "the fusion must not be the longcat_flash_ngram form"
    );
}

/// The embedding forward on tiny tables equals the word lookup plus the
/// fused projection of the concatenated lookups over `1 + n`, computed here
/// from the loaded tables; the HF per-embedder projections concatenate into
/// that one projection.
#[test]
fn ngram_embedding_uses_the_fused_projection_and_fusion() {
    let args = tiny_config(8, 1, 2);
    let raw = tiny_weights(&args, 8);
    let parts: Vec<UniquePtr<MlxArray>> = (0..4)
        .map(|i| mlxcel_core::copy(&raw[&format!("language_model.model.oe_embed_proj{i}.weight")]))
        .collect();
    let weights = sanitize_weights(raw, &args).unwrap();
    let fused = &weights["model.ngram_embeddings.post_proj.weight"];
    assert_eq!(mlxcel_core::array_shape(fused), vec![HIDDEN, HIDDEN]);
    for (i, part) in parts.iter().enumerate() {
        let cols = slice_axis(fused, 1, 4 * i as i32, 4 * (i as i32 + 1));
        assert_eq!(
            to_vec(&cols),
            to_vec(part),
            "post_proj columns of embedder {i}"
        );
    }
    assert!(weights.contains_key("model.ngram_embeddings.word_embeddings.weight"));
    assert!(weights.contains_key("model.ngram_embeddings.embedders.3.weight"));
    assert!(!weights.keys().any(|k| k.starts_with("language_model.")));

    let embedding =
        SparseNgramEmbedding::from_weights(&weights, &args, "model.ngram_embeddings").unwrap();
    let input = ids(&TOKENS[..5]);
    let mut context = None;
    let got = to_vec(&embedding.forward(&input, &mut context));

    // Reference from the same tables.
    let ctx = mlxcel_core::astype(&input, mlxcel_core::dtype::INT64);
    let shifted: HashMap<usize, UniquePtr<MlxArray>> = (2..=3)
        .map(|i| (i, NgramEmbedding::shift_right(&ctx, i - 1)))
        .collect();
    let mut lookups = Vec::new();
    for i in 2..=3usize {
        for j in 0..2usize {
            let index = (i - 2) * 2 + j;
            let ids = NgramEmbedding::hashed_ids(
                &ctx,
                &shifted,
                &embedding.vocab_mods[&(i, j)],
                i,
                (embedding.m + index * 2 + 1) as i64,
                5,
            );
            lookups.push(embedding.embedders[index].forward(&ids));
        }
    }
    let refs: Vec<&MlxArray> = lookups.iter().map(|a| &**a).collect();
    let proj = embedding
        .post_proj
        .forward(&mlxcel_core::concatenate_many(&refs, -1));
    let word = embedding.word_embeddings.forward(&input);
    let proj = to_vec(&proj);
    let word = to_vec(&word);
    let want: Vec<f32> = word.iter().zip(&proj).map(|(w, p)| w + p / 5.0).collect();
    assert!(max_abs_diff(&got, &want) < 1e-5);
    let ngram_form: Vec<f32> = word.iter().zip(&proj).map(|(w, p)| (w + p) / 5.0).collect();
    assert!(max_abs_diff(&got, &ngram_form) > 1e-2);
}

/// Scores that rank the sink and local columns last: the forced columns are
/// still selected, the rest of the budget goes to the best past keys, and no
/// future key is selected while past keys remain. Disabling the forced
/// columns selects by score alone and drops them.
#[test]
fn lsa_selection_forces_sink_and_local_columns() {
    // One query at position 9 over 10 keys; scores rise with distance from the
    // query except the two worst, columns 0 (sink) and 9 (local).
    let scores: Vec<f32> = vec![-5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -6.0];
    let scores = mlxcel_core::from_slice_f32(&scores, &[1, 1, 1, 10]);
    let forced = LsaSelection {
        topk: 4,
        init_tokens: 1,
        local_tokens: 1,
    };
    let mut got = to_vec_i32(&lsa_top_indices(&scores, 9, forced));
    got.sort_unstable();
    assert_eq!(got, vec![0, 7, 8, 9]);

    let unforced = LsaSelection {
        topk: 4,
        init_tokens: 0,
        local_tokens: 0,
    };
    let mut got = to_vec_i32(&lsa_top_indices(&scores, 9, unforced));
    got.sort_unstable();
    assert_eq!(
        got,
        vec![5, 6, 7, 8],
        "the arm without forced columns differs"
    );

    // A two-column local window ending at the query keeps columns 8 and 9.
    let local = LsaSelection {
        topk: 4,
        init_tokens: 0,
        local_tokens: 3,
    };
    let scores = mlxcel_core::from_slice_f32(
        &[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, -1.0, -2.0, -3.0],
        &[1, 1, 1, 10],
    );
    let mut got = to_vec_i32(&lsa_top_indices(&scores, 9, local));
    got.sort_unstable();
    assert_eq!(got, vec![0, 7, 8, 9]);
}

/// Multi-token selection is causal: a query never ranks a future key above a
/// past one, a sink column after the query is not forced, and the prefill
/// mask keeps exactly the selected past keys.
#[test]
fn lsa_prefill_selection_and_mask_are_causal() {
    // Three queries at positions 3..6 over 6 keys, future keys scored highest.
    let mut values = Vec::new();
    for _ in 0..3 {
        values.extend([1.0f32, 2.0, 3.0, 4.0, 100.0, 200.0]);
    }
    let scores = mlxcel_core::from_slice_f32(&values, &[1, 1, 3, 6]);
    let selection = LsaSelection {
        topk: 3,
        init_tokens: 1,
        local_tokens: 1,
    };
    let indices = lsa_top_indices(&scores, 3, selection);
    let got = to_vec_i32(&indices);
    let rows: Vec<Vec<i32>> = got
        .chunks(3)
        .map(|r| {
            let mut r = r.to_vec();
            r.sort_unstable();
            r
        })
        .collect();
    assert_eq!(
        rows[0],
        vec![0, 2, 3],
        "query 3: sink 0, local 3, best past 2"
    );
    assert_eq!(rows[1], vec![0, 3, 4]);
    assert_eq!(rows[2], vec![0, 4, 5]);

    let mask = to_vec(&sparse_prefill_mask(
        &indices,
        3,
        6,
        mlxcel_core::dtype::FLOAT32,
    ));
    for (q, r) in rows.iter().enumerate() {
        for col in 0..6 {
            let keep = mask[q * 6 + col] == 0.0;
            assert_eq!(keep, r.contains(&(col as i32)), "query {q} col {col}");
        }
    }

    // Fewer past keys than `topk`: the extra picks are future keys, and the
    // mask drops them.
    let scores = mlxcel_core::from_slice_f32(&[1.0, 5.0, 4.0, 3.0], &[1, 1, 1, 4]);
    let indices = lsa_top_indices(
        &scores,
        1,
        LsaSelection {
            topk: 3,
            init_tokens: 0,
            local_tokens: 0,
        },
    );
    let mask = to_vec(&sparse_prefill_mask(
        &indices,
        1,
        4,
        mlxcel_core::dtype::FLOAT32,
    ));
    assert_eq!(mask[0], 0.0);
    assert_eq!(mask[1], 0.0);
    assert!(mask[2].is_infinite() && mask[3].is_infinite());
}

/// Prefill, chunked prefill and single-token decode all reproduce one full
/// forward with the selection active (the window passes `index_topk` at the
/// fifth key), so the decode gather and the prefill mask select the same
/// keys.
#[test]
fn incremental_steps_match_the_full_forward_with_lsa_active() {
    let model = tiny_model();
    let _ = model.make_caches();
    let full = model.forward(&ids(&TOKENS), &mut [], None);

    for pieces in [
        vec![3usize, 2, 1, 1, 1, 1, 1, 1],
        vec![1; TOKENS.len()],
        vec![6, 5],
    ] {
        let _ = model.make_caches();
        let mut worst = 0.0f32;
        let mut pos = 0usize;
        for piece in &pieces {
            let out = model.forward(&ids(&TOKENS[pos..pos + piece]), &mut [], None);
            for j in 0..*piece {
                let diff = max_abs_diff(&row(&out, j as i32), &row(&full, (pos + j) as i32));
                worst = worst.max(diff);
            }
            pos += piece;
        }
        assert!(worst <= 1e-3, "{pieces:?}: max abs diff {worst}");
    }
}

/// The selection changes the logits: the same weights with `index_topk` at
/// the sequence length (every key kept) differ once the window passes the
/// fixture's `index_topk`, and agree before it.
#[test]
fn the_selection_is_active_on_the_fixture() {
    let sparse = tiny_model();
    let _ = sparse.make_caches();
    let sparse_logits = sparse.forward(&ids(&TOKENS), &mut [], None);

    let mut args = tiny_config(8, 1, 2);
    args.index_topk = 64;
    let weights = sanitize_weights(tiny_weights(&args, 8), &args).unwrap();
    let dense = LongcatFlashSparseModel::from_weights(&weights, &args).unwrap();
    let _ = dense.make_caches();
    let dense_logits = dense.forward(&ids(&TOKENS), &mut [], None);

    for pos in 0..INDEX_TOPK {
        assert!(max_abs_diff(&row(&sparse_logits, pos), &row(&dense_logits, pos)) < 1e-5);
    }
    let late = max_abs_diff(
        &row(&sparse_logits, TOKENS.len() as i32 - 1),
        &row(&dense_logits, TOKENS.len() as i32 - 1),
    );
    assert!(
        late > 1e-4,
        "the top-k selection must change late positions: {late}"
    );
}

/// The engine path: greedy decode through `DirectEngine` matches a greedy
/// decode that recomputes the whole sequence from fresh state each step.
#[test]
fn engine_decode_matches_recompute_greedy() {
    let model = tiny_model();
    let greedy = SamplingConfig::greedy();
    let engine = DirectEngine::new(&model, 0)
        .run(&TOKENS[..6], 6, &greedy)
        .expect("engine run");

    let mut seq = TOKENS[..6].to_vec();
    let mut want = Vec::new();
    for _ in 0..6 {
        let _ = model.make_caches();
        let logits = model.forward(&ids(&seq), &mut [], None);
        let last = row(&logits, seq.len() as i32 - 1);
        let next =
            last.iter()
                .enumerate()
                .fold(0, |best, (i, v)| if *v > last[best] { i } else { best }) as i32;
        want.push(next);
        seq.push(next);
    }
    assert_eq!(engine, want);
}

/// `last_row_logits` projects only the sampled row and advances the state
/// exactly as `forward` does.
#[test]
fn last_row_logits_match_the_full_forward_row() {
    let model = tiny_model();
    let _ = model.make_caches();
    let full = model.forward(&ids(&TOKENS[..7]), &mut [], None);
    let next = model.forward(&ids(&TOKENS[7..8]), &mut [], None);

    let _ = model.make_caches();
    let last = model
        .last_row_logits(&ids(&TOKENS[..7]), None, None, &mut [], None, 6)
        .expect("token path");
    assert_eq!(mlxcel_core::array_shape(&last)[1], 1);
    assert!(max_abs_diff(&row(&last, 0), &row(&full, 6)) < 1e-5);
    let after = model.forward(&ids(&TOKENS[7..8]), &mut [], None);
    assert!(max_abs_diff(&row(&after, 0), &row(&next, 0)) < 1e-5);
}

fn holds_quantized_latent(model: &LongcatFlashSparseModel) -> bool {
    let state = model.state.borrow();
    !state.layers.is_empty()
        && state.layers.iter().all(|c| {
            c.latent.iter().all(|cache| cache.holds_quantized_latent())
                && !c.index.holds_quantized_latent()
        })
}

fn max_rel_error(got: &[f32], want: &[f32]) -> f32 {
    let scale = want
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        .max(1e-6);
    got.iter()
        .zip(want)
        .fold(0.0f32, |acc, (g, w)| acc.max((g - w).abs() / scale))
}

/// The family is a quantized-latent family: a model built from its config
/// opts in, keeps an injected `int8` table, builds both latent caches of every
/// layer in it (the indexer keys stay fp16), and replaces a Turbo layer with
/// fp16. The default table is fp16.
#[test]
fn the_quantized_latent_modes_are_accepted_and_fp16_stays_the_default() {
    assert!(mlxcel_core::mla::stores_quantized_latent_pair(MODEL_TYPE));
    let model = tiny_model_with(QUANTIZABLE_RANK, 1, 2);
    assert!(model.quantized_latent.get());
    let n = model.num_layers();
    assert_eq!(
        model.kv_cache_layer_modes(),
        Some(vec![KVCacheMode::Fp16; n])
    );
    let _ = model.make_caches();
    mlxcel_core::eval(&model.forward(&ids(&TOKENS[..6]), &mut [], None));
    assert!(!holds_quantized_latent(&model));

    let mut modes = vec![KVCacheMode::Int8; n];
    modes[0] = KVCacheMode::Turbo4Asym;
    model.set_kv_cache_layer_modes(modes);
    let mut want = vec![KVCacheMode::Int8; n];
    want[0] = KVCacheMode::Fp16;
    assert_eq!(model.kv_cache_layer_modes(), Some(want));

    model.set_kv_cache_layer_modes(vec![KVCacheMode::Int8; n]);
    mlxcel_core::eval(&model.forward(&ids(&TOKENS[..6]), &mut [], None));
    assert!(holds_quantized_latent(&model));
    for layer in model.state.borrow().layers.iter() {
        assert_eq!(layer.index.mode, KVCacheMode::Fp16);
    }

    with_mla_latent_quant(MlaLatentQuant::Fp16, || {
        let model = tiny_model_with(QUANTIZABLE_RANK, 1, 2);
        assert!(!model.quantized_latent.get());
        model.set_kv_cache_layer_modes(vec![KVCacheMode::Int8; n]);
        assert_eq!(
            model.kv_cache_layer_modes(),
            Some(vec![KVCacheMode::Fp16; n])
        );
    });
}

/// Teacher-forced decode with the selection active over the quantized latent,
/// at 8 and at 4 bits: every step gathers the selected rows from the packed
/// window and reads them absorbed, and the logits track the fp16 latent
/// within the codec's bound.
#[test]
fn sparse_decode_gathers_the_quantized_latent_and_tracks_fp16() {
    let prefill = 5;
    let fp16 = tiny_model_with(QUANTIZABLE_RANK, 1, 2);
    let _ = fp16.make_caches();
    mlxcel_core::eval(&fp16.forward(&ids(&TOKENS[..prefill]), &mut [], None));
    let want: Vec<Vec<f32>> = TOKENS[prefill..]
        .iter()
        .map(|t| row(&fp16.forward(&ids(&[*t]), &mut [], None), 0))
        .collect();

    for (option, bits, tol) in [
        (MlaLatentQuant::Auto, 8, 5e-2f32),
        (MlaLatentQuant::Int4, 4, 2.5e-1),
    ] {
        with_mla_latent_quant(option, || {
            let model = tiny_model_with(QUANTIZABLE_RANK, 1, 2);
            model.set_kv_cache_layer_modes(vec![KVCacheMode::Int8; model.num_layers()]);
            let _ = model.make_caches();
            mlxcel_core::eval(&model.forward(&ids(&TOKENS[..prefill]), &mut [], None));
            for layer in model.state.borrow().layers.iter() {
                for cache in &layer.latent {
                    assert_eq!(cache.stored_latent_bits(), Some(bits));
                }
            }
            let before = mlxcel_core::mla::stats::snapshot();
            let got: Vec<Vec<f32>> = TOKENS[prefill..]
                .iter()
                .map(|t| row(&model.forward(&ids(&[*t]), &mut [], None), 0))
                .collect();
            let after = mlxcel_core::mla::stats::snapshot();
            let delta = |path| after.get(path).saturating_sub(before.get(path));
            let steps = (TOKENS.len() - prefill) as u64;
            assert!(
                delta(MlaDecodePath::AbsorbedComposed) + delta(MlaDecodePath::AbsorbedFused)
                    >= steps * 2 * model.num_layers() as u64,
                "{bits}-bit: every sparse decode step reads the quantized latent absorbed"
            );
            assert_eq!(delta(MlaDecodePath::Decompressed), 0);
            for (step, (got, want)) in got.iter().zip(&want).enumerate() {
                let err = max_rel_error(got, want);
                assert!(
                    err > 0.0,
                    "{bits}-bit step {step}: must not be the fp16 arm"
                );
                assert!(err < tol, "{bits}-bit step {step}: relative error {err}");
            }
        });
    }
}

/// A gathered quantized decode step equals the dense step over the same
/// stored rows dequantized: the gather reads the active layout, it does not
/// re-select.
#[test]
fn a_quantized_gather_reads_the_rows_it_selected() {
    let rows = 7;
    let window = {
        let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
        let ckv = mlxcel_core::from_slice_f32(
            &pseudo_random(5, (rows * QUANTIZABLE_RANK) as usize, 1.0),
            &[1, 1, rows, QUANTIZABLE_RANK],
        );
        let kpe = mlxcel_core::from_slice_f32(
            &pseudo_random(9, (rows * ROPE) as usize, 1.0),
            &[1, 1, rows, ROPE],
        );
        let geometry =
            crate::models::mla_batched::mla_geometry(HEADS, QUANTIZABLE_RANK, NOPE, ROPE, V_DIM);
        crate::models::mla_batched::store_latent_step(
            &mut cache, geometry, ckv, kpe, MODEL_TYPE, true,
        )
    };
    assert!(window.is_quantized());
    let indices = mlxcel_core::from_slice_i32(&[6, 0, 3], &[1, 1, 1, 3]);
    let gathered = window.gather_rows(&indices);
    assert!(gathered.is_quantized());
    let (all_ckv, all_kpe) = window.into_dense();
    let (ckv, kpe) = gathered.into_dense();
    for (slot, source) in [6, 0, 3].into_iter().enumerate() {
        let slot = slot as i32;
        assert_eq!(
            to_vec(&slice_axis(&ckv, 2, slot, slot + 1)),
            to_vec(&slice_axis(&all_ckv, 2, source, source + 1))
        );
        assert_eq!(
            to_vec(&slice_axis(&kpe, 2, slot, slot + 1)),
            to_vec(&slice_axis(&all_kpe, 2, source, source + 1))
        );
    }
}

/// An `int8` engine run stores the quantized latent and decodes the same
/// greedy tokens as fp16 on this fixture.
#[test]
fn an_int8_engine_run_matches_fp16_greedy() {
    let greedy = SamplingConfig::greedy();
    let fp16_model = tiny_model_with(QUANTIZABLE_RANK, 1, 2);
    let fp16 = DirectEngine::new(&fp16_model, 0)
        .run(&TOKENS[..6], 6, &greedy)
        .expect("fp16 run");
    let int8_model = tiny_model_with(QUANTIZABLE_RANK, 1, 2);
    let int8 = DirectEngine::new(&int8_model, 0)
        .with_kv_cache_mode(KVCacheMode::Int8)
        .run(&TOKENS[..6], 6, &greedy)
        .expect("int8 run");
    assert!(holds_quantized_latent(&int8_model));
    assert!(!holds_quantized_latent(&fp16_model));
    assert_eq!(int8, fp16);
}

/// The mlx-community layout (already stacked, decomposed and fused, rooted
/// at `language_model.`) and the HF layout build the same model.
#[test]
fn mlx_and_hf_layouts_build_the_same_model() {
    let args = tiny_config(8, 1, 2);
    let converted = sanitize_weights(tiny_weights(&args, 8), &args).unwrap();
    let mut rooted = WeightMap::new();
    for (key, value) in &converted {
        rooted.insert(format!("language_model.{key}"), mlxcel_core::copy(value));
    }
    let resanitized = sanitize_weights(rooted, &args).unwrap();
    let mut keys: Vec<_> = resanitized.keys().cloned().collect();
    let mut want: Vec<_> = converted.keys().cloned().collect();
    keys.sort();
    want.sort();
    assert_eq!(keys, want);

    let a = LongcatFlashSparseModel::from_weights(&converted, &args).unwrap();
    let b = LongcatFlashSparseModel::from_weights(&resanitized, &args).unwrap();
    let _ = a.make_caches();
    let _ = b.make_caches();
    let la = a.forward(&ids(&TOKENS), &mut [], None);
    let lb = b.forward(&ids(&TOKENS), &mut [], None);
    assert!(max_abs_diff(&to_vec(&la), &to_vec(&lb)) < 1e-6);
}

/// Config shapes this port would run wrongly are refused by name.
#[test]
fn unsupported_configs_are_refused() {
    let base = tiny_config(8, 1, 2);
    assert!(base.validate().is_ok());
    assert!(base.uses_lsa());

    let mut bad = base.clone();
    bad.attention_method = "NSA".into();
    assert!(bad.validate().unwrap_err().contains("attention_method"));

    let mut bad = base.clone();
    bad.index_topk = 0;
    assert!(bad.validate().unwrap_err().contains("index_topk"));

    let mut bad = base.clone();
    bad.index_head_dim = 2;
    assert!(bad.validate().unwrap_err().contains("index_head_dim"));

    let mut bad = base.clone();
    bad.zero_expert_type = "zero".into();
    assert!(bad.validate().unwrap_err().contains("zero_expert_type"));

    let mut bad = base.clone();
    bad.q_lora_rank = 0;
    assert!(bad.validate().unwrap_err().contains("q_lora_rank"));

    // Missing indexer weights under LSA fail the load, naming the key.
    let mut weights = sanitize_weights(tiny_weights(&base, 8), &base).unwrap();
    weights.remove("model.layers.1.self_attn.0.indexer.wk.weight");
    let err = LongcatFlashSparseModel::from_weights(&weights, &base)
        .err()
        .expect("missing indexer refused");
    assert!(err.contains("model.layers.1.self_attn.0.indexer"), "{err}");

    // Dense MLA ignores the selection rule and needs no indexer.
    let mut dense = base.clone();
    dense.attention_method = "MLA".into();
    let mut weights = sanitize_weights(tiny_weights(&dense, 8), &dense).unwrap();
    weights.retain(|k, _| !k.contains(".indexer."));
    let model = LongcatFlashSparseModel::from_weights(&weights, &dense).unwrap();
    let _ = model.make_caches();
    mlxcel_core::eval(&model.forward(&ids(&TOKENS), &mut [], None));
}

/// The original config has no `model_type`; it parses to this family.
#[test]
fn a_config_without_model_type_parses() {
    let config: LongcatFlashSparseConfig = serde_json::from_str(
        r#"{
            "architectures": ["LongcatCausalLM"],
            "hidden_size": 16, "ffn_hidden_size": 24, "moe_topk": 2,
            "expert_ffn_hidden_size": 8, "n_routed_experts": 2, "num_layers": 1,
            "vocab_size": 32, "max_position_embeddings": 64, "num_attention_heads": 2,
            "kv_lora_rank": 8, "q_lora_rank": 8, "qk_rope_head_dim": 4,
            "qk_nope_head_dim": 4, "v_head_dim": 4, "routed_scaling_factor": 1.0,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0
        }"#,
    )
    .unwrap();
    assert_eq!(config.model_type, MODEL_TYPE);
    assert!(config.uses_lsa());
    assert!(config.mla_scale_q_lora && config.mla_scale_kv_lora);
    assert!(config.indexer_rope_interleave);
}
