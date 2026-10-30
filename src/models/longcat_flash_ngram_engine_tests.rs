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

//! LongCat-Flash and LongCat-Flash-Ngram on the engine path (#2544).
//!
//! One struct serves both families. It cannot batch, so the engine hands
//! `forward` an empty cache slice and the model must carry its dual-sub-layer
//! KV caches and the n-gram token context across calls itself, resetting
//! both in `make_caches`. The reference here is a full-sequence forward on
//! fresh state: prefill, chunked prefill and single-token decode must all
//! reproduce it.

use mlxcel_core::engine::DirectEngine;
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::utils::{array_to_vec_f32, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{LongcatFlashNgramConfig, LongcatFlashNgramModel, sanitize_weights};

const VOCAB: i32 = 32;
const HIDDEN: i32 = 16;
const HEADS: i32 = 2;
const KV_LORA: i32 = 8;
const Q_LORA: i32 = 8;
const ROPE: i32 = 4;
const NOPE: i32 = 4;
const V_DIM: i32 = 4;
const FFN: i32 = 24;
const EXPERT_FFN: i32 = 8;
const ROUTED: i32 = 2;
const ZERO: i32 = 1;

fn tiny_config(model_type: &str) -> LongcatFlashNgramConfig {
    let json = format!(
        r#"{{
            "model_type": "{model_type}",
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
            "kv_lora_rank": {KV_LORA},
            "q_lora_rank": {Q_LORA},
            "qk_rope_head_dim": {ROPE},
            "qk_nope_head_dim": {NOPE},
            "v_head_dim": {V_DIM},
            "routed_scaling_factor": 1.0,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "mla_scale_q_lora": true,
            "mla_scale_kv_lora": true,
            "norm_topk_prob": true,
            "ngram_vocab_size_ratio": 2,
            "emb_neighbor_num": 3,
            "emb_split_num": 2
        }}"#
    );
    serde_json::from_str(&json).expect("parse tiny longcat config")
}

/// Byte-reproducible fixture values, independent of MLX's RNG.
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

/// Checkpoint-layout f32 weights (per-expert MLPs, fused `kv_b_proj`, the
/// ngram variant's `embed_tokens` under its pre-rename key), inserted in a
/// fixed order so every run builds the same model (#1265). The model is
/// built through `sanitize_weights` exactly as `load` does.
fn tiny_model(model_type: &str) -> LongcatFlashNgramModel {
    let args = tiny_config(model_type);
    let mut weights = WeightMap::new();
    let mut seed = 101u64;
    let mut put = |name: String, shape: &[i32], scale: f32| {
        seed += 1;
        let n: i32 = shape.iter().product();
        let values = pseudo_random(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), n as usize, scale);
        weights.insert(name, mlxcel_core::from_slice_f32(&values, shape));
    };
    put("model.embed_tokens.weight".into(), &[VOCAB, HIDDEN], 0.8);
    put("lm_head.weight".into(), &[VOCAB, HIDDEN], 0.6);
    if args.is_ngram_model() {
        let m = (args.ngram_vocab_size_ratio * args.vocab_size) as i32;
        let embedders = (args.emb_split_num * (args.emb_neighbor_num - 1)) as i32;
        let emb_dim = HIDDEN / embedders;
        for i in 0..embedders {
            put(
                format!("model.ngram_embeddings.embedders.{i}.weight"),
                &[m + 2 * i + 1, emb_dim],
                0.8,
            );
            put(
                format!("model.ngram_embeddings.post_projs.{i}.weight"),
                &[HIDDEN, emb_dim],
                0.5,
            );
        }
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
                &[KV_LORA + ROPE, HIDDEN],
                0.5,
            );
            put(
                format!("{a}.kv_b_proj.weight"),
                &[HEADS * (NOPE + V_DIM), KV_LORA],
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

    let mut norm_names = vec![("model.norm.weight".to_string(), HIDDEN)];
    for l in 0..args.num_layers {
        for i in 0..2 {
            let p = format!("model.layers.{l}");
            norm_names.push((format!("{p}.self_attn.{i}.q_a_layernorm.weight"), Q_LORA));
            norm_names.push((format!("{p}.self_attn.{i}.kv_a_layernorm.weight"), KV_LORA));
            norm_names.push((format!("{p}.input_layernorm.{i}.weight"), HIDDEN));
            norm_names.push((format!("{p}.post_attention_layernorm.{i}.weight"), HIDDEN));
        }
    }
    for (name, dim) in &norm_names {
        put(name.clone(), &[*dim], 0.1);
    }
    // Norm gains of 1 plus small noise keep activations in range without
    // making every gain identical.
    let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
    for (name, _) in norm_names {
        let w = weights.remove(&name).expect("norm weight present");
        weights.insert(name, mlxcel_core::add(&w, &one));
    }

    let weights = sanitize_weights(weights, &args).expect("sanitize tiny longcat weights");
    LongcatFlashNgramModel::from_weights(&weights, &args).expect("build tiny longcat")
}

fn ids(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

/// Logits of sequence position `pos` of a `[1, L, V]` forward output.
fn row(logits: &MlxArray, pos: i32) -> Vec<f32> {
    let r = slice_axis(logits, 1, pos, pos + 1);
    let r = mlxcel_core::astype(&r, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&r);
    array_to_vec_f32(&r)
}

fn argmax(values: &[f32]) -> i32 {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best as i32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Greedy decode that recomputes the whole sequence from fresh state at every
/// step, the reference a correct incremental decode must reproduce.
fn recompute_greedy(model: &LongcatFlashNgramModel, prompt: &[i32], steps: usize) -> Vec<i32> {
    let mut seq = prompt.to_vec();
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        let _ = model.make_caches();
        let logits = model.forward(&ids(&seq), &mut [], None);
        let next = argmax(&row(&logits, seq.len() as i32 - 1));
        out.push(next);
        seq.push(next);
    }
    out
}

const TOKENS: [i32; 11] = [3, 17, 9, 25, 4, 30, 12, 7, 21, 1, 14];

/// Prefill 3 tokens, then a 2-token chunk, then single-token steps, and
/// compare every new position against one full-sequence forward. The chunk
/// covers chunked prefill (offset > 0 with L > 1) and the n-gram context
/// crossing a multi-token boundary.
fn assert_incremental_matches_full(model_type: &str) {
    let model = tiny_model(model_type);
    assert_eq!(
        tiny_config(model_type).is_ngram_model(),
        model_type.ends_with("ngram")
    );

    let _ = model.make_caches();
    let full = model.forward(&ids(&TOKENS), &mut [], None);

    let _ = model.make_caches();
    let mut worst = 0.0f32;
    let mut pos = 0usize;
    for piece in [3usize, 2, 1, 1, 1, 1, 1, 1] {
        let out = model.forward(&ids(&TOKENS[pos..pos + piece]), &mut [], None);
        for j in 0..piece {
            let diff = max_abs_diff(&row(&out, j as i32), &row(&full, (pos + j) as i32));
            worst = worst.max(diff);
        }
        pos += piece;
    }
    assert_eq!(pos, TOKENS.len());
    assert!(
        worst <= 1e-3,
        "{model_type}: chunked prefill plus single-token steps diverged from the full forward: \
         max abs diff {worst}"
    );
}

#[test]
fn longcat_flash_incremental_logits_match_full_forward() {
    assert_incremental_matches_full("longcat_flash");
}

#[test]
fn longcat_flash_ngram_incremental_logits_match_full_forward() {
    assert_incremental_matches_full("longcat_flash_ngram");
}

fn assert_consecutive_runs_match_recompute(model_type: &str) {
    let model = tiny_model(model_type);
    let greedy = SamplingConfig::greedy();
    let prompt = &TOKENS[..6];
    let mut engine = DirectEngine::new(&model, 0);
    let first = engine.run(prompt, 8, &greedy).expect("first run");
    let second = engine.run(prompt, 8, &greedy).expect("second run");
    assert_eq!(
        first, second,
        "{model_type}: the second run inherited state"
    );

    let reference = recompute_greedy(&model, prompt, 8);
    assert!(
        !first.is_empty(),
        "{model_type}: the run produced no tokens"
    );
    assert_eq!(
        first,
        reference[..first.len()],
        "{model_type}: engine decode drifted from the recompute reference"
    );
}

#[test]
fn longcat_flash_consecutive_direct_engine_runs_match_recompute() {
    assert_consecutive_runs_match_recompute("longcat_flash");
}

#[test]
fn longcat_flash_ngram_consecutive_direct_engine_runs_match_recompute() {
    assert_consecutive_runs_match_recompute("longcat_flash_ngram");
}

/// A chunked engine prefill must land on the same tokens as a one-piece one.
#[test]
fn longcat_flash_ngram_chunked_engine_prefill_matches_single_piece() {
    let model = tiny_model("longcat_flash_ngram");
    let greedy = SamplingConfig::greedy();
    let prompt = &TOKENS[..9];
    let whole = DirectEngine::new(&model, 0)
        .run(prompt, 6, &greedy)
        .expect("single-piece run");
    let chunked = DirectEngine::new(&model, 4)
        .run(prompt, 6, &greedy)
        .expect("chunked run");
    assert_eq!(whole, chunked);
}

/// With persistent internal caches, pad positions would land in state no
/// trim reaches (#1755), so neither variant may take padded prefill.
#[test]
fn longcat_flash_refuses_padded_prefill() {
    for model_type in ["longcat_flash", "longcat_flash_ngram"] {
        let model = tiny_model(model_type);
        assert!(!model.supports_batching(), "{model_type}");
        assert!(!model.supports_padded_prefill(), "{model_type}");
    }
}

// ---------------------------------------------------------------------------
// `--kv-cache-mode` on the latent caches (#2766)
// ---------------------------------------------------------------------------

/// Both LongCat variants serve FP16 KV only: their MLA caches hold a latent
/// pair, which no quantized mode supports. A quantized request is refused
/// and the model reports FP16 for every layer, so `logit_trace` and the
/// server's startup line report what the caches really hold.
fn assert_reports_fp16_only(model_type: &str) {
    use mlxcel_core::cache::KVCacheMode;

    let model = tiny_model(model_type);
    let n = model.num_layers();
    model.set_kv_cache_layer_modes(vec![KVCacheMode::Int8; n]);
    assert_eq!(
        model.kv_cache_layer_modes(),
        Some(vec![KVCacheMode::Fp16; n]),
        "{model_type} must report the FP16 it actually stores"
    );
    assert!(mlxcel_core::mla::caches_mla_latent_pair(model_type));

    // The refused request changes nothing: the same greedy run either way.
    let greedy = SamplingConfig::greedy();
    let fp16 = DirectEngine::new(&tiny_model(model_type), 0)
        .run(&TOKENS[..6], 6, &greedy)
        .expect("fp16 run");
    let int8 = DirectEngine::new(&model, 0)
        .with_kv_cache_mode(KVCacheMode::Int8)
        .run(&TOKENS[..6], 6, &greedy)
        .expect("int8 request run");
    assert_eq!(int8, fp16);
}

#[test]
fn longcat_flash_reports_fp16_only_kv() {
    assert_reports_fp16_only("longcat_flash");
}

#[test]
fn longcat_flash_ngram_reports_fp16_only_kv() {
    assert_reports_fp16_only("longcat_flash_ngram");
}
