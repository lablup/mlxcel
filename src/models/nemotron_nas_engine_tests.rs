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

//! NemotronNAS on the engine path (#2544).
//!
//! The family cannot batch, so its pool entry is model-owned and
//! `Engine::prefill` hands `forward` an empty cache slice. The model has to
//! keep one `KVCache` per attention layer itself and reset them in
//! `make_caches`. These tests drive it the way the engine does: through
//! `DirectEngine`, and through `forward` with an empty slice.

use mlxcel_core::engine::DirectEngine;
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::utils::{array_to_vec_f32, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{NemotronNASConfig, NemotronNASModel};

const VOCAB: i32 = 32;
const HIDDEN: i32 = 16;
const HEADS: i32 = 4;
const HEAD_DIM: i32 = HIDDEN / HEADS;
const INTER: i32 = 24;

/// Five blocks covering every attention subblock kind: full attention at two
/// GQA widths, a `no_op` block (no cache), and a `replace_with_linear` block
/// (a cache slot it never touches), with each FFN kind mixed in.
const MIXED_BLOCKS: &str = r#"
    {"attention": {"n_heads_in_group": 2}, "ffn": {"ffn_mult": 1.0}},
    {"attention": {"no_op": true}, "ffn": {"ffn_mult": 1.0}},
    {"attention": {"replace_with_linear": true}, "ffn": {"replace_with_linear": true}},
    {"attention": {"n_heads_in_group": 4}, "ffn": {"no_op": true}},
    {"attention": {"n_heads_in_group": 1}, "ffn": {"ffn_mult": 1.0}}
"#;

/// The same kinds with a `replace_with_linear` block first, so the first
/// cache slot is one no block ever writes.
const LINEAR_FIRST_BLOCKS: &str = r#"
    {"attention": {"replace_with_linear": true}, "ffn": {"ffn_mult": 1.0}},
    {"attention": {"no_op": true}, "ffn": {"ffn_mult": 1.0}},
    {"attention": {"n_heads_in_group": 2}, "ffn": {"replace_with_linear": true}},
    {"attention": {"n_heads_in_group": 4}, "ffn": {"no_op": true}},
    {"attention": {"n_heads_in_group": 1}, "ffn": {"ffn_mult": 1.0}}
"#;

fn config_with_blocks(blocks: &str) -> NemotronNASConfig {
    let json = format!(
        r#"{{
            "model_type": "nemotron-nas",
            "hidden_size": {HIDDEN},
            "num_hidden_layers": 5,
            "num_attention_heads": {HEADS},
            "rms_norm_eps": 1e-5,
            "vocab_size": {VOCAB},
            "rope_theta": 10000.0,
            "max_position_embeddings": 256,
            "tie_word_embeddings": false,
            "block_configs": [{blocks}]
        }}"#
    );
    serde_json::from_str(&json).expect("parse tiny nemotron-nas config")
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

/// f32 weights for a tiny config, inserted in a fixed order so every run
/// builds the same model (a `WeightMap` walk would not, #1265).
fn tiny_weights(config: &NemotronNASConfig) -> WeightMap {
    let mut weights = WeightMap::new();
    let mut seed = 11u64;
    let mut put = |name: String, shape: &[i32], scale: f32| {
        seed += 1;
        let n: i32 = shape.iter().product();
        let values = pseudo_random(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), n as usize, scale);
        weights.insert(name, mlxcel_core::from_slice_f32(&values, shape));
    };
    let ones = |shape: &[i32]| mlxcel_core::full_f32(shape, 1.0, mlxcel_core::dtype::FLOAT32);

    put("model.embed_tokens.weight".into(), &[VOCAB, HIDDEN], 0.8);
    put("lm_head.weight".into(), &[VOCAB, HIDDEN], 0.6);
    let mut norms = vec![("model.norm.weight".to_string(), ones(&[HIDDEN]))];

    for (i, block) in config.block_configs.iter().enumerate() {
        let p = format!("model.layers.{i}");
        let attn = &block.attention;
        if !attn.no_op {
            norms.push((format!("{p}.input_layernorm.weight"), ones(&[HIDDEN])));
        }
        if attn.replace_with_linear {
            put(
                format!("{p}.self_attn.linear.weight"),
                &[HIDDEN, HIDDEN],
                0.3,
            );
        } else if !attn.no_op {
            let kv = HEADS / attn.n_heads_in_group.unwrap_or(1) as i32 * HEAD_DIM;
            put(
                format!("{p}.self_attn.q_proj.weight"),
                &[HIDDEN, HIDDEN],
                0.5,
            );
            put(format!("{p}.self_attn.k_proj.weight"), &[kv, HIDDEN], 0.5);
            put(format!("{p}.self_attn.v_proj.weight"), &[kv, HIDDEN], 0.5);
            put(
                format!("{p}.self_attn.o_proj.weight"),
                &[HIDDEN, HIDDEN],
                0.4,
            );
        }
        let ffn = &block.ffn;
        if !ffn.no_op {
            norms.push((
                format!("{p}.post_attention_layernorm.weight"),
                ones(&[HIDDEN]),
            ));
        }
        if ffn.replace_with_linear {
            put(format!("{p}.mlp.linear.weight"), &[HIDDEN, HIDDEN], 0.3);
        } else if !ffn.no_op {
            put(format!("{p}.mlp.gate_proj.weight"), &[INTER, HIDDEN], 0.4);
            put(format!("{p}.mlp.up_proj.weight"), &[INTER, HIDDEN], 0.4);
            put(format!("{p}.mlp.down_proj.weight"), &[HIDDEN, INTER], 0.4);
        }
    }
    for (name, w) in norms {
        weights.insert(name, w);
    }
    weights
}

fn tiny_model() -> NemotronNASModel {
    model_with_blocks(MIXED_BLOCKS)
}

fn model_with_blocks(blocks: &str) -> NemotronNASModel {
    let config = config_with_blocks(blocks);
    let weights = tiny_weights(&config);
    NemotronNASModel::from_weights(config, weights).expect("build tiny nemotron-nas")
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

/// Greedy decode by recomputing the whole sequence from fresh caches at every
/// step: no state carried between calls, so it is the reference a correct
/// incremental decode must reproduce.
fn recompute_greedy(model: &NemotronNASModel, prompt: &[i32], steps: usize) -> Vec<i32> {
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

const PROMPT: [i32; 6] = [3, 17, 9, 25, 4, 30];

#[test]
fn nemotron_nas_direct_engine_run_completes() {
    let model = tiny_model();
    assert!(!model.supports_batching());
    assert!(!model.supports_padded_prefill());
    let tokens = DirectEngine::new(&model, 0)
        .run(&PROMPT, 8, &SamplingConfig::greedy())
        .expect("DirectEngine::run on NemotronNAS");
    assert!(!tokens.is_empty(), "the run produced no tokens");
    assert!(tokens.iter().all(|t| (0..VOCAB).contains(t)));
}

#[test]
fn nemotron_nas_consecutive_direct_engine_runs_match_recompute() {
    let model = tiny_model();
    let greedy = SamplingConfig::greedy();
    let mut engine = DirectEngine::new(&model, 0);
    let first = engine.run(&PROMPT, 8, &greedy).expect("first run");
    let second = engine.run(&PROMPT, 8, &greedy).expect("second run");
    assert_eq!(first, second, "the second run continued on the first's KV");

    let reference = recompute_greedy(&model, &PROMPT, 8);
    assert!(!first.is_empty());
    assert_eq!(
        first,
        reference[..first.len()],
        "incremental decode drifted"
    );
}

const TOKENS: [i32; 10] = [3, 17, 9, 25, 4, 30, 12, 7, 21, 1];

/// Feed `TOKENS` in `pieces` through `forward` with an empty slice, as the
/// engine does, and compare every position the pieces produce against one
/// full-sequence forward on fresh caches.
fn worst_incremental_diff(model: &NemotronNASModel, pieces: &[usize]) -> f32 {
    let _ = model.make_caches();
    let full = model.forward(&ids(&TOKENS), &mut [], None);

    let _ = model.make_caches();
    let mut worst = 0.0f32;
    let mut pos = 0usize;
    for &piece in pieces {
        let out = model.forward(&ids(&TOKENS[pos..pos + piece]), &mut [], None);
        for j in 0..piece {
            let diff = max_abs_diff(&row(&out, j as i32), &row(&full, (pos + j) as i32));
            worst = worst.max(diff);
        }
        pos += piece;
    }
    assert_eq!(pos, TOKENS.len());
    worst
}

#[test]
fn nemotron_nas_incremental_logits_match_full_forward() {
    let worst = worst_incremental_diff(&tiny_model(), &[4, 1, 1, 1, 1, 1, 1]);
    assert!(
        worst <= 1e-3,
        "prefill plus single-token steps diverged from the full forward: max abs diff {worst}"
    );
}

/// A chunk after the first prefill piece needs a mask as wide as the live
/// KV, read from a slot a full attention block writes, not from the first
/// slot, which belongs to a linear replacement block here.
#[test]
fn nemotron_nas_linear_first_chunked_prefill_matches_full_forward() {
    let model = model_with_blocks(LINEAR_FIRST_BLOCKS);
    let worst = worst_incremental_diff(&model, &[3, 3, 1, 1, 1, 1]);
    assert!(
        worst <= 1e-3,
        "chunked prefill diverged from the full forward: max abs diff {worst}"
    );
}
