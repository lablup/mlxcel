// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Unit tests for the Laguna DFlash drafter: config contract, sanitizer,
//! context window trimming, and the causal block mask.

use super::cache::LagunaDFlashContextCache;
use super::config::LagunaDFlashConfig;
use super::drafter::LagunaDFlashDrafter;
use super::sanitize::{expected_weight_keys, sanitize_weights};
use crate::dtype;
use crate::ffi;
use crate::layers::{Embedding, UnifiedEmbedding};
use crate::weights::WeightMap;
use serde_json::json;
use std::collections::BTreeSet;

/// The published `poolside/Laguna-XS-2.1-DFlash` config, verbatim (no
/// `sliding_windows` key).
fn xs_2_1_config() -> serde_json::Value {
    json!({
        "attention_bias": false, "head_dim": 128, "hidden_act": "silu", "hidden_size": 2048,
        "intermediate_size": 8192, "max_position_embeddings": 262144, "model_type": "laguna",
        "num_attention_heads": 64, "num_hidden_layers": 5, "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06, "rope_theta": 500000.0, "sliding_window": 512,
        "vocab_size": 100352,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention"],
        "gating": "per-head", "architectures": ["DFlashLagunaForCausalLM"],
        "draft_vocab_size": 100352, "torch_dtype": "bfloat16",
        "eagle_aux_hidden_state_layer_ids": [2, 14, 26, 34, 40],
        "dflash_config": {"block_size": 16, "mask_token_id": 12, "num_target_layers": 40,
                          "target_layer_ids": [1, 13, 25, 33, 39], "causal": true},
        "num_experts": 0
    })
}

#[test]
fn xs_2_1_config_loads_and_defaults_sliding_windows() {
    let cfg = LagunaDFlashConfig::from_json(&xs_2_1_config()).expect("XS 2.1 config loads");
    assert_eq!(cfg.num_hidden_layers, 5);
    assert_eq!(cfg.sliding_windows, vec![512; 5]);
    assert_eq!(cfg.target_layer_ids, vec![1, 13, 25, 33, 39]);
    assert_eq!(cfg.num_target_layers, 40);
    assert_eq!(cfg.mask_token_id, 12);
    assert_eq!(cfg.block_size, 16);
    assert_eq!(cfg.head_dim, 128);
    assert!((cfg.rope_theta - 500_000.0).abs() < 1.0);
    assert!(LagunaDFlashConfig::is_laguna_dflash_config(&xs_2_1_config()));
    assert!(super::super::dflash::is_dflash_drafter_config(
        &xs_2_1_config()
    ));
}

#[test]
fn config_rejects_bad_contract() {
    let mut cases: Vec<(&str, serde_json::Value)> = Vec::new();
    let mut c = xs_2_1_config();
    c["dflash_config"]["causal"] = json!(false);
    cases.push(("causal", c));
    let mut c = xs_2_1_config();
    c["draft_vocab_size"] = json!(100000);
    cases.push(("draft_vocab_size", c));
    let mut c = xs_2_1_config();
    c["dflash_config"]["target_layer_ids"] = json!([1, 13, 13, 33, 39]);
    cases.push(("target_layer_ids", c));
    let mut c = xs_2_1_config();
    c["gating"] = json!("per-element");
    cases.push(("gating", c));
    let mut c = xs_2_1_config();
    c["layer_types"][2] = json!("full_attention");
    cases.push(("layer_types", c));
    let mut c = xs_2_1_config();
    c["dflash_config"]["num_target_layers"] = json!(4);
    cases.push(("num_target_layers", c));
    let mut c = xs_2_1_config();
    c["dflash_config"]["mask_token_id"] = json!(100352);
    cases.push(("mask_token_id", c));
    let mut c = xs_2_1_config();
    c["sliding_windows"] = json!([512, 512]);
    cases.push(("sliding_windows", c));
    for (field, cfg) in cases {
        let err = LagunaDFlashConfig::from_json(&cfg).expect_err(field);
        assert!(
            err.contains(field),
            "{field}: error must name the offending field, got {err}"
        );
    }
    // Target mismatch is a separate, bind-time check.
    let cfg = LagunaDFlashConfig::from_json(&xs_2_1_config()).unwrap();
    assert!(cfg.validate_target(40, 100352).is_ok());
    assert!(cfg.validate_target(48, 100352).is_err());
    assert!(cfg.validate_target(40, 100353).is_err());
}

/// Tiny drafter geometry: hidden 8, 2 heads of 4, 1 kv head, 2 layers,
/// window 4, over a 4-layer target with hidden 8 captured at layers [1, 3].
fn tiny_config_json() -> serde_json::Value {
    tiny_config_json_with_layers(2)
}

/// [`tiny_config_json`] with `layers` drafter layers over target layers
/// `[1, 3, ...]` (one captured target layer per drafter layer).
fn tiny_config_json_with_layers(layers: usize) -> serde_json::Value {
    let target_layer_ids: Vec<usize> = (0..layers).map(|i| 2 * i + 1).collect();
    json!({
        "model_type": "laguna", "architectures": ["DFlashLagunaForCausalLM"],
        "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": layers,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
        "rms_norm_eps": 1e-6, "vocab_size": 32, "draft_vocab_size": 32,
        "rope_theta": 10000.0, "sliding_window": 4,
        "layer_types": vec!["sliding_attention"; layers], "gating": "per-head",
        "dflash_config": {"block_size": 4, "mask_token_id": 31,
                          "num_target_layers": 2 * layers,
                          "target_layer_ids": target_layer_ids, "causal": true},
        "num_experts": 0
    })
}

pub(crate) fn tiny_config() -> LagunaDFlashConfig {
    LagunaDFlashConfig::from_json(&tiny_config_json()).expect("tiny config")
}

fn tiny_single_layer_config() -> LagunaDFlashConfig {
    LagunaDFlashConfig::from_json(&tiny_config_json_with_layers(1)).expect("single-layer config")
}

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((self.0 >> 40) as f32) / ((1u64 << 24) as f32);
        (unit * 2.0 - 1.0) * scale
    }
}

fn rand_array(rng: &mut Lcg, shape: &[i32], scale: f32) -> cxx::UniquePtr<ffi::MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32(scale)).collect();
    ffi::from_slice_f32(&data, shape)
}

fn ones(n: i32) -> cxx::UniquePtr<ffi::MlxArray> {
    ffi::ones(&[n], dtype::FLOAT32)
}

/// Random f32 weights in the published (fused) layout for [`tiny_config`].
pub(crate) fn tiny_weights(cfg: &LagunaDFlashConfig, seed: u64) -> WeightMap {
    let mut rng = Lcg(seed);
    let h = cfg.hidden_size as i32;
    let d = cfg.head_dim as i32;
    let nh = cfg.num_attention_heads as i32;
    let kv = cfg.num_key_value_heads as i32;
    let ii = cfg.intermediate_size as i32;
    let n = cfg.target_layer_ids.len() as i32;
    let mut w: WeightMap = std::collections::HashMap::new();
    w.insert("fc.weight".into(), rand_array(&mut rng, &[h, n * h], 0.3));
    w.insert("hidden_norm.weight".into(), ones(h));
    w.insert("norm.weight".into(), ones(h));
    for i in 0..n {
        w.insert(format!("aux_hidden_norms.{i}.weight"), ones(h));
    }
    for i in 0..cfg.num_hidden_layers {
        let p = format!("layers.{i}");
        w.insert(format!("{p}.input_layernorm.weight"), ones(h));
        w.insert(format!("{p}.post_attention_layernorm.weight"), ones(h));
        w.insert(
            format!("{p}.self_attn.qkv_proj.weight"),
            rand_array(&mut rng, &[(nh + 2 * kv) * d, h], 0.3),
        );
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rand_array(&mut rng, &[h, nh * d], 0.3),
        );
        w.insert(
            format!("{p}.self_attn.g_proj.weight"),
            rand_array(&mut rng, &[nh, h], 0.3),
        );
        w.insert(format!("{p}.self_attn.q_norm.weight"), ones(d));
        w.insert(format!("{p}.self_attn.k_norm.weight"), ones(d));
        w.insert(
            format!("{p}.mlp.gate_proj.weight"),
            rand_array(&mut rng, &[ii, h], 0.3),
        );
        w.insert(
            format!("{p}.mlp.up_proj.weight"),
            rand_array(&mut rng, &[ii, h], 0.3),
        );
        w.insert(
            format!("{p}.mlp.down_proj.weight"),
            rand_array(&mut rng, &[h, ii], 0.3),
        );
    }
    w
}

fn key_set(w: &WeightMap) -> BTreeSet<String> {
    w.keys().cloned().collect()
}

#[test]
fn sanitize_fuses_split_qkv_and_rejects_mixed() {
    let cfg = tiny_config();
    let h = cfg.hidden_size as i32;
    let d = cfg.head_dim as i32;
    let nh = cfg.num_attention_heads as i32;
    let kv = cfg.num_key_value_heads as i32;

    // Split spelling with a `model.` prefix: q rows 1.0, k rows 2.0, v rows 3.0.
    let mut w = tiny_weights(&cfg, 1);
    let fused_key = "layers.0.self_attn.qkv_proj.weight".to_string();
    w.remove(&fused_key);
    let fill = |v: f32, rows: i32| -> cxx::UniquePtr<ffi::MlxArray> {
        ffi::from_slice_f32(&vec![v; (rows * h) as usize], &[rows, h])
    };
    w.insert(
        "model.layers.0.self_attn.q_proj.weight".into(),
        fill(1.0, nh * d),
    );
    w.insert(
        "model.layers.0.self_attn.k_proj.weight".into(),
        fill(2.0, kv * d),
    );
    w.insert(
        "model.layers.0.self_attn.v_proj.weight".into(),
        fill(3.0, kv * d),
    );
    sanitize_weights(&mut w, &cfg).expect("split q/k/v fuses");
    assert_eq!(key_set(&w), expected_weight_keys(&cfg));
    let fused = w.get(&fused_key).unwrap();
    assert_eq!(ffi::array_shape(fused), vec![(nh + 2 * kv) * d, h]);
    ffi::eval(fused);
    let data = crate::utils::array_to_vec_f32(fused);
    let row = |r: i32| data[(r * h) as usize];
    assert_eq!(row(0), 1.0);
    assert_eq!(row(nh * d - 1), 1.0);
    assert_eq!(row(nh * d), 2.0);
    assert_eq!(row(nh * d + kv * d), 3.0);

    // Both spellings on one layer.
    let mut w = tiny_weights(&cfg, 1);
    w.insert("layers.1.self_attn.q_proj.weight".into(), fill(1.0, nh * d));
    w.insert("layers.1.self_attn.k_proj.weight".into(), fill(2.0, kv * d));
    w.insert("layers.1.self_attn.v_proj.weight".into(), fill(3.0, kv * d));
    let err = sanitize_weights(&mut w, &cfg).expect_err("mixed spellings are rejected");
    assert!(err.contains("both"), "{err}");

    // Incomplete split set.
    let mut w = tiny_weights(&cfg, 1);
    w.remove("layers.1.self_attn.qkv_proj.weight");
    w.insert("layers.1.self_attn.q_proj.weight".into(), fill(1.0, nh * d));
    w.insert("layers.1.self_attn.k_proj.weight".into(), fill(2.0, kv * d));
    let err = sanitize_weights(&mut w, &cfg).expect_err("incomplete split set is rejected");
    assert!(err.contains("incomplete"), "{err}");

    // A stray key is an error, and so is a missing one.
    let mut w = tiny_weights(&cfg, 1);
    w.insert("embed_tokens.weight".into(), fill(0.0, 4));
    let err = sanitize_weights(&mut w, &cfg).expect_err("stray key");
    assert!(
        err.contains("unexpected") && err.contains("embed_tokens.weight"),
        "{err}"
    );
    let mut w = tiny_weights(&cfg, 1);
    w.remove("aux_hidden_norms.1.weight");
    let err = sanitize_weights(&mut w, &cfg).expect_err("missing key");
    assert!(
        err.contains("missing") && err.contains("aux_hidden_norms.1.weight"),
        "{err}"
    );

    // Quantization sidecars next to a projection are allowed.
    let mut w = tiny_weights(&cfg, 1);
    w.insert("fc.scales".into(), fill(1.0, 1));
    w.insert("layers.0.mlp.up_proj.biases".into(), fill(1.0, 1));
    sanitize_weights(&mut w, &cfg).expect("sidecars are part of the layout");
}

#[test]
fn context_trim_keeps_window_minus_one() {
    // A 600-position context on a window-512 layer: 511 positions survive
    // and the offset advances past all 600.
    let mut cache = LagunaDFlashContextCache::new(512);
    let k = ffi::zeros(&[1, 1, 600, 4], dtype::FLOAT32);
    let v = ffi::zeros(&[1, 1, 600, 4], dtype::FLOAT32);
    let (keys, _) = cache.update_and_fetch(k, v);
    assert_eq!(ffi::array_shape(&keys)[2], 511);
    assert_eq!(cache.len(), 511);
    assert_eq!(cache.offset(), 600);
    // The attention drops rows before projection and advances the offset
    // by the dropped count; a following short append keeps the newest 511.
    let mut cache = LagunaDFlashContextCache::new(512);
    cache.advance(89);
    let k = ffi::zeros(&[1, 1, 511, 4], dtype::FLOAT32);
    let v = ffi::zeros(&[1, 1, 511, 4], dtype::FLOAT32);
    cache.update_and_fetch(k, v);
    assert_eq!(cache.offset(), 600);
    let k = ffi::zeros(&[1, 1, 5, 4], dtype::FLOAT32);
    let v = ffi::zeros(&[1, 1, 5, 4], dtype::FLOAT32);
    let (keys, _) = cache.update_and_fetch(k, v);
    assert_eq!(ffi::array_shape(&keys)[2], 511);
    assert_eq!(cache.offset(), 605);
}

/// Drafter bound to a random embedding standing in for the target's.
pub(crate) fn bound_tiny_drafter(seed: u64) -> LagunaDFlashDrafter {
    bound_drafter_for(tiny_config(), seed)
}

fn bound_drafter_for(cfg: LagunaDFlashConfig, seed: u64) -> LagunaDFlashDrafter {
    let weights = tiny_weights(&cfg, seed);
    let mut drafter = LagunaDFlashDrafter::from_weights(&weights, cfg.clone()).expect("builds");
    let mut rng = Lcg(seed ^ 0x9e37);
    let embed = rand_array(
        &mut rng,
        &[cfg.vocab_size as i32, cfg.hidden_size as i32],
        0.5,
    );
    drafter
        .model
        .bind_target_embedding(UnifiedEmbedding::Regular(Embedding::new(embed)));
    drafter
}

fn logits_for(
    drafter: &mut LagunaDFlashDrafter,
    block: &[i32],
    hidden: &ffi::MlxArray,
) -> Vec<f32> {
    let mut caches = drafter.model.make_cache();
    let inputs = ffi::from_slice_i32(block, &[1, block.len() as i32]);
    let logits = drafter.model.forward(&inputs, hidden, &mut caches);
    let logits = ffi::astype(&logits, dtype::FLOAT32);
    ffi::eval(&logits);
    crate::utils::array_to_vec_f32(&logits)
}

#[test]
fn draft_block_is_causal_inside_the_proposal() {
    let mut drafter = bound_tiny_drafter(3);
    let cfg = drafter.model.config.clone();
    let vocab = cfg.vocab_size;
    let n = cfg.target_layer_ids.len() as i32;
    let mut rng = Lcg(11);
    let hidden = rand_array(&mut rng, &[1, 3, n * cfg.hidden_size as i32], 1.0);
    let block_len = 6usize;
    let base: Vec<i32> = vec![5, 31, 31, 31, 31, 31];
    let base_logits = logits_for(&mut drafter, &base, &hidden);
    for j in 1..block_len {
        let mut altered = base.clone();
        altered[j] = 7;
        let logits = logits_for(&mut drafter, &altered, &hidden);
        for pos in 0..block_len {
            let mut diff = 0f32;
            for v in 0..vocab {
                diff = diff.max((logits[pos * vocab + v] - base_logits[pos * vocab + v]).abs());
            }
            if pos < j {
                assert!(
                    diff < 1e-4,
                    "changing position {j} changed position {pos} by {diff}"
                );
            } else if pos - j < cfg.sliding_windows[0] {
                // Rows further than the window only see the change through
                // deeper layers, so only direct visibility is asserted.
                assert!(
                    diff > 1e-4,
                    "changing position {j} left position {pos} unchanged"
                );
            }
        }
    }
}

#[test]
fn draft_block_sees_the_context_within_its_window() {
    // Window 4: the block row j at position T + j attends context rows
    // >= T + j - 3. A single drafter layer keeps the visibility direct (a
    // second layer would carry the change through earlier block rows).
    let mut drafter = bound_drafter_for(tiny_single_layer_config(), 5);
    let cfg = drafter.model.config.clone();
    let vocab = cfg.vocab_size;
    let n = cfg.target_layer_ids.len() as i32;
    let width = n * cfg.hidden_size as i32;
    let t = 6i32;
    let mut rng = Lcg(17);
    let base_hidden = rand_array(&mut rng, &[1, t, width], 1.0);
    let base_data = {
        ffi::eval(&base_hidden);
        crate::utils::array_to_vec_f32(&base_hidden)
    };
    let block: Vec<i32> = vec![5, 31, 31, 31];
    let base_logits = logits_for(&mut drafter, &block, &base_hidden);
    let perturb = |row: i32| -> cxx::UniquePtr<ffi::MlxArray> {
        let mut data = base_data.clone();
        for c in 0..width {
            data[(row * width + c) as usize] += 1.5;
        }
        ffi::from_slice_f32(&data, &[1, t, width])
    };
    let max_diff = |a: &[f32], b: &[f32], pos: usize| -> f32 {
        (0..vocab)
            .map(|v| (a[pos * vocab + v] - b[pos * vocab + v]).abs())
            .fold(0f32, f32::max)
    };
    // Newest context row (position 5): visible to block rows 0..3
    // (positions 6, 7, 8 need rows >= 3, 4, 5; position 9 needs >= 6, so
    // row 5 is outside the last block row's window).
    let logits = logits_for(&mut drafter, &block, &perturb(5));
    assert!(max_diff(&logits, &base_logits, 0) > 1e-4);
    assert!(max_diff(&logits, &base_logits, 2) > 1e-4);
    assert!(
        max_diff(&logits, &base_logits, 3) < 1e-4,
        "block row 3 (position 9) must not see context position 5 with window 4"
    );
    // Context row 1 lies before every block row's window (row 0 at position
    // 6 needs rows >= 3) and is dropped before projection.
    let logits = logits_for(&mut drafter, &block, &perturb(1));
    for pos in 0..block.len() {
        assert!(
            max_diff(&logits, &base_logits, pos) < 1e-4,
            "context row 1 is outside the window of block row {pos}"
        );
    }
}

#[test]
fn draft_block_depends_on_rope() {
    // RoPE must reach the attention: changing the base changes the logits
    // of every block row that attends more than itself.
    let mut drafter = bound_tiny_drafter(9);
    let cfg = drafter.model.config.clone();
    let n = cfg.target_layer_ids.len() as i32;
    let mut rng = Lcg(23);
    let hidden = rand_array(&mut rng, &[1, 3, n * cfg.hidden_size as i32], 1.0);
    let block: Vec<i32> = vec![5, 31, 31, 31];
    let base = logits_for(&mut drafter, &block, &hidden);
    for layer in &mut drafter.model.layers {
        layer.self_attn.rope_base = 1.0;
    }
    let changed = logits_for(&mut drafter, &block, &hidden);
    let diff = base
        .iter()
        .zip(&changed)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        diff > 1e-3,
        "changing the RoPE base must change the logits (diff {diff})"
    );
}

#[test]
fn from_weights_rejects_projection_rows_that_disagree_with_the_config() {
    let cfg = tiny_config();
    let mut w = tiny_weights(&cfg, 4);
    let h = cfg.hidden_size as i32;
    w.insert(
        "layers.0.self_attn.qkv_proj.weight".into(),
        ffi::zeros(&[h, h], dtype::FLOAT32),
    );
    let msg = match LagunaDFlashDrafter::from_weights(&w, cfg.clone()) {
        Ok(_) => panic!("a qkv_proj row mismatch must be a load error"),
        Err(e) => format!("{e}"),
    };
    assert!(
        msg.contains("qkv_proj.weight") && msg.contains("rows"),
        "{msg}"
    );

    let mut w = tiny_weights(&cfg, 4);
    w.insert(
        "fc.weight".into(),
        ffi::zeros(&[h + 1, 2 * h], dtype::FLOAT32),
    );
    let msg = match LagunaDFlashDrafter::from_weights(&w, cfg) {
        Ok(_) => panic!("an fc row mismatch must be a load error"),
        Err(e) => format!("{e}"),
    };
    assert!(msg.contains("fc.weight"), "{msg}");
}
