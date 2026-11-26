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

//! Random tiny Inkling 2 Mini: 6 layers in blocks of 2, a 3-pass core over
//! layers 2..4, sliding layers 0, 1, 3, 4 with a 4-token window, so a
//! 90-token sequence front-trims every sliding visit.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::json;

use super::Inkling2Model;
use super::config::Inkling2Config;

const H: i32 = 16;
const HEADS: i32 = 2;
const KV: i32 = 1;
const HD: i32 = 64;
const REL: i32 = 4;
const WINDOW: i32 = 4;
const EXTENT: i32 = 8;
const E: i32 = 4;
const I: i32 = 8;
const V: i32 = 32;
const RANK: i32 = 2;

struct Rng(u64);

impl Rng {
    fn array(&mut self, shape: &[i32], scale: f32) -> UniquePtr<MlxArray> {
        let count: i32 = shape.iter().product();
        let values: Vec<f32> = (0..count)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
            })
            .collect();
        mlxcel_core::from_slice_f32(&values, shape)
    }
}

fn config(passes: usize) -> Inkling2Config {
    let raw = json!({
        "model_type": "inkling2_loop",
        "text_config": {
            "model_type": "inkling2_loop_text",
            "hidden_size": H, "num_hidden_layers": 6, "vocab_size": V, "unpadded_vocab_size": V - 2,
            "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": HD,
            "swa_num_attention_heads": HEADS, "swa_num_key_value_heads": KV, "swa_head_dim": HD,
            "sliding_window_size": WINDOW, "local_layer_ids": [0, 1, 3, 4],
            "d_rel": REL, "rel_extent": EXTENT, "log_scaling_n_floor": 4,
            "sconv_kernel_size": 4, "intermediate_size": I,
            "n_routed_experts": E, "num_experts_per_tok": 2,
            "n_layers_per_attn_res_block": 2,
            "loop_config": {"core_start": 2, "core_end": 4, "num_passes": passes, "gate_rank": RANK}
        }
    });
    Inkling2Config::from_json(&raw.to_string()).unwrap()
}

fn put(w: &mut WeightMap, key: String, shape: &[i32], scale: f32, rng: &mut Rng) {
    w.insert(key, rng.array(shape, scale));
}

fn weights(seed: u64, zero_controllers: bool) -> WeightMap {
    let mut rng = Rng(seed);
    let mut w = WeightMap::new();
    let m = "language_model.model";
    put(
        &mut w,
        format!("{m}.embed_tokens.weight"),
        &[V, H],
        1.0,
        &mut rng,
    );
    for name in ["embed_norm", "norm"] {
        w.insert(
            format!("{m}.{name}.weight"),
            mlxcel_core::ones(&[H], mlxcel_core::dtype::FLOAT32),
        );
    }
    put(
        &mut w,
        format!("{m}.final_input.query"),
        &[H],
        1.0,
        &mut rng,
    );
    put(
        &mut w,
        format!("{m}.message_queries"),
        &[2, 4, H],
        1.0,
        &mut rng,
    );
    put(
        &mut w,
        "language_model.lm_head.weight".into(),
        &[V, H],
        0.5,
        &mut rng,
    );
    for layer in 0..6 {
        let p = format!("{m}.layers.{layer}");
        let sliding = layer != 2 && layer != 5;
        if layer > 0 {
            put(
                &mut w,
                format!("{p}.attention_input.query"),
                &[if sliding { 1 } else { 3 }, H],
                1.0,
                &mut rng,
            );
        }
        put(&mut w, format!("{p}.mlp_input.query"), &[H], 1.0, &mut rng);
        let norms: &[&str] = if sliding {
            &["input_layernorm"]
        } else {
            &[
                "q_input_layernorm",
                "k_input_layernorm",
                "v_input_layernorm",
            ]
        };
        for norm in norms.iter().chain(&["post_attention_layernorm"]) {
            w.insert(
                format!("{p}.{norm}.weight"),
                mlxcel_core::ones(&[H], mlxcel_core::dtype::FLOAT32),
            );
        }
        put(
            &mut w,
            format!("{p}.attn_sconv.conv.weight"),
            &[H, 4, 1],
            0.3,
            &mut rng,
        );
        put(
            &mut w,
            format!("{p}.mlp_sconv.conv.weight"),
            &[H, 4, 1],
            0.3,
            &mut rng,
        );
        let a = format!("{p}.self_attn");
        for (name, rows) in [
            ("q_proj", HEADS * HD),
            ("k_proj", KV * HD),
            ("v_proj", KV * HD),
            ("r_proj", HEADS * REL),
            ("gate_proj", HEADS * HD),
            ("sink_proj", HEADS),
        ] {
            put(
                &mut w,
                format!("{a}.{name}.weight"),
                &[rows, H],
                0.4,
                &mut rng,
            );
        }
        put(
            &mut w,
            format!("{a}.o_proj.weight"),
            &[H, HEADS * HD],
            0.4,
            &mut rng,
        );
        put(&mut w, format!("{a}.q_scale"), &[HEADS, HD], 2.0, &mut rng);
        put(
            &mut w,
            format!("{a}.rel_scale"),
            &[HEADS, REL],
            2.0,
            &mut rng,
        );
        put(
            &mut w,
            format!("{a}.rel_proj"),
            &[REL, if sliding { WINDOW } else { EXTENT }],
            1.0,
            &mut rng,
        );
        for conv in ["k_sconv", "v_sconv"] {
            if sliding {
                put(
                    &mut w,
                    format!("{a}.{conv}.conv.weight"),
                    &[KV * HD, 4, 1],
                    0.3,
                    &mut rng,
                );
            } else {
                put(
                    &mut w,
                    format!("{a}.{conv}.static_weight"),
                    &[4, KV * HD],
                    0.3,
                    &mut rng,
                );
                put(
                    &mut w,
                    format!("{a}.{conv}.weight_proj.weight"),
                    &[KV * 4, H],
                    0.3,
                    &mut rng,
                );
            }
        }
        put(
            &mut w,
            format!("{p}.mlp.gate_weight"),
            &[E, H],
            0.5,
            &mut rng,
        );
        put(
            &mut w,
            format!("{p}.mlp.e_score_correction_bias"),
            &[E],
            0.1,
            &mut rng,
        );
        put(
            &mut w,
            format!("{p}.mlp.switch_mlp.gate_proj.weight"),
            &[E, I, H],
            0.4,
            &mut rng,
        );
        put(
            &mut w,
            format!("{p}.mlp.switch_mlp.up_proj.weight"),
            &[E, I, H],
            0.4,
            &mut rng,
        );
        put(
            &mut w,
            format!("{p}.mlp.switch_mlp.down_proj.weight"),
            &[E, H, I],
            0.4,
            &mut rng,
        );
        if (2..4).contains(&layer) {
            for controller in ["attn", "k", "v", "mlp"] {
                let c = format!("{p}.controllers.{controller}");
                put(
                    &mut w,
                    format!("{c}.down_weight"),
                    &[RANK, 2 * H],
                    0.5,
                    &mut rng,
                );
                let scale = if zero_controllers { 0.0 } else { 1.0 };
                put(
                    &mut w,
                    format!("{c}.forget_weight"),
                    &[H, RANK],
                    scale,
                    &mut rng,
                );
                put(
                    &mut w,
                    format!("{c}.gain_weight"),
                    &[H, RANK],
                    scale,
                    &mut rng,
                );
            }
        }
    }
    w
}

pub(super) fn model(passes: usize, zero_controllers: bool) -> Inkling2Model {
    Inkling2Model::from_weights(config(passes), weights(7, zero_controllers), None).unwrap()
}

fn tokens(count: usize, seed: usize) -> Vec<i32> {
    (0..count)
        .map(|i| ((i * 7 + seed * 3 + 1) % (V as usize - 2)) as i32)
        .collect()
}

fn ids(values: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(values, &[1, values.len() as i32])
}

fn rows(logits: &MlxArray) -> Vec<Vec<f32>> {
    mlxcel_core::eval(logits);
    let width = *mlxcel_core::array_shape(logits).last().unwrap() as usize;
    array_to_vec_f32(logits)
        .chunks(width)
        .map(<[f32]>::to_vec)
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Feed `chunks` one after another through fresh state; all logits rows.
fn run(model: &Inkling2Model, chunks: &[&[i32]]) -> Vec<Vec<f32>> {
    let mut caches = model.make_caches();
    chunks
        .iter()
        .flat_map(|chunk| rows(&model.forward(&ids(chunk), &mut caches, None)))
        .collect()
}

#[test]
fn layer_visits_follow_the_loop() {
    assert_eq!(model(3, false).visit_count(), 2 + 3 * 2 + 2);
    assert_eq!(model(1, false).visit_count(), 6);
}

#[test]
fn decode_after_prefill_matches_one_full_forward() {
    let model = model(3, false);
    let sequence = tokens(90, 1);
    let full = run(&model, &[&sequence]);
    let mut chunks: Vec<&[i32]> = vec![&sequence[..80]];
    chunks.extend(sequence[80..].chunks(1));
    let incremental = run(&model, &chunks);
    assert_eq!(full.len(), 90);
    assert_eq!(
        full[0].len(),
        V as usize - 2,
        "logits keep only the unpadded vocabulary"
    );
    for (row, (a, b)) in full.iter().zip(&incremental).enumerate() {
        assert!(max_diff(a, b) < 2e-4, "row {row}: {}", max_diff(a, b));
    }
}

#[test]
fn chunked_prefill_matches_single_prefill() {
    let model = model(3, false);
    let sequence = tokens(85, 2);
    let single = run(&model, &[&sequence]);
    let chunked = run(
        &model,
        &[&sequence[..30], &sequence[30..77], &sequence[77..]],
    );
    for (row, (a, b)) in single.iter().zip(&chunked).enumerate() {
        assert!(max_diff(a, b) < 2e-4, "row {row}: {}", max_diff(a, b));
    }
}

#[test]
fn last_row_logits_equal_the_full_forward_row() {
    let model = model(3, false);
    let sequence = tokens(20, 3);
    let full = run(&model, &[&sequence]);
    let mut caches = model.make_caches();
    let last = model
        .last_row_logits(&ids(&sequence), None, None, &mut caches, None, 19)
        .unwrap();
    assert!(max_diff(&rows(&last)[0], &full[19]) < 1e-5);
}

#[test]
fn the_loop_and_its_controllers_change_the_output() {
    let sequence = tokens(24, 4);
    let looped = run(&model(3, false), &[&sequence]);
    let one_pass = run(&model(1, false), &[&sequence]);
    let silent = run(&model(3, true), &[&sequence]);
    let last = looped.len() - 1;
    assert!(
        max_diff(&looped[last], &one_pass[last]) > 1e-2,
        "extra passes must matter"
    );
    assert!(
        max_diff(&looped[last], &silent[last]) > 1e-3,
        "controllers must matter"
    );
}

/// The `AttentionSinkKVCache` equivalence (upstream f2956e68): after the
/// sliding visits have front-trimmed, a rejected tail can be trimmed and
/// re-decoded to the same logits, and a trim past the kept margin is refused
/// without touching the state.
#[test]
fn trim_rewinds_kv_and_convolution_state_exactly() {
    let model = model(3, false);
    let sequence = tokens(95, 5);
    let mut caches = model.make_caches();
    let _ = rows(&model.forward(&ids(&sequence[..85]), &mut caches, None));
    let first: Vec<_> = sequence[85..]
        .iter()
        .map(|&token| rows(&model.forward(&ids(&[token]), &mut caches, None)).remove(0))
        .collect();
    model.trim_state(None, 10).unwrap();
    for (step, &token) in sequence[85..].iter().enumerate() {
        let again = rows(&model.forward(&ids(&[token]), &mut caches, None)).remove(0);
        assert!(max_diff(&again, &first[step]) < 1e-5, "step {step}");
    }
    let error = model.trim_state(None, 40).unwrap_err();
    assert!(error.contains("Inkling 2 visit"), "{error}");
    let next = rows(&model.forward(&ids(&[3]), &mut caches, None)).remove(0);
    let mut reference = model.make_caches();
    let mut whole = sequence.clone();
    whole.push(3);
    let expected = rows(&model.forward(&ids(&whole), &mut reference, None))
        .pop()
        .unwrap();
    assert!(
        max_diff(&next, &expected) < 2e-4,
        "a refused trim must not change the state"
    );
}

/// Prompt-lookup verify rounds are allowed: a verify block longer than the
/// convolution margin can still be rejected whole and re-decoded to the same
/// logits (issue #3359 makes the grouped expert kernel reachable this way).
#[test]
fn a_rejected_verify_block_rewinds_exactly() {
    let model = model(3, false);
    assert!(model.supports_speculative_verify_rewind());
    assert!(
        mlxcel_core::speculative::prompt_lookup::prompt_lookup_unsupported_reason(&model).is_none()
    );
    let sequence = tokens(60, 6);
    let mut caches = model.make_caches();
    let _ = rows(&model.forward(&ids(&sequence[..40]), &mut caches, None));
    let draft: Vec<i32> = tokens(20, 9);
    {
        let _speculative = mlxcel_core::cache::DecodeLookaheadAppendScope::enter();
        let _ = rows(&model.forward(&ids(&draft), &mut caches, None));
    }
    model.trim_state(None, 20).unwrap();
    let again = rows(&model.forward(&ids(&sequence[40..]), &mut caches, None));
    let full = run(&model, &[&sequence]);
    for (row, (a, b)) in again.iter().zip(&full[40..]).enumerate() {
        assert!(max_diff(a, b) < 2e-4, "row {row}: {}", max_diff(a, b));
    }
}
