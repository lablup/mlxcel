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

//! Unit and reference-parity tests for the DeepSeek-V4 port.
//!
//! The numeric fixtures (`HC_*`, `YARN_FREQS`, `LSW_*`, `SSP_*`, `SC_*`,
//! `OC_*`) were computed in float64 by replaying the reference math from
//! `references/mlx-vlm/mlx_vlm/models/deepseek_v4/` (`_hc_split_sinkhorn_ops`,
//! `DeepseekV4RoPE.__init__`, `_limited_swiglu`, `_score_func`,
//! `_simple_compress_kv`, `_overlap_compress_kv`) outside MLX, so a port bug
//! and a fixture bug cannot share a cause.

use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;

use super::compress::{PoolingCache, overlap_compress_kv, pool_visible_counts, simple_compress_kv};
use super::indexer::Indexer;
use super::moe::{MoEGate, limited_swiglu};
use super::rope::{V4Rope, v4_rope_base_freqs, v4_rope_padded_freqs};
use super::*;

// Deterministic pseudo-noise so quantization-free fixtures stay varied and
// finite without a random dependency.
fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((state >> 8) as f32 / (1 << 24) as f32) - 0.5) * 0.5
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let denom = e.abs().max(1.0);
        assert!(
            (a - e).abs() / denom < tol,
            "{what}[{i}]: got {a}, expected {e} (tol {tol})"
        );
    }
}

fn tiny_args_json() -> &'static str {
    r#"{
        "model_type": "deepseek_v4",
        "vocab_size": 32,
        "hidden_size": 8,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "q_lora_rank": 6,
        "qk_rope_head_dim": 4,
        "o_groups": 2,
        "o_lora_rank": 4,
        "moe_intermediate_size": 8,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "num_hash_layers": 1,
        "norm_topk_prob": true,
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "compress_ratios": [0, 128, 4],
        "compress_rope_theta": 160000.0,
        "sliding_window": 32,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 3,
        "hc_eps": 1e-6,
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 4,
        "index_block": 2,
        "index_keep": 2,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "rope_scaling": {"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
                          "beta_fast": 32, "beta_slow": 1},
        "eos_token_id": 1
    }"#
}

fn tiny_args() -> ModelArgs {
    let args: ModelArgs = serde_json::from_str(tiny_args_json()).expect("tiny config parses");
    args.normalized().expect("tiny config validates")
}

fn put(weights: &mut WeightMap, name: &str, shape: &[i32], seed: u32) {
    let n: i32 = shape.iter().product();
    weights.insert(
        name.to_string(),
        mlxcel_core::from_slice_f32(&noise(n as usize, seed), shape),
    );
}

fn put_ones(weights: &mut WeightMap, name: &str, n: i32) {
    weights.insert(
        name.to_string(),
        mlxcel_core::from_slice_f32(&vec![1.0; n as usize], &[n]),
    );
}

/// A full raw-f32 weight map for [`tiny_args`], exercising the 2-D `wo_a`
/// reshape and the int64 `tid2eid` cast the real checkpoint needs.
fn tiny_weight_map(args: &ModelArgs) -> WeightMap {
    let mut w = WeightMap::new();
    let hidden = args.hidden_size as i32;
    let hc = args.hc_mult as i32;
    let mix = (2 + hc) * hc;
    let heads = args.num_attention_heads as i32;
    let head_dim = args.head_dim as i32;
    let q_lora = args.q_lora_rank as i32;
    let o_groups = args.o_groups as i32;
    let o_lora = args.o_lora_rank as i32;
    let experts = args.n_routed_experts as i32;
    let inter = args.moe_intermediate_size as i32;
    let idx_heads = args.index_n_heads as i32;
    let idx_hd = args.index_head_dim as i32;

    put(
        &mut w,
        "model.embed_tokens.weight",
        &[args.vocab_size as i32, hidden],
        1,
    );
    put_ones(&mut w, "model.norm.weight", hidden);
    put(&mut w, "model.hc_head.fn", &[hc, hc * hidden], 2);
    put(&mut w, "model.hc_head.base", &[hc], 3);
    put_ones(&mut w, "model.hc_head.scale", 1);
    put(
        &mut w,
        "lm_head.weight",
        &[args.vocab_size as i32, hidden],
        4,
    );

    for (i, &ratio) in args.compress_ratios.iter().enumerate() {
        let seed = (i as u32 + 2) * 1000;
        let p = format!("model.layers.{i}");
        put(
            &mut w,
            &format!("{p}.attn.wq_a.weight"),
            &[q_lora, hidden],
            seed + 1,
        );
        put_ones(&mut w, &format!("{p}.attn.q_norm.weight"), q_lora);
        put(
            &mut w,
            &format!("{p}.attn.wq_b.weight"),
            &[heads * head_dim, q_lora],
            seed + 2,
        );
        put(
            &mut w,
            &format!("{p}.attn.wkv.weight"),
            &[head_dim, hidden],
            seed + 3,
        );
        put_ones(&mut w, &format!("{p}.attn.kv_norm.weight"), head_dim);
        // 2-D on purpose: sanitize must reshape into [o_groups, o_lora, -1].
        put(
            &mut w,
            &format!("{p}.attn.wo_a.weight"),
            &[o_groups * o_lora, heads * head_dim / o_groups],
            seed + 4,
        );
        put(
            &mut w,
            &format!("{p}.attn.wo_b.weight"),
            &[hidden, o_groups * o_lora],
            seed + 5,
        );
        put(&mut w, &format!("{p}.attn.attn_sink"), &[heads], seed + 6);
        put_ones(&mut w, &format!("{p}.attn_norm.weight"), hidden);
        put_ones(&mut w, &format!("{p}.ffn_norm.weight"), hidden);
        for hc_name in ["attn_hc", "ffn_hc"] {
            put(
                &mut w,
                &format!("{p}.{hc_name}.fn"),
                &[mix, hc * hidden],
                seed + 7,
            );
            put(&mut w, &format!("{p}.{hc_name}.base"), &[mix], seed + 8);
            put_ones(&mut w, &format!("{p}.{hc_name}.scale"), 3);
        }
        if ratio > 0 {
            let out_dim = head_dim * if ratio == 4 { 2 } else { 1 };
            let c = format!("{p}.attn.compressor");
            put(
                &mut w,
                &format!("{c}.wkv.weight"),
                &[out_dim, hidden],
                seed + 9,
            );
            put(
                &mut w,
                &format!("{c}.wgate.weight"),
                &[out_dim, hidden],
                seed + 10,
            );
            put(
                &mut w,
                &format!("{c}.ape"),
                &[ratio as i32, out_dim],
                seed + 11,
            );
            put_ones(&mut w, &format!("{c}.norm.weight"), head_dim);
        }
        if ratio == 4 {
            let ix = format!("{p}.attn.indexer");
            put(
                &mut w,
                &format!("{ix}.wq_b.weight"),
                &[idx_heads * idx_hd, q_lora],
                seed + 12,
            );
            put(
                &mut w,
                &format!("{ix}.weights_proj.weight"),
                &[idx_heads, hidden],
                seed + 13,
            );
            let c = format!("{ix}.compressor");
            put(
                &mut w,
                &format!("{c}.wkv.weight"),
                &[2 * idx_hd, hidden],
                seed + 14,
            );
            put(
                &mut w,
                &format!("{c}.wgate.weight"),
                &[2 * idx_hd, hidden],
                seed + 15,
            );
            put(&mut w, &format!("{c}.ape"), &[4, 2 * idx_hd], seed + 16);
            put_ones(&mut w, &format!("{c}.norm.weight"), idx_hd);
        }
        put(
            &mut w,
            &format!("{p}.ffn.gate.weight"),
            &[experts, hidden],
            seed + 17,
        );
        if i < args.num_hash_layers {
            // int64 on purpose: the checkpoint ships the table as I64.
            let table: Vec<i64> = (0..args.vocab_size as i64 * 2)
                .map(|v| (v * 7 + 3) % i64::from(experts))
                .collect();
            w.insert(
                format!("{p}.ffn.gate.tid2eid"),
                mlxcel_core::from_slice_i64(&table, &[args.vocab_size as i32, 2]),
            );
        } else {
            put(
                &mut w,
                &format!("{p}.ffn.gate.e_score_correction_bias"),
                &[experts],
                seed + 18,
            );
        }
        for name in ["gate_proj", "up_proj", "down_proj"] {
            let (out, inp) = if name == "down_proj" {
                (hidden, inter)
            } else {
                (inter, hidden)
            };
            put(
                &mut w,
                &format!("{p}.ffn.switch_mlp.{name}.weight"),
                &[experts, out, inp],
                seed + 19,
            );
            put(
                &mut w,
                &format!("{p}.ffn.shared_experts.{name}.weight"),
                &[out, inp],
                seed + 20,
            );
        }
    }
    w
}

// ---------------------------------------------------------------------------
// Config parsing and validation.
// ---------------------------------------------------------------------------

#[test]
fn deepseek_v4_default_compress_ratios_match_reference_post_init() {
    let args: ModelArgs = serde_json::from_str(
        r#"{"model_type":"deepseek_v4","vocab_size":1000,"hidden_size":64,
            "num_hidden_layers":5,"num_attention_heads":8,"head_dim":16,
            "qk_rope_head_dim":4,"index_head_dim":8,"o_groups":2,"o_lora_rank":4,
            "q_lora_rank":8,"n_routed_experts":4,"num_experts_per_tok":2,
            "num_hash_layers":1}"#,
    )
    .expect("parse");
    let args = args.normalized().expect("normalize");
    assert_eq!(args.compress_ratios, vec![0, 128, 4, 128, 0]);
}

#[test]
fn deepseek_v4_config_rejects_hostile_values() {
    let base: serde_json::Value = serde_json::from_str(tiny_args_json()).expect("parse");
    let cases: &[(&str, serde_json::Value)] = &[
        ("num_key_value_heads", serde_json::json!(2)),
        ("compress_ratios", serde_json::json!([0, 7, 4])),
        ("compress_ratios", serde_json::json!([0, 128])),
        ("scoring_func", serde_json::json!("softplus2")),
        ("num_experts_per_tok", serde_json::json!(5)),
        ("num_experts_per_tok", serde_json::json!(0)),
        ("hc_mult", serde_json::json!(0)),
        ("hc_eps", serde_json::json!(0.0)),
        ("o_groups", serde_json::json!(3)),
        ("head_dim", serde_json::json!(2)),
        ("qk_rope_head_dim", serde_json::json!(3)),
        (
            "rope_scaling",
            serde_json::json!({"type": "linear", "factor": 2}),
        ),
        ("sliding_window", serde_json::json!(0)),
        ("num_hash_layers", serde_json::json!(4)),
    ];
    for (field, value) in cases {
        let mut cfg = base.clone();
        cfg[field] = value.clone();
        let args: ModelArgs = serde_json::from_value(cfg).expect("parse");
        assert!(
            args.normalized().is_err(),
            "config with hostile {field} was accepted"
        );
    }
    // Positive control: the unmodified config validates.
    let args: ModelArgs = serde_json::from_value(base).expect("parse");
    assert!(args.normalized().is_ok());
}

#[test]
fn deepseek_v4_expert_quantization_override_is_per_path() {
    let mut cfg: serde_json::Value = serde_json::from_str(tiny_args_json()).expect("parse");
    cfg["quantization"] = serde_json::json!({
        "group_size": 64, "bits": 4, "mode": "affine",
        "model.layers.0.ffn.switch_mlp.gate_proj":
            {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    });
    let args: ModelArgs = serde_json::from_value(cfg).expect("parse");
    let args = args.normalized().expect("validate");
    let (gs, bits, mode) = args.expert_quantization("model.layers.0.ffn.switch_mlp.gate_proj");
    assert_eq!((gs, bits, mode.as_str()), (32, 4, "mxfp4"));
    let (gs, bits, mode) = args.expert_quantization("model.layers.0.ffn.switch_mlp.up_proj");
    assert_eq!((gs, bits, mode.as_str()), (64, 4, "affine"));
}

#[test]
fn deepseek_v4_config_rejects_hostile_quantization_overrides() {
    for (group_size, bits, _) in crate::models::switch_layers::HOSTILE_QUANT_PARAMS {
        let mut cfg: serde_json::Value = serde_json::from_str(tiny_args_json()).expect("parse");
        cfg["quantization"] = serde_json::json!({
            "group_size": 64, "bits": 4,
            "model.layers.0.ffn.switch_mlp.gate_proj":
                {"group_size": group_size, "bits": bits}
        });
        let args: ModelArgs = serde_json::from_value(cfg).expect("parse");
        assert!(
            args.normalized().is_err(),
            "hostile override ({group_size}, {bits}) was accepted"
        );
    }
    for mode in crate::models::switch_layers::HOSTILE_QUANT_MODES {
        let mut cfg: serde_json::Value = serde_json::from_str(tiny_args_json()).expect("parse");
        cfg["quantization"] = serde_json::json!({
            "group_size": 64, "bits": 4,
            "model.layers.0.ffn.switch_mlp.gate_proj":
                {"group_size": 64, "bits": 4, "mode": mode}
        });
        let args: ModelArgs = serde_json::from_value(cfg).expect("parse");
        assert!(
            args.normalized().is_err(),
            "hostile override mode {mode:?} was accepted"
        );
    }
    // Positive control: a valid mxfp4 override passes.
    let mut cfg: serde_json::Value = serde_json::from_str(tiny_args_json()).expect("parse");
    cfg["quantization"] = serde_json::json!({
        "group_size": 64, "bits": 4,
        "model.layers.0.ffn.switch_mlp.gate_proj":
            {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    });
    let args: ModelArgs = serde_json::from_value(cfg).expect("parse");
    assert!(args.normalized().is_ok());
}

// ---------------------------------------------------------------------------
// RoPE tables.
// ---------------------------------------------------------------------------

// Computed by replaying DeepseekV4RoPE.__init__ in float64:
// dims=8, base=160000, yarn factor=16, orig=65536, beta 32/1.
const YARN_FREQS: &[f32] = &[1.0, 20.0, 581.818_2, 21333.333];

#[test]
fn v4_rope_yarn_freqs_match_reference() {
    let scaling = RopeScalingV4 {
        scaling_type: Some("yarn".to_string()),
        factor: Some(16.0),
        original_max_position_embeddings: Some(65536),
        beta_fast: Some(32.0),
        beta_slow: Some(1.0),
    };
    let freqs = v4_rope_base_freqs(8, 160000.0, Some(&scaling)).expect("yarn freqs");
    assert_close(&freqs, YARN_FREQS, 1e-4, "yarn freqs");

    // No scaling: plain base**(2i/dims) wavelengths.
    let plain = v4_rope_base_freqs(4, 10000.0, None).expect("plain freqs");
    assert_close(&plain, &[1.0, 100.0], 1e-5, "plain freqs");
}

#[test]
fn v4_rope_padded_freqs_prefix_negate_and_scale() {
    let base = vec![1.0_f32, 20.0];
    // head_dim 12, dims 4 -> 4 leading inf pairs.
    let fwd = v4_rope_padded_freqs(&base, 4, 12, 1, false).expect("fwd");
    assert_eq!(fwd.len(), 6);
    assert!(fwd[..4].iter().all(|v| v.is_infinite() && *v > 0.0));
    assert_close(&fwd[4..], &[1.0, 20.0], 1e-6, "fwd tail");

    let inv = v4_rope_padded_freqs(&base, 4, 12, 1, true).expect("inv");
    assert_close(&inv[4..], &[-1.0, -20.0], 1e-6, "inverse tail negated");
    // The inf prefix must NOT be negated into -inf (a -inf frequency would
    // rotate; the reference negates only after padding decides the pairs).
    assert!(inv[..4].iter().all(|v| v.is_infinite() && *v > 0.0));

    let pooled = v4_rope_padded_freqs(&base, 4, 4, 4, false).expect("pooled");
    assert_close(&pooled, &[0.25, 5.0], 1e-6, "freq_scale divides the table");
}

#[test]
fn v4_rope_forward_then_inverse_is_identity() {
    let rope = V4Rope::new(4, 10000.0, None, 1, &[(8, false), (8, true)]).expect("rope");
    let x = mlxcel_core::from_slice_f32(&noise(2 * 8, 7), &[1, 1, 2, 8]);
    let roped = rope.apply(&x, 37, false);
    let back = rope.apply(&roped, 37, true);
    assert_close(
        &array_to_vec_f32(&back),
        &array_to_vec_f32(&x),
        1e-4,
        "rope inverse undoes forward",
    );
    // And the rotation must actually do something at a nonzero offset.
    let moved = array_to_vec_f32(&roped);
    let orig = array_to_vec_f32(&x);
    assert!(
        moved.iter().zip(&orig).any(|(a, b)| (a - b).abs() > 1e-3),
        "rope at offset 37 must rotate the tail lanes"
    );
    // Leading (nope) lanes are inf-frequency and must be untouched.
    assert_close(&moved[..4], &orig[..4], 1e-6, "nope lanes unrotated");
}

// ---------------------------------------------------------------------------
// HyperConnections: float64 replay of `_hc_split_sinkhorn_ops` with
// hc=4, hidden=2, iters=5, eps=1e-6 over HC_X and the HC_FN/BASE/SCALE
// parameters below.
// ---------------------------------------------------------------------------

const HC_X: &[f32] = &[0.5, -1.0, 1.5, 0.25, -0.75, 2.0, 0.1, -0.3];
const HC_POST: &[f32] = &[0.82569532, 0.78207842, 0.764_055_8, 0.772_175_2];
const HC_COMB: &[f32] = &[
    0.17882462, 0.21570121, 0.26847933, 0.33699383, 0.18062745, 0.23122135, 0.27858402, 0.30956618,
    0.27885287, 0.27492428, 0.2457538, 0.20046805, 0.36169405, 0.27815216, 0.20718185, 0.15297094,
];
const HC_COLLAPSED: &[f32] = &[0.7802578, 0.36594218];
const HC_SCALE: &[f32] = &[1.1, 0.9, 1.3];

fn hc_fixture_weights() -> WeightMap {
    let hc = 4;
    let d = 2;
    let mix = (2 + hc) * hc;
    let fn_vals: Vec<f32> = (0..mix)
        .flat_map(|i| {
            (0..hc * d).map(move |j| ((0.3 * i as f64 + 0.7 * j as f64).sin() * 0.5) as f32)
        })
        .collect();
    let base_vals: Vec<f32> = (0..mix)
        .map(|i| ((0.11 * i as f64).cos() * 0.2) as f32)
        .collect();
    let mut w = WeightMap::new();
    w.insert(
        "hc.fn".to_string(),
        mlxcel_core::from_slice_f32(&fn_vals, &[mix, hc * d]),
    );
    w.insert(
        "hc.base".to_string(),
        mlxcel_core::from_slice_f32(&base_vals, &[mix]),
    );
    w.insert(
        "hc.scale".to_string(),
        mlxcel_core::from_slice_f32(HC_SCALE, &[3]),
    );
    w
}

#[test]
fn hyper_connection_matches_reference_sinkhorn_math() {
    let weights = hc_fixture_weights();
    let hc = hyper::HyperConnection::from_weights(&weights, "hc", 2, 4, 5, 1e-6, 1e-6)
        .expect("hc fixture loads");
    let x = mlxcel_core::from_slice_f32(HC_X, &[1, 1, 4, 2]);
    let (collapsed, post, comb) = hc.forward(&x);
    assert_close(
        &array_to_vec_f32(&collapsed),
        HC_COLLAPSED,
        2e-4,
        "hc collapsed",
    );
    assert_close(&array_to_vec_f32(&post), HC_POST, 2e-4, "hc post");
    let comb_v = array_to_vec_f32(&comb);
    assert_close(&comb_v, HC_COMB, 2e-4, "hc comb");
    // Sinkhorn output must be near doubly stochastic.
    for r in 0..4 {
        let row: f32 = comb_v[r * 4..(r + 1) * 4].iter().sum();
        assert!((row - 1.0).abs() < 1e-3, "comb row {r} sums to {row}");
        let col: f32 = (0..4).map(|c| comb_v[c * 4 + r]).sum();
        assert!((col - 1.0).abs() < 1e-3, "comb col {r} sums to {col}");
    }
}

#[test]
fn hc_expand_folds_output_back_into_widened_residual() {
    let weights = hc_fixture_weights();
    let hc = hyper::HyperConnection::from_weights(&weights, "hc", 2, 4, 5, 1e-6, 1e-6)
        .expect("hc fixture loads");
    let residual = mlxcel_core::from_slice_f32(HC_X, &[1, 1, 4, 2]);
    let (_, post, comb) = hc.forward(&residual);
    let x_out = mlxcel_core::from_slice_f32(&[0.3, -0.6], &[1, 1, 2]);
    let expanded = hyper::hc_expand(&x_out, &residual, &post, &comb);
    let got = array_to_vec_f32(&expanded);

    let post_v = array_to_vec_f32(&post);
    let comb_v = array_to_vec_f32(&comb);
    let x_v = [0.3_f32, -0.6];
    let mut expected = vec![0.0_f32; 8];
    for h in 0..4 {
        for d in 0..2 {
            // comb.T @ residual: entry [h][r] of comb.T is comb[r][h].
            let mixed: f32 = (0..4).map(|r| comb_v[r * 4 + h] * HC_X[r * 2 + d]).sum();
            expected[h * 2 + d] = post_v[h] * x_v[d] + mixed;
        }
    }
    assert_close(&got, &expected, 2e-4, "hc_expand");
}

#[test]
fn hyper_head_collapse_is_sigmoid_gated_sum() {
    // fn = 0, base = 0, scale = 1 -> pre = sigmoid(0) + eps per lane, so the
    // head must collapse to (0.5 + eps) * sum over the widened axis.
    let mut w = WeightMap::new();
    w.insert(
        "head.fn".to_string(),
        mlxcel_core::from_slice_f32(&[0.0; 8], &[2, 4]),
    );
    w.insert(
        "head.base".to_string(),
        mlxcel_core::from_slice_f32(&[0.0; 2], &[2]),
    );
    w.insert(
        "head.scale".to_string(),
        mlxcel_core::from_slice_f32(&[1.0], &[1]),
    );
    let head = hyper::HyperHead::from_weights(&w, "head", 2, 2, 1e-6, 1e-6).expect("head");
    let x = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let out = array_to_vec_f32(&head.forward(&x));
    assert_close(
        &out,
        &[(1.0 + 3.0) * 0.500001, (2.0 + 4.0) * 0.500001],
        1e-4,
        "hyper head",
    );
}

// ---------------------------------------------------------------------------
// Pooling cache and compressors.
// ---------------------------------------------------------------------------

#[test]
fn pooling_cache_prompt_then_decode_emits_full_windows_only() {
    let mut pool = PoolingCache::new(4);
    let vals: Vec<f32> = (0..12).map(|v| v as f32).collect(); // [1, 6, 2]
    let kv = mlxcel_core::from_slice_f32(&vals, &[1, 6, 2]);
    let gate = mlxcel_core::from_slice_f32(&vals, &[1, 6, 2]);

    // Prompt: 6 tokens = one full window + remainder 2, base offset 0.
    let (r_kv, _r_gate, base) = pool.accumulate_windows(&kv, &gate, 0);
    assert_eq!(mlxcel_core::array_shape(&r_kv), vec![1, 4, 2]);
    assert_eq!(base, 0);
    assert_eq!(pool.remainder, 2);
    assert_close(
        &array_to_vec_f32(&r_kv),
        &vals[..8],
        1e-6,
        "ready window rows",
    );

    // Decode token 7: window still incomplete.
    let t6 = mlxcel_core::from_slice_f32(&[12.0, 13.0], &[1, 1, 2]);
    let (r_kv, _, _) = pool.accumulate_windows(&t6, &t6, 6);
    assert_eq!(mlxcel_core::array_shape(&r_kv)[1], 0);
    assert_eq!(pool.remainder, 3);

    // Decode token 8 completes the window [4..8): base = 7 - 4 + 1 = 4.
    let t7 = mlxcel_core::from_slice_f32(&[14.0, 15.0], &[1, 1, 2]);
    let (r_kv, _, base) = pool.accumulate_windows(&t7, &t7, 7);
    assert_eq!(mlxcel_core::array_shape(&r_kv), vec![1, 4, 2]);
    assert_eq!(base, 4);
    assert_eq!(pool.remainder, 0);
    assert_close(
        &array_to_vec_f32(&r_kv),
        &[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        1e-6,
        "decode-completed window",
    );

    // The pooled buffer grows only through update_and_fetch.
    let px = mlxcel_core::from_slice_f32(&[1.0, 2.0], &[1, 1, 2]);
    let pooled = pool.update_and_fetch(px);
    assert_eq!(mlxcel_core::array_shape(&pooled), vec![1, 1, 2]);
    let empty = mlxcel_core::zeros(&[1, 0, 2], mlxcel_core::dtype::FLOAT32);
    let pooled = pool.update_and_fetch(empty);
    assert_eq!(mlxcel_core::array_shape(&pooled), vec![1, 1, 2]);
}

#[test]
fn pool_visibility_counts_match_reference_make_mask() {
    // Query at absolute position offset + j sees pooled rows
    // < (offset + 1 + j) / ratio.
    assert_eq!(pool_visible_counts(4, 0, 4, 10), vec![0, 0, 0, 1]);
    assert_eq!(pool_visible_counts(3, 6, 4, 10), vec![1, 2, 2]);
    assert_eq!(pool_visible_counts(2, 100, 4, 3), vec![3, 3]);
}

// Float64 replay of `_simple_compress_kv` (nw=2, r=2, d=2).
const SC_KV: &[f32] = &[0.5, -1.0, 1.0, 2.0, -0.5, 0.25, 2.0, -1.5];
const SC_GATE: &[f32] = &[0.2, -0.4, 0.6, 0.1, -0.3, 0.8, 0.4, 0.5];
const SC_APE: &[f32] = &[0.05, -0.1, 0.2, 0.3];
const SC_OUT: &[f32] = &[0.8170678, 1.1328485, 1.2514179, -0.66871358];

#[test]
fn simple_compress_matches_reference_math() {
    let kv = mlxcel_core::from_slice_f32(SC_KV, &[1, 2, 2, 2]);
    let gate = mlxcel_core::from_slice_f32(SC_GATE, &[1, 2, 2, 2]);
    let ape = mlxcel_core::from_slice_f32(SC_APE, &[2, 2]);
    let out = simple_compress_kv(&kv, &gate, &ape);
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 2, 2]);
    assert_close(&array_to_vec_f32(&out), SC_OUT, 2e-4, "simple compress");
}

// Float64 replay of `_overlap_compress_kv` (nw=2, r=2, d=4 -> out 2).
const OC_KV: &[f32] = &[
    0.5, -1.0, 0.3, 0.7, 1.0, 2.0, -0.2, 0.4, -0.5, 0.25, 0.9, -0.6, 2.0, -1.5, 0.1, 1.2,
];
const OC_GATE: &[f32] = &[
    0.2, -0.4, 0.5, -0.1, 0.6, 0.1, -0.7, 0.9, -0.3, 0.8, 0.2, 0.4, 0.4, 0.5, -0.2, 0.3,
];
const OC_APE: &[f32] = &[0.05, -0.1, 0.15, -0.2, 0.2, 0.3, -0.25, 0.1];
const OC_OUT: &[f32] = &[0.21600919, 0.46424951, 0.756_068_8, 0.713_791_5];

#[test]
fn overlap_compress_matches_reference_math() {
    let kv = mlxcel_core::from_slice_f32(OC_KV, &[1, 2, 2, 4]);
    let gate = mlxcel_core::from_slice_f32(OC_GATE, &[1, 2, 2, 4]);
    let ape = mlxcel_core::from_slice_f32(OC_APE, &[2, 4]);
    let out = overlap_compress_kv(&kv, &gate, &ape);
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 2, 2]);
    assert_close(&array_to_vec_f32(&out), OC_OUT, 2e-4, "overlap compress");
}

// ---------------------------------------------------------------------------
// MoE: limited SwiGLU, sqrtsoftplus scoring, bias contract, hash routing.
// ---------------------------------------------------------------------------

// Float64 replay of `_limited_swiglu` at limit 10.
const LSW_GATE: &[f32] = &[-20.0, -5.0, 0.0, 5.0, 20.0];
const LSW_UP: &[f32] = &[12.0, -12.0, 3.0, -3.0, 0.5];
const LSW_OUT: &[f32] = &[-4.1223072e-07, 0.33464255, 0.0, -14.899607, 4.999773];

#[test]
fn limited_swiglu_clamps_match_reference() {
    let gate = mlxcel_core::from_slice_f32(LSW_GATE, &[5]);
    let up = mlxcel_core::from_slice_f32(LSW_UP, &[5]);
    let out = limited_swiglu(&gate, &up, 10.0);
    assert_close(&array_to_vec_f32(&out), LSW_OUT, 2e-4, "limited swiglu");

    // limit <= 0 disables the clamp entirely.
    let unclamped = limited_swiglu(&gate, &up, 0.0);
    let v = array_to_vec_f32(&unclamped);
    let silu20 = 20.0_f32 / (1.0 + (-20.0_f32).exp());
    assert!((v[4] - silu20 * 0.5).abs() < 1e-3, "unclamped tail");
}

// Float64 replay of `_score_func("sqrtsoftplus")`.
const SSP_IN: &[f32] = &[-4.0, -1.0, 0.0, 0.5, 3.0];
const SSP_OUT: &[f32] = &[0.13472167, 0.55969785, 0.83255461, 0.986_953_4, 1.7460204];

#[test]
fn moe_gate_bias_selects_but_unbiased_scores_weight() {
    // hidden = 5, experts = 5: x = one-hot rows so logits = gate row dot,
    // with gate weight = I * SSP_IN scaled so scores are exactly
    // sqrtsoftplus(SSP_IN) per expert for x = all-ones? Simpler: hidden = 1,
    // gate weight column = SSP_IN, x = [1] -> logits = SSP_IN.
    let mut cfg: serde_json::Value = serde_json::from_str(tiny_args_json()).expect("parse");
    cfg["hidden_size"] = serde_json::json!(1);
    cfg["n_routed_experts"] = serde_json::json!(5);
    cfg["num_experts_per_tok"] = serde_json::json!(2);
    // hidden=1 fails head/o_group checks only if attention reads it; gate
    // math needs a valid config, so keep attention fields consistent.
    cfg["head_dim"] = serde_json::json!(8);
    let args: ModelArgs = serde_json::from_value(cfg).expect("parse");
    let args = args.normalized().expect("validate");

    let mut w = WeightMap::new();
    w.insert(
        "gate.weight".to_string(),
        mlxcel_core::from_slice_f32(SSP_IN, &[5, 1]),
    );
    // Bias pushes expert 0 (lowest score) into the top-2 and pushes expert 4
    // (highest score) out.
    w.insert(
        "gate.e_score_correction_bias".to_string(),
        mlxcel_core::from_slice_f32(&[10.0, 0.0, 0.0, 0.0, -10.0], &[5]),
    );
    let gate = MoEGate::from_weights(&w, &args, "gate", false).expect("gate");

    let x = mlxcel_core::from_slice_f32(&[1.0], &[1, 1, 1]);
    let ids = mlxcel_core::from_slice_i32(&[0], &[1, 1]);
    let (inds, weights) = gate.forward(&x, &ids);
    let mut got_inds = array_to_vec_f32(&mlxcel_core::astype(&inds, mlxcel_core::dtype::FLOAT32));
    got_inds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(got_inds, vec![0.0, 3.0], "bias must steer selection");

    // Weights come from the UNBIASED sqrtsoftplus scores, renormalised and
    // scaled by routed_scaling_factor = 1.5.
    let w_v = array_to_vec_f32(&weights);
    let inds_v = array_to_vec_f32(&mlxcel_core::astype(&inds, mlxcel_core::dtype::FLOAT32));
    let s: Vec<f32> = inds_v.iter().map(|&i| SSP_OUT[i as usize]).collect();
    let sum: f32 = s.iter().sum();
    let expected: Vec<f32> = s.iter().map(|v| v / (sum + 1e-20) * 1.5).collect();
    assert_close(&w_v, &expected, 2e-4, "unbiased weighting contract");
}

#[test]
fn moe_gate_hash_routing_takes_indices_from_tid2eid() {
    let args = tiny_args();
    let mut w = WeightMap::new();
    put(&mut w, "gate.weight", &[4, 8], 99);
    let table: Vec<i64> = (0..64).map(|v| (v * 5 + 1) % 4).collect();
    w.insert(
        "gate.tid2eid".to_string(),
        mlxcel_core::from_slice_i64(&table, &[32, 2]),
    );
    let gate = MoEGate::from_weights(&w, &args, "gate", true).expect("hash gate");

    let x = mlxcel_core::from_slice_f32(&noise(2 * 8, 5), &[1, 2, 8]);
    let ids = mlxcel_core::from_slice_i32(&[7, 21], &[1, 2]);
    let (inds, weights) = gate.forward(&x, &ids);
    assert_eq!(mlxcel_core::array_shape(&inds), vec![1, 2, 2]);
    let got = array_to_vec_f32(&mlxcel_core::astype(&inds, mlxcel_core::dtype::FLOAT32));
    let expected: Vec<f32> = [7_i64, 21]
        .iter()
        .flat_map(|&t| {
            [
                table[(t * 2) as usize] as f32,
                table[(t * 2 + 1) as usize] as f32,
            ]
        })
        .collect();
    assert_close(&got, &expected, 1e-6, "hash indices come from the table");
    // Weights stay positive and finite (renormalised sqrtsoftplus).
    assert!(
        array_to_vec_f32(&weights)
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
    );
}

// ---------------------------------------------------------------------------
// HiSA: hierarchical selection must agree with the flat fallback.
// ---------------------------------------------------------------------------

fn indexer_fixture(args: &ModelArgs) -> WeightMap {
    let mut w = WeightMap::new();
    let hidden = args.hidden_size as i32;
    let q_lora = args.q_lora_rank as i32;
    let idx_heads = args.index_n_heads as i32;
    let idx_hd = args.index_head_dim as i32;
    put(&mut w, "ix.wq_b.weight", &[idx_heads * idx_hd, q_lora], 11);
    put(&mut w, "ix.weights_proj.weight", &[idx_heads, hidden], 12);
    put(
        &mut w,
        "ix.compressor.wkv.weight",
        &[2 * idx_hd, hidden],
        13,
    );
    put(
        &mut w,
        "ix.compressor.wgate.weight",
        &[2 * idx_hd, hidden],
        14,
    );
    put(&mut w, "ix.compressor.ape", &[4, 2 * idx_hd], 15);
    put_ones(&mut w, "ix.compressor.norm.weight", idx_hd);
    w
}

fn selection_rows(sel: &MlxArray, k: usize) -> Vec<Vec<i32>> {
    let v = array_to_vec_f32(&mlxcel_core::astype(sel, mlxcel_core::dtype::FLOAT32));
    v.chunks(k)
        .map(|c| {
            let mut r: Vec<i32> = c.iter().map(|f| *f as i32).collect();
            r.sort_unstable();
            r
        })
        .collect()
}

/// Strictly positive inputs: every q / pooled entry is in (0.75, 1.25) and
/// every head weight positive, so the ReLU is the identity, all scores are
/// unique with probability 1, and argpartition tie-breaking cannot mask a
/// real disagreement between the selection paths.
fn positive(n: usize, seed: u32) -> Vec<f32> {
    noise(n, seed).iter().map(|v| 1.0 + v).collect()
}

#[test]
fn hisa_selection_agrees_with_flat_fallback() {
    // block=2, keep=6 keeps every one of the 6 blocks of Np = 12 pooled
    // rows, so the hierarchy must select exactly what the flat path selects
    // on the same scores.
    let mut cfg: serde_json::Value = serde_json::from_str(tiny_args_json()).expect("parse");
    cfg["index_keep"] = serde_json::json!(6);
    let args: ModelArgs = serde_json::from_value(cfg).expect("parse");
    let args = args.normalized().expect("validate");
    let weights = indexer_fixture(&args);
    let ix = Indexer::from_weights(&weights, &args, "ix", 4).expect("indexer");

    let (h, l, d, np, k) = (2, 48, 8, 12, 4);
    let q = mlxcel_core::from_slice_f32(&positive((h * l * d) as usize, 41), &[1, h, l, d]);
    let pooled = mlxcel_core::from_slice_f32(&positive((np * d) as usize, 42), &[1, np, d]);
    let w = mlxcel_core::from_slice_f32(&positive((l * h) as usize, 43), &[1, l, h]);
    let counts = pool_visible_counts(l, 0, 4, np);

    let sel_h = ix.hisa_select_batched(&q, &pooled, &w, k, &counts);
    let sel_f = ix.flat_select(&q, &pooled, &w, k, Some(&counts));
    assert_eq!(mlxcel_core::array_shape(&sel_h), vec![1, l, k]);
    let rows_h = selection_rows(&sel_h, k as usize);
    let rows_f = selection_rows(&sel_f, k as usize);
    for j in 0..l as usize {
        // Rows with fewer than k visible pooled positions pick arbitrary
        // masked filler; the sparse mask hides those downstream, so parity is
        // only meaningful where at least k rows are visible.
        if counts[j] >= k {
            assert_eq!(
                rows_h[j], rows_f[j],
                "hierarchical and flat selection diverge at row {j}"
            );
            assert!(
                rows_h[j].iter().all(|&i| i < counts[j]),
                "row {j} selected an invisible pooled row"
            );
        }
    }

    // Decode fast path (L == 1, no mask): same selection as flat over the
    // full prefix.
    let q1 = mlxcel_core::from_slice_f32(&positive((h * d) as usize, 51), &[1, h, 1, d]);
    let w1 = mlxcel_core::from_slice_f32(&positive(h as usize, 52), &[1, 1, h]);
    let sel_h = ix.hisa_select_decode(&q1, &pooled, &w1, k);
    let sel_f = ix.flat_select(&q1, &pooled, &w1, k, None);
    assert_eq!(
        selection_rows(&sel_h, k as usize),
        selection_rows(&sel_f, k as usize),
        "decode fast path diverges from flat"
    );
}

#[test]
fn indexer_forward_dispatches_and_respects_visibility() {
    // End-to-end through the projections and pooling: 48 tokens at ratio 4
    // produce Np = 12 >= index_block * index_keep, so the batched HiSA path
    // runs; the selection must stay inside each row's visible prefix
    // wherever at least k rows are visible.
    let mut cfg: serde_json::Value = serde_json::from_str(tiny_args_json()).expect("parse");
    cfg["index_keep"] = serde_json::json!(6);
    let args: ModelArgs = serde_json::from_value(cfg).expect("parse");
    let args = args.normalized().expect("validate");
    let weights = indexer_fixture(&args);
    let ix = Indexer::from_weights(&weights, &args, "ix", 4).expect("indexer");
    let rope =
        V4Rope::new(4, 160000.0, args.rope_scaling.as_ref(), 1, &[(8, false)]).expect("rope");

    let l = 48;
    let x = mlxcel_core::from_slice_f32(&noise((l * 8) as usize, 21), &[1, l, 8]);
    let q_res = mlxcel_core::from_slice_f32(&noise((l * 6) as usize, 22), &[1, l, 6]);
    let mut pool = PoolingCache::new(4);
    let sel = ix
        .forward(&x, &q_res, &rope, &mut pool, 0)
        .expect("prefill selects");
    assert_eq!(mlxcel_core::array_shape(&sel), vec![1, l, 4]);
    let rows = selection_rows(&sel, 4);
    let counts = pool_visible_counts(l, 0, 4, 12);
    for (j, row) in rows.iter().enumerate() {
        if counts[j] >= 4 {
            assert!(
                row.iter().all(|&i| i >= 0 && i < counts[j]),
                "row {j} selected outside its visible prefix: {row:?}"
            );
        }
    }

    // A decode step continues from the same pooling cache.
    let x1 = mlxcel_core::from_slice_f32(&noise(8, 31), &[1, 1, 8]);
    let q1 = mlxcel_core::from_slice_f32(&noise(6, 32), &[1, 1, 6]);
    let sel = ix
        .forward(&x1, &q1, &rope, &mut pool, l)
        .expect("decode selects");
    assert_eq!(mlxcel_core::array_shape(&sel), vec![1, 1, 4]);
    let row = &selection_rows(&sel, 4)[0];
    assert!(row.iter().all(|&i| (0..12).contains(&i)));
}

// ---------------------------------------------------------------------------
// Sanitize and coverage.
// ---------------------------------------------------------------------------

#[test]
fn sanitize_maps_legacy_plane_onto_canonical_names() {
    let args = tiny_args();
    let mut legacy = WeightMap::new();
    put(&mut legacy, "embed.weight", &[32, 8], 1);
    put(&mut legacy, "norm.weight", &[8], 2);
    put(&mut legacy, "head.weight", &[32, 8], 3);
    put(&mut legacy, "hc_head_fn", &[2, 16], 4);
    put(&mut legacy, "layers.0.hc_attn_fn", &[8, 16], 5);
    put(&mut legacy, "layers.0.hc_ffn_scale", &[3], 6);
    put(&mut legacy, "layers.0.ffn.gate.bias", &[4], 7);
    put(
        &mut legacy,
        "layers.0.ffn.shared_experts.w1.weight",
        &[8, 8],
        8,
    );
    put(&mut legacy, "mtp.some.tensor", &[2], 9);
    put(&mut legacy, "layers.7.attn.wq_a.weight", &[6, 8], 10);
    for e in 0..4 {
        put(
            &mut legacy,
            &format!("layers.0.ffn.experts.{e}.w2.weight"),
            &[8, 8],
            20 + e,
        );
    }
    let out = sanitize::sanitize_weights(&legacy, &args).expect("sanitize");
    for key in [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.hc_head.fn",
        "model.layers.0.attn_hc.fn",
        "model.layers.0.ffn_hc.scale",
        "model.layers.0.ffn.gate.e_score_correction_bias",
        "model.layers.0.ffn.shared_experts.gate_proj.weight",
        "model.layers.0.ffn.switch_mlp.down_proj.weight",
    ] {
        assert!(out.contains_key(key), "missing canonical key {key}");
    }
    assert!(
        !out.keys().any(|k| k.starts_with("mtp.")),
        "mtp.* must be dropped"
    );
    assert!(
        !out.keys().any(|k| k.contains("layers.7")),
        "layers beyond num_hidden_layers must be dropped"
    );
    let stacked = out
        .get("model.layers.0.ffn.switch_mlp.down_proj.weight")
        .expect("stacked experts");
    assert_eq!(mlxcel_core::array_shape(stacked), vec![4, 8, 8]);
}

#[test]
fn sanitize_reshapes_wo_a_and_casts_tid2eid() {
    let args = tiny_args();
    let w = tiny_weight_map(&args);
    let out = sanitize::sanitize_weights(&w, &args).expect("sanitize");
    let wo_a = out.get("model.layers.0.attn.wo_a.weight").expect("wo_a");
    assert_eq!(mlxcel_core::array_shape(wo_a), vec![2, 4, 8]);
    let tid = out.get("model.layers.0.ffn.gate.tid2eid").expect("tid2eid");
    assert_eq!(mlxcel_core::array_dtype(tid), mlxcel_core::dtype::INT32);
}

#[test]
fn weight_coverage_rejects_missing_and_unknown_tensors() {
    let args = tiny_args();
    let full = tiny_weight_map(&args);
    let sanitized = sanitize::sanitize_weights(&full, &args).expect("sanitize");
    sanitize::validate_weight_coverage(&sanitized, &args).expect("tiny map covers the config");

    let mut missing = WeightMap::new();
    for (k, v) in sanitized.iter() {
        if k != "model.layers.2.attn.indexer.wq_b.weight" {
            missing.insert(k.clone(), mlxcel_core::copy(v));
        }
    }
    let err = sanitize::validate_weight_coverage(&missing, &args).unwrap_err();
    assert!(
        err.contains("model.layers.2.attn.indexer.wq_b.weight"),
        "error must name the missing tensor: {err}"
    );

    let mut unknown = WeightMap::new();
    for (k, v) in sanitized.iter() {
        unknown.insert(k.clone(), mlxcel_core::copy(v));
    }
    unknown.insert(
        "model.layers.0.attn.kv_b_proj.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0], &[1]),
    );
    let err = sanitize::validate_weight_coverage(&unknown, &args).unwrap_err();
    assert!(
        err.contains("kv_b_proj"),
        "error must name the unknown tensor: {err}"
    );
}

// ---------------------------------------------------------------------------
// End-to-end: a tiny synthetic model over all three attention kinds.
// ---------------------------------------------------------------------------

#[test]
fn tiny_model_prefill_and_decode_produce_finite_logits() {
    let args = tiny_args();
    let weights = tiny_weight_map(&args);
    let model = DeepSeekV4Model::from_weights(&weights, &args).expect("tiny model builds");

    // 26-token prefill: the ratio-4 layer accumulates Np = 6 > index_topk =
    // 4, driving the sparse split-softmax path and the batched HiSA
    // selection; the ratio-128 layer runs with an empty pool; layer 0 stays
    // local. Layer 0 routes by tid2eid (hash), the rest by bias.
    let prompt: Vec<i32> = (0..26).map(|v| v % 31).collect();
    let input = mlxcel_core::from_slice_i32(&prompt, &[1, 26]);
    let mut caches = model.make_internal_caches();
    let logits = model.forward_with_caches(&input, &mut caches);
    assert_eq!(mlxcel_core::array_shape(&logits), vec![1, 26, 32]);
    let v = array_to_vec_f32(&mlxcel_core::utils::slice_axis(&logits, 1, 25, 26));
    assert!(
        v.iter().all(|x| x.is_finite()),
        "prefill logits must be finite"
    );

    // Three decode steps continue from the same caches (rotating window,
    // pooling remainders, HiSA decode fast path once Np >= block * keep).
    for (step, &tok) in [3_i32, 14, 8].iter().enumerate() {
        let input = mlxcel_core::from_slice_i32(&[tok], &[1, 1]);
        let logits = model.forward_with_caches(&input, &mut caches);
        assert_eq!(mlxcel_core::array_shape(&logits), vec![1, 1, 32]);
        let v = array_to_vec_f32(&logits);
        assert!(
            v.iter().all(|x| x.is_finite()),
            "decode step {step} logits must be finite"
        );
    }

    // The local rotating caches advanced in lockstep across all layers.
    for cache in &caches {
        assert_eq!(cache.local.offset, 29);
    }
    assert!(caches[0].pool.is_none() && caches[0].idx_pool.is_none());
    assert!(caches[1].pool.is_some() && caches[1].idx_pool.is_none());
    assert!(caches[2].pool.is_some() && caches[2].idx_pool.is_some());
}

#[test]
fn tiny_model_chunked_prefill_matches_single_pass() {
    let args = tiny_args();
    let weights = tiny_weight_map(&args);
    let model = DeepSeekV4Model::from_weights(&weights, &args).expect("tiny model builds");

    // 18 tokens keep the sparse layer's Np at 4 <= index_topk, so both the
    // single pass and the chunked pass stay on the DENSE local+pooled path
    // (above index_topk the top-k indexer can legitimately tie-break exact
    // zero ReLU scores differently between differently-shaped calls). The
    // 16 + 2 split also keeps every completed overlap window inside one
    // call: the reference's `_overlap_compress_kv` shifts the first feature
    // half one window back WITHIN the ready batch, so a ratio-4 window
    // completed at a chunk boundary sees a zero/-inf overlap prefix instead
    // of its predecessor. That is reference behavior, not a chunking bug
    // here; see the deepseek_v4_compress module docs.
    let prompt: Vec<i32> = (0..18).map(|v| (v * 3 + 1) % 31).collect();

    let input = mlxcel_core::from_slice_i32(&prompt, &[1, 18]);
    let mut caches = model.make_internal_caches();
    let full = model.forward_with_caches(&input, &mut caches);
    let full_last = array_to_vec_f32(&mlxcel_core::utils::slice_axis(&full, 1, 17, 18));

    // Same prompt in two chunks (16 + 2) through fresh caches: the pooling
    // remainder path and the rotating continuation append must reproduce the
    // single-pass last-position logits.
    let mut caches = model.make_internal_caches();
    let chunk_a = mlxcel_core::from_slice_i32(&prompt[..16], &[1, 16]);
    let _ = model.forward_with_caches(&chunk_a, &mut caches);
    let chunk_b = mlxcel_core::from_slice_i32(&prompt[16..], &[1, 2]);
    let chunked = model.forward_with_caches(&chunk_b, &mut caches);
    let chunked_last = array_to_vec_f32(&mlxcel_core::utils::slice_axis(&chunked, 1, 1, 2));

    assert_close(&chunked_last, &full_last, 5e-3, "chunked prefill parity");
}

#[test]
fn real_checkpoint_config_shape_parses_with_both_quantization_keys() {
    // The real DeepSeek-V4-Flash-4bit config.json carries BOTH `quantization`
    // and `quantization_config` (identical blocks). serde must accept that
    // rather than erroring on a duplicate field, and the 44-entry
    // compress_ratios list (one extra for the dropped MTP layer) must be
    // truncated to num_hidden_layers before validation, as the reference
    // __post_init__ does.
    let mut cfg: serde_json::Value = serde_json::from_str(tiny_args_json()).expect("parse");
    let quant = serde_json::json!({
        "group_size": 64, "bits": 4, "mode": "affine",
        "model.layers.0.ffn.switch_mlp.gate_proj":
            {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    });
    cfg["quantization"] = quant.clone();
    cfg["quantization_config"] = quant;
    cfg["compress_ratios"] = serde_json::json!([0, 128, 4, 0]);
    let args: ModelArgs =
        serde_json::from_value(cfg).expect("config with both quantization keys must parse");
    let args = args
        .normalized()
        .expect("over-length compress_ratios must truncate to num_hidden_layers");
    assert_eq!(args.compress_ratios, vec![0, 128, 4]);
    let (gs, bits, mode) = args.expert_quantization("model.layers.0.ffn.switch_mlp.gate_proj");
    assert_eq!((gs, bits, mode.as_str()), (32, 4, "mxfp4"));
}

#[test]
fn eos_token_id_parses_from_config() {
    let args = tiny_args();
    assert_eq!(args.eos_token_ids(), vec![1]);
}
