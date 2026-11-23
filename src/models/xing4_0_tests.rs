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

//! Xing4.0 hyper connection, sanitize, config and model tests (issue #2356).

use super::*;
use crate::models::deepseek_v3::tests::{
    ramp, test_config_dense_direct_q, tiny_dense_model_weights,
};
use mlxcel_core::utils::array_to_vec_f32;

// ---- Scalar reference of the hyper connection ------------------------------

/// The variants a port could silently fall into. `Xing` is the reference;
/// every other arm is the DeepSeek-V4 behaviour (or an ordering slip) for one
/// item of the issue's list.
#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Xing,
    /// Item 3: `pre = sigmoid(..) + hc_eps` (DeepSeek-V4).
    PreEps,
    /// Item 5: softmax rows, `+ hc_eps`, then the same row/column count
    /// (DeepSeek-V4's initial column pass plus `iters - 1` rounds).
    SoftmaxSinkhorn,
    /// Item 5: column normalization before row normalization.
    ColumnFirst,
}

struct RefOut {
    post: Vec<f64>,
    comb: Vec<Vec<f64>>,
    collapsed: Vec<f64>,
}

struct RefParams {
    hc: usize,
    d: usize,
    hc_fn: Vec<f32>,
    base: Vec<f32>,
    scale: Vec<f32>,
    iters: usize,
    hc_eps: f64,
    norm_eps: f64,
    clamp: (f64, f64),
}

fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

/// `x` is `[hc][d]` for one position.
fn reference(p: &RefParams, x: &[Vec<f64>], variant: Variant) -> RefOut {
    let (hc, d) = (p.hc, p.d);
    let flat: Vec<f64> = x.iter().flatten().copied().collect();
    let ms = flat.iter().map(|v| v * v).sum::<f64>() / flat.len() as f64;
    let inv = 1.0 / (ms + p.norm_eps).sqrt();
    let normed: Vec<f64> = flat.iter().map(|v| v * inv).collect();
    let mix_len = (2 + hc) * hc;
    let mix: Vec<f64> = (0..mix_len)
        .map(|r| {
            (0..hc * d)
                .map(|c| p.hc_fn[r * hc * d + c] as f64 * normed[c])
                .sum()
        })
        .collect();
    let s = |i: usize| p.scale[i] as f64;
    let b = |i: usize| p.base[i] as f64;
    let pre: Vec<f64> = (0..hc)
        .map(|i| {
            let v = sigmoid(mix[i] * s(0) + b(i));
            if variant == Variant::PreEps {
                v + p.hc_eps
            } else {
                v
            }
        })
        .collect();
    let post: Vec<f64> = (0..hc)
        .map(|i| 2.0 * sigmoid(mix[hc + i] * s(1) + b(hc + i)))
        .collect();
    let mut comb: Vec<Vec<f64>> = (0..hc)
        .map(|r| {
            (0..hc)
                .map(|c| {
                    let k = 2 * hc + r * hc + c;
                    (mix[k] * s(2) + b(k)).clamp(p.clamp.0, p.clamp.1)
                })
                .collect()
        })
        .collect();
    let row_norm = |m: &mut Vec<Vec<f64>>| {
        for row in m.iter_mut() {
            let sum: f64 = row.iter().sum();
            row.iter_mut().for_each(|v| *v /= sum + p.hc_eps);
        }
    };
    let col_norm = |m: &mut Vec<Vec<f64>>| {
        let sums: Vec<f64> = (0..hc).map(|c| m.iter().map(|row| row[c]).sum()).collect();
        for row in m.iter_mut() {
            for (v, sum) in row.iter_mut().zip(&sums) {
                *v /= sum + p.hc_eps;
            }
        }
    };
    for row in comb.iter_mut() {
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        row.iter_mut().for_each(|v| *v = (*v - max).exp());
    }
    match variant {
        Variant::SoftmaxSinkhorn => {
            for row in comb.iter_mut() {
                let sum: f64 = row.iter().sum();
                row.iter_mut().for_each(|v| *v = *v / sum + p.hc_eps);
            }
            col_norm(&mut comb);
            for _ in 0..p.iters.saturating_sub(1) {
                row_norm(&mut comb);
                col_norm(&mut comb);
            }
        }
        Variant::ColumnFirst => {
            for _ in 0..p.iters {
                col_norm(&mut comb);
                row_norm(&mut comb);
            }
        }
        _ => {
            for _ in 0..p.iters {
                row_norm(&mut comb);
                col_norm(&mut comb);
            }
        }
    }
    let collapsed: Vec<f64> = (0..d)
        .map(|j| (0..hc).map(|i| pre[i] * x[i][j]).sum())
        .collect();
    RefOut {
        post,
        comb,
        collapsed,
    }
}

fn max_abs_diff(a: &[f64], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - *y as f64).abs())
        .fold(0.0, f64::max)
}

const HC: usize = 2;
const D: usize = 3;

fn hc_test_config(iters: usize, hc_eps: f32) -> Xing40Config {
    let mut base = test_config_dense_direct_q();
    base.hidden_size = D;
    Xing40Config {
        base,
        hc_mult: HC,
        hc_sinkhorn_iters: iters,
        hc_eps,
        mhc_h_res_clamp_min: -30.0,
        mhc_h_res_clamp_max: 30.0,
        eos_token_ids: vec![2],
    }
}

/// Hand-picked weights: `hc_fn` from the shared ramp scaled up so the gates
/// leave the linear region, a non-trivial base and three distinct scales.
fn hc_weights() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mix = (2 + HC) * HC;
    let hc_fn: Vec<f32> = ramp(mix * HC * D).iter().map(|v| v * 4.0).collect();
    let base: Vec<f32> = (0..mix).map(|i| 0.1 * i as f32 - 0.3).collect();
    let scale = vec![0.7, 1.3, 2.1];
    (hc_fn, base, scale)
}

fn hc_module(config: &Xing40Config) -> XingHyperConnection {
    let (hc_fn, base, scale) = hc_weights();
    let mix = ((2 + HC) * HC) as i32;
    let mut weights = WeightMap::new();
    weights.insert(
        "hc.hc_fn".into(),
        mlxcel_core::from_slice_f32(&hc_fn, &[mix, (HC * D) as i32]),
    );
    weights.insert(
        "hc.hc_base".into(),
        mlxcel_core::from_slice_f32(&base, &[mix]),
    );
    weights.insert(
        "hc.hc_scale".into(),
        mlxcel_core::from_slice_f32(&scale, &[3]),
    );
    XingHyperConnection::from_weights(&weights, "hc", config).expect("tiny hc must load")
}

fn hc_input() -> (Vec<Vec<f64>>, UniquePtr<MlxArray>) {
    let rows = vec![vec![0.9, -0.4, 1.7], vec![-1.2, 0.6, 0.3]];
    let data: Vec<f32> = rows.iter().flatten().map(|v| *v as f32).collect();
    let arr = mlxcel_core::from_slice_f32(&data, &[1, 1, HC as i32, D as i32]);
    (rows, arr)
}

fn run_case(iters: usize, hc_eps: f32) -> (RefParams, Vec<f32>, Vec<f32>, Vec<f32>) {
    let config = hc_test_config(iters, hc_eps);
    let module = hc_module(&config);
    let (_, x) = hc_input();
    let (post, comb, collapsed) = module.forward(&x);
    let (hc_fn, base, scale) = hc_weights();
    let params = RefParams {
        hc: HC,
        d: D,
        hc_fn,
        base,
        scale,
        iters,
        hc_eps: hc_eps as f64,
        norm_eps: config.base.rms_norm_eps as f64,
        clamp: (-30.0, 30.0),
    };
    (
        params,
        array_to_vec_f32(&post),
        array_to_vec_f32(&comb),
        array_to_vec_f32(&collapsed),
    )
}

/// The connection matches the scalar reference, and each DeepSeek-V4 variant
/// of items 3 and 5 is far from what it computes. A large `hc_eps` and a short
/// Sinkhorn make the eps placement and the normalization order visible.
#[test]
fn hyper_connection_matches_the_xing_reference_and_not_the_v4_variants() {
    let (params, post, comb, collapsed) = run_case(2, 0.05);
    let (x, _) = hc_input();
    let want = reference(&params, &x, Variant::Xing);
    let flat_comb = |r: &RefOut| r.comb.iter().flatten().copied().collect::<Vec<f64>>();

    assert!(
        max_abs_diff(&want.post, &post) < 1e-5,
        "post {post:?} vs {:?}",
        want.post
    );
    assert!(
        max_abs_diff(&flat_comb(&want), &comb) < 1e-5,
        "comb {comb:?}"
    );
    assert!(
        max_abs_diff(&want.collapsed, &collapsed) < 1e-5,
        "collapsed {collapsed:?} vs {:?}",
        want.collapsed
    );

    let pre_eps = reference(&params, &x, Variant::PreEps);
    assert!(
        max_abs_diff(&pre_eps.collapsed, &collapsed) > 1e-2,
        "item 3: the V4 `+ hc_eps` on pre must be absent"
    );
    for variant in [Variant::SoftmaxSinkhorn, Variant::ColumnFirst] {
        let other = reference(&params, &x, variant);
        assert!(
            max_abs_diff(&flat_comb(&other), &comb) > 1e-3,
            "item 5: a V4-style or column-first Sinkhorn must not match"
        );
    }
}

/// At the released 20 rounds and `hc_eps` 1e-6 the connection still matches
/// the reference, and `comb` is doubly stochastic.
#[test]
fn hyper_connection_matches_the_reference_at_released_settings() {
    let (params, post, comb, collapsed) = run_case(20, 1e-6);
    let (x, _) = hc_input();
    let want = reference(&params, &x, Variant::Xing);
    assert!(max_abs_diff(&want.post, &post) < 1e-5);
    assert!(max_abs_diff(&want.collapsed, &collapsed) < 1e-5);
    for r in 0..HC {
        let row: f32 = (0..HC).map(|c| comb[r * HC + c]).sum();
        let col: f32 = (0..HC).map(|c| comb[c * HC + r]).sum();
        assert!((row - 1.0).abs() < 1e-3 && (col - 1.0).abs() < 1e-3);
    }
}

/// Item 7: the expand is `post * out + comb @ x`, checked on a non-symmetric
/// `comb` against hand-computed values; `comb.T @ x` gives different ones.
#[test]
fn expand_uses_comb_times_x_not_its_transpose() {
    let comb = [[0.8f32, 0.2], [0.35, 0.65]];
    let x = [[1.0f32, 2.0], [-3.0, 0.5]];
    let post = [1.5f32, 0.25];
    let out = [0.4f32, -0.6];

    let got = hc_expand(
        &mlxcel_core::from_slice_f32(&out, &[1, 1, 2]),
        &mlxcel_core::from_slice_f32(&x.concat(), &[1, 1, 2, 2]),
        &mlxcel_core::from_slice_f32(&post, &[1, 1, 2]),
        &mlxcel_core::from_slice_f32(&comb.concat(), &[1, 1, 2, 2]),
    );
    let got = array_to_vec_f32(&got);

    let expand = |transpose: bool| -> Vec<f64> {
        let mut v = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                let mixed: f64 = (0..2)
                    .map(|k| {
                        let c = if transpose { comb[k][i] } else { comb[i][k] };
                        c as f64 * x[k][j] as f64
                    })
                    .sum();
                v.push(post[i] as f64 * out[j] as f64 + mixed);
            }
        }
        v
    };
    // Hand-computed: row 0 = 1.5*[0.4,-0.6] + 0.8*[1,2] + 0.2*[-3,0.5].
    assert!(max_abs_diff(&[0.8, 0.8], &got[..2]) < 1e-6, "{got:?}");
    assert!(max_abs_diff(&expand(false), &got) < 1e-6);
    assert!(max_abs_diff(&expand(true), &got) > 1e-1);
}

/// Item 8: the head is the plain mean over the streams, with no weights.
#[test]
fn head_collapses_by_mean() {
    let x = mlxcel_core::from_slice_f32(&[1.0, 3.0, -2.0, 5.0], &[1, 1, 2, 2]);
    let got = array_to_vec_f32(&hc_head(&x));
    assert_eq!(got, vec![-0.5, 4.0]);
}

/// Item 9: embeddings enter repeated across the streams.
#[test]
fn widen_repeats_each_row_hc_times() {
    let h = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let w = hc_widen(&h, 3);
    assert_eq!(mlxcel_core::array_shape(&w), vec![1, 2, 3, 2]);
    assert_eq!(
        array_to_vec_f32(&w),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
    );
}

// ---- Config ------------------------------------------------------------

fn config_json(extra: serde_json::Value) -> serde_json::Value {
    let mut v = serde_json::json!({
        "model_type": "xing4_0",
        "vocab_size": 11, "hidden_size": 8, "intermediate_size": 8,
        "moe_intermediate_size": 8, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 2,
        "kv_lora_rank": 8, "q_lora_rank": null, "qk_rope_head_dim": 4,
        "v_head_dim": 4, "qk_nope_head_dim": 4,
        "hc_mult": 2, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
        "mhc_h_res_clamp_min": -30, "mhc_h_res_clamp_max": 30,
        "eos_token_id": 2
    });
    for (k, val) in extra.as_object().unwrap() {
        v[k] = val.clone();
    }
    v
}

#[test]
fn config_reads_the_hyper_fields_and_eos() {
    let cfg: Xing40Config = serde_json::from_value(config_json(serde_json::json!({
        "hc_mult": 4, "hc_sinkhorn_iters": 7, "eos_token_id": [2, 5]
    })))
    .unwrap();
    assert_eq!(cfg.hc_mult, 4);
    assert_eq!(cfg.hc_sinkhorn_iters, 7);
    assert_eq!(cfg.eos_token_ids, vec![2, 5]);
    assert_eq!(cfg.base.hidden_size, 8);
    assert_eq!(cfg.mhc_h_res_clamp_min, -30.0);
}

#[test]
fn config_refuses_eos_ids_outside_the_i32_range() {
    // Narrowing with `as i32` wrapped 4294967298 to 2 and accepted -1 (#3299).
    for (eos, needle) in [
        (serde_json::json!(4_294_967_298_u64), "4294967298"),
        (serde_json::json!([-1]), "-1"),
        (serde_json::json!([2, 2_147_483_648_u64]), "2147483648"),
    ] {
        let err = serde_json::from_value::<Xing40Config>(config_json(
            serde_json::json!({ "eos_token_id": eos }),
        ))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("eos_token_id") && err.contains(needle),
            "{err}"
        );
    }
    let mut cfg = hc_test_config(20, 1e-6);
    assert!(cfg.validate().is_ok());
    cfg.eos_token_ids = vec![2, -1];
    assert!(cfg.validate().unwrap_err().contains("eos_token_id -1"));
    let max: Xing40Config = serde_json::from_value(config_json(
        serde_json::json!({ "eos_token_id": [0, 2_147_483_647] }),
    ))
    .unwrap();
    assert_eq!(max.eos_token_ids, vec![0, i32::MAX]);
}

#[test]
fn config_refuses_unusable_hyper_settings() {
    for (patch, needle) in [
        (serde_json::json!({"hc_mult": 0}), "hc_mult"),
        (serde_json::json!({"hc_mult": 65}), "hc_mult"),
        (
            serde_json::json!({"hc_sinkhorn_iters": 100000}),
            "hc_sinkhorn_iters",
        ),
        (serde_json::json!({"hc_eps": -1.0}), "hc_eps"),
        (
            serde_json::json!({"mhc_h_res_clamp_min": 5, "mhc_h_res_clamp_max": -5}),
            "mhc_h_res_clamp",
        ),
    ] {
        let err = serde_json::from_value::<Xing40Config>(config_json(patch))
            .expect_err("must refuse")
            .to_string();
        assert!(err.contains(needle), "{err}");
    }
}

// ---- Weights, sanitize, model -----------------------------------------------

fn tiny_config(rope_scaling: Option<serde_json::Value>) -> Xing40Config {
    let mut base = test_config_dense_direct_q();
    base.model_type = "xing4_0".into();
    base.rope_scaling = rope_scaling.map(|v| serde_json::from_value(v).unwrap());
    Xing40Config {
        base,
        hc_mult: 2,
        hc_sinkhorn_iters: 20,
        hc_eps: 1e-6,
        mhc_h_res_clamp_min: -30.0,
        mhc_h_res_clamp_max: 30.0,
        eos_token_ids: vec![2],
    }
}

fn insert_hc(weights: &mut WeightMap, prefix: &str, cfg: &Xing40Config, layer: usize) {
    let hc = cfg.hc_mult as i32;
    let hidden = cfg.base.hidden_size as i32;
    let mix = (2 + hc) * hc;
    let n = (mix * hc * hidden) as usize;
    let fn_data: Vec<f32> = ramp(n + layer).into_iter().skip(layer).collect();
    weights.insert(
        format!("{prefix}.hc_fn"),
        mlxcel_core::from_slice_f32(&fn_data, &[mix, hc * hidden]),
    );
    let base: Vec<f32> = (0..mix).map(|i| 0.05 * i as f32 - 0.2).collect();
    weights.insert(
        format!("{prefix}.hc_base"),
        mlxcel_core::from_slice_f32(&base, &[mix]),
    );
    weights.insert(
        format!("{prefix}.hc_scale"),
        mlxcel_core::from_slice_f32(&[0.5, 1.0, 1.5], &[3]),
    );
}

fn tiny_weights(cfg: &Xing40Config) -> WeightMap {
    let mut weights = tiny_dense_model_weights(&cfg.base);
    for l in 0..cfg.base.num_hidden_layers {
        insert_hc(&mut weights, &format!("model.layers.{l}.attn_hc"), cfg, l);
        insert_hc(&mut weights, &format!("model.layers.{l}.ffn_hc"), cfg, l);
    }
    weights
}

fn tiny_model(cfg: &Xing40Config) -> Xing40Model {
    let weights = Xing40Model::sanitize_weights(tiny_weights(cfg), cfg).expect("sanitize");
    Xing40Model::from_weights(&weights, cfg).expect("tiny model must load")
}

/// The mlx-vlm `language_model.` root is mapped onto the text layout and the
/// MTP layer `model.layers.{num_hidden_layers}.*` is removed (item 10).
#[test]
fn sanitize_strips_the_wrapper_and_the_mtp_layer() {
    let cfg = tiny_config(None);
    let mut raw = tiny_weights(&cfg);
    let n = cfg.base.num_hidden_layers;
    crate::models::deepseek_v3::tests::insert_dense_layer_weights(&mut raw, &cfg.base, n);
    insert_hc(&mut raw, &format!("model.layers.{n}.attn_hc"), &cfg, n);
    let mut wrapped = WeightMap::new();
    for (k, v) in raw.iter() {
        wrapped.insert(format!("language_model.{k}"), mlxcel_core::copy(v));
    }

    let out = Xing40Model::sanitize_weights(wrapped, &cfg).expect("sanitize");
    assert!(!out.keys().any(|k| k.starts_with("language_model.")));
    assert!(
        !out.keys()
            .any(|k| k.starts_with(&format!("model.layers.{n}."))),
        "MTP layer {n} must be dropped"
    );
    for l in 0..n {
        assert!(out.contains_key(&format!("model.layers.{l}.self_attn.embed_q.weight")));
        assert!(out.contains_key(&format!("model.layers.{l}.attn_hc.hc_fn")));
    }
    assert!(out.contains_key("lm_head.weight"));
    Xing40Model::from_weights(&out, &cfg).expect("sanitized map must load");
}

#[test]
fn load_refuses_an_hc_fn_of_the_wrong_shape() {
    let cfg = tiny_config(None);
    let mut weights = Xing40Model::sanitize_weights(tiny_weights(&cfg), &cfg).unwrap();
    weights.insert(
        "model.layers.1.ffn_hc.hc_fn".into(),
        mlxcel_core::from_slice_f32(&ramp(8 * 8), &[8, 8]),
    );
    let err = Xing40Model::from_weights(&weights, &cfg)
        .err()
        .expect("must refuse");
    assert!(err.contains("model.layers.1.ffn_hc.hc_fn"), "{err}");
}

#[test]
fn tiny_model_forward_is_finite_and_last_row_matches() {
    let cfg = tiny_config(None);
    let model = tiny_model(&cfg);
    assert_eq!(model.layers.len(), cfg.base.num_hidden_layers);
    let input = mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[1, 4]);

    let mut caches = model.make_caches_impl();
    let full = model.forward_impl(&input, &mut caches, None);
    assert_eq!(mlxcel_core::array_shape(&full), vec![1, 4, 11]);
    let full = array_to_vec_f32(&full);
    assert!(full.iter().all(|v| v.is_finite()));

    let mut caches = model.make_caches_impl();
    let last = model
        .last_row_logits(&input, None, None, &mut caches, None, 3)
        .expect("text path projects the last row");
    let last = array_to_vec_f32(&last);
    let diff = last
        .iter()
        .zip(&full[3 * 11..])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(diff < 1e-5, "last row diverges by {diff}");
    assert_eq!(model.eos_token_ids(), vec![2]);
}

#[test]
fn tiny_model_batched_decode_matches_sequential() {
    let _runtime = crate::initialize_runtime();
    let model = tiny_model(&tiny_config(None));
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        &model,
        &[&[1], &[2, 3], &[4, 5, 6]],
        &[7, 8, 9],
        1e-5,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
        &model,
        &[1, 2, 3],
        4,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(&model);
}

fn yarn_block() -> serde_json::Value {
    serde_json::json!({
        "type": "yarn", "factor": 64, "original_max_position_embeddings": 4,
        "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0
    })
}

/// YaRN installs a table on every attention, changes the logits, and the
/// batched decode path uses the same table as the single-sequence one.
#[test]
fn yarn_rope_scaling_installs_a_table_on_every_attention() {
    let _runtime = crate::initialize_runtime();
    let plain = tiny_model(&tiny_config(None));
    assert!(
        plain
            .layers
            .iter()
            .all(|l| l.self_attn.rope_table.is_none())
    );

    let mut cfg = tiny_config(Some(yarn_block()));
    cfg.base.max_position_embeddings = 256;
    let yarn = tiny_model(&cfg);
    assert!(yarn.layers.iter().all(|l| l.self_attn.rope_table.is_some()));

    let input = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5, 6, 7, 8], &[1, 8]);
    let a = array_to_vec_f32(&plain.forward_impl(&input, &mut plain.make_caches_impl(), None));
    let b = array_to_vec_f32(&yarn.forward_impl(&input, &mut yarn.make_caches_impl(), None));
    let diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(diff > 1e-4, "the YaRN table must change the rotation");

    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        &yarn,
        &[&[1], &[2, 3], &[4, 5, 6]],
        &[7, 8, 9],
        1e-5,
    );
}

/// The table path reproduces `fast_rope` when handed the plain
/// `base^(2i/d)` frequencies, so the DeepSeek-V3 attention's opt-in table
/// changes nothing but the frequencies.
#[test]
fn plain_frequency_table_reproduces_the_default_rope() {
    let cfg = tiny_config(None);
    let weights = Xing40Model::sanitize_weights(tiny_weights(&cfg), &cfg).unwrap();
    let make = || {
        DeepSeekV3Attention::from_weights(&weights, &cfg.base, "model.layers.0.self_attn")
            .expect("attention")
    };
    let dims = cfg.base.qk_rope_head_dim;
    let freqs: Vec<f32> = (0..dims / 2)
        .map(|i| cfg.base.rope_theta.powf((2 * i) as f32 / dims as f32))
        .collect();
    let tabled = make().with_rope_table(
        mlxcel_core::from_slice_f32(&freqs, &[(dims / 2) as i32]),
        1.0,
    );
    let default = make();
    let x = mlxcel_core::from_slice_f32(&ramp(5 * cfg.base.hidden_size), &[1, 5, 8]);
    let a = array_to_vec_f32(&default.forward(&x, &mut KVCache::new(), None));
    let b = array_to_vec_f32(&tabled.forward(&x, &mut KVCache::new(), None));
    let diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(diff < 1e-5, "table path diverges by {diff}");
}

/// The OptiQ export stores the MLA `embed_q` at 8 bits under a 4-bit
/// top-level block. The multi-linear loader solves the width from the shapes
/// instead of storing the declared 4, which `quantized_matmul` aborts on.
#[test]
fn mla_multi_linear_takes_a_per_path_width_from_the_shapes() {
    use mlxcel_core::layers::MultiLinear;
    let shape = [2, 8, 64];
    let w = mlxcel_core::from_slice_f32(&ramp(2 * 8 * 64), &shape);
    let mut weights = WeightMap::new();
    weights.insert(
        "p.weight".into(),
        mlxcel_core::quantize_weights_w(&w, 64, 8),
    );
    weights.insert(
        "p.scales".into(),
        mlxcel_core::quantize_weights_scales(&w, 64, 8),
    );
    weights.insert(
        "p.biases".into(),
        mlxcel_core::quantize_weights_biases(&w, 64, 8),
    );
    let layer = MultiLinear::from_weights(&weights, "p", 64, 4).expect("per-path width loads");
    match &layer {
        MultiLinear::Quantized(q) => assert_eq!(q.bits, 8),
        MultiLinear::Regular(_) => panic!("must stay quantized"),
    }
    let x = mlxcel_core::from_slice_f32(&ramp(2 * 3 * 64), &[2, 3, 64]);
    let got = array_to_vec_f32(&layer.forward(&x));
    let want = array_to_vec_f32(&mlxcel_core::matmul(
        &x,
        &mlxcel_core::transpose_axes(&w, &[0, 2, 1]),
    ));
    let diff = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff < 2e-2,
        "8-bit plane decodes within quantization error ({diff})"
    );
}

/// The OptiQ export also stores some routed-expert planes at 8 bits under the
/// 4-bit default. Each plane's width is solved from its own shapes, so
/// `gather_qmm` sees a consistent triple per projection.
#[test]
fn routed_expert_planes_take_per_path_widths_from_the_shapes() {
    let cfg = test_config_dense_direct_q();
    let (e, hidden, inter) = (2, 64, 64);
    let mut weights = WeightMap::new();
    for (proj, shape, bits) in [
        ("gate_proj", [e, inter, hidden], 4),
        ("up_proj", [e, inter, hidden], 4),
        ("down_proj", [e, hidden, inter], 8),
    ] {
        let w = mlxcel_core::from_slice_f32(&ramp((e * inter * hidden) as usize), &shape);
        weights.insert(
            format!("p.{proj}.weight"),
            mlxcel_core::quantize_weights_w(&w, 64, bits),
        );
        weights.insert(
            format!("p.{proj}.scales"),
            mlxcel_core::quantize_weights_scales(&w, 64, bits),
        );
        weights.insert(
            format!("p.{proj}.biases"),
            mlxcel_core::quantize_weights_biases(&w, 64, bits),
        );
    }
    let glu = crate::models::deepseek_v3::SwitchGLU::from_weights(&weights, &cfg, "p")
        .expect("mixed-width expert planes load");
    assert_eq!((glu.bits, glu.up_bits, glu.down_bits), (4, 4, 8));
    let x = mlxcel_core::from_slice_f32(&ramp(3 * hidden as usize), &[3, hidden]);
    let idx = mlxcel_core::from_slice_i32(&[0, 1, 1, 0, 0, 1], &[3, 2]);
    let out = array_to_vec_f32(&glu.forward(&x, &idx));
    assert_eq!(out.len(), 3 * 2 * hidden as usize);
    assert!(out.iter().all(|v| v.is_finite()));
}

/// A hostile expert plane must be refused at load, not handed to `gather_qmm`:
/// shapes that solve to 1 bit (no `gather_qmm` kernel) or to no integer width
/// at all are both load errors naming the plane.
#[test]
fn routed_expert_planes_with_unrunnable_widths_are_refused() {
    use crate::models::switch_layers::insert_stacked_quantized_expert_plane;
    let cfg = test_config_dense_direct_q();
    // gs 64, in 64: an honest 4-bit plane packs 8 words into 1 group.
    for (down_packed_in, needle) in [(2, "1-bit"), (3, "down_proj")] {
        let mut weights = WeightMap::new();
        insert_stacked_quantized_expert_plane(&mut weights, "p.gate_proj", 2, 64, 8, 1);
        insert_stacked_quantized_expert_plane(&mut weights, "p.up_proj", 2, 64, 8, 1);
        insert_stacked_quantized_expert_plane(
            &mut weights,
            "p.down_proj",
            2,
            64,
            down_packed_in,
            1,
        );
        let err = crate::models::deepseek_v3::SwitchGLU::from_weights(&weights, &cfg, "p")
            .err()
            .expect("an unrunnable expert plane must be refused at load");
        assert!(err.contains("p.down_proj"), "{err}");
        assert!(err.contains(needle), "{err}");
    }
}
