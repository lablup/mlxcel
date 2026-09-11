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

//! Checkpoint-free tests for the Laguna loader: per-layer RoPE resolution,
//! router math, the per-head output gate, the compressed-tensors NVFP4
//! sanitize, the pre-stacked `gate_up_proj` split, and prefill causality on
//! both layer types.

use super::laguna::{FULL_ATTENTION, LagunaModel, ModelArgs, RouterScoreFunc, SLIDING_ATTENTION};
use super::laguna_layers::LagunaCache;
use super::laguna_layers::{Attention, GateMode, SparseMoeBlock, router_scores, router_select};
use super::laguna_sanitize::sanitize_weights;
use super::sanitize::f32_to_f8_e4m3;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// `poolside/Laguna-XS-2.1` config trimmed to the fields the loader reads.
const XS_2_1_CONFIG: &str = r#"{
    "model_type": "laguna",
    "architectures": ["LagunaForCausalLM"],
    "vocab_size": 100352,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 8,
    "num_attention_heads": 48,
    "num_attention_heads_per_layer": [48, 64, 64, 64, 48, 64, 64, 64],
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
    "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
                    "full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    "sliding_window": 512,
    "rope_parameters": {
        "full_attention": {"rope_theta": 500000.0, "rope_type": "yarn", "factor": 32.0,
                           "original_max_position_embeddings": 8192, "beta_slow": 1.0, "beta_fast": 64.0,
                           "attention_factor": 1.3465735902799727, "partial_rotary_factor": 0.5},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0}
    },
    "gating": "per-head",
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "norm_topk_prob": true,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [0],
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse"],
    "moe_routed_scaling_factor": 2.5,
    "moe_router_logit_softcapping": 0.0,
    "tie_word_embeddings": false,
    "bos_token_id": 2,
    "eos_token_id": [2, 24],
    "quantization_config": {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
        "config_groups": {"group_0": {"format": "nvfp4-pack-quantized",
            "weights": {"num_bits": 4, "type": "float", "group_size": 16}}}}
}"#;

fn xs_config() -> ModelArgs {
    serde_json::from_str(XS_2_1_CONFIG).expect("parse laguna config")
}

fn f32_array(data: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(data, shape)
}

fn to_vec(arr: &MlxArray) -> Vec<f32> {
    let f32_arr = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f32_arr);
    mlxcel_core::utils::array_to_vec_f32(&f32_arr)
}

fn to_u32(arr: &MlxArray) -> Vec<u32> {
    mlxcel_core::eval(arr);
    mlxcel_core::array_to_raw_bytes(arr)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn laguna_config_parses_and_resolves_schedules() {
    let args = xs_config();
    assert_eq!(args.head_dim(), 128);
    assert_eq!(args.num_heads_for_layer(0).unwrap(), 48);
    assert_eq!(args.num_heads_for_layer(1).unwrap(), 64);
    assert!(!args.is_sliding(0));
    assert!(args.is_sliding(1));
    assert!(!args.is_moe_layer(0));
    assert!(args.is_moe_layer(1));
    assert!(args.declares_compressed_tensors_nvfp4());
    assert_eq!(args.quant_spec(false), super::laguna::QuantSpec::NVFP4);
    assert_eq!(args.router_score_func().unwrap(), RouterScoreFunc::Sigmoid);
    assert_eq!(GateMode::from_config(&args.gating), GateMode::PerHead);
    assert_eq!(
        GateMode::from_config(&serde_json::json!(true)),
        GateMode::PerHead
    );
    assert_eq!(
        GateMode::from_config(&serde_json::json!("per-element")),
        GateMode::PerElement
    );
    assert_eq!(
        GateMode::from_config(&serde_json::json!(false)),
        GateMode::None
    );

    // Without `mlp_layer_types` the step / only-layers schedule applies.
    let mut args = args;
    args.mlp_layer_types = None;
    assert!(!args.is_moe_layer(0));
    assert!(args.is_moe_layer(1));
    args.decoder_sparse_step = 2;
    assert!(args.is_moe_layer(1));
    assert!(!args.is_moe_layer(2));
}

#[test]
fn layer_rope_full_uses_yarn_partial_and_sliding_uses_default() {
    let args = xs_config();
    let full = args.layer_rope("full_attention");
    assert_eq!(full.rotated_dims, 64);
    let yarn = full.yarn.as_ref().expect("full layers use YaRN");
    assert!(
        (yarn.mscale - 1.346_573_6).abs() < 1e-6,
        "mscale {}",
        yarn.mscale
    );
    assert_eq!(mlxcel_core::array_shape(&yarn.freqs), vec![32]);

    let sliding = args.layer_rope(SLIDING_ATTENTION);
    assert_eq!(sliding.rotated_dims, 128);
    assert!(sliding.yarn.is_none());
    assert!((sliding.base - 10000.0).abs() < 1e-3);

    // XS.2 style: the factor and the original length live at the top level.
    let mut xs2 = args;
    xs2.rope_parameters = serde_json::json!({
        "original_max_position_embeddings": 4096,
        "full_attention": {"rope_theta": 500000.0, "rope_type": "yarn", "factor": 64.0,
                           "beta_slow": 1.0, "beta_fast": 64.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0}
    });
    xs2.partial_rotary_factor = Some(0.5);
    let full = xs2.layer_rope("full_attention");
    assert_eq!(full.rotated_dims, 64);
    let derived = full.yarn.as_ref().expect("yarn").mscale;
    assert!(
        (derived - (0.1 * 64f32.ln() + 1.0)).abs() < 1e-5,
        "{derived}"
    );

    // A missing entry falls back to the top-level `rope_theta`.
    let mut bare = xs2;
    bare.rope_parameters = serde_json::Value::Null;
    bare.rope_theta = Some(123456.0);
    let rope = bare.layer_rope(SLIDING_ATTENTION);
    assert!(rope.yarn.is_none());
    assert!((rope.base - 123456.0).abs() < 1.0);
    assert_eq!(
        rope.rotated_dims, 64,
        "top-level partial factor still applies"
    );
}

#[test]
fn router_sigmoid_selects_with_bias_but_weights_without_bias() {
    // Expert 2 has the top score but a large negative bias pushes it out of
    // the top-2; expert 3 has the lowest score and a bias that lifts it in.
    let logits = f32_array(&[1.0, 0.5, 3.0, -2.0], &[1, 4]);
    let bias = f32_array(&[0.0, 0.0, -10.0, 10.0], &[4]);
    let scores = router_scores(&logits, RouterScoreFunc::Sigmoid);
    let score_vals = to_vec(&scores);
    let (indices, weights) = router_select(&scores, Some(&bias), 2, true, 2.5);
    let mut idx = to_vec(&indices)
        .into_iter()
        .map(|v| v as usize)
        .collect::<Vec<_>>();
    let w = to_vec(&weights);
    let mut pairs: Vec<(usize, f32)> = idx.iter().copied().zip(w.iter().copied()).collect();
    pairs.sort_by_key(|p| p.0);
    idx.sort_unstable();
    assert_eq!(idx, vec![0, 3], "selection follows scores + bias");

    let s0 = score_vals[0];
    let s3 = score_vals[3];
    let sum = s0 + s3;
    assert!((pairs[0].1 - 2.5 * s0 / sum).abs() < 1e-5, "{pairs:?}");
    assert!((pairs[1].1 - 2.5 * s3 / sum).abs() < 1e-5, "{pairs:?}");
    // The weights are bias-free sigmoid scores, so they are far from what a
    // bias-inclusive renormalization would give.
    assert!((s3 - 1.0 / (1.0 + 2f32.exp())).abs() < 1e-5);
}

/// The legacy spelling. A checkpoint predating `moe_router_score_func` says
/// `moe_router_use_sigmoid: false`, which selects softmax scoring; anything
/// else, including the key being absent, is sigmoid.
#[test]
fn legacy_router_use_sigmoid_flag_resolves_the_score_function() {
    let resolve = |legacy: Option<bool>, explicit: Option<&str>| {
        let mut args = xs_config();
        args.moe_router_use_sigmoid = legacy;
        args.moe_router_score_func = explicit.map(str::to_string);
        args.router_score_func()
    };
    assert_eq!(
        resolve(Some(false), None).unwrap(),
        RouterScoreFunc::Softmax
    );
    assert_eq!(resolve(Some(true), None).unwrap(), RouterScoreFunc::Sigmoid);
    assert_eq!(resolve(None, None).unwrap(), RouterScoreFunc::Sigmoid);
    // The explicit key wins over the legacy one, and an unknown name is an
    // error rather than a silent fallback to sigmoid.
    assert_eq!(
        resolve(Some(false), Some("sqrtsoftplus")).unwrap(),
        RouterScoreFunc::SqrtSoftplus
    );
    assert!(resolve(None, Some("bogus")).is_err());
}

#[test]
fn router_sqrtsoftplus_matches_formula() {
    let z = [-100.0f32, -3.0, 0.0, 2.5, 30.0];
    let logits = f32_array(&z, &[1, 5]);
    let scores = to_vec(&router_scores(&logits, RouterScoreFunc::SqrtSoftplus));
    for (i, &zi) in z.iter().enumerate() {
        let expected = (zi.exp().ln_1p()).sqrt();
        assert!(scores[i].is_finite(), "z = {zi} produced {}", scores[i]);
        let tol = 1e-6f32.max(expected * 1e-5);
        assert!(
            (scores[i] - expected).abs() < tol,
            "z = {zi}: got {}, expected {expected}",
            scores[i]
        );
    }
    let softmax = to_vec(&router_scores(&logits, RouterScoreFunc::Softmax));
    assert!((softmax.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

/// Router rows for the routed-block tests: token `[1, 0]` sees logits
/// `[6, 3, 0]` and token `[0, 1]` sees `[-4, 8, 0]`. The correction bias
/// lifts only expert 2.
const ROUTER_ROWS: [[f32; 2]; 3] = [[6.0, -4.0], [3.0, 8.0], [0.0, 0.0]];
const ROUTER_BIAS: [f32; 3] = [0.0, 0.0, 0.85];
const ROUTED_SCALE: f64 = 2.5;

/// A two-layer config whose layer 1 is one routed block: hidden 2, three
/// experts of width 2 at top-2, a width-2 shared expert.
fn routed_block_args(score_func: &str) -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "laguna", "vocab_size": 8, "hidden_size": 2, "intermediate_size": 4,
        "num_hidden_layers": 2, "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 2,
        "num_experts": 3, "num_experts_per_tok": 2, "moe_intermediate_size": 2,
        "shared_expert_intermediate_size": 2, "norm_topk_prob": true,
        "mlp_layer_types": ["dense", "sparse"], "moe_routed_scaling_factor": ROUTED_SCALE,
        "moe_router_score_func": score_func
    }))
    .expect("routed block config")
}

/// Expert and shared-expert weights for [`routed_block_args`], row-major:
/// `gate` / `up` are `[E, I, H]`, `down` is `[E, H, I]`, and the shared
/// expert's are the `E = 1` case.
struct RoutedBlockWeights {
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    shared_gate: Vec<f32>,
    shared_up: Vec<f32>,
    shared_down: Vec<f32>,
}

impl RoutedBlockWeights {
    fn new() -> Self {
        let mut rng = Lcg(0x1765);
        let mut draw = |n: usize| (0..n).map(|_| rng.next_f32(1.0)).collect::<Vec<f32>>();
        Self {
            gate: draw(12),
            up: draw(12),
            down: draw(12),
            shared_gate: draw(4),
            shared_up: draw(4),
            shared_down: draw(4),
        }
    }

    fn weight_map(&self) -> WeightMap {
        let mut w: WeightMap = std::collections::HashMap::new();
        let rows: Vec<f32> = ROUTER_ROWS.iter().flatten().copied().collect();
        w.insert("mlp.gate.proj.weight".into(), f32_array(&rows, &[3, 2]));
        w.insert(
            "mlp.gate.e_score_correction_bias".into(),
            f32_array(&ROUTER_BIAS, &[3]),
        );
        for (leaf, data) in [
            ("gate_proj", &self.gate),
            ("up_proj", &self.up),
            ("down_proj", &self.down),
        ] {
            w.insert(
                format!("mlp.switch_mlp.{leaf}.weight"),
                f32_array(data, &[3, 2, 2]),
            );
        }
        for (leaf, data) in [
            ("gate_proj", &self.shared_gate),
            ("up_proj", &self.shared_up),
            ("down_proj", &self.shared_down),
        ] {
            w.insert(
                format!("mlp.shared_expert.{leaf}.weight"),
                f32_array(data, &[2, 2]),
            );
        }
        w
    }

    /// `down · (silu(gate · x) * (up · x))` for one width-2 SwiGLU.
    fn swiglu(gate: &[f32], up: &[f32], down: &[f32], x: [f64; 2]) -> [f64; 2] {
        let dot = |row: &[f32]| row[0] as f64 * x[0] + row[1] as f64 * x[1];
        let act: Vec<f64> = (0..2)
            .map(|i| {
                let g = dot(&gate[i * 2..i * 2 + 2]);
                g / (1.0 + (-g).exp()) * dot(&up[i * 2..i * 2 + 2])
            })
            .collect();
        [0, 1].map(|h| down[h * 2] as f64 * act[0] + down[h * 2 + 1] as f64 * act[1])
    }

    /// Plain-f64 reference for the routed block on one token: select the top
    /// two experts on `score + bias`, weight them by their bias-free scores
    /// renormalized over the pair and scaled, then add the shared expert.
    /// Returns the output and the chosen experts in ascending order.
    fn reference(&self, x: [f64; 2], score: fn(f64) -> f64) -> ([f64; 2], Vec<usize>) {
        let scores: Vec<f64> = ROUTER_ROWS
            .iter()
            .map(|row| score(row[0] as f64 * x[0] + row[1] as f64 * x[1]))
            .collect();
        let biased = |e: usize| scores[e] + ROUTER_BIAS[e] as f64;
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| biased(b).total_cmp(&biased(a)));
        let mut chosen = order[..2].to_vec();
        chosen.sort_unstable();
        let total: f64 = chosen.iter().map(|&e| scores[e]).sum();
        let mut y = Self::swiglu(&self.shared_gate, &self.shared_up, &self.shared_down, x);
        for &e in &chosen {
            let s = e * 4..e * 4 + 4;
            let out = Self::swiglu(&self.gate[s.clone()], &self.up[s.clone()], &self.down[s], x);
            let weight = ROUTED_SCALE * scores[e] / total;
            y[0] += weight * out[0];
            y[1] += weight * out[1];
        }
        (y, chosen)
    }
}

/// `moe_router_score_func: "sqrtsoftplus"` must reach the routed block's
/// forward pass, not just parse. `router_sqrtsoftplus_matches_formula` checks
/// `router_scores` alone and `legacy_router_use_sigmoid_flag_resolves_the_score_function`
/// checks the config key alone; neither fails if the block ignores the key.
///
/// The case is built so the two score functions route differently. Sigmoid
/// scores saturate near 1, so on token `[1, 0]` a bias of 0.85 lifts expert 2
/// past expert 1 (0.95 + 0 < 0.5 + 0.85); sqrt-softplus keeps growing with the
/// logit, so the same bias falls short (1.75 > 0.83 + 0.85). On token `[0, 1]`
/// both pick experts 1 and 2 but weight them 2:1 under sigmoid and about
/// 3.4:1 under sqrt-softplus.
#[test]
fn sqrtsoftplus_router_routes_and_weights_a_forward_pass() {
    let sigmoid: fn(f64) -> f64 = |z| 1.0 / (1.0 + (-z).exp());
    let sqrt_softplus: fn(f64) -> f64 = |z| z.exp().ln_1p().sqrt();
    let params = RoutedBlockWeights::new();
    let weights = params.weight_map();
    let tokens = [[1.0f64, 0.0], [0.0, 1.0]];
    let x = f32_array(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    // The references themselves disagree, on the chosen experts for token 0
    // and on the output for both tokens, by far more than the tolerance.
    assert_eq!(params.reference(tokens[0], sigmoid).1, vec![0, 2]);
    assert_eq!(params.reference(tokens[0], sqrt_softplus).1, vec![0, 1]);
    assert_eq!(params.reference(tokens[1], sigmoid).1, vec![1, 2]);
    assert_eq!(params.reference(tokens[1], sqrt_softplus).1, vec![1, 2]);
    for token in tokens {
        let a = params.reference(token, sigmoid).0;
        let b = params.reference(token, sqrt_softplus).0;
        let gap = (a[0] - b[0]).abs().max((a[1] - b[1]).abs());
        assert!(gap > 1e-2, "token {token:?}: the arms differ by only {gap}");
    }

    for (name, score) in [("sigmoid", sigmoid), ("sqrtsoftplus", sqrt_softplus)] {
        let args = routed_block_args(name);
        let quant = args.quant_spec(false);
        let block = SparseMoeBlock::from_weights(&weights, &args, &quant, "mlp")
            .expect("routed block builds");
        let got = to_vec(&block.forward(&x));
        for (t, token) in tokens.iter().enumerate() {
            let want = params.reference(*token, score).0;
            for h in 0..2 {
                let g = got[t * 2 + h] as f64;
                assert!(
                    (g - want[h]).abs() < 1e-4 * (1.0 + want[h].abs()),
                    "{name}: token {t} feature {h} is {g}, the reference is {}",
                    want[h]
                );
            }
        }
    }

    // End to end: the key reaches every routed layer through
    // `LagunaModel::from_weights`, and the logits move with it.
    let tiny_logits = |score_func: &str| {
        let mut config: serde_json::Value = serde_json::from_str(TINY_CONFIG).unwrap();
        config["moe_router_score_func"] = serde_json::json!(score_func);
        let args: ModelArgs = serde_json::from_value(config).unwrap();
        let model = LagunaModel::from_weights(&tiny_weights(&args), &args).expect("tiny model");
        let ids = mlxcel_core::from_slice_i32(&[1, 5, 9, 3, 7, 2], &[1, 6]);
        let mut caches = model.make_caches();
        to_vec(&model.forward_with_caches(&ids, &mut caches))
    };
    let sigmoid_logits = tiny_logits("sigmoid");
    let sqrt_softplus_logits = tiny_logits("sqrtsoftplus");
    assert!(sqrt_softplus_logits.iter().all(|v| v.is_finite()));
    let moved = sigmoid_logits
        .iter()
        .zip(&sqrt_softplus_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        moved > 1e-3,
        "sqrtsoftplus left the logits unchanged ({moved})"
    );
}

/// Two-head, head_dim 2 attention over a 4-wide hidden state with an identity
/// `o_proj` and `q/k/v` projections that keep the values in range.
fn gate_test_args(gating: serde_json::Value) -> ModelArgs {
    serde_json::from_str(&format!(
        r#"{{
        "model_type": "laguna", "vocab_size": 16, "hidden_size": 4, "intermediate_size": 8,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
        "layer_types": ["sliding_attention"], "sliding_window": 8,
        "rope_parameters": {{"sliding_attention": {{"rope_type": "default", "rope_theta": 10000.0}}}},
        "gating": {gating}
    }}"#
    ))
    .unwrap()
}

fn gate_test_weights() -> WeightMap {
    let mut w: WeightMap = std::collections::HashMap::new();
    let mut q = vec![0.0f32; 16];
    for i in 0..4 {
        q[i * 4 + i] = 1.0;
    }
    w.insert("attn.q_proj.weight".into(), f32_array(&q, &[4, 4]));
    let kv = [0.3, -0.2, 0.5, 0.1, -0.4, 0.6, 0.2, 0.7];
    w.insert("attn.k_proj.weight".into(), f32_array(&kv, &[2, 4]));
    w.insert("attn.v_proj.weight".into(), f32_array(&kv, &[2, 4]));
    w.insert("attn.o_proj.weight".into(), f32_array(&q, &[4, 4]));
    w.insert("attn.q_norm.weight".into(), f32_array(&[1.0, 1.0], &[2]));
    w.insert("attn.k_norm.weight".into(), f32_array(&[1.0, 1.0], &[2]));
    // g_proj on an all-ones input: head 0 -> 0, head 1 -> 10.
    w.insert(
        "attn.g_proj.weight".into(),
        f32_array(&[0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5], &[2, 4]),
    );
    w
}

#[test]
fn per_head_gate_applies_softplus_per_head() {
    let x = f32_array(&[1.0; 12], &[1, 3, 4]);
    let run = |gating: serde_json::Value| {
        let args = gate_test_args(gating);
        let quant = args.quant_spec(false);
        let rope = args.layer_rope(SLIDING_ATTENTION);
        let attn = Attention::from_weights(&gate_test_weights(), &args, &quant, "attn", 0, &rope)
            .expect("attention builds");
        let mut cache = LagunaCache::Rotating(mlxcel_core::layers::RotatingKVCache::new(8));
        to_vec(&attn.forward(&x, &mut cache))
    };
    let plain = run(serde_json::json!(false));
    let gated = run(serde_json::json!("per-head"));
    let g0 = 2f32.ln();
    let g1 = 10f32.exp().ln_1p();
    assert!(
        plain.iter().any(|v| v.abs() > 1e-3),
        "attention output is non-trivial"
    );
    for (i, (p, g)) in plain.iter().zip(gated.iter()).enumerate() {
        let scale = if i % 4 < 2 { g0 } else { g1 };
        assert!(
            (g - p * scale).abs() < 1e-4 * scale.max(1.0),
            "feature {i}: gated {g}, plain {p}, scale {scale}"
        );
    }
}

#[test]
fn gate_width_mismatch_is_rejected() {
    let args = gate_test_args(serde_json::json!("per-element"));
    let quant = args.quant_spec(false);
    let rope = args.layer_rope(SLIDING_ATTENTION);
    let err = Attention::from_weights(&gate_test_weights(), &args, &quant, "attn", 0, &rope)
        .err()
        .expect("a 2-wide gate cannot be per-element over 4 features");
    assert!(err.contains("g_proj"), "{err}");
}

/// Layer 0 full, layer 1 sliding with a 3-token window; two query heads share
/// one key/value head of width 2 over a 4-wide hidden state, and no gate.
fn sink_test_args(sinks_enabled: bool) -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "laguna", "vocab_size": 16, "hidden_size": 4, "intermediate_size": 8,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
        "layer_types": ["full_attention", "sliding_attention"], "sliding_window": 3,
        "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
        "gating": false,
        "swa_attention_sink_enabled": sinks_enabled
    }))
    .expect("sink test config")
}

/// `v_proj` rows for the sink tests; the value of token `x` is `SINK_V · x`.
const SINK_V: [[f32; 4]; 2] = [[0.5, -0.25, 1.0, 0.0], [0.0, 1.0, -0.5, 0.25]];

/// A zero `q_proj` makes every attention logit 0, so each query row's softmax
/// is uniform over the `n` keys it can see: every value gets `1 / n` without a
/// sink and `1 / (n + exp(sink_h))` with one. `o_proj` is the identity.
fn sink_test_weights(sink: Option<[f32; 2]>) -> WeightMap {
    let mut w: WeightMap = std::collections::HashMap::new();
    let mut eye = vec![0.0f32; 16];
    for i in 0..4 {
        eye[i * 4 + i] = 1.0;
    }
    w.insert("attn.q_proj.weight".into(), f32_array(&[0.0; 16], &[4, 4]));
    w.insert(
        "attn.k_proj.weight".into(),
        f32_array(&[0.3, -0.2, 0.5, 0.1, -0.4, 0.6, 0.2, 0.7], &[2, 4]),
    );
    let v: Vec<f32> = SINK_V.iter().flatten().copied().collect();
    w.insert("attn.v_proj.weight".into(), f32_array(&v, &[2, 4]));
    w.insert("attn.o_proj.weight".into(), f32_array(&eye, &[4, 4]));
    w.insert("attn.q_norm.weight".into(), f32_array(&[1.0, 1.0], &[2]));
    w.insert("attn.k_norm.weight".into(), f32_array(&[1.0, 1.0], &[2]));
    if let Some(sink) = sink {
        w.insert("attn.sink".into(), f32_array(&sink, &[2]));
    }
    w
}

/// `swa_attention_sink_enabled` gives each sliding layer a per-head sink
/// logit that joins the softmax denominator, on the prefill path (masked,
/// window-clipped) and on the single-token decode path (no mask). No published
/// checkpoint sets the flag, so this is the only coverage the arm has.
#[test]
fn sliding_attention_sink_joins_the_softmax_on_prefill_and_decode() {
    let sink = [0.0f32, 3f32.ln()];
    let window = 3usize;
    let mut rng = Lcg(0x5117);
    let x: Vec<f32> = (0..24).map(|_| rng.next_f32(1.0)).collect();
    let values: Vec<[f64; 2]> = x
        .chunks_exact(4)
        .map(|t| SINK_V.map(|row| (0..4).map(|i| row[i] as f64 * t[i] as f64).sum()))
        .collect();

    // Token t of the six attends to tokens max(0, t - 2)..=t: a five-token
    // prefill through the window mask, then one decode step.
    let expected = |sink: Option<[f32; 2]>| -> Vec<f64> {
        let mut out = Vec::new();
        for t in 0..values.len() {
            let seen = &values[(t + 1).saturating_sub(window)..=t];
            for h in 0..2 {
                let denom = seen.len() as f64 + sink.map_or(0.0, |s| (s[h] as f64).exp());
                for d in 0..2 {
                    out.push(seen.iter().map(|v| v[d]).sum::<f64>() / denom);
                }
            }
        }
        out
    };
    let run = |args: &ModelArgs, weights: &WeightMap| -> Vec<f32> {
        let quant = args.quant_spec(false);
        let rope = args.layer_rope(SLIDING_ATTENTION);
        let attn = Attention::from_weights(weights, args, &quant, "attn", 1, &rope)
            .expect("sliding layer builds");
        let mut cache = LagunaCache::Rotating(mlxcel_core::layers::RotatingKVCache::new(
            args.sliding_window as i32,
        ));
        let mut out = to_vec(&attn.forward(&f32_array(&x[..20], &[1, 5, 4]), &mut cache));
        out.extend(to_vec(
            &attn.forward(&f32_array(&x[20..], &[1, 1, 4]), &mut cache),
        ));
        out
    };
    let check = |label: &str, got: &[f32], want: &[f64]| {
        assert_eq!(got.len(), want.len(), "{label}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (*g as f64 - w).abs() < 1e-4 * (1.0 + w.abs()),
                "{label}: token {} head {} dim {}: got {g}, expected {w}",
                i / 4,
                (i / 2) % 2,
                i % 2
            );
        }
    };

    let with_sink = expected(Some(sink));
    let without = expected(None);
    assert!(
        with_sink
            .iter()
            .zip(&without)
            .any(|(a, b)| (a - b).abs() > 0.05),
        "the sink must move the expected output"
    );
    check(
        "sinks enabled",
        &run(&sink_test_args(true), &sink_test_weights(Some(sink))),
        &with_sink,
    );
    // With the flag off, a `sink` tensor in the checkpoint is not read.
    check(
        "sinks disabled",
        &run(&sink_test_args(false), &sink_test_weights(Some(sink))),
        &without,
    );

    // Only sliding layers carry a sink: a full layer builds without one even
    // with the flag on, and an enabled sliding layer refuses to build without it.
    let args = sink_test_args(true);
    let quant = args.quant_spec(false);
    let full = Attention::from_weights(
        &sink_test_weights(None),
        &args,
        &quant,
        "attn",
        0,
        &args.layer_rope(FULL_ATTENTION),
    )
    .expect("a full-attention layer never reads a sink");
    assert!(full.sinks.is_none());
    let err = Attention::from_weights(
        &sink_test_weights(None),
        &args,
        &quant,
        "attn",
        1,
        &args.layer_rope(SLIDING_ATTENTION),
    )
    .err()
    .expect("an enabled sliding layer needs its sink");
    assert!(err.contains("attn.sink"), "{err}");
}

fn compressed_tensors_weights(num_experts: usize) -> (WeightMap, Vec<f32>, Vec<u8>) {
    let mut w: WeightMap = std::collections::HashMap::new();
    let mut globals = Vec::new();
    let mut scale_bytes = Vec::new();
    // Not all powers of two: 1.75 is `1.11b * 2^0`, which fills every E4M3
    // mantissa bit, and `3 * 2^-9` is an E4M3 subnormal. A power-of-two-only
    // set would pass the round-trip assertion without ever exercising the
    // mantissa or the subnormal range.
    let scale_values = [1.0f32, 1.75, 0.5, 3.0 * 2f32.powi(-9)];
    for e in 0..num_experts {
        let base = format!("model.layers.1.mlp.experts.{e}.gate_proj");
        // out = 4 rows, in = 16 -> 8 packed bytes per row, 1 block scale per row.
        let packed: Vec<u8> = (0..32)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(e as u8))
            .collect();
        w.insert(
            format!("{base}.weight_packed"),
            mlxcel_core::from_bytes(&packed, &[4, 8], mlxcel_core::dtype::UINT8),
        );
        // The loader promotes F8_E4M3 to f16; feed E4M3-representable values.
        let scales = f32_array(&scale_values, &[4, 1]);
        let scales = mlxcel_core::astype(&scales, mlxcel_core::dtype::FLOAT16);
        w.insert(format!("{base}.weight_scale"), scales);
        let g = 0.75 + e as f32;
        globals.push(g);
        w.insert(format!("{base}.weight_global_scale"), f32_array(&[g], &[1]));
        w.insert(
            format!("{base}.input_global_scale"),
            f32_array(&[3.0], &[1]),
        );
        scale_bytes.extend(scale_values.iter().map(|&v| f32_to_f8_e4m3(v)));
    }
    w.insert(
        "model.layers.1.self_attn.k_scale".into(),
        f32_array(&[1.0], &[1]),
    );
    w.insert(
        "model.layers.1.self_attn.v_scale".into(),
        f32_array(&[1.0], &[1]),
    );
    w.insert(
        "model.layers.1.self_attn.rotary_emb.inv_freq".into(),
        f32_array(&[1.0, 0.5], &[2]),
    );
    w.insert(
        "model.layers.1.mlp.gate.weight".into(),
        f32_array(&[0.0; 8], &[2, 4]),
    );
    w.insert(
        "model.layers.1.mlp.experts.e_score_correction_bias".into(),
        f32_array(&[0.0; 2], &[2]),
    );
    (w, globals, scale_bytes)
}

#[test]
fn sanitize_compressed_tensors_nvfp4_stacks_experts_and_keeps_global_scale() {
    let mut args = xs_config();
    args.num_hidden_layers = 2;
    args.num_experts = 2;
    let (mut w, globals, expected_scale_bytes) = compressed_tensors_weights(2);
    let expected_words: Vec<u32> = (0..2)
        .flat_map(|e| {
            (0..32u8)
                .map(move |i| i.wrapping_mul(7).wrapping_add(e))
                .collect::<Vec<u8>>()
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<u32>>()
        })
        .collect();

    let transcoded = sanitize_weights(&mut w, &args).expect("sanitize");
    assert_eq!(transcoded, 1);

    let weight = w
        .get("model.layers.1.mlp.switch_mlp.gate_proj.weight")
        .expect("stacked weight");
    assert_eq!(mlxcel_core::array_dtype(weight), mlxcel_core::dtype::UINT32);
    assert_eq!(mlxcel_core::array_shape(weight), vec![2, 4, 2]);
    assert_eq!(to_u32(weight), expected_words);

    let scales = w
        .get("model.layers.1.mlp.switch_mlp.gate_proj.scales")
        .expect("stacked scales");
    assert_eq!(mlxcel_core::array_dtype(scales), mlxcel_core::dtype::UINT8);
    assert_eq!(mlxcel_core::array_shape(scales), vec![2, 4, 1]);
    mlxcel_core::eval(scales);
    assert_eq!(
        mlxcel_core::array_to_raw_bytes(scales),
        expected_scale_bytes
    );

    let global = w
        .get("model.layers.1.mlp.switch_mlp.gate_proj.global_scale")
        .expect("global scale sidecar");
    assert_eq!(
        mlxcel_core::array_dtype(global),
        mlxcel_core::dtype::FLOAT32
    );
    let got = to_vec(global);
    for (g, expected) in got.iter().zip(globals.iter().map(|g| 1.0 / g)) {
        assert!((g - expected).abs() < 1e-6, "{got:?}");
    }

    for gone in [
        "model.layers.1.mlp.experts.0.gate_proj.weight_packed",
        "model.layers.1.mlp.experts.0.gate_proj.weight_scale",
        "model.layers.1.mlp.experts.0.gate_proj.weight_global_scale",
        "model.layers.1.mlp.experts.0.gate_proj.input_global_scale",
        "model.layers.1.self_attn.k_scale",
        "model.layers.1.self_attn.v_scale",
        "model.layers.1.self_attn.rotary_emb.inv_freq",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.experts.e_score_correction_bias",
    ] {
        assert!(!w.contains_key(gone), "{gone} should be removed");
    }
    assert!(w.contains_key("model.layers.1.mlp.gate.proj.weight"));
    assert!(w.contains_key("model.layers.1.mlp.gate.e_score_correction_bias"));

    // Idempotent: a second pass leaves the key set and the words unchanged.
    let before: std::collections::BTreeSet<String> = w.keys().cloned().collect();
    let again = sanitize_weights(&mut w, &args).expect("second sanitize");
    assert_eq!(again, 0);
    let after: std::collections::BTreeSet<String> = w.keys().cloned().collect();
    assert_eq!(before, after);
    assert_eq!(
        to_u32(
            w.get("model.layers.1.mlp.switch_mlp.gate_proj.weight")
                .unwrap()
        ),
        expected_words
    );
}

#[test]
fn sanitize_transcodes_dense_shared_expert_triplet() {
    let mut args = xs_config();
    args.num_hidden_layers = 1;
    args.num_experts = 0;
    let mut w: WeightMap = std::collections::HashMap::new();
    let packed: Vec<u8> = (0..16).collect();
    w.insert(
        "model.layers.0.mlp.shared_expert.up_proj.weight_packed".into(),
        mlxcel_core::from_bytes(&packed, &[2, 8], mlxcel_core::dtype::UINT8),
    );
    w.insert(
        "model.layers.0.mlp.shared_expert.up_proj.weight_scale".into(),
        mlxcel_core::from_bytes(&[0x38, 0x30], &[2, 1], mlxcel_core::dtype::UINT8),
    );
    w.insert(
        "model.layers.0.mlp.shared_expert.up_proj.weight_global_scale".into(),
        f32_array(&[4.0], &[1]),
    );
    assert_eq!(sanitize_weights(&mut w, &args).unwrap(), 1);
    let weight = w
        .get("model.layers.0.mlp.shared_expert.up_proj.weight")
        .expect("dense weight");
    assert_eq!(mlxcel_core::array_shape(weight), vec![2, 2]);
    assert_eq!(
        to_u32(weight),
        vec![0x0302_0100, 0x0706_0504, 0x0b0a_0908, 0x0f0e_0d0c]
    );
    let scales = w
        .get("model.layers.0.mlp.shared_expert.up_proj.scales")
        .unwrap();
    mlxcel_core::eval(scales);
    assert_eq!(mlxcel_core::array_to_raw_bytes(scales), vec![0x38, 0x30]);
    let global = to_vec(
        w.get("model.layers.0.mlp.shared_expert.up_proj.global_scale")
            .unwrap(),
    );
    assert_eq!(global, vec![0.25]);
    assert!(!w.contains_key("model.layers.0.mlp.shared_expert.up_proj.weight_packed"));
}

#[test]
fn sanitize_splits_prestacked_gate_up() {
    let mut args = xs_config();
    args.num_hidden_layers = 1;
    args.num_experts = 2;
    args.quantization_config = None;
    let mut w: WeightMap = std::collections::HashMap::new();
    // [E=2, 2I=4, in/8=2] words, expert-major then row-major.
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    w.insert(
        "model.layers.0.mlp.switch_mlp.gate_up_proj.weight".into(),
        f32_array(&data, &[2, 4, 2]),
    );
    w.insert(
        "model.layers.0.mlp.switch_mlp.gate_up_proj.scales".into(),
        f32_array(&data, &[2, 4, 2]),
    );
    sanitize_weights(&mut w, &args).unwrap();
    let gate = w
        .get("model.layers.0.mlp.switch_mlp.gate_proj.weight")
        .expect("gate half");
    let up = w
        .get("model.layers.0.mlp.switch_mlp.up_proj.weight")
        .expect("up half");
    assert_eq!(mlxcel_core::array_shape(gate), vec![2, 2, 2]);
    assert_eq!(mlxcel_core::array_shape(up), vec![2, 2, 2]);
    assert_eq!(to_vec(gate), vec![0., 1., 2., 3., 8., 9., 10., 11.]);
    assert_eq!(to_vec(up), vec![4., 5., 6., 7., 12., 13., 14., 15.]);
    assert!(w.contains_key("model.layers.0.mlp.switch_mlp.up_proj.scales"));
    assert!(!w.contains_key("model.layers.0.mlp.switch_mlp.gate_up_proj.weight"));
}

#[test]
fn sanitize_stacks_bf16_experts_and_pops_tied_head() {
    let mut args = xs_config();
    args.num_hidden_layers = 1;
    args.num_experts = 2;
    args.tie_word_embeddings = true;
    args.quantization_config = None;
    let mut w: WeightMap = std::collections::HashMap::new();
    for e in 0..2 {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            w.insert(
                format!("model.layers.0.mlp.experts.{e}.{proj}.weight"),
                f32_array(&[e as f32; 4], &[2, 2]),
            );
        }
    }
    w.insert("lm_head.weight".into(), f32_array(&[0.0; 4], &[2, 2]));
    sanitize_weights(&mut w, &args).unwrap();
    let stacked = w
        .get("model.layers.0.mlp.switch_mlp.down_proj.weight")
        .expect("stacked");
    assert_eq!(mlxcel_core::array_shape(stacked), vec![2, 2, 2]);
    assert_eq!(to_vec(stacked), vec![0., 0., 0., 0., 1., 1., 1., 1.]);
    assert!(!w.contains_key("model.layers.0.mlp.experts.0.down_proj.weight"));
    assert!(!w.contains_key("lm_head.weight"));
}

/// Tiny hybrid config: two full and two sliding layers, a 6-token window so
/// a 96-token prompt exercises the window band, a dense layer 0, and 4
/// routed experts plus a shared expert elsewhere.
const TINY_CONFIG: &str = r#"{
    "model_type": "laguna", "vocab_size": 32, "hidden_size": 16, "intermediate_size": 24,
    "num_hidden_layers": 4, "num_attention_heads": 2, "num_attention_heads_per_layer": [2, 4, 2, 4],
    "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-6,
    "layer_types": ["full_attention", "sliding_attention", "full_attention", "sliding_attention"],
    "sliding_window": 6,
    "rope_parameters": {
        "full_attention": {"rope_theta": 500000.0, "rope_type": "yarn", "factor": 32.0,
                           "original_max_position_embeddings": 64, "beta_slow": 1.0, "beta_fast": 64.0,
                           "attention_factor": 1.3465735902799727, "partial_rotary_factor": 0.5},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0}
    },
    "gating": "per-head",
    "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 8,
    "shared_expert_intermediate_size": 8, "norm_topk_prob": true,
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "moe_routed_scaling_factor": 2.5, "tie_word_embeddings": false, "eos_token_id": [2, 24]
}"#;

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

fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let mut rng = Lcg(0x1347);
    let mut w: WeightMap = std::collections::HashMap::new();
    let mut rand = |shape: &[i32], scale: f32| {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.next_f32(scale)).collect();
        f32_array(&data, shape)
    };
    let h = args.hidden_size as i32;
    let d = args.head_dim() as i32;
    let kv = args.num_key_value_heads as i32;
    let v = args.vocab_size as i32;
    w.insert("model.embed_tokens.weight".into(), rand(&[v, h], 0.5));
    w.insert(
        "model.norm.weight".into(),
        f32_array(&vec![1.0; h as usize], &[h]),
    );
    w.insert("lm_head.weight".into(), rand(&[v, h], 0.3));
    for i in 0..args.num_hidden_layers {
        let nh = args.num_heads_for_layer(i).unwrap() as i32;
        let p = format!("model.layers.{i}");
        w.insert(
            format!("{p}.self_attn.q_proj.weight"),
            rand(&[nh * d, h], 0.3),
        );
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rand(&[kv * d, h], 0.3),
        );
        w.insert(
            format!("{p}.self_attn.v_proj.weight"),
            rand(&[kv * d, h], 0.3),
        );
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rand(&[h, nh * d], 0.3),
        );
        w.insert(format!("{p}.self_attn.g_proj.weight"), rand(&[nh, h], 0.3));
        w.insert(
            format!("{p}.self_attn.q_norm.weight"),
            f32_array(&vec![1.0; d as usize], &[d]),
        );
        w.insert(
            format!("{p}.self_attn.k_norm.weight"),
            f32_array(&vec![1.0; d as usize], &[d]),
        );
        w.insert(
            format!("{p}.input_layernorm.weight"),
            f32_array(&vec![1.0; h as usize], &[h]),
        );
        w.insert(
            format!("{p}.post_attention_layernorm.weight"),
            f32_array(&vec![1.0; h as usize], &[h]),
        );
        if args.is_moe_layer(i) {
            let e = args.num_experts as i32;
            let mi = args.moe_intermediate_size as i32;
            let si = args.shared_expert_intermediate_size.unwrap() as i32;
            w.insert(format!("{p}.mlp.gate.proj.weight"), rand(&[e, h], 0.5));
            w.insert(
                format!("{p}.mlp.gate.e_score_correction_bias"),
                rand(&[e], 0.2),
            );
            w.insert(
                format!("{p}.mlp.switch_mlp.gate_proj.weight"),
                rand(&[e, mi, h], 0.3),
            );
            w.insert(
                format!("{p}.mlp.switch_mlp.up_proj.weight"),
                rand(&[e, mi, h], 0.3),
            );
            w.insert(
                format!("{p}.mlp.switch_mlp.down_proj.weight"),
                rand(&[e, h, mi], 0.3),
            );
            w.insert(
                format!("{p}.mlp.shared_expert.gate_proj.weight"),
                rand(&[si, h], 0.3),
            );
            w.insert(
                format!("{p}.mlp.shared_expert.up_proj.weight"),
                rand(&[si, h], 0.3),
            );
            w.insert(
                format!("{p}.mlp.shared_expert.down_proj.weight"),
                rand(&[h, si], 0.3),
            );
        } else {
            let ii = args.intermediate_size as i32;
            w.insert(format!("{p}.mlp.gate_proj.weight"), rand(&[ii, h], 0.3));
            w.insert(format!("{p}.mlp.up_proj.weight"), rand(&[ii, h], 0.3));
            w.insert(format!("{p}.mlp.down_proj.weight"), rand(&[h, ii], 0.3));
        }
    }
    w
}

pub(crate) fn tiny_model() -> (LagunaModel, ModelArgs) {
    let args: ModelArgs = serde_json::from_str(TINY_CONFIG).expect("tiny config");
    let weights = tiny_weights(&args);
    let model = LagunaModel::from_weights(&weights, &args).expect("tiny laguna builds");
    (model, args)
}

#[test]
fn tiny_model_builds_mixed_caches_and_eos() {
    let (model, _) = tiny_model();
    let caches = model.make_caches();
    let kinds: Vec<bool> = caches.iter().map(LagunaCache::is_rotating).collect();
    assert_eq!(kinds, vec![false, true, false, true]);
    assert_eq!(model.eos_token_ids, vec![2, 24]);
    assert!(model.lm_head.is_some());
}

#[test]
fn prefill_is_causal_on_both_layer_types() {
    let (model, args) = tiny_model();
    let n = 96usize;
    let mut rng = Lcg(7);
    let base: Vec<i32> = (0..n)
        .map(|_| (rng.next_f32(1.0).abs() * (args.vocab_size as f32 - 1.0)) as i32)
        .collect();
    let split = 40usize;
    let mut altered = base.clone();
    for (i, tok) in altered.iter_mut().enumerate().skip(split) {
        *tok = (*tok + 7 + i as i32) % args.vocab_size as i32;
    }
    assert_ne!(base[split..], altered[split..]);

    let run = |tokens: &[i32]| {
        let ids = mlxcel_core::from_slice_i32(tokens, &[1, n as i32]);
        let mut caches = model.make_caches();
        let logits = model.forward_with_caches(&ids, &mut caches);
        (to_vec(&logits), caches)
    };
    let (a, caches_a) = run(&base);
    let (b, _) = run(&altered);
    let vocab = args.vocab_size;
    assert!(a.iter().all(|v| v.is_finite()), "finite logits");
    let mut max_diff = 0f32;
    for pos in 0..split {
        for v in 0..vocab {
            max_diff = max_diff.max((a[pos * vocab + v] - b[pos * vocab + v]).abs());
        }
    }
    assert!(
        max_diff < 1e-3,
        "positions before {split} changed by {max_diff}"
    );
    let mut tail_diff = 0f32;
    for pos in split..n {
        for v in 0..vocab {
            tail_diff = tail_diff.max((a[pos * vocab + v] - b[pos * vocab + v]).abs());
        }
    }
    assert!(tail_diff > 1e-3, "altered tail must change its own logits");
    assert!(caches_a.iter().all(|c| c.offset() == n as i32));
}

#[test]
fn decode_matches_prefill_logits() {
    // The last-position logits of a single-pass prefill must equal the
    // logits of prefilling the prefix and decoding the last token, on both
    // cache kinds and beyond the sliding window.
    let (model, args) = tiny_model();
    let tokens: Vec<i32> = (0..20)
        .map(|i| (i * 5 + 3) % args.vocab_size as i32)
        .collect();
    let vocab = args.vocab_size;
    let ids = mlxcel_core::from_slice_i32(&tokens, &[1, 20]);
    let mut caches = model.make_caches();
    let full = to_vec(&model.forward_with_caches(&ids, &mut caches));
    let last_full = &full[19 * vocab..];

    let prefix = mlxcel_core::from_slice_i32(&tokens[..19], &[1, 19]);
    let mut caches = model.make_caches();
    let _ = model.forward_with_caches(&prefix, &mut caches);
    let last = mlxcel_core::from_slice_i32(&tokens[19..], &[1, 1]);
    let decoded = to_vec(&model.forward_with_caches(&last, &mut caches));
    let max_diff = last_full
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_diff < 2e-3,
        "decode diverged from prefill by {max_diff}"
    );
}

#[test]
fn router_weight_on_input_is_rejected() {
    let mut args: ModelArgs = serde_json::from_str(TINY_CONFIG).unwrap();
    args.moe_apply_router_weight_on_input = true;
    let weights = tiny_weights(&args);
    let err = LagunaModel::from_weights(&weights, &args)
        .err()
        .expect("must reject");
    assert!(err.contains("moe_apply_router_weight_on_input"), "{err}");
}

/// The per-expert `global_scale` sidecar must multiply the routed matmul
/// output of exactly the selected expert, on both the unsorted gather layout
/// (small token counts) and the sorted layout (`gather_sort` output).
#[test]
fn switch_linear_applies_per_expert_global_scale_after_gather_qmm() {
    use super::switch_layers::{SwitchLinear, gather_sort};
    let (out_dim, in_dim, experts) = (8i32, 64i32, 2i32);
    let mut rng = Lcg(21);
    let mut weight_parts = Vec::new();
    let mut scale_parts = Vec::new();
    let mut bias_parts = Vec::new();
    for _ in 0..experts {
        let n = (out_dim * in_dim) as usize;
        let data: Vec<f32> = (0..n).map(|_| rng.next_f32(0.5)).collect();
        let dense = f32_array(&data, &[out_dim, in_dim]);
        let dense = mlxcel_core::astype(&dense, mlxcel_core::dtype::FLOAT16);
        let q = mlxcel_core::quantize_weights(&dense, 64, 4);
        weight_parts.push(mlxcel_core::quantized_weights_w(&q));
        scale_parts.push(mlxcel_core::quantized_weights_scales(&q));
        bias_parts.push(mlxcel_core::quantized_weights_biases(&q));
    }
    let mut w: WeightMap = std::collections::HashMap::new();
    w.insert(
        "sw.weight".into(),
        mlxcel_core::stack_owned(&weight_parts, 0),
    );
    w.insert(
        "sw.scales".into(),
        mlxcel_core::stack_owned(&scale_parts, 0),
    );
    w.insert("sw.biases".into(), mlxcel_core::stack_owned(&bias_parts, 0));
    let plain = SwitchLinear::from_weights_with_mode(&w, "sw", 64, 4, "affine").unwrap();
    w.insert("sw.global_scale".into(), f32_array(&[0.5, 2.0], &[2]));
    let scaled = SwitchLinear::from_weights_with_mode(&w, "sw", 64, 4, "affine").unwrap();
    assert!(
        scaled.quantized_parts().is_none(),
        "fused paths must skip scaled planes"
    );

    let tokens = 3i32;
    let x_data: Vec<f32> = (0..(tokens * in_dim) as usize)
        .map(|_| rng.next_f32(1.0))
        .collect();
    let x = mlxcel_core::astype(
        &f32_array(&x_data, &[tokens, 1, 1, in_dim]),
        mlxcel_core::dtype::FLOAT16,
    );
    // Token 0 -> experts [0, 1], token 1 -> [1, 0], token 2 -> [1, 1].
    let indices = mlxcel_core::from_slice_i32(&[0, 1, 1, 0, 1, 1], &[tokens, 2]);
    let expected_factor = [0.5f32, 2.0, 2.0, 0.5, 2.0, 2.0];

    let base = to_vec(&plain.forward(&x, &indices, false));
    let got = to_vec(&scaled.forward(&x, &indices, false));
    assert_eq!(base.len(), (tokens * 2 * out_dim) as usize);
    for (i, (b, g)) in base.iter().zip(got.iter()).enumerate() {
        let factor = expected_factor[i / out_dim as usize];
        assert!(
            (g - b * factor).abs() < 2e-3 * (1.0 + b.abs()),
            "slot {i}: {g} vs {b} * {factor}"
        );
    }

    // Sorted layout: `[tokens * k, 1, in]` rows with a flat index vector.
    let (sorted_x, sorted_idx, _inv) = gather_sort(&x, &indices);
    let base_sorted = to_vec(&plain.forward(&sorted_x, &sorted_idx, true));
    let got_sorted = to_vec(&scaled.forward(&sorted_x, &sorted_idx, true));
    let idx_sorted = to_vec(&sorted_idx);
    for (i, (b, g)) in base_sorted.iter().zip(got_sorted.iter()).enumerate() {
        let expert = idx_sorted[i / out_dim as usize] as usize;
        let factor = [0.5f32, 2.0][expert];
        assert!(
            (g - b * factor).abs() < 2e-3 * (1.0 + b.abs()),
            "sorted slot {i}"
        );
    }

    // A sidecar whose length does not match the expert count is refused.
    w.insert("sw.global_scale".into(), f32_array(&[0.5, 2.0, 1.0], &[3]));
    assert!(SwitchLinear::from_weights_with_mode(&w, "sw", 64, 4, "affine").is_err());
}

/// `config.json` is untrusted input on the `mlxcel generate -m <org>/<repo>`
/// path. Each value here reaches MLX as an out-of-range `argpartition` pivot or
/// as a NaN that never throws, and an MLX C++ exception crossing the cxx bridge
/// aborts the process rather than failing the load.
#[test]
fn validate_rejects_routing_values_that_abort_inside_mlx() {
    let base: ModelArgs = serde_json::from_str(TINY_CONFIG).unwrap();
    base.validate().expect("the tiny config is valid");

    // The serde default. A config that declares sparse layers through
    // `mlp_layer_types` but omits `num_experts_per_tok` yields 0, and
    // `kth = num_experts - 0` is one past the end of the score row.
    let mut args = base.clone();
    args.num_experts_per_tok = 0;
    let err = args.validate().expect_err("k = 0 must be rejected");
    assert!(err.contains("num_experts_per_tok"), "{err}");

    for bad_k in [base.num_experts + 1, usize::MAX] {
        let mut args = base.clone();
        args.num_experts_per_tok = bad_k;
        assert!(args.validate().is_err(), "k = {bad_k}");
    }

    let mut args = base.clone();
    args.num_experts = 0;
    assert!(args.validate().is_err());

    let mut args = base.clone();
    args.num_hidden_layers = 1_000_000;
    assert!(args.validate().is_err());

    for bad in [f32::INFINITY, f32::NAN] {
        let mut args = base.clone();
        args.moe_routed_scaling_factor = bad;
        assert!(args.validate().is_err(), "routed_scaling_factor {bad}");
        let mut args = base.clone();
        args.moe_router_logit_softcapping = bad;
        assert!(args.validate().is_err(), "softcapping {bad}");
    }

    // A `head_dim` below 2 used to panic inside `Ord::clamp` (which asserts
    // `min <= max`) while resolving the partial-rotary width.
    for bad_dim in [0usize, 1] {
        let mut args = base.clone();
        args.head_dim = Some(bad_dim);
        assert!(args.validate().is_err(), "head_dim {bad_dim}");
        // And the helper itself no longer panics on the way there.
        let _ = args.layer_rope(SLIDING_ATTENTION).rotated_dims;
    }

    // A dense-only config never touches the router, so the MoE keys stay free.
    let mut args = base.clone();
    args.mlp_layer_types = Some(vec!["dense".into(); args.num_hidden_layers]);
    args.num_experts = 0;
    args.num_experts_per_tok = 0;
    args.validate().expect("dense-only config needs no router");
}

/// The partition pivot is clamped into the score row the router actually
/// emitted, so no `num_experts_per_tok` can reach `argpartition` out of range.
#[test]
fn router_select_clamps_the_partition_pivot_into_the_score_row() {
    let scores = f32_array(&[0.1, 0.9, 0.5, 0.3], &[1, 4]);

    // k = 0 used to compute `kth = 4` on a 4-wide row, which MLX rejects by
    // throwing; the clamp keeps one expert.
    let (indices, weights) = router_select(&scores, None, 0, true, 1.0);
    assert_eq!(mlxcel_core::array_shape(&indices), vec![1, 1]);
    assert_eq!(to_vec(&indices), vec![1.0]);
    assert_eq!(to_vec(&weights), vec![1.0]);

    // k past the row width used to slice a negative start out of the partition
    // order, silently returning fewer than k experts instead of failing.
    let (indices, _) = router_select(&scores, None, 9, true, 1.0);
    assert_eq!(mlxcel_core::array_shape(&indices), vec![1, 4]);

    // The in-range case is untouched.
    let (indices, _) = router_select(&scores, None, 2, true, 1.0);
    let mut chosen = to_vec(&indices);
    chosen.sort_by(f32::total_cmp);
    assert_eq!(chosen, vec![1.0, 2.0]);
}

/// The router width and the stacked expert planes come from different tensors,
/// and `router_select` sizes the index range from the router. Planes shorter
/// than the router are an out-of-bounds `gather_qmm` read on an ordinary token.
#[test]
fn a_router_wider_than_the_expert_planes_is_rejected() {
    let args: ModelArgs = serde_json::from_str(TINY_CONFIG).unwrap();
    LagunaModel::from_weights(&tiny_weights(&args), &args).expect("baseline builds");

    for proj in ["gate_proj", "up_proj", "down_proj"] {
        let mut w = tiny_weights(&args);
        let key = format!("model.layers.1.mlp.switch_mlp.{proj}.weight");
        let shape = mlxcel_core::array_shape(w.get(&key).expect("stacked plane"));
        let short = vec![shape[0] - 1, shape[1], shape[2]];
        let n: i32 = short.iter().product();
        w.insert(key, f32_array(&vec![0.1; n as usize], &short));
        let err = LagunaModel::from_weights(&w, &args)
            .err()
            .expect("short plane must be rejected");
        assert!(err.contains("expert planes"), "{proj}: {err}");
    }

    // A correction bias that cannot broadcast onto the score row is refused at
    // load; MLX would throw on the add instead.
    let mut w = tiny_weights(&args);
    w.insert(
        "model.layers.1.mlp.gate.e_score_correction_bias".into(),
        f32_array(&[0.0; 3], &[3]),
    );
    let err = LagunaModel::from_weights(&w, &args)
        .err()
        .expect("bias width must match");
    assert!(err.contains("e_score_correction_bias"), "{err}");
}

/// An expert plane past the declared count means the config under-reports the
/// checkpoint. Stacking only the declared experts leaves the router able to
/// index past the stack, and it sends every leftover plane through the dense
/// transcode as a linear nothing ever reads.
#[test]
fn sanitize_rejects_an_expert_past_the_declared_count() {
    let mut args = xs_config();
    args.num_hidden_layers = 2;
    args.num_experts = 2;
    let (mut w, _, _) = compressed_tensors_weights(3);
    let err = sanitize_weights(&mut w, &args).expect_err("3 planes, 2 declared");
    assert!(err.contains("num_experts = 2"), "{err}");

    // Same rule on the per-expert bf16 path.
    let mut args = xs_config();
    args.num_hidden_layers = 1;
    args.num_experts = 1;
    args.quantization_config = None;
    let mut w: WeightMap = std::collections::HashMap::new();
    for e in 0..2 {
        w.insert(
            format!("model.layers.0.mlp.experts.{e}.down_proj.weight"),
            f32_array(&[e as f32; 4], &[2, 2]),
        );
    }
    let err = sanitize_weights(&mut w, &args).expect_err("2 planes, 1 declared");
    assert!(err.contains("num_experts = 1"), "{err}");
}

/// Expert planes that disagree on shape or dtype abort inside
/// `mlx::core::stack`, which throws and is therefore an uncatchable abort
/// mid-load rather than a load error.
#[test]
fn sanitize_rejects_mismatched_expert_planes() {
    let mut args = xs_config();
    args.num_hidden_layers = 2;
    args.num_experts = 2;
    let (mut w, _, _) = compressed_tensors_weights(2);
    // Expert 1 declares twice the input width. It is self-consistent on its own
    // (16 codes per block scale) and only the cross-expert check catches it.
    let packed: Vec<u8> = (0..64).map(|i| i as u8).collect();
    w.insert(
        "model.layers.1.mlp.experts.1.gate_proj.weight_packed".into(),
        mlxcel_core::from_bytes(&packed, &[4, 16], mlxcel_core::dtype::UINT8),
    );
    w.insert(
        "model.layers.1.mlp.experts.1.gate_proj.weight_scale".into(),
        mlxcel_core::astype(&f32_array(&[1.0; 8], &[4, 2]), mlxcel_core::dtype::FLOAT16),
    );
    let err = sanitize_weights(&mut w, &args).expect_err("unstackable planes");
    assert!(err.contains("share one shape and dtype"), "{err}");
}
