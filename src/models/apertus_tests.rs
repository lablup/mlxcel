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

//! Unit tests for Apertus (Swiss AI) config parsing and the xIELU activation
//! math.
//!
//! These cover the checkpoint-free surface: that `config.json` deserializes
//! into `ModelArgs` with the Apertus deltas (xIELU `qk_norm`/`post_norm`
//! flags, llama3 `rope_scaling`, untied embeddings) and that the pure-scalar
//! activation pieces (`softplus`, the xIELU branch formula) match
//! hand-computed values. The MLX op path (`apertus_xielu`) and end-to-end
//! generation are validated against a real `mlx-community` Apertus checkpoint,
//! which needs a Metal device and is not exercised here.

use super::apertus::{ModelArgs, softplus};

/// A trimmed `Apertus-8B-Instruct-2509` config, with the fields the loader
/// reads. Mirrors the real checkpoint's `config.json`.
const APERTUS_8B_CONFIG: &str = r#"{
    "model_type": "apertus",
    "attention_bias": false,
    "hidden_size": 4096,
    "intermediate_size": 21504,
    "max_position_embeddings": 65536,
    "mlp_bias": false,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "post_norm": false,
    "qk_norm": true,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
        "type": "llama3"
    },
    "rope_theta": 12000000,
    "tie_word_embeddings": false,
    "vocab_size": 131072,
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

#[test]
fn apertus_config_parses_core_fields() {
    let args: ModelArgs = serde_json::from_str(APERTUS_8B_CONFIG).expect("parse apertus config");
    assert_eq!(args.model_type, "apertus");
    assert_eq!(args.hidden_size, 4096);
    assert_eq!(args.num_hidden_layers, 32);
    assert_eq!(args.intermediate_size, 21504);
    assert_eq!(args.num_attention_heads, 32);
    assert_eq!(args.num_key_value_heads, 8);
    assert_eq!(args.vocab_size, 131072);
    assert_eq!(args.rope_theta, 12_000_000.0);
}

#[test]
fn apertus_delta_flags_parse() {
    let args: ModelArgs = serde_json::from_str(APERTUS_8B_CONFIG).expect("parse apertus config");
    // QK-norm on, post_norm off, embeddings untied for Apertus-8B.
    assert!(args.qk_norm);
    assert!(!args.post_norm);
    assert!(!args.tie_word_embeddings);
}

#[test]
fn apertus_head_dim_derives_from_hidden_and_heads() {
    let args: ModelArgs = serde_json::from_str(APERTUS_8B_CONFIG).expect("parse apertus config");
    // head_dim is omitted from the config, so it derives from hidden/heads.
    assert_eq!(args.head_dim(), 4096 / 32);
    assert_eq!(args.head_dim(), 128);
}

#[test]
fn apertus_rope_scaling_llama3_fields_parse() {
    let args: ModelArgs = serde_json::from_str(APERTUS_8B_CONFIG).expect("parse apertus config");
    let scaling = args
        .rope_scaling
        .as_ref()
        .expect("apertus ships rope_scaling");
    assert_eq!(
        scaling.get("rope_type").and_then(|v| v.as_str()),
        Some("llama3")
    );
    assert_eq!(scaling.get("factor").and_then(|v| v.as_f64()), Some(8.0));
    assert_eq!(
        scaling.get("low_freq_factor").and_then(|v| v.as_f64()),
        Some(1.0)
    );
    assert_eq!(
        scaling.get("high_freq_factor").and_then(|v| v.as_f64()),
        Some(4.0)
    );
    assert_eq!(
        scaling
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_u64()),
        Some(8192)
    );
}

#[test]
fn apertus_quantization_is_read_from_config() {
    let args: ModelArgs = serde_json::from_str(APERTUS_8B_CONFIG).expect("parse apertus config");
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}

#[test]
fn apertus_optional_flags_default_when_absent() {
    // A minimal config drops the delta flags and quantization; the loader must
    // fall back to safe defaults (qk_norm off, post_norm off, untied, 4-bit/64).
    let minimal = r#"{
        "model_type": "apertus",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-05,
        "vocab_size": 100
    }"#;
    let args: ModelArgs = serde_json::from_str(minimal).expect("parse minimal apertus config");
    assert!(!args.qk_norm);
    assert!(!args.post_norm);
    assert!(!args.tie_word_embeddings);
    assert!(args.quantization.is_none());
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
    assert_eq!(args.rope_theta, 10_000.0); // default when omitted
    assert!(args.rope_scaling.is_none());
}

#[test]
fn softplus_matches_reference() {
    // softplus(0) = ln(2).
    assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-6);

    // Round-trip: the checkpoint stores alpha in inverse-softplus form, so
    // softplus(log(exp(a) - 1)) recovers `a` (here a = 0.8, the XieLU init).
    let raw = (0.8f32.exp() - 1.0).ln();
    assert!((softplus(raw) - 0.8).abs() < 1e-5);

    // Large inputs stay numerically stable (softplus(x) -> x).
    assert!((softplus(40.0) - 40.0).abs() < 1e-3);
}

/// Pure-scalar reference for the xIELU branch formula, sharing the real
/// `softplus`. `apertus_xielu` runs the same math on MLX arrays; this checks
/// the algebra (and the `alpha_n = beta + softplus(raw)` offset) on the CPU
/// without a Metal device.
fn xielu_scalar(x: f32, alpha_p_raw: f32, alpha_n_raw: f32, beta: f32, eps: f32) -> f32 {
    let alpha_p = softplus(alpha_p_raw);
    let alpha_n = beta + softplus(alpha_n_raw);
    if x > 0.0 {
        alpha_p * x * x + beta * x
    } else {
        let clamped = x.min(eps);
        // expm1(clamped) = exp(clamped) - 1.
        ((clamped.exp() - 1.0) - x) * alpha_n + beta * x
    }
}

#[test]
fn xielu_formula_matches_hand_values() {
    let beta = 0.5f32;
    let eps = -1e-6f32;
    // Choose raw scalars so softplus(raw) = 0.8 for both alpha_p and alpha_n's
    // softplus term; then alpha_p = 0.8 and alpha_n = beta + 0.8 = 1.3.
    let raw = (0.8f32.exp() - 1.0).ln();

    // Positive branch: alpha_p * x^2 + beta * x = 0.8 * 4 + 0.5 * 2 = 4.2.
    let pos = xielu_scalar(2.0, raw, raw, beta, eps);
    assert!((pos - 4.2).abs() < 1e-4, "positive branch: got {pos}");

    // Negative branch at x = -1.0, eps ~= 0 so clamped = -1.0:
    // (exp(-1) - 1 - (-1)) * 1.3 + 0.5 * (-1)
    //   = (exp(-1)) * 1.3 - 0.5.
    let expected_neg = (-1.0f32).exp() * 1.3 - 0.5;
    let neg = xielu_scalar(-1.0, raw, raw, beta, eps);
    assert!(
        (neg - expected_neg).abs() < 1e-4,
        "negative branch: got {neg}, expected {expected_neg}"
    );

    // Continuity-ish sanity: at x = 0 the activation is exactly 0
    // (negative branch, clamped = eps, expm1(eps) ~= eps, so the term is tiny).
    let at_zero = xielu_scalar(0.0, raw, raw, beta, eps);
    assert!(
        at_zero.abs() < 1e-3,
        "x=0 should be near zero: got {at_zero}"
    );
}
