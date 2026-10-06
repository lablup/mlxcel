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

//! Unit tests for Apertus (Swiss AI) config parsing and the xIELU activation
//! math.
//!
//! These cover the checkpoint-free surface: that `config.json` deserializes
//! into `ModelArgs` with the Apertus deltas (xIELU `qk_norm`/`post_norm`
//! flags, llama3 `rope_scaling`, untied embeddings) and that the pure-scalar
//! activation pieces (`softplus`, the xIELU branch formula) match
//! hand-computed values. `fused_xielu_kernel_matches_graph_every_dtype` adds
//! the GPU check of the fused xIELU kernel against the MLX op path
//! (`apertus_xielu`) in f32, f16 and bf16. End-to-end generation is validated
//! against a real `mlx-community` Apertus checkpoint, which is not exercised
//! here.

use super::apertus::{ModelArgs, apertus_xielu, softplus};
use mlxcel_core::dtype;
use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};

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

/// The MLX `apertus_xielu` path factors the shared `beta * x` term out of both
/// branches and adds it once after the per-element select:
///   `result = where(x > 0, alpha_p*x^2, (expm1(min(x,eps))-x)*alpha_n) + beta*x`
/// rather than adding `beta * x` inside each branch. This mirrors that factored
/// structure on scalars and asserts it equals the branch-local reference
/// (`xielu_scalar`) bit-for-bit in f32, which is the algebraic guarantee behind
/// "greedy temp-0 output unchanged": the select only chooses which value
/// `beta * x` is added to, so factoring the add out cannot change any result.
fn xielu_scalar_factored(x: f32, alpha_p_raw: f32, alpha_n_raw: f32, beta: f32, eps: f32) -> f32 {
    let alpha_p = softplus(alpha_p_raw);
    let alpha_n = beta + softplus(alpha_n_raw);
    let core = if x > 0.0 {
        alpha_p * x * x
    } else {
        let clamped = x.min(eps);
        ((clamped.exp() - 1.0) - x) * alpha_n
    };
    core + beta * x
}

#[test]
fn xielu_factored_beta_x_matches_branch_local() {
    let beta = 0.5f32;
    let eps = -1e-6f32;
    let raw = (0.8f32.exp() - 1.0).ln();

    // Across positive, negative, and zero inputs the factored form (beta * x
    // added once after the select) must equal the branch-local form bit-for-bit.
    for &x in &[-4.0f32, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0, 12.5] {
        let branch_local = xielu_scalar(x, raw, raw, beta, eps);
        let factored = xielu_scalar_factored(x, raw, raw, beta, eps);
        assert_eq!(
            branch_local.to_bits(),
            factored.to_bits(),
            "factored xIELU diverged at x={x}: branch_local={branch_local}, factored={factored}"
        );
    }
}

/// Normalized (RMS, max) deviation budget per dtype, the one
/// `fused_norm_parity_tests.rs` uses: about one ulp of the RMS element on
/// average, a few on the largest.
fn xielu_tolerance(dt: i32) -> (f64, f64) {
    if dt == dtype::FLOAT32 {
        (1e-6, 1e-5)
    } else if dt == dtype::FLOAT16 {
        (2e-3, 1.2e-2)
    } else {
        (1.6e-2, 7e-2)
    }
}

fn as_f32_vec(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(arr, dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The fused xIELU kernel (Metal, and ROCm since #2069) against the
/// elementwise graph it replaces, on the same backend and the same inputs, in
/// f32, f16 and bf16: within the `fused_norm_parity_tests.rs` budget, and
/// reported as byte-identical or not. Inputs cover the edge values of
/// `fused_xielu_matches_elementwise_bit_for_bit` plus a pseudo-random spread
/// over [-12, 12), where `alpha_p * x^2` still fits in f16. On ROCm every
/// dtype must also be byte-identical.
#[test]
fn fused_xielu_kernel_matches_graph_every_dtype() {
    let backend = gpu_backend_kind();
    if matches!(backend, GpuBackendKind::Metal | GpuBackendKind::Rocm) {
        assert!(
            mlxcel_core::fused_xielu_kernel_available(),
            "{backend:?} has a fused xIELU port, so the kernel path must be taken"
        );
    } else {
        eprintln!("skipping: {backend:?} has no fused xIELU port (elementwise fallback)");
        return;
    }
    let (alpha_p, alpha_n, beta, eps) = (0.8731f32, 0.6042f32, 0.5f32, -1e-6f32);
    let mut vals: Vec<f32> = vec![
        0.0, -0.0, 1e-7, -1e-7, 5e-7, -5e-7, 1e-6, -1e-6, 2e-6, -2e-6, 1e-4, -1e-4, 0.001, -0.001,
        0.5, -0.5, 1.0, -1.0, 2.5, -2.5, 8.0, -8.0, 42.0, -42.0, 90.0, -90.0,
    ];
    let mut state = 0x2069_u64;
    for _ in 0..16384 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
        vals.push(unit * 24.0 - 12.0);
    }
    let n = vals.len() as i32;
    let x_f32 = mlxcel_core::from_slice_f32(&vals, &[1, 1, n]);
    for dt in [dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
        let x = mlxcel_core::astype(&x_f32, dt);
        let graph = apertus_xielu(&x, alpha_p, alpha_n, beta, eps);
        let fused = mlxcel_core::fused_xielu(&x, alpha_p, alpha_n, beta, eps)
            .expect("fused_xielu falls back rather than refusing");
        assert_eq!(mlxcel_core::array_dtype(&fused), dt);
        let (g, f) = (as_f32_vec(&graph), as_f32_vec(&fused));
        let mut diff_sq = 0f64;
        let mut ref_sq = 0f64;
        let mut max_abs = 0f64;
        let mut differing = 0usize;
        for (a, b) in f.iter().zip(&g) {
            let d = (*a as f64) - (*b as f64);
            diff_sq += d * d;
            ref_sq += (*b as f64) * (*b as f64);
            max_abs = max_abs.max(d.abs());
            differing += usize::from(a.to_bits() != b.to_bits());
        }
        let ref_rms = (ref_sq / g.len() as f64).sqrt().max(1e-20);
        let nrms = (diff_sq / g.len() as f64).sqrt() / ref_rms;
        let nmax = max_abs / ref_rms;
        println!(
            "xielu parity [{backend:?}, dtype {dt}]: nrms={nrms:e} nmax={nmax:e}, \
             {differing} of {n} elements differ in bits"
        );
        let (rms_budget, max_budget) = xielu_tolerance(dt);
        assert!(
            nrms < rms_budget && nmax < max_budget,
            "fused xIELU deviates from the graph (dtype {dt}): nrms={nrms:e} nmax={nmax:e}"
        );
        // ROCm's kernel rounds each intermediate where the ROCm graph does,
        // so it is held to byte identity in every dtype (the claim
        // docs/environment-variables.md makes). Metal is pinned to identity
        // for bf16 by `apertus::tests::fused_xielu_matches_elementwise_bit_for_bit`
        // and to the tolerance above for f32 and f16.
        if backend == GpuBackendKind::Rocm {
            assert_eq!(
                differing, 0,
                "fused xIELU on ROCm must be byte-identical to the graph (dtype {dt}): \
                 {differing} of {n} elements differ"
            );
        }
    }
}
