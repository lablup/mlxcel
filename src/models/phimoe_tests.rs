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

//! Unit tests for PhiMoE.
//!
//! The config tests are checkpoint-free. The sparsemixer and fused-expert
//! tests build small `MlxArray`s and run on the default device; numeric
//! correctness of the whole forward pass is validated end-to-end against a
//! real `mlx-community` checkpoint.

use super::phimoe::{ModelArgs, sparsemixer};
use crate::models::switch_layers::{SwitchGLU, moe_weighted_sum};
use mlxcel_core::weights::WeightMap;

/// Trimmed config matching the `phi-3.5-moe-instruct` HuggingFace checkpoint
/// at `mlx-community/Phi-3.5-MoE-instruct-4bit`. The checkpoint spells the
/// norm epsilon `rms_norm_eps` (the key mlx-lm's `phimoe.py` reads), not
/// `layer_norm_eps` (issue #2465).
const PHI35_MOE_4BIT_CONFIG: &str = r#"{
    "model_type": "phimoe",
    "vocab_size": 32064,
    "hidden_size": 4096,
    "intermediate_size": 6400,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 16,
    "num_experts_per_tok": 2,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

/// Unquantized (bf16) config to confirm the quantization defaults.
const PHI35_MOE_BF16_CONFIG: &str = r#"{
    "model_type": "phimoe",
    "vocab_size": 32064,
    "hidden_size": 4096,
    "intermediate_size": 6400,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 16,
    "num_experts_per_tok": 2,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0
}"#;

#[test]
fn phimoe_4bit_config_parses() {
    let args: ModelArgs =
        serde_json::from_str(PHI35_MOE_4BIT_CONFIG).expect("parse 4-bit phimoe config");
    assert_eq!(args.model_type, "phimoe");
    assert_eq!(args.vocab_size, 32064);
    assert_eq!(args.hidden_size, 4096);
    assert_eq!(args.intermediate_size, 6400);
    assert_eq!(args.num_hidden_layers, 32);
    assert_eq!(args.num_attention_heads, 32);
    assert_eq!(args.num_key_value_heads, 8);
    assert_eq!(args.num_local_experts, 16);
    assert_eq!(args.num_experts_per_tok, 2);
    assert_eq!(args.layer_norm_eps, 1e-5);
    assert_eq!(args.rope_theta, 10000.0);
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}

#[test]
fn phimoe_head_dim_derives_correctly() {
    let args: ModelArgs =
        serde_json::from_str(PHI35_MOE_4BIT_CONFIG).expect("parse 4-bit phimoe config");
    // 4096 hidden / 32 attention heads = 128 head_dim
    assert_eq!(args.head_dim(), 128);
}

#[test]
fn phimoe_num_kv_heads_accessor() {
    let args: ModelArgs =
        serde_json::from_str(PHI35_MOE_4BIT_CONFIG).expect("parse 4-bit phimoe config");
    assert_eq!(args.num_kv_heads(), 8);
}

#[test]
fn phimoe_unquantized_config_uses_quantization_defaults() {
    let args: ModelArgs =
        serde_json::from_str(PHI35_MOE_BF16_CONFIG).expect("parse bf16 phimoe config");
    assert!(args.quantization.is_none());
    // Defaults: group_size=64, bits=4
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}

/// The `layer_norm_eps` spelling stays accepted for configs that use it
/// (issue #2465).
#[test]
fn phimoe_layer_norm_eps_spelling_still_accepted() {
    let args: ModelArgs = serde_json::from_str(
        r#"{
            "model_type": "phimoe",
            "layer_norm_eps": 1e-5
        }"#,
    )
    .expect("parse layer_norm_eps spelling");
    assert_eq!(args.layer_norm_eps, 1e-5);
}

/// Neither key: the 1e-6 default stands (issue #2465).
#[test]
fn phimoe_norm_eps_defaults_without_either_key() {
    let args: ModelArgs =
        serde_json::from_str(r#"{ "model_type": "phimoe" }"#).expect("parse minimal config");
    assert_eq!(args.layer_norm_eps, 1e-6);
}

// ── sparsemixer routing (#2379) ─────────────────────────────────────────────

fn values_f32(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    let arr = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&arr);
    mlxcel_core::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn values_i32(arr: &mlxcel_core::MlxArray) -> Vec<i32> {
    let arr = mlxcel_core::astype(arr, mlxcel_core::dtype::INT32);
    mlxcel_core::eval(&arr);
    mlxcel_core::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Upstream's case (ml-explore/mlx-lm#1939). Row 0 has a clear winner each
/// round, so both scores are 1.0. Row 1's runner-up sits within `2 * eps` of
/// the top, so the first round's softmax is shared between them (0.5025) and
/// the second round is alone again (1.0). Softmax over the selected logits,
/// the routing this replaced, would give [0.881, 0.119] and [0.5025, 0.4975].
#[test]
fn sparsemixer_matches_the_reference_case() {
    let gates = mlxcel_core::from_slice_f32(&[3.0, 1.0, 0.0, -1.0, 3.0, 2.99, 0.0, -1.0], &[2, 4]);
    let (indices, scores) = sparsemixer(&gates, 2, 0.01);
    assert_eq!(mlxcel_core::array_shape(&indices), vec![2, 2]);
    assert_eq!(values_i32(&indices), vec![0, 1, 0, 1]);
    let scores = values_f32(&scores);
    for (got, want) in scores.iter().zip([1.0f32, 1.0, 0.5025, 1.0]) {
        assert!((got - want).abs() < 1e-4, "scores {scores:?}");
    }
}

/// Larger `top_k` never picks an expert twice, and a token-major `[n, E]`
/// input routes each row independently.
#[test]
fn sparsemixer_picks_distinct_experts() {
    let gates = mlxcel_core::from_slice_f32(&[0.5, 2.0, -1.0, 1.9, 0.1, 0.2, 0.3, 0.4], &[2, 4]);
    let (indices, _) = sparsemixer(&gates, 4, 0.01);
    for row in values_i32(&indices).chunks(4) {
        let mut sorted = row.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3], "row {row:?}");
    }
}

/// The fused single-token expert kernel must weight each expert by the
/// sparsemixer score as given. The scores here sum to 1.5025, so a kernel that
/// renormalized them would land 1.5x away from the graph path.
#[test]
fn fused_expert_kernel_keeps_sparsemixer_scores() {
    if !mlxcel_core::fused_moe_kernels_available() {
        eprintln!("skipping: no fused MoE kernel on this backend");
        return;
    }
    const E: i32 = 4;
    const DIN: i32 = 256;
    const DFF: i32 = 128;
    const GS: i32 = 64;
    const BITS: i32 = 4;

    // Deterministic host-side weights so the test does not depend on the RNG.
    let pseudo = |n: usize, seed: f32| -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.618_034 + seed).sin() * 0.05)
            .collect()
    };
    let mut weights = WeightMap::new();
    for (name, rows, cols, seed) in [
        ("gate_proj", DFF, DIN, 0.1f32),
        ("up_proj", DFF, DIN, 0.7),
        ("down_proj", DIN, DFF, 1.3),
    ] {
        let full = mlxcel_core::from_slice_f32(
            &pseudo((E * rows * cols) as usize, seed),
            &[E * rows, cols],
        );
        let full = mlxcel_core::astype(&full, mlxcel_core::dtype::BFLOAT16);
        let w = mlxcel_core::quantize_weights_w(&full, GS, BITS);
        let sc = mlxcel_core::quantize_weights_scales(&full, GS, BITS);
        let bi = mlxcel_core::quantize_weights_biases(&full, GS, BITS);
        let packed = *mlxcel_core::array_shape(&w).last().unwrap();
        let groups = *mlxcel_core::array_shape(&sc).last().unwrap();
        let prefix = format!("moe.switch_mlp.{name}");
        weights.insert(
            format!("{prefix}.weight"),
            mlxcel_core::reshape(&w, &[E, rows, packed]),
        );
        weights.insert(
            format!("{prefix}.scales"),
            mlxcel_core::reshape(&sc, &[E, rows, groups]),
        );
        weights.insert(
            format!("{prefix}.biases"),
            mlxcel_core::reshape(&bi, &[E, rows, groups]),
        );
    }
    let experts = SwitchGLU::from_weights(&weights, "moe.switch_mlp", GS, BITS)
        .expect("quantized experts load");

    let x = mlxcel_core::from_slice_f32(&pseudo(DIN as usize, 2.9), &[1, DIN]);
    let x = mlxcel_core::astype(
        &mlxcel_core::multiply_scalar(&x, 20.0),
        mlxcel_core::dtype::BFLOAT16,
    );
    let gates = mlxcel_core::from_slice_f32(&[3.0, 2.99, 0.0, -1.0], &[1, 4]);
    let (indices, scores) = sparsemixer(&gates, 2, 0.01);
    let scores_host = values_f32(&scores);
    assert!((scores_host.iter().sum::<f32>() - 1.5025).abs() < 1e-3);

    let fused = experts
        .forward_fused_kernel(&x, &indices, &scores)
        .expect("fused kernel accepts this shape");
    let graph = moe_weighted_sum(
        &experts.forward(&x, &indices),
        &scores,
        mlxcel_core::dtype::BFLOAT16,
    );
    let fused = values_f32(&fused);
    let graph = values_f32(&graph);
    assert_eq!(fused.len(), graph.len());
    let norm = graph.iter().map(|v| v * v).sum::<f32>().sqrt();
    let diff = fused
        .iter()
        .zip(&graph)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    assert!(norm > 0.0, "degenerate expert output");
    assert!(
        diff / norm < 0.02,
        "fused vs graph relative error {}",
        diff / norm
    );
}

// ── LongRoPE and routing config (#2379) ─────────────────────────────────────

/// The `rope_scaling` block `mlx-community/Phi-3.5-MoE-instruct-4bit` ships,
/// with the factor lists shortened to a recognizable pattern.
fn longrope_config(rope_scaling: &str) -> String {
    format!(
        r#"{{
            "model_type": "phimoe",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_local_experts": 16,
            "num_experts_per_tok": 2,
            "max_position_embeddings": 131072,
            "original_max_position_embeddings": 4096,
            "router_jitter_noise": 0.01,
            "rope_theta": 10000.0,
            "rope_scaling": {rope_scaling}
        }}"#
    )
}

fn longrope_block() -> String {
    let short: Vec<String> = (0..64).map(|_| "1.0".to_string()).collect();
    let long: Vec<String> = (0..64).map(|i| format!("{}", 1.0 + i as f64)).collect();
    format!(
        r#"{{"type": "longrope", "short_factor": [{}], "long_factor": [{}],
            "short_mscale": 1.243163121016122, "long_mscale": 1.243163121016122,
            "original_max_position_embeddings": 4096}}"#,
        short.join(","),
        long.join(",")
    )
}

#[test]
fn longrope_block_builds_both_tables() {
    let args: ModelArgs = serde_json::from_str(&longrope_config(&longrope_block())).unwrap();
    assert_eq!(args.max_position_embeddings, 131072);
    assert_eq!(args.original_max_position_embeddings(), 4096);
    assert!((args.router_jitter_noise - 0.01).abs() < 1e-9);

    let su = args.su_rope().expect("a longrope block builds SuRope");
    // The regime threshold is the trained context.
    assert_eq!(su.regime(4096), 0);
    assert_eq!(su.regime(4097), 1);

    // A sequence inside the trained context rotates with the short table and
    // one past it with the long table; both carry the configured mscale.
    let short = su.table_for(0, 512);
    let long = su.table_for(4096, 8);
    assert!((short.scale - 1.243_163_1).abs() < 1e-5);
    assert!((long.scale - 1.243_163_1).abs() < 1e-5);
    let short_freqs = values_f32(&short.freqs);
    let long_freqs = values_f32(&long.freqs);
    assert_eq!(short_freqs.len(), 64);
    assert_eq!(long_freqs.len(), 64);
    // freq[i] = factor[i] * theta^(2i / 128): short factors are all 1.0, long
    // factor i is 1 + i.
    for i in [0usize, 1, 33, 63] {
        let base = 10000f32.powf(2.0 * i as f32 / 128.0);
        assert!((short_freqs[i] - base).abs() / base < 1e-5, "short {i}");
        let want = (1.0 + i as f32) * base;
        assert!((long_freqs[i] - want).abs() / want < 1e-5, "long {i}");
    }
}

#[test]
fn null_or_absent_rope_scaling_keeps_plain_rope() {
    let null: ModelArgs = serde_json::from_str(&longrope_config("null")).unwrap();
    assert!(null.rope_scaling.is_none());
    assert!(null.su_rope().is_none());

    // The trimmed fixture carries no `rope_scaling` key (the real
    // Phi-3.5-MoE checkpoint ships `longrope`; this stands in for Phi-mini-MoE).
    let absent: ModelArgs = serde_json::from_str(PHI35_MOE_4BIT_CONFIG).unwrap();
    assert!(absent.su_rope().is_none());
    assert_eq!(absent.max_position_embeddings, 131072);
    assert!((absent.router_jitter_noise - 0.01).abs() < 1e-9);
}

#[test]
fn more_experts_per_token_than_experts_is_rejected() {
    let mut args: ModelArgs = serde_json::from_str(PHI35_MOE_4BIT_CONFIG).unwrap();
    assert!(args.validate().is_ok());
    args.num_experts_per_tok = 17;
    let err = args.validate().unwrap_err();
    assert!(err.contains("num_experts_per_tok (17)"), "{err}");
    // Zero rounds would leave sparsemixer nothing to concatenate.
    args.num_experts_per_tok = 0;
    assert!(args.validate().is_err());
}
