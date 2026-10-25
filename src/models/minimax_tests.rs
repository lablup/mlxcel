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

//! Unit tests for the MiniMax model config and fused decode-MoE dispatch.
//!
//! These cover the checkpoint-free surface: config parsing, quantization
//! accessors, and the gating conditions for the fused single-token decode
//! kernel (#268 / #304). The MLX op path and end-to-end generation require a
//! Metal device and a fitting checkpoint (MiniMax-Text-01 ~456B exceeds 128 GB;
//! runtime validation is deferred until a smaller variant or larger-memory
//! machine is available).

use super::minimax::{MiniMaxModel, ModelArgs};
use crate::models::switch_layers::fused_moe_enabled_from;
use mlxcel_core::weights::WeightMap;

/// Trimmed config for a quantized MiniMax-M2 style checkpoint.
const MINIMAX_CONFIG_4BIT: &str = r#"{
    "model_type": "minimax_text_01",
    "vocab_size": 200064,
    "hidden_size": 6144,
    "intermediate_size": 16384,
    "num_hidden_layers": 80,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "num_experts_per_tok": 8,
    "num_local_experts": 256,
    "rms_norm_eps": 1e-05,
    "rope_theta": 4000000.0,
    "rotary_dim": 64,
    "head_dim": 128,
    "tie_word_embeddings": false,
    "use_qk_norm": true,
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

/// Trimmed config for a non-quantized MiniMax checkpoint (bf16/f16 weights).
const MINIMAX_CONFIG_UNQUANT: &str = r#"{
    "model_type": "minimax_text_01",
    "vocab_size": 200064,
    "hidden_size": 6144,
    "intermediate_size": 16384,
    "num_hidden_layers": 80,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "num_experts_per_tok": 8,
    "num_local_experts": 256,
    "rms_norm_eps": 1e-05,
    "rope_theta": 4000000.0,
    "rotary_dim": 64,
    "head_dim": 128,
    "tie_word_embeddings": false,
    "use_qk_norm": true
}"#;

#[test]
fn parses_moe_topology_fields() {
    let args: ModelArgs = serde_json::from_str(MINIMAX_CONFIG_4BIT).expect("parse minimax config");
    assert_eq!(args.model_type, "minimax_text_01");
    assert_eq!(args.num_hidden_layers, 80);
    assert_eq!(args.num_local_experts, 256);
    assert_eq!(args.num_experts_per_tok, 8);
    assert_eq!(args.rms_norm_eps, 1e-5);
    assert!(args.use_qk_norm);
    assert!(!args.tie_word_embeddings);
}

#[test]
fn quantization_accessors_resolve_correctly() {
    let args: ModelArgs = serde_json::from_str(MINIMAX_CONFIG_4BIT).expect("parse minimax config");
    // Expert projections use the top-level 4-bit default.
    assert_eq!(args.bits(), 4);
    assert_eq!(args.group_size(), 64);
    // MiniMax gate (router) uses 8-bit regardless of the expert bit width.
    assert_eq!(args.gate_bits(), 8);
}

#[test]
fn unquantized_config_gate_bits_equals_bits() {
    let args: ModelArgs =
        serde_json::from_str(MINIMAX_CONFIG_UNQUANT).expect("parse minimax unquant config");
    // Without a quantization block, bits() falls back to 4 and gate_bits() to bits().
    assert_eq!(args.bits(), 4);
    assert_eq!(args.gate_bits(), 4);
}

#[test]
fn rope_and_head_dim_parse() {
    let args: ModelArgs = serde_json::from_str(MINIMAX_CONFIG_4BIT).expect("parse minimax config");
    assert_eq!(args.head_dim, 128);
    assert_eq!(args.rotary_dim, 64);
    assert_eq!(args.rope_theta, 4_000_000.0_f32);
}

/// Confirms the fused-kernel gate conditions that `SparseMoeBlock::forward`
/// evaluates before attempting `forward_fused_kernel`.
///
/// The dispatch branch is:
///   `if mlxcel_core::array_shape(&x_flat)[0] == 1 && fused_moe_enabled()`
///
/// The test verifies:
/// - `fused_moe_enabled_from(None)` returns true (default-on as of #282), so a
///   single-token input on an unset `MLXCEL_FUSED_MOE` will attempt the kernel.
/// - `fused_moe_enabled_from(Some("0"))` returns false, so the gate is skipped
///   and the `SwitchGLU` + `moe_weighted_sum` fallback is taken regardless of
///   token count.
/// - The multi-token guard (`shape[0] != 1`) independently blocks the kernel,
///   independently of the env flag, which is modelled here by checking that the
///   compound `n_tokens == 1 && enabled` evaluates correctly for the two cases.
#[test]
fn fused_dispatch_gate_conditions_for_minimax() {
    // Single-token + default env: kernel is attempted.
    let n_tokens: i32 = 1;
    let fused_on = fused_moe_enabled_from(None);
    assert!(
        n_tokens == 1 && fused_on,
        "single-token + default env should attempt the fused kernel"
    );

    // Single-token + MLXCEL_FUSED_MOE=0: kernel is skipped.
    let fused_off = fused_moe_enabled_from(Some("0"));
    assert!(
        !(n_tokens == 1 && fused_off),
        "single-token + MLXCEL_FUSED_MOE=0 must skip the fused kernel"
    );

    // Multi-token prefill + default env: kernel is skipped regardless of env.
    let n_tokens_prefill: i32 = 128;
    assert!(
        !(n_tokens_prefill == 1 && fused_on),
        "prefill (n_tokens > 1) must skip the fused kernel"
    );
}

fn tiny_args() -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "minimax_text_01",
        "vocab_size": 24,
        "hidden_size": 8,
        "intermediate_size": 6,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_experts_per_tok": 2,
        "num_local_experts": 4,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rotary_dim": 2,
        "head_dim": 4,
        "tie_word_embeddings": false,
        "use_qk_norm": true
    }))
    .unwrap()
}

fn tensor(shape: &[i32], phase: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as usize + phase * 7) as f32).cos() * 0.055)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let h = args.hidden_size as i32;
    let q = args.num_attention_heads as i32 * args.head_dim as i32;
    let kv = args.num_key_value_heads as i32 * args.head_dim as i32;
    let e = args.num_local_experts as i32;
    let d = args.intermediate_size as i32;
    let mut weights = WeightMap::new();
    let mut phase = 1;
    let mut put = |key: String, shape: &[i32]| {
        weights.insert(key, tensor(shape, phase));
        phase += 1;
    };
    put(
        "model.embed_tokens.weight".into(),
        &[args.vocab_size as i32, h],
    );
    put("model.norm.weight".into(), &[h]);
    put("lm_head.weight".into(), &[args.vocab_size as i32, h]);
    for layer in 0..args.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        let a = format!("{p}.self_attn");
        put(format!("{a}.q_proj.weight"), &[q, h]);
        put(format!("{a}.k_proj.weight"), &[kv, h]);
        put(format!("{a}.v_proj.weight"), &[kv, h]);
        put(format!("{a}.o_proj.weight"), &[h, q]);
        put(format!("{a}.q_norm.weight"), &[q]);
        put(format!("{a}.k_norm.weight"), &[kv]);
        put(format!("{p}.input_layernorm.weight"), &[h]);
        put(format!("{p}.post_attention_layernorm.weight"), &[h]);
        let m = format!("{p}.block_sparse_moe");
        put(format!("{m}.gate.weight"), &[e, h]);
        put(format!("{m}.e_score_correction_bias"), &[e]);
        put(format!("{m}.switch_mlp.gate_proj.weight"), &[e, d, h]);
        put(format!("{m}.switch_mlp.up_proj.weight"), &[e, d, h]);
        put(format!("{m}.switch_mlp.down_proj.weight"), &[e, h, d]);
    }
    weights
}

#[test]
fn minimax_native_batched_decode_matches_three_routed_rows() {
    let args = tiny_args();
    let model = MiniMaxModel::from_weights(&tiny_weights(&args), &args).unwrap();
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        &model,
        &[&[1, 2], &[3, 4, 5], &[6, 7, 8, 9]],
        &[10, 11, 12],
        3e-4,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
        &model,
        &[2, 4, 6],
        8,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(&model);
}
