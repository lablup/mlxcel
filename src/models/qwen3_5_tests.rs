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

//! Unit tests for the Qwen 3.5 DFlash speculative-decoding hooks.
//!
//! These tests cover the *standalone* helpers — `rebuild_with_zero_tail`,
//! `zero_per_row_kv_tail`, and `sanitize_weights`'s MTP-stripping path — without
//! requiring a real model checkpoint. The end-to-end `forward_speculative` +
//! `rollback_speculative_cache` round trip is exercised by integration tests
//! that need a real Qwen 3.5 model and are gated behind hardware availability.

use super::qwen3_5::{
    MRopePositionSource, Qwen35Config, Qwen35UnsupportedConfig, mrope_position_source,
    rebuild_with_zero_tail, sanitize_weights, validate_qwen35_wrapper_config, zero_per_row_kv_tail,
};
use mlxcel_core::dtype;
use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;

fn assert_allclose(actual: &mlxcel_core::MlxArray, expected: &mlxcel_core::MlxArray) {
    let close = mlxcel_core::allclose(actual, expected, 1e-3, 1e-3);
    mlxcel_core::eval(&close);
    assert!(
        mlxcel_core::item_bool(&close),
        "tensors differ beyond tolerance"
    );
}

/// Build a synthetic Qwen 3.5 config with the smallest valid shape so we
/// can exercise the weight-sanitizer's MTP-stripping path without loading
/// a real checkpoint.
fn make_tiny_config() -> Qwen35Config {
    Qwen35Config {
        model_type: "qwen3_5".to_string(),
        hidden_size: 8,
        num_hidden_layers: 2,
        intermediate_size: 16,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: Some(4),
        linear_num_value_heads: 2,
        linear_num_key_heads: 2,
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        norm_topk_prob: true,
        rope_parameters: None,
        full_attention_interval: 4,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
        attention_bias: false,
        vocab_size: 32,
        quantization: None,
        mlp_only_layers: vec![],
        output_gate_type: None,
    }
}

// ---------------------------------------------------------------------------
// `rebuild_with_zero_tail` — the row-fixated KV tail zeroing primitive used
// during per-row speculative rollback.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires serial MLX execution"]
fn rebuild_with_zero_tail_zeroes_only_target_row_tail() {
    // [B=2, H=2, S=4, D=2], filled with row-distinct values so any cross-
    // row corruption is immediately visible.
    let row0 = vec![1.0_f32; 16];
    let row1 = vec![2.0_f32; 16];
    let mut data = Vec::with_capacity(32);
    data.extend(row0);
    data.extend(row1);
    let tensor = mlxcel_core::from_slice_f32(&data, &[2, 2, 4, 2]);

    // Zero the tail of row 0 starting at seq index 2 (length 2 to zero out).
    let out = rebuild_with_zero_tail(&tensor, &[2, 2, 4, 2], 0, 2, 4, dtype::FLOAT32);
    mlxcel_core::eval(&out);

    // Row 0, positions 0..2 must still be 1.0; positions 2..4 must be 0.0.
    let row0_head = mlxcel_core::slice(&out, &[0, 0, 0, 0], &[1, 2, 2, 2]);
    let row0_tail = mlxcel_core::slice(&out, &[0, 0, 2, 0], &[1, 2, 4, 2]);
    let row1_all = mlxcel_core::slice(&out, &[1, 0, 0, 0], &[2, 2, 4, 2]);

    let expected_head = mlxcel_core::from_slice_f32(&[1.0_f32; 8], &[1, 2, 2, 2]);
    let expected_tail = mlxcel_core::from_slice_f32(&[0.0_f32; 8], &[1, 2, 2, 2]);
    let expected_row1 = mlxcel_core::from_slice_f32(&[2.0_f32; 16], &[1, 2, 4, 2]);

    assert_allclose(&row0_head, &expected_head);
    assert_allclose(&row0_tail, &expected_tail);
    assert_allclose(&row1_all, &expected_row1);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn rebuild_with_zero_tail_handles_last_row() {
    // 3 rows, zero the tail of row 2 (the last one) — exercises the
    // `bi + 1 < batch` skip branch.
    let mut data = Vec::with_capacity(24);
    for v in [1.0_f32, 2.0, 3.0] {
        data.extend(vec![v; 8]);
    }
    let tensor = mlxcel_core::from_slice_f32(&data, &[3, 2, 2, 2]);

    let out = rebuild_with_zero_tail(&tensor, &[3, 2, 2, 2], 2, 1, 2, dtype::FLOAT32);
    mlxcel_core::eval(&out);

    let row0 = mlxcel_core::slice(&out, &[0, 0, 0, 0], &[1, 2, 2, 2]);
    let row1 = mlxcel_core::slice(&out, &[1, 0, 0, 0], &[2, 2, 2, 2]);
    let row2_head = mlxcel_core::slice(&out, &[2, 0, 0, 0], &[3, 2, 1, 2]);
    let row2_tail = mlxcel_core::slice(&out, &[2, 0, 1, 0], &[3, 2, 2, 2]);

    assert_allclose(
        &row0,
        &mlxcel_core::from_slice_f32(&[1.0_f32; 8], &[1, 2, 2, 2]),
    );
    assert_allclose(
        &row1,
        &mlxcel_core::from_slice_f32(&[2.0_f32; 8], &[1, 2, 2, 2]),
    );
    assert_allclose(
        &row2_head,
        &mlxcel_core::from_slice_f32(&[3.0_f32; 4], &[1, 2, 1, 2]),
    );
    assert_allclose(
        &row2_tail,
        &mlxcel_core::from_slice_f32(&[0.0_f32; 4], &[1, 2, 1, 2]),
    );
}

#[test]
#[ignore = "requires serial MLX execution"]
fn rebuild_with_zero_tail_no_op_when_start_equals_kv_len() {
    // start == kv_len means "zero an empty tail" — must return a copy that
    // matches the original.
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let tensor = mlxcel_core::from_slice_f32(&data, &[1, 2, 2, 2]);
    let out = rebuild_with_zero_tail(&tensor, &[1, 2, 2, 2], 0, 2, 2, dtype::FLOAT32);
    mlxcel_core::eval(&out);
    assert_allclose(&out, &tensor);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn rebuild_with_zero_tail_preserves_input_dtype() {
    // Apple Silicon precision rule: the zeroed buffer must keep the original
    // KV dtype — promoting bf16/f16 to f32 here would silently corrupt the
    // verify-pass path. (docs/apple-silicon-precision.md.)
    let data = vec![1.0_f32; 16];
    let tensor_f32 = mlxcel_core::from_slice_f32(&data, &[2, 2, 2, 2]);
    let tensor_bf16 = mlxcel_core::astype(&tensor_f32, dtype::BFLOAT16);

    let out = rebuild_with_zero_tail(&tensor_bf16, &[2, 2, 2, 2], 0, 1, 2, dtype::BFLOAT16);
    mlxcel_core::eval(&out);

    assert_eq!(mlxcel_core::array_dtype(&out), dtype::BFLOAT16);
}

// ---------------------------------------------------------------------------
// `zero_per_row_kv_tail` — wraps `rebuild_with_zero_tail` and applies it to
// both K and V of a `KVCache`.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires serial MLX execution"]
fn zero_per_row_kv_tail_zeroes_both_k_and_v() {
    // Synthesize a KVCache with B=2, H=2, S=4, D=2; distinct rows.
    let mut kv = KVCache::new();
    let row0 = [1.0_f32; 16];
    let row1 = [3.0_f32; 16];
    let mut k_data = Vec::with_capacity(32);
    k_data.extend(row0.iter());
    k_data.extend(row1.iter());
    let v_data: Vec<f32> = k_data.iter().map(|x| x + 10.0).collect();
    let k = mlxcel_core::from_slice_f32(&k_data, &[2, 2, 4, 2]);
    let v = mlxcel_core::from_slice_f32(&v_data, &[2, 2, 4, 2]);
    kv.update(k, v);

    zero_per_row_kv_tail(&mut kv, 0, 2, 4);
    mlxcel_core::eval(kv.keys.as_ref().unwrap());
    mlxcel_core::eval(kv.values.as_ref().unwrap());

    // Row 0 tail (positions 2..4) must be zero in BOTH K and V.
    let k_tail = mlxcel_core::slice(kv.keys.as_ref().unwrap(), &[0, 0, 2, 0], &[1, 2, 4, 2]);
    let v_tail = mlxcel_core::slice(kv.values.as_ref().unwrap(), &[0, 0, 2, 0], &[1, 2, 4, 2]);
    let zero_tail = mlxcel_core::from_slice_f32(&[0.0_f32; 8], &[1, 2, 2, 2]);
    assert_allclose(&k_tail, &zero_tail);
    assert_allclose(&v_tail, &zero_tail);

    // Row 0 head (positions 0..2) and row 1 (all positions) must be unchanged.
    let k_head = mlxcel_core::slice(kv.keys.as_ref().unwrap(), &[0, 0, 0, 0], &[1, 2, 2, 2]);
    let v_head = mlxcel_core::slice(kv.values.as_ref().unwrap(), &[0, 0, 0, 0], &[1, 2, 2, 2]);
    assert_allclose(
        &k_head,
        &mlxcel_core::from_slice_f32(&[1.0_f32; 8], &[1, 2, 2, 2]),
    );
    assert_allclose(
        &v_head,
        &mlxcel_core::from_slice_f32(&[11.0_f32; 8], &[1, 2, 2, 2]),
    );

    let k_row1 = mlxcel_core::slice(kv.keys.as_ref().unwrap(), &[1, 0, 0, 0], &[2, 2, 4, 2]);
    let v_row1 = mlxcel_core::slice(kv.values.as_ref().unwrap(), &[1, 0, 0, 0], &[2, 2, 4, 2]);
    assert_allclose(
        &k_row1,
        &mlxcel_core::from_slice_f32(&[3.0_f32; 16], &[1, 2, 4, 2]),
    );
    assert_allclose(
        &v_row1,
        &mlxcel_core::from_slice_f32(&[13.0_f32; 16], &[1, 2, 4, 2]),
    );
}

#[test]
#[ignore = "requires serial MLX execution"]
fn zero_per_row_kv_tail_no_op_on_empty_cache() {
    let mut kv = KVCache::new();
    // No keys / values populated — must not panic.
    zero_per_row_kv_tail(&mut kv, 0, 0, 4);
    assert!(kv.keys.is_none());
    assert!(kv.values.is_none());
}

// ---------------------------------------------------------------------------
// `sanitize_weights` MTP stripping — the acceptance criterion that Qwen 3.5
// checkpoints' `mtp.*` weights are dropped without breaking the existing
// load path. mirrors the mlx-lm / mlx-vlm behavior.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires serial MLX execution"]
fn sanitize_weights_drops_mtp_keys() {
    let mut weights = WeightMap::new();
    // Insert a few legitimate keys and several mtp.* keys that must be stripped.
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 8], &[2, 4]),
    );
    weights.insert(
        "model.norm.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0_f32; 4], &[4]),
    );
    weights.insert(
        "mtp.layers.0.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 16], &[4, 4]),
    );
    weights.insert(
        "mtp.embed_tokens.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 8], &[2, 4]),
    );
    weights.insert(
        "lm_head.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 8], &[2, 4]),
    );

    let config = make_tiny_config();
    let sanitized = sanitize_weights(weights, &config);

    // mtp.* must be absent; legitimate keys must remain.
    assert!(
        sanitized.keys().all(|k| !k.starts_with("mtp.")),
        "mtp.* keys should have been removed; found: {:?}",
        sanitized.keys().collect::<Vec<_>>()
    );
    assert!(sanitized.contains_key("model.embed_tokens.weight"));
    assert!(sanitized.contains_key("model.norm.weight"));
    // tie_word_embeddings is false in the tiny config, so lm_head stays.
    assert!(sanitized.contains_key("lm_head.weight"));
}

#[test]
#[ignore = "requires serial MLX execution"]
fn sanitize_weights_drops_lm_head_when_tied_embeddings() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 8], &[2, 4]),
    );
    weights.insert(
        "model.norm.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0_f32; 4], &[4]),
    );
    weights.insert(
        "lm_head.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 8], &[2, 4]),
    );

    let mut config = make_tiny_config();
    config.tie_word_embeddings = true;
    let sanitized = sanitize_weights(weights, &config);

    assert!(
        !sanitized.contains_key("lm_head.weight"),
        "lm_head.weight should have been dropped when tie_word_embeddings is true"
    );
    assert!(sanitized.contains_key("model.embed_tokens.weight"));
}

// ---------------------------------------------------------------------------
// `sanitize_weights` idempotency (issue #776): the norm `+1.0` shift must be
// layout-gated on the conv1d weight shape alone. A bundled `mtp.*` tensor
// must never force a second shift on an already-converted checkpoint, and a
// genuinely raw checkpoint must still shift exactly once.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires serial MLX execution"]
fn sanitize_weights_is_idempotent_on_already_converted_checkpoint_with_stray_mtp_key() {
    let mut weights = WeightMap::new();
    // Already-converted conv1d layout: `[out, kW, 1]` (last dim == 1).
    weights.insert(
        "model.layers.0.linear_attn.conv1d.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 16], &[4, 4, 1]),
    );
    // Already-shifted norm gammas (a raw gamma of 0.5 becomes 1.5 after a
    // single +1.0 shift; a second shift would produce 2.5).
    weights.insert(
        "model.norm.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.5_f32; 4], &[4]),
    );
    weights.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.5_f32; 8], &[8]),
    );
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 8], &[2, 4]),
    );
    // A bundled MTP tensor, mirroring a future converted repo that ships the
    // MTP head alongside an already-sanitized main checkpoint.
    weights.insert(
        "mtp.layers.0.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 16], &[4, 4]),
    );

    let config = make_tiny_config();
    let sanitized = sanitize_weights(weights, &config);

    // mtp.* must still be stripped.
    assert!(sanitized.keys().all(|k| !k.starts_with("mtp.")));

    // The norm gammas must be UNCHANGED (no second shift), not 2.5.
    assert_allclose(
        sanitized.get("model.norm.weight").unwrap(),
        &mlxcel_core::from_slice_f32(&[1.5_f32; 4], &[4]),
    );
    assert_allclose(
        sanitized
            .get("model.layers.0.input_layernorm.weight")
            .unwrap(),
        &mlxcel_core::from_slice_f32(&[1.5_f32; 8], &[8]),
    );

    // The already-converted conv1d weight must not be re-transposed: shape
    // stays `[out, kW, 1]`.
    let conv1d_shape = mlxcel_core::array_shape(
        sanitized
            .get("model.layers.0.linear_attn.conv1d.weight")
            .unwrap(),
    );
    assert_eq!(conv1d_shape, vec![4, 4, 1]);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn sanitize_weights_shifts_norms_exactly_once_on_raw_layout() {
    let mut weights = WeightMap::new();
    // Raw torch/HF conv1d layout: `[out, in, kW]` (last dim = kernel size != 1).
    weights.insert(
        "model.layers.0.linear_attn.conv1d.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 16], &[4, 1, 4]),
    );
    weights.insert(
        "model.norm.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.5_f32; 4], &[4]),
    );
    weights.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.5_f32; 8], &[8]),
    );
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 8], &[2, 4]),
    );
    // A raw checkpoint that bundles the MTP head. This is the direction in which
    // dropping the `has_mtp` term could have lost coverage, so pin it: the raw
    // conv1d layout alone must still drive the shift.
    weights.insert(
        "mtp.layers.0.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 16], &[4, 4]),
    );

    let config = make_tiny_config();
    let sanitized = sanitize_weights(weights, &config);

    assert!(sanitized.keys().all(|k| !k.contains("mtp.")));

    // The norm gammas must be shifted exactly once: 0.5 + 1.0 = 1.5.
    assert_allclose(
        sanitized.get("model.norm.weight").unwrap(),
        &mlxcel_core::from_slice_f32(&[1.5_f32; 4], &[4]),
    );
    assert_allclose(
        sanitized
            .get("model.layers.0.input_layernorm.weight")
            .unwrap(),
        &mlxcel_core::from_slice_f32(&[1.5_f32; 8], &[8]),
    );

    // The raw conv1d weight must be transposed to `[out, kW, 1]`.
    let conv1d_shape = mlxcel_core::array_shape(
        sanitized
            .get("model.layers.0.linear_attn.conv1d.weight")
            .unwrap(),
    );
    assert_eq!(conv1d_shape, vec![4, 4, 1]);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn sanitize_weights_does_not_shift_norms_on_a_degenerate_conv1d_shape() {
    // A conv1d tensor of rank < 3 carries no layout information. The gate and the
    // transpose share `is_raw_conv1d_layout`, so such a tensor must read as
    // converted: nothing is transposed, and therefore nothing may be shifted.
    // Gating on `shape.last() != Some(&1)` instead would shift every norm here
    // while transposing no conv1d at all, which is the corruption issue #776 is
    // about.
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.linear_attn.conv1d.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 16], &[4, 4]),
    );
    weights.insert(
        "model.norm.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.5_f32; 4], &[4]),
    );
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0_f32; 8], &[2, 4]),
    );

    let config = make_tiny_config();
    let sanitized = sanitize_weights(weights, &config);

    assert_allclose(
        sanitized.get("model.norm.weight").unwrap(),
        &mlxcel_core::from_slice_f32(&[1.5_f32; 4], &[4]),
    );
    let conv1d_shape = mlxcel_core::array_shape(
        sanitized
            .get("model.layers.0.linear_attn.conv1d.weight")
            .unwrap(),
    );
    assert_eq!(conv1d_shape, vec![4, 4]);
}

// ---------------------------------------------------------------------------
// Config shape, pinned to the published Qwen3.8-27B checkpoint (#1163).
// ---------------------------------------------------------------------------

/// The `text_config` object of `mlx-community/Qwen3.8-27B-4bit`
/// (revision `3e6447f0`), reproduced inline so the test runs without the
/// 16 GB checkpoint. Trimmed only of the 64-entry `layer_types` array, which
/// mlxcel ignores in favour of the positional
/// `!(i + 1).is_multiple_of(full_attention_interval)` rule.
fn qwen3_8_27b_text_config() -> serde_json::Value {
    serde_json::json!({
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attn_output_gate": true,
        "bos_token_id": 248044,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 17408,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 262144,
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "output_gate_type": "swish",
        "pad_token_id": null,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default"
        },
        "tie_word_embeddings": false,
        "use_cache": true,
        "vocab_size": 248320
    })
}

/// The published `config.json` of `mlx-community/Qwen3.8-27B-4bit` minus the
/// `vision_config` and `quantization_config` objects, which are parsed by
/// other types.
fn qwen3_8_27b_wrapper_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "do_sample": true,
        "eos_token_id": [248046, 248044],
        "generation_config": {
            "bos_token_id": 248044,
            "do_sample": true,
            "eos_token_id": [248046, 248044],
            "pad_token_id": 248044,
            "temperature": 1.0,
            "top_k": 20,
            "top_p": 0.95
        },
        "image_token_id": 248056,
        "language_model_only": false,
        "model_type": "qwen3_5",
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        "temperature": 1.0,
        "text_config": qwen3_8_27b_text_config(),
        "tie_word_embeddings": false,
        "top_k": 20,
        "top_p": 0.95,
        "transformers_version": "5.8.0.dev0",
        "video_token_id": 248057,
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053
    })
}

/// Qwen3.8-27B loads on the `qwen3_5` path only because `Qwen35Config` has no
/// `deny_unknown_fields`: every key the generation added is dropped rather
/// than rejected. That is load-bearing and undocumented, so pin it. If a
/// future serde change starts rejecting `attn_output_gate`, `layer_types`,
/// `mamba_ssm_dtype`, `mtp_num_hidden_layers`, the top-level
/// `partial_rotary_factor`, or the nested `generation_config`, this test fails
/// instead of a user's load.
#[test]
fn qwen3_8_27b_text_config_parses_and_ignores_the_new_keys() {
    let config: Qwen35Config = serde_json::from_value(qwen3_8_27b_text_config())
        .expect("the published Qwen3.8-27B text_config must deserialize as Qwen35Config");

    assert_eq!(config.model_type, "qwen3_5_text");
    assert_eq!(config.hidden_size, 5120);
    assert_eq!(config.num_hidden_layers, 64);
    assert_eq!(config.num_attention_heads, 24);
    assert_eq!(config.num_key_value_heads, 4);
    assert_eq!(config.head_dim, Some(256));
    assert_eq!(config.vocab_size, 248320);
    assert_eq!(config.full_attention_interval, 4);
    assert_eq!(config.linear_num_value_heads, 48);
    assert_eq!(config.linear_num_key_heads, 16);
    assert!(!config.tie_word_embeddings);

    // `mlp_only_layers` is absent in Qwen3.8 (Qwen3.5 ships `[]`). The serde
    // default must keep it empty, and `is_moe_layer` must stay unreachable
    // because the checkpoint is dense (`num_experts` absent).
    assert!(config.mlp_only_layers.is_empty());
    assert_eq!(config.num_experts, 0);
    assert!(
        (0..config.num_hidden_layers).all(|i| !config.is_moe_layer(i)),
        "a dense checkpoint must not route any layer through the MoE branch"
    );

    // The generation added `output_gate_type`; mlxcel now reads it.
    assert_eq!(config.output_gate_type.as_deref(), Some("swish"));
    assert_eq!(config.mrope_interleaved(), Some(true));

    // `partial_rotary_factor` is duplicated at the text_config top level and
    // inside `rope_parameters`. mlxcel reads the nested one; both agree here,
    // and the top-level copy stays ignored.
    assert_eq!(config.rope_dims(), 64);

    // `rope_theta` is 1e7 for this family, not the 1e5 fallback.
    assert_eq!(
        config
            .rope_parameters
            .as_ref()
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(serde_json::Value::as_f64),
        Some(10_000_000.0)
    );

    // The positional linear/full schedule must reproduce the checkpoint's own
    // `layer_types` array: 16 full-attention layers out of 64, every fourth.
    assert_eq!(
        (0..config.num_hidden_layers)
            .filter(|&i| !config.is_linear_layer(i))
            .count(),
        16
    );

    config
        .validate_supported()
        .expect("the published checkpoint must pass validation");
}

/// The whole `config.json`, including keys nested outside `text_config`, must
/// keep parsing on the wrapper side too.
#[test]
fn qwen3_8_27b_wrapper_config_is_accepted() {
    let wrapper = qwen3_8_27b_wrapper_config();

    // `generation_config` nested inside config.json is new in Qwen3.8 and is
    // ignored by the loader (the sampler reads generation_config.json).
    assert!(wrapper.get("generation_config").is_some());
    // `language_model_only: false` is the only value with a code path.
    assert_eq!(
        wrapper.get("language_model_only").and_then(|v| v.as_bool()),
        Some(false)
    );
    validate_qwen35_wrapper_config(&wrapper).expect("language_model_only=false must load");

    // Qwen3.5 omits the key entirely; that must stay valid.
    let mut without = wrapper.clone();
    without
        .as_object_mut()
        .expect("wrapper config is an object")
        .remove("language_model_only");
    validate_qwen35_wrapper_config(&without).expect("an absent language_model_only must load");
}

// ---------------------------------------------------------------------------
// `validate_supported` — silently ignored keys are now named errors (#1163).
// ---------------------------------------------------------------------------

fn config_with(text_overrides: serde_json::Value) -> Qwen35Config {
    let mut value = qwen3_8_27b_text_config();
    let object = value.as_object_mut().expect("text config is an object");
    for (key, override_value) in text_overrides
        .as_object()
        .expect("overrides must be an object")
    {
        if override_value.is_null() {
            object.remove(key);
        } else {
            object.insert(key.clone(), override_value.clone());
        }
    }
    serde_json::from_value(value).expect("overridden config must still deserialize")
}

#[test]
fn output_gate_type_accepts_silu_swish_and_absent() {
    for gate in ["swish", "silu"] {
        config_with(serde_json::json!({ "output_gate_type": gate }))
            .validate_supported()
            .unwrap_or_else(|e| panic!("output_gate_type={gate:?} must load: {e}"));
    }
    // Every Qwen3.5 checkpoint omits the key; absent must remain valid.
    let absent = config_with(serde_json::json!({ "output_gate_type": null }));
    assert_eq!(absent.output_gate_type, None);
    absent
        .validate_supported()
        .expect("an absent output_gate_type must load");
}

#[test]
fn output_gate_type_sigmoid_is_a_named_error() {
    let err = config_with(serde_json::json!({ "output_gate_type": "sigmoid" }))
        .validate_supported()
        .expect_err("sigmoid has no code path and must not load");
    assert_eq!(
        err,
        Qwen35UnsupportedConfig::OutputGateType("sigmoid".to_string())
    );
    let message = err.to_string();
    assert!(
        message.contains("output_gate_type") && message.contains("sigmoid"),
        "error must name the key and the offending value, got: {message}"
    );
}

#[test]
fn mrope_interleaved_accepts_true_and_absent() {
    config_with(serde_json::json!({
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "rope_theta": 10000000
        }
    }))
    .validate_supported()
    .expect("mrope_interleaved=true is the implemented layout");

    // Qwen3.5 omits the key.
    let absent = config_with(serde_json::json!({
        "rope_parameters": {
            "mrope_section": [11, 11, 10],
            "rope_theta": 10000000
        }
    }));
    assert_eq!(absent.mrope_interleaved(), None);
    absent
        .validate_supported()
        .expect("an absent mrope_interleaved must load");
}

#[test]
fn mrope_interleaved_false_is_a_named_error() {
    let err = config_with(serde_json::json!({
        "rope_parameters": {
            "mrope_interleaved": false,
            "mrope_section": [11, 11, 10],
            "rope_theta": 10000000
        }
    }))
    .validate_supported()
    .expect_err("InterleavedMRoPE is hardcoded, so a non-interleaved config must not load");
    assert_eq!(err, Qwen35UnsupportedConfig::MropeInterleavedDisabled);
}

#[test]
fn language_model_only_true_is_a_named_error() {
    let mut wrapper = qwen3_8_27b_wrapper_config();
    wrapper
        .as_object_mut()
        .expect("wrapper config is an object")
        .insert("language_model_only".to_string(), serde_json::json!(true));
    let err = validate_qwen35_wrapper_config(&wrapper)
        .expect_err("a vision-stripped build has no code path and must not load");
    assert_eq!(err, Qwen35UnsupportedConfig::LanguageModelOnly);
    let message = err.to_string();
    assert!(
        message.contains("language_model_only"),
        "error must name the key, got: {message}"
    );
}

// ---------------------------------------------------------------------------
// `mrope_position_source` — chunked-prefill position_ids reuse (#1163).
// ---------------------------------------------------------------------------

/// Chunked prefill calls forward repeatedly with a growing `cache_offset`
/// while the stored `position_ids` covers the whole prompt. Each call must
/// slice its own window instead of reusing the head of the tensor or
/// recomputing. Mirrors Blaizzy/mlx-vlm#1741.
#[test]
fn mrope_position_source_slices_each_chunk_of_a_chunked_prefill() {
    // A 1024-token prompt whose MRoPE ids were computed once, then consumed
    // in 256-token chunks.
    let stored = [3, 1, 1024];
    let mut offset = 0;
    while offset < 1024 {
        assert_eq!(
            mrope_position_source(Some(&stored), 1, 256, offset),
            MRopePositionSource::SliceStored {
                start: offset,
                end: offset + 256,
            },
            "chunk at offset {offset} must slice its own window"
        );
        offset += 256;
    }

    // Decode step immediately after the prefill: the stored tensor is exactly
    // long enough for the last position, so it is still reusable.
    assert_eq!(
        mrope_position_source(Some(&stored), 1, 1, 1023),
        MRopePositionSource::SliceStored {
            start: 1023,
            end: 1024
        }
    );
}

#[test]
fn mrope_position_source_recomputes_when_the_window_runs_past_the_stored_ids() {
    let stored = [3, 1, 1024];
    // First decode step past the prompt: 1024 + 1 > 1024.
    assert_eq!(
        mrope_position_source(Some(&stored), 1, 1, 1024),
        MRopePositionSource::Recompute
    );
    // A chunk that starts inside the stored range but overruns it.
    assert_eq!(
        mrope_position_source(Some(&stored), 1, 256, 900),
        MRopePositionSource::Recompute
    );
}

/// Blaizzy/mlx-vlm#1040: a following request with a different batch size must
/// not reuse the stored ids, or the broadcast downstream is silently wrong.
#[test]
fn mrope_position_source_rejects_a_batch_mismatch_and_a_wrong_rank() {
    let stored = [3, 4, 1024];
    assert_eq!(
        mrope_position_source(Some(&stored), 1, 8, 0),
        MRopePositionSource::Recompute
    );
    assert_eq!(
        mrope_position_source(Some(&stored), 4, 8, 0),
        MRopePositionSource::SliceStored { start: 0, end: 8 }
    );

    // A 2-D tensor is not a `[3, batch, len]` MRoPE array.
    assert_eq!(
        mrope_position_source(Some(&[1, 1024]), 1, 8, 0),
        MRopePositionSource::Recompute
    );
}

#[test]
fn mrope_position_source_recomputes_when_nothing_is_stored() {
    assert_eq!(
        mrope_position_source(None, 1, 8, 0),
        MRopePositionSource::Recompute
    );
}
