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
    Qwen35Config, rebuild_with_zero_tail, sanitize_weights, zero_per_row_kv_tail,
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
