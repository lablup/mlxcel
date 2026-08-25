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

//! Tests for the pieces both late-interaction families share.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

use super::{
    DEFAULT_EMBEDDING_DIM, QUERY_AUGMENTATION_TOKENS, apply_dense_projection_override,
    embedding_dim, format_query, project_and_normalize, reject_lora_only_checkpoint,
};
use crate::models::embedding_test_support::{Rng, mlx_test_guard, temp_dir, to_vec};

#[test]
fn embedding_dim_defaults_to_128_and_reads_the_config() {
    assert_eq!(embedding_dim(&json!({})), DEFAULT_EMBEDDING_DIM);
    assert_eq!(embedding_dim(&json!({"embedding_dim": 128})), 128);
    assert_eq!(embedding_dim(&json!({"embedding_dim": 64})), 64);
    // A zero or non-numeric value falls back rather than producing a
    // zero-width vector the engine would then reject.
    assert_eq!(embedding_dim(&json!({"embedding_dim": 0})), 128);
    assert_eq!(embedding_dim(&json!({"embedding_dim": "128"})), 128);
}

#[test]
fn format_query_appends_ten_augmentation_tokens() {
    let formatted = format_query("What was the total revenue in 2023?", "<|endoftext|>");
    assert!(formatted.starts_with("Query: What was the total revenue in 2023?"));
    assert_eq!(
        formatted.matches("<|endoftext|>").count(),
        QUERY_AUGMENTATION_TOKENS
    );
    assert!(formatted.ends_with(&"<|endoftext|>".repeat(QUERY_AUGMENTATION_TOKENS)));

    let smol = format_query("hi", "<end_of_utterance>");
    assert_eq!(
        smol,
        format!(
            "Query: hi{}",
            "<end_of_utterance>".repeat(QUERY_AUGMENTATION_TOKENS)
        )
    );
}

#[test]
fn lora_only_checkpoint_is_rejected_with_the_merge_instruction() {
    let dir = temp_dir("col_lora_only");
    std::fs::write(dir.join("adapter_model.safetensors"), b"stub").unwrap();
    let error = format!(
        "{:#}",
        reject_lora_only_checkpoint(&dir).expect_err("a LoRA-only directory must be rejected")
    );
    assert!(
        error.contains("merge the adapter into the base checkpoint first"),
        "{error}"
    );

    // A merged export keeps the adapter next to the base shard and loads.
    std::fs::write(dir.join("model.safetensors"), b"stub").unwrap();
    assert!(reject_lora_only_checkpoint(&dir).is_ok());
    std::fs::remove_dir_all(&dir).unwrap();

    // A plain base checkpoint with no adapter at all is fine.
    let base = temp_dir("col_base_only");
    std::fs::write(base.join("model.safetensors"), b"stub").unwrap();
    assert!(reject_lora_only_checkpoint(&base).is_ok());
    std::fs::remove_dir_all(&base).unwrap();
}

#[test]
fn sanitize_prefers_1_dense_projection() {
    let _guard = mlx_test_guard();
    let mut rng = Rng::new(0xC0_1D);
    let mut weights = WeightMap::new();
    // The base repository's untrained projection, plus a quantized pair that
    // must not survive the override.
    rng.insert(&mut weights, "linear.weight", &[4, 3], 1.0);
    rng.insert(&mut weights, "linear.bias", &[4], 1.0);
    rng.insert(&mut weights, "linear.scales", &[4, 1], 1.0);
    rng.insert(&mut weights, "linear.biases", &[4, 1], 1.0);
    // The trained module folder.
    weights.insert(
        "1_Dense.linear.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0; 12], &[4, 3]),
    );
    weights.insert(
        "1_Dense.linear.bias".to_string(),
        mlxcel_core::from_slice_f32(&[0.5; 4], &[4]),
    );

    assert!(apply_dense_projection_override(&mut weights, "linear"));
    assert!(!weights.contains_key("1_Dense.linear.weight"));
    assert!(!weights.contains_key("1_Dense.linear.bias"));
    assert!(
        !weights.contains_key("linear.scales") && !weights.contains_key("linear.biases"),
        "a dense module folder must not leave the root projection's quantization behind"
    );
    assert_eq!(to_vec(&weights["linear.weight"]), vec![1.0; 12]);
    assert_eq!(to_vec(&weights["linear.bias"]), vec![0.5; 4]);

    // With no module folder the root projection is kept untouched.
    let mut only_root = WeightMap::new();
    only_root.insert(
        "custom_text_proj.weight".to_string(),
        mlxcel_core::from_slice_f32(&[2.0; 6], &[2, 3]),
    );
    assert!(!apply_dense_projection_override(
        &mut only_root,
        "custom_text_proj"
    ));
    assert_eq!(to_vec(&only_root["custom_text_proj.weight"]), vec![2.0; 6]);
}

#[test]
fn padding_rows_are_zero_and_real_rows_unit_norm() {
    let _guard = mlx_test_guard();
    let (batch, length, hidden, dim) = (2i32, 4i32, 3i32, 5i32);

    let mut rng = Rng::new(0x5EED);
    let mut weights = WeightMap::new();
    rng.insert(&mut weights, "linear.weight", &[dim, hidden], 0.8);
    rng.insert(&mut weights, "linear.bias", &[dim], 0.3);
    let projection = UnifiedLinear::from_weights(&weights, "linear", 0, 0).unwrap();

    let hidden_states = rng.tensor(&[batch, length, hidden], 1.5);
    // Row 0 is full length; row 1 has two real tokens then two padding ones.
    let mask = mlxcel_core::from_slice_i32(&[1, 1, 1, 1, 1, 1, 0, 0], &[batch, length]);

    let out = project_and_normalize(&hidden_states, &projection, &mask);
    mlxcel_core::eval(&out);
    assert_eq!(
        mlxcel_core::array_shape(&out),
        vec![batch, length, dim],
        "the projection keeps the token axis"
    );

    let values = to_vec(&out);
    let width = dim as usize;
    let real = [true, true, true, true, true, true, false, false];
    for (row, &is_real) in real.iter().enumerate() {
        let slice = &values[row * width..(row + 1) * width];
        let norm = slice.iter().map(|v| v * v).sum::<f32>().sqrt();
        if is_real {
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "real row {row} has norm {norm}, expected 1"
            );
        } else {
            assert!(
                slice.iter().all(|&v| v == 0.0),
                "padding row {row} is not zeroed: {slice:?}"
            );
        }
    }
}

#[test]
fn an_all_zero_projection_row_stays_finite() {
    let _guard = mlx_test_guard();
    // A zero hidden state with a bias-free projection produces a zero
    // vector; the epsilon must keep it at zero instead of dividing by zero.
    let mut weights = WeightMap::new();
    weights.insert(
        "linear.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0, -1.0, 0.5, 0.25], &[2, 2]),
    );
    let projection = UnifiedLinear::from_weights(&weights, "linear", 0, 0).unwrap();
    let hidden = mlxcel_core::from_slice_f32(&[0.0, 0.0], &[1, 1, 2]);
    let mask = mlxcel_core::from_slice_i32(&[1], &[1, 1]);

    let out = project_and_normalize(&hidden, &projection, &mask);
    mlxcel_core::eval(&out);
    let values = to_vec(&out);
    assert!(
        values.iter().all(|v| v.is_finite()),
        "a zero row must not produce NaN: {values:?}"
    );
    assert_eq!(values, vec![0.0, 0.0]);
}
