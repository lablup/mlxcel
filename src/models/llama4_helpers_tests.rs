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

use crate::models::llama4::TextArgs;
use crate::models::llama4_helpers::{create_chunked_attention_mask, get_weight_copy};
use mlxcel_core::weights::WeightMap;
use std::sync::{Mutex, OnceLock};

fn test_guard() -> &'static Mutex<()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(()))
}

fn scalar_at(mask: &mlxcel_core::MlxArray, row: i32, col: i32) -> f32 {
    let cell = mlxcel_core::slice(mask, &[row, col], &[row + 1, col + 1]);
    let cell = mlxcel_core::reshape(&cell, &[1]);
    mlxcel_core::eval(&cell);
    mlxcel_core::item_f32(&cell)
}

#[test]
#[ignore = "requires serial MLX execution"]
fn chunked_attention_mask_blocks_other_chunks_and_future_tokens() {
    let _guard = test_guard().lock().unwrap();
    let mask = create_chunked_attention_mask(3, 0, 0, 2);

    assert_eq!(mlxcel_core::array_shape(&mask), vec![3, 3]);
    assert_eq!(scalar_at(&mask, 0, 0), 0.0);
    assert!(scalar_at(&mask, 0, 1).is_infinite() && scalar_at(&mask, 0, 1).is_sign_negative());
    assert_eq!(scalar_at(&mask, 1, 1), 0.0);
    assert!(scalar_at(&mask, 1, 2).is_infinite() && scalar_at(&mask, 1, 2).is_sign_negative());
    assert!(scalar_at(&mask, 2, 0).is_infinite() && scalar_at(&mask, 2, 0).is_sign_negative());
    assert_eq!(scalar_at(&mask, 2, 2), 0.0);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn get_weight_copy_returns_new_buffer_and_reports_missing_keys() {
    let _guard = test_guard().lock().unwrap();
    let original = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let mut weights = WeightMap::new();
    weights.insert("layers.0.test".to_string(), original);

    let copied = get_weight_copy(&weights, "layers.0.test").unwrap();
    assert_eq!(mlxcel_core::array_shape(&copied), vec![2, 2]);
    assert_ne!(
        copied.as_ref().unwrap() as *const _,
        weights["layers.0.test"].as_ref().unwrap() as *const _
    );

    let missing = get_weight_copy(&weights, "layers.1.test");
    assert!(missing.is_err());
}

#[test]
#[ignore = "requires serial MLX execution"]
fn text_args_quantization_defaults_are_available_to_helpers() {
    let _guard = test_guard().lock().unwrap();
    let args = TextArgs {
        model_type: "llama4".to_string(),
        hidden_size: 128,
        num_hidden_layers: 2,
        intermediate_size: 256,
        intermediate_size_mlp: 128,
        num_attention_heads: 4,
        num_key_value_heads: 1,
        rms_norm_eps: 1e-5,
        vocab_size: 32000,
        head_dim: 32,
        max_position_embeddings: 1024,
        attention_chunk_size: 8,
        interleave_moe_layer_step: 4,
        num_local_experts: 8,
        num_experts_per_tok: 2,
        attention_bias: false,
        use_qk_norm: true,
        rope_theta: 500000.0,
        attn_temperature_tuning: 0,
        floor_scale: 1,
        attn_scale: 1.0,
        group_size: None,
        bits: None,
    };

    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}
