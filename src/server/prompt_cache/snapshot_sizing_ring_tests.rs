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

//! Sizing of the sliding-window families' ring-only state checkpoints
//! (issue #3255).

use super::*;
use mlxcel_core::cache::KVCacheMode;
use serde_json::json;

/// `gemma-4-12b-it` reduced to the fields the estimator reads: 48 layers, 40
/// sliding with 8 KV heads of 256, 8 full with 1 KV head of 512.
fn gemma4_12b_config() -> Value {
    let layer_types: Vec<&str> = (0..48)
        .map(|i| {
            if (i + 1) % 6 == 0 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 512,
            "sliding_window": 1024,
            "layer_types": layer_types,
            "hidden_size": 3840
        }
    })
}

#[test]
fn a_gemma4_ring_checkpoint_is_every_sliding_ring_at_its_window() {
    let rec = recommend_ring_checkpoint_capacity_from_config(
        &gemma4_12b_config(),
        KVCacheMode::Fp16,
        None,
    )
    .expect("Gemma 4 stores ring checkpoints");
    // 40 layers x 1024 tokens x 8 heads x 256 x (K and V) x 2 bytes = 320 MiB,
    // the figure in #3255.
    assert_eq!(rec.entry_bytes, 40 * 1024 * 8 * 256 * 2 * 2);
    assert_eq!(rec.entry_bytes, 320 << 20);
    assert_eq!(
        rec.capacity_bytes,
        rec.entry_bytes * MODEL_AWARE_STATE_CHECKPOINT_TARGET_ENTRIES as usize
    );
    let tight = recommend_ring_checkpoint_capacity_from_config(
        &gemma4_12b_config(),
        KVCacheMode::Fp16,
        Some(8 << 30),
    )
    .unwrap();
    assert_eq!(tight.capacity_bytes, 1 << 30, "an eighth of what is left");
}

#[test]
fn ring_checkpoints_are_sized_only_for_their_families_and_fp16() {
    assert!(
        recommend_ring_checkpoint_capacity_from_config(
            &gemma4_12b_config(),
            KVCacheMode::Int8,
            None
        )
        .is_none(),
        "quantized caches are not snapshotted"
    );
    let llama = json!({
        "model_type": "llama",
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "sliding_window": 512
    });
    assert!(
        recommend_ring_checkpoint_capacity_from_config(&llama, KVCacheMode::Fp16, None).is_none()
    );
    // The hybrid families' recommendation does not change: it still ignores
    // the attention-only Gemma 4.
    assert!(recommend_state_checkpoint_capacity_from_config(&gemma4_12b_config(), None).is_none());
}
