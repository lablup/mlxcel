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

/// Bytes one ring row costs on the fixture: 40 sliding layers x 8 heads x
/// 256 x (K and V) x 2 bytes.
const ROW: usize = 40 * 8 * 256 * 2 * 2;

#[test]
fn a_gemma4_ring_checkpoint_is_every_sliding_ring_after_its_last_piece() {
    // A decode-shaped ring is the window: 320 MiB, the figure in #3255.
    let decode = recommend_ring_checkpoint_capacity_from_config(
        &gemma4_12b_config(),
        KVCacheMode::Fp16,
        1,
        None,
    )
    .expect("Gemma 4 stores ring checkpoints");
    assert_eq!(decode.entry_bytes, 1024 * ROW);
    assert_eq!(decode.entry_bytes, 320 << 20);
    // At a history boundary the ring still holds the whole last prefill
    // piece after the newest `window - 1` rows: 3071 rows at a 2048 chunk,
    // 960 MiB minus a row, which is what the server stores.
    let rec = recommend_ring_checkpoint_capacity_from_config(
        &gemma4_12b_config(),
        KVCacheMode::Fp16,
        2048,
        None,
    )
    .unwrap();
    assert_eq!(rec.entry_bytes, (1024 - 1 + 2048) * ROW);
    assert_eq!(
        rec.capacity_bytes,
        rec.entry_bytes * MODEL_AWARE_STATE_CHECKPOINT_TARGET_ENTRIES as usize
    );
    let tight = recommend_ring_checkpoint_capacity_from_config(
        &gemma4_12b_config(),
        KVCacheMode::Fp16,
        2048,
        Some(8 << 30),
    )
    .unwrap();
    assert_eq!(tight.capacity_bytes, 1 << 30, "an eighth of what is left");

    // gemma-4-12b-it ships as `gemma4_unified` / `gemma4_unified_text`.
    let mut unified = gemma4_12b_config();
    unified["model_type"] = json!("gemma4_unified");
    unified["text_config"]["model_type"] = json!("gemma4_unified_text");
    let rec = recommend_ring_checkpoint_capacity_from_config(&unified, KVCacheMode::Fp16, 1, None)
        .expect("the Unified checkpoint stores ring checkpoints too");
    assert_eq!(rec.entry_bytes, 320 << 20);
}

#[test]
fn ring_checkpoints_are_sized_only_for_their_families_and_fp16() {
    assert!(
        recommend_ring_checkpoint_capacity_from_config(
            &gemma4_12b_config(),
            KVCacheMode::Int8,
            2048,
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
        recommend_ring_checkpoint_capacity_from_config(&llama, KVCacheMode::Fp16, 2048, None)
            .is_none()
    );
    // The hybrid families' recommendation does not change: it still ignores
    // the attention-only Gemma 4.
    assert!(recommend_state_checkpoint_capacity_from_config(&gemma4_12b_config(), None).is_none());
}
