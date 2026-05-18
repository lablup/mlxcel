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

//! `granularity: attention_head` unit tests for `PruneOp`, including
//! the GQA-safe policy and the quantized-Affine layout.

use super::super::{PruneOp, PruneSelector};
use super::{make_cfg, read_f32_2d};
use crate::{SurgeryOp, WeightMap};
use mlxcel_core as ffi;
use mlxcel_core::dtype;

#[test]
fn attention_head_prune_zeros_q_proj_row_slice() {
    // num_heads = 4, head_dim = 8, so q_proj is [32, 16]. Zero
    // head ids [1, 3]: rows [8..16) and [24..32) must be zero;
    // rows [0..8) and [16..24) must remain ones.
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        ffi::ones(&[32, 16], dtype::FLOAT32),
    );
    let op = PruneOp::new(
        "model.layers.0.self_attn.*",
        PruneSelector::AttentionHead {
            head_ids: vec![1, 3],
        },
    )
    .expect("compile");
    let cfg = make_cfg(4, 4, 32, 64, 1);
    op.apply(&mut weights, &cfg).expect("apply");

    let (shape, floats) = read_f32_2d(&weights, "model.layers.0.self_attn.q_proj.weight");
    assert_eq!(shape, vec![32, 16]);
    let row_sum = |r: usize| -> f32 { floats[r * 16..(r + 1) * 16].iter().sum() };
    // Head 0 rows [0..8): nonzero.
    for r in 0..8 {
        assert!(row_sum(r) > 0.0, "row {r} (head 0) should be nonzero");
    }
    // Head 1 rows [8..16): zero.
    for r in 8..16 {
        assert!(row_sum(r) == 0.0, "row {r} (head 1) should be zero");
    }
    // Head 2 rows [16..24): nonzero.
    for r in 16..24 {
        assert!(row_sum(r) > 0.0, "row {r} (head 2) should be nonzero");
    }
    // Head 3 rows [24..32): zero.
    for r in 24..32 {
        assert!(row_sum(r) == 0.0, "row {r} (head 3) should be zero");
    }
}

#[test]
fn attention_head_prune_zeros_o_proj_column_slice() {
    // o_proj shape [hidden=32, num_heads*head_dim=32]. Head 2 is
    // axis-1 cols [16..24).
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        ffi::ones(&[32, 32], dtype::FLOAT32),
    );
    let op = PruneOp::new(
        "model.layers.0.self_attn.*",
        PruneSelector::AttentionHead { head_ids: vec![2] },
    )
    .expect("compile");
    let cfg = make_cfg(4, 4, 32, 64, 1);
    op.apply(&mut weights, &cfg).expect("apply");

    let (shape, floats) = read_f32_2d(&weights, "model.layers.0.self_attn.o_proj.weight");
    assert_eq!(shape, vec![32, 32]);
    for row in 0..32 {
        for col in 0..32 {
            let v = floats[row * 32 + col];
            if (16..24).contains(&col) {
                assert!(v == 0.0, "col {col} of row {row} should be zero");
            } else {
                assert!(v == 1.0, "col {col} of row {row} should be one");
            }
        }
    }
}

#[test]
fn attention_head_prune_skips_kv_proj_under_gqa_policy() {
    // num_heads=8, num_kv_heads=2 (GQA). k_proj shape
    // [num_kv_heads*head_dim=16, 32]. We expect the op to leave it
    // untouched and still succeed because q_proj is matched too.
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        ffi::ones(&[64, 32], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        ffi::ones(&[16, 32], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        ffi::ones(&[16, 32], dtype::FLOAT32),
    );

    let op = PruneOp::new(
        "model.layers.0.self_attn.*",
        PruneSelector::AttentionHead { head_ids: vec![0] },
    )
    .expect("compile");
    let cfg = make_cfg(8, 2, 64, 128, 1);
    op.apply(&mut weights, &cfg).expect("apply");

    // q_proj head 0 rows [0..8) zero.
    let (_, floats) = read_f32_2d(&weights, "model.layers.0.self_attn.q_proj.weight");
    for r in 0..8 {
        let s: f32 = floats[r * 32..(r + 1) * 32].iter().sum();
        assert!(s == 0.0, "q_proj row {r} should be zero");
    }
    // k_proj untouched (all ones).
    let (_, floats) = read_f32_2d(&weights, "model.layers.0.self_attn.k_proj.weight");
    assert!(
        floats.iter().all(|&x| x == 1.0),
        "k_proj must be untouched under GQA policy"
    );
    // v_proj untouched (all ones).
    let (_, floats) = read_f32_2d(&weights, "model.layers.0.self_attn.v_proj.weight");
    assert!(
        floats.iter().all(|&x| x == 1.0),
        "v_proj must be untouched under GQA policy"
    );
}

#[test]
fn attention_head_prune_errors_when_only_kv_matched() {
    // Pattern matches only k_proj — there is no Q/O for the op to
    // touch, so it must error rather than silently no-op.
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        ffi::ones(&[16, 32], dtype::FLOAT32),
    );
    let op = PruneOp::new(
        "model.layers.0.self_attn.k_proj.*",
        PruneSelector::AttentionHead { head_ids: vec![0] },
    )
    .expect("compile");
    let cfg = make_cfg(8, 2, 64, 128, 1);
    let err = op.apply(&mut weights, &cfg).expect_err("kv-only must err");
    assert!(format!("{err}").contains("Q or O"));
}

#[test]
fn attention_head_prune_rejects_out_of_range_head_id() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        ffi::ones(&[32, 16], dtype::FLOAT32),
    );
    let op = PruneOp::new(
        "model.layers.0.self_attn.*",
        PruneSelector::AttentionHead { head_ids: vec![99] },
    )
    .expect("compile");
    let cfg = make_cfg(4, 4, 32, 64, 1);
    let err = op.apply(&mut weights, &cfg).expect_err("oor head");
    assert!(format!("{err}").contains("head_id 99"));
}

#[test]
fn attention_head_uses_text_config_for_vlm() {
    // VLMs typically nest LM dims under `text_config`. The op
    // should pick them up automatically.
    let mut weights = WeightMap::new();
    weights.insert(
        "language_model.model.layers.0.self_attn.q_proj.weight".to_string(),
        ffi::ones(&[32, 16], dtype::FLOAT32),
    );
    let op = PruneOp::new(
        "language_model.model.layers.0.self_attn.*",
        PruneSelector::AttentionHead { head_ids: vec![0] },
    )
    .expect("compile");
    let cfg = serde_json::json!({
        "vision_config": {"some": "thing"},
        "text_config": {
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
        }
    });
    op.apply(&mut weights, &cfg)
        .expect("apply on VLM-style cfg");

    let (_, floats) = read_f32_2d(
        &weights,
        "language_model.model.layers.0.self_attn.q_proj.weight",
    );
    // Head 0 = rows [0..8) zero.
    let s: f32 = floats[0..8 * 16].iter().sum();
    assert!(s == 0.0);
    // Heads 1..3 = rows [8..32) nonzero.
    let s: f32 = floats[8 * 16..].iter().sum();
    assert!(s > 0.0);
}

#[test]
fn quantized_attention_head_zeros_weight_scales_biases() {
    // Synthetic 4-bit affine layout: q_proj.weight u32 packed
    // [num_heads*head_dim=32, hidden_size/8 = 16/8=2], scales/
    // biases f32 [32, hidden_size/group_size = 16/8 = 2].
    // Zero head 1 (rows [8..16)) and verify all three sibling
    // tensors have those rows zeroed.
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        ffi::ones(&[32, 2], dtype::UINT32),
    );
    weights.insert(
        "model.layers.0.self_attn.q_proj.scales".to_string(),
        ffi::ones(&[32, 2], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.self_attn.q_proj.biases".to_string(),
        ffi::ones(&[32, 2], dtype::FLOAT32),
    );

    let op = PruneOp::new(
        "model.layers.0.self_attn.q_proj.*",
        PruneSelector::AttentionHead { head_ids: vec![1] },
    )
    .expect("compile");
    let cfg = make_cfg(4, 4, 32, 64, 1);
    op.apply(&mut weights, &cfg).expect("apply");

    // Verify scales rows [8..16) are zero by reading raw bytes
    // (FLOAT32 = 4 bytes per element).
    let arr = weights
        .get("model.layers.0.self_attn.q_proj.scales")
        .unwrap();
    mlxcel_core::eval(arr);
    let bytes = ffi::array_to_raw_bytes(arr);
    let row_bytes = 2 * 4;
    for r in 8..16 {
        let start = r * row_bytes;
        for b in 0..row_bytes {
            assert_eq!(bytes[start + b], 0, "scales row {r} byte {b} must be zero");
        }
    }
    // Verify rows [0..8) are not zero (still ones).
    for r in 0..8 {
        let start = r * row_bytes;
        let mut all_zero = true;
        for b in 0..row_bytes {
            if bytes[start + b] != 0 {
                all_zero = false;
                break;
            }
        }
        assert!(!all_zero, "scales row {r} must not be zeroed");
    }
}

#[test]
fn axis_alignment_aligned_o_proj_slice_succeeds() {
    // hidden=14, heads=2 -> head_dim=7. o_proj shape [hidden=14,
    // num_heads*head_dim=14]. Pruning head 0 would zero cols
    // [0..7). Since head_dim divides cleanly into the column count,
    // the alignment guard fires "0 mod 7 == 0" (aligned) and the
    // prune succeeds. This pins the head-aligned case.
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        ffi::ones(&[14, 14], dtype::FLOAT32),
    );
    let op = PruneOp::new(
        "model.layers.0.self_attn.*",
        PruneSelector::AttentionHead { head_ids: vec![0] },
    )
    .expect("compile");
    let cfg = make_cfg(2, 2, 14, 28, 1);
    op.apply(&mut weights, &cfg).expect("apply aligned");
}
