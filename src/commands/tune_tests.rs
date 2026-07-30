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

//! Unit tests for `mlxcel tune` argument and matrix construction (issue #906).
//!
//! The profiling itself is a GPU run and is exercised by benchmark passes, not
//! here. What is worth pinning is the pure logic: list parsing, the head
//! geometry a checkpoint config yields, and the scattered block table the
//! synthetic sweep hands to the kernel.

use super::{
    HeadGeometry, head_geometry_from_config, parse_usize_list, row_offsets_for, scattered_rows,
};

const FALLBACK: HeadGeometry = HeadGeometry {
    head_dim: 128,
    q_heads: 32,
    kv_heads: 8,
};

// ── List parsing ─────────────────────────────────────────────────────────────

#[test]
fn parse_usize_list_accepts_whitespace_and_trailing_commas() {
    assert_eq!(parse_usize_list("1,4,16").expect("parsed"), vec![1, 4, 16]);
    assert_eq!(parse_usize_list(" 8 , 32 ").expect("parsed"), vec![8, 32]);
    assert_eq!(parse_usize_list("1,2,").expect("parsed"), vec![1, 2]);
}

#[test]
fn parse_usize_list_rejects_empty_and_zero_and_garbage() {
    assert!(parse_usize_list("").is_err());
    assert!(parse_usize_list(",,").is_err());
    assert!(parse_usize_list("0").is_err());
    assert!(parse_usize_list("4,zero").is_err());
}

// ── Head geometry ────────────────────────────────────────────────────────────

#[test]
fn head_geometry_prefers_the_explicit_head_dim() {
    let config = serde_json::json!({
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 5120,
    });
    let g = head_geometry_from_config(&config, FALLBACK);
    assert_eq!(
        g,
        HeadGeometry {
            head_dim: 128,
            q_heads: 40,
            kv_heads: 8
        }
    );
}

#[test]
fn head_geometry_derives_head_dim_from_hidden_size() {
    let config = serde_json::json!({
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
    });
    let g = head_geometry_from_config(&config, FALLBACK);
    assert_eq!(g.head_dim, 128);
}

#[test]
fn head_geometry_defaults_kv_heads_to_mha() {
    // A config with no `num_key_value_heads` is multi-head attention, so the
    // KV head count equals the query head count, not the fallback's GQA value.
    let config = serde_json::json!({
        "num_attention_heads": 16,
        "hidden_size": 2048,
    });
    let g = head_geometry_from_config(&config, FALLBACK);
    assert_eq!(g.q_heads, 16);
    assert_eq!(g.kv_heads, 16);
}

#[test]
fn head_geometry_falls_back_on_an_empty_config() {
    let g = head_geometry_from_config(&serde_json::json!({}), FALLBACK);
    assert_eq!(g, FALLBACK);
}

#[test]
fn head_geometry_never_yields_a_zero_dimension() {
    // A hostile or truncated config must not produce a zero-sized launch.
    let config = serde_json::json!({
        "num_attention_heads": 0,
        "num_key_value_heads": 0,
        "head_dim": 0,
    });
    let g = head_geometry_from_config(&config, FALLBACK);
    assert!(g.head_dim >= 1 && g.q_heads >= 1 && g.kv_heads >= 1);
}

// ── Synthetic block table ────────────────────────────────────────────────────

#[test]
fn scattered_rows_are_unique_and_descending() {
    let rows = scattered_rows(2, 4);
    assert_eq!(rows.len(), 8);
    // 2x pool slack, reverse order: 15, 14, ... 8.
    assert_eq!(rows, vec![15, 14, 13, 12, 11, 10, 9, 8]);
    let mut sorted = rows.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), rows.len(), "physical rows must not alias");
}

#[test]
fn scattered_rows_stay_inside_the_pool() {
    let batch = 4;
    let blocks = 7;
    let pool = batch * blocks * 2;
    for r in scattered_rows(batch, blocks) {
        assert!(r >= 0 && (r as usize) < pool);
    }
}

#[test]
fn row_offsets_bracket_every_sequence() {
    let offsets = row_offsets_for(3, 5);
    assert_eq!(offsets, vec![0, 5, 10, 15]);
    assert_eq!(offsets.len(), 4);
    assert_eq!(
        offsets.last().copied().unwrap_or(0) as usize,
        scattered_rows(3, 5).len()
    );
}
