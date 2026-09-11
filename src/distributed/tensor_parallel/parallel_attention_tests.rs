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

use super::*;

// ---------------------------------------------------------------------------
// AttentionType classification
// ---------------------------------------------------------------------------

#[test]
fn test_attention_type_mha() {
    let config = TPAttentionConfig {
        tp_rank: 0,
        tp_size: 1,
        total_heads: 32,
        total_kv_heads: 32,
        head_dim: 128,
        sliding_window: None,
    };
    assert_eq!(config.attention_type(), AttentionType::MHA);
    assert_eq!(config.gqa_group_size(), 1);
}

#[test]
fn test_attention_type_gqa() {
    let config = TPAttentionConfig {
        tp_rank: 0,
        tp_size: 1,
        total_heads: 32,
        total_kv_heads: 8,
        head_dim: 128,
        sliding_window: None,
    };
    assert_eq!(config.attention_type(), AttentionType::GQA);
    assert_eq!(config.gqa_group_size(), 4);
}

#[test]
fn test_attention_type_mqa() {
    let config = TPAttentionConfig {
        tp_rank: 0,
        tp_size: 1,
        total_heads: 32,
        total_kv_heads: 1,
        head_dim: 128,
        sliding_window: None,
    };
    assert_eq!(config.attention_type(), AttentionType::MQA);
    assert_eq!(config.gqa_group_size(), 32);
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[test]
fn test_validate_valid_config() {
    let config = TPAttentionConfig {
        tp_rank: 0,
        tp_size: 2,
        total_heads: 32,
        total_kv_heads: 8,
        head_dim: 128,
        sliding_window: None,
    };
    assert!(validate_tp_attention_config(&config).is_ok());
}

#[test]
fn test_validate_rank_out_of_range() {
    let config = TPAttentionConfig {
        tp_rank: 4,
        tp_size: 4,
        total_heads: 32,
        total_kv_heads: 8,
        head_dim: 128,
        sliding_window: None,
    };
    assert!(validate_tp_attention_config(&config).is_err());
}

#[test]
fn test_validate_heads_not_divisible_by_tp() {
    let config = TPAttentionConfig {
        tp_rank: 0,
        tp_size: 3,
        total_heads: 32,
        total_kv_heads: 8,
        head_dim: 128,
        sliding_window: None,
    };
    assert!(validate_tp_attention_config(&config).is_err());
}

#[test]
fn test_validate_invalid_gqa_ratio() {
    let config = TPAttentionConfig {
        tp_rank: 0,
        tp_size: 2,
        total_heads: 32,
        total_kv_heads: 6,
        head_dim: 128,
        sliding_window: None,
    };
    assert!(validate_tp_attention_config(&config).is_err());
}

#[test]
fn test_validate_zero_tp_size() {
    let config = TPAttentionConfig {
        tp_rank: 0,
        tp_size: 0,
        total_heads: 32,
        total_kv_heads: 8,
        head_dim: 128,
        sliding_window: None,
    };
    assert!(validate_tp_attention_config(&config).is_err());
}

#[test]
fn test_validate_zero_heads() {
    let config = TPAttentionConfig {
        tp_rank: 0,
        tp_size: 1,
        total_heads: 0,
        total_kv_heads: 8,
        head_dim: 128,
        sliding_window: None,
    };
    assert!(validate_tp_attention_config(&config).is_err());
}

#[test]
fn test_validate_kv_heads_not_divisible_by_tp() {
    // 4 KV heads with tp_size=8: valid (replicated), but 6 KV heads
    // with tp_size=4 and total_heads=24 would fail GQA ratio.
    let config = TPAttentionConfig {
        tp_rank: 0,
        tp_size: 4,
        total_heads: 32,
        total_kv_heads: 8,
        head_dim: 128,
        sliding_window: None,
    };
    assert!(validate_tp_attention_config(&config).is_ok());

    // KV heads >= tp_size but not divisible: should fail.
    let config2 = TPAttentionConfig {
        tp_rank: 0,
        tp_size: 4,
        total_heads: 24,
        total_kv_heads: 6,
        head_dim: 128,
        sliding_window: None,
    };
    assert!(validate_tp_attention_config(&config2).is_err());
}

// ---------------------------------------------------------------------------
// Head assignment
// ---------------------------------------------------------------------------

#[test]
fn test_head_assignment_single_rank() {
    let range = head_assignment(0, 1, 32);
    assert_eq!(range, 0..32);
}

#[test]
fn test_head_assignment_two_ranks() {
    assert_eq!(head_assignment(0, 2, 32), 0..16);
    assert_eq!(head_assignment(1, 2, 32), 16..32);
}

#[test]
fn test_head_assignment_four_ranks() {
    assert_eq!(head_assignment(0, 4, 32), 0..8);
    assert_eq!(head_assignment(1, 4, 32), 8..16);
    assert_eq!(head_assignment(2, 4, 32), 16..24);
    assert_eq!(head_assignment(3, 4, 32), 24..32);
}

// ---------------------------------------------------------------------------
// KV head assignment
// ---------------------------------------------------------------------------

#[test]
fn test_kv_assignment_mha_sharded() {
    // MHA with 32 KV heads, tp_size=4: shard KV heads.
    let assign = kv_head_assignment(0, 4, 32);
    assert_eq!(assign, KVAssignment::Sharded(0..8));
    assert!(!assign.is_replicated());
    assert_eq!(assign.num_heads(), 8);

    let assign3 = kv_head_assignment(3, 4, 32);
    assert_eq!(assign3, KVAssignment::Sharded(24..32));
}

#[test]
fn test_kv_assignment_gqa_shardable() {
    // GQA with 8 KV heads, tp_size=4: shard KV heads (2 per rank).
    let assign = kv_head_assignment(0, 4, 8);
    assert_eq!(assign, KVAssignment::Sharded(0..2));
    assert_eq!(assign.num_heads(), 2);
}

#[test]
fn test_kv_assignment_gqa_replicated() {
    // GQA with 4 KV heads, tp_size=8: replicate all KV heads.
    let assign = kv_head_assignment(0, 8, 4);
    assert_eq!(assign, KVAssignment::Replicated(0..4));
    assert!(assign.is_replicated());
    assert_eq!(assign.num_heads(), 4);

    // All ranks get the same assignment.
    for rank in 0..8 {
        let a = kv_head_assignment(rank, 8, 4);
        assert_eq!(a, KVAssignment::Replicated(0..4));
    }
}

#[test]
fn test_kv_assignment_mqa() {
    // MQA with 1 KV head, tp_size=4: replicate on all ranks.
    let assign = kv_head_assignment(0, 4, 1);
    assert_eq!(assign, KVAssignment::Replicated(0..1));
    assert!(assign.is_replicated());
    assert_eq!(assign.num_heads(), 1);
}

#[test]
fn test_kv_assignment_single_rank() {
    // tp_size=1: always "sharded" (range covers everything).
    let assign = kv_head_assignment(0, 1, 8);
    assert_eq!(assign, KVAssignment::Sharded(0..8));
}

// ---------------------------------------------------------------------------
// Compute local attention shapes
// ---------------------------------------------------------------------------

#[test]
fn test_local_shapes_mha_tp2() {
    // Llama 3.1 8B: 32 heads, 8 KV heads, head_dim=128, tp_size=2
    let config = TPAttentionConfig {
        tp_rank: 0,
        tp_size: 2,
        total_heads: 32,
        total_kv_heads: 8,
        head_dim: 128,
        sliding_window: None,
    };
    let meta = compute_local_attention_shapes(&config).unwrap();

    assert_eq!(meta.local_n_heads, 16);
    assert_eq!(meta.local_n_kv_heads, 4);
    assert_eq!(meta.q_head_range, 0..16);
    assert_eq!(meta.kv_assignment, KVAssignment::Sharded(0..4));
    assert_eq!(meta.local_q_dim, 16 * 128);
    assert_eq!(meta.local_k_dim, 4 * 128);
    assert_eq!(meta.local_v_dim, 4 * 128);
    assert_eq!(meta.local_o_input_dim, 16 * 128);
    assert!(meta.needs_allreduce);
    assert_eq!(meta.attention_type, AttentionType::GQA);
}

#[test]
fn test_local_shapes_mqa_tp4() {
    // MQA model: 32 Q heads, 1 KV head, tp_size=4
    let config = TPAttentionConfig {
        tp_rank: 2,
        tp_size: 4,
        total_heads: 32,
        total_kv_heads: 1,
        head_dim: 64,
        sliding_window: None,
    };
    let meta = compute_local_attention_shapes(&config).unwrap();

    assert_eq!(meta.local_n_heads, 8);
    assert_eq!(meta.local_n_kv_heads, 1); // Replicated
    assert_eq!(meta.q_head_range, 16..24);
    assert!(meta.kv_assignment.is_replicated());
    assert_eq!(meta.local_q_dim, 8 * 64);
    assert_eq!(meta.local_k_dim, 64);
    assert_eq!(meta.local_v_dim, 64);
    assert!(meta.needs_allreduce);
    assert_eq!(meta.attention_type, AttentionType::MQA);
}

#[test]
fn test_local_shapes_single_rank() {
    let config = TPAttentionConfig {
        tp_rank: 0,
        tp_size: 1,
        total_heads: 32,
        total_kv_heads: 8,
        head_dim: 128,
        sliding_window: None,
    };
    let meta = compute_local_attention_shapes(&config).unwrap();

    assert_eq!(meta.local_n_heads, 32);
    assert_eq!(meta.local_n_kv_heads, 8);
    assert!(!meta.needs_allreduce);
}

#[test]
fn test_local_shapes_with_sliding_window() {
    // Gemma 3 style: sliding window + GQA
    let config = TPAttentionConfig {
        tp_rank: 0,
        tp_size: 2,
        total_heads: 16,
        total_kv_heads: 8,
        head_dim: 256,
        sliding_window: Some(4096),
    };
    let meta = compute_local_attention_shapes(&config).unwrap();

    assert_eq!(meta.sliding_window, Some(4096));
    assert_eq!(meta.local_n_heads, 8);
    assert_eq!(meta.local_n_kv_heads, 4);
}

// ---------------------------------------------------------------------------
// requires_allreduce_after_o_proj
// ---------------------------------------------------------------------------

#[test]
fn test_allreduce_required() {
    assert!(!requires_allreduce_after_o_proj(1));
    assert!(requires_allreduce_after_o_proj(2));
    assert!(requires_allreduce_after_o_proj(4));
    assert!(requires_allreduce_after_o_proj(8));
}

// ---------------------------------------------------------------------------
// Full head coverage verification
// ---------------------------------------------------------------------------

#[test]
fn test_verify_head_coverage_mha_tp4() {
    let all_meta = compute_all_rank_metadata(4, 32, 32, 128, None).unwrap();
    assert!(verify_head_coverage(&all_meta, 32, 32).is_ok());
}

#[test]
fn test_verify_head_coverage_gqa_tp4() {
    let all_meta = compute_all_rank_metadata(4, 32, 8, 128, None).unwrap();
    assert!(verify_head_coverage(&all_meta, 32, 8).is_ok());
}

#[test]
fn test_verify_head_coverage_mqa_tp4() {
    let all_meta = compute_all_rank_metadata(4, 32, 1, 64, None).unwrap();
    assert!(verify_head_coverage(&all_meta, 32, 1).is_ok());
}

#[test]
fn test_verify_head_coverage_gqa_replicated_tp8() {
    // 4 KV heads with 8 ranks: KV heads replicated
    let all_meta = compute_all_rank_metadata(8, 64, 4, 128, None).unwrap();
    assert!(verify_head_coverage(&all_meta, 64, 4).is_ok());

    // Verify every rank has replicated KV heads
    for meta in &all_meta {
        assert!(meta.kv_assignment.is_replicated());
        assert_eq!(meta.local_n_kv_heads, 4);
    }
}

// ---------------------------------------------------------------------------
// Real model configurations
// ---------------------------------------------------------------------------

#[test]
fn test_llama3_1_8b_tp2() {
    // Llama 3.1 8B: 32 Q heads, 8 KV heads, head_dim=128
    let all_meta = compute_all_rank_metadata(2, 32, 8, 128, None).unwrap();
    assert!(verify_head_coverage(&all_meta, 32, 8).is_ok());

    assert_eq!(all_meta[0].local_n_heads, 16);
    assert_eq!(all_meta[0].local_n_kv_heads, 4);
    assert_eq!(all_meta[0].attention_type, AttentionType::GQA);
    assert_eq!(all_meta[0].local_q_dim, 2048);
    assert_eq!(all_meta[0].local_k_dim, 512);
}

#[test]
fn test_qwen2_5_7b_tp4() {
    // Qwen 2.5 7B: 28 Q heads, 4 KV heads, head_dim=128
    let all_meta = compute_all_rank_metadata(4, 28, 4, 128, None).unwrap();
    assert!(verify_head_coverage(&all_meta, 28, 4).is_ok());

    assert_eq!(all_meta[0].local_n_heads, 7);
    assert_eq!(all_meta[0].local_n_kv_heads, 1);
    assert_eq!(all_meta[0].kv_assignment, KVAssignment::Sharded(0..1));
}

#[test]
fn test_gemma3_4b_sliding_window_tp2() {
    // Gemma 3 4B: 8 Q heads, 4 KV heads, head_dim=256, sliding_window=4096
    let all_meta = compute_all_rank_metadata(2, 8, 4, 256, Some(4096)).unwrap();
    assert!(verify_head_coverage(&all_meta, 8, 4).is_ok());

    assert_eq!(all_meta[0].local_n_heads, 4);
    assert_eq!(all_meta[0].local_n_kv_heads, 2);
    assert_eq!(all_meta[0].sliding_window, Some(4096));
}

#[test]
fn test_mistral_7b_tp2() {
    // Mistral 7B: 32 Q heads, 8 KV heads, head_dim=128, sliding_window=4096
    let all_meta = compute_all_rank_metadata(2, 32, 8, 128, Some(4096)).unwrap();
    assert!(verify_head_coverage(&all_meta, 32, 8).is_ok());

    assert_eq!(all_meta[0].local_n_heads, 16);
    assert_eq!(all_meta[0].local_n_kv_heads, 4);
    assert_eq!(all_meta[0].sliding_window, Some(4096));
}

// ---------------------------------------------------------------------------
// KVAssignment Display / utility
// ---------------------------------------------------------------------------

#[test]
fn test_kv_assignment_head_range() {
    let sharded = KVAssignment::Sharded(4..8);
    assert_eq!(sharded.head_range(), &(4..8));
    assert_eq!(sharded.num_heads(), 4);
    assert!(!sharded.is_replicated());

    let replicated = KVAssignment::Replicated(0..2);
    assert_eq!(replicated.head_range(), &(0..2));
    assert_eq!(replicated.num_heads(), 2);
    assert!(replicated.is_replicated());
}

#[test]
fn test_attention_type_display() {
    assert_eq!(format!("{}", AttentionType::MHA), "MHA");
    assert_eq!(format!("{}", AttentionType::GQA), "GQA");
    assert_eq!(format!("{}", AttentionType::MQA), "MQA");
}
