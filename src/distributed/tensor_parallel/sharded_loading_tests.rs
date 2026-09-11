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

use std::collections::HashMap;

use super::*;
use crate::distributed::tensor_parallel::shard_strategy::{
    CommPattern, LayerShardPlan, ModelShardPlan, ShardStrategy,
};

fn make_plan(tp_size: usize) -> ModelShardPlan {
    ModelShardPlan {
        tp_size,
        num_layers: 2,
        layer_plans: vec![
            LayerShardPlan {
                weight_pattern: "model.layers.{}.self_attn.q_proj.weight".to_string(),
                strategy: ShardStrategy::ColumnParallel,
                shard_axis: 0,
                comm_pattern: CommPattern::None,
            },
            LayerShardPlan {
                weight_pattern: "model.layers.{}.self_attn.o_proj.weight".to_string(),
                strategy: ShardStrategy::RowParallel,
                shard_axis: 1,
                comm_pattern: CommPattern::AllReduce,
            },
            LayerShardPlan {
                weight_pattern: "model.layers.{}.mlp.gate_proj.weight".to_string(),
                strategy: ShardStrategy::ColumnParallel,
                shard_axis: 0,
                comm_pattern: CommPattern::None,
            },
        ],
        embedding_strategy: ShardStrategy::Replicated,
        lm_head_strategy: ShardStrategy::Replicated,
        architecture: "llama".to_string(),
    }
}

// ---------------------------------------------------------------------------
// compute_shard_spec tests
// ---------------------------------------------------------------------------

#[test]
fn test_replicated_weight_returns_full_range() {
    let plan = make_plan(4);
    let shape = [768, 768];
    // LayerNorm is not in the plan, so it should be replicated.
    let spec =
        compute_shard_spec("model.layers.0.input_layernorm.weight", &shape, &plan, 0).unwrap();
    assert!(spec.is_replicated());
    assert_eq!(spec.start_index, 0);
    assert_eq!(spec.end_index, 768);
}

#[test]
fn test_column_parallel_even_split() {
    let plan = make_plan(4);
    let shape = [768, 768];
    let spec =
        compute_shard_spec("model.layers.0.self_attn.q_proj.weight", &shape, &plan, 0).unwrap();
    assert_eq!(spec.strategy, ShardStrategy::ColumnParallel);
    assert_eq!(spec.shard_axis, 0);
    assert_eq!(spec.start_index, 0);
    assert_eq!(spec.end_index, 192);
    assert!(!spec.padded);

    let spec3 =
        compute_shard_spec("model.layers.0.self_attn.q_proj.weight", &shape, &plan, 3).unwrap();
    assert_eq!(spec3.start_index, 576);
    assert_eq!(spec3.end_index, 768);
}

#[test]
fn test_row_parallel_even_split() {
    let plan = make_plan(4);
    let shape = [768, 768];
    let spec =
        compute_shard_spec("model.layers.1.self_attn.o_proj.weight", &shape, &plan, 2).unwrap();
    assert_eq!(spec.strategy, ShardStrategy::RowParallel);
    assert_eq!(spec.shard_axis, 1);
    assert_eq!(spec.start_index, 384);
    assert_eq!(spec.end_index, 576);
}

#[test]
fn test_non_divisible_dimension_remainder_distribution() {
    let plan = make_plan(4);
    // 770 / 4 = 192 remainder 2. First 2 ranks get 193, last 2 get 192.
    let shape = [770, 768];

    let spec0 =
        compute_shard_spec("model.layers.0.self_attn.q_proj.weight", &shape, &plan, 0).unwrap();
    assert_eq!(spec0.start_index, 0);
    assert_eq!(spec0.end_index, 193);
    assert_eq!(spec0.shard_size(), 193);

    let spec1 =
        compute_shard_spec("model.layers.0.self_attn.q_proj.weight", &shape, &plan, 1).unwrap();
    assert_eq!(spec1.start_index, 193);
    assert_eq!(spec1.end_index, 386);
    assert_eq!(spec1.shard_size(), 193);

    let spec2 =
        compute_shard_spec("model.layers.0.self_attn.q_proj.weight", &shape, &plan, 2).unwrap();
    assert_eq!(spec2.start_index, 386);
    assert_eq!(spec2.end_index, 578);
    assert_eq!(spec2.shard_size(), 192);

    let spec3 =
        compute_shard_spec("model.layers.0.self_attn.q_proj.weight", &shape, &plan, 3).unwrap();
    assert_eq!(spec3.start_index, 578);
    assert_eq!(spec3.end_index, 770);
    assert_eq!(spec3.shard_size(), 192);

    // Total should equal original dimension.
    assert_eq!(
        spec0.shard_size() + spec1.shard_size() + spec2.shard_size() + spec3.shard_size(),
        770
    );
}

#[test]
fn test_embedding_replicated_by_default() {
    let plan = make_plan(4);
    let shape = [32000, 4096];
    let spec = compute_shard_spec("model.embed_tokens.weight", &shape, &plan, 0).unwrap();
    assert!(spec.is_replicated());
}

#[test]
fn test_embedding_vocab_parallel() {
    let mut plan = make_plan(4);
    plan.embedding_strategy = ShardStrategy::VocabParallel;
    let shape = [32000, 4096];
    let spec = compute_shard_spec("model.embed_tokens.weight", &shape, &plan, 1).unwrap();
    assert_eq!(spec.strategy, ShardStrategy::VocabParallel);
    assert_eq!(spec.shard_axis, 0);
    assert_eq!(spec.start_index, 8000);
    assert_eq!(spec.end_index, 16000);
}

#[test]
fn test_lm_head_vocab_parallel() {
    let mut plan = make_plan(2);
    plan.lm_head_strategy = ShardStrategy::VocabParallel;
    let shape = [32000, 4096];
    let spec = compute_shard_spec("lm_head.weight", &shape, &plan, 0).unwrap();
    assert_eq!(spec.strategy, ShardStrategy::VocabParallel);
    assert_eq!(spec.start_index, 0);
    assert_eq!(spec.end_index, 16000);
}

#[test]
fn test_rank_out_of_range() {
    let plan = make_plan(4);
    let shape = [768, 768];
    let result = compute_shard_spec("model.layers.0.self_attn.q_proj.weight", &shape, &plan, 5);
    assert!(result.is_err());
}

#[test]
fn test_empty_shape_error() {
    let plan = make_plan(4);
    let shape: [usize; 0] = [];
    let result = compute_shard_spec("model.layers.0.self_attn.q_proj.weight", &shape, &plan, 0);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// compute_sharded_shape tests
// ---------------------------------------------------------------------------

#[test]
fn test_sharded_shape_replicated() {
    let spec = ShardSpec {
        rank: 0,
        tp_size: 4,
        shard_axis: 0,
        start_index: 0,
        end_index: 768,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::Replicated,
    };
    assert_eq!(compute_sharded_shape(&[768, 768], &spec), vec![768, 768]);
}

#[test]
fn test_sharded_shape_column_parallel() {
    let spec = ShardSpec {
        rank: 0,
        tp_size: 4,
        shard_axis: 0,
        start_index: 0,
        end_index: 192,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::ColumnParallel,
    };
    assert_eq!(compute_sharded_shape(&[768, 768], &spec), vec![192, 768]);
}

#[test]
fn test_sharded_shape_row_parallel() {
    let spec = ShardSpec {
        rank: 1,
        tp_size: 2,
        shard_axis: 1,
        start_index: 384,
        end_index: 768,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::RowParallel,
    };
    assert_eq!(compute_sharded_shape(&[768, 768], &spec), vec![768, 384]);
}

// ---------------------------------------------------------------------------
// shard_tensor_data tests
// ---------------------------------------------------------------------------

#[test]
fn test_shard_axis0_contiguous() {
    // 4x4 tensor of u8, shard into 2 ranks on axis 0.
    // [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
    let data: Vec<u8> = (0..16).collect();
    let shape = [4, 4];
    let spec = ShardSpec {
        rank: 0,
        tp_size: 2,
        shard_axis: 0,
        start_index: 0,
        end_index: 2,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::ColumnParallel,
    };
    let result = shard_tensor_data(&data, &shape, 1, &spec).unwrap();
    assert_eq!(result, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    let spec1 = ShardSpec {
        rank: 1,
        tp_size: 2,
        shard_axis: 0,
        start_index: 2,
        end_index: 4,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::ColumnParallel,
    };
    let result1 = shard_tensor_data(&data, &shape, 1, &spec1).unwrap();
    assert_eq!(result1, vec![8, 9, 10, 11, 12, 13, 14, 15]);
}

#[test]
fn test_shard_axis1_strided() {
    // 4x4 tensor of u8, shard into 2 ranks on axis 1.
    let data: Vec<u8> = (0..16).collect();
    let shape = [4, 4];

    let spec = ShardSpec {
        rank: 0,
        tp_size: 2,
        shard_axis: 1,
        start_index: 0,
        end_index: 2,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::RowParallel,
    };
    let result = shard_tensor_data(&data, &shape, 1, &spec).unwrap();
    // Rows: [0,1], [4,5], [8,9], [12,13]
    assert_eq!(result, vec![0, 1, 4, 5, 8, 9, 12, 13]);

    let spec1 = ShardSpec {
        rank: 1,
        tp_size: 2,
        shard_axis: 1,
        start_index: 2,
        end_index: 4,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::RowParallel,
    };
    let result1 = shard_tensor_data(&data, &shape, 1, &spec1).unwrap();
    // Rows: [2,3], [6,7], [10,11], [14,15]
    assert_eq!(result1, vec![2, 3, 6, 7, 10, 11, 14, 15]);
}

#[test]
fn test_shard_replicated_returns_full_data() {
    let data: Vec<u8> = (0..16).collect();
    let shape = [4, 4];
    let spec = ShardSpec {
        rank: 0,
        tp_size: 2,
        shard_axis: 0,
        start_index: 0,
        end_index: 4,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::Replicated,
    };
    let result = shard_tensor_data(&data, &shape, 1, &spec).unwrap();
    assert_eq!(result, data);
}

#[test]
fn test_shard_with_dtype_size_2() {
    // 2x4 tensor of float16 (2 bytes each) = 16 bytes total.
    let data: Vec<u8> = (0..16).collect();
    let shape = [2, 4];
    // Shard axis 0 into 2 ranks.
    let spec = ShardSpec {
        rank: 1,
        tp_size: 2,
        shard_axis: 0,
        start_index: 1,
        end_index: 2,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::ColumnParallel,
    };
    let result = shard_tensor_data(&data, &shape, 2, &spec).unwrap();
    // Row 1 = bytes 8..16
    assert_eq!(result, vec![8, 9, 10, 11, 12, 13, 14, 15]);
}

#[test]
fn test_shard_data_too_short() {
    let data: Vec<u8> = vec![0; 10]; // Too short for 4x4 tensor.
    let shape = [4, 4];
    let spec = ShardSpec {
        rank: 0,
        tp_size: 2,
        shard_axis: 0,
        start_index: 0,
        end_index: 2,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::ColumnParallel,
    };
    let result = shard_tensor_data(&data, &shape, 1, &spec);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// compute_byte_ranges tests
// ---------------------------------------------------------------------------

#[test]
fn test_byte_ranges_axis0() {
    let shape = [8, 4];
    let spec = ShardSpec {
        rank: 1,
        tp_size: 2,
        shard_axis: 0,
        start_index: 4,
        end_index: 8,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::ColumnParallel,
    };
    let br = compute_byte_ranges(&shape, 2, &spec).unwrap();
    assert_eq!(br.ranges.len(), 1);
    // 4 rows * 4 cols * 2 bytes = 32 bytes starting at offset 32.
    assert_eq!(br.ranges[0], (32, 64));
    assert_eq!(br.total_bytes, 32);
}

#[test]
fn test_byte_ranges_axis1() {
    let shape = [4, 8];
    let spec = ShardSpec {
        rank: 0,
        tp_size: 2,
        shard_axis: 1,
        start_index: 0,
        end_index: 4,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::RowParallel,
    };
    let br = compute_byte_ranges(&shape, 2, &spec).unwrap();
    // 4 outer rows, each reading 4 * 2 = 8 bytes from a stride of 16 bytes.
    assert_eq!(br.ranges.len(), 4);
    assert_eq!(br.ranges[0], (0, 8));
    assert_eq!(br.ranges[1], (16, 24));
    assert_eq!(br.ranges[2], (32, 40));
    assert_eq!(br.ranges[3], (48, 56));
    assert_eq!(br.total_bytes, 32);
}

#[test]
fn test_byte_ranges_replicated() {
    let shape = [4, 4];
    let spec = ShardSpec {
        rank: 0,
        tp_size: 2,
        shard_axis: 0,
        start_index: 0,
        end_index: 4,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::Replicated,
    };
    let br = compute_byte_ranges(&shape, 4, &spec).unwrap();
    assert_eq!(br.ranges.len(), 1);
    assert_eq!(br.ranges[0], (0, 64));
    assert_eq!(br.total_bytes, 64);
}

// ---------------------------------------------------------------------------
// validate_sharded_memory tests
// ---------------------------------------------------------------------------

#[test]
fn test_memory_validation_consistent() {
    let plan = make_plan(2);
    let mut shapes = HashMap::new();
    let mut dtype_sizes = HashMap::new();

    // Sharded weight: q_proj (column-parallel on axis 0)
    shapes.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        vec![768, 768],
    );
    dtype_sizes.insert("model.layers.0.self_attn.q_proj.weight".to_string(), 2);

    // Replicated weight: layernorm
    shapes.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        vec![768],
    );
    dtype_sizes.insert("model.layers.0.input_layernorm.weight".to_string(), 2);

    let report = validate_sharded_memory(&plan, &shapes, &dtype_sizes).unwrap();

    // q_proj: 768*768*2 = 1_179_648 bytes total, each rank gets 384*768*2 = 589_824
    // layernorm: 768*2 = 1_536 bytes, replicated on both ranks
    assert_eq!(report.total_original_bytes, 1_179_648 + 1_536);
    assert_eq!(report.replicated_bytes, 1_536);
    assert_eq!(report.sharded_bytes, 1_179_648);
    assert_eq!(report.num_sharded_weights, 1);
    assert_eq!(report.num_replicated_weights, 1);
    assert!(report.is_consistent());

    // Each rank: 589_824 (sharded) + 1_536 (replicated) = 591_360
    assert_eq!(report.per_rank_bytes[0], 591_360);
    assert_eq!(report.per_rank_bytes[1], 591_360);
}

#[test]
fn test_memory_validation_savings_ratio() {
    let plan = make_plan(4);
    let mut shapes = HashMap::new();
    let mut dtype_sizes = HashMap::new();

    // Only sharded weights for simplicity.
    shapes.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        vec![1024, 1024],
    );
    dtype_sizes.insert("model.layers.0.self_attn.q_proj.weight".to_string(), 2);

    let report = validate_sharded_memory(&plan, &shapes, &dtype_sizes).unwrap();
    // Each rank gets 1/4 of the weight.
    let savings = report.savings_ratio();
    assert!(savings > 0.7, "expected > 0.7 savings, got {savings}");
}

#[test]
fn test_memory_validation_non_divisible() {
    let plan = make_plan(4);
    let mut shapes = HashMap::new();
    let mut dtype_sizes = HashMap::new();

    // 770 / 4 = 192 r2. Ranks 0,1 get 193; ranks 2,3 get 192.
    shapes.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        vec![770, 768],
    );
    dtype_sizes.insert("model.layers.0.self_attn.q_proj.weight".to_string(), 2);

    let report = validate_sharded_memory(&plan, &shapes, &dtype_sizes).unwrap();
    assert!(report.is_consistent());
    // Sum of sharded portions should equal original.
    let sum: usize = report.per_rank_bytes.iter().sum();
    assert_eq!(sum, 770 * 768 * 2);
}

// ---------------------------------------------------------------------------
// dtype_byte_size tests
// ---------------------------------------------------------------------------

#[test]
fn test_dtype_byte_sizes() {
    assert_eq!(dtype_byte_size("float32"), 4);
    assert_eq!(dtype_byte_size("float16"), 2);
    assert_eq!(dtype_byte_size("bfloat16"), 2);
    assert_eq!(dtype_byte_size("int8"), 1);
    assert_eq!(dtype_byte_size("float64"), 8);
    assert_eq!(dtype_byte_size("unknown"), 2); // default
}

// ---------------------------------------------------------------------------
// 3D tensor sharding tests
// ---------------------------------------------------------------------------

#[test]
fn test_shard_3d_tensor_axis0() {
    // Shape [4, 2, 3], dtype_size=1, shard axis 0 into 2 ranks.
    let data: Vec<u8> = (0..24).collect();
    let shape = [4, 2, 3];

    let spec = ShardSpec {
        rank: 0,
        tp_size: 2,
        shard_axis: 0,
        start_index: 0,
        end_index: 2,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::ColumnParallel,
    };
    let result = shard_tensor_data(&data, &shape, 1, &spec).unwrap();
    // First 2 "slices" of axis 0: elements 0..12
    assert_eq!(result, &data[0..12]);
}

#[test]
fn test_shard_3d_tensor_axis1() {
    // Shape [2, 4, 3], dtype_size=1, shard axis 1 into 2 ranks.
    let data: Vec<u8> = (0..24).collect();
    let shape = [2, 4, 3];

    let spec = ShardSpec {
        rank: 0,
        tp_size: 2,
        shard_axis: 1,
        start_index: 0,
        end_index: 2,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::RowParallel,
    };
    let result = shard_tensor_data(&data, &shape, 1, &spec).unwrap();
    // Outer dim = 2, each outer slice has 4*3=12 bytes.
    // From each outer slice, take first 2 rows of axis 1 = 2*3=6 bytes.
    // Slice 0: [0,1,2,3,4,5], Slice 1: [12,13,14,15,16,17]
    assert_eq!(result, vec![0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]);
}

// ---------------------------------------------------------------------------
// compute_shard_range edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_shard_range_tp1() {
    let (start, end, padded, pad_count) = compute_shard_range(100, 1, 0);
    assert_eq!((start, end), (0, 100));
    assert!(!padded);
    assert_eq!(pad_count, 0);
}

#[test]
fn test_shard_range_dim_less_than_tp() {
    // 3 elements across 4 ranks: ranks 0,1,2 get 1 each, rank 3 gets 0.
    let (s0, e0, _, _) = compute_shard_range(3, 4, 0);
    let (s1, e1, _, _) = compute_shard_range(3, 4, 1);
    let (s2, e2, _, _) = compute_shard_range(3, 4, 2);
    let (s3, e3, p3, _) = compute_shard_range(3, 4, 3);
    assert_eq!((s0, e0), (0, 1));
    assert_eq!((s1, e1), (1, 2));
    assert_eq!((s2, e2), (2, 3));
    assert_eq!((s3, e3), (3, 3)); // empty shard
    assert!(p3); // marked as padded (empty)
}

// ---------------------------------------------------------------------------
// ShardSpec helper tests
// ---------------------------------------------------------------------------

#[test]
fn test_shard_spec_shard_size() {
    let spec = ShardSpec {
        rank: 0,
        tp_size: 4,
        shard_axis: 0,
        start_index: 0,
        end_index: 192,
        padded: false,
        pad_count: 0,
        strategy: ShardStrategy::ColumnParallel,
    };
    assert_eq!(spec.shard_size(), 192);
    assert!(!spec.is_replicated());
}

// ---------------------------------------------------------------------------
// Expert-parallel tests
// ---------------------------------------------------------------------------

#[test]
fn test_expert_parallel_shard_spec() {
    let plan = ModelShardPlan {
        tp_size: 2,
        num_layers: 1,
        layer_plans: vec![LayerShardPlan {
            weight_pattern: "model.layers.{}.mlp.experts".to_string(),
            strategy: ShardStrategy::ExpertParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::AllReduce,
        }],
        embedding_strategy: ShardStrategy::Replicated,
        lm_head_strategy: ShardStrategy::Replicated,
        architecture: "mixtral".to_string(),
    };

    // 8 experts across 2 ranks.
    let shape = [8, 4096, 4096];
    let spec0 = compute_shard_spec("model.layers.0.mlp.experts", &shape, &plan, 0).unwrap();
    assert_eq!(spec0.strategy, ShardStrategy::ExpertParallel);
    assert_eq!(spec0.start_index, 0);
    assert_eq!(spec0.end_index, 4);

    let spec1 = compute_shard_spec("model.layers.0.mlp.experts", &shape, &plan, 1).unwrap();
    assert_eq!(spec1.start_index, 4);
    assert_eq!(spec1.end_index, 8);
}
