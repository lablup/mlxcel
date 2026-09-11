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

#[test]
fn shard_strategy_display() {
    assert_eq!(ShardStrategy::ColumnParallel.to_string(), "column_parallel");
    assert_eq!(ShardStrategy::RowParallel.to_string(), "row_parallel");
    assert_eq!(ShardStrategy::ExpertParallel.to_string(), "expert_parallel");
    assert_eq!(ShardStrategy::VocabParallel.to_string(), "vocab_parallel");
    assert_eq!(ShardStrategy::Replicated.to_string(), "replicated");
}

#[test]
fn comm_pattern_display() {
    assert_eq!(CommPattern::None.to_string(), "none");
    assert_eq!(CommPattern::AllReduce.to_string(), "all_reduce");
    assert_eq!(CommPattern::AllGather.to_string(), "all_gather");
    assert_eq!(CommPattern::ReduceScatter.to_string(), "reduce_scatter");
}

#[test]
fn model_shard_plan_expand_weight_names() {
    let plan = ModelShardPlan {
        tp_size: 2,
        num_layers: 3,
        layer_plans: vec![
            LayerShardPlan {
                weight_pattern: "model.layers.{}.self_attn.q_proj.weight".to_string(),
                strategy: ShardStrategy::ColumnParallel,
                shard_axis: 0,
                comm_pattern: CommPattern::None,
            },
            LayerShardPlan {
                weight_pattern: "model.embed_tokens.weight".to_string(),
                strategy: ShardStrategy::VocabParallel,
                shard_axis: 0,
                comm_pattern: CommPattern::AllGather,
            },
        ],
        embedding_strategy: ShardStrategy::VocabParallel,
        lm_head_strategy: ShardStrategy::Replicated,
        architecture: "test".to_string(),
    };

    let expanded = plan.expand_weight_names();
    // 3 layers for the templated pattern + 1 non-templated
    assert_eq!(expanded.len(), 4);
    assert_eq!(expanded[0].0, "model.layers.0.self_attn.q_proj.weight");
    assert_eq!(expanded[1].0, "model.layers.1.self_attn.q_proj.weight");
    assert_eq!(expanded[2].0, "model.layers.2.self_attn.q_proj.weight");
    assert_eq!(expanded[3].0, "model.embed_tokens.weight");
}

#[test]
fn plan_for_weight_pattern_match() {
    let plan = ModelShardPlan {
        tp_size: 2,
        num_layers: 32,
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
        ],
        embedding_strategy: ShardStrategy::Replicated,
        lm_head_strategy: ShardStrategy::Replicated,
        architecture: "test".to_string(),
    };

    // Match layer 15 Q proj
    let found = plan.plan_for_weight("model.layers.15.self_attn.q_proj.weight");
    assert!(found.is_some());
    assert_eq!(found.unwrap().strategy, ShardStrategy::ColumnParallel);

    // Match layer 31 O proj
    let found = plan.plan_for_weight("model.layers.31.self_attn.o_proj.weight");
    assert!(found.is_some());
    assert_eq!(found.unwrap().strategy, ShardStrategy::RowParallel);

    // No match for unknown weight
    let found = plan.plan_for_weight("model.layers.0.input_layernorm.weight");
    assert!(found.is_none());
}

#[test]
fn plan_for_weight_exact_match() {
    let plan = ModelShardPlan {
        tp_size: 2,
        num_layers: 1,
        layer_plans: vec![LayerShardPlan {
            weight_pattern: "model.embed_tokens.weight".to_string(),
            strategy: ShardStrategy::VocabParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::AllGather,
        }],
        embedding_strategy: ShardStrategy::VocabParallel,
        lm_head_strategy: ShardStrategy::Replicated,
        architecture: "test".to_string(),
    };

    let found = plan.plan_for_weight("model.embed_tokens.weight");
    assert!(found.is_some());
    assert_eq!(found.unwrap().strategy, ShardStrategy::VocabParallel);
}

#[test]
fn summary_format() {
    let plan = ModelShardPlan {
        tp_size: 4,
        num_layers: 32,
        layer_plans: vec![
            LayerShardPlan {
                weight_pattern: "q".to_string(),
                strategy: ShardStrategy::ColumnParallel,
                shard_axis: 0,
                comm_pattern: CommPattern::None,
            },
            LayerShardPlan {
                weight_pattern: "o".to_string(),
                strategy: ShardStrategy::RowParallel,
                shard_axis: 1,
                comm_pattern: CommPattern::AllReduce,
            },
        ],
        embedding_strategy: ShardStrategy::Replicated,
        lm_head_strategy: ShardStrategy::VocabParallel,
        architecture: "llama".to_string(),
    };

    let summary = plan.summary();
    assert!(summary.contains("TP-4"));
    assert!(summary.contains("llama"));
    assert!(summary.contains("32 layers"));
    assert!(summary.contains("1 col-parallel"));
    assert!(summary.contains("1 row-parallel"));
    assert!(summary.contains("embedding=replicated"));
    assert!(summary.contains("lm_head=vocab_parallel"));
}
