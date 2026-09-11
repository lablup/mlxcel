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
use crate::distributed::tensor_parallel::config::{EmbeddingMode, MoeShardMode, ShardConfig};
use crate::distributed::tensor_parallel::shard_strategy::{CommPattern, ShardStrategy};

/// Helper: assert a plan has the expected attention patterns.
fn assert_has_attention_plans(plan: &ModelShardPlan) {
    let q = plan.plan_for_weight("model.layers.0.self_attn.q_proj.weight");
    assert!(q.is_some(), "missing Q projection plan");
    assert_eq!(q.unwrap().strategy, ShardStrategy::ColumnParallel);

    let k = plan.plan_for_weight("model.layers.0.self_attn.k_proj.weight");
    assert!(k.is_some(), "missing K projection plan");
    assert_eq!(k.unwrap().strategy, ShardStrategy::ColumnParallel);

    let v = plan.plan_for_weight("model.layers.0.self_attn.v_proj.weight");
    assert!(v.is_some(), "missing V projection plan");
    assert_eq!(v.unwrap().strategy, ShardStrategy::ColumnParallel);

    let o = plan.plan_for_weight("model.layers.0.self_attn.o_proj.weight");
    assert!(o.is_some(), "missing O projection plan");
    assert_eq!(o.unwrap().strategy, ShardStrategy::RowParallel);
    assert_eq!(o.unwrap().comm_pattern, CommPattern::AllReduce);
}

/// Helper: assert a plan has the expected FFN gate/up/down patterns.
fn assert_has_ffn_plans(plan: &ModelShardPlan) {
    let gate = plan.plan_for_weight("model.layers.0.mlp.gate_proj.weight");
    assert!(gate.is_some(), "missing gate_proj plan");
    assert_eq!(gate.unwrap().strategy, ShardStrategy::ColumnParallel);

    let up = plan.plan_for_weight("model.layers.0.mlp.up_proj.weight");
    assert!(up.is_some(), "missing up_proj plan");
    assert_eq!(up.unwrap().strategy, ShardStrategy::ColumnParallel);

    let down = plan.plan_for_weight("model.layers.0.mlp.down_proj.weight");
    assert!(down.is_some(), "missing down_proj plan");
    assert_eq!(down.unwrap().strategy, ShardStrategy::RowParallel);
    assert_eq!(down.unwrap().comm_pattern, CommPattern::AllReduce);
}

// ---- tp_size=1 returns replicated plan ----

#[test]
fn tp_size_1_returns_replicated() {
    let config = ShardConfig::with_tp_size(1);
    let plan = generate_shard_plan("llama", 32, &config).unwrap();
    assert_eq!(plan.tp_size, 1);
    assert!(plan.layer_plans.is_empty());
    assert_eq!(plan.embedding_strategy, ShardStrategy::Replicated);
    assert_eq!(plan.lm_head_strategy, ShardStrategy::Replicated);
}

// ---- Llama family ----

#[test]
fn llama_plan_has_correct_strategies() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("llama", 32, &config).unwrap();
    assert_eq!(plan.tp_size, 2);
    assert_eq!(plan.num_layers, 32);
    assert_eq!(plan.architecture, "llama");
    assert_has_attention_plans(&plan);
    assert_has_ffn_plans(&plan);
}

#[test]
fn llama_plan_with_vocab_parallel_embedding() {
    let config = ShardConfig {
        tp_size: 4,
        embedding_mode: EmbeddingMode::VocabParallel,
        lm_head_mode: EmbeddingMode::VocabParallel,
        ..Default::default()
    };
    let plan = generate_shard_plan("llama", 32, &config).unwrap();
    assert_eq!(plan.embedding_strategy, ShardStrategy::VocabParallel);
    assert_eq!(plan.lm_head_strategy, ShardStrategy::VocabParallel);
}

#[test]
fn mistral_uses_llama_plan() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("mistral", 32, &config).unwrap();
    assert_eq!(plan.architecture, "llama");
    assert_has_attention_plans(&plan);
    assert_has_ffn_plans(&plan);
}

// ---- Llama 4 (MoE) ----

#[test]
fn llama4_plan_has_moe() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("llama4", 48, &config).unwrap();
    assert_eq!(plan.architecture, "llama4");
    assert_has_attention_plans(&plan);
    // Has both dense FFN and MoE expert plans
    assert_has_ffn_plans(&plan);
    // MoE expert plan present
    let expert = plan.plan_for_weight("model.layers.0.feed_forward.experts");
    assert!(expert.is_some());
    assert_eq!(expert.unwrap().strategy, ShardStrategy::ExpertParallel);
}

// ---- Qwen ----

#[test]
fn qwen_plan() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("qwen2", 32, &config).unwrap();
    assert_eq!(plan.architecture, "qwen");
    assert_has_attention_plans(&plan);
    assert_has_ffn_plans(&plan);
}

#[test]
fn qwen_moe_plan_expert_parallel() {
    let config = ShardConfig {
        tp_size: 2,
        moe_mode: MoeShardMode::ExpertParallel,
        ..Default::default()
    };
    let plan = generate_shard_plan("qwen2_moe", 24, &config).unwrap();
    assert_eq!(plan.architecture, "qwen_moe");
    assert_has_attention_plans(&plan);
    // MoE gate is replicated
    let gate = plan.plan_for_weight("model.layers.0.mlp.gate.weight");
    assert!(gate.is_some());
    assert_eq!(gate.unwrap().strategy, ShardStrategy::Replicated);
    // Experts are expert-parallel
    let experts = plan.plan_for_weight("model.layers.0.mlp.experts");
    assert!(experts.is_some());
    assert_eq!(experts.unwrap().strategy, ShardStrategy::ExpertParallel);
}

#[test]
fn qwen_moe_plan_within_expert() {
    let config = ShardConfig {
        tp_size: 2,
        moe_mode: MoeShardMode::WithinExpert,
        ..Default::default()
    };
    let plan = generate_shard_plan("qwen3_moe", 24, &config).unwrap();
    // Within-expert: individual expert FFN weights are sharded
    let found = plan
        .layer_plans
        .iter()
        .any(|p| p.weight_pattern.contains("experts.*.gate_proj"));
    assert!(found, "should have within-expert gate_proj pattern");
}

// ---- Gemma ----

#[test]
fn gemma_plan() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("gemma3", 26, &config).unwrap();
    assert_eq!(plan.architecture, "gemma");
    assert_has_attention_plans(&plan);
    assert_has_ffn_plans(&plan);
}

// ---- Phi ----

#[test]
fn phi_plan_has_both_naming_conventions() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("phi3", 32, &config).unwrap();
    assert_eq!(plan.architecture, "phi");
    assert_has_attention_plans(&plan);
    // Phi includes both gate/up/down and fc1/fc2 patterns
    let fc1 = plan.plan_for_weight("model.layers.0.mlp.fc1.weight");
    assert!(fc1.is_some());
    assert_eq!(fc1.unwrap().strategy, ShardStrategy::ColumnParallel);
    let fc2 = plan.plan_for_weight("model.layers.0.mlp.fc2.weight");
    assert!(fc2.is_some());
    assert_eq!(fc2.unwrap().strategy, ShardStrategy::RowParallel);
}

// ---- Mixtral ----

#[test]
fn mixtral_plan() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("mixtral", 32, &config).unwrap();
    assert_eq!(plan.architecture, "mixtral");
    assert_has_attention_plans(&plan);
    // MoE through block_sparse_moe
    let gate = plan.plan_for_weight("model.layers.0.block_sparse_moe.gate.weight");
    assert!(gate.is_some());
}

// ---- DeepSeek ----

#[test]
fn deepseek_dense_plan() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("deepseek", 32, &config).unwrap();
    assert_eq!(plan.architecture, "deepseek");
    assert_has_attention_plans(&plan);
    assert_has_ffn_plans(&plan);
}

#[test]
fn deepseek_moe_plan_has_mla() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("deepseek_v3", 61, &config).unwrap();
    assert_eq!(plan.architecture, "deepseek_moe");

    // MLA-specific projections
    let q_a = plan.plan_for_weight("model.layers.0.self_attn.q_a_proj.weight");
    assert!(q_a.is_some());
    assert_eq!(q_a.unwrap().strategy, ShardStrategy::ColumnParallel);

    // KV-A is replicated (shared compressed KV)
    let kv_a = plan.plan_for_weight("model.layers.0.self_attn.kv_a_proj_with_mqa.weight");
    assert!(kv_a.is_some());
    assert_eq!(kv_a.unwrap().strategy, ShardStrategy::Replicated);

    // O projection is row-parallel
    let o = plan.plan_for_weight("model.layers.0.self_attn.o_proj.weight");
    assert!(o.is_some());
    assert_eq!(o.unwrap().strategy, ShardStrategy::RowParallel);

    // Shared experts
    let shared_gate = plan.plan_for_weight("model.layers.0.mlp.shared_experts.gate_proj.weight");
    assert!(shared_gate.is_some());
    assert_eq!(shared_gate.unwrap().strategy, ShardStrategy::ColumnParallel);
}

// ---- Cohere ----

#[test]
fn cohere_plan() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("cohere2", 40, &config).unwrap();
    assert_eq!(plan.architecture, "cohere");
    assert_has_attention_plans(&plan);
    assert_has_ffn_plans(&plan);
}

// ---- StarCoder2 ----

#[test]
fn starcoder2_plan() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("starcoder2", 40, &config).unwrap();
    assert_eq!(plan.architecture, "starcoder2");
    assert_has_attention_plans(&plan);
    let fc1 = plan.plan_for_weight("model.layers.0.mlp.fc1.weight");
    assert!(fc1.is_some());
    assert_eq!(fc1.unwrap().strategy, ShardStrategy::ColumnParallel);
}

// ---- SSM models are replicated ----

#[test]
fn ssm_models_are_replicated() {
    let config = ShardConfig::with_tp_size(2);
    for arch in &[
        "mamba",
        "mamba2",
        "falcon_mamba",
        "jamba",
        "nemotron_h",
        "rwkv7",
        "recurrent_gemma",
        "qwen3_next",
    ] {
        let plan = generate_shard_plan(arch, 32, &config).unwrap();
        assert_eq!(plan.tp_size, 1, "SSM model {arch} should be replicated");
        assert!(
            plan.layer_plans.is_empty(),
            "SSM model {arch} should have no layer plans"
        );
    }
}

// ---- Kimi K3 has no tensor-parallel port (issue #1334); it is replicated
// next to its `kimi_linear` sibling rather than sharded like a dense
// transformer would be. ----

#[test]
fn kimi_k3_is_replicated() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("kimi_k3", 61, &config).unwrap();
    assert_eq!(plan.tp_size, 1, "kimi_k3 (hybrid) should be replicated");
    assert_eq!(plan.architecture, "kimi_k3");
    assert!(plan.layer_plans.is_empty());
}

// ---- Unknown architecture falls back to generic ----

#[test]
fn unknown_architecture_uses_generic_fallback() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("some_future_model", 24, &config).unwrap();
    assert_eq!(plan.architecture, "some_future_model");
    assert_has_attention_plans(&plan);
    assert_has_ffn_plans(&plan);
}

// ---- Invalid config rejected ----

#[test]
fn invalid_tp_size_rejected() {
    let config = ShardConfig::with_tp_size(3);
    let result = generate_shard_plan("llama", 32, &config);
    assert!(result.is_err());
}

// ---- Weight expansion ----

#[test]
fn expand_weight_names_count() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("llama", 4, &config).unwrap();
    let expanded = plan.expand_weight_names();
    // 7 patterns (4 attention + 3 FFN), each expanded for 4 layers = 28
    assert_eq!(expanded.len(), 7 * 4);
}

// ---- Summary includes architecture info ----

#[test]
fn plan_summary_contains_key_info() {
    let config = ShardConfig::with_tp_size(4);
    let plan = generate_shard_plan("llama", 32, &config).unwrap();
    let summary = plan.summary();
    assert!(summary.contains("TP-4"));
    assert!(summary.contains("llama"));
    assert!(summary.contains("32 layers"));
}

// ---- OLMo uses llama plan ----

#[test]
fn olmo_family_uses_llama_plan() {
    let config = ShardConfig::with_tp_size(2);
    for arch in &["olmo", "olmo2", "olmo3"] {
        let plan = generate_shard_plan(arch, 32, &config).unwrap();
        assert_eq!(plan.architecture, "llama");
        assert_has_attention_plans(&plan);
        assert_has_ffn_plans(&plan);
    }
}

// ---- OLMoE ----

#[test]
fn olmoe_plan() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("olmoe", 16, &config).unwrap();
    assert_eq!(plan.architecture, "olmoe");
    assert_has_attention_plans(&plan);
}

// ---- PhiMoE ----

#[test]
fn phimoe_plan() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("phimoe", 32, &config).unwrap();
    assert_eq!(plan.architecture, "phimoe");
    assert_has_attention_plans(&plan);
    let gate = plan.plan_for_weight("model.layers.0.block_sparse_moe.gate.weight");
    assert!(gate.is_some());
}

// ---- Qwen 3.5 ----

#[test]
fn qwen3_5_plan() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("qwen3_5", 36, &config).unwrap();
    assert_eq!(plan.architecture, "qwen3_5");
    assert_has_attention_plans(&plan);
    assert_has_ffn_plans(&plan);
    assert!(
        plan.plan_for_weight("model.layers.0.linear_attn.in_proj_qkv.weight")
            .is_some()
    );
    assert!(
        plan.plan_for_weight("model.layers.0.linear_attn.out_proj.weight")
            .is_some()
    );
}

#[test]
fn qwen3_5_moe_plan() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("qwen3_5_moe", 28, &config).unwrap();
    assert_eq!(plan.architecture, "qwen_moe");
    assert_has_attention_plans(&plan);
}

#[test]
fn qwen3_next_is_replicated() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("qwen3_next", 32, &config).unwrap();
    assert_eq!(plan.tp_size, 1, "qwen3_next (hybrid) should be replicated");
    assert!(plan.layer_plans.is_empty());
}

#[test]
fn deepseek_moe_plan_has_dense_ffn() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("deepseek_v3", 61, &config).unwrap();
    // Dense FFN layers (first few layers) should also be sharded
    let gate = plan.plan_for_weight("model.layers.0.mlp.gate_proj.weight");
    assert!(gate.is_some(), "dense FFN gate_proj should be in plan");
    assert_eq!(gate.unwrap().strategy, ShardStrategy::ColumnParallel);
    let down = plan.plan_for_weight("model.layers.0.mlp.down_proj.weight");
    assert!(down.is_some(), "dense FFN down_proj should be in plan");
    assert_eq!(down.unwrap().strategy, ShardStrategy::RowParallel);
}
