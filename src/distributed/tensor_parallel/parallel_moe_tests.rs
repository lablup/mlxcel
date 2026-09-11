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
// MoEParallelMode
// ---------------------------------------------------------------------------

#[test]
fn test_mode_display() {
    assert_eq!(
        format!("{}", MoEParallelMode::ExpertParallel),
        "expert_parallel"
    );
    assert_eq!(
        format!("{}", MoEParallelMode::WithinExpertSharding),
        "within_expert_sharding"
    );
}

#[test]
fn test_mode_default() {
    assert_eq!(MoEParallelMode::default(), MoEParallelMode::ExpertParallel);
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn base_config() -> TPMoEConfig {
    TPMoEConfig {
        tp_rank: 0,
        tp_size: 2,
        num_experts: 8,
        experts_per_token: 2,
        expert_intermediate_size: 14336,
        hidden_size: 4096,
        has_shared_expert: false,
        shared_expert_intermediate_size: None,
        mode: MoEParallelMode::ExpertParallel,
        activation: FFNActivationType::SiLU,
    }
}

#[test]
fn test_validate_valid_expert_parallel() {
    assert!(validate_tp_moe_config(&base_config()).is_ok());
}

#[test]
fn test_validate_valid_within_expert() {
    let config = TPMoEConfig {
        mode: MoEParallelMode::WithinExpertSharding,
        ..base_config()
    };
    assert!(validate_tp_moe_config(&config).is_ok());
}

#[test]
fn test_validate_rank_out_of_range() {
    let config = TPMoEConfig {
        tp_rank: 2,
        ..base_config()
    };
    assert!(validate_tp_moe_config(&config).is_err());
}

#[test]
fn test_validate_zero_experts() {
    let config = TPMoEConfig {
        num_experts: 0,
        ..base_config()
    };
    assert!(validate_tp_moe_config(&config).is_err());
}

#[test]
fn test_validate_experts_per_token_exceeds_total() {
    let config = TPMoEConfig {
        experts_per_token: 10,
        ..base_config()
    };
    assert!(validate_tp_moe_config(&config).is_err());
}

#[test]
fn test_validate_expert_parallel_insufficient_experts() {
    // 4 experts with tp_size=8: not enough for expert parallel
    let config = TPMoEConfig {
        tp_size: 8,
        num_experts: 4,
        mode: MoEParallelMode::ExpertParallel,
        ..base_config()
    };
    assert!(validate_tp_moe_config(&config).is_err());
}

#[test]
fn test_validate_within_expert_not_divisible() {
    let config = TPMoEConfig {
        tp_size: 3,
        expert_intermediate_size: 14336,
        mode: MoEParallelMode::WithinExpertSharding,
        ..base_config()
    };
    assert!(validate_tp_moe_config(&config).is_err());
}

#[test]
fn test_validate_shared_expert_not_divisible() {
    let config = TPMoEConfig {
        has_shared_expert: true,
        shared_expert_intermediate_size: Some(14335), // not divisible by 2
        ..base_config()
    };
    assert!(validate_tp_moe_config(&config).is_err());
}

// ---------------------------------------------------------------------------
// Expert assignment
// ---------------------------------------------------------------------------

#[test]
fn test_expert_assignment_balanced() {
    // 8 experts, 4 ranks: 2 experts per rank
    for rank in 0..4 {
        let assignment = compute_expert_assignment(rank, 4, 8);
        assert_eq!(assignment.num_local_experts(), 2);
        assert_eq!(assignment.total_experts, 8);
    }

    let a0 = compute_expert_assignment(0, 4, 8);
    assert_eq!(a0.expert_indices, vec![0, 4]);
    assert!(a0.owns_expert(0));
    assert!(a0.owns_expert(4));
    assert!(!a0.owns_expert(1));

    let a1 = compute_expert_assignment(1, 4, 8);
    assert_eq!(a1.expert_indices, vec![1, 5]);

    let a2 = compute_expert_assignment(2, 4, 8);
    assert_eq!(a2.expert_indices, vec![2, 6]);

    let a3 = compute_expert_assignment(3, 4, 8);
    assert_eq!(a3.expert_indices, vec![3, 7]);
}

#[test]
fn test_expert_assignment_unbalanced() {
    // 7 experts, 4 ranks: ranks 0-2 get 2 experts, rank 3 gets 1
    let a0 = compute_expert_assignment(0, 4, 7);
    assert_eq!(a0.expert_indices, vec![0, 4]);

    let a1 = compute_expert_assignment(1, 4, 7);
    assert_eq!(a1.expert_indices, vec![1, 5]);

    let a2 = compute_expert_assignment(2, 4, 7);
    assert_eq!(a2.expert_indices, vec![2, 6]);

    let a3 = compute_expert_assignment(3, 4, 7);
    assert_eq!(a3.expert_indices, vec![3]);
}

#[test]
fn test_expert_assignment_single_rank() {
    let a = compute_expert_assignment(0, 1, 8);
    assert_eq!(a.num_local_experts(), 8);
    assert_eq!(a.expert_indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn test_expert_assignment_one_per_rank() {
    // 4 experts, 4 ranks: exactly 1 each
    for rank in 0..4 {
        let a = compute_expert_assignment(rank, 4, 4);
        assert_eq!(a.num_local_experts(), 1);
        assert_eq!(a.expert_indices, vec![rank]);
    }
}

// ---------------------------------------------------------------------------
// MoE shapes — Expert Parallel
// ---------------------------------------------------------------------------

#[test]
fn test_moe_shapes_expert_parallel_tp2() {
    // Mixtral 8x7B: 8 experts, top-2, intermediate=14336, hidden=4096
    let config = TPMoEConfig {
        tp_rank: 0,
        tp_size: 2,
        num_experts: 8,
        experts_per_token: 2,
        expert_intermediate_size: 14336,
        hidden_size: 4096,
        has_shared_expert: false,
        shared_expert_intermediate_size: None,
        mode: MoEParallelMode::ExpertParallel,
        activation: FFNActivationType::SiLU,
    };
    let meta = compute_local_moe_shapes(&config).unwrap();

    assert_eq!(meta.tp_rank, 0);
    assert_eq!(meta.mode, MoEParallelMode::ExpertParallel);
    assert_eq!(meta.expert_assignment.num_local_experts(), 4);
    assert_eq!(meta.expert_assignment.expert_indices, vec![0, 2, 4, 6]);
    // In expert-parallel, each expert is unsharded (full intermediate).
    assert_eq!(meta.expert_ffn_meta.local_intermediate_size, 14336);
    assert!(!meta.expert_ffn_meta.needs_allreduce);
    assert!(!meta.has_shared_expert);
    assert!(meta.shared_expert_ffn_meta.is_none());
    assert_eq!(meta.router_weight_shape, [4096, 8]);
    assert!(meta.router_replicated);
    assert_eq!(meta.experts_per_token, 2);
    assert!(meta.needs_all_to_all);
    assert!(!meta.needs_allreduce);
}

#[test]
fn test_moe_shapes_expert_parallel_single_rank() {
    let config = TPMoEConfig {
        tp_rank: 0,
        tp_size: 1,
        num_experts: 8,
        experts_per_token: 2,
        expert_intermediate_size: 14336,
        hidden_size: 4096,
        has_shared_expert: false,
        shared_expert_intermediate_size: None,
        mode: MoEParallelMode::ExpertParallel,
        activation: FFNActivationType::SiLU,
    };
    let meta = compute_local_moe_shapes(&config).unwrap();

    assert_eq!(meta.expert_assignment.num_local_experts(), 8);
    assert!(!meta.needs_all_to_all); // single rank
    assert!(!meta.needs_allreduce);
}

// ---------------------------------------------------------------------------
// MoE shapes — Within-Expert Sharding
// ---------------------------------------------------------------------------

#[test]
fn test_moe_shapes_within_expert_tp2() {
    let config = TPMoEConfig {
        tp_rank: 0,
        tp_size: 2,
        num_experts: 8,
        experts_per_token: 2,
        expert_intermediate_size: 14336,
        hidden_size: 4096,
        has_shared_expert: false,
        shared_expert_intermediate_size: None,
        mode: MoEParallelMode::WithinExpertSharding,
        activation: FFNActivationType::SiLU,
    };
    let meta = compute_local_moe_shapes(&config).unwrap();

    assert_eq!(meta.mode, MoEParallelMode::WithinExpertSharding);
    // Within-expert: expert FFN is sharded
    assert_eq!(meta.expert_ffn_meta.local_intermediate_size, 7168);
    assert!(meta.expert_ffn_meta.needs_allreduce);
    assert!(!meta.needs_all_to_all);
    assert!(meta.needs_allreduce);
}

#[test]
fn test_moe_shapes_within_expert_tp4() {
    let config = TPMoEConfig {
        tp_rank: 3,
        tp_size: 4,
        num_experts: 8,
        experts_per_token: 2,
        expert_intermediate_size: 14336,
        hidden_size: 4096,
        has_shared_expert: false,
        shared_expert_intermediate_size: None,
        mode: MoEParallelMode::WithinExpertSharding,
        activation: FFNActivationType::SiLU,
    };
    let meta = compute_local_moe_shapes(&config).unwrap();

    assert_eq!(meta.expert_ffn_meta.local_intermediate_size, 3584);
    assert_eq!(meta.expert_ffn_meta.intermediate_range, 10752..14336);
}

// ---------------------------------------------------------------------------
// Shared expert (DeepSeek-style)
// ---------------------------------------------------------------------------

#[test]
fn test_moe_shapes_with_shared_expert() {
    // DeepSeek V3: 256 experts, top-8, shared expert with different intermediate
    let config = TPMoEConfig {
        tp_rank: 0,
        tp_size: 8,
        num_experts: 256,
        experts_per_token: 8,
        expert_intermediate_size: 2048,
        hidden_size: 7168,
        has_shared_expert: true,
        shared_expert_intermediate_size: Some(18432),
        mode: MoEParallelMode::ExpertParallel,
        activation: FFNActivationType::SiLU,
    };
    let meta = compute_local_moe_shapes(&config).unwrap();

    assert!(meta.has_shared_expert);
    let shared = meta.shared_expert_ffn_meta.as_ref().unwrap();
    // Shared expert is TP-sharded across 8 ranks: 18432 / 8 = 2304
    assert_eq!(shared.local_intermediate_size, 2304);
    assert!(shared.needs_allreduce);
    assert_eq!(shared.hidden_size, 7168);

    // Routed experts: expert-parallel, each unsharded
    assert_eq!(meta.expert_ffn_meta.local_intermediate_size, 2048);
    assert!(!meta.expert_ffn_meta.needs_allreduce);
    assert_eq!(meta.expert_assignment.num_local_experts(), 32); // 256/8

    assert_eq!(meta.router_weight_shape, [7168, 256]);
}

#[test]
fn test_moe_shapes_shared_expert_default_size() {
    // When shared_expert_intermediate_size is None, use expert_intermediate_size
    let config = TPMoEConfig {
        tp_rank: 0,
        tp_size: 2,
        num_experts: 8,
        experts_per_token: 2,
        expert_intermediate_size: 14336,
        hidden_size: 4096,
        has_shared_expert: true,
        shared_expert_intermediate_size: None,
        mode: MoEParallelMode::ExpertParallel,
        activation: FFNActivationType::SiLU,
    };
    let meta = compute_local_moe_shapes(&config).unwrap();

    let shared = meta.shared_expert_ffn_meta.as_ref().unwrap();
    assert_eq!(shared.local_intermediate_size, 7168); // 14336/2
}

// ---------------------------------------------------------------------------
// Expert coverage verification
// ---------------------------------------------------------------------------

#[test]
fn test_verify_expert_coverage_tp2() {
    let config = base_config();
    let all_meta = compute_all_rank_moe_metadata(&config).unwrap();
    assert!(verify_expert_coverage(&all_meta, 8).is_ok());
}

#[test]
fn test_verify_expert_coverage_tp4() {
    let config = TPMoEConfig {
        tp_size: 4,
        ..base_config()
    };
    let all_meta = compute_all_rank_moe_metadata(&config).unwrap();
    assert!(verify_expert_coverage(&all_meta, 8).is_ok());
}

#[test]
fn test_verify_expert_coverage_tp8() {
    let config = TPMoEConfig {
        tp_size: 8,
        ..base_config()
    };
    let all_meta = compute_all_rank_moe_metadata(&config).unwrap();
    assert!(verify_expert_coverage(&all_meta, 8).is_ok());
}

// ---------------------------------------------------------------------------
// Real model configurations
// ---------------------------------------------------------------------------

#[test]
fn test_mixtral_8x7b_tp2() {
    // Mixtral 8x7B: 8 experts, top-2, intermediate=14336, hidden=4096
    let config = TPMoEConfig {
        tp_rank: 0,
        tp_size: 2,
        num_experts: 8,
        experts_per_token: 2,
        expert_intermediate_size: 14336,
        hidden_size: 4096,
        has_shared_expert: false,
        shared_expert_intermediate_size: None,
        mode: MoEParallelMode::ExpertParallel,
        activation: FFNActivationType::SiLU,
    };
    let all_meta = compute_all_rank_moe_metadata(&config).unwrap();
    assert!(verify_expert_coverage(&all_meta, 8).is_ok());

    assert_eq!(all_meta[0].expert_assignment.num_local_experts(), 4);
    assert_eq!(all_meta[1].expert_assignment.num_local_experts(), 4);
}

#[test]
fn test_deepseek_v3_tp8() {
    // DeepSeek V3: 256 experts, top-8, expert_intermediate=2048, hidden=7168
    // Shared expert: intermediate=18432
    let config = TPMoEConfig {
        tp_rank: 0,
        tp_size: 8,
        num_experts: 256,
        experts_per_token: 8,
        expert_intermediate_size: 2048,
        hidden_size: 7168,
        has_shared_expert: true,
        shared_expert_intermediate_size: Some(18432),
        mode: MoEParallelMode::ExpertParallel,
        activation: FFNActivationType::SiLU,
    };
    let all_meta = compute_all_rank_moe_metadata(&config).unwrap();
    assert!(verify_expert_coverage(&all_meta, 256).is_ok());

    // Each rank: 256/8 = 32 experts
    for meta in &all_meta {
        assert_eq!(meta.expert_assignment.num_local_experts(), 32);
        assert!(meta.has_shared_expert);
        let shared = meta.shared_expert_ffn_meta.as_ref().unwrap();
        assert_eq!(shared.local_intermediate_size, 2304);
    }
}

#[test]
fn test_llama4_scout_tp4() {
    // Llama 4 Scout: 16 experts, top-1, intermediate=8192, hidden=5120
    let config = TPMoEConfig {
        tp_rank: 0,
        tp_size: 4,
        num_experts: 16,
        experts_per_token: 1,
        expert_intermediate_size: 8192,
        hidden_size: 5120,
        has_shared_expert: false,
        shared_expert_intermediate_size: None,
        mode: MoEParallelMode::ExpertParallel,
        activation: FFNActivationType::SiLU,
    };
    let all_meta = compute_all_rank_moe_metadata(&config).unwrap();
    assert!(verify_expert_coverage(&all_meta, 16).is_ok());

    // 16/4 = 4 experts per rank
    for meta in &all_meta {
        assert_eq!(meta.expert_assignment.num_local_experts(), 4);
        assert_eq!(meta.experts_per_token, 1);
    }
}

// ---------------------------------------------------------------------------
// Communication requirements
// ---------------------------------------------------------------------------

#[test]
fn test_communication_expert_parallel() {
    let config = TPMoEConfig {
        tp_size: 4,
        ..base_config()
    };
    let meta = compute_local_moe_shapes(&config).unwrap();
    assert!(meta.needs_all_to_all);
    assert!(!meta.needs_allreduce);
}

#[test]
fn test_communication_within_expert() {
    let config = TPMoEConfig {
        mode: MoEParallelMode::WithinExpertSharding,
        ..base_config()
    };
    let meta = compute_local_moe_shapes(&config).unwrap();
    assert!(!meta.needs_all_to_all);
    assert!(meta.needs_allreduce);
}

#[test]
fn test_communication_single_rank() {
    let config = TPMoEConfig {
        tp_rank: 0,
        tp_size: 1,
        num_experts: 8,
        experts_per_token: 2,
        expert_intermediate_size: 14336,
        hidden_size: 4096,
        has_shared_expert: false,
        shared_expert_intermediate_size: None,
        mode: MoEParallelMode::ExpertParallel,
        activation: FFNActivationType::SiLU,
    };
    let meta = compute_local_moe_shapes(&config).unwrap();
    assert!(!meta.needs_all_to_all);
    assert!(!meta.needs_allreduce);
}

// ---------------------------------------------------------------------------
// Router replication
// ---------------------------------------------------------------------------

#[test]
fn test_router_always_replicated() {
    let config = base_config();
    let all_meta = compute_all_rank_moe_metadata(&config).unwrap();
    for meta in &all_meta {
        assert!(meta.router_replicated);
        assert_eq!(meta.router_weight_shape, [4096, 8]);
    }
}
