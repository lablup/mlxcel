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
// Activation type
// ---------------------------------------------------------------------------

#[test]
fn test_activation_display() {
    assert_eq!(format!("{}", FFNActivationType::SiLU), "silu");
    assert_eq!(format!("{}", FFNActivationType::GELU), "gelu");
    assert_eq!(format!("{}", FFNActivationType::GELUApprox), "gelu_approx");
    assert_eq!(format!("{}", FFNActivationType::ReLU), "relu");
    assert_eq!(
        format!("{}", FFNActivationType::ReLUSquared),
        "relu_squared"
    );
}

#[test]
fn test_activation_properties() {
    for act in [
        FFNActivationType::SiLU,
        FFNActivationType::GELU,
        FFNActivationType::GELUApprox,
        FFNActivationType::ReLU,
        FFNActivationType::ReLUSquared,
    ] {
        assert!(act.is_elementwise());
        assert!(act.is_gated());
    }
}

#[test]
fn test_activation_default() {
    assert_eq!(FFNActivationType::default(), FFNActivationType::SiLU);
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[test]
fn test_validate_valid_config() {
    let config = TPFFNConfig {
        tp_rank: 0,
        tp_size: 2,
        intermediate_size: 14336,
        hidden_size: 4096,
        activation: FFNActivationType::SiLU,
    };
    assert!(validate_tp_ffn_config(&config).is_ok());
}

#[test]
fn test_validate_rank_out_of_range() {
    let config = TPFFNConfig {
        tp_rank: 4,
        tp_size: 4,
        intermediate_size: 14336,
        hidden_size: 4096,
        activation: FFNActivationType::SiLU,
    };
    assert!(validate_tp_ffn_config(&config).is_err());
}

#[test]
fn test_validate_zero_tp_size() {
    let config = TPFFNConfig {
        tp_rank: 0,
        tp_size: 0,
        intermediate_size: 14336,
        hidden_size: 4096,
        activation: FFNActivationType::SiLU,
    };
    assert!(validate_tp_ffn_config(&config).is_err());
}

#[test]
fn test_validate_zero_intermediate() {
    let config = TPFFNConfig {
        tp_rank: 0,
        tp_size: 1,
        intermediate_size: 0,
        hidden_size: 4096,
        activation: FFNActivationType::SiLU,
    };
    assert!(validate_tp_ffn_config(&config).is_err());
}

#[test]
fn test_validate_zero_hidden() {
    let config = TPFFNConfig {
        tp_rank: 0,
        tp_size: 1,
        intermediate_size: 14336,
        hidden_size: 0,
        activation: FFNActivationType::SiLU,
    };
    assert!(validate_tp_ffn_config(&config).is_err());
}

#[test]
fn test_validate_intermediate_not_divisible() {
    let config = TPFFNConfig {
        tp_rank: 0,
        tp_size: 3,
        intermediate_size: 14336,
        hidden_size: 4096,
        activation: FFNActivationType::SiLU,
    };
    assert!(validate_tp_ffn_config(&config).is_err());
}

// ---------------------------------------------------------------------------
// Local FFN shapes
// ---------------------------------------------------------------------------

#[test]
fn test_local_shapes_single_rank() {
    let config = TPFFNConfig {
        tp_rank: 0,
        tp_size: 1,
        intermediate_size: 14336,
        hidden_size: 4096,
        activation: FFNActivationType::SiLU,
    };
    let meta = compute_local_ffn_shapes(&config).unwrap();

    assert_eq!(meta.tp_rank, 0);
    assert_eq!(meta.local_intermediate_size, 14336);
    assert_eq!(meta.intermediate_range, 0..14336);
    assert_eq!(meta.hidden_size, 4096);
    assert_eq!(meta.gate_proj_shape, [4096, 14336]);
    assert_eq!(meta.up_proj_shape, [4096, 14336]);
    assert_eq!(meta.down_proj_shape, [14336, 4096]);
    assert!(!meta.needs_allreduce);
    assert_eq!(meta.activation, FFNActivationType::SiLU);
}

#[test]
fn test_local_shapes_tp2() {
    // Llama 3.1 8B: intermediate_size=14336, hidden_size=4096
    let config = TPFFNConfig {
        tp_rank: 0,
        tp_size: 2,
        intermediate_size: 14336,
        hidden_size: 4096,
        activation: FFNActivationType::SiLU,
    };
    let meta = compute_local_ffn_shapes(&config).unwrap();

    assert_eq!(meta.local_intermediate_size, 7168);
    assert_eq!(meta.intermediate_range, 0..7168);
    assert_eq!(meta.gate_proj_shape, [4096, 7168]);
    assert_eq!(meta.up_proj_shape, [4096, 7168]);
    assert_eq!(meta.down_proj_shape, [7168, 4096]);
    assert!(meta.needs_allreduce);

    // Rank 1
    let config1 = TPFFNConfig {
        tp_rank: 1,
        ..config
    };
    let meta1 = compute_local_ffn_shapes(&config1).unwrap();
    assert_eq!(meta1.intermediate_range, 7168..14336);
    assert_eq!(meta1.local_intermediate_size, 7168);
}

#[test]
fn test_local_shapes_tp4() {
    // Llama 3.1 70B: intermediate_size=28672, hidden_size=8192
    let config = TPFFNConfig {
        tp_rank: 2,
        tp_size: 4,
        intermediate_size: 28672,
        hidden_size: 8192,
        activation: FFNActivationType::SiLU,
    };
    let meta = compute_local_ffn_shapes(&config).unwrap();

    assert_eq!(meta.local_intermediate_size, 7168);
    assert_eq!(meta.intermediate_range, 14336..21504);
    assert_eq!(meta.gate_proj_shape, [8192, 7168]);
    assert_eq!(meta.up_proj_shape, [8192, 7168]);
    assert_eq!(meta.down_proj_shape, [7168, 8192]);
    assert!(meta.needs_allreduce);
}

// ---------------------------------------------------------------------------
// Full coverage verification
// ---------------------------------------------------------------------------

#[test]
fn test_verify_intermediate_coverage_tp2() {
    let all_meta = compute_all_rank_ffn_metadata(2, 14336, 4096, FFNActivationType::SiLU).unwrap();
    assert!(verify_intermediate_coverage(&all_meta, 14336).is_ok());
}

#[test]
fn test_verify_intermediate_coverage_tp4() {
    let all_meta = compute_all_rank_ffn_metadata(4, 28672, 8192, FFNActivationType::SiLU).unwrap();
    assert!(verify_intermediate_coverage(&all_meta, 28672).is_ok());
}

#[test]
fn test_verify_intermediate_coverage_tp8() {
    let all_meta = compute_all_rank_ffn_metadata(8, 28672, 8192, FFNActivationType::GELU).unwrap();
    assert!(verify_intermediate_coverage(&all_meta, 28672).is_ok());
}

// ---------------------------------------------------------------------------
// Real model configurations
// ---------------------------------------------------------------------------

#[test]
fn test_llama3_1_8b_ffn_tp2() {
    // Llama 3.1 8B: intermediate=14336, hidden=4096, SiLU
    let all_meta = compute_all_rank_ffn_metadata(2, 14336, 4096, FFNActivationType::SiLU).unwrap();
    assert!(verify_intermediate_coverage(&all_meta, 14336).is_ok());

    assert_eq!(all_meta[0].local_intermediate_size, 7168);
    assert_eq!(all_meta[1].local_intermediate_size, 7168);
    assert!(all_meta[0].needs_allreduce);
    assert!(all_meta[1].needs_allreduce);
}

#[test]
fn test_qwen2_5_7b_ffn_tp4() {
    // Qwen 2.5 7B: intermediate=18944, hidden=3584, SiLU
    let all_meta = compute_all_rank_ffn_metadata(4, 18944, 3584, FFNActivationType::SiLU).unwrap();
    assert!(verify_intermediate_coverage(&all_meta, 18944).is_ok());

    assert_eq!(all_meta[0].local_intermediate_size, 4736);
    for meta in &all_meta {
        assert!(meta.needs_allreduce);
        assert_eq!(meta.hidden_size, 3584);
    }
}

#[test]
fn test_gemma3_4b_ffn_tp2() {
    // Gemma 3 4B: intermediate=16384, hidden=2560, GELU
    let all_meta = compute_all_rank_ffn_metadata(2, 16384, 2560, FFNActivationType::GELU).unwrap();
    assert!(verify_intermediate_coverage(&all_meta, 16384).is_ok());

    assert_eq!(all_meta[0].local_intermediate_size, 8192);
    assert_eq!(all_meta[0].activation, FFNActivationType::GELU);
}

#[test]
fn test_deepseek_v3_ffn_tp8() {
    // DeepSeek V3 MLP expert: intermediate=18432, hidden=7168, SiLU
    let all_meta = compute_all_rank_ffn_metadata(8, 18432, 7168, FFNActivationType::SiLU).unwrap();
    assert!(verify_intermediate_coverage(&all_meta, 18432).is_ok());

    assert_eq!(all_meta[0].local_intermediate_size, 2304);
    for meta in &all_meta {
        assert!(meta.needs_allreduce);
    }
}

#[test]
fn test_mixtral_expert_ffn_tp2() {
    // Mixtral 8x7B expert: intermediate=14336, hidden=4096, SiLU
    let all_meta = compute_all_rank_ffn_metadata(2, 14336, 4096, FFNActivationType::SiLU).unwrap();
    assert!(verify_intermediate_coverage(&all_meta, 14336).is_ok());
    assert_eq!(all_meta[0].local_intermediate_size, 7168);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_ffn_shapes_equal_split() {
    // intermediate_size == tp_size: each rank gets 1 column
    let config = TPFFNConfig {
        tp_rank: 0,
        tp_size: 4,
        intermediate_size: 4,
        hidden_size: 16,
        activation: FFNActivationType::ReLU,
    };
    let meta = compute_local_ffn_shapes(&config).unwrap();
    assert_eq!(meta.local_intermediate_size, 1);
    assert_eq!(meta.intermediate_range, 0..1);
    assert_eq!(meta.gate_proj_shape, [16, 1]);
}

#[test]
fn test_gelu_approx_activation() {
    let config = TPFFNConfig {
        tp_rank: 0,
        tp_size: 2,
        intermediate_size: 8192,
        hidden_size: 2048,
        activation: FFNActivationType::GELUApprox,
    };
    let meta = compute_local_ffn_shapes(&config).unwrap();
    assert_eq!(meta.activation, FFNActivationType::GELUApprox);
    assert_eq!(meta.local_intermediate_size, 4096);
}
