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

//! Unit tests for the DeepSeek v1 per-expert `SwitchLinear` loader, covering
//! the `baidu/Unlimited-OCR` raw-checkpoint path (`experts.{idx}.{proj}`
//! stacked into `switch_mlp` via the shared
//! `switch_layers::stack_individual_experts` helper).

use super::{ModelArgs, SwitchLinear};
use mlxcel_core::weights::WeightMap;

fn test_args(n_routed_experts: Option<usize>) -> ModelArgs {
    ModelArgs {
        model_type: "deepseek".to_string(),
        vocab_size: 8,
        hidden_size: 8,
        intermediate_size: 8,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        max_position_embeddings: 16,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        moe_intermediate_size: None,
        n_shared_experts: None,
        n_routed_experts,
        num_experts_per_tok: None,
        moe_layer_freq: 1,
        first_k_dense_replace: 0,
        routed_scaling_factor: 1.0,
        attention_bias: false,
        group_size: None,
        bits: None,
    }
}

fn insert_expert(weights: &mut WeightMap, root: &str, idx: usize, out: i32, in_dim: i32) {
    weights.insert(
        format!("{root}.experts.{idx}.gate_proj.weight"),
        mlxcel_core::from_slice_f32(&vec![0.0; (out * in_dim) as usize], &[out, in_dim]),
    );
}

/// `baidu/Unlimited-OCR`-style truncated checkpoint: the config declares 4
/// routed experts but the shard only carries experts 0..2 contiguously (a gap
/// at index 3, e.g. a dropped middle/trailing shard). Before the cross-check,
/// `stack_individual_experts` silently stacked only the 3 experts it found,
/// which would let the router's top-k gather index expert 3 out of bounds at
/// inference instead of failing at load time.
#[test]
fn switch_linear_errors_when_stacked_experts_fall_short_of_config_count() {
    // (`SwitchLinear` holds non-Debug MlxArray handles, so match on the Result
    // rather than using `expect_err`.)
    let root = "model.layers.0.mlp";
    let mut weights = WeightMap::new();
    for e in 0..3 {
        insert_expert(&mut weights, root, e, 4, 4);
    }
    let args = test_args(Some(4));

    let err = match SwitchLinear::from_weights(
        &weights,
        &args,
        &format!("{root}.switch_mlp.gate_proj"),
    ) {
        Ok(_) => panic!("stacking fewer experts than n_routed_experts declares must error"),
        Err(e) => e,
    };
    assert!(
        err.contains('4') && err.contains('3'),
        "error should name both the declared (4) and found (3) counts: {err}"
    );
}

/// The same declared count with every expert present must still load
/// (the cross-check only rejects a shortfall, never an exact match).
#[test]
fn switch_linear_accepts_full_expert_count() {
    let root = "model.layers.0.mlp";
    let mut weights = WeightMap::new();
    for e in 0..4 {
        insert_expert(&mut weights, root, e, 4, 4);
    }
    let args = test_args(Some(4));

    let sl = SwitchLinear::from_weights(&weights, &args, &format!("{root}.switch_mlp.gate_proj"))
        .expect("stacking exactly n_routed_experts experts must succeed");
    assert_eq!(sl.num_experts(), 4);
}
