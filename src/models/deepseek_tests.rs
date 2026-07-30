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

/// The same minimal config with an explicit quantization pair. DeepSeek v1
/// declares `group_size` / `bits` as flat top-level keys rather than a nested
/// `quantization` block, so those are the two fields the guard reads.
fn quant_args(group_size: i32, bits: i32) -> ModelArgs {
    let mut args = test_args(None);
    args.group_size = Some(group_size);
    args.bits = Some(bits);
    args
}

/// `SwitchLinear::from_stacked_parts` stores the declared pair verbatim on the
/// quantized variant and hands it straight to `gather_qmm`, which crosses the
/// cxx bridge as `UniquePtr<MlxArray>` rather than `Result`. A C++ throw there
/// is an uncatchable `std::terminate`,
/// so losing the bound turns a rejected load into an uncatchable abort at the
/// first routed forward pass in production. This test asserts on the load
/// result rather than running a forward pass, so a regression fails cleanly
/// here instead of aborting the test binary.
#[test]
fn deepseek_switch_linear_rejects_quantization_params_that_would_abort_gather_qmm() {
    // Honest 4-bit expert geometry: `packed_in * 32 == bits * num_groups *
    // group_size` (8 * 32 == 4 * 1 * 64), so the positive control below is a
    // plane MLX can actually describe.
    const EXPERTS: i32 = 3;
    const OUT: i32 = 4;
    const PACKED_IN: i32 = 8;
    const NUM_GROUPS: i32 = 1;
    const GROUP_SIZE: i32 = 64;
    const BITS: i32 = 4;
    const PREFIX: &str = "model.layers.0.mlp.switch_mlp.gate_proj";

    let mut weights = WeightMap::new();
    crate::models::switch_layers::insert_stacked_quantized_expert_plane(
        &mut weights,
        PREFIX,
        EXPERTS,
        OUT,
        PACKED_IN,
        NUM_GROUPS,
    );

    // Positive control first, so a guard that rejected every quantized plane
    // could not pass this test.
    SwitchLinear::from_weights(&weights, &quant_args(GROUP_SIZE, BITS), PREFIX)
        .expect("honest 4-bit expert plane must load");

    for (group_size, bits, field) in crate::models::switch_layers::HOSTILE_QUANT_PARAMS {
        let err = match SwitchLinear::from_weights(&weights, &quant_args(group_size, bits), PREFIX)
        {
            Ok(_) => panic!(
                "(group_size {group_size}, bits {bits}) must be refused at load, \
                 not stored for gather_qmm"
            ),
            Err(e) => e,
        };
        assert!(
            err.contains(field),
            "(group_size {group_size}, bits {bits}) must be blamed on {field}, got: {err}"
        );
    }

    // A bf16 expert plane carries no packing at all, so the declared pair is
    // irrelevant there and must not gate the non-quantized fallback.
    let mut regular = WeightMap::new();
    regular.insert(
        format!("{PREFIX}.weight"),
        mlxcel_core::ones(&[EXPERTS, OUT, PACKED_IN], mlxcel_core::dtype::BFLOAT16),
    );
    SwitchLinear::from_weights(&regular, &quant_args(0, 0), PREFIX)
        .expect("a bf16 expert plane must not be gated on quantization params");
}
