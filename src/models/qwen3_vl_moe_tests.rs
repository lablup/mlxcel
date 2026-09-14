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

//! Qwen3-VL-MoE expert tests: the loader's bound on declared quantization
//! params (issue #958) and the fused-kernel `dff` decline (issue #1884), both
//! reached through the shared `switch_layers::SwitchGLU` the family now holds.

use super::{Qwen3VLMoeConfig, SparseMoeBlock};
use crate::models::switch_layers::{
    HONEST_EXPERT_GROUP_SIZE, HOSTILE_QUANT_PARAMS, insert_honest_affine_swiglu_experts,
    insert_stacked_quantized_expert_plane,
};
use mlxcel_core::weights::WeightMap;

/// Honest 4-bit expert geometry: `packed_in * 32 == bits * num_groups *
/// group_size` (8 * 32 == 4 * 1 * 64), so the positive control below is a plane
/// MLX can actually describe.
const EXPERTS: i32 = 3;
const OUT: i32 = 4;
const PACKED_IN: i32 = 8;
const NUM_GROUPS: i32 = 1;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

/// The decoder layer prefix `SparseMoeBlock::from_weights` is handed, and the
/// first expert plane its shared loader reads under it.
const LAYER_PREFIX: &str = "model.layers.0";
const PREFIX: &str = "model.layers.0.mlp.switch_mlp.gate_proj";

/// Smallest Qwen3-VL-MoE text config that parses. `quantization: None` leaves
/// the declared pair at the config type's `0` / `0` fallback, which is itself a
/// hostile pair and must still load a non-quantized plane.
fn config_with(quantization: Option<(i32, i32)>) -> Qwen3VLMoeConfig {
    let mut config = serde_json::json!({
        "hidden_size": OUT,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 1,
        "vocab_size": 8,
        "num_experts": EXPERTS,
        "num_experts_per_tok": 1,
    });
    if let Some((group_size, bits)) = quantization {
        config["quantization"] = serde_json::json!({ "group_size": group_size, "bits": bits });
    }
    serde_json::from_value(config).expect("test qwen3_vl_moe config must parse")
}

fn insert_router(weights: &mut WeightMap) {
    let n = (EXPERTS * OUT) as usize;
    weights.insert(
        format!("{LAYER_PREFIX}.mlp.gate.weight"),
        mlxcel_core::from_slice_f32(&vec![0.1f32; n], &[EXPERTS, OUT]),
    );
}

/// The pair this loader stores is handed straight to `gather_qmm`, which
/// crosses the cxx bridge as `UniquePtr<MlxArray>` rather than `Result`. A C++
/// throw there is an uncatchable `std::terminate`,
/// so losing the bound turns a rejected load into an uncatchable abort at the
/// first routed forward pass in production. This test asserts on the load
/// result rather than running a forward pass, so a regression fails cleanly
/// here instead of aborting the test binary.
///
/// It drives the family's own block loader rather than the shared
/// `SwitchLinear` directly, so it also pins that `SparseMoeBlock::from_weights`
/// routes the declared pair from `Qwen3VLMoeConfig` into a bounded loader.
#[test]
fn qwen3_vl_moe_switch_linear_rejects_quantization_params_that_would_abort_gather_qmm() {
    let mut weights = WeightMap::new();
    for leaf in ["gate_proj", "up_proj", "down_proj"] {
        insert_stacked_quantized_expert_plane(
            &mut weights,
            &format!("{LAYER_PREFIX}.mlp.switch_mlp.{leaf}"),
            EXPERTS,
            OUT,
            PACKED_IN,
            NUM_GROUPS,
        );
    }
    // A non-quantized router, so the declared pair can only reach the experts.
    insert_router(&mut weights);

    // Positive control first, so a guard that rejected every quantized plane
    // could not pass this test.
    if let Err(e) = SparseMoeBlock::from_weights(
        &weights,
        &config_with(Some((GROUP_SIZE, BITS))),
        LAYER_PREFIX,
    ) {
        panic!("honest 4-bit expert planes must load: {e}");
    }

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        let err = match SparseMoeBlock::from_weights(
            &weights,
            &config_with(Some((group_size, bits))),
            LAYER_PREFIX,
        ) {
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
        assert!(
            err.contains(PREFIX),
            "the load error must name the offending tensor {PREFIX}, got: {err}"
        );
    }

    // A bf16 expert plane carries no packing and no `.scales`, so the declared
    // pair is inert on the `gather_mm` path and must not gate the load, even at
    // the unset `0` / `0` fallback.
    let mut dense = WeightMap::new();
    let n = (EXPERTS * OUT * PACKED_IN) as usize;
    for leaf in ["gate_proj", "up_proj", "down_proj"] {
        dense.insert(
            format!("{LAYER_PREFIX}.mlp.switch_mlp.{leaf}.weight"),
            mlxcel_core::from_slice_f32(&vec![0.0f32; n], &[EXPERTS, OUT, PACKED_IN]),
        );
    }
    insert_router(&mut dense);
    if let Err(e) = SparseMoeBlock::from_weights(&dense, &config_with(None), LAYER_PREFIX) {
        panic!("a non-quantized expert plane must load with an unset pair: {e}");
    }
}

/// Qwen3-VL-MoE built the Qwen3-MoE expert copy by hand before issue #1884, so
/// it inherited that copy's missing `dff` bound. Through the family's own block
/// loader, experts of Dff 8256 (above the 4096 Metal and 8192 CUDA defaults)
/// must now decline the fused kernel, while Dff 64 on the same geometry must
/// still dispatch it. The positive half launches the kernel, which aborts on a
/// backend without a port (issue #1803), so it is gated on
/// `custom_kernels_available()`.
#[test]
fn qwen3_vl_moe_fused_kernel_declines_experts_wider_than_the_dff_bound() {
    if std::env::var_os("MLXCEL_FUSED_MOE_MAX_DFF").is_some() {
        eprintln!(
            "skipping: MLXCEL_FUSED_MOE_MAX_DFF is set, so the default bound this test pins is \
             not in force"
        );
        return;
    }
    let hidden = HONEST_EXPERT_GROUP_SIZE;
    let block = |dff: i32| {
        let mut weights = WeightMap::new();
        insert_honest_affine_swiglu_experts(
            &mut weights,
            &format!("{LAYER_PREFIX}.mlp.switch_mlp"),
            2,
            hidden,
            dff,
        );
        weights.insert(
            format!("{LAYER_PREFIX}.mlp.gate.weight"),
            mlxcel_core::from_slice_f32(&vec![0.1f32; (2 * hidden) as usize], &[2, hidden]),
        );
        SparseMoeBlock::from_weights(&weights, &config_with(Some((hidden, 4))), LAYER_PREFIX)
            .expect("honest affine 4-bit experts must load")
    };
    let x = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&vec![0.05f32; hidden as usize], &[1, hidden]),
        mlxcel_core::dtype::BFLOAT16,
    );
    let indices = mlxcel_core::from_slice_i32(&[0, 1], &[1, 2]);
    let scores = mlxcel_core::from_slice_f32(&[0.5, 0.5], &[1, 2]);

    assert!(
        block(8256)
            .experts
            .forward_fused_kernel(&x, &indices, &scores)
            .is_none(),
        "Dff 8256 is above both default bounds, so the fused kernel must decline"
    );

    if !mlxcel_core::custom_kernels_available() {
        eprintln!(
            "skipping the positive control: this backend has no fused MoE kernel port (#1803)"
        );
        return;
    }
    let dispatched = block(hidden)
        .experts
        .forward_fused_kernel(&x, &indices, &scores)
        .expect("Dff 64 is below both bounds, so the fused kernel must dispatch");
    mlxcel_core::eval(&dispatched);
    assert_eq!(
        mlxcel_core::array_shape(&dispatched)
            .iter()
            .product::<i32>(),
        hidden
    );
}
