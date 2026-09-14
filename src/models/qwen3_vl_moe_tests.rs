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

//! Regression test for the bound on declared quantization params that the
//! Qwen3-VL-MoE expert loader applies before it stores them on a quantized
//! expert plane (issue #958). Since issue #1884 that loader is the shared
//! `switch_layers::SwitchGLU::from_weights`.

use super::{Qwen3VLMoeConfig, SparseMoeBlock};
use crate::models::switch_layers::{HOSTILE_QUANT_PARAMS, insert_stacked_quantized_expert_plane};
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
