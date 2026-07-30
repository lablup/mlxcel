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

//! Regression test for the bound on declared quantization params that
//! `kimi_linear::MultiLinear::from_weights` applies before it stores them on a
//! quantized per-head MLA projection (issue #958).

use super::MultiLinear;
use crate::models::switch_layers::{HOSTILE_QUANT_PARAMS, insert_stacked_quantized_expert_plane};
use mlxcel_core::weights::WeightMap;

/// Honest 4-bit geometry: `packed_in * 32 == bits * num_groups * group_size`
/// (8 * 32 == 4 * 1 * 64), so the positive control below is a projection MLX can
/// actually describe. `MultiLinear` is a per-head projection rather than an
/// expert plane, but its tensors carry the same `[heads, out, packed_in]` /
/// `[heads, out, num_groups]` shapes the shared fixture helper builds.
const HEADS: i32 = 3;
const OUT: i32 = 4;
const PACKED_IN: i32 = 8;
const NUM_GROUPS: i32 = 1;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

const PREFIX: &str = "model.layers.0.self_attn.embed_q";

/// The pair this loader stores is handed straight to `quantized_matmul`, which
/// crosses the cxx bridge as `UniquePtr<MlxArray>` rather than `Result`. A C++
/// throw there is an uncatchable `std::terminate`, so losing the bound would
/// abort the whole test binary with SIGABRT at the first MLA forward pass
/// instead of failing cleanly at load. Issue #958.
#[test]
fn kimi_linear_multi_linear_rejects_quantization_params_that_would_abort_quantized_matmul() {
    let mut weights = WeightMap::new();
    insert_stacked_quantized_expert_plane(&mut weights, PREFIX, HEADS, OUT, PACKED_IN, NUM_GROUPS);

    // Positive control first, so a guard that rejected every quantized
    // projection could not pass this test.
    match MultiLinear::from_weights(&weights, PREFIX, GROUP_SIZE, BITS) {
        Ok(_) => {}
        Err(e) => panic!("honest 4-bit per-head projection must load: {e}"),
    }

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        let err = match MultiLinear::from_weights(&weights, PREFIX, group_size, bits) {
            Ok(_) => panic!(
                "(group_size {group_size}, bits {bits}) must be refused at load, \
                 not stored for quantized_matmul"
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

    // A bf16 projection carries no packing and no `.scales`, so the declared
    // pair is inert on the dense `matmul` path and must not gate the load.
    let mut dense = WeightMap::new();
    let n = (HEADS * OUT * PACKED_IN) as usize;
    dense.insert(
        format!("{PREFIX}.weight"),
        mlxcel_core::from_slice_f32(&vec![0.0f32; n], &[HEADS, OUT, PACKED_IN]),
    );
    match MultiLinear::from_weights(&dense, PREFIX, 0, 0) {
        Ok(_) => {}
        Err(e) => panic!("a non-quantized per-head projection must load with an unset pair: {e}"),
    }
}
