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

use super::{KimiLinearConfig, KimiLinearModel, LinearAttnConfig, MultiLinear};
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
/// throw there is an uncatchable `std::terminate`,
/// so losing the bound turns a rejected load into an uncatchable abort at the
/// first routed forward pass in production. This test asserts on the load
/// result rather than running a forward pass, so a regression fails cleanly
/// here instead of aborting the test binary.
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

/// A minimal config where every layer is an MLA layer (`kda_layers` empty, so
/// `is_linear_layer` is always false) and MoE stacking is inert (`num_experts`
/// 0), so `sanitize_weights` only exercises the `kv_b_proj` decomposition this
/// test targets.
fn mla_config(kv_lora_rank: usize) -> KimiLinearConfig {
    KimiLinearConfig {
        model_type: "kimi_linear".to_string(),
        vocab_size: 32,
        hidden_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        intermediate_size: 64,
        head_dim: 8,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        linear_attn_config: LinearAttnConfig {
            kda_layers: vec![],
            num_heads: 4,
            head_dim: 8,
            short_conv_kernel_size: 4,
        },
        num_experts: 0,
        moe_intermediate_size: 0,
        kv_lora_rank,
        tie_word_embeddings: true,
        qk_nope_head_dim: Some(4),
        qk_rope_head_dim: Some(4),
        v_head_dim: Some(4),
        mla_use_nope: false,
        num_experts_per_token: 1,
        num_shared_experts: 0,
        moe_router_activation_func: "sigmoid".to_string(),
        moe_renormalize: true,
        routed_scaling_factor: 1.0,
        first_k_dense_replace: 0,
        moe_layer_freq: 1,
        use_grouped_topk: true,
        num_expert_group: 1,
        topk_group: 1,
        quantization: None,
    }
}

/// The MLA `kv_b_proj` pair `KimiLinearModel::sanitize_weights` decomposes is
/// solved from `kv_lora_rank` and two tensor axes rather than declared, and
/// every input is untrusted: `kv_lora_rank` comes from `config.json` and the
/// axes from the checkpoint. Before issue #958 the naive arithmetic divided by
/// both without checking them, so a `kv_lora_rank` of 0 panicked on integer
/// division and a solved pair outside anything MLX can describe reached
/// `dequantize`, which crosses the cxx bridge as `UniquePtr<MlxArray>` rather
/// than `Result` and therefore aborts during weight sanitization rather than
/// failing the load.
///
/// This drives the real `KimiLinearModel::sanitize_weights` rather than the
/// shared helper directly, so the guard is exercised where a checkpoint
/// reaches it. The assertions are on the returned `Result`, so a regression
/// fails cleanly here instead of aborting the test binary.
#[test]
fn kimi_linear_sanitize_rejects_a_kv_lora_rank_no_packing_can_describe() {
    // Honest affine 4-bit geometry: packed_in * 32 == bits * num_groups *
    // group_size (2 * 32 == 4 * 1 * 16), with num_heads 4 and
    // qk_nope_head_dim = v_head_dim = 4 so rows = 4 * 8 = 32.
    let build = |config: &KimiLinearConfig| {
        let rows = (config.num_attention_heads * (config.qk_nope() + config.v_head())) as i32;
        let mut weights = WeightMap::new();
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer_idx}.self_attn.kv_b_proj");
            // UINT32, not a float dtype: `dequantize` rejects any other packed
            // dtype by throwing, and that throw is the uncatchable abort this
            // guard exists to keep out of the forward path. A float fixture here
            // takes the test binary down on the positive control.
            weights.insert(
                format!("{prefix}.weight"),
                mlxcel_core::zeros(&[rows, 2], mlxcel_core::dtype::UINT32),
            );
            weights.insert(
                format!("{prefix}.scales"),
                mlxcel_core::zeros(&[rows, 1], mlxcel_core::dtype::FLOAT32),
            );
            weights.insert(
                format!("{prefix}.biases"),
                mlxcel_core::zeros(&[rows, 1], mlxcel_core::dtype::FLOAT32),
            );
        }
        weights
    };

    // Positive control first, so a guard that rejected every quantized
    // kv_b_proj could not pass this test.
    let honest = mla_config(16);
    let sanitized = match KimiLinearModel::sanitize_weights(build(&honest), &honest) {
        Ok(w) => w,
        Err(e) => panic!("an honest quantized kv_b_proj must still sanitize: {e}"),
    };
    assert!(sanitized.contains_key("model.layers.0.self_attn.embed_q.weight"));

    // A zero `kv_lora_rank` is Rust integer division by zero on the very
    // first solve, so it has to be refused before the division rather than
    // after.
    let zero_rank = mla_config(0);
    let err = KimiLinearModel::sanitize_weights(build(&zero_rank), &zero_rank)
        .err()
        .unwrap_or_else(|| panic!("kv_lora_rank 0 must be refused, not divided by"));
    assert!(err.contains("kv_lora_rank"), "unhelpful error: {err}");

    // A large `kv_lora_rank` truncates the solved bit width to 0, which is
    // the divisor MLX would then divide by.
    let wide_rank = mla_config(4096);
    let err = KimiLinearModel::sanitize_weights(build(&wide_rank), &wide_rank)
        .err()
        .unwrap_or_else(|| panic!("a solved bit width of 0 must be refused"));
    assert!(err.contains("bits"), "unhelpful error: {err}");

    // A non-quantized kv_b_proj carries no packing, so the pair is never
    // solved and a `kv_lora_rank` that no packing could describe must not gate
    // it. The tensor is built at `wide_rank`'s own width so it still satisfies
    // the separate shape cross-check below the solve, which would otherwise
    // reject this for an unrelated reason.
    let mut float_only = WeightMap::new();
    let rows = (wide_rank.num_attention_heads * (wide_rank.qk_nope() + wide_rank.v_head())) as i32;
    float_only.insert(
        "model.layers.0.self_attn.kv_b_proj.weight".to_string(),
        mlxcel_core::zeros(
            &[rows, wide_rank.kv_lora_rank as i32],
            mlxcel_core::dtype::FLOAT32,
        ),
    );
    match KimiLinearModel::sanitize_weights(float_only, &wide_rank) {
        Ok(_) => {}
        Err(e) => panic!("a float kv_b_proj must not be gated on quantization params: {e}"),
    }
}
