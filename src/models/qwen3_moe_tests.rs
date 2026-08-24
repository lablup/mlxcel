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

//! Regression test for the bound on declared quantization params that the
//! Qwen3-MoE `SwitchLinear` loader applies before it stores them on a quantized
//! expert plane (issue #958).

use mlxcel_core::weights::WeightMap;

use super::{Attention, DecoderLayer, ModelArgs, Qwen3MoeModel, SwitchLinear};
use crate::models::switch_layers::{HOSTILE_QUANT_PARAMS, insert_stacked_quantized_expert_plane};

/// Honest 4-bit expert geometry: `packed_in * 32 == bits * num_groups *
/// group_size` (8 * 32 == 4 * 1 * 64), so the positive control below is a plane
/// MLX can actually describe.
const EXPERTS: i32 = 3;
const OUT: i32 = 4;
const PACKED_IN: i32 = 8;
const NUM_GROUPS: i32 = 1;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

const PREFIX: &str = "model.layers.0.mlp.switch_mlp.gate_proj";

/// Smallest Qwen3-MoE config that parses, varying only the declared
/// quantization pair. Qwen3-MoE carries it in a nested `quantization` block.
/// Built through serde rather than a struct literal so every
/// `#[serde(default)]` field fills itself in.
fn args_with(group_size: i32, bits: i32) -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "qwen3_moe",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_experts": 3,
        "num_experts_per_tok": 1,
        "decoder_sparse_step": 1,
        "moe_intermediate_size": 8,
        "rms_norm_eps": 1e-5,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "quantization": { "group_size": group_size, "bits": bits },
    }))
    .expect("test config must parse")
}

fn moe_args_with_rope(rope_scaling: Option<serde_json::Value>) -> ModelArgs {
    let mut config = serde_json::json!({
        "model_type": "qwen3_moe",
        "vocab_size": 8,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_experts": 0,
        "num_experts_per_tok": 1,
        "decoder_sparse_step": 1,
        "moe_intermediate_size": 8,
        "rms_norm_eps": 1e-5,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "tie_word_embeddings": false
    });
    if let Some(block) = rope_scaling {
        config["rope_scaling"] = block;
    }
    serde_json::from_value(config).expect("test qwen3_moe config must parse")
}

fn values(len: usize, start: f32, step: f32) -> Vec<f32> {
    (0..len).map(|idx| start + idx as f32 * step).collect()
}

fn insert_tensor(weights: &mut WeightMap, name: &str, values: Vec<f32>, shape: &[i32]) {
    weights.insert(
        name.to_string(),
        mlxcel_core::from_slice_f32(&values, shape),
    );
}

fn make_moe_weight_map() -> WeightMap {
    let mut weights = WeightMap::new();
    insert_tensor(
        &mut weights,
        "model.embed_tokens.weight",
        values(32, 0.01, 0.01),
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.q_proj.weight",
        values(16, 0.02, 0.01),
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.k_proj.weight",
        values(16, 0.20, -0.01),
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.v_proj.weight",
        values(16, 0.03, 0.005),
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.o_proj.weight",
        values(16, 0.04, 0.005),
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.q_norm.weight",
        vec![1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.k_norm.weight",
        vec![1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.gate_proj.weight",
        values(32, 0.05, 0.005),
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.up_proj.weight",
        values(32, 0.06, 0.005),
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.down_proj.weight",
        values(32, 0.07, 0.005),
        &[4, 8],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.input_layernorm.weight",
        vec![1.0, 1.0, 1.0, 1.0],
        &[4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.post_attention_layernorm.weight",
        vec![1.0, 1.0, 1.0, 1.0],
        &[4],
    );
    insert_tensor(
        &mut weights,
        "model.norm.weight",
        vec![1.0, 1.0, 1.0, 1.0],
        &[4],
    );
    insert_tensor(
        &mut weights,
        "lm_head.weight",
        values(32, 0.08, 0.005),
        &[8, 4],
    );
    weights
}

fn assert_close(a: &mlxcel_core::MlxArray, b: &mlxcel_core::MlxArray) {
    mlxcel_core::eval(a);
    mlxcel_core::eval(b);
    let close = mlxcel_core::allclose(a, b, 1e-5, 1e-5);
    assert!(mlxcel_core::item_bool(&close));
}

fn linear_block() -> serde_json::Value {
    serde_json::json!({"rope_type": "linear", "factor": 8.0})
}

fn llama3_block() -> serde_json::Value {
    serde_json::json!({
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192
    })
}

/// The pair this loader stores is handed straight to `gather_qmm`, which
/// crosses the cxx bridge as `UniquePtr<MlxArray>` rather than `Result`. A C++
/// throw there is an uncatchable `std::terminate`,
/// so losing the bound turns a rejected load into an uncatchable abort at the
/// first routed forward pass in production. This test asserts on the load
/// result rather than running a forward pass, so a regression fails cleanly
/// here instead of aborting the test binary.
#[test]
fn qwen3_moe_switch_linear_rejects_quantization_params_that_would_abort_gather_qmm() {
    let mut weights = WeightMap::new();
    insert_stacked_quantized_expert_plane(
        &mut weights,
        PREFIX,
        EXPERTS,
        OUT,
        PACKED_IN,
        NUM_GROUPS,
    );

    // Positive control first, so a guard that rejected every quantized plane
    // could not pass this test.
    SwitchLinear::from_weights(&weights, &args_with(GROUP_SIZE, BITS), PREFIX)
        .expect("honest 4-bit expert plane must load");

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        let err = match SwitchLinear::from_weights(&weights, &args_with(group_size, bits), PREFIX) {
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
    SwitchLinear::from_weights(&regular, &args_with(0, 0), PREFIX)
        .expect("a bf16 expert plane must not be gated on quantization params");
}

#[test]
fn qwen3_moe_rope_scaling_kind_reads_linear_and_type_precedence() {
    let legacy = moe_args_with_rope(Some(serde_json::json!({"type": "linear", "factor": 8.0})));
    assert_eq!(legacy.rope_scaling_kind().scale(), 0.125);
    assert!(legacy.rope_scaling_kind().freqs().is_none());

    let modern = moe_args_with_rope(Some(linear_block()));
    assert_eq!(modern.rope_scaling_kind().scale(), 0.125);
    assert!(modern.rope_scaling_kind().freqs().is_none());

    let both = moe_args_with_rope(Some(
        serde_json::json!({"type": "default", "rope_type": "linear", "factor": 8.0}),
    ));
    assert_eq!(both.rope_scaling_kind().scale(), 1.0);
    assert!(both.rope_scaling_kind().freqs().is_none());
}

#[test]
fn qwen3_moe_rope_scaling_kind_warns_to_plain_table_for_unusable_blocks() {
    for block in [
        None,
        Some(serde_json::json!({"rope_type": "default"})),
        Some(serde_json::json!({"rope_type": "linear"})),
        Some(serde_json::json!({"rope_type": "linear", "factor": 0.0})),
        Some(serde_json::json!({"rope_type": "linear", "factor": -8.0})),
        Some(serde_json::json!({"rope_type": "dynamic", "factor": 8.0})),
    ] {
        let args = moe_args_with_rope(block);
        let kind = args.rope_scaling_kind();
        assert_eq!(kind.scale(), 1.0);
        assert!(kind.freqs().is_none());
    }
}

#[test]
fn qwen3_moe_rope_scaling_reaches_model_layer_and_attention_constructors() {
    let weights = make_moe_weight_map();
    let args = moe_args_with_rope(Some(linear_block()));

    let attention = Attention::from_weights(&weights, &args, "model.layers.0.self_attn").unwrap();
    assert_eq!(attention.rope_scale, 0.125);
    assert!(attention.rope_freqs.is_none());
    assert!(attention.fused_qk_norm_launcher_usable());

    let layer = DecoderLayer::from_weights(&weights, &args, 0).unwrap();
    assert_eq!(layer.self_attn.rope_scale, 0.125);
    assert!(layer.self_attn.rope_freqs.is_none());

    let model = Qwen3MoeModel::from_weights(&weights, &args).unwrap();
    assert_eq!(model.layers[0].self_attn.rope_scale, 0.125);
    assert!(model.layers[0].self_attn.rope_freqs.is_none());

    let table_args = moe_args_with_rope(Some(llama3_block()));
    let table_model = Qwen3MoeModel::from_weights(&weights, &table_args).unwrap();
    let table_attn = &table_model.layers[0].self_attn;
    assert_eq!(table_attn.rope_scale, 1.0);
    assert!(table_attn.rope_freqs.is_some());
    assert!(!table_attn.fused_qk_norm_launcher_usable());
}

#[test]
fn qwen3_moe_graph_rope_uses_linear_scale_and_frequency_tables() {
    let weights = make_moe_weight_map();
    let q =
        mlxcel_core::from_slice_f32(&[0.1, -0.2, 0.3, -0.1, 0.4, 0.5, -0.3, 0.2], &[1, 2, 2, 2]);
    let k =
        mlxcel_core::from_slice_f32(&[0.2, 0.1, -0.4, 0.3, 0.6, -0.2, 0.7, -0.5], &[1, 2, 2, 2]);

    let linear_args = moe_args_with_rope(Some(linear_block()));
    let linear = Attention::from_weights(&weights, &linear_args, "model.layers.0.self_attn")
        .expect("linear attention must load");
    let (linear_q, linear_k) = linear.apply_rope(&q, &k, 7);
    let want_q = mlxcel_core::fast_rope(&q, linear.rope_dims, false, linear.rope_base, 0.125, 7);
    let want_k = mlxcel_core::fast_rope(&k, linear.rope_dims, false, linear.rope_base, 0.125, 7);
    assert_close(&linear_q, &want_q);
    assert_close(&linear_k, &want_k);

    let table_args = moe_args_with_rope(Some(llama3_block()));
    let table = Attention::from_weights(&weights, &table_args, "model.layers.0.self_attn")
        .expect("llama3-table attention must load");
    let freqs = table.rope_freqs.as_ref().expect("llama3 must build freqs");
    let (table_q, table_k) = table.apply_rope(&q, &k, 7);
    let want_q = mlxcel_core::fast_rope_with_freqs(&q, table.rope_dims, false, 1.0, 7, freqs);
    let want_k = mlxcel_core::fast_rope_with_freqs(&k, table.rope_dims, false, 1.0, 7, freqs);
    assert_close(&table_q, &want_q);
    assert_close(&table_k, &want_k);
}
