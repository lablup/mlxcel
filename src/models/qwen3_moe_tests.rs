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

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;

use super::{Attention, DecoderLayer, MLPType, ModelArgs, Qwen3MoeModel, SwitchLinear};
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
fn qwen3_moe_rope_warnings_use_the_checkpoint_name_when_available() {
    let mut args = moe_args_with_rope(Some(serde_json::json!({"rope_type": "dynamic"})));
    assert_eq!(args.model_label(), "qwen3_moe");
    args.set_checkpoint_label(std::path::Path::new("models/vendor-qwen3-moe-scaled"));
    assert_eq!(args.model_label(), "vendor-qwen3-moe-scaled");
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

/// Config for a one-layer routed model: hidden 4, two 2-wide heads, three
/// experts of width 8 with top-2 routing, so every decode step exercises the
/// router, `gather_mm` over `tokens * top_k` slots and the weighted combine.
fn moe_expert_args() -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "qwen3_moe",
        "vocab_size": 8,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_experts": 3,
        "num_experts_per_tok": 2,
        "norm_topk_prob": true,
        "decoder_sparse_step": 1,
        "moe_intermediate_size": 8,
        "rms_norm_eps": 1e-5,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "tie_word_embeddings": false
    }))
    .expect("test qwen3_moe expert config must parse")
}

/// [`make_moe_weight_map`] with the dense MLP replaced by a router plus three
/// stacked (non-quantized) expert planes, so `DecoderLayer` builds an
/// `MLPType::MoE` layer.
fn make_moe_expert_weight_map() -> WeightMap {
    let mut weights = make_moe_weight_map();
    for leaf in ["gate_proj", "up_proj", "down_proj"] {
        weights.remove(&format!("model.layers.0.mlp.{leaf}.weight"));
    }
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.gate.weight",
        values(12, 0.30, -0.04),
        &[3, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.switch_mlp.gate_proj.weight",
        values(96, 0.05, 0.003),
        &[3, 8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.switch_mlp.up_proj.weight",
        values(96, -0.06, 0.004),
        &[3, 8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.switch_mlp.down_proj.weight",
        values(96, 0.07, -0.002),
        &[3, 4, 8],
    );
    weights
}

fn assert_close_tol(a: &mlxcel_core::MlxArray, b: &mlxcel_core::MlxArray, tol: f64) {
    mlxcel_core::eval(a);
    mlxcel_core::eval(b);
    let close = mlxcel_core::allclose(a, b, tol, tol);
    assert!(
        mlxcel_core::item_bool(&close),
        "arrays differ beyond {tol}: {:?} vs {:?}",
        mlxcel_core::utils::array_to_vec_f32(a),
        mlxcel_core::utils::array_to_vec_f32(b)
    );
}

/// Prefill `prompt` into a fresh cache set and return the caches.
fn prefilled_caches(model: &Qwen3MoeModel, prompt: &[i32]) -> Vec<KVCache> {
    let mut caches = model.make_caches();
    let ids = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let logits = model.forward(&ids, &mut caches, None);
    mlxcel_core::eval(&logits);
    caches
}

/// The batched decode step (#1616) must reproduce, row for row, what each
/// sequence produces when decoded alone: different prompt lengths give the two
/// rows different RoPE offsets, and top-2 routing over three experts gives the
/// MoE block a `[2, 2]` slot grid, so the per-row RoPE, the per-sequence cache
/// update and the multi-token `gather_mm` path are all on the compared graph.
#[test]
fn qwen3_moe_batched_decode_matches_each_sequence_decoded_alone() {
    let weights = make_moe_expert_weight_map();
    let args = moe_expert_args();
    let model = Qwen3MoeModel::from_weights(&weights, &args).expect("routed model must load");
    assert!(matches!(model.layers[0].mlp, MLPType::MoE(_)));

    let prompts: [&[i32]; 2] = [&[1, 2, 3], &[4, 5]];
    let next = [6, 7];

    let mut expected = Vec::new();
    for (prompt, tok) in prompts.iter().zip(next) {
        let mut caches = prefilled_caches(&model, prompt);
        let step = mlxcel_core::from_slice_i32(&[tok], &[1, 1]);
        let logits = model.forward(&step, &mut caches, None);
        mlxcel_core::eval(&logits);
        assert_eq!(caches[0].offset, prompt.len() as i32 + 1);
        expected.push(logits);
    }

    let mut caches_a = prefilled_caches(&model, prompts[0]);
    let mut caches_b = prefilled_caches(&model, prompts[1]);
    let step = mlxcel_core::from_slice_i32(&next, &[2, 1]);
    let logits = {
        let mut batch: Vec<&mut [KVCache]> = vec![caches_a.as_mut_slice(), caches_b.as_mut_slice()];
        LanguageModel::forward_batched(&model, &step, &mut batch, None)
    };
    mlxcel_core::eval(&logits);
    assert_eq!(mlxcel_core::array_shape(&logits), vec![2, 1, 8]);

    for (i, want) in expected.iter().enumerate() {
        let row = mlxcel_core::slice(&logits, &[i as i32, 0, 0], &[i as i32 + 1, 1, 8]);
        assert_close_tol(&row, want, 1e-5);
    }
    // Each row's own cache advanced by exactly the one batched token.
    assert_eq!(caches_a[0].offset, 4);
    assert_eq!(caches_b[0].offset, 3);
}

/// A one-row batch keeps the exact single-sequence graph rather than the
/// batched forward, so B=1 through the scheduler's batched entry point stays
/// byte-for-byte the same computation as `forward`.
#[test]
fn qwen3_moe_forward_batched_single_row_is_the_single_sequence_forward() {
    let weights = make_moe_expert_weight_map();
    let args = moe_expert_args();
    let model = Qwen3MoeModel::from_weights(&weights, &args).expect("routed model must load");

    let prompt: &[i32] = &[2, 4, 6];
    let mut alone = prefilled_caches(&model, prompt);
    let step = mlxcel_core::from_slice_i32(&[1], &[1, 1]);
    let want = model.forward(&step, &mut alone, None);

    let mut batched = prefilled_caches(&model, prompt);
    let got = {
        let mut batch: Vec<&mut [KVCache]> = vec![batched.as_mut_slice()];
        LanguageModel::forward_batched(&model, &step, &mut batch, None)
    };
    assert_eq!(mlxcel_core::array_shape(&got), vec![1, 1, 8]);
    assert_close_tol(&got, &want, 0.0);
    assert_eq!(batched[0].offset, alone[0].offset);
}

/// An empty batch is the scheduler's guard step and must not dispatch a
/// forward: it returns the trait default's empty logits.
#[test]
fn qwen3_moe_forward_batched_empty_batch_returns_empty_logits() {
    let weights = make_moe_expert_weight_map();
    let args = moe_expert_args();
    let model = Qwen3MoeModel::from_weights(&weights, &args).expect("routed model must load");
    let step = mlxcel_core::from_slice_i32(&[], &[0, 1]);
    let mut batch: Vec<&mut [KVCache]> = Vec::new();
    let got = LanguageModel::forward_batched(&model, &step, &mut batch, None);
    assert_eq!(mlxcel_core::array_shape(&got), vec![0, 1, 1]);
}
