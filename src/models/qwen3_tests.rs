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

//! Regression tests for Qwen3 `rope_scaling` propagation (#1388).

use mlxcel_core::weights::WeightMap;

use super::{Attention, ModelArgs, Qwen3Model, TransformerBlock};

fn args_with(rope_scaling: Option<serde_json::Value>) -> ModelArgs {
    let mut config = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": 8,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "tie_word_embeddings": false
    });
    if let Some(block) = rope_scaling {
        config["rope_scaling"] = block;
    }
    serde_json::from_value(config).expect("test qwen3 config must parse")
}

fn tensor(values: &[f32], shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::from_slice_f32(values, shape)
}

fn insert_tensor(weights: &mut WeightMap, name: &str, values: &[f32], shape: &[i32]) {
    weights.insert(name.to_string(), tensor(values, shape));
}

fn make_weight_map() -> WeightMap {
    let mut weights = WeightMap::new();
    insert_tensor(
        &mut weights,
        "model.embed_tokens.weight",
        &[
            0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30,
            1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70,
            2.80, 2.90, 3.00, 3.10,
        ],
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.q_proj.weight",
        &[
            0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.k_proj.weight",
        &[
            0.7, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.2, 0.4, 0.3, 0.2, 0.1,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.v_proj.weight",
        &[
            0.05, 0.10, 0.15, 0.20, 0.10, 0.15, 0.20, 0.25, 0.15, 0.20, 0.25, 0.30, 0.20, 0.25,
            0.30, 0.35,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.o_proj.weight",
        &[
            0.20, 0.10, 0.30, 0.40, 0.10, 0.30, 0.20, 0.40, 0.40, 0.30, 0.10, 0.20, 0.30, 0.40,
            0.20, 0.10,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.q_norm.weight",
        &[1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.k_norm.weight",
        &[1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.gate_proj.weight",
        &[
            0.10, 0.20, 0.30, 0.40, 0.20, 0.30, 0.40, 0.50, 0.30, 0.40, 0.50, 0.60, 0.40, 0.50,
            0.60, 0.70, 0.50, 0.60, 0.70, 0.80, 0.60, 0.70, 0.80, 0.90, 0.70, 0.80, 0.90, 1.00,
            0.80, 0.90, 1.00, 1.10,
        ],
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.up_proj.weight",
        &[
            0.15, 0.25, 0.35, 0.45, 0.25, 0.35, 0.45, 0.55, 0.35, 0.45, 0.55, 0.65, 0.45, 0.55,
            0.65, 0.75, 0.55, 0.65, 0.75, 0.85, 0.65, 0.75, 0.85, 0.95, 0.75, 0.85, 0.95, 1.05,
            0.85, 0.95, 1.05, 1.15,
        ],
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.down_proj.weight",
        &[
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.50, 0.55,
        ],
        &[4, 8],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.input_layernorm.weight",
        &[1.0, 1.0, 1.0, 1.0],
        &[4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.post_attention_layernorm.weight",
        &[1.0, 1.0, 1.0, 1.0],
        &[4],
    );
    insert_tensor(
        &mut weights,
        "model.norm.weight",
        &[1.0, 1.0, 1.0, 1.0],
        &[4],
    );
    insert_tensor(
        &mut weights,
        "lm_head.weight",
        &[
            0.10, 0.20, 0.30, 0.40, 0.20, 0.30, 0.40, 0.50, 0.30, 0.40, 0.50, 0.60, 0.40, 0.50,
            0.60, 0.70, 0.50, 0.60, 0.70, 0.80, 0.60, 0.70, 0.80, 0.90, 0.70, 0.80, 0.90, 1.00,
            0.80, 0.90, 1.00, 1.10,
        ],
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

#[test]
fn qwen3_rope_scaling_kind_reads_linear_spellings_and_type_precedence() {
    let legacy = args_with(Some(serde_json::json!({"type": "linear", "factor": 8.0})));
    assert_eq!(legacy.rope_scaling_kind().scale(), 0.125);
    assert!(legacy.rope_scaling_kind().freqs().is_none());

    let modern = args_with(Some(linear_block()));
    assert_eq!(modern.rope_scaling_kind().scale(), 0.125);
    assert!(modern.rope_scaling_kind().freqs().is_none());

    let both = args_with(Some(
        serde_json::json!({"type": "default", "rope_type": "linear", "factor": 8.0}),
    ));
    assert_eq!(both.rope_scaling_kind().scale(), 1.0);
    assert!(both.rope_scaling_kind().freqs().is_none());
}

#[test]
fn qwen3_rope_warnings_use_the_checkpoint_name_when_available() {
    let mut args = args_with(Some(serde_json::json!({"rope_type": "yarn"})));
    assert_eq!(args.model_label(), "qwen3");
    args.set_checkpoint_label(std::path::Path::new("models/vendor-qwen3-scaled"));
    assert_eq!(args.model_label(), "vendor-qwen3-scaled");
}

#[test]
fn qwen3_rope_scaling_kind_warns_to_plain_table_for_unusable_blocks() {
    for block in [
        None,
        Some(serde_json::json!({"rope_type": "default"})),
        Some(serde_json::json!({"rope_type": "linear"})),
        Some(serde_json::json!({"rope_type": "linear", "factor": 0.0})),
        Some(serde_json::json!({"rope_type": "linear", "factor": -8.0})),
        Some(serde_json::json!({"rope_type": "yarn"})),
    ] {
        let args = args_with(block);
        let kind = args.rope_scaling_kind();
        assert_eq!(kind.scale(), 1.0);
        assert!(kind.freqs().is_none());
    }
}

#[test]
fn qwen3_rope_scaling_kind_builds_the_yarn_table_since_1472() {
    let args = args_with(Some(serde_json::json!({
        "rope_type": "yarn",
        "factor": 8.0,
        "original_max_position_embeddings": 32768
    })));
    let kind = args.rope_scaling_kind();
    assert!(kind.freqs().is_some(), "a usable yarn block builds a table");
    assert_eq!(kind.scale(), 1.0);
    assert!(
        kind.attn_scale() > 1.0,
        "yarn carries its temperature mscale"
    );
}

#[test]
fn qwen3_rope_scaling_reaches_model_block_and_attention_constructors() {
    let weights = make_weight_map();
    let args = args_with(Some(linear_block()));

    let attention = Attention::from_weights(&weights, &args, "model.layers.0.self_attn").unwrap();
    assert_eq!(attention.rope_scale, 0.125);
    assert!(attention.rope_freqs.is_none());
    assert!(attention.fused_qk_norm_launcher_usable());

    // Pipeline stages call this direct block constructor instead of
    // `Qwen3Model::from_weights`, so it must resolve the args on its own.
    let block = TransformerBlock::from_weights(&weights, &args, 0).unwrap();
    assert_eq!(block.self_attn.rope_scale, 0.125);
    assert!(block.self_attn.rope_freqs.is_none());

    let model = Qwen3Model::from_weights(&weights, &args).unwrap();
    assert_eq!(model.layers[0].self_attn.rope_scale, 0.125);
    assert!(model.layers[0].self_attn.rope_freqs.is_none());

    let table_args = args_with(Some(llama3_block()));
    let table_model = Qwen3Model::from_weights(&weights, &table_args).unwrap();
    let table_attn = &table_model.layers[0].self_attn;
    assert_eq!(table_attn.rope_scale, 1.0);
    assert!(table_attn.rope_freqs.is_some());
    assert!(!table_attn.fused_qk_norm_launcher_usable());
    assert_eq!(
        mlxcel_core::array_shape(table_attn.rope_freqs.as_ref().unwrap()),
        vec![1]
    );
}

#[test]
fn qwen3_graph_rope_uses_linear_scale_and_frequency_tables() {
    let weights = make_weight_map();
    let q =
        mlxcel_core::from_slice_f32(&[0.1, -0.2, 0.3, -0.1, 0.4, 0.5, -0.3, 0.2], &[1, 2, 2, 2]);
    let k =
        mlxcel_core::from_slice_f32(&[0.2, 0.1, -0.4, 0.3, 0.6, -0.2, 0.7, -0.5], &[1, 2, 2, 2]);

    let linear_args = args_with(Some(linear_block()));
    let linear = Attention::from_weights(&weights, &linear_args, "model.layers.0.self_attn")
        .expect("linear attention must load");
    let (linear_q, linear_k) = linear.apply_rope(&q, &k, 7);
    let want_q = mlxcel_core::fast_rope(&q, linear.rope_dims, false, linear.rope_base, 0.125, 7);
    let want_k = mlxcel_core::fast_rope(&k, linear.rope_dims, false, linear.rope_base, 0.125, 7);
    assert_close(&linear_q, &want_q);
    assert_close(&linear_k, &want_k);

    let table_args = args_with(Some(llama3_block()));
    let table = Attention::from_weights(&weights, &table_args, "model.layers.0.self_attn")
        .expect("llama3-table attention must load");
    let freqs = table.rope_freqs.as_ref().expect("llama3 must build freqs");
    let (table_q, table_k) = table.apply_rope(&q, &k, 7);
    let want_q = mlxcel_core::fast_rope_with_freqs(&q, table.rope_dims, false, 1.0, 7, freqs);
    let want_k = mlxcel_core::fast_rope_with_freqs(&k, table.rope_dims, false, 1.0, 7, freqs);
    assert_close(&table_q, &want_q);
    assert_close(&table_k, &want_k);
}

#[test]
fn qwen3_batched_rope_uses_linear_scale_and_frequency_tables() {
    let weights = make_weight_map();
    let q =
        mlxcel_core::from_slice_f32(&[0.1, -0.2, 0.3, -0.1, 0.4, 0.5, -0.3, 0.2], &[2, 2, 1, 2]);
    let k =
        mlxcel_core::from_slice_f32(&[0.2, 0.1, -0.4, 0.3, 0.6, -0.2, 0.7, -0.5], &[2, 2, 1, 2]);
    let offsets = [3, 9];

    let linear_args = args_with(Some(linear_block()));
    let linear = Attention::from_weights(&weights, &linear_args, "model.layers.0.self_attn")
        .expect("linear attention must load");
    let (linear_q, linear_k) = linear.apply_rope_batched(&q, &k, &offsets);
    let want_q = mlxcel_core::fast_rope_batched(
        &q,
        linear.rope_dims,
        false,
        linear.rope_base,
        0.125,
        &offsets,
    );
    let want_k = mlxcel_core::fast_rope_batched(
        &k,
        linear.rope_dims,
        false,
        linear.rope_base,
        0.125,
        &offsets,
    );
    assert_close(&linear_q, &want_q);
    assert_close(&linear_k, &want_k);

    let table_args = args_with(Some(llama3_block()));
    let table = Attention::from_weights(&weights, &table_args, "model.layers.0.self_attn")
        .expect("llama3-table attention must load");
    let freqs = table.rope_freqs.as_ref().expect("llama3 must build freqs");
    let (table_q, table_k) = table.apply_rope_batched(&q, &k, &offsets);
    let want_q =
        mlxcel_core::fast_rope_batched_with_freqs(&q, table.rope_dims, false, 1.0, &offsets, freqs);
    let want_k =
        mlxcel_core::fast_rope_batched_with_freqs(&k, table.rope_dims, false, 1.0, &offsets, freqs);
    assert_close(&table_q, &want_q);
    assert_close(&table_k, &want_k);
}
