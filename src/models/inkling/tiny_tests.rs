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

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::json;

use super::{InklingConfig, InklingModel};

fn matrix(rows: i32, columns: i32, seed: usize) -> UniquePtr<MlxArray> {
    let values: Vec<f32> = (0..rows as usize * columns as usize)
        .map(|index| (((index + seed) % 17) as f32 - 8.0) / 50.0)
        .collect();
    mlxcel_core::from_slice_f32(&values, &[rows, columns])
}

fn conv(channels: i32, kernel: i32, seed: usize) -> UniquePtr<MlxArray> {
    let values: Vec<f32> = (0..channels as usize * kernel as usize)
        .map(|index| (((index + seed) % 7) as f32 - 3.0) / 40.0)
        .collect();
    mlxcel_core::from_slice_f32(&values, &[channels, kernel, 1])
}

fn tiny_model() -> InklingModel {
    const HIDDEN: i32 = 4;
    const HEADS: i32 = 2;
    const KV_HEADS: i32 = 1;
    const HEAD_DIM: i32 = 2;
    const D_REL: i32 = 2;
    const DENSE: i32 = 6;
    const VOCAB: i32 = 8;
    const KERNEL: i32 = 2;
    let config: InklingConfig = serde_json::from_value(json!({
        "vocab_size": VOCAB,
        "text_config": {
            "hidden_size": HIDDEN,
            "num_hidden_layers": 2,
            "vocab_size": VOCAB,
            "unpadded_vocab_size": VOCAB,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "swa_num_attention_heads": HEADS,
            "swa_num_key_value_heads": KV_HEADS,
            "swa_head_dim": HEAD_DIM,
            "sliding_window_size": 3,
            "layer_types": ["hybrid_sliding", "hybrid"],
            "d_rel": D_REL,
            "rel_extent": 4,
            "log_scaling_n_floor": 2,
            "log_scaling_alpha": 0.1,
            "sconv_kernel_size": KERNEL,
            "dense_mlp_idx": 2,
            "mlp_layer_types": ["dense", "dense"],
            "intermediate_size": 2,
            "dense_intermediate_size": DENSE,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "n_shared_experts": 1
        }
    }))
    .unwrap();
    let mut weights = WeightMap::new();
    weights.insert("model.embed_tokens.weight".into(), matrix(VOCAB, HIDDEN, 1));
    weights.insert(
        "model.embed_norm.weight".into(),
        mlxcel_core::ones(&[HIDDEN], dtype::FLOAT32),
    );
    weights.insert(
        "model.norm.weight".into(),
        mlxcel_core::ones(&[HIDDEN], dtype::FLOAT32),
    );
    weights.insert("lm_head.weight".into(), matrix(VOCAB, HIDDEN, 5));
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            weights.insert(
                format!("{prefix}.{name}"),
                mlxcel_core::ones(&[HIDDEN], dtype::FLOAT32),
            );
        }
        weights.insert(
            format!("{prefix}.attn_sconv.conv.weight"),
            conv(HIDDEN, KERNEL, layer + 2),
        );
        weights.insert(
            format!("{prefix}.mlp_sconv.conv.weight"),
            conv(HIDDEN, KERNEL, layer + 4),
        );
        let attn = format!("{prefix}.self_attn");
        weights.insert(
            format!("{attn}.q_proj.weight"),
            matrix(4, HIDDEN, layer + 1),
        );
        weights.insert(
            format!("{attn}.k_proj.weight"),
            matrix(2, HIDDEN, layer + 2),
        );
        weights.insert(
            format!("{attn}.v_proj.weight"),
            matrix(2, HIDDEN, layer + 3),
        );
        weights.insert(
            format!("{attn}.r_proj.weight"),
            matrix(4, HIDDEN, layer + 4),
        );
        weights.insert(
            format!("{attn}.o_proj.weight"),
            matrix(HIDDEN, 4, layer + 5),
        );
        weights.insert(
            format!("{attn}.q_norm.weight"),
            mlxcel_core::ones(&[HEAD_DIM], dtype::FLOAT32),
        );
        weights.insert(
            format!("{attn}.k_norm.weight"),
            mlxcel_core::ones(&[HEAD_DIM], dtype::FLOAT32),
        );
        weights.insert(
            format!("{attn}.k_sconv.conv.weight"),
            conv(2, KERNEL, layer + 6),
        );
        weights.insert(
            format!("{attn}.v_sconv.conv.weight"),
            conv(2, KERNEL, layer + 7),
        );
        let extent = if layer == 0 { 3 } else { 4 };
        weights.insert(format!("{attn}.rel_proj"), matrix(D_REL, extent, layer + 8));
        let mlp = format!("{prefix}.mlp");
        weights.insert(
            format!("{mlp}.gate_proj.weight"),
            matrix(DENSE, HIDDEN, layer + 9),
        );
        weights.insert(
            format!("{mlp}.up_proj.weight"),
            matrix(DENSE, HIDDEN, layer + 10),
        );
        weights.insert(
            format!("{mlp}.down_proj.weight"),
            matrix(HIDDEN, DENSE, layer + 11),
        );
        weights.insert(
            format!("{mlp}.global_scale"),
            mlxcel_core::ones(&[1], dtype::FLOAT32),
        );
    }
    InklingModel::from_weights(config, weights).unwrap()
}

fn forward(model: &InklingModel, tokens: &[i32]) -> UniquePtr<MlxArray> {
    let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    let mut caches = model.make_caches();
    model.forward(&input, &mut caches, None)
}

#[test]
fn tiny_hybrid_prefill_is_causal_and_matches_incremental_decode() {
    let tokens = [1, 2, 3, 4, 5];
    let full = forward(&tiny_model(), &tokens);

    let changed = forward(&tiny_model(), &[1, 2, 6, 7, 0]);
    let prefix_full = mlxcel_core::utils::slice_axis(&full, 1, 0, 2);
    let prefix_changed = mlxcel_core::utils::slice_axis(&changed, 1, 0, 2);
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &prefix_full,
        &prefix_changed,
        1e-5,
        1e-5,
    )));

    let incremental_model = tiny_model();
    let mut caches = incremental_model.make_caches();
    let mut incremental: Option<UniquePtr<MlxArray>> = None;
    for token in tokens {
        let input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
        let output = incremental_model.forward(&input, &mut caches, None);
        incremental = Some(match incremental {
            Some(previous) => mlxcel_core::concatenate(&previous, &output, 1),
            None => output,
        });
    }
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &full,
        incremental.as_deref().unwrap(),
        2e-4,
        2e-4,
    )));
}
