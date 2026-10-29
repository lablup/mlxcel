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

use mlxcel_core::weights::WeightMap;

use super::*;

fn map(entries: &[(&str, &[i32], i32)]) -> WeightMap {
    entries
        .iter()
        .map(|(k, shape, dt)| {
            (
                (*k).to_string(),
                mlxcel_core::astype(&mlxcel_core::zeros(shape, dtype::FLOAT32), *dt),
            )
        })
        .collect()
}

#[test]
fn text_encoder_keys_map_onto_the_llada2_layout() {
    let raw = map(&[
        (
            "model.language_model.word_embeddings.weight",
            &[4, 2],
            dtype::BFLOAT16,
        ),
        (
            "model.language_model.layers.1.mlp.experts.gate_proj",
            &[2, 3, 2],
            dtype::BFLOAT16,
        ),
        (
            "model.language_model.layers.1.mlp.gate.expert_bias",
            &[2],
            dtype::FLOAT32,
        ),
        (LM_HEAD_KEY, &[4, 2], dtype::BFLOAT16),
    ]);
    let out = sanitize_text_encoder_weights(raw);
    let mut keys: Vec<&String> = out.keys().collect();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "model.layers.1.mlp.gate.expert_bias",
            "model.layers.1.mlp.switch_mlp.gate_proj.weight",
            "model.word_embeddings.weight",
        ]
    );
}

fn sigvq_config() -> SigVqConfig {
    serde_json::from_value(serde_json::json!({
        "hidden_size": 4, "intermediate_size": 8, "num_attention_heads": 2,
        "num_hidden_layers": 1, "patch_size": 2, "image_size": 4,
        "codebook_size": 3, "codebook_embed_dim": 5, "semantic_embed_dim": 6
    }))
    .unwrap()
}

#[test]
fn sigvq_sanitize_renames_filters_and_permutes_once() {
    let raw = || {
        map(&[
            (
                "prior_projector.net.0.proj.weight",
                &[6, 6],
                dtype::BFLOAT16,
            ),
            ("prior_projector.net.2.weight", &[6, 6], dtype::BFLOAT16),
            (
                "visual.patch_embed.proj.weight",
                &[4, 3, 2, 2],
                dtype::BFLOAT16,
            ),
            ("vqmodel.quant_conv.weight", &[5, 4, 1, 1], dtype::BFLOAT16),
        ])
    };
    let config = sigvq_config();
    let prior = sanitize_sigvq_weights(raw(), &config, false, ConvLayout::PytorchNchw).unwrap();
    let mut keys: Vec<&String> = prior.keys().collect();
    keys.sort();
    assert_eq!(
        keys,
        vec!["prior_projector.0.weight", "prior_projector.1.weight"]
    );

    let full = sanitize_sigvq_weights(raw(), &config, true, ConvLayout::PytorchNchw).unwrap();
    assert_eq!(
        mlxcel_core::array_shape(&full["visual.patch_embed.proj.weight"]),
        vec![4, 2, 2, 3]
    );
    assert_eq!(
        mlxcel_core::array_shape(&full["vqmodel.quant_conv.weight"]),
        vec![5, 1, 1, 4]
    );
    // A kernel already in MLX layout but labelled PyTorch is refused, not
    // permuted a second time.
    let mut mlx = raw();
    mlx.insert(
        "visual.patch_embed.proj.weight".into(),
        mlxcel_core::zeros(&[4, 2, 2, 3], dtype::BFLOAT16),
    );
    assert!(sanitize_sigvq_weights(mlx, &config, true, ConvLayout::PytorchNchw).is_err());
}

#[test]
fn transformer_sanitize_handles_patch_one_keys_and_fused_projections() {
    let raw = map(&[
        ("all_x_embedder.1-1.weight", &[4, 2], dtype::BFLOAT16),
        (
            "all_final_layer.1-1.adaLN_modulation.1.weight",
            &[4, 2],
            dtype::BFLOAT16,
        ),
        ("t_embedder.mlp.0.weight", &[2, 2], dtype::BFLOAT16),
        ("t_embedder.mlp.2.bias", &[2], dtype::BFLOAT16),
        ("layers.0.attention.to_qkv.weight", &[6, 2], dtype::BFLOAT16),
        ("layers.0.feed_forward.w13.weight", &[8, 2], dtype::BFLOAT16),
    ]);
    let out = sanitize_transformer_weights(raw);
    let mut keys: Vec<&String> = out.keys().collect();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "final_layer.adaLN_modulation.0.weight",
            "layers.0.attention.to_k.weight",
            "layers.0.attention.to_q.weight",
            "layers.0.attention.to_v.weight",
            "layers.0.feed_forward.w1.weight",
            "layers.0.feed_forward.w3.weight",
            "t_embedder.linear1.weight",
            "t_embedder.linear2.bias",
            "x_embedder.weight",
        ]
    );
    assert_eq!(
        mlxcel_core::array_shape(&out["layers.0.attention.to_q.weight"]),
        vec![2, 2]
    );
    assert_eq!(
        mlxcel_core::array_shape(&out["layers.0.feed_forward.w3.weight"]),
        vec![4, 2]
    );
}

#[test]
fn fp8_and_mixed_dtypes_are_refused() {
    let fp8 = map(&[
        ("layers.0.attention.to_q.weight", &[2, 2], dtype::UINT8),
        (
            "layers.0.attention.to_q.weight_scale_inv",
            &[1, 1],
            dtype::FLOAT32,
        ),
    ]);
    let err = check_component_dtypes("transformer", &fp8).unwrap_err();
    assert!(format!("{err}").contains("FP8"), "{err}");
    let mixed = map(&[("a", &[2], dtype::BFLOAT16), ("b", &[2], dtype::FLOAT16)]);
    assert!(check_component_dtypes("queryformer", &mixed).is_err());
    let ok = map(&[("a", &[2], dtype::BFLOAT16), ("ids", &[2], dtype::INT32)]);
    assert!(check_component_dtypes("queryformer", &ok).is_ok());
}
