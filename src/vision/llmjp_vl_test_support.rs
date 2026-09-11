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

//! Test-only fixtures for the LLM-jp-VL runtime gates: a tiny SigLIP tower, a
//! tiny `mlp1` connector, tiny Llama and Qwen3 decoders, and the deterministic
//! weights that back them. Scaled so `(image_size / patch_size)^2 *
//! downsample_ratio^2` is exactly 1, which makes one tile contribute one
//! feature token and keeps the merge assertions readable.
//!
//! Used by: `crate::vision::llmjp_vl` tests.

use mlxcel_core::weights::WeightMap;

use crate::models::embedding_test_support::Rng;
use crate::vision::config::{VisionConfig, VisionHiddenActivation};
use crate::vision::encoders::siglip::SigLipVisionModel;
use crate::vision::internvl::InternVLConnector;
use crate::vision::llmjp_vl::LlmJpVlModel;
use crate::vision::llmjp_vl_text::LlmJpTextModel;
use crate::vision::processors::internvl::InternVLProcessor;

pub(crate) const TEXT_HIDDEN: usize = 8;
pub(crate) const TEXT_LAYERS: usize = 2;
pub(crate) const TEXT_HEADS: usize = 2;
pub(crate) const TEXT_KV_HEADS: usize = 1;
pub(crate) const TEXT_HEAD_DIM: usize = 4;
pub(crate) const TEXT_INTERMEDIATE: usize = 16;
pub(crate) const VOCAB: usize = 24;

pub(crate) const VIT_HIDDEN: usize = 8;
pub(crate) const VIT_LAYERS: usize = 1;
pub(crate) const VIT_HEADS: usize = 2;
pub(crate) const VIT_INTERMEDIATE: usize = 16;
pub(crate) const VIT_IMAGE_SIZE: usize = 4;
pub(crate) const VIT_PATCH_SIZE: usize = 2;

pub(crate) const IMG_PAD: i32 = 14;
pub(crate) const IMG_START: i32 = 15;
pub(crate) const IMG_END: i32 = 16;

pub(crate) fn tiny_vision_config() -> VisionConfig {
    VisionConfig {
        model_type: "siglip_vision_model".to_string(),
        num_hidden_layers: VIT_LAYERS,
        hidden_size: VIT_HIDDEN,
        intermediate_size: VIT_INTERMEDIATE,
        num_attention_heads: VIT_HEADS,
        patch_size: VIT_PATCH_SIZE,
        image_size: VIT_IMAGE_SIZE,
        num_channels: 3,
        layer_norm_eps: 1e-6,
        hidden_act: VisionHiddenActivation::GeluPytorchTanh,
    }
}

/// SigLIP tower weights under the checkpoint's real prefix, in the HF conv
/// layout `[out, in, kH, kW]` the released bf16 originals ship.
pub(crate) fn tiny_vision_weights(rng: &mut Rng, prefix: &str) -> WeightMap {
    let mut w = WeightMap::new();
    let h = VIT_HIDDEN as i32;
    let patches = ((VIT_IMAGE_SIZE / VIT_PATCH_SIZE) * (VIT_IMAGE_SIZE / VIT_PATCH_SIZE)) as i32;
    rng.insert(
        &mut w,
        &format!("{prefix}.embeddings.patch_embedding.weight"),
        &[h, 3, VIT_PATCH_SIZE as i32, VIT_PATCH_SIZE as i32],
        0.2,
    );
    rng.insert(
        &mut w,
        &format!("{prefix}.embeddings.patch_embedding.bias"),
        &[h],
        0.1,
    );
    rng.insert(
        &mut w,
        &format!("{prefix}.embeddings.position_embedding.weight"),
        &[patches, h],
        0.1,
    );
    for i in 0..VIT_LAYERS {
        let p = format!("{prefix}.encoder.layers.{i}");
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            rng.insert(
                &mut w,
                &format!("{p}.self_attn.{proj}.weight"),
                &[h, h],
                0.2,
            );
            rng.insert(&mut w, &format!("{p}.self_attn.{proj}.bias"), &[h], 0.1);
        }
        rng.insert(
            &mut w,
            &format!("{p}.mlp.fc1.weight"),
            &[VIT_INTERMEDIATE as i32, h],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.mlp.fc1.bias"),
            &[VIT_INTERMEDIATE as i32],
            0.1,
        );
        rng.insert(
            &mut w,
            &format!("{p}.mlp.fc2.weight"),
            &[h, VIT_INTERMEDIATE as i32],
            0.2,
        );
        rng.insert(&mut w, &format!("{p}.mlp.fc2.bias"), &[h], 0.1);
        for ln in ["layer_norm1", "layer_norm2"] {
            rng.insert(&mut w, &format!("{p}.{ln}.weight"), &[h], 0.1);
            rng.insert(&mut w, &format!("{p}.{ln}.bias"), &[h], 0.1);
        }
    }
    rng.insert(
        &mut w,
        &format!("{prefix}.post_layernorm.weight"),
        &[h],
        0.1,
    );
    rng.insert(&mut w, &format!("{prefix}.post_layernorm.bias"), &[h], 0.1);
    // The attention-pooling head the loader drops. Present here so the tower
    // builder is exercised against a weight map that still carries it.
    rng.insert(&mut w, &format!("{prefix}.head.probe"), &[1, 1, h], 0.1);
    w
}

/// `mlp1` weights: LayerNorm over `VIT_HIDDEN * 4`, then two Linears into the
/// text width.
/// The same tower weights with `patch_embedding.weight` pre-transposed into the
/// MLX `[O, kH, kW, I]` layout an already-converted checkpoint would ship.
pub(crate) fn tiny_vision_weights_mlx_layout(rng: &mut Rng, prefix: &str) -> WeightMap {
    let mut w = tiny_vision_weights(rng, prefix);
    let key = format!("{prefix}.embeddings.patch_embedding.weight");
    let hf = w.remove(&key).expect("patch embedding present");
    w.insert(key, mlxcel_core::transpose_axes(&hf, &[0, 2, 3, 1]));
    w
}

/// Build only the tower, from a caller-supplied weight map.
pub(crate) fn build_vision_model(weights: &WeightMap) -> SigLipVisionModel {
    SigLipVisionModel::from_weights_with_quant_and_gelu(
        weights,
        &tiny_vision_config(),
        "vision_backbone.vision_model",
        64,
        4,
        false,
    )
    .expect("tiny SigLIP tower builds")
}

pub(crate) fn tiny_connector_weights(rng: &mut Rng) -> WeightMap {
    let mut w = WeightMap::new();
    let shuffled = (VIT_HIDDEN * 4) as i32;
    let hidden = TEXT_HIDDEN as i32;
    rng.insert(&mut w, "mlp1.0.weight", &[shuffled], 0.3);
    rng.insert(&mut w, "mlp1.0.bias", &[shuffled], 0.3);
    rng.insert(&mut w, "mlp1.1.weight", &[hidden, shuffled], 0.2);
    rng.insert(&mut w, "mlp1.1.bias", &[hidden], 0.1);
    rng.insert(&mut w, "mlp1.3.weight", &[hidden, hidden], 0.2);
    rng.insert(&mut w, "mlp1.3.bias", &[hidden], 0.1);
    w
}

pub(crate) fn tiny_llama_args() -> crate::models::llama3::ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "llama",
        "hidden_size": TEXT_HIDDEN,
        "num_hidden_layers": TEXT_LAYERS,
        "intermediate_size": TEXT_INTERMEDIATE,
        "num_attention_heads": TEXT_HEADS,
        "num_key_value_heads": TEXT_KV_HEADS,
        "head_dim": TEXT_HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB,
        "rope_theta": 500000.0,
        "attention_bias": false,
        "tie_word_embeddings": false,
    }))
    .expect("tiny llama args parse")
}

pub(crate) fn tiny_qwen3_args() -> crate::models::qwen3::ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": TEXT_HIDDEN,
        "num_hidden_layers": TEXT_LAYERS,
        "intermediate_size": TEXT_INTERMEDIATE,
        "num_attention_heads": TEXT_HEADS,
        "num_key_value_heads": TEXT_KV_HEADS,
        "head_dim": TEXT_HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": true,
    }))
    .expect("tiny qwen3 args parse")
}

/// Decoder weights in the sanitized (`model.*`) layout both backbones expect
/// after `strip_language_model_prefix`. `qk_norm` adds the Qwen3-only per-head
/// RMSNorm pair.
pub(crate) fn tiny_text_weights(rng: &mut Rng, qk_norm: bool, tied: bool) -> WeightMap {
    let mut w = WeightMap::new();
    let h = TEXT_HIDDEN as i32;
    let hd = TEXT_HEAD_DIM as i32;
    let q_out = TEXT_HEADS as i32 * hd;
    let kv_out = TEXT_KV_HEADS as i32 * hd;
    let inter = TEXT_INTERMEDIATE as i32;

    rng.insert(&mut w, "model.embed_tokens.weight", &[VOCAB as i32, h], 0.5);
    for i in 0..TEXT_LAYERS {
        let p = format!("model.layers.{i}");
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.q_proj.weight"),
            &[q_out, h],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.k_proj.weight"),
            &[kv_out, h],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.v_proj.weight"),
            &[kv_out, h],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.o_proj.weight"),
            &[h, q_out],
            0.2,
        );
        if qk_norm {
            rng.insert(&mut w, &format!("{p}.self_attn.q_norm.weight"), &[hd], 0.1);
            rng.insert(&mut w, &format!("{p}.self_attn.k_norm.weight"), &[hd], 0.1);
        }
        rng.insert(
            &mut w,
            &format!("{p}.mlp.gate_proj.weight"),
            &[inter, h],
            0.2,
        );
        rng.insert(&mut w, &format!("{p}.mlp.up_proj.weight"), &[inter, h], 0.2);
        rng.insert(
            &mut w,
            &format!("{p}.mlp.down_proj.weight"),
            &[h, inter],
            0.2,
        );
        rng.insert(&mut w, &format!("{p}.input_layernorm.weight"), &[h], 0.1);
        rng.insert(
            &mut w,
            &format!("{p}.post_attention_layernorm.weight"),
            &[h],
            0.1,
        );
    }
    rng.insert(&mut w, "model.norm.weight", &[h], 0.1);
    if !tied {
        rng.insert(&mut w, "lm_head.weight", &[VOCAB as i32, h], 0.2);
    }
    w
}

pub(crate) fn build_text_model(model_type: &str) -> LlmJpTextModel {
    let mut rng = Rng::new(0x5EED_1362_5EED_1362);
    match model_type {
        "qwen3" => {
            let weights = tiny_text_weights(&mut rng, true, true);
            LlmJpTextModel::Qwen3(
                crate::models::Qwen3Model::from_weights(&weights, &tiny_qwen3_args())
                    .expect("tiny qwen3 backbone builds"),
            )
        }
        _ => {
            let weights = tiny_text_weights(&mut rng, false, false);
            LlmJpTextModel::Llama(
                crate::models::Llama3Model::from_weights(&weights, &tiny_llama_args())
                    .expect("tiny llama backbone builds"),
            )
        }
    }
}

pub(crate) fn build_model(model_type: &str, layer_norm_eps: f32) -> LlmJpVlModel {
    let mut rng = Rng::new(0x00C0_FFEE_1362);
    let vision_config = tiny_vision_config();
    let vision_weights = tiny_vision_weights(&mut rng, "vision_backbone.vision_model");
    let vision_model = SigLipVisionModel::from_weights_with_quant_and_gelu(
        &vision_weights,
        &vision_config,
        "vision_backbone.vision_model",
        64,
        4,
        false,
    )
    .expect("tiny SigLIP tower builds");

    let connector_weights = tiny_connector_weights(&mut rng);
    let connector =
        InternVLConnector::from_weights(&connector_weights, "mlp1", layer_norm_eps, 0.5, 64, 4)
            .expect("tiny mlp1 connector builds");

    let mut processor = InternVLProcessor::new(VIT_IMAGE_SIZE, 1, 12, true);
    processor.mean = [0.5; 3];
    processor.std = [0.5; 3];

    LlmJpVlModel {
        text_model: build_text_model(model_type),
        vision_model,
        connector,
        processor,
        image_context_token_id: IMG_PAD,
        img_start_token_id: IMG_START,
        img_end_token_id: IMG_END,
        // (4/2)^2 * 0.5^2 = 1 feature token per tile at this scale.
        num_image_token: 1,
        eos_token_ids: vec![2, 11],
        image_seq_length: 256,
        model_max_length: 4096,
        max_dynamic_patch: 12,
    }
}

pub(crate) fn tile_pixels(tiles: i32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let count = tiles * 3 * VIT_IMAGE_SIZE as i32 * VIT_IMAGE_SIZE as i32;
    let data: Vec<f32> = (0..count).map(|i| (i as f32 * 0.17).sin()).collect();
    mlxcel_core::from_slice_f32(
        &data,
        &[tiles, 3, VIT_IMAGE_SIZE as i32, VIT_IMAGE_SIZE as i32],
    )
}

pub(crate) fn to_vec(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    let f = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    mlxcel_core::utils::array_to_vec_f32(&f)
}
