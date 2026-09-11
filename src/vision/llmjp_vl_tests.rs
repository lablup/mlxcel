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

//! LLM-jp-VL runtime gates.
//!
//! Four properties carry this port. The tile budget must reproduce upstream's
//! integer arithmetic, including the clamp at both ends. The connector must
//! normalize at torch's default epsilon, not the vision config's, because the
//! two differ by an order of magnitude here. Both text backbones must load
//! behind one runtime and report their own layer counts. And the runtime must
//! declare a server-side cache layout with a non-zero layer count: a VLM whose
//! `sequence_state_layout()` reports zero layers passes every CLI test (the CLI
//! builds its own caches) while the server scheduler hands it an empty cache
//! vector and the decoder silently runs nothing.

use mlxcel_core::cache::SequenceStateBackend;
use mlxcel_core::generate::LanguageModel;

use super::{LLMJP_MLP1_LAYER_NORM_EPS, image_tile_budget};
use crate::models::embedding_test_support::Rng;
use crate::vision::encoders::VisionEncoder;
use crate::vision::llmjp_vl_test_support::{
    IMG_END, IMG_PAD, IMG_START, TEXT_HIDDEN, TEXT_LAYERS, build_model, build_vision_model,
    tile_pixels, tiny_vision_config, tiny_vision_weights, tiny_vision_weights_mlx_layout, to_vec,
};

#[test]
fn image_budget_formula_caps_at_12_and_floors_at_1() {
    // Upstream: max_num = (( 4096 - text ) // images - 2) // 256 - 1, clamped.
    // A short prompt saturates at the checkpoint's max_dynamic_patch.
    assert_eq!(image_tile_budget(4096, 20, 1, 256, 12), 12);
    assert_eq!(image_tile_budget(4096, 499, 1, 256, 12), 12);

    // (4096 - 1000) // 1 - 2 = 3094; 3094 // 256 - 1 = 12 - 1 = 11.
    assert_eq!(image_tile_budget(4096, 1000, 1, 256, 12), 11);

    // Two images halve the per-image budget: (4096 - 20) // 2 - 2 = 2036;
    // 2036 // 256 - 1 = 7 - 1 = 6.
    assert_eq!(image_tile_budget(4096, 20, 2, 256, 12), 6);

    // A prompt that has already eaten the context floors at one tile rather
    // than going negative.
    assert_eq!(image_tile_budget(4096, 4090, 1, 256, 12), 1);
    assert_eq!(image_tile_budget(4096, 9000, 1, 256, 12), 1);

    // The cap follows the checkpoint, not the constant 12.
    assert_eq!(image_tile_budget(4096, 20, 1, 256, 6), 6);
}

#[test]
fn connector_uses_torch_default_layer_norm_eps() {
    // Upstream builds `nn.LayerNorm(vit_hidden * 4)` with no `eps`, so the
    // connector normalizes at torch's 1e-5 while `vision_config.layer_norm_eps`
    // is 1e-6. The InternVL loader passes the config value at the equivalent
    // call site; this family must not.
    assert_eq!(LLMJP_MLP1_LAYER_NORM_EPS, 1e-5);
    assert_ne!(
        LLMJP_MLP1_LAYER_NORM_EPS,
        tiny_vision_config().layer_norm_eps
    );

    // The epsilon is observable in the output, so the two are not
    // interchangeable in practice either.
    let ids = mlxcel_core::from_slice_i32(&[1, IMG_PAD, 3], &[1, 3]);
    let pixels = tile_pixels(1);
    let torch_default = build_model("llama", LLMJP_MLP1_LAYER_NORM_EPS);
    let vision_config_eps = build_model("llama", 1e-6);
    let a = to_vec(
        &torch_default
            .get_input_embeddings(&ids, &pixels)
            .inputs_embeds,
    );
    let b = to_vec(
        &vision_config_eps
            .get_input_embeddings(&ids, &pixels)
            .inputs_embeds,
    );
    assert_eq!(a.len(), b.len());
    let max_diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 0.0,
        "the two epsilons must not produce identical embeddings"
    );
}

#[test]
fn text_model_selection_by_llm_config_model_type() {
    // Both released backbones load behind one runtime and report their own
    // identity and depth.
    let llama = build_model("llama", LLMJP_MLP1_LAYER_NORM_EPS);
    assert_eq!(llama.text_model.backbone_name(), "llama");
    assert_eq!(LanguageModel::num_layers(&llama), TEXT_LAYERS);

    let qwen3 = build_model("qwen3", LLMJP_MLP1_LAYER_NORM_EPS);
    assert_eq!(qwen3.text_model.backbone_name(), "qwen3");
    assert_eq!(LanguageModel::num_layers(&qwen3), TEXT_LAYERS);

    // The two backbones are genuinely different graphs: same tiny shapes, same
    // seeded weights, different logits.
    let ids = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let mut llama_caches = LanguageModel::make_caches(&llama);
    let mut qwen3_caches = LanguageModel::make_caches(&qwen3);
    let llama_logits = to_vec(&LanguageModel::forward(
        &llama,
        &ids,
        &mut llama_caches,
        None,
    ));
    let qwen3_logits = to_vec(&LanguageModel::forward(
        &qwen3,
        &ids,
        &mut qwen3_caches,
        None,
    ));
    assert_eq!(llama_logits.len(), qwen3_logits.len());
    assert!(
        llama_logits
            .iter()
            .zip(&qwen3_logits)
            .any(|(a, b)| (a - b).abs() > 1e-5),
        "the llama and qwen3 arms must not compute the same thing"
    );
    assert!(
        llama_logits.iter().all(|v| v.is_finite()),
        "llama logits must be finite"
    );
    assert!(
        qwen3_logits.iter().all(|v| v.is_finite()),
        "qwen3 logits must be finite"
    );
}

#[test]
fn image_features_land_only_on_image_pad_positions() {
    let model = build_model("qwen3", LLMJP_MLP1_LAYER_NORM_EPS);
    let tokens = vec![1, IMG_START, IMG_PAD, IMG_END, 5, 6];
    let ids = mlxcel_core::from_slice_i32(&tokens, &[1, tokens.len() as i32]);
    let pixels = tile_pixels(1);

    let base = to_vec(&model.text_model.get_embed_tokens(&ids));
    let merged = to_vec(&model.get_input_embeddings(&ids, &pixels).inputs_embeds);
    assert_eq!(base.len(), merged.len());

    for position in 0..tokens.len() {
        let range = position * TEXT_HIDDEN..(position + 1) * TEXT_HIDDEN;
        let changed = base[range.clone()]
            .iter()
            .zip(merged[range].iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert_eq!(
            changed,
            position == 2,
            "position {position} changed={changed}; only the <|image_pad|> position may change"
        );
    }
}

#[test]
fn the_server_cache_layout_follows_the_text_backbone() {
    // The VLM trap this guards: an unoverridden `sequence_state_layout()` plus
    // `supports_batching() == false` hands the server scheduler an empty cache
    // vector, and the decoder runs zero layers while every CLI test still
    // passes (the CLI builds its own caches).
    for backbone in ["llama", "qwen3"] {
        let model = build_model(backbone, LLMJP_MLP1_LAYER_NORM_EPS);
        let layout = LanguageModel::sequence_state_layout(&model);
        assert_eq!(
            layout.num_layers, TEXT_LAYERS,
            "{backbone}: sequence state must carry one entry per decoder layer"
        );
        assert_eq!(layout.backend, SequenceStateBackend::DenseKvCache);
        assert!(
            LanguageModel::supports_batching(&model),
            "{backbone}: the text backbone batches, so the VLM must too"
        );
        assert_eq!(
            LanguageModel::make_caches(&model).len(),
            TEXT_LAYERS,
            "{backbone}: make_caches must not hand back an empty vector"
        );
    }
}

#[test]
fn the_three_framing_ids_are_suppressed_from_the_output() {
    let model = build_model("llama", LLMJP_MLP1_LAYER_NORM_EPS);
    let suppressed = LanguageModel::output_suppressed_token_ids(&model);
    for id in [IMG_PAD, IMG_START, IMG_END] {
        assert!(suppressed.contains(&id), "id {id} must be suppressed");
    }
    // The stop ids must stay emittable, or generation could never terminate.
    for id in LanguageModel::eos_token_ids(&model) {
        assert!(
            !suppressed.contains(&id),
            "stop id {id} must not be suppressed"
        );
    }
}

#[test]
fn either_patch_embedding_conv_layout_produces_the_same_features() {
    // The released bf16 originals ship the HF `[O, I, kH, kW]` layout
    // (`[1152, 3, 16, 16]`); an MLX conversion would ship `[O, kH, kW, I]`.
    // `VisionEmbeddings::from_weights` sanitizes the first into the second, so
    // both must reach the encoder as the same weight and the tower must emit
    // identical features. The fixture also carries a `head.probe` tensor, the
    // attention-pooling head the loader drops: the tower never reads it, so its
    // presence changes nothing here either.
    let mut rng = Rng::new(0x00C0_FFEE_1362);
    let hf_layout = tiny_vision_weights(&mut rng, "vision_backbone.vision_model");
    let mut rng = Rng::new(0x00C0_FFEE_1362);
    let mlx_layout = tiny_vision_weights_mlx_layout(&mut rng, "vision_backbone.vision_model");
    assert!(
        hf_layout.contains_key("vision_backbone.vision_model.head.probe"),
        "the fixture must carry the pooling head the loader drops"
    );

    let pixels = tile_pixels(1);
    // The tower is channels-last, so transpose the processor's [N, C, H, W].
    let pixels = mlxcel_core::transpose_axes(&pixels, &[0, 2, 3, 1]);
    let from_hf = to_vec(
        &build_vision_model(&hf_layout)
            .forward(&pixels)
            .hidden_states,
    );
    let from_mlx = to_vec(
        &build_vision_model(&mlx_layout)
            .forward(&pixels)
            .hidden_states,
    );

    assert_eq!(from_hf.len(), from_mlx.len());
    let max_diff = from_hf
        .iter()
        .zip(&from_mlx)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff == 0.0,
        "the two conv layouts must reach the encoder as the same weight (max diff {max_diff})"
    );
    assert!(
        from_hf.iter().all(|v| v.is_finite()),
        "features must be finite"
    );
}
