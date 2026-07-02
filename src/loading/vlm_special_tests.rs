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

use super::{
    cap_molmo2_vit_num_layers, dequantize_moondream3_weight, flatten_phi4mm_patch_embedding,
    inherit_quantization_if_missing, llama4_mm_tokens_per_image, llama4_quantization_params,
    llama4_token_ids, llama4_vision_prefix, minicpmv4_6_text_weights, molmo2_max_crops,
    moondream2_text_config_value, moondream3_text_config_value, moondream3_vision_config_value,
    parse_molmo2_vit_layers, phi3_num_crops, phi4_siglip_text_config_value,
    phi4mm_text_config_value, phi4mm_vision_config_value, remap_minicpmo_text_weights,
    remap_minicpmv4_6_weights, resolve_moondream2_eos_token_id, rewrite_molmo2_weight_key,
    rewrite_moondream2_weight_key, rewrite_moondream3_weight_key, rewrite_phi3_weight_key,
    rewrite_phi4_siglip_weight_key, rewrite_phi4mm_vision_key,
    should_transpose_phi3_patch_embedding,
};
use crate::moondream2_prompt::Moondream2PromptStyle;
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

#[test]
fn remap_minicpmo_text_weights_strips_language_model_prefix() {
    let mut weights = WeightMap::new();
    weights.insert(
        "language_model.model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );
    weights.insert(
        "language_model.lm_head.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );
    weights.insert(
        "vision_tower.embeddings.patch_embedding.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );

    let remapped = remap_minicpmo_text_weights(&weights);
    assert!(remapped.contains_key("model.embed_tokens.weight"));
    assert!(remapped.contains_key("lm_head.weight"));
    assert!(remapped.contains_key("vision_tower.embeddings.patch_embedding.weight"));
}

#[test]
fn remap_minicpmv4_6_weights_maps_all_namespaces_and_skips_position_ids() {
    let mut weights = WeightMap::new();
    for key in [
        "model.language_model.model.layers.0.input_layernorm.weight",
        "model.lm_head.weight",
        "model.vision_tower.vit_merger.linear_1.weight",
        "model.vision_tower.encoder.layers.0.layer_norm1.weight",
        "model.vpm.post_layernorm.weight",
        "model.vit_merger.pre_norm.weight",
        "model.merger.mlp.0.linear_1.weight",
        // Backward-compat prefixes.
        "model.llm.model.norm.weight",
        "model.visual.embeddings.patch_embedding.weight",
        // Must be dropped.
        "model.vision_tower.embeddings.position_ids",
    ] {
        weights.insert(key.to_string(), mlxcel_core::ones(&[2, 2], dtype::FLOAT32));
    }

    let remapped = remap_minicpmv4_6_weights(&weights);

    // language_model / lm_head namespaces.
    assert!(remapped.contains_key("language_model.model.layers.0.input_layernorm.weight"));
    assert!(remapped.contains_key("lm_head.weight"));
    // vit_merger (both `vision_tower.vit_merger.*` and bare `vit_merger.*`).
    assert!(remapped.contains_key("vit_merger.linear_1.weight"));
    assert!(remapped.contains_key("vit_merger.pre_norm.weight"));
    // vision_tower (encoder, vpm-aliased, and visual-aliased).
    assert!(remapped.contains_key("vision_tower.encoder.layers.0.layer_norm1.weight"));
    assert!(remapped.contains_key("vision_tower.post_layernorm.weight"));
    assert!(remapped.contains_key("vision_tower.embeddings.patch_embedding.weight"));
    // merger.
    assert!(remapped.contains_key("merger.mlp.0.linear_1.weight"));
    // llm.* backward-compat maps into the language_model namespace.
    assert!(remapped.contains_key("language_model.model.norm.weight"));
    // position_ids dropped.
    assert!(!remapped.keys().any(|k| k.contains("position_ids")));
}

#[test]
fn minicpmv4_6_text_weights_strips_language_model_prefix_and_keeps_lm_head() {
    let mut weights = WeightMap::new();
    weights.insert(
        "language_model.model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );
    weights.insert(
        "language_model.model.layers.0.input_layernorm.weight".to_string(),
        mlxcel_core::ones(&[2], dtype::FLOAT32),
    );
    weights.insert(
        "lm_head.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );
    // Vision weights must NOT leak into the text weight map.
    weights.insert(
        "vision_tower.encoder.layers.0.layer_norm1.weight".to_string(),
        mlxcel_core::ones(&[2], dtype::FLOAT32),
    );
    weights.insert(
        "merger.mlp.0.linear_1.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );

    let text = minicpmv4_6_text_weights(&weights);
    // The Qwen35Model loader expects the de-prefixed `model.*` namespace.
    assert!(text.contains_key("model.embed_tokens.weight"));
    assert!(text.contains_key("model.layers.0.input_layernorm.weight"));
    assert!(text.contains_key("lm_head.weight"));
    assert!(!text.contains_key("vision_tower.encoder.layers.0.layer_norm1.weight"));
    assert!(!text.contains_key("merger.mlp.0.linear_1.weight"));
}

#[test]
fn rewrite_moondream3_weight_key_strips_model_prefix_and_skips_region_branch() {
    assert_eq!(
        rewrite_moondream3_weight_key("model.text.wte"),
        Some("text.wte.weight".to_string())
    );
    assert_eq!(
        rewrite_moondream3_weight_key("model.text.blocks.4.attn.qkv.weight.packed"),
        Some("text.blocks.4.attn.qkv.weight.packed".to_string())
    );
    assert_eq!(
        rewrite_moondream3_weight_key("model.region.coord_encoder.weight"),
        None
    );
}

#[test]
fn moondream3_text_and_vision_config_helpers_fill_default_shapes() {
    let text = moondream3_text_config_value(&json!({
        "text_group_size": 64,
        "expert_group_size": 32,
        "quantization_config": {"quant_method": "int4"}
    }));
    let vision = moondream3_vision_config_value(&json!({}));

    assert_eq!(text["model_type"], "moondream3");
    assert_eq!(text["group_size"], 64);
    assert_eq!(text["moe"]["expert_group_size"], 32);
    assert_eq!(text["bits"], 4);
    assert_eq!(vision["crop_size"], 378);
    assert_eq!(vision["enc_patch_size"], 14);
}

#[test]
fn rewrite_moondream2_weight_key_maps_unified_checkpoint_layout() {
    // Text tower: strip `model.`, add `.weight` to the tied embedding.
    assert_eq!(
        rewrite_moondream2_weight_key("model.text.wte"),
        Some("text.wte.weight".to_string())
    );
    assert_eq!(
        rewrite_moondream2_weight_key("model.text.blocks.0.ln.weight"),
        Some("text.blocks.0.ln.weight".to_string())
    );
    assert_eq!(
        rewrite_moondream2_weight_key("model.text.blocks.3.attn.qkv.weight"),
        Some("text.blocks.3.attn.qkv.weight".to_string())
    );
    assert_eq!(
        rewrite_moondream2_weight_key("model.text.post_ln.weight"),
        Some("text.post_ln.weight".to_string())
    );
    assert_eq!(
        rewrite_moondream2_weight_key("model.text.lm_head.bias"),
        Some("text.lm_head.bias".to_string())
    );

    // Vision tower: raw pos_emb keeps no `.weight`, blocks keep ln1/ln2.
    assert_eq!(
        rewrite_moondream2_weight_key("model.vision.patch_emb.weight"),
        Some("vision.patch_emb.weight".to_string())
    );
    assert_eq!(
        rewrite_moondream2_weight_key("model.vision.pos_emb"),
        Some("vision.pos_emb".to_string())
    );
    assert_eq!(
        rewrite_moondream2_weight_key("model.vision.blocks.0.ln1.weight"),
        Some("vision.blocks.0.ln1.weight".to_string())
    );
    assert_eq!(
        rewrite_moondream2_weight_key("model.vision.proj_mlp.fc1.weight"),
        Some("vision.proj_mlp.fc1.weight".to_string())
    );

    // Region head and position-id buffers are dropped; already-stripped keys
    // pass through unchanged.
    assert_eq!(
        rewrite_moondream2_weight_key("model.region.coord_encoder.weight"),
        None
    );
    assert_eq!(
        rewrite_moondream2_weight_key("model.text.position_ids"),
        None
    );
    assert_eq!(
        rewrite_moondream2_weight_key("vision.blocks.0.attn.qkv.weight"),
        Some("vision.blocks.0.attn.qkv.weight".to_string())
    );
}

#[test]
fn moondream2_text_config_helper_fills_dense_phi_shapes() {
    let text = moondream2_text_config_value(
        &json!({
            "quantization": {"group_size": 32, "bits": 8}
        }),
        50256,
    );

    assert_eq!(text["model_type"], "moondream2");
    assert_eq!(text["dim"], 2048);
    assert_eq!(text["n_heads"], 32);
    assert_eq!(text["partial_rotary_factor"], 0.5);
    assert_eq!(text["rope_theta"], 10000.0);
    assert_eq!(text["group_size"], 32);
    assert_eq!(text["bits"], 8);
    // The resolved special-token id feeds both bos and eos.
    assert_eq!(text["eos_token_id"], 50256);
    assert_eq!(text["bos_token_id"], 50256);

    // No quantization block -> fp16 defaults (group_size 64 / bits 4).
    let text_default = moondream2_text_config_value(&json!({}), 50256);
    assert_eq!(text_default["group_size"], 64);
    assert_eq!(text_default["bits"], 4);
}

#[test]
fn resolve_moondream2_eos_falls_back_to_endoftext_id_for_legacy_era() {
    // A legacy-era moondream2 checkpoint: `model_type: "moondream1"` with an
    // empty nested `config` and no top-level `eos_token_id`. Without a
    // tokenizer the resolver must fall back to the GPT-2 `<|endoftext|>` id
    // (50256).
    let real_config = json!({
        "architectures": ["HfMoondream"],
        "model_type": "moondream1",
        "config": {},
        "torch_dtype": "bfloat16"
    });
    assert_eq!(
        resolve_moondream2_eos_token_id(
            &real_config,
            None,
            Moondream2PromptStyle::LegacyQuestionAnswer
        ),
        50256
    );
}

#[test]
fn resolve_moondream2_eos_is_zero_for_starmie_era_despite_stale_tokenizer_config() {
    // The real 2025-06-21 checkpoint: config carries no explicit ids, and the
    // STALE legacy tokenizer_config.json still maps `<|endoftext|>` to 50256.
    // For starmie-era weights the true stop token is id 0
    // (`<|endoftext|>` in moondream/starmie-v1); trusting the stale sidecar
    // makes generation run past the real EOS into degenerate repetition.
    let real_config = json!({ "model_type": "moondream1", "config": {} });
    let stale_tokenizer_config = json!({
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "tokenizer_class": "CodeGenTokenizer",
        "added_tokens_decoder": {
            "50256": { "content": "<|endoftext|>", "special": true }
        }
    });
    assert_eq!(
        resolve_moondream2_eos_token_id(
            &real_config,
            Some(&stale_tokenizer_config),
            Moondream2PromptStyle::StarmieTemplates
        ),
        0
    );
}

#[test]
fn resolve_moondream2_eos_reads_id_from_tokenizer_config_for_legacy_era() {
    // The legacy moondream2 tokenizer_config.json shape: `eos_token` is the
    // `<|endoftext|>` string and `added_tokens_decoder` maps id 50256 to it.
    let real_config = json!({ "model_type": "moondream1", "config": {} });
    let tokenizer_config = json!({
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "tokenizer_class": "CodeGenTokenizer",
        "added_tokens_decoder": {
            "50256": { "content": "<|endoftext|>", "special": true }
        }
    });
    assert_eq!(
        resolve_moondream2_eos_token_id(
            &real_config,
            Some(&tokenizer_config),
            Moondream2PromptStyle::LegacyQuestionAnswer
        ),
        50256
    );
}

#[test]
fn resolve_moondream2_eos_prefers_explicit_config_ids() {
    for style in [
        Moondream2PromptStyle::StarmieTemplates,
        Moondream2PromptStyle::LegacyQuestionAnswer,
    ] {
        // An explicit top-level id wins over the era/tokenizer/fallback.
        let top_level = json!({ "eos_token_id": 7 });
        assert_eq!(resolve_moondream2_eos_token_id(&top_level, None, style), 7);

        // A nested `config.eos_token_id` is honored when the top level is
        // absent.
        let nested = json!({ "config": { "eos_token_id": 11 } });
        assert_eq!(resolve_moondream2_eos_token_id(&nested, None, style), 11);
    }
}

/// Enumerate the exact tensor-key list of the real `vikhyatk/moondream2`
/// checkpoint (2025-06-21 revision, 592 tensors): 24 text blocks x 10 keys,
/// 5 top-level text keys, 27 vision blocks x 12 keys, 9 top-level vision
/// keys, and 14 region keys.
fn real_moondream2_checkpoint_keys() -> Vec<String> {
    let mut keys = Vec::new();

    for layer in 0..24 {
        for suffix in [
            "attn.proj.bias",
            "attn.proj.weight",
            "attn.qkv.bias",
            "attn.qkv.weight",
            "ln.bias",
            "ln.weight",
            "mlp.fc1.bias",
            "mlp.fc1.weight",
            "mlp.fc2.bias",
            "mlp.fc2.weight",
        ] {
            keys.push(format!("model.text.blocks.{layer}.{suffix}"));
        }
    }
    for suffix in [
        "lm_head.bias",
        "lm_head.weight",
        "post_ln.bias",
        "post_ln.weight",
        "wte",
    ] {
        keys.push(format!("model.text.{suffix}"));
    }

    for layer in 0..27 {
        for suffix in [
            "attn.proj.bias",
            "attn.proj.weight",
            "attn.qkv.bias",
            "attn.qkv.weight",
            "ln1.bias",
            "ln1.weight",
            "ln2.bias",
            "ln2.weight",
            "mlp.fc1.bias",
            "mlp.fc1.weight",
            "mlp.fc2.bias",
            "mlp.fc2.weight",
        ] {
            keys.push(format!("model.vision.blocks.{layer}.{suffix}"));
        }
    }
    for suffix in [
        "patch_emb.bias",
        "patch_emb.weight",
        "pos_emb",
        "post_ln.bias",
        "post_ln.weight",
        "proj_mlp.fc1.bias",
        "proj_mlp.fc1.weight",
        "proj_mlp.fc2.bias",
        "proj_mlp.fc2.weight",
    ] {
        keys.push(format!("model.vision.{suffix}"));
    }

    for suffix in [
        "coord_decoder.fc1.bias",
        "coord_decoder.fc1.weight",
        "coord_decoder.fc2.bias",
        "coord_decoder.fc2.weight",
        "coord_encoder.bias",
        "coord_encoder.weight",
        "coord_features",
        "size_decoder.fc1.bias",
        "size_decoder.fc1.weight",
        "size_decoder.fc2.bias",
        "size_decoder.fc2.weight",
        "size_encoder.bias",
        "size_encoder.weight",
        "size_features",
    ] {
        keys.push(format!("model.region.{suffix}"));
    }

    assert_eq!(keys.len(), 592, "real checkpoint tensor count");
    keys
}

/// The full weight-name set the Moondream2 text + (shared Moondream3) vision
/// loaders consume: every `UnifiedLinear`/`LayerNorm` weight+bias, the raw
/// `vision.pos_emb`, and the embedding under `text.wte.weight`.
fn moondream2_loader_required_keys() -> std::collections::BTreeSet<String> {
    let mut keys = std::collections::BTreeSet::new();

    for layer in 0..24 {
        for suffix in [
            "ln.weight",
            "ln.bias",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ] {
            keys.insert(format!("text.blocks.{layer}.{suffix}"));
        }
    }
    for key in [
        "text.wte.weight",
        "text.post_ln.weight",
        "text.post_ln.bias",
        "text.lm_head.weight",
        "text.lm_head.bias",
    ] {
        keys.insert(key.to_string());
    }

    for layer in 0..27 {
        for suffix in [
            "ln1.weight",
            "ln1.bias",
            "ln2.weight",
            "ln2.bias",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ] {
            keys.insert(format!("vision.blocks.{layer}.{suffix}"));
        }
    }
    for key in [
        "vision.patch_emb.weight",
        "vision.patch_emb.bias",
        "vision.pos_emb",
        "vision.post_ln.weight",
        "vision.post_ln.bias",
        "vision.proj_mlp.fc1.weight",
        "vision.proj_mlp.fc1.bias",
        "vision.proj_mlp.fc2.weight",
        "vision.proj_mlp.fc2.bias",
    ] {
        keys.insert(key.to_string());
    }

    keys
}

#[test]
fn rewrite_moondream2_weight_key_covers_the_real_checkpoint_exactly() {
    // Guard the remap against the real 2025-06-21 checkpoint contract: every
    // region tensor is dropped, and the remaining 578 tensors remap onto
    // EXACTLY the key set the text/vision loaders request, with no leftovers
    // (a tensor silently landing in an unused slot would zero-initialize
    // nothing today but would mask future drift) and no misses (a missing
    // slot fails the load).
    let remapped: std::collections::BTreeSet<String> = real_moondream2_checkpoint_keys()
        .iter()
        .filter_map(|key| rewrite_moondream2_weight_key(key))
        .collect();

    let required = moondream2_loader_required_keys();

    let missing: Vec<_> = required.difference(&remapped).collect();
    assert!(
        missing.is_empty(),
        "loader-required keys not produced by the remap: {missing:?}"
    );

    let unused: Vec<_> = remapped.difference(&required).collect();
    assert!(
        unused.is_empty(),
        "remapped keys no loader consumes: {unused:?}"
    );

    assert_eq!(remapped.len(), 592 - 14, "region tensors must be dropped");
}

#[test]
fn dequantize_moondream3_weight_restores_interleaved_uint4_rows() {
    let mut packed_bytes = [0u8; 128];
    packed_bytes[0] = 0x1F;
    packed_bytes[1] = 0x20;
    let packed_i32: Vec<i32> = packed_bytes.iter().map(|&value| value as i32).collect();
    let packed = mlxcel_core::from_slice_i32(&packed_i32, &[1, 128]);
    let packed = mlxcel_core::astype(&packed, dtype::UINT8);
    let scale = mlxcel_core::ones(&[2, 1], dtype::FLOAT32);
    let zero = mlxcel_core::zeros(&[2, 1], dtype::FLOAT32);

    let dequantized = dequantize_moondream3_weight(&packed, &scale, &zero, &[2, 128]);
    assert_eq!(mlxcel_core::array_shape(&dequantized), vec![2, 128]);
    mlxcel_core::eval(&dequantized);
    let total = mlxcel_core::sum_all(&mlxcel_core::astype(&dequantized, dtype::FLOAT32));
    mlxcel_core::eval(&total);
    assert!(mlxcel_core::item_f32(&total) > 0.0);
}

#[test]
fn rewrite_phi3_weight_key_skips_position_ids_and_maps_known_prefixes() {
    assert_eq!(
        rewrite_phi3_weight_key("model.embed_tokens.weight"),
        Some("model.embed_tokens.weight".to_string())
    );
    assert_eq!(
        rewrite_phi3_weight_key(
            "model.vision_embed_tokens.img_processor.vision_model.embeddings.patch_embedding.weight"
        ),
        Some("vision_tower.vision_model.embeddings.patch_embedding.weight".to_string())
    );
    assert_eq!(
        rewrite_phi3_weight_key("model.vision_embed_tokens.img_projection.0.weight"),
        Some("img_projection.0.weight".to_string())
    );
    assert_eq!(
        rewrite_phi3_weight_key("model.vision_embed_tokens.glb_GN"),
        Some("glb_GN".to_string())
    );
    assert_eq!(rewrite_phi3_weight_key("model.position_ids"), None);
}

#[test]
fn phi3_patch_embedding_transpose_detection_matches_layout_expectations() {
    assert!(!should_transpose_phi3_patch_embedding(&[1024, 14, 14, 3]));
    assert!(should_transpose_phi3_patch_embedding(&[14, 14, 3, 1024]));
    assert!(!should_transpose_phi3_patch_embedding(&[1024, 196]));
}

#[test]
fn flatten_phi4mm_patch_embedding_flattens_both_layouts_to_same_shape() {
    // PyTorch layout [out, in, kH, kW]: transpose to channel-last, then flatten.
    let pytorch = mlxcel_core::ones(&[1024, 3, 14, 14], dtype::FLOAT32);
    let flat = flatten_phi4mm_patch_embedding(&pytorch);
    assert_eq!(mlxcel_core::array_shape(&flat), vec![1024, 3 * 14 * 14]);

    // Already channel-last [out, kH, kW, in]: skip the transpose (issue #428),
    // flatten to the same [out, in*kH*kW] shape without double-converting.
    let channel_last = mlxcel_core::ones(&[1024, 14, 14, 3], dtype::FLOAT32);
    let flat = flatten_phi4mm_patch_embedding(&channel_last);
    assert_eq!(mlxcel_core::array_shape(&flat), vec![1024, 14 * 14 * 3]);

    // Non-4D weights are copied through unchanged.
    let already_flat = mlxcel_core::ones(&[1024, 196], dtype::FLOAT32);
    let out = flatten_phi4mm_patch_embedding(&already_flat);
    assert_eq!(mlxcel_core::array_shape(&out), vec![1024, 196]);
}

#[test]
fn phi3_num_crops_prefers_preprocessor_then_config_then_default() {
    assert_eq!(
        phi3_num_crops(
            &json!({"vision_config": {"num_crops": 8}}),
            Some(&json!({"num_crops": 4}))
        ),
        4
    );
    assert_eq!(
        phi3_num_crops(
            &json!({"vision_config": {"num_crops": 8}}),
            Some(&json!({}))
        ),
        4
    );
    assert_eq!(
        phi3_num_crops(&json!({"vision_config": {"num_crops": 8}}), None),
        8
    );
    assert_eq!(phi3_num_crops(&json!({}), None), 16);
}

#[test]
fn rewrite_phi4_siglip_weight_key_keeps_text_keys_and_remaps_multimodal_prefixes() {
    assert_eq!(
        rewrite_phi4_siglip_weight_key("model.layers.0.self_attn.qkv_proj.weight"),
        Some("model.layers.0.self_attn.qkv_proj.weight".to_string())
    );
    assert_eq!(
        rewrite_phi4_siglip_weight_key(
            "model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight"
        ),
        Some(
            "vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight".to_string()
        )
    );
    assert_eq!(
        rewrite_phi4_siglip_weight_key("model.mm_projector.0.weight"),
        Some("mm_projector_linear1.weight".to_string())
    );
    assert_eq!(rewrite_phi4_siglip_weight_key("model.position_ids"), None);
}

#[test]
fn phi4_siglip_text_config_value_inherits_root_text_fields() {
    let text_config = phi4_siglip_text_config_value(&json!({
        "model_type": "phi4-siglip",
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_hidden_layers": 40,
        "intermediate_size": 17920,
        "vocab_size": 100352,
        "rope_theta": 500000.0,
        "quantization": {"group_size": 64, "bits": 4},
        "vision_config": {"hidden_size": 1152}
    }))
    .unwrap();

    assert_eq!(text_config["hidden_size"], 5120);
    assert_eq!(text_config["num_attention_heads"], 40);
    assert_eq!(text_config["quantization"]["group_size"], 64);
}

#[test]
fn rewrite_phi4mm_vision_key_maps_multimodal_prefixes_and_skips_audio() {
    assert_eq!(
        rewrite_phi4mm_vision_key(
            "model.embed_tokens_extend.image_embed.img_processor.embeddings.patch_embedding.weight"
        ),
        Some(
            "vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight".to_string()
        )
    );
    assert_eq!(
        rewrite_phi4mm_vision_key("model.embed_tokens_extend.image_embed.img_projection.0.weight"),
        Some("mm_projector_linear1.weight".to_string())
    );
    assert_eq!(
        rewrite_phi4mm_vision_key("model.layers.0.self_attn.qkv_proj.base_layer.weight"),
        Some("model.layers.0.self_attn.qkv_proj.base_layer.weight".to_string())
    );
    assert_eq!(
        rewrite_phi4mm_vision_key(
            "model.embed_tokens_extend.audio_embed.audio_projection.speech.0.weight"
        ),
        None
    );
}

#[test]
fn phi4mm_text_config_value_inherits_root_text_fields() {
    let text_config = phi4mm_text_config_value(&json!({
        "model_type": "phi4mm",
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "num_hidden_layers": 32,
        "intermediate_size": 8192,
        "vocab_size": 200064,
        "partial_rotary_factor": 0.75,
        "tie_word_embeddings": true
    }))
    .unwrap();

    assert_eq!(text_config["model_type"], "phi4mm");
    assert_eq!(text_config["partial_rotary_factor"], 0.75);
    assert_eq!(text_config["tie_word_embeddings"], true);
}

#[test]
fn phi4mm_vision_config_value_uses_crop_size_defaults() {
    let vision_config = phi4mm_vision_config_value(&json!({
        "embd_layer": {
            "image_embd_layer": {
                "crop_size": 448
            }
        }
    }));

    assert_eq!(vision_config["patch_size"], 14);
    assert_eq!(vision_config["image_size"], 448);
    assert_eq!(vision_config["num_patches"], 1024);
}

#[test]
fn molmo2_helpers_clamp_layer_count_and_parse_defaults() {
    assert_eq!(cap_molmo2_vit_num_layers(27), 25);
    assert_eq!(cap_molmo2_vit_num_layers(12), 12);
    assert_eq!(parse_molmo2_vit_layers(&json!({})), vec![-3, -9]);
    assert_eq!(
        parse_molmo2_vit_layers(&json!({"vit_layers": [-1, -7, 3]})),
        vec![-1, -7, 3]
    );
}

#[test]
fn rewrite_molmo2_weight_key_maps_text_vision_and_lm_head_prefixes() {
    assert_eq!(
        rewrite_molmo2_weight_key("model.transformer.layers.0.self_attn.q_proj.weight"),
        "language_model.model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        rewrite_molmo2_weight_key(
            "model.vision_backbone.transformer.resblocks.0.attn.q_proj.weight"
        ),
        "vision_tower.transformer.0.attn.q_proj.weight"
    );
    assert_eq!(
        rewrite_molmo2_weight_key("lm_head.weight"),
        "language_model.lm_head.weight"
    );
}

#[test]
fn molmo2_max_crops_uses_default_when_preprocessor_is_missing() {
    assert_eq!(molmo2_max_crops(None), 8);
    assert_eq!(molmo2_max_crops(Some(&json!({"max_crops": 12}))), 12);
}

#[test]
fn inherit_quantization_if_missing_copies_top_level_quantization_once() {
    let mut text_config = json!({
        "hidden_size": 4096
    });
    let full_config = json!({
        "quantization": {"group_size": 128, "bits": 8}
    });

    inherit_quantization_if_missing(&mut text_config, &full_config).unwrap();
    assert_eq!(text_config["quantization"]["group_size"], 128);
    assert_eq!(text_config["quantization"]["bits"], 8);

    let mut explicit = json!({
        "quantization": {"group_size": 64, "bits": 4}
    });
    inherit_quantization_if_missing(&mut explicit, &full_config).unwrap();
    assert_eq!(explicit["quantization"]["group_size"], 64);
    assert_eq!(explicit["quantization"]["bits"], 4);
}

#[test]
fn inherit_quantization_if_missing_rejects_non_object_text_config() {
    let mut text_config = json!(5);
    let full_config = json!({
        "quantization": {"group_size": 128, "bits": 8}
    });

    let err = inherit_quantization_if_missing(&mut text_config, &full_config)
        .unwrap_err()
        .to_string();
    assert!(err.contains("special VLM text_config"));
}

#[test]
fn llama4_helpers_cover_prefix_detection_defaults_and_token_math() {
    let mut vision_model_weights = WeightMap::new();
    vision_model_weights.insert(
        "vision_model.patch_embedding.linear.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );
    assert_eq!(llama4_vision_prefix(&vision_model_weights), "vision_model");

    let mut tower_weights = WeightMap::new();
    tower_weights.insert(
        "vision_tower.patch_embedding.linear.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );
    assert_eq!(llama4_vision_prefix(&tower_weights), "vision_tower");

    assert_eq!(llama4_quantization_params(&json!({})), (64, 4));
    assert_eq!(
        llama4_quantization_params(&json!({"quantization": {"group_size": 128, "bits": 6}})),
        (128, 6)
    );
    assert_eq!(llama4_token_ids(&json!({})), (200092, 200018));
    assert_eq!(
        llama4_token_ids(&json!({"image_token_index": 7, "text_config": {"pad_token_id": 9}})),
        (7, 9)
    );
}

#[test]
fn llama4_mm_tokens_per_image_applies_pixel_shuffle_ratio() {
    let config: crate::vision::encoders::llama4::Llama4VisionConfig =
        serde_json::from_value(json!({
            "hidden_size": 1024,
            "image_size": 1120,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "patch_size": 14,
            "pixel_shuffle_ratio": 0.5
        }))
        .unwrap();

    assert_eq!(llama4_mm_tokens_per_image(&config), 1600);
}
