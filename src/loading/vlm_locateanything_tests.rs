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

//! Unit tests for the LocateAnything loader: config normalization, weight-key
//! remapping, and stop-token resolution.

use super::*;

/// The `vision_config` block verbatim from `mlx-community/LocateAnything-3B-4bit`.
fn real_vision_config() -> serde_json::Value {
    serde_json::json!({
        "_attn_implementation_autoset": true,
        "_name_or_path": "moonshotai/MoonViT-SO-400M",
        "auto_map": {
            "AutoConfig": "moonshotai/MoonViT-SO-400M--configuration_moonvit.MoonViTConfig",
            "AutoModel": "moonshotai/MoonViT-SO-400M--modeling_moonvit.MoonVitPretrainedModel"
        },
        "hidden_size": 1152,
        "init_pos_emb_height": 64,
        "init_pos_emb_width": 64,
        "intermediate_size": 4304,
        "merge_kernel_size": [2, 2],
        "model_type": "moonvit",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
        "torch_dtype": "bfloat16"
    })
}

/// The `text_config` block verbatim from `mlx-community/LocateAnything-3B-4bit`.
fn real_text_config() -> serde_json::Value {
    serde_json::json!({
        "_attn_implementation_autoset": true,
        "_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "architectures": ["Qwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "block_size": 6,
        "bos_token_id": 151643,
        "causal_attn": false,
        "eos_token_id": 151645,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 32768,
        "max_window_layers": 70,
        "model_type": "qwen2",
        "null_token_id": 152678,
        "num_attention_heads": 16,
        "num_hidden_layers": 36,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_scaling": null,
        "rope_theta": 1000000.0,
        "sliding_window": 32768,
        "switch_token_id": 152679,
        "text_mask_token_id": 151676,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "use_cache": false,
        "use_sliding_window": false,
        "vocab_size": 152681
    })
}

fn real_full_config() -> serde_json::Value {
    serde_json::json!({
        "_attn_implementation": "magi",
        "architectures": ["LocateAnythingForConditionalGeneration"],
        "box_end_token_id": 151669,
        "box_start_token_id": 151668,
        "coord_end_token_id": 152677,
        "coord_start_token_id": 151677,
        "eos_token_id": 151645,
        "image_token_index": 151665,
        "mlp_connector_layers": 2,
        "model_type": "locateanything",
        "none_token_id": 4064,
        "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
        "ref_end_token_id": 151673,
        "ref_start_token_id": 151672,
        "text_config": real_text_config(),
        "vision_config": real_vision_config()
    })
}

fn one() -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::ones(&[1], mlxcel_core::dtype::FLOAT32)
}

#[test]
fn parses_the_real_vision_config_into_the_shared_moonvit_config() {
    let raw: LocateAnythingVisionConfig =
        serde_json::from_value(real_vision_config()).expect("real vision_config must parse");
    assert_eq!(raw.merge_kernel(), [2, 2]);

    let cfg = raw.to_moonvit_config().expect("moonvit conversion");
    assert_eq!(cfg.model_type, "moonvit");
    assert_eq!(cfg.depth, 27);
    assert_eq!(cfg.embed_dim, 1152);
    assert_eq!(cfg.hidden_size, 1152);
    assert_eq!(cfg.num_heads, 16);
    assert_eq!(cfg.intermediate_size, 4304);
    assert_eq!(cfg.patch_size, 14);
    assert_eq!(cfg.num_channels, 3);
    assert_eq!(cfg.init_pos_emb_height, 64);
    assert_eq!(cfg.init_pos_emb_width, 64);
    assert_eq!(cfg.spatial_merge_size, 2);
    // The two deltas from Kimi-VL's MoonViT.
    assert_eq!(cfg.layer_norm_eps, 1e-5);
    assert_eq!(cfg.mlp_activation, MoonViTMlpActivation::GeluTanh);
    // head_dim = 1152 / 16 = 72, which the 2D rope requires to be divisible by 4.
    assert_eq!((cfg.embed_dim / cfg.num_heads) % 4, 0);
}

#[test]
fn connector_input_dim_matches_the_checkpoint_layer_norm_width() {
    let raw: LocateAnythingVisionConfig =
        serde_json::from_value(real_vision_config()).expect("parse");
    let merge = raw.merge_kernel();
    let input_dim = raw.hidden_size * merge[0] * merge[1];
    // `multi_modal_projector.layer_norm.weight` is [4608] in the real
    // safetensors index.
    assert_eq!(input_dim, 4608);
    assert_eq!(
        connector_input_dim(raw.hidden_size, merge).expect("real config must resolve"),
        4608
    );
}

#[test]
fn an_overflowing_connector_input_width_is_an_error_rather_than_a_truncation() {
    // 1152 * 65536 * 65536 is 1152 * 2^32, whose low 32 bits are zero. A bare
    // `as i32` yielded `input_dim == 0`, which reaches the connector's
    // `reshape(image_features, &[-1, 0])` and throws inside MLX; a C++ throw
    // across the cxx boundary aborts the process instead of unwinding.
    let err = connector_input_dim(1152, [65_536, 65_536]).expect_err("must refuse");
    let err = err.to_string();
    assert!(err.contains("1152"), "error must name hidden_size: {err}");
    assert!(err.contains("65536"), "error must name the merge: {err}");

    // Saturating the multiply itself, not just the narrowing cast.
    assert!(connector_input_dim(usize::MAX, [2, 2]).is_err());
    // Just past i32::MAX, where the product is exact but unrepresentable.
    assert!(connector_input_dim(i32::MAX as usize + 1, [1, 1]).is_err());
    // A zero width is as fatal as a truncated one and is refused too.
    assert!(connector_input_dim(0, [2, 2]).is_err());

    // The largest representable width still resolves.
    assert_eq!(
        connector_input_dim(i32::MAX as usize, [1, 1]).expect("i32::MAX fits"),
        i32::MAX
    );
}

#[test]
fn rejects_a_non_moonvit_vision_tower() {
    let mut value = real_vision_config();
    value["model_type"] = serde_json::json!("siglip");
    let raw: LocateAnythingVisionConfig = serde_json::from_value(value).expect("parse");
    let err = raw.to_moonvit_config().expect_err("must refuse");
    assert!(err.contains("moonvit"), "unexpected error: {err}");
}

#[test]
fn rejects_a_non_square_merge_kernel() {
    let mut value = real_vision_config();
    value["merge_kernel_size"] = serde_json::json!([2, 4]);
    let raw: LocateAnythingVisionConfig = serde_json::from_value(value).expect("parse");
    let err = raw.to_moonvit_config().expect_err("must refuse");
    assert!(err.contains("square"), "unexpected error: {err}");
}

#[test]
fn rejects_a_patch_size_outside_the_documented_range() {
    // `patch_size` sizes both the processor's resize and the MoonViT conv
    // patch-embed, so an out-of-range value is refused here rather than clamped
    // on one side only. Uncapped it drives an unbounded resize.
    for bad in [0usize, MAX_PATCH_SIZE + 1, 100_000, usize::MAX] {
        let mut value = real_vision_config();
        value["patch_size"] = serde_json::json!(bad);
        let raw: LocateAnythingVisionConfig = serde_json::from_value(value).expect("parse");
        let err = raw
            .to_moonvit_config()
            .expect_err("patch_size must be refused");
        assert!(err.contains("patch_size"), "unexpected error: {err}");
        assert!(
            err.contains(&bad.to_string()),
            "the error must name the offending value {bad}: {err}"
        );
    }

    // The boundary itself is accepted.
    let mut value = real_vision_config();
    value["patch_size"] = serde_json::json!(MAX_PATCH_SIZE);
    let raw: LocateAnythingVisionConfig = serde_json::from_value(value).expect("parse");
    assert_eq!(
        raw.to_moonvit_config()
            .expect("the boundary is valid")
            .patch_size,
        MAX_PATCH_SIZE
    );
}

#[test]
fn rejects_a_merge_kernel_axis_outside_the_documented_range() {
    // 65536 * 65536 is exactly 2^32: square and non-zero, so it passes both the
    // `.max(1)` floor and the square-kernel check, yet narrowing it to i32
    // yields a zero divisor. The range check refuses it before either.
    for bad in [
        serde_json::json!([65_536, 65_536]),
        serde_json::json!([0, 0]),
        serde_json::json!([MAX_MERGE_KERNEL + 1, MAX_MERGE_KERNEL + 1]),
    ] {
        let mut value = real_vision_config();
        value["merge_kernel_size"] = bad.clone();
        let raw: LocateAnythingVisionConfig = serde_json::from_value(value).expect("parse");
        let err = raw
            .to_moonvit_config()
            .expect_err("merge_kernel_size must be refused");
        assert!(
            err.contains("merge_kernel_size"),
            "unexpected error for {bad}: {err}"
        );
    }

    // The boundary itself is accepted, and is still required to be square.
    let mut value = real_vision_config();
    value["merge_kernel_size"] = serde_json::json!([MAX_MERGE_KERNEL, MAX_MERGE_KERNEL]);
    let raw: LocateAnythingVisionConfig = serde_json::from_value(value).expect("parse");
    assert_eq!(
        raw.to_moonvit_config()
            .expect("the boundary is valid")
            .spatial_merge_size,
        MAX_MERGE_KERNEL
    );
}

#[test]
fn the_loader_and_processor_geometry_bounds_stay_equal() {
    // The loader rejects what the processor would otherwise clamp. If the two
    // ceilings ever drift apart, the loader would accept a geometry the
    // processor then quietly lowered, desyncing the emitted patch grid from the
    // MoonViT tower that was built from the unlowered value.
    use crate::vision::processors::locateanything as processor;
    assert_eq!(MAX_PATCH_SIZE, processor::MAX_PATCH_SIZE);
    assert_eq!(MAX_MERGE_KERNEL, processor::MAX_MERGE_KERNEL);
}

#[test]
fn an_absent_merge_kernel_axis_still_defaults_instead_of_being_rejected() {
    // Only axes the config actually declares are range-checked; a short vector
    // keeps `merge_kernel()`'s square-fill behaviour.
    let mut value = real_vision_config();
    value["merge_kernel_size"] = serde_json::json!([3]);
    let raw: LocateAnythingVisionConfig = serde_json::from_value(value).expect("parse");
    assert_eq!(raw.merge_kernel(), [3, 3]);
    let cfg = raw.to_moonvit_config().expect("a declared 3 is in range");
    assert_eq!(cfg.spatial_merge_size, 3);
}

#[test]
fn the_released_geometry_sits_inside_the_new_range_checks() {
    // Guards the ceilings themselves: lowering either constant below the
    // shipped checkpoint's geometry would reject the very model this loader
    // exists for.
    let raw: LocateAnythingVisionConfig =
        serde_json::from_value(real_vision_config()).expect("parse");
    assert!((1..=MAX_PATCH_SIZE).contains(&raw.patch_size));
    assert!(
        raw.merge_kernel_size
            .iter()
            .all(|axis| (1..=MAX_MERGE_KERNEL).contains(axis))
    );
    assert!(
        raw.to_moonvit_config().is_ok(),
        "the released config must keep parsing unchanged"
    );
}

#[test]
fn parses_the_real_text_config_with_inherited_quantization() {
    let full = real_full_config();
    let mut text_value = full.get("text_config").cloned().unwrap();
    text_value
        .as_object_mut()
        .unwrap()
        .insert("quantization".to_string(), full["quantization"].clone());

    let args: models::llama3::ModelArgs =
        serde_json::from_value(text_value).expect("real text_config must parse");
    assert_eq!(args.model_type, "qwen2");
    assert_eq!(args.hidden_size, 2048);
    assert_eq!(args.num_hidden_layers, 36);
    assert_eq!(args.num_attention_heads, 16);
    assert_eq!(args.num_key_value_heads, Some(2));
    assert!(args.tie_word_embeddings);
    assert_eq!(args.rope_theta, 1_000_000.0);
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
    // `rope_scaling: null` must not be a parse error.
    assert!(args.rope_scaling.is_none());
}

#[test]
fn already_converted_mlx_keys_pass_through_unchanged() {
    // These are the exact key spellings in the released MLX conversion's
    // `model.safetensors.index.json`.
    let mut raw = WeightMap::new();
    for key in [
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj.bias",
        "language_model.model.norm.weight",
        "multi_modal_projector.layer_norm.weight",
        "multi_modal_projector.linear_1.weight",
        "multi_modal_projector.linear_2.bias",
        "vision_tower.blocks.0.attn.wqkv.weight",
        "vision_tower.blocks.0.attn.wo.bias",
        "vision_tower.blocks.0.mlp.fc0.weight",
        "vision_tower.final_layernorm.weight",
        "vision_tower.patch_embed.pos_emb.weight",
    ] {
        raw.insert(key.to_string(), one());
    }
    let n = raw.len();

    let out = remap_locateanything_weights(raw, true);
    assert_eq!(out.len(), n, "no key should be dropped or duplicated");
    assert!(out.contains_key("vision_tower.blocks.0.attn.wqkv.weight"));
    assert!(out.contains_key("multi_modal_projector.linear_1.weight"));
    assert!(out.contains_key("language_model.model.embed_tokens.weight"));
}

#[test]
fn remaps_unconverted_upstream_prefixes() {
    let mut raw = WeightMap::new();
    raw.insert(
        "vision_model.encoder.blocks.0.wqkv.weight".to_string(),
        one(),
    );
    raw.insert("vision_model.encoder.blocks.0.wo.bias".to_string(), one());
    raw.insert("vision_model.final_layernorm.weight".to_string(), one());
    raw.insert("mlp1.0.weight".to_string(), one());
    raw.insert("mlp1.1.bias".to_string(), one());
    raw.insert("mlp1.3.weight".to_string(), one());
    raw.insert("language_model.model.norm.weight".to_string(), one());
    raw.insert(
        "language_model.model.layers.0.self_attn.rotary_emb.inv_freq".to_string(),
        one(),
    );

    let out = remap_locateanything_weights(raw, true);

    assert!(out.contains_key("vision_tower.blocks.0.attn.wqkv.weight"));
    assert!(out.contains_key("vision_tower.blocks.0.attn.wo.bias"));
    assert!(out.contains_key("vision_tower.final_layernorm.weight"));
    assert!(out.contains_key("multi_modal_projector.layer_norm.weight"));
    assert!(out.contains_key("multi_modal_projector.linear_1.bias"));
    assert!(out.contains_key("multi_modal_projector.linear_2.weight"));
    assert!(out.contains_key("language_model.model.norm.weight"));
    assert!(!out.keys().any(|k| k.contains("rotary_emb")));
}

#[test]
fn drops_the_tied_lm_head_only_when_embeddings_are_tied() {
    let mut tied = WeightMap::new();
    tied.insert("language_model.lm_head.weight".to_string(), one());
    tied.insert("language_model.model.norm.weight".to_string(), one());
    let out = remap_locateanything_weights(tied, true);
    assert!(!out.contains_key("language_model.lm_head.weight"));
    assert!(out.contains_key("language_model.model.norm.weight"));

    let mut untied = WeightMap::new();
    untied.insert("language_model.lm_head.weight".to_string(), one());
    untied.insert("language_model.model.norm.weight".to_string(), one());
    let out = remap_locateanything_weights(untied, false);
    assert!(
        out.contains_key("language_model.lm_head.weight"),
        "an untied head must survive"
    );
}

#[test]
fn transposes_a_pytorch_conv_patch_embed_weight() {
    let mut raw = WeightMap::new();
    // PyTorch layout [out, in, kH, kW].
    raw.insert(
        "vision_model.patch_embed.proj.weight".to_string(),
        mlxcel_core::from_slice_f32(&vec![0.0; 8 * 3 * 2 * 2], &[8, 3, 2, 2]),
    );
    let out = remap_locateanything_weights(raw, true);
    let w = out.get("vision_tower.patch_embed.proj.weight").unwrap();
    assert_eq!(mlxcel_core::array_shape(w), vec![8, 2, 2, 3]);
}

#[test]
fn leaves_an_already_channel_last_patch_embed_weight_alone() {
    // The released checkpoint stores [1152, 14, 14, 3] already.
    let mut raw = WeightMap::new();
    raw.insert(
        "vision_tower.patch_embed.proj.weight".to_string(),
        mlxcel_core::from_slice_f32(&vec![0.0; 8 * 2 * 2 * 3], &[8, 2, 2, 3]),
    );
    let out = remap_locateanything_weights(raw, true);
    let w = out.get("vision_tower.patch_embed.proj.weight").unwrap();
    assert_eq!(mlxcel_core::array_shape(w), vec![8, 2, 2, 3]);
}

#[test]
fn resolves_qwen2_stop_tokens_from_the_real_config() {
    let added = serde_json::json!({ "<|im_end|>": 151645, "<|endoftext|>": 151643 });
    let ids = resolve_eos_token_ids(&real_full_config(), Some(&added));
    assert_eq!(ids, vec![151645], "configured eos plus deduped <|im_end|>");
}

#[test]
fn falls_back_to_qwen2_defaults_without_an_eos_in_config() {
    let config = serde_json::json!({ "model_type": "locateanything" });
    let ids = resolve_eos_token_ids(&config, None);
    assert_eq!(ids, DEFAULT_EOS_TOKEN_IDS.to_vec());
}

#[test]
fn resolves_the_image_framing_tokens_from_added_tokens() {
    let added = serde_json::json!({
        "<IMG_CONTEXT>": 151665,
        "<img>": 151666,
        "</img>": 151667
    });
    assert_eq!(
        resolve_added_token_id(Some(&added), "<img>", DEFAULT_IMG_START_TOKEN_ID),
        151_666
    );
    assert_eq!(
        resolve_added_token_id(Some(&added), "</img>", DEFAULT_IMG_END_TOKEN_ID),
        151_667
    );
    // Missing file falls back to the released-checkpoint defaults.
    assert_eq!(
        resolve_added_token_id(None, "<img>", DEFAULT_IMG_START_TOKEN_ID),
        DEFAULT_IMG_START_TOKEN_ID
    );
}

// --- bf16 ordering vs mixed-precision densification (fix commit 5ad34371) -

const MIXED_GROUP: i32 = 64;

/// A deterministic bf16 dense plane of shape `[out, MIXED_GROUP]`, matching
/// how the released checkpoint's tensors arrive: `mlx_lm` quantizes from a
/// bf16 source, so the quantizer's scales inherit bf16 too.
fn bf16_dense_plane(out: i32, seed: i32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n = (out * MIXED_GROUP) as usize;
    let data: Vec<f32> = (0..n)
        .map(|i| (((i as i32 * 37 + seed * 11) % 97) as f32 - 48.0) * 0.01)
        .collect();
    let f32_plane = mlxcel_core::from_slice_f32(&data, &[out, MIXED_GROUP]);
    mlxcel_core::astype(&f32_plane, mlxcel_core::dtype::BFLOAT16)
}

/// Insert `{prefix}.{weight,scales,biases}` quantized at `bits` from a bf16
/// source, so `.scales` carries bf16 exactly as the released checkpoint's
/// mixed-precision layers do.
fn insert_bf16_sourced_quantized(
    weights: &mut WeightMap,
    prefix: &str,
    out: i32,
    seed: i32,
    bits: i32,
) {
    let dense = bf16_dense_plane(out, seed);
    let q = mlxcel_core::quantize_weights_with_mode(&dense, MIXED_GROUP, bits, "affine");
    let scales = mlxcel_core::quantized_weights_scales(&q);
    assert_eq!(
        mlxcel_core::array_dtype(&scales),
        mlxcel_core::dtype::BFLOAT16,
        "test setup assumption: quantizing a bf16 source keeps bf16 scales"
    );
    weights.insert(
        format!("{prefix}.weight"),
        mlxcel_core::quantized_weights_w(&q),
    );
    weights.insert(format!("{prefix}.scales"), scales);
    if mlxcel_core::quantized_weights_has_biases(&q) {
        weights.insert(
            format!("{prefix}.biases"),
            mlxcel_core::quantized_weights_biases(&q),
        );
    }
}

/// Regression for commit 5ad34371: `densify_mixed_precision_qkv` must run
/// before the bf16 -> f16 conversion pass. MLX's `dequantize` returns an
/// array carrying the scales' dtype, so a mixed-precision layer densified
/// *after* conversion would insert new bf16 planes with nothing left to
/// convert them, which was the M5 JIT crash both passes exist to avoid.
#[test]
fn densified_mixed_precision_planes_are_still_converted_to_f16() {
    let prefix = "language_model.model.layers.0.self_attn";
    let mut weights = WeightMap::new();
    insert_bf16_sourced_quantized(&mut weights, &format!("{prefix}.q_proj"), 8, 1, 8);
    insert_bf16_sourced_quantized(&mut weights, &format!("{prefix}.k_proj"), 4, 2, 4);
    insert_bf16_sourced_quantized(&mut weights, &format!("{prefix}.v_proj"), 4, 3, 4);
    // An ordinary bf16 tensor untouched by densification (the MoonViT tower),
    // so the test also proves the pass still does its normal job alongside
    // the densified planes.
    weights.insert(
        "vision_tower.blocks.0.mlp.fc0.weight".to_string(),
        bf16_dense_plane(4, 9),
    );

    let densified = reconcile_mixed_precision_weights(&mut weights, MIXED_GROUP, 4, "affine", true)
        .expect("reconcile");
    assert_eq!(
        densified, 1,
        "exactly the mixed-precision layer needed densifying"
    );

    for proj in ["q_proj", "k_proj", "v_proj"] {
        assert!(
            !weights.contains_key(&format!("{prefix}.{proj}.scales")),
            "{proj} must be dense after densification"
        );
        let w = weights.get(&format!("{prefix}.{proj}.weight")).unwrap();
        assert_eq!(
            mlxcel_core::array_dtype(w),
            mlxcel_core::dtype::FLOAT16,
            "{proj}'s densified plane must be converted to f16, not left at the bf16 \
             dtype `dequantize` gave it from the bf16 scales"
        );
    }

    let tower = weights.get("vision_tower.blocks.0.mlp.fc0.weight").unwrap();
    assert_eq!(
        mlxcel_core::array_dtype(tower),
        mlxcel_core::dtype::FLOAT16,
        "the ordinary bf16 tower tensor must still convert"
    );
}

/// With the conversion flag off (the value the real call site passes on
/// non-Apple hardware), densification still normalizes the mixed layer, but
/// nothing is converted: the planes stay at whatever dtype `dequantize`
/// produced. This is the control for the test above: it proves the f16
/// dtype there comes from the conversion pass, not from densification
/// itself.
#[test]
fn densification_runs_even_when_bf16_conversion_is_skipped() {
    let prefix = "language_model.model.layers.0.self_attn";
    let mut weights = WeightMap::new();
    insert_bf16_sourced_quantized(&mut weights, &format!("{prefix}.q_proj"), 8, 1, 8);
    insert_bf16_sourced_quantized(&mut weights, &format!("{prefix}.k_proj"), 4, 2, 4);
    insert_bf16_sourced_quantized(&mut weights, &format!("{prefix}.v_proj"), 4, 3, 4);

    let densified =
        reconcile_mixed_precision_weights(&mut weights, MIXED_GROUP, 4, "affine", false)
            .expect("reconcile");
    assert_eq!(densified, 1);

    for proj in ["q_proj", "k_proj", "v_proj"] {
        assert!(!weights.contains_key(&format!("{prefix}.{proj}.scales")));
        let w = weights.get(&format!("{prefix}.{proj}.weight")).unwrap();
        assert_eq!(
            mlxcel_core::array_dtype(w),
            mlxcel_core::dtype::BFLOAT16,
            "{proj} stays bf16 when the conversion pass is off"
        );
    }
}
