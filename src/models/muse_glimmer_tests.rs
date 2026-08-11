use super::*;
use mlxcel_core::weights::WeightMap;

fn test_config() -> MuseGlimmerTextConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "muse_glimmer_text",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "vocab_size": 8,
        "tie_word_embeddings": false,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 4,
        "qk_scale_factor": 3.87,
        "output_multiplier": 0.19611613513818404,
        "final_logit_softcapping": 20.0,
        "layer_rope_theta": [500000.0, null]
    }))
    .unwrap()
}

fn full_config_value() -> serde_json::Value {
    let vision_layers = (0..50)
        .map(|idx| MuseGlimmerVisionConfig::expected_layer_type(idx, 50))
        .collect::<Vec<_>>();
    serde_json::json!({
        "model_type": "muse_glimmer",
        "image_token_id": 200092,
        "video_token_id": 200091,
        "out_hidden_size": 6144,
        "projector_hidden_size": 4096,
        "projector_hidden_act": "gelu",
        "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "vocab_size": 8,
            "tie_word_embeddings": false,
            "layer_types": ["sliding_attention", "full_attention"]
        },
        "vision_config": {
            "model_type": "muse_glimmer_vision",
            "hidden_act": "gelu",
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "layer_norm_eps": 1e-5,
            "layer_types": vision_layers,
            "max_position_embeddings": 1024,
            "merge_size": 2,
            "num_attention_heads": 16,
            "num_hidden_layers": 50,
            "patch_size": 14,
            "patch_temporal": 2,
            "pos_emb_height": 32,
            "pos_emb_width": 32,
            "rope_parameters": {"rope_theta": 10000.0}
        }
    })
}

fn tensor(shape: &[i32], value: f32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let len = shape.iter().product::<i32>() as usize;
    mlxcel_core::from_slice_f32(&vec![value; len], shape)
}

fn add_weight(weights: &mut WeightMap, key: impl Into<String>, shape: &[i32], value: f32) {
    weights.insert(key.into(), tensor(shape, value));
}

fn to_vec_f32(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    mlxcel_core::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

fn synthetic_weights(config: &MuseGlimmerTextConfig) -> WeightMap {
    let mut weights = WeightMap::new();
    let root = "model.language_model";
    add_weight(
        &mut weights,
        format!("{root}.embed_tokens.weight"),
        &[config.vocab_size as i32, config.hidden_size as i32],
        0.01,
    );
    add_weight(
        &mut weights,
        "lm_head.weight",
        &[config.vocab_size as i32, config.hidden_size as i32],
        0.01,
    );
    add_weight(
        &mut weights,
        format!("{root}.norm.weight"),
        &[config.hidden_size as i32],
        1.0,
    );

    for layer in 0..config.num_hidden_layers {
        let prefix = format!("{root}.layers.{layer}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            add_weight(
                &mut weights,
                format!("{prefix}.{norm}.weight"),
                &[config.hidden_size as i32],
                1.0,
            );
        }
        for (proj, out_dim) in [
            ("q_proj", config.num_attention_heads * config.head_dim),
            ("gate_proj", config.num_attention_heads * config.head_dim),
            ("o_proj", config.hidden_size),
            ("k_proj", config.num_key_value_heads * config.head_dim),
            ("v_proj", config.num_key_value_heads * config.head_dim),
        ] {
            add_weight(
                &mut weights,
                format!("{prefix}.self_attn.{proj}.weight"),
                &[out_dim as i32, config.hidden_size as i32],
                0.01,
            );
        }
        add_weight(
            &mut weights,
            format!("{prefix}.mlp.gate_proj.weight"),
            &[config.intermediate_size as i32, config.hidden_size as i32],
            0.01,
        );
        add_weight(
            &mut weights,
            format!("{prefix}.mlp.up_proj.weight"),
            &[config.intermediate_size as i32, config.hidden_size as i32],
            0.01,
        );
        add_weight(
            &mut weights,
            format!("{prefix}.mlp.down_proj.weight"),
            &[config.hidden_size as i32, config.intermediate_size as i32],
            0.01,
        );
    }
    weights
}

#[test]
fn full_config_parses_pinned_vision_and_projector_fields() {
    let config: MuseGlimmerConfig = serde_json::from_value(full_config_value()).unwrap();
    config.validate().unwrap();
    assert_eq!(config.image_token_id, Some(DEFAULT_IMAGE_TOKEN_ID));
    assert_eq!(config.video_token_id, Some(DEFAULT_VIDEO_TOKEN_ID));
    assert_eq!(config.out_hidden_size, 6144);
    assert_eq!(config.projector_hidden_size, 4096);
    assert_eq!(config.projector_hidden_act, "gelu");

    let vision = &config.vision_config;
    assert_eq!(vision.hidden_size, 1536);
    assert_eq!(vision.intermediate_size, 8960);
    assert_eq!(vision.num_hidden_layers, 50);
    assert_eq!(vision.num_attention_heads, 16);
    assert_eq!(vision.patch_size, 14);
    assert_eq!(vision.patch_temporal, 2);
    assert_eq!(vision.merge_size, 2);
    assert_eq!(vision.pos_emb_height, 32);
    assert_eq!(vision.pos_emb_width, 32);
    assert_eq!(vision.max_position_embeddings, 1024);
    assert_eq!(vision.rope_theta(), 10000.0);
    assert!(vision.is_window_layer(0));
    assert!(!vision.is_window_layer(3));
    assert!(vision.is_window_layer(48));
    assert!(!vision.is_window_layer(49));
}

#[test]
fn vision_config_rejects_non_pinned_schedule_or_missing_final_full_layer() {
    let mut config: MuseGlimmerConfig = serde_json::from_value(full_config_value()).unwrap();
    config.vision_config.layer_types[0] = "full_attention".to_string();
    let err = config.validate().unwrap_err();
    assert!(err.contains("layer 0"));

    let mut config: MuseGlimmerConfig = serde_json::from_value(full_config_value()).unwrap();
    config.vision_config.layer_types[49] = "window_attention".to_string();
    let err = config.validate().unwrap_err();
    assert!(err.contains("layer 49"));
}

#[test]
fn config_validates_layer_pattern_and_rope_selection() {
    let config = test_config();
    config.validate().unwrap();
    assert!(config.is_sliding_layer(0));
    assert!(!config.is_sliding_layer(1));
    assert_eq!(config.rope_theta_for_layer(0), Some(500000.0));
    assert_eq!(config.rope_theta_for_layer(1), None);
}

#[test]
fn config_rejects_unknown_layer_type() {
    let mut config = test_config();
    config.layer_types[1] = "linear_attention".to_string();
    let err = config.validate().unwrap_err();
    assert!(err.contains("unsupported layer_type"));
}

#[test]
fn text_model_builds_from_published_vlm_roots_and_uses_mixed_caches() {
    let config = test_config();
    let weights = synthetic_weights(&config);
    let model = MuseGlimmerTextModel::from_weights(
        &weights,
        &config,
        "model.language_model",
        "lm_head",
        vec![200001, 200008],
        vec![200092, 200091, 200018],
    )
    .unwrap();

    let caches = model.make_muse_caches();
    assert!(matches!(caches[0], MuseCache::Rotating(_)));
    assert!(matches!(caches[1], MuseCache::Standard(_)));
}

#[test]
fn token_embeddings_apply_the_reference_weightless_rms_norm() {
    let config = test_config();
    let weights = synthetic_weights(&config);
    let model = MuseGlimmerTextModel::from_weights(
        &weights,
        &config,
        "model.language_model",
        "lm_head",
        vec![200001, 200008],
        vec![200092, 200091, 200018],
    )
    .unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1], &[1, 1]);
    let embeddings = model.get_embed_tokens(&input_ids);
    let values = to_vec_f32(&embeddings);
    assert_eq!(values.len(), config.hidden_size);
    assert!(
        values.iter().all(|value| (0.95..=1.05).contains(value)),
        "weightless embedding norm was not applied: {values:?}"
    );
}

#[test]
fn reduced_decoder_forward_runs_and_softcaps_logits() {
    let config = test_config();
    let weights = synthetic_weights(&config);
    let model = MuseGlimmerTextModel::from_weights(
        &weights,
        &config,
        "model.language_model",
        "lm_head",
        vec![200001, 200008],
        vec![200092, 200091, 200018],
    )
    .unwrap();
    let mut caches = model.make_muse_caches();
    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let logits = model.forward_with_muse_caches(&input_ids, None, &mut caches, None);
    mlxcel_core::eval(&logits);
    assert_eq!(mlxcel_core::array_shape(&logits), vec![1, 2, 8]);

    let raw = mlxcel_core::from_slice_f32(&[0.0, 100.0, -100.0], &[1, 3]);
    let capped = MuseGlimmerTextModel::softcap_logits(&raw, 0.2, Some(20.0));
    mlxcel_core::eval(&capped);
    let values = to_vec_f32(&capped);
    assert!(values[1] < 20.0);
    assert!(values[2] > -20.0);
}

#[path = "muse_glimmer_batch_tests.rs"]
mod batch_tests;
