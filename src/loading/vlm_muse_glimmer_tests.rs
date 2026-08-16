use super::*;
use image::{DynamicImage, Rgb, RgbImage};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::models::muse_glimmer::{
    MuseGlimmerConfig, MuseGlimmerTextConfig, MuseGlimmerVisionConfig,
};
use crate::models::muse_glimmer_config::MuseQuantization;
use crate::models::{
    DEFAULT_IMAGE_END_TOKEN_ID, DEFAULT_IMAGE_PLACEHOLDER_TOKEN_ID, DEFAULT_IMAGE_START_TOKEN_ID,
};
use crate::vision::encoders::muse_glimmer::MUSE_GLIMMER_VISION_TOWER_ROOT;
use crate::vision::encoders::muse_glimmer_fusion::{
    MUSE_GLIMMER_VISION_ADAPTER_ROOT, MUSE_GLIMMER_VISION_PROJECTION_ROOT,
};
use crate::vision::processors::muse_glimmer::MuseGlimmerImageProcessor;
use crate::vlm_runtime::{PreparedVlmEmbeddings, VlmPreparationSummary};
use crate::{LoadedModel, VlmRuntimeRef};

fn tensor(shape: &[i32], value: f32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let len = shape.iter().product::<i32>() as usize;
    mlxcel_core::from_slice_f32(&vec![value; len], shape)
}

fn put(weights: &mut WeightMap, key: impl Into<String>, shape: &[i32], value: f32) {
    weights.insert(key.into(), tensor(shape, value));
}

#[allow(clippy::field_reassign_with_default)]
fn tiny_config() -> MuseGlimmerConfig {
    let mut vision_config = MuseGlimmerVisionConfig::default();
    vision_config.hidden_size = 4;
    vision_config.intermediate_size = 8;
    vision_config.num_attention_heads = 1;
    vision_config.patch_size = 1;
    vision_config.patch_temporal = 2;
    vision_config.merge_size = 2;
    vision_config.pos_emb_height = 2;
    vision_config.pos_emb_width = 2;
    vision_config.num_hidden_layers = 0;
    vision_config.layer_types = Vec::new();

    MuseGlimmerConfig {
        text_config: MuseGlimmerTextConfig {
            model_type: "muse_glimmer_text".to_string(),
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            post_norm_eps: 1e-8,
            vocab_size: 16,
            tie_word_embeddings: false,
            layer_types: Vec::new(),
            sliding_window: 8,
            qk_scale_factor: 1.0,
            output_multiplier: 1.0,
            final_logit_softcapping: None,
            layer_rope_theta: Vec::new(),
            rope_parameters: None,
            quantization: None,
        },
        vision_config,
        image_token_id: Some(MUSE_GLIMMER_IMAGE_TOKEN_ID),
        video_token_id: Some(MUSE_GLIMMER_VIDEO_TOKEN_ID),
        out_hidden_size: 16,
        projector_hidden_size: 8,
        projector_hidden_act: "gelu".to_string(),
    }
}

fn tiny_weights(config: &MuseGlimmerConfig) -> WeightMap {
    let mut weights = WeightMap::new();
    let text = MUSE_GLIMMER_LANGUAGE_ROOT;
    put(
        &mut weights,
        format!("{text}.embed_tokens.weight"),
        &[
            config.text_config.vocab_size as i32,
            config.text_config.hidden_size as i32,
        ],
        0.01,
    );
    put(
        &mut weights,
        format!("{text}.norm.weight"),
        &[config.text_config.hidden_size as i32],
        1.0,
    );
    put(
        &mut weights,
        format!("{MUSE_GLIMMER_LM_HEAD_ROOT}.weight"),
        &[
            config.text_config.vocab_size as i32,
            config.text_config.hidden_size as i32,
        ],
        0.01,
    );

    let vision = MUSE_GLIMMER_VISION_TOWER_ROOT;
    put(
        &mut weights,
        format!("{vision}.patch_embedder.patch_embedding.weight"),
        &[config.vision_config.hidden_size as i32, 6],
        0.02,
    );
    put(
        &mut weights,
        format!("{vision}.patch_embedder.position_embedding_table.weight"),
        &[4, config.vision_config.hidden_size as i32],
        0.0,
    );
    for norm in ["ln_pre", "ln_post"] {
        put(
            &mut weights,
            format!("{vision}.{norm}.weight"),
            &[config.vision_config.hidden_size as i32],
            1.0,
        );
        put(
            &mut weights,
            format!("{vision}.{norm}.bias"),
            &[config.vision_config.hidden_size as i32],
            0.0,
        );
    }

    put(
        &mut weights,
        format!("{MUSE_GLIMMER_VISION_ADAPTER_ROOT}.fc1.weight"),
        &[
            config.projector_hidden_size as i32,
            config.out_hidden_size as i32,
        ],
        0.03,
    );
    put(
        &mut weights,
        format!("{MUSE_GLIMMER_VISION_ADAPTER_ROOT}.fc2.weight"),
        &[
            config.projector_hidden_size as i32,
            config.projector_hidden_size as i32,
        ],
        0.02,
    );
    put(
        &mut weights,
        format!("{MUSE_GLIMMER_VISION_PROJECTION_ROOT}.weight"),
        &[
            config.text_config.hidden_size as i32,
            config.projector_hidden_size as i32,
        ],
        0.04,
    );
    weights
}

fn tiny_image() -> DynamicImage {
    let img = RgbImage::from_fn(2, 2, |x, y| {
        let base = (x + y * 2) as u8;
        Rgb([base, base.saturating_add(10), base.saturating_add(20)])
    });
    DynamicImage::ImageRgb8(img)
}

fn tiny_vlm_model(eos: Vec<i32>) -> crate::vision::MuseGlimmerVlmModel {
    let config = tiny_config();
    let weights = tiny_weights(&config);
    let processor = MuseGlimmerImageProcessor::from_vision_config(&config.vision_config);
    match build_muse_glimmer_vlm_from_weights(&weights, &config, processor, eos) {
        Ok(model) => model,
        Err(err) => panic!("synthetic Muse Glimmer VLM failed to build: {err}"),
    }
}

fn with_temp_muse_config_dir(test: impl FnOnce(&Path)) {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("mlxcel_muse_loader_{nanos}"));
    if let Err(err) = fs::create_dir_all(&dir) {
        panic!("failed to create test dir: {err}");
    }
    if let Err(err) = fs::write(
        dir.join("config.json"),
        r#"{"model_type":"muse_glimmer","architectures":["MuseGlimmerForConditionalGeneration"],"text_config":{"model_type":"muse_glimmer_text","hidden_size":8,"intermediate_size":16,"num_hidden_layers":0,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":8,"rms_norm_eps":1e-6,"vocab_size":16,"tie_word_embeddings":false,"layer_types":[]},"vision_config":{"model_type":"muse_glimmer_vision_model"}}"#,
    ) {
        panic!("failed to write config: {err}");
    }

    test(&dir);

    if let Err(err) = fs::remove_dir_all(dir) {
        panic!("failed to remove test dir: {err}");
    }
}

fn to_vec_f32(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    mlxcel_core::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn published_weight_roots_match_checkpoint_index_keys() {
    let language_key = format!("{MUSE_GLIMMER_LANGUAGE_ROOT}.layers.0.self_attn.q_proj.weight");
    let embed_key = format!("{MUSE_GLIMMER_LANGUAGE_ROOT}.embed_tokens.weight");
    let lm_head_key = format!("{MUSE_GLIMMER_LM_HEAD_ROOT}.weight");

    assert_eq!(
        language_key,
        "model.language_model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(embed_key, "model.language_model.embed_tokens.weight");
    assert_eq!(lm_head_key, "lm_head.weight");
}

#[test]
fn synthetic_muse_vlm_builder_exposes_typed_methods() {
    let model = tiny_vlm_model(vec![7, 8]);

    assert_eq!(model.image_token_id(), MUSE_GLIMMER_IMAGE_TOKEN_ID);
    assert_eq!(model.video_token_id(), MUSE_GLIMMER_VIDEO_TOKEN_ID);
    assert_eq!(model.pad_token_id(), MUSE_GLIMMER_PAD_TOKEN_ID);
    assert_eq!(LanguageModel::eos_token_ids(&model), vec![7, 8]);
    assert!(model.reject_video_inputs().is_err());
    assert_eq!(model.text_sequence_state_layout().num_layers, 0);
    model.prepare_text_sequence_state(SequenceId::from_raw(9));
    model.release_text_sequence_state(SequenceId::from_raw(9));
    model.reset_text_runtime_state();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let embeddings = match model.text_embeddings(&input_ids) {
        Ok(embeddings) => embeddings,
        Err(err) => panic!("text embeddings failed: {err}"),
    };
    assert_eq!(mlxcel_core::array_shape(&embeddings), vec![1, 2, 8]);

    let (pixel_values, grid) = model.preprocess_images(&[tiny_image()]);
    assert_eq!(grid, vec![(1, 2, 2)]);
    assert_eq!(mlxcel_core::array_shape(&pixel_values), vec![4, 6]);

    let features = match model.encode_and_fuse_images(&pixel_values, &grid) {
        Ok(features) => features,
        Err(err) => panic!("image encode+fuse failed: {err}"),
    };
    assert_eq!(mlxcel_core::array_shape(&features), vec![1, 8]);
    assert!(to_vec_f32(&features).iter().all(|value| value.is_finite()));
}

#[test]
fn muse_glimmer_loaded_model_delegates_text_and_runtime_capabilities() {
    let loaded = LoadedModel::MuseGlimmerVLM(tiny_vlm_model(vec![7, 8]));

    assert!(loaded.is_vlm());
    assert!(loaded.vision_module().is_none());
    assert!(loaded.image_token_block_info().is_none());
    match loaded.vlm_runtime() {
        Some(VlmRuntimeRef::MuseGlimmer(model)) => {
            assert_eq!(model.image_token_id(), MUSE_GLIMMER_IMAGE_TOKEN_ID);
        }
        _ => panic!("Muse Glimmer VLM runtime reference was not exposed"),
    }

    assert_eq!(LanguageModel::num_layers(&loaded), 0);
    assert_eq!(LanguageModel::eos_token_ids(&loaded), vec![7, 8]);
    assert_eq!(
        LanguageModel::output_suppressed_token_ids(&loaded),
        vec![
            MUSE_GLIMMER_IMAGE_TOKEN_ID,
            MUSE_GLIMMER_VIDEO_TOKEN_ID,
            MUSE_GLIMMER_PAD_TOKEN_ID,
        ]
    );
    assert_eq!(LanguageModel::sequence_state_layout(&loaded).num_layers, 0);
    assert!(LanguageModel::supports_batching(&loaded));
    assert!(!LanguageModel::supports_padded_prefill(&loaded));
    LanguageModel::prepare_sequence_state(&loaded, SequenceId::from_raw(11));
    LanguageModel::release_sequence_state_by_id(&loaded, SequenceId::from_raw(11));
    LanguageModel::reset_runtime_state(&loaded);

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let embeddings = match LanguageModel::embed_tokens(&loaded, &input_ids) {
        Some(embeddings) => embeddings,
        None => panic!("LoadedModel Muse Glimmer must expose token embeddings"),
    };
    assert_eq!(mlxcel_core::array_shape(&embeddings), vec![1, 2, 8]);
}

#[test]
fn muse_glimmer_text_only_request_bypasses_vision_cleanly() {
    let loaded = LoadedModel::MuseGlimmerVLM(tiny_vlm_model(vec![7, 8]));
    let mut prompt_tokens = vec![1, 2, 3];
    let prepared = match crate::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &loaded,
        &mut prompt_tokens,
        "prompt",
        &[],
        |_text, _add_special| Vec::new(),
    ) {
        Ok(prepared) => prepared,
        Err(err) => panic!("Muse Glimmer text-only preparation failed: {err}"),
    };

    assert!(prepared.is_none());
    assert_eq!(prompt_tokens, vec![1, 2, 3]);
}

#[test]
fn muse_glimmer_image_request_expands_and_scatters_shared_runtime_embeddings() {
    let loaded = LoadedModel::MuseGlimmerVLM(tiny_vlm_model(vec![7, 8]));
    let mut prompt_tokens = vec![1, DEFAULT_IMAGE_PLACEHOLDER_TOKEN_ID, 2];
    let prepared = match crate::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &loaded,
        &mut prompt_tokens,
        "prompt",
        &[tiny_image()],
        |_text, _add_special| Vec::new(),
    ) {
        Ok(Some(prepared)) => prepared,
        Ok(None) => panic!("Muse Glimmer image request unexpectedly bypassed vision"),
        Err(err) => panic!("Muse Glimmer image request preparation failed: {err}"),
    };

    assert_eq!(
        prompt_tokens,
        vec![
            1,
            DEFAULT_IMAGE_START_TOKEN_ID,
            MUSE_GLIMMER_IMAGE_TOKEN_ID,
            DEFAULT_IMAGE_END_TOKEN_ID,
            2,
        ]
    );
    assert_muse_summary(prepared, 1, 1, 5);
}

#[test]
fn muse_glimmer_request_rejects_placeholder_image_mismatch() {
    let loaded = LoadedModel::MuseGlimmerVLM(tiny_vlm_model(vec![7, 8]));
    let mut prompt_tokens = vec![1, 2];
    let err = match crate::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &loaded,
        &mut prompt_tokens,
        "prompt",
        &[tiny_image()],
        |_text, _add_special| Vec::new(),
    ) {
        Ok(_) => panic!("Muse Glimmer accepted image input without a placeholder"),
        Err(err) => err.to_string(),
    };

    assert!(err.contains("0 image placeholders"));
    assert!(err.contains("1 images were processed"));
}

#[test]
fn muse_glimmer_request_rejects_reserved_video_token() {
    let loaded = LoadedModel::MuseGlimmerVLM(tiny_vlm_model(vec![7, 8]));
    let mut prompt_tokens = vec![1, MUSE_GLIMMER_VIDEO_TOKEN_ID, 2];
    let err = match crate::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &loaded,
        &mut prompt_tokens,
        "prompt",
        &[],
        |_text, _add_special| Vec::new(),
    ) {
        Ok(_) => panic!("Muse Glimmer accepted a reserved video token"),
        Err(err) => err.to_string(),
    };

    assert!(err.contains("does not support video inputs"));
}

fn assert_muse_summary(
    prepared: PreparedVlmEmbeddings,
    image_blocks: usize,
    image_tokens: usize,
    total_tokens: usize,
) {
    assert_eq!(
        mlxcel_core::array_shape(&prepared.embeddings.inputs_embeds),
        vec![1, total_tokens as i32, 8]
    );
    assert!(prepared.embeddings.attention_mask_4d.is_none());
    assert_eq!(
        prepared.preparation,
        Some(VlmPreparationSummary::MuseGlimmer {
            image_blocks,
            image_tokens,
            total_tokens,
        })
    );
}

#[test]
fn muse_glimmer_xla_image_preprocessor_rejects_family_explicitly() {
    with_temp_muse_config_dir(|dir| {
        let err = match crate::load_xla_image_preprocessor(dir) {
            Ok(_) => panic!("Muse Glimmer XLA image preprocessing must be rejected"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("Muse Glimmer VLM"));
        assert!(err.contains("OpenXLA image execution"));
    });
}

#[test]
fn muse_quantization_contract_accepts_text_sidecars_and_rejects_unsafe_layouts() {
    let mut config = tiny_config();
    config.text_config.quantization = Some(MuseQuantization {
        group_size: 64,
        bits: 4,
    });
    let mut weights = tiny_weights(&config);
    put(&mut weights, "lm_head.scales", &[16, 1], 0.1);
    put(&mut weights, "lm_head.biases", &[16, 1], 0.0);
    put(
        &mut weights,
        "model.vision_adapter.fc1.scales",
        &[8, 1],
        0.1,
    );
    put(
        &mut weights,
        "model.vision_adapter.fc1.biases",
        &[8, 1],
        0.0,
    );
    assert!(ensure_supported_muse_weight_map(&weights, &config).is_ok());

    let dense_config = tiny_config();
    let err = ensure_supported_muse_weight_map(&weights, &dense_config)
        .map(|_| "unexpected success".to_string())
        .unwrap_or_else(|err| err.to_string());
    assert!(err.contains("declares no quantization contract"));

    put(
        &mut weights,
        "model.vision_tower.patch_embedder.patch_embedding.scales",
        &[4, 1],
        0.1,
    );
    let err = ensure_supported_muse_weight_map(&weights, &config)
        .map(|_| "unexpected success".to_string())
        .unwrap_or_else(|err| err.to_string());
    assert!(err.contains("vision-tower sidecar"));
}

#[test]
fn builder_rejects_video_temporal_layouts() {
    let weights = tiny_weights(&tiny_config());
    let mut config = tiny_config();
    config.vision_config.patch_temporal = 4;
    let processor = MuseGlimmerImageProcessor::from_vision_config(&config.vision_config);
    let err = build_muse_glimmer_vlm_from_weights(&weights, &config, processor, Vec::new())
        .map(|_| "unexpected success".to_string())
        .unwrap_or_else(|err| err.to_string());
    assert!(err.contains("video temporal layouts"));
}

#[test]
fn pinned_weight_index_classifies_each_source_weight_once() {
    let model_dir = Path::new("models/mlx/muse-glimmer-30b");
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        crate::test_support::pinned_checkpoint::skip_or_fail_pinned_checkpoint(
            "pinned_weight_index_classifies_each_source_weight_once",
            &format!(
                "pinned Muse Glimmer checkpoint index not present at {}",
                index_path.display()
            ),
        );
        return;
    }

    let inventory = match read_muse_weight_inventory_from_index(model_dir) {
        Ok(inventory) => inventory,
        Err(err) => panic!("Muse Glimmer pinned index classification failed: {err}"),
    };
    assert_eq!(
        inventory,
        MuseWeightInventory {
            language_model: 626,
            vision_tower: 806,
            vision_adapter: 2,
            vision_projection: 1,
            lm_head: 1,
            total: 1436,
        }
    );
}

#[test]
fn muse_weight_classifier_normalizes_mlx_vlm_roots_and_rejects_unknowns() {
    assert_eq!(
        classify_muse_weight_key("model.vision_tower.layers.0.attn.q_proj.weight"),
        Some(MuseWeightRoot::VisionTower)
    );
    assert_eq!(
        classify_muse_weight_key("model.vision_adapter.fc1.weight"),
        Some(MuseWeightRoot::VisionAdapter)
    );
    assert_eq!(
        classify_muse_weight_key("model.vision_projection.weight"),
        Some(MuseWeightRoot::VisionProjection)
    );
    assert_eq!(
        classify_muse_weight_key("model.language_model.embed_tokens.weight"),
        Some(MuseWeightRoot::LanguageModel)
    );
    assert_eq!(
        classify_muse_weight_key("lm_head.weight"),
        Some(MuseWeightRoot::LmHead)
    );
    assert_eq!(classify_muse_weight_key("model.mm_projector.weight"), None);
    assert_eq!(
        normalize_muse_weight_key("language_model.model.layers.0.mlp.up_proj.scales"),
        "model.language_model.layers.0.mlp.up_proj.scales"
    );
    assert_eq!(
        normalize_muse_weight_key("language_model.lm_head.biases"),
        "lm_head.biases"
    );
    assert_eq!(
        normalize_muse_weight_key("vision_tower.layers.0.attn.q_proj.weight"),
        "model.vision_tower.layers.0.attn.q_proj.weight"
    );
    assert_eq!(
        classify_muse_weight_key("language_model.model.embed_tokens.scales"),
        Some(MuseWeightRoot::LanguageModel)
    );
    assert_eq!(
        classify_muse_weight_key("language_model.lm_head.weight"),
        Some(MuseWeightRoot::LmHead)
    );
}

#[test]
fn root_quantization_is_inherited_into_muse_text_config() {
    let quantization = serde_json::json!({
        "group_size": 64,
        "bits": 4,
        "mode": "affine"
    });
    let mut config = serde_json::json!({
        "quantization": quantization,
        "quantization_config": quantization,
        "text_config": {}
    });
    if let Err(err) = inherit_muse_text_quantization(&mut config) {
        panic!("root Muse quantization should be inherited: {err}");
    }
    assert_eq!(
        config["text_config"]["quantization"],
        serde_json::json!({"group_size": 64, "bits": 4, "mode": "affine"})
    );

    config["quantization_config"]["bits"] = serde_json::json!(8);
    let err = inherit_muse_text_quantization(&mut config)
        .map(|_| "unexpected success".to_string())
        .unwrap_or_else(|err| err);
    assert!(err.contains("disagree"));

    let mut unsupported_mode = serde_json::json!({
        "quantization": {"group_size": 64, "bits": 4, "mode": "mxfp4"},
        "text_config": {}
    });
    let err = inherit_muse_text_quantization(&mut unsupported_mode)
        .map(|_| "unexpected success".to_string())
        .unwrap_or_else(|err| err);
    assert!(err.contains("must be \"affine\""));

    let mut non_string_mode = serde_json::json!({
        "quantization": {"group_size": 64, "bits": 4, "mode": 4},
        "text_config": {}
    });
    let err = inherit_muse_text_quantization(&mut non_string_mode)
        .map(|_| "unexpected success".to_string())
        .unwrap_or_else(|err| err);
    assert!(err.contains("must be a string"));
}

#[test]
fn muse_affine_quantization_rejects_missing_biases_and_global_scale() {
    let mut config = tiny_config();
    config.text_config.quantization = Some(MuseQuantization {
        group_size: 64,
        bits: 4,
    });
    let mut weights = tiny_weights(&config);
    put(&mut weights, "lm_head.scales", &[16, 1], 0.1);

    let err = ensure_supported_muse_weight_map(&weights, &config)
        .map(|_| "unexpected success".to_string())
        .unwrap_or_else(|err| err.to_string());
    assert!(err.contains("no matching lm_head.biases"));

    put(&mut weights, "lm_head.biases", &[16, 1], 0.0);
    put(&mut weights, "lm_head.global_scale", &[1], 1.0);
    let err = ensure_supported_muse_weight_map(&weights, &config)
        .map(|_| "unexpected success".to_string())
        .unwrap_or_else(|err| err.to_string());
    assert!(err.contains("does not support global-scale sidecar"));
}
