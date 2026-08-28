use super::*;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::weights::WeightMap;
use std::io::Cursor;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use crate::LoadedModel;
use crate::models::muse_glimmer_config::{DEFAULT_PAD_TOKEN_ID, DEFAULT_VIDEO_TOKEN_ID};
use crate::models::{
    DEFAULT_IMAGE_TOKEN_ID, MuseGlimmerConfig, MuseGlimmerTextConfig, MuseGlimmerTextModel,
    MuseGlimmerTextWrapper, MuseGlimmerVisionConfig,
};
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy, ReasoningBudgetOverride};
use crate::server::model_provider::GenerateEvent;
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;
use crate::vision::encoders::muse_glimmer::{
    MUSE_GLIMMER_VISION_TOWER_ROOT, MuseGlimmerVisionTower,
};
use crate::vision::encoders::muse_glimmer_fusion::{
    MUSE_GLIMMER_VISION_ADAPTER_ROOT, MUSE_GLIMMER_VISION_PROJECTION_ROOT, MuseGlimmerVisionFusion,
};
use crate::vision::muse_glimmer_vlm::MuseGlimmerVlmModel;
use crate::vision::processors::muse_glimmer::MuseGlimmerImageProcessor;

const LANGUAGE_ROOT: &str = "model.language_model";
const LM_HEAD_ROOT: &str = "lm_head";
pub(super) const EOS_ID: i32 = 200_099;
pub(super) const GENERATED_TOKEN_ID: i32 = 12;
const VOCAB_SIZE: usize = 200_100;

pub(super) struct StreamSummary {
    pub tokens: Vec<String>,
    pub result: crate::server::GenerationResult,
}

fn tensor(shape: &[i32], value: f32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let len = shape.iter().product::<i32>() as usize;
    mlxcel_core::from_slice_f32(&vec![value; len], shape)
}

fn put(weights: &mut WeightMap, key: impl Into<String>, shape: &[i32], value: f32) {
    weights.insert(key.into(), tensor(shape, value));
}

fn lm_head_weight(hidden: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut values = vec![0.0f32; VOCAB_SIZE * hidden];
    let row_start = GENERATED_TOKEN_ID as usize * hidden;
    for value in &mut values[row_start..row_start + hidden] {
        *value = 1.0;
    }
    mlxcel_core::from_slice_f32(&values, &[VOCAB_SIZE as i32, hidden as i32])
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
            num_hidden_layers: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            post_norm_eps: 1e-8,
            vocab_size: VOCAB_SIZE,
            tie_word_embeddings: false,
            layer_types: vec![
                "sliding_attention".to_string(),
                "full_attention".to_string(),
            ],
            sliding_window: 8,
            qk_scale_factor: 1.0,
            output_multiplier: 1.0,
            final_logit_softcapping: None,
            layer_rope_theta: vec![Some(500_000.0), None],
            rope_parameters: None,
            quantization: None,
        },
        vision_config,
        image_token_id: Some(DEFAULT_IMAGE_TOKEN_ID),
        video_token_id: Some(DEFAULT_VIDEO_TOKEN_ID),
        out_hidden_size: 16,
        projector_hidden_size: 8,
        projector_hidden_act: "gelu".to_string(),
    }
}

fn tiny_weights(config: &MuseGlimmerConfig) -> WeightMap {
    let mut weights = WeightMap::new();
    put(
        &mut weights,
        format!("{LANGUAGE_ROOT}.embed_tokens.weight"),
        &[
            config.text_config.vocab_size as i32,
            config.text_config.hidden_size as i32,
        ],
        0.01,
    );
    put(
        &mut weights,
        format!("{LANGUAGE_ROOT}.norm.weight"),
        &[config.text_config.hidden_size as i32],
        1.0,
    );
    weights.insert(
        format!("{LM_HEAD_ROOT}.weight"),
        lm_head_weight(config.text_config.hidden_size),
    );

    for layer in 0..config.text_config.num_hidden_layers {
        let prefix = format!("{LANGUAGE_ROOT}.layers.{layer}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            put(
                &mut weights,
                format!("{prefix}.{norm}.weight"),
                &[config.text_config.hidden_size as i32],
                1.0,
            );
        }
        for (proj, rows) in [
            ("q_proj", 8),
            ("gate_proj", 8),
            ("o_proj", 8),
            ("k_proj", 8),
            ("v_proj", 8),
        ] {
            put(
                &mut weights,
                format!("{prefix}.self_attn.{proj}.weight"),
                &[rows, config.text_config.hidden_size as i32],
                0.01,
            );
        }
        put(
            &mut weights,
            format!("{prefix}.mlp.gate_proj.weight"),
            &[16, 8],
            0.01,
        );
        put(
            &mut weights,
            format!("{prefix}.mlp.up_proj.weight"),
            &[16, 8],
            0.01,
        );
        put(
            &mut weights,
            format!("{prefix}.mlp.down_proj.weight"),
            &[8, 16],
            0.01,
        );
    }

    put(
        &mut weights,
        format!("{MUSE_GLIMMER_VISION_TOWER_ROOT}.patch_embedder.patch_embedding.weight"),
        &[config.vision_config.hidden_size as i32, 6],
        0.02,
    );
    put(
        &mut weights,
        format!("{MUSE_GLIMMER_VISION_TOWER_ROOT}.patch_embedder.position_embedding_table.weight"),
        &[4, config.vision_config.hidden_size as i32],
        0.0,
    );
    for norm in ["ln_pre", "ln_post"] {
        put(
            &mut weights,
            format!("{MUSE_GLIMMER_VISION_TOWER_ROOT}.{norm}.weight"),
            &[config.vision_config.hidden_size as i32],
            1.0,
        );
        put(
            &mut weights,
            format!("{MUSE_GLIMMER_VISION_TOWER_ROOT}.{norm}.bias"),
            &[config.vision_config.hidden_size as i32],
            0.0,
        );
    }
    put(
        &mut weights,
        format!("{MUSE_GLIMMER_VISION_ADAPTER_ROOT}.fc1.weight"),
        &[8, config.out_hidden_size as i32],
        0.03,
    );
    put(
        &mut weights,
        format!("{MUSE_GLIMMER_VISION_ADAPTER_ROOT}.fc2.weight"),
        &[8, 8],
        0.02,
    );
    put(
        &mut weights,
        format!("{MUSE_GLIMMER_VISION_PROJECTION_ROOT}.weight"),
        &[config.text_config.hidden_size as i32, 8],
        0.04,
    );
    weights
}

fn tiny_muse_model() -> MuseGlimmerVlmModel {
    let config = tiny_config();
    let weights = tiny_weights(&config);
    let text_model = match MuseGlimmerTextModel::from_weights(
        &weights,
        &config.text_config,
        LANGUAGE_ROOT,
        LM_HEAD_ROOT,
        vec![EOS_ID],
        vec![
            DEFAULT_IMAGE_TOKEN_ID,
            DEFAULT_VIDEO_TOKEN_ID,
            DEFAULT_PAD_TOKEN_ID,
        ],
    ) {
        Ok(model) => model,
        Err(err) => panic!("synthetic Muse text model failed to build: {err}"),
    };
    let vision_tower = match MuseGlimmerVisionTower::from_weights(&weights, &config.vision_config) {
        Ok(model) => model,
        Err(err) => panic!("synthetic Muse vision tower failed to build: {err}"),
    };
    let vision_fusion = match MuseGlimmerVisionFusion::from_weights(&weights, &config) {
        Ok(model) => model,
        Err(err) => panic!("synthetic Muse vision fusion failed to build: {err}"),
    };
    let processor = MuseGlimmerImageProcessor::from_vision_config(&config.vision_config);
    match MuseGlimmerVlmModel::new(
        MuseGlimmerTextWrapper::new(text_model),
        vision_tower,
        vision_fusion,
        processor,
        &config,
    ) {
        Ok(model) => model,
        Err(err) => panic!("synthetic Muse VLM failed to build: {err}"),
    }
}

pub(super) fn scheduler(parallelism: usize) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        LoadedModel::MuseGlimmerVLM(tiny_muse_model()),
        muse_tokenizer(),
        vec![EOS_ID],
        rx,
        parallelism,
        8,
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        0,
        false,
        PreemptionPolicy::default(),
        parallelism,
        DecodeStorageBackend::Dense,
    );
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    sched
}

pub(super) fn options(max_tokens: usize) -> ServerGenerateOptions {
    ServerGenerateOptions {
        max_tokens,
        sampling: SamplingConfig::greedy(),
        stop_sequences: None,
        ignore_eos: false,
        priority: RequestPriority::Normal,
        logprobs: Default::default(),
        reasoning_budget: ReasoningBudgetOverride::InheritServerDefault,
        thinking_enter_block_on_start: false,
        prompt_cache_ctx: None,
        structured: None,
        image_soft_tokens: None,
    }
}

pub(super) fn enqueue(
    sched: &mut BatchScheduler,
    prompt_tokens: Vec<i32>,
    image: Vec<u8>,
    max_tokens: usize,
) -> mpsc::Receiver<GenerateEvent> {
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt_tokens),
        options(max_tokens),
        vec![image],
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
    );
    rx
}

pub(super) fn image_bytes(width: u32, height: u32, seed: u8) -> Vec<u8> {
    let image = DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
        let base = seed.wrapping_add((x + y * width) as u8);
        Rgb([base, base.wrapping_add(31), base.wrapping_add(67)])
    }));
    let mut cursor = Cursor::new(Vec::new());
    if let Err(err) = image.write_to(&mut cursor, ImageFormat::Png) {
        panic!("synthetic Muse image failed to encode: {err}");
    }
    cursor.into_inner()
}

pub(super) fn collect_stream(rx: &mpsc::Receiver<GenerateEvent>) -> StreamSummary {
    let mut tokens = Vec::new();
    loop {
        match rx.recv_timeout(Duration::from_secs(2)) {
            Ok(GenerateEvent::Token(text)) => tokens.push(text),
            Ok(GenerateEvent::TokenWithLogprobs(text, _)) => tokens.push(text),
            Ok(GenerateEvent::Prefill(_)) => {}
            Ok(GenerateEvent::Done(result)) => return StreamSummary { tokens, result },
            Ok(GenerateEvent::Error(error)) => panic!("unexpected generation error: {error}"),
            Err(err) => panic!("generation did not finish: {err}"),
        }
    }
}

pub(super) fn collect_result_only(
    rx: &mpsc::Receiver<GenerateEvent>,
) -> crate::server::GenerationResult {
    collect_stream(rx).result
}

pub(super) fn muse_state_offsets(
    sched: &BatchScheduler,
    seq_id: SequenceId,
) -> Option<Vec<(bool, i32, i32)>> {
    match &sched.model {
        LoadedModel::MuseGlimmerVLM(model) => model.text.sequence_cache_summaries(seq_id),
        _ => panic!("scheduler test must use Muse Glimmer VLM"),
    }
}

fn muse_tokenizer() -> MlxcelTokenizer {
    let json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 12, "content": "<0x61>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
            {"id": 200080, "content": "<|image_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 200081, "content": "<|image_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 200090, "content": "<|image|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 200091, "content": "<|video|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 200092, "content": "<|patch|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 200099, "content": "<|eos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": {"type": "ByteFallback"},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": true,
            "vocab": {
                "Hello": 2,
                "World": 3,
                "<0x61>": 12,
                "<|image_start|>": 200080,
                "<|image_end|>": 200081,
                "<|image|>": 200090,
                "<|video|>": 200091,
                "<|patch|>": 200092,
                "<|eos|>": 200099
            },
            "merges": []
        }
    }"#;
    match tokenizers::Tokenizer::from_bytes(json.as_bytes()) {
        Ok(tokenizer) => MlxcelTokenizer::HuggingFace(tokenizer),
        Err(err) => panic!("failed to build Muse scheduler test tokenizer: {err}"),
    }
}
