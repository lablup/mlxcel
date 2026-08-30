use super::*;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use mlxcel_core::cache::{SequenceId, SequenceStateBackend};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::weights::WeightMap;
use std::io::Cursor;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use crate::LoadedModel;
use crate::models::muse_glimmer_config::{DEFAULT_PAD_TOKEN_ID, DEFAULT_VIDEO_TOKEN_ID};
use crate::models::{
    DEFAULT_IMAGE_END_TOKEN_ID, DEFAULT_IMAGE_PLACEHOLDER_TOKEN_ID, DEFAULT_IMAGE_START_TOKEN_ID,
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
const EOS_ID: i32 = 200_099;
const VOCAB_SIZE: usize = 200_100;

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
    put(
        &mut weights,
        format!("{LM_HEAD_ROOT}.weight"),
        &[
            config.text_config.vocab_size as i32,
            config.text_config.hidden_size as i32,
        ],
        0.01,
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

fn scheduler() -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        LoadedModel::MuseGlimmerVLM(tiny_muse_model()),
        MlxcelTokenizer::stub(),
        vec![EOS_ID],
        rx,
        1,
        8,
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        0,
        false,
        PreemptionPolicy::default(),
        1,
        DecodeStorageBackend::Dense,
    );
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    sched
}

fn options(max_tokens: usize) -> ServerGenerateOptions {
    ServerGenerateOptions {
        max_tokens,
        sampling: SamplingConfig::greedy(),
        stop_sequences: None,
        ignore_eos: false,
        priority: RequestPriority::Normal,
        logprobs: Default::default(),
        reasoning_budget: ReasoningBudgetOverride::InheritServerDefault,
        thinking_enter_block_on_start: false,
        reasoning_control: None,
        prompt_cache_ctx: None,
        structured: None,
        image_soft_tokens: None,
    }
}

fn enqueue(
    sched: &mut BatchScheduler,
    prompt_tokens: Vec<i32>,
    images: Vec<Vec<u8>>,
) -> mpsc::Receiver<GenerateEvent> {
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt_tokens),
        options(1),
        images,
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    rx
}

fn tiny_image() -> DynamicImage {
    let img = RgbImage::from_fn(2, 2, |x, y| {
        let base = (x + y * 2) as u8;
        Rgb([base, base.saturating_add(10), base.saturating_add(20)])
    });
    DynamicImage::ImageRgb8(img)
}

fn tiny_image_bytes() -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    if let Err(err) = tiny_image().write_to(&mut cursor, ImageFormat::Png) {
        panic!("synthetic Muse image failed to encode: {err}");
    }
    cursor.into_inner()
}

fn receive_done(rx: &mpsc::Receiver<GenerateEvent>) -> crate::server::GenerationResult {
    loop {
        match rx.recv_timeout(Duration::from_secs(2)) {
            Ok(GenerateEvent::Token(_))
            | Ok(GenerateEvent::TokenWithLogprobs(_, _))
            | Ok(GenerateEvent::Prefill(_)) => {}
            Ok(GenerateEvent::Done(result)) => return result,
            Ok(GenerateEvent::Error(error)) => panic!("unexpected generation error: {error}"),
            Err(err) => panic!("generation did not finish: {err}"),
        }
    }
}

fn receive_error(rx: &mpsc::Receiver<GenerateEvent>) -> String {
    match rx.recv_timeout(Duration::from_secs(2)) {
        Ok(GenerateEvent::Error(error)) => error,
        Ok(GenerateEvent::Token(text)) => panic!("unexpected token event: {text:?}"),
        Ok(GenerateEvent::TokenWithLogprobs(text, _)) => {
            panic!("unexpected token+logprobs event: {text:?}")
        }
        Ok(GenerateEvent::Done(_)) => panic!("unexpected done event"),
        Ok(GenerateEvent::Prefill(_)) => panic!("unexpected prefill event"),
        Err(err) => panic!("generation did not reject: {err}"),
    }
}

fn muse_state_offsets(sched: &BatchScheduler, seq_id: SequenceId) -> Option<Vec<(bool, i32, i32)>> {
    match &sched.model {
        LoadedModel::MuseGlimmerVLM(model) => model.text.sequence_cache_summaries(seq_id),
        _ => panic!("scheduler test must use Muse Glimmer VLM"),
    }
}

#[test]
fn muse_scheduler_admits_text_only_with_model_owned_sequence_state() {
    let mut sched = scheduler();
    assert_eq!(
        sched.model.sequence_state_layout().backend,
        SequenceStateBackend::ModelOwned
    );

    let rx = enqueue(&mut sched, vec![1, 2, 3], Vec::new());
    let seq_id = SequenceId::from_raw(0);
    assert_eq!(sched.prefill_queue.len(), 1);
    assert_eq!(
        muse_state_offsets(&sched, seq_id),
        Some(vec![(true, 0, 0), (false, 0, 0)])
    );

    match sched.decide_action() {
        BatchSchedulerAction::Prefill(id) => sched.execute_prefill(id),
        other => panic!("expected prefill action for Muse text request, got {other:?}"),
    }
    let result = receive_done(&rx);
    assert_eq!(result.prompt_tokens, 3);
    assert_eq!(sched.prefill_queue.len(), 0);
    assert!(sched.active_batch.is_empty());
    assert_eq!(muse_state_offsets(&sched, seq_id), None);
}

#[test]
fn muse_scheduler_admits_single_image_with_expanded_usage_and_embeddings() {
    let mut sched = scheduler();
    let rx = enqueue(
        &mut sched,
        vec![10, DEFAULT_IMAGE_PLACEHOLDER_TOKEN_ID, 11],
        vec![tiny_image_bytes()],
    );
    let seq_id = SequenceId::from_raw(0);

    let Some(mut seq) = sched.prefill_queue.dequeue() else {
        let err = receive_error(&rx);
        panic!("Muse image request was not admitted: {err}");
    };
    assert_eq!(seq.seq_id, seq_id);
    assert_eq!(
        seq.prompt_tokens,
        vec![
            10,
            DEFAULT_IMAGE_START_TOKEN_ID,
            DEFAULT_IMAGE_TOKEN_ID,
            DEFAULT_IMAGE_END_TOKEN_ID,
            11,
        ]
    );
    let Some(embeddings) = seq.vlm_embeddings.as_ref() else {
        panic!("Muse image request was admitted without prepared embeddings");
    };
    assert_eq!(
        mlxcel_core::array_shape(&embeddings.inputs_embeds),
        vec![1, 5, 8]
    );

    if let Err(err) = BatchScheduler::begin_prefill(&mut seq) {
        panic!("Muse image request failed prefill transition: {err}");
    }
    sched.execute_full_prefill(seq);
    let result = receive_done(&rx);
    assert_eq!(result.prompt_tokens, 5);
    assert_eq!(sched.prefill_queue.len(), 0);
    assert!(sched.active_batch.is_empty());
    assert_eq!(muse_state_offsets(&sched, seq_id), None);
}

#[test]
fn muse_scheduler_rejects_media_cardinality_mismatch_before_generation() {
    let mut missing_image = scheduler();
    let rx = enqueue(
        &mut missing_image,
        vec![1, DEFAULT_IMAGE_PLACEHOLDER_TOKEN_ID, 2],
        Vec::new(),
    );
    let err = receive_error(&rx);
    assert!(err.contains("image marker tokens"));
    assert_eq!(missing_image.prefill_queue.len(), 0);
    assert!(missing_image.active_batch.is_empty());

    let mut missing_placeholder = scheduler();
    let rx = enqueue(
        &mut missing_placeholder,
        vec![1, 2],
        vec![tiny_image_bytes()],
    );
    let err = receive_error(&rx);
    assert!(err.contains("0 image placeholders"));
    assert!(err.contains("1 images were processed"));
    assert_eq!(missing_placeholder.prefill_queue.len(), 0);
    assert!(missing_placeholder.active_batch.is_empty());
}
