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

//! Issue #2625: an admission that aborts between media preparation and the
//! per-sequence binds must leave every model fallback slot empty.
//!
//! `prepare_request_vlm_embeddings` writes a request's media state (Qwen VL
//! MRoPE deltas and DeepStack features, Mllama cross-attention states, ...)
//! into model-level fallback slots before a sequence id exists, and admission
//! moves them under the request's id with `bind_*_to_sequence`. When
//! admission returns in between, nothing drained the slots, so the next
//! admitted request bound the failed request's state as its own; a text-only
//! request never overwrites them and so decoded against another request's
//! image.
//!
//! The tests write the fallback slots directly, standing in for the media
//! preparation, then drive the real `enqueue_request` through each abort
//! path and check what the next text-only request binds.

use super::*;
use crate::models::qwen3_vl::qwen3_vl_tests::tiny_qwen3_vl_model;
use crate::vision::mllama_vl::tests::{image_states, tiny_wrapper};

/// Scheduler with room for one active and one queued sequence, so the cache
/// pool holds exactly two sequences.
fn scheduler(model: LoadedModel, vlm_prefix_cache: bool) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let scheduler = BatchScheduler::with_config(
        model,
        MlxcelTokenizer::stub(),
        vec![0],
        rx,
        1,
        1,
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        0,
        false,
        PreemptionPolicy::default(),
        1,
        DecodeStorageBackend::Dense,
    )
    .with_vlm_prefix_cache(vlm_prefix_cache);
    install_thread_local_default_stream(scheduler.generation_stream.as_ref());
    scheduler
}

/// A Qwen3-VL model loaded without a vision tower (#1367): text requests run,
/// and an image request fails inside media preparation.
fn text_only_qwen3_vl() -> LoadedModel {
    LoadedModel::Qwen3VL(crate::vision::Qwen3VLModel {
        text_model: tiny_qwen3_vl_model(),
        vision_encoder: None,
        text_only_path: Some("tiny-qwen3-vl".to_string()),
        processor: crate::vision::processors::qwen2_vl::Qwen2VLProcessor::new(14, 2, 2),
        image_token_id: 30,
        video_token_id: 31,
        vision_start_token_id: 29,
        spatial_merge_size: 2,
    })
}

fn qwen3_vl(scheduler: &BatchScheduler) -> &crate::vision::Qwen3VLModel {
    let LoadedModel::Qwen3VL(model) = scheduler.engine.model() else {
        panic!("the scheduler must hold the Qwen3-VL model");
    };
    model
}

/// Write what an image request's preparation leaves behind on Qwen3-VL: an
/// MRoPE fallback entry with a nonzero delta and a DeepStack fallback entry.
fn seed_qwen3_vl_fallback(model: &crate::vision::Qwen3VLModel) {
    model
        .text_model
        .set_mrope_state(mlxcel_core::from_slice_i32(&[0; 9], &[3, 1, 3]), 7);
    let mask = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[0.0, 1.0, 0.0], &[1, 3]),
        mlxcel_core::dtype::BOOL,
    );
    model.text_model.set_deepstack_state(
        mask,
        vec![mlxcel_core::full_f32(
            &[1, 8],
            0.5,
            mlxcel_core::dtype::FLOAT32,
        )],
    );
}

/// Assert `seq_id` was bound text-only: no MRoPE delta or positions, and no
/// DeepStack features.
fn assert_qwen3_vl_bound_text_only(model: &crate::vision::Qwen3VLModel, seq_id: SequenceId) {
    let entry = model
        .text_model
        .take_mrope_entry(seq_id)
        .expect("admission binds an MRoPE entry for every Qwen3-VL sequence");
    assert_eq!(
        entry.rope_deltas, None,
        "the text-only request inherited the aborted request's MRoPE delta"
    );
    assert!(
        entry.position_ids.is_none(),
        "the text-only request inherited the aborted request's MRoPE positions"
    );
    assert!(
        model.text_model.take_deepstack_entry(seq_id).is_none(),
        "the text-only request inherited the aborted request's DeepStack features"
    );
}

fn png_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    image::DynamicImage::new_rgb8(28, 28)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("encode a tiny PNG");
    bytes
}

fn enqueue(scheduler: &mut BatchScheduler, images: Vec<Vec<u8>>) -> mpsc::Receiver<GenerateEvent> {
    let mut options = super::scheduler_model_owned_cache_tests::options();
    options.prompt_cache_ctx = None;
    let (tx, rx) = mpsc::channel();
    scheduler.enqueue_request(
        "prompt".into(),
        Some(vec![1, 2, 3]),
        options,
        images,
        vec![],
        vec![],
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    rx
}

/// Allocate sequences until the pool refuses one, and return the held ids.
fn exhaust_pool(scheduler: &mut BatchScheduler) -> Vec<SequenceId> {
    let mut held = Vec::new();
    while let Ok(id) = scheduler.allocate_sequence_state() {
        held.push(id);
        assert!(held.len() <= 8, "the cache pool never filled");
    }
    held
}

fn expect_error(rx: &mpsc::Receiver<GenerateEvent>) -> String {
    match rx.try_recv() {
        Ok(GenerateEvent::Error(message)) => message,
        Ok(_) => panic!("admission must answer the aborted request with an error"),
        Err(err) => panic!("admission sent nothing to the aborted request: {err}"),
    }
}

/// The id of the one request waiting in the prefill queue.
fn queued_seq_id(scheduler: &BatchScheduler) -> SequenceId {
    assert_eq!(scheduler.prefill_queue.len(), 1, "one request is queued");
    scheduler
        .prefill_queue
        .peek()
        .expect("the text-only request was queued")
        .seq_id
}

/// The headline path: the media state is prepared, the cache pool is full,
/// admission answers `Server busy`. The next text-only request must not bind
/// the aborted request's MRoPE and DeepStack state.
#[test]
fn allocation_failure_after_media_preparation_clears_qwen_vl_fallback() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut scheduler = scheduler(text_only_qwen3_vl(), true);
    let mut held = exhaust_pool(&mut scheduler);
    assert_eq!(held.len(), 2, "the pool holds one active and one queued");

    seed_qwen3_vl_fallback(qwen3_vl(&scheduler));
    let busy = enqueue(&mut scheduler, vec![]);
    let message = expect_error(&busy);
    assert!(message.starts_with("Server busy"), "got: {message}");

    scheduler.release_sequence_caches(held.pop().unwrap());
    let _next = enqueue(&mut scheduler, vec![]);
    let seq_id = queued_seq_id(&scheduler);
    assert_qwen3_vl_bound_text_only(qwen3_vl(&scheduler), seq_id);
}

/// The early VLM-prefix-cache preparation fails after a family wrote its
/// fallback (here: the image reaches a model without a vision tower).
#[test]
fn early_preparation_error_clears_qwen_vl_fallback() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut scheduler = scheduler(text_only_qwen3_vl(), true);

    seed_qwen3_vl_fallback(qwen3_vl(&scheduler));
    let failed = enqueue(&mut scheduler, vec![png_bytes()]);
    let message = expect_error(&failed);
    assert!(message.contains("without a vision tower"), "got: {message}");
    assert!(scheduler.prefill_queue.is_empty());

    let _next = enqueue(&mut scheduler, vec![]);
    let seq_id = queued_seq_id(&scheduler);
    assert_qwen3_vl_bound_text_only(qwen3_vl(&scheduler), seq_id);
}

/// Without VLM prefix sharing the media is prepared after allocation; a
/// preparation error releases the sequence and must also drop the fallback.
#[test]
fn late_preparation_error_clears_qwen_vl_fallback() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut scheduler = scheduler(text_only_qwen3_vl(), false);

    seed_qwen3_vl_fallback(qwen3_vl(&scheduler));
    let failed = enqueue(&mut scheduler, vec![png_bytes()]);
    let message = expect_error(&failed);
    assert!(message.contains("without a vision tower"), "got: {message}");
    assert_eq!(
        scheduler.engine.pool().active_count(),
        0,
        "the failed request's sequence is released"
    );

    let _next = enqueue(&mut scheduler, vec![]);
    let seq_id = queued_seq_id(&scheduler);
    assert_qwen3_vl_bound_text_only(qwen3_vl(&scheduler), seq_id);
}

/// Llama 3.2 Vision keeps the image out of the token stream; its prepared
/// cross-attention states must not reach the next text-only request either.
#[test]
fn allocation_failure_after_media_preparation_clears_mllama_image_state() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut scheduler = scheduler(LoadedModel::MllamaVLM(tiny_wrapper()), true);
    let mut held = exhaust_pool(&mut scheduler);

    let LoadedModel::MllamaVLM(mllama) = scheduler.engine.model() else {
        unreachable!();
    };
    mllama.set_cross_attention_states(image_states(2625));
    assert!(mllama.has_cross_attention_states());

    let busy = enqueue(&mut scheduler, vec![]);
    let message = expect_error(&busy);
    assert!(message.starts_with("Server busy"), "got: {message}");

    scheduler.release_sequence_caches(held.pop().unwrap());
    let _next = enqueue(&mut scheduler, vec![]);
    let seq_id = queued_seq_id(&scheduler);
    let LoadedModel::MllamaVLM(mllama) = scheduler.engine.model() else {
        unreachable!();
    };
    assert!(
        !mllama.has_image_state_for_sequence(seq_id),
        "the text-only request inherited the aborted request's image"
    );
    assert!(!mllama.has_cross_attention_states());
}
