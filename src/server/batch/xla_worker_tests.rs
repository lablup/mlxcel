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

use std::collections::BTreeSet;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use image::{DynamicImage, ImageFormat};

use super::*;
use crate::FakeHostMultimodalPreprocessor;
use crate::server::request_options::{RequestOptionOverrides, build_server_generate_options};
use crate::server::{GenerationResult, ServerConfig};

#[derive(Default)]
struct FakeServingEngine {
    next_id: u64,
    active: BTreeSet<u64>,
    text_submissions: Vec<Vec<i32>>,
    prepared_submissions: Vec<(Vec<i32>, usize)>,
}

impl XlaServingEngine for FakeServingEngine {
    fn b_max(&self) -> usize {
        4
    }

    fn is_idle(&self) -> bool {
        self.active.is_empty()
    }

    fn pending_len(&self) -> usize {
        0
    }

    fn active_len(&self) -> usize {
        self.active.len()
    }

    fn submit(
        &mut self,
        prompt: &[i32],
        _max_new_tokens: usize,
        _params: SampleParams,
    ) -> Result<u64, String> {
        self.text_submissions.push(prompt.to_vec());
        Ok(self.activate())
    }

    fn submit_prepared(
        &mut self,
        prepared: PreparedPrefill,
        _max_new_tokens: usize,
        _params: SampleParams,
    ) -> Result<u64, String> {
        self.prepared_submissions
            .push((prepared.token_ids, prepared.sequence_len));
        Ok(self.activate())
    }

    fn cancel(&mut self, req_id: u64) -> bool {
        self.active.remove(&req_id)
    }

    fn pump(&mut self) -> Result<Vec<EngineEvent>, String> {
        let ids: Vec<_> = self.active.iter().copied().collect();
        self.active.clear();
        let mut events = Vec::with_capacity(ids.len() * 2);
        for req_id in ids {
            events.push(EngineEvent::Token { req_id, token: 0 });
            events.push(EngineEvent::Finished {
                req_id,
                reason: XlaFinishReason::Length,
            });
        }
        Ok(events)
    }
}

impl FakeServingEngine {
    fn activate(&mut self) -> u64 {
        let req_id = self.next_id;
        self.next_id += 1;
        self.active.insert(req_id);
        req_id
    }
}

fn png_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    DynamicImage::new_rgb8(2, 2)
        .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
        .unwrap();
    bytes
}

fn options(max_tokens: usize) -> ServerGenerateOptions {
    build_server_generate_options(
        &ServerConfig::default(),
        RequestOptionOverrides {
            max_tokens: Some(max_tokens),
            ..RequestOptionOverrides::default()
        },
    )
}

fn media(images: usize, audio: usize, videos: usize) -> MediaRequestMetadata {
    MediaRequestMetadata::from_resolved(images, audio, videos)
}

fn image_stage() -> ImagePreprocessStage {
    ImagePreprocessStage::spawn_with_loader(2, || {
        Ok(Some(Box::new(FakeHostMultimodalPreprocessor {
            image_token_id: -200,
            tokens_per_image: 3,
            hidden_size: 4,
            max_sequence_len: 32,
        })))
    })
    .unwrap()
    .unwrap()
}

fn worker(image_preprocessor: Option<ImagePreprocessStage>) -> XlaServeWorker<FakeServingEngine> {
    let (_request_tx, request_rx) = mpsc::channel();
    XlaServeWorker {
        engine: FakeServingEngine::default(),
        tokenizer: MlxcelTokenizer::stub(),
        request_rx,
        states: HashMap::new(),
        batch_metrics: Arc::new(BatchMetrics::new()),
        batch_observability: Arc::new(BatchObservability::new()),
        image_preprocessor,
        pending_images: HashMap::new(),
        next_image_job_id: 0,
        shutdown: false,
    }
}

fn wait_for_preprocessing(worker: &mut XlaServeWorker<FakeServingEngine>) {
    for _ in 0..500 {
        worker.drain_preprocessed();
        if worker.pending_images.is_empty() {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("timed out waiting for image preprocessing");
}

fn receive_done(rx: &mpsc::Receiver<GenerateEvent>) -> GenerationResult {
    loop {
        match rx.recv_timeout(Duration::from_secs(1)).unwrap() {
            GenerateEvent::Done(result) => return result,
            GenerateEvent::Token(_) | GenerateEvent::TokenWithLogprobs(_, _) => {}
            GenerateEvent::Error(error) => panic!("unexpected generation failure: {error}"),
        }
    }
}

#[test]
fn mixed_text_and_image_admission_keeps_public_and_effective_lengths_distinct() {
    let mut worker = worker(Some(image_stage()));
    let (text_tx, text_rx) = mpsc::channel();
    let (image_tx, image_rx) = mpsc::channel();

    worker.admit(
        String::new(),
        Some(vec![10, 11]),
        options(1),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        media(0, 0, 0),
        text_tx,
        Arc::new(AtomicBool::new(false)),
    );
    worker.admit(
        String::new(),
        Some(vec![20, -200, 21]),
        options(1),
        vec![png_bytes()],
        Vec::new(),
        Vec::new(),
        media(1, 0, 0),
        image_tx,
        Arc::new(AtomicBool::new(false)),
    );

    assert_eq!(worker.engine.text_submissions, vec![vec![10, 11]]);
    assert_eq!(worker.states.len(), 1);
    wait_for_preprocessing(&mut worker);
    assert_eq!(
        worker.engine.prepared_submissions,
        vec![(vec![20, -200, -200, -200, 21], 5)]
    );
    assert_eq!(worker.states.len(), 2);
    assert_eq!(
        worker
            .batch_observability
            .total_prefill_tokens
            .load(Ordering::Relaxed),
        7
    );

    let events = worker.engine.pump().unwrap();
    worker.dispatch(events);
    let text_result = receive_done(&text_rx);
    let image_result = receive_done(&image_rx);
    assert_eq!(text_result.prompt_tokens, 2);
    assert_eq!(image_result.prompt_tokens, 3);
    assert_eq!(text_result.completion_tokens, 1);
    assert_eq!(image_result.completion_tokens, 1);
    assert_eq!(
        worker
            .batch_metrics
            .total_sequences_processed
            .load(Ordering::Relaxed),
        2
    );
}

#[test]
fn malformed_image_failure_does_not_disturb_active_text_request() {
    let mut worker = worker(Some(image_stage()));
    let (text_tx, text_rx) = mpsc::channel();
    let (image_tx, image_rx) = mpsc::channel();

    worker.admit(
        String::new(),
        Some(vec![1, 2]),
        options(1),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        media(0, 0, 0),
        text_tx,
        Arc::new(AtomicBool::new(false)),
    );
    worker.admit(
        String::new(),
        Some(vec![3, -200, 4]),
        options(1),
        vec![b"not-an-image".to_vec()],
        Vec::new(),
        Vec::new(),
        media(1, 0, 0),
        image_tx,
        Arc::new(AtomicBool::new(false)),
    );

    wait_for_preprocessing(&mut worker);
    let GenerateEvent::Error(error) = image_rx.recv_timeout(Duration::from_secs(1)).unwrap() else {
        panic!("malformed image must fail only its request");
    };
    assert!(error.contains("image preprocessing failed"));
    assert_eq!(worker.states.len(), 1);
    assert_eq!(worker.engine.active_len(), 1);

    let events = worker.engine.pump().unwrap();
    worker.dispatch(events);
    assert_eq!(receive_done(&text_rx).prompt_tokens, 2);

    let (reuse_tx, reuse_rx) = mpsc::channel();
    worker.admit(
        String::new(),
        Some(vec![5, -200, 6]),
        options(1),
        vec![png_bytes()],
        Vec::new(),
        Vec::new(),
        media(1, 0, 0),
        reuse_tx,
        Arc::new(AtomicBool::new(false)),
    );
    wait_for_preprocessing(&mut worker);
    let events = worker.engine.pump().unwrap();
    worker.dispatch(events);
    assert_eq!(receive_done(&reuse_rx).prompt_tokens, 3);
}

#[test]
fn cancelled_image_does_not_disturb_active_text_or_prevent_slot_reuse() {
    let mut worker = worker(Some(image_stage()));
    let (text_tx, text_rx) = mpsc::channel();
    let (cancelled_tx, cancelled_rx) = mpsc::channel();

    worker.admit(
        String::new(),
        Some(vec![1, 2]),
        options(1),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        media(0, 0, 0),
        text_tx,
        Arc::new(AtomicBool::new(false)),
    );
    worker.admit(
        String::new(),
        Some(vec![3, -200, 4]),
        options(1),
        vec![png_bytes()],
        Vec::new(),
        Vec::new(),
        media(1, 0, 0),
        cancelled_tx,
        Arc::new(AtomicBool::new(true)),
    );
    wait_for_preprocessing(&mut worker);
    assert!(cancelled_rx.try_recv().is_err());
    assert_eq!(worker.states.len(), 1);

    let events = worker.engine.pump().unwrap();
    worker.dispatch(events);
    assert_eq!(receive_done(&text_rx).prompt_tokens, 2);

    let (reuse_tx, reuse_rx) = mpsc::channel();
    worker.admit(
        String::new(),
        Some(vec![5, -200, 6]),
        options(1),
        vec![png_bytes()],
        Vec::new(),
        Vec::new(),
        media(1, 0, 0),
        reuse_tx,
        Arc::new(AtomicBool::new(false)),
    );
    wait_for_preprocessing(&mut worker);
    let events = worker.engine.pump().unwrap();
    worker.dispatch(events);
    assert_eq!(receive_done(&reuse_rx).prompt_tokens, 3);
}

#[test]
fn unsupported_output_features_are_rejected_before_engine_admission() {
    assert!(
        admission::validate_xla_output_features(false, true)
            .unwrap_err()
            .contains("structured")
    );

    let mut worker = worker(Some(image_stage()));
    let (logprobs_tx, logprobs_rx) = mpsc::channel();
    let mut logprobs_options = options(1);
    logprobs_options.logprobs.enabled = true;
    worker.admit(
        String::new(),
        Some(vec![1]),
        logprobs_options,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        media(0, 0, 0),
        logprobs_tx,
        Arc::new(AtomicBool::new(false)),
    );
    let GenerateEvent::Error(error) = logprobs_rx.recv().unwrap() else {
        panic!("logprobs must stay explicitly unsupported");
    };
    assert!(error.contains("does not support logprobs"));

    let (audio_tx, audio_rx) = mpsc::channel();
    worker.admit(
        String::new(),
        Some(vec![1]),
        options(1),
        Vec::new(),
        vec![vec![1, 2, 3]],
        Vec::new(),
        media(0, 1, 0),
        audio_tx,
        Arc::new(AtomicBool::new(false)),
    );
    let GenerateEvent::Error(error) = audio_rx.recv().unwrap() else {
        panic!("audio must be explicitly unsupported");
    };
    assert!(error.contains("does not support audio"));
    assert!(worker.engine.text_submissions.is_empty());
    assert!(worker.engine.prepared_submissions.is_empty());
}

#[test]
fn declared_audio_video_and_unqualified_images_never_fall_back_to_text() {
    let mut worker = worker(None);

    for (media, expected) in [
        (MediaRequestMetadata::new(0, 1, 0, 0, 0, 0), "audio"),
        (MediaRequestMetadata::new(0, 0, 1, 0, 0, 0), "video"),
    ] {
        let (tx, rx) = mpsc::channel();
        worker.admit(
            String::new(),
            Some(vec![1]),
            options(1),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            media,
            tx,
            Arc::new(AtomicBool::new(false)),
        );
        let GenerateEvent::Error(error) = rx.recv().unwrap() else {
            panic!("{expected} declaration must be rejected");
        };
        assert!(error.contains(expected));
    }

    let (partial_tx, partial_rx) = mpsc::channel();
    worker.admit(
        String::new(),
        Some(vec![1]),
        options(1),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        MediaRequestMetadata::new(1, 0, 0, 0, 0, 0),
        partial_tx,
        Arc::new(AtomicBool::new(false)),
    );
    let GenerateEvent::Error(error) = partial_rx.recv().unwrap() else {
        panic!("dropped image declaration must be rejected");
    };
    assert!(error.contains("refusing text fallback"));

    let (unsupported_tx, unsupported_rx) = mpsc::channel();
    worker.admit(
        String::new(),
        Some(vec![1, -200, 2]),
        options(1),
        vec![png_bytes()],
        Vec::new(),
        Vec::new(),
        media(1, 0, 0),
        unsupported_tx,
        Arc::new(AtomicBool::new(false)),
    );
    let GenerateEvent::Error(error) = unsupported_rx.recv().unwrap() else {
        panic!("image input without qualified preprocessor must be rejected");
    };
    assert!(error.contains("does not support image input"));
    assert!(worker.engine.text_submissions.is_empty());
    assert!(worker.engine.prepared_submissions.is_empty());
}

#[test]
fn pending_preprocess_poll_timeout_does_not_shutdown_worker() {
    let (request_tx, request_rx) = mpsc::channel();
    let mut worker = XlaServeWorker {
        engine: FakeServingEngine::default(),
        tokenizer: MlxcelTokenizer::stub(),
        request_rx,
        states: HashMap::new(),
        batch_metrics: Arc::new(BatchMetrics::new()),
        batch_observability: Arc::new(BatchObservability::new()),
        image_preprocessor: Some(image_stage()),
        pending_images: HashMap::new(),
        next_image_job_id: 0,
        shutdown: false,
    };
    let (response_tx, _response_rx) = mpsc::channel();
    worker.admit(
        String::new(),
        Some(vec![1, -200, 2]),
        options(1),
        vec![png_bytes()],
        Vec::new(),
        Vec::new(),
        media(1, 0, 0),
        response_tx,
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(worker.pending_images.len(), 1);

    worker.drain_incoming(true);

    assert!(!worker.shutdown);
    drop(request_tx);
}
