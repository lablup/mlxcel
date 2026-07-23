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

//! Request admission and asynchronous image-preprocessing completion.

use std::time::Duration;

use super::super::xla_preprocess::{
    ImagePreprocessJob, ImagePreprocessOutcome, ImagePreprocessResult,
};
use super::*;

pub(super) fn validate_xla_output_features(logprobs: bool, structured: bool) -> Result<(), String> {
    if logprobs {
        return Err(
            "the OpenXLA backend does not support logprobs (no logit readback)".to_string(),
        );
    }
    if structured {
        return Err(
            "the OpenXLA backend does not support structured / JSON-schema output".to_string(),
        );
    }
    Ok(())
}

impl<E: XlaServingEngine> XlaServeWorker<E> {
    /// Validate, tokenize, and submit one request. Images enter the bounded
    /// preprocessing stage; text reaches the engine directly.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn admit(
        &mut self,
        prompt: String,
        prompt_token_ids: Option<Vec<i32>>,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<crate::server::media::ResolvedVideo>,
        media: MediaRequestMetadata,
        response_tx: mpsc::Sender<GenerateEvent>,
        cancelled: Arc<AtomicBool>,
    ) {
        if let Err(error) = media.validate_xla_raw_counts(images.len(), audio.len(), videos.len()) {
            let _ = response_tx.send(GenerateEvent::Error(error));
            return;
        }
        if let Err(error) =
            validate_xla_output_features(options.logprobs.enabled, options.structured.is_some())
        {
            let _ = response_tx.send(GenerateEvent::Error(error));
            return;
        }
        let prompt_tokens = match prompt_token_ids {
            Some(tokens) => tokens,
            None => {
                let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
                match self.tokenizer.encode(&prompt, add_special) {
                    Ok(ids) => ids.into_iter().map(|token| token as i32).collect(),
                    Err(error) => {
                        let _ = response_tx
                            .send(GenerateEvent::Error(format!("Tokenization error: {error}")));
                        return;
                    }
                }
            }
        };
        let params = sample_params(&options);
        let stop_sequences = options.stop_sequences.unwrap_or_default();
        let start = Instant::now();

        if images.is_empty() {
            self.submit_text(
                prompt_tokens,
                options.max_tokens,
                params,
                stop_sequences,
                start,
                response_tx,
                cancelled,
            );
            return;
        }

        let Some(stage) = self.image_preprocessor.as_ref() else {
            let _ = response_tx.send(GenerateEvent::Error(
                "the loaded OpenXLA model/runtime bundle does not support image input".to_string(),
            ));
            return;
        };
        let Some(next_job_id) = self.next_image_job_id.checked_add(1) else {
            let _ = response_tx.send(GenerateEvent::Error(
                "OpenXLA image preprocessing request id overflowed".to_string(),
            ));
            return;
        };
        let job_id = self.next_image_job_id;
        self.next_image_job_id = next_job_id;
        let job = ImagePreprocessJob {
            job_id,
            token_ids: prompt_tokens.clone(),
            expected_image_count: media.declared_images,
            images,
            cancelled: cancelled.clone(),
        };
        match stage.try_submit(job) {
            Ok(()) => {
                self.pending_images.insert(
                    job_id,
                    PendingImageState {
                        response_tx,
                        cancelled,
                        prompt_tokens,
                        params,
                        stop_sequences,
                        max_tokens: options.max_tokens,
                        start,
                    },
                );
            }
            Err(mpsc::TrySendError::Full(_)) => {
                let _ = response_tx.send(GenerateEvent::Error(
                    "OpenXLA image preprocessing queue is full; retry later".to_string(),
                ));
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                let _ = response_tx.send(GenerateEvent::Error(
                    "OpenXLA image preprocessor is unavailable".to_string(),
                ));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_text(
        &mut self,
        prompt_tokens: Vec<i32>,
        max_tokens: usize,
        params: SampleParams,
        stop_sequences: Vec<String>,
        start: Instant,
        response_tx: mpsc::Sender<GenerateEvent>,
        cancelled: Arc<AtomicBool>,
    ) {
        if max_tokens == 0 {
            self.finish_zero_budget(prompt_tokens, start, response_tx);
            return;
        }
        let prompt_len = prompt_tokens.len();
        match self.engine.submit(&prompt_tokens, max_tokens, params) {
            Ok(req_id) => {
                self.batch_observability.record_prefill_start(prompt_len);
                self.states.insert(
                    req_id,
                    ServeState {
                        response_tx,
                        cancelled,
                        detok: StreamingDecodeState::new(&self.tokenizer, &prompt_tokens),
                        stop: StopMatcher::new(stop_sequences),
                        start,
                        prompt_token_count: prompt_len,
                        effective_prefill_len: prompt_len,
                        max_tokens,
                        generated_tokens: 0,
                    },
                );
            }
            Err(error) => {
                let _ = response_tx.send(GenerateEvent::Error(format!(
                    "OpenXLA submit failed: {error}"
                )));
            }
        }
    }

    fn finish_zero_budget(
        &self,
        prompt_tokens: Vec<i32>,
        start: Instant,
        response_tx: mpsc::Sender<GenerateEvent>,
    ) {
        let prompt_len = prompt_tokens.len();
        let detok = StreamingDecodeState::new(&self.tokenizer, &prompt_tokens);
        let _ = response_tx.send(GenerateEvent::Done(detok.finish(start, prompt_len, 0)));
    }

    fn handle_preprocessed(&mut self, result: ImagePreprocessResult) {
        let Some(state) = self.pending_images.remove(&result.job_id) else {
            return;
        };
        if state.cancelled.load(Ordering::Acquire) {
            return;
        }
        let prepared = match result.outcome {
            ImagePreprocessOutcome::Prepared(prepared) => prepared,
            ImagePreprocessOutcome::Cancelled => return,
            ImagePreprocessOutcome::Failed(error) => {
                let _ = state.response_tx.send(GenerateEvent::Error(format!(
                    "OpenXLA image preprocessing failed: {error}"
                )));
                return;
            }
        };
        if state.max_tokens == 0 {
            self.finish_zero_budget(state.prompt_tokens, state.start, state.response_tx);
            return;
        }

        let effective_prefill_len = prepared.sequence_len;
        let prompt_token_count = state.prompt_tokens.len();
        match self
            .engine
            .submit_prepared(prepared, state.max_tokens, state.params)
        {
            Ok(req_id) => {
                // Internal throughput/KV accounting uses the expanded prepared
                // length; the API result below retains the logical prompt count.
                self.batch_observability
                    .record_prefill_start(effective_prefill_len);
                self.states.insert(
                    req_id,
                    ServeState {
                        response_tx: state.response_tx,
                        cancelled: state.cancelled,
                        detok: StreamingDecodeState::new(&self.tokenizer, &state.prompt_tokens),
                        stop: StopMatcher::new(state.stop_sequences),
                        start: state.start,
                        prompt_token_count,
                        effective_prefill_len,
                        max_tokens: state.max_tokens,
                        generated_tokens: 0,
                    },
                );
            }
            Err(error) => {
                let _ = state.response_tx.send(GenerateEvent::Error(format!(
                    "OpenXLA prepared-prefill admission failed: {error}"
                )));
            }
        }
    }

    pub(super) fn drain_preprocessed(&mut self) {
        loop {
            let received = match self.image_preprocessor.as_ref() {
                Some(stage) => stage.try_recv(),
                None => return,
            };
            match received {
                Ok(result) => self.handle_preprocessed(result),
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.image_preprocessor = None;
                    for (_, state) in self.pending_images.drain() {
                        let _ = state.response_tx.send(GenerateEvent::Error(
                            "OpenXLA image preprocessor stopped unexpectedly".to_string(),
                        ));
                    }
                    return;
                }
            }
        }
    }

    fn handle(&mut self, request: ModelRequest) {
        match request {
            ModelRequest::Generate {
                prompt,
                prompt_token_ids,
                options,
                images,
                audio,
                videos,
                media,
                response_tx,
                cancelled,
            } => self.admit(
                prompt,
                prompt_token_ids,
                options,
                images,
                audio,
                videos,
                media,
                response_tx,
                cancelled,
            ),
            ModelRequest::Shutdown => {
                self.shutdown = true;
                for state in self.pending_images.values() {
                    state.cancelled.store(true, Ordering::Release);
                }
                self.pending_images.clear();
            }
        }
    }

    /// Block on the request channel only when no image result is outstanding.
    /// With preprocessing in flight, a short timeout keeps the scheduler asleep
    /// without delaying completion or active decode rows.
    pub(super) fn drain_incoming(&mut self, block: bool) {
        if block {
            let request = if self.pending_images.is_empty() {
                match self.request_rx.recv() {
                    Ok(request) => request,
                    Err(_) => {
                        self.shutdown = true;
                        return;
                    }
                }
            } else {
                match self.request_rx.recv_timeout(Duration::from_millis(2)) {
                    Ok(request) => request,
                    Err(mpsc::RecvTimeoutError::Timeout) => return,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        self.shutdown = true;
                        return;
                    }
                }
            };
            self.handle(request);
        }
        loop {
            match self.request_rx.try_recv() {
                Ok(request) => self.handle(request),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.shutdown = true;
                    break;
                }
            }
        }
    }
}

fn sample_params(options: &ServerGenerateOptions) -> SampleParams {
    let sampling = &options.sampling;
    SampleParams {
        temperature: sampling.temperature,
        top_k: sampling.top_k.max(0) as usize,
        top_p: sampling.top_p,
        min_p: sampling.min_p,
        seed: sampling.seed,
        repetition_penalty: sampling.repetition_penalty,
        frequency_penalty: sampling.frequency_penalty,
        presence_penalty: sampling.presence_penalty,
        dry_multiplier: sampling.dry_multiplier,
        dry_base: sampling.dry_base,
        dry_allowed_length: sampling.dry_allowed_length,
        dry_penalty_last_n: sampling.dry_penalty_last_n,
        dry_sequence_breakers: sampling.dry_sequence_breakers.clone(),
    }
}
