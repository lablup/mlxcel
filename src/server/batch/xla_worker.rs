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

//! OpenXLA / IREE serve worker (issue #449 M3 Stage 2c).
//!
//! [`XlaServeWorker`] adapts the `mlxcel-xla` continuous-batching engine
//! ([`XlaBatchEngine`]) to the server's [`BatchEngine`](super::BatchEngine)
//! contract, so `ModelProvider` drives the OpenXLA backend through the same
//! request/event seam as the MLX [`BatchScheduler`](super::BatchScheduler). The
//! serve loop drains [`ModelRequest`]s, tokenizes each prompt, submits it to the
//! engine, pumps the engine one step at a time, and maps each per-request
//! [`EngineEvent`] back to a [`GenerateEvent`] on that request's channel, reusing
//! the server's [`StreamingDecodeState`] for byte-fallback-safe detokenization.
//!
//! Text and qualified LLaVA image requests share this path. Image decoding and
//! host vision execution run through a bounded preprocessing stage, then enter
//! the same engine as an owned prepared prefill. This path honors `max_tokens`,
//! the model's EOS ids,
//! sampling (temperature / top-k / top-p / min-p / seed, #449 M3 Stage 2d), the
//! history-based penalties (repetition / frequency / presence / DRY, #449 M3
//! Stage 2d, applied host-side in the engine's sampler with the same math and
//! order as the MLX path), and stop strings (#449 M3 Stage 2d): a
//! [`StopMatcher`] withholds any decoded tail that could begin a stop string and
//! ends the request at the earliest full match, excluding the stop string and
//! everything after it from the output (the same rule as
//! [`apply_stop_sequences`](crate::server::anthropic_translator::apply_stop_sequences),
//! applied incrementally so it is safe across token boundaries). Requests that
//! need features the engine cannot provide are rejected with a clear error rather
//! than served wrong: logprobs (no logit readback), structured / JSON-schema
//! output (no constraint masking), and unsupported audio/video inputs.
//!
//! Compiled only under `xla-iree` (real IREE execution). The MLX serving path is
//! untouched.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::session::PreparedPrefill;
use mlxcel_xla::{EngineEvent, FinishReason as XlaFinishReason, SampleParams, XlaBatchEngine};

use super::BatchEngine;
use super::observability::BatchObservability;
use super::stop_matcher::StopMatcher;
use super::xla_preprocess::ImagePreprocessStage;
use crate::server::ServerGenerateOptions;
use crate::server::media::MediaRequestMetadata;
use crate::server::model_provider::model_worker::StreamingDecodeState;
use crate::server::model_provider::{GenerateEvent, ModelRequest};
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;

#[path = "xla_worker_admission.rs"]
mod admission;
#[cfg(test)]
#[path = "xla_worker_tests.rs"]
mod tests;

pub(crate) trait XlaServingEngine {
    fn b_max(&self) -> usize;
    fn is_idle(&self) -> bool;
    fn pending_len(&self) -> usize;
    fn active_len(&self) -> usize;
    fn submit(
        &mut self,
        prompt: &[i32],
        max_new_tokens: usize,
        params: SampleParams,
    ) -> Result<u64, String>;
    fn submit_prepared(
        &mut self,
        prepared: PreparedPrefill,
        max_new_tokens: usize,
        params: SampleParams,
    ) -> Result<u64, String>;
    fn cancel(&mut self, req_id: u64) -> bool;
    fn pump(&mut self) -> Result<Vec<EngineEvent>, String>;
}

impl XlaServingEngine for XlaBatchEngine {
    fn b_max(&self) -> usize {
        XlaBatchEngine::b_max(self)
    }

    fn is_idle(&self) -> bool {
        XlaBatchEngine::is_idle(self)
    }

    fn pending_len(&self) -> usize {
        XlaBatchEngine::pending_len(self)
    }

    fn active_len(&self) -> usize {
        XlaBatchEngine::active_len(self)
    }

    fn submit(
        &mut self,
        prompt: &[i32],
        max_new_tokens: usize,
        params: SampleParams,
    ) -> Result<u64, String> {
        XlaBatchEngine::submit(self, prompt, max_new_tokens, params)
            .map_err(|error| error.to_string())
    }

    fn submit_prepared(
        &mut self,
        prepared: PreparedPrefill,
        max_new_tokens: usize,
        params: SampleParams,
    ) -> Result<u64, String> {
        XlaBatchEngine::submit_prepared(self, prepared, max_new_tokens, params)
            .map_err(|error| error.to_string())
    }

    fn cancel(&mut self, req_id: u64) -> bool {
        XlaBatchEngine::cancel(self, req_id)
    }

    fn pump(&mut self) -> Result<Vec<EngineEvent>, String> {
        XlaBatchEngine::pump(self)
    }
}

/// Per-active-request state, keyed by the engine's request id.
struct ServeState {
    response_tx: mpsc::Sender<GenerateEvent>,
    cancelled: Arc<AtomicBool>,
    detok: StreamingDecodeState,
    /// Streaming-safe stop-string matcher. Inactive (a pass-through) when the
    /// request set no stop strings, so those requests stream exactly as before.
    stop: StopMatcher,
    start: Instant,
    /// Public/API usage keeps the original logical prompt count.
    prompt_token_count: usize,
    /// Internal prefill throughput and KV positions use the expanded prepared
    /// length. Kept separately so future metrics cannot conflate the two.
    effective_prefill_len: usize,
    max_tokens: usize,
    /// Tokens the engine has generated for this request (one per `Token` event,
    /// counted even when detok withholds the piece), reported to `BatchMetrics`
    /// when the sequence completes.
    generated_tokens: usize,
}

struct PendingImageState {
    response_tx: mpsc::Sender<GenerateEvent>,
    cancelled: Arc<AtomicBool>,
    prompt_tokens: Vec<i32>,
    params: SampleParams,
    stop_sequences: Vec<String>,
    max_tokens: usize,
    start: Instant,
}

/// Server-side worker that serves requests through the OpenXLA continuous-batching
/// engine. Built and run on a single worker thread (see
/// `model_worker::spawn_xla_model_worker`).
pub(crate) struct XlaServeWorker<E = XlaBatchEngine> {
    engine: E,
    tokenizer: MlxcelTokenizer,
    request_rx: mpsc::Receiver<ModelRequest>,
    /// Active requests, keyed by the engine req id `submit` returned.
    states: HashMap<u64, ServeState>,
    /// Batch metrics surfaced by the `/metrics` endpoint, populated the same way
    /// the MLX `BatchScheduler` populates them (active count + queue depth gauges,
    /// per-sequence completion), so the OpenXLA serve path is observable too.
    batch_metrics: Arc<BatchMetrics>,
    /// Cumulative serve counters (`/metrics`): sequences started/completed and
    /// prefill/decode token throughput. The cache-pool / paged gauges are
    /// MLX-specific and stay zero for this path (it has neither).
    batch_observability: Arc<BatchObservability>,
    image_preprocessor: Option<ImagePreprocessStage>,
    pending_images: HashMap<u64, PendingImageState>,
    next_image_job_id: u64,
    shutdown: bool,
}

impl XlaServeWorker<XlaBatchEngine> {
    pub(crate) fn new(
        engine: XlaBatchEngine,
        tokenizer: MlxcelTokenizer,
        model_path: std::path::PathBuf,
        request_rx: mpsc::Receiver<ModelRequest>,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
    ) -> Result<Self, String> {
        let image_preprocessor = ImagePreprocessStage::spawn_for_model(model_path, engine.b_max())?;
        Ok(Self {
            engine,
            tokenizer,
            request_rx,
            states: HashMap::new(),
            batch_metrics,
            batch_observability,
            image_preprocessor,
            pending_images: HashMap::new(),
            next_image_job_id: 0,
            shutdown: false,
        })
    }
}

impl<E: XlaServingEngine> XlaServeWorker<E> {
    /// Refresh the active-count and queue-depth gauges from the engine. Cheap
    /// (two atomic stores), called each serve iteration so `/metrics` tracks the
    /// live batch.
    fn update_gauges(&self) {
        self.batch_metrics
            .set_active_count(self.engine.active_len());
        self.batch_metrics
            .set_queue_depth(self.engine.pending_len());
    }

    /// Drop any requests whose client cancelled, freeing their engine slots. A
    /// cancelled request emits no further events (the caller initiated it).
    fn evict_cancelled(&mut self) {
        let ids: Vec<u64> = self
            .states
            .iter()
            .filter(|(_, s)| s.cancelled.load(Ordering::Relaxed))
            .map(|(&id, _)| id)
            .collect();
        for id in ids {
            self.engine.cancel(id);
            self.states.remove(&id);
        }
        let jobs: Vec<u64> = self
            .pending_images
            .iter()
            .filter(|(_, state)| state.cancelled.load(Ordering::Relaxed))
            .map(|(&job_id, _)| job_id)
            .collect();
        for job_id in jobs {
            self.pending_images.remove(&job_id);
        }
    }

    /// Map one engine step's events onto the per-request channels.
    fn dispatch(&mut self, events: Vec<EngineEvent>) {
        for ev in events {
            match ev {
                EngineEvent::Token { req_id, token } => {
                    // Decode the token, run it through the request's stop matcher,
                    // and emit only what is safe to stream. `Some(keep)` means a
                    // stop string matched: finalize the request keeping `keep`
                    // bytes (everything already streamed, before the stop string).
                    let stop_keep = {
                        let Some(state) = self.states.get_mut(&req_id) else {
                            continue;
                        };
                        // Count the generated token even if detok withholds the
                        // piece (a mid-multibyte token), so the completion metric
                        // reflects what the engine produced.
                        state.generated_tokens += 1;
                        let Some(piece) = state.detok.on_token(token, &self.tokenizer) else {
                            continue;
                        };
                        if state.stop.is_active() {
                            let chunk = state.stop.push(&piece);
                            if !chunk.emit.is_empty() {
                                let _ = state.response_tx.send(GenerateEvent::Token(chunk.emit));
                            }
                            chunk.stopped.then(|| state.stop.emitted_len())
                        } else {
                            let _ = state.response_tx.send(GenerateEvent::Token(piece));
                            None
                        }
                    };
                    if let Some(keep) = stop_keep {
                        self.finalize_stop(req_id, keep);
                    }
                }
                EngineEvent::Finished { req_id, reason } => {
                    if let Some(state) = self.states.remove(&req_id) {
                        let ServeState {
                            response_tx,
                            mut detok,
                            mut stop,
                            start,
                            prompt_token_count,
                            effective_prefill_len,
                            max_tokens,
                            generated_tokens,
                            ..
                        } = state;
                        tracing::debug!(
                            prompt_token_count,
                            effective_prefill_len,
                            "OpenXLA request completed with distinct public and KV prefill lengths"
                        );
                        self.batch_metrics
                            .record_sequence_completed(generated_tokens);
                        self.batch_observability.record_sequence_completed();
                        // Release any tail held back as a potential stop-string
                        // prefix; ending on EOS/length means it never completed
                        // one, so it is real output and must be streamed first.
                        let tail = stop.flush();
                        if !tail.is_empty() {
                            let _ = response_tx.send(GenerateEvent::Token(tail));
                        }
                        // Then release any tail the incremental detokenizer held
                        // back (a final token carrying complete text plus a
                        // trailing incomplete UTF-8 byte). It is the last output,
                        // so it follows the stop-matcher tail (issue #633).
                        if let Some(detok_tail) = detok.flush(&self.tokenizer) {
                            let _ = response_tx.send(GenerateEvent::Token(detok_tail));
                        }
                        let mut result = detok.finish(start, prompt_token_count, max_tokens);
                        // The engine knows the authoritative reason; prefer it over
                        // the count-based inference in `finish`.
                        result.finish_reason = match reason {
                            XlaFinishReason::Stop => "stop",
                            XlaFinishReason::Length => "length",
                        }
                        .to_string();
                        let _ = response_tx.send(GenerateEvent::Done(result));
                    }
                }
            }
        }
    }

    /// Finalize a request that matched a stop string: free its engine slot and
    /// send a terminal `Done` whose text is truncated to `keep_bytes` (the bytes
    /// already streamed, i.e. everything before the stop string). The matcher
    /// withheld the stop string and everything after it, so the non-streaming
    /// result matches what was streamed; the finish reason is `stop`.
    fn finalize_stop(&mut self, req_id: u64, keep_bytes: usize) {
        self.engine.cancel(req_id);
        if let Some(state) = self.states.remove(&req_id) {
            let ServeState {
                response_tx,
                detok,
                start,
                prompt_token_count,
                max_tokens,
                generated_tokens,
                ..
            } = state;
            self.batch_metrics
                .record_sequence_completed(generated_tokens);
            self.batch_observability.record_sequence_completed();
            let mut result =
                detok.finish_truncated(keep_bytes, start, prompt_token_count, max_tokens);
            result.finish_reason = "stop".to_string();
            let _ = response_tx.send(GenerateEvent::Done(result));
        }
    }

    /// Send a terminal `Error` to every in-flight request (used when the engine
    /// fails fatally) and clear them.
    fn fail_all(&mut self, msg: &str) {
        for (_, state) in self.states.drain() {
            let _ = state
                .response_tx
                .send(GenerateEvent::Error(msg.to_string()));
        }
        for (_, state) in self.pending_images.drain() {
            let _ = state
                .response_tx
                .send(GenerateEvent::Error(msg.to_string()));
        }
    }
}

impl<E: XlaServingEngine> BatchEngine for XlaServeWorker<E> {
    fn serve(&mut self) {
        tracing::info!(
            "OpenXLA serve worker starting (continuous batching, B_max={}, sampling, stop strings)",
            self.engine.b_max()
        );
        loop {
            self.drain_preprocessed();
            self.evict_cancelled();

            // If the engine has no work, block for the next request so the thread
            // does not spin; otherwise pick up any newly queued requests and pump.
            let block = self.engine.is_idle() && !self.shutdown;
            self.drain_incoming(block);
            self.drain_preprocessed();
            self.evict_cancelled();
            // Reflect admits/cancels (and a drained-to-idle batch) in the gauges.
            self.update_gauges();

            if self.engine.is_idle() {
                if self.shutdown {
                    break;
                }
                // Idle but not shutdown (e.g. everything drained was cancelled or a
                // zero-budget request): loop and block again.
                continue;
            }

            match self.engine.pump() {
                Ok(events) => {
                    // Each `Token` event is one token produced this step across the
                    // active batch, so the count is the step's decode width.
                    let decoded = events
                        .iter()
                        .filter(|e| matches!(e, EngineEvent::Token { .. }))
                        .count();
                    if decoded > 0 {
                        self.batch_observability.record_decode_step(decoded);
                    }
                    self.dispatch(events);
                    // Reflect any sequences that completed this step.
                    self.update_gauges();
                }
                Err(err) => {
                    tracing::error!("OpenXLA engine step failed: {err}");
                    self.fail_all(&format!("OpenXLA engine step failed: {err}"));
                    break;
                }
            }
        }
        tracing::info!("OpenXLA serve worker stopped");
    }
}
