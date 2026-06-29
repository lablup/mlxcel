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
//! Scope: text-only. This path honors `max_tokens`, the model's EOS ids,
//! sampling (temperature / top-k / top-p / min-p / seed, #449 M3 Stage 2d), and
//! stop strings (#449 M3 Stage 2d): a [`StopMatcher`] withholds any decoded tail
//! that could begin a stop string and ends the request at the earliest full
//! match, excluding the stop string and everything after it from the output
//! (the same rule as
//! [`apply_stop_sequences`](crate::server::anthropic_translator::apply_stop_sequences),
//! applied incrementally so it is safe across token boundaries). The
//! history-based penalties (repetition / frequency / presence / DRY) are not
//! applied (logged once). Requests that need features the engine cannot provide
//! are rejected with a clear error rather than served wrong: logprobs (no logit
//! readback), structured / JSON-schema output (no constraint masking), and
//! multimodal inputs (text-only).
//!
//! Compiled only under `xla-iree` (real IREE execution). The MLX serving path is
//! untouched.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_xla::{EngineEvent, FinishReason as XlaFinishReason, SampleParams, XlaBatchEngine};

use super::BatchEngine;
use super::stop_matcher::StopMatcher;
use crate::server::ServerGenerateOptions;
use crate::server::model_provider::model_worker::StreamingDecodeState;
use crate::server::model_provider::{GenerateEvent, ModelRequest};
use crate::tokenizer::MlxcelTokenizer;

/// Per-active-request state, keyed by the engine's request id.
struct ServeState {
    response_tx: mpsc::Sender<GenerateEvent>,
    cancelled: Arc<AtomicBool>,
    detok: StreamingDecodeState,
    /// Streaming-safe stop-string matcher. Inactive (a pass-through) when the
    /// request set no stop strings, so those requests stream exactly as before.
    stop: StopMatcher,
    start: Instant,
    prompt_token_count: usize,
    max_tokens: usize,
}

/// Server-side worker that serves requests through the OpenXLA continuous-batching
/// engine. Built and run on a single worker thread (see
/// `model_worker::spawn_xla_model_worker`).
pub(crate) struct XlaServeWorker {
    engine: XlaBatchEngine,
    tokenizer: MlxcelTokenizer,
    request_rx: mpsc::Receiver<ModelRequest>,
    /// Active requests, keyed by the engine req id `submit` returned.
    states: HashMap<u64, ServeState>,
    shutdown: bool,
    warned_penalties: bool,
}

impl XlaServeWorker {
    pub(crate) fn new(
        engine: XlaBatchEngine,
        tokenizer: MlxcelTokenizer,
        request_rx: mpsc::Receiver<ModelRequest>,
    ) -> Self {
        Self {
            engine,
            tokenizer,
            request_rx,
            states: HashMap::new(),
            shutdown: false,
            warned_penalties: false,
        }
    }

    /// Validate, tokenize, and submit one `Generate` request, or send a terminal
    /// `Error` / empty `Done` when it cannot be served as submitted.
    fn admit(
        &mut self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<crate::server::media::ResolvedVideo>,
        response_tx: mpsc::Sender<GenerateEvent>,
        cancelled: Arc<AtomicBool>,
    ) {
        // Reject what the engine genuinely cannot serve, rather than serve it wrong.
        if !images.is_empty() || !audio.is_empty() || !videos.is_empty() {
            let _ = response_tx.send(GenerateEvent::Error(
                "the OpenXLA backend is text-only; multimodal inputs are not supported".to_string(),
            ));
            return;
        }
        if options.logprobs.enabled {
            let _ = response_tx.send(GenerateEvent::Error(
                "the OpenXLA backend does not support logprobs (greedy argmax, no logit readback)"
                    .to_string(),
            ));
            return;
        }
        if options.structured.is_some() {
            let _ = response_tx.send(GenerateEvent::Error(
                "the OpenXLA backend does not support structured / JSON-schema output".to_string(),
            ));
            return;
        }

        // Sampling: temperature / top-k / top-p / min-p / seed are honored; stop
        // strings are enforced below by a per-request `StopMatcher`. The
        // history-based penalties (repetition / frequency / presence / DRY) are
        // not applied, so warn once when a request asks for them.
        let sampling = &options.sampling;
        let params = SampleParams {
            temperature: sampling.temperature,
            top_k: sampling.top_k.max(0) as usize,
            top_p: sampling.top_p,
            min_p: sampling.min_p,
            seed: sampling.seed,
        };
        let uses_penalties = sampling.repetition_penalty != 1.0
            || sampling.frequency_penalty != 0.0
            || sampling.presence_penalty != 0.0
            || sampling.dry_multiplier != 0.0;
        if uses_penalties && !self.warned_penalties {
            tracing::warn!(
                "the OpenXLA backend applies temperature / top-k / top-p / min-p only; \
                 repetition / frequency / presence penalties and DRY are ignored"
            );
            self.warned_penalties = true;
        }

        let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
        let token_ids = match self.tokenizer.encode(&prompt, add_special) {
            Ok(ids) => ids,
            Err(err) => {
                let _ =
                    response_tx.send(GenerateEvent::Error(format!("Tokenization error: {err}")));
                return;
            }
        };
        let prompt_tokens: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();

        // A zero token budget asks for no generation: return an empty result so the
        // route still sees usage counts, without touching the engine.
        if options.max_tokens == 0 {
            let detok = StreamingDecodeState::new(&self.tokenizer, &prompt_tokens);
            let result = detok.finish(Instant::now(), prompt_tokens.len(), 0);
            let _ = response_tx.send(GenerateEvent::Done(result));
            return;
        }

        let detok = StreamingDecodeState::new(&self.tokenizer, &prompt_tokens);
        match self
            .engine
            .submit(&prompt_tokens, options.max_tokens, params)
        {
            Ok(req_id) => {
                self.states.insert(
                    req_id,
                    ServeState {
                        response_tx,
                        cancelled,
                        detok,
                        stop: StopMatcher::new(options.stop_sequences.unwrap_or_default()),
                        start: Instant::now(),
                        prompt_token_count: prompt_tokens.len(),
                        max_tokens: options.max_tokens,
                    },
                );
            }
            Err(err) => {
                let _ = response_tx.send(GenerateEvent::Error(format!(
                    "OpenXLA submit failed: {err}"
                )));
            }
        }
    }

    fn handle(&mut self, req: ModelRequest) {
        match req {
            ModelRequest::Generate {
                prompt,
                options,
                images,
                audio,
                videos,
                response_tx,
                cancelled,
            } => self.admit(
                prompt,
                options,
                images,
                audio,
                videos,
                response_tx,
                cancelled,
            ),
            ModelRequest::Shutdown => self.shutdown = true,
        }
    }

    /// Pull requests off the channel. When `block` is set (the engine is idle so
    /// there is nothing else to do), block for the first one; then drain any more
    /// that are already queued without blocking.
    fn drain_incoming(&mut self, block: bool) {
        if block {
            match self.request_rx.recv() {
                Ok(req) => self.handle(req),
                // Sender dropped: treat as shutdown.
                Err(_) => {
                    self.shutdown = true;
                    return;
                }
            }
        }
        loop {
            match self.request_rx.try_recv() {
                Ok(req) => self.handle(req),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.shutdown = true;
                    break;
                }
            }
        }
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
                            max_tokens,
                            ..
                        } = state;
                        // Release any tail held back as a potential stop-string
                        // prefix; ending on EOS/length means it never completed
                        // one, so it is real output and must be streamed first.
                        let tail = stop.flush();
                        if !tail.is_empty() {
                            let _ = response_tx.send(GenerateEvent::Token(tail));
                        }
                        detok.flush(&self.tokenizer);
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
                ..
            } = state;
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
    }
}

impl BatchEngine for XlaServeWorker {
    fn serve(&mut self) {
        tracing::info!(
            "OpenXLA serve worker starting (continuous batching, B_max={}, sampling, stop strings)",
            self.engine.b_max()
        );
        loop {
            self.evict_cancelled();

            // If the engine has no work, block for the next request so the thread
            // does not spin; otherwise pick up any newly queued requests and pump.
            let block = self.engine.is_idle() && !self.shutdown;
            self.drain_incoming(block);
            self.evict_cancelled();

            if self.engine.is_idle() {
                if self.shutdown {
                    break;
                }
                // Idle but not shutdown (e.g. everything drained was cancelled or a
                // zero-budget request): loop and block again.
                continue;
            }

            match self.engine.pump() {
                Ok(events) => self.dispatch(events),
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
