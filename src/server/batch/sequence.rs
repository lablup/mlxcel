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

//! Sequence lifecycle state machine and per-request context.
//!
//! [`SequenceState`] models the valid lifecycle:
//!
//! ```text
//! Queued --> Prefilling --> Decoding --> Finished(reason)
//!   ^\           \                         ^
//!   | \           \--- Finished(Cancelled/Error)
//!   |  \----- Finished(Cancelled/Error)
//!   |
//!   +--- (eviction: Decoding -> Queued for re-prefill)
//! ```
//!
//! Any non-terminal state may transition to `Finished(Cancelled)` or
//! `Finished(Error)` so the scheduler can abort sequences when a client
//! disconnects or an error occurs during prefill.
//!
//! When preemptive eviction is enabled, `Decoding -> Queued` is also a
//! valid transition so an evicted sequence can be re-queued for re-prefill.
//!
//! [`SequenceInfo`] bundles every piece of context needed by the batch
//! scheduler and generation loop: prompt tokens, sampling config, VLM
//! embeddings, generated output, response channel, and timing data.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::sampling::{LogprobsConfig, SamplerState, TokenLogprobData};

use super::generation_bounds::{BoundStop, GenerationBounds};
use super::stop_matcher::StopMatcher;
use crate::server::model_provider::model_worker::StreamingDecodeState;
use crate::server::model_provider::{GenerateEvent, PrefillStats, TokenMeta};
use crate::server::thinking_budget::ThinkingState;
use crate::vision::merge::InputEmbeddings;

// ---------------------------------------------------------------------------
// Request priority
// ---------------------------------------------------------------------------

/// Priority level for a generation request.
///
/// Higher-priority requests are prefilled before lower-priority ones.
/// Within the decode batch, all sequences receive equal treatment regardless
/// of priority.
///
/// Set via the `X-Priority` HTTP header (default: `Normal`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum RequestPriority {
    /// Low priority (e.g., batch processing jobs).
    Low = 0,
    /// Default priority for interactive requests.
    #[default]
    Normal = 1,
    /// High priority (e.g., latency-sensitive interactive chat).
    High = 2,
}

impl RequestPriority {
    /// Parse a priority string from an HTTP header value.
    ///
    /// Accepts "high", "normal", "low" (case-insensitive). Returns `None`
    /// for unrecognized values so the caller can fall back to the default.
    pub fn from_header(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "high" => Some(Self::High),
            "normal" => Some(Self::Normal),
            "low" => Some(Self::Low),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

/// Lifecycle state of a single generation sequence.
///
/// Valid transitions:
///
/// ```text
/// Queued --> Prefilling --> Decoding --> Finished(reason)
/// ```
///
/// Additionally, any non-terminal state may transition to
/// `Finished(Cancelled)` or `Finished(Error)` for early abort (e.g. client
/// disconnect, prefill failure).
///
/// The scheduler is the sole authority for driving transitions; HTTP handlers
/// only observe the current state.
#[derive(Debug)]
#[non_exhaustive]
pub enum SequenceState {
    /// Waiting in the prefill queue.
    Queued,
    /// Currently being prefilled (prompt processing).
    Prefilling,
    /// In the active decode batch, generating tokens.
    Decoding,
    /// Generation complete.
    Finished(FinishReason),
}

/// Server-wide context-retention policy (b10621 `--context-shift` /
/// `--keep`, #1472).
///
/// `context_shift` off (the default, upstream's too) makes the resolved KV
/// bound a hard stop: an over-long prompt is refused at admission and a
/// generation that reaches the bound finishes with `truncated: true` and
/// `stop_type: "limit"`. On, the scheduler makes room by discarding tokens
/// after a retained prefix instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ContextRetentionPolicy {
    /// Whether the scheduler may shift (front-discard) a bounded sequence's
    /// context rather than stopping it.
    pub context_shift: bool,
    /// Default retained leading tokens on a shift (`-1` = the whole initial
    /// prompt), overridable per request as `n_keep`.
    pub n_keep: i64,
}

/// Per-sequence resolved context-retention state (#1472).
///
/// Resolved once at enqueue, when the prompt length is known, so the `-1` =
/// keep-the-whole-prompt form and the BOS accounting are already folded in by
/// the time a trim runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SequenceRetention {
    /// Leading tokens retained across a context shift, mirroring b10621's
    /// slot arithmetic: the request's `n_keep` (falling back to the server's
    /// `--keep`), with `-1` resolved to the prompt length and one token added
    /// when the tokenizer prepends BOS. Clamped to `bound - 4` at trim time,
    /// as upstream clamps to `n_ctx - 4`.
    pub keep: i64,
    /// Tokens past `keep` to discard per shift; `0` (upstream's default)
    /// resolves to half of the non-retained window at trim time.
    pub discard: i64,
    /// Set when generation stopped because the next token would not fit the
    /// KV bound with shifting disabled. Reported as `truncated: true` with
    /// `stop_type: "limit"`, b10621's stop-at-the-bound shape.
    pub context_exhausted: bool,
}

/// Why a sequence finished generating.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishReason {
    /// An EOS token was emitted.
    Stop,
    /// A request-supplied string stop sequence matched the decoded text
    /// (issue #1466). Distinct from [`Stop`](Self::Stop) so the response layer
    /// can report b10621's `stop_type: "word"` rather than `"eos"`; the string
    /// that matched is held by the sequence's
    /// [`StopMatcher`](super::stop_matcher::StopMatcher), which is the single
    /// source of truth for it on every serving path.
    StopSequence,
    /// `max_tokens` was reached.
    Length,
    /// The N-gram loop detector tripped: the raw generated stream collapsed
    /// into a short repeated pattern, so generation was ended early. Mapped to
    /// the OpenAI `finish_reason` string `"stop"` (an early, non-error stop,
    /// matching vLLM's behavior) by the completion result builder, which keys
    /// the wire string off `completion_tokens < max_tokens`.
    RepetitionLoop,
    /// The client disconnected or cancelled the request.
    Cancelled,
    /// An internal error occurred during generation.
    Error(String),
}

impl SequenceState {
    /// Attempt to transition from the current state to `next`.
    ///
    /// Returns `Ok(())` if the transition is valid, or `Err` with a message
    /// describing the illegal transition.
    ///
    /// Valid transitions:
    /// - `Queued -> Prefilling` (scheduler begins prefill)
    /// - `Prefilling -> Decoding` (prefill complete, enter decode loop)
    /// - `Prefilling -> Finished(Stop | StopSequence | Length | RepetitionLoop)` (completed on first token)
    /// - `Decoding -> Finished(*)` (normal completion)
    /// - `Decoding -> Queued` (preemptive eviction for re-prefill)
    /// - Any non-terminal state -> `Finished(Cancelled | Error)` (early abort)
    pub fn transition_to(&mut self, next: SequenceState) -> Result<(), String> {
        let valid = match (&*self, &next) {
            // Normal forward progression
            (SequenceState::Queued, SequenceState::Prefilling) => true,
            (SequenceState::Prefilling, SequenceState::Decoding) => true,
            (SequenceState::Prefilling, SequenceState::Finished(FinishReason::Stop)) => true,
            // The first sampled token is decoded during prefill, so a stop
            // string short enough to complete on it finishes the request there.
            (SequenceState::Prefilling, SequenceState::Finished(FinishReason::StopSequence)) => {
                true
            }
            (SequenceState::Prefilling, SequenceState::Finished(FinishReason::Length)) => true,
            (SequenceState::Prefilling, SequenceState::Finished(FinishReason::RepetitionLoop)) => {
                true
            }
            (SequenceState::Decoding, SequenceState::Finished(_)) => true,
            // Preemptive eviction: a decoding sequence can be evicted and
            // re-queued for re-prefill.
            (SequenceState::Decoding, SequenceState::Queued) => true,
            // Early abort: any non-terminal state can transition to
            // Finished(Cancelled) or Finished(Error) so the scheduler can
            // clean up sequences when a client disconnects or an error occurs
            // during prefill.
            (_, SequenceState::Finished(FinishReason::Cancelled | FinishReason::Error(_)))
                if !self.is_finished() =>
            {
                true
            }
            _ => false,
        };

        if valid {
            *self = next;
            Ok(())
        } else {
            Err(format!("invalid state transition: {self:?} -> {next:?}"))
        }
    }

    /// Returns `true` if the sequence has reached a terminal state.
    pub fn is_finished(&self) -> bool {
        matches!(self, SequenceState::Finished(_))
    }
}

// ---------------------------------------------------------------------------
// Per-request context
// ---------------------------------------------------------------------------

/// Full context for a single generation request.
///
/// Created when the HTTP handler enqueues a request and consumed by the batch
/// scheduler through prefill and decode phases. The `response_tx` channel
/// carries streaming `GenerateEvent` values back to the HTTP handler.
pub struct SequenceInfo {
    /// Unique identifier assigned by the `CachePool`.
    pub seq_id: SequenceId,
    /// Resolved context-retention state for this sequence (#1472). See
    /// [`SequenceRetention`].
    pub retention: SequenceRetention,
    /// Current lifecycle state.
    pub state: SequenceState,

    // -- Request context --
    /// Tokenized prompt (integer token IDs).
    pub prompt_tokens: Vec<i32>,
    /// Sampling parameters for this request.
    pub sampling: SamplingConfig,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// EOS token IDs that terminate generation.
    pub eos_token_ids: Vec<i32>,
    /// Request priority for prefill ordering and eviction decisions.
    pub priority: RequestPriority,
    /// Runtime-LoRA scale snapshot taken at admission (#1439). Sequences
    /// with different snapshots never share a forward (b10621's
    /// `can_batch_with` rule); the scheduler applies the executing group's
    /// snapshot to the shared handles before its forwards.
    pub lora_scales: Option<std::sync::Arc<Vec<f32>>>,
    /// Log probability configuration for this request.
    pub logprobs_config: LogprobsConfig,

    // -- VLM / multimodal context (optional) --
    /// Pre-computed vision-language embeddings for VLM requests.
    pub vlm_embeddings: Option<InputEmbeddings>,
    /// Raw image bytes for VLM requests (empty for text-only).
    pub images: Vec<Vec<u8>>,
    /// Raw audio bytes for audio-language models (empty for non-audio).
    pub audio: Vec<Vec<u8>>,

    // -- Generation state --
    /// Tokens produced so far during decode.
    pub generated_tokens: Vec<i32>,
    /// Accumulated decoded text.
    pub generated_text: String,
    /// Streaming decode helper for incremental text emission.
    /// Used by `BatchScheduler` during prefill and decode steps.
    pub(crate) decode_state: StreamingDecodeState,
    /// b10621's per-request generation bounds (#1477): `n_indent` and
    /// `t_max_predict_ms`. Inert, and free, for a request that sends neither.
    pub(crate) bounds: GenerationBounds,
    /// Streaming-safe matcher for the request's string stop sequences
    /// (issue #1466). Every decoded piece passes through it before reaching
    /// `response_tx`, so a piece that could still turn into a stop string is
    /// held back rather than streamed. Inactive
    /// ([`StopMatcher::default`](super::stop_matcher::StopMatcher::default))
    /// when the request supplied no `stop` field, in which case it is an exact
    /// pass-through and the emitted bytes are unchanged.
    pub(crate) stop_matcher: StopMatcher,
    /// Incrementally maintained token history for penalty-based sampling.
    /// Initialized from prompt tokens during prefill, then appended per decode
    /// step. Avoids O(prompt_len + generated_len) Vec reconstruction on every
    /// decode step. Empty when `sampling.needs_token_history()` is false.
    pub(crate) token_history: Vec<i32>,
    /// Incremental per-sequence sampler state for history-based penalties
    /// (repetition, frequency/presence). Created lazily on the first decode
    /// step that applies one of those penalties and dropped with the sequence
    /// (so the default no-penalty path never allocates it). DRY is not
    /// state-backed and keeps reading `token_history` directly. Kept in sync
    /// with `token_history` by `sample_token_optimized_with_state`.
    pub(crate) sampler_state: Option<SamplerState>,
    /// Merged EOS token IDs (model + per-request stop tokens), computed once
    /// during prefill and reused for every decode step. Avoids redundant
    /// allocation on every step.
    pub(crate) merged_eos: Vec<i32>,

    // -- Chunked prefill state --
    /// Token offset into `prompt_tokens` for chunked prefill.
    /// When 0 the prefill has not started; when == `prompt_tokens.len()` it
    /// is complete. The scheduler advances this in increments of
    /// `prefill_chunk_size`.
    pub prefill_offset: usize,

    // -- prompt prefix cache --
    /// Token count from the head of `prompt_tokens` that was already present
    /// in the adopted KV cache. When `> 0`, prefill **skips** those leading
    /// tokens and feeds only `prompt_tokens[prefill_start_offset..]` to the
    /// model. When `0` (the default) the prefill processes the entire prompt
    /// exactly as before the scheduler gained cache-adoption support.
    pub prefill_start_offset: usize,

    /// Mirror of `prefill_start_offset` at adopt time — the counter reported
    /// to observability / `usage.cached_tokens`. Stored separately so that
    /// the prefill path can rewrite `prefill_start_offset` for bookkeeping
    /// (e.g. chunked prefill continuation math) without affecting the stat.
    pub already_cached_tokens: usize,

    // -- thinking-token budget --
    /// Per-sequence thinking-budget state. Drives forced `</think>` injection
    /// when the in-block token count reaches the configured cap. Set to
    /// [`crate::server::thinking_budget::ThinkingState::disabled`] when the
    /// model has no `<think>` / `</think>` tokens, when the request supplies
    /// no budget, or when the server default is unbounded.
    pub(crate) thinking: ThinkingState,

    // -- structured-output constraint --
    /// Optional `llguidance` matcher state for constrained decoding. When
    /// `Some(_)`, the scheduler computes a per-step token mask before sampling
    /// and consumes the sampled token after sampling so the partial output
    /// stays JSON-Schema-conforming. `None` means generation is unconstrained
    /// (the existing behavior preserved bit-for-bit when the request does not
    /// supply `response_format`).
    pub(crate) structured: Option<
        std::sync::Arc<std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>>,
    >,

    // -- Response channel --
    /// Sender for streaming events back to the HTTP handler.
    pub response_tx: mpsc::Sender<GenerateEvent>,

    // -- Cancellation --
    /// Shared flag set to `true` by the SSE sender when the client disconnects.
    /// The `BatchScheduler` checks this flag in `finalize_completed()` and
    /// transitions the sequence to `Finished(Cancelled)`.
    pub cancelled: Arc<AtomicBool>,

    // -- Timing --
    /// Wall-clock time when the request was received.
    pub created_at: Instant,
    /// Wall-clock time when prefill started (set on `Queued -> Prefilling`).
    pub prefill_start: Option<Instant>,
    /// Wall-clock time when the first decode token was produced.
    pub first_token_time: Option<Instant>,
}

impl SequenceInfo {
    /// Convenience: returns the number of generated tokens so far.
    pub fn completion_count(&self) -> usize {
        self.generated_tokens.len()
    }

    /// Returns `true` if this request carries VLM image data.
    pub fn is_vlm_request(&self) -> bool {
        self.vlm_embeddings.is_some() || !self.images.is_empty()
    }

    /// Stamp the first decoded token and publish the prefill snapshot
    /// (issue #1441).
    ///
    /// Called from every path that produces the first token (prefill
    /// completion and the speculative burst driver), and idempotent, so the
    /// snapshot reaches the client exactly once, before the first
    /// [`GenerateEvent::Token`] of the request. `prompt_ms` uses `created_at`
    /// as its origin, which is the same origin
    /// [`take_generation_result`](Self::take_generation_result) later hands to
    /// `finish_with_cache`, so the streamed prefill figures and the final
    /// frame's cannot disagree.
    pub(crate) fn mark_first_token(&mut self) {
        if self.first_token_time.is_some() {
            return;
        }
        let now = Instant::now();
        self.first_token_time = Some(now);
        let _ = self.response_tx.send(GenerateEvent::Prefill(PrefillStats {
            prompt_tokens: self.prompt_tokens.len(),
            cached_tokens: self.already_cached_tokens,
            prompt_ms: now.duration_since(self.created_at).as_millis() as u64,
            processed: self.prompt_tokens.len(),
            first_token: true,
        }));
    }

    /// Publish a mid-prefill progress observation (#1477).
    ///
    /// b10621 sends one `prompt_progress` frame when a slot starts processing
    /// its prompt and one per batch iteration after that, so a client watching
    /// a long prompt sees `processed` climb to `total` before the first content
    /// frame. `processed` counts cache-supplied tokens too, which is why a
    /// fully cached prompt opens at `processed == cache` rather than at zero
    /// (measured on the pinned binary).
    ///
    /// Emitted unconditionally: the native route forwards it only under
    /// `return_progress`, and every other consumer already ignores the variant.
    pub(crate) fn report_prefill_progress(&self, processed: usize) {
        let _ = self.response_tx.send(GenerateEvent::Prefill(PrefillStats {
            prompt_tokens: self.prompt_tokens.len(),
            cached_tokens: self.already_cached_tokens,
            prompt_ms: self.created_at.elapsed().as_millis() as u64,
            processed: processed.min(self.prompt_tokens.len()),
            first_token: false,
        }));
    }

    /// Stream one newly decoded piece to the client through the request's stop
    /// matcher, returning the stop string that matched, if this piece completed
    /// one (issue #1466).
    ///
    /// This is the single funnel every serving path uses to reach `response_tx`
    /// with generated text: classic decode, the fused batched decode, prefill's
    /// first token, and the speculative burst driver. Routing all of them
    /// through one method is what makes the streamed concatenation equal the
    /// non-streaming text, because the held-back ambiguous tail lives in one
    /// place instead of being re-derived per call site.
    ///
    /// With no stop strings configured the matcher is a pass-through, so the
    /// event sent is byte-for-byte the one the call site sent before.
    ///
    /// A caller that receives `Some(word)` must finish the sequence with
    /// [`FinishReason::StopSequence`] and stop feeding tokens.
    /// `token_id` is the token this piece was decoded from, carried out on the
    /// event as [`TokenMeta`] so the native streaming frame can report b10621's
    /// `tokens` and `tokens_predicted` without re-deriving either from the text
    /// (#1477). `None` from a call site with no single token behind the piece.
    pub(crate) fn stream_decoded_text(
        &mut self,
        text: String,
        token_id: Option<i32>,
        logprobs: Option<TokenLogprobData>,
    ) -> Option<String> {
        let meta = TokenMeta {
            token_id,
            decoded: Some(self.generated_tokens.len()),
        };
        // No stop strings: send the piece verbatim, exactly as every call site
        // did before #1466, without building a `StopChunk` per decoded token.
        // This is the overwhelmingly common case on the decode hot path.
        if !self.stop_matcher.is_active() {
            self.bounds.observe(&text, &text);
            let event = match logprobs {
                Some(lp) => GenerateEvent::TokenWithLogprobs(text, meta, lp),
                None => GenerateEvent::Token(text, meta),
            };
            let _ = self.response_tx.send(event);
            return None;
        }

        // b10621 sends one partial frame per decoded token even while a stop
        // string is being matched: that frame carries empty content and the
        // token's own id, and its `tokens_predicted` still counts the token
        // (measured against the pinned binary, #1477). Emitting nothing here
        // would drop the id and make the frame count disagree with the token
        // count. Consumers that accumulate text ignore an empty piece.
        let chunk = self.stop_matcher.push(&text);
        // The generation bounds see the raw decoded text (what upstream appends
        // to `generated_text`) and the emitted part separately (what arms its
        // `has_new_line`); the two differ only mid stop-string match (#1477).
        self.bounds.observe(&text, &chunk.emit);
        let event = match logprobs {
            Some(lp) => GenerateEvent::TokenWithLogprobs(chunk.emit, meta, lp),
            None => GenerateEvent::Token(chunk.emit, meta),
        };
        let _ = self.response_tx.send(event);
        chunk.matched
    }

    /// Whether a b10621 generation bound ended this request on the step just
    /// streamed (#1477).
    ///
    /// Read right after [`stream_decoded_text`](Self::stream_decoded_text) by
    /// every decode path, alongside the stop-string result: a string stop wins
    /// when both land on the same piece, because upstream evaluates the stop
    /// strings first and stops feeding tokens there.
    pub(crate) fn bound_stopped(&self) -> bool {
        self.bounds.fired().is_some()
    }

    /// Close the text stream at the end of generation: push the incremental
    /// detokenizer's held tail (if any) through the stop matcher, then release
    /// whatever the matcher itself was holding.
    ///
    /// The tail can itself complete a stop string, because `StreamingDecodeState`
    /// withholds a piece whose trailing bytes are an incomplete UTF-8 sequence,
    /// so text that completes one can first become visible here. The finish is
    /// then reclassified as [`FinishReason::StopSequence`], which the caller must
    /// not have to remember: this method is the only place that closes the
    /// stream, so folding the reclassification in makes it unmissable.
    ///
    /// A `Cancelled` or `Error` finish is never reclassified. Those outcomes
    /// gate the prompt-cache donate-back and the wire response, and a stop string
    /// that happens to surface while a request is failing does not make it a
    /// healthy stop.
    ///
    /// After a stop string has already matched the matcher is inert, so neither
    /// the tail nor the held text is emitted and the client never sees a byte
    /// past the match.
    pub(crate) fn close_text_stream(&mut self, tail: Option<String>) {
        let tail_matched = tail
            .and_then(|t| self.stream_decoded_text(t, None, None))
            .is_some();
        if !self.stop_matcher.has_matched() {
            let held = self.stop_matcher.flush();
            if !held.is_empty() {
                // A flush spans whatever the matcher was holding, so no single
                // token id describes it; the count is still exact.
                let meta = TokenMeta {
                    token_id: None,
                    decoded: Some(self.generated_tokens.len()),
                };
                let _ = self.response_tx.send(GenerateEvent::Token(held, meta));
            }
        }
        if tail_matched
            && matches!(
                self.state,
                SequenceState::Finished(
                    FinishReason::Stop | FinishReason::Length | FinishReason::RepetitionLoop
                )
            )
        {
            self.state = SequenceState::Finished(FinishReason::StopSequence);
        }
    }

    /// The stop string that ended this request, or `None` when no string stop
    /// sequence matched.
    pub(crate) fn matched_stop(&self) -> Option<&str> {
        self.stop_matcher.matched()
    }

    /// Build the final [`GenerationResult`] for this sequence, taking the
    /// incremental decode state with it.
    ///
    /// When a string stop sequence matched, the accumulated text is truncated to
    /// the bytes that actually reached the client, so the non-streaming `text`
    /// equals the concatenation of the streamed chunks, and the finish is
    /// stamped as [`StopKind::Word`] carrying the matched string. Otherwise this
    /// is exactly the `finish_with_cache` the call sites used before.
    ///
    /// Takes `&mut self` rather than `self` because every call site still needs
    /// the sequence afterwards (prompt-cache donate-back reads `prompt_tokens`
    /// and `generated_tokens`). The decode state is swapped out for an empty
    /// one, which is never read again.
    ///
    /// [`StopKind::Word`]: crate::server::model_provider::StopKind::Word
    pub(crate) fn take_generation_result(
        &mut self,
        tokenizer: &crate::tokenizer::MlxcelTokenizer,
        cached_tokens: usize,
    ) -> crate::server::model_provider::GenerationResult {
        let state = std::mem::replace(
            &mut self.decode_state,
            StreamingDecodeState::new(tokenizer, &[]),
        );
        let prompt_len = self.prompt_tokens.len();
        let mut result = match self.stop_matcher.matched() {
            Some(word) => state.finish_stopped_by_word(
                word.to_string(),
                self.stop_matcher.emitted_len(),
                self.created_at,
                prompt_len,
                self.max_tokens,
                cached_tokens,
            ),
            None => {
                state.finish_with_cache(self.created_at, prompt_len, self.max_tokens, cached_tokens)
            }
        };
        // A context-bound stop (#1472) presents as b10621 presents it: the
        // OpenAI string is "length" and the native pair is `stop_type:
        // "limit"` with `truncated: true`, which the finish helpers cannot
        // derive because the token count sits below `max_tokens`.
        if self.retention.context_exhausted
            && !matches!(
                result.stop_kind,
                crate::server::model_provider::StopKind::Word(_)
            )
        {
            result.stop_kind = crate::server::model_provider::StopKind::ContextExhausted;
            result.finish_reason = "length".to_string();
        }
        // A fired generation bound presents as b10621 presents it: `stop_type:
        // "limit"` and the OpenAI string `"length"`, because the request ran
        // into a bound it was given rather than reaching an end of sequence
        // (#1477). A string stop sequence takes precedence, as upstream's
        // ordering does.
        if !matches!(
            result.stop_kind,
            crate::server::model_provider::StopKind::Word(_)
        ) && let Some(bound) = self.bounds.fired()
        {
            if let BoundStop::Indent { keep_bytes } = bound {
                // Upstream's "cut the last line": the text is erased from the
                // first character after the offending line's leading
                // whitespace. It does NOT re-truncate the stream, which has
                // already sent those bytes, so the two shapes disagree there
                // exactly as upstream's do.
                truncate_result_text(&mut result, keep_bytes);
            }
            result.stop_kind = crate::server::model_provider::StopKind::Limit;
            result.finish_reason = "length".to_string();
        }
        // b10621's `return_tokens` reports the raw sampled ids, including the
        // token that completed a string stop sequence even though its text is
        // excluded from `content` (measured against the pinned binary, #1477).
        // `generated_tokens` is exactly that sequence.
        result.generated_token_ids = self.generated_tokens.clone();
        result
    }
}

/// Truncate a finished result's text to `keep_bytes`, flooring to a character
/// boundary (#1477).
///
/// The bound is computed over the same byte sequence the text carries, so the
/// floor is defensive rather than expected; truncating mid-character would
/// panic and is not worth risking on a stop rule.
fn truncate_result_text(
    result: &mut crate::server::model_provider::GenerationResult,
    keep_bytes: usize,
) {
    if keep_bytes >= result.text.len() {
        return;
    }
    let mut cut = keep_bytes;
    while cut > 0 && !result.text.is_char_boundary(cut) {
        cut -= 1;
    }
    result.text.truncate(cut);
}

// We cannot derive `Debug` automatically because `InputEmbeddings` contains
// `UniquePtr<MlxArray>` which is not `Debug`. A manual implementation keeps
// the struct debuggable in logs.
impl std::fmt::Debug for SequenceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequenceInfo")
            .field("seq_id", &self.seq_id)
            .field("state", &self.state)
            .field("priority", &self.priority)
            .field("prompt_tokens_len", &self.prompt_tokens.len())
            .field("max_tokens", &self.max_tokens)
            .field("generated_tokens_len", &self.generated_tokens.len())
            .field("prefill_offset", &self.prefill_offset)
            .field("is_vlm", &self.is_vlm_request())
            .field("structured", &self.structured.is_some())
            .field("created_at", &self.created_at)
            .field("prefill_start", &self.prefill_start)
            .field("first_token_time", &self.first_token_time)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Scheduler action
// ---------------------------------------------------------------------------

/// Decision emitted by the batch scheduler each tick.
///
/// The scheduler inspects the prefill queue and active batch, then returns one
/// of these actions for the execution engine to carry out.
#[derive(Debug)]
#[non_exhaustive]
pub enum BatchSchedulerAction {
    /// Prefill the next queued request.
    Prefill(SequenceId),
    /// Run one decode step for the listed active sequences.
    Decode(Vec<SequenceId>),
    /// Run one round of the in-flight tick-cooperative speculative slice
    /// (issue #734). Emitted only while a B=1 MTP slice job is parked on
    /// the scheduler; alternates with `Decode` / `Prefill` under
    /// contention so classic rows advance between speculative rounds.
    SpeculativeRound,
    /// Decode the listed active sequences **and** advance the parked chunked
    /// prefill by one chunk, in that order, within a single tick (issue #908
    /// prototype). Emitted only when `MLXCEL_MIXED_STEP` is set; with the
    /// variable unset the scheduler never produces this variant and a chunked
    /// prefill sharing a tick with decode work resolves to `Decode`.
    MixedStep(Vec<SequenceId>),
    /// No work available -- the engine should block until a new request
    /// arrives.
    Idle,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SequenceState transition tests --

    #[test]
    fn valid_transition_queued_to_prefilling() {
        let mut state = SequenceState::Queued;
        assert!(state.transition_to(SequenceState::Prefilling).is_ok());
        assert!(matches!(state, SequenceState::Prefilling));
    }

    #[test]
    fn valid_transition_prefilling_to_decoding() {
        let mut state = SequenceState::Prefilling;
        assert!(state.transition_to(SequenceState::Decoding).is_ok());
        assert!(matches!(state, SequenceState::Decoding));
    }

    #[test]
    fn valid_transition_decoding_to_finished_stop() {
        let mut state = SequenceState::Decoding;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Stop))
                .is_ok()
        );
        assert!(matches!(state, SequenceState::Finished(FinishReason::Stop)));
    }

    #[test]
    fn valid_transition_decoding_to_finished_length() {
        let mut state = SequenceState::Decoding;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Length))
                .is_ok()
        );
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Length)
        ));
    }

    #[test]
    fn valid_transition_decoding_to_finished_repetition_loop() {
        let mut state = SequenceState::Decoding;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::RepetitionLoop))
                .is_ok()
        );
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::RepetitionLoop)
        ));
    }

    #[test]
    fn valid_transition_prefilling_to_finished_repetition_loop() {
        let mut state = SequenceState::Prefilling;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::RepetitionLoop))
                .is_ok()
        );
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::RepetitionLoop)
        ));
    }

    // -- decode-site loop-detection wiring --
    //
    // These replicate the exact guard the scheduler decode sites apply after
    // pushing each token, driving the real `SequenceState` machine so the
    // wiring (detector + transition), not just the pure helper, is exercised.

    fn apply_loop_guard(
        state: &mut SequenceState,
        generated: &[i32],
        cfg: &mlxcel_core::LoopDetectionConfig,
    ) {
        if !state.is_finished()
            && mlxcel_core::detect_repetition_loop(generated, cfg)
            && let Err(err) =
                state.transition_to(SequenceState::Finished(FinishReason::RepetitionLoop))
        {
            panic!("unexpected transition error: {err}");
        }
    }

    #[test]
    fn degenerate_stream_finishes_with_repetition_loop() {
        let cfg = mlxcel_core::LoopDetectionConfig::new(1, 20, 4);
        let mut state = SequenceState::Decoding;
        // Single-token collapse: token 5 repeated four times at the tail.
        apply_loop_guard(&mut state, &[1, 2, 5, 5, 5, 5], &cfg);
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::RepetitionLoop)
        ));
    }

    #[test]
    fn below_threshold_stream_keeps_decoding() {
        let cfg = mlxcel_core::LoopDetectionConfig::new(1, 20, 4);
        let mut state = SequenceState::Decoding;
        // Only three repeats: below min_count, sequence keeps decoding.
        apply_loop_guard(&mut state, &[1, 2, 5, 5, 5], &cfg);
        assert!(matches!(state, SequenceState::Decoding));
    }

    #[test]
    fn disabled_loop_detection_keeps_decoding_on_obvious_loop() {
        let cfg = mlxcel_core::LoopDetectionConfig::disabled();
        let mut state = SequenceState::Decoding;
        apply_loop_guard(&mut state, &[5, 5, 5, 5, 5, 5, 5, 5], &cfg);
        assert!(matches!(state, SequenceState::Decoding));
    }

    #[test]
    fn loop_guard_does_not_override_already_finished_length() {
        let cfg = mlxcel_core::LoopDetectionConfig::new(1, 20, 4);
        // Length limit fired first; the loop guard must not re-transition a
        // finished sequence (the `is_finished()` precheck protects it).
        let mut state = SequenceState::Finished(FinishReason::Length);
        apply_loop_guard(&mut state, &[5, 5, 5, 5], &cfg);
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Length)
        ));
    }

    #[test]
    fn valid_transition_decoding_to_finished_cancelled() {
        let mut state = SequenceState::Decoding;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Cancelled))
                .is_ok()
        );
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Cancelled)
        ));
    }

    #[test]
    fn valid_transition_decoding_to_finished_error() {
        let mut state = SequenceState::Decoding;
        let reason = FinishReason::Error("out of memory".to_string());
        assert!(state.transition_to(SequenceState::Finished(reason)).is_ok());
        match &state {
            SequenceState::Finished(FinishReason::Error(msg)) => {
                assert_eq!(msg, "out of memory");
            }
            other => panic!("expected Finished(Error), got {other:?}"),
        }
    }

    // -- Early cancellation from non-Decoding states --

    #[test]
    fn valid_transition_queued_to_finished_cancelled() {
        let mut state = SequenceState::Queued;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Cancelled))
                .is_ok()
        );
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Cancelled)
        ));
    }

    #[test]
    fn valid_transition_queued_to_finished_error() {
        let mut state = SequenceState::Queued;
        let reason = FinishReason::Error("validation failed".to_string());
        assert!(state.transition_to(SequenceState::Finished(reason)).is_ok());
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Error(_))
        ));
    }

    #[test]
    fn valid_transition_prefilling_to_finished_cancelled() {
        let mut state = SequenceState::Prefilling;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Cancelled))
                .is_ok()
        );
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Cancelled)
        ));
    }

    #[test]
    fn valid_transition_prefilling_to_finished_error() {
        let mut state = SequenceState::Prefilling;
        let reason = FinishReason::Error("OOM during prefill".to_string());
        assert!(state.transition_to(SequenceState::Finished(reason)).is_ok());
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Error(_))
        ));
    }

    // -- Preemptive eviction transition --

    #[test]
    fn valid_transition_decoding_to_queued_for_eviction() {
        let mut state = SequenceState::Decoding;
        assert!(state.transition_to(SequenceState::Queued).is_ok());
        assert!(matches!(state, SequenceState::Queued));
    }

    #[test]
    fn eviction_allows_full_re_lifecycle() {
        // Decoding -> Queued (eviction) -> Prefilling -> Decoding -> Finished
        let mut state = SequenceState::Decoding;
        state.transition_to(SequenceState::Queued).unwrap();
        state.transition_to(SequenceState::Prefilling).unwrap();
        state.transition_to(SequenceState::Decoding).unwrap();
        state
            .transition_to(SequenceState::Finished(FinishReason::Stop))
            .unwrap();
        assert!(state.is_finished());
    }

    // -- Still-invalid transitions --

    #[test]
    fn invalid_transition_queued_to_decoding() {
        let mut state = SequenceState::Queued;
        let result = state.transition_to(SequenceState::Decoding);
        assert!(result.is_err());
        // State should remain unchanged after a failed transition.
        assert!(matches!(state, SequenceState::Queued));
    }

    #[test]
    fn invalid_transition_queued_to_finished_stop() {
        // Queued can only go to Finished via Cancelled or Error, not Stop/Length
        let mut state = SequenceState::Queued;
        let result = state.transition_to(SequenceState::Finished(FinishReason::Stop));
        assert!(result.is_err());
        assert!(matches!(state, SequenceState::Queued));
    }

    #[test]
    fn invalid_transition_queued_to_finished_length() {
        let mut state = SequenceState::Queued;
        let result = state.transition_to(SequenceState::Finished(FinishReason::Length));
        assert!(result.is_err());
        assert!(matches!(state, SequenceState::Queued));
    }

    #[test]
    fn invalid_transition_prefilling_to_queued() {
        let mut state = SequenceState::Prefilling;
        let result = state.transition_to(SequenceState::Queued);
        assert!(result.is_err());
        assert!(matches!(state, SequenceState::Prefilling));
    }

    #[test]
    fn valid_transition_prefilling_to_finished_stop() {
        let mut state = SequenceState::Prefilling;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Stop))
                .is_ok()
        );
        assert!(matches!(state, SequenceState::Finished(FinishReason::Stop)));
    }

    #[test]
    fn valid_transition_prefilling_to_finished_length() {
        let mut state = SequenceState::Prefilling;
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Length))
                .is_ok()
        );
        assert!(matches!(
            state,
            SequenceState::Finished(FinishReason::Length)
        ));
    }

    #[test]
    fn invalid_transition_decoding_to_prefilling() {
        let mut state = SequenceState::Decoding;
        let result = state.transition_to(SequenceState::Prefilling);
        assert!(result.is_err());
        assert!(matches!(state, SequenceState::Decoding));
    }

    #[test]
    fn invalid_transition_finished_to_anything() {
        let mut state = SequenceState::Finished(FinishReason::Stop);
        assert!(state.transition_to(SequenceState::Queued).is_err());
        assert!(state.transition_to(SequenceState::Prefilling).is_err());
        assert!(state.transition_to(SequenceState::Decoding).is_err());
        // Also cannot re-finish (e.g. Finished -> Finished(Cancelled))
        assert!(
            state
                .transition_to(SequenceState::Finished(FinishReason::Cancelled))
                .is_err()
        );
    }

    #[test]
    fn is_finished_returns_correct_values() {
        assert!(!SequenceState::Queued.is_finished());
        assert!(!SequenceState::Prefilling.is_finished());
        assert!(!SequenceState::Decoding.is_finished());
        assert!(SequenceState::Finished(FinishReason::Stop).is_finished());
        assert!(SequenceState::Finished(FinishReason::Length).is_finished());
        assert!(SequenceState::Finished(FinishReason::Cancelled).is_finished());
        assert!(SequenceState::Finished(FinishReason::Error("err".into())).is_finished());
    }

    // -- Full lifecycle test --

    #[test]
    fn full_lifecycle_queued_through_finished() {
        let mut state = SequenceState::Queued;

        state.transition_to(SequenceState::Prefilling).unwrap();
        assert!(matches!(state, SequenceState::Prefilling));

        state.transition_to(SequenceState::Decoding).unwrap();
        assert!(matches!(state, SequenceState::Decoding));

        state
            .transition_to(SequenceState::Finished(FinishReason::Stop))
            .unwrap();
        assert!(state.is_finished());
    }

    // -- FinishReason equality --

    #[test]
    fn finish_reason_equality() {
        assert_eq!(FinishReason::Stop, FinishReason::Stop);
        assert_eq!(FinishReason::Length, FinishReason::Length);
        assert_eq!(FinishReason::Cancelled, FinishReason::Cancelled);
        assert_eq!(
            FinishReason::Error("x".into()),
            FinishReason::Error("x".into())
        );
        assert_ne!(FinishReason::Stop, FinishReason::Length);
        assert_ne!(
            FinishReason::Error("a".into()),
            FinishReason::Error("b".into())
        );
    }

    // -- BatchSchedulerAction --

    #[test]
    fn batch_scheduler_action_debug() {
        // Ensure Debug is implemented (compile-time check via formatting).
        let action = BatchSchedulerAction::Idle;
        let _ = format!("{action:?}");
    }
}
