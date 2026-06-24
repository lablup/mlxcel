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
use mlxcel_core::sampling::{LogprobsConfig, SamplerState};

use crate::server::model_provider::GenerateEvent;
use crate::server::model_provider::model_worker::StreamingDecodeState;
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

/// Why a sequence finished generating.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishReason {
    /// An EOS token was emitted.
    Stop,
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
    /// - `Prefilling -> Finished(Stop | Length | RepetitionLoop)` (completed on first token)
    /// - `Decoding -> Finished(*)` (normal completion)
    /// - `Decoding -> Queued` (preemptive eviction for re-prefill)
    /// - Any non-terminal state -> `Finished(Cancelled | Error)` (early abort)
    pub fn transition_to(&mut self, next: SequenceState) -> Result<(), String> {
        let valid = match (&*self, &next) {
            // Normal forward progression
            (SequenceState::Queued, SequenceState::Prefilling) => true,
            (SequenceState::Prefilling, SequenceState::Decoding) => true,
            (SequenceState::Prefilling, SequenceState::Finished(FinishReason::Stop)) => true,
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
