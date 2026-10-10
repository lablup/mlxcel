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

//! The raw-completion client of the engine (ADR 0007 "Raw completion" row,
//! epic #2166 Phase 6, #2176): one sequence driven through [`Engine::open`],
//! [`Engine::prefill`], [`Engine::step`] and [`Engine::close`] with no
//! scheduler in between. It is what `mlxcel generate` decodes on, what the
//! parity harness's `d:engine` arm runs (#2172), what `mlxcel-bench-decode`
//! times, and what the inference session wraps. There is no other
//! single-sequence decode loop.
//!
//! It uses the same prefill plan, sampler, finish step and merged EOS set the
//! server's B=1 row uses, so a divergence between this client and the
//! server's dense B=1 arm is scheduler policy, not engine execution.
//!
//! After the first token it decodes on the lookahead pipeline, the split
//! step ([`Engine::submit`], [`Engine::finish_rows`], [`Engine::unwind_appends`])
//! with step n+1's forward overlapping step n's host read, whenever the
//! scheduler's eligibility rules admit the sequence or the sequence is a
//! model-owned family's that the close releases (`direct_decode`). A sampler
//! the fused draw cannot run (history penalties, DRY, mirostat, adaptive-p)
//! keeps the forward ahead on the per-row pipeline ([`Engine::submit_forward`],
//! [`Engine::draw_row`], `direct_decode_rows`). Everything else, and every
//! request under `MLXCEL_FORCE_SYNC`, takes the synchronous [`Engine::step`].

use std::borrow::Cow;
use std::time::Instant;

use super::direct_decode::DecodeState;
use super::lookahead::force_sync_requested;
use super::{Engine, EngineError, PrefillStep, SequenceSpec, StepRowHooks};
use crate::cache::{KVCacheMode, SequenceId};
use crate::decode_finish::{FinishCause, FinishHooks};
use crate::ffi::MlxThreadLocalStream;
use crate::generate::{
    GenerationStats, LanguageModel, SamplingConfig, TtftPhases, pad_embeddings,
    prefill_tile_alignment_enabled, resolve_kv_cache_layer_modes, ttft_profile_enabled,
};
use crate::generation_policy::{merged_eos_token_ids, seed_rng_if_needed};
use crate::prefill_plan::{PrefillCaps, PrefillInput, PrefillPlan, prefill_chunk_len};
use crate::sampling::TokenBiasMap;
use crate::sampling_row_step::LogitMask;
use crate::streams::{install_thread_local_default_stream, shared_thread_local_generation_stream};
use crate::{MlxArray, UniquePtr};

/// The vocabulary width of `[1, T, vocab]` logits, checked rather than
/// indexed: a model that returns another rank, a wider batch, or fewer than
/// `positions` positions is reported instead of panicking in a release
/// build. `positions` is how many leading positions the caller reads.
pub(super) fn logits_vocab(shape: &[i32], positions: usize) -> Result<i32, String> {
    match shape {
        &[1, t, vocab] if vocab > 0 && usize::try_from(t).is_ok_and(|t| t >= positions) => {
            Ok(vocab)
        }
        _ => Err(format!(
            "unexpected logits shape {shape:?}, wanted [1, >= {positions}, vocab]"
        )),
    }
}

/// Whether a row outcome's token reaches a streaming callback: the finish
/// step appended it (so not an EOS) and it is not the looping token a
/// repetition loop withholds, which stays in the stream but is not streamed.
///
/// Used by: `DirectEngine::generate`, `DirectEngine::generate_with_drafter`.
pub(super) fn delivers_to_callback(
    generated: &[i32],
    len_before: usize,
    outcome: &super::RowOutcome,
) -> bool {
    generated.len() > len_before && outcome.finish != Some(FinishCause::RepetitionLoop)
}

/// Hooks for a bare sequence: no stop strings, no generation or context
/// bound, no structured-output constraint and no thinking budget, which is
/// what a raw completion carries.
#[derive(Debug, Clone, Copy, Default)]
pub struct BareHooks;

impl FinishHooks for BareHooks {
    fn stop_text(&mut self, _token: i32, _generated_len: usize) -> bool {
        false
    }

    fn bound_stopped(&self) -> bool {
        false
    }

    fn context_bound_due(&self, _generated_len: usize) -> bool {
        false
    }
}

impl StepRowHooks for BareHooks {
    fn logit_mask(&mut self) -> Option<&mut dyn LogitMask> {
        None
    }

    fn override_token(&mut self, sampled: i32) -> i32 {
        sampled
    }

    fn consume_sampled(&mut self, _sampled: i32) -> Result<bool, String> {
        Ok(false)
    }
}

/// Why a raw completion stopped before it finished.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DirectEngineError {
    #[error("open: {0}")]
    Open(EngineError),
    #[error("prefill: {0}")]
    Prefill(EngineError),
    #[error("prefill eval: {0}")]
    PrefillEval(String),
    #[error("prefill trim: {0}")]
    PrefillTrim(String),
    #[error("the prefill plan had no pieces (empty prompt)")]
    EmptyPrompt,
    #[error("first token: {0}")]
    FirstToken(String),
    #[error("step: {0}")]
    Step(EngineError),
    #[error("step: {0}")]
    Row(String),
}

/// One raw completion.
#[derive(Clone, Copy)]
pub struct DirectRequest<'a> {
    pub prompt_tokens: &'a [i32],
    /// Pre-merged input embeddings (a VLM prefill); the prompt is then one
    /// embedding piece, never chunked.
    pub embeddings: Option<&'a MlxArray>,
    /// A caller-supplied prefill mask for the embeddings.
    pub mask: Option<&'a MlxArray>,
    pub max_tokens: usize,
    pub sampling: &'a SamplingConfig,
}

impl<'a> DirectRequest<'a> {
    /// A text completion.
    #[must_use]
    pub fn text(prompt_tokens: &'a [i32], max_tokens: usize, sampling: &'a SamplingConfig) -> Self {
        Self {
            prompt_tokens,
            embeddings: None,
            mask: None,
            max_tokens,
            sampling,
        }
    }
}

/// The tokens a raw completion produced and how long its phases took.
#[derive(Debug, Clone, Default)]
pub struct DirectRun {
    pub tokens: Vec<i32>,
    /// `prefill_time_ms` runs from the first prefill piece through the first
    /// token's evaluation; `decode_time_ms` covers the step loop.
    pub stats: GenerationStats,
}

impl DirectRun {
    /// The run of a zero budget: nothing is prefilled or emitted.
    pub(super) fn empty(prompt_tokens: usize) -> Self {
        Self {
            tokens: Vec::new(),
            stats: GenerationStats {
                prompt_tokens,
                ..GenerationStats::default()
            },
        }
    }
}

/// An [`Engine`] over one model, decoding one request at a time.
pub struct DirectEngine<M: LanguageModel> {
    engine: Engine<M>,
    prefill_chunk: usize,
    kv_cache_mode: KVCacheMode,
    /// The request's effective token bias, applied to every sampling config
    /// handed to [`DirectEngine::generate`] unless that config carries its own
    /// non-empty map (the caller's wins, as it always has).
    token_bias: TokenBiasMap,
    /// The per-thread generation stream the server worker and the CLI share.
    generation_stream: Option<UniquePtr<MlxThreadLocalStream>>,
    /// Keep every decode step on the synchronous chain (the lookahead
    /// pipeline's kill switch, [`super::lookahead::FORCE_SYNC_ENV`], probed
    /// once at construction).
    force_sync: bool,
    /// Print the `[TTFT]` line when `MLXCEL_PROFILE_TTFT` is set; off by
    /// default so a warmup run does not print a cold line of its own.
    report_ttft: bool,
}

impl<M: LanguageModel> DirectEngine<M> {
    /// Wrap `model`; `prefill_chunk` is the piece size of the prefill plan
    /// (the server's `--prefill-chunk-size`, `0` for one forward).
    pub fn new(model: M, prefill_chunk: usize) -> Self {
        Self {
            engine: Engine::with_capacity(model, 1),
            prefill_chunk,
            kv_cache_mode: KVCacheMode::Fp16,
            token_bias: TokenBiasMap::default(),
            generation_stream: shared_thread_local_generation_stream(),
            force_sync: force_sync_requested(),
            report_ttft: false,
        }
    }

    /// Wrap `model` with the one chunk policy (`MLXCEL_PREFILL_CHUNK`, else
    /// [`crate::prefill_plan::DEFAULT_PREFILL_CHUNK`]).
    pub fn with_default_chunk(model: M) -> Self {
        Self::new(model, prefill_chunk_len())
    }

    /// The KV cache quantization mode of every sequence this client opens,
    /// with the Boundary-V upgrade (`MLXCEL_KV_BOUNDARY_V_LAYERS`) for the
    /// Turbo4 modes, applied to the pool caches and injected into a
    /// model-owned family's own caches exactly as the scheduler does.
    #[must_use]
    pub fn with_kv_cache_mode(mut self, mode: KVCacheMode) -> Self {
        self.kv_cache_mode = mode;
        self
    }

    /// Attach the request's token bias (see the field).
    #[must_use]
    pub fn with_token_bias(mut self, bias: TokenBiasMap) -> Self {
        self.token_bias = bias;
        self
    }

    /// Force (`true`) or allow (`false`) the synchronous decode chain,
    /// overriding the `MLXCEL_FORCE_SYNC` probe. The pipelined and the
    /// synchronous loop emit the same stream for every greedy request; see
    /// `direct_decode` for when the pipeline applies.
    #[must_use]
    pub fn with_force_sync(mut self, force_sync: bool) -> Self {
        self.force_sync = force_sync;
        self
    }

    /// Opt this client into the `[TTFT]` line (`MLXCEL_PROFILE_TTFT` must
    /// also be set). `mlxcel generate --profile` and `mlxcel-bench-decode`'s
    /// measured pass set it; their warmups do not.
    #[must_use]
    pub fn with_ttft_report(mut self, report: bool) -> Self {
        self.report_ttft = report;
        self
    }

    /// Whether this run prints the `[TTFT]` line.
    fn ttft_profiled(&self) -> bool {
        self.report_ttft && ttft_profile_enabled()
    }

    pub fn model(&self) -> &M {
        self.engine.model()
    }

    /// Whether every decode step stays on the synchronous chain.
    pub fn force_sync(&self) -> bool {
        self.force_sync
    }

    pub fn kv_cache_mode(&self) -> KVCacheMode {
        self.kv_cache_mode
    }

    pub fn token_bias(&self) -> &TokenBiasMap {
        &self.token_bias
    }

    /// Hand the model back.
    pub fn into_model(self) -> M {
        self.engine.into_parts().0
    }

    pub(super) fn engine(&self) -> &Engine<M> {
        &self.engine
    }

    pub(super) fn engine_mut(&mut self) -> &mut Engine<M> {
        &mut self.engine
    }

    pub(super) fn generation_stream(&self) -> Option<&UniquePtr<MlxThreadLocalStream>> {
        self.generation_stream.as_ref()
    }

    /// Open a sequence under this client's KV mode. The model's own modes go
    /// in before [`Engine::open`]: a model-owned family (Gemma 3/4, Llama 4,
    /// Qwen 3.5, their VLM wrappers) builds the sequence's caches in
    /// `prepare_sequence_state` from the modes it holds at that moment, and
    /// a later `set_kv_cache_layer_modes` rebuilds only its fallback slot.
    pub(super) fn open_sequence(&mut self) -> Result<SequenceId, DirectEngineError> {
        self.inject_model_kv_cache_modes();
        let id = self
            .engine
            .open(SequenceSpec::default())
            .map_err(DirectEngineError::Open)?;
        self.apply_pool_kv_cache_mode(id);
        Ok(id)
    }

    pub(super) fn close_sequence(&mut self, id: SequenceId) {
        self.engine.close(id);
    }

    /// Record a finished prefill in `id`'s pool entry as the scheduler does:
    /// the pool offset is the caller's after a prefill, `prompt_len + 1` once
    /// the first token is sampled, and every continuing step advances it.
    pub(super) fn mark_prefilled(&mut self, id: SequenceId, prompt_len: usize) {
        if let Some(set) = self.engine.pool_mut().get_mut(id) {
            set.prompt_len = prompt_len;
            set.current_offset = prompt_len as i32 + 1;
        }
    }

    /// The fixed work between the first token and the decode loop.
    pub(super) fn prepare_decode(&mut self, id: SequenceId, max_tokens: usize) {
        self.prepare_turbo4_delegated_before_decode(id, max_tokens);
    }

    /// Prefill `prompt_tokens` and decode up to `max_tokens` under
    /// `sampling`, returning the generated tokens.
    pub fn run(
        &mut self,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> Result<Vec<i32>, DirectEngineError> {
        self.generate(
            &DirectRequest::text(prompt_tokens, max_tokens, sampling),
            |_| true,
        )
        .map(|run| run.tokens)
    }

    /// Run one completion: open a sequence, prefill it under the prefill
    /// plan, sample the first token, step until the finish step says so or
    /// `on_token` returns `false`, and close the sequence.
    ///
    /// `on_token` receives every emitted token in order: never an EOS, and
    /// not a repetition loop's looping token (which stays in the returned
    /// stream but is withheld from the callback, as the CLI loop always
    /// did), but the token that spends the budget. The sequence is closed
    /// on every return path, so a failed run leaves no state behind. A zero
    /// `max_tokens` emits nothing and runs no forward.
    pub fn generate<F: FnMut(i32) -> bool>(
        &mut self,
        request: &DirectRequest<'_>,
        on_token: F,
    ) -> Result<DirectRun, DirectEngineError> {
        if request.max_tokens == 0 {
            return Ok(DirectRun::empty(request.prompt_tokens.len()));
        }
        let profile_ttft = self.ttft_profiled();
        let setup_start = Instant::now();
        install_thread_local_default_stream(self.generation_stream.as_ref());
        let sampling = self.compose_sampling(request.sampling);
        let id = self.open_sequence()?;
        let setup_ns = if profile_ttft {
            setup_start.elapsed().as_nanos()
        } else {
            0
        };
        let result = self.run_open_sequence(id, request, &sampling, setup_ns, on_token);
        self.engine.close(id);
        result
    }

    /// The effective sampling config: the caller's own non-empty bias wins,
    /// else the cached map is injected, else the config is borrowed as is.
    pub(super) fn compose_sampling<'a>(
        &self,
        sampling: &'a SamplingConfig,
    ) -> Cow<'a, SamplingConfig> {
        if self.token_bias.is_empty() || !sampling.token_bias.is_empty() {
            Cow::Borrowed(sampling)
        } else {
            let mut cloned = sampling.clone();
            cloned.token_bias = self.token_bias.clone();
            Cow::Owned(cloned)
        }
    }

    /// Hand the model its resolved per-layer modes (a no-op table for
    /// `Fp16`), as the scheduler does at configuration time.
    fn inject_model_kv_cache_modes(&self) {
        let model_modes =
            resolve_kv_cache_layer_modes(self.kv_cache_mode, self.model().num_layers());
        self.model().set_kv_cache_layer_modes(model_modes);
    }

    /// Resolve the per-layer modes for `id`'s pool caches, which the pool
    /// allocates in `Fp16`.
    fn apply_pool_kv_cache_mode(&mut self, id: SequenceId) {
        if self.kv_cache_mode == KVCacheMode::Fp16 {
            return;
        }
        let Some(caches) = self.engine.pool_mut().get_caches_mut(id) else {
            return;
        };
        if caches.is_empty() {
            return;
        }
        let layer_modes = resolve_kv_cache_layer_modes(self.kv_cache_mode, caches.len());
        for (cache, mode) in caches.iter_mut().zip(layer_modes) {
            cache.mode = mode;
        }
    }

    /// Turbo4Delegated caches fold their prefill handoff before decode; a
    /// `max_tokens <= 1` run is prefill-only from the cache's point of view.
    fn prepare_turbo4_delegated_before_decode(&mut self, id: SequenceId, max_tokens: usize) {
        if max_tokens <= 1 {
            return;
        }
        if let Some(caches) = self.engine.pool_mut().get_caches_mut(id) {
            for cache in caches.iter_mut() {
                cache.prepare_turbo4_delegated_for_decode();
            }
        }
    }

    pub(super) fn run_open_sequence<F: FnMut(i32) -> bool>(
        &mut self,
        id: SequenceId,
        request: &DirectRequest<'_>,
        sampling: &SamplingConfig,
        setup_ns: u128,
        mut on_token: F,
    ) -> Result<DirectRun, DirectEngineError> {
        let profile_ttft = self.ttft_profiled();
        let prompt_tokens = request.prompt_tokens;
        let max_tokens = request.max_tokens;
        let eos = merged_eos_token_ids(self.model().eos_token_ids(), &sampling.stop_token_ids);
        let mut state = DecodeState::new(id, prompt_tokens, sampling, eos, max_tokens);

        let prefill_start = Instant::now();
        let build_start = profile_ttft.then(Instant::now);
        let logits = match request.embeddings {
            Some(embeddings) => {
                self.prefill_embeddings(id, prompt_tokens, embeddings, request.mask)?
            }
            None => self.prefill_text(id, prompt_tokens)?,
        };
        let build_ns = build_start.map_or(0, |t| t.elapsed().as_nanos());

        // The server reseeds right before a row's first token is sampled.
        seed_rng_if_needed(sampling);
        let eval_start = profile_ttft.then(Instant::now);
        let first = self.engine.complete_prefill(&logits, &mut state.row());
        let eval_ns = eval_start.map_or(0, |t| t.elapsed().as_nanos());
        self.mark_prefilled(id, prompt_tokens.len());
        let post_start = profile_ttft.then(Instant::now);
        self.prepare_turbo4_delegated_before_decode(id, max_tokens);
        let post_ns = post_start.map_or(0, |t| t.elapsed().as_nanos());
        let prefill_time = prefill_start.elapsed();
        if profile_ttft {
            // The engine samples and evaluates the first token in one
            // per-row chain (`complete_prefill`), so the sampler graph build
            // is reported inside `eval` rather than as its own phase.
            let phases = TtftPhases {
                setup_ns,
                build_ns,
                sample_ns: 0,
                eval_ns,
                post_ns,
            };
            eprintln!(
                "{}",
                phases.format_line(prompt_tokens.len(), prefill_time.as_nanos())
            );
        }
        crate::clear_memory_cache();

        let decode_start = Instant::now();
        if let Some(error) = first.error {
            return Err(DirectEngineError::FirstToken(error.message().to_string()));
        }
        let mut continue_decode = true;
        if delivers_to_callback(&state.generated, 0, &first) {
            continue_decode = on_token(first.token);
        }
        if first.finish.is_none() && continue_decode {
            // Prefill is encoded by now; the decode loop raises the
            // command-buffer input budget around pipelined steps only (see
            // `DecodeCommandBufferBudget` and `direct_decode`).
            self.decode(&mut state, &mut on_token)?;
        }
        let decode_time = decode_start.elapsed();

        let generated = state.generated;
        let prompt_count = prompt_tokens.len();
        let gen_count = generated.len();
        let prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        let decode_ms = decode_time.as_secs_f64() * 1000.0;
        let stats = GenerationStats {
            prompt_tokens: prompt_count,
            generated_tokens: gen_count,
            prefill_time_ms: prefill_ms,
            decode_time_ms: decode_ms,
            prefill_tok_per_sec: if prefill_ms > 0.0 {
                prompt_count as f64 / (prefill_ms / 1000.0)
            } else {
                0.0
            },
            decode_tok_per_sec: if decode_ms > 0.0 {
                gen_count as f64 / (decode_ms / 1000.0)
            } else {
                0.0
            },
        };
        Ok(DirectRun {
            tokens: generated,
            stats,
        })
    }

    /// Every piece of the cold text plan the scheduler would build for this
    /// prompt, returning the `[1, 1, vocab]` logits of its last position.
    pub(super) fn prefill_text(
        &mut self,
        id: SequenceId,
        prompt_tokens: &[i32],
    ) -> Result<UniquePtr<MlxArray>, DirectEngineError> {
        let caps = PrefillCaps::for_model(self.model(), prefill_tile_alignment_enabled());
        let plan = PrefillPlan::new(prompt_tokens.len(), self.prefill_chunk, caps);
        // A model that picks its RoPE table from the whole prompt's length
        // makes that choice once for the prompt, not once per piece.
        let _span = crate::prefill_span::announce(prompt_tokens.len() as i32);
        let mut last_logits = None;
        for piece in plan.pieces() {
            let tokens = &prompt_tokens[piece.range.clone()];
            // A fresh sequence's KV state holds exactly the earlier pieces.
            let (input, pad_mask) =
                super::piece_input(&plan, piece, tokens, piece.range.start as i32);
            let outcome = self.prefill_piece(&PrefillStep {
                seq_id: id,
                input: &input,
                embeddings: None,
                mask: pad_mask.as_deref(),
                last_pos: piece.last_real_pos(),
                trim_excess: piece.trim_after().unwrap_or(0) as i32,
                eval: !plan.is_terminal(piece),
            })?;
            if !plan.is_terminal(piece) {
                // One piece's transients are released before the next
                // piece's graph is built (#672).
                crate::clear_memory_cache();
            }
            last_logits = Some(outcome);
        }
        last_logits.ok_or(DirectEngineError::EmptyPrompt)
    }

    /// The one piece of an embedding-input plan: the pre-merged embeddings at
    /// the real sequence length, padded to the accelerator tile only when the
    /// model opts in and no caller mask fixes the shape.
    fn prefill_embeddings(
        &mut self,
        id: SequenceId,
        prompt_tokens: &[i32],
        embeddings: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>, DirectEngineError> {
        let caps = PrefillCaps::for_model(self.model(), prefill_tile_alignment_enabled())
            .with_input(PrefillInput::Embeddings {
                executor_pads: mask.is_none(),
            });
        let plan = PrefillPlan::new(prompt_tokens.len(), self.prefill_chunk, caps);
        let piece = plan
            .pieces()
            .first()
            .ok_or(DirectEngineError::EmptyPrompt)?;
        debug_assert!(plan.is_single_pass());
        let _span = crate::prefill_span::announce(prompt_tokens.len() as i32);
        let (input, pad_mask) = super::piece_input(&plan, piece, prompt_tokens, 0);
        let padded_embeddings;
        let effective_embeddings = if piece.is_padded() {
            padded_embeddings = pad_embeddings(embeddings, piece.padded_len);
            &*padded_embeddings
        } else {
            embeddings
        };
        self.prefill_piece(&PrefillStep {
            seq_id: id,
            input: &input,
            embeddings: Some(effective_embeddings),
            mask: mask.or(pad_mask.as_deref()),
            last_pos: piece.last_real_pos(),
            trim_excess: piece.trim_after().unwrap_or(0) as i32,
            // The embedding graph is evaluated before `after_prefill` may
            // swap the weights it references (the engine runs that hook).
            eval: true,
        })
    }

    /// Per-target-token log-likelihoods over `prompt_tokens`: entry `i` is
    /// `log P(prompt_tokens[i + 1] | prompt_tokens[..=i])`, so the result has
    /// one entry fewer than the window (and is empty for a window shorter
    /// than two tokens). One prefill-only scoring pass on a fresh sequence
    /// under this client's KV mode, so a Turbo mode measures perplexity with
    /// its V-compression in effect; the sequence is closed afterwards.
    ///
    /// The window is not padded: tile alignment is a decode optimization and
    /// the position-to-target mapping below wants the unpadded shape.
    ///
    /// Used by: `MlxInferenceSession::evaluate_loglikelihoods`, the
    /// wikitext-2 perplexity gates.
    pub fn loglikelihoods(&mut self, prompt_tokens: &[i32]) -> Result<Vec<f32>, DirectEngineError> {
        if prompt_tokens.len() < 2 {
            return Ok(Vec::new());
        }
        install_thread_local_default_stream(self.generation_stream.as_ref());
        let id = self.open_sequence()?;
        let result = self.score_window(id, prompt_tokens);
        self.engine.close(id);
        crate::clear_memory_cache();
        result
    }

    fn score_window(
        &mut self,
        id: SequenceId,
        prompt_tokens: &[i32],
    ) -> Result<Vec<f32>, DirectEngineError> {
        let actual_len = prompt_tokens.len();
        let input = crate::from_slice_i32(prompt_tokens, &[1, actual_len as i32]);
        let _span = crate::prefill_span::announce(actual_len as i32);
        let outcome = self
            .engine
            .score(&PrefillStep {
                seq_id: id,
                input: &input,
                embeddings: None,
                mask: None,
                last_pos: actual_len - 1,
                trim_excess: 0,
                eval: true,
            })
            .map_err(DirectEngineError::Prefill)?;
        outcome.eval.map_err(DirectEngineError::PrefillEval)?;
        outcome.trim.map_err(DirectEngineError::PrefillTrim)?;
        let logits = outcome.logits;

        // `[1, T, vocab]`: slice the context positions `[0, T-1)`, log-softmax
        // in fp32 (fp16 underflows on extreme negative logits) and gather each
        // position's next token.
        let logits_shape = crate::ffi::array_shape(&logits);
        let vocab =
            logits_vocab(&logits_shape, actual_len - 1).map_err(DirectEngineError::PrefillEval)?;
        let context_logits =
            crate::ffi::slice(&logits, &[0, 0, 0], &[1, (actual_len - 1) as i32, vocab]);
        let context_f32 = crate::ffi::astype(&context_logits, crate::dtype::FLOAT32);
        let logprobs = crate::ffi::log_softmax(&context_f32, -1);
        let targets: Vec<i32> = prompt_tokens[1..].to_vec();
        let target_arr = crate::from_slice_i32(&targets, &[1, (actual_len - 1) as i32, 1]);
        let gathered = crate::ffi::take_along_axis(&logprobs, &target_arr, -1);
        crate::try_eval(&gathered).map_err(|e| DirectEngineError::PrefillEval(e.to_string()))?;
        let bytes = crate::ffi::array_to_raw_bytes(&gathered);
        debug_assert_eq!(bytes.len(), (actual_len - 1) * 4);
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn prefill_piece(
        &mut self,
        step: &PrefillStep<'_>,
    ) -> Result<UniquePtr<MlxArray>, DirectEngineError> {
        let outcome = self
            .engine
            .prefill(step)
            .map_err(DirectEngineError::Prefill)?;
        outcome.eval.map_err(DirectEngineError::PrefillEval)?;
        outcome.trim.map_err(DirectEngineError::PrefillTrim)?;
        Ok(outcome.logits)
    }
}
