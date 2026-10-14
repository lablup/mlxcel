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

//! The inference-session contract (issue #448, ADR 0004).
//!
//! PR #446 drew the compute-backend seam at the model-load boundary: a backend
//! returned the concrete MLX [`LoadedModel`](crate) and the caller drove
//! generation. ADR 0004 reframes that seam to the inference-session level,
//! because the non-MLX targets we want (FuriosaAI, Tenstorrent, OpenXLA) are
//! graph-compiler backends that keep KV state and sampling inside their own
//! compiled graph and never produce an `MlxArray`.
//!
//! This module hosts the core contract. A backend produces an *inference
//! session* that owns its own per-sequence KV state and runs generation
//! token-in / token-out with on-device sampling. That is the single-sequence
//! contract the CLI `generate` / `chat` paths consume and the contract a future
//! non-MLX backend (issue #449, a separate default-off crate) will implement.
//!
//! Two layers live here:
//!
//! - [`InferenceSession`] is the engine-neutral, object-safe contract. It
//!   exposes the conceptual token-level primitives ([`InferenceSession::prefill`]
//!   and [`InferenceSession::decode_step`]) plus capability advertisement
//!   ([`InferenceSession::capabilities`]). It is the shape a compiler-family
//!   backend fills in.
//! - [`MlxInferenceSession`] is the MLX implementation: a one-sequence client
//!   of the batch-native engine (ADR 0007, #2176). Every generation method
//!   runs the raw-completion client, [`crate::engine::DirectEngine`], over the
//!   borrowed model, so the exact decode loop, prefill plan, sampler and
//!   finish step the server's B=1 row runs are what a caller gets whether it
//!   reaches them through the session, through `mlxcel generate` or through
//!   the server. The per-token forward stays inside the engine (no per-op
//!   virtual dispatch), and the concrete hot types ([`KVCache`](crate::layers::KVCache),
//!   the cache pool) are never type-erased.
//!
//! # Why the MLX session keeps fused entry points, not hand-rolled steps
//!
//! The engine client drives a whole completion (open, prefill, step until the
//! finish step, close) and carries the KV mode, the token bias and the
//! prefill plan with it. Re-expressing that as a hand-written `prefill` +
//! `decode_step` loop would fork it, so [`MlxInferenceSession`] exposes the
//! fused `generate_*` methods and treats [`InferenceSession::prefill`] /
//! [`InferenceSession::decode_step`] as the engine-neutral contract reserved
//! for the compiler-family backend rather than the MLX fast path.
//!
//! # Single sequence only
//!
//! The session guarantees single-sequence generation. Cross-sequence batched
//! serving stays an MLX capability driven by the server `BatchScheduler` through
//! the retained `load_model` path (PR #446), not through this session. The
//! [`SessionCapabilities`] a session advertises lets a caller gate on what a
//! given backend supports; the MLX single-sequence session advertises only the
//! single-sequence and multimodal-prefill capabilities it actually provides.

use crate::cache::KVCacheMode;
use crate::engine::{DirectEngine, DirectRequest, DirectRun};
use crate::ffi::MlxArray;
use crate::generate::{GenerationStats, LanguageModel, SamplingConfig};
use crate::sampling::TokenBiasMap;

// Keep the prepared-prefill DTO discoverable from the session boundary while
// its implementation remains in a focused module below the 500-line file cap.
pub use crate::multimodal::{
    OwnedTensor, PreparedAdapterMode, PreparedAttentionBias, PreparedModality, PreparedPositions,
    PreparedPrefill, PreparedPrefillError, PreparedTensorDType,
};

/// Capabilities a backend inference session advertises so the control plane can
/// gate features a given backend does not support yet.
///
/// These describe what the *session* (single-sequence engine) offers, not what
/// the backend can do through other paths. The MLX backend, for example,
/// supports cross-request batching through the server `BatchScheduler` and the
/// `load_model` path, but its single-sequence [`MlxInferenceSession`] reports
/// `batched_serving == false` because batching is not a property of this
/// session. Backend-level batching support is advertised separately by the
/// compute backend (see `ComputeBackend::supports_batched_serving` in the root
/// crate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SessionCapabilities {
    /// The session can run more than one sequence in a single batched forward.
    pub batched_serving: bool,
    /// The session manages a paged KV block table and pool.
    pub paged_kv: bool,
    /// The session can run speculative (draft + verify) decode internally.
    pub speculative_decode: bool,
    /// The session accepts pre-computed input embeddings (image / audio / video
    /// prefill) in addition to token ids.
    pub multimodal: bool,
}

impl SessionCapabilities {
    /// The conservative floor: a plain single-sequence engine with no batching,
    /// paging, speculative decode, or multimodal prefill. This is the only
    /// capability set the core session contract guarantees; richer backends
    /// turn individual flags on.
    #[must_use]
    pub const fn single_sequence() -> Self {
        Self {
            batched_serving: false,
            paged_kv: false,
            speculative_decode: false,
            multimodal: false,
        }
    }

    /// Advertise owned/native multimodal prefill on a single-sequence session.
    #[must_use]
    pub const fn with_multimodal(mut self) -> Self {
        self.multimodal = true;
        self
    }
}

/// The engine-neutral, single-sequence inference contract (ADR 0004).
///
/// A backend produces a session that owns its per-sequence KV state and runs
/// generation token-in / token-out with sampling done on-device. The contract
/// is intentionally object-safe (no generic method parameters, no closures in
/// signatures) so a future backend can be held behind `Box<dyn
/// InferenceSession>` if cross-crate dispatch ever needs it.
///
/// [`prefill`](Self::prefill) and [`decode_step`](Self::decode_step) define the
/// conceptual token-level primitive shape a compiler-family backend (issue #449)
/// implements directly: prefill the prompt, then advance one token per step with
/// KV and sampling kept inside the backend. The MLX reference session
/// ([`MlxInferenceSession`]) instead drives the CLI through its fused
/// `generate_*` entry points (see the module docs), so on MLX these two methods
/// return a reserved-contract error rather than a parallel decode loop.
pub trait InferenceSession {
    /// What this session can do, so the control plane can gate features.
    fn capabilities(&self) -> SessionCapabilities;

    /// Seed the session's KV state with the prompt tokens (conceptual contract).
    ///
    /// The error type is `String` to keep `mlxcel-core` free of an `anyhow`
    /// dependency, matching the crate's other fallible entry points.
    ///
    /// # Errors
    ///
    /// Backends that do not expose token-level stepping (the MLX reference
    /// session) return an error pointing at the fused generation entry points.
    fn prefill(&mut self, token_ids: &[i32]) -> Result<(), String>;

    /// Seed KV from an owned, validated embeddings payload and return the first
    /// sampled token produced at its effective final position.
    ///
    /// Returning the first token keeps the expanded embedding length distinct
    /// from logical token history: the backend seeds every prepared position
    /// exactly once, then decode begins at `sequence_len`.
    ///
    /// # Errors
    ///
    /// Backends without an owned prepared-prefill entry return an explicit
    /// unsupported error. Implementations must never fall back to token prefill
    /// while silently discarding this payload.
    fn prefill_prepared(&mut self, prepared: &PreparedPrefill) -> Result<i32, String>;

    /// Advance generation by one token given the previously emitted token, and
    /// return the next sampled token id (conceptual contract).
    ///
    /// # Errors
    ///
    /// Backends that do not expose token-level stepping (the MLX reference
    /// session) return an error pointing at the fused generation entry points.
    fn decode_step(&mut self, token: i32) -> Result<i32, String>;
}

/// The MLX single-sequence inference session.
///
/// A one-sequence client of the engine: each generation method opens a
/// [`DirectEngine`] over the borrowed model with this session's KV mode and
/// token bias, runs one completion on a fresh sequence and closes it, so
/// nothing carries over between calls and model-owned state (Gemma 3/4,
/// Llama 4, Qwen 3.5) is reset through the engine's open and close. The
/// methods keep their infallible shape: an engine failure (an MLX throw at the
/// fallible boundary, #822) is reported on stderr and yields the tokens the
/// run had produced, which is none.
pub struct MlxInferenceSession {
    kv_cache_mode: KVCacheMode,
    token_bias: TokenBiasMap,
}

impl MlxInferenceSession {
    /// Create a session with an FP16 KV cache (default). The engine sizes a
    /// sequence's caches from the model, so `_num_layers` is not read; it
    /// stays for the `ComputeBackend::create_session` shape the OpenXLA
    /// session needs.
    #[must_use]
    pub fn new(_num_layers: usize) -> Self {
        Self {
            kv_cache_mode: KVCacheMode::Fp16,
            token_bias: TokenBiasMap::default(),
        }
    }

    /// Create a session with the given KV cache quantization mode (with the
    /// Boundary-V upgrade for the Turbo4 modes, applied per sequence).
    #[must_use]
    pub fn new_with_kv_mode(_num_layers: usize, kv_cache_mode: KVCacheMode) -> Self {
        Self {
            kv_cache_mode,
            token_bias: TokenBiasMap::default(),
        }
    }

    /// Attach a pre-resolved token-bias map, applied to every sampling config
    /// handed to the generation methods unless that config carries its own.
    /// An empty map is a no-op.
    #[must_use]
    pub fn with_token_bias(mut self, bias: TokenBiasMap) -> Self {
        self.token_bias = bias;
        self
    }

    /// The capabilities of the MLX single-sequence session: it does not batch,
    /// page, or speculate internally, but it does accept multimodal embedding
    /// prefill through the `*_with_embeddings` methods.
    #[must_use]
    pub fn capabilities(&self) -> SessionCapabilities {
        SessionCapabilities {
            multimodal: true,
            ..SessionCapabilities::single_sequence()
        }
    }

    /// Kept for callers that reset between runs: every generation method
    /// opens and closes its own engine sequence, so there is nothing to
    /// reset here.
    pub fn reset_with_model<M: LanguageModel + ?Sized>(&mut self, _model: &M) {}

    /// The cached token-bias map (used by tests to assert wiring).
    #[must_use]
    pub fn token_bias(&self) -> &TokenBiasMap {
        &self.token_bias
    }

    /// The KV cache mode every sequence this session opens is built with.
    #[must_use]
    pub fn kv_cache_mode(&self) -> KVCacheMode {
        self.kv_cache_mode
    }

    fn client<'m, M: LanguageModel>(&self, model: &'m M) -> DirectEngine<&'m M> {
        DirectEngine::with_default_chunk(model)
            .with_kv_cache_mode(self.kv_cache_mode)
            .with_token_bias(self.token_bias.clone())
    }

    fn run<M: LanguageModel, F: FnMut(i32) -> bool>(
        &self,
        model: &M,
        request: &DirectRequest<'_>,
        on_token: F,
    ) -> DirectRun {
        self.client(model)
            .generate(request, on_token)
            .unwrap_or_else(|err| {
                eprintln!("generation failed: {err}");
                DirectRun::default()
            })
    }

    /// Greedy / sampled generation.
    pub fn generate<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> Vec<i32> {
        self.generate_streaming(model, prompt_tokens, max_tokens, sampling, |_| true)
    }

    /// Streaming generation with a per-token callback: every token the finish
    /// step appends reaches it, and `false` ends the run.
    pub fn generate_streaming<M: LanguageModel, F: FnMut(i32) -> bool>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        on_token: F,
    ) -> Vec<i32> {
        self.run(
            model,
            &DirectRequest::text(prompt_tokens, max_tokens, sampling),
            on_token,
        )
        .tokens
    }

    /// Streaming generation seeded with pre-computed input embeddings (VLM /
    /// audio prefill).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_embeddings<M: LanguageModel, F: FnMut(i32) -> bool>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        input_embeddings: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        max_tokens: usize,
        sampling: &SamplingConfig,
        on_token: F,
    ) -> Vec<i32> {
        self.run(
            model,
            &DirectRequest {
                prompt_tokens,
                embeddings: input_embeddings,
                mask,
                max_tokens,
                sampling,
            },
            on_token,
        )
        .tokens
    }

    /// Generation with profiling stats (prefill and decode phases as the
    /// engine client times them).
    pub fn generate_with_stats<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> (Vec<i32>, GenerationStats) {
        let run = self.run(
            model,
            &DirectRequest::text(prompt_tokens, max_tokens, sampling),
            |_| true,
        );
        (run.tokens, run.stats)
    }

    /// Embedding-prefill generation with profiling stats.
    pub fn generate_with_stats_and_embeddings<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        input_embeddings: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> (Vec<i32>, GenerationStats) {
        let run = self.run(
            model,
            &DirectRequest {
                prompt_tokens,
                embeddings: input_embeddings,
                mask,
                max_tokens,
                sampling,
            },
            |_| true,
        );
        (run.tokens, run.stats)
    }

    /// Per-target-token log-likelihoods over the prompt window (perplexity
    /// evaluation): one prefill-only scoring pass on the engine under this
    /// session's KV mode ([`DirectEngine::loglikelihoods`]).
    pub fn evaluate_loglikelihoods<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
    ) -> Vec<f32> {
        self.client(model)
            .loglikelihoods(prompt_tokens)
            .unwrap_or_else(|err| {
                eprintln!("log-likelihood evaluation failed: {err}");
                Vec::new()
            })
    }
}

impl InferenceSession for MlxInferenceSession {
    fn capabilities(&self) -> SessionCapabilities {
        MlxInferenceSession::capabilities(self)
    }

    fn prefill(&mut self, _token_ids: &[i32]) -> Result<(), String> {
        Err(RESERVED_STEP_CONTRACT.to_string())
    }

    fn prefill_prepared(&mut self, _prepared: &PreparedPrefill) -> Result<i32, String> {
        Err(
            "MlxInferenceSession does not consume the owned PreparedPrefill contract; use its native MLX embedding-prefill entry points"
                .to_string(),
        )
    }

    fn decode_step(&mut self, _token: i32) -> Result<i32, String> {
        Err(RESERVED_STEP_CONTRACT.to_string())
    }
}

/// Message returned by the MLX session's conceptual token-level primitives.
/// The CLI never hits this path (it drives the fused `generate_*` methods); it
/// documents, for a reader or a future backend author, that `prefill` /
/// `decode_step` are the engine-neutral contract a compiler-family backend
/// (issue #449) fills in rather than the MLX fast path.
const RESERVED_STEP_CONTRACT: &str = "MlxInferenceSession drives generation through its fused generate_* entry \
     points (generate, generate_streaming, generate_with_stats, and the \
     embedding-prefill variants); the token-level prefill / decode_step \
     primitives are the engine-neutral contract reserved for the StableHLO/MLIR \
     compiler-family backend (ADR 0004, issue #449) and are not the MLX path";

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
