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

//! The root-crate inference-session seam (issue #448, ADR 0004).
//!
//! [`Session`] is the value a [`ComputeBackend`](super::ComputeBackend)
//! produces. It mirrors the [`Backend`](super::Backend) selection enum: a single
//! [`Session::Mlx`] variant under default features, so every `match` over it
//! collapses to its one arm. Each generation method is a single-arm `match`
//! marked `#[inline]`, so the session dispatch folds away entirely at compile
//! time and the CLI `generate` / `chat` paths call the wrapped
//! [`MlxInferenceSession`] (and through it the existing `CxxGenerator`) with no
//! runtime indirection added on the hot path. The per-token forward stays inside
//! the session method; the concrete KV types are never type-erased.
//!
//! # Extension point for the non-MLX backend (issue #449)
//!
//! The compiler-family backend (OpenXLA / StableHLO, ADR 0004 Track B) lands in
//! its own default-off crate. When it does, it implements the engine-neutral
//! [`InferenceSession`](mlxcel_core::session::InferenceSession) contract in
//! `mlxcel-core`, and this enum gains a `cfg`-gated variant wrapping it (or, if
//! cross-crate generics over the streaming closure prove awkward, this seam
//! becomes a `Box<dyn InferenceSession>` boundary that drives the object-safe
//! `prefill` / `decode_step` primitives). It is left as a single MLX variant
//! today because there is no second backend to wrap yet; over-engineering the
//! dispatch now would add cost the default build pays for nothing.

use mlxcel_core::MlxArray;
use mlxcel_core::generate::{GenerationStats, LanguageModel, SamplingConfig};
#[cfg(feature = "xla-backend")]
use mlxcel_core::session::InferenceSession;
use mlxcel_core::session::{MlxInferenceSession, SessionCapabilities};
#[cfg(feature = "xla-backend")]
use mlxcel_xla::XlaInferenceSession;

/// The inference session selected for a load.
///
/// Under default features this enum has a single variant, [`Session::Mlx`].
/// Because the dispatch is a single-arm `match`, the seam adds no runtime
/// indirection: `session.generate(...)` inlines to the wrapped
/// [`MlxInferenceSession::generate`], which delegates verbatim to the existing
/// `CxxGenerator`.
pub enum Session {
    /// The MLX single-sequence session, wrapping `CxxGenerator`.
    Mlx(MlxInferenceSession),
    /// The OpenXLA / StableHLO compiler-family session (issue #449), compiled
    /// only under the `xla-backend` feature. It owns its KV and samples
    /// on-device, so it self-drives the engine-neutral prefill / decode_step
    /// contract and ignores the threaded MLX model and sampling config (greedy).
    #[cfg(feature = "xla-backend")]
    Xla(XlaInferenceSession),
}

impl Session {
    /// Wrap an [`MlxInferenceSession`] as the active session.
    #[inline]
    #[must_use]
    pub fn mlx(session: MlxInferenceSession) -> Self {
        Session::Mlx(session)
    }

    /// Wrap an [`XlaInferenceSession`] as the active session.
    #[cfg(feature = "xla-backend")]
    #[inline]
    #[must_use]
    pub fn xla(session: XlaInferenceSession) -> Self {
        Session::Xla(session)
    }

    /// What this session can do, so the control plane can gate features.
    #[inline]
    #[must_use]
    pub fn capabilities(&self) -> SessionCapabilities {
        match self {
            Session::Mlx(s) => s.capabilities(),
            #[cfg(feature = "xla-backend")]
            Session::Xla(s) => s.capabilities(),
        }
    }

    /// Reset generator-owned and model-owned caches for a fresh prefill.
    #[inline]
    pub fn reset_with_model<M: LanguageModel + ?Sized>(&mut self, model: &M) {
        match self {
            Session::Mlx(s) => s.reset_with_model(model),
            #[cfg(feature = "xla-backend")]
            Session::Xla(_s) => {
                // The XLA session resets its own KV on the next prefill; the MLX
                // model is not used by this engine.
            }
        }
    }

    /// Greedy / sampled generation.
    #[inline]
    pub fn generate<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> Vec<i32> {
        match self {
            Session::Mlx(s) => s.generate(model, prompt_tokens, max_tokens, sampling),
            // The XLA session self-drives via prefill / decode_step and ignores
            // the MLX model and the sampling config (greedy). It surfaces an
            // empty result until the engine is wired in; the control plane uses
            // the object-safe contract directly for error-visible paths.
            #[cfg(feature = "xla-backend")]
            Session::Xla(s) => s
                .generate_greedy(prompt_tokens, max_tokens, &[])
                .unwrap_or_default(),
        }
    }

    /// Streaming generation with a per-token callback (the closure stays
    /// generic, preserving lookahead pipelining).
    #[inline]
    pub fn generate_streaming<M: LanguageModel, F: FnMut(i32) -> bool>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        on_token: F,
    ) -> Vec<i32> {
        match self {
            Session::Mlx(s) => {
                s.generate_streaming(model, prompt_tokens, max_tokens, sampling, on_token)
            }
            #[cfg(feature = "xla-backend")]
            Session::Xla(s) => s
                .generate_streaming_greedy(prompt_tokens, max_tokens, &[], on_token)
                .unwrap_or_default(),
        }
    }

    /// Streaming generation seeded with pre-computed input embeddings (VLM /
    /// audio prefill).
    #[inline]
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
        match self {
            Session::Mlx(s) => s.generate_streaming_with_embeddings(
                model,
                prompt_tokens,
                input_embeddings,
                mask,
                max_tokens,
                sampling,
                on_token,
            ),
            // The XLA session is single-sequence text; it has no multimodal
            // capability, so it ignores the input embeddings and drives the token
            // stream from the prompt ids.
            #[cfg(feature = "xla-backend")]
            Session::Xla(s) => s
                .generate_streaming_greedy(prompt_tokens, max_tokens, &[], on_token)
                .unwrap_or_default(),
        }
    }

    /// Generation with profiling stats.
    #[inline]
    pub fn generate_with_stats<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> (Vec<i32>, GenerationStats) {
        match self {
            Session::Mlx(s) => s.generate_with_stats(model, prompt_tokens, max_tokens, sampling),
            #[cfg(feature = "xla-backend")]
            Session::Xla(s) => (
                s.generate_greedy(prompt_tokens, max_tokens, &[])
                    .unwrap_or_default(),
                GenerationStats::default(),
            ),
        }
    }

    /// Embedding-prefill generation with profiling stats.
    #[inline]
    pub fn generate_with_stats_and_embeddings<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        input_embeddings: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> (Vec<i32>, GenerationStats) {
        match self {
            Session::Mlx(s) => s.generate_with_stats_and_embeddings(
                model,
                prompt_tokens,
                input_embeddings,
                mask,
                max_tokens,
                sampling,
            ),
            #[cfg(feature = "xla-backend")]
            Session::Xla(s) => (
                s.generate_greedy(prompt_tokens, max_tokens, &[])
                    .unwrap_or_default(),
                GenerationStats::default(),
            ),
        }
    }

    /// Per-target-token log-likelihoods over the prompt window.
    #[inline]
    pub fn evaluate_loglikelihoods<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
    ) -> Vec<f32> {
        match self {
            Session::Mlx(s) => s.evaluate_loglikelihoods(model, prompt_tokens),
            // Perplexity evaluation is not part of the single-sequence XLA
            // session; it returns no per-token scores.
            #[cfg(feature = "xla-backend")]
            Session::Xla(_s) => Vec::new(),
        }
    }
}
