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

//! Feature-gated scaffold slot for a future non-MLX forward-execution engine.
//!
//! Compiled only under the `experimental-backend` feature, so shipping
//! binaries (Apple Silicon, CUDA) never compile any of it. This module marks
//! the plug-in point where a non-MLX engine (the motivating target is
//! FuriosaAI's TCP / RNGD, which cannot route through the MLX bridge at all)
//! implements [`ComputeBackend`] with its own forward execution, KV cache
//! representation, and weight loading.
//!
//! Issue #338 lands the seam only. No kernels, runtime glue, or weight loading
//! live here. A separate hardware-feasibility gate (go / no-go on real RNGD
//! hardware) must precede any kernel work, so every method here reports that
//! the engine is not implemented yet rather than pretending to load a model.
//! When a real engine arrives it will likely live in its own feature-gated
//! crate and be constructed here.
//!
//! Note for the next implementer (issue #449): the CLI generation path now goes
//! through [`ComputeBackend::create_session`], which returns a [`Session`]
//! wrapping the engine-neutral
//! [`InferenceSession`](mlxcel_core::session::InferenceSession) contract. A
//! genuinely non-MLX engine implements that contract's token-level `prefill` /
//! `decode_step` primitives in its own default-off crate and plugs in here.
//! [`ComputeBackend::load_model`] still returns the concrete [`LoadedModel`] for
//! the server batched path, because the batch scheduler pattern-matches its
//! variants for multimodal dispatch; abstracting that path over a backend-owned
//! KV representation is a later phase (ADR 0004), not part of the first non-MLX
//! session.

use std::path::Path;

use anyhow::Result;
use mlxcel_core::TokenBiasMap;
use mlxcel_core::cache::KVCacheMode;

use super::{ComputeBackend, Session};
use crate::LoadedModel;
use crate::distributed::ShardConfig;
use crate::tokenizer::MlxcelTokenizer;

/// Placeholder handle for the not-yet-implemented non-MLX backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExperimentalBackend;

impl ExperimentalBackend {
    /// Construct the scaffold backend handle.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

fn not_implemented<T>() -> Result<T> {
    anyhow::bail!(
        "the experimental non-MLX compute backend is a seam scaffold only \
         (issue #338 landed the backend boundary, not any engine); no \
         forward-execution engine is wired in yet"
    )
}

impl ComputeBackend for ExperimentalBackend {
    fn name(&self) -> &'static str {
        "experimental"
    }

    fn load_model(&self, _model_path: &Path) -> Result<(LoadedModel, MlxcelTokenizer)> {
        not_implemented()
    }

    fn load_model_with_adapter(
        &self,
        _model_path: &Path,
        _adapter_path: &Path,
    ) -> Result<(LoadedModel, MlxcelTokenizer)> {
        not_implemented()
    }

    fn load_model_with_tensor_parallel(
        &self,
        _model_path: &Path,
        _adapter_path: Option<&Path>,
        _shard_config: &ShardConfig,
    ) -> Result<(LoadedModel, MlxcelTokenizer)> {
        not_implemented()
    }

    fn create_session(
        &self,
        _num_layers: usize,
        _kv_cache_mode: KVCacheMode,
        _token_bias: TokenBiasMap,
    ) -> Result<Session> {
        // No engine is wired in, so the scaffold cannot produce a session. The
        // error mirrors the load-boundary stub: a session is the new contract a
        // real non-MLX engine (issue #449) will satisfy here.
        not_implemented()
    }

    fn supports_batched_serving(&self) -> bool {
        false
    }
}
