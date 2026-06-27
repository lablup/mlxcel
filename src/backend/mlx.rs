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

//! The MLX forward-execution engine behind the [`ComputeBackend`] seam.
//!
//! This is a behavior-preserving wrapper: every method delegates to the
//! existing `crate::loading` entry points unchanged, so the same loading and
//! forward code runs whether a caller reaches it directly or through the seam.
//! No loading logic is reimplemented here. The MLX path keeps all of its
//! concrete hot types and KV optimizations; the seam only routes the load
//! call.

use std::path::Path;

use anyhow::Result;
use mlxcel_core::TokenBiasMap;
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::session::MlxInferenceSession;

use super::{ComputeBackend, Session};
use crate::LoadedModel;
use crate::distributed::ShardConfig;
use crate::loading;
use crate::tokenizer::MlxcelTokenizer;

/// Zero-sized handle to the MLX forward-execution engine.
///
/// Holds no state: MLX device and allocator setup live in the runtime, not the
/// backend. Being zero-sized is what lets [`Backend`](super::Backend) fold to a
/// compile-time constant under default features.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MlxBackend;

impl MlxBackend {
    /// Construct the MLX backend handle.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl ComputeBackend for MlxBackend {
    #[inline]
    fn name(&self) -> &'static str {
        "mlx"
    }

    #[inline]
    fn load_model(&self, model_path: &Path) -> Result<(LoadedModel, MlxcelTokenizer)> {
        loading::load_model(model_path)
    }

    #[inline]
    fn load_model_with_adapter(
        &self,
        model_path: &Path,
        adapter_path: &Path,
    ) -> Result<(LoadedModel, MlxcelTokenizer)> {
        loading::load_model_with_adapter(model_path, adapter_path)
    }

    #[inline]
    fn load_model_with_tensor_parallel(
        &self,
        model_path: &Path,
        adapter_path: Option<&Path>,
        shard_config: &ShardConfig,
    ) -> Result<(LoadedModel, MlxcelTokenizer)> {
        loading::load_model_with_tensor_parallel(model_path, adapter_path, shard_config)
    }

    #[inline]
    fn create_session(
        &self,
        _model_path: &Path,
        num_layers: usize,
        kv_cache_mode: KVCacheMode,
        token_bias: TokenBiasMap,
    ) -> Result<Session> {
        // The MLX engine already loaded the model at `load_model`, so it does not
        // need the model directory here.
        // Wrap the existing `CxxGenerator` (inside `MlxInferenceSession`) with
        // the same KV mode and token bias the CLI used before the seam, so the
        // generation methods delegate verbatim and CLI output is byte-identical.
        Ok(Session::mlx(
            MlxInferenceSession::new_with_kv_mode(num_layers, kv_cache_mode)
                .with_token_bias(token_bias),
        ))
    }

    #[inline]
    fn supports_batched_serving(&self) -> bool {
        // MLX serves batched requests through the server `BatchScheduler` and
        // the retained `load_model` path; that capability is unchanged by the
        // single-sequence session.
        true
    }
}
