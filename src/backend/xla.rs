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

//! OpenXLA / StableHLO compute backend (issue #449, ADR 0004 Track B).
//!
//! Compiled only under the `xla-backend` feature, so Apple-Silicon and CUDA
//! shipping builds never compile it. This is the real non-MLX engine slot the
//! `experimental` scaffold reserved: it produces a [`Session`] backed by
//! [`mlxcel_xla::XlaInferenceSession`], which fills in the engine-neutral
//! `InferenceSession` contract (token-level `prefill` / `decode_step` with
//! on-device sampling).
//!
//! # Why `load_model` is not the entry point here
//!
//! The MLX [`load_model`](ComputeBackend::load_model) boundary returns the
//! concrete MLX [`LoadedModel`] for the server batched-serving path. The OpenXLA
//! backend has no `LoadedModel`: it runs a compiled StableHLO graph and owns its
//! KV and sampling inside the session, so it drives generation through
//! [`create_session`](ComputeBackend::create_session) and the self-contained
//! `prefill` / `decode_step` loop, and `load_model` returns an error pointing the
//! caller at the session path. Threading the model directory and config into
//! session creation (today `create_session` takes only `num_layers`) is part of
//! the control-plane wiring milestone, together with the IREE execution path.

use std::path::Path;

use anyhow::Result;
use mlxcel_core::TokenBiasMap;
use mlxcel_core::cache::KVCacheMode;
use mlxcel_xla::XlaInferenceSession;

use super::{ComputeBackend, Session};
use crate::LoadedModel;
use crate::distributed::ShardConfig;
use crate::tokenizer::MlxcelTokenizer;

/// Handle for the OpenXLA / StableHLO compiler-family backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct XlaBackend;

impl XlaBackend {
    /// Construct the OpenXLA backend handle.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

fn load_unsupported<T>() -> Result<T> {
    anyhow::bail!(
        "the OpenXLA backend drives generation through its self-contained \
         inference session (create_session, then prefill / decode_step), not the \
         MLX load-boundary path; load_model is the MLX batched-serving entry and \
         is not provided by this backend"
    )
}

impl ComputeBackend for XlaBackend {
    fn name(&self) -> &'static str {
        "xla"
    }

    fn load_model(&self, _model_path: &Path) -> Result<(LoadedModel, MlxcelTokenizer)> {
        load_unsupported()
    }

    fn load_model_with_adapter(
        &self,
        _model_path: &Path,
        _adapter_path: &Path,
    ) -> Result<(LoadedModel, MlxcelTokenizer)> {
        load_unsupported()
    }

    fn load_model_with_tensor_parallel(
        &self,
        _model_path: &Path,
        _adapter_path: Option<&Path>,
        _shard_config: &ShardConfig,
    ) -> Result<(LoadedModel, MlxcelTokenizer)> {
        load_unsupported()
    }

    fn create_session(
        &self,
        model_path: &Path,
        num_layers: usize,
        _kv_cache_mode: KVCacheMode,
        _token_bias: TokenBiasMap,
    ) -> Result<Session> {
        // The OpenXLA engine drives generation from the session, so it loads its
        // own weights and config from the model directory here (rather than the
        // MLX `load_model` boundary). Under the `iree` feature this compiles the
        // bundled prefill / decode_step graphs and uploads the weights resident;
        // without it, the session loads but `prefill` / `decode_step` report that
        // the `iree` feature is off. KV mode and token bias do not apply: the
        // session owns its own KV and samples greedily on-device.
        let session =
            XlaInferenceSession::load(model_path, num_layers).map_err(|e| anyhow::anyhow!(e))?;
        Ok(Session::xla(session))
    }

    fn supports_batched_serving(&self) -> bool {
        false
    }
}
