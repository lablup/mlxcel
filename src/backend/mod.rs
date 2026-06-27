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

//! Compute-backend seam: abstracts *who* executes `forward()`.
//!
//! Every model in mlxcel implements the [`LanguageModel`] forward contract
//! (`mlxcel-core/src/generate.rs`) and runs that contract through the MLX C++
//! bridge. MLX owns the device abstraction (Metal, CUDA, CPU) and the
//! dynamic-graph executor. Adding an accelerator that *already* has an MLX
//! backend is cheap, but there is no way today to host an execution engine
//! that MLX does not cover at all, where forward computation, KV cache
//! representation, and weight loading happen outside MLX (the motivating
//! target is FuriosaAI's TCP / RNGD, whose `furiosa-opt` toolchain compiles to
//! a virtual ISA and cannot route through MLX).
//!
//! A [`ComputeBackend`] abstracts the forward-execution engine. It has two
//! layers, drawn at the altitudes ADR 0004 settled on (issue #448):
//!
//! - **Session layer (core, single sequence).** A backend produces an
//!   inference [`Session`] that owns its own KV state and runs generation
//!   token-in / token-out with on-device sampling. This is the contract the CLI
//!   `generate` / `chat` paths consume and the one a future non-MLX backend
//!   (issue #449, a separate default-off crate) implements. The MLX session
//!   wraps the existing `CxxGenerator`, so the same decode loop and sampling
//!   run and CLI output stays byte-identical.
//! - **Extended layer (MLX-only, load boundary).** The server batch scheduler
//!   does cross-sequence batched forward and owns [`LoadedModel`] directly, so
//!   the load-boundary entry ([`ComputeBackend::load_model`], returning
//!   `(LoadedModel, MlxcelTokenizer)`) is retained unchanged for
//!   `src/server/model_worker.rs` and the scheduler. Batched serving is treated
//!   as an MLX capability the single-sequence session does not cover yet (the
//!   KV / batching abstraction ADR 0004 defers).
//!
//! # Why the seam abstracts the *engine*, not individual ops
//!
//! An op-level seam (parametrizing every model over a tensor type) would lose
//! MLX graph fusion and `mx.compile` and add real overhead in the inner loop,
//! and it does not fit a graph-compiler backend's whole-graph model. So the
//! session method runs the per-token forward internally and the MLX path keeps
//! its concrete hot types (KV cache tensors, paged-KV blocks, prompt-cache
//! detach / adopt) exposed and untouched.
//!
//! # Codegen equivalence when the optional backend is off (the default)
//!
//! [`Backend`] is an enum that has a *single* variant ([`Backend::Mlx`]) under
//! default features. [`MlxBackend`] is a zero-sized type, so the single-variant
//! enum is itself zero-sized with no discriminant. [`select_backend`] is an
//! `#[inline]` constructor that always returns that one variant with no
//! environment read and no branch, and every [`Backend`] method is a
//! single-arm `match` marked `#[inline]`. After inlining the dispatch folds
//! away entirely: `select_backend().load_model(p)` lowers to a direct call to
//! the existing MLX loader, and `backend.create_session(...).generate(...)`
//! lowers to a direct call into the wrapped `CxxGenerator`, identical to the
//! pre-seam build. The returned [`Session`] is itself a single-variant enum
//! whose per-method `match` collapses the same way, so no runtime indirection
//! is added on the generation hot path. Shipping binaries (Apple Silicon, CUDA)
//! compile no extra backend code because the optional `experimental-backend`
//! module and enum variant are `cfg`-gated off.

use std::path::Path;

use anyhow::Result;
use mlxcel_core::TokenBiasMap;
use mlxcel_core::cache::KVCacheMode;

use crate::LoadedModel;
use crate::distributed::ShardConfig;
use crate::tokenizer::MlxcelTokenizer;

pub mod mlx;
pub use mlx::MlxBackend;

pub mod session;
pub use session::Session;

#[cfg(feature = "experimental-backend")]
pub mod experimental;

#[cfg(feature = "xla-backend")]
pub mod xla;
#[cfg(feature = "xla-backend")]
pub use xla::XlaBackend;

/// The forward-execution engine seam.
///
/// A `ComputeBackend` produces two things: a single-sequence inference
/// [`Session`] for the CLI generation path ([`create_session`](Self::create_session)),
/// and, for the server batched path, the loaded forward executor at the load
/// boundary ([`load_model`](Self::load_model), returning [`LoadedModel`], which
/// implements the [`LanguageModel`](mlxcel_core::generate::LanguageModel)
/// forward contract). The backend abstracts *which engine* runs generation, not
/// individual ops.
///
/// The session method runs the per-token forward internally, and the load
/// boundary keeps returning the concrete `LoadedModel`, so the MLX path's
/// specialized hot paths (paged KV, prompt-cache detach / adopt, concrete cache
/// tensors) are never forced through a generic op interface. Those stay on the
/// concrete MLX path behind [`MlxBackend`].
pub trait ComputeBackend {
    /// Stable identifier for diagnostics and logging (e.g. `"mlx"`).
    fn name(&self) -> &'static str;

    /// Load a model from a directory (or a file, whose parent directory is
    /// used) and return the forward executor plus its tokenizer.
    fn load_model(&self, model_path: &Path) -> Result<(LoadedModel, MlxcelTokenizer)>;

    /// Load a model with LoRA adapter weights fused in.
    fn load_model_with_adapter(
        &self,
        model_path: &Path,
        adapter_path: &Path,
    ) -> Result<(LoadedModel, MlxcelTokenizer)>;

    /// Load a model under a tensor-parallel shard configuration.
    fn load_model_with_tensor_parallel(
        &self,
        model_path: &Path,
        adapter_path: Option<&Path>,
        shard_config: &ShardConfig,
    ) -> Result<(LoadedModel, MlxcelTokenizer)>;

    /// Produce a single-sequence inference [`Session`] for an already-loaded
    /// model (issue #448, ADR 0004).
    ///
    /// The session owns its own KV state and runs generation token-in /
    /// token-out. `num_layers` and `kv_cache_mode` come from the loaded model;
    /// `token_bias` is the pre-resolved language / suppression bias the CLI
    /// threads in (empty is a zero-overhead no-op). A backend with no engine
    /// wired in (the experimental scaffold) returns an error here.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend cannot construct a session.
    fn create_session(
        &self,
        num_layers: usize,
        kv_cache_mode: KVCacheMode,
        token_bias: TokenBiasMap,
    ) -> Result<Session>;

    /// Whether this backend supports cross-sequence batched serving (the server
    /// `BatchScheduler` path). The single-sequence [`Session`] never batches;
    /// this advertises the backend-level capability the extended load-boundary
    /// path provides. MLX returns `true`; a backend with no batched engine
    /// returns `false`.
    fn supports_batched_serving(&self) -> bool;
}

/// The compute backend selected for this process.
///
/// Under default features this enum has a single variant, [`Backend::Mlx`].
/// Because [`MlxBackend`] is zero-sized, `Backend` is zero-sized with no
/// discriminant and every `match` over it collapses to its one arm, so the
/// dispatch added by the seam folds away at compile time. The optional
/// `Experimental` variant is `cfg`-gated and exists only when the
/// `experimental-backend` feature is enabled.
pub enum Backend {
    /// The MLX forward-execution engine: the path every shipping build uses.
    Mlx(MlxBackend),
    /// Scaffold slot for an optional non-MLX engine (issue #338 lands the seam
    /// only; no kernels are wired in here).
    #[cfg(feature = "experimental-backend")]
    Experimental(experimental::ExperimentalBackend),
    /// The OpenXLA / StableHLO compiler-family engine (issue #449), compiled
    /// only under the `xla-backend` feature.
    #[cfg(feature = "xla-backend")]
    Xla(xla::XlaBackend),
}

impl Backend {
    /// Stable identifier for the active backend.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Mlx(b) => b.name(),
            #[cfg(feature = "experimental-backend")]
            Backend::Experimental(b) => b.name(),
            #[cfg(feature = "xla-backend")]
            Backend::Xla(b) => b.name(),
        }
    }

    /// Load a model from a directory through the active backend.
    #[inline]
    pub fn load_model(&self, model_path: &Path) -> Result<(LoadedModel, MlxcelTokenizer)> {
        match self {
            Backend::Mlx(b) => b.load_model(model_path),
            #[cfg(feature = "experimental-backend")]
            Backend::Experimental(b) => b.load_model(model_path),
            #[cfg(feature = "xla-backend")]
            Backend::Xla(b) => b.load_model(model_path),
        }
    }

    /// Load a model with a LoRA adapter through the active backend.
    #[inline]
    pub fn load_model_with_adapter(
        &self,
        model_path: &Path,
        adapter_path: &Path,
    ) -> Result<(LoadedModel, MlxcelTokenizer)> {
        match self {
            Backend::Mlx(b) => b.load_model_with_adapter(model_path, adapter_path),
            #[cfg(feature = "experimental-backend")]
            Backend::Experimental(b) => b.load_model_with_adapter(model_path, adapter_path),
            #[cfg(feature = "xla-backend")]
            Backend::Xla(b) => b.load_model_with_adapter(model_path, adapter_path),
        }
    }

    /// Load a model under a tensor-parallel shard configuration through the
    /// active backend.
    #[inline]
    pub fn load_model_with_tensor_parallel(
        &self,
        model_path: &Path,
        adapter_path: Option<&Path>,
        shard_config: &ShardConfig,
    ) -> Result<(LoadedModel, MlxcelTokenizer)> {
        match self {
            Backend::Mlx(b) => {
                b.load_model_with_tensor_parallel(model_path, adapter_path, shard_config)
            }
            #[cfg(feature = "experimental-backend")]
            Backend::Experimental(b) => {
                b.load_model_with_tensor_parallel(model_path, adapter_path, shard_config)
            }
            #[cfg(feature = "xla-backend")]
            Backend::Xla(b) => {
                b.load_model_with_tensor_parallel(model_path, adapter_path, shard_config)
            }
        }
    }

    /// Produce a single-sequence inference [`Session`] through the active
    /// backend. Under default features the dispatch folds to a direct
    /// [`MlxBackend::create_session`].
    #[inline]
    pub fn create_session(
        &self,
        num_layers: usize,
        kv_cache_mode: KVCacheMode,
        token_bias: TokenBiasMap,
    ) -> Result<Session> {
        match self {
            Backend::Mlx(b) => b.create_session(num_layers, kv_cache_mode, token_bias),
            #[cfg(feature = "experimental-backend")]
            Backend::Experimental(b) => b.create_session(num_layers, kv_cache_mode, token_bias),
            #[cfg(feature = "xla-backend")]
            Backend::Xla(b) => b.create_session(num_layers, kv_cache_mode, token_bias),
        }
    }

    /// Whether the active backend supports cross-sequence batched serving.
    #[inline]
    #[must_use]
    pub fn supports_batched_serving(&self) -> bool {
        match self {
            Backend::Mlx(b) => b.supports_batched_serving(),
            #[cfg(feature = "experimental-backend")]
            Backend::Experimental(b) => b.supports_batched_serving(),
            #[cfg(feature = "xla-backend")]
            Backend::Xla(b) => b.supports_batched_serving(),
        }
    }
}

/// Select the compute backend for this process.
///
/// Under default features this is a compile-time constant: it always returns
/// the single [`Backend::Mlx`] variant with no environment read and no branch,
/// so the call inlines to a direct [`MlxBackend`] construction and the seam
/// adds no runtime dispatch on the load path or, transitively, the hot
/// `forward()` path.
#[cfg(not(any(feature = "experimental-backend", feature = "xla-backend")))]
#[inline]
#[must_use]
pub fn select_backend() -> Backend {
    Backend::Mlx(MlxBackend::new())
}

/// Select the compute backend for this process.
///
/// With the `experimental-backend` feature enabled, an optional non-MLX engine
/// can be selected at runtime (e.g. via an environment switch). This selection
/// logic is compiled only under the feature, so default builds carry none of
/// it. The seam currently has no non-MLX engine wired in, so any non-MLX
/// selection still resolves to a scaffold that reports it is not implemented.
#[cfg(any(feature = "experimental-backend", feature = "xla-backend"))]
#[must_use]
pub fn select_backend() -> Backend {
    // The plug-in point for the optional non-MLX backends. Each opt-in arm is
    // compiled only under its own feature; with neither feature this function is
    // the const MLX form above. Any selection but an explicit opt-in falls back
    // to MLX.
    match std::env::var("MLXCEL_BACKEND").ok().as_deref() {
        #[cfg(feature = "experimental-backend")]
        Some("experimental") => Backend::Experimental(experimental::ExperimentalBackend::new()),
        #[cfg(feature = "xla-backend")]
        Some("xla") => Backend::Xla(xla::XlaBackend::new()),
        _ => Backend::Mlx(MlxBackend::new()),
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
