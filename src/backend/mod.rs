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
//! This module introduces the seam and nothing else. A [`ComputeBackend`]
//! abstracts the forward-execution engine at the model-load boundary: it
//! produces the loaded forward executor ([`LoadedModel`], which is an
//! `impl LanguageModel`) for a given model spec. It abstracts the *engine*,
//! not individual ops.
//!
//! # Why the seam sits at model load, not at `forward()`
//!
//! The forward entry boundary (`LanguageModel::forward`, once per token in
//! decode and once per chunk in prefill) is already the right contract: a
//! non-MLX backend implements the *same* `LanguageModel` contract with a
//! different engine behind it. The seam therefore lives one level up, at the
//! point that decides which engine constructs the executor. It does NOT wrap
//! `forward()` itself and is NOT an op-level abstraction. An op-level seam
//! would lose MLX graph fusion and `mx.compile` and add real overhead in the
//! inner loop. The MLX path keeps its concrete hot types (KV cache tensors,
//! paged-KV blocks, prompt-cache detach / adopt) exposed and untouched.
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
//! the existing MLX loader, identical to the pre-seam build. Shipping binaries
//! (Apple Silicon, CUDA) compile no extra backend code because the optional
//! `experimental-backend` module and enum variant are `cfg`-gated off.

use std::path::Path;

use anyhow::Result;

use crate::LoadedModel;
use crate::distributed::ShardConfig;
use crate::tokenizer::MlxcelTokenizer;

pub mod mlx;
pub use mlx::MlxBackend;

#[cfg(feature = "experimental-backend")]
pub mod experimental;

/// The forward-execution engine seam.
///
/// A `ComputeBackend` produces the loaded forward executor for a model spec.
/// The executor is a [`LoadedModel`], which implements the
/// [`LanguageModel`](mlxcel_core::generate::LanguageModel) forward contract, so
/// the backend abstracts *which engine* runs `forward()`, not individual ops.
///
/// The trait is drawn narrowly on purpose: it covers only the load boundary so
/// that the MLX path's specialized hot paths (paged KV, prompt-cache detach /
/// adopt, concrete cache tensors) are never forced through a generic
/// interface. Those stay on the concrete MLX path behind [`MlxBackend`].
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
        }
    }

    /// Load a model from a directory through the active backend.
    #[inline]
    pub fn load_model(&self, model_path: &Path) -> Result<(LoadedModel, MlxcelTokenizer)> {
        match self {
            Backend::Mlx(b) => b.load_model(model_path),
            #[cfg(feature = "experimental-backend")]
            Backend::Experimental(b) => b.load_model(model_path),
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
#[cfg(not(feature = "experimental-backend"))]
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
#[cfg(feature = "experimental-backend")]
#[must_use]
pub fn select_backend() -> Backend {
    // The plug-in point for a future non-MLX backend. Until an engine is wired
    // in (a separate hardware-feasibility gate must precede any kernel work),
    // every selection but the explicit experimental opt-in falls back to MLX.
    match std::env::var("MLXCEL_BACKEND").ok().as_deref() {
        Some("experimental") => Backend::Experimental(experimental::ExperimentalBackend::new()),
        _ => Backend::Mlx(MlxBackend::new()),
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
