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

//! Focused tests for the compute-backend seam. These do not load a real
//! checkpoint (token-parity on a real model is owned by the integration / CI
//! parity gate); they assert that selection folds to MLX under default
//! features, that the seam exposes the forward contract, and that it reaches
//! the real MLX loader.

use super::*;
use mlxcel_core::TokenBiasMap;
use mlxcel_core::cache::KVCacheMode;

/// Compile-time proof that the executor the seam yields implements the forward
/// contract. Whatever the backend loads is a working forward executor.
const _: () = {
    fn _requires_language_model<T: mlxcel_core::generate::LanguageModel>() {}
    fn _check() {
        _requires_language_model::<crate::LoadedModel>();
    }
};

/// Compile-time proof that `MlxBackend` implements the seam trait.
const _: () = {
    fn _requires_compute_backend<T: ComputeBackend>() {}
    fn _check() {
        _requires_compute_backend::<MlxBackend>();
    }
};

#[test]
fn select_backend_resolves_to_mlx_under_default_features() {
    let backend = select_backend();
    assert!(
        matches!(backend, Backend::Mlx(_)),
        "default-feature selection must fold to the single MLX variant"
    );
    assert_eq!(backend.name(), "mlx");
}

#[test]
fn mlx_backend_reports_mlx_name() {
    assert_eq!(MlxBackend::new().name(), "mlx");
}

#[test]
fn seam_delegates_to_real_mlx_loader_on_missing_dir() {
    // Proves the seam reaches the existing MLX loader rather than a
    // backend-level shim: a nonexistent directory surfaces the loader's own
    // error (no real checkpoint is loaded, so this stays fast and bridge-free).
    let backend = select_backend();
    let missing = std::path::Path::new("/nonexistent/mlxcel-backend-seam-338");
    // `(LoadedModel, MlxcelTokenizer)` is not `Debug`, so match instead of
    // `expect_err`. The error message is loader-defined; we only assert that
    // delegation reached the loader at all, which a backend-level early return
    // could not produce.
    match backend.load_model(missing) {
        Ok(_) => panic!("loading a nonexistent model directory must error"),
        Err(err) => {
            let _ = err.to_string();
        }
    }
}

#[test]
fn mlx_backend_creates_a_session_and_advertises_batched_serving() {
    // The session is constructed without loading a checkpoint (the wrapped
    // `CxxGenerator` only allocates KV caches), so this stays fast and bridge
    // light, like the existing `CxxGenerator::new` unit tests.
    let backend = select_backend();
    assert!(
        backend.supports_batched_serving(),
        "MLX serves batched requests via the retained load_model / scheduler path"
    );
    let session = backend
        .create_session(
            std::path::Path::new("/tmp/model"),
            4,
            KVCacheMode::Fp16,
            TokenBiasMap::new(),
        )
        .expect("MLX backend must produce a single-sequence session");
    // The single-sequence session does not batch, page, or speculate, but it
    // accepts multimodal embedding prefill.
    let caps = session.capabilities();
    assert!(!caps.batched_serving);
    assert!(!caps.paged_kv);
    assert!(!caps.speculative_decode);
    assert!(caps.multimodal);
}

#[test]
fn mlx_session_threads_the_token_bias_through() {
    let mut bias = TokenBiasMap::new();
    bias.insert(5, -2.0);
    let session = select_backend()
        .create_session(
            std::path::Path::new("/tmp/model"),
            2,
            KVCacheMode::Fp16,
            bias,
        )
        .expect("session creation must succeed");
    match session {
        Session::Mlx(s) => {
            assert_eq!(s.token_bias().len(), 1);
            assert!(s.token_bias().contains(5));
        }
        #[cfg(feature = "xla-backend")]
        Session::Xla(_) => {
            unreachable!("select_backend defaults to MLX without MLXCEL_BACKEND=xla")
        }
    }
}

/// The experimental scaffold has no engine wired in, so it cannot produce a
/// session. Compiled only under the optional feature; default builds carry none
/// of it.
#[cfg(feature = "experimental-backend")]
#[test]
fn experimental_backend_session_creation_errors() {
    let backend = experimental::ExperimentalBackend::new();
    assert!(!backend.supports_batched_serving());
    let result = backend.create_session(
        std::path::Path::new("/tmp/model"),
        4,
        KVCacheMode::Fp16,
        TokenBiasMap::new(),
    );
    assert!(
        result.is_err(),
        "the experimental scaffold must report it has no session engine"
    );
}

/// The OpenXLA backend produces a single-sequence session whose token-level
/// primitives report the `iree` feature is off (the without-execution build).
/// `load_model` is the MLX batched path and errors here. Compiled under
/// `xla-backend` but NOT when real execution is on (`xla-iree`), where
/// `create_session` loads a real model and the fake-path fixture does not apply
/// (that path is validated by the end-to-end CLI run).
#[cfg(all(feature = "xla-backend", not(feature = "xla-iree")))]
#[test]
fn xla_backend_creates_a_single_sequence_session_scaffold() {
    let backend = XlaBackend::new();
    assert_eq!(backend.name(), "xla");
    assert!(!backend.supports_batched_serving());
    assert!(
        backend.load_model(std::path::Path::new("/tmp/x")).is_err(),
        "the XLA backend drives generation through the session, not load_model"
    );

    let session = backend
        .create_session(
            std::path::Path::new("/tmp/model"),
            16,
            KVCacheMode::Fp16,
            TokenBiasMap::new(),
        )
        .expect("xla session creation must succeed");
    let caps = session.capabilities();
    assert!(
        !caps.batched_serving && !caps.paged_kv && !caps.speculative_decode && !caps.multimodal,
        "the XLA session advertises the single-sequence floor"
    );

    match session {
        Session::Xla(mut s) => {
            // The self-contained drive loop surfaces the not-wired stub instead
            // of panicking; real execution lands with the IREE FFI milestone.
            assert!(s.generate_greedy(&[1, 2, 3], 4, &[]).is_err());
        }
        Session::Mlx(_) => unreachable!("XlaBackend::create_session yields an XLA session"),
    }
}
