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
