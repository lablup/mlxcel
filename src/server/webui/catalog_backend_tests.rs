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

//! Which registry column the catalog reads for each GPU backend (issue #1886).

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};

use super::{backend_status_for, family_for_registry_id, runnable_on_backend};
use crate::models::registry::{BackendStatus, BackendSupport};

use BackendStatus::{Partial, Supported, Unsupported};

#[test]
fn each_backend_kind_reads_its_own_column() {
    // Every column holds a different status, so a kind wired to a neighbour's
    // column returns the wrong value in at least one of the two layouts.
    let layouts = [
        BackendSupport {
            metal: Supported,
            cuda: Partial,
            rocm: Unsupported,
        },
        BackendSupport {
            metal: Unsupported,
            cuda: Supported,
            rocm: Partial,
        },
    ];
    for backends in layouts {
        assert_eq!(
            backend_status_for(GpuBackendKind::Metal, &backends),
            backends.metal
        );
        assert_eq!(
            backend_status_for(GpuBackendKind::Cuda, &backends),
            backends.cuda
        );
        assert_eq!(
            backend_status_for(GpuBackendKind::Rocm, &backends),
            backends.rocm
        );
    }
}

#[test]
fn no_gpu_backend_is_unsupported_even_when_every_column_is_supported() {
    let backends = BackendSupport {
        metal: Supported,
        cuda: Supported,
        rocm: Supported,
    };
    assert_eq!(
        backend_status_for(GpuBackendKind::None, &backends),
        Unsupported
    );
}

#[test]
fn every_gpu_backend_kind_has_a_registry_column() {
    let backends = BackendSupport {
        metal: Supported,
        cuda: Supported,
        rocm: Supported,
    };
    for kind in GpuBackendKind::ALL {
        if kind == GpuBackendKind::None {
            continue;
        }
        assert_eq!(
            backend_status_for(kind, &backends),
            Supported,
            "GpuBackendKind::{kind:?} reads no registry column; add a `BackendSupport` column \
             for it and a `backend_status_for` arm that returns it"
        );
    }
}

#[test]
fn runnable_on_backend_follows_the_resolved_backend() {
    // qwen3 is runnable in every backend column (the registry tests pin that),
    // so the answer depends only on whether a GPU backend resolved. On a Metal
    // host this runs the runtime selection, not only the pure helper above.
    let qwen3 = family_for_registry_id("qwen3").expect("registry has a qwen3 family");
    let kind = gpu_backend_kind();
    assert_eq!(
        runnable_on_backend(&qwen3),
        kind != GpuBackendKind::None,
        "qwen3 runnability on {kind:?}"
    );
    assert_eq!(
        runnable_on_backend(&qwen3),
        matches!(
            backend_status_for(kind, &qwen3.backends),
            Supported | Partial
        )
    );
}
