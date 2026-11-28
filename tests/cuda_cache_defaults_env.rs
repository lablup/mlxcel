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

//! The two CUDA cache defaults reach MLX through the real appliers (#2803).
//!
//! `apply_cuda_graph_cache_default` and `apply_cuda_sdpa_cache_default` set
//! `MLX_CUDA_GRAPH_CACHE_SIZE` and `MLX_CUDA_SDPA_CACHE_SIZE` to 2000 when the
//! operator left them unset, and MLX has to read that value instead of its own
//! defaults of 400 and 256.
//!
//! A separate test binary, because MLX resolves a setting once per process.
//! `mlx::core::config::get` stores the first value it returns for a name, the
//! caller's default included, and answers from that store from then on. The
//! CUDA backend makes that first read itself, with its own default: for the
//! graph cache when a command encoder is constructed (`graph_cache_` in the
//! `device.cpp` overlay), for the attention plan cache at the first cuDNN
//! attention call (`sdpa_cache()`).
//!
//! This check used to live in `mlxcel-core`'s lib suite. The CUDA gate runs
//! that suite single threaded in name order, so `ffi_tests::*` had built the
//! device long before `native_env_tests::*` ran, and the check read 400 on
//! every host with a CUDA device. Here nothing has touched MLX before the
//! appliers run. That is also their contract in production: once, at the top
//! of `main`, before any MLX op.
//!
//! Keep this the only test in the binary: a second one that evaluates an
//! array, or reads either name, would bring the order dependence back.
//!
//! ```sh
//! cargo test --features cuda --test cuda_cache_defaults_env
//! ```

#![cfg(feature = "cuda")]

use mlxcel_core::hardware::{apply_cuda_graph_cache_default, apply_cuda_sdpa_cache_default};
use mlxcel_core::native_env::{mlx_env_int, native_getenv, remove_native_var};

/// What `mlx_env_int` answers for a name that MLX finds unset.
const ABSENT: i32 = -7;

#[test]
fn the_cuda_cache_defaults_reach_mlx() {
    for (name, apply) in [
        (
            "MLX_CUDA_GRAPH_CACHE_SIZE",
            apply_cuda_graph_cache_default as fn(),
        ),
        (
            "MLX_CUDA_SDPA_CACHE_SIZE",
            apply_cuda_sdpa_cache_default as fn(),
        ),
    ] {
        // An exported value would win over the default; the default is what
        // this checks.
        // SAFETY: the only test in this binary, so no other thread reads or
        // writes the environment.
        unsafe { remove_native_var(name) };
        apply();
        assert_eq!(
            native_getenv(name).as_deref(),
            Some(&b"2000"[..]),
            "{name} in the C runtime"
        );
        assert_eq!(mlx_env_int(name, ABSENT), 2000, "{name} as MLX reads it");
    }
}
