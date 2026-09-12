// Copyright 2026 Lablup Inc.
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

//! ROCm rpath emission, shared by every crate that links a ROCm build (#1802).
//!
//! `mlxcel-core` links the ROCm shared libraries, but a dependency's
//! `cargo:rustc-link-arg` does not reach the crates that link it, and ROCm
//! installs do not always register their library directory with the dynamic
//! loader. Without an rpath of their own, the test and binary targets of
//! `mlxcel`, `mlxcel-surgery` and `mlxcel-xla` build and then fail to start
//! with "libamdhip64.so.7: cannot open shared object file".
//!
//! Every such crate calls [`emit`] from its own build script. The default must
//! stay in step with `rocm_path` in `src/lib/mlxcel-core/build.rs`.

use std::env;
use std::path::PathBuf;

/// Add the ROCm library directory to this crate's link arguments when the
/// `rocm` feature is on. A no-op otherwise, so Apple Silicon, CUDA and CI
/// builds are unaffected.
pub fn emit() {
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    if env::var_os("CARGO_FEATURE_ROCM").is_none() {
        return;
    }
    let rocm = env::var_os("ROCM_PATH")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm"));
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        rocm.join("lib").display()
    );
}
