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

//! Whether the CUDA JIT PTX disk cache's test-only hooks are compiled in
//! (issue #2765).
//!
//! The `jit_module.cpp` overlay reads `MLXCEL_PTX_CACHE_TEST_TMP_SUFFIX` so
//! `tests/cuda_ptx_cache_trust.rs` can plant a file at the exact path a
//! publish opens. A shipped binary has no use for it, so `build.rs` compiles
//! it in only for the Cargo profiles tests run under and tells CMake through
//! `MLXCEL_PTX_CACHE_TEST_HOOKS`.
//!
//! The profile is read from the build script's `OUT_DIR`, which is
//! `<target>/[<triple>/]<profile>/build/<package>-<hash>/out`. Neither of the
//! obvious signals works: Cargo sets `PROFILE` to `release` for every profile
//! that inherits from it (`test-fast` included), and `test-fast` turns
//! `debug_assertions` off. A Cargo feature would work but changes the
//! package's metadata hash and with it `OUT_DIR`, so alternating test and
//! binary builds would rebuild MLX from scratch each time.
//!
//! Included by path from `build.rs` and from `mlxcel-mlx-pin`, which runs the
//! unit tests without building MLX.

use std::path::Path;

/// The Cargo profile directories whose builds compile the hooks in: `debug`
/// (the `dev` and `test` profiles) and `test-fast` (`make verify-test*` and
/// CI's test gates). Everything else, `release` above all, leaves them out.
pub const PROFILES_WITH_TEST_HOOKS: &[&str] = &["debug", "test-fast"];

/// The profile directory name `out_dir` sits in, or `None` when `out_dir`
/// does not have the `<profile>/build/<package>-<hash>/out` layout.
pub fn profile_dir_name(out_dir: &Path) -> Option<&str> {
    let mut ancestors = out_dir.ancestors();
    let out = ancestors.next()?;
    let _package = ancestors.next()?;
    let build = ancestors.next()?;
    let profile = ancestors.next()?;
    if out.file_name()? != "out" || build.file_name()? != "build" {
        return None;
    }
    profile.file_name()?.to_str()
}

/// Whether a build whose script runs with `out_dir` as `OUT_DIR` compiles the
/// PTX cache test hooks in. False for any profile not listed in
/// [`PROFILES_WITH_TEST_HOOKS`] and for a layout this does not recognize.
pub fn test_hooks_enabled(out_dir: &Path) -> bool {
    profile_dir_name(out_dir).is_some_and(|profile| PROFILES_WITH_TEST_HOOKS.contains(&profile))
}

#[cfg(test)]
#[path = "ptx_cache_test_hooks_tests.rs"]
mod tests;
