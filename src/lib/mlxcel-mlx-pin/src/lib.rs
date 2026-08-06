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

//! Unit-test host for `mlxcel-core`'s build-script MLX pin resolution (#1047).
//!
//! This crate has no production role and nothing depends on it. It exists so
//! the pure logic behind the single-source-of-truth MLX pin (CMake `GIT_TAG`
//! parsing, the `_deps/` staleness decision, and the fetched-HEAD comparison)
//! can be covered by real unit tests without compiling `mlxcel-core`, whose
//! build script builds MLX C++ from source.
//!
//! The module below is the same file `mlxcel-core/build.rs` includes, reached
//! by path rather than copied, so the tested code and the built code cannot
//! drift apart. See `src/lib/mlxcel-mlx-pin/Cargo.toml` for why this is a
//! path include rather than a `[build-dependencies]` entry.

#[path = "../../mlxcel-core/build_support/mlx_pin.rs"]
pub mod mlx_pin;
