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

// Build script for `mlxcel-surgery`. It exists only to emit the ROCm rpath
// (issue #1802) and does nothing on any other build: this crate is pure Rust
// and needs no native toolchain.
//
// Its test binaries link mlxcel-core, which links the ROCm shared libraries,
// and a dependency's `cargo:rustc-link-arg` does not reach the crate that links
// it. Without this the test binaries build and then fail to start with
// "libamdhip64.so.7: cannot open shared object file".

#[path = "../mlxcel-core/build_support/rocm_rpath.rs"]
mod rocm_rpath;

fn main() {
    rocm_rpath::emit();
}
