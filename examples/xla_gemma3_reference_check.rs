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

//! Standalone pinned Gemma3 eager-MLX-CUDA versus IREE-local-task boundary gate.
//!
//! This executable intentionally avoids the Rust libtest harness. It accepts
//! command-line arguments first and falls back to the historical environment
//! variables used by the ignored test.
//!
//! ```text
//! IREE_DIST=/path/to/iree-dist \
//! cargo run --example xla_gemma3_reference_check \
//!   --features cuda,xla-reference-diagnostics -- \
//!   --model /path/to/gemma-3-4b-it-4bit \
//!   --image tests/fixtures/test_image.png \
//!   --device local-task
//! ```

use std::path::PathBuf;

fn argument(flag: &str) -> Option<String> {
    let args = std::env::args().collect::<Vec<_>>();
    args.iter()
        .position(|argument| argument == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn required_path(flag: &str, variable: &str) -> PathBuf {
    argument(flag)
        .or_else(|| std::env::var(variable).ok())
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("missing required {flag} or {variable}"))
}

fn main() {
    let model = required_path("--model", "MLXCEL_GEMMA3_FIXTURE");
    let image = argument("--image")
        .or_else(|| std::env::var("MLXCEL_GEMMA3_IMAGE").ok())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tests/fixtures/test_image.png"));
    let device = argument("--device")
        .or_else(|| std::env::var("MLXCEL_XLA_DEVICE").ok())
        .unwrap_or_else(|| "local-task".to_string());

    mlxcel::run_gemma3_eager_mlx_iree_prepared_boundary(&model, &image, &device);
}
