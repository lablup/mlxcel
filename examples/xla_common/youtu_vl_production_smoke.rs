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

//! Production-path smoke test for the Youtu-VL oracle gate (issue #3364).
//!
//! The 27-stage oracle drives the continuous-batching engine, which compiles
//! the checkpoint's `[out, in]` projection layout. The CLI drives a different
//! engine: the single-sequence [`XlaInferenceSession`], which compiles the
//! transposed `[in, out]` layout and its own prefill and decode modules. That
//! session repeated token 35 forever while the gate passed, so this smoke test
//! loads the session exactly as the CLI does (default precision, the gate's
//! capacity, the production decode loop) and requires its greedy tokens to
//! match the gated ones for the image prompt and for a text-only prompt.

use std::path::Path;

use mlxcel::PreparedPrefill;
use mlxcel_xla::{XlaInferenceSession, YoutuVlReferenceDiagnosticEngine};

/// Text-only prompt of the issue's reproduction.
pub const TEXT_PROMPT: &str = "The capital of France is";

/// Load the production single-sequence session the CLI builds.
///
/// The session selects its device from `MLXCEL_XLA_DEVICE` like the CLI, so
/// the caller must have pointed it at the device the gate runs on.
pub fn load_production_session(
    model: &Path,
    layers: usize,
    context_capacity: usize,
    device: &str,
) -> XlaInferenceSession {
    let session_device = std::env::var("MLXCEL_XLA_DEVICE")
        .unwrap_or_else(|_| mlxcel_xla::default_device().to_string());
    assert_eq!(
        session_device, device,
        "set MLXCEL_XLA_DEVICE to the --device value so the production session \
         runs on the gated device"
    );
    XlaInferenceSession::load_with_context_capacity(model, layers, context_capacity)
        .expect("load the production Youtu-VL OpenXLA session")
}

/// Image prompt: the production prepared-embeddings path must reproduce the
/// HF greedy tokens the gate's engine already matched.
pub fn check_image_prompt(
    session: &mut XlaInferenceSession,
    prepared: &PreparedPrefill,
    hf_tokens: &[i32],
) -> Vec<i32> {
    let tokens = session
        .generate_prepared_greedy(prepared, hf_tokens.len(), &[])
        .expect("production Youtu-VL prepared greedy generation");
    assert_eq!(
        tokens, hf_tokens,
        "production session image-prompt greedy tokens differ from the HF reference"
    );
    tokens
}

/// Text-only prompt, gated half: the batch engine fed the prompt token ids.
/// Run before the production session loads so the two engines' resident
/// weights are never held at once.
pub fn gated_text_tokens(
    engine: &mut YoutuVlReferenceDiagnosticEngine,
    prompt: &[i32],
    max_new_tokens: usize,
) -> Vec<i32> {
    engine
        .greedy_from_tokens(prompt, max_new_tokens)
        .expect("gated Youtu-VL text-only greedy generation")
}

/// Text-only prompt, production half: the session's token-prefill path must
/// match the gated tokens.
pub fn check_text_prompt(
    session: &mut XlaInferenceSession,
    prompt: &[i32],
    gated: &[i32],
) -> Vec<i32> {
    let production = session
        .generate_greedy(prompt, gated.len(), &[])
        .expect("production Youtu-VL text-only greedy generation");
    assert_eq!(
        production, gated,
        "production session text-only greedy tokens differ from the gated engine"
    );
    production
}
