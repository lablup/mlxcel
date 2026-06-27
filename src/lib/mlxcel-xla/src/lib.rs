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

//! OpenXLA / StableHLO compiler-family inference backend (issue #449, ADR 0004
//! Track B). Default-off: the root crate compiles this only under the
//! `xla-backend` feature, so Apple-Silicon and CUDA shipping builds never touch
//! it.
//!
//! This crate hosts [`XlaInferenceSession`], the engine that fills in the
//! engine-neutral [`InferenceSession`](mlxcel_core::session::InferenceSession)
//! contract for the StableHLO/MLIR compiler family. The validated route (issue
//! #449 Phase 0 to 2b) is: a model is authored once as a StableHLO graph (the
//! Rust emitter, issue #451, or the JAX reference), `iree-compile` lowers
//! `prefill` and a single-token `decode_step` to a vmfb, and the IREE runtime
//! executes them with weights resident and sampling (argmax) on-device, so a
//! step returns a token id rather than logits.
//!
//! # Milestone state
//!
//! This is the crate + seam scaffold (Phase 3 M1). It wires the session into the
//! `ComputeBackend` seam and implements the self-contained greedy drive loop
//! ([`XlaInferenceSession::generate_greedy`] /
//! [`XlaInferenceSession::generate_streaming_greedy`]) over the object-safe
//! [`InferenceSession::prefill`] / [`InferenceSession::decode_step`] primitives.
//! The actual graph execution (the IREE runtime C API via FFI, loading the vmfb
//! and uploading weights) is the next milestone, so [`InferenceSession::prefill`]
//! and [`InferenceSession::decode_step`] return [`NOT_WIRED`] for now. Nothing in
//! the default build depends on this crate.

use std::path::{Path, PathBuf};

use mlxcel_core::session::{InferenceSession, SessionCapabilities};

/// Error returned by the scaffold's token-level primitives until the IREE
/// execution path is wired in (Phase 3, the milestone after this one).
pub const NOT_WIRED: &str = "the OpenXLA inference session is a crate and seam scaffold (issue #449 \
     Phase 3 M1); graph execution through the IREE runtime C API is the next \
     milestone, so prefill / decode_step are not bound to an engine yet";

/// The single-sequence OpenXLA inference session.
///
/// Owns its per-sequence KV state and runs generation token-in / token-out with
/// on-device sampling, the shape ADR 0004 reserves for a compiler-family backend.
/// In this milestone it holds the load inputs and the drive loop; the compiled
/// graphs, resident weights, device handles, and live KV land with the execution
/// milestone.
pub struct XlaInferenceSession {
    model_path: PathBuf,
    num_layers: usize,
}

impl XlaInferenceSession {
    /// Prepare a session for a model directory.
    ///
    /// In a later milestone this compiles or loads the exported `prefill` and
    /// `decode_step` vmfbs and uploads the (optionally int4) weights to the
    /// device. For now it records the inputs so the seam wiring is exercisable.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be prepared (none in this
    /// milestone).
    pub fn load(model_path: &Path, num_layers: usize) -> Result<Self, String> {
        Ok(Self {
            model_path: model_path.to_path_buf(),
            num_layers,
        })
    }

    /// Layer count the session was loaded with.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// The model directory backing this session.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Self-contained greedy generation over the object-safe contract.
    ///
    /// Seeds KV with the prompt prefix, then advances one token per step until an
    /// EOS id or the token budget. This is the drive-path the control plane uses
    /// for a backend whose session owns its KV and samples on-device, so it does
    /// not thread an MLX model. Sampling is greedy (argmax inside `decode_step`).
    ///
    /// # Errors
    ///
    /// Propagates the `prefill` / `decode_step` error (currently [`NOT_WIRED`]),
    /// or errors on an empty prompt.
    pub fn generate_greedy(
        &mut self,
        prompt_tokens: &[i32],
        max_new_tokens: usize,
        eos_token_ids: &[i32],
    ) -> Result<Vec<i32>, String> {
        self.generate_streaming_greedy(prompt_tokens, max_new_tokens, eos_token_ids, |_| true)
    }

    /// Self-contained greedy generation with a per-token callback.
    ///
    /// Invokes `on_token` with each newly generated token; returning `false`
    /// stops generation (the token that triggered the stop is still included).
    ///
    /// # Errors
    ///
    /// Propagates the `prefill` / `decode_step` error (currently [`NOT_WIRED`]),
    /// or errors on an empty prompt.
    pub fn generate_streaming_greedy<F: FnMut(i32) -> bool>(
        &mut self,
        prompt_tokens: &[i32],
        max_new_tokens: usize,
        eos_token_ids: &[i32],
        mut on_token: F,
    ) -> Result<Vec<i32>, String> {
        if prompt_tokens.is_empty() {
            return Err("XLA generation requires a non-empty prompt".to_string());
        }
        // Seed KV with all but the last prompt token; the last token is the first
        // input to decode_step, which returns the first generated token. This
        // keeps decode_step's "given the previously emitted token" contract exact
        // and avoids double-counting the last prompt token in the cache.
        let split = prompt_tokens.len() - 1;
        self.prefill(&prompt_tokens[..split])?;
        let mut current = prompt_tokens[split];
        let mut out = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let next = self.decode_step(current)?;
            out.push(next);
            let keep_going = on_token(next);
            if !keep_going || eos_token_ids.contains(&next) {
                break;
            }
            current = next;
        }
        Ok(out)
    }
}

impl InferenceSession for XlaInferenceSession {
    fn capabilities(&self) -> SessionCapabilities {
        SessionCapabilities::single_sequence()
    }

    fn prefill(&mut self, _token_ids: &[i32]) -> Result<(), String> {
        Err(NOT_WIRED.to_string())
    }

    fn decode_step(&mut self, _token: i32) -> Result<i32, String> {
        Err(NOT_WIRED.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> XlaInferenceSession {
        XlaInferenceSession::load(Path::new("/tmp/xla-model"), 16).unwrap()
    }

    #[test]
    fn capabilities_are_single_sequence() {
        let c = session().capabilities();
        assert!(!c.batched_serving);
        assert!(!c.paged_kv);
        assert!(!c.speculative_decode);
        assert!(!c.multimodal);
    }

    #[test]
    fn load_records_inputs() {
        let s = session();
        assert_eq!(s.num_layers(), 16);
        assert_eq!(s.model_path(), Path::new("/tmp/xla-model"));
    }

    #[test]
    fn token_primitives_report_not_wired() {
        let mut s = session();
        assert_eq!(s.prefill(&[1, 2, 3]).unwrap_err(), NOT_WIRED);
        assert_eq!(s.decode_step(1).unwrap_err(), NOT_WIRED);
    }

    #[test]
    fn greedy_surfaces_the_stub_error_not_a_panic() {
        let mut s = session();
        assert_eq!(
            s.generate_greedy(&[1, 2, 3], 8, &[2]).unwrap_err(),
            NOT_WIRED
        );
    }

    #[test]
    fn empty_prompt_is_rejected() {
        let mut s = session();
        assert!(s.generate_greedy(&[], 8, &[2]).is_err());
    }
}
