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
//! Phase 3 M2 wires real execution behind the `iree` feature: [`prefill`] seeds
//! the KV cache through the bucketed prefill graph and [`decode_step`] advances
//! one token through the decode graph, both driven by the IREE runtime C API via
//! the [`iree`] module's FFI shim, with the model weights resident on the device
//! and the next-token argmax computed on-device. The drive loop
//! ([`XlaInferenceSession::generate_greedy`] /
//! [`generate_streaming_greedy`](XlaInferenceSession::generate_streaming_greedy))
//! sits on top of those object-safe primitives.
//!
//! Without the `iree` feature the crate is pure Rust (no native toolchain, no
//! IREE distribution needed, so CI builds `--features xla-backend` unchanged) and
//! `prefill` / `decode_step` return [`NOT_WIRED`]. Nothing in the default build
//! depends on this crate.
//!
//! [`prefill`]: InferenceSession::prefill
//! [`decode_step`]: InferenceSession::decode_step

use std::path::{Path, PathBuf};

use mlxcel_core::session::{InferenceSession, SessionCapabilities};

#[cfg(feature = "iree")]
mod iree;

// The continuous-batching engine (#449 M3 Stage 2b). Present under `iree` (real
// execution) and under `test` (so its backend-neutral Scheduler bookkeeping is
// unit-tested without the IREE runtime, which the crate's own tests cannot link).
// Absent from a plain `--features xla-backend` build, so CI is unaffected.
#[cfg(any(feature = "iree", test))]
#[cfg_attr(not(feature = "iree"), allow(dead_code))]
mod batch;

#[cfg(feature = "iree")]
pub use batch::{EngineEvent, FinishReason, XlaBatchEngine, XlaReferenceEngine};

/// Error returned by the token-level primitives when the crate is built without
/// the `iree` feature (no IREE execution path compiled in).
pub const NOT_WIRED: &str = "the OpenXLA inference session was built without the `iree` feature; rebuild \
     mlxcel-xla with `--features iree` (and IREE_DIST pointing at an extracted iree \
     dist) to enable StableHLO / IREE execution of prefill / decode_step";

/// Read the model's EOS token ids from `generation_config.json` (a single int or
/// a list). Only parsed under `iree`, where `serde_json` is available and the
/// decode loop actually runs; otherwise empty (execution errors out anyway).
#[cfg(feature = "iree")]
pub(crate) fn read_eos(model_path: &Path) -> Vec<i32> {
    let p = model_path.join("generation_config.json");
    let Ok(s) = std::fs::read_to_string(p) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) else {
        return Vec::new();
    };
    match v.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => {
            n.as_i64().map(|x| vec![x as i32]).unwrap_or_default()
        }
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(serde_json::Value::as_i64)
            .map(|x| x as i32)
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(not(feature = "iree"))]
fn read_eos(_model_path: &Path) -> Vec<i32> {
    Vec::new()
}

/// The single-sequence OpenXLA inference session.
///
/// Owns its per-sequence KV state and runs generation token-in / token-out with
/// on-device sampling, the shape ADR 0004 reserves for a compiler-family backend.
/// Under the `iree` feature it holds the live IREE engine ([`iree::IreeLlama`])
/// and the running cache length; without it, it holds only the load inputs and
/// the token primitives report [`NOT_WIRED`].
pub struct XlaInferenceSession {
    model_path: PathBuf,
    num_layers: usize,
    eos_token_ids: Vec<i32>,
    #[cfg(feature = "iree")]
    engine: iree::IreeLlama,
    #[cfg(feature = "iree")]
    cache_len: i32,
}

impl XlaInferenceSession {
    /// Prepare a session for a model directory.
    ///
    /// Under the `iree` feature this verifies the model architecture, compiles
    /// the bundled `prefill` / `decode_step` graphs, and uploads the weights as
    /// resident device buffers (on `MLXCEL_XLA_DEVICE`, default `"local-task"`).
    /// Without the feature it only records the inputs so the seam wiring stays
    /// exercisable.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be prepared: under `iree`, an
    /// unsupported model architecture, a missing IREE distribution, an
    /// `iree-compile` failure, or a weight-loading failure.
    pub fn load(model_path: &Path, num_layers: usize) -> Result<Self, String> {
        let eos_token_ids = read_eos(model_path);
        #[cfg(feature = "iree")]
        {
            let device =
                std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".to_string());
            let engine = iree::IreeLlama::load(model_path, &device)?;
            Ok(Self {
                model_path: model_path.to_path_buf(),
                num_layers,
                eos_token_ids,
                engine,
                cache_len: 0,
            })
        }
        #[cfg(not(feature = "iree"))]
        {
            Ok(Self {
                model_path: model_path.to_path_buf(),
                num_layers,
                eos_token_ids,
            })
        }
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

    /// The model's EOS token ids (from `generation_config.json`), so the drive
    /// loop and the seam can stop on them.
    #[must_use]
    pub fn eos_token_ids(&self) -> &[i32] {
        &self.eos_token_ids
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
    /// Propagates the `prefill` / `decode_step` error, or errors on an empty
    /// prompt.
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
    /// Propagates the `prefill` / `decode_step` error, or errors on an empty
    /// prompt.
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

    #[cfg(feature = "iree")]
    fn prefill(&mut self, token_ids: &[i32]) -> Result<(), String> {
        self.engine.prefill_seed(token_ids)?;
        self.cache_len = token_ids.len() as i32;
        Ok(())
    }

    #[cfg(not(feature = "iree"))]
    fn prefill(&mut self, _token_ids: &[i32]) -> Result<(), String> {
        Err(NOT_WIRED.to_string())
    }

    #[cfg(feature = "iree")]
    fn decode_step(&mut self, token: i32) -> Result<i32, String> {
        let next = self.engine.decode(token, self.cache_len)?;
        self.cache_len += 1;
        Ok(next)
    }

    #[cfg(not(feature = "iree"))]
    fn decode_step(&mut self, _token: i32) -> Result<i32, String> {
        Err(NOT_WIRED.to_string())
    }
}

// These scaffold tests cover the without-`iree` behavior (load records inputs;
// the token primitives report NOT_WIRED). Under `--features iree`, `load`
// constructs a real engine from a real model directory, so the fake-path fixture
// does not apply; that path is validated by the end-to-end CLI run on GB10.
#[cfg(all(test, not(feature = "iree")))]
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
