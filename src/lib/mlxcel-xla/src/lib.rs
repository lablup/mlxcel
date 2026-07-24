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

use mlxcel_core::session::{InferenceSession, PreparedPrefill, SessionCapabilities};

mod context;
#[cfg(any(feature = "iree", test))]
#[cfg_attr(not(feature = "iree"), allow(dead_code))]
mod prepared;
mod prepared_deepstack;
mod prepared_gemma3n;

#[cfg(feature = "iree")]
mod iree;

// The continuous-batching engine (#449 M3 Stage 2b). Present under `iree` (real
// execution) and under `test` (so its backend-neutral Scheduler bookkeeping is
// unit-tested without the IREE runtime, which the crate's own tests cannot link).
// Absent from a plain `--features xla-backend` build, so CI is unaffected.
#[cfg(any(feature = "iree", test))]
#[cfg_attr(not(feature = "iree"), allow(dead_code))]
mod batch;

// Host-side token sampler (#449 M3 Stage 2d). Pure Rust; present under `iree` (the
// engine samples with it) and under `test` (unit-tested without the IREE runtime).
#[cfg(any(feature = "iree", test))]
#[cfg_attr(not(feature = "iree"), allow(dead_code))]
mod sampler;

// Safetensors weight-dtype widening (#449 M3 Stage 2d). Pure Rust bf16/f16 -> f32
// converters; present under `iree` (the loader widens weights with them) and under
// `test` (the f16 conversion is unit-tested without the IREE runtime).
#[cfg(any(feature = "iree", test))]
#[cfg_attr(not(feature = "iree"), allow(dead_code))]
mod weights;

// Per-architecture checkpoint tensor naming (#449 M3 Stage 2d; generalized in
// #499). Pure Rust; present under `iree` (the loader orders weights with it) and
// under `test` (the naming schemes are unit-tested without the IREE runtime).
#[cfg(any(feature = "iree", test))]
#[cfg_attr(not(feature = "iree"), allow(dead_code))]
mod weight_names;

// Rust-native StableHLO emitter (#449 M3 Stage 2d, ported from the #451 spike).
// Pure Rust; present under `iree` (the engine emits its graphs from config.json at
// load) and under `test` (the byte-exact regression runs without the IREE
// runtime). A faithful port keeps some entry points the production path does not
// call (uniform-B batched emit, the hard-coded reference config), so dead_code is
// allowed module-wide.
#[cfg(any(feature = "iree", test))]
#[allow(dead_code)]
mod emitter;

// Reusable per-architecture validation harness (issue #496). Pure Rust; present
// under `iree` (so the harness is available to tooling) and under `test` (the
// byte-exact structural gate runs here). The engine never calls it, so dead_code
// is allowed under a non-test `iree` build, matching the sibling modules above.
#[cfg(any(feature = "iree", test))]
#[allow(dead_code)]
mod validation;

#[cfg(feature = "iree")]
pub use batch::{EngineEvent, FinishReason, XlaAdmissionError, XlaBatchEngine, XlaReferenceEngine};
#[cfg(feature = "diagnostics")]
pub use batch::{
    Gemma3nAllLayerDiagnosticRun, Gemma3nCanonicalDiagnosticRun, Gemma3nPrefixDecodeDiagnosticRun,
    LlavaReferenceDiagnosticEngine, LlavaReferenceDiagnosticRun, run_gemma3n_all_layer_diagnostics,
    run_gemma3n_canonical_diagnostics, run_gemma3n_prefix_decode_diagnostic,
};
pub use context::{
    CONTEXT_CAPACITY_ENV, ContextCapacityError, DEFAULT_CONTEXT_CAPACITY,
    context_capacity_from_env, validate_request_capacity,
};
#[cfg(all(feature = "diagnostics", xla_iree_cuda))]
pub use emitter::{
    run_gemma3n_altup_correct_diagnostic_probe, run_gemma3n_altup_predict_diagnostic_probe,
    run_gemma3n_attention_diagnostic_probe, run_gemma3n_decode_attention_diagnostic_probe,
    run_gemma3n_dense_mlp_diagnostic_probe, run_gemma3n_initial_altup_diagnostic_probe,
    run_gemma3n_ple_diagnostic_probe, run_gemma3n_ple_injection_all_planes_diagnostic_probe,
    run_gemma3n_ple_injection_diagnostic_probe, run_gemma3n_post_attention_diagnostic_probe,
    run_gemma3n_qmv_diagnostic_probe, run_gemma3n_rms_diagnostic_probe,
    run_gemma3n_sdpa_vector_context_diagnostic_probe,
};
#[cfg(feature = "diagnostics")]
pub use iree::PreparedPrefillDiagnostics;
#[cfg(any(test, feature = "diagnostics"))]
#[must_use]
pub fn llava_diagnostic_device_memory_note(device: &str) -> &'static str {
    if device == "cuda" {
        "IREE CUDA reports N/A for dedicated memory on GB10 unified memory"
    } else {
        "the IREE C runtime integration exposes no portable device-allocation counter"
    }
}
#[cfg(feature = "diagnostics")]
pub fn dequantize_gemma3n_affine_diagnostic(
    packed: &[u8],
    scales: &[u8],
    biases: &[u8],
    out: usize,
    in_packed: usize,
    bits: usize,
    group_size: usize,
) -> Result<Vec<f32>, String> {
    weights::dequantize_affine_bf16_fused(packed, scales, biases, out, in_packed, bits, group_size)
}
#[cfg(feature = "diagnostics")]
pub use emitter::{Gemma3nDiagnosticLayout, Gemma3nDiagnosticSegment};
#[cfg(feature = "iree")]
pub use prepared::PreparedInputError;
pub use prepared_deepstack::{DeepStackFeatures, DeepStackInputError, DeepStackPreparedPrefill};
pub use prepared_gemma3n::{Gemma3nDensePle, Gemma3nDensePleError, Gemma3nPreparedPrefill};
#[cfg(feature = "iree")]
pub use sampler::SampleParams;

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
    context_capacity: usize,
    eos_token_ids: Vec<i32>,
    #[cfg(feature = "iree")]
    engine: iree::IreeLlama,
    #[cfg(feature = "iree")]
    cache_len: i32,
}

/// The default HAL device when `MLXCEL_XLA_DEVICE` is unset.
///
/// On Apple Silicon (macOS) the `xla-iree` runtime is built with the Metal
/// driver and Metal is the GPU, so default to `"metal"` (the dev/parity path,
/// faster than the CPU fallback). Elsewhere default to the portable multithreaded
/// CPU (`"local-task"`). CUDA is never auto-selected: it needs a cuda build and
/// device, so set `MLXCEL_XLA_DEVICE=cuda` explicitly. Override the default on any
/// platform with `MLXCEL_XLA_DEVICE`.
#[must_use]
pub fn default_device() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "metal"
    }
    #[cfg(not(target_os = "macos"))]
    {
        "local-task"
    }
}

impl XlaInferenceSession {
    /// Prepare a session for a model directory.
    ///
    /// Under the `iree` feature this verifies the model architecture, compiles
    /// the bundled `prefill` / `decode_step` graphs, and uploads the weights as
    /// resident device buffers (on `MLXCEL_XLA_DEVICE`, default
    /// [`default_device`]: `"metal"` on Apple Silicon, `"local-task"` elsewhere).
    /// Without the feature it only records the inputs so the seam wiring stays
    /// exercisable.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be prepared: under `iree`, an
    /// unsupported model architecture, a missing IREE distribution, an
    /// `iree-compile` failure, or a weight-loading failure.
    pub fn load(model_path: &Path, num_layers: usize) -> Result<Self, String> {
        let context_capacity = context_capacity_from_env()?;
        Self::load_with_context_capacity(model_path, num_layers, context_capacity)
    }

    /// Prepare a session with an explicitly selected static context capacity.
    pub fn load_with_context_capacity(
        model_path: &Path,
        num_layers: usize,
        context_capacity: usize,
    ) -> Result<Self, String> {
        let context_capacity = context::validate_context_capacity_value(context_capacity)?;
        let eos_token_ids = read_eos(model_path);
        #[cfg(feature = "iree")]
        {
            let device =
                std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| default_device().to_string());
            let engine = iree::IreeLlama::load(model_path, &device, context_capacity)?;
            Ok(Self {
                model_path: model_path.to_path_buf(),
                num_layers,
                context_capacity,
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
                context_capacity,
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

    /// Static sequence capacity compiled into this session's graph and KV cache.
    #[must_use]
    pub fn context_capacity(&self) -> usize {
        self.context_capacity
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
        validate_request_capacity(prompt_tokens.len(), max_new_tokens, self.context_capacity)
            .map_err(|err| err.to_string())?;
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

    /// Seed KV from a complete token prompt and return the prefill argmax.
    ///
    /// This mirrors the batched slot-seed convention and is distinct from the
    /// prefix-prefill-plus-decode generation loop.
    pub fn prefill_first_token(&mut self, prompt_tokens: &[i32]) -> Result<i32, String> {
        if prompt_tokens.is_empty() {
            return Err("prefill_first_token requires a non-empty prompt".to_string());
        }
        if prompt_tokens.len() > self.context_capacity {
            return Err(format!(
                "prefill length {} exceeds context_capacity={}",
                prompt_tokens.len(),
                self.context_capacity
            ));
        }
        #[cfg(feature = "iree")]
        {
            let first = self.engine.prefill_first(prompt_tokens)?;
            self.cache_len = i32::try_from(prompt_tokens.len())
                .map_err(|_| "prefill length does not fit i32".to_string())?;
            Ok(first)
        }
        #[cfg(not(feature = "iree"))]
        {
            Err(NOT_WIRED.to_string())
        }
    }

    /// Greedy generation seeded by an owned prepared-embeddings payload.
    pub fn generate_prepared_greedy(
        &mut self,
        prepared: &PreparedPrefill,
        max_new_tokens: usize,
        eos_token_ids: &[i32],
    ) -> Result<Vec<i32>, String> {
        self.generate_prepared_streaming_greedy(prepared, max_new_tokens, eos_token_ids, |_| true)
    }

    /// Seed a Gemma3n session through its distinct embeddings-plus-dense-PLE
    /// entry and return the first generated token.
    pub fn prefill_gemma3n_prepared(
        &mut self,
        request: &Gemma3nPreparedPrefill,
    ) -> Result<i32, String> {
        #[cfg(feature = "iree")]
        {
            let first = self.engine.prefill_gemma3n_prepared_first(request)?;
            self.cache_len = i32::try_from(request.prepared().sequence_len)
                .map_err(|_| "prepared sequence length does not fit i32".to_string())?;
            Ok(first)
        }
        #[cfg(not(feature = "iree"))]
        {
            let _ = request;
            Err(NOT_WIRED.to_string())
        }
    }

    /// Seed a session through the distinct sparse DeepStack embeddings entry.
    pub fn prefill_deepstack_prepared(
        &mut self,
        request: &DeepStackPreparedPrefill,
    ) -> Result<i32, String> {
        #[cfg(feature = "iree")]
        {
            let first = self.engine.prefill_deepstack_prepared_first(request)?;
            self.cache_len = i32::try_from(request.prepared().sequence_len)
                .map_err(|_| "prepared sequence length does not fit i32".to_string())?;
            Ok(first)
        }
        #[cfg(not(feature = "iree"))]
        {
            let _ = request;
            Err(NOT_WIRED.to_string())
        }
    }

    /// Greedy generation seeded by compact prefill-only DeepStack additions.
    pub fn generate_deepstack_prepared_greedy(
        &mut self,
        request: &DeepStackPreparedPrefill,
        max_new_tokens: usize,
        eos_token_ids: &[i32],
    ) -> Result<Vec<i32>, String> {
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        validate_request_capacity(
            request.prepared().sequence_len,
            max_new_tokens,
            self.context_capacity,
        )
        .map_err(|error| error.to_string())?;
        let first = self.prefill_deepstack_prepared(request)?;
        let mut out = Vec::with_capacity(max_new_tokens);
        out.push(first);
        let mut current = first;
        while out.len() < max_new_tokens && !eos_token_ids.contains(&current) {
            current = self.decode_step(current)?;
            out.push(current);
        }
        Ok(out)
    }

    /// Greedy generation seeded by Gemma3n post-scale embeddings and dense PLE.
    pub fn generate_gemma3n_prepared_greedy(
        &mut self,
        request: &Gemma3nPreparedPrefill,
        max_new_tokens: usize,
        eos_token_ids: &[i32],
    ) -> Result<Vec<i32>, String> {
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        validate_request_capacity(
            request.prepared().sequence_len,
            max_new_tokens,
            self.context_capacity,
        )
        .map_err(|error| error.to_string())?;
        let first = self.prefill_gemma3n_prepared(request)?;
        let mut out = Vec::with_capacity(max_new_tokens);
        out.push(first);
        let mut current = first;
        while out.len() < max_new_tokens && !eos_token_ids.contains(&current) {
            current = self.decode_step(current)?;
            out.push(current);
        }
        Ok(out)
    }

    /// Streaming greedy generation from prepared embeddings. The embeddings
    /// entry seeds all `sequence_len` positions and returns the first generated
    /// token, so decode starts at the expanded length without replaying a
    /// placeholder or logical token.
    pub fn generate_prepared_streaming_greedy<F: FnMut(i32) -> bool>(
        &mut self,
        prepared: &PreparedPrefill,
        max_new_tokens: usize,
        eos_token_ids: &[i32],
        mut on_token: F,
    ) -> Result<Vec<i32>, String> {
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        validate_request_capacity(prepared.sequence_len, max_new_tokens, self.context_capacity)
            .map_err(|error| error.to_string())?;
        let first = self.prefill_prepared(prepared)?;
        let mut out = Vec::with_capacity(max_new_tokens);
        out.push(first);
        if !on_token(first) || eos_token_ids.contains(&first) {
            return Ok(out);
        }
        let mut current = first;
        while out.len() < max_new_tokens {
            let next = self.decode_step(current)?;
            out.push(next);
            if !on_token(next) || eos_token_ids.contains(&next) {
                break;
            }
            current = next;
        }
        Ok(out)
    }
}

impl InferenceSession for XlaInferenceSession {
    fn capabilities(&self) -> SessionCapabilities {
        #[cfg(feature = "iree")]
        {
            SessionCapabilities::single_sequence().with_multimodal()
        }
        #[cfg(not(feature = "iree"))]
        {
            SessionCapabilities::single_sequence()
        }
    }

    #[cfg(feature = "iree")]
    fn prefill(&mut self, token_ids: &[i32]) -> Result<(), String> {
        if token_ids.len() > self.context_capacity {
            return Err(format!(
                "prefill length {} exceeds context_capacity={}",
                token_ids.len(),
                self.context_capacity
            ));
        }
        self.engine.prefill_seed(token_ids)?;
        self.cache_len = token_ids.len() as i32;
        Ok(())
    }

    #[cfg(not(feature = "iree"))]
    fn prefill(&mut self, _token_ids: &[i32]) -> Result<(), String> {
        Err(NOT_WIRED.to_string())
    }

    #[cfg(feature = "iree")]
    fn prefill_prepared(&mut self, prepared: &PreparedPrefill) -> Result<i32, String> {
        let first = self.engine.prefill_prepared_first(prepared)?;
        self.cache_len = i32::try_from(prepared.sequence_len)
            .map_err(|_| "prepared sequence length does not fit i32".to_string())?;
        Ok(first)
    }

    #[cfg(not(feature = "iree"))]
    fn prefill_prepared(&mut self, _prepared: &PreparedPrefill) -> Result<i32, String> {
        Err(NOT_WIRED.to_string())
    }

    #[cfg(feature = "iree")]
    fn decode_step(&mut self, token: i32) -> Result<i32, String> {
        if self.cache_len < 0 || self.cache_len as usize >= self.context_capacity {
            return Err(format!(
                "decode position {} is outside context_capacity={}",
                self.cache_len, self.context_capacity
            ));
        }
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
    use mlxcel_core::session::{
        OwnedTensor, PreparedAttentionBias, PreparedPositions, PreparedTensorDType,
    };

    fn session() -> XlaInferenceSession {
        XlaInferenceSession::load(Path::new("/tmp/xla-model"), 16).unwrap()
    }

    fn prepared() -> PreparedPrefill {
        PreparedPrefill::new(
            vec![1],
            OwnedTensor::new(vec![0; 4], PreparedTensorDType::Float32, vec![1, 1, 1]).unwrap(),
            PreparedPositions::Sequential {
                start: 0,
                length: 1,
            },
            PreparedAttentionBias {
                tensor: OwnedTensor::new(
                    vec![0; 4],
                    PreparedTensorDType::Float32,
                    vec![1, 1, 1, 1],
                )
                .unwrap(),
                causal: true,
            },
            Vec::new(),
        )
        .unwrap()
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
        assert_eq!(s.context_capacity(), DEFAULT_CONTEXT_CAPACITY);
    }

    #[test]
    fn explicit_context_capacity_is_recorded_and_validated() {
        let s =
            XlaInferenceSession::load_with_context_capacity(Path::new("/tmp/xla-model"), 16, 1024)
                .unwrap();
        assert_eq!(s.context_capacity(), 1024);
        assert!(
            XlaInferenceSession::load_with_context_capacity(Path::new("/tmp/xla-model"), 16, 0,)
                .is_err()
        );
    }

    #[test]
    fn token_primitives_report_not_wired() {
        let mut s = session();
        assert_eq!(s.prefill(&[1, 2, 3]).unwrap_err(), NOT_WIRED);
        assert_eq!(s.prefill_prepared(&prepared()).unwrap_err(), NOT_WIRED);
        assert_eq!(s.decode_step(1).unwrap_err(), NOT_WIRED);
    }

    #[test]
    fn greedy_surfaces_the_stub_error_not_a_panic() {
        let mut s = session();
        assert_eq!(
            s.generate_greedy(&[1, 2, 3], 8, &[2]).unwrap_err(),
            NOT_WIRED
        );
        assert_eq!(
            s.generate_prepared_greedy(&prepared(), 8, &[2])
                .unwrap_err(),
            NOT_WIRED
        );
    }

    #[test]
    fn greedy_rejects_context_overflow_before_the_unwired_backend() {
        let mut s =
            XlaInferenceSession::load_with_context_capacity(Path::new("/tmp/xla-model"), 16, 8)
                .unwrap();
        let err = s.generate_greedy(&[1, 2, 3, 4, 5], 4, &[2]).unwrap_err();
        assert!(err.contains("effective_prompt_len=5"));
        assert!(err.contains("max_new_tokens=4"));
        assert!(err.contains("context_capacity=8"));
    }

    #[test]
    fn empty_prompt_is_rejected() {
        let mut s = session();
        assert!(s.generate_greedy(&[], 8, &[2]).is_err());
    }

    #[test]
    fn default_device_is_metal_on_apple_silicon_else_cpu() {
        let d = default_device();
        #[cfg(target_os = "macos")]
        assert_eq!(d, "metal");
        #[cfg(not(target_os = "macos"))]
        assert_eq!(d, "local-task");
    }

    #[test]
    fn llava_diagnostics_share_the_production_embeddings_prefill() {
        let c = include_str!("../csrc/xla_iree.c");
        let diagnostic = c
            .split("int xla_llama_prefill_embeddings_slot_diagnostics(")
            .nth(1)
            .expect("diagnostic C ABI")
            .split("int xla_llama_prefill_embeddings_ple(")
            .next()
            .expect("bounded diagnostic C ABI");
        assert!(diagnostic.contains("xla_llama_prefill_embeddings_impl("));
        assert!(!diagnostic.contains("iree_runtime_call_invoke"));

        let rust = include_str!("iree.rs");
        let ffi = rust
            .split("fn xla_llama_prefill_embeddings_slot_diagnostics(")
            .next()
            .expect("diagnostic Rust FFI");
        assert!(
            ffi.lines()
                .rev()
                .take(2)
                .any(|line| line.contains("cfg(feature = \"diagnostics\")")),
            "the diagnostic ABI must remain feature-gated"
        );
    }

    #[test]
    fn llava_device_memory_note_does_not_label_cpu_as_cuda() {
        assert!(super::llava_diagnostic_device_memory_note("cuda").contains("GB10"));
        assert!(!super::llava_diagnostic_device_memory_note("local-sync").contains("CUDA"));
        assert!(!super::llava_diagnostic_device_memory_note("local-task").contains("GB10"));
    }
}
