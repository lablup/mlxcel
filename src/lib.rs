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

//! mlxcel - High-performance LLM inference on Apple Silicon
//!
//! This crate provides efficient inference for Large Language Models using
//! direct MLX C++ bindings via mlxcel-core.

pub mod audio;
pub mod backend;
pub mod cli;
pub mod distributed;
pub mod downloader;
pub mod execution;
pub mod lang_bias;
pub mod lora;
pub mod models;
pub mod multimodal;
pub mod reasoning_stream;
pub mod server;
#[cfg(feature = "surgery")]
pub mod surgery;
pub mod tokenizer;
pub mod vision;

mod loaded_model;
mod loaded_model_capabilities;
mod loading;
mod model_metadata;

// Fail-fast wrapper shared by every core inference worker thread (issue #375).
// `pub(crate)` so both `server::model_provider::model_worker` and
// `distributed::pipeline::remote_service` can name it.
pub(crate) mod worker_failfast;

// Crate-wide helpers for `#[cfg(test)]` paths. Provides the single shared
// `ENV_LOCK` that every env-mutating test in this crate must acquire; see `test_support::env_lock` for the rationale. `pub(crate)` so
// that test modules at any depth (e.g. `crate::server::cli_input::tests`)
// can name it as `crate::test_support::env_lock`.
#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
#[path = "model_metadata_tests.rs"]
mod model_metadata_tests;

#[cfg(test)]
#[path = "lang_analyzer_tests.rs"]
mod lang_analyzer_tests;

#[cfg(test)]
#[path = "sampling_observability_tests.rs"]
mod sampling_observability_tests;

// Re-export mlxcel-core generate module
pub use execution::kv_cache_advisor;
pub use execution::memory_estimate;
pub use execution::quant_advisor;
pub use execution::runtime::{RuntimeDevice, RuntimeSetup, initialize_runtime};
pub use execution::sampling;
pub use mlxcel_core::generate;
pub use mlxcel_core::generate::{
    CxxGenerator, DecodeBatchContext, DecodeStorageBackend, GenerationStats, LanguageModel,
    SamplingConfig,
};
pub use mlxcel_core::speculative::SpeculativeGenerator;
pub use multimodal::host_preprocessor::{
    FakeHostMultimodalPreprocessor, HostMultimodalPreprocessor, HostPreprocessorError,
    LlavaHostPreprocessor, load_xla_image_preprocessor,
};
pub use multimodal::{
    internvl_prompt, kimi_vl_prompt, minicpmo_prompt, moondream2_prompt, moondream3_prompt,
    phi3v_prompt, phi4_siglip_prompt, phi4mm_prompt, pixtral_prompt, qwen_vl, smolvlm_prompt,
    video, vlm_prompt, vlm_runtime, youtu_vl_prompt,
};

// Re-export the compute-backend seam (issue #338, reframed to a session engine
// in issue #448 / ADR 0004). Control-plane callers in both the library and the
// `mlxcel` / `mlxcel-server` binaries reach model loading through
// `select_backend()` so the forward-execution engine is chosen at one boundary.
// The CLI generation path obtains a single-sequence `Session` from the backend;
// the server batched path keeps using `load_model`. Under default features both
// fold to the single MLX variant with no runtime dispatch.
pub use backend::{Backend, ComputeBackend, MlxBackend, Session, select_backend};
pub use mlxcel_core::session::{
    InferenceSession, MlxInferenceSession, OwnedTensor, PreparedAttentionBias, PreparedModality,
    PreparedPositions, PreparedPrefill, PreparedPrefillError, PreparedTensorDType,
    SessionCapabilities,
};
pub use server::ImageInputLimits;

/// Return the image admission limits shared by CLI and server requests.
#[must_use]
pub fn current_image_input_limits() -> ImageInputLimits {
    server::current_image_input_limits()
}

/// Decode already-bounded image payloads with the same limits as the server.
pub fn decode_image_payloads_with_limits(
    images: &[Vec<u8>],
    limits: ImageInputLimits,
) -> anyhow::Result<Vec<image::DynamicImage>> {
    server::model_provider::model_worker::decode_request_images_with_limits(images, limits)
}

// Re-export split modules
pub use loaded_model::LoadedModel;
pub use loaded_model_capabilities::VlmRuntimeRef;
pub use loading::{
    context_window_from_config, load_model, load_model_with_adapter,
    load_model_with_tensor_parallel, load_qwen3_omni_speech, read_eos_token_ids,
    read_model_context_window,
};
