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
pub mod cli;
pub mod distributed;
pub mod downloader;
pub mod execution;
pub mod lang_bias;
pub mod lora;
pub mod models;
pub mod multimodal;
pub mod server;
#[cfg(feature = "surgery")]
pub mod surgery;
pub mod tokenizer;
pub mod vision;

mod loaded_model;
mod loaded_model_capabilities;
mod loading;
mod model_metadata;

// Crate-wide helpers for `#[cfg(test)]` paths. Provides the single shared
// `ENV_LOCK` that every env-mutating test in this crate must acquire (issue
// #573); see `test_support::env_lock` for the rationale. `pub(crate)` so
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
pub use multimodal::{
    internvl_prompt, minicpmo_prompt, moondream3_prompt, phi3v_prompt, phi4_siglip_prompt,
    phi4mm_prompt, qwen_vl, video, vlm_prompt, vlm_runtime, youtu_vl_prompt,
};

// Re-export split modules
pub use loaded_model::LoadedModel;
pub use loaded_model_capabilities::VlmRuntimeRef;
pub use loading::{
    load_model, load_model_with_adapter, load_model_with_tensor_parallel, read_eos_token_ids,
};
