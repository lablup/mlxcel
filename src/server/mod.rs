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

//! OpenAI/llama-server compatible HTTP server for mlxcel

pub mod anthropic_translator;
pub mod app;
pub mod audio_model;
pub(crate) mod audio_worker;
pub mod batch;
mod chat_request;
pub mod chat_template;
pub mod chat_template_kwargs;
mod cli_input;
mod config;
pub mod conversation_store;
mod cors;
pub(crate) mod diffusion_worker;
pub mod kokoro_tts;
mod media;
pub mod model_provider;
pub mod prompt_cache;
mod request_options;
pub mod responses_store;
pub mod responses_translator;
pub mod router_front;
pub mod routes;
pub mod speculative_dispatch;
mod startup;
mod state;
mod streaming;
pub mod streaming_anthropic;
pub mod streaming_responses;
pub mod structured;
pub mod thinking_budget;
pub mod tool_calls;
pub mod types;
pub mod whisper_stt;

pub use app::create_app;
pub use audio_model::{
    AudioModelError, AudioModelKind, AudioModelProvider, AudioSynthesizeInput,
    AudioSynthesizeOutput, AudioTranscribeInput, AudioTranscribeOutput,
};
pub use chat_template::ChatTemplateProcessor;
pub use chat_template_kwargs::{
    ChatTemplateKwargs, ChatTemplateKwargsError, LLAMA_ARG_CHAT_TEMPLATE_KWARGS,
    env_fallback_chat_template_kwargs,
};
pub use cli_input::{
    ServerStartupInput, env_fallback_apc_block_size, env_fallback_apc_enabled,
    env_fallback_apc_hash, env_fallback_apc_num_blocks, env_fallback_cache_type_k,
    env_fallback_cache_type_v, env_fallback_kv_bits, env_fallback_kv_group_size,
    env_fallback_kv_quant_scheme, env_fallback_kv_skip_last_layer, env_fallback_lang_bias,
    env_fallback_lang_bias_include_byte_fragments, env_fallback_prompt_cache_capacity_bytes,
    env_fallback_prompt_cache_enabled, env_fallback_prompt_cache_max_entries,
    env_fallback_prompt_cache_min_prefix, env_fallback_prompt_cache_ttl,
    env_fallback_reasoning_budget, long_cli_flag_was_set, resolve_batch_kv_quant_config,
    resolve_kv_cache_mode,
};
pub use config::{
    DecodeStorageBackend, PipelineParallelRuntimeConfig, PreemptionPolicy,
    RemotePipelineStageConfig, ServerConfig, ServerGenerateOptions,
};
pub(crate) use media::current_image_input_limits;
pub use media::{
    DEFAULT_MAX_IMAGE_DECODE_ALLOC_BYTES, DEFAULT_MAX_IMAGE_HEIGHT, DEFAULT_MAX_IMAGE_PAYLOAD_SIZE,
    DEFAULT_MAX_IMAGE_WIDTH, DEFAULT_MAX_IMAGES_PER_REQUEST, ImageInputLimits,
};
pub use model_provider::{GenerationResult, ModelProvider};
pub use prompt_cache::{
    ApcBlockHash, ApcConfig, ApcHashAlgo, BlockHashChain, CacheEntry, DEFAULT_APC_BLOCK_SIZE,
    HYBRID_SSM_MODEL_TYPES, InsertError as PromptCacheInsertError, MultimodalDigest,
    PromptCacheConfig, PromptCacheKey, PromptCacheStats, PromptCacheStore, detect_hybrid_ssm,
    detect_hybrid_ssm_from_path, is_hybrid_ssm_model_type, multimodal_digest,
    multimodal_digest_from_vecs,
};
pub use speculative_dispatch::{SpeculativeDispatch, SpeculativeDispatchError};
pub use startup::{
    MIN_PARALLEL_CONTEXT_SIZE, ServerStartupConfig, effective_parallel_context_slots,
    resolve_parallel_context_size, start_server,
};
pub use state::{AppState, BatchMetrics, Metrics, ModelMediaSupport};
