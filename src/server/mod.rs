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
pub mod auth;
pub mod batch;
mod chat_request;
pub mod chat_template;
pub mod chat_template_json;
pub mod chat_template_kwargs;
mod cli_input;
pub(crate) mod completion_control;
mod config;
pub mod conversation_store;
mod cors;
pub(crate) mod diffusion_worker;
mod dry_breakers;
pub mod embedding_model;
pub mod embedding_worker;
pub(crate) mod florence2_worker;
pub mod gcp_compat;
mod http_timeout;
pub mod infill;
pub mod kokoro_tts;
mod listen;
pub mod logging;
mod media;
mod media_net;
pub mod media_root;
pub mod model_meta;
pub mod model_provider;
pub mod model_source;
pub mod prompt_cache;
mod read_budget;
pub mod reasoning_format;
mod request_options;
pub mod rerank_model;
pub mod rerank_worker;
pub mod responses_store;
pub mod responses_translator;
pub mod router_cache;
pub mod router_front;
pub mod router_models;
pub mod router_presets;
pub mod router_server;
pub mod routes;
pub mod slot_persist;
pub mod slots_state;
pub mod speculative_dispatch;
mod startup;
mod state;
pub(crate) mod stream_session;
mod streaming;
pub mod streaming_anthropic;
pub mod streaming_responses;
pub mod structured;
pub mod thinking_budget;
pub mod tls;
pub mod tool_calls;
pub mod transport;
pub mod types;
pub mod whisper_stt;

pub use app::create_app;
pub use audio_model::{
    AudioModelError, AudioModelKind, AudioModelProvider, AudioSynthesizeInput,
    AudioSynthesizeOutput, AudioTranscribeInput, AudioTranscribeOutput,
};
pub use auth::{ApiKeys, resolve_api_keys};
pub use chat_template::ChatTemplateProcessor;
pub use chat_template_kwargs::{
    ChatTemplateKwargs, ChatTemplateKwargsError, LLAMA_ARG_CHAT_TEMPLATE_KWARGS,
    env_fallback_chat_template_kwargs,
};
pub use cli_input::{
    ServerStartupInput, env_fallback_apc_block_size, env_fallback_apc_enabled,
    env_fallback_apc_hash, env_fallback_apc_num_blocks, env_fallback_api_key_files,
    env_fallback_api_keys, env_fallback_batch_size, env_fallback_cache_type_k,
    env_fallback_cache_type_v, env_fallback_cors_credentials, env_fallback_draft_model,
    env_fallback_embedding_model, env_fallback_endpoint_slots, env_fallback_kv_bits,
    env_fallback_kv_group_size, env_fallback_kv_quant_scheme, env_fallback_kv_skip_last_layer,
    env_fallback_lang_bias, env_fallback_lang_bias_include_byte_fragments, env_fallback_log_file,
    env_fallback_prompt_cache_capacity_bytes, env_fallback_prompt_cache_enabled,
    env_fallback_prompt_cache_max_entries, env_fallback_prompt_cache_min_prefix,
    env_fallback_prompt_cache_snapshot_capacity_bytes,
    env_fallback_prompt_cache_snapshot_max_entries, env_fallback_prompt_cache_snapshot_ttl,
    env_fallback_prompt_cache_ttl, env_fallback_reasoning_budget, env_fallback_reranker_model,
    env_fallback_ubatch_size, long_cli_flag_was_set, resolve_batch_kv_quant_config,
    resolve_compat_toggle, resolve_kv_cache_mode,
};
pub use config::{
    DecodeStorageBackend, PipelineParallelRuntimeConfig, PreemptionPolicy, ReasoningAliasField,
    RemotePipelineStageConfig, ServerConfig, ServerGenerateOptions,
};
pub use cors::{CorsPolicy, OriginPolicy};
pub use media::{
    DEFAULT_MAX_IMAGE_DECODE_ALLOC_BYTES, DEFAULT_MAX_IMAGE_HEIGHT, DEFAULT_MAX_IMAGE_PAYLOAD_SIZE,
    DEFAULT_MAX_IMAGE_WIDTH, DEFAULT_MAX_IMAGES_PER_REQUEST, ImageInputLimits,
    configure_media_admission, media_admission_disabled,
};
pub(crate) use media::{current_image_input_limits, media_capability_rejection};
pub use media_net::{configure_private_media_urls, private_media_urls_allowed_from_env};
pub use model_provider::{GenerationResult, ModelProvider};
pub use model_source::{
    LLAMA_ARG_OFFLINE, LlamaModelSourceArgs, ResolvedModelSource, env_fallback_offline,
    parse_hf_repo, parse_model_aliases, resolve_llama_model_source, superseded_model_notice,
};
pub use prompt_cache::{
    ApcBlockHash, ApcConfig, ApcHashAlgo, BlockHashChain, CacheEntry, DEFAULT_APC_BLOCK_SIZE,
    HYBRID_SSM_MODEL_TYPES, InsertError as PromptCacheInsertError, MultimodalDigest,
    PromptCacheConfig, PromptCacheKey, PromptCacheStats, PromptCacheStore, detect_hybrid_ssm,
    detect_hybrid_ssm_from_path, is_hybrid_ssm_model_type, multimodal_digest,
    multimodal_digest_from_vecs,
};
pub use reasoning_format::{ReasoningFormat, ShapedResponse, shape_response};
pub use speculative_dispatch::{SpeculativeDispatch, SpeculativeDispatchError};
pub use startup::{
    MIN_PARALLEL_CONTEXT_SIZE, ServerStartupConfig, effective_parallel_context_slots,
    ensure_chat_template_is_not_a_builtin_name, is_b10621_builtin_chat_template,
    resolve_parallel_context_size, start_server,
};
pub use state::{AppState, BatchMetrics, Metrics, ModelMediaSupport};

#[cfg(test)]
mod llama_compat_tests;

#[cfg(test)]
mod max_tokens_route_tests;

#[cfg(test)]
mod muse_glimmer_template_tests;

#[cfg(test)]
mod reasoning_effort_tests;

#[cfg(test)]
mod responses_input_parts_tests;

#[cfg(test)]
mod muse_atem_roundtrip_tests;

#[cfg(test)]
mod muse_atem_stream_support;

#[cfg(test)]
mod muse_atem_stream_chat_tests;

#[cfg(test)]
mod muse_atem_stream_responses_tests;

#[cfg(test)]
mod muse_atem_stream_anthropic_tests;
