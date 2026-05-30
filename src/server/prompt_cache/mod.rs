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

//! Cross-request prompt-prefix KV cache.
//!
//! This module holds the shared, thread-safe store that retains detached
//! KV caches keyed by `(model_id, lora_id, template_sig, session_key,
//! token_prefix_hash)`. The store hands caches out to freshly arriving
//! requests so that common prompt prefixes (system prompts, long context
//! windows, tool-calling preambles, etc.) don't get re-prefilled per
//! request.
//!
//! ## Sub-issues
//!
//! * — [`mlxcel_core::cache::DetachedCacheSet`] and the
//!   [`mlxcel_core::cache::CachePool`] detach/adopt API (upstream).
//! * — this module, PromptCacheStore itself.
//! * — radix trie inside [`store::PromptCacheStore::lookup_longest_prefix`]
//!   (via [`trie`]).
//! * — full `template_sig` wiring.
//! * — metrics bridge to [`super::state::BatchMetrics`].
//! * — CLI/env wiring for [`PromptCacheConfig`].
//!
//! Integration: [`super::startup`] constructs a shared `Arc<PromptCacheStore>`
//! when [`PromptCacheConfig::enabled`] is true and installs it on both
//! [`super::state::AppState`] and [`super::model_provider::ModelProvider`].
//! When disabled the store slot stays `None` so no memory is reserved.

mod apc_lookup;
pub mod block_hash;
pub mod entry;
pub mod hybrid_ssm;
pub mod key;
mod lookup;
pub mod metrics;
pub mod policy;
pub mod store;
mod trie;
mod types;

#[cfg(test)]
mod apc_integration_tests;
#[cfg(test)]
mod prefix_matcher_tests;

pub use apc_lookup::ApcStoreStats;
pub use block_hash::{
    ApcBlockHash, ApcHashAlgo, BlockHashChain, DEFAULT_APC_BLOCK_SIZE, ParseApcHashError,
};
pub use entry::CacheEntry;
pub use hybrid_ssm::{
    HYBRID_SSM_MODEL_TYPES, detect_hybrid_ssm, detect_hybrid_ssm_from_path,
    is_hybrid_ssm_model_type,
};
pub use key::{
    ANONYMOUS_SESSION_SENTINEL, MultimodalDigest, PromptCacheKey, PromptCacheKeyDigest,
    multimodal_digest, multimodal_digest_from_vecs, resolve_session_key, template_sig,
    tools_digest,
};
pub use metrics::{AtomicPromptCacheMetrics, NoopPromptCacheMetrics, PromptCacheMetrics};
pub use policy::{ApcConfig, PromptCacheConfig, PromptCacheStats};
pub use store::PromptCacheStore;
pub use types::{BucketKey, InsertError};
