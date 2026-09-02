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

//! Public-facing types owned by [`super::store::PromptCacheStore`] —
//! insertion errors and bucket identity keys.
//!
//! These types are factored out so the `store.rs` file stays focused on
//! the store itself (locking, LRU, metrics bookkeeping) and sub-issue
//! reviewers can audit the store's surface without wading through pure
//! data definitions.

use thiserror::Error;

use super::key::{MultimodalDigest, PromptCacheKey};

/// Insert-time failure mode.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum InsertError {
    /// The feature is disabled (either via config or via store construction).
    #[error("prompt cache is disabled")]
    Disabled,
    /// The token prefix being inserted is shorter than
    /// [`super::policy::PromptCacheConfig::min_prefix_tokens`].
    #[error("prompt cache: prefix is too short ({got} < {min_required})")]
    PrefixTooShort { got: usize, min_required: usize },
    /// The single entry exceeds the store's configured byte budget on its
    /// own, so no amount of eviction could make room for it.
    #[error(
        "prompt cache: entry size {entry_bytes} exceeds capacity {capacity_bytes} (cannot fit even alone)"
    )]
    OversizedEntry {
        entry_bytes: usize,
        capacity_bytes: usize,
    },
}

/// Composition key (model/lora/template/session) kept alongside the digest
/// so lookups can disambiguate partial prefix collisions.
///
/// The key identifies a *bucket* — a set of entries that share the same
/// model/lora/template/session and can therefore share a KV-cache prefix.
/// Token prefixes distinguish entries *within* a bucket. The session-
/// sensitive bucket is the strict identity used for metrics / tie-breaks;
/// see [`SessionlessBucketKey`] for the cross-session fallback index.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BucketKey {
    pub model_id: String,
    pub lora_id: Option<String>,
    pub template_sig: String,
    pub mm_digest: MultimodalDigest,
    /// See [`PromptCacheKey::rope_regime`]. Two requests on opposite sides of a
    /// position-selected RoPE table must not share a bucket (#1358).
    pub rope_regime: Option<u8>,
    pub session_key: Option<String>,
}

impl BucketKey {
    /// Build a bucket key from a [`PromptCacheKey`] (drops the token prefix
    /// since it does not participate in bucket identity).
    pub fn from_key(key: &PromptCacheKey<'_>) -> Self {
        Self {
            model_id: key.model_id.to_string(),
            lora_id: key.lora_id.map(str::to_string),
            template_sig: key.template_sig.to_string(),
            mm_digest: key.mm_digest,
            rope_regime: key.rope_regime,
            session_key: key.session_key.map(str::to_string),
        }
    }
}

/// Session-independent bucket identity. Used as the index key for the
/// per-model/lora/template radix trie that powers
/// [`super::store::PromptCacheStore::lookup_longest_prefix`]. Two entries
/// with identical `(model_id, lora_id, template_sig, mm_digest)` but different
/// `session_key`s share the same trie, so cross-session prefix reuse is
/// possible only when the rendered template and resolved multimodal payload
/// match — subject to the tie-break rules documented on the lookup method.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct SessionlessBucketKey {
    pub model_id: String,
    pub lora_id: Option<String>,
    pub template_sig: String,
    pub mm_digest: MultimodalDigest,
    /// See [`PromptCacheKey::rope_regime`]. This is the radix trie's index key,
    /// so without the regime here a cross-session prefix lookup would still
    /// reach entries encoded under the other table (#1358).
    pub rope_regime: Option<u8>,
}

impl SessionlessBucketKey {
    pub(super) fn from_key(key: &PromptCacheKey<'_>) -> Self {
        Self {
            model_id: key.model_id.to_string(),
            lora_id: key.lora_id.map(str::to_string),
            template_sig: key.template_sig.to_string(),
            mm_digest: key.mm_digest,
            rope_regime: key.rope_regime,
        }
    }
}
