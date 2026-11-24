// Copyright 2025-2026 Lablup Inc.
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

//! Recurrent models on the OpenXLA backend never share prefix-hash caches
//! (epic #564, issue #3341).
//!
//! A saved token prefix cannot restore recurrent state, and the OpenXLA serve
//! engine has no state snapshot to adopt instead. The guard runs twice on the
//! same decision ([`xla_recurrent_family`]):
//!
//! 1. `startup::resolve_prompt_cache_store` turns the whole prompt cache and
//!    APC off before the store is built ([`disable_prompt_cache_for_xla_recurrent`]),
//!    so neither `AppState` nor the provider holds a store and the cache
//!    routes report it disabled.
//! 2. `ModelProvider`'s OpenXLA arm drops any store another caller hands it
//!    ([`xla_provider_prompt_cache`]).
//!
//! The decision is the union of the emitter's `Config::has_recurrent_state()`
//! and the shared [`HYBRID_SSM_MODEL_TYPES`](crate::server::prompt_cache::HYBRID_SSM_MODEL_TYPES)
//! list, so a recurrent family the emitter cannot load yet is refused too.

use std::path::Path;
use std::sync::Arc;

use crate::server::prompt_cache::{
    PromptCacheConfig, PromptCacheStore, recurrent_prefix_sharing_family,
};

/// Whether this process serves through the OpenXLA engine. The same test as
/// `ModelProvider::new_with_server_config_and_prompt_cache`'s OpenXLA arm.
pub(crate) fn xla_serve_backend_selected() -> bool {
    std::env::var("MLXCEL_BACKEND").ok().as_deref() == Some("xla")
}

/// The family logged when `config.json` cannot be read or parsed, so the
/// guard cannot tell whether the model is recurrent.
pub(crate) const UNDETECTED_FAMILY: &str = "unreadable config.json";

/// The family to log when `model_path` must stay out of prefix sharing on the
/// OpenXLA backend, or `None` when it may share prefixes. A `config.json`
/// that cannot be read or parsed fails closed (the cache stays off): the
/// guard cannot rule out recurrent state, and the model load reports the
/// error itself.
pub(crate) fn xla_recurrent_family(model_path: &Path) -> Option<String> {
    let Ok(text) = std::fs::read_to_string(model_path.join("config.json")) else {
        return Some(UNDETECTED_FAMILY.to_string());
    };
    // Tolerant of HF's bare `Infinity` (Mamba2's `time_step_limit`).
    let Ok(config) = mlxcel_xla::parse_hf_config_json(&text) else {
        return Some(UNDETECTED_FAMILY.to_string());
    };
    let threads_state = mlxcel_xla::config_json_has_recurrent_state(&text).unwrap_or(false);
    recurrent_prefix_sharing_family(&config, threads_state)
}

/// Turn the prompt cache and APC off for a recurrent OpenXLA model. Returns
/// the family it was turned off for, after logging one line naming it.
pub(crate) fn disable_prompt_cache_for_xla_recurrent(
    prompt_cache: &mut PromptCacheConfig,
    model_path: &Path,
) -> Option<String> {
    let family = xla_recurrent_family(model_path)?;
    tracing::info!(
        model_type = %family,
        "OpenXLA recurrent model ({family}): prompt cache and APC disabled, because a saved \
         token prefix cannot restore recurrent state"
    );
    prompt_cache.enabled = false;
    prompt_cache.apc.enabled = false;
    Some(family)
}

/// The store `ModelProvider`'s OpenXLA arm may attach: `store` unless the
/// model is recurrent.
pub(crate) fn xla_provider_prompt_cache(
    model_path: &Path,
    store: Option<Arc<PromptCacheStore>>,
) -> Option<Arc<PromptCacheStore>> {
    let store = store?;
    if let Some(family) = xla_recurrent_family(model_path) {
        tracing::info!(
            model_type = %family,
            "OpenXLA recurrent model ({family}): prompt-cache store not attached"
        );
        return None;
    }
    Some(store)
}

#[cfg(test)]
#[path = "xla_prompt_cache_guard_tests.rs"]
mod tests;
