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

//! Model-aware defaults for the exact-prefix snapshot cache.
//!
//! Snapshot-only families such as Qwen3.8 carry model-owned recurrent state.
//! Their per-entry footprint can exceed the old fixed 512 MiB default, so a
//! single healthy insert followed by LRU enforcement could evict the previous
//! live conversation snapshot and produce a 0% multi-turn hit rate. This module
//! derives a bounded startup default from `config.json`; operator-provided caps
//! still win.

use std::path::Path;

use serde_json::Value;

use crate::execution::config_fields::{
    HEAD_DIM_KEYS, HIDDEN_SIZE_KEYS, LAYER_COUNT_KEYS, NUM_HEADS_KEYS, get_u64, text_config,
};
use crate::execution::kv_arch::{KvArchKind, estimate_kv_arch_from_config};
use crate::execution::memory_estimate::{QuantHint, estimate_total_memory};

/// Context tokens used when a server context size is not useful or is larger
/// than the snapshot-store sizing target.
pub const MODEL_AWARE_SNAPSHOT_CONTEXT_TOKENS: u64 = 8192;
/// Target number of representative snapshots for the implicit default.
pub const MODEL_AWARE_SNAPSHOT_TARGET_ENTRIES: u64 = 6;
/// Fraction of currently available host/unified memory available to the
/// implicit snapshot-cache raise. This keeps a metadata-derived default from
/// consuming most of a constrained machine while explicit operator caps remain
/// authoritative.
pub const MODEL_AWARE_SNAPSHOT_AVAILABLE_MEMORY_DENOMINATOR: u64 = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotCapacityRecommendation {
    pub capacity_bytes: usize,
    pub entry_bytes: usize,
    pub representative_tokens: u64,
    pub kv_bytes_at_representative_tokens: usize,
    pub fixed_state_bytes: usize,
    pub available_ceiling_bytes: Option<usize>,
    pub target_entries: u64,
    pub architecture: String,
}

#[must_use]
pub fn recommend_model_snapshot_capacity(
    model_dir: &Path,
    context_size: usize,
    current_capacity_bytes: usize,
) -> Option<SnapshotCapacityRecommendation> {
    let representative_tokens = representative_tokens(context_size);
    let config_str = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let config: Value = serde_json::from_str(&config_str).ok()?;
    let available_memory_bytes = estimate_total_memory(
        model_dir,
        representative_tokens,
        1,
        QuantHint::Default,
        false,
    )
    .available_bytes;
    let available_memory_bytes = (available_memory_bytes > 0).then_some(available_memory_bytes);
    recommend_model_snapshot_capacity_from_config(
        &config,
        context_size,
        current_capacity_bytes,
        available_memory_bytes,
    )
}

#[must_use]
pub fn recommend_model_snapshot_capacity_from_config(
    config: &Value,
    context_size: usize,
    current_capacity_bytes: usize,
    available_memory_bytes: Option<u64>,
) -> Option<SnapshotCapacityRecommendation> {
    let representative_tokens = representative_tokens(context_size);
    let arch = estimate_kv_arch_from_config(config, representative_tokens, false, 1)?;
    let fixed_state_bytes = qwen3_next_fixed_state_bytes(config);
    if !snapshot_family_is_model_aware(config, arch.kind, fixed_state_bytes) {
        return None;
    }

    let kv_bytes = usize::try_from(arch.total_bytes).unwrap_or(usize::MAX);
    let entry_bytes = kv_bytes.saturating_add(fixed_state_bytes);
    if entry_bytes == 0 {
        return None;
    }

    let target_capacity = entry_bytes
        .saturating_mul(usize::try_from(MODEL_AWARE_SNAPSHOT_TARGET_ENTRIES).unwrap_or(usize::MAX));
    let available_ceiling_bytes = available_memory_bytes.and_then(|bytes| {
        bytes
            .checked_div(MODEL_AWARE_SNAPSHOT_AVAILABLE_MEMORY_DENOMINATOR)
            .and_then(|ceiling| usize::try_from(ceiling).ok())
            .filter(|ceiling| *ceiling > 0)
    });
    let bounded_capacity = available_ceiling_bytes
        .map(|ceiling| target_capacity.min(ceiling))
        .unwrap_or(target_capacity)
        .max(current_capacity_bytes);

    Some(SnapshotCapacityRecommendation {
        capacity_bytes: bounded_capacity,
        entry_bytes,
        representative_tokens,
        kv_bytes_at_representative_tokens: kv_bytes,
        fixed_state_bytes,
        available_ceiling_bytes,
        target_entries: MODEL_AWARE_SNAPSHOT_TARGET_ENTRIES,
        architecture: arch.detail,
    })
}

#[must_use]
pub fn representative_tokens(context_size: usize) -> u64 {
    let context_size = u64::try_from(context_size).unwrap_or(MODEL_AWARE_SNAPSHOT_CONTEXT_TOKENS);
    context_size
        .max(crate::server::prompt_cache::PromptCacheConfig::DEFAULT_MIN_PREFIX_TOKENS as u64)
        .min(MODEL_AWARE_SNAPSHOT_CONTEXT_TOKENS)
}

fn model_type(config: &Value) -> String {
    let text = text_config(config);
    text.get("model_type")
        .and_then(Value::as_str)
        .or_else(|| config.get("model_type").and_then(Value::as_str))
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn snapshot_family_is_model_aware(
    config: &Value,
    kind: KvArchKind,
    fixed_state_bytes: usize,
) -> bool {
    if fixed_state_bytes > 0 {
        return true;
    }
    let mt = model_type(config);
    if matches!(kind, KvArchKind::Hybrid | KvArchKind::PureSsm) {
        return true;
    }
    matches!(
        mt.as_str(),
        "gemma4"
            | "jamba"
            | "falcon_h1"
            | "lfm2"
            | "bailing_moe_linear"
            | "nemotron_h"
            | "mamba"
            | "mamba2"
            | "granitemoehybrid"
            | "muse_glimmer"
            | "plamo2"
    )
}

fn qwen3_next_fixed_state_bytes(config: &Value) -> usize {
    let text = text_config(config);
    let mt = model_type(config);
    if !matches!(mt.as_str(), "qwen3_next" | "qwen3_5" | "qwen3_5_moe") {
        return 0;
    }
    let Some(num_layers) = get_u64(text, LAYER_COUNT_KEYS) else {
        return 0;
    };
    let Some(interval) = get_u64(text, &["full_attention_interval"]).filter(|v| *v > 0) else {
        return 0;
    };
    let linear_layers = (0..num_layers)
        .filter(|layer| !(layer + 1).is_multiple_of(interval))
        .count() as u64;
    if linear_layers == 0 {
        return 0;
    }

    let num_heads = get_u64(text, NUM_HEADS_KEYS).unwrap_or(1).max(1);
    let key_head_dim = get_u64(text, &["linear_key_head_dim"])
        .or_else(|| get_u64(text, HEAD_DIM_KEYS))
        .or_else(|| {
            get_u64(text, HIDDEN_SIZE_KEYS).and_then(|hidden| hidden.checked_div(num_heads))
        })
        .unwrap_or(128);
    let value_head_dim = get_u64(text, &["linear_value_head_dim"]).unwrap_or(key_head_dim);
    let linear_num_key_heads = get_u64(text, &["linear_num_key_heads"]).unwrap_or(16);
    let linear_num_value_heads = get_u64(text, &["linear_num_value_heads"]).unwrap_or(64);
    let conv_kernel_dim = get_u64(text, &["linear_conv_kernel_dim"]).unwrap_or(4);

    let value_dim = linear_num_value_heads.saturating_mul(value_head_dim);
    let key_dim = linear_num_key_heads.saturating_mul(key_head_dim);
    let state_elems = value_dim.saturating_mul(key_head_dim);
    let conv_elems = conv_kernel_dim
        .saturating_sub(1)
        .saturating_mul(key_dim.saturating_mul(2).saturating_add(value_dim));
    let bytes_per_layer = state_elems.saturating_add(conv_elems).saturating_mul(2);
    usize::try_from(bytes_per_layer.saturating_mul(linear_layers)).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::prompt_cache::PromptCacheConfig;
    use serde_json::json;

    #[test]
    fn qwen38_27b_default_holds_multiple_representative_snapshots() {
        let cfg = json!({
            "model_type": "qwen3_next",
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4
        });

        let rec = recommend_model_snapshot_capacity_from_config(
            &cfg,
            131_072,
            PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            Some(64 * 1024 * 1024 * 1024),
        )
        .expect("qwen3-next recommendation");

        assert_eq!(
            rec.representative_tokens,
            MODEL_AWARE_SNAPSHOT_CONTEXT_TOKENS
        );
        assert_eq!(rec.kv_bytes_at_representative_tokens, 536_870_912);
        assert_eq!(rec.fixed_state_bytes, 78_446_592);
        assert_eq!(rec.entry_bytes, 615_317_504);
        assert_eq!(rec.capacity_bytes, 3_691_905_024);
        assert!(rec.capacity_bytes > PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES);
    }

    #[test]
    fn available_memory_clamps_implicit_raise_but_not_below_fallback() {
        let cfg = json!({
            "model_type": "qwen3_next",
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4
        });

        let rec = recommend_model_snapshot_capacity_from_config(
            &cfg,
            8192,
            PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            Some(8 * 1024 * 1024 * 1024),
        )
        .expect("qwen3-next recommendation");

        assert_eq!(rec.available_ceiling_bytes, Some(2 * 1024 * 1024 * 1024));
        assert_eq!(rec.capacity_bytes, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn standard_attention_keeps_legacy_default() {
        let cfg = json!({
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128
        });

        assert!(
            recommend_model_snapshot_capacity_from_config(
                &cfg,
                8192,
                PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
                Some(64 * 1024 * 1024 * 1024),
            )
            .is_none()
        );
    }
}
