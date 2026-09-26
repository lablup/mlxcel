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
use crate::execution::kv_arch::{
    KvArchKind, estimate_kv_arch_from_config, estimate_kv_arch_unwindowed_from_config,
};
use crate::execution::memory_estimate::{QuantHint, estimate_total_memory};

/// Context tokens used when a server context size is not useful or is larger
/// than the snapshot-store sizing target.
pub const MODEL_AWARE_SNAPSHOT_CONTEXT_TOKENS: u64 = 8192;
/// Target number of representative snapshots for the implicit default.
pub const MODEL_AWARE_SNAPSHOT_TARGET_ENTRIES: u64 = 6;
/// Fraction of the memory left after the model's own estimated footprint
/// (weights, KV at the representative length, runtime headroom) available to
/// the implicit snapshot-cache raise. This keeps a metadata-derived default from
/// consuming most of a constrained machine while explicit operator caps remain
/// authoritative.
pub const MODEL_AWARE_SNAPSHOT_AVAILABLE_MEMORY_DENOMINATOR: u64 = 4;
/// The one-entry floor may use up to this fraction of available memory.
///
/// Below one representative entry the store rejects every snapshot of that
/// size as "cannot fit even alone", which is a 0% hit rate, not a smaller
/// cache. The floor keeps one entry whenever it takes no more than half of
/// what is free; a host tighter than that keeps the quarter ceiling.
pub const MODEL_AWARE_SNAPSHOT_FLOOR_MEMORY_DENOMINATOR: u64 = 2;

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
    let config = read_config(model_dir)?;
    recommend_model_snapshot_capacity_from_config(
        &config,
        context_size,
        current_capacity_bytes,
        memory_left_after_model(model_dir, representative_tokens(context_size)),
    )
}

/// Model-aware default for the KV prompt-cache store (the longest-prefix
/// trie that dense-KV families donate finished sequences into).
///
/// Same shape as the snapshot recommendation, for the same failure: the
/// store's fixed 2 GiB default holds one entry of about 16k tokens on an 8B
/// Llama and far fewer on a 70B, and an entry larger than the whole store is
/// rejected, so every later turn of that conversation reports `cached=0`
/// (measured: Llama 3.1 8B, a 17k-token conversation, 2.24 GB per entry).
/// The entry is sized window-capped, because the KV store receives a
/// sequence at completion, after any sliding layer has wrapped.
///
/// Returns `None` for snapshot-only families, which never donate into this
/// store, and for families without a KV cache.
#[must_use]
pub fn recommend_model_kv_store_capacity(
    model_dir: &Path,
    context_size: usize,
    current_capacity_bytes: usize,
) -> Option<SnapshotCapacityRecommendation> {
    let config = read_config(model_dir)?;
    recommend_model_kv_store_capacity_from_config(
        &config,
        context_size,
        current_capacity_bytes,
        memory_left_after_model(model_dir, representative_tokens(context_size)),
    )
}

#[must_use]
pub fn recommend_model_kv_store_capacity_from_config(
    config: &Value,
    context_size: usize,
    current_capacity_bytes: usize,
    available_memory_bytes: Option<u64>,
) -> Option<SnapshotCapacityRecommendation> {
    let representative_tokens = representative_tokens(context_size);
    let arch = estimate_kv_arch_from_config(config, representative_tokens, false, 1)?;
    let fixed_state_bytes = qwen3_next_fixed_state_bytes(config);
    if matches!(arch.kind, KvArchKind::PureSsm)
        || snapshot_family_is_model_aware(config, arch.kind, fixed_state_bytes)
    {
        return None;
    }
    let entry_bytes = usize::try_from(arch.total_bytes).unwrap_or(usize::MAX);
    if entry_bytes == 0 {
        return None;
    }
    let (capacity_bytes, available_ceiling_bytes) =
        bounded_capacity(entry_bytes, current_capacity_bytes, available_memory_bytes);
    Some(SnapshotCapacityRecommendation {
        capacity_bytes,
        entry_bytes,
        representative_tokens,
        kv_bytes_at_representative_tokens: entry_bytes,
        fixed_state_bytes: 0,
        available_ceiling_bytes,
        target_entries: MODEL_AWARE_SNAPSHOT_TARGET_ENTRIES,
        architecture: arch.detail,
    })
}

fn read_config(model_dir: &Path) -> Option<Value> {
    let config_str = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    serde_json::from_str(&config_str).ok()
}

/// Memory left once the model itself is resident, the budget both stores
/// are sized from.
///
/// `available_bytes` is measured before the weights load, so on a
/// unified-memory host it still counts the weights' own share: a 31B Gemma 4
/// on a 64 GB Mac would otherwise be allowed a quarter of memory the weights
/// already occupy. An unknown budget (`0`) stays `None`; a known one that the
/// model exhausts stays `Some(0)`, which keeps the compiled-in fallback
/// instead of lifting the ceiling.
fn memory_left_after_model(model_dir: &Path, representative_tokens: u64) -> Option<u64> {
    let estimate = estimate_total_memory(
        model_dir,
        representative_tokens,
        1,
        QuantHint::Default,
        false,
    );
    (estimate.available_bytes > 0).then(|| estimate.slack_bytes())
}

/// Six representative entries, clamped to a quarter of the memory left,
/// floored at one entry when that fits in half of it, and never below the
/// current (compiled-in) capacity. Returns the capacity and the ceiling.
fn bounded_capacity(
    entry_bytes: usize,
    current_capacity_bytes: usize,
    available_memory_bytes: Option<u64>,
) -> (usize, Option<usize>) {
    let target_capacity = entry_bytes
        .saturating_mul(usize::try_from(MODEL_AWARE_SNAPSHOT_TARGET_ENTRIES).unwrap_or(usize::MAX));
    // A zero ceiling is kept as zero: it means the model leaves no room, and
    // the `max` below then falls back to the compiled-in default rather than
    // treating "no room" as "no limit".
    let available_ceiling_bytes = available_memory_bytes.map(|bytes| {
        bytes
            .checked_div(MODEL_AWARE_SNAPSHOT_AVAILABLE_MEMORY_DENOMINATOR)
            .and_then(|ceiling| usize::try_from(ceiling).ok())
            .unwrap_or(usize::MAX)
    });
    // At least one representative entry when memory allows it; see
    // `MODEL_AWARE_SNAPSHOT_FLOOR_MEMORY_DENOMINATOR`.
    let one_entry_floor = match available_memory_bytes {
        Some(bytes) => {
            let room = bytes
                .checked_div(MODEL_AWARE_SNAPSHOT_FLOOR_MEMORY_DENOMINATOR)
                .and_then(|room| usize::try_from(room).ok())
                .unwrap_or(0);
            if entry_bytes <= room { entry_bytes } else { 0 }
        }
        None => entry_bytes,
    };
    let capacity = available_ceiling_bytes
        .map(|ceiling| target_capacity.min(ceiling))
        .unwrap_or(target_capacity)
        .max(one_entry_floor)
        .max(current_capacity_bytes);
    (capacity, available_ceiling_bytes)
}

#[must_use]
pub fn recommend_model_snapshot_capacity_from_config(
    config: &Value,
    context_size: usize,
    current_capacity_bytes: usize,
    available_memory_bytes: Option<u64>,
) -> Option<SnapshotCapacityRecommendation> {
    let representative_tokens = representative_tokens(context_size);
    // Unwindowed on purpose: the history-boundary snapshot is captured right
    // after a prefill forward, before any sliding layer wraps, so a
    // sliding-window family's largest entry holds every layer at full length
    // (Gemma 4 31B: 2.97 GB at 3289 tokens, where the window-capped figure
    // predicts about a third of that). Sizing from the capped figure left the
    // store unable to hold a single entry.
    let arch = estimate_kv_arch_unwindowed_from_config(config, representative_tokens, false, 1)?;
    let fixed_state_bytes = qwen3_next_fixed_state_bytes(config);
    if !snapshot_family_is_model_aware(config, arch.kind, fixed_state_bytes) {
        return None;
    }

    let kv_bytes = usize::try_from(arch.total_bytes).unwrap_or(usize::MAX);
    let entry_bytes = kv_bytes.saturating_add(fixed_state_bytes);
    if entry_bytes == 0 {
        return None;
    }

    let (bounded_capacity, available_ceiling_bytes) =
        bounded_capacity(entry_bytes, current_capacity_bytes, available_memory_bytes);

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
    // `0` is the server's "model default" (no `--ctx-size`), not a zero-token
    // window. Clamping it up to the minimum prefix sized the store for a
    // 32-token snapshot on every default launch, which left the raise far
    // below one real entry.
    if context_size == 0 {
        return MODEL_AWARE_SNAPSHOT_CONTEXT_TOKENS;
    }
    let context_size = u64::try_from(context_size).unwrap_or(MODEL_AWARE_SNAPSHOT_CONTEXT_TOKENS);
    context_size
        .max(crate::server::prompt_cache::PromptCacheConfig::DEFAULT_MIN_PREFIX_TOKENS as u64)
        .min(MODEL_AWARE_SNAPSHOT_CONTEXT_TOKENS)
}

/// Both `model_type` spellings a checkpoint may carry, lowercased and with a
/// trailing `_text` removed.
///
/// A VLM checkpoint names the wrapper at the top level and the decoder inside
/// `text_config` (`gemma4` / `gemma4_text`, `qwen3_5` / `qwen3_5_text`,
/// `granite4_vision` / `granitemoehybrid`). Reading only the decoder spelling
/// missed every Gemma 4 and Qwen 3.5 VLM checkpoint, which left Gemma 4 on the
/// 512 MiB fallback and zeroed Qwen 3.5's recurrent-state term.
fn model_types(config: &Value) -> Vec<String> {
    let text = text_config(config);
    let mut types = Vec::with_capacity(2);
    for raw in [text.get("model_type"), config.get("model_type")]
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        let lower = raw.to_ascii_lowercase();
        let normalized = lower.strip_suffix("_text").unwrap_or(&lower).to_string();
        if !types.contains(&normalized) {
            types.push(normalized);
        }
    }
    types
}

fn snapshot_family_is_model_aware(
    config: &Value,
    kind: KvArchKind,
    fixed_state_bytes: usize,
) -> bool {
    if fixed_state_bytes > 0 {
        return true;
    }
    if matches!(kind, KvArchKind::Hybrid | KvArchKind::PureSsm) {
        return true;
    }
    // Families whose server state is model-owned and reused only through
    // snapshots (`supports_snapshot_reuse`). The attention-only ones among
    // them (Gemma 3/4, Llama 4, AFMoE, Inkling) park a full KV state per
    // entry, which is what outgrows the fixed default on a large checkpoint.
    model_types(config).iter().any(|mt| {
        matches!(
            mt.as_str(),
            "gemma3"
                | "gemma4"
                | "gemma4_unified"
                | "llama4"
                | "afmoe"
                | "inkling"
                | "inkling_mm_model"
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
    })
}

fn qwen3_next_fixed_state_bytes(config: &Value) -> usize {
    let text = text_config(config);
    if !model_types(config)
        .iter()
        .any(|mt| matches!(mt.as_str(), "qwen3_next" | "qwen3_5" | "qwen3_5_moe"))
    {
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

    /// `config.json` of `gemma-4-31b-it-qat-4bit`, reduced to the fields the
    /// estimator reads. The decoder type sits in `text_config` as
    /// `gemma4_text`, which the old single-spelling lookup never matched.
    fn gemma4_31b_vlm_config() -> Value {
        let layer_types: Vec<&str> = (0..60)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect();
        json!({
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "num_hidden_layers": 60,
                "num_attention_heads": 32,
                "num_key_value_heads": 16,
                "num_global_key_value_heads": 4,
                "head_dim": 256,
                "global_head_dim": 512,
                "sliding_window": 1024,
                "layer_types": layer_types,
                "hidden_size": 5376
            }
        })
    }

    #[test]
    fn gemma4_vlm_checkpoint_gets_a_boundary_snapshot_sized_default() {
        let rec = recommend_model_snapshot_capacity_from_config(
            &gemma4_31b_vlm_config(),
            262_144,
            PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            Some(100 * 1024 * 1024 * 1024),
        )
        .expect("gemma4 VLM checkpoints must get the model-aware default");

        // Unwrapped boundary snapshot at 8192 tokens: 901_120 bytes/token,
        // the figure measured on the real checkpoint (2_966_980_240 bytes
        // for 3289 tokens) to within 2%.
        assert_eq!(rec.entry_bytes, 901_120 * 8192);
        let measured_per_token = 2_966_980_240f64 / 3289.0;
        let estimated_per_token = rec.entry_bytes as f64 / 8192.0;
        assert!((measured_per_token / estimated_per_token - 1.0).abs() < 0.02);
        // Six entries would be 44 GB; the quarter-of-available ceiling wins.
        assert_eq!(rec.capacity_bytes, 25 * 1024 * 1024 * 1024);
        assert!(rec.capacity_bytes >= rec.entry_bytes);
    }

    #[test]
    fn a_tight_ceiling_still_holds_one_entry_when_memory_allows() {
        // 24 GiB free: the quarter ceiling (6 GiB) is below one 7.4 GB entry,
        // which would reject every snapshot. One entry fits in half of it.
        let rec = recommend_model_snapshot_capacity_from_config(
            &gemma4_31b_vlm_config(),
            8192,
            PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            Some(24 * 1024 * 1024 * 1024),
        )
        .expect("recommendation");
        assert_eq!(rec.capacity_bytes, rec.entry_bytes);

        // 12 GiB free: one entry would take more than half, keep the ceiling.
        let rec = recommend_model_snapshot_capacity_from_config(
            &gemma4_31b_vlm_config(),
            8192,
            PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            Some(12 * 1024 * 1024 * 1024),
        )
        .expect("recommendation");
        assert_eq!(rec.capacity_bytes, 3 * 1024 * 1024 * 1024);
    }

    #[test]
    fn a_model_that_exhausts_memory_keeps_the_fallback() {
        // Known budget, nothing left after the weights: no raise at all.
        let rec = recommend_model_snapshot_capacity_from_config(
            &gemma4_31b_vlm_config(),
            8192,
            PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            Some(0),
        )
        .expect("recommendation");
        assert_eq!(rec.available_ceiling_bytes, Some(0));
        assert_eq!(
            rec.capacity_bytes,
            PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES
        );
    }

    #[test]
    fn qwen35_vlm_checkpoint_counts_its_recurrent_state() {
        // `qwen3.5-*` checkpoints spell the decoder `qwen3_5_text`.
        let cfg = json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
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
            }
        });
        let rec = recommend_model_snapshot_capacity_from_config(
            &cfg,
            131_072,
            PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            Some(64 * 1024 * 1024 * 1024),
        )
        .expect("qwen3.5 VLM recommendation");
        assert_eq!(rec.fixed_state_bytes, 78_446_592);
    }

    #[test]
    fn an_unset_context_size_sizes_for_the_representative_window() {
        // `context_size == 0` is the default launch (no `--ctx-size`).
        assert_eq!(
            representative_tokens(0),
            MODEL_AWARE_SNAPSHOT_CONTEXT_TOKENS
        );
        assert_eq!(representative_tokens(4096), 4096);
        assert_eq!(
            representative_tokens(131_072),
            MODEL_AWARE_SNAPSHOT_CONTEXT_TOKENS
        );
        let rec = recommend_model_snapshot_capacity_from_config(
            &gemma4_31b_vlm_config(),
            0,
            PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            Some(100 * 1024 * 1024 * 1024),
        )
        .expect("recommendation");
        assert_eq!(
            rec.representative_tokens,
            MODEL_AWARE_SNAPSHOT_CONTEXT_TOKENS
        );
    }

    #[test]
    fn dense_kv_store_default_holds_a_long_conversation() {
        // Llama 3.1 8B: 32 layers x 8 KV heads x 128 x K/V x bf16 = 128 KiB
        // per token. The fixed 2 GiB store rejected a 17k-token entry
        // (measured 2.24 GB); six 8192-token entries is 6 GiB.
        let cfg = json!({
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128
        });
        let rec = recommend_model_kv_store_capacity_from_config(
            &cfg,
            0,
            PromptCacheConfig::DEFAULT_CAPACITY_BYTES,
            Some(100 * 1024 * 1024 * 1024),
        )
        .expect("dense models get a KV-store recommendation");
        assert_eq!(rec.entry_bytes, 128 * 1024 * 8192);
        assert_eq!(rec.capacity_bytes, 6 * 1024 * 1024 * 1024);
        assert!(rec.capacity_bytes > 2_243_952_640);
    }

    #[test]
    fn kv_store_default_skips_snapshot_only_families() {
        assert!(
            recommend_model_kv_store_capacity_from_config(
                &gemma4_31b_vlm_config(),
                0,
                PromptCacheConfig::DEFAULT_CAPACITY_BYTES,
                Some(100 * 1024 * 1024 * 1024),
            )
            .is_none()
        );
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
