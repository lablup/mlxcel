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

//! Unit tests for [`super`] (`snapshot_sizing.rs`).

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

// Real-checkpoint geometry for the attention-cache snapshot families (#1761).
// Each config is the on-disk `config.json` reduced to the fields the estimator
// reads, keeping the wrapper / `text_config` split and the `_text` spellings
// the checkpoints under `/home/inureyes/models/mlx` actually carry.

/// `gemma-3-4b-it-4bit`: the text config omits the head fields, so the
/// Gemma 3 defaults (4 KV heads x 256) are what is being sized.
fn gemma3_4b_vlm_config() -> Value {
    json!({
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 2560,
            "num_hidden_layers": 34,
            "sliding_window": 1024
        }
    })
}

/// `llama-4-scout-17b-16e-instruct-4bit`: 48 layers x 8 KV heads x 128,
/// chunked attention with no `sliding_window`, so it classifies Standard.
fn llama4_scout_config() -> Value {
    json!({
        "model_type": "llama4",
        "text_config": {
            "model_type": "llama4_text",
            "num_hidden_layers": 48,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_size": 5120,
            "attention_chunk_size": 8192
        }
    })
}

/// `gemma-4-12b-it-4bit`: 40 sliding layers (8 x 256) and 8 global layers
/// with their own geometry (1 x 512).
fn gemma4_unified_12b_config() -> Value {
    let layer_types: Vec<&str> = (0..48)
        .map(|i| {
            if (i + 1) % 6 == 0 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    json!({
        "model_type": "gemma4_unified",
        "text_config": {
            "model_type": "gemma4_unified_text",
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 512,
            "hidden_size": 3840,
            "sliding_window": 1024,
            "attention_k_eq_v": true,
            "layer_types": layer_types
        }
    })
}

/// `muse-glimmer-30b-4bit`: 52 layers x 2 KV heads x 128, 13 of them global.
fn muse_glimmer_30b_config() -> Value {
    let layer_types: Vec<&str> = (0..52)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    json!({
        "model_type": "muse_glimmer",
        "text_config": {
            "model_type": "muse_glimmer_text",
            "num_hidden_layers": 52,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "hidden_size": 6656,
            "sliding_window": 2048,
            "layer_types": layer_types
        }
    })
}

/// AFMoE (Trinity) carries no `text_config`; its `model_type` is top-level.
fn afmoe_config() -> Value {
    json!({
        "model_type": "afmoe",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_size": 2048,
        "sliding_window": 2048,
        "global_attn_every_n_layers": 4
    })
}

/// Per-token bytes of an unwindowed bf16 snapshot, K and V.
const fn kv_bytes_per_token(layers: usize, kv_heads: usize, head_dim: usize) -> usize {
    layers * kv_heads * head_dim * 2 * 2
}

#[test]
fn attention_cache_families_take_the_model_aware_path() {
    let tokens = MODEL_AWARE_SNAPSHOT_CONTEXT_TOKENS as usize;
    let cases = [
        (
            "gemma3_text",
            gemma3_4b_vlm_config(),
            kv_bytes_per_token(34, 4, 256),
        ),
        (
            "llama4_text",
            llama4_scout_config(),
            kv_bytes_per_token(48, 8, 128),
        ),
        (
            "gemma4_unified_text",
            gemma4_unified_12b_config(),
            kv_bytes_per_token(40, 8, 256) + kv_bytes_per_token(8, 1, 512),
        ),
        (
            "muse_glimmer_text",
            muse_glimmer_30b_config(),
            kv_bytes_per_token(52, 2, 128),
        ),
        ("afmoe", afmoe_config(), kv_bytes_per_token(32, 4, 128)),
    ];
    for (spelling, cfg, per_token) in cases {
        let rec = recommend_model_snapshot_capacity_from_config(
            &cfg,
            0,
            PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            Some(100 * 1024 * 1024 * 1024),
        )
        .unwrap_or_else(|| panic!("{spelling} must get the model-aware snapshot default"));
        // Unwindowed: a boundary snapshot holds every layer at full length.
        assert_eq!(rec.entry_bytes, per_token * tokens, "{spelling}");
        // Six representative entries fit under the quarter-of-75-GiB ceiling.
        assert_eq!(rec.capacity_bytes, rec.entry_bytes * 6, "{spelling}");
        assert!(
            rec.capacity_bytes > PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            "{spelling}"
        );
        // Snapshot-only families never donate into the dense KV store.
        assert!(
            recommend_model_kv_store_capacity_from_config(
                &cfg,
                0,
                PromptCacheConfig::DEFAULT_CAPACITY_BYTES,
                Some(100 * 1024 * 1024 * 1024),
            )
            .is_none(),
            "{spelling}"
        );
    }
}

#[test]
fn llama4_scout_entry_matches_the_measured_per_token_footprint() {
    // #1752 validation run: a 2352-token Scout completion entry stored
    // 503_317_008 bytes, a 2560-token step-aligned buffer at 192 KiB/token.
    let rec = recommend_model_snapshot_capacity_from_config(
        &llama4_scout_config(),
        0,
        PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
        Some(100 * 1024 * 1024 * 1024),
    )
    .expect("recommendation");
    let estimated_per_token = rec.entry_bytes as f64 / rec.representative_tokens as f64;
    let measured_per_token = 503_317_008f64 / 2560.0;
    assert!((measured_per_token / estimated_per_token - 1.0).abs() < 0.01);
    // The old fixed 512 MiB bucket could not hold one 8192-token entry.
    assert!(rec.entry_bytes > PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES);
}

#[test]
fn text_suffix_spellings_match_their_wrapper_spellings() {
    for (wrapper, text) in [
        ("gemma3", "gemma3_text"),
        ("gemma4", "gemma4_text"),
        ("gemma4_unified", "gemma4_unified_text"),
        ("llama4", "llama4_text"),
        ("muse_glimmer", "muse_glimmer_text"),
    ] {
        // Decoder spelling alone (no wrapper `model_type`), as a text-only
        // conversion of the checkpoint would carry it.
        let text_only = json!({
            "text_config": {
                "model_type": text,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 64
            }
        });
        assert_eq!(model_types(&text_only), vec![wrapper.to_string()]);
        assert!(
            snapshot_family_is_model_aware(&text_only, KvArchKind::Standard, 0),
            "{text}"
        );
        // Mixed-case spelling normalizes the same way.
        let upper = json!({ "model_type": text.to_ascii_uppercase() });
        assert_eq!(model_types(&upper), vec![wrapper.to_string()]);
    }
}

#[test]
fn per_token_estimates_match_real_checkpoint_boundary_snapshots() {
    // First-turn `Boundary` entries (prefill from token 0, no sliding layer
    // wrapped yet) logged by `mlxcel-server` on GB10 for #1761, as
    // (config, token_len, entry_bytes).
    let measured = [
        (
            "gemma3_text",
            gemma3_4b_vlm_config(),
            2614u64,
            368_174_024u64,
        ),
        (
            "gemma4_unified_text",
            gemma4_unified_12b_config(),
            2614,
            902_694_208,
        ),
        (
            "muse_glimmer_text",
            muse_glimmer_30b_config(),
            2771,
            148_150_600,
        ),
    ];
    for (spelling, cfg, tokens, bytes) in measured {
        let rec = recommend_model_snapshot_capacity_from_config(
            &cfg,
            0,
            PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            Some(100 * 1024 * 1024 * 1024),
        )
        .expect("recommendation");
        let estimated_per_token = rec.entry_bytes as f64 / rec.representative_tokens as f64;
        let measured_per_token = bytes as f64 / tokens as f64;
        assert!(
            (measured_per_token / estimated_per_token - 1.0).abs() < 0.02,
            "{spelling}: measured {measured_per_token} vs estimated {estimated_per_token}"
        );
    }
}
