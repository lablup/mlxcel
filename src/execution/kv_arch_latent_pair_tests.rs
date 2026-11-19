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

//! Sizing of the quantized latent on the `DeepSeekV3Attention` families
//! (issue #3160), and LongCat's two latent caches per layer (issue #3161).

use serde_json::{Value, json};

use super::*;

/// Moonlight-16B / Kimi-VL text geometry, cut to four layers.
fn v3_text(model_type: &str) -> Value {
    json!({
        "model_type": model_type,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "hidden_size": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
    })
}

#[test]
fn deepseek_v3_int8_latent_is_672_bytes_per_token_per_layer() {
    let e = estimate_kv_arch_from_config(&v3_text("deepseek_v3"), 2048, KVCacheMode::Int8, 1)
        .expect("estimate");
    assert_eq!(e.kind, KvArchKind::MlaCompressed);
    assert_eq!(e.marginal_bytes_per_token, 4 * 672);
    assert_eq!(e.total_bytes, 4 * 672 * 2048);
    assert!(e.detail.contains("affine-quantized"), "{}", e.detail);

    // The fp16 latent of the same checkpoint is unchanged.
    let fp16 = estimate_kv_arch_from_config(&v3_text("deepseek_v3"), 2048, KVCacheMode::Fp16, 1)
        .expect("estimate");
    assert_eq!(fp16.marginal_bytes_per_token, 4 * 1152);
}

/// Kimi-VL nests the `deepseek_v3` decoder under `text_config`; the sizing
/// reads the decoder and lands on the same rate.
#[test]
fn kimi_vl_sizes_its_deepseek_v3_decoder_at_the_quantized_rate() {
    let cfg = json!({ "model_type": "kimi_vl", "text_config": v3_text("deepseek_v3") });
    let e = estimate_kv_arch_from_config(&cfg, 2048, KVCacheMode::Int8, 1).expect("estimate");
    assert_eq!(e.marginal_bytes_per_token, 4 * 672);
    for family in [
        "kimi_k2",
        "joyai_llm_flash",
        "youtu",
        "youtu_llm",
        "youtu_vl",
    ] {
        let e = estimate_kv_arch_from_config(&v3_text(family), 2048, KVCacheMode::Int8, 1)
            .expect("estimate");
        assert_eq!(e.marginal_bytes_per_token, 4 * 672, "{family}");
    }
}

/// `--kv-bits 8` with the default `--kv-skip-last-layer`: three quantized
/// latent layers and one fp16 latent layer.
#[test]
fn deepseek_v3_int8_latent_with_an_fp16_last_layer() {
    let sizing = KvSizing {
        mode: KVCacheMode::Int8,
        fp16_last_layer: true,
    };
    let e =
        estimate_kv_arch_from_config(&v3_text("deepseek_v3"), 2048, sizing, 1).expect("estimate");
    assert_eq!(e.marginal_bytes_per_token, 3 * 672 + 1152);
}

/// A LongCat decoder layer holds two MLA attentions, each with its own latent
/// cache (#3161), so its rate is twice that of a one-cache quantized-latent
/// family with the same geometry, at either mode. Before #3161 the estimate
/// counted one cache per layer and undersized LongCat by half.
#[test]
fn longcat_counts_two_latent_caches_per_layer() {
    for mode in [KVCacheMode::Fp16, KVCacheMode::Int8] {
        let one_cache = estimate_kv_arch_from_config(&v3_text("deepseek_v3"), 2048, mode, 1)
            .expect("estimate")
            .marginal_bytes_per_token;
        for family in ["longcat_flash", "longcat_flash_ngram"] {
            let mut cfg = v3_text(family);
            cfg["num_layers"] = cfg["num_hidden_layers"].take();
            let e = estimate_kv_arch_from_config(&cfg, 2048, mode, 1).expect("estimate");
            assert_eq!(e.marginal_bytes_per_token, 2 * one_cache, "{family} {mode}");
        }
    }
    let mut cfg = v3_text("longcat_flash");
    cfg["num_layers"] = cfg["num_hidden_layers"].take();
    let fp16 = estimate_kv_arch_from_config(&cfg, 2048, KVCacheMode::Fp16, 1).expect("estimate");
    assert_eq!(fp16.marginal_bytes_per_token, 2 * 4 * 1152);
    // Both caches hold the affine latent under int8 (672 bytes per row).
    let int8 = estimate_kv_arch_from_config(&cfg, 2048, KVCacheMode::Int8, 1).expect("estimate");
    assert_eq!(int8.marginal_bytes_per_token, 2 * 4 * 672);
}

/// A family that still writes the cache directly gets no affine rate, and a
/// rank the codec cannot group falls back to the element-width estimate.
#[test]
fn only_quantizable_view_families_get_the_affine_rate() {
    let others = mlxcel_core::mla::MLA_LATENT_CACHE_FAMILIES
        .iter()
        .filter(|family| !mlxcel_core::mla::stores_quantized_latent_pair(family));
    for &family in others {
        let e = estimate_kv_arch_from_config(&v3_text(family), 2048, KVCacheMode::Int8, 1)
            .expect("estimate");
        assert_ne!(e.marginal_bytes_per_token, 4 * 672, "{family}");
        assert!(!e.detail.contains("affine-quantized"), "{}", e.detail);
    }

    let mut odd = v3_text("deepseek_v3");
    odd["kv_lora_rank"] = json!(500);
    let e = estimate_kv_arch_from_config(&odd, 2048, KVCacheMode::Int8, 1).expect("estimate");
    assert!(!e.detail.contains("affine-quantized"), "{}", e.detail);
}
