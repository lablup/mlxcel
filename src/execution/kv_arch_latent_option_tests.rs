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

//! KV sizing under `--mla-latent-quant` (issue #3179): the estimate follows
//! the precision the run stores, and the advisor never follows it.

use mlxcel_core::cache::turbo::resolve_kv_cache_mode_for_model;
use mlxcel_core::mla::MlaLatentQuant::{Auto, Fp16, Int4, Int8};
use mlxcel_core::mla::{MLA_LATENT_CACHE_FAMILIES, with_mla_latent_quant};
use serde_json::{Value, json};

use super::*;
use crate::execution::kv_cache_advisor::{KvContextRange, recommend_kv_cache_mode};

/// Shipping latent geometry (rank 512, rope 64) over four layers.
fn mla_text(model_type: &str) -> Value {
    json!({
        "model_type": model_type,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "hidden_size": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
    })
}

/// Bytes per token per layer the estimate reports for the mode the run
/// resolves `requested` to.
fn rate(model_type: &str, requested: KVCacheMode) -> u64 {
    let (mode, _) = resolve_kv_cache_mode_for_model(requested, model_type);
    let estimate =
        estimate_kv_arch_from_config(&mla_text(model_type), 2048, mode, 1).expect("estimate");
    assert_eq!(estimate.kind, KvArchKind::MlaCompressed, "{model_type}");
    estimate.marginal_bytes_per_token / 4
}

/// Acceptance: a rank-512 / rope-64 4-bit latent is sized at 416 bytes per
/// token per layer, whatever KV cache mode was requested, and a DSA slot adds
/// its int8 indexer key (128 packed bytes and two groups of scale and bias).
#[test]
fn int4_sizes_416_bytes_per_token_per_layer_plus_the_int8_indexer_key() {
    with_mla_latent_quant(Int4, || {
        for family in ["deepseek_v3", "kimi_vl", "glm4_moe_lite"] {
            for requested in [
                KVCacheMode::Fp16,
                KVCacheMode::Int8,
                KVCacheMode::Turbo4Asym,
            ] {
                assert_eq!(rate(family, requested), 416, "{family}/{requested}");
            }
        }
        for family in ["deepseek_v32", "deepseek_v3.2", "glm_moe_dsa"] {
            assert_eq!(rate(family, KVCacheMode::Fp16), 416 + 136, "{family}");
        }
        // DeepSeek V2 holds the latent while it absorbs.
        if mlxcel_core::mla::absorbed_enabled() {
            assert_eq!(rate("deepseek_v2", KVCacheMode::Fp16), 416);
        }
        // LongCat has two latent caches per decoder layer.
        assert_eq!(rate("longcat_flash", KVCacheMode::Fp16), 2 * 416);
    });
}

/// Acceptance: a forced `int8` on GLM-4.7-Flash and on a DSA type is sized at
/// the int8 rate the default gate withholds.
#[test]
fn a_forced_int8_sizes_the_int8_latent_on_the_gated_off_families() {
    with_mla_latent_quant(Int8, || {
        assert_eq!(rate("glm4_moe_lite", KVCacheMode::Fp16), 672);
        assert_eq!(rate("deepseek_v32", KVCacheMode::Fp16), 808);
        assert_eq!(rate("glm_moe_dsa", KVCacheMode::Int8), 808);
        assert_eq!(rate("deepseek_v3", KVCacheMode::Fp16), 672);
    });
    with_mla_latent_quant(Auto, || {
        assert_eq!(rate("glm4_moe_lite", KVCacheMode::Int8), 1152);
        assert_eq!(rate("deepseek_v32", KVCacheMode::Int8), 1408);
        assert_eq!(rate("deepseek_v3", KVCacheMode::Int8), 672);
        assert_eq!(rate("deepseek_v3", KVCacheMode::Fp16), 1152);
    });
    with_mla_latent_quant(Fp16, || {
        assert_eq!(rate("deepseek_v3", KVCacheMode::Int8), 1152);
        assert_eq!(rate("deepseek_v32", KVCacheMode::Int8), 1408);
    });
}

/// Acceptance: the advisor never suggests an explicit value. Its advice under
/// every option value is the advice under `auto`.
#[test]
fn the_advisor_gives_the_default_gates_advice_under_every_option() {
    let advice = |family: &str| -> Vec<_> {
        [KvArchKind::MlaCompressed, KvArchKind::MlaDecompressed]
            .into_iter()
            .flat_map(|kind| {
                KvContextRange::all()
                    .into_iter()
                    .map(move |range| recommend_kv_cache_mode(kind, family, range))
            })
            .map(|a| (a.suggested, a.also_consider, a.rationale))
            .collect()
    };
    for &family in MLA_LATENT_CACHE_FAMILIES
        .iter()
        .chain(&["deepseek_v2", "llama"])
    {
        let default = with_mla_latent_quant(Auto, || advice(family));
        for option in [Fp16, Int8, Int4] {
            assert_eq!(
                with_mla_latent_quant(option, || advice(family)),
                default,
                "{family}/{option}"
            );
        }
        for (_, _, rationale) in &default {
            assert!(
                !rationale.contains("mla-latent-quant"),
                "{family}: {rationale}"
            );
        }
    }
    // GLM stays advised fp16 even while the option forces its latent to int8.
    with_mla_latent_quant(Int8, || {
        let a = recommend_kv_cache_mode(
            KvArchKind::MlaCompressed,
            "glm4_moe_lite",
            KvContextRange::Long,
        );
        assert_eq!(a.suggested, KVCacheMode::Fp16);
    });
}
