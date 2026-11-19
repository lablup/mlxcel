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

//! Sizing of the DSA families' `[kv_latent | indexer key]` cache slot (issue
//! #3163).

use serde_json::{Value, json};

use super::*;

/// DeepSeek-V3.2 attention geometry, cut to four layers. `index_head_dim` is
/// left to its default of 128 unless a test sets it.
fn dsa_text(model_type: &str) -> Value {
    json!({
        "model_type": model_type,
        "num_hidden_layers": 4,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "hidden_size": 7168,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
    })
}

/// The fp16 slot is 640 wide, not 512: the indexer key is cached beside the
/// latent. Before #3163 these families were sized at 1152 bytes per token per
/// layer, 256 short of what the cache holds.
#[test]
fn dsa_fp16_slot_counts_the_indexer_key() {
    for family in ["deepseek_v32", "deepseek_v3.2", "glm_moe_dsa"] {
        let e = estimate_kv_arch_from_config(&dsa_text(family), 2048, KVCacheMode::Fp16, 1)
            .expect("estimate");
        assert_eq!(e.kind, KvArchKind::MlaCompressed, "{family}");
        assert_eq!(e.marginal_bytes_per_token, 4 * (640 + 64) * 2, "{family}");
        assert_eq!(e.total_bytes, 4 * 1408 * 2048, "{family}");
        assert!(e.detail.contains("704 elems"), "{family}: {}", e.detail);
    }
}

/// The DSA families are not quantized-latent families (the top-k overlap gate
/// of #3163 was missed), so an `int8` request is not sized at the affine rate
/// of 808 bytes: the run resolves it to fp16 at startup.
#[test]
fn dsa_int8_request_gets_no_affine_rate() {
    for family in ["deepseek_v32", "deepseek_v3.2", "glm_moe_dsa"] {
        let e = estimate_kv_arch_from_config(&dsa_text(family), 2048, KVCacheMode::Int8, 1)
            .expect("estimate");
        assert_ne!(e.marginal_bytes_per_token, 4 * 808, "{family}");
        assert!(
            !e.detail.contains("affine-quantized"),
            "{family}: {}",
            e.detail
        );
    }
    // The rate the slot would cost, for the day the families are enabled.
    assert_eq!(
        mlxcel_core::mla::quantized_latent_bytes_per_token(640, 64, 8, 64),
        808
    );
}

/// `index_head_dim` is read from the config when it is there.
#[test]
fn dsa_slot_follows_a_declared_index_head_dim() {
    let mut cfg = dsa_text("deepseek_v32");
    cfg["index_head_dim"] = json!(64);
    let e = estimate_kv_arch_from_config(&cfg, 2048, KVCacheMode::Fp16, 1).expect("estimate");
    assert_eq!(e.marginal_bytes_per_token, 4 * (512 + 64 + 64) * 2);
}

/// A latent-pair family with no indexer keeps the latent-only width.
#[test]
fn non_dsa_latent_pair_families_keep_the_latent_only_width() {
    for family in ["deepseek_v3", "glm4_moe_lite", "kimi_k3"] {
        let e = estimate_kv_arch_from_config(&dsa_text(family), 2048, KVCacheMode::Fp16, 1)
            .expect("estimate");
        assert_eq!(e.marginal_bytes_per_token, 4 * 1152, "{family}");
    }
}
