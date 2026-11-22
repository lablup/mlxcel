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

//! Tests for the compressed-latent MLA cache view: byte accounting, the mode
//! rule, the latent-family list and the wrap/supports contract.

use super::*;

const LITE: MlaGeometry = MlaGeometry {
    num_heads: 16,
    kv_lora_rank: 512,
    qk_nope_head_dim: 128,
    qk_rope_head_dim: 64,
    v_head_dim: 128,
};

/// DeepSeek-V3 geometry: 128 heads make the decompressed layout eight times
/// worse again while the latent side does not move at all, which is the
/// whole point of the accounting.
const V3: MlaGeometry = MlaGeometry {
    num_heads: 128,
    kv_lora_rank: 512,
    qk_nope_head_dim: 128,
    qk_rope_head_dim: 64,
    v_head_dim: 128,
};

#[test]
fn latent_bytes_per_token_is_the_issue_s_576_floats() {
    assert_eq!(latent_bytes_per_token(&LITE, 1), 576);
    assert_eq!(latent_bytes_per_token(&LITE, 2), 1152);
    // Independent of head count, unlike the decompressed layout.
    assert_eq!(
        latent_bytes_per_token(&V3, 2),
        latent_bytes_per_token(&LITE, 2)
    );
}

#[test]
fn decompressed_bytes_per_token_scales_with_head_count() {
    // 16 heads * (192 key + 128 value) = 5120 elements.
    assert_eq!(decompressed_bytes_per_token(&LITE, 1), 5120);
    assert_eq!(decompressed_bytes_per_token(&V3, 1), 40960);
}

#[test]
fn absorption_reduces_the_per_token_cache_by_the_head_count_ratio() {
    let before = decompressed_bytes_per_token(&LITE, 2) as f64;
    let after = latent_bytes_per_token(&LITE, 2) as f64;
    assert!(
        (before / after - 8.888).abs() < 0.01,
        "DeepSeek-V2-Lite ratio was {}",
        before / after
    );
    let before_v3 = decompressed_bytes_per_token(&V3, 2) as f64;
    let after_v3 = latent_bytes_per_token(&V3, 2) as f64;
    assert!(
        (before_v3 / after_v3 - 71.1).abs() < 0.1,
        "DeepSeek-V3 ratio was {}",
        before_v3 / after_v3
    );
}

#[test]
fn wrap_declines_a_turbo_kv_cache() {
    let mut cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
    let err = MlaLatentCache::wrap(&mut cache, LITE).unwrap_err();
    assert!(err.contains("fp16 or int8"), "{err}");
    // `supports` must give the same answer, since the fallback branch in a
    // family reads it instead of `wrap`.
    assert!(MlaLatentCache::supports(&cache, LITE).is_err());
}

#[test]
fn wrap_accepts_the_affine_latent_mode_at_its_byte_rate() {
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let view = MlaLatentCache::wrap(&mut cache, LITE).unwrap();
    assert_eq!(view.bytes_per_token(2), 672);
}

#[test]
fn a_rank_that_is_not_whole_groups_declines_the_quantized_latent() {
    // The family then takes its decompressed fallback, exactly as it does
    // for a mode the latent cannot hold.
    const ODD: MlaGeometry = MlaGeometry {
        kv_lora_rank: 96,
        ..LITE
    };
    let cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let err = MlaLatentCache::supports(&cache, ODD).unwrap_err();
    assert!(err.contains("kv_lora_rank 96"), "{err}");
    assert!(MlaLatentCache::supports(&KVCache::new(), ODD).is_ok());
}

#[test]
fn supports_declines_a_paged_backed_fp16_cache() {
    // The mode check passes (FP16), so the paged decline is the one that
    // must fire: the pool's single head dim cannot hold the asymmetric
    // latent split. `kv_arch` relies on `deepseek_v2` never reaching this
    // branch (the family does not opt into the paged backend), so the
    // runtime side of that contract is pinned here (#3153).
    use crate::cache::{PagedBlockPool, PagedKvLayout, PagedSequenceState};
    use std::cell::RefCell;
    use std::rc::Rc;

    let layout = PagedKvLayout::uniform(1, 16, 16 * 2 * 576 * 2).expect("valid layout");
    let pool = Rc::new(RefCell::new(PagedBlockPool::new(layout.clone())));
    let state = Rc::new(RefCell::new(PagedSequenceState::new(&layout)));
    let mut cache = KVCache::new_paged(pool, state, 0);
    assert_eq!(cache.mode, KVCacheMode::Fp16);
    assert!(cache.is_paged_backed());

    let err = MlaLatentCache::supports(&cache, LITE).unwrap_err();
    assert!(err.contains("paged-backed"), "{err}");
    assert!(!err.contains("FP16 KV cache"), "{err}");
    assert!(MlaLatentCache::wrap(&mut cache, LITE).is_err());
}

#[test]
fn a_latent_cache_stays_trimmable_for_speculative_decode() {
    // Speculative decode gates on `can_trim_prompt_cache` and then rejects
    // the tokens the draft got wrong with `trim`. Both operate on the
    // sequence axis, which is axis 2 in the latent layout exactly as in the
    // decompressed one, so a rejected draft token removes one latent row.
    // If that were not true, a rejection would silently corrupt the cache.
    const G: MlaGeometry = MlaGeometry {
        num_heads: 4,
        kv_lora_rank: 32,
        qk_nope_head_dim: 16,
        qk_rope_head_dim: 8,
        v_head_dim: 12,
    };
    let mut cache = KVCache::new();
    {
        let mut view = MlaLatentCache::wrap(&mut cache, G).unwrap();
        let ckv = crate::ffi::zeros(&[1, 1, 6, 32], crate::dtype::FLOAT16);
        let kpe = crate::ffi::zeros(&[1, 1, 6, 8], crate::dtype::FLOAT16);
        view.update_and_fetch(ckv, kpe);
        assert_eq!(view.seq_len(), 6);
    }
    assert!(crate::cache::can_trim_prompt_cache(std::slice::from_ref(
        &cache
    )));
    assert_eq!(cache.trim(2), 2);
    let view = MlaLatentCache::wrap(&mut cache, G).unwrap();
    assert_eq!(view.seq_len(), 4);
}

/// Every family in `MLA_LATENT_CACHE_FAMILIES` always caches a latent
/// pair, and `deepseek_v2` does not. `deepseek_v2` has a decompressed
/// fallback, so a quantized mode there lands on ordinary per-head K/V rows
/// and must keep working.
#[test]
fn only_the_fallback_less_families_are_latent_families() {
    for family in [
        "glm4_moe_lite",
        "deepseek_v3",
        "kimi_k2",
        "joyai_llm_flash",
        "kimi_vl",
        "kimi_k25",
        "youtu",
        "youtu_llm",
        "youtu_vl",
        "deepseek_v32",
        "deepseek_v3.2",
        "glm_moe_dsa",
        "kimi_linear",
        "kimi_k3",
        "longcat_flash",
        "longcat_flash_ngram",
    ] {
        assert!(caches_mla_latent_pair(family), "{family}");
    }
    assert_eq!(MLA_LATENT_CACHE_FAMILIES.len(), 16);
    assert!(!caches_mla_latent_pair("deepseek_v2"));
    assert!(!caches_mla_latent_pair("llama"));
    assert!(!caches_mla_latent_pair(""));
}

/// The DSA trio needs its own entries, not coverage inherited from
/// `deepseek_v3`. `caches_mla_latent_pair` matches exactly, so a family
/// whose `model_type` merely extends an entry is not covered by it, and
/// `glm_moe_dsa` shares nothing textual with the model it wraps.
#[test]
fn the_dsa_families_are_listed_in_their_own_right() {
    for family in ["deepseek_v32", "deepseek_v3.2", "glm_moe_dsa"] {
        assert!(
            MLA_LATENT_CACHE_FAMILIES.contains(&family),
            "{family} must be listed explicitly; `deepseek_v3` does not cover it"
        );
    }
}

/// Exact match, not a prefix walk: the latent packing belongs to one
/// concrete attention implementation, and a neighbouring `model_type` that
/// happens to share a prefix must not inherit the answer.
#[test]
fn latent_family_lookup_is_exact_and_case_insensitive() {
    assert!(caches_mla_latent_pair("  GLM4_MoE_Lite  "));
    assert!(caches_mla_latent_pair("  DeepSeek_V3.2  "));
    assert!(!caches_mla_latent_pair("glm4_moe_lite_vl"));
    assert!(!caches_mla_latent_pair("glm4_moe"));
    assert!(!caches_mla_latent_pair("glm_moe"));
    assert!(!caches_mla_latent_pair("deepseek_v32_vl"));
}

/// The rule `MlaLatentCache::supports` has always applied, now callable on
/// its own so the model-level resolver states it once rather than twice.
#[test]
fn latent_layout_accepts_fp16_and_int8_and_no_turbo_mode() {
    for mode in [KVCacheMode::Fp16, KVCacheMode::Int8] {
        assert!(latent_layout_supports_mode(mode).is_ok(), "{mode}");
    }
    for mode in [
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo3Asym,
        KVCacheMode::Turbo4,
        KVCacheMode::Turbo4Delegated,
    ] {
        let err = latent_layout_supports_mode(mode).unwrap_err();
        assert!(err.contains("fp16 or int8"), "{err}");
        assert!(err.contains(&mode.to_string()), "{err}");
    }
}

#[test]
fn wrap_accepts_the_default_fp16_cache() {
    let mut cache = KVCache::new();
    let view = MlaLatentCache::wrap(&mut cache, LITE).unwrap();
    assert_eq!(view.seq_len(), 0);
    assert_eq!(view.bytes_per_token(2), 1152);
}
