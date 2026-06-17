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

//! Unit tests for [`crate::execution::kv_cache_advisor`].

use super::*;

/// The five KV-cache modes the issue scopes the advisor to. `Turbo4Delegated`
/// is intentionally excluded.
const ALLOWED_MODES: [KVCacheMode; 5] = [
    KVCacheMode::Fp16,
    KVCacheMode::Int8,
    KVCacheMode::Turbo4Asym,
    KVCacheMode::Turbo4,
    KVCacheMode::Turbo3Asym,
];

const ALL_ARCH_KINDS: [KvArchKind; 6] = [
    KvArchKind::Standard,
    KvArchKind::SlidingWindow,
    KvArchKind::MlaCompressed,
    KvArchKind::MlaDecompressed,
    KvArchKind::Hybrid,
    KvArchKind::PureSsm,
];

#[test]
fn context_range_bucketing() {
    assert_eq!(KvContextRange::from_ctx_len(0), KvContextRange::Short);
    assert_eq!(KvContextRange::from_ctx_len(4_096), KvContextRange::Short);
    assert_eq!(KvContextRange::from_ctx_len(4_097), KvContextRange::Medium);
    assert_eq!(KvContextRange::from_ctx_len(16_384), KvContextRange::Medium);
    assert_eq!(KvContextRange::from_ctx_len(32_768), KvContextRange::Medium);
    assert_eq!(KvContextRange::from_ctx_len(32_769), KvContextRange::Long);
    assert_eq!(KvContextRange::from_ctx_len(131_072), KvContextRange::Long);
}

#[test]
fn short_context_is_always_fp16() {
    for kind in ALL_ARCH_KINDS {
        let advice = recommend_kv_cache_mode(kind, "llama", KvContextRange::Short);
        assert_eq!(
            advice.suggested,
            KVCacheMode::Fp16,
            "short context for {kind:?} should keep fp16"
        );
        assert_eq!(advice.also_consider, None);
    }
}

#[test]
fn standard_medium_suggests_turbo4_asym() {
    let advice = recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Medium);
    assert_eq!(advice.suggested, KVCacheMode::Turbo4Asym);
    assert_eq!(advice.also_consider, Some(KVCacheMode::Int8));
}

#[test]
fn standard_long_non_allowlisted_never_symmetric_turbo4() {
    // Llama is a dense Q4_K_M-style family and is NOT on the allowlist.
    let advice = recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Long);
    assert_eq!(advice.suggested, KVCacheMode::Turbo4Asym);
    assert_ne!(advice.suggested, KVCacheMode::Turbo4);
    assert_eq!(advice.also_consider, Some(KVCacheMode::Turbo3Asym));
}

#[test]
fn allowlisted_family_long_suggests_symmetric_turbo4() {
    for mt in ["qwen3_5", "qwen3_5_moe", "qwen3_next"] {
        let advice = recommend_kv_cache_mode(KvArchKind::Standard, mt, KvContextRange::Long);
        assert_eq!(
            advice.suggested,
            KVCacheMode::Turbo4,
            "allowlisted family {mt} at long context should suggest turbo4"
        );
        assert_eq!(advice.also_consider, Some(KVCacheMode::Turbo4Asym));
    }
}

/// The most important safety invariant: symmetric Turbo4 is suggested only
/// for allowlisted families, never otherwise, in any field, at any range.
#[test]
fn symmetric_turbo4_only_for_allowlisted_families() {
    let off_allowlist = ["llama", "qwen2", "gemma3", "mistral", "qwen3", "phi3", ""];
    for kind in ALL_ARCH_KINDS {
        for range in KvContextRange::all() {
            for mt in off_allowlist {
                let advice = recommend_kv_cache_mode(kind, mt, range);
                assert_ne!(
                    advice.suggested,
                    KVCacheMode::Turbo4,
                    "{kind:?}/{mt}/{range:?} must not suggest symmetric turbo4"
                );
                assert_ne!(
                    advice.also_consider,
                    Some(KVCacheMode::Turbo4),
                    "{kind:?}/{mt}/{range:?} must not list symmetric turbo4"
                );
            }
        }
    }
}

#[test]
fn pure_ssm_is_always_fp16() {
    for range in KvContextRange::all() {
        let advice = recommend_kv_cache_mode(KvArchKind::PureSsm, "mamba", range);
        assert_eq!(advice.suggested, KVCacheMode::Fp16);
        assert_eq!(advice.also_consider, None);
    }
}

#[test]
fn mla_uses_int8_never_turbo() {
    for kind in [KvArchKind::MlaCompressed, KvArchKind::MlaDecompressed] {
        // Short stays fp16.
        assert_eq!(
            recommend_kv_cache_mode(kind, "deepseek_v3", KvContextRange::Short).suggested,
            KVCacheMode::Fp16
        );
        for range in [KvContextRange::Medium, KvContextRange::Long] {
            let advice = recommend_kv_cache_mode(kind, "deepseek_v3", range);
            assert_eq!(advice.suggested, KVCacheMode::Int8);
            // MLA must never get a Walsh-Hadamard Turbo mode.
            for mode in [
                KVCacheMode::Turbo4,
                KVCacheMode::Turbo4Asym,
                KVCacheMode::Turbo3Asym,
            ] {
                assert_ne!(advice.suggested, mode);
                assert_ne!(advice.also_consider, Some(mode));
            }
        }
    }
}

/// Every recommendation is one of the five KV-cache modes named in the
/// issue, and never `Turbo4Delegated`. Because the output is always a
/// `KVCacheMode` (a KV-cache-only storage setting), it cannot imply a
/// bf16 to f16 promotion of quantized model weights (the #289 landmine).
#[test]
fn recommendation_is_always_an_allowed_kv_cache_mode() {
    let model_types = ["llama", "qwen3_5", "deepseek_v3", "mamba", "gemma3", ""];
    for kind in ALL_ARCH_KINDS {
        for range in KvContextRange::all() {
            for mt in model_types {
                let advice = recommend_kv_cache_mode(kind, mt, range);
                assert!(
                    ALLOWED_MODES.contains(&advice.suggested),
                    "{kind:?}/{mt}/{range:?} suggested an out-of-scope mode: {:?}",
                    advice.suggested
                );
                assert_ne!(advice.suggested, KVCacheMode::Turbo4Delegated);
                if let Some(also) = advice.also_consider {
                    assert!(
                        ALLOWED_MODES.contains(&also),
                        "{kind:?}/{mt}/{range:?} also_consider out-of-scope: {also:?}"
                    );
                    assert_ne!(also, KVCacheMode::Turbo4Delegated);
                }
            }
        }
    }
}

#[test]
fn advise_returns_empty_for_missing_config() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(advise_kv_cache_modes(tmp.path()).is_empty());
}

#[test]
fn advise_classifies_standard_config_for_all_ranges() {
    let cfg = serde_json::json!({
        "model_type": "llama",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128
    });
    let (kind, mt) = arch_and_type_from_config(&cfg).unwrap();
    assert_eq!(kind, KvArchKind::Standard);
    assert_eq!(mt, "llama");

    let advices: Vec<_> = KvContextRange::all()
        .into_iter()
        .map(|r| recommend_kv_cache_mode(kind, &mt, r))
        .collect();
    assert_eq!(advices.len(), 3);
    assert_eq!(advices[0].context_range, KvContextRange::Short);
    assert_eq!(advices[0].suggested, KVCacheMode::Fp16);
    assert_eq!(advices[1].suggested, KVCacheMode::Turbo4Asym);
    assert_eq!(advices[2].suggested, KVCacheMode::Turbo4Asym);
}

/// Computing advice must not change the runtime default. With no opt-in
/// flags, the resolver still returns `Fp16`.
#[test]
fn default_kv_cache_mode_unchanged_without_opt_in() {
    use crate::cli::turbo_args::resolve_kv_cache_mode;

    // Build advice (the new behavior) ...
    let _ = recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Long);
    // ... the default resolution is still fp16.
    let resolved = resolve_kv_cache_mode(None, None, None).unwrap();
    assert_eq!(resolved, KVCacheMode::Fp16);
}

#[test]
fn render_block_states_advisory_and_no_weight_change() {
    let advices = vec![
        recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Short),
        recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Medium),
        recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Long),
    ];
    let block = render_kv_cache_advice(&advices);
    assert!(block.contains("advisory"));
    assert!(block.contains("opt-in") || block.contains("opt in"));
    assert!(block.contains("default"));
    // The bf16->f16 guarantee must be visible to the user.
    assert!(block.contains("never the model weights"));
    // All three ranges appear.
    assert!(block.contains("short"));
    assert!(block.contains("medium"));
    assert!(block.contains("long"));
    // No em dashes in user-facing output.
    assert!(!block.contains('\u{2014}'));
}

#[test]
fn render_block_empty_for_no_advice() {
    assert_eq!(render_kv_cache_advice(&[]), "");
}

#[test]
fn headline_includes_also_consider_when_present() {
    let advice = recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Medium);
    let line = advice.headline();
    assert!(line.contains("fp16+turbo4"));
    assert!(line.contains("also benchmark"));
    assert!(line.contains("int8"));
}
