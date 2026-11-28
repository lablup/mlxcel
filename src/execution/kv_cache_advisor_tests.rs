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

//! Unit tests for [`crate::execution::kv_cache_advisor`].

use super::*;

/// The KV-cache modes the advisor may emit. `Turbo3Asym` is intentionally
/// excluded: its measured decode throughput (about 0.01-0.07x of fp16) makes it
/// a documented memory-extremis last resort, never an advisory pick.
const ALLOWED_MODES: [KVCacheMode; 5] = [
    KVCacheMode::Fp16,
    KVCacheMode::Int8,
    KVCacheMode::Turbo4Delegated,
    KVCacheMode::Turbo4Asym,
    KVCacheMode::Turbo4,
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
fn standard_medium_suggests_int8_then_delegated() {
    let advice = recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Medium);
    assert_eq!(advice.suggested, KVCacheMode::Int8);
    assert_eq!(advice.also_consider, Some(KVCacheMode::Turbo4Delegated));
}

#[test]
fn standard_long_non_allowlisted_never_symmetric_turbo4() {
    // Llama is a dense Q4_K_M-style family and is NOT on the allowlist.
    let advice = recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Long);
    assert_eq!(advice.suggested, KVCacheMode::Turbo4Delegated);
    assert_ne!(advice.suggested, KVCacheMode::Turbo4);
    assert_eq!(advice.also_consider, Some(KVCacheMode::Turbo4Asym));
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
        assert_eq!(advice.also_consider, Some(KVCacheMode::Turbo4Delegated));
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
    // `deepseek_v2` without absorption keeps the decompressed layout, so int8
    // there lands on ordinary per-head K/V rows and is a real recommendation.
    // The absorbed (latent) case is
    // `absorbed_deepseek_v2_is_advised_int8_for_its_quantized_latent`, and a
    // latent-pair family is covered by
    // `latent_mla_families_are_advised_fp16_because_the_binary_refuses_the_rest`.
    for kind in [KvArchKind::MlaDecompressed] {
        // Short stays fp16.
        assert_eq!(
            recommend_kv_cache_mode(kind, "deepseek_v2", KvContextRange::Short).suggested,
            KVCacheMode::Fp16
        );
        for range in [KvContextRange::Medium, KvContextRange::Long] {
            let advice = recommend_kv_cache_mode(kind, "deepseek_v2", range);
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

/// DeepSeek V2 sized as the latent (absorbed decode on) is advised int8 at
/// medium and long context: the mode stores the latent affine-quantized and
/// keeps absorbed decode (issue #3155). Short context keeps fp16, and no Turbo
/// mode is ever named, since those are substituted with fp16 there.
#[test]
fn absorbed_deepseek_v2_is_advised_int8_for_its_quantized_latent() {
    let short = recommend_kv_cache_mode(
        KvArchKind::MlaCompressed,
        "deepseek_v2",
        KvContextRange::Short,
    );
    assert_eq!(short.suggested, KVCacheMode::Fp16);
    assert_eq!(short.also_consider, None);
    for range in [KvContextRange::Medium, KvContextRange::Long] {
        let advice = recommend_kv_cache_mode(KvArchKind::MlaCompressed, "deepseek_v2", range);
        assert_eq!(advice.suggested, KVCacheMode::Int8, "{range:?}");
        assert_eq!(advice.also_consider, None, "{range:?}");
        assert!(advice.rationale.contains("672"), "{}", advice.rationale);
        // The advice must name a mode the binary keeps for this model.
        let (resolved, _) = mlxcel_core::cache::turbo::resolve_kv_cache_mode_for_model_with(
            advice.suggested,
            "deepseek_v2",
            true,
        );
        assert_eq!(resolved, advice.suggested, "{range:?}");
    }
}

/// `mlxcel recommend` must not advise a mode the binary refuses.
///
/// Every family in `MLA_LATENT_CACHE_FAMILIES` has any non-fp16 request
/// substituted back to fp16 before a cache is built (issue #1350), and
/// `kv_arch` classifies anything carrying `kv_lora_rank` as MLA, so without
/// this branch the advisor recommended int8 at medium and long context for
/// exactly the families that reject it.
#[test]
fn latent_mla_families_are_advised_fp16_because_the_binary_refuses_the_rest() {
    for &family in mlxcel_core::mla::MLA_LATENT_CACHE_FAMILIES {
        if mlxcel_core::mla::stores_quantized_latent_pair(family) {
            continue;
        }
        for kind in [KvArchKind::MlaCompressed, KvArchKind::MlaDecompressed] {
            for range in KvContextRange::all() {
                let advice = recommend_kv_cache_mode(kind, family, range);
                assert_eq!(
                    advice.suggested,
                    KVCacheMode::Fp16,
                    "{family}/{kind:?}/{range:?} advised a mode the binary refuses"
                );
                assert_eq!(advice.also_consider, None, "{family}/{kind:?}/{range:?}");
            }
        }
    }
}

/// The `DeepSeekV3Attention` families keep int8 as the affine latent (#3160),
/// so they are advised it at medium and long context, and the advice is a mode
/// the resolver keeps for them.
#[test]
fn quantized_latent_pair_families_are_advised_int8_at_medium_and_long() {
    for &family in mlxcel_core::mla::QUANTIZED_LATENT_FAMILIES {
        for kind in [KvArchKind::MlaCompressed, KvArchKind::MlaDecompressed] {
            let short = recommend_kv_cache_mode(kind, family, KvContextRange::Short);
            assert_eq!(short.suggested, KVCacheMode::Fp16, "{family}/{kind:?}");
            for range in [KvContextRange::Medium, KvContextRange::Long] {
                let advice = recommend_kv_cache_mode(kind, family, range);
                assert_eq!(advice.suggested, KVCacheMode::Int8, "{family}/{range:?}");
                assert_eq!(advice.also_consider, None, "{family}/{range:?}");
                assert!(advice.rationale.contains("672"), "{}", advice.rationale);
                for absorbed in [false, true] {
                    let (resolved, _) =
                        mlxcel_core::cache::turbo::resolve_kv_cache_mode_for_model_with(
                            advice.suggested,
                            family,
                            absorbed,
                        );
                    assert_eq!(resolved, advice.suggested, "{family}/{range:?}");
                }
            }
        }
    }
}

/// Every recommendation is one of the allowed KV-cache modes, and never
/// `Turbo3Asym` (a documented memory-extremis last resort, not an advisory
/// pick). Because the output is always a `KVCacheMode` (a KV-cache-only storage
/// setting), it cannot imply a bf16 to f16 promotion of quantized model weights
/// (the #289 landmine).
#[test]
fn recommendation_is_always_an_allowed_kv_cache_mode() {
    let model_types = [
        "llama",
        "qwen3_5",
        "deepseek_v2",
        "deepseek_v3",
        "glm_moe_dsa",
        "mamba",
        "gemma3",
        "",
    ];
    for kind in ALL_ARCH_KINDS {
        for range in KvContextRange::all() {
            for mt in model_types {
                let advice = recommend_kv_cache_mode(kind, mt, range);
                assert!(
                    ALLOWED_MODES.contains(&advice.suggested),
                    "{kind:?}/{mt}/{range:?} suggested an out-of-scope mode: {:?}",
                    advice.suggested
                );
                assert_ne!(advice.suggested, KVCacheMode::Turbo3Asym);
                if let Some(also) = advice.also_consider {
                    assert!(
                        ALLOWED_MODES.contains(&also),
                        "{kind:?}/{mt}/{range:?} also_consider out-of-scope: {also:?}"
                    );
                    assert_ne!(also, KVCacheMode::Turbo3Asym);
                }
            }
        }
    }
}

#[test]
fn advise_returns_empty_for_missing_config() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(advise_kv_cache_modes(tmp.path(), KVCacheMode::Fp16).is_empty());
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
    let (kind, mt) = arch_and_type_from_config(&cfg, KVCacheMode::Fp16).unwrap();
    assert_eq!(kind, KvArchKind::Standard);
    assert_eq!(mt, "llama");

    let advices: Vec<_> = KvContextRange::all()
        .into_iter()
        .map(|r| recommend_kv_cache_mode(kind, &mt, r))
        .collect();
    assert_eq!(advices.len(), 3);
    assert_eq!(advices[0].context_range, KvContextRange::Short);
    assert_eq!(advices[0].suggested, KVCacheMode::Fp16);
    assert_eq!(advices[1].suggested, KVCacheMode::Int8);
    assert_eq!(advices[2].suggested, KVCacheMode::Turbo4Delegated);
}

/// Both OpenAI-era families used to classify as unavailable, so
/// `advise_kv_cache_modes_from_config` returned an empty vector and `mlxcel
/// inspect` printed no KV-cache-mode section for them at all (#927).
#[test]
fn openai_era_naming_produces_advice() {
    let gpt2 = serde_json::json!({
        "model_type": "gpt2",
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
    });
    let gpt_bigcode = serde_json::json!({
        "model_type": "gpt_bigcode",
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 2048,
        "multi_query": true,
    });

    for (name, cfg) in [("gpt2", gpt2), ("gpt_bigcode", gpt_bigcode)] {
        let (kind, mt) = arch_and_type_from_config(&cfg, KVCacheMode::Fp16)
            .unwrap_or_else(|| panic!("{name} should classify"));
        assert_eq!(kind, KvArchKind::Standard, "{name} kind");
        assert_eq!(mt, name, "{name} model_type");

        let advices = advise_kv_cache_modes_from_config(&cfg, KVCacheMode::Fp16);
        assert_eq!(advices.len(), 3, "{name} advice count");
        // head_dim is 64 (gpt2) / 128 (gpt_bigcode), both powers of two, so
        // the Turbo guard does not downgrade the long-context suggestion.
        assert_eq!(advices[2].suggested, KVCacheMode::Turbo4Delegated, "{name}");
    }
}

/// `read_head_dim` derives from `n_embd` / `n_head` now that it shares the
/// classifier's alias lists. Both families are powers of two, so nothing is
/// downgraded, but the guard must be deriving a real value rather than
/// defaulting permissively through `None`.
#[test]
fn openai_era_naming_derives_head_dim_for_turbo_guard() {
    let gpt2 = serde_json::json!({ "n_embd": 768, "n_head": 12 });
    assert_eq!(read_head_dim(&gpt2), Some(64));

    let gpt_bigcode = serde_json::json!({ "n_embd": 2048, "n_head": 16 });
    assert_eq!(read_head_dim(&gpt_bigcode), Some(128));
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
    assert!(line.contains("int8"));
    assert!(line.contains("also benchmark"));
    assert!(line.contains("turbo4-delegated"));
}

#[test]
fn non_power_of_two_head_dim_withholds_turbo() {
    // Phi-2-style: hidden_size=2560, num_attention_heads=32 -> head_dim=80 (not power of two).
    let cfg = serde_json::json!({
        "model_type": "phi",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "hidden_size": 2560
    });
    // Verify head_dim derivation.
    assert_eq!(read_head_dim(&cfg), Some(80));
    assert!(!80u64.is_power_of_two());

    let advices = advise_kv_cache_modes_from_config(&cfg, KVCacheMode::Fp16);
    assert_eq!(advices.len(), 3);

    // None should suggest or offer a Walsh-Hadamard Turbo mode.
    for advice in &advices {
        assert!(
            !is_walsh_hadamard_turbo(advice.suggested),
            "Expected no Turbo suggestion for non-pow2 head dim, got {:?}",
            advice.suggested
        );
        if let Some(ac) = advice.also_consider {
            assert!(
                !is_walsh_hadamard_turbo(ac),
                "Expected no Turbo in also_consider for non-pow2 head dim, got {ac:?}"
            );
        }
    }
    // short -> fp16, medium -> int8, long -> int8.
    assert_eq!(advices[0].suggested, KVCacheMode::Fp16);
    assert_eq!(advices[1].suggested, KVCacheMode::Int8);
    assert_eq!(advices[2].suggested, KVCacheMode::Int8);
}

/// Alternate config naming (OLMo / MPT-style `d_model` + `n_heads`) must not
/// bypass the non-power-of-two guard. `read_head_dim` mirrors the classifier's
/// field-name coverage, so a non-power-of-two head dim still withholds Turbo.
#[test]
fn alternate_naming_non_power_of_two_head_dim_withholds_turbo() {
    // MPT-30B-style: d_model=7168, n_heads=64 -> head_dim=112 (not power of two).
    let cfg = serde_json::json!({
        "model_type": "mpt",
        "n_layers": 48,
        "n_heads": 64,
        "d_model": 7168
    });
    // Without the field-name alignment this returned None, defaulting
    // turbo_ok=true and slipping head_dim 112 past the guard.
    assert_eq!(read_head_dim(&cfg), Some(112));
    assert!(!112u64.is_power_of_two());

    let advices = advise_kv_cache_modes_from_config(&cfg, KVCacheMode::Fp16);
    assert!(!advices.is_empty());
    for advice in &advices {
        assert!(
            !is_walsh_hadamard_turbo(advice.suggested),
            "alternate-naming non-pow2 head dim must withhold Turbo, got {:?}",
            advice.suggested
        );
        if let Some(ac) = advice.also_consider {
            assert!(
                !is_walsh_hadamard_turbo(ac),
                "alternate-naming non-pow2 head dim must withhold Turbo in also_consider, got {ac:?}"
            );
        }
    }
}

/// Alternate naming with a power-of-two head dim (OLMo-1B: d_model=2048,
/// n_heads=16 -> 128) still derives the head dim and keeps Turbo eligible.
#[test]
fn alternate_naming_power_of_two_head_dim_keeps_turbo() {
    let cfg = serde_json::json!({
        "model_type": "olmo",
        "n_layers": 16,
        "n_heads": 16,
        "d_model": 2048
    });
    assert_eq!(read_head_dim(&cfg), Some(128));
    assert!(128u64.is_power_of_two());

    let advices = advise_kv_cache_modes_from_config(&cfg, KVCacheMode::Fp16);
    assert_eq!(advices.len(), 3);
    // olmo is not on the symmetric allowlist; pow2 head dim keeps the
    // Walsh-Hadamard turbo4-delegated at long context.
    assert_eq!(advices[2].suggested, KVCacheMode::Turbo4Delegated);
}

#[test]
fn power_of_two_head_dim_keeps_turbo() {
    // Llama-style: hidden_size=4096, num_attention_heads=32 -> head_dim=128 (power of two).
    let cfg = serde_json::json!({
        "model_type": "llama",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096
    });
    assert_eq!(read_head_dim(&cfg), Some(128));
    assert!(128u64.is_power_of_two());

    let advices = advise_kv_cache_modes_from_config(&cfg, KVCacheMode::Fp16);
    assert!(advices.len() >= 2);
    // Pow2 head dim keeps Turbo eligible: long (non-allowlisted) leads with the
    // Walsh-Hadamard turbo4-delegated rather than being downgraded to int8.
    assert_eq!(advices[2].suggested, KVCacheMode::Turbo4Delegated);
    assert_eq!(advices[2].also_consider, Some(KVCacheMode::Turbo4Asym));
}

/// DeepSeek V2-Lite reduced to the fields the classifier reads.
fn deepseek_v2_lite_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "deepseek_v2",
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
    })
}

/// Without absorbed decode a run in a Turbo mode builds decompressed per-head
/// K/V for DeepSeek V2, so the advice is classified as `MlaDecompressed` and
/// the medium/long buckets advise per-token int8 (#3153). With absorbed decode
/// on, the Turbo request resolves back to the fp16 latent at startup, so the
/// advice is the latent's: fp16 at short context and the quantized latent
/// (int8, #3155) beyond it, never a mode that would grow the cache.
#[test]
fn deepseek_v2_under_a_turbo_mode_is_advised_by_its_resolved_layout() {
    let cfg = deepseek_v2_lite_config();
    let (kind, mt) = arch_and_type_from_config(&cfg, KVCacheMode::Turbo4Asym).unwrap();
    assert_eq!(mt, "deepseek_v2");
    let advices = advise_kv_cache_modes_from_config(&cfg, KVCacheMode::Turbo4Asym);
    assert_eq!(advices.len(), 3);
    if mlxcel_core::mla::absorbed_enabled() {
        assert_eq!(kind, KvArchKind::MlaCompressed);
        for advice in &advices {
            assert_eq!(advice.arch_kind, KvArchKind::MlaCompressed);
        }
        assert_eq!(advices[0].suggested, KVCacheMode::Fp16);
        assert_eq!(advices[1].suggested, KVCacheMode::Int8);
        assert_eq!(advices[2].suggested, KVCacheMode::Int8);
    } else {
        assert_eq!(kind, KvArchKind::MlaDecompressed);
        for advice in &advices {
            assert_eq!(advice.arch_kind, KvArchKind::MlaDecompressed);
        }
        assert_eq!(advices[0].suggested, KVCacheMode::Fp16);
        assert_eq!(advices[1].suggested, KVCacheMode::Int8);
        assert_eq!(advices[2].suggested, KVCacheMode::Int8);
    }

    // FP16 follows the process absorption decision.
    let (fp16_kind, _) = arch_and_type_from_config(&cfg, KVCacheMode::Fp16).unwrap();
    let expected = if mlxcel_core::mla::absorbed_enabled() {
        KvArchKind::MlaCompressed
    } else {
        KvArchKind::MlaDecompressed
    };
    assert_eq!(fp16_kind, expected);
}

/// DeepSeek-VL2 nests the same decoder under `language_config` with no
/// `model_type`, so it is advised exactly as `deepseek_v2` for every mode.
#[test]
fn deepseek_vl2_is_advised_as_its_deepseek_v2_decoder() {
    let v2 = deepseek_v2_lite_config();
    let mut language = v2.clone();
    language
        .as_object_mut()
        .expect("object")
        .remove("model_type");
    let vl2 = serde_json::json!({ "model_type": "deepseek_vl_v2", "language_config": language });
    for mode in [
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
    ] {
        let (vl2_kind, vl2_mt) = arch_and_type_from_config(&vl2, mode).unwrap();
        let (v2_kind, _) = arch_and_type_from_config(&v2, mode).unwrap();
        assert_eq!(vl2_mt, "deepseek_v2", "{mode}");
        assert_eq!(vl2_kind, v2_kind, "{mode}");
        let vl2_advice: Vec<_> = advise_kv_cache_modes_from_config(&vl2, mode)
            .into_iter()
            .map(|a| a.suggested)
            .collect();
        let v2_advice: Vec<_> = advise_kv_cache_modes_from_config(&v2, mode)
            .into_iter()
            .map(|a| a.suggested)
            .collect();
        assert_eq!(vl2_advice, v2_advice, "{mode}");
    }
}
