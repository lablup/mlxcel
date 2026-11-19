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

//! Which latent-pair families may store their MLA latent quantized (issue
//! #3160).
//!
//! Every family in [`crate::mla::MLA_LATENT_CACHE_FAMILIES`] caches the
//! `(kv_latent, k_pe)` pair unconditionally. A family moves onto the affine
//! int8 latent of issue #3155 once its cache write goes through [`crate::mla::MlaLatentCache`] and
//! every read of the returned [`crate::mla::LatentWindow`] handles the
//! quantized variant. [`QUANTIZED_LATENT_FAMILIES`] lists the families that
//! have done so; the rest keep the fp16-only rule.
//!
//! Kept beside [`crate::mla::cache`] rather than inside it so the gate and its
//! tests stay in one small file.

use crate::cache::KVCacheMode;
use crate::mla::cache::caches_mla_latent_pair;
use crate::mla::latent_layout_supports_mode;

/// Latent-pair `model_type` values whose cache stores the latent through
/// [`crate::mla::MlaLatentCache`], so `int8` keeps an affine group-64 latent
/// with fp16 rope keys.
///
/// The eight types that run `DeepSeekV3Attention`: `DeepSeekV3Model`
/// (`deepseek_v3`, and the `kimi_k2` / `joyai_llm_flash` checkpoints detection
/// routes to it), the Kimi-VL and Kimi K2.5 wrappers whose text decoder is
/// that model, and the Youtu decoders whose layers hold that attention. Every
/// entry is also in [`crate::mla::MLA_LATENT_CACHE_FAMILIES`].
///
/// Used by: [`latent_pair_supports_mode`], [`stores_quantized_latent_pair`].
pub static QUANTIZED_LATENT_FAMILIES: &[&str] = &[
    "deepseek_v3",
    "kimi_k2",
    "joyai_llm_flash",
    "kimi_vl",
    "kimi_k25",
    "youtu",
    "youtu_llm",
    "youtu_vl",
];

/// Whether `model_type` is a latent-pair family that stores its latent through
/// the quantizing view. Same exact, trimmed, case-insensitive match as
/// [`caches_mla_latent_pair`].
///
/// Used by: `cache::turbo::allowlist` (the rule 1 warning),
/// `mlxcel::cli::turbo_args::model_stores_quantized_latent`,
/// `mlxcel::execution::kv_arch` (sizing) and
/// `mlxcel::execution::kv_cache_advisor`.
#[must_use]
pub fn stores_quantized_latent_pair(model_type: &str) -> bool {
    let needle = model_type.trim().to_ascii_lowercase();
    QUANTIZED_LATENT_FAMILIES
        .iter()
        .any(|&family| needle == family)
}

/// The modes a latent-pair family's cache may hold.
///
/// A family in [`QUANTIZED_LATENT_FAMILIES`] follows
/// [`latent_layout_supports_mode`] (fp16 or the affine `int8` latent, no
/// Turbo mode). Any other latent-pair family holds fp16 only. A `model_type`
/// outside [`crate::mla::MLA_LATENT_CACHE_FAMILIES`] has no latent-pair
/// constraint and is accepted; the other resolver rules apply to it. Returns
/// the operator-facing reason on refusal.
///
/// Used by: `cache::turbo::allowlist::resolve_kv_cache_mode_for_model_with`
/// (rule 1).
pub fn latent_pair_supports_mode(model_type: &str, mode: KVCacheMode) -> Result<(), String> {
    if mode == KVCacheMode::Fp16 || !caches_mla_latent_pair(model_type) {
        return Ok(());
    }
    latent_layout_supports_mode(mode)?;
    if stores_quantized_latent_pair(model_type) {
        return Ok(());
    }
    Err(format!(
        "mla: this family stores its latent pair as fp16; the quantized latent is available on \
         deepseek_v2 and on {}",
        QUANTIZED_LATENT_FAMILIES.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mla::MLA_LATENT_CACHE_FAMILIES;

    const TURBO: [KVCacheMode; 4] = [
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
        KVCacheMode::Turbo3Asym,
        KVCacheMode::Turbo4Delegated,
    ];

    #[test]
    fn every_quantized_family_is_a_latent_pair_family() {
        for &family in QUANTIZED_LATENT_FAMILIES {
            assert!(MLA_LATENT_CACHE_FAMILIES.contains(&family), "{family}");
            assert!(stores_quantized_latent_pair(family), "{family}");
        }
        assert!(stores_quantized_latent_pair("  Kimi_VL "));
        assert!(!stores_quantized_latent_pair("deepseek_v2"));
        assert!(!stores_quantized_latent_pair("deepseek_v32"));
        assert!(!stores_quantized_latent_pair("glm4_moe_lite"));
        assert!(!stores_quantized_latent_pair("deepseek_v3_vl"));
    }

    #[test]
    fn quantized_families_accept_fp16_and_int8_and_refuse_turbo() {
        for &family in QUANTIZED_LATENT_FAMILIES {
            assert!(latent_pair_supports_mode(family, KVCacheMode::Fp16).is_ok());
            assert!(latent_pair_supports_mode(family, KVCacheMode::Int8).is_ok());
            for mode in TURBO {
                let err = latent_pair_supports_mode(family, mode).unwrap_err();
                assert!(err.contains("fp16 or int8"), "{family}/{mode}: {err}");
            }
        }
    }

    #[test]
    fn other_latent_pair_families_hold_fp16_only() {
        for &family in MLA_LATENT_CACHE_FAMILIES {
            if stores_quantized_latent_pair(family) {
                continue;
            }
            assert!(latent_pair_supports_mode(family, KVCacheMode::Fp16).is_ok());
            let err = latent_pair_supports_mode(family, KVCacheMode::Int8).unwrap_err();
            assert!(err.contains("deepseek_v3"), "{family}: {err}");
            for mode in TURBO {
                assert!(latent_pair_supports_mode(family, mode).is_err(), "{family}");
            }
        }
    }

    #[test]
    fn non_latent_types_have_no_latent_pair_constraint() {
        for family in ["llama", "deepseek_v2", ""] {
            assert!(latent_pair_supports_mode(family, KVCacheMode::Int8).is_ok());
            assert!(latent_pair_supports_mode(family, KVCacheMode::Turbo4Asym).is_ok());
        }
    }
}
