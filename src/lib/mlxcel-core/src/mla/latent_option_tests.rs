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

//! The MLA latent option's resolution rules (issue #3179). No test here
//! touches the process-wide value: each passes the option explicitly or
//! installs it on its own thread with `with_mla_latent_quant`.

use super::*;
use crate::cache::turbo::{
    resolve_kv_cache_mode_for_model_with, resolve_kv_cache_mode_for_model_with_option,
};
use crate::mla::{
    MLA_LATENT_CACHE_FAMILIES, latent_pair_supports_mode_with, stores_quantized_latent_pair_with,
};

use MlaLatentQuant::{Auto, Fp16, Int4, Int8};

const MODES: [KVCacheMode; 6] = [
    KVCacheMode::Fp16,
    KVCacheMode::Int8,
    KVCacheMode::Turbo4Asym,
    KVCacheMode::Turbo4,
    KVCacheMode::Turbo3Asym,
    KVCacheMode::Turbo4Delegated,
];

/// A listed family, an unlisted one, a DSA one, `deepseek_v2` and a model
/// with no latent cache.
const SAMPLE: [&str; 5] = [
    "deepseek_v3",
    "glm4_moe_lite",
    "deepseek_v32",
    "deepseek_v2",
    "llama",
];

#[test]
fn the_four_values_parse_and_anything_else_lists_them() {
    for (text, want) in [
        ("auto", Auto),
        ("fp16", Fp16),
        ("int8", Int8),
        ("int4", Int4),
    ] {
        assert_eq!(MlaLatentQuant::parse(text), Ok(want));
        assert_eq!(want.as_str(), text);
        assert_eq!(text.parse::<MlaLatentQuant>(), Ok(want));
    }
    assert_eq!(MlaLatentQuant::parse(" INT4 "), Ok(Int4));
    let err = MlaLatentQuant::parse("int2").unwrap_err();
    assert!(err.contains("auto, fp16, int8, int4"), "{err}");
    assert_eq!(MlaLatentQuant::default(), Auto);
    assert!(!Auto.is_explicit() && Fp16.is_explicit() && Int4.is_explicit());
}

#[test]
fn the_cli_value_wins_and_a_bad_one_names_the_flag() {
    assert_eq!(resolve_mla_latent_quant(Some("int4")), Ok(Int4));
    let err = resolve_mla_latent_quant(Some("q4")).unwrap_err();
    assert!(err.starts_with(MLA_LATENT_QUANT_FLAG), "{err}");
}

/// Acceptance: with the option at `auto`, resolution is what it was before
/// the option existed, for a listed family, an unlisted family and
/// `deepseek_v2`.
#[test]
fn auto_resolves_exactly_as_the_default_gate() {
    use KVCacheMode::{Fp16 as F, Int8 as I};
    // Listed: int8 keeps the affine 8-bit latent, fp16 stays fp16.
    assert_eq!(effective_latent_bits("deepseek_v3", I, Auto), Some(8));
    assert_eq!(effective_latent_bits("deepseek_v3", F, Auto), None);
    assert_eq!(effective_latent_bits(" Kimi_Linear ", I, Auto), Some(8));
    // Unlisted latent-pair families keep fp16 under either request.
    for family in [
        "glm4_moe_lite",
        "deepseek_v32",
        "deepseek_v3.2",
        "glm_moe_dsa",
    ] {
        assert_eq!(effective_latent_bits(family, I, Auto), None, "{family}");
        assert_eq!(effective_latent_bits(family, F, Auto), None, "{family}");
    }
    // deepseek_v2 follows the request (its absorbed state is the caller's).
    assert_eq!(effective_latent_bits("deepseek_v2", I, Auto), Some(8));
    assert_eq!(effective_latent_bits("deepseek_v2", F, Auto), None);
    // A Turbo request selects no latent width anywhere.
    for family in SAMPLE {
        assert_eq!(
            effective_latent_bits(family, KVCacheMode::Turbo4Asym, Auto),
            None
        );
    }

    for &family in MLA_LATENT_CACHE_FAMILIES
        .iter()
        .chain(&["deepseek_v2", "llama", ""])
    {
        let listed = crate::mla::QUANTIZED_LATENT_FAMILIES.contains(&family);
        assert_eq!(stores_quantized_latent_pair_with(family, Auto), listed);
        for mode in MODES {
            for absorbed in [false, true] {
                assert_eq!(
                    resolve_kv_cache_mode_for_model_with_option(mode, family, absorbed, Auto),
                    resolve_kv_cache_mode_for_model_with(mode, family, absorbed),
                    "{family}/{mode}/{absorbed}"
                );
            }
        }
    }
    assert_eq!(latent_cache_bits_with(KVCacheMode::Int8, Auto), Some(8));
    assert_eq!(latent_cache_bits_with(KVCacheMode::Fp16, Auto), None);
}

#[test]
fn an_explicit_value_decides_every_mla_family_and_no_other_model() {
    for &family in MLA_LATENT_CACHE_FAMILIES.iter().chain(&["deepseek_v2"]) {
        for mode in MODES {
            assert_eq!(effective_latent_bits(family, mode, Fp16), None);
            assert_eq!(effective_latent_bits(family, mode, Int8), Some(8));
            assert_eq!(effective_latent_bits(family, mode, Int4), Some(4));
        }
    }
    for mode in MODES {
        for option in [Fp16, Int8, Int4] {
            assert_eq!(effective_latent_bits("llama", mode, option), None);
        }
    }
    for &family in MLA_LATENT_CACHE_FAMILIES {
        assert!(!stores_quantized_latent_pair_with(family, Fp16), "{family}");
        assert!(stores_quantized_latent_pair_with(family, Int8), "{family}");
        assert!(stores_quantized_latent_pair_with(family, Int4), "{family}");
        assert!(latent_pair_supports_mode_with(family, KVCacheMode::Int8, Int4).is_ok());
        let err = latent_pair_supports_mode_with(family, KVCacheMode::Int8, Fp16).unwrap_err();
        assert!(err.contains(MLA_LATENT_QUANT_FLAG), "{err}");
        // A Turbo mode is refused by the layout whatever the option says.
        assert!(latent_pair_supports_mode_with(family, KVCacheMode::Turbo4, Int8).is_err());
    }
    // deepseek_v2 is not a latent-pair family; its latent is the view's.
    assert!(!stores_quantized_latent_pair_with("deepseek_v2", Int8));
    assert_eq!(latent_cache_bits_with(KVCacheMode::Int8, Int4), Some(4));
    assert_eq!(latent_cache_bits_with(KVCacheMode::Int8, Int8), Some(8));
    assert_eq!(latent_cache_bits_with(KVCacheMode::Int8, Fp16), Some(8));
    assert_eq!(latent_cache_bits_with(KVCacheMode::Fp16, Int4), None);
}

#[test]
fn the_resolver_builds_int8_storage_for_a_forced_latent_and_fp16_otherwise() {
    use KVCacheMode::{Fp16 as F, Int8 as I};
    for &family in MLA_LATENT_CACHE_FAMILIES {
        for mode in MODES {
            for absorbed in [false, true] {
                let resolve = |option| {
                    resolve_kv_cache_mode_for_model_with_option(mode, family, absorbed, option)
                };
                assert_eq!(resolve(Fp16), (F, None), "{family}/{mode}");
                assert_eq!(resolve(Int8), (I, None), "{family}/{mode}");
                assert_eq!(resolve(Int4), (I, None), "{family}/{mode}");
            }
        }
    }
    // deepseek_v2 holds a latent only while it absorbs.
    for mode in MODES {
        let v2 = |absorbed, option| {
            resolve_kv_cache_mode_for_model_with_option(mode, "deepseek_v2", absorbed, option)
        };
        assert_eq!(v2(true, Fp16), (F, None));
        assert_eq!(v2(true, Int4), (I, None));
        for option in [Fp16, Int8, Int4] {
            assert_eq!(
                v2(false, option),
                resolve_kv_cache_mode_for_model_with(mode, "deepseek_v2", false)
            );
            // A model with no latent cache ignores the option: `fp16` leaves
            // its int8 layers int8.
            assert_eq!(
                resolve_kv_cache_mode_for_model_with_option(mode, "llama", true, option),
                resolve_kv_cache_mode_for_model_with(mode, "llama", true)
            );
        }
    }
}

#[test]
fn auto_prints_nothing_and_a_model_without_a_latent_gets_one_ignored_line() {
    for family in SAMPLE {
        for mode in MODES {
            assert!(latent_option_notices(family, mode, true, Auto).is_empty());
        }
    }
    for option in [Fp16, Int8, Int4] {
        let lines = latent_option_notices("llama", KVCacheMode::Int8, true, option);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].contains("is ignored") && lines[0].contains("no MLA latent cache"));
        let lines = latent_option_notices("deepseek_v2", KVCacheMode::Fp16, false, option);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].contains("absorbed MLA decode is off"), "{lines:?}");
    }
}

#[test]
fn the_experimental_line_quotes_the_measurement_for_the_family_and_width() {
    let only = |family: &str, mode, option| {
        let lines = latent_option_notices(family, mode, true, option);
        assert_eq!(lines.len(), 1, "{family}/{option}: {lines:?}");
        lines.into_iter().next().unwrap()
    };
    let f = KVCacheMode::Fp16;
    // int8 on a listed family and on deepseek_v2 is the shipped precision.
    assert!(latent_option_notices("deepseek_v3", f, true, Int8).is_empty());
    assert!(latent_option_notices("deepseek_v2", f, true, Int8).is_empty());
    assert!(latent_option_notices("deepseek_v3", f, true, Fp16).is_empty());

    let glm = only("glm4_moe_lite", f, Int8);
    assert!(glm.contains("experimental") && glm.contains("1 decided-position disagreement"));
    assert!(!glm.contains("0.86"), "{glm}");

    let dsa = only("deepseek_v32", f, Int8);
    assert!(
        dsa.contains("top-k overlap") && dsa.contains("0.99"),
        "{dsa}"
    );
    assert!(!dsa.contains("stays int8"), "{dsa}");

    for family in ["deepseek_v3", "deepseek_v2", "kimi_linear", "longcat_flash"] {
        let int4 = only(family, f, Int4);
        assert!(
            int4.contains("0.86 logits") && int4.contains("composed path"),
            "{int4}"
        );
    }
    let dsa4 = only("glm_moe_dsa", f, Int4);
    assert!(
        dsa4.contains("0.86 logits") && dsa4.contains("top-k overlap"),
        "{dsa4}"
    );
    // The int8 key slot's note is not repeated as if it described int4; the
    // int4 note says the 4-bit slot is unmeasured on the real checkpoint.
    assert!(dsa4.contains("not been quality-measured"), "{dsa4}");
    assert!(!dsa4.contains("with an int8 key slot passed"), "{dsa4}");
    assert!(!dsa.contains("not been quality-measured"), "{dsa}");
    assert!(
        dsa4.contains("indexer key stored beside the latent stays int8"),
        "{dsa4}"
    );
    let glm4 = only("glm4_moe_lite", f, Int4);
    assert!(glm4.contains("0.86 logits"), "{glm4}");
    // GLM's int8 result describes int8 only.
    assert!(!glm4.contains("1 decided-position"), "{glm4}");
    // The #3179 measurements: GLM's own int4 result, and the int4 traces on
    // Moonlight and LongCat for every family.
    assert!(glm4.contains("3 decided-position disagreements"), "{glm4}");
    assert!(glm4.contains("-19.5% and -16.1%"), "{glm4}");
    assert!(glm4.contains("0.84 and 1.13 logits"), "{glm4}");
    let glm8 = only("glm4_moe_lite", f, Int8);
    assert!(!glm8.contains("3 decided-position"), "{glm8}");
    assert_eq!(experimental_latent_notes("glm4_moe_lite", 4).len(), 3);
    assert_eq!(experimental_latent_notes("glm4_moe_lite", 8).len(), 1);
    assert_eq!(experimental_latent_notes("deepseek_v32", 4).len(), 3);
    assert_eq!(experimental_latent_notes("deepseek_v32", 8).len(), 1);
    assert_eq!(experimental_latent_notes("deepseek_v3", 4).len(), 2);
    assert_eq!(experimental_latent_notes("deepseek_v3", 8).len(), 0);
    assert_eq!(
        experimental_latent_notes("longcat_flash_sparse", 4).len(),
        3
    );
    assert_eq!(
        experimental_latent_notes("longcat_flash_sparse", 8).len(),
        0
    );
}

#[test]
fn a_quantized_mode_that_implies_another_precision_gets_one_line_naming_both() {
    let i = KVCacheMode::Int8;
    let lines = latent_option_notices("deepseek_v3", i, true, Fp16);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].contains("int8 KV cache mode") && lines[0].contains("--kv-bits"));
    assert!(lines[0].contains("disagree") && lines[0].contains("stored as fp16"));

    // int8 mode with an int4 latent: the disagreement and the experimental line.
    let lines = latent_option_notices("deepseek_v3", i, true, Int4);
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(lines[0].contains("stored as int4"), "{lines:?}");
    assert!(lines[1].contains("experimental"), "{lines:?}");

    // Agreeing flags print no disagreement.
    assert!(latent_option_notices("deepseek_v3", i, true, Int8).is_empty());
    // A Turbo mode never names a latent precision, so any explicit value differs.
    let lines = latent_option_notices("deepseek_v3", KVCacheMode::Turbo4Asym, true, Int8);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].contains("disagree"), "{lines:?}");
}

#[test]
fn the_thread_override_is_scoped_and_restored() {
    let before = mla_latent_quant();
    let inside = with_mla_latent_quant(Int4, || {
        assert_eq!(latent_cache_bits(KVCacheMode::Int8), Some(4));
        assert!(crate::mla::stores_quantized_latent_pair("glm4_moe_lite"));
        let nested = with_mla_latent_quant(Fp16, mla_latent_quant);
        assert_eq!(nested, Fp16);
        mla_latent_quant()
    });
    assert_eq!(inside, Int4);
    assert_eq!(mla_latent_quant(), before);
    // Another thread never sees this thread's override.
    let other = with_mla_latent_quant(Int8, || {
        std::thread::spawn(mla_latent_quant).join().unwrap()
    });
    assert_eq!(other, before);
}

/// A measurement is taken at one width, so a note keyed on two would print
/// one width's result as if it described the other (review item for #3189).
#[test]
fn every_note_is_keyed_on_the_one_width_it_was_measured_at() {
    for note in EXPERIMENTAL_LATENT_NOTES {
        assert_eq!(note.bits.len(), 1, "{}", note.text);
        assert!(matches!(note.bits[0], 4 | 8), "{}", note.text);
    }
}

/// One rule, one case rule: an explicit value decides a latent-pair family
/// and an absorbing `deepseek_v2` whatever the `config.json` casing, and
/// nothing else (review item for #3189).
#[test]
fn whether_the_option_decides_a_model_ignores_case_and_follows_absorption() {
    for spelling in ["deepseek_v2", "DeepSeek_V2", " DEEPSEEK_V2 "] {
        assert!(holds_mla_latent(spelling, true), "{spelling:?}");
        assert!(!holds_mla_latent(spelling, false), "{spelling:?}");
        for option in [Fp16, Int8, Int4] {
            assert!(
                option_decides_latent(spelling, true, option),
                "{spelling:?}"
            );
            assert!(
                !option_decides_latent(spelling, false, option),
                "{spelling:?}"
            );
        }
        assert!(!option_decides_latent(spelling, true, Auto), "{spelling:?}");
        // The resolver reads the same rule, so the casing reaches the cache.
        assert_eq!(
            resolve_kv_cache_mode_for_model_with_option(KVCacheMode::Fp16, spelling, true, Int4),
            (KVCacheMode::Int8, None),
            "{spelling:?}"
        );
        assert!(latent_option_notices(spelling, KVCacheMode::Int8, true, Int8).is_empty());
    }
    for spelling in ["glm4_moe_lite", "GLM4_MOE_LITE"] {
        for absorbed in [false, true] {
            assert!(
                option_decides_latent(spelling, absorbed, Int4),
                "{spelling}"
            );
        }
    }
    for option in [Auto, Fp16, Int8, Int4] {
        assert!(!option_decides_latent("llama", true, option));
    }
}

#[test]
fn a_non_utf8_or_unknown_environment_value_is_an_error_naming_the_variable() {
    use std::env::VarError;
    assert_eq!(option_from_env_value(Err(VarError::NotPresent)), Ok(Auto));
    assert_eq!(option_from_env_value(Ok("  ".to_string())), Ok(Auto));
    assert_eq!(option_from_env_value(Ok(" INT4 ".to_string())), Ok(Int4));
    let bad = option_from_env_value(Ok("q4".to_string())).unwrap_err();
    assert!(bad.starts_with(MLA_LATENT_QUANT_ENV) && bad.contains("auto, fp16, int8, int4"));
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let raw = std::ffi::OsString::from_vec(vec![b'i', b'n', b't', 0xff]);
        let err = option_from_env_value(Err(VarError::NotUnicode(raw))).unwrap_err();
        assert!(err.starts_with(MLA_LATENT_QUANT_ENV), "{err}");
        assert!(err.contains("not valid UTF-8"), "{err}");
    }
}
