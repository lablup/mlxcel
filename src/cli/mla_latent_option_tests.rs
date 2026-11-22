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

//! `--mla-latent-quant` at the entry points (issue #3179): parsing, the mode
//! a model directory resolves to under each value, and the lines it prints.
//! Each test installs the option on its own thread.

use clap::Parser;
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::mla::MlaLatentQuant::{self, Auto, Fp16, Int4, Int8};
use mlxcel_core::mla::with_mla_latent_quant;

use super::*;
use crate::cli::turbo_args::{
    TurboKvCacheArgs, model_stores_quantized_latent, resolve_effective_kv_cache_mode,
};

#[derive(Parser, Debug)]
struct Harness {
    #[command(flatten)]
    turbo: TurboKvCacheArgs,
}

fn model_dir(config: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("config.json"), config).expect("write config.json");
    dir
}

fn dir_of(model_type: &str) -> tempfile::TempDir {
    model_dir(&format!("{{\"model_type\": \"{model_type}\"}}"))
}

#[test]
fn the_flag_parses_the_four_values_and_rejects_the_rest_listing_them() {
    for (text, want) in [
        ("auto", Auto),
        ("fp16", Fp16),
        ("int8", Int8),
        ("int4", Int4),
    ] {
        let parsed = Harness::try_parse_from(["x", "--mla-latent-quant", text]).expect(text);
        assert_eq!(parsed.turbo.mla_latent_quant, Some(want));
    }
    assert_eq!(
        Harness::try_parse_from(["x"])
            .unwrap()
            .turbo
            .mla_latent_quant,
        None
    );
    let err = Harness::try_parse_from(["x", "--mla-latent-quant", "q4"])
        .unwrap_err()
        .to_string();
    assert!(err.contains("auto, fp16, int8, int4"), "{err}");
}

/// Acceptance: unset, a listed family, an unlisted family and `deepseek_v2`
/// resolve fp16 and int8 exactly as before the option existed.
#[test]
fn auto_leaves_resolution_as_it_was() {
    with_mla_latent_quant(Auto, || {
        let resolve =
            |family: &str, mode| resolve_effective_kv_cache_mode(mode, dir_of(family).path());
        for family in [
            "deepseek_v3",
            "glm4_moe_lite",
            "deepseek_v32",
            "hy_v4",
            "deepseek_v2",
            "llama",
        ] {
            assert_eq!(
                resolve(family, KVCacheMode::Fp16),
                (KVCacheMode::Fp16, None)
            );
            assert!(mla_latent_option_notices(KVCacheMode::Int8, dir_of(family).path()).is_empty());
            assert!(!option_decides_model(dir_of(family).path()));
        }
        assert_eq!(
            resolve("deepseek_v3", KVCacheMode::Int8),
            (KVCacheMode::Int8, None)
        );
        assert_eq!(
            resolve("deepseek_v2", KVCacheMode::Int8),
            (KVCacheMode::Int8, None)
        );
        for family in ["glm4_moe_lite", "deepseek_v32", "hy_v4"] {
            let (mode, warning) = resolve(family, KVCacheMode::Int8);
            assert_eq!(mode, KVCacheMode::Fp16, "{family}");
            assert!(
                warning
                    .expect("the substitution is announced")
                    .contains("fp16")
            );
        }
        let v3 = dir_of("deepseek_v3");
        assert_eq!(model_latent_bits(KVCacheMode::Int8, v3.path()), Some(8));
        assert_eq!(
            latent_codec_label(8),
            "affine group-64 MLA latent, fp16 rope keys"
        );
    });
}

#[test]
fn an_explicit_value_decides_the_mode_of_every_mla_model_directory() {
    for family in [
        "deepseek_v3",
        "glm4_moe_lite",
        "deepseek_v32",
        "hy_v4",
        "kimi_linear",
        "longcat_flash",
    ] {
        let dir = dir_of(family);
        for requested in [
            KVCacheMode::Fp16,
            KVCacheMode::Int8,
            KVCacheMode::Turbo4Asym,
        ] {
            let resolve = |option: MlaLatentQuant| {
                with_mla_latent_quant(option, || {
                    let (mode, warning) = resolve_effective_kv_cache_mode(requested, dir.path());
                    assert!(warning.is_none(), "{family}/{option}: {warning:?}");
                    (mode, model_latent_bits(mode, dir.path()))
                })
            };
            assert_eq!(resolve(Fp16), (KVCacheMode::Fp16, None), "{family}");
            assert_eq!(resolve(Int8), (KVCacheMode::Int8, Some(8)), "{family}");
            assert_eq!(resolve(Int4), (KVCacheMode::Int8, Some(4)), "{family}");
        }
        with_mla_latent_quant(Int4, || {
            assert!(option_decides_model(dir.path()), "{family}");
            assert!(model_stores_quantized_latent(KVCacheMode::Int8, dir.path()));
            assert!(
                latent_codec_label(4).contains("4-bit") && latent_codec_label(4).contains("int4")
            );
        });
    }
    // A model with no latent cache keeps its own mode under every value:
    // `fp16` leaves its int8 layers int8.
    let llama = dir_of("llama");
    for option in [Fp16, Int8, Int4] {
        with_mla_latent_quant(option, || {
            assert_eq!(
                resolve_effective_kv_cache_mode(KVCacheMode::Int8, llama.path()),
                (KVCacheMode::Int8, None)
            );
            assert!(!option_decides_model(llama.path()));
            assert_eq!(model_latent_bits(KVCacheMode::Int8, llama.path()), None);
            let lines = mla_latent_option_notices(KVCacheMode::Int8, llama.path());
            assert_eq!(lines.len(), 1, "{lines:?}");
            assert!(lines[0].contains("is ignored"), "{lines:?}");
        });
    }
}

/// A forced latent on a rank the codec cannot group keeps fp16 from startup,
/// with the existing whole-groups reason; under `auto` that check stays the
/// DSA families' alone.
#[test]
fn a_forced_latent_on_an_ungroupable_rank_falls_back_to_fp16() {
    let odd = model_dir(r#"{"model_type": "deepseek_v3", "kv_lora_rank": 500}"#);
    let dsa =
        model_dir(r#"{"model_type": "deepseek_v32", "kv_lora_rank": 512, "index_head_dim": 100}"#);
    for dir in [&odd, &dsa] {
        for option in [Int8, Int4] {
            with_mla_latent_quant(option, || {
                let (mode, warning) =
                    resolve_effective_kv_cache_mode(KVCacheMode::Fp16, dir.path());
                assert_eq!(mode, KVCacheMode::Fp16);
                let warning = warning.expect("the fallback is announced");
                assert!(warning.contains("multiples of that"), "{warning}");
            });
        }
    }
    with_mla_latent_quant(Auto, || {
        let (mode, warning) = resolve_effective_kv_cache_mode(KVCacheMode::Int8, odd.path());
        assert_eq!((mode, warning), (KVCacheMode::Int8, None));
    });
}

/// Acceptance: each warning is emitted once. The lines for one model are
/// distinct, and announcing them again prints nothing new.
#[test]
fn each_line_is_announced_once_per_process() {
    let glm = dir_of("glm4_moe_lite");
    with_mla_latent_quant(Int4, || {
        let lines = mla_latent_option_notices(KVCacheMode::Int8, glm.path());
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines[0].contains("disagree") && lines[1].contains("experimental"));
        // A line unique to this test, so no other test announced it first.
        let line = format!("{} [once-test]", lines[1]);
        assert!(first_announcement(&line));
        assert!(!first_announcement(&line));
        // The public path is idempotent too (it prints to stderr at most once).
        announce_mla_latent_option(KVCacheMode::Int8, glm.path());
        announce_mla_latent_option(KVCacheMode::Int8, glm.path());
        for line in lines {
            assert!(!first_announcement(&line), "already announced: {line}");
        }
    });
}

/// `mlxcel inspect` and the preflights name the width the latent is stored at,
/// not the `int8` storage that holds it (review item for #3189).
#[test]
fn the_kv_dtype_label_names_a_narrower_latent_and_the_flag_that_chose_it() {
    let glm = dir_of("glm4_moe_lite");
    let v3 = dir_of("deepseek_v3");
    let llama = dir_of("llama");
    with_mla_latent_quant(Int4, || {
        for dir in [&glm, &v3] {
            let label = kv_dtype_label(KVCacheMode::Int8, dir.path());
            assert_eq!(label, "int4 (MLA latent, --mla-latent-quant int4)");
        }
        // A model the option does not decide keeps its own int8 caches.
        assert_eq!(kv_dtype_label(KVCacheMode::Int8, llama.path()), "int8");
        assert_eq!(kv_dtype_label(KVCacheMode::Fp16, v3.path()), "fp16");
    });
    for option in [Auto, Int8] {
        with_mla_latent_quant(option, || {
            assert_eq!(kv_dtype_label(KVCacheMode::Int8, v3.path()), "int8");
        });
    }
    with_mla_latent_quant(Int4, || {
        let estimate = crate::execution::memory_estimate::estimate_total_memory(
            v3.path(),
            1024,
            1,
            crate::execution::memory_estimate::QuantHint::Fp16,
            KVCacheMode::Int8,
        );
        let banner = crate::execution::memory_estimate::format_estimate(v3.path(), &estimate);
        assert!(
            banner.contains("KV dtype:        int4 (MLA latent"),
            "{banner}"
        );
    });
}
