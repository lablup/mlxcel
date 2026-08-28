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

use super::SpecCompatArgs;

fn args() -> SpecCompatArgs {
    SpecCompatArgs::default()
}

#[test]
fn defaults_are_inert() {
    args().ensure_inert().expect("an empty surface is inert");
}

#[test]
fn spec_type_resolves_the_translatable_values_and_rejects_the_rest() {
    // `none` disables speculation (b10621 semantics: the explicit selector
    // stops the draft-sidecar type inference, so a configured draft model
    // runs no speculation).
    let mut a = args();
    a.spec_type = Some("none".to_string());
    let resolution = a.resolved_spec_type().expect("none resolves");
    assert!(resolution.disable_speculation);
    assert_eq!(resolution.draft_kind, None);
    a.ensure_inert().expect("none passes the inertness gate");

    // draft-mtp / draft-dflash are exact --draft-kind translations.
    for (value, kind) in [("draft-mtp", "mtp"), ("draft-dflash", "dflash")] {
        let mut a = args();
        a.spec_type = Some(value.to_string());
        let resolution = a.resolved_spec_type().expect("translatable draft type");
        assert_eq!(resolution.draft_kind, Some(kind));
        assert!(!resolution.disable_speculation);
    }

    // Every n-gram mode, the untranslatable draft types, multi-subsystem
    // lists, none-plus-a-type combinations, and unknown values all fail with
    // the option named.
    for mode in [
        "ngram-simple",
        "ngram-map-k",
        "ngram-map-k4v",
        "ngram-mod",
        "draft-simple",
        "draft-eagle3",
        "draft-dspark",
        "draft-mtp,draft-dflash",
        "none,draft-mtp",
        "wibble",
    ] {
        let mut a = args();
        a.spec_type = Some(mode.to_string());
        let rejection = a
            .ensure_inert()
            .expect_err("untranslatable selectors must fail startup");
        assert_eq!(rejection.option, "--spec-type", "mode {mode}");
    }
}

#[test]
fn no_backend_sampling_half_is_rejected() {
    // The --no- spelling is the operator-active half of the pair: it moves
    // draft sampling to a host-side CPU sampler, which mlxcel cannot do.
    let mut a = args();
    a.no_spec_draft_backend_sampling = true;
    assert_eq!(
        a.ensure_inert().expect_err("CPU draft sampling").option,
        "--no-spec-draft-backend-sampling"
    );
}

#[test]
fn draft_sampler_thresholds_accept_only_their_inert_defaults() {
    for (set, ok, bad) in [
        (0usize, "0", "2"), // --spec-draft-n-min
        (1, "0.0", "0.5"),  // --spec-draft-p-min
        (2, "0.10", "0.4"), // --spec-draft-p-split
    ] {
        let mut a = args();
        match set {
            0 => a.spec_draft_n_min = Some(ok.to_string()),
            1 => a.spec_draft_p_min = Some(ok.to_string()),
            _ => a.spec_draft_p_split = Some(ok.to_string()),
        }
        a.ensure_inert().expect("the upstream default is inert");
        let mut a = args();
        match set {
            0 => a.spec_draft_n_min = Some(bad.to_string()),
            1 => a.spec_draft_p_min = Some(bad.to_string()),
            _ => a.spec_draft_p_split = Some(bad.to_string()),
        }
        a.ensure_inert()
            .expect_err("a non-default threshold must fail startup");
    }
}

#[test]
fn ggml_draft_placement_values_are_rejected() {
    // Reject-any-value knobs: every member of the placement family, so the
    // manifest test pointers cover exactly what they claim.
    let reject_any: &[(&str, fn(&mut SpecCompatArgs, String))] = &[
        ("--spec-draft-cpu-mask", |a, v| {
            a.spec_draft_cpu_mask = Some(v)
        }),
        ("--spec-draft-cpu-mask-batch", |a, v| {
            a.spec_draft_cpu_mask_batch = Some(v);
        }),
        ("--spec-draft-cpu-range", |a, v| {
            a.spec_draft_cpu_range = Some(v);
        }),
        ("--spec-draft-device", |a, v| a.spec_draft_device = Some(v)),
        ("--spec-draft-override-tensor", |a, v| {
            a.spec_draft_override_tensor = Some(v);
        }),
        ("--spec-draft-n-cpu-moe", |a, v| {
            a.spec_draft_n_cpu_moe = Some(v);
        }),
    ];
    for (option, set) in reject_any {
        let mut a = args();
        set(&mut a, "anything".to_string());
        assert_eq!(&a.ensure_inert().expect_err(option).option, option);
    }

    let mut a = args();
    a.spec_draft_cpu_moe = true;
    assert_eq!(
        a.ensure_inert().expect_err("moe").option,
        "--spec-draft-cpu-moe"
    );

    // Inert-at-default knobs: the default passes, any other value fails.
    type Setter = fn(&mut SpecCompatArgs, String);
    let inert_pairs: &[(&str, Setter, &str, &str)] = &[
        (
            "--spec-draft-cpu-strict",
            |a, v| a.spec_draft_cpu_strict = Some(v),
            "0",
            "1",
        ),
        (
            "--spec-draft-cpu-strict-batch",
            |a, v| a.spec_draft_cpu_strict_batch = Some(v),
            "0",
            "1",
        ),
        (
            "--spec-draft-poll",
            |a, v| a.spec_draft_poll = Some(v),
            "50",
            "10",
        ),
        (
            "--spec-draft-poll-batch",
            |a, v| a.spec_draft_poll_batch = Some(v),
            "50",
            "10",
        ),
        (
            "--spec-draft-prio",
            |a, v| a.spec_draft_prio = Some(v),
            "0",
            "2",
        ),
        (
            "--spec-draft-prio-batch",
            |a, v| a.spec_draft_prio_batch = Some(v),
            "0",
            "2",
        ),
        (
            "--spec-draft-threads",
            |a, v| a.spec_draft_threads = Some(v),
            "-1",
            "8",
        ),
        (
            "--spec-draft-threads-batch",
            |a, v| a.spec_draft_threads_batch = Some(v),
            "-1",
            "8",
        ),
    ];
    for (option, set, inert, active) in inert_pairs {
        let mut a = args();
        set(&mut a, (*inert).to_string());
        a.ensure_inert()
            .unwrap_or_else(|r| panic!("{option} {inert} must be inert: {r}"));
        let mut a = args();
        set(&mut a, (*active).to_string());
        assert_eq!(&a.ensure_inert().expect_err(option).option, option);
    }
}

#[test]
fn spec_draft_hf_is_rejected_with_the_model_draft_alternative() {
    let mut a = args();
    a.spec_draft_hf = Some("org/gguf-draft".to_string());
    let rejection = a
        .ensure_inert()
        .expect_err("GGUF draft pulls are unsupported");
    assert_eq!(rejection.option, "--spec-draft-hf");
    assert!(
        rejection.to_string().contains("--model-draft"),
        "{rejection}"
    );
}

#[test]
fn draft_ngl_full_offload_spellings_are_inert_and_partial_offload_is_rejected() {
    // auto, all, and the historical negative count all mean "everything on
    // the accelerator", which is what mlxcel already does.
    for inert in ["auto", "all", "-1", "-99"] {
        let mut a = args();
        a.spec_draft_ngl = Some(inert.to_string());
        a.ensure_inert()
            .expect("full offload is what mlxcel already does");
    }
    let mut a = args();
    a.spec_draft_ngl = Some("20".to_string());
    a.ensure_inert()
        .expect_err("a partial draft offload must fail startup");
}

#[test]
fn draft_kv_cache_types_accept_only_f16() {
    let mut a = args();
    a.spec_draft_type_k = Some("f16".to_string());
    a.spec_draft_type_v = Some("f16".to_string());
    a.ensure_inert().expect("f16 is mlxcel's draft KV dtype");
    let mut a = args();
    a.spec_draft_type_k = Some("q8_0".to_string());
    a.ensure_inert()
        .expect_err("a quantized GGML draft cache must fail startup");
}

#[test]
fn ngram_tuning_knobs_are_inert_while_no_ngram_selector_can_be_chosen() {
    // The n-gram tuning knobs only take effect when --spec-type selects an
    // n-gram mode, which mlxcel rejects, so any tuning value is inert
    // exactly as it is upstream with a non-n-gram selector.
    let mut a = args();
    a.spec_ngram_simple_min_hits = Some("7".to_string());
    a.spec_ngram_map_k_size_m = Some("64".to_string());
    a.spec_ngram_mod_n_max = Some("128".to_string());
    a.ensure_inert()
        .expect("tuning values are inert without an n-gram selector");
}

#[test]
fn lookup_caches_are_rejected() {
    let mut a = args();
    a.lookup_cache_static = Some("/tmp/cache.bin".to_string());
    assert_eq!(
        a.ensure_inert().expect_err("lookup").option,
        "--lookup-cache-static"
    );
    let mut a = args();
    a.lookup_cache_dynamic = Some("/tmp/cache.bin".to_string());
    assert_eq!(
        a.ensure_inert().expect_err("lookup").option,
        "--lookup-cache-dynamic"
    );
}

#[test]
fn backend_sampling_positive_half_is_always_inert() {
    let mut a = args();
    a.spec_draft_backend_sampling = true;
    a.ensure_inert()
        .expect("accelerator sampling is what mlxcel already does");
}
