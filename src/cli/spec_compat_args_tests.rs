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
fn spec_type_none_is_inert_and_ngram_modes_are_rejected() {
    let mut a = args();
    a.spec_type = Some("none".to_string());
    a.ensure_inert().expect("none is the inert selector");

    for mode in [
        "ngram-simple",
        "ngram-map-k",
        "ngram-map-k4v",
        "ngram-mod",
        "draft,ngram-simple",
    ] {
        let mut a = args();
        a.spec_type = Some(mode.to_string());
        let rejection = a
            .ensure_inert()
            .expect_err("non-none selectors must fail startup");
        assert_eq!(rejection.option, "--spec-type");
        assert!(
            rejection.to_string().contains("--model-draft"),
            "{rejection}"
        );
    }
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
    let mut a = args();
    a.spec_draft_cpu_mask = Some("0xff".to_string());
    assert_eq!(
        a.ensure_inert().expect_err("mask").option,
        "--spec-draft-cpu-mask"
    );

    let mut a = args();
    a.spec_draft_device = Some("cuda0".to_string());
    assert_eq!(
        a.ensure_inert().expect_err("device").option,
        "--spec-draft-device"
    );

    let mut a = args();
    a.spec_draft_cpu_moe = true;
    assert_eq!(
        a.ensure_inert().expect_err("moe").option,
        "--spec-draft-cpu-moe"
    );

    // Inert-by-value knobs.
    let mut a = args();
    a.spec_draft_poll = Some("50".to_string());
    a.spec_draft_prio = Some("0".to_string());
    a.spec_draft_threads = Some("-1".to_string());
    a.ensure_inert().expect("upstream defaults are inert");
    let mut a = args();
    a.spec_draft_poll = Some("10".to_string());
    a.ensure_inert().expect_err("a non-default poll must fail");
}

#[test]
fn draft_ngl_auto_and_all_are_inert_and_partial_offload_is_rejected() {
    for inert in ["auto", "all"] {
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
fn ngram_tuning_knobs_are_inert_while_the_selector_is_pinned_to_none() {
    // With --spec-type pinned to none, b10621 itself ignores every tuning
    // value, so any value is inert here too.
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
}

#[test]
fn backend_sampling_is_always_inert() {
    let mut a = args();
    a.spec_draft_backend_sampling = true;
    a.ensure_inert()
        .expect("accelerator sampling is what mlxcel already does");
}
