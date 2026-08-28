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

//! Classification tests for the b10621 multimodal projector / media group
//! (issue #1451).

use super::*;

fn args() -> MultimodalCompatArgs {
    MultimodalCompatArgs::default()
}

#[test]
fn a_bare_argument_set_is_inert() {
    assert_eq!(args().rejection(), None);
    assert_eq!(args().media_admission(), MediaAdmission::Auto);
    assert!(args().image_token_bounds().expect("no bounds").is_empty());
}

#[test]
fn mmproj_file_is_refused_with_an_mlx_replacement() {
    let mut a = args();
    a.mmproj = Some("/models/qwen2-vl-mmproj-f16.gguf".into());
    let rejection = a.rejection().expect("a projector file is refused");
    assert_eq!(rejection.option, "--mmproj");
    let rendered = rejection.to_string();
    assert!(rendered.contains("qwen2-vl-mmproj-f16.gguf"), "{rendered}");
    assert!(
        rendered.contains("integrated MLX VLM checkpoint"),
        "{rendered}"
    );
    // The diagnostic has to name a concrete replacement, not just say no.
    assert!(rendered.contains("--model"), "{rendered}");
}

#[test]
fn mmproj_url_is_refused_for_the_same_reason() {
    let mut a = args();
    a.mmproj_url = Some("https://example.invalid/mmproj.gguf".into());
    let rejection = a.rejection().expect("a projector URL is refused");
    assert_eq!(rejection.option, "--mmproj-url");
}

#[test]
fn an_empty_projector_value_from_an_unset_variable_is_ignored() {
    // `LLAMA_ARG_MMPROJ=` arrives as an empty string; refusing to start over one
    // would make the compatibility surface worse than ignoring it.
    let mut a = args();
    a.mmproj = Some("   ".into());
    a.mmproj_url = Some(String::new());
    assert_eq!(a.rejection(), None);
}

#[test]
fn projector_placement_accepts_only_the_inert_half() {
    let mut on = args();
    on.mmproj_offload = true;
    assert_eq!(on.rejection(), None, "the default half asks for nothing");

    let mut off = args();
    off.no_mmproj_offload = true;
    let rejection = off.rejection().expect("host offload is refused");
    assert_eq!(rejection.option, "--no-mmproj-offload");
    assert!(rejection.to_string().contains("one MLX device"));
}

#[test]
fn mmproj_device_none_and_a_named_device_are_both_refused() {
    let mut none = args();
    none.mmproj_device = Some("none".into());
    let rejection = none.rejection().expect("`none` is refused");
    assert_eq!(rejection.option, "--mmproj-device");
    assert!(
        rejection.alternative.is_none(),
        "`none` asks for a host projector, for which there is no alternative"
    );

    let mut named = args();
    named.mmproj_device = Some("CUDA0".into());
    let rejection = named.rejection().expect("a device name is refused");
    assert!(rejection.to_string().contains("MLXCEL_DEVICE"));
}

#[test]
fn mtmd_batch_max_tokens_is_inert_at_the_upstream_default_only() {
    for inert in ["1024", " 1024 ", "+1024"] {
        let mut a = args();
        a.mtmd_batch_max_tokens = Some(inert.into());
        assert_eq!(a.rejection(), None, "{inert} is b10621's own default");
    }
    let mut a = args();
    a.mtmd_batch_max_tokens = Some("512".into());
    let rejection = a.rejection().expect("a narrower batch is refused");
    assert_eq!(rejection.option, "--mtmd-batch-max-tokens");
    assert!(
        rejection
            .to_string()
            .contains("single vision-tower forward")
    );
}

#[test]
fn image_token_bounds_parse_upstreams_value_domain() {
    let mut a = args();
    a.image_min_tokens = Some("64".into());
    a.image_max_tokens = Some("1280".into());
    let bounds = a.image_token_bounds().expect("both halves parse");
    assert_eq!(bounds.min_tokens, Some(64));
    assert_eq!(bounds.max_tokens, Some(1280));
    assert!(!bounds.is_empty());
}

#[test]
fn a_non_positive_image_token_bound_means_use_the_models_own() {
    // Upstream only treats `custom_image_*_tokens > 0` as a custom bound, so
    // `0` and a negative value are the sentinel for "read it from the model"
    // rather than an error.
    for sentinel in ["0", "-1"] {
        let mut a = args();
        a.image_min_tokens = Some(sentinel.into());
        let bounds = a.image_token_bounds().expect("a sentinel is accepted");
        assert_eq!(bounds.min_tokens, None, "{sentinel}");
    }
}

#[test]
fn a_non_integer_image_token_bound_is_refused() {
    let mut a = args();
    a.image_max_tokens = Some("lots".into());
    let rejection = a.rejection().expect("a non-integer is refused");
    assert_eq!(rejection.option, "--image-max-tokens");
}

#[test]
fn a_maximum_below_the_minimum_is_refused_as_upstream_throws() {
    let mut a = args();
    a.image_min_tokens = Some("512".into());
    a.image_max_tokens = Some("64".into());
    let rejection = a.rejection().expect("an inverted budget is refused");
    assert_eq!(rejection.option, "--image-max-tokens");
    assert!(rejection.to_string().contains("below the minimum"));
}

#[test]
fn no_mmproj_disables_media_admission_on_both_spellings() {
    for mutate in [
        (|a: &mut MultimodalCompatArgs| a.no_mmproj = true) as fn(&mut MultimodalCompatArgs),
        |a: &mut MultimodalCompatArgs| a.no_mmproj_auto = true,
    ] {
        let mut a = args();
        mutate(&mut a);
        assert_eq!(a.media_admission(), MediaAdmission::Disabled);
        assert!(a.media_admission().is_disabled());
        // Disabling admission is honored, not refused.
        assert_eq!(a.rejection(), None);
    }
    let mut auto = args();
    auto.mmproj_auto = true;
    assert_eq!(auto.media_admission(), MediaAdmission::Auto);
}

#[test]
fn media_root_resolution_requires_an_existing_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut a = args();
    a.media_path = Some(dir.path().to_path_buf());
    let resolved = a
        .resolve_media_root()
        .expect("an existing directory resolves")
        .expect("some root");
    assert_eq!(
        resolved,
        std::fs::canonicalize(dir.path()).expect("canonical tempdir"),
        "the stored root must be canonical so containment compares like with like"
    );

    let mut missing = args();
    missing.media_path = Some(dir.path().join("nope"));
    assert!(missing.resolve_media_root().is_err());

    let file = dir.path().join("a.png");
    std::fs::write(&file, b"x").expect("write");
    let mut not_a_dir = args();
    not_a_dir.media_path = Some(file);
    let err = not_a_dir
        .resolve_media_root()
        .expect_err("a file is not a media root");
    assert!(err.contains("not a directory"), "{err}");
}

#[test]
fn no_media_path_leaves_local_files_disabled() {
    assert_eq!(args().resolve_media_root().expect("no root"), None);
}

#[test]
fn every_projector_flag_reports_in_a_fixed_order() {
    // A command line carrying several unsupported values must always name the
    // same one, so a deployment script's failure message does not depend on
    // clap's field order changing.
    let mut a = args();
    a.mmproj = Some("p.gguf".into());
    a.mmproj_url = Some("https://example.invalid/p.gguf".into());
    a.no_mmproj_offload = true;
    a.mmproj_device = Some("none".into());
    a.mtmd_batch_max_tokens = Some("1".into());
    assert_eq!(a.rejection().expect("refused").option, "--mmproj");
}

// ── environment vocabulary matches b10621 ───────────────────────────────────

/// Set `pairs` for the duration of `body`, restoring the previous values.
///
/// Serialized against every other environment-mutating test in this binary;
/// `cargo test` runs them all in one process and Rust 2024 makes an unguarded
/// `set_var` next to a concurrent read undefined behavior.
fn with_env<T>(pairs: &[(&str, Option<&str>)], body: impl FnOnce() -> T) -> T {
    let _guard = crate::test_support::env_lock::env_lock();
    let saved: Vec<(String, Option<String>)> = pairs
        .iter()
        .map(|(k, _)| ((*k).to_owned(), std::env::var(k).ok()))
        .collect();
    unsafe {
        for (key, value) in pairs {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
    let out = body();
    unsafe {
        for (key, value) in &saved {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
    out
}

#[test]
fn llama_arg_mmproj_auto_resolves_through_b10621s_own_vocabulary() {
    // clap's boolish parser accepts a wider vocabulary than b10621's
    // `parse_bool_value` and errors outside it, so the pair is resolved at
    // runtime instead of being bound by clap.
    for (value, expected) in [
        ("off", MediaAdmission::Disabled),
        ("false", MediaAdmission::Disabled),
        ("0", MediaAdmission::Disabled),
        ("on", MediaAdmission::Auto),
        ("1", MediaAdmission::Auto),
    ] {
        let admission = with_env(
            &[
                ("LLAMA_ARG_MMPROJ_AUTO", Some(value)),
                ("LLAMA_ARG_NO_MMPROJ_AUTO", None),
            ],
            || {
                let mut a = args();
                a.apply_env_bindings().expect("a recognized value");
                a.media_admission()
            },
        );
        assert_eq!(admission, expected, "LLAMA_ARG_MMPROJ_AUTO={value}");
    }
}

#[test]
fn the_no_alias_variable_means_false_as_upstream_defines_it() {
    let admission = with_env(
        &[
            ("LLAMA_ARG_MMPROJ_AUTO", Some("on")),
            ("LLAMA_ARG_NO_MMPROJ_AUTO", Some("whatever")),
        ],
        || {
            let mut a = args();
            a.apply_env_bindings().expect("the alias wins");
            a.media_admission()
        },
    );
    assert_eq!(admission, MediaAdmission::Disabled);
}

#[test]
fn an_unrecognized_boolean_value_stops_startup_as_upstream_throws() {
    let outcome = with_env(
        &[
            ("LLAMA_ARG_MMPROJ_OFFLOAD", Some("sometimes")),
            ("LLAMA_ARG_NO_MMPROJ_OFFLOAD", None),
        ],
        || args().apply_env_bindings(),
    );
    let (var, raw) = outcome.expect_err("an unrecognized value is an error");
    assert_eq!(var, "LLAMA_ARG_MMPROJ_OFFLOAD");
    assert_eq!(raw, "sometimes");
}

#[test]
fn llama_arg_mmproj_offload_off_reaches_the_rejection() {
    let rejection = with_env(
        &[
            ("LLAMA_ARG_MMPROJ_OFFLOAD", Some("off")),
            ("LLAMA_ARG_NO_MMPROJ_OFFLOAD", None),
        ],
        || {
            let mut a = args();
            a.apply_env_bindings().expect("a recognized value");
            a.rejection()
        },
    );
    assert_eq!(
        rejection.expect("host offload is refused").option,
        "--no-mmproj-offload"
    );
}

#[test]
fn an_explicit_command_line_half_wins_over_the_environment() {
    let admission = with_env(
        &[
            ("LLAMA_ARG_MMPROJ_AUTO", Some("off")),
            ("LLAMA_ARG_NO_MMPROJ_AUTO", None),
        ],
        || {
            let mut a = args();
            a.mmproj_auto = true;
            a.apply_env_bindings().expect("no parse needed");
            a.media_admission()
        },
    );
    assert_eq!(admission, MediaAdmission::Auto);
}
