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

//! Unit tests for the b10621 RoPE / YaRN flag group.

use super::*;

fn args() -> RopeOverrideArgs {
    RopeOverrideArgs::default()
}

#[test]
fn an_empty_group_resolves_to_no_override() {
    assert_eq!(args().resolve().expect("no flags is not an error"), None);
}

#[test]
fn the_yarn_sentinels_are_accepted_and_carry_no_override() {
    // A deployment script that spells out b10621's own defaults is asking for
    // the checkpoint's behavior, and must keep working.
    let group = RopeOverrideArgs {
        yarn_ext_factor: Some(-1.0),
        yarn_attn_factor: Some(-1.0),
        yarn_beta_fast: Some(-1.0),
        yarn_beta_slow: Some(-1.0),
        yarn_orig_ctx: Some(0),
        ..args()
    };
    assert!(
        !group.yarn_knobs().any_set(),
        "sentinels resolve to no knobs"
    );
    assert_eq!(group.resolve().expect("sentinels only"), None);
}

#[test]
fn a_real_yarn_value_becomes_a_knob_on_the_override() {
    // Since #1472 a non-sentinel value is a knob for the YaRN table builder,
    // not a refusal; with no YaRN rotation in force it stays inert, exactly as
    // it does in b10621.
    let group = RopeOverrideArgs {
        yarn_beta_fast: Some(40.0),
        ..args()
    };
    let over = group
        .resolve()
        .expect("a YaRN knob is serveable")
        .expect("a knob installs an override");
    assert_eq!(over.yarn_knobs().beta_fast, Some(40.0));
    assert_eq!(over.yarn_knobs().beta_slow, None);
}

#[test]
fn every_non_sentinel_yarn_flag_resolves_onto_the_knobs() {
    let group = RopeOverrideArgs {
        yarn_ext_factor: Some(1.0),
        yarn_attn_factor: Some(0.5),
        yarn_beta_slow: Some(1.5),
        yarn_beta_fast: Some(32.0),
        yarn_orig_ctx: Some(4096),
        ..args()
    };
    let over = group.resolve().expect("valid").expect("an override");
    let knobs = over.yarn_knobs();
    assert_eq!(knobs.ext_factor, Some(1.0));
    assert_eq!(knobs.attn_factor, Some(0.5));
    assert_eq!(knobs.beta_slow, Some(1.5));
    assert_eq!(knobs.beta_fast, Some(32.0));
    assert_eq!(knobs.orig_ctx, Some(4096));
}

#[test]
fn a_zero_ext_factor_is_a_real_request_not_a_sentinel() {
    // b10621 documents `0.0 = full interpolation`, which is a genuine YaRN
    // setting; only `-1.0` means "leave it alone".
    let group = RopeOverrideArgs {
        yarn_ext_factor: Some(0.0),
        ..args()
    };
    let over = group.resolve().expect("valid").expect("an override");
    assert_eq!(over.yarn_knobs().ext_factor, Some(0.0));
}

#[test]
fn a_yarn_knob_outside_its_domain_is_refused_at_startup() {
    let group = RopeOverrideArgs {
        yarn_beta_fast: Some(f32::NAN),
        ..args()
    };
    let err = group.resolve().expect_err("NaN cannot tune a rotation");
    assert!(err.contains("--yarn-beta-fast"), "{err}");

    let group = RopeOverrideArgs {
        yarn_orig_ctx: Some(-8),
        ..args()
    };
    let err = group.resolve().expect_err("a negative context is refused");
    assert!(err.contains("--yarn-orig-ctx -8"), "{err}");
}

#[test]
fn the_rope_flags_resolve_into_an_override() {
    let group = RopeOverrideArgs {
        rope_scaling: Some("linear".to_string()),
        rope_scale: Some(4.0),
        ..args()
    };
    let over = group
        .resolve()
        .expect("valid")
        .expect("a linear override was requested");
    assert_eq!(over.freq_scale(), Some(0.25));
}

#[test]
fn an_invalid_rope_scale_is_still_refused_next_to_a_yarn_knob() {
    let group = RopeOverrideArgs {
        rope_scale: Some(-1.0),
        yarn_beta_fast: Some(32.0),
        ..args()
    };
    let err = group.resolve().expect_err("the scale half is invalid");
    assert!(err.contains("--rope-scale"), "{err}");
}

#[test]
fn rope_scaling_yarn_resolves_into_an_override() {
    let group = RopeOverrideArgs {
        rope_scaling: Some("yarn".to_string()),
        rope_scale: Some(4.0),
        ..args()
    };
    let over = group
        .resolve()
        .expect("yarn is serveable since #1472")
        .expect("an override");
    assert_eq!(
        over.scaling_type(),
        Some(crate::models::rope_overrides::RopeScalingTypeOverride::Yarn)
    );
    assert_eq!(over.freq_scale(), Some(0.25));
}

#[test]
fn the_group_parses_from_a_command_line_on_a_bare_parser() {
    use clap::Parser;

    #[derive(Parser)]
    #[command(allow_negative_numbers = true)]
    struct Probe {
        #[command(flatten)]
        rope: RopeOverrideArgs,
    }

    let parsed = Probe::try_parse_from([
        "probe",
        "--rope-scaling",
        "linear",
        "--rope-scale",
        "8",
        "--rope-freq-base",
        "1000000",
        "--yarn-ext-factor",
        "-1",
    ])
    .expect("the b10621 spellings parse");
    assert_eq!(parsed.rope.rope_scaling.as_deref(), Some("linear"));
    assert_eq!(parsed.rope.rope_scale, Some(8.0));
    assert_eq!(parsed.rope.rope_freq_base, Some(1_000_000.0));
    // The negative sentinel has to survive argv parsing, not just resolution.
    assert_eq!(parsed.rope.yarn_ext_factor, Some(-1.0));
}
