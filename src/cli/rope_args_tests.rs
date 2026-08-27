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
    group
        .resolve_yarn_request()
        .expect("the sentinels mean \"use the model's own values\"");
    assert_eq!(group.resolve().expect("sentinels only"), None);
}

#[test]
fn a_real_yarn_value_is_refused_with_the_flag_named() {
    let group = RopeOverrideArgs {
        yarn_beta_fast: Some(32.0),
        ..args()
    };
    let err = group
        .resolve()
        .expect_err("mlxcel's shared RoPE path has no YaRN arm");
    assert!(err.contains("--yarn-beta-fast 32"), "{err}");
    assert!(err.contains("#1472"), "{err}");
}

#[test]
fn every_non_sentinel_yarn_flag_is_named_in_one_message() {
    // One restart should tell the operator about all of them, not one per run.
    let group = RopeOverrideArgs {
        yarn_ext_factor: Some(1.0),
        yarn_attn_factor: Some(0.5),
        yarn_beta_slow: Some(1.0),
        yarn_beta_fast: Some(32.0),
        yarn_orig_ctx: Some(4096),
        ..args()
    };
    let err = group.resolve().expect_err("five real values");
    for expected in [
        "--yarn-ext-factor",
        "--yarn-attn-factor",
        "--yarn-beta-slow",
        "--yarn-beta-fast",
        "--yarn-orig-ctx 4096",
    ] {
        assert!(err.contains(expected), "{expected} missing from: {err}");
    }
}

#[test]
fn a_zero_ext_factor_is_a_real_request_not_a_sentinel() {
    // b10621 documents `0.0 = full interpolation`, which is a genuine YaRN
    // setting; only `-1.0` means "leave it alone".
    let group = RopeOverrideArgs {
        yarn_ext_factor: Some(0.0),
        ..args()
    };
    let err = group.resolve().expect_err("0.0 is full interpolation");
    assert!(err.contains("--yarn-ext-factor 0"), "{err}");
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
fn yarn_is_checked_before_the_rope_flags() {
    // Both halves are wrong here; the YaRN diagnostic is the more specific of
    // the two and must be the one the operator sees.
    let group = RopeOverrideArgs {
        rope_scale: Some(-1.0),
        yarn_beta_fast: Some(32.0),
        ..args()
    };
    let err = group.resolve().expect_err("both halves are invalid");
    assert!(err.contains("--yarn-beta-fast"), "{err}");
}

#[test]
fn rope_scaling_yarn_is_refused_even_with_no_yarn_flags() {
    let group = RopeOverrideArgs {
        rope_scaling: Some("yarn".to_string()),
        ..args()
    };
    let err = group.resolve().expect_err("yarn has no arm on this path");
    assert!(err.contains("--rope-scaling yarn"), "{err}");
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
