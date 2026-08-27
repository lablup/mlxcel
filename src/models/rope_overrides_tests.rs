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

//! Unit tests for the b10621 RoPE runtime override.
//!
//! Everything here exercises the pure decision table
//! ([`RopeRuntimeOverride::from_flags`] and
//! [`RopeRuntimeOverride::apply_to_spec`]). The process-wide slot is
//! deliberately untested at unit level: it is a `OnceLock` shared by every test
//! binary thread, so a test that installed one would decide what every other
//! test in the same process rotates with. The seam that reads it is covered
//! end-to-end instead, in `tests/llama_compat_rope_scaling_override.rs` and
//! `tests/llama_compat_rope_freq_base.rs`, one configuration per test binary.

use super::*;

fn spec(rope_type: &str, factor: Option<f32>) -> RopeScalingSpec {
    RopeScalingSpec {
        rope_type: Some(rope_type.to_string()),
        factor,
        ..RopeScalingSpec::default()
    }
}

#[test]
fn no_flags_means_no_override() {
    assert_eq!(
        RopeRuntimeOverride::from_flags(None, None, None, None).expect("no flags is not an error"),
        None
    );
}

#[test]
fn rope_scale_is_the_reciprocal_of_rope_freq_scale() {
    // b10621: `--rope-scale N` sets rope_freq_scale = 1/N, `--rope-freq-scale
    // N` sets it to N. Both spellings of "expand the context 8x" must land on
    // the same stored value.
    let from_scale = RopeRuntimeOverride::from_flags(None, Some(8.0), None, None)
        .expect("valid")
        .expect("an override");
    let from_freq = RopeRuntimeOverride::from_flags(None, None, Some(0.125), None)
        .expect("valid")
        .expect("an override");
    assert_eq!(from_scale.freq_scale(), Some(0.125));
    assert_eq!(from_freq.freq_scale(), Some(0.125));
}

#[test]
fn the_two_scale_spellings_may_agree_but_not_disagree() {
    RopeRuntimeOverride::from_flags(None, Some(8.0), Some(0.125), None)
        .expect("reciprocal values agree");

    let err = RopeRuntimeOverride::from_flags(None, Some(8.0), Some(0.5), None)
        .expect_err("non-reciprocal values are ambiguous");
    assert!(err.contains("--rope-scale"), "{err}");
    assert!(err.contains("--rope-freq-scale"), "{err}");
}

#[test]
fn yarn_is_refused_at_the_edge_with_a_reason() {
    let err = RopeRuntimeOverride::from_flags(Some("yarn"), None, None, None)
        .expect_err("yarn has no representation on this path");
    assert!(err.contains("not implemented"), "{err}");
    // Naming the families that DO serve YaRN from their own config is what
    // stops the message reading as "mlxcel cannot do YaRN at all".
    assert!(err.contains("DeepSeek"), "{err}");
}

#[test]
fn an_unknown_scheme_names_the_accepted_domain() {
    let err = RopeRuntimeOverride::from_flags(Some("dynamic"), None, None, None)
        .expect_err("b10621 accepts three spellings");
    assert!(err.contains("{none,linear,yarn}"), "{err}");
}

#[test]
fn non_positive_and_non_finite_scalars_are_refused() {
    for (scale, freq_scale, base) in [
        (Some(0.0), None, None),
        (Some(-2.0), None, None),
        (None, Some(f32::NAN), None),
        (None, Some(f32::INFINITY), None),
        (None, None, Some(0.0)),
        (None, None, Some(-10000.0)),
    ] {
        let err = RopeRuntimeOverride::from_flags(None, scale, freq_scale, base).expect_err(
            "a non-positive or non-finite factor produces NaN logits with no error anywhere",
        );
        assert!(err.contains("positive finite"), "{err}");
    }
}

#[test]
fn scaling_none_drops_the_checkpoints_own_block() {
    let over = RopeRuntimeOverride::from_flags(Some("none"), None, None, None)
        .expect("valid")
        .expect("an override");
    let declared = spec("llama3", Some(8.0));
    assert_eq!(
        over.apply_to_spec(Some(&declared)).expect("none applies"),
        None
    );
}

#[test]
fn scaling_linear_with_a_scale_sets_the_reciprocal_factor() {
    let over = RopeRuntimeOverride::from_flags(Some("linear"), Some(4.0), None, None)
        .expect("valid")
        .expect("an override");
    let resolved = over
        .apply_to_spec(None)
        .expect("linear applies")
        .expect("a block");
    assert_eq!(resolved.rope_type(), "linear");
    assert_eq!(resolved.factor, Some(4.0));
}

#[test]
fn scaling_linear_without_a_scale_keeps_the_checkpoints_factor() {
    let over = RopeRuntimeOverride::from_flags(Some("linear"), None, None, None)
        .expect("valid")
        .expect("an override");
    let declared = spec("llama3", Some(8.0));
    let resolved = over
        .apply_to_spec(Some(&declared))
        .expect("linear applies")
        .expect("a block");
    assert_eq!(resolved.rope_type(), "linear");
    assert_eq!(resolved.factor, Some(8.0));
}

#[test]
fn scaling_linear_on_a_checkpoint_with_no_block_is_a_no_op_factor() {
    let over = RopeRuntimeOverride::from_flags(Some("linear"), None, None, None)
        .expect("valid")
        .expect("an override");
    let resolved = over
        .apply_to_spec(None)
        .expect("linear applies")
        .expect("a block");
    assert_eq!(resolved.factor, Some(1.0));
}

#[test]
fn a_bare_scale_defaults_to_linear_when_the_model_names_no_scheme() {
    // b10621's help text: "defaults to linear unless specified by the model".
    let over = RopeRuntimeOverride::from_flags(None, Some(2.0), None, None)
        .expect("valid")
        .expect("an override");
    for declared in [
        None,
        Some(spec("default", None)),
        Some(spec("linear", Some(4.0))),
    ] {
        let resolved = over
            .apply_to_spec(declared.as_ref())
            .expect("linear applies")
            .expect("a block");
        assert_eq!(resolved.rope_type(), "linear");
        assert_eq!(resolved.factor, Some(2.0));
    }
}

#[test]
fn a_bare_scale_over_a_banded_scheme_is_refused_rather_than_composed() {
    // The one combination with no defensible answer: llama.cpp multiplies its
    // own rope_freq_scale into the banded rotation, mlxcel's llama3 table has
    // no such multiplier, and silently dropping either half changes the
    // rotation without saying so.
    let over = RopeRuntimeOverride::from_flags(None, Some(2.0), None, None)
        .expect("valid")
        .expect("an override");
    let declared = spec("llama3", Some(8.0));
    let err = over
        .apply_to_spec(Some(&declared))
        .expect_err("a scale on top of llama3 is ambiguous");
    assert!(err.contains("llama3"), "{err}");
    assert!(err.contains("--rope-scaling linear"), "{err}");
}

#[test]
fn a_base_only_override_leaves_the_block_alone() {
    let over = RopeRuntimeOverride::from_flags(None, None, None, Some(1_000_000.0))
        .expect("valid")
        .expect("an override");
    let declared = spec("llama3", Some(8.0));
    let resolved = over
        .apply_to_spec(Some(&declared))
        .expect("a base override touches no block")
        .expect("a block");
    assert_eq!(resolved.rope_type(), "llama3");
    assert_eq!(resolved.factor, Some(8.0));
    assert_eq!(over.apply_to_base(500_000.0), 1_000_000.0);
}

#[test]
fn a_base_is_left_at_the_checkpoints_value_when_no_base_flag_was_given() {
    let over = RopeRuntimeOverride::from_flags(Some("none"), None, None, None)
        .expect("valid")
        .expect("an override");
    assert_eq!(over.apply_to_base(500_000.0), 500_000.0);
}

#[test]
fn describe_names_every_flag_that_was_set() {
    let over = RopeRuntimeOverride::from_flags(Some("linear"), Some(4.0), None, Some(10_000.0))
        .expect("valid")
        .expect("an override");
    let described = over.describe();
    assert!(described.contains("--rope-scaling linear"), "{described}");
    assert!(described.contains("--rope-freq-scale 0.25"), "{described}");
    assert!(described.contains("--rope-freq-base 10000"), "{described}");
}
