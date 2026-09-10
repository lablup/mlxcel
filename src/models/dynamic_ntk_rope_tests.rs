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

//! Unit tests for the shared dynamic NTK schedule (#1324).
//!
//! The property that matters most here is invisible from any output check: a
//! dynamic block must leave the *position* alone. The schedule that shipped
//! rotated every position at twice its value, and the result is fluent text on
//! every prompt, at every length, that simply is not the model the weights
//! describe. So the position scale is asserted directly, per mode, rather than
//! inferred from a generation.
//!
//! The base half is the mirror image: it is a no-op on every prompt shorter
//! than `max_position_embeddings`, which is every prompt anyone runs by hand,
//! so it is checked against the closed form at lengths that actually cross the
//! boundary.

use super::{DynamicNtkRope, DynamicNtkRopeMode};
use crate::models::rope_utils::RopeScalingSpec;

/// The InternLM3 validation checkpoint's geometry
/// (`mlx-community/internlm3-8b-instruct-4bit`).
const DIMS: i32 = 128;
const BASE: f32 = 50_000_000.0;
const MAX_POS: usize = 32768;

// `the_rescale_log_fires_once_per_schedule_not_once_per_process` asserts on the
// process-global dedup set inside `log_dynamic_rescale_once`, so it reserves
// the geometries `(64, {111_111.0, 222_222.0, 333_333.0}, 1024, {2.0, 4.0})`.
// A new test that crosses `max_position_embeddings` on one of those would make
// that test's first assertion depend on execution order. Use the constants
// above, or a geometry no other test names.

/// Parse a `rope_scaling` block the way a `config.json` delivers it.
fn spec(json: &str) -> RopeScalingSpec {
    serde_json::from_str(json).unwrap_or_else(|err| panic!("block must parse: {err}\n{json}"))
}

fn build(json: Option<&str>) -> Result<DynamicNtkRope, String> {
    let parsed = json.map(spec);
    DynamicNtkRope::from_scaling(
        DIMS,
        BASE,
        false,
        MAX_POS,
        parsed.as_ref(),
        "internlm3-8b-4bit",
    )
}

fn built(json: Option<&str>) -> DynamicNtkRope {
    build(json).expect("block must resolve")
}

/// Assert `value` is within `1e-3` relative of `expected`.
///
/// The reference figures are computed in f64; `base_for` evaluates the same
/// expression in f32 to match `fast_rope`'s parameter type, which costs about
/// 1e-7 relative at these magnitudes. The tolerance is four orders of magnitude
/// looser than that and still tight enough that using the wrong `factor`, or
/// skipping the clamp, fails it by orders of magnitude.
fn assert_close(value: f32, expected: f64, what: &str) {
    let rel = ((value as f64) - expected).abs() / expected.abs();
    assert!(
        rel < 1e-3,
        "{what}: got {value}, expected {expected} (relative error {rel:.3e})"
    );
}

// The position scale, per mode. This is the defect #1324 fixed.

#[test]
fn an_absent_block_leaves_positions_unscaled() {
    let rope = built(None);
    assert_eq!(rope.mode(), DynamicNtkRopeMode::Default);
    assert_eq!(rope.scale(), 1.0);
    assert_eq!(rope.base_for(1), BASE);
    assert_eq!(rope.base_for(1_000_000), BASE);
}

#[test]
fn a_dynamic_block_leaves_positions_unscaled() {
    // The whole point of the issue. `internlm3` returned 2.0 here, so the token
    // at absolute position p was rotated as if it sat at 2p, at every context
    // length, on the checkpoint the family is validated against.
    let rope = built(Some(r#"{"factor": 6.0, "rope_type": "dynamic"}"#));
    assert_eq!(rope.mode(), DynamicNtkRopeMode::Dynamic { factor: 6.0 });
    assert_eq!(rope.scale(), 1.0);
}

#[test]
fn a_linear_block_divides_positions_by_the_factor() {
    let rope = built(Some(r#"{"factor": 4.0, "rope_type": "linear"}"#));
    assert_eq!(rope.mode(), DynamicNtkRopeMode::Linear { factor: 4.0 });
    assert_eq!(rope.scale(), 0.25);
    // Linear never touches the base, at any length.
    assert_eq!(rope.base_for(100), BASE);
    assert_eq!(rope.base_for(1_000_000), BASE);
}

#[test]
fn a_default_block_is_the_unscaled_schedule() {
    let rope = built(Some(r#"{"rope_type": "default"}"#));
    assert_eq!(rope.mode(), DynamicNtkRopeMode::Default);
    assert_eq!(rope.scale(), 1.0);
}

// Which key the scheme is read from.

#[test]
fn both_spellings_of_the_scheme_key_resolve() {
    // InternLM3 checkpoints spell it `rope_type`; InternLM2 checkpoints spell
    // it `type` (`models/internlm2_5-7b-chat-4bit` ships `{"type": "dynamic",
    // "factor": 2.0}`). One helper serves both families, so it has to read
    // both, and a config carrying both keys has to parse rather than hit
    // serde's `duplicate field`.
    assert_eq!(
        built(Some(r#"{"factor": 6.0, "rope_type": "dynamic"}"#)).mode(),
        DynamicNtkRopeMode::Dynamic { factor: 6.0 }
    );
    assert_eq!(
        built(Some(r#"{"type": "dynamic", "factor": 2.0}"#)).mode(),
        DynamicNtkRopeMode::Dynamic { factor: 2.0 }
    );
    assert_eq!(
        built(Some(
            r#"{"type": "linear", "rope_type": "linear", "factor": 2.0}"#
        ))
        .mode(),
        DynamicNtkRopeMode::Linear { factor: 2.0 }
    );
}

// The base schedule.

#[test]
fn the_dynamic_base_does_not_move_at_or_below_max_position() {
    // `seq_len` is clamped up to `max_position_embeddings`, so the ratio is
    // exactly 1.0 and the base is bit-identical to `rope_theta` for every
    // sequence that fits. A short prompt therefore cannot distinguish a correct
    // dynamic schedule from a plain one through the base; only the position
    // scale separates them there.
    let rope = built(Some(r#"{"factor": 6.0, "rope_type": "dynamic"}"#));
    assert_eq!(rope.base_for(1), BASE);
    assert_eq!(rope.base_for(100), BASE);
    assert_eq!(rope.base_for(MAX_POS as i32), BASE);
    // Zero and negative lengths cannot arise from `L + offset`, but the clamp
    // has to hold for them rather than producing a base below `rope_theta`.
    assert_eq!(rope.base_for(0), BASE);
    assert_eq!(rope.base_for(-5), BASE);
}

#[test]
fn the_dynamic_base_grows_past_max_position() {
    // base * ((f * seq / M) - (f - 1)) ^ (d / (d - 2)), computed in f64 for
    // f = 6, d = 128, M = 32768, base = 5e7.
    let rope = built(Some(r#"{"factor": 6.0, "rope_type": "dynamic"}"#));
    assert_close(
        rope.base_for(32769),
        50_009_300.608_753,
        "just past the boundary",
    );
    assert_close(rope.base_for(40000), 117_777_118.662_832, "40000");
    assert_close(rope.base_for(65536), 360_979_300.432_547, "65536");
}

#[test]
fn the_dynamic_base_grows_past_max_position_for_internlm2_geometry() {
    // `models/internlm2_5-7b-chat-4bit`: factor 2.0, rope_theta 1e6, same head_dim and
    // max_position_embeddings. Its block was dropped at deserialization before
    // this change, so the base never left 1e6 at any length.
    let rope = DynamicNtkRope::from_scaling(
        128,
        1_000_000.0,
        false,
        32768,
        Some(&spec(r#"{"type": "dynamic", "factor": 2.0}"#)),
        "internlm2_5-7b-chat-4bit",
    )
    .expect("block must resolve");
    assert_eq!(rope.base_for(32768), 1_000_000.0);
    assert_close(rope.base_for(40000), 1_449_795.741_993, "40000");
    assert_close(rope.base_for(65536), 3_052_773.674_881, "65536");
}

#[test]
fn the_dynamic_base_is_monotonic_in_sequence_length() {
    let rope = built(Some(r#"{"factor": 6.0, "rope_type": "dynamic"}"#));
    let mut previous = rope.base_for(MAX_POS as i32);
    for seq in [40000, 65536, 131_072, 262_144] {
        let next = rope.base_for(seq);
        assert!(
            next > previous,
            "base must grow with sequence length: {seq} gave {next}, previous was {previous}"
        );
        previous = next;
    }
}

// Blocks that cannot be served.

#[test]
fn an_unimplemented_scheme_is_a_load_error() {
    // Both upstream `__post_init__` implementations raise on anything outside
    // {"linear", "dynamic"}, and neither InternLM `ModelArgs` is reachable from
    // a VLM `text_config`, so a load error here cannot take a working
    // checkpoint offline the way it would on the shared Llama path.
    for json in [
        r#"{"rope_type": "yarn", "factor": 40.0}"#,
        r#"{"rope_type": "llama3", "factor": 8.0}"#,
        r#"{"type": "longrope", "factor": 4.0}"#,
    ] {
        let err = build(Some(json)).expect_err("scheme must be rejected");
        assert!(
            err.contains("not implemented"),
            "error must name the problem: {err}"
        );
        assert!(
            err.contains("internlm3-8b-4bit"),
            "error must name the checkpoint: {err}"
        );
    }
}

#[test]
fn a_scaled_scheme_without_a_usable_factor_is_a_load_error() {
    for json in [
        r#"{"rope_type": "dynamic"}"#,
        r#"{"rope_type": "linear"}"#,
        r#"{"rope_type": "dynamic", "factor": "6.0"}"#,
    ] {
        let err = build(Some(json)).expect_err("missing factor must be rejected");
        assert!(err.contains("no numeric factor"), "unexpected error: {err}");
    }

    for json in [
        r#"{"rope_type": "dynamic", "factor": 0.0}"#,
        r#"{"rope_type": "linear", "factor": -2.0}"#,
    ] {
        let err = build(Some(json)).expect_err("unusable factor must be rejected");
        assert!(
            err.contains("not a positive finite number"),
            "unexpected error: {err}"
        );
    }
}

#[test]
fn a_dynamic_block_with_a_zero_max_position_is_a_load_error() {
    // The base schedule divides by `max_position_embeddings`. Zero would make
    // every effective base `inf`, every logit `NaN`, and nothing on the path
    // throws.
    let err = DynamicNtkRope::from_scaling(
        DIMS,
        BASE,
        false,
        0,
        Some(&spec(r#"{"rope_type": "dynamic", "factor": 6.0}"#)),
        "broken-checkpoint",
    )
    .expect_err("zero max_position_embeddings must be rejected");
    assert!(
        err.contains("max_position_embeddings"),
        "unexpected error: {err}"
    );

    // Only the dynamic branch needs it, so the other two still build.
    assert!(build_with_max_pos(0, None).is_ok());
    assert!(build_with_max_pos(0, Some(r#"{"rope_type": "linear", "factor": 2.0}"#)).is_ok());
}

fn build_with_max_pos(max_pos: usize, json: Option<&str>) -> Result<DynamicNtkRope, String> {
    let parsed = json.map(spec);
    DynamicNtkRope::from_scaling(DIMS, BASE, false, max_pos, parsed.as_ref(), "probe")
}

// What `apply` actually hands MLX.

/// Read an array back as raw bytes, for bit-exact comparison.
fn bytes(a: &mlxcel_core::MlxArray) -> Vec<u8> {
    mlxcel_core::eval(a);
    mlxcel_core::array_to_raw_bytes(a)
}

/// A `[1, 2, 8, 16]` float32 tensor of ascending values.
fn sample() -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n = 2 * 8 * 16;
    let vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 1.0).collect();
    mlxcel_core::from_slice_f32(&vals, &[1, 2, 8, 16])
}

#[test]
fn apply_uses_unit_scale_and_the_plain_base_inside_max_position() {
    let rope = DynamicNtkRope::from_scaling(
        16,
        BASE,
        false,
        MAX_POS,
        Some(&spec(r#"{"factor": 6.0, "rope_type": "dynamic"}"#)),
        "probe",
    )
    .expect("block must resolve");

    let x = sample();
    let got = rope.apply(&x, 3, 11);
    let want = mlxcel_core::fast_rope(&x, 16, false, BASE, 1.0, 3);
    assert_eq!(
        bytes(&got),
        bytes(&want),
        "a dynamic schedule inside max_position_embeddings is the plain schedule"
    );

    // And it is not the schedule that shipped.
    let shipped = mlxcel_core::fast_rope(&x, 16, false, BASE, 2.0, 3);
    assert_ne!(
        bytes(&got),
        bytes(&shipped),
        "scale 2.0 must not produce the same rotation as scale 1.0"
    );
}

#[test]
fn apply_uses_the_rescaled_base_past_max_position() {
    // `max_position_embeddings` of 8 puts an 11-position forward past the
    // boundary, which is the only cheap way to exercise the branch that a
    // 32768-token prompt would otherwise be needed for.
    let rope = DynamicNtkRope::from_scaling(
        16,
        BASE,
        false,
        8,
        Some(&spec(r#"{"factor": 6.0, "rope_type": "dynamic"}"#)),
        "probe",
    )
    .expect("block must resolve");

    let expected_base = rope.base_for(11);
    assert!(
        expected_base > BASE,
        "11 positions past a max of 8 must rescale the base, got {expected_base}"
    );

    let x = sample();
    let got = rope.apply(&x, 3, 11);
    let want = mlxcel_core::fast_rope(&x, 16, false, expected_base, 1.0, 3);
    assert_eq!(bytes(&got), bytes(&want));
}

#[test]
fn apply_divides_positions_for_a_linear_block() {
    let rope = DynamicNtkRope::from_scaling(
        16,
        BASE,
        false,
        MAX_POS,
        Some(&spec(r#"{"factor": 2.0, "rope_type": "linear"}"#)),
        "probe",
    )
    .expect("block must resolve");

    let x = sample();
    let got = rope.apply(&x, 3, 11);
    let want = mlxcel_core::fast_rope(&x, 16, false, BASE, 0.5, 3);
    assert_eq!(bytes(&got), bytes(&want));
}

#[test]
fn apply_honors_the_traditional_flag() {
    let rope = DynamicNtkRope::from_scaling(16, BASE, true, MAX_POS, None, "probe")
        .expect("absent block must resolve");
    let x = sample();
    let got = rope.apply(&x, 0, 8);
    let want = mlxcel_core::fast_rope(&x, 16, true, BASE, 1.0, 0);
    assert_eq!(bytes(&got), bytes(&want));
    let interleaved = mlxcel_core::fast_rope(&x, 16, false, BASE, 1.0, 0);
    assert_ne!(bytes(&got), bytes(&interleaved));
}

#[test]
fn the_rescale_log_fires_once_per_schedule_not_once_per_process() {
    // Distinct geometry per schedule so this test cannot collide with the
    // `apply` tests above, which cross the boundary on the shared constants
    // and so populate the same dedup set.
    let probe = |dims: i32, base: f32, max_pos: usize, factor: f64| {
        DynamicNtkRope::from_scaling(
            dims,
            base,
            false,
            max_pos,
            Some(&spec(&format!(
                r#"{{"rope_type": "dynamic", "factor": {factor}}}"#
            ))),
            "dedup-probe",
        )
        .expect("block must resolve")
    };

    let rope = probe(64, 111_111.0, 1024, 2.0);
    let past = 2048;
    assert!(
        rope.log_dynamic_rescale_once(past, rope.base_for(past)),
        "the first crossing must log"
    );
    assert!(
        !rope.log_dynamic_rescale_once(past, rope.base_for(past)),
        "the same schedule must not log again"
    );
    // `Copy`, so the caller's per-layer rebuild is a distinct value with the
    // same parameters. It must still be deduped.
    assert!(
        !probe(64, 111_111.0, 1024, 2.0).log_dynamic_rescale_once(past, 0.0),
        "an equal schedule rebuilt per layer must not log again"
    );

    // A second checkpoint in the same process still gets its line, which is
    // what a process-wide flag would have swallowed.
    assert!(
        probe(64, 222_222.0, 1024, 2.0).log_dynamic_rescale_once(past, 0.0),
        "a different base must log"
    );
    assert!(
        probe(64, 111_111.0, 1024, 4.0).log_dynamic_rescale_once(past, 0.0),
        "a different factor must log"
    );

    // Inside the boundary, and any non-dynamic mode, never logs.
    assert!(
        !probe(64, 333_333.0, 1024, 2.0).log_dynamic_rescale_once(1024, 333_333.0),
        "seq_len == max_position_embeddings is a no-op, so it must not log"
    );
    let linear = DynamicNtkRope::from_scaling(
        64,
        333_333.0,
        false,
        1024,
        Some(&spec(r#"{"rope_type": "linear", "factor": 2.0}"#)),
        "dedup-probe",
    )
    .expect("block must resolve");
    assert!(
        !linear.log_dynamic_rescale_once(2048, 333_333.0),
        "a linear schedule never rescales the base, so it must not log"
    );
}
