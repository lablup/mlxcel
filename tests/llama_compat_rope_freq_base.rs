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

//! End-to-end contract for llama-server b10621's `--rope-freq-base` override
//! (issue #1450). One configuration per process; see the header of
//! `llama_compat_rope_scaling_override.rs` for why.
//!
//! # The assertion without a reference run
//!
//! A process that has installed a base override cannot also produce the
//! un-overridden table to compare against, so this checks a property instead:
//! under `--rope-freq-base B`, two resolves that declare *different* bases must
//! produce the *same* table, because both were replaced by `B`. Without the
//! override those two tables differ in every entry, so the property is not
//! satisfiable by accident. A third resolve pins the absolute value by
//! declaring `B` itself.

use mlxcel::cli::rope_args::RopeOverrideArgs;
use mlxcel::models::rope_overrides;
use mlxcel::models::rope_utils::{RopeScalingKind, RopeScalingSpec};

const DIMS: usize = 64;
const OVERRIDE_BASE: f32 = 1_000_000.0;

fn llama3_spec() -> RopeScalingSpec {
    RopeScalingSpec {
        rope_type: Some("llama3".to_string()),
        factor: Some(32.0),
        low_freq_factor: Some(1.0),
        high_freq_factor: Some(4.0),
        original_max_position_embeddings: Some(8192.0),
        ..RopeScalingSpec::default()
    }
}

/// Read a resolved kind's frequency table as `f32`.
fn table(kind: &RopeScalingKind) -> Vec<f32> {
    let freqs = kind.freqs().expect("a llama3 block builds a table");
    let f32s = mlxcel_core::astype(freqs, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f32s);
    mlxcel_core::array_to_raw_bytes(&f32s)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn rope_freq_base_replaces_the_checkpoints_rope_theta_in_the_table() {
    // Without an override, two different declared bases give two different
    // tables. Establishing that first is what makes the post-install equality
    // meaningful rather than vacuous.
    let before_500k = table(&RopeScalingKind::resolve(
        Some(&llama3_spec()),
        DIMS,
        500_000.0,
        None,
        "probe",
    ));
    let before_10k = table(&RopeScalingKind::resolve(
        Some(&llama3_spec()),
        DIMS,
        10_000.0,
        None,
        "probe",
    ));
    assert_eq!(before_500k.len(), DIMS / 2);
    assert!(
        before_500k
            .iter()
            .zip(&before_10k)
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "the declared base must matter before anything overrides it"
    );

    let requested = RopeOverrideArgs {
        rope_freq_base: Some(OVERRIDE_BASE),
        ..RopeOverrideArgs::default()
    }
    .resolve()
    .expect("--rope-freq-base is serveable")
    .expect("it is an override");
    rope_overrides::install(Some(requested)).expect("first install in this process");

    let after_500k = table(&RopeScalingKind::resolve(
        Some(&llama3_spec()),
        DIMS,
        500_000.0,
        None,
        "probe",
    ));
    let after_10k = table(&RopeScalingKind::resolve(
        Some(&llama3_spec()),
        DIMS,
        10_000.0,
        None,
        "probe",
    ));
    let at_override = table(&RopeScalingKind::resolve(
        Some(&llama3_spec()),
        DIMS,
        OVERRIDE_BASE,
        None,
        "probe",
    ));

    assert_eq!(
        after_500k, after_10k,
        "under --rope-freq-base the checkpoint's own rope_theta must not reach the table"
    );
    assert_eq!(
        after_500k, at_override,
        "the table must be the one the requested base builds, not merely a consistent one"
    );
    assert_ne!(
        after_500k, before_500k,
        "the override must actually move the frequencies"
    );

    assert_eq!(rope_overrides::rejection(), None);
    rope_overrides::verify_applied("probe").expect("the override reached the rotation");
}
