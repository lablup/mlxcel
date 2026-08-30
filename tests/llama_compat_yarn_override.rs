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

//! End-to-end contract for llama-server b10621's `--rope-scaling yarn` and the
//! five `--yarn-*` knobs (issue #1472).
//!
//! One test binary for one process-wide configuration, for the same reason
//! `llama_compat_rope_scaling_override.rs` is: the override is a `OnceLock`,
//! so a second configuration in the same process would decide what the first
//! one rotates with.
//!
//! What is asserted, in server execution order: an uninstalled process builds
//! no YaRN table for a scaling-free checkpoint, the flag group resolves the
//! yarn request into an installed override, an unconsumed override refuses to
//! serve, the seam then builds the YaRN table with the knobs applied, and the
//! verification passes.

use mlxcel::cli::rope_args::RopeOverrideArgs;
use mlxcel::models::rope_overrides;
use mlxcel::models::rope_utils::RopeScalingKind;

const DIMS: usize = 64;
const BASE: f32 = 500_000.0;
const TRAIN_CTX: f32 = 8192.0;

fn table(kind: &RopeScalingKind) -> Vec<f32> {
    let freqs = kind.freqs().expect("a yarn request builds a table");
    let f32s = mlxcel_core::astype(freqs, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f32s);
    mlxcel_core::array_to_raw_bytes(&f32s)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn rope_scaling_yarn_with_knobs_reaches_the_rotation_and_is_verified() {
    // 1. Nothing installed: a checkpoint with no scaling block stays on the
    //    plain table.
    assert!(rope_overrides::installed().is_none());
    let bare = RopeScalingKind::resolve(None, DIMS, BASE, Some(TRAIN_CTX), "probe");
    assert!(bare.freqs().is_none(), "no block, no override, no table");
    assert_eq!(bare.attn_scale(), 1.0);

    // 2. Resolve the yarn request through the same flag group the server
    //    binaries flatten. The sentinels stay sentinels; the real values
    //    become knobs.
    let requested = RopeOverrideArgs {
        rope_scaling: Some("yarn".to_string()),
        rope_scale: Some(4.0),
        yarn_orig_ctx: Some(4096),
        yarn_beta_fast: Some(40.0),
        yarn_ext_factor: Some(-1.0),
        ..RopeOverrideArgs::default()
    }
    .resolve()
    .expect("--rope-scaling yarn is serveable since #1472")
    .expect("it is an override");
    assert_eq!(requested.yarn_knobs().orig_ctx, Some(4096));
    assert_eq!(requested.yarn_knobs().beta_fast, Some(40.0));
    assert_eq!(
        requested.yarn_knobs().ext_factor,
        None,
        "-1.0 is b10621's sentinel, not a value"
    );
    rope_overrides::install(Some(requested)).expect("first install in this process");

    // 3. Installed but not yet consumed: refusing to serve is the point.
    let refusal = rope_overrides::verify_applied("some-architecture-off-the-seam")
        .expect_err("an override that reached no RoPE path must not be served");
    assert!(refusal.contains("--rope-scaling yarn"), "{refusal}");
    assert!(refusal.contains("--yarn-orig-ctx 4096"), "{refusal}");

    // 4. The seam consumes it: the scaling-free checkpoint now rotates on a
    //    YaRN table, at unit position scale, with the temperature mscale.
    let overridden = RopeScalingKind::resolve(None, DIMS, BASE, Some(TRAIN_CTX), "probe");
    let yarn_table = table(&overridden);
    assert_eq!(yarn_table.len(), DIMS / 2);
    assert_eq!(overridden.scale(), 1.0);
    let expected_mscale = 0.1 * 4.0f32.ln() + 1.0;
    assert!(
        (overridden.attn_scale() - expected_mscale).abs() < 1e-6,
        "attn_scale {} != {expected_mscale}",
        overridden.attn_scale()
    );

    // The table is a genuine interpolation: the fastest pair extrapolates
    // (divisor unchanged) and the slowest pair interpolates (divisor scaled
    // by the factor), so the flags demonstrably tuned the rotation.
    let plain: Vec<f32> = (0..DIMS / 2)
        .map(|i| BASE.powf(2.0 * i as f32 / DIMS as f32))
        .collect();
    assert!((yarn_table[0] - plain[0]).abs() < 1e-6);
    let last = yarn_table.len() - 1;
    let interpolated = plain[last] * 4.0;
    assert!(
        (yarn_table[last] - interpolated).abs() / interpolated < 1e-5,
        "{} != {interpolated}",
        yarn_table[last]
    );

    assert!(rope_overrides::applications() > 0);

    // 5. Verification passes and no seam recorded a rejection.
    assert_eq!(rope_overrides::rejection(), None);
    rope_overrides::verify_applied("probe").expect("the override reached the rotation");
}
