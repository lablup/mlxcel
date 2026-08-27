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

//! End-to-end contract for llama-server b10621's `--rope-scaling` override
//! (issue #1450).
//!
//! # Why this is a whole test binary for one test
//!
//! The override is process-wide by construction: every model family reads its
//! own `config.json` inside its own loader, so there is no argument to thread
//! from the CLI down to the rotation (see `mlxcel::models::rope_overrides`).
//! `install` is therefore a `OnceLock`, and a second test in the same process
//! that installed a different value would decide what the first one rotates
//! with. `docs/benchmarks.md` handles the same problem for `MLXCEL_QMV_WIDE`
//! the same way: one configuration per process.
//!
//! Cargo gives each `tests/*.rs` file its own binary, so a file with exactly
//! one test is exactly one configuration. The sibling file
//! `llama_compat_rope_freq_base.rs` covers `--rope-freq-base` in its own
//! process for the same reason, and the pure decision table (which needs no
//! global at all) is unit-tested in `src/models/rope_overrides_tests.rs`.
//!
//! # What is asserted
//!
//! The whole lifecycle, in the order a server executes it: an uninstalled
//! process is unaffected, an installed-but-unconsumed override refuses to
//! serve, a seam consumes it and the rotation actually changes, and the
//! verification then passes.

use mlxcel::cli::rope_args::RopeOverrideArgs;
use mlxcel::models::rope_overrides::{self, RopeRuntimeOverride};
use mlxcel::models::rope_utils::{RopeScalingKind, RopeScalingSpec};

/// The `rope_scaling` block Llama 3.1 / 3.2 checkpoints ship.
fn llama3_spec() -> RopeScalingSpec {
    RopeScalingSpec {
        rope_type: Some("llama3".to_string()),
        factor: Some(32.0),
        low_freq_factor: Some(1.0),
        high_freq_factor: Some(4.0),
        original_max_position_embeddings: Some(8192.0),
    }
}

const DIMS: usize = 64;
const BASE: f32 = 500_000.0;

#[test]
fn rope_scaling_none_replaces_the_checkpoints_banded_table_and_is_verified() {
    // 1. Nothing installed: the checkpoint's own block decides, and the
    //    post-load verification has nothing to check.
    assert!(
        rope_overrides::installed().is_none(),
        "this binary must start with no override installed"
    );
    let declared = RopeScalingKind::resolve(Some(&llama3_spec()), DIMS, BASE, "probe");
    assert!(
        declared.freqs().is_some(),
        "a llama3 block builds a frequency table when nothing overrides it"
    );
    assert_eq!(rope_overrides::applications(), 0);
    rope_overrides::verify_applied("probe")
        .expect("with no override installed there is nothing to verify");

    // 2. Install what `--rope-scaling none` resolves to, through the same
    //    flag group the server binaries flatten, so this test cannot pass
    //    against a value the CLI could never produce.
    let requested = RopeOverrideArgs {
        rope_scaling: Some("none".to_string()),
        ..RopeOverrideArgs::default()
    }
    .resolve()
    .expect("--rope-scaling none is serveable")
    .expect("it is an override");
    assert_eq!(
        requested,
        RopeRuntimeOverride::from_flags(Some("none"), None, None, None)
            .expect("valid")
            .expect("an override"),
    );
    rope_overrides::install(Some(requested)).expect("first install in this process");

    // 3. Installed but not yet consumed: refusing to serve is the point. A
    //    server that started here would answer every request with the
    //    checkpoint's own rotation and say nothing.
    let refusal = rope_overrides::verify_applied("some-architecture-off-the-seam")
        .expect_err("an override that reached no RoPE path must not be served");
    assert!(
        refusal.contains("some-architecture-off-the-seam"),
        "{refusal}"
    );
    assert!(refusal.contains("--rope-scaling none"), "{refusal}");
    // The message has to say which families DO honor it, or an operator has no
    // way to tell "wrong flag" from "wrong checkpoint".
    assert!(refusal.contains("Qwen3"), "{refusal}");

    // 4. The seam consumes it, and the rotation changes: the banded table is
    //    gone and the plain `base^(2i/d)` rotation is back.
    let overridden = RopeScalingKind::resolve(Some(&llama3_spec()), DIMS, BASE, "probe");
    assert!(
        overridden.freqs().is_none(),
        "--rope-scaling none must drop the checkpoint's llama3 table"
    );
    assert_eq!(
        overridden.scale(),
        1.0,
        "the plain table rotates at unit position scale"
    );
    assert!(rope_overrides::applications() > 0);

    // 5. Verification now passes, and no seam recorded a rejection.
    assert_eq!(rope_overrides::rejection(), None);
    rope_overrides::verify_applied("probe").expect("the override reached the rotation");

    // 6. Re-installing the same value is accepted (the server's model-switch
    //    path re-enters startup), a different one is not.
    rope_overrides::install(Some(requested)).expect("the same value installs idempotently");
    rope_overrides::install(None).expect_err("a different value must not silently replace it");
}
