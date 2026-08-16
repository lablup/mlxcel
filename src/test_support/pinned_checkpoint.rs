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

//! Shared gate for tests that contract-check a pinned, machine-local
//! checkpoint against a published shape or inventory.
//!
//! These checkpoints run tens of gigabytes and are absent on most machines,
//! so the tests that use them skip when the checkpoint cannot be read. That
//! skip must stay narrow: only availability and readability problems (an
//! absent file, an unreadable file, JSON that does not parse, a truncated
//! header) may skip. A genuine contract violation, such as a shape or
//! inventory that disagrees with the published contract, must always fail
//! its own assertion instead of routing through this helper.
//!
//! On a machine that owns the checkpoint, an unconditional skip is a silent
//! loss of the only coverage that exists for the contract, so setting
//! `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` turns that skip into a hard failure.
//! This module only ever turns a skip into a failure; it must never be used
//! to turn a failure into a skip.

/// Report that a pinned checkpoint `test_name` depends on is not usable, and
/// either fail or skip depending on `MLXCEL_REQUIRE_PINNED_CHECKPOINTS`.
///
/// Machines that never downloaded the checkpoint skip, so the rest of the
/// suite still runs there. Machines that own it can set
/// `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` to turn the skip into a failure, so
/// a half-downloaded or corrupted checkpoint cannot quietly disable this
/// contract check forever on the one machine positioned to enforce it.
///
/// `test_name` is the caller's own `#[test]` function name, so the skip or
/// failure message still identifies which test is affected even though the
/// gate itself is shared across call sites.
pub(crate) fn skip_or_fail_pinned_checkpoint(test_name: &str, reason: &str) {
    // The crate-wide env lock serializes this read against tests that mutate
    // the process environment with `unsafe set_var`; on Rust 2024 an
    // unsynchronized concurrent read of the env block is undefined behavior.
    // Hold the guard only for the read, and drop it before the assertion so
    // a failing assertion here cannot poison the mutex for later tests.
    let required = {
        let _env_guard = crate::test_support::env_lock::env_lock();
        std::env::var("MLXCEL_REQUIRE_PINNED_CHECKPOINTS").is_ok_and(|value| value == "1")
    };
    assert!(
        !required,
        "MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1 but the pinned checkpoint {test_name} needs is not \
         usable: {reason}"
    );
    eprintln!("Skipping {test_name}: {reason}");
}
