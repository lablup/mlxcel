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

//! Unit tests for cold-last-level-cache rotating buffers (issue #906).

use super::{
    DEFAULT_LLC_BYTES, LLC_BYTES_ENV, MAX_ROTATION, ROTATION_HEADROOM, Rotation, apple_slc_bytes,
    cuda_l2_bytes, last_level_cache_bytes, rotation_count_for,
};
use crate::hardware::AppleSiliconGen;
use crate::test_support::env_lock::env_lock;

const MIB: u64 = 1024 * 1024;

// ── Cache-size estimation ────────────────────────────────────────────────────

#[test]
fn apple_slc_grows_with_the_performance_core_tier() {
    assert_eq!(apple_slc_bytes(AppleSiliconGen::M1, 4), 8 * MIB);
    assert_eq!(apple_slc_bytes(AppleSiliconGen::M1, 6), 24 * MIB);
    assert_eq!(apple_slc_bytes(AppleSiliconGen::M1, 8), 48 * MIB);
    // M1 Ultra: two dies, so the largest bucket.
    assert_eq!(apple_slc_bytes(AppleSiliconGen::M1, 16), 96 * MIB);
    assert_eq!(apple_slc_bytes(AppleSiliconGen::M5, 16), 96 * MIB);
}

#[test]
fn unknown_silicon_falls_back_to_the_conservative_floor() {
    assert_eq!(
        apple_slc_bytes(AppleSiliconGen::Unknown, 64),
        DEFAULT_LLC_BYTES
    );
}

#[test]
fn cuda_l2_is_an_explicitly_unimplemented_stub() {
    // Documented as not implemented; the override env var is the CUDA path
    // until an FFI helper for cudaDeviceProp::l2CacheSize exists.
    assert_eq!(cuda_l2_bytes(), None);
}

#[test]
fn env_override_wins_over_the_detected_estimate() {
    let _guard = env_lock();
    // SAFETY: the crate-wide env lock is held for the whole mutation window,
    // so no other test in this binary reads or writes the environment here.
    unsafe { std::env::set_var(LLC_BYTES_ENV, "12345678") };
    let observed = last_level_cache_bytes();
    // SAFETY: same lock window as the set above.
    unsafe { std::env::remove_var(LLC_BYTES_ENV) };
    assert_eq!(observed, 12_345_678);
}

#[test]
fn a_nonsense_env_override_is_ignored() {
    let _guard = env_lock();
    // SAFETY: the crate-wide env lock is held for the whole mutation window.
    unsafe { std::env::set_var(LLC_BYTES_ENV, "not-a-number") };
    let observed = last_level_cache_bytes();
    // SAFETY: same lock window as the set above.
    unsafe { std::env::remove_var(LLC_BYTES_ENV) };
    assert!(observed > 0, "fell back to the detected estimate");
}

// ── Rotation sizing ──────────────────────────────────────────────────────────

#[test]
fn rotation_covers_twice_the_cache() {
    // 8 MiB working set against a 96 MiB cache needs 2 * 96 / 8 = 24 buffers.
    assert_eq!(rotation_count_for(8 * MIB, 96 * MIB), 24);
    assert_eq!(ROTATION_HEADROOM, 2);
}

#[test]
fn rotation_rounds_up_so_the_set_always_exceeds_the_window() {
    // 5 MiB into 2 * 8 MiB = 16 MiB needs 3.2 buffers, so 4.
    assert_eq!(rotation_count_for(5 * MIB, 8 * MIB), 4);
}

#[test]
fn a_working_set_larger_than_the_window_needs_no_rotation() {
    assert_eq!(rotation_count_for(512 * MIB, 96 * MIB), 1);
}

#[test]
fn rotation_is_capped_so_a_tiny_working_set_cannot_explode() {
    assert_eq!(rotation_count_for(1, 96 * MIB), MAX_ROTATION);
}

#[test]
fn a_zero_working_set_is_not_a_division_by_zero() {
    assert_eq!(rotation_count_for(0, 96 * MIB), 1);
}

// ── Rotation iteration ───────────────────────────────────────────────────────

#[test]
fn rotation_cycles_round_robin() {
    let mut r = Rotation::new(3);
    let seen: Vec<usize> = (0..7).map(|_| r.next_index()).collect();
    assert_eq!(seen, vec![0, 1, 2, 0, 1, 2, 0]);
}

#[test]
fn rotation_reset_restarts_the_cycle() {
    let mut r = Rotation::new(3);
    assert_eq!(r.next_index(), 0);
    assert_eq!(r.next_index(), 1);
    r.reset();
    assert_eq!(r.next_index(), 0);
}

#[test]
fn a_single_buffer_rotation_is_the_warm_mode() {
    let mut r = Rotation::new(1);
    assert!(!r.is_rotating());
    assert_eq!(r.mode_tag(), "warm");
    assert_eq!(r.next_index(), 0);
    assert_eq!(r.next_index(), 0);
}

#[test]
fn a_zero_count_degrades_to_one_buffer() {
    let r = Rotation::new(0);
    assert_eq!(r.count(), 1);
    assert!(!r.is_rotating());
}

#[test]
fn a_rotating_set_reports_the_cold_mode_tag() {
    let r = Rotation::new(8);
    assert!(r.is_rotating());
    assert_eq!(r.mode_tag(), "cold-l2");
    assert_eq!(r.count(), 8);
}
