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

//! Arithmetic and diagnostic tests for the b10621 image-token budget
//! (issue #1451).
//!
//! The process-wide `INSTALL`ed override is deliberately never set here: a
//! `OnceLock` set by one test would leak into every other test in the same
//! binary. Everything these tests need is reachable through the explicit-value
//! entry points, which is why [`ImageTokenOverride::apply`] and
//! [`unapplied_diagnostic`] take their inputs rather than reading the global.

use super::*;

/// Qwen2-VL geometry: 14-pixel patches merged 2x2, so `patch_area == 784`.
const PATCH: usize = 14;
const MERGE: usize = 2;
const PATCH_AREA: usize = PATCH * PATCH * MERGE * MERGE;

#[test]
fn patch_area_matches_upstreams_definition() {
    assert_eq!(PATCH_AREA, 784);
    assert_eq!(patch_area(PATCH, MERGE), Some(784));
}

#[test]
fn both_halves_convert_tokens_to_pixels() {
    let over = ImageTokenOverride::from_bounds(Some(8), Some(1024)).expect("an override");
    let (min, max) = over.apply(PATCH, MERGE, 4 * 784, 16_384 * 784);
    assert_eq!(min, 8 * PATCH_AREA);
    assert_eq!(max, 1024 * PATCH_AREA);
}

#[test]
fn a_half_that_was_not_requested_keeps_the_checkpoints_own_bound() {
    let declared_min = 4 * PATCH_AREA;
    let declared_max = 16_384 * PATCH_AREA;

    let min_only = ImageTokenOverride::from_bounds(Some(32), None).expect("an override");
    assert_eq!(
        min_only.apply(PATCH, MERGE, declared_min, declared_max),
        (32 * PATCH_AREA, declared_max)
    );

    let max_only = ImageTokenOverride::from_bounds(None, Some(256)).expect("an override");
    assert_eq!(
        max_only.apply(PATCH, MERGE, declared_min, declared_max),
        (declared_min, 256 * PATCH_AREA)
    );
}

#[test]
fn neither_half_means_no_override_at_all() {
    assert!(ImageTokenOverride::from_bounds(None, None).is_none());
}

#[test]
fn a_degenerate_geometry_leaves_the_declared_bounds_alone() {
    // A zero patch size would make `patch_area` zero and collapse every image
    // to a single patch; returning the declared bounds is the safe reading.
    let over = ImageTokenOverride::from_bounds(Some(8), Some(64)).expect("an override");
    assert_eq!(over.apply(0, MERGE, 111, 222), (111, 222));
    assert_eq!(over.apply(PATCH, 0, 111, 222), (111, 222));
    assert_eq!(over.apply(usize::MAX, 2, 111, 222), (111, 222));
}

#[test]
fn an_overflowing_token_count_falls_back_to_the_declared_bound() {
    let over = ImageTokenOverride::from_bounds(Some(u32::MAX), None).expect("an override");
    let (min, _) = over.apply(usize::MAX / 3, 1, 999, 1000);
    assert_eq!(min, 999);
}

#[test]
fn describe_renders_what_the_operator_typed() {
    let both = ImageTokenOverride::from_bounds(Some(8), Some(64)).expect("an override");
    assert_eq!(
        both.describe(),
        "--image-min-tokens 8 --image-max-tokens 64"
    );
    let min_only = ImageTokenOverride::from_bounds(Some(8), None).expect("an override");
    assert_eq!(min_only.describe(), "--image-min-tokens 8");
}

#[test]
fn the_unapplied_diagnostic_names_the_checkpoint_and_the_flag() {
    let over = ImageTokenOverride::from_bounds(None, Some(256)).expect("an override");
    let message = unapplied_diagnostic("llava-1.5-7b-4bit", &over);
    assert!(message.contains("llava-1.5-7b-4bit"), "{message}");
    assert!(message.contains("--image-max-tokens 256"), "{message}");
    // The operator has to be told what to do, not only that it failed.
    assert!(message.contains("Drop the flag"), "{message}");
    assert!(message.contains("dynamic resolution"), "{message}");
}

#[test]
fn resolve_pixel_bounds_is_a_pass_through_without_an_installed_override() {
    // Nothing installs an override in this binary, so this exercises the hot
    // path every ordinary preprocess takes.
    assert!(installed().is_none());
    assert_eq!(resolve_pixel_bounds(PATCH, MERGE, 7, 9), (7, 9));
}

#[test]
fn installing_the_same_value_twice_is_accepted() {
    // `install(None)` is what an ordinary startup does, and the model-switch
    // path re-enters it; both must succeed.
    assert!(install(None).is_ok());
    assert!(install(None).is_ok());
}
