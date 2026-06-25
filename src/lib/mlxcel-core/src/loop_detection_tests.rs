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

use super::{LoopDetectionConfig, MAX_EFFECTIVE_PATTERN_SIZE, detect_repetition_loop};

/// The conservative starting threshold the server applies for the Gemma 4
/// amplifier case (`min_pattern_size=1, max_pattern_size=20, min_count=4`).
fn recommended() -> LoopDetectionConfig {
    LoopDetectionConfig::new(1, 20, 4)
}

// -- enablement / disabled no-op --

#[test]
fn default_is_disabled() {
    let cfg = LoopDetectionConfig::default();
    assert!(!cfg.is_enabled());
    assert_eq!(cfg, LoopDetectionConfig::disabled());
}

#[test]
fn disabled_config_never_fires_even_on_obvious_loop() {
    let stream = vec![7, 7, 7, 7, 7, 7, 7, 7];
    assert!(!detect_repetition_loop(
        &stream,
        &LoopDetectionConfig::disabled()
    ));
    // max_pattern_size == 0 disables regardless of min_count.
    assert!(!detect_repetition_loop(
        &stream,
        &LoopDetectionConfig::new(1, 0, 4)
    ));
}

#[test]
fn min_count_below_two_disables() {
    let stream = vec![5, 5, 5, 5, 5, 5];
    assert!(!LoopDetectionConfig::new(1, 20, 0).is_enabled());
    assert!(!LoopDetectionConfig::new(1, 20, 1).is_enabled());
    assert!(!detect_repetition_loop(
        &stream,
        &LoopDetectionConfig::new(1, 20, 1)
    ));
}

// -- single-token loop boundary (p = 1) --

#[test]
fn single_token_loop_fires_at_exactly_min_count() {
    let cfg = recommended(); // min_count = 4

    // 3 repeats: below threshold, must not fire.
    assert!(!detect_repetition_loop(&[9, 9, 9], &cfg));
    // 3 repeats with unrelated prefix: still below threshold.
    assert!(!detect_repetition_loop(&[1, 2, 9, 9, 9], &cfg));

    // Exactly 4 repeats at the tail: fires.
    assert!(detect_repetition_loop(&[9, 9, 9, 9], &cfg));
    assert!(detect_repetition_loop(&[1, 2, 3, 9, 9, 9, 9], &cfg));

    // 5 repeats: still fires (tail of 4 is all 9).
    assert!(detect_repetition_loop(&[9, 9, 9, 9, 9], &cfg));
}

#[test]
fn single_token_loop_requires_consecutive_tail() {
    let cfg = recommended();
    // Four 9s but interrupted: the tail is not four consecutive 9s.
    assert!(!detect_repetition_loop(&[9, 9, 0, 9, 9], &cfg));
    // Trailing non-loop token breaks the tail pattern.
    assert!(!detect_repetition_loop(&[9, 9, 9, 9, 0], &cfg));
}

// -- multi-token loops (p = 2, 3) --

#[test]
fn two_token_loop_fires() {
    let cfg = recommended();
    // [3,4] repeated 4 times.
    let stream = vec![3, 4, 3, 4, 3, 4, 3, 4];
    assert!(detect_repetition_loop(&stream, &cfg));

    // [3,4] repeated only 3 times: below min_count for p=2, and p=1 sees
    // alternating tokens so it cannot fire either.
    let below = vec![3, 4, 3, 4, 3, 4];
    assert!(!detect_repetition_loop(&below, &cfg));
}

#[test]
fn three_token_loop_fires_with_prefix() {
    let cfg = recommended();
    // unrelated prefix then [1,2,3] x4.
    let stream = vec![8, 8, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
    assert!(detect_repetition_loop(&stream, &cfg));
}

#[test]
fn min_pattern_size_above_one_skips_single_token_loop() {
    // Only scan p in 2..=3: a single-token loop must not trigger.
    let cfg = LoopDetectionConfig::new(2, 3, 4);
    assert!(!detect_repetition_loop(&[5, 5, 5, 5, 5, 5], &cfg));
    // But a two-token loop still fires.
    assert!(detect_repetition_loop(&[5, 6, 5, 6, 5, 6, 5, 6], &cfg));
}

// -- no false positives --

#[test]
fn non_repeating_stream_does_not_fire() {
    let cfg = recommended();
    let stream: Vec<i32> = (0..200).collect();
    assert!(!detect_repetition_loop(&stream, &cfg));
}

#[test]
fn empty_and_short_streams_do_not_fire() {
    let cfg = recommended();
    assert!(!detect_repetition_loop(&[], &cfg));
    assert!(!detect_repetition_loop(&[1], &cfg));
    assert!(!detect_repetition_loop(&[1, 1], &cfg));
}

#[test]
fn natural_short_repeats_below_threshold_do_not_fire() {
    let cfg = recommended();
    // "the the the" style: token 42 appears 3 times, but not 4 in a row.
    let stream = vec![10, 42, 11, 42, 12, 42, 13];
    assert!(!detect_repetition_loop(&stream, &cfg));
}

// -- normalization rules --

#[test]
fn min_pattern_size_zero_treated_as_one() {
    // min_pattern_size = 0 should behave like 1 and catch single-token loops.
    let cfg = LoopDetectionConfig::new(0, 20, 4);
    assert!(detect_repetition_loop(&[7, 7, 7, 7], &cfg));
}

#[test]
fn min_pattern_size_clamped_to_max() {
    // min_pattern_size (5) > max_pattern_size (3): clamp min down to 3 so the
    // scan range stays well-formed (p in 3..=3) instead of being empty.
    let cfg = LoopDetectionConfig::new(5, 3, 2);
    // [1,2,3] repeated twice -> p=3 fires.
    assert!(detect_repetition_loop(&[1, 2, 3, 1, 2, 3], &cfg));
    // A two-token loop must not fire: p=2 is below the clamped min of 3, and
    // [5,6,5,6,5,6] is not a single 3-token block repeated.
    assert!(!detect_repetition_loop(&[5, 6, 5, 6, 5, 6], &cfg));
}

#[test]
fn observed_cjk_collapse_shape_fires() {
    // Mirrors the issue repro: a long clean prefix then a single garbage token
    // (stand-in for `様`) repeating to fill the thinking budget.
    let mut stream: Vec<i32> = (100..160).collect();
    stream.extend(std::iter::repeat_n(255, 30));
    assert!(detect_repetition_loop(&stream, &recommended()));
}

// -- effective max_pattern_size cap (CPU-DoS guard) --
//
// `LoopDetectionConfig::new(1, <huge>, 2)` is exactly what BOTH untrusted
// override surfaces produce: the per-request path (`loop_detection_from_request`,
// from the OpenAI `max_pattern_size` chat field) and the global-env triple
// (`MLXCEL_LOOP_DETECTION`). Neither clamps the value; the detector caps the
// effective scan at `MAX_EFFECTIVE_PATTERN_SIZE`. These detector-level tests
// therefore cover both override paths at once.

#[test]
fn huge_max_pattern_size_still_fires_and_terminates() {
    // A per-request / global-env override with an unbounded max_pattern_size.
    let cfg = LoopDetectionConfig::new(1, usize::MAX, 2);

    // Still fires on a real short loop: a single token repeated many times.
    let loop_stream = vec![9; 256];
    assert!(detect_repetition_loop(&loop_stream, &cfg));

    // Does NOT fire on a long strictly-increasing, non-repeating stream, and the
    // scan terminates promptly (the cap bounds the work despite usize::MAX).
    let non_repeating: Vec<i32> = (0..400).collect();
    assert!(!detect_repetition_loop(&non_repeating, &cfg));
}

#[test]
fn cap_bounds_detected_pattern_length() {
    // Same huge max as the untrusted overrides; min set to the cap so only the
    // capped pattern length is in range.
    let cfg = LoopDetectionConfig::new(MAX_EFFECTIVE_PATTERN_SIZE, usize::MAX, 2);

    // A pattern of length exactly MAX_EFFECTIVE_PATTERN_SIZE repeated min_count
    // (2) times: this length is scanned, so it fires.
    let block_at_cap: Vec<i32> = (0..MAX_EFFECTIVE_PATTERN_SIZE as i32).collect();
    let mut at_cap = block_at_cap.clone();
    at_cap.extend_from_slice(&block_at_cap);
    assert!(detect_repetition_loop(&at_cap, &cfg));

    // A pattern of length MAX_EFFECTIVE_PATTERN_SIZE + 1 repeated min_count (2)
    // times: this length is beyond the cap and never scanned, so it does NOT
    // fire even though the stream is a genuine two-times repeat.
    let block_above_cap: Vec<i32> = (0..(MAX_EFFECTIVE_PATTERN_SIZE as i32 + 1)).collect();
    let mut above_cap = block_above_cap.clone();
    above_cap.extend_from_slice(&block_above_cap);
    assert!(!detect_repetition_loop(&above_cap, &cfg));
}

#[test]
fn min_pattern_size_above_cap_does_not_panic() {
    // min_pattern_size well above the cap with a huge max_pattern_size: the min
    // clamps down to the capped max, so the only scanned pattern size is the cap
    // itself (p == MAX_EFFECTIVE_PATTERN_SIZE) and the range stays well-formed.
    // A single-token loop shorter than that pattern's window
    // (MAX_EFFECTIVE_PATTERN_SIZE * min_count) is below the scan window, so the
    // detector breaks out without matching: it must return false without
    // panicking (the guarantee here is no panic / well-formed range).
    let cfg = LoopDetectionConfig::new(MAX_EFFECTIVE_PATTERN_SIZE + 100, usize::MAX, 2);
    let single_token_loop = vec![3; MAX_EFFECTIVE_PATTERN_SIZE];
    assert!(!detect_repetition_loop(&single_token_loop, &cfg));
}
