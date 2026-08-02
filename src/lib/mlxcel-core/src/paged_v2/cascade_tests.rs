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

//! Host-side tests for cascade detection and planning (issue #903).
//!
//! No MLX and no GPU: everything here is a decision made from a
//! [`PagedCsrView`], and keeping it that way is what lets the rules that pick a
//! subgroup be checked exhaustively at unit-test cost. The launch itself is
//! covered by `cascade_launch_tests`.

use super::*;

const PAGE: i32 = 32;

/// Build a view from `(physical rows, last_page_len, first_page_offset)` per
/// request. Row numbers are the identity mapping of block ids to pool rows, so
/// two requests naming the same row hold the same block, which is exactly the
/// sharing the detector reads.
fn view(page_size: i32, requests: &[(Vec<i32>, i32, i32)]) -> PagedCsrView {
    let mut v = PagedCsrView {
        page_size,
        indices: Vec::new(),
        indptr: vec![0],
        last_page_len: Vec::new(),
        first_page_offset: Vec::new(),
        seq_lens: Vec::new(),
        rope_offsets: Vec::new(),
    };
    for (rows, lpl, fpo) in requests {
        v.indices.extend_from_slice(rows);
        v.indptr.push(v.indices.len() as i32);
        v.last_page_len.push(*lpl);
        v.first_page_offset.push(*fpo);
        let len = if rows.is_empty() {
            0
        } else {
            (rows.len() as i32 - 1) * page_size + lpl - fpo
        };
        v.seq_lens.push(len);
        v.rope_offsets.push(len);
    }
    v.validate().expect("test view must be well formed");
    v
}

/// Four requests behind one shared prompt: pages 0..shared are the same rows,
/// then each has `private` rows of its own.
fn shared_prompt(shared: usize, private: usize, members: usize) -> PagedCsrView {
    let requests: Vec<(Vec<i32>, i32, i32)> = (0..members)
        .map(|m| {
            let mut rows: Vec<i32> = (0..shared as i32).collect();
            rows.extend((0..private as i32).map(|p| 1000 + (m as i32) * 100 + p));
            (rows, PAGE, 0)
        })
        .collect();
    view(PAGE, &requests)
}

#[test]
fn the_env_gate_honours_both_spellings_and_falls_back_to_the_default() {
    assert!(parse_cascade_enabled(Some("1")));
    assert!(parse_cascade_enabled(Some(" ON ")));
    assert!(parse_cascade_enabled(Some("yes")));
    assert!(!parse_cascade_enabled(Some("0")));
    assert!(!parse_cascade_enabled(Some("off")));
    assert!(!parse_cascade_enabled(Some("false")));
    // The issue's documented kill switch is `=0`, which disables the feature
    // regardless of which way `DEFAULT_CASCADE_ENABLED` points.
    assert!(!parse_cascade_enabled(Some("0")));
    // Unset and unparseable both take the shipped default.
    assert_eq!(parse_cascade_enabled(None), DEFAULT_CASCADE_ENABLED);
    assert_eq!(parse_cascade_enabled(Some("")), DEFAULT_CASCADE_ENABLED);
    assert_eq!(parse_cascade_enabled(Some("maybe")), DEFAULT_CASCADE_ENABLED);
}

#[test]
fn thresholds_parse_only_non_negative_integers() {
    assert_eq!(parse_threshold(None, 16), 16);
    assert_eq!(parse_threshold(Some("  "), 16), 16);
    assert_eq!(parse_threshold(Some("-4"), 16), 16);
    assert_eq!(parse_threshold(Some("nope"), 16), 16);
    assert_eq!(parse_threshold(Some(" 64 "), 16), 64);
    assert_eq!(parse_threshold(Some("0"), 16), 0);
}

#[test]
fn a_shared_prompt_across_the_whole_batch_is_detected() {
    let v = shared_prompt(64, 3, 4);
    let group = detect_shared_prefix(&v, 16, 2).expect("64 shared pages across 4 requests");
    assert_eq!(group.shared_pages, 64);
    assert_eq!(group.members, vec![0, 1, 2, 3]);
    assert_eq!(group.saved_page_reads(), 64 * 3);
}

#[test]
fn the_span_stops_one_page_short_of_the_shortest_member() {
    // Two requests sharing every page they have. The last page of a request is
    // the only one that may be partially filled, and level 0 declares its pages
    // full, so the span must never reach it.
    let v = view(PAGE, &[(vec![0, 1, 2, 3], 7, 0), (vec![0, 1, 2, 3, 4], PAGE, 0)]);
    let group = detect_shared_prefix(&v, 1, 2).expect("a shared prefix exists");
    assert_eq!(group.shared_pages, 3, "the 4th page of request 0 is partial");
    assert_eq!(group.members, vec![0, 1]);
}

#[test]
fn both_thresholds_are_enforced() {
    let v = shared_prompt(20, 2, 3);
    assert!(detect_shared_prefix(&v, 21, 2).is_none(), "span below floor");
    assert!(detect_shared_prefix(&v, 16, 4).is_none(), "too few members");
    assert!(detect_shared_prefix(&v, 20, 3).is_some());
    // A member floor below 2 is nonsense (there is no duplication with one
    // member) and is refused rather than interpreted.
    assert!(detect_shared_prefix(&v, 1, 1).is_none());
    assert!(detect_shared_prefix(&v, 0, 2).is_none());
}

#[test]
fn a_mid_page_window_start_is_excluded_from_every_group() {
    // Request 1 has a sliding window trimmed into the middle of its first page,
    // so its page list no longer starts on a block boundary and the shared span
    // could not be stated in whole pages.
    let mut v = shared_prompt(20, 2, 3);
    v.first_page_offset[1] = 5;
    v.seq_lens[1] -= 5;
    v.validate().expect("still structurally valid");
    let group = detect_shared_prefix(&v, 16, 2).expect("requests 0 and 2 still share");
    assert_eq!(group.members, vec![0, 2]);
    // With only one eligible request left there is nothing to hoist.
    v.first_page_offset[2] = 5;
    v.seq_lens[2] -= 5;
    v.validate().expect("still structurally valid");
    assert!(detect_shared_prefix(&v, 16, 2).is_none());
}

#[test]
fn the_subgroup_that_saves_the_most_reads_wins() {
    // Group A: 2 requests sharing 40 pages  -> 40 saved page reads.
    // Group B: 3 requests sharing 30 pages  -> 60 saved page reads.
    let mut requests: Vec<(Vec<i32>, i32, i32)> = Vec::new();
    for m in 0..2 {
        let mut rows: Vec<i32> = (0..40).collect();
        rows.push(5000 + m);
        requests.push((rows, PAGE, 0));
    }
    for m in 0..3 {
        let mut rows: Vec<i32> = (100..130).collect();
        rows.push(6000 + m);
        requests.push((rows, PAGE, 0));
    }
    let v = view(PAGE, &requests);
    let group = detect_shared_prefix(&v, 16, 2).expect("both groups clear the floor");
    assert_eq!(group.members, vec![2, 3, 4]);
    assert_eq!(group.shared_pages, 30);
    assert_eq!(group.saved_page_reads(), 60);
}

#[test]
fn requests_that_merely_look_similar_do_not_group() {
    // Same page count, same lengths, different physical rows: no sharing, so
    // nothing to hoist. Detection is by identity of the pool row, which within
    // one view is identity of the block.
    let v = view(
        PAGE,
        &[
            ((0..40).collect(), PAGE, 0),
            ((100..140).collect(), PAGE, 0),
        ],
    );
    assert!(detect_shared_prefix(&v, 16, 2).is_none());
}

#[test]
fn the_plan_splits_the_same_pages_the_flat_view_would_have_read() {
    let v = shared_prompt(64, 3, 4);
    let group = detect_shared_prefix(&v, 16, 2).expect("group");
    let plan = build_cascade_plan(&v, group).expect("plan");

    assert_eq!(plan.prefix_view.batch(), 1);
    assert_eq!(plan.prefix_view.total_pages(), 64);
    assert_eq!(plan.prefix_view.seq_lens[0], 64 * PAGE);
    assert_eq!(plan.shared_tokens(), 64 * PAGE as usize);

    for r in 0..v.batch() {
        // Every request's two levels cover its whole visible range, exactly
        // once, with no page counted twice and none dropped.
        let level0 = plan.prefix_view.seq_lens[0];
        assert_eq!(plan.suffix_view.seq_lens[r] + level0, v.seq_lens[r]);
        assert_eq!(
            plan.suffix_view.pages_for(r) + plan.prefix_view.total_pages(),
            v.pages_for(r)
        );
        let begin = plan.suffix_view.indptr[r] as usize;
        let end = plan.suffix_view.indptr[r + 1] as usize;
        let flat_begin = v.indptr[r] as usize + 64;
        assert_eq!(
            &plan.suffix_view.indices[begin..end],
            &v.indices[flat_begin..v.indptr[r + 1] as usize]
        );
    }
}

#[test]
fn a_mixed_batch_gives_non_members_a_group_of_one() {
    // Requests 0 and 2 share; request 1 does not. Its level-1 range is its full
    // range and its merge group is a single row, which the merge kernel
    // resolves to the identity.
    let mut requests: Vec<(Vec<i32>, i32, i32)> = Vec::new();
    let mut a: Vec<i32> = (0..20).collect();
    a.push(900);
    requests.push((a, PAGE, 0));
    requests.push(((500..508).collect(), 11, 0));
    let mut c: Vec<i32> = (0..20).collect();
    c.push(901);
    requests.push((c, 7, 0));
    let v = view(PAGE, &requests);

    let group = detect_shared_prefix(&v, 16, 2).expect("0 and 2 share 20 pages");
    assert_eq!(group.members, vec![0, 2]);
    let plan = build_cascade_plan(&v, group).expect("plan");

    // concat(level 1 [3 rows], level 0 [2 rows]) reordered to 0, 3, 1, 2, 4.
    assert_eq!(plan.merge_order, vec![0, 3, 1, 2, 4]);
    assert_eq!(plan.o_indptr, vec![0, 2, 3, 5]);
    assert_eq!(plan.member_rows, vec![0, 2]);
    assert!(!plan.members_are_whole_batch());

    // The non-member keeps its whole range at level 1.
    assert_eq!(plan.suffix_view.seq_lens[1], v.seq_lens[1]);
    assert_eq!(plan.suffix_view.pages_for(1), v.pages_for(1));
}

#[test]
fn the_whole_batch_case_skips_the_member_gather() {
    let v = shared_prompt(32, 2, 4);
    let group = detect_shared_prefix(&v, 16, 2).expect("group");
    let plan = build_cascade_plan(&v, group).expect("plan");
    assert!(plan.members_are_whole_batch());
    assert_eq!(plan.merge_order, vec![0, 4, 1, 5, 2, 6, 3, 7]);
    assert_eq!(plan.o_indptr, vec![0, 2, 4, 6, 8]);
}

#[test]
fn the_builder_refuses_a_group_the_detector_would_never_produce() {
    let v = shared_prompt(20, 2, 3);
    // Unsorted members break both the merge grouping and the binary searches.
    let err = build_cascade_plan(
        &v,
        CascadeGroup {
            shared_pages: 10,
            members: vec![2, 0],
        },
    )
    .unwrap_err();
    assert!(err.contains("ascending"), "{err}");

    // A span that swallows a member's last page would declare a partial page
    // full at level 0.
    let err = build_cascade_plan(
        &v,
        CascadeGroup {
            shared_pages: 22,
            members: vec![0, 1],
        },
    )
    .unwrap_err();
    assert!(err.contains("does not exceed"), "{err}");

    // An out-of-range member.
    let err = build_cascade_plan(
        &v,
        CascadeGroup {
            shared_pages: 10,
            members: vec![0, 9],
        },
    )
    .unwrap_err();
    assert!(err.contains("outside the batch"), "{err}");

    // A zero-page span.
    let err = build_cascade_plan(
        &v,
        CascadeGroup {
            shared_pages: 0,
            members: vec![0, 1],
        },
    )
    .unwrap_err();
    assert!(err.contains("zero pages"), "{err}");
}

#[test]
fn an_empty_or_single_request_batch_never_cascades() {
    let empty = PagedCsrView {
        page_size: PAGE,
        ..PagedCsrView::default()
    };
    assert!(detect_shared_prefix(&empty, 1, 2).is_none());
    let single = view(PAGE, &[((0..40).collect(), PAGE, 0)]);
    assert!(detect_shared_prefix(&single, 1, 2).is_none());
}
